"""MSA module for AlphaFold3 -- the whole four-block stack as one launch plan.

4-block MSA module: each block runs OPM -> MSA row attention -> transition ->
PairBlock (the last block skips the MSA-side update).

Reference: openfold3/core/model/latent/msa_module.py MSAModuleStack

What this operator actually costs
--------------------------------
The captured shape is ``m: bf16[1, 8, 16, 64]``, ``z: bf16[1, 16, 16, 128]`` --
80 KiB of live state and a fraction of a microsecond of B200 tensor-core work
for the whole stack.  Composing the frozen L2 winners through their
``nn.Module`` forwards issues **63 kernels** (53 Triton + 10 eager residual
adds) and measures, on this box:

    harness 362 us    host enqueue 354 us    device (graph replay) 180 us

and this file measures **163 us at 19 us of host** on the same box, a 46.9x
speedup against the eager reference where the composition scores 20.4x.

i.e. it is *host*-bound with 174 us to spare on the device.  The harness zeroes
a 253 MiB L2-flush buffer before its start event, which buys the CPU only a
~70 us head start, so the remaining ~280 us of Python is on the clock.

Two facts, both measured here (``dev/launch_probe.py``, 48 dependent trivial
kernels), set the design:

=========================  ==========  ===========  =========
variant                     device us    us/launch    host us
=========================  ==========  ===========  =========
eager, no PDL                  152.5        3.18        147.0
graph, no PDL                   75.8        1.58          1.9
eager, PDL                      68.0        1.42        112.9
**graph + PDL**                 69.4        1.45          1.9
=========================  ==========  ===========  =========

Programmatic dependent launch keeps its ~1.4 us device floor *inside* a CUDA
graph, and the graph takes host cost to nothing.  They are not alternatives:
the plan below is one graph of PDL-chained kernels, and gets both.

So this file owns the whole stack rather than composing four module towers:

* **one plan, built once.**  Every block's weights are folded into two
  contiguous arenas (pre-transposed and pre-concatenated where the kernels want
  that), every intermediate buffer is allocated once, and the complete argument
  list of every kernel is built once.  A forward patches *four* pointers -- the
  caller's four inputs, into one staging kernel -- and replays the graph.
* **the ten residual adds are gone.**  ``z += OPM``, ``m += msa_att`` and
  ``m += transition`` are folded into the producing kernel's epilogue (or, for
  the two MSA-side ones, into the *consumer's* prologue where that also removes
  a buffer).  Worth 10 launches and 16.2 us of device time on its own.
* **the last block's MSA side is not emitted at all.**  ``skip_msa_update`` is
  a plan-time fact, not a runtime branch.
* **PDL everywhere, with each kernel's loads sorted around its wait.**  A load
  of state written two or more grids back is already final when a grid starts
  (grid k+2 cannot start until k+1 has run its own ``gdc_wait``), so it belongs
  *above* ``gdc_wait()`` where its latency overlaps the producer's tail.  In a
  chain this long almost everything qualifies: only the immediately preceding
  kernel's output has to be read after the wait.  The exceptions are the loads
  that cross a stream boundary (below), which have to wait.
* **the graph is a DAG, not a chain.**  Once a block's OPM has landed, its MSA
  update and its pair stack are independent -- they share only a read-only
  ``z`` -- and the *next* block's OPM depends on ``m``, not on ``z``, so it can
  run a block early too.  All three go on a second stream and hide behind the
  pair stack they run beside.  Critical path: 53 nodes -> 41.
* **the L2 flush is *not* worth fighting.**  The harness flushes 253 MiB before
  its start event, and the body costs ~23 us more cold than warm.  It looks like
  weight latency and is not: neither a ``prefetch.global.L2`` of all 5.9 MiB nor
  a kernel that genuinely *reads* all of it ahead of time (``_warm``, still here
  behind ``MSA_WARM=1``) moves that gap by a measurable microsecond.  Both are
  off.  See ITERATIONS.md; do not re-derive either.

The schedule, then, is:

===============================  =============================================
critical path (main stream)      second stream
===============================  =============================================
``OPM_0``  (``z += opm(m)``)
``pair_0`` kernels 1..9          ``att_0``, ``tr_h_0``, ``tr_o_0``,
                                 ``OPM_1`` (writes ``u_1``)
``pair_0`` kernel 10 ``+ u_1``   <-- join
``pair_1`` kernels 1..9          ``att_1``, ``tr_h_1``, ``tr_o_1``, ``OPM_2``
``pair_1`` kernel 10 ``+ u_2``   <-- join
``pair_2`` kernels 1..9          ``att_2``, ``tr_h_2``, ``tr_o_2``, ``OPM_3``
``pair_2`` kernel 10 ``+ u_3``   <-- join
``pair_3`` kernels 1..10
===============================  =============================================

41 nodes of critical path out of 53 in the graph, plus one staging and one
output kernel outside it.  The side chain is ~13 us of work against the ~29 us
the pair stack's first nine kernels take, so it is genuinely free.  Measured, on
the harness: serial 208 us -> fork 185 -> fork+hoist 165 (and the uncaptured
straight-line PDL chain, the fallback, 222 us at 241 us of host).

The pair stack is the frozen L2 ``PairBlock`` winner's own ten kernels and its
``_Plan``, spliced into this plan's launch list rather than driven through
``PairBlock.forward``: its buffers are chained so that block *i*'s pair output
*is* block *i+1*'s ``z`` input, and no pointer in it ever changes again.  Only
its last kernel is re-emitted here (``_sg_out_u``), to add ``u``.  The
MSA-attention kernel is likewise the frozen L2 kernel verbatim.  The outer
product mean and the MSA transition are re-emitted because their residual adds
fold into them.

Numerics
--------
The eager baseline rounds every stage to bf16 on its way through memory, and a
four-block stack amplifies any deviation from *that* rounding: the composition
of the L2 winners already lands at 0.9956 of elements inside atol=rtol=1e-2,
against a 0.99 requirement.  So every fused residual rounds the operator's
output to bf16 *before* adding it, exactly where the reference's store did,
rather than keeping the fp32 accumulator: ``(z + res.to(bf16)).to(bf16)`` and
not ``(z + res).to(bf16)``.  Every reduction and accumulator is fp32.

Fallback
--------
Anything the plan cannot prove -- a shape, dtype, device or Triton feature off
the fast path, grad enabled, a relocated parameter -- goes to ``_reference``,
which is the baseline composition over the frozen L2 submodules and is correct
everywhere.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

from ..L2.alphafold3_msa_attention import MSARowAttentionWithPairBias
from ..L2.alphafold3_outer_product_mean import OuterProductMean
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition

# The frozen winners' kernels and launch machinery, reused verbatim.
from ..L2.alphafold3_msa_attention import _fused as _msa_attn
from ..L2.alphafold3_outer_product_mean import _load as _opm_load
from ..L2.alphafold3_outer_product_mean import _norm as _opm_norm
from ..L2.alphafold3_pair_block import _ARGV0, _Call, _Plan as _PairPlan
from ..L2.alphafold3_pair_block import _ln, _sig, _Unsupported
from ..L2.alphafold3_pair_block import _SB, _ST, _TK, _WARPS4

try:  # Triton 3.6+; probed, not assumed
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except Exception:  # pragma: no cover - older Triton
    _HAS_PDL = False


__targets__ = ["MSAModuleStack"]

_FAST_DTYPES = (torch.bfloat16,)

# PDL is the device-side lever, the graph is the host-side one, and they
# compose (see the module docstring's table). ``_GRAPH`` falling back to False
# leaves a correct -- just host-heavier -- straight-line PDL chain.
_PDL = _HAS_PDL
_GRAPH = True

# Outer product mean: one CTA per (pair block, c_z channel). Inherited from the
# frozen L2 winner, whose sweep is in its own ITERATIONS.md.
_OPM_WARPS, _OPM_STAGES = 4, 2

# MSA transition (c_in=64, hidden=256, 128 rows here): row tile, column tile,
# warps, stages for the SwiGLU stage and the output stage. The frozen L2
# SwiGLUTransition swept exactly this shape ((`lns`,128,64,256) and
# (`out`,128,256,64) in its table); the pre-transposed weight layout here is the
# only difference.
_TH_BM, _TH_BN, _TH_WARPS, _TH_STAGES = 16, 32, 8, 3
_TO_BM, _TO_BN, _TO_WARPS, _TO_STAGES = 16, 16, 8, 4

# Staging / output kernels: elements per program, and a floor on the program
# count (which only matters when the -- default off -- arena prefetch is on, to
# spread its instructions thin).
_IO_BL = 128
_IO_NP = 512

def _flag(name: str, default: str) -> bool:
    """An environment override, so a sweep cannot silently become the default."""
    return os.environ.get(name, default) != "0"


_GRAPH = _flag("MSA_GRAPH", "1") and _GRAPH

# The schedule knobs, read at *plan build* time rather than import time so a
# sweep can rebuild a plan under a different setting inside one process. Each
# ships at its measured winner; ITERATIONS.md has what every one of them bought.
#
# ``MSA_PF``     ``prefetch.global.L2`` of every weight arena from the staging
#                kernel. **Measured worthless** (143.36 against 143.30 us on the
#                flushed body) and negative on the harness, because the staging
#                kernel is outside the graph and on the critical path. Off.
# ``MSA_FORK``   Run each block's MSA side on a second stream inside the graph.
#                The MSA update and the pair stack are independent once the
#                block's OPM has landed -- they share only a read-only ``z`` --
#                so three kernels per block hide behind the pair stack's ten
#                instead of extending the chain. A captured graph is a DAG, so
#                the concurrency is real on replay; with the knob off the same
#                order runs serially, which is still correct. Worth 22 us.
# ``MSA_HOIST``  Take the outer product mean off the critical path as well. Its
#                input is ``m``, not ``z``, so block i+1's OPM can run on the
#                side stream *during* block i, writing ``u`` to a buffer of its
#                own; the ``z += u`` add then rides in the epilogue of block i's
#                last pair kernel and costs no launch. Block 0 keeps the
#                in-place form -- there is nothing earlier to hide it behind.
# ``MSA_WARM``   Pull every weight arena into L2 with a load-based kernel at the
#                head of the first block's side stream, where its duration is
#                hidden. The flushed body does cost ~23 us more than the warm
#                one -- but this closes **none** of it (164.77 us harness /
#                143.26 flushed with, 164.70 / 143.30 without), so the gap is not
#                weight latency. Off. Kept because it is the disproof.
# ``MSA_MEARLY`` Hoist the OPM's ``m`` load above its ``gdc_wait()``. Measured
#                worth 0.0 us, and the plan disables it in both schedules where
#                its "written two grids back" argument stops holding.
# ``MSA_WARM_LAST`` Put the warm-up at the *tail* of the first block's side chain
#                rather than its head. Also worth nothing (143.42 vs 143.26).
_KNOBS = (("MSA_PF", "0"), ("MSA_FORK", "1"), ("MSA_HOIST", "1"),
          ("MSA_WARM", "0"), ("MSA_MEARLY", "1"), ("MSA_WARM_LAST", "0"))

# How many arenas ``_warm`` can take. Six covers four blocks (this plan's two
# plus one per pair block); a deeper stack falls back to warming a prefix.
_WARM_NA = 6
_WARM_BL = 2048


# ---------------------------------------------------------------------------
# Staging in, output out, and the L2 warm-up
# ---------------------------------------------------------------------------
@triton.jit
def _pf_at(addr, off):
    """Fire-and-forget ``prefetch.global.L2`` of the 128 B lines at ``addr+off``.

    ``addr`` is a raw device address loaded from a table rather than a kernel
    pointer argument, because the arenas being warmed belong to five different
    allocations (this plan's two, plus one per pair block) and the count is a
    plan-time fact. There is no result, so nothing in the issuing program
    depends on it: the requests are in flight while the program retires.
    """
    tl.inline_asm_elementwise("prefetch.global.L2 [$1];", "=r,l",
                              [addr + off.to(tl.int64)],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ingest(SM, SZ, SMM, SPM, DST, PTRS, NLINE,
            NM: tl.constexpr, NZ: tl.constexpr, NMM: tl.constexpr,
            NPM: tl.constexpr, OM: tl.constexpr, OZ: tl.constexpr,
            OMM: tl.constexpr, OPM_: tl.constexpr,
            NAR: tl.constexpr, BL: tl.constexpr, PFPP: tl.constexpr,
            PDL: tl.constexpr):
    """Copy the caller's four inputs into the plan's arena, and warm L2.

    The four input pointers are the only thing that moves between calls (the
    harness hands out a fresh slot of a shifting pool every iteration), so staging
    them here is what makes every *other* kernel's argument list constant --
    which is what lets the rest of the stack be captured once as a graph.

    The same launch carries the whole stack's L2 warm-up. Doing it with
    prefetches rather than loads is deliberate: a load-based warm-up would make
    this kernel's own duration the cost, while prefetches ride out behind it.
    """
    pid = tl.program_id(0)
    o = pid * BL + tl.arange(0, BL)
    if PFPP > 0:
        for a in tl.static_range(NAR):
            base = tl.load(PTRS + a)
            nl = tl.load(NLINE + a)
            _pf_at(base, tl.minimum(pid * PFPP + tl.arange(0, PFPP),
                                    nl - 1).to(tl.int64) * 128)
    if PDL:
        gdc_wait()
    m = o < NM
    tl.store(DST + OM + o, tl.load(SM + o, mask=m, other=0), mask=m)
    m = o < NZ
    tl.store(DST + OZ + o, tl.load(SZ + o, mask=m, other=0), mask=m)
    m = o < NMM
    tl.store(DST + OMM + o, tl.load(SMM + o, mask=m, other=0), mask=m)
    m = o < NPM
    tl.store(DST + OPM_ + o, tl.load(SPM + o, mask=m, other=0), mask=m)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _warm(A0, A1, A2, A3, A4, A5, SINK,
          N0: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
          N3: tl.constexpr, N4: tl.constexpr, N5: tl.constexpr,
          BL: tl.constexpr, PDL: tl.constexpr):
    """Read every weight arena. **Default off: measured worth 0.0 us.**

    Built to test the obvious explanation for the ~23 us the body costs cold
    against warm -- 5.9 MiB of weights the harness's L2 flush leaves in HBM. It
    is affordable (it runs on the first block's side stream, which has ~16 us of
    slack against that block's pair chain) and it provably executes, but the
    flushed body does not move: 143.26 us with against 143.30 without, and the
    harness reports 164.77 against 164.70. A fire-and-forget
    ``prefetch.global.L2`` of the same arenas was equally worthless.

    So the cold gap is **not** weight latency. The remaining candidate is first
    touch of the ~7 MiB of intermediates this stack writes per forward, which no
    warm-up can avoid, because those lines have to be written anyway. Kept in the
    file because it is the measurement that rules the weights out.

    The sum is stored so the loads are not dead code; ``SINK`` is one fp32 per
    program and is never read. Unused arena slots are passed ``N = 0`` and fold
    away at compile time.
    """
    pid = tl.program_id(0)
    o = pid * BL + tl.arange(0, BL)
    acc = tl.zeros([BL], tl.float32)
    if PDL:
        gdc_wait()
    if N0 > 0:
        acc += tl.load(A0 + o, mask=o < N0, other=0).to(tl.float32)
    if N1 > 0:
        acc += tl.load(A1 + o, mask=o < N1, other=0).to(tl.float32)
    if N2 > 0:
        acc += tl.load(A2 + o, mask=o < N2, other=0).to(tl.float32)
    if N3 > 0:
        acc += tl.load(A3 + o, mask=o < N3, other=0).to(tl.float32)
    if N4 > 0:
        acc += tl.load(A4 + o, mask=o < N4, other=0).to(tl.float32)
    if N5 > 0:
        acc += tl.load(A5 + o, mask=o < N5, other=0).to(tl.float32)
    tl.store(SINK + pid, tl.sum(acc, 0))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _egress(SM, SZ, DST, NM: tl.constexpr, NZ: tl.constexpr,
            BL: tl.constexpr, PDL: tl.constexpr):
    """Copy the plan's final ``m`` and ``z`` into one freshly allocated output.

    The graph's nodes hold constant pointers, so the last kernels write plan-
    owned buffers; this one launch moves both into memory the caller owns, so
    nothing returned is ever aliased between calls. One allocation and two
    contiguous views, not two allocations.
    """
    pid = tl.program_id(0)
    o = pid * BL + tl.arange(0, BL)
    if PDL:
        gdc_wait()
    m = o < NM
    tl.store(DST + o, tl.load(SM + o, mask=m, other=0), mask=m)
    m = o < NZ
    tl.store(DST + NM + o, tl.load(SZ + o, mask=m, other=0), mask=m)


# ---------------------------------------------------------------------------
# Outer product mean, with the block's z residual in the epilogue
# ---------------------------------------------------------------------------
@triton.jit
def _opm_u(M, MASK, OUT, LNW, LNB, W1, W2, WOUT, BOUT,
           S: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
           H: tl.constexpr, Z: tl.constexpr,
           SB: tl.constexpr, CB: tl.constexpr, HB: tl.constexpr,
           PB: tl.constexpr, LN_EPS: tl.constexpr, EPS: tl.constexpr,
           PDL: tl.constexpr, MEARLY: tl.constexpr, RESID: tl.constexpr):
    """``OuterProductMean(m, msa_mask)``, one CTA per ``c_z`` channel.

    With ``RESID`` the block's ``z`` residual is added in place -- read and
    write land on exactly the same addresses in the same program, so no second
    buffer is needed and the add costs no launch. Without it the result is
    stored on its own as ``u``, for a consumer further down the chain to add;
    that is what lets this kernel run a block early on the side stream.

    The reassociation, the ``(residue, seq)`` row order that puts the MSA axis
    inside the pair dot's ``K``, and the tile shape are the frozen L2 winner's;
    what is new here is that ``OUT`` is read as well as written, so the block's
    residual add costs no launch and no round trip. Read and write land on
    exactly the same addresses in the same program, so the update is safe in
    place and needs no second ``z`` buffer.

    ``MEARLY`` hoists the MSA tile and mask loads above ``gdc_wait()``, which in
    a straight-line chain is sound for every block but the first (``m`` is
    eleven grids back, and a PDL chain makes anything written two or more grids
    back final before this grid starts). It is **measured worth 0.0 us** --
    147.33 against 147.36 us on the flushed body -- so the plan simply turns it
    off in the two schedules where the argument stops holding: with a fork the
    load would cross an event join, and with the OPM hoisted onto the side
    stream its ``m`` is one grid back rather than eleven. Kept as a knob only
    because it is the shape of argument that *is* worth 1.5 us elsewhere in this
    chain.

    This kernel covers only the shape the plan validated: one pair block over
    the whole residue axis (so a single layer-norm tile serves both sides of
    the outer product), no padding on any axis, batch 1, mask and affine
    present. Anything else takes the reference path.
    """
    z = tl.program_id(1)
    lp = W1.dtype.element_ty
    od = OUT.dtype.element_ty
    cid = tl.arange(0, CB)
    hid = tl.arange(0, HB)

    # Weights first: all of them are plan-owned and final, so they issue before
    # the wait and their (cold, after the harness's flush) latency overlaps
    # whatever is still draining.
    woff = hid[:, None] * C + cid[None, :]
    w1 = tl.load(W1 + woff)
    w2 = tl.load(W2 + woff)
    wz = tl.load(WOUT + z * (H * H) + hid[:, None] * H + hid[None, :])
    bo = tl.load(BOUT + z).to(tl.float32)
    lnw = tl.load(LNW + cid).to(tl.float32)
    lnb = tl.load(LNB + cid).to(tl.float32)
    if MEARLY:
        x, mk = _opm_load(M, MASK, 0, 0, 0, S, R, C, SB, CB, PB,
                          True, False, False)
    if PDL:
        gdc_wait()
    if not MEARLY:
        x, mk = _opm_load(M, MASK, 0, 0, 0, S, R, C, SB, CB, PB,
                          True, False, False)

    # Y[z] = W1^T W[z]: weights only, so it fills the shadow of the tile load.
    yz = tl.dot(tl.trans(w1), wz).to(lp)
    lhs, mkr = _opm_norm(x, mk, lnw, lnb, C, CB, SB, PB, LN_EPS,
                         True, True, False, lp)
    t2 = tl.reshape(tl.dot(lhs, yz).to(lp), [PB, SB * HB])
    b2 = tl.reshape(tl.dot(lhs, tl.trans(w2)).to(lp), [PB, SB * HB])
    nrm = tl.sum(mkr[:, None, :] * mkr[None, :, :], 2)
    res = (tl.dot(t2, tl.trans(b2)) + bo) / (nrm + EPS)

    ri = tl.arange(0, PB)
    ooff = ri[:, None] * (R * Z) + ri[None, :] * Z + z
    # The reference materializes the OPM result in bf16 and *then* adds z, so
    # round before the add rather than adding the fp32 accumulator: over four
    # blocks the difference is amplified enough to matter against the 1e-2
    # tolerance.
    if RESID:
        tl.store(OUT + ooff,
                 (tl.load(OUT + ooff).to(tl.float32)
                  + res.to(od).to(tl.float32)).to(od))
    else:
        tl.store(OUT + ooff, res.to(od))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# MSA transition, with both MSA-side residuals folded in
# ---------------------------------------------------------------------------
@triton.jit
def _tr_hidden(M, A, LNW, LNB, WT, HOUT,
               C: tl.constexpr, CT: tl.constexpr, BM: tl.constexpr,
               BN: tl.constexpr, EPS: tl.constexpr, DT: tl.constexpr,
               PDL: tl.constexpr):
    """``m1 = m + msa_att``, then ``SiLU(ln(m1) @ Wa) * (ln(m1) @ Wb)``.

    The attention residual is consumed here instead of in the attention's own
    epilogue, which keeps that kernel the frozen L2 winner verbatim and -- more
    usefully -- means ``m1`` never reaches memory at all: the output stage
    recomputes it from the same two operands, both of which are old enough by
    then to be loaded above its wait.

    ``m`` is the previous block's output (or, in the first block, the staging
    kernel's, three grids back), so it is final and loads before the wait;
    ``A`` is the immediately preceding kernel's output and cannot.
    """
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    cb = tl.program_id(1) * BN
    c = tl.arange(0, C)
    n = cb + tl.arange(0, BN)
    wa = tl.load(WT + c[:, None] * (2 * CT) + n[None, :])
    wb = tl.load(WT + c[:, None] * (2 * CT) + (CT + n)[None, :])
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    off = rm[:, None] * C + c[None, :]
    mv = tl.load(M + off).to(tl.float32)
    if PDL:
        gdc_wait()
    m1 = (mv + tl.load(A + off).to(tl.float32)).to(DT)
    xl = _ln(m1.to(tl.float32), lw, lb, C, EPS).to(DT)
    # Both projections round through bf16 before the gate, and the gated value
    # rounds again -- the three points the reference's own stores round at.
    ga = tl.dot(xl, wa).to(DT).to(tl.float32)
    gb = tl.dot(xl, wb).to(DT).to(tl.float32)
    tl.store(HOUT + rm[:, None] * CT + n[None, :],
             ((ga * _sig(ga)).to(DT).to(tl.float32) * gb).to(DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _tr_out(HB_, M, A, WOT, OUT,
            C: tl.constexpr, CT: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, TB: tl.constexpr, DT: tl.constexpr,
            PDL: tl.constexpr):
    """``m2 = m1 + linear_out(hidden)``, closing the block's MSA side.

    ``m`` and the attention output are two grids back, so the residual -- which
    is the only thing here that is not the predecessor's output -- is loaded
    before the wait and recomputed rather than read from a materialized ``m1``.
    """
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    o = rm[:, None] * C + col[None, :]
    m1 = (tl.load(M + o).to(tl.float32)
          + tl.load(A + o).to(tl.float32)).to(DT).to(tl.float32)
    if PDL:
        gdc_wait()
    acc = tl.zeros([BM, BN], tl.float32)
    for t in range(0, CT, TB):
        k = t + tl.arange(0, TB)
        acc += tl.dot(tl.load(HB_ + rm[:, None] * CT + k[None, :]),
                      tl.load(WOT + k[:, None] * C + col[None, :]))
    tl.store(OUT + o, (m1 + acc.to(DT).to(tl.float32)).to(DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _sg_out_u(HB_, MK, WOT, Z, U, OUT,
              C: tl.constexpr, CT: tl.constexpr, BR: tl.constexpr,
              CB: tl.constexpr, TB: tl.constexpr, DT: tl.constexpr,
              PDL: tl.constexpr):
    """The pair stack's last kernel, plus the *next* block's OPM residual.

    Otherwise the frozen L2 ``_sg_out``: ``linear_out`` of the pair transition,
    the mask, and the block's own residual, with the reduction walked in ``TB``
    chunks so Triton pipelines the weight slices. The one addition is ``+ u``,
    the next block's outer product mean, which was computed on the side stream
    while this block's pair chain was running. That is the whole reason the OPM
    is off the critical path: the add it needs costs no launch here.

    ``u`` is rounded into ``z`` only after this block's own residual has been
    rounded, which is the order the reference's two stores impose.

    ``U`` is loaded *after* ``gdc_wait()`` even though it is a load of something
    this grid did not write: it arrives across an event join from the side
    stream, and the wait is what makes the producer's store visible.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.program_id(1) * CB + tl.arange(0, CB)
    o = rows[:, None] * C + col[None, :]
    mk = tl.load(MK + rows)[:, None].to(tl.float32)
    zv = tl.load(Z + o).to(tl.float32)
    if PDL:
        gdc_wait()
    uv = tl.load(U + o).to(tl.float32)
    acc = tl.zeros([BR, CB], tl.float32)
    for t in range(0, CT, TB):
        k = t + tl.arange(0, TB)
        acc += tl.dot(tl.load(HB_ + rows[:, None] * CT + k[None, :]),
                      tl.load(WOT + k[:, None] * C + col[None, :]))
    tl.store(OUT + o, ((zv + acc * mk).to(DT).to(tl.float32) + uv).to(DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _opm_tile_ok(r: int) -> bool:
    """Does one outer-product-mean CTA cover the whole residue axis?

    ``tl.dot`` fixes the pair dot's M and N at >= 16, and the shared
    layer-norm tile -- one tile feeding both sides of the outer product -- only
    exists when a single pair block spans the axis. Below 16 residues neither
    holds and the reference takes over.
    """
    return r >= 16 and _pow2(r)


class _StackPlan:
    """Every kernel of the whole stack, pre-bound, in one CUDA graph.

    Built on the first forward rather than in ``__init__`` so the weights that
    get folded into the arenas are the ones a checkpoint load left behind, and
    invalidated by ``_apply`` / ``_load_from_state_dict`` on the owning module.
    """

    def __init__(self, mod, m, z, msa_mask, pair_mask):
        pf, fork, hoist, warm_on, mearly, warm_last = (
            _flag(n, d) for n, d in _KNOBS)
        dt, dev = m.dtype, m.device
        S, R, CM = m.shape[-3], m.shape[-2], m.shape[-1]
        N, CZ = z.shape[-2], z.shape[-1]
        blocks = mod.blocks
        b0 = blocks[0]
        opm = b0.outer_product_mean
        att = b0.msa_att_row
        tr = b0.msa_transition

        H, ZC = opm.c_hidden, opm.c_z
        SB = triton.next_power_of_2(S)
        CB = triton.next_power_of_2(CM)
        HB = triton.next_power_of_2(H)
        CT = tr.n * tr.c_in
        AH, AD = att.no_heads, att.c_hidden
        AHD = AH * AD
        BCZ = triton.next_power_of_2(CZ)
        BCM = triton.next_power_of_2(CM)
        BHD = max(16, triton.next_power_of_2(AHD))
        BR = max(16, triton.next_power_of_2(R))

        # One shape family only: the plan is the point, and every condition
        # here removes a runtime branch from a kernel. Anything else is the
        # reference's problem, and the reference is correct everywhere.
        if not (m.ndim == 4 and z.ndim == 4 and m.shape[0] == 1
                and m.is_contiguous() and z.is_contiguous()
                and msa_mask.is_contiguous() and pair_mask.is_contiguous()
                and z.dtype is dt and msa_mask.dtype is dt
                and pair_mask.dtype is dt
                and tuple(msa_mask.shape) == tuple(m.shape[:-1])
                and tuple(pair_mask.shape) == tuple(z.shape[:-1])
                and z.shape[0] == 1 and R == N and z.shape[-3] == N
                and CM == tr.c_in and CZ == ZC and CM == opm.c_m
                and SB == S and CB == CM and HB == H and N == 16
                and _pow2(S) and _pow2(CM) and _pow2(CZ) and _pow2(H)
                and _pow2(CT) and CT % _TH_BN == 0 and CM % _TO_BN == 0
                and (S * R) % _TH_BM == 0 and (S * R) % _TO_BM == 0
                and BR == R and BCZ == CZ and BCM == CM and AHD == CM
                and _opm_tile_ok(R) and S * R * CB <= 32768
                and all(b.opm_first for b in blocks)
                and all(b.skip_msa_update == (i == len(blocks) - 1)
                        for i, b in enumerate(blocks))):
            raise _Unsupported
        PB = R
        eps_ln = opm.layer_norm.eps

        DT = tl.bfloat16
        rows = S * R
        keep = []          # storages the raw pointers below must outlive

        # ---- weight arenas ------------------------------------------------
        # Two, because the layer-norm affines the attention kernel wants are
        # fp32 and everything else is the input dtype. Contiguous so the
        # staging kernel can warm each with one flat prefetch range.
        pend_lp, pend_f32 = [], []

        def lp(t):
            t = t.detach().to(dt).contiguous()
            pend_lp.append(t)
            return t

        def lpt(*ws):
            """Pre-transposed ``[K, sum(N_i)]`` view of concatenated rows."""
            return lp(torch.cat(ws, 0).t())

        def f32(p, n):
            buf = torch.zeros((n,), dtype=torch.float32)
            if p is not None:
                buf[:p.shape[0]] = p.detach().float().cpu()
            pend_f32.append(buf)
            return buf

        W = []
        for i, b in enumerate(blocks):
            o = b.outer_product_mean
            w = {"o_lnw": lp(o.layer_norm.weight), "o_lnb": lp(o.layer_norm.bias),
                 "o_w1": lp(o.linear_1.weight), "o_w2": lp(o.linear_2.weight),
                 "o_wo": lp(o.linear_out.weight), "o_bo": lp(o.linear_out.bias)}
            if not b.skip_msa_update:
                a, t = b.msa_att_row, b.msa_transition
                # linear_z transposed with each head's column replicated
                # c_hidden times, so the pair projection lands straight on the
                # (head, channel) axis the value tensor uses -- the frozen L2
                # attention kernel's layout, reproduced here.
                wzd = torch.zeros((BCZ, BHD), dtype=dt)
                wzd[:CZ, :AHD] = a.linear_z.weight.detach().to(dt).cpu(
                ).t().repeat_interleave(AD, dim=1)
                wvgt = torch.zeros((2, BCM, BHD), dtype=dt)
                wvgt[0, :CM, :AHD] = a.linear_v.weight.detach().to(dt).cpu().t()
                wvgt[1, :CM, :AHD] = a.linear_g.weight.detach().to(dt).cpu().t()
                wot = torch.zeros((BHD, BCM), dtype=dt)
                wot[:AHD, :CM] = a.linear_o.weight.detach().to(dt).cpu().t()
                pend_lp += [wzd, wvgt, wot]
                w.update(a_wzd=wzd, a_wvgt=wvgt, a_wot=wot,
                         a_lzw=f32(a.layer_norm_z.weight, BCZ),
                         a_lzb=f32(a.layer_norm_z.bias, BCZ),
                         a_lmw=f32(a.layer_norm_m.weight, BCM),
                         a_lmb=f32(a.layer_norm_m.bias, BCM),
                         t_wab=lpt(t.swiglu.linear_a.weight,
                                   t.swiglu.linear_b.weight),
                         t_wo=lpt(t.linear_out.weight),
                         t_lnw=f32(t.layer_norm.weight, CM),
                         t_lnb=f32(t.layer_norm.bias, CM))
                if (a.linear_z.weight.shape != (AH, CZ)
                        or a.linear_v.weight.shape != (AHD, CM)
                        or a.linear_o.weight.shape != (CM, AHD)
                        or t.swiglu.linear_a.weight.shape != (CT, CM)
                        or t.linear_out.weight.shape != (CM, CT)):
                    raise _Unsupported
            W.append(w)

        def pack(pend, dtype):
            arena = torch.empty(sum(t.numel() for t in pend), dtype=dtype,
                                device=dev)
            off, remap = 0, {}
            for t in pend:
                v = arena[off:off + t.numel()].view(t.shape)
                v.copy_(t)
                remap[id(t)] = v
                off += t.numel()
            return arena, remap

        alp, rlp = pack(pend_lp, dt)
        af32, rf32 = pack(pend_f32, torch.float32)
        W = [{k: rlp.get(id(v), rf32.get(id(v), v)) for k, v in w.items()}
             for w in W]
        keep += [alp, af32]
        self.weights = W

        # ---- input staging arena and the state buffers --------------------
        NM, NZ, NMM, NPM = m.numel(), z.numel(), msa_mask.numel(), pair_mask.numel()
        OM, OZ, OMM = 0, NM, NM + NZ
        OPM_ = OMM + NMM
        ina = torch.empty(OPM_ + NPM, dtype=dt, device=dev)
        self.ina = ina
        sm = ina[OM:OM + NM].view(m.shape)
        sz = ina[OZ:OZ + NZ].view(z.shape)
        smm = ina[OMM:OMM + NMM].view(msa_mask.shape)
        spm = ina[OPM_:OPM_ + NPM].view(pair_mask.shape)
        keep.append(ina)

        def buf(*shape):
            t = torch.empty(shape, dtype=dt, device=dev)
            keep.append(t)
            return t

        # ``m`` needs a fresh destination per block (the block's own OPM and
        # attention still read the old one); ``att``/``hidden`` are dead by the
        # end of the block and are reused across all of them.
        mbuf = [sm] + [buf(rows, CM) for _ in range(len(blocks) - 1)]
        attb = buf(rows, CM)
        hidb = buf(rows, CT)

        K, G, A, NW, ST, PD = [], [], [], [], [], []

        def add(kern, grid, args, warps, stages=1, pdl=_PDL):
            """Queue a kernel. ``stages=None`` leaves Triton's own default,
            which is what the frozen L2 ``_Plan`` passes for the pair kernels --
            the one spliced kernel here has to match it or it is not the same
            kernel any more."""
            K.append(kern)
            G.append(grid)
            A.append(args)
            NW.append(warps)
            ST.append(stages)
            PD.append(pdl)

        # ---- the stack, in launch order ----------------------------------
        # With a fork, the OPM's ``m`` load crosses an event join, so the hoist
        # above its ``gdc_wait()`` is not sound (measured: nondeterministic
        # output). Everything else this plan hoists stays inside one stream.
        hoist = hoist and len(blocks) > 1
        # ...and with ``hoist`` the OPM follows the transition it depends on by
        # a single grid on the side stream, so the hoist is unsound there too.
        mearly = mearly and not fork and not hoist
        R2 = N * N
        ubuf = [None] + [buf(R2, CZ) for _ in range(len(blocks) - 1)]

        idx_opm, idx_side, pair_plans = [], [], []
        zref = sz
        for i, b in enumerate(blocks):
            w = W[i]
            # Block 0's OPM has nothing earlier to hide behind, so it keeps the
            # in-place residual. Every later block's writes ``u`` instead and
            # runs a block early, on the side stream.
            resid = not (hoist and i > 0)
            idx_opm.append(len(K))
            add(_opm_u, (1, CZ, 1),
                [mbuf[i], smm, zref if resid else ubuf[i],
                 w["o_lnw"], w["o_lnb"], w["o_w1"], w["o_w2"], w["o_wo"],
                 w["o_bo"],
                 S, R, CM, H, CZ, SB, CB, HB, PB, eps_ln, opm.eps, _PDL,
                 i > 0 and mearly, resid],
                _OPM_WARPS, _OPM_STAGES)
            idx_side.append(None if b.skip_msa_update else len(K))
            if not b.skip_msa_update:
                a, t = b.msa_att_row, b.msa_transition
                add(_msa_attn, (rows,),
                    [zref, spm, mbuf[i], w["a_wzd"], w["a_wvgt"], w["a_wot"],
                     w["a_lzw"], w["a_lzb"], w["a_lmw"], w["a_lmb"], attb,
                     S, R, CZ, CM, BR, BCZ, BCM, BHD, BR == R, BCZ == CZ,
                     BCM == CM, float(a.inf), a.layer_norm_z.eps,
                     a.layer_norm_m.eps, True, True, True, True, True, _PDL],
                    min(8, max(1, (BR * max(BCZ, BCM, BHD)) // 256)))
                add(_tr_hidden, (rows // _TH_BM, CT // _TH_BN),
                    [mbuf[i], attb, w["t_lnw"], w["t_lnb"], w["t_wab"], hidb,
                     CM, CT, _TH_BM, _TH_BN, t.layer_norm.eps, DT, _PDL],
                    _TH_WARPS, _TH_STAGES)
                add(_tr_out, (rows // _TO_BM, CM // _TO_BN),
                    [hidb, mbuf[i], attb, w["t_wo"], mbuf[i + 1],
                     CM, CT, _TO_BM, _TO_BN, CT, DT, _PDL],
                    _TO_WARPS, _TO_STAGES)
            # The pair stack is the frozen L2 winner's own plan, spliced in:
            # its ``z`` is this block's (already OPM-updated) buffer and its
            # output buffer is the next block's ``z``, so nothing in its ten
            # kernels' argument lists ever changes again.
            pp = _PairPlan(b.pair_stack, zref.view(1, N, N, CZ),
                           spm.view(1, N, N))
            if pp.graph is not None or len(pp.calls) != 10:
                raise _Unsupported
            pair_plans.append(pp)
            zref = pp.scratch["out"].view(1, N, N, CZ)
        self.pair_plans = pair_plans

        # The replacement for each block's last pair kernel: same work, plus
        # the next block's OPM residual. Built off the pair plan's own weight
        # and scratch buffers so there is one copy of them, not two.
        # Arenas, in the order ``_warm`` reads them: this plan's two, then one
        # per pair block.
        arenas = [alp, af32] + [pp.arena for pp in pair_plans]
        idx_warm = None
        if warm_on and idx_side[0] is not None and len(arenas) <= _WARM_NA:
            nel = [t.numel() for t in arenas]
            pad = arenas + [alp] * (_WARM_NA - len(arenas))
            nel += [0] * (_WARM_NA - len(nel))
            sink = torch.empty(-(-max(nel) // _WARM_BL), dtype=torch.float32,
                               device=dev)
            keep.append(sink)
            idx_warm = len(K)
            # Out of the PDL chain entirely, and it is the only kernel here
            # that can be: it reads nothing any other kernel writes, so it has
            # no wait to do. It also must not trigger dependents, or the side
            # chain behind it would be allowed to start before its loads had
            # issued -- which would defeat the whole point.
            add(_warm, (sink.numel(),),
                pad + [sink] + nel + [_WARM_BL, False], 4, 1, False)

        idx_sgu = []
        if hoist:
            pt = blocks[0].pair_stack.pair_transition
            CTZ, SBp, STp = pt.n * pt.c_in, min(_SB, R2), min(_ST, CZ)
            TBp = min(_TK, CTZ)
            for i in range(len(blocks) - 1):
                pp = pair_plans[i]
                idx_sgu.append(len(K))
                add(_sg_out_u, (R2 // SBp, CZ // STp),
                    [pp.scratch["hb"], spm, pp.weights["wot"],
                     pp.scratch["z5"], ubuf[i + 1], pp.scratch["out"],
                     CZ, CTZ, SBp, STp, TBp, DT, _PDL],
                    _WARPS4, None)
        self.keep = keep

        # ---- compile, then bake the addresses in --------------------------
        dix = dev.index if dev.index is not None else _cur_device()
        compiled = [
            k[g](*a, num_warps=w, launch_pdl=p,
                 **({} if s is None else {"num_stages": s}))
            for k, g, a, w, s, p in zip(K, G, A, NW, ST, PD)]
        if not all(_Call.usable(c) for c in compiled):
            raise _Unsupported
        self.compiled = compiled
        raw = [[x.data_ptr() if torch.is_tensor(x) else x for x in a]
               for a in A]
        mine = [_Call(c, g, r, dix) for c, g, r in zip(compiled, G, raw)]

        # ---- the schedule -------------------------------------------------
        # Segments in *enqueue* order, each tagged with the stream it goes on.
        # A side segment is enqueued before the main segment it hides behind and
        # joined after it, which is what makes the captured graph a DAG. With
        # ``hoist`` the join lands one kernel earlier -- before the pair stack's
        # last kernel, which is the one that consumes ``u``.
        segs, labels = [], []
        plab = [getattr(c, "name", "?") for c in pair_plans[0].compiled]

        def emit(side, cs, lab):
            segs.append((side, cs))
            labels.extend(lab)

        if hoist:
            emit(False, [mine[idx_opm[0]]], ["b0.opm"])
        for i, b in enumerate(blocks):
            if not hoist:
                emit(False, [mine[idx_opm[i]]], [f"b{i}.opm"])
            side, slab = [], []
            if i == 0 and idx_warm is not None and not warm_last:
                side.append(mine[idx_warm])
                slab.append("warm")
            if idx_side[i] is not None:
                side += mine[idx_side[i]:idx_side[i] + 3]
                slab += [f"b{i}.{t}" for t in ("att", "tr_h", "tr_o")]
            if hoist and i + 1 < len(blocks):
                side.append(mine[idx_opm[i + 1]])
                slab.append(f"b{i + 1}.opm")
            if i == 0 and idx_warm is not None and warm_last:
                side.append(mine[idx_warm])
                slab.append("warm")
            if side:
                emit(fork, side, slab)
            pc = pair_plans[i].calls
            if hoist and i + 1 < len(blocks):
                emit(False, pc[:-1], [f"b{i}.{t}" for t in plab[:-1]])
                emit(False, [mine[idx_sgu[i]]], [f"b{i}._sg_out_u"])
            else:
                emit(False, pc, [f"b{i}.{t}" for t in plab])
        self.segs = segs
        self.body = [c for _, cs in segs for c in cs]
        # Diagnostics only: ``dev/prefix.py`` truncates the chain by index.
        self.labels = labels

        # ---- staging and output, outside the graph ------------------------
        nline, ptrs = [], []
        for t in [alp, af32] + [p.arena for p in pair_plans]:
            ptrs.append(t.data_ptr())
            nline.append((t.numel() * t.element_size() + 127) // 128)
        pt = torch.tensor(ptrs, dtype=torch.int64, device=dev)
        nl = torch.tensor(nline, dtype=torch.int32, device=dev)
        keep += [pt, nl]
        nmax = max(NM, NZ, NMM, NPM)
        # The program floor only exists to spread the (default off) prefetch;
        # without it, programs past what the copies need would do nothing.
        np_io = max(-(-nmax // _IO_BL), _IO_NP if pf else 1)
        pfpp = triton.next_power_of_2(-(-max(nline) // np_io)) if pf else 0
        ing = _ingest[(np_io,)](
            m, z, msa_mask, pair_mask, ina, pt, nl,
            NM, NZ, NMM, NPM, OM, OZ, OMM, OPM_, len(ptrs), _IO_BL, pfpp,
            _PDL, num_warps=4, launch_pdl=_PDL)
        zout = pair_plans[-1].scratch["out"]
        mout = mbuf[-1]
        flat = torch.empty(NM + NZ, dtype=dt, device=dev)
        egr = _egress[(max(-(-max(NM, NZ) // _IO_BL), 1),)](
            mout, zout, flat, NM, NZ, _IO_BL, _PDL,
            num_warps=4, launch_pdl=_PDL)
        if not (_Call.usable(ing) and _Call.usable(egr)):
            raise _Unsupported
        self.compiled += [ing, egr]
        self.ing = _Call(ing, (np_io,),
                         [m.data_ptr(), z.data_ptr(), msa_mask.data_ptr(),
                          pair_mask.data_ptr(), ina.data_ptr(), pt.data_ptr(),
                          nl.data_ptr(), NM, NZ, NMM, NPM, OM, OZ, OMM, OPM_,
                          len(ptrs), _IO_BL, pfpp, _PDL], dix)
        self.egr = _Call(egr, (max(-(-max(NM, NZ) // _IO_BL), 1),),
                         [mout.data_ptr(), zout.data_ptr(), flat.data_ptr(),
                          NM, NZ, _IO_BL, _PDL], dix)
        self.dix = dix
        self.stream = _raw_stream(dix)
        self.mshape, self.zshape = tuple(m.shape), tuple(z.shape)
        self.mmshape, self.pmshape = tuple(msa_mask.shape), tuple(pair_mask.shape)
        self.dtype, self.device = dt, dev
        self.nm, self.nz = NM, NZ
        self.graph = None
        if _GRAPH:
            self._capture()
            if self.graph is None:
                # Capture left every body argv naming the (dead) capture
                # stream; the straight-line fallback launches them directly.
                st = _raw_stream(dix)
                for c in self.body:
                    c.argv[3] = st

    def _capture(self):
        """Capture the fixed-pointer body -- everything but staging and output.

        Every argument in the body is a constant by now, so the only per-call
        host work left is two eager launches and a replay: ~19 us against ~354
        for the module composition this replaces. PDL survives capture (the
        module docstring's table), so this is not a trade against the
        device-side floor.

        Side segments are captured on a second stream, forked with
        ``wait_stream`` before the main segment they hide behind and joined
        after it, which is what records the DAG. The join deliberately lands
        *before* the last main segment of each block -- that kernel is the one
        that consumes the side stream's ``u``.

        Falls back to the straight-line PDL chain if capture fails, which is
        correct because the enqueue order is itself a valid serial order.
        """
        body = self.body
        try:
            g = torch.cuda.CUDAGraph()
            warm = torch.cuda.Stream()
            side = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                st = _raw_stream(self.dix)
                for c in body:
                    c.argv[3] = st
                for _ in range(3):
                    for c in body:
                        c.run(*c.argv)
            torch.cuda.current_stream().wait_stream(warm)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                cap = torch.cuda.current_stream()
                cst = _raw_stream(self.dix)
                join = False
                for is_side, cs in self.segs:
                    if is_side:
                        side.wait_stream(cap)
                        with torch.cuda.stream(side):
                            sst = _raw_stream(self.dix)
                            for c in cs:
                                c.argv[3] = sst
                                c.run(*c.argv)
                        join = True
                        continue
                    for c in cs:
                        c.argv[3] = cst
                        c.run(*c.argv)
                    if join:
                        cap.wait_stream(side)
                        join = False
                assert not join
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - a graph must never lose the op
            self.graph = None
        else:
            self.graph = g

    def matches(self, m, z, msa_mask, pair_mask) -> bool:
        # Alignment is not optional: Triton specialized every compiled kernel
        # on the 16-byte alignment of the pointers it was built with.
        return (m.dtype is self.dtype and m.device == self.device
                and z.device == self.device and msa_mask.device == self.device
                and pair_mask.device == self.device
                and z.dtype is self.dtype and msa_mask.dtype is self.dtype
                and pair_mask.dtype is self.dtype
                and tuple(m.shape) == self.mshape
                and tuple(z.shape) == self.zshape
                and tuple(msa_mask.shape) == self.mmshape
                and tuple(pair_mask.shape) == self.pmshape
                and m.is_contiguous() and z.is_contiguous()
                and msa_mask.is_contiguous() and pair_mask.is_contiguous()
                and not ((m.data_ptr() | z.data_ptr() | msa_mask.data_ptr()
                          | pair_mask.data_ptr()) & 15))

    def run(self, m, z, msa_mask, pair_mask):
        ing, egr = self.ing, self.egr
        av = ing.argv
        av[_ARGV0] = m.data_ptr()
        av[_ARGV0 + 1] = z.data_ptr()
        av[_ARGV0 + 2] = msa_mask.data_ptr()
        av[_ARGV0 + 3] = pair_mask.data_ptr()
        out = torch.empty(self.nm + self.nz, dtype=self.dtype,
                          device=self.device)
        op = out.data_ptr()
        if op & 15:
            return None
        egr.argv[_ARGV0 + 2] = op
        stream = _raw_stream(self.dix)
        g = self.graph
        if stream != self.stream:
            self.stream = stream
            ing.argv[3] = stream
            egr.argv[3] = stream
            if g is None:
                for c in self.body:
                    c.argv[3] = stream
        ing.run(*ing.argv)
        if g is None:
            for c in self.body:
                c.run(*c.argv)
        else:
            g.replay()
        egr.run(*egr.argv)
        return (out[:self.nm].view(self.mshape),
                out[self.nm:].view(self.zshape))


class MSAModuleBlock(nn.Module):
    """Single block of AF3 Algorithm 8.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        transition_n: Transition layer scale
        msa_dropout: MSA dropout rate
        pair_dropout: Pair dropout rate
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
        eps: float = 1e-3,
        last_block: bool = False,
    ):
        super().__init__()
        self.opm_first = opm_first
        self.skip_msa_update = last_block and opm_first

        if not self.skip_msa_update:
            self.msa_att_row = MSARowAttentionWithPairBias(
                c_m=c_m, c_z=c_z,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                inf=inf,
            )

            self.msa_transition = SwiGLUTransition(c_in=c_m, n=transition_n)

        self.outer_product_mean = OuterProductMean(
            c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm, eps=eps,
        )

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        if self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        if not self.skip_msa_update:
            m = m + self.msa_att_row(m, z=z, mask=pair_mask)
            m = m + self.msa_transition(m)

        if not self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        return m, z


class MSAModuleStack(nn.Module):
    """AF3 Algorithm 8: MSA module stack.

    Args:
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        c_hidden_msa_att: Hidden dim in MSA attention
        c_hidden_opm: Hidden dim in outer product mean
        c_hidden_mul: Hidden dim in triangle multiplication
        c_hidden_pair_att: Hidden dim in triangle attention
        no_heads_msa: Heads for MSA attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of MSA module blocks
        transition_n: Transition scale
        opm_first: Whether OPM comes before MSA attention
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_m=c_m, c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                inf=inf,
                eps=eps,
                last_block=(i == no_blocks - 1),
            )
            for i in range(no_blocks)
        ])
        self._plan: _StackPlan | None = None
        self._no_fuse = False
        self._rejected: set = set()

    # The plan folds every parameter into its arenas, so any wholesale change
    # to them invalidates it. Both hooks fire before the first forward in
    # normal use (``.to(device)``, then a checkpoint load), which is exactly
    # why the plan is built lazily.
    def _apply(self, *args, **kwargs):
        self._plan = None
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._plan = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _reference(self, m, z, msa_mask, pair_mask):
        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)
        return m, z

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        plan = self._plan
        if plan is not None and plan.matches(m, z, msa_mask, pair_mask):
            out = plan.run(m, z, msa_mask, pair_mask)
            # ``run`` declines only on a misaligned output allocation, which the
            # caching allocator does not produce; take the reference rather than
            # rebuilding a plan that would decline again.
            return (out if out is not None
                    else self._reference(m, z, msa_mask, pair_mask))

        if (not self._no_fuse and not kwargs and m.is_cuda
                and m.dtype in _FAST_DTYPES and not torch.is_grad_enabled()
                and z is not None and msa_mask is not None
                and pair_mask is not None):
            key = (m.dtype, m.device, tuple(m.shape), tuple(z.shape),
                   tuple(msa_mask.shape), tuple(pair_mask.shape))
            if key not in self._rejected:
                try:
                    plan = _StackPlan(self, m, z, msa_mask, pair_mask)
                except _Unsupported:
                    # Remembered per shape rather than latched for the module,
                    # so one odd shape does not cost every later one the plan.
                    self._rejected.add(key)
                except Exception:  # noqa: BLE001 - never lose the operator
                    self._no_fuse = True
                else:
                    self._plan = plan
                    if plan.matches(m, z, msa_mask, pair_mask):
                        out = plan.run(m, z, msa_mask, pair_mask)
                        if out is not None:
                            return out

        return self._reference(m, z, msa_mask, pair_mask)
