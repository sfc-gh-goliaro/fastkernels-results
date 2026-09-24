"""TP-aware embedding and LM head (L2 operators), for B200 / sm_100.

The baseline's two classes collapse, under ``fastkernels bench``, into three regimes. The
measurements in this workspace say what each one wants, and in all three the answer turned
out to be "what torch already does" -- which took rather more work to establish than to
state, and the reasoning is recorded below so it is checkable rather than merely asserted:

* ``ParallelLMHead`` at M = 1, 60, 64 reads the whole vocabulary matrix (622-756 MB of
  bf16) exactly once and reaches only 64-69 % of the 7.672 TB/s peak implied by this
  B200's memory clock (3.996 GHz) and bus width (7680 bits), at one CTA per SM because
  cuBLAS's 384-wide N tile needs 184-224 KB of dynamic shared memory. This is the one
  regime with any headroom, and a weight-streaming kernel was written for it, is correct,
  and beats cuBLAS on a probe -- but not in the harness, by 2 us either way across three
  runs. Delegated, with the kernel and its measurements kept in place; see the table below.
* ``ParallelLMHead`` at M = 494 and M = 16384 is tensor-pipe-bound -- 93.5 % and 99.95 %
  of ``sm__pipe_tensor_cycles_active``, with 16.1 and 513.7 waves per SM. Neither the
  wave-quantisation defect nor the bandwidth gap that motivated this file is present, and
  at M = 494 the arithmetic alone needs ~227 us at the sustained pipe rate against a
  single-read memory floor near 91 us. Delegated, and not as a concession.
* ``VocabParallelEmbedding`` is a row gather sitting 2-6 us above a 7.14 us harness
  measurement floor on four of five shapes. A no-regression target -- but one that needed
  an active decision, and it is the one substantive change in this file: the hot path calls
  ATen rather than the frozen L1 CUDA gather, which the probe puts at 0.83x on n = 1 and
  0.95x on n = 16384. In-harness the decision reads between 1.00x and 1.04x at n = 16384
  depending on how contended the machine is -- 1.04x when busy, 1.003x when quiet -- and
  never a regression. Small, but it is the difference between choosing the kernel and
  inheriting it, and the baseline's own route has no per-shape table with which to decline.

## What was measured, and what closed each direction

Everything below is from this workspace, in a harness-faithful loop (252 MB L2-flush
buffer zeroed before each iteration, CUDA-event median of 50 after 10 warmups, ``x`` in a
256-byte shifting pool because it is a forward argument, the weight unpooled because it is
module state). cuBLAS was re-timed in the same process as every challenger, because
absolute latencies on this machine move with contention while ratios within a run do not.

**A streaming-read probe** (``tools/probe_stream_ceiling.py``). A kernel that only reads the
weight and accumulates in fp32 -- no shared-memory staging, no tensor cores, one fp32 stored
per CTA, and a self-check against ``W.float().sum()`` so it cannot be skipping work. Best of
42 configurations, per shape and against cuBLAS re-timed in the same process: 5.377 vs
5.194 TB/s at M = 1, 5.474 vs 5.189 at M = 60, 5.581 vs 5.224 at M = 64, i.e. 1.036x /
1.055x / 1.068x.

As percentages of that clock-derived peak: 70.1 %, 71.3 %, 72.7 %. **All three are below the
plan's ~6.0 TB/s abort threshold**, so the prescribed outcome is parity -- which is what shipped.
The kernel work recorded below ran *after* the threshold was already missed: it is exploration
past an uncleared gate rather than a continuation the gate authorised, and it produced no
admission.

What this probe is, stated carefully because it was read too strongly once: the best rate
achieved by *this one* reduction kernel, which carries a loop-carried fp32 accumulator and a
final reduction. It is neither a ceiling on the memory system nor a lower bound on it -- it
bounds nothing about the hardware. The streaming GEMM below reaches 5.68 TB/s at M = 1, which
shows only that this kernel was not the best available kernel. What the probe is good for is
sizing the prize, and on that it was right: the prize is a few per cent, not the 31-36 % that
"64-69 % of peak" suggests, because 7.672 TB/s is a clock-derived interface peak rather than
anything this machine hands out to a streaming read.

The two mechanisms the draft proposed, reconciled separately and never multiplied. **The
partial-wave tail** is worth ``1/(1 - (1 - 100/148)/3)`` = 1.121x at M = 60: cuBLAS launches
grid 396 on 148 SMs at one CTA per SM, so three waves whose last holds 100 blocks, and the
recoverable idle fraction is 0.1081. That is arithmetic under the naive synchronous-wave model,
not a measurement. **Per-SM streaming efficiency** is worth 1.055x: 71.3 % measured against
cuBLAS's 67.6 % at M = 60. The plausible combined speedup is bounded at about **1.07x**, the
largest composite any configuration reached. Multiplying the two -- 1.121 x 1.055 = 1.183x --
double-counts, because the 1.055x figure was itself measured at grid 2368 whose own tail is
already amortised, so it is a composite and not a pure efficiency term.

The same sweep says something about *where* the few per cent comes from, and it is not
mainly wave alignment. A grid of exactly 148 CTAs has no tail by construction and is the
*worst* configuration measured, at 1.067 TB/s, 4.9x slower than cuBLAS: on this machine
bandwidth follows concurrency. That observation does not by itself refute cuBLAS's
partial-wave tail, and it should not be quoted as if it did -- the probe uses no shared
memory and 4 warps per CTA, so its own wave is 148 x (resident CTAs per SM) and grid = 148
underfills residency, whereas cuBLAS is genuinely capped at one CTA per SM. The two are not
like for like. What the streaming tiles do differently is directly measurable, and it was
measured: dropping from cuBLAS's 188-230 KB of shared memory to 61-74 KB does put three CTAs on
an SM where cuBLAS gets one (``launch__occupancy_limit_shared_mem`` 3 against 1). The mechanism
the design was built on was delivered. It bought nothing -- see ``profile/lmhead_stream_dormant/``
and the paragraph below.

**The streaming GEMM** (``tools/probe_stream_gemm_full.py``, and the two earlier probes it
supersedes). N-only parallelisation, both operands through TMA, fp32 accumulate, ``M`` as a
compile-time constant, swept over BM x BN x BK x num_stages x num_warps x warp_specialize
with cuBLAS timed *interleaved* with each candidate in one loop -- flush, time cuBLAS,
flush, time the candidate, alternating -- because baselines on this machine drift more than
10 us between processes, which is more than the whole admission margin. Best per shape:

    M       cuBLAS   candidate  ratio   margin   tile
    1       119.8 us   109.5 us  1.094x  +10.3 us  BM=16 BN=64  BK=128 ns=4 nw=4
    60      121.9 us   117.7 us  1.035x   +4.1 us  BM=64 BN=128 BK=64  ns=3 nw=4
    64      144.4 us   138.1 us  1.045x   +6.2 us  BM=64 BN=128 BK=64  ns=3 nw=4

Two earlier, coarser sweeps had concluded the opposite -- 0.934-0.971x -- for two reasons
worth recording so the conclusion is not re-derived wrongly either way. They shared a
``BM = 64`` tile across all three shapes, which wastes 4x the accumulator and shared memory
at M = 1; and they timed cuBLAS in a separate pass, so process-to-process drift of more than
10 us swamped a 4 us signal.

And then the harness disagreed with all of them: in-harness the same three tiles read
1.018x / 0.983x / 1.014x, reproducibly. The table above is a probe result and the table on
``_MEASURED_FAST`` is the score; where they disagree the score wins, and here it does
disagree, by 2-8 us in the candidate's favour on the probe. What differs is not obvious from
either loop's source -- both flush L2, both use CUDA events and a median, both pool ``x`` --
and it was not worth chasing further once the sign of the answer was settled three times.
The transferable part is the discipline, not the number: a probe chooses tiles, and only the
harness admits them.

ncu then answered the remaining question -- whether the design's mechanism worked at all -- and
the answer closes the direction rather than leaving it open
(``profile/lmhead_stream_dormant/REPORT.md``, six ``--set full`` captures taken in one session).
The mechanism worked: three CTAs per SM against cuBLAS's one. It made things *worse*. The
achieved read percentage falls in all three cases -- 62.37 vs 68.11 at M = 1, 64.71 vs 66.06 at
M = 60, 66.87 vs 68.22 at M = 64 -- on identical traffic, both reading the weight exactly once
to within 0.1 MB. Three CTAs at 36-79 registers move the weight less well than one CTA at 255
registers with 217 KB of staging, which means cuBLAS's shared-memory footprint is not the defect
the draft read it as: it is how a single CTA keeps enough bytes in flight, and the occupancy it
costs is a price paid deliberately. Neither is the kernel secretly compute-bound (tensor pipe
25.6-56.8 %), and at M = 60 and M = 64 the two have the same waves per SM to two decimals while
the streaming kernel is still slower, so within these captures the partial-wave tail is not what
separates them either. Scoped to what was actually profiled: none of the configurations measured
here has 4 us in it, and the mechanism the design rested on was shown not to pay. That is a
statement about these configurations, not a proof that no arrangement of a streaming GEMM could
ever win. A second, deeper profiling run with PM sampling and source counters is in
``profile/lmhead_stream_pm/``, which is where the tail claim is tested against a time series
rather than against static wave counts.

On the emitted PTX (``profile/lmhead_stream_gemm/full_sweep.json``): the ``BM = 64`` tiles
lower to ``tcgen05.mma.cta_group`` with a TMEM-backed accumulator, and the ``BM = 16`` tile
at M = 1 lowers to ``mma.sync.aligned.m16n8k16`` instead. That is still a tensor-core MMA
and not an FMA fallback, but it is not the tcgen05 path, and it should not be described as
one. It does not matter here: at M = 1 the kernel is entirely DRAM-bound -- 5.68 TB/s of
weight against 39.8 GFLOP of arithmetic -- so the MMA form is not what sets the time.

**cuBLAS layout and API shopping**, the cheapest experiment in the sequence. Each alternative was
put behind an exact ``(M, K, N)`` entry in ``_MEASURED_FAST_API`` and scored by its own
``python validate.py`` run -- not by a probe, which cannot admit anything -- with the outputs
archived as ``bench_results/runs/api_*.txt`` and the launch grids in
``profile/lmhead_api_variants/``. In-harness, per shape:

    mode      M=1     M=60    M=64    M=494   M=16384   verdict
    mm       1.01x    1.01x   1.01x   1.00x   1.00x     best +2.0 us, under the 4 us gate
    addmm    0.93x    0.88x   0.89x   0.74x   0.87x     bias epilogue, zeros-fill, elementwise
    swap_t   0.98x    0.44x   0.75x   0.17x   0.58x     see below
    mv       0.98x      -       -       -       -       M = 1 only

``mm`` and ``mv`` reach the *identical kernel at the identical grid* as ``F.linear`` -- 396 at
M = 1 and 60, 428 at M = 64, 2376 at M = 494, 76032 at M = 16384 -- so they are the same dispatch
under another name and can only lose the wrapper's cost, which they do by less than the gate.

The role-swapped ``matmul(W, x.T)`` is where the interesting failures are. At M = 60 and M = 494
it falls off the sm100 path onto an Ampere-generation ``cutlass_80`` kernel: grid 594 at 350.4 us
against ``F.linear``'s grid 396 at 120.1 us, and grid 2374 at 1615.8 us against 2376 at 285.3 us.
At M = 64 it instead reaches a *different and marginally faster* variant,
``nvjet_sm100_tst_64x384_64x3_1x2_h_bz_TNT`` at grid 428, **141.2 us against F.linear's
143.2 us** -- and the ``.T.contiguous()`` that returning ``[M, N]`` requires more than gives that
back. Admitting a role swap on the product alone would be measuring the wrong thing, so the
0.75x above is timed including the copy, as it must be.

**The gather** (``tools/probe_l1_gather.py``, ``tools/probe_vector_gather.py``). ATen
against the frozen L1 CUDA gather on the five scored shapes, all five bit-exact:

    n      dim   ATen      frozen L1   ratio   floor-implied ceiling
    1      4096   9.25 us   11.15 us    0.83x   1.30x
    60     2048  11.26 us   11.30 us    1.00x   1.58x
    470    4096  11.26 us   11.30 us    1.00x   1.58x
    1000   4096  13.31 us   13.31 us    1.00x   1.86x
    16384  4096  35.95 us   37.89 us    0.95x   1.11x at torch's own 5.3 TB/s copy rate

Four of the five are floor-dominated, so there is nothing to win. The one shape with content
is n = 16384: 134 MB of stores, which ATen writes at 4.65 TB/s net of the floor. The persistent
16-byte-vectorised store path below was routed to by an exact ``(16384, 4096)`` entry and scored
by three ``python validate.py`` runs (``bench_results/runs/gather_vec_run*.txt``): **33.8 us
against ATen's 35.8 us, 1.059x, +2.0 us, identical in all three runs and bit-exact.** It is
genuinely faster, and it does not clear the gate -- +2.0 us is one ~2.04 us quantisation step
against a 4 us margin, and the margin is not moved to fit the result. Consistent with the plan's
own bound of 1.11-1.22x for 134 MB of stores at torch's copy rate. So ATen keeps every shape --
because the win is unadmittable, not because ATen won.

## Why the host side is cheap, and where that permission ends

Timing zeroes the 252 MB L2-flush buffer before ``start.record()``, which is enqueued ahead
of the measured window and buys the host a large head start: the same gather called at
three different Python depths reads 13.31 us in all three cases
(``tools/probe_floor.py``). So a table lookup costs nothing and flattening Python frames
is worth nothing. What that does *not* license is compilation, autotuning, or a thread
inside the timed region. Compilation happens at import and only for the entries a table
actually enables, so with all three tables empty nothing here compiles at all -- ``_init_kernel``
and ``_init_vector_gather`` both return before building. The sweeps that would choose those
configurations ran offline, in ``tools/``, and only a frozen winner would reach this file.

## Contract notes

These are why the file is longer than the work it does. The L4 models and the engine reach
much deeper than ``forward``:

* ``embedding_op.emb.weight`` is the state_dict key the harness shares weights through,
  with ``load_state_dict(..., strict=False)`` inside a bare ``except: pass``. A renamed key
  is ignored silently, leaving the candidate on unrelated random weights, so the frozen
  ``Embedding`` submodule stays and keeps owning the parameter.
* The weight is reassigned *after* construction -- four L4 models retie it and the harness
  mutates ``p.data`` -- so nothing weight-derived may be cached in ``__init__``, and
  ``forward`` reads it from the module every call.
* Neither class may grow a ``.weight`` attribute. Three weight-tying branches in L4
  currently raise ``AttributeError`` and are dead; adding ``.weight`` would silently
  activate them.
* ``bias`` is accepted and ignored, as in the baseline. A real bias parameter would change
  the state_dict, and the harness would overwrite it anyway -- it rewrites any float
  parameter whose absolute maximum falls below 1e-6.
* Every ``__init__`` parameter is splatted by name by the bench, including the inert
  ``params_dtype``, ``org_num_embeddings``, and ``padding_size``, so all the names must
  survive. ``forward``'s parameter must stay named ``x``.
* ``linear_op`` is called directly from outside (``L4/mamba.py`` does
  ``self.lm_head.linear_op(hidden, weight)``), so it stays an attribute module with the
  ``(input, weight, bias=None)`` signature. It is the frozen L1 ``Matmul``, which means
  this file inherits whatever L1 has been measured to be best; its table is empty today,
  so that is ``F.linear``.
* ``project`` and ``linear_op`` are both reached inside CUDA graph capture regions. Nothing
  here builds a per-call descriptor or reads host state that a capture could stale, so both
  capture and replay correctly; ``tools/test_contract.py`` captures ``project`` and replays
  it against fresh input contents to check that rather than assume it.
* The tp > 1, prefill, and mixed-batch branches are unreachable under the bench
  (``parallel_embedding`` is not in ``_DISTRIBUTED_OPS``, so ``_tp_size()`` is 1) but are
  reached from thirteen L4 models and several engine paths, so they are preserved verbatim
  rather than simplified away.
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from ....infra.context import get_context
from ....infra.tp import _tp_size, _tp_rank
from ..L1.linear import Matmul
from ..L1.embedding import Embedding
from ..L1.allreduce import AllReduce

# ---------------------------------------------------------------------------
# The weight-streaming projection: tile table, kernel, guard, dispatch.
# ---------------------------------------------------------------------------


class Tile:
    """One frozen streaming configuration, plus the in-harness margin that admitted it."""

    __slots__ = ("bm", "bn", "bk", "num_warps", "num_stages", "margin_us")

    def __init__(self, bm, bn, bk, num_warps, num_stages, margin_us=0.0):
        self.bm, self.bn, self.bk = bm, bn, bk
        self.num_warps, self.num_stages = num_warps, num_stages
        # Worst absolute latency win, in microseconds, over the repeated in-harness runs
        # that admitted this shape. Not a ratio: the admission gate is absolute, because
        # the harness's readings quantise in ~2.04 us steps regardless of case size.
        self.margin_us = margin_us

    def __repr__(self):
        return (f"Tile({self.bm}, {self.bn}, {self.bk}, num_warps={self.num_warps}, "
                f"num_stages={self.num_stages}, margin_us={self.margin_us:.1f})")


# Keyed ``(M, K, N)`` -- the exact problem, never a neighbourhood of it. An entry earns its
# place one way only: at least three independent ``validate.py`` runs whose *worst* reading
# still beats the baseline by 4 us, which is about twice the ~2.04 us step the harness's
# readings quantise in on these 120-160 us cases.
#
# **Empty, and this is the interesting result in the file.** The kernel below is correct and
# is genuinely faster than cuBLAS when timed by a probe. It is not faster in the harness, and
# the harness is what scores. Three runs, and the three shapes agree to within 0.1 us
# run-to-run (``tools/admit.py`` over ``bench_results/runs/kernel_run{1,2,3}.txt``):
#
#     shape                  candidate   baseline   ratio    worst margin
#     (1, 2048, 151936)       113.7 us   115.7 us   1.018x       +2.0 us
#     (60, 2048, 151936)      121.9 us   119.8 us   0.983x       -2.1 us
#     (64, 2304, 163840)      142.3 us   144.4 us   1.014x       +2.0 us
#
# Two of the three win by exactly one quantisation step, which is not a measurement; the
# third loses by one, reproducibly, in all three runs. A ``tools/`` probe of the same three
# tiles -- same kernel, same tile parameters, cuBLAS interleaved with the candidate inside
# one timing loop so drift is common-mode -- reported +10.3 / +4.1 / +6.2 us instead. Both
# numbers are real; they measure different things, and only one of them is the score. The
# gap between them is the reason admission is defined in-harness rather than by probe, and
# it is worth more than the kernel would have been.
#
# The tiles that the probe chose, kept here so the negative is reproducible by putting them
# back: ``(1, 2048, 151936) -> Tile(16, 64, 128, 4, 4)``,
# ``(60, 2048, 151936) -> Tile(64, 128, 64, 4, 3)``,
# ``(64, 2304, 163840) -> Tile(64, 128, 64, 4, 3)``. ``BM = 16`` for M = 1 matters -- a shared
# ``BM = 64`` tile holds accumulators for 64 rows when one exists, and dropping to the
# smallest ``tl.dot`` M cuts shared memory to 10 KB, i.e. 22 blocks per SM against cuBLAS's
# 1. ``warp_specialize`` is off in all three, and that is a correctness requirement rather
# than a tuning choice: see the note on the kernel below.
#
# M = 494 and M = 16384 are absent for a different and firmer reason -- they are
# tensor-pipe-bound at 93.5 % and 99.95 %, so there was never anything here for them.
_MEASURED_FAST: dict[tuple[int, int, int], Tile] = {}

# The kernel's own domain, kept independent of the table so a bad table entry cannot route
# a problem the kernel is not written for.
_M_MAX = 64          # above this the problem is tensor-pipe-bound, not read-bound
_K_MIN = 256         # short-K cases are already at the harness floor
_ALIGN_BYTES = 16    # TMA wants a 16-byte-aligned base; the pool hands out 256 B slots

# Fast-path entries per key. Plain ints incremented on the host: no threads, no device sync,
# nothing the integrity guards watch. This is what distinguishes "the fast path ran and
# tied" from "the fast path never ran".
_FASTPATH_HITS: dict[tuple[int, int, int], int] = {}

_KERNEL = None                                  # the compiled launcher, or None
_KERNEL_STATUS = "disabled:not-initialised"


def _build_kernel():
    """Compile the bf16 weight-streaming GEMM and return a launcher, or raise.

    Called at import rather than on first use: it keeps compilation out of the timed region
    and out of the three correctness forwards, and it happens while the worker is still
    producing output, clear of the watchdog that only watches the log's mtime.
    """
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def _stream_nt_gemm(x_desc, w_desc, C, N, K, M_CONST: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        NS: tl.constexpr):
        """C[M,N] = x[M,K] @ W[N,K].T, fp32 accumulate, parallelised over N only.

        Both operands come through TMA, and that is a correctness requirement rather than a
        performance preference. A plain masked ``tl.load`` for ``x`` inside a
        ``tl.range(..., warp_specialize=True)`` loop is silently wrong in 9 of 15 tile
        configurations on this toolchain -- match ratios down to 0.0154, with plausible
        timings -- because warp specialisation schedules the specialised region assuming its
        loads are async bulk-tensor operations. Two things follow, and both are load-bearing:
        every load in the loop is a descriptor load, and ``warp_specialize`` stays off in
        every admitted tile. ``profile/lmhead_stream_gemm/correctness.json`` has the sweep.

        Using a descriptor for ``x`` is what also lets the row mask go: a TMA box whose row
        extent exceeds the tensor's rows is legal here and zero-fills beyond the extent
        (``profile/lmhead_stream_ptx/preconditions.json``), so a ``BM = 16`` box over a
        1-row ``x`` reads the row that exists and zeroes the rest, contributing nothing.
        ``M_CONST`` is the real row count as a compile-time constant, so the store mask
        folds away instead of being evaluated per call.

        No split-K: the weight is already read exactly once, and a reduction pass over an
        18-150 MB partial buffer would add more DRAM traffic than the latency it saves.
        """
        off_n = tl.program_id(0) * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in tl.range(0, K, BK, num_stages=NS):
            acc = tl.dot(x_desc.load([0, k]), tl.trans(w_desc.load([off_n, k])), acc)
        rm = tl.arange(0, BM)
        rn = off_n + tl.arange(0, BN)
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty),
                 mask=(rm < M_CONST)[:, None] & (rn < N)[None, :])

    def launch(x2: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
               tile: Tile) -> torch.Tensor:
        # Descriptors are built per call, not cached: the harness's shifting pool hands out a
        # fresh data_ptr for every iteration, so a cached descriptor would address the
        # previous iteration's buffer. Building them on the host rather than with
        # tl.make_tensor_descriptor avoids a tensormap_create + fenceproxy_acquire pair in
        # every CTA, which measured ~2 us of pure per-CTA overhead in the frozen L1 study.
        # The host pays for this outside the measured window -- the harness zeroes a 252 MB
        # L2-flush buffer before start.record(), which buys a large head start.
        M, K = x2.shape
        N = weight.shape[0]
        _stream_nt_gemm[(triton.cdiv(N, tile.bn),)](
            TensorDescriptor.from_tensor(x2, [tile.bm, tile.bk]),
            TensorDescriptor.from_tensor(weight, [tile.bn, tile.bk]),
            out, N, K, M_CONST=M, BM=tile.bm, BN=tile.bn, BK=tile.bk,
            NS=tile.num_stages, num_warps=tile.num_warps)
        return out

    return launch


def _check_tile(tile: Tile, M: int) -> None:
    """Reject a malformed table entry loudly, once, instead of per call.

    TMA block shapes and ``tl.arange(0, BN)`` both need positive powers of two; ``tl.dot``
    needs ``BM >= 16``; and a ``BM`` below the row count would silently drop rows. ``_plan``
    deliberately does not re-check the tile it looks up -- it runs on every dispatch and
    stays to integer comparisons.
    """
    for name, value in (("bm", tile.bm), ("bn", tile.bn), ("bk", tile.bk)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"tile.{name}={value} is not a positive power of two")
    if tile.bm < 16:
        raise ValueError(f"{tile!r} has bm < 16, which tl.dot cannot lower")
    if tile.bm < M:
        raise ValueError(f"{tile!r} has bm < M={M}, which would drop rows")
    if tile.num_warps <= 0 or tile.num_stages <= 0:
        raise ValueError(f"{tile!r} has a non-positive num_warps/num_stages")


def _compile_for(table) -> None:
    """Force compilation of every admitted tile on dummy operands, at import.

    Triton compiles on first launch, so each admitted configuration is launched once here
    rather than during the harness's first correctness forward. Signatures are deduplicated
    because compilation is charged against a 1200 s per-operator import budget. With the table
    empty there is nothing to compile and this is never reached.
    """
    if _KERNEL is None:
        return
    seen = set()
    for (M, K, N), tile in table.items():
        _check_tile(tile, M)
        sig = (M, tile.bm, tile.bn, tile.bk, tile.num_warps, tile.num_stages)
        if sig in seen:
            continue
        seen.add(sig)
        x2 = torch.zeros((M, K), device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros((N, K), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        _KERNEL(x2, weight, out, tile)
        del x2, weight, out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _init_kernel() -> None:
    """Resolve the kernel once, at import. Any failure degrades to torch."""
    global _KERNEL, _KERNEL_STATUS
    if not _MEASURED_FAST:
        _KERNEL_STATUS = "disabled:no-enabled-shapes"
        return
    if not torch.cuda.is_available():
        _KERNEL_STATUS = "disabled:no-cuda-device"
        return
    try:
        _KERNEL = _build_kernel()
        _compile_for(_MEASURED_FAST)
        _KERNEL_STATUS = "built"
    except Exception as exc:  # compiler, driver, or architecture rejected the kernel
        _KERNEL = None
        _KERNEL_STATUS = f"failed:{type(exc).__name__}: {exc}"
        # One line, at import, on stderr: the bench worker sends this to the per-operator
        # log, so a swallowed build failure stays visible instead of hiding behind a silent
        # 1.00x.
        print(f"[candidate L2/parallel_embedding] streaming kernel unavailable, "
              f"delegating to torch: {_KERNEL_STATUS}", file=sys.stderr, flush=True)


_init_kernel()


def _plan(input: torch.Tensor, weight: torch.Tensor, bias) -> Tile | None:
    """Return the tile to run this projection with, or None to delegate.

    Pure and cheap: integer and attribute checks only, no allocation and no CUDA sync. Every
    predicate guards something the kernel relies on, and the exact shape must additionally
    be in the measured table -- which is the last gate, not the first.
    """
    if _KERNEL is None:
        return None
    # The kernel has no bias epilogue, and the baseline never builds a bias anyway.
    if bias is not None:
        return None
    # An inference-only kernel: it builds no graph, so grad mode delegates.
    if torch.is_grad_enabled():
        return None
    # bf16 only. Everything else is unmeasured here.
    if input.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return None
    # Lazy negation and conjugation are *metadata*: the bit says "the logical value is the
    # negation of what is in storage", and torch resolves it inside the operators that read
    # the tensor. A TMA descriptor is built from a raw pointer and reads the physical bytes,
    # so it would silently compute ``raw @ w.T`` where ``F.linear`` computes ``-raw @ w.T``.
    # Verified: a ``torch._neg_view`` input matches the physical storage at 1.0000 and the
    # logical answer at 0.0043. Nothing re-derived from the pointer or the strides can reveal
    # this, so it has to be asked about directly.
    if input.is_neg() or weight.is_neg() or input.is_conj() or weight.is_conj():
        return None
    # One kernel, one device: every operand has to live where the launch goes -- and Triton
    # launches on the *current* device and its stream, not on the operand's. Agreement between
    # the operands is therefore not sufficient; both must also be the current device, or the
    # launch goes somewhere else. Unreachable under the bench, which gives each worker one
    # device, but reachable from the multi-GPU L4 paths.
    if not input.is_cuda or weight.device != input.device:
        return None
    if input.device.index != torch.cuda.current_device():
        return None
    if weight.dim() != 2 or input.dim() < 1:
        return None
    K = weight.shape[1]
    N = weight.shape[0]
    if input.shape[-1] != K:
        return None
    # Bounds first, so nothing below divides by a zero K -- F.linear accepts K == 0 and must
    # keep handling it. K % 8 and N % 8 are what TMA's 16-byte-aligned global strides need
    # for bf16.
    if K < _K_MIN or N <= 0 or K % 8 or N % 8:
        return None
    # A hidden .contiguous() would cost an extra launch, worth more than the kernel can win
    # back; the descriptors also require a unit last stride and a packed row stride.
    if not input.is_contiguous() or weight.stride(-1) != 1 or weight.stride(0) != K:
        return None
    M = input.numel() // K
    if M <= 0 or M > _M_MAX:
        return None
    # TMA needs 16-byte-aligned bases. The shifting pool hands out 256-byte-aligned slots,
    # but that is a property of the pool, not a guarantee.
    if input.data_ptr() % _ALIGN_BYTES or weight.data_ptr() % _ALIGN_BYTES:
        return None
    # ``project`` and ``linear_op`` are both called inside capture regions by the engine and
    # Eagle3 paths. The descriptors are host-built per call from the live data_ptr, so a
    # capture would bake one iteration's addresses into the graph; decline instead of
    # replaying against a stale buffer.
    if torch.cuda.is_current_stream_capturing():
        return None
    return _MEASURED_FAST.get((M, K, N))


def _stream(input: torch.Tensor, weight: torch.Tensor, tile: Tile) -> torch.Tensor:
    K = weight.shape[1]
    N = weight.shape[0]
    M = input.numel() // K
    key = (M, K, N)
    _FASTPATH_HITS[key] = _FASTPATH_HITS.get(key, 0) + 1
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    _KERNEL(input.reshape(M, K), weight, out, tile)
    return out.view(*input.shape[:-1], N)


# ---------------------------------------------------------------------------
# Alternative torch entry points for the same product.
#
# The cheapest experiment available, and worth having a table for even though it is empty:
# asking torch for ``x @ W.T`` a different way costs nothing to try, and if some other entry
# point reached a better-quantised ``nvjet`` tile it would be a free win. Each of these was
# routed through this table one at a time and measured by ``python validate.py`` -- not by a
# probe, which cannot admit anything -- with the full output archived under
# ``bench_results/runs/api_*.txt``. Grids and kernel names for every dispatch that differs
# from the baseline's are in ``profile/lmhead_api_variants/``.
#
# All four lose in-harness. Three of them dispatch to the *identical* nvjet kernel as
# ``F.linear``, so they are the same call by another name and lose only the wrapper's cost;
# ``addmm`` adds a bias epilogue plus the zero tensor it needs; and the role-swapped
# ``matmul(W, x.T)`` either falls off the sm100 path onto an Ampere-generation
# ``cutlass_80_tensorop_bf16`` kernel or, where it does not, pays for the transposed-output
# copy that returning ``[M, N]`` requires. The per-shape readings are in ``benchmark.csv``.
API_MM = "mm"            # torch.mm on an explicitly reshaped x
API_ADDMM = "addmm"      # torch.addmm, accumulating into a zero tensor
API_SWAP_T = "swap_t"    # role-swapped NT GEMM, plus the copy that transposes the result back
API_MV = "mv"            # torch.mv, M == 1 only

_MEASURED_FAST_API: dict[tuple[int, int, int], str] = {}

# True when some projection table has an entry, i.e. when there is a dispatch decision to make at
# all. Computed once at import, because with every table empty the guards below have nothing to
# find and walking them is pure cost: at M = 1 the extra frames measured a systematic ~1.9 us
# in-harness against the baseline across three runs -- one reading quantum, and the difference
# between sitting exactly on the 0.98x floor and clearing it. The candidate should not be
# measurably slower than the baseline for deciding to do nothing.
_HAS_PROJECTION_FAST_PATH = bool(_MEASURED_FAST) or bool(_MEASURED_FAST_API)


def _api_mm(input, weight, M, K, N):
    return torch.mm(input.reshape(M, K), weight.T)


def _api_addmm(input, weight, M, K, N):
    # addmm needs something to accumulate into. A broadcast scalar zero is the cheapest thing
    # that keeps the result equal to F.linear's, and its allocation is part of the variant's
    # honest cost rather than something to hide outside the timed region.
    zero = torch.zeros((), device=input.device, dtype=input.dtype)
    return torch.addmm(zero, input.reshape(M, K), weight.T)


def _api_swap_t(input, weight, M, K, N):
    # The product comes out [N, M]; returning [M, N] costs a real copy, and this variant may only
    # ever be judged on the post-copy number.
    return torch.matmul(weight, input.reshape(M, K).T).T.contiguous()


def _api_mv(input, weight, M, K, N):
    return torch.mv(weight, input.reshape(K))


_API_IMPLS = {API_MM: _api_mm, API_ADDMM: _api_addmm,
              API_SWAP_T: _api_swap_t, API_MV: _api_mv}


def _plan_api(input: torch.Tensor, weight: torch.Tensor, bias) -> str | None:
    """Return the alternative entry point to use for this call, or None to fall through.

    Cheap and pure. The predicates are the ones these alternatives actually need, which is a
    shorter list than the streaming kernel's -- they are all torch calls, so alignment,
    contiguity of ``x`` and graph capture are torch's problem, not ours. What does matter is
    that the reshape is meaningful, that no bias is involved, and that ``mv`` only ever sees
    a single row.
    """
    if not _MEASURED_FAST_API or bias is not None:
        return None
    if weight.dim() != 2 or input.dim() < 1:
        return None
    K = weight.shape[1]
    if K <= 0 or input.shape[-1] != K:
        return None
    M = input.numel() // K
    if M <= 0:
        return None
    mode = _MEASURED_FAST_API.get((M, K, weight.shape[0]))
    if mode is None or mode not in _API_IMPLS:
        return None
    # Equality, not identity: these are strings, and only CPython's interning of the
    # current literals would make `is` work. A table built at runtime would break it.
    if mode == API_MV and M != 1:
        return None
    return mode


class LogitMatmul(Matmul):
    """The baseline's functional linear, with the measured-only fast paths in front of it.

    Subclasses the frozen L1 winner rather than replacing it, so the fallback is whatever L1
    has been measured to be best -- today ``F.linear``, its own table being empty -- and the
    ``(input, weight, bias=None)`` signature that ``L4/mamba.py`` calls directly is inherited
    rather than restated. Stays a plain ``nn.Module`` with no parameters, so the bench's
    submodule walk and the weight loader see what they saw before; in particular it does not
    look like an fp8 linear, which is matched on a ``.weight`` parameter this class does not
    have.

    Two tables sit in front of the fallback and both are empty, so the first thing ``forward``
    does is skip them: with nothing admitted anywhere there is no decision to make, and walking
    the guards to reach that conclusion measured a systematic ~1.9 us at M = 1 -- one reading
    quantum, against a baseline that is two frames to this file's eight. When a table *does* have
    an entry they are consulted in cost order, because an alternative torch entry point is free
    if it wins and so is asked before the custom kernel.
    """

    def forward(self, input, weight, bias=None):
        if not _HAS_PROJECTION_FAST_PATH:
            return super().forward(input, weight, bias)
        mode = _plan_api(input, weight, bias)
        if mode is not None:
            K = weight.shape[1]
            N = weight.shape[0]
            M = input.numel() // K
            _FASTPATH_HITS[(M, K, N)] = _FASTPATH_HITS.get((M, K, N), 0) + 1
            return _API_IMPLS[mode](input, weight, M, K, N).view(*input.shape[:-1], N)
        tile = _plan(input, weight, bias)
        if tile is None:
            return super().forward(input, weight, bias)
        return _stream(input, weight, tile)


# The two row-gather implementations available for a given shape. ATen is the default and
# not a member of this table, because it is also the fallback: an unrecognised name in the
# table below degrades to it rather than raising.
#
# ``L1_GATHER`` is the frozen ``candidate/L1/embedding.py`` CUDA kernel, reached through the
# submodule that already owns the parameter. It is bit-exact on all five scored shapes and
# slower on two of them, which is the whole reason this dispatch exists: routing
# unconditionally through ``self.embedding_op(x)``, as the baseline does, would take its
# 0.83x on n = 1 and 0.95x on n = 16384. Its own docstring says as much -- it targets narrow
# rows, and "for a wide row ATen already does the right thing". These rows are 4 KB and 8 KB.
L1_GATHER = "l1"

# One behavioural difference from the baseline follows from choosing ATen, and it is worth
# stating rather than leaving to be discovered. Lazy negation is *metadata*:
# ``torch._neg_view(t)`` shares the storage of ``t`` and has the logical value of ``-t``. The
# frozen L1 gather reads ``const_data_ptr()`` directly, so given a neg-bit index tensor it
# returns the row at the *physical* index, while ATen honours the logical one. On
# ``torch._neg_view(tensor([1]))`` the frozen kernel returns ``weight[1]`` and ATen raises an
# out-of-range device assertion -- which is the right answer for a logical index of -1. So
# routing to ATen diverges from the baseline here in the direction of being correct. It cannot
# affect scoring: the harness skips any case whose baseline raises, and the bench never
# produces a neg-bit index. The same metadata is why ``_plan`` refuses a neg-bit operand
# outright instead of trusting a pointer re-derived from it.
#
# ``VEC_GATHER`` is the store-bound stretch goal: a 16-byte-vectorised row gather on an
# SM-count-sized persistent grid, which is what the plan proposed for the one embedding shape
# with real content (n = 16384, dim = 4096 -- 134 MB of stores, which ATen writes at 4.65 TB/s
# net of the harness floor, against torch's own 5.3 TB/s copy rate). It was routed to by an
# exact ``(16384, 4096)`` entry and measured by ``python validate.py`` three times; the readings
# are in ``benchmark.csv`` and the archived runs are ``bench_results/runs/gather_vec_run*.txt``.
# It won by one ~2.04 us timing quantum -- 33.8 us against ATen's 35.8 us, identically in all
# three runs and bit-exact -- and that is short of the 4 us admission margin, so the entry is
# gone. Not a tie and not an ATen win: a real but unadmittable one.
VEC_GATHER = "vec"

# Keyed ``(n, embedding_dim)`` -- the exact problem, never a neighbourhood of it. Empty, for two
# different measured reasons that are worth keeping distinct. Against the frozen L1 gather, ATen
# won or tied on every scored shape. Against the vectorised store path it did *not*: that path won
# at n = 16384 by 2.0 us, reproducibly, and 2.0 us is one reading quantum against a 4 us admission
# margin, so it is unadmittable rather than unwanted. An empty table is a measured decision here,
# not an unfinished one.
_MEASURED_FAST_GATHER: dict[tuple[int, int], str] = {}

_VEC_GATHER_BLOCK = 1024      # bf16 elements per program: 128 threads x 8 = 16 B per thread
_VEC_GATHER_WAVES = 16        # persistent grid of 16 x SM count

_VEC_GATHER = None
_VEC_GATHER_STATUS = "disabled:not-requested"


def _build_vector_gather():
    """Compile the persistent vectorised row gather and return a launcher, or raise."""
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_persistent(W, Idx, Out, dim, n_rows, n_src_rows, n_tiles,
                           BLOCK: tl.constexpr, GRID: tl.constexpr):
        """out[i, :] = weight[idx[i], :], one program per (row, column block), strided.

        The grid is sized to the SM count rather than to the work, so the tile loop runs
        inside a resident program instead of paying a block launch per 2 KB of output. Stores
        are ``BLOCK`` contiguous bf16 per program, which at 128 threads is 16 bytes each.

        The index load is range-checked and the weight load is masked on the result, so an
        out-of-range index cannot issue an illegal access. Note the semantic difference this
        creates against ATen, which raises a device-side assert for such an index where this
        writes a zero row: that difference is the reason this path would need an in-range
        guarantee from its caller before it could ship, and it is moot here because the shape
        was measured and rejected.
        """
        per_row = dim // BLOCK
        off = tl.arange(0, BLOCK)
        for tile in range(tl.program_id(0), n_tiles, GRID):
            row = tile // per_row
            col = (tile % per_row) * BLOCK
            r = tl.load(Idx + row).to(tl.int64)
            ok = (r >= 0) & (r < n_src_rows) & (row < n_rows)
            src = tl.load(W + r * dim + col + off, mask=ok, other=0.0)
            tl.store(Out + row.to(tl.int64) * dim + col + off, src)

    def launch(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n_rows = indices.numel()
        dim = weight.shape[1]
        out = torch.empty((*indices.shape, dim), device=weight.device, dtype=weight.dtype)
        grid = _VEC_GATHER_WAVES * torch.cuda.get_device_properties(
            weight.device).multi_processor_count
        n_tiles = n_rows * (dim // _VEC_GATHER_BLOCK)
        _gather_persistent[(grid,)](
            weight, indices.reshape(-1), out, dim, n_rows, weight.shape[0], n_tiles,
            BLOCK=_VEC_GATHER_BLOCK, GRID=grid, num_warps=4)
        return out

    return launch


def _init_vector_gather() -> None:
    """Build the vectorised gather at import, but only if some shape routes to it."""
    global _VEC_GATHER, _VEC_GATHER_STATUS
    if VEC_GATHER not in _MEASURED_FAST_GATHER.values():
        _VEC_GATHER_STATUS = "disabled:no-enabled-shapes"
        return
    if not torch.cuda.is_available():
        _VEC_GATHER_STATUS = "disabled:no-cuda-device"
        return
    try:
        _VEC_GATHER = _build_vector_gather()
        for (n, dim), mode in _MEASURED_FAST_GATHER.items():
            if mode != VEC_GATHER:
                continue
            if dim % _VEC_GATHER_BLOCK:
                raise ValueError(f"dim={dim} is not a multiple of {_VEC_GATHER_BLOCK}")
            w = torch.zeros((max(n, 1) + 1, dim), device="cuda", dtype=torch.bfloat16)
            idx = torch.zeros((n,), device="cuda", dtype=torch.int64)
            _VEC_GATHER(w, idx)
            del w, idx
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _VEC_GATHER_STATUS = "built"
    except Exception as exc:
        _VEC_GATHER = None
        _VEC_GATHER_STATUS = f"failed:{type(exc).__name__}: {exc}"
        print(f"[candidate L2/parallel_embedding] vector gather unavailable, delegating to "
              f"ATen: {_VEC_GATHER_STATUS}", file=sys.stderr, flush=True)


def _gather_is_vectorisable(weight: torch.Tensor, indices: torch.Tensor) -> bool:
    """Preconditions the vectorised gather relies on. Pure, no allocation, no sync."""
    if _VEC_GATHER is None:
        return False
    if torch.is_grad_enabled():
        return False
    if not weight.is_cuda or indices.device != weight.device:
        return False
    if weight.device.index != torch.cuda.current_device():
        return False
    if weight.dim() != 2 or weight.shape[1] % _VEC_GATHER_BLOCK:
        return False
    if not weight.is_contiguous() or not indices.is_contiguous():
        return False
    # Lazy negation is metadata a raw-pointer kernel cannot see; see the note below.
    if weight.is_neg() or indices.is_neg() or weight.is_conj():
        return False
    if indices.dtype not in (torch.int64, torch.int32):
        return False
    if indices.numel() == 0:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return True

# ---------------------------------------------------------------------------
# The fused tensor-parallel shard gather.
#
# At tp > 1 each rank owns a contiguous slice of the vocabulary, so the baseline does five
# things to one index tensor: build a range mask, subtract the shard base under that mask,
# gather, zero the rows that belong to another rank, and all-reduce. The first four are one
# pass over the same data and materialise three intermediates the size of the output -- the
# mask, the shifted indices, and the pre-masked gather -- so they fuse into a single kernel
# that reads a global id and writes either its row or zeros. The all-reduce stays exactly
# where it was and there is still exactly one of them.
#
# This is unscored: ``parallel_embedding`` is not in the bench's ``_DISTRIBUTED_OPS``, so
# ``_tp_size()`` returns 1 and this path is never timed here. It is reached from the L4 models,
# which is why it is implemented and verified rather than left as the baseline's five ops --
# but also why it is not tuned. Semantics are the baseline's, exactly: an out-of-shard id
# yields a zero row, which is what ``mask * (x - vocab_start)`` followed by
# ``mask.unsqueeze(-1) * y`` produces (it gathers row 0 and then zeroes it).
#
# Unlike the store-bound gather above, this kernel needs no separate range check for safety:
# the shard test *is* the bounds test. An id inside ``[vocab_start, vocab_end)`` maps into
# ``[0, per_partition)`` by construction, and everything else takes the zero path.
_SHARD_GATHER_BLOCK = 1024
_SHARD_GATHER_WAVES = 16        # persistent grid of 16 x SM count, capped at the tile count
# A single host-side counter, so a test can tell "the fused path ran" from "the fallback ran and
# agreed with it". A plain int, no threads, nothing the integrity guards watch.
_SHARD_GATHER_HITS = [0]
_SHARD_GATHER = None
_SHARD_GATHER_STATUS = "disabled:not-initialised"


def _build_shard_gather():
    """Compile the fused shard gather and return a launcher, or raise."""
    import triton
    import triton.language as tl

    @triton.jit
    def _shard_gather_kernel(W, Idx, Out, dim, n_rows, vocab_start, vocab_end, n_tiles,
                             BLOCK: tl.constexpr, GRID: tl.constexpr):
        per_row = tl.cdiv(dim, BLOCK)
        off = tl.arange(0, BLOCK)
        for tile in range(tl.program_id(0), n_tiles, GRID):
            row = tile // per_row
            col = (tile % per_row) * BLOCK
            cols = col + off
            in_dim = cols < dim
            gid = tl.load(Idx + row).to(tl.int64)
            local = gid - vocab_start
            mine = (gid >= vocab_start) & (gid < vocab_end)
            # A non-local row reads nothing and stores zeros, which is the baseline's result
            # for it. Masking the load on ``mine`` is what makes the shard test double as the
            # bounds check.
            src = tl.load(W + local * dim + cols, mask=mine & in_dim, other=0.0)
            tl.store(Out + row.to(tl.int64) * dim + cols, src, mask=in_dim)

    def launch(weight, indices, vocab_start, vocab_end):
        flat = indices.reshape(-1)
        n_rows = flat.numel()
        dim = weight.shape[1]
        out = torch.empty((*indices.shape, dim), device=weight.device, dtype=weight.dtype)
        if n_rows == 0 or dim == 0:
            return out
        sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        per_row = (dim + _SHARD_GATHER_BLOCK - 1) // _SHARD_GATHER_BLOCK
        n_tiles = n_rows * per_row
        grid = min(_SHARD_GATHER_WAVES * sms, n_tiles)
        _shard_gather_kernel[(grid,)](
            weight, flat, out, dim, n_rows, vocab_start, vocab_end, n_tiles,
            BLOCK=_SHARD_GATHER_BLOCK, GRID=grid, num_warps=4)
        return out

    return launch


_SHARD_GATHER_TRIED = False


def _init_shard_gather() -> None:
    """Build the fused shard gather once. Any failure degrades to the baseline's ops.

    Called at import *and* on the first tp > 1 forward, because the ordering cannot be relied on:
    ``_tp_size()`` reads ``dist.get_world_size()``, so whether this module knows it is sharded
    depends on whether the process group was initialised before it was imported. An
    import-time-only build would silently leave the fused path dormant for any caller that imports
    first and initialises later -- correct, since the fallback is the baseline's own composition,
    but pointless.

    Building on first call would be wrong for the *scored* path, and is refused there: it would
    land inside the harness's correctness forwards or its warmups. It is fine here because this
    path exists only at tp > 1, which the bench never reaches -- ``parallel_embedding`` is not in
    ``_DISTRIBUTED_OPS`` -- so nothing timed can be behind it. The single attempt is what matters:
    a failed build is not retried per call.
    """
    global _SHARD_GATHER, _SHARD_GATHER_STATUS, _SHARD_GATHER_TRIED
    if _SHARD_GATHER_TRIED:
        return
    if not torch.cuda.is_available():
        _SHARD_GATHER_STATUS = "disabled:no-cuda-device"
        return
    if _tp_size() <= 1:
        # Not an outcome, just "not yet": leave _SHARD_GATHER_TRIED unset so a later tp > 1 forward
        # still gets its chance. At tp = 1 that forward never happens.
        _SHARD_GATHER_STATUS = "deferred:single-rank-at-import"
        return
    _SHARD_GATHER_TRIED = True
    try:
        _SHARD_GATHER = _build_shard_gather()
        _SHARD_GATHER_STATUS = "built"
    except Exception as exc:
        _SHARD_GATHER = None
        _SHARD_GATHER_STATUS = f"failed:{type(exc).__name__}: {exc}"
        print(f"[candidate L2/parallel_embedding] fused shard gather unavailable, using the "
              f"baseline's ops: {_SHARD_GATHER_STATUS}", file=sys.stderr, flush=True)


def _shard_gather_ok(weight: torch.Tensor, indices: torch.Tensor) -> bool:
    """Preconditions the fused shard gather relies on. Pure, no allocation, no sync."""
    if _SHARD_GATHER is None:
        return False
    if torch.is_grad_enabled():
        return False
    if not weight.is_cuda or indices.device != weight.device:
        return False
    if weight.device.index != torch.cuda.current_device():
        return False
    if weight.dim() != 2 or weight.shape[1] <= 0:
        return False
    if not weight.is_contiguous() or not indices.is_contiguous():
        return False
    if weight.is_neg() or indices.is_neg() or weight.is_conj():
        return False
    if indices.dtype not in (torch.int64, torch.int32):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return True


def _shard_gather(weight, x, vocab_start, vocab_end):
    """Rows of ``weight`` for the ids this rank owns, zeros for the rest.

    The fused kernel when its preconditions hold, and otherwise the baseline's exact
    composition -- which is also what documents the semantics the kernel has to match.
    """
    _init_shard_gather()
    if _shard_gather_ok(weight, x):
        _SHARD_GATHER_HITS[0] += 1
        return _SHARD_GATHER(weight, x, vocab_start, vocab_end)
    mask = (x >= vocab_start) & (x < vocab_end)
    y = torch.embedding(weight, mask * (x - vocab_start))
    return mask.unsqueeze(-1) * y



_init_vector_gather()
_init_shard_gather()


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__()
        tp, rank = _tp_size(), _tp_rank()
        assert num_embeddings % tp == 0
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.padding_size = padding_size
        self.embedding_dim = embedding_dim
        self.per_partition = num_embeddings // tp
        self.vocab_start = self.per_partition * rank
        self.vocab_end = self.vocab_start + self.per_partition
        self.tp_size = tp
        # Resolved exactly as the baseline resolves it, and equally inert: the parameter's
        # dtype comes from nn.Embedding's own default and the harness casts it afterwards.
        # The keyword has to keep existing because the bench passes it by name.
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.embedding_op = Embedding(self.per_partition, embedding_dim)
        self.embedding_op.emb.weight.weight_loader = self._weight_loader
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def _gather(self, weight, x):
        """One row gather, chosen per shape from the measured table. ATen unless told otherwise."""
        if not _MEASURED_FAST_GATHER:
            return torch.embedding(weight, x)
        mode = _MEASURED_FAST_GATHER.get((x.numel(), weight.shape[-1]))
        if mode == L1_GATHER:
            return self.embedding_op(x)
        if mode == VEC_GATHER and _gather_is_vectorisable(weight, x):
            return _VEC_GATHER(weight, x)
        return torch.embedding(weight, x)

    def forward(self, x):
        # Read the weight afresh from the module: four L4 models retie this parameter after
        # construction and the harness mutates its ``.data`` in place.
        weight = self.embedding_op.emb.weight
        if self.tp_size == 1:
            return self._gather(weight, x)
        # tp > 1 is unreachable under this bench -- ``parallel_embedding`` is not in
        # ``_DISTRIBUTED_OPS``, so ``_tp_size()`` is 1 -- but it is reached from thirteen L4
        # models. One all-reduce, as in the baseline; what is fused is everything before it.
        y = _shard_gather(weight, x, self.vocab_start, self.vocab_end)
        return self.allreduce(y)


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 bias: bool = False,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__(num_embeddings, embedding_dim,
                         params_dtype=params_dtype,
                         org_num_embeddings=org_num_embeddings,
                         padding_size=padding_size)
        # ``bias`` is accepted and ignored, as in the baseline. Materialising one would add a
        # state_dict entry the baseline does not have, and the harness would overwrite it
        # anyway -- it rewrites any float parameter whose absolute maximum falls below 1e-6.
        self.linear_op = LogitMatmul()

    def project(self, x):
        """Linear projection only (no gather). Used inside CUDA graph."""
        ctx = get_context()
        if ctx.is_mixed:
            x = x[ctx.logit_indices].contiguous()
        elif ctx.is_prefill:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return self.linear_op(x, self.embedding_op.emb.weight)

    def gather_logits(self, logits):
        """Gather partial logits from all ranks. Used outside CUDA graph."""
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if _tp_rank() == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if _tp_rank() == 0 else logits
        return logits

    def gather_greedy(self, logits):
        """Fast path for greedy: local argmax + small allgather.

        Instead of gathering full vocab logits (~31MB/rank), gather only
        the (max_val, max_idx) per sequence (~2KB/rank).
        Returns token IDs directly on rank 0, None on other ranks.
        """
        if self.tp_size <= 1:
            return None

        rank = _tp_rank()
        local_max_vals, local_max_idxs = logits.max(dim=-1)
        local_max_idxs = local_max_idxs + self.vocab_start

        info = torch.stack([local_max_vals, local_max_idxs.float()], dim=-1)
        gathered = [torch.empty_like(info) for _ in range(self.tp_size)]
        dist.all_gather(gathered, info)
        if rank == 0:
            all_info = torch.stack(gathered, dim=0)
            all_vals = all_info[:, :, 0]
            all_idxs = all_info[:, :, 1].long()
            best_rank = all_vals.argmax(dim=0)
            bs = logits.size(0)
            token_ids = all_idxs[best_rank, torch.arange(bs, device=logits.device)]
            return token_ids
        return None

    def forward(self, x):
        logits = self.project(x)
        return self.gather_logits(logits)
