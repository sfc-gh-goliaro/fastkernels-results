"""Oasis timestep embedding for B200 / sm_100: the whole forward in one host call.

The baseline is not compute-bound. Its useful work is ~16 MFLOP and its weights are
5 MiB (~0.7 us of B200 HBM traffic), yet it measures 56-77 us flat in ``M`` (the spread is
host load on a shared machine, not the shape) because it issues twelve eager torch kernels
and the timed window is paced by *host* dispatch, ~3 us fixed plus ~4 us per torch op. Ten
of those twelve kernels exist only to build an ``M x 256`` table of sines and cosines; the
two that matter are ~2 us of memory traffic scheduled as 7.5-9.6 us CUTLASS tiles whose
``M`` dimension is 95% padding at ``M <= 6``.

So the first lever is structural: one Python-visible call into a CUDA extension that issues
two device kernels, instead of twelve dispatches. That is most of the win, and it moves the
bottleneck rather than removing it -- once the dispatches are gone the window is
7 us of harness overhead, ~12-18 us of kernel device time and ~5.5 us of launch and
inter-kernel gap, so the *second* lever is the kernels themselves. What mattered there was
memory-level parallelism per warp, not occupancy; see ``TILE`` in the CUDA source and
``profile/tse_v2_hoisted_tile/REPORT.md``.

The binding constraint is numeric, not structural. ``allow_tf32`` is on and
``float32_matmul_precision`` is ``'high'`` in this build, so the reference both linears
are scored against rounds its GEMM operands to TF32 and accumulates in fp32. An
*exact*-fp32 candidate is therefore incorrect: it misses the harness's
``atol=1e-5, rtol=1e-3`` bound on ~14% of elements (matched 0.852-0.859 against a 0.99
gate) on every scored shape. This file rounds both operands with ``cvt.rn.tf32.f32``, the
conversion the tensor-core path itself applies, which makes every individual product
bit-identical to the reference's -- a TF32 value carries 11 significant bits, so a product of two
is 22 bits and is exact in fp32.

Rounding the operands is necessary but not sufficient. What is left is the summation *order*, and
it is not a small residual to be tolerated: the first stage's activation is rounded back to TF32
before the second GEMM reads it, so a ~1e-7 relative difference in the first stage gets quantised
up to a full TF32 ulp on roughly one element in 5 000, and each of those perturbs every output in
its row. That is what held a scalar implementation to ``matched_ratio`` 0.9997.

So the first stage does not approximate the reference's order, it *reproduces* it. Its reference
is ``cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4``, and ``s1688gemm`` is
``mma.sync.m16n8k8`` with TF32 operands; chaining that instruction over K in ascending steps of 8
gives a bitwise-equal stage. Measured, on this B200 / torch 2.11 / CUDA 13 build: the chained
instruction is bitwise equal to ``F.linear`` at both K = 256 and K = 1024
(``tests/probe_mma.py``). The kernel-name evidence establishes the opcode family; the bitwise
equality is what establishes the order, and only for the build it was measured on.

And the gap is not merely one of *order*. Fed products of 2^24, 1 and -2^24, a sequential fp32 fma
chain loses the small term and returns 0 while the instruction returns 1 -- the tensor core carries
more than fp32 precision through its 8-product sum and rounds once at the end. So no arrangement
of scalar fp32 fmas can reproduce the reference, which is why rearranging the reduction tree never
moved the scalar path off 0.9997.

The second stage stays scalar, because making it exact too costs 42% of the speed for a property
nothing requires -- see ``_MMA2``.

Three facts about that conversion were established by measurement
(``tests/probe_numeric.py``, ``tests/probe_tie_rule.py``), not assumed:

* The rounding mode has to be ``.rn`` (ties to even), not the ``.rna`` (ties away from
  zero) the design draft specified. Ties are one fp32 encoding in 8192, which at
  K = 256 is ~0.04 per output column -- rare, but each is worth a full TF32 ulp of one
  operand. Against cuBLAS on the scored geometry ``.rna`` deviates by up to 5.5e-5 and
  ``.rn`` by 1.1e-6, and end to end ``.rna`` costs ``matched_ratio`` 0.9969 where ``.rn``
  gives 0.9997 against a 0.99 gate.
* ``"=r"`` and ``"=f"`` destinations produce identical results on this compiler. ``"=r"``
  is kept because the instruction writes a ``.b32`` and because it is the constraint
  the frozen ``candidate/L1/fp8_linear.py`` uses for its ``cvt``.
* accumulating in fp64 instead of fp32 changes nothing (``tests/probe_accum.py``), which is why
  the residual error was mis-diagnosed at first as accumulation *width*. Widening the accumulator
  around a scalar chain cannot help: what has to change is the primitive doing the 8-product sum.

``cosf``/``sinf``/``expf`` are the accurate libdevice forms and ``--use_fast_math`` is
off, because ``torch.cos`` on fp32 CUDA calls the same routines and the embedding stage
is bitwise identical to ``timestep_embedding`` only as long as that holds.

Everything unproven delegates to the baseline computation: an unbuilt extension, an
unexpected shape or dtype, autograd, autocast, a changed matmul precision policy, or
``torch.compile`` all route to ``timestep_embedding`` plus the three ``mlp`` layers,
which is correct by construction. A build or dispatch problem can cost speed but can
never produce a wrong answer.
"""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

# ---------------------------------------------------------------------------
# Fast-path domain.
#
# Every bound here is something the kernels or the numeric argument actually rely on,
# not a guess:
#
# _M_MIN = 2   At M = 1 cuBLAS drops to an *exact*-fp32 GEMV instead of the TF32
#              tensor-core path, and the polarity of the argument above flips: the
#              TF32-emulating pipeline scores 0.869 there while exact fp32 scores
#              1.000. M = 1 is not a scored shape, so it delegates rather than guess.
# _M_MAX = 8   M is a template parameter so the M accumulators live in registers;
#              2..8 are the instantiations that exist. The scored envelope is 2..6.
# _K_MULTIPLE  32 lanes x float4 = 128 elements per load step, for both K dimensions.
# _SMEM_FLOATS Both kernels stage an M x K plane in shared memory; 8192 floats is
#              32 KiB, inside the 48 KiB a block gets without an opt-in attribute.
# ---------------------------------------------------------------------------
_M_MIN = 2
_M_MAX = 8
_K_MULTIPLE = 128
_SMEM_FLOATS = 8192
_ALIGN_BYTES = 16

# The (frequency_embedding_size, hidden_size) pairs the fast path is allowed to serve.
#
# An allow-list rather than the divisibility test this started as, because divisibility is
# not what makes the numerics work. Two shapes that both satisfy F % 128 == H % 128 == 0 can
# behave completely differently:
#
#   (256, 1024)  the scored envelope           matched 1.0000
#   (4096, 128)  admitted by divisibility      matched 0.9609  -- FAILS the 0.99 gate
#
# The reference is a TF32 tensor-core GEMM in both cases (checked directly), so the
# difference is accumulation depth: at K = 4096 the reference's own summation error is 60x
# what it is at K = 256, and this kernel's per-lane chain is 128 products long instead of 8.
# There is no cheap predicate for "the accumulation orders stay close enough", so the rule
# is measurement: a pair is admitted only if tests/test_candidate.py scores it against the
# real baseline, and that test iterates this set.
_VALIDATED_SHAPES = frozenset({
    (256, 1024),   # the scored envelope
    (128, 256),
    (256, 1152),
    (512, 512),
})

# Launch geometry. Zero means "derive the covering grid in C++", which is free there
# and would cost Python arithmetic on a path whose whole budget is a few microseconds.
#
# ``_KWARPS1``/``_KWARPS2`` are how many warps cooperate on one output column in each
# kernel. Splitting K multiplies the CTA count, which ncu's occupancy rule suggested was
# the fix for a grid of 128 CTAs on 148 SMs (0.14 waves, 12% achieved occupancy). It was
# measured and it is not: at KWARPS = 2/8 the kernels got *slower* (stage 2 10.1 -> 12.0 us
# at M = 6), because the extra warps came at the cost of the shared-memory reuse and two
# more barriers per column. The real limiter was memory-level parallelism inside each
# warp, which ``TILE`` in the CUDA source addresses instead. Both are kept at 1 and the
# split survives as a template parameter the sweep can re-measure.
#
# Each warp's share of K must be a whole number of 128-element steps, so KWARPS is capped
# by K/128 -- 2 for F = 256, 8 for H = 1024 -- and must divide the warps per block. The
# C++ entry clamps a request down to the largest legal value.
_G1 = 0
_G2 = 0
_THREADS = 256
_KWARPS1 = 1
_KWARPS2 = 1

# Whether each stage uses the tensor-core (`mma.sync.m16n8k8`) kernel instead of the scalar one.
#
# This is a *numeric* decision before it is a performance one. `mma.sync.m16n8k8` with TF32
# operands is the instruction cuBLAS's `s1688gemm` is built from, and issuing it over the same K
# in the same ascending order makes the stage bitwise equal to the reference. The scalar path
# cannot: its fma chain differs from the instruction's internal summation by ~1e-8 on about half
# of all elements, and `h = tf32(silu(y))` amplifies that into a full-ulp jump on roughly one h
# element in 5 000, which is what kept `matched_ratio` at 0.9997 instead of 1.0.
#
# The shipped combination is tensor-core stage 1 and scalar stage 2, chosen by measurement
# (metrics/mode_sweep.json, 25 variants interleaved in one process):
#
#   stage 1   stage 2   geomean   worst matched   max_abs    bitwise
#   scalar    scalar     3.72x      0.9997*       7.5e-06    no
#   mma       scalar     3.02x      1.000000      8.6e-07    no
#   mma       mma        1.75x      1.000000      0.0        YES
#                                   (*over the 175-case sweep; 1.0 over the five scored shapes)
#
# Exactness in stage 1 is what matters and it is nearly free. It drops max_abs by 18x -- because
# it removes the quantisation straddle, not because it shrinks an error -- and costs 19% of the
# speed. Making stage 2 exact as well buys bitwise equality for another 42%, which would put the
# operator under the speed floor the whole exercise exists to clear, so it stays a selectable mode
# with its measurement recorded rather than the default.
_MMA1 = 1
_MMA2 = 0

# How many steps of the weight row each lane loads before consuming any of them -- the
# memory-level-parallelism knob, and the one that mattered most. Measured: raising the second
# kernel's tile from 1 to 8 took it from 12.0 us to 6.6 us of device time at M = 6.
#
# A "step" is 128 elements on the scalar path (32 lanes x float4) and 8 on the tensor-core path
# (one mma), so the two tiles are not in the same units. TILE1 = 4 is the tensor-core stage 1's
# value and was worth 2.45x -> 3.02x on its own; TILE2 = 8 gives the scalar stage 2 its whole
# share of K in flight at H = 1024.
#
# Overridable through the environment purely so the two builds can be compared in one
# session; the shipped values are the defaults.
def _tile_from_env(name: str, default: int) -> int:
    """A malformed override falls back to the default instead of raising out of the import.

    This is a test-only knob on a module the harness imports, so a stray value in the environment
    must not be able to turn the whole operator into an ImportError.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not 1 <= value <= 16:
        print(f"[{_EXT_BASENAME}] ignoring {name}={raw!r}, using {default}",
              file=sys.stderr, flush=True)
        return default
    return value


_TILE1 = _tile_from_env("FK_TSE_TILE1", 4)
_TILE2 = _tile_from_env("FK_TSE_TILE2", 8)

# Whether the weight loads carry `L1::no_allocate`. Compile-time: the policy is a qualifier on the
# load instruction. Measured, see metrics/experiment_sweep.json.
_CACHE_HINTS = 1

# The `__launch_bounds__` ceiling. Compile-time, and swept together with `-maxrregcount`.
_LAUNCH_BOUNDS = 512

# Whether stage 1 gathers the embedding from the exact [128, F] table instead of computing the
# trig. Bit-identical either way -- the table is built with the baseline's own expression -- so this
# is purely a speed question, and out-of-range `t` still goes through the in-kernel computation.
_EMB_TABLE = 0

# Whether the epilogue stages its stores through shared memory so a warp writes contiguously
# instead of one lane writing M values strided by H. Measured, see metrics/experiment_sweep.json.
_COALESCED = 0

# Whether the scalar stage 2 reads the hidden activation from global memory instead of staging it in
# shared.
#
# Off, and the reason is a disagreement between two measurements that is worth recording. The
# interleaved in-process harness put this 15% *ahead* of the staged version (2.024x against 1.752x,
# `metrics/pdl_coop_sweep.json`), and `python validate.py` then put the same build at 1.95x
# geometric mean against the staged version's 2.24x -- below AC-7's 2.0x floor.
#
# Both were run correctly. The in-process harness is the right tool for ranking variants against
# each other, because it sees the same clocks; but it is a *proxy* for the official window, and here
# the proxy and the gate disagreed. The gate wins: it is what the criterion is written against. So
# the staged version ships and this stays a recorded, selectable alternative rather than a claim.
_H_GLOBAL = 0

_MAX_PERIOD = 10000

_EXT_BASENAME = "fk_oasis_tse"
_DEFAULT_ARCH = "10.0"

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace {

// Calls that reached the kernels. Relaxed and host-read only, so it costs one
// non-contended increment per call and nothing the harness's integrity guards watch.
// A benchmark row measured with this at zero is a measurement of the torch fallback.
std::atomic<int64_t> g_fastpath_calls{0};

// Lanes per K step. 32 lanes x float4 = 128 elements, which is why both K dimensions
// must be multiples of 128. A sweep over 8 / 16 / 32 measured 32 best by 40%
// (metrics/geometry_sweep.json), so it is fixed rather than templated -- one axis fewer
// in a template space that already carries M and KWARPS.
constexpr int kLanes = 32;
constexpr int kStepElems = kLanes * 4;

// The conversion the tensor-core path applies to its operands.
//
// .rn (ties to even), not .rna (ties away from zero). The two differ on one fp32
// encoding in 8192, which sounds negligible and is not: at K = 256 that is ~0.04 ties
// per output column, each worth a full TF32 ulp, and the reference rounds ties to even.
// Measured against cuBLAS on the scored geometry (tests/probe_tie_rule.py), .rna is 50x
// further away -- max deviation 5.5e-5 against .rn's 1.1e-6 -- and end to end .rna costs
// matched_ratio 0.9969 where .rn gives 0.9997.
//
// The destination is a .b32, hence the integer constraint; "=f" was measured to behave
// identically on this compiler, but the ISA writes an untyped word and the frozen
// candidate/L1/fp8_linear.py uses an integer constraint for its own cvt.
__device__ __forceinline__ float to_tf32(const float x) {
  unsigned r;
  asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __int_as_float(r);
}

// The raw sinusoid element, before any TF32 rounding: exactly what the reference computes.
//
// Factored out and used by *both* stage-1 kernels and by the `embedding_raw` entry point below,
// so the exhaustive bitwise check runs against the same device code the shipped path runs. A test
// that compiles its own copy of this arithmetic proves nothing about the kernel.
//
// cosf/sinf are the accurate libdevice routines torch.cos/torch.sin lower to, and freqs arrives
// built by the baseline's own expression, so the pair is bit-identical to timestep_embedding's.
__device__ __forceinline__ void embedding_pair(const int64_t* __restrict__ t,
                                               const float* __restrict__ freqs, const int row,
                                               const int col, float& cos_out, float& sin_out) {
  const float a = static_cast<float>(t[row]) * freqs[col];
  cos_out = cosf(a);
  sin_out = sinf(a);
}

// The weight rows are streamed exactly once each -- the ncu traffic counters put both kernels at
// 1.01-1.02x of the weight-size floor -- so nothing is gained by letting them allocate L1 lines,
// and the staged plane and the bias are what would be evicted to make room. `L1::no_allocate` says
// "read this but do not keep it". Selected by TSE_CACHE_HINTS so the plain load stays measurable;
// the idiom follows candidate/L1/gelu.py.
#define TSE_CACHE_HINTS TSE_CACHE_HINTS_VALUE

__device__ __forceinline__ float4 load_weight4(const float4* __restrict__ p) {
#if TSE_CACHE_HINTS
  float4 r;
  asm("ld.global.L1::no_allocate.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
      : "l"(p));
  return r;
#else
  return *p;
#endif
}

__device__ __forceinline__ unsigned load_weight_bits(const float* __restrict__ p) {
#if TSE_CACHE_HINTS
  unsigned r;
  asm("ld.global.L1::no_allocate.b32 %0, [%1];" : "=r"(r) : "l"(p));
  return __float_as_uint(__int_as_float(static_cast<int>(r)));
#else
  return __float_as_uint(*p);
#endif
}

// Number of timesteps the exact embedding table covers. The harness materialises `t` with
// `randint(0, 128)`, so this is its whole domain -- but the kernel must still be right outside it,
// which is what the branch in `embedding_from_table` is for.
constexpr int kTableRows = 128;

// The embedding for one (row, col), taken from the precomputed table when `t[row]` is inside the
// table's domain and computed otherwise.
//
// The table is built with the baseline's own torch expression, so a gather is bit-identical to the
// computation by construction -- there is no accuracy question, only a speed one. The branch is
// what keeps the kernel correct for any `t`: no host read, no fallback to Python, and the
// out-of-range case still goes through the same `embedding_pair` the non-table path uses.
__device__ __forceinline__ void embedding_from_table(const int64_t* __restrict__ t,
                                                     const float* __restrict__ freqs,
                                                     const float* __restrict__ table,
                                                     const int F, const int row, const int col,
                                                     float& cos_out, float& sin_out) {
  const int64_t ti = t[row];
  if (ti >= 0 && ti < kTableRows) {
    const float* __restrict__ r = table + static_cast<size_t>(ti) * F;
    cos_out = r[col];
    sin_out = r[(F >> 1) + col];
  } else {
    embedding_pair(t, freqs, row, col, cos_out, sin_out);
  }
}

// Accumulate this lane's share of one output column's dot product.
//
// fmaf rather than a separate multiply and add: both operands are TF32-rounded, so the
// product carries 22 significant bits and is exact in fp32. A single rounding of
// a*b + c and a rounding of the (exact) product followed by a rounding of the sum are
// therefore the same number, and fmaf is one instruction.
//
// The A operand is passed as a plain pointer with row stride K, which is either shared
// memory (kernel 1's staged embedding) or global memory (kernel 2's activation, which
// is 24 KiB and stays in L1 after the first CTA on an SM touches it).
// TILE is how many 128-element steps of the B row this lane loads *before* it consumes
// any of them, and it is the single most important number in this file.
//
// Written as a plain loop -- load a step, accumulate it, next step -- the kernel keeps
// exactly one global load in flight per warp, because the fma consumes b immediately and
// the loop bound is a runtime value the compiler will not unroll across. ncu measured the
// consequence: 4.6% of peak DRAM throughput and half of all stalls on long_scoreboard.
// The whole problem is 5 MiB, which is smaller than this machine's bandwidth-delay
// product, so the only way to go fast is to have essentially all of it in flight at once:
// 4 MiB / 16 B per lane / 32768 threads = 8 loads per lane. Hence TILE = 8 for the second
// kernel (K = 1024, one step per warp per 128 elements) and TILE = 2 for the first
// (K = 256), which is exactly enough to cover the scored geometry with no remainder.
//
// Anything with a different step count still runs correctly through the remainder loop,
// just with less of K in flight.
template <int M, int TILE>
__device__ __forceinline__ void accumulate_chunk(const float* __restrict__ a,
                                                 const float4* __restrict__ b_row,
                                                 const int K, const int k_begin,
                                                 const int k_count, const int lane,
                                                 float (&acc)[M]) {
  const int steps = k_count / kStepElems;
  int s = 0;
  for (; s + TILE <= steps; s += TILE) {
    float4 b[TILE];
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
      b[j] = load_weight4(&b_row[(k_begin + (s + j) * kStepElems + lane * 4) >> 2]);
    }
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
      const int k = k_begin + (s + j) * kStepElems + lane * 4;
      const float b0 = to_tf32(b[j].x);
      const float b1 = to_tf32(b[j].y);
      const float b2 = to_tf32(b[j].z);
      const float b3 = to_tf32(b[j].w);
#pragma unroll
      for (int i = 0; i < M; ++i) {
        const float* __restrict__ ai = a + i * K + k;
        acc[i] = fmaf(b0, ai[0], acc[i]);
        acc[i] = fmaf(b1, ai[1], acc[i]);
        acc[i] = fmaf(b2, ai[2], acc[i]);
        acc[i] = fmaf(b3, ai[3], acc[i]);
      }
    }
  }
  for (; s < steps; ++s) {
    const int k = k_begin + s * kStepElems + lane * 4;
    const float4 b = load_weight4(&b_row[k >> 2]);
    const float b0 = to_tf32(b.x);
    const float b1 = to_tf32(b.y);
    const float b2 = to_tf32(b.z);
    const float b3 = to_tf32(b.w);
#pragma unroll
    for (int i = 0; i < M; ++i) {
      const float* __restrict__ ai = a + i * K + k;
      acc[i] = fmaf(b0, ai[0], acc[i]);
      acc[i] = fmaf(b1, ai[1], acc[i]);
      acc[i] = fmaf(b2, ai[2], acc[i]);
      acc[i] = fmaf(b3, ai[3], acc[i]);
    }
  }
}

// Steps of B hoisted per kernel. Substituted at build time from the Python constants, so
// a different tile is a different extension (the build name carries a hash of the source)
// rather than a silently stale .so.
constexpr int kTileStage1 = TSE_TILE1;
constexpr int kTileStage2 = TSE_TILE2;

// Reduce the M accumulators across the 32 lanes of a warp, into lane 0.
template <int M>
__device__ __forceinline__ void warp_reduce(float (&acc)[M]) {
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1) {
#pragma unroll
    for (int i = 0; i < M; ++i) {
      acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off, kLanes);
    }
  }
}

// Combine the KWARPS per-warp partials of one output column, in warp index order so the
// summation is deterministic. Returns true for the single thread that owns the epilogue.
//
// Every thread in the block must reach the barriers, which is why the callers run a
// uniform iteration count and pass n_valid rather than returning early.
template <int M, int KWARPS>
__device__ __forceinline__ bool combine_partials(float (&acc)[M], float* __restrict__ part,
                                                 const int col_local, const int kw,
                                                 const int lane) {
  if (KWARPS == 1) {
    return lane == 0;
  }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < M; ++i) {
      part[(col_local * KWARPS + kw) * M + i] = acc[i];
    }
  }
  __syncthreads();
  const bool owner = (kw == 0 && lane == 0);
  if (owner) {
#pragma unroll
    for (int i = 0; i < M; ++i) {
      float sum = part[(col_local * KWARPS) * M + i];
#pragma unroll
      for (int w = 1; w < KWARPS; ++w) {
        sum += part[(col_local * KWARPS + w) * M + i];
      }
      acc[i] = sum;
    }
  }
  return owner;
}

// Kernel 1: sinusoidal embedding -> silu(emb @ W1^T + b1), TF32-rounded into h.
//
// KWARPS warps cooperate on each output column, so the grid is KWARPS times larger than
// the one-warp-per-column arrangement it replaces. That arrangement launched 128 CTAs on
// 148 SMs -- 0.14 waves, 12% achieved occupancy against 75% theoretical -- and ncu
// attributed 34% of its stalls to long_scoreboard, i.e. global-load latency with nothing
// resident to hide it behind. Splitting K is how the extra warps are found: there are
// only H output columns to go around, and H is 1024.
//
// The embedding is rebuilt by every CTA. That replicates M*(F/2) sin/cos pairs across the
// grid, but they run concurrently on different SMs, so it costs throughput on a machine
// with SMs to spare rather than latency.
template <int M, int KWARPS>
__global__ void __launch_bounds__(TSE_LAUNCH_BOUNDS) tse_stage1(
    const int64_t* __restrict__ t, const float* __restrict__ freqs,
    const float* __restrict__ table, const float* __restrict__ w1,
    const float* __restrict__ b1, float* __restrict__ h, const int F, const int H) {
  extern __shared__ __align__(16) float smem[];
  float* __restrict__ emb_s = smem;                  // M x F
  float* __restrict__ part = smem + M * F;           // (warps) x M

  const int nthreads = blockDim.x;
  const int half = F >> 1;
  for (int idx = threadIdx.x; idx < M * half; idx += nthreads) {
    const int row = idx / half;
    const int col = idx - row * half;
    float c, sn;
    if (table != nullptr) {
      embedding_from_table(t, freqs, table, F, row, col, c, sn);
    } else {
      embedding_pair(t, freqs, row, col, c, sn);
    }
    // Rounded here so the GEMM below reads TF32 operands, which is what the reference does to
    // them too.
    emb_s[row * F + col] = to_tf32(c);
    emb_s[row * F + half + col] = to_tf32(sn);
  }
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int cols = nthreads / (kLanes * KWARPS);     // output columns per CTA
  const int col_local = warp / KWARPS;
  const int kw = warp % KWARPS;
  const int k_count = F / KWARPS;
  const int k_begin = kw * k_count;

  // Uniform across the block, so every thread reaches the same barriers inside
  // combine_partials even when the grid does not divide the column count evenly.
  const int groups = (H + cols - 1) / cols;
  const int iters = (groups > static_cast<int>(blockIdx.x))
                        ? (groups - static_cast<int>(blockIdx.x) + static_cast<int>(gridDim.x) - 1)
                              / static_cast<int>(gridDim.x)
                        : 0;
  for (int it = 0; it < iters; ++it) {
    const int n = (static_cast<int>(blockIdx.x) + it * static_cast<int>(gridDim.x)) * cols
                  + col_local;
    float acc[M];
#pragma unroll
    for (int i = 0; i < M; ++i) {
      acc[i] = 0.0f;
    }
    if (n < H) {
      accumulate_chunk<M, kTileStage1>(emb_s, reinterpret_cast<const float4*>(w1 + static_cast<size_t>(n) * F),
                          F, k_begin, k_count, lane, acc);
    }
    warp_reduce<M>(acc);
    const bool owner = combine_partials<M, KWARPS>(acc, part, col_local, kw, lane);
    if (owner && n < H) {
      const float bias = b1[n];
#pragma unroll
      for (int i = 0; i < M; ++i) {
        // The bias is a GEMM epilogue term, not an operand, so it stays fp32. The
        // activation is at::silu's own expression; the TF32 rounding that follows is
        // what the reference's second GEMM does to this value anyway.
        const float y = acc[i] + bias;
        h[static_cast<size_t>(i) * H + n] = to_tf32(y / (1.0f + expf(-y)));
      }
    }
    if (KWARPS > 1) {
      __syncthreads();  // the partial buffer is reused by the next iteration
    }
  }
}

// Kernel 2: out = h @ W2^T + b2.
//
// h is staged into shared memory, which is worth a barrier and 24 KiB because every
// column this CTA owns reads all of it: with KWARPS = 1 that is warps-per-block reuses of
// each element. Reading it from global instead was measured 2 us slower per call. h was
// just written by kernel 1, so the staging load itself comes out of L2.
template <int M, int KWARPS>
__global__ void __launch_bounds__(TSE_LAUNCH_BOUNDS) tse_stage2(
    const float* __restrict__ h, const float* __restrict__ w2, const float* __restrict__ b2,
    float* __restrict__ out, const int M_times_H, const int H, const int coalesced,
    const int h_global) {
  extern __shared__ __align__(16) float smem[];
  // The staged plane is not allocated at all when h comes from global, so the partial buffer sits
  // at the base of the allocation rather than after it.
  float* __restrict__ h_s = smem;                    // M x H, absent when h_global
  float* __restrict__ part = smem + (h_global ? 0 : M_times_H);   // (warps) x M

  const int nthreads = blockDim.x;
  // Staging h costs a barrier and up to 32 KiB of shared memory, and buys reuse across the columns
  // a block owns. Reading it straight from global instead measured 15% faster: h is at most 32 KiB,
  // kernel 1 has just written it so it is in L2, and every block reads the same addresses so it
  // stays in L1 -- the reuse was already free, and the shared memory was costing residency.
  const float* __restrict__ a_src = h;
  if (!h_global) {
    const int words = M_times_H / 4;
    const float4* __restrict__ src = reinterpret_cast<const float4*>(h);
    float4* __restrict__ dst = reinterpret_cast<float4*>(h_s);
    for (int idx = static_cast<int>(threadIdx.x); idx < words; idx += nthreads) {
      dst[idx] = src[idx];
    }
    __syncthreads();
    a_src = h_s;
  }

  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int cols = nthreads / (kLanes * KWARPS);
  const int col_local = warp / KWARPS;
  const int kw = warp % KWARPS;
  const int k_count = H / KWARPS;
  const int k_begin = kw * k_count;

  const int groups = (H + cols - 1) / cols;
  const int iters = (groups > static_cast<int>(blockIdx.x))
                        ? (groups - static_cast<int>(blockIdx.x) + static_cast<int>(gridDim.x) - 1)
                              / static_cast<int>(gridDim.x)
                        : 0;
  for (int it = 0; it < iters; ++it) {
    const int n = (static_cast<int>(blockIdx.x) + it * static_cast<int>(gridDim.x)) * cols
                  + col_local;
    float acc[M];
#pragma unroll
    for (int i = 0; i < M; ++i) {
      acc[i] = 0.0f;
    }
    if (n < H) {
      accumulate_chunk<M, kTileStage2>(
          a_src, reinterpret_cast<const float4*>(w2 + static_cast<size_t>(n) * H), H, k_begin,
          k_count, lane, acc);
    }
    warp_reduce<M>(acc);
    const bool owner = combine_partials<M, KWARPS>(acc, part, col_local, kw, lane);
    if (coalesced) {
      // Park the results in shared memory, then let the whole block write them out with
      // consecutive threads on consecutive addresses. The direct store has one lane per column
      // writing M values strided by H, i.e. M scattered 4-byte stores per warp.
      if (owner) {
#pragma unroll
        for (int i = 0; i < M; ++i) {
          part[col_local * M + i] = acc[i];
        }
      }
      __syncthreads();
      const int n_base = (static_cast<int>(blockIdx.x) + it * static_cast<int>(gridDim.x)) * cols;
      for (int idx = static_cast<int>(threadIdx.x); idx < M * cols; idx += nthreads) {
        const int i = idx / cols;
        const int c = idx - i * cols;
        const int nn = n_base + c;
        if (nn < H) {
          out[static_cast<size_t>(i) * H + nn] = part[c * M + i] + b2[nn];
        }
      }
      __syncthreads();
    } else {
      if (owner && n < H) {
        const float bias = b2[n];
#pragma unroll
        for (int i = 0; i < M; ++i) {
          out[static_cast<size_t>(i) * H + n] = acc[i] + bias;
        }
      }
      if (KWARPS > 1) {
        __syncthreads();
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Tensor-core path.
//
// `mma.sync.m16n8k8` with TF32 operands is the instruction cuBLAS's `s1688gemm` -- the kernel
// the first linear is scored against -- is built from. Issuing it over the same K in the same
// ascending order makes the sequence of hardware operations identical, and the result therefore
// *bitwise* equal to the reference rather than merely close.
//
// That is not an optimisation, it is the only way to reach the accuracy the plan asks for. The
// scalar path above cannot: a scalar fma chain over the same 8 products differs from the
// instruction's internal summation on about half of all elements (tests/probe_mma.py), by ~1e-8
// each. That is tiny, but `h = tf32(silu(y))` quantises at 4.9e-4 relative, so roughly one h
// element in 5 000 crosses a rounding boundary and jumps a full ulp -- and each one perturbs
// every output in its row. Bitwise-equal stages remove the mechanism instead of shrinking it.
__device__ __forceinline__ unsigned to_tf32_bits(const float x) {
  unsigned r;
  asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void mma_m16n8k8(const unsigned (&a)[4], const unsigned (&b)[2],
                                            float (&acc)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The m16n8k8 fragment layout, from the PTX ISA. Validated by tests/probe_mma.py against an
// exact fp64 product rather than trusted:
//   group = lane / 4, slot = lane % 4
//   A[16][K] row-major : a0=(group, slot)   a1=(group+8, slot)
//                        a2=(group, slot+4) a3=(group+8, slot+4)
//   B[K][8] col-major, i.e. b[n][k] in memory : b0=(slot, group)  b1=(slot+4, group)
//   C/D[16][8]         : c0=(group, 2*slot)   c1=(group, 2*slot+1)
//                        c2=(group+8, 2*slot) c3=(group+8, 2*slot+1)
//
// The tile is 16 rows but M is at most 8, so rows 8..15 never exist and a1/a3 are statically zero.
// That is safe for a stronger reason than "adding zero is exact" -- it is not, in general: 0 x Inf
// is NaN and -0 + +0 is +0. It is safe because the padded rows land in *separate* accumulator
// elements (c2/c3, rows 8..15) which are never stored, and nothing from them reaches the rows that
// do exist. So the staged plane stays M x K rather than being padded to 16 and the shared-memory
// budget is unchanged.
// Four floats of padding per row of the staged plane.
//
// Without it the A-fragment load is a 6-way bank conflict and the tensor-core kernels are
// unusable: lane `t` reads `a_s[group*LD + k0 + slot]` with `group = t/4`, and every K this
// operator sees is a multiple of 128, so `group*K % 32 == 0` and all M groups land on the same
// bank. At LD = K + 4 the groups step 4 banks apart, the 32 lanes cover all 32 banks exactly
// once, and the access is conflict-free.
constexpr int kSmemPad = 4;

template <int M>
__device__ __forceinline__ void load_a_frag(const float* __restrict__ a_s, const int lda,
                                            const int k0, const int group, const int slot,
                                            unsigned (&a)[4]) {
  const bool lo = group < M;
  const bool hi = group + 8 < M;
  a[0] = lo ? to_tf32_bits(a_s[static_cast<size_t>(group) * lda + k0 + slot]) : 0u;
  a[2] = lo ? to_tf32_bits(a_s[static_cast<size_t>(group) * lda + k0 + slot + 4]) : 0u;
  a[1] = hi ? to_tf32_bits(a_s[static_cast<size_t>(group + 8) * lda + k0 + slot]) : 0u;
  a[3] = hi ? to_tf32_bits(a_s[static_cast<size_t>(group + 8) * lda + k0 + slot + 4]) : 0u;
}

// b0 and b1 are 16 bytes apart in the same row, so a lane's two loads fall in one 32-byte
// sector and a warp's eight groups fetch eight fully-used sectors per step: the same total
// traffic as the scalar path's float4 loads, at a quarter the request width.
__device__ __forceinline__ void load_b_frag(const float* __restrict__ b_mat, const int K,
                                            const int n0, const int k0, const int group,
                                            const int slot, unsigned (&b)[2]) {
  const size_t row = static_cast<size_t>(n0 + group) * K + k0 + slot;
  b[0] = to_tf32_bits(__uint_as_float(load_weight_bits(b_mat + row)));
  b[1] = to_tf32_bits(__uint_as_float(load_weight_bits(b_mat + row + 4)));
}

// One warp, eight output columns, K walked in ascending steps of 8. TILE steps of B are hoisted
// into registers before any of them is consumed, for the same reason the scalar path hoists --
// and hoisting cannot perturb the result, because it reorders loads and not accumulations.
template <int M, int TILE>
__device__ __forceinline__ void mma_columns(const float* __restrict__ a_s, const int lda,
                                            const float* __restrict__ b_mat, const int K,
                                            const int n0, const int group, const int slot,
                                            float (&acc)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc[i] = 0.0f;
  }
  const int steps = K >> 3;
  int s = 0;
  for (; s + TILE <= steps; s += TILE) {
    unsigned bb[TILE][2];
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
      load_b_frag(b_mat, K, n0, (s + j) << 3, group, slot, bb[j]);
    }
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
      unsigned a[4];
      load_a_frag<M>(a_s, lda, (s + j) << 3, group, slot, a);
      mma_m16n8k8(a, bb[j], acc);
    }
  }
  for (; s < steps; ++s) {
    unsigned a[4], b[2];
    load_a_frag<M>(a_s, lda, s << 3, group, slot, a);
    load_b_frag(b_mat, K, n0, s << 3, group, slot, b);
    mma_m16n8k8(a, b, acc);
  }
}

// Kernel 1, tensor-core form: the same embedding stage, then the GEMM as chained mma.
template <int M>
__global__ void __launch_bounds__(TSE_LAUNCH_BOUNDS) tse_stage1_mma(
    const int64_t* __restrict__ t, const float* __restrict__ freqs,
    const float* __restrict__ table, const float* __restrict__ w1,
    const float* __restrict__ b1, float* __restrict__ h, const int F, const int H) {
  extern __shared__ __align__(16) float emb_s[];  // M x (F + kSmemPad)

  const int nthreads = blockDim.x;
  const int half = F >> 1;
  const int lda = F + kSmemPad;
  for (int idx = threadIdx.x; idx < M * half; idx += nthreads) {
    const int row = idx / half;
    const int col = idx - row * half;
    float c, sn;
    if (table != nullptr) {
      embedding_from_table(t, freqs, table, F, row, col, c, sn);
    } else {
      embedding_pair(t, freqs, row, col, c, sn);
    }
    emb_s[row * lda + col] = to_tf32(c);
    emb_s[row * lda + half + col] = to_tf32(sn);
  }
  __syncthreads();

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int warps = nthreads >> 5;
  const int group = lane >> 2;
  const int slot = lane & 3;
  const int stride = static_cast<int>(gridDim.x) * warps * 8;

  for (int n0 = (static_cast<int>(blockIdx.x) * warps + warp) * 8; n0 < H; n0 += stride) {
    float acc[4];
    mma_columns<M, kTileStage1>(emb_s, lda, w1, F, n0, group, slot, acc);
    if (group < M) {
      const int n = n0 + 2 * slot;
      const float y0 = acc[0] + b1[n];
      const float y1 = acc[1] + b1[n + 1];
      h[static_cast<size_t>(group) * H + n] = to_tf32(y0 / (1.0f + expf(-y0)));
      h[static_cast<size_t>(group) * H + n + 1] = to_tf32(y1 / (1.0f + expf(-y1)));
    }
  }
}

// Kernel 2, tensor-core form.
template <int M>
__global__ void __launch_bounds__(TSE_LAUNCH_BOUNDS) tse_stage2_mma(
    const float* __restrict__ h, const float* __restrict__ w2, const float* __restrict__ b2,
    float* __restrict__ out, const int M_times_H, const int H) {
  extern __shared__ __align__(16) float h_s[];  // M x (H + kSmemPad)

  const int nthreads = blockDim.x;
  const int lda = H + kSmemPad;
  {
    // Row by row rather than one flat copy, because the padded stride is not a multiple of the
    // float4 width. H is a multiple of 128 and the pad is 4 floats, so every row start stays
    // 16-byte aligned and the vector loads survive.
    const int words_per_row = H / 4;
    for (int idx = static_cast<int>(threadIdx.x); idx < M * words_per_row; idx += nthreads) {
      const int row = idx / words_per_row;
      const int w = idx - row * words_per_row;
      reinterpret_cast<float4*>(h_s + row * lda)[w] =
          reinterpret_cast<const float4*>(h + static_cast<size_t>(row) * H)[w];
    }
    __syncthreads();
  }

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int warps = nthreads >> 5;
  const int group = lane >> 2;
  const int slot = lane & 3;
  const int stride = static_cast<int>(gridDim.x) * warps * 8;

  for (int n0 = (static_cast<int>(blockIdx.x) * warps + warp) * 8; n0 < H; n0 += stride) {
    float acc[4];
    mma_columns<M, kTileStage2>(h_s, lda, w2, H, n0, group, slot, acc);
    if (group < M) {
      const int n = n0 + 2 * slot;
      out[static_cast<size_t>(group) * H + n] = acc[0] + b2[n];
      out[static_cast<size_t>(group) * H + n + 1] = acc[1] + b2[n + 1];
    }
  }
}

// Test-only companion to `embedding_raw`. Deliberately calls the same `embedding_pair` the two
// stage-1 kernels call, so what it writes is what they consume before rounding.
__global__ void embedding_raw_kernel(const int64_t* __restrict__ t,
                                     const float* __restrict__ freqs, float* __restrict__ out,
                                     const int F, const int half, const int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    const int row = idx / half;
    const int col = idx - row * half;
    float c, sn;
    embedding_pair(t, freqs, row, col, c, sn);
    out[static_cast<size_t>(row) * F + col] = c;
    out[static_cast<size_t>(row) * F + half + col] = sn;
  }
}

struct Geometry {
  int g1;
  int g2;
  int threads;
  int kwarps1;
  int kwarps2;
  int mma1;
  int mma2;
  int emb_table;
  int coalesced;
  int h_global;
};

// Bit layout of the packed geometry, in one place so the Python and C++ sides cannot drift:
//   g1 : 0-15    g2 : 16-31    threads : 32-41    kwarps1 : 42-45
//   kwarps2 : 46-49   mma1 : 50   mma2 : 51   emb_table : 52   coalesced : 53
//   h_global : 54
Geometry unpack_geometry(const int64_t packed) {
  Geometry geom;
  geom.g1 = static_cast<int>(packed & 0xFFFF);
  geom.g2 = static_cast<int>((packed >> 16) & 0xFFFF);
  geom.threads = static_cast<int>((packed >> 32) & 0x3FF);
  geom.kwarps1 = static_cast<int>((packed >> 42) & 0xF);
  geom.kwarps2 = static_cast<int>((packed >> 46) & 0xF);
  geom.mma1 = static_cast<int>((packed >> 50) & 0x1);
  geom.mma2 = static_cast<int>((packed >> 51) & 0x1);
  geom.emb_table = static_cast<int>((packed >> 52) & 0x1);
  geom.coalesced = static_cast<int>((packed >> 53) & 0x1);
  geom.h_global = static_cast<int>((packed >> 54) & 0x1);
  return geom;
}

// The largest power-of-two warp split that is legal for this K and block shape: it must be
// one of the values dispatch instantiates, divide the warps per block, and give each warp a
// whole number of 128-element steps.
//
// Rounding down to a power of two first matters: dispatch only implements {1, 2, 4, 8}, so a
// request of 3 that happened to satisfy the divisibility tests (threads = 96, K = 384) used
// to survive the loop and then raise from the switch instead of running.
int clamp_kwarps(const int requested, const int K, const int warps) {
  int kw = 8;
  while (kw > 1 && kw > requested) {
    kw >>= 1;
  }
  if (kw > warps) {
    kw = warps;
  }
  while (kw > 1 && (warps % kw != 0 || (K / kw) % kStepElems != 0)) {
    kw >>= 1;
  }
  return kw < 1 ? 1 : kw;
}

// The two launches, in order, on the caller's stream.
//
// The stages are dispatched *independently* -- each picks scalar or tensor-core, and the scalar
// one picks its own KWARPS -- rather than as one combined template. That is what keeps the
// instantiation count at 7 x (4 + 1) per stage instead of the product over both, and it is also
// the shape the measurement wants: the two stages have different K and turned out to prefer
// different implementations.
struct Launch {
  int grid;
  int threads;
  size_t smem;
  cudaStream_t stream;
};

template <int M, int KWARPS>
void launch_stage1_scalar(const at::Tensor& t, const at::Tensor& freqs, const float* table,
                          const at::Tensor& w1, const at::Tensor& b1, at::Tensor& h, const int F,
                          const int H, const Launch& L) {
  tse_stage1<M, KWARPS><<<L.grid, L.threads, L.smem, L.stream>>>(
      t.const_data_ptr<int64_t>(), freqs.const_data_ptr<float>(), table,
      w1.const_data_ptr<float>(), b1.const_data_ptr<float>(), h.data_ptr<float>(), F, H);
}

template <int M>
void launch_stage1_mma(const at::Tensor& t, const at::Tensor& freqs, const float* table,
                       const at::Tensor& w1, const at::Tensor& b1, at::Tensor& h, const int F,
                       const int H, const Launch& L) {
  tse_stage1_mma<M><<<L.grid, L.threads, L.smem, L.stream>>>(
      t.const_data_ptr<int64_t>(), freqs.const_data_ptr<float>(), table,
      w1.const_data_ptr<float>(), b1.const_data_ptr<float>(), h.data_ptr<float>(), F, H);
}

template <int M, int KWARPS>
void launch_stage2_scalar(const at::Tensor& h, const at::Tensor& w2, const at::Tensor& b2,
                          at::Tensor& out, const int M_times_H, const int H, const Launch& L,
                          const int coalesced, const int h_global) {
  tse_stage2<M, KWARPS><<<L.grid, L.threads, L.smem, L.stream>>>(
      h.const_data_ptr<float>(), w2.const_data_ptr<float>(), b2.const_data_ptr<float>(),
      out.data_ptr<float>(), M_times_H, H, coalesced, h_global);
}

template <int M>
void launch_stage2_mma(const at::Tensor& h, const at::Tensor& w2, const at::Tensor& b2,
                       at::Tensor& out, const int M_times_H, const int H, const Launch& L) {
  tse_stage2_mma<M><<<L.grid, L.threads, L.smem, L.stream>>>(
      h.const_data_ptr<float>(), w2.const_data_ptr<float>(), b2.const_data_ptr<float>(),
      out.data_ptr<float>(), M_times_H, H);
}

// Columns one CTA covers, which fixes the covering grid. The scalar kernels give each column a
// group of KWARPS warps; the tensor-core kernels give each warp eight columns.
int columns_per_block(const int warps, const int kwarps, const bool mma) {
  return mma ? warps * 8 : warps / kwarps;
}

template <int M>
void launch_pair(const at::Tensor& t, const at::Tensor& freqs, const at::Tensor& w1,
                 const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2,
                 at::Tensor& h, at::Tensor& out, const int F, const int H,
                 const Geometry& geom, cudaStream_t stream) {
  const int warps = geom.threads / kLanes;
  const int cols1 = columns_per_block(warps, geom.kwarps1, geom.mma1 != 0);
  const int cols2 = columns_per_block(warps, geom.kwarps2, geom.mma2 != 0);

  Launch L1{geom.g1 > 0 ? geom.g1 : (H + cols1 - 1) / cols1, geom.threads,
            sizeof(float) * (geom.mma1
                                 ? static_cast<size_t>(M) * (F + kSmemPad)
                                 : static_cast<size_t>(M) * F + static_cast<size_t>(warps) * M),
            stream};
  const size_t stage2_plane = geom.mma2 ? static_cast<size_t>(M) * (H + kSmemPad)
                                        : (geom.h_global ? 0 : static_cast<size_t>(M) * H);
  Launch L2{geom.g2 > 0 ? geom.g2 : (H + cols2 - 1) / cols2, geom.threads,
            sizeof(float) * (stage2_plane
                             + (geom.mma2 ? 0 : static_cast<size_t>(warps) * M)),
            stream};

  // The table lives after the frequencies in the same cached tensor, so a mode that does not want
  // it costs nothing and there is no extra pybind argument on the hot path.
  const float* table = geom.emb_table && freqs.numel() >= F / 2 + kTableRows * F
                           ? freqs.const_data_ptr<float>() + F / 2
                           : nullptr;
  if (geom.mma1) {
    launch_stage1_mma<M>(t, freqs, table, w1, b1, h, F, H, L1);
  } else {
    switch (geom.kwarps1) {
      case 1: launch_stage1_scalar<M, 1>(t, freqs, table, w1, b1, h, F, H, L1); break;
      case 2: launch_stage1_scalar<M, 2>(t, freqs, table, w1, b1, h, F, H, L1); break;
      case 4: launch_stage1_scalar<M, 4>(t, freqs, table, w1, b1, h, F, H, L1); break;
      default: launch_stage1_scalar<M, 8>(t, freqs, table, w1, b1, h, F, H, L1); break;
    }
  }
  if (geom.mma2) {
    launch_stage2_mma<M>(h, w2, b2, out, M * H, H, L2);
  } else {
    switch (geom.kwarps2) {
      case 1: launch_stage2_scalar<M, 1>(h, w2, b2, out, M * H, H, L2, geom.coalesced,
                                                 geom.h_global); break;
      case 2: launch_stage2_scalar<M, 2>(h, w2, b2, out, M * H, H, L2, geom.coalesced,
                                                 geom.h_global); break;
      case 4: launch_stage2_scalar<M, 4>(h, w2, b2, out, M * H, H, L2, geom.coalesced,
                                                 geom.h_global); break;
      default: launch_stage2_scalar<M, 8>(h, w2, b2, out, M * H, H, L2, geom.coalesced,
                                                 geom.h_global); break;
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch(const int M, const at::Tensor& t, const at::Tensor& freqs, const at::Tensor& w1,
              const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2, at::Tensor& h,
              at::Tensor& out, const int F, const int H, const Geometry& geom,
              cudaStream_t stream) {
#define TSE_M_CASE(MM)                                                                    \
  case MM:                                                                                \
    launch_pair<MM>(t, freqs, w1, b1, w2, b2, h, out, F, H, geom, stream);                 \
    return;
  switch (M) {
    TSE_M_CASE(2)
    TSE_M_CASE(3)
    TSE_M_CASE(4)
    TSE_M_CASE(5)
    TSE_M_CASE(6)
    TSE_M_CASE(7)
    TSE_M_CASE(8)
    default:
      TORCH_CHECK_VALUE(false, "oasis_tse: no kernel instantiated for M = ", M);
  }
#undef TSE_M_CASE
}

// The guards below re-check, in C++, what the Python predicate already established.
// TORCH_CHECK_VALUE surfaces as a Python ValueError, so "this input is not for me"
// stays distinguishable from a real CUDA error, which arrives as a RuntimeError.
Geometry validate(const at::Tensor& t, const at::Tensor& freqs, const at::Tensor& w1,
                  const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2,
                  const int64_t packed_geometry, int& M, int& F, int& H) {
  TORCH_CHECK_VALUE(t.is_cuda() && t.dim() == 1 && t.scalar_type() == at::kLong &&
                        t.is_contiguous(),
                    "oasis_tse: t must be a contiguous 1-D int64 CUDA tensor");
  TORCH_CHECK_VALUE(w1.dim() == 2 && w2.dim() == 2 && b1.dim() == 1 && b2.dim() == 1,
                    "oasis_tse: weights must be 2-D and biases 1-D");
  TORCH_CHECK_VALUE(freqs.dim() == 1, "oasis_tse: freqs must be 1-D");

  // Range-checked as int64 *before* narrowing. This is a public pybind entry point, so it
  // can be called without the Python predicate in front of it, and a length of 2^32 + 2
  // would otherwise wrap to M = 2, pass every check, and silently return two rows.
  const int64_t m64 = t.size(0);
  const int64_t f64 = w1.size(1);
  const int64_t h64 = w1.size(0);
  TORCH_CHECK_VALUE(m64 >= 2 && m64 <= 8, "oasis_tse: M must be in [2, 8], got ", m64);
  TORCH_CHECK_VALUE(f64 > 0 && f64 <= (1 << 20) && h64 > 0 && h64 <= (1 << 20),
                    "oasis_tse: F and H must be in (0, 2^20], got ", f64, " and ", h64);
  M = static_cast<int>(m64);
  F = static_cast<int>(f64);
  H = static_cast<int>(h64);

  TORCH_CHECK_VALUE(M >= 2 && M <= 8, "oasis_tse: M must be in [2, 8], got ", M);
  // Positive as well as divisible: F = H = 0 passes every modulo and then derives a grid of
  // zero blocks, which the driver rejects with "invalid argument" instead of falling back.
  TORCH_CHECK_VALUE(F > 0 && H > 0, "oasis_tse: F and H must be positive, got ", F, " and ",
                    H);
  TORCH_CHECK_VALUE(F % kStepElems == 0 && H % kStepElems == 0,
                    "oasis_tse: F and H must be multiples of 128, got ", F, " and ", H);
  TORCH_CHECK_VALUE(w2.size(0) == H && w2.size(1) == H && b1.size(0) == H && b2.size(0) == H,
                    "oasis_tse: second linear must be H x H with H-wide biases");
  // Either the frequency vector alone, or the frequency vector followed by the exact
  // [kTableRows, F] embedding table in the same allocation.
  TORCH_CHECK_VALUE(freqs.size(0) == F / 2 || freqs.size(0) == F / 2 + kTableRows * F,
                    "oasis_tse: the embedding source must be F/2 entries, optionally followed by "
                    "a ", kTableRows, " x ", F, " table; got ", freqs.size(0));
  TORCH_CHECK_VALUE(M * F <= 8192 && M * H <= 8192,
                    "oasis_tse: a staged plane exceeds the shared-memory budget");

  for (const at::Tensor* p : {&freqs, &w1, &b1, &w2, &b2}) {
    TORCH_CHECK_VALUE(p->is_cuda() && p->device() == t.device(),
                      "oasis_tse: every operand must be on t's device");
    TORCH_CHECK_VALUE(p->scalar_type() == at::kFloat, "oasis_tse: operands must be float32");
    TORCH_CHECK_VALUE(p->is_contiguous(), "oasis_tse: operands must be contiguous");
    TORCH_CHECK_VALUE(reinterpret_cast<uintptr_t>(p->const_data_ptr()) % 16 == 0,
                      "oasis_tse: operands must be 16-byte aligned for float4 loads");
  }

  Geometry geom = unpack_geometry(packed_geometry);

  TORCH_CHECK_VALUE(geom.threads >= kLanes && geom.threads <= 512 &&
                        geom.threads % kLanes == 0,
                    "oasis_tse: threads must be a multiple of 32 in [32, 512], got ",
                    geom.threads);
  const int warps = geom.threads / kLanes;
  // The tensor-core kernels give each warp an 8-column tile, so H must divide into those.
  TORCH_CHECK_VALUE(!((geom.mma1 || geom.mma2) && H % 8 != 0),
                    "oasis_tse: the tensor-core path needs H to be a multiple of 8, got ", H);
  // Each warp's K chunk must be a whole number of 128-element steps, and the warps of a
  // block must split evenly into column groups.
  TORCH_CHECK_VALUE(geom.kwarps1 >= 1 && geom.kwarps2 >= 1,
                    "oasis_tse: kwarps must be positive");
  // The packed values are a *request*, reduced here to the largest legal split. Doing it
  // in C++ rather than in the Python predicate keeps the predicate to attribute reads and
  // keeps the module shape-general: F = 128 cannot split at all, H = 256 splits by 2, and
  // the scored (256, 1024) splits by 2 and 8. The alternative -- deriving it in Python --
  // would put integer arithmetic on a path whose whole budget is a few microseconds, and
  // rejecting instead would silently narrow the admitted set to one shape.
  geom.kwarps1 = clamp_kwarps(geom.kwarps1, F, warps);
  geom.kwarps2 = clamp_kwarps(geom.kwarps2, H, warps);
  return geom;
}

// One stream-ordered allocation, split into the workspace and the output. Taken inside
// the device guard so it lands on t's device, and on the current stream so it is
// ordered against whatever the caller queued before this call.
std::vector<at::Tensor> run(const at::Tensor& t, const at::Tensor& freqs, const at::Tensor& w1,
                            const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2,
                            const int64_t packed_geometry) {
  int M = 0, F = 0, H = 0;
  const Geometry geom = validate(t, freqs, w1, b1, w2, b2, packed_geometry, M, F, H);

  const c10::cuda::CUDAGuard guard(t.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Two allocations rather than one split into views. One would save an allocator call,
  // which measured as nothing (the caching allocator hands back a warm block with no kernel
  // and no sync, and the harness's host thread runs far ahead of the device anyway), and it
  // would cost every returned output a reference to the discarded h plane -- a caller that
  // retains outputs would pin twice the memory it can see. h is released stream-ordered when
  // this function returns.
  at::Tensor h = at::empty({M, H}, w1.options());
  at::Tensor out = at::empty({M, H}, w1.options());

  g_fastpath_calls.fetch_add(1, std::memory_order_relaxed);
  dispatch(M, t, freqs, w1, b1, w2, b2, h, out, F, H, geom, stream);
  return {out, h};
}

}  // namespace

at::Tensor tse_forward(const at::Tensor& t, const at::Tensor& freqs, const at::Tensor& w1,
                       const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2,
                       const int64_t packed_geometry) {
  return run(t, freqs, w1, b1, w2, b2, packed_geometry)[0];
}

// Same launches, also returning the hidden activation, so the two stages can be scored
// against silu(F.linear(...)) and F.linear(...) separately. Test-only.
std::vector<at::Tensor> tse_forward_stages(const at::Tensor& t, const at::Tensor& freqs,
                                           const at::Tensor& w1, const at::Tensor& b1,
                                           const at::Tensor& w2, const at::Tensor& b2,
                                           const int64_t packed_geometry) {
  return run(t, freqs, w1, b1, w2, b2, packed_geometry);
}

// (g1, g2, threads, kwarps1, kwarps2) as the kernels would actually be launched for this
// (F, H). Lets the geometry sweep record what ran rather than what was asked for.
std::vector<int64_t> effective_geometry(const int64_t packed_geometry, const int64_t F,
                                        const int64_t H) {
  Geometry geom = unpack_geometry(packed_geometry);
  const int warps = geom.threads / kLanes;
  geom.kwarps1 = clamp_kwarps(geom.kwarps1, static_cast<int>(F), warps);
  geom.kwarps2 = clamp_kwarps(geom.kwarps2, static_cast<int>(H), warps);
  const int cols1 = columns_per_block(warps, geom.kwarps1, geom.mma1 != 0);
  const int cols2 = columns_per_block(warps, geom.kwarps2, geom.mma2 != 0);
  return {geom.g1 > 0 ? geom.g1 : (H + cols1 - 1) / cols1,
          geom.g2 > 0 ? geom.g2 : (H + cols2 - 1) / cols2,
          geom.threads, geom.kwarps1, geom.kwarps2, geom.mma1, geom.mma2, geom.emb_table,
          geom.coalesced, geom.h_global};
}

// Test-only: the raw, un-rounded embedding, written by the production `embedding_pair` helper.
//
// This exists because the only other way to see the kernel's embedding was through
// `tf32(silu(tf32(emb)))`, and that transform is *not* injective -- 625 of the encodings reachable
// from t in [0, 128) collapse several distinct TF32 inputs (0.8984375 and 0.89892578125 among
// them), so equality after it does not establish equality before it.
at::Tensor embedding_raw(const at::Tensor& t, const at::Tensor& freqs, const int64_t F) {
  TORCH_CHECK_VALUE(t.is_cuda() && t.dim() == 1 && t.scalar_type() == at::kLong &&
                        t.is_contiguous(),
                    "embedding_raw: t must be a contiguous 1-D int64 CUDA tensor");
  TORCH_CHECK_VALUE(F > 0 && F % 2 == 0 && freqs.numel() >= F / 2,
                    "embedding_raw: freqs must hold at least F/2 entries");
  const at::cuda::CUDAGuard guard(t.device());
  const int M = static_cast<int>(t.size(0));
  auto out = at::empty({M, F}, freqs.options());
  const int half = static_cast<int>(F) / 2;
  const int total = M * half;
  if (total > 0) {
    embedding_raw_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        t.const_data_ptr<int64_t>(), freqs.const_data_ptr<float>(), out.data_ptr<float>(),
        static_cast<int>(F), half, total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

int64_t fastpath_calls() { return g_fastpath_calls.load(std::memory_order_relaxed); }

void reset_fastpath_calls() { g_fastpath_calls.store(0, std::memory_order_relaxed); }
"""

_CPP_SOURCE = r"""
at::Tensor tse_forward(const at::Tensor& t, const at::Tensor& freqs, const at::Tensor& w1,
                       const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2,
                       int64_t packed_geometry);
std::vector<at::Tensor> tse_forward_stages(const at::Tensor& t, const at::Tensor& freqs,
                                           const at::Tensor& w1, const at::Tensor& b1,
                                           const at::Tensor& w2, const at::Tensor& b2,
                                           int64_t packed_geometry);
std::vector<int64_t> effective_geometry(int64_t packed_geometry, int64_t F, int64_t H);
at::Tensor embedding_raw(const at::Tensor& t, const at::Tensor& freqs, int64_t F);
int64_t fastpath_calls();
void reset_fastpath_calls();
"""


# Bit layout, mirrored by ``unpack_geometry`` in the CUDA source.
_GEOMETRY_FIELDS = (
    ("g1", 0, 16), ("g2", 16, 16), ("threads", 32, 10),
    ("kwarps1", 42, 4), ("kwarps2", 46, 4), ("mma1", 50, 1), ("mma2", 51, 1),
    ("emb_table", 52, 1), ("coalesced", 53, 1), ("h_global", 54, 1),
)


def pack_geometry(g1: int = _G1, g2: int = _G2, threads: int = _THREADS,
                  kwarps1: int = _KWARPS1, kwarps2: int = _KWARPS2,
                  mma1: int = _MMA1, mma2: int = _MMA2,
                  emb_table: int = _EMB_TABLE, coalesced: int = _COALESCED,
                  h_global: int = _H_GLOBAL) -> int:
    """Fold the launch geometry into one integer.

    Seven separate pybind arguments would be seven more casts on a path whose whole budget is a
    few microseconds, and the geometry never changes between calls, so it is packed once at
    import instead.
    """
    values = {"g1": g1, "g2": g2, "threads": threads, "kwarps1": kwarps1,
              "kwarps2": kwarps2, "mma1": mma1, "mma2": mma2, "emb_table": emb_table,
              "coalesced": coalesced, "h_global": h_global}
    packed = 0
    for name, shift, width in _GEOMETRY_FIELDS:
        value = values[name]
        if not 0 <= value < (1 << width):
            raise ValueError(f"{name}={value} does not fit in {width} bits")
        packed |= value << shift
    return packed


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
#: The compute capability the extension is compiled for, as a ``(major, minor)`` tuple, or
#: ``None`` when no device was visible at import. Only one architecture is compiled -- seven
#: would be roughly six times the cold-build time for six binaries that never run -- so a
#: call on a device of a *different* capability has no kernel image and would raise
#: ``cudaErrorNoKernelImageForDevice`` rather than fall back. The predicate checks for it.
_BUILT_FOR_CAPABILITY: tuple[int, int] | None = None


def _gencode_flag() -> str:
    """``-gencode`` for the visible device.

    Passed as a flag rather than through ``TORCH_CUDA_ARCH_LIST``: once a caller
    supplies an ``arch``-bearing flag, ``cpp_extension`` adds none of its own, so this
    pins the build to one architecture without mutating the environment other imports
    in this process compile under. The ambient list in this workspace names seven.
    """
    global _BUILT_FOR_CAPABILITY
    arch = _DEFAULT_ARCH
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}.{minor}"
            _BUILT_FOR_CAPABILITY = (major, minor)
    except Exception:  # noqa: BLE001 - no device is not a reason to fail the import
        pass
    if _BUILT_FOR_CAPABILITY is None:
        major, _, minor = arch.partition(".")
        _BUILT_FOR_CAPABILITY = (int(major), int(minor or 0))
    tag = arch.replace(".", "")
    return f"-gencode=arch=compute_{tag},code=sm_{tag}"


#: The source before the tile substitution below. Kept so ``tests/sweep_variants.py`` can
#: build several tile variants in one process and measure them interleaved -- on this
#: shared machine, the same binary measured 11.4 us and 17.7 us of device time in two runs
#: minutes apart, so variants compared across runs prove nothing.
_CUDA_SOURCE_TEMPLATE = _CUDA_SOURCE


def substitute_tiles(source: str, tile1: int, tile2: int, *, cache_hints: int = None,
                    launch_bounds: int = None) -> str:
    """Fill in the compile-time constants. Each combination is therefore its own extension.

    ``cache_hints`` and ``launch_bounds`` are compile-time because a cache-policy qualifier is part
    of the load instruction and ``__launch_bounds__`` is part of the kernel signature -- neither can
    be a runtime knob, so both are swept by building several extensions.
    """
    values = {
        "TSE_TILE1": tile1,
        "TSE_TILE2": tile2,
        "TSE_CACHE_HINTS_VALUE": _CACHE_HINTS if cache_hints is None else cache_hints,
        "TSE_LAUNCH_BOUNDS": _LAUNCH_BOUNDS if launch_bounds is None else launch_bounds,
    }
    for name, value in (("TSE_TILE1", values["TSE_TILE1"]), ("TSE_TILE2", values["TSE_TILE2"])):
        if not 1 <= value <= 16:
            raise ValueError(f"{name}={value} is outside the supported range 1..16")
    if values["TSE_CACHE_HINTS_VALUE"] not in (0, 1):
        raise ValueError("cache_hints must be 0 or 1")
    if not 32 <= values["TSE_LAUNCH_BOUNDS"] <= 1024:
        raise ValueError("launch_bounds must be in [32, 1024]")
    # Longest first, so TSE_TILE1 does not eat the prefix of another token.
    for name in sorted(values, key=len, reverse=True):
        source = source.replace(name, str(values[name]))
    return source


_CUDA_SOURCE = substitute_tiles(_CUDA_SOURCE_TEMPLATE, _TILE1, _TILE2)

# The name carries a hash of the source *and of the compile flags*, so an edit can never silently
# reuse a stale .so out of the shared extension cache (freshness there is decided by mtime) and
# this build can never collide with another operator's. The flags are in the digest because only
# one architecture is compiled: without them, two hosts of different compute capability would
# agree on the name and disagree on the contents.
_BUILD_FLAGS = ("-O3", "-lineinfo", _gencode_flag())
_SRC_DIGEST = hashlib.sha256(
    (_CUDA_SOURCE + _CPP_SOURCE + "|".join(_BUILD_FLAGS)).encode()).hexdigest()[:12]
_EXT_NAME = f"{_EXT_BASENAME}_{_SRC_DIGEST}"


def _build_directory() -> Path:
    """Where the extension is built. Workspace-local, so nothing lands in the shared cache.

    ``FK_TSE_BUILD_ROOT`` redirects it, and exists for one reason: with a fixed
    ``build_directory``, ``TORCH_EXTENSIONS_DIR`` has no effect, so there was no way to make the
    ``nvcc``-absent test face a genuinely *cold* build -- it kept finding the warm ``.so`` and so
    proved only that an already-built extension imports. Pointing this at a fresh directory is
    what makes that test test something.
    """
    root = os.environ.get("FK_TSE_BUILD_ROOT")
    base = Path(root) if root else Path(__file__).resolve().parents[2] / ".torch_extensions"
    path = base / _EXT_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build():
    """Compile and load the extension, or return ``None`` after reporting why not.

    Called at import, never from ``forward``: ninja spawns subprocesses, and the
    harness rejects a candidate whose thread count grows during the timing window.
    """
    from torch.utils.cpp_extension import load_inline

    build_dir = _build_directory()
    cold = not (build_dir / f"{_EXT_NAME}.so").exists()
    if cold:
        # The bench worker's stall watchdog watches its log's mtime, so a cold build
        # has to say something before it spends a minute in nvcc.
        print(f"[{_EXT_NAME}] cold build starting (nvcc, single arch) ...",
              file=sys.stderr, flush=True)
    try:
        ext = load_inline(
            name=_EXT_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["tse_forward", "tse_forward_stages", "effective_geometry",
                       "embedding_raw", "fastpath_calls", "reset_fastpath_calls"],
            # -lineinfo so ncu can attribute SASS to source. --use_fast_math is
            # deliberately absent: it would substitute __cosf/__sinf/__expf and the
            # embedding would stop being bitwise equal to timestep_embedding.
            extra_cuda_cflags=list(_BUILD_FLAGS),
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - a build failure must stay importable
        print(f"[{_EXT_NAME}] CUDA extension unavailable, delegating to the baseline "
              f"computation (1.00x): {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None
    if cold:
        print(f"[{_EXT_NAME}] build done", file=sys.stderr, flush=True)
    return ext


_EXT = _build()

#: Empty exactly when the fused path is live; otherwise the reason it is not. A
#: benchmark row recorded with this non-empty is a measurement of the fallback.
EXTENSION_STATUS = "" if _EXT is not None else "build failed (see stderr at import)"

#: Calls that took the torch fallback. A plain module-level int on a path that already
#: costs tens of microseconds; the fast-path counter lives in C++ so it costs nothing.
DELEGATED_CALLS = 0

_PACKED_GEOMETRY = pack_geometry()


def fastpath_calls() -> int:
    """Calls served by the fused kernels since import (or the last reset)."""
    return 0 if _EXT is None else _EXT.fastpath_calls()


def _report_counters() -> None:
    """One line on stderr at interpreter exit, if this module was used at all.

    The bench worker runs in a subprocess and sends stderr to the per-operator log, so
    this is what makes "the recorded speedup was the kernel, not the fallback" checkable
    after the fact rather than assumed. Registered at import; it creates no thread and
    runs long after any timing window.
    """
    fast = fastpath_calls()
    if fast or DELEGATED_CALLS:
        print(f"[{_EXT_NAME}] fast-path calls={fast} delegated={DELEGATED_CALLS} "
              f"status={EXTENSION_STATUS or 'live'}", file=sys.stderr, flush=True)


atexit.register(_report_counters)


def reset_counters() -> None:
    global DELEGATED_CALLS
    DELEGATED_CALLS = 0
    if _EXT is not None:
        _EXT.reset_fastpath_calls()


# ---------------------------------------------------------------------------
# Frequency cache
#
# Keyed on everything that changes the vector, and deliberately *not* a registered
# buffer: as a buffer it would enter state_dict() and collide with the harness's shared
# load, and a later .to(dtype) could downcast it.
# ---------------------------------------------------------------------------
_FREQS_CACHE: dict[tuple, torch.Tensor] = {}

#: The frequency vector concatenated with the exact embedding table, for the table mode.
_EMB_SOURCE_CACHE: dict[tuple, torch.Tensor] = {}

#: Timesteps the exact table covers, mirroring ``kTableRows`` in the CUDA source. The harness draws
#: ``t`` from ``randint(0, 128)``, so this is its whole domain -- the kernel still handles anything
#: outside it, in-kernel, without reading ``t`` on the host.
_TABLE_ROWS = 128

# Whether the extension has a kernel image for a given device index, resolved for *every* visible
# device at import (see ``_resolve_device_support``).
#
# Filled eagerly rather than on first use because the predicate is required to read only attributes
# and integers: ``get_device_capability`` is a driver query, and doing it lazily meant the very
# first admitted call did something the predicate promises not to do. Memoisation hid that in any
# warmed benchmark, which is the reason to fix it rather than to excuse it.
#
# An index absent from the map is declined, so the predicate stays a dict lookup even for a device
# that somehow appeared after import.
_DEVICE_SUPPORTED: dict[int | None, bool] = {}


# Every per-module hook dict torch consults on a forward call. Named explicitly rather than
# discovered, so a new hook kind in a future release shows up as a test failure here rather than
# as a silently skipped hook.
_HOOK_ATTRS = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
)

# The global registries, resolved once. torch consults these on every module call, so a hook
# installed with ``register_module_forward_hook`` applies to submodules this path bypasses.
_GLOBAL_HOOK_REGISTRIES = tuple(
    d for d in (
        getattr(nn.modules.module, "_global_forward_pre_hooks", None),
        getattr(nn.modules.module, "_global_forward_hooks", None),
        getattr(nn.modules.module, "_global_backward_pre_hooks", None),
        getattr(nn.modules.module, "_global_backward_hooks", None),
    ) if d is not None
)


def _has_hooks(module: nn.Module) -> bool:
    """Whether ``module`` or any of its ``mlp`` children carries a hook.

    Only the container and its three children are inspected -- one level, four modules, six
    dict lookups each -- because those are exactly the modules the fused path replaces.
    """
    for mod in (module, *module.mlp):
        for attr in _HOOK_ATTRS:
            if getattr(mod, attr, None):
                return True
    return False


def _global_hooks_present() -> bool:
    return any(registry for registry in _GLOBAL_HOOK_REGISTRIES)


def _resolve_device_support() -> None:
    """Fill ``_DEVICE_SUPPORTED`` for every visible device. Called once, at import."""
    try:
        if not torch.cuda.is_available():
            return
        count = torch.cuda.device_count()
    except Exception:  # noqa: BLE001 - no usable driver means nothing is admitted
        return
    for index in range(count):
        try:
            ok = torch.cuda.get_device_capability(index) == _BUILT_FOR_CAPABILITY
        except Exception:  # noqa: BLE001 - an unqueryable device is not one to launch on
            ok = False
        _DEVICE_SUPPORTED[index] = ok
    # A tensor built as plain "cuda" reports the current device's index, so None should not
    # normally reach the predicate; mapped anyway so it is a lookup rather than a miss.
    try:
        _DEVICE_SUPPORTED[None] = _DEVICE_SUPPORTED.get(torch.cuda.current_device(), False)
    except Exception:  # noqa: BLE001
        _DEVICE_SUPPORTED[None] = False


def _embedding_source(device: torch.device, dim: int, max_period: int) -> torch.Tensor:
    """The frequency vector, followed by the exact ``[128, dim]`` embedding table.

    One allocation for both, so the table mode needs no extra kernel argument: the C++ side finds
    the table at offset ``dim // 2`` when the tensor is long enough. Keyed on everything that
    changes it, and deliberately not a registered buffer -- as a buffer it would enter
    ``state_dict()`` and collide with the harness's shared load, and ``.to(dtype)`` could downcast
    it.

    The table is built by the baseline's own ``timestep_embedding``, so gathering a row is
    bit-identical to computing it. That is the point: the table is a speed experiment, not an
    accuracy one, and it cannot drift from the computed path because it *is* the computed path.
    """
    key = (device.type, device.index, dim, max_period, "table")
    cached = _EMB_SOURCE_CACHE.get(key)
    if cached is None:
        half = dim // 2
        freqs = _freqs(device, half, max_period)
        rows = torch.arange(0, _TABLE_ROWS, dtype=torch.int64, device=device)
        table = OasisTimestepEmbedder.timestep_embedding(rows, dim, max_period)
        cached = torch.cat([freqs, table.reshape(-1)])
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        _EMB_SOURCE_CACHE[key] = cached
    return cached


def _freqs(device: torch.device, half: int, max_period: int) -> torch.Tensor:
    key = (device.type, device.index, half, max_period)
    cached = _FREQS_CACHE.get(key)
    if cached is None:
        # The baseline's own expression, evaluated on the target device, so the
        # embedding the kernel builds is bit-identical to timestep_embedding's rather
        # than merely close to it.
        cached = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
            / half,
        )
        if device.type == "cuda":
            # Written once here and read forever after, possibly from another stream.
            # One synchronization at fill time, outside any timed region, makes it safe
            # everywhere afterwards; the baseline gets that ordering for free by
            # rebuilding the vector on every call.
            torch.cuda.current_stream(device).synchronize()
        _FREQS_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Dispatch predicate
# ---------------------------------------------------------------------------
# Resolved once at import so the predicate is attribute reads and integer comparisons.
# torch.compiler.is_compiling is guarded because it has moved between releases.
_is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)

# Forward-mode AD is detected by the *dual level*, which is -1 outside
# ``fwAD.dual_level()`` and >= 0 inside it. Not by ``torch._C._is_fwd_grad_enabled()``:
# that is a global mode flag which reads True in a default process, so guarding on it
# disabled the fast path unconditionally.
_forward_ad = None
try:
    import torch.autograd.forward_ad as _forward_ad
    if not isinstance(getattr(_forward_ad, "_current_level", None), int):
        _forward_ad = None
except Exception:  # noqa: BLE001 - absence just means one guard fewer
    _forward_ad = None


_resolve_device_support()


# Resolved once, so the scored path has no branch: the two cache lookups have the same signature.
def _embedding_source_for(device: torch.device, dim: int) -> torch.Tensor:
    if _EMB_TABLE:
        return _embedding_source(device, dim, _MAX_PERIOD)
    return _freqs(device, dim // 2, _MAX_PERIOD)


def _admits(module: "OasisTimestepEmbedder", t: torch.Tensor) -> bool:
    """Whether the fused path may serve this call. Pure, cheap, and never synchronizes.

    Only tensor and module attributes and Python integers are read -- in particular
    never ``t``'s *values*, which would mean a device-to-host copy inside the timed
    window. Everything this returns False for is served by the baseline computation.

    The whole body runs under ``try``/``except`` and any exception means "not for me". A
    Python ``try`` costs nothing when nothing is raised, and it turns every way a module can
    fail to look like this one -- an ``mlp`` whose entries have no ``.weight``, a subclass
    with an exotic property, a parameter replaced by something that is not a tensor -- into
    a fallback instead of an ``AttributeError`` escaping from ``forward``.
    """
    try:
        return _admits_inner(module, t)
    except Exception:  # noqa: BLE001 - anything unexpected means delegate
        return False


def _admits_inner(module: "OasisTimestepEmbedder", t: torch.Tensor) -> bool:
    if _EXT is None:
        return False
    # An inference-only kernel: it builds no autograd graph and knows nothing about
    # forward-mode duals.
    if torch.is_grad_enabled():
        return False
    if _forward_ad is not None and _forward_ad._current_level >= 0:
        return False
    # Guard the fast path out of a traced graph rather than break the trace.
    if _is_compiling is not None and _is_compiling():
        return False
    # Autocast would change what the reference computes; so would a matmul precision
    # policy other than the TF32 one this kernel emulates. Under "highest" or with
    # allow_tf32 off, the reference becomes exact fp32 and this kernel is the wrong
    # answer, not a faster one.
    #
    # Measured cost of the five global reads above, on this build: dual level 18 ns,
    # is_grad_enabled 32 ns, is_compiling 45 ns, is_autocast_enabled 47 ns,
    # allow_tf32 170 ns -- ~310 ns total, ~2% of the target window. The allow_tf32 read
    # dominates and would be nearly free inside the C++ entry point, but moving it there
    # means signalling "not for me" by raising, and a Python exception costs far more
    # than the 170 ns it would save. So all of them stay here.
    if torch.is_autocast_enabled():
        return False
    if not torch.backends.cuda.matmul.allow_tf32:
        return False

    if (t.dtype is not torch.int64 or not t.is_cuda or t.dim() != 1
            or not t.is_contiguous()):
        return False
    m = t.shape[0]
    if m < _M_MIN or m > _M_MAX:
        return False

    layers = module.mlp
    if len(layers) != 3:
        return False
    lin1, lin2 = layers[0], layers[2]
    # The fused path does not call any of these three modules -- it reads their parameters and
    # reimplements what they compute -- so it is only equivalent to them if they are the exact
    # implementations it was written against. Types are compared with ``is``, not ``isinstance``:
    # a Linear subclass can carry the same weight and bias and mean something different by
    # ``forward``, and the fused call would silently skip it.
    #
    # Same for the embedding: a subclass overriding ``timestep_embedding`` is honoured by the
    # fallback and would be ignored here, which is exactly the drift the fallback exists to
    # prevent.
    if module.timestep_embedding is not OasisTimestepEmbedder.timestep_embedding:
        return False
    if type(lin1) is not Linear or type(lin2) is not Linear or type(layers[1]) is not SiLU:
        return False
    # Hooks observe or rewrite what a module returns, and skipping the module skips them. Any
    # hook on the container, on the three children, or installed globally means the delegated
    # path and the fused path no longer agree.
    if _has_hooks(module) or _global_hooks_present():
        return False
    w1, b1 = lin1.weight, lin1.bias
    w2, b2 = lin2.weight, lin2.bias
    if b1 is None or b2 is None:
        return False
    # Dimensions come from weight.shape, never from .in_features/.out_features: the
    # frozen L1 Linear takes those as constructor arguments and registers only
    # weight/bias, so reading them would raise AttributeError on every scored call.
    if w1.dim() != 2 or w2.dim() != 2:
        return False
    h, f = w1.shape
    if f != module.frequency_embedding_size:
        return False
    # Both the geometry the kernels can load and the numerics they reproduce. The
    # divisibility is what the float4 load scheme needs; the allow-list is what has actually
    # been scored against the baseline. A zero H or F would also pass every modulo and then
    # derive a zero grid, which is an invalid launch rather than a wrong answer -- the
    # allow-list excludes it too.
    if (f, h) not in _VALIDATED_SHAPES:
        return False
    if f % _K_MULTIPLE or h % _K_MULTIPLE:
        return False
    if m * f > _SMEM_FLOATS or m * h > _SMEM_FLOATS:
        return False
    if w2.shape != (h, h) or b1.shape != (h,) or b2.shape != (h,):
        return False
    device = t.device
    # Only one architecture is compiled, so a device of a different capability has no kernel
    # image for these kernels and the launch would raise instead of degrading.
    # A dict lookup: the map is filled at import precisely so this stays one.
    if not _DEVICE_SUPPORTED.get(device.index, False):
        return False
    for p in (w1, b1, w2, b2):
        if (p.dtype is not torch.float32 or p.device != device or not p.is_contiguous()
                or p.data_ptr() % _ALIGN_BYTES):
            return False
    return True


class OasisTimestepEmbedder(nn.Module):
    """Timestep embedder with a fused single-host-call fast path.

    The parameter tree is the baseline's, exactly: the harness shares weights with
    ``load_state_dict(..., strict=False)``, so a renamed or reshaped parameter would
    silently *not* be shared and the module would return garbage rather than fail. The
    ``ModuleList`` indices matter too -- ``tasks/baseline/L3/oasis_dit.py`` reaches
    ``self.t_embedder.mlp[0].weight`` and ``.mlp[2].weight`` directly.

    Nothing is derived from ``weight`` in ``__init__``. The harness moves the module,
    casts its parameters, and only *then* loads the baseline's state dict, so anything
    precomputed at construction would be stale by the time ``forward`` ran.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Linear(frequency_embedding_size, hidden_size, bias=True),
                SiLU(),
                Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """The baseline's expression, unchanged.

        Public API, and the single source of the fallback's embedding -- so the fused
        and delegated paths cannot drift apart.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if _admits(self, t):
            lin1, lin2 = self.mlp[0], self.mlp[2]
            return _EXT.tse_forward(
                t,
                _embedding_source_for(t.device, self.frequency_embedding_size),
                lin1.weight, lin1.bias, lin2.weight, lin2.bias,
                _PACKED_GEOMETRY,
            )
        global DELEGATED_CALLS
        DELEGATED_CALLS += 1
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        if (x.dtype is torch.float32 and type(self.mlp[1]) is SiLU
                and not _has_hooks(self) and not _global_hooks_present()
                and _is_compiling is not None and _is_compiling()):
            # Under torch.compile the fast path guards itself out, and the delegated path then
            # runs the frozen L1 SiLU -- a pybind extension that Dynamo marks as skipped, so
            # ``fullgraph=True`` raises where the *baseline* traces cleanly. That asymmetry is
            # the candidate being worse than what it replaces, not merely unoptimised.
            #
            # F.silu is bitwise identical to what that module computes for fp32: its C++
            # returns at::silu(x) for every dtype other than bfloat16 and float16, and
            # tests/probe_numeric.py checks the equality directly. So this substitution is
            # exact for fp32 -- which is the dtype this operator is scored in -- and is not
            # applied to the low-precision dtypes where the L1 module's approximate form is
            # deliberately different.
            #
            # Guarded on the exact type and on the absence of hooks, because the substitution is
            # only sound for the module it was measured against. Applied unconditionally it is a
            # wrong answer: with ``mlp[1] = nn.ReLU()`` every output changes, by up to 0.236.
            #
            # Only the activation is substituted; the frozen L1 Linear is pure Python plus
            # F.linear, so it traces as it is and stays in the path.
            return self.mlp[2](torch.nn.functional.silu(self.mlp[0](x)))
        for layer in self.mlp:
            x = layer(x)
        return x
