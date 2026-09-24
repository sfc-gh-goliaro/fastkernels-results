"""Tensor concatenation on B200 as one batched contiguous byte copy, launched so
its own launch cost overlaps the kernel before it, with a ``torch.cat`` fallback
for everything the byte-copy kernel does not claim.

What this is
------------
A drop-in replacement for ``torch.cat`` on the shapes YOLOv10's neck produces,
backed by one CUDA C++ translation unit JIT-compiled at import through
``load_inline``. Concatenating contiguous tensors along dimension ``d`` is a
batched contiguous byte copy: flatten to ``outer = prod(shape[:d])`` rows, and
input ``j`` contributes ``row_j = prod(shape[j][d:]) * itemsize`` bytes per row at
byte offset ``off_j = sum_{i<j} row_i`` inside an output row of
``row_out = sum_j row_j`` bytes. Working in bytes rather than elements is what
makes the kernel dtype-agnostic, and it is legitimate precisely because concat
performs no arithmetic -- there is no value to round, promote or canonicalize.

Any input the kernel does not claim -- non-CUDA, mixed dtype, non-contiguous,
zero-numel, gradient-tracking, a live transform or mode stack, an ambiguous memory
format, a dtype ``torch.cat`` itself refuses -- goes to ``torch.cat``, and so does
everything if the build fails. "The claimed regime" below lists every condition and
why it is there.

Where the time goes, and what was actually won
----------------------------------------------
The benchmark's timed window is ``pool.next() + module(xs)``, so two ``copy_``
kernels that re-materialise the inputs into shifted pool slots sit inside every
measurement and are not under this module's control. Roughly 73% of the window is
that harness overhead, so a hypothetically free concat would score about 1.37x.
``torch.cat`` already costs what a single contiguous ``copy_`` of the same bytes
costs, to within 0.05 us on all five scored cases -- there is no second pass and no
wasted traffic to remove. Four of the five cases are latency-bound rather than
bandwidth-bound: 32x more bytes costs only 1.5x more time, and the smallest case
moves 0.59 MiB in 4.19 us, about 2% of achievable bandwidth.

So the memory system was never the opportunity. The launch was.

**The supported claim is a reproduced speedup of about 1.16-1.22x geomean, with no
scored case below 1.10x in any run.** Eleven official runs of the shipped launch
configuration: geomean 1.2194, 1.2174, 1.1599, 1.1623, 1.2214, 1.2184, 1.2184, 1.1894,
1.1812, 1.1935 and 1.2254, with ``max_abs_error == 0.0`` on all five cases in every one
and no scored case below 1.108x in any of them. The spread
tracks device clock state -- the 1.16 pair ran where the *baseline* geomean was
0.01955 ms against 0.01602 ms for the others -- not the candidate.

A paired, randomized, block-balanced A/B of this candidate inside one process puts it
at 1.1900 geomean, with an A/A control (``torch.cat`` against itself) and a B/B
control (this kernel against itself) measured the same way. Two of those ten controls
have intervals excluding zero, which is why the test is the ratio of each margin to
its own worst control rather than whether an interval clears zero: those ratios run
from 19x on the noisiest case to 1488x. ``docs/phase1-findings.md`` has the table.

Nearly all of that comes from one thing, and it is not the kernel body: see
"Programmatic dependent launch" below. The structural advantages over ATen are real
and measurable in NCU but worth close to nothing at these sizes. The numbers, the
protocol and every retired idea are in ``docs/phase1-findings.md``; per-run results
are in ``benchmark.csv`` and ``solutions.jsonl``. One place, so they cannot drift.

Programmatic dependent launch
-----------------------------
The kernel is launched with ``cudaLaunchKernelEx`` carrying
``cudaLaunchAttributeProgrammaticStreamSerialization``, and calls
``cudaGridDependencySynchronize()`` once, ahead of every global access and outside
any divergent branch. This lets the grid be launched and staged while the preceding
kernel in the stream is still running, so the fixed launch and scheduling cost is
paid concurrently instead of afterwards. It is worth about 2.0 us on the four
latency-bound cases and 1.6 us on the largest -- against a whole-operator marginal
cost of 4.1-6.2 us, which is why it dominates everything else here.

That it is *real device time* and not an artifact of where the end event lands was
checked separately: over the five cases the mean CUDA-event gain is 2.318 us and
the mean **wall-clock** gain for the identical loop with a single synchronize at
the end is 2.237 us. The same kernel measured with nothing in flight ahead of it
gains only 0.398 us, which is what "the overlap is with the predecessor" predicts.

Correctness is demonstrated rather than assumed, and the demonstration needed two
attempts. A hazard probe in which the preceding kernel spins for 2 ms and only then
writes the input could not make an *unsynchronized* PDL grid read stale data in any
regime -- one spinning CTA, 148 spinning CTAs, or a large async host-to-device copy
-- so it had no power and proved nothing. The reason is that without an explicit
``cudaTriggerProgrammaticLaunchCompletion()`` the hardware raises the completion
signal only as the predecessor finishes, after its writes. With the predecessor
triggering first and *then* spinning and writing, the probe acquires power and the
result is unambiguous: the unsynchronized control reads stale data, while the
synchronized probe and this module's own ``forward`` both read fresh data on every
trial. ``profile/probe_pdl_hazard.py``.

The one call covers both hazards. It waits for the predecessor grid to complete, so
the input bytes it wrote are visible, and so are its reads finished -- which
matters because the caching allocator may hand back, as this call's output, storage
a predecessor is still reading. The wait is pointer-independent, and the address
arithmetic ahead of it touches only kernel parameters.

Chosen configuration, and why
-----------------------------
16 bytes per thread (one ``uint4``), one vector per thread, 128 threads per block,
``grid.y = outer`` and ``grid.x`` the *sum* over inputs of each input's own block
count.

Block 128 is chosen for SM coverage on the smallest scored case: it has
``outer = 1`` and rows of 6400 and 12800 vectors, so 128 threads launches
50 + 100 = 150 CTAs against 148 SMs where 256 threads would launch 75 and leave
half the device idle. (Block size does not change per-thread register count, so a
register-footprint argument would be worthless here.) ATen also picks 128. A paired
sweep of 20 block-size x vectors-per-thread configurations against the real
candidate found nothing measurably better: the best geomean was 1.2222 at block 64
with two vectors per thread against 1.2205 for the shipped choice, a 0.14%
difference well inside the run-to-run spread, and the shipped choice is the best of
those that cover the SMs.

The input count is a template parameter, not a struct field. The per-CTA prefix
scan sits on the critical path ahead of the first load, and leaving the bound at
``kMaxInputs`` made it eight compares instead of one at ``K = 2``; that alone cost
about 0.08 us on the cases where a thread does nothing but one 16-byte load and one
16-byte store, which was enough to put three of the five cases *below* ``torch.cat``
before it was fixed.

Structure, and how it differs from ATen's ``CatArrayBatchedCopy_vectorized``
---------------------------------------------------------------------------
ATen launches ``grid.y = K`` with ``grid.x`` sized by the *largest* input, so on a
2:1 channel split roughly a quarter of its CTAs launch only to exit, and it
recovers the outer index per element by integer division. Here ``grid.x`` is
``sum_j ceil(vec_j / (block * vecs_per_thread))``, so every CTA issues memory
traffic, and ``grid.y`` carries the row index so no division appears in the address
path at all. Each CTA resolves which input it belongs to exactly once, warp-
uniformly, by scanning a ``K+1``-entry prefix table passed by value in the kernel
parameter bank; the inner loop is then a branch-free ``dst[i] = src[i]`` over a
single source with one bounds check for the partial trailing block.

NCU confirms all of it (``profile/concat_candidate_vs_aten/REPORT.md``): 150 CTAs
against ATen's 200 on the smallest case and 4,800 against 6,400 on the largest, and
100% of this kernel's warps issue a global load against 75% of ATen's, which is the
no-op CTAs measured rather than inferred. 18 registers against 20. Identical global
load and store request and sector counts, L2 theoretical sectors exactly equal to
ideal with zero excessive sectors on both sides, and DRAM reads agreeing to within
512 bytes -- not byte-identical, which the run-to-run range does not support. 16
sectors per request on both sides, i.e. a fully coalesced 128-bit access. And in
SASS exactly one ``LDG.E.128`` and one ``STG.E.128`` per thread with no ``IDIV``,
``MUFU`` or float conversion anywhere in the index path.

On instruction count the shipped instantiation and the plain one differ, and the
figure has to be attributed: ``<2,1,16,true>`` executes **25.0%** fewer warp
instructions than ATen (30,000 against 40,000 on the smallest case) and
``<2,1,16,false>`` executes **26.5%** fewer (29,400). The gap between them is the
one extra instruction the dependency acquire costs -- 50.0 warp instructions per
warp against 49.0 -- so the shipped kernel pays exactly one instruction for the
~2 us the attribute buys.

And it is worth almost nothing. A 48-configuration sweep of an earlier hand-written
variant never beat ``torch.cat``, and capping the grid never helped, because these
shapes are latency-bound. The structure is kept because it costs nothing and closes
the only structural arguments ATen's kernel leaves open -- not because it is where
the speedup came from.

The claimed regime, and why each condition is there
---------------------------------------------------
The custom path runs only when a cheap host-side predicate holds. Everything else
reaches ``torch.cat``, so behaviour outside the claimed regime is unchanged *by
construction* rather than by re-implementation:

* ``1 <= len(xs) <= 8`` -- the prefix table is passed by value.
* every member is exactly ``torch.Tensor`` -- a subclass has semantics of its own.
* all CUDA on one device, one dtype -- one guard, one launch, no promotion.
* a dtype ``torch.cat`` itself accepts. The byte kernel is dtype-agnostic and that
  is the hazard: on torch 2.11 ``cat_cuda`` is not implemented for ``int4``,
  ``uint1``, ``uint2`` or ``uint4``, and copying their bytes would turn a
  ``NotImplementedError`` into a silent success. The supported set is probed at
  import by asking ``torch.cat``, not hard-coded, so it cannot go stale.
* a real allocation. A tensor can be exactly ``torch.Tensor``, CUDA, strided and
  contiguous and still have ``data_ptr() == 0`` -- an escaped functional tensor
  from ``torch._to_functional_tensor`` does -- and zero passes an alignment test.
* an ``int`` dimension. ``torch.cat`` takes a dimension *name* on a named tensor
  and raises its own ``TypeError`` for a float or a bool, so normalising a non-int
  here would produce a different error than the baseline's.
* all classically contiguous -- the byte-slab identity requires it. This also
  routes channels_last inputs to ``torch.cat``, which propagates that memory format
  to its output; a plain contiguous allocation would not.
* memory format unambiguous. A 4-D tensor with ``C == 1`` or ``H*W == 1`` is
  contiguous *and* channels_last at once, and ``torch.cat``'s output stride
  metadata for such inputs is not necessarily a plain contiguous layout.
* shapes identical except on the concat dimension -- malformed input must reach
  ``torch.cat`` and raise *its* error, not a different one from here.
* every ``row_j``, every ``off_j`` and every ``data_ptr`` a multiple of 16 -- the
  128-bit path. A scalar tail path is possible but no captured case needs one, so
  declining is honest and free.
* ``outer <= 65535`` -- the ``grid.y`` limit.
* no member with ``requires_grad`` -- there is no backward formula here, so
  silently detaching a graph would not be a drop-in.
* no live ``TorchFunctionMode`` / ``TorchDispatchMode``, functorch transform,
  forward-mode AD dual level, tracing or compilation. A mode stack would observe
  ``aten::cat`` from the baseline and only ``aten::empty`` from here. The dual level
  needs its own check: a forward-mode dual tensor has ``type(t) is torch.Tensor``
  and ``requires_grad == False``, so the ``requires_grad`` guard does not exclude
  it.
* nothing zero-numel and nothing 0-dimensional. ``torch.cat`` raises on a 0-dim
  member, and it *skips* a legacy ``(0,)`` 1-D empty tensor without applying its
  shape-compatibility check -- ``cat([randn(2,3,4), empty(0)], 1)`` succeeds and
  returns shape ``(2,3,4)``. Declining every zero-numel input is simpler and
  strictly safer than reproducing that rule, and no captured case has one.
* the empty list delegates, because ``torch.cat([])`` raises ``ValueError`` and
  returning an empty tensor instead would not be a drop-in.
* ``len(xs) == 1`` is claimed, and returns a fresh copy: ``torch.cat`` does not
  alias its input for a single-element list, so neither may this.

The predicate costs 3.3-3.5 us of host time per call, measured with
``time.perf_counter``, against 40-80 us of host slack per benchmark iteration --
the 253 MiB L2 flush sits inside the timing loop but outside the CUDA events, and
nothing synchronises inside the loop, so the GPU carries queued work while the CPU
runs ahead. That the measurement can *see* host-side starvation at all was
established by inflating the host cost until it does: the window is unmoved at
+40 us of injected host work per call and moves sharply at +80 us. Nothing in the
predicate allocates, synchronises or calls into CUDA.

The op is deliberately *not* registered through ``torch.library``. Registration
exists to make tracing and ``FakeTensor`` work, and every input under tracing,
compilation or a live mode stack is already delegated to ``torch.cat``, which has
that fidelity by definition. Registering would add a dispatch layer to the hot path
and buy nothing.

Deliberate limitations
----------------------
* Misaligned or non-contiguous inputs are not gathered and there is no scalar tail
  path; they go to ``torch.cat``.
* ``torch.jit.script`` works on the baseline and does not work here -- TorchScript
  cannot resolve the JIT-compiled extension handle. ``torch.jit.trace`` and
  ``torch.compile`` are fine, because both are detected and delegated.
* A profiler trace shows ``concat_rows_kernel`` and ``aten::empty`` where the
  baseline shows ``aten::cat``. Inherent to replacing the kernel; noted because it
  is a real observability difference for anyone reading a trace.
* A single row large enough to need 2^31 CTAs would raise from the extension rather
  than delegate. That needs a contiguous row of about 4.4 TB, so it is a
  portability note rather than a reachable case; the predicate bounds ``outer`` but
  not ``grid.x``.
* The 32-byte access path exists and is byte-exact but is switched off. It measured
  better in a standalone restatement of the kernel and then lost by 3% over two
  official runs; a paired comparison switching only the width inside this kernel
  put it at a tie on four cases and 7% behind on the largest. It stays reachable
  through ``set_launch_config`` so that experiment remains reproducible against the
  shipped code rather than against a lookalike.
* ``launch_count`` / ``launch_geometries`` on the extension, and the
  ``FASTKERNELS_CONCAT_TRACE`` environment variable that dumps them at process
  exit, exist so a test -- and a scored benchmark run -- can prove the custom kernel
  actually ran rather than infer it from output correctness, which delegation would
  also satisfy. ``forward`` never reads them.
* Those diagnostic counters and the launch configuration are plain globals with no
  synchronisation. The benchmark runs each operator in a single-threaded worker, and
  every caller here holds the GIL through the binding, so nothing races in practice --
  but if two Python threads ever called ``forward`` concurrently on a build whose
  binding releases the GIL, ``record_launch`` could lose a count or interleave a
  geometry entry. The counters are diagnostics, so a lost count would misreport
  evidence rather than corrupt a result; ``set_launch_config`` is an experiment hook
  and is not meant to be called concurrently with ``forward`` at all.
* The grid-range guard bounds the CTA count against the *widest* grid any
  configuration ``set_launch_config`` accepts, not against the shipped one, so its
  answer stays correct whatever an experiment has set. That makes it conservative: it
  would decline a row somewhere above 1.1 TB that the shipped configuration could in
  principle have launched.
* ``concat_path`` and ``concat_rows_mismapped`` are test hooks. The first reports
  which path the *real* predicate chooses for a given input without doing any work;
  the second builds a deliberately mis-seeded offset table so a test can confirm the
  byte-exactness comparison has the power to detect a mapping bug. ``forward`` calls
  neither.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

_MAX_INPUTS = 8
_VECTOR_BYTES = 16
_BLOCK = 128
_VECS_PER_THREAD = 1
# Programmatic Dependent Launch. On by measurement: it is worth about 2 us of real
# device time per call and it is the single largest effect found here. See
# "Programmatic dependent launch" in the module docstring for what was measured
# and what the residual risk is.
_USE_PDL = True
# Ceiling on the access width, and the reason it is 16 rather than 32.
#
# The kernel can issue 32 bytes per thread as a single ld.global.v4.u64 /
# st.global.v4.u64 pair whenever every row length and every base pointer is
# 32-byte aligned, which holds for every captured shape. A standalone restatement
# of the kernel measured that as roughly 9% faster on the largest case, and it was
# adopted on that reading. It did not survive: two official runs came out 3% below
# the 16-byte configuration, and a paired comparison that switches *only* the width
# inside this kernel put 32 bytes at a tie on four cases and about 7% behind on the
# largest one -- the opposite of what the standalone comparison said, because those
# restatements also differed in their bounds checks and staging.
#
# So the width stays at 16 and the wide path is reachable only through
# set_launch_config, which is what let the experiment run against the shipped
# kernel rather than against a lookalike. metrics/access_width_ab.json has the
# numbers.
_MAX_ACCESS_BYTES = 16

# Which path a call takes. Returned by the concat_path test hook, never consulted
# by forward. Distinct codes rather than a bool so a test can assert *why* an
# input was declined -- an untested decline reason would otherwise look tested.
PATH_CUSTOM = 0
PATH_NO_EXTENSION = 1
PATH_NOT_A_SEQUENCE = 2
PATH_BAD_LENGTH = 3
PATH_TRANSFORM_ACTIVE = 4
PATH_NOT_PLAIN_TENSOR = 5
PATH_EXOTIC_TENSOR = 6
PATH_NOT_CUDA = 7
PATH_DEVICE_MISMATCH = 8
PATH_DTYPE_MISMATCH = 9
PATH_REQUIRES_GRAD = 10
PATH_ZERO_DIM = 11
PATH_ZERO_NUMEL = 12
PATH_NOT_CONTIGUOUS = 13
PATH_AMBIGUOUS_FORMAT = 14
PATH_DIM_OUT_OF_RANGE = 15
PATH_SHAPE_MISMATCH = 16
PATH_MISALIGNED = 17
PATH_OUTER_TOO_LARGE = 18
PATH_NON_INT_DIM = 19
PATH_DTYPE_UNSUPPORTED = 20
PATH_NO_STORAGE = 21
PATH_GRID_TOO_LARGE = 22

PATH_NAMES = {
    PATH_CUSTOM: "custom",
    PATH_NO_EXTENSION: "delegate:no_extension",
    PATH_NOT_A_SEQUENCE: "delegate:not_a_sequence",
    PATH_BAD_LENGTH: "delegate:bad_length",
    PATH_TRANSFORM_ACTIVE: "delegate:transform_active",
    PATH_NOT_PLAIN_TENSOR: "delegate:not_plain_tensor",
    PATH_EXOTIC_TENSOR: "delegate:exotic_tensor",
    PATH_NOT_CUDA: "delegate:not_cuda",
    PATH_DEVICE_MISMATCH: "delegate:device_mismatch",
    PATH_DTYPE_MISMATCH: "delegate:dtype_mismatch",
    PATH_REQUIRES_GRAD: "delegate:requires_grad",
    PATH_ZERO_DIM: "delegate:zero_dim",
    PATH_ZERO_NUMEL: "delegate:zero_numel",
    PATH_NOT_CONTIGUOUS: "delegate:not_contiguous",
    PATH_AMBIGUOUS_FORMAT: "delegate:ambiguous_memory_format",
    PATH_DIM_OUT_OF_RANGE: "delegate:dim_out_of_range",
    PATH_SHAPE_MISMATCH: "delegate:shape_mismatch",
    PATH_MISALIGNED: "delegate:misaligned",
    PATH_OUTER_TOO_LARGE: "delegate:outer_too_large",
    PATH_NON_INT_DIM: "delegate:non_int_dim",
    PATH_DTYPE_UNSUPPORTED: "delegate:dtype_torch_cat_refuses",
    PATH_NO_STORAGE: "delegate:no_storage",
    PATH_GRID_TOO_LARGE: "delegate:grid_too_large",
}

_CPP_SOURCE = """
#include <torch/extension.h>
#include <vector>
at::Tensor concat_rows(std::vector<at::Tensor> xs, int64_t dim);
at::Tensor concat_rows_mismapped(std::vector<at::Tensor> xs, int64_t dim);
void set_launch_config(int64_t block, int64_t vecs_per_thread, int64_t use_pdl,
                       int64_t max_access_bytes);
std::vector<int64_t> get_launch_config();
int64_t plan_grid_total(std::vector<int64_t> row_bytes, int64_t block,
                        int64_t vecs_per_thread, int64_t access_bytes);
int64_t launch_count();
std::vector<std::vector<int64_t>> launch_geometries();
void reset_counters();
int64_t pdl_supported();
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int kMaxInputs = 8;
constexpr int kVectorBytes = 16;       // the narrowest width the fast path claims
constexpr int kWideVectorBytes = 32;   // used when every row and pointer allows it
constexpr int64_t kMaxOuter = 65535;   // grid.y limit

// The two access widths, as a load and a store each so a multi-vector inner loop
// can still issue all its loads before any of its stores.
//
// 32 bytes needs inline PTX: there is no 32-byte builtin vector type, and a struct
// of two uint4 compiles to two 128-bit instructions rather than the single
// 256-bit instruction this exists to get. Measured worth about 9% on the largest
// scored case and nothing on the four latency-bound ones.
template <int kBytes> struct Access;

template <> struct Access<16> {
    using Reg = uint4;
    static __device__ __forceinline__ Reg load(const void* s) {
        return *static_cast<const uint4*>(s);
    }
    static __device__ __forceinline__ void store(void* d, const Reg& v) {
        *static_cast<uint4*>(d) = v;
    }
};

template <> struct Access<32> {
    struct Reg { uint64_t a, b, c, d; };
    static __device__ __forceinline__ Reg load(const void* s) {
        Reg r;
        asm volatile("ld.global.v4.u64 {%0,%1,%2,%3}, [%4];"
                     : "=l"(r.a), "=l"(r.b), "=l"(r.c), "=l"(r.d)
                     : "l"(s) : "memory");
        return r;
    }
    static __device__ __forceinline__ void store(void* d, const Reg& v) {
        asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};"
                     :: "l"(d), "l"(v.a), "l"(v.b), "l"(v.c), "l"(v.d) : "memory");
    }
};

// The entire launch plan, passed by value so every CTA reads it out of the
// kernel parameter bank instead of chasing a pointer into global memory.
//
// Row extents and offsets are 64-bit because a row can exceed 2^31 vectors. The
// CTA prefix is 32-bit on purpose: it is a grid.x index, which the launch limit
// already caps below 2^31, and every entry is computed in 64-bit on the host and
// range-checked before it is narrowed, so it cannot wrap silently. That keeps the
// per-CTA prefix scan a 32-bit compare, which on the smallest scored case is a
// measurable part of the kernel's critical path.
// Row extents and offsets are counted in whichever access width the launch chose,
// so the kernel never divides or scales them.
struct CopyPlan {
    const uint8_t* src[kMaxInputs];
    int64_t src_row_vecs[kMaxInputs];        // vectors input j contributes per row
    int64_t dst_row_off[kMaxInputs];         // where input j starts in an output row
    int32_t cta_prefix[kMaxInputs + 1];      // cumulative CTA count; [k] is grid.x
    uint8_t* dst;
    int64_t dst_row_vecs;                    // output row stride, in vectors
};

// One CTA copies one contiguous run of one input's slab within one row.
//
// blockIdx.y is the row, so the row index never has to be recovered by integer
// division. blockIdx.x is a global CTA index across all inputs, resolved to an
// input by a linear scan of the prefix table: kInputs-1 predicated adds,
// evaluated once and warp-uniformly, against ATen's per-element division.
//
// The input count is a template parameter rather than a struct field because the
// scan sits on the critical path ahead of the first load. On the benched shapes
// K is 2, so this is one compare; leaving it a runtime bound made it eight, and
// that alone cost about 0.08 us on the cases where a thread does nothing but one
// 16-byte load and one 16-byte store.
template <int kInputs, int kVecsPerThread, int kAccessBytes, bool kUsePdl>
__global__ void concat_rows_kernel(const CopyPlan p) {
    using Wide = Access<kAccessBytes>;
    const int cta = static_cast<int>(blockIdx.x);

    int j = 0;
#pragma unroll
    for (int t = 1; t < kInputs; ++t) {
        j += (cta >= p.cta_prefix[t]) ? 1 : 0;
    }

    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int64_t row_vecs = p.src_row_vecs[j];
    const int64_t block_base = static_cast<int64_t>(cta - p.cta_prefix[j])
                               * blockDim.x * kVecsPerThread;
    const uint8_t* __restrict__ src = p.src[j] + row * row_vecs * kAccessBytes;
    uint8_t* __restrict__ dst = p.dst
        + (row * p.dst_row_vecs + p.dst_row_off[j]) * kAccessBytes;

    if (kUsePdl) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        // Must precede every global access, not only the loads: the preceding
        // kernel in the stream both writes the bytes read here and may still be
        // reading memory the caching allocator has already handed back as the
        // destination. Programmatic serialization lets this grid be scheduled
        // early; this is what makes it wait before touching anything.
        cudaGridDependencySynchronize();
#endif
    }

    // Loads first, then stores. Written as one fused loop instead, each store
    // waits on its own load and the round trips serialize: at two vectors per
    // thread that cost about 2 us on the cases where latency is the whole cost.
    // At one vector per thread this compiles to the same single load and store.
    int64_t index[kVecsPerThread];
    bool active[kVecsPerThread];
    typename Wide::Reg staged[kVecsPerThread];
#pragma unroll
    for (int v = 0; v < kVecsPerThread; ++v) {
        index[v] = block_base + static_cast<int64_t>(threadIdx.x)
                   + static_cast<int64_t>(v) * blockDim.x;
        active[v] = index[v] < row_vecs;
        if (active[v]) staged[v] = Wide::load(src + index[v] * kAccessBytes);
    }
#pragma unroll
    for (int v = 0; v < kVecsPerThread; ++v) {
        if (active[v]) Wide::store(dst + index[v] * kAccessBytes, staged[v]);
    }
}

// Chosen by measurement; mutable so an experiment can sweep them without
// rebuilding. Read once per call on the host side.
int64_t g_block = 128;
int64_t g_vecs_per_thread = 1;
bool g_use_pdl = false;
// The widest access the launcher may pick. Alignment decides per call whether it
// actually can; this is only a ceiling, so an experiment can pin the narrow width.
int64_t g_max_access_bytes = kWideVectorBytes;

// Evidence that the custom kernel ran, rather than an inference from output
// correctness -- delegation to torch.cat produces identical output, so a
// correctness check alone cannot tell the two apart. Recorded here, in the one
// function that launches, and read only through the accessors below.
constexpr int kMaxGeometries = 16;
constexpr int kGeometryFields = 8;
int64_t g_geometry[kMaxGeometries][kGeometryFields];
int g_geometry_count = 0;
int64_t g_launch_count = 0;

void record_launch(const int64_t (&fields)[kGeometryFields]) {
    ++g_launch_count;
    for (int i = 0; i < g_geometry_count; ++i) {
        if (std::memcmp(g_geometry[i], fields, sizeof(fields)) == 0) return;
    }
    if (g_geometry_count < kMaxGeometries) {
        std::memcpy(g_geometry[g_geometry_count++], fields, sizeof(fields));
    }
}

// The one place the CTA arithmetic lives. The launcher calls it with the live
// configuration; the host predicate calls it through plan_grid_total below with the
// most conservative one. Keeping a second copy of the ceil-and-sum in Python is what
// let an earlier version claim the two could not disagree while they were in fact
// independent.
//
// Everything is 64-bit here and nothing is narrowed: the caller decides whether the
// total fits the 32-bit prefix table, so an over-range plan is visible rather than
// wrapped.
struct GridPlan {
    int64_t prefix[kMaxInputs + 1];
    int64_t total_ctas;
};

GridPlan compute_grid_plan(const int64_t* row_bytes, int64_t k, int64_t block,
                           int64_t vecs_per_thread, int64_t access_bytes) {
    TORCH_CHECK(k >= 1 && k <= kMaxInputs, "compute_grid_plan: bad input count ", k);
    TORCH_CHECK(block > 0 && vecs_per_thread > 0 && access_bytes > 0,
                "compute_grid_plan: bad launch configuration");
    GridPlan plan = {};
    const int64_t per_cta = block * vecs_per_thread;
    int64_t total = 0;
    for (int64_t j = 0; j < k; ++j) {
        plan.prefix[j] = total;
        const int64_t vectors = row_bytes[j] / access_bytes;
        total += (vectors + per_cta - 1) / per_cta;
    }
    plan.prefix[k] = total;
    plan.total_ctas = total;
    return plan;
}

template <int kInputs, int kVecsPerThread, int kAccessBytes>
void launch_one(const CopyPlan& p, dim3 grid, int block, bool use_pdl,
                cudaStream_t stream) {
    if (use_pdl) {
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = dim3(static_cast<unsigned>(block), 1, 1);
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
        AT_CUDA_CHECK(cudaLaunchKernelEx(
            &config,
            concat_rows_kernel<kInputs, kVecsPerThread, kAccessBytes, true>, p));
        return;
    }
    concat_rows_kernel<kInputs, kVecsPerThread, kAccessBytes, false>
        <<<grid, static_cast<unsigned>(block), 0, stream>>>(p);
}

template <int kInputs, int kVecsPerThread>
void launch_for_width(const CopyPlan& p, dim3 grid, int block, int access_bytes,
                      bool use_pdl, cudaStream_t stream) {
    switch (access_bytes) {
        case 16:
            launch_one<kInputs, kVecsPerThread, 16>(p, grid, block, use_pdl, stream);
            break;
        case 32:
            launch_one<kInputs, kVecsPerThread, 32>(p, grid, block, use_pdl, stream);
            break;
        default:
            TORCH_CHECK(false, "concat_rows: access width must be 16 or 32 bytes");
    }
}

template <int kInputs>
void launch_for_inputs(const CopyPlan& p, dim3 grid, int block,
                       int vecs_per_thread, int access_bytes, bool use_pdl,
                       cudaStream_t stream) {
    switch (vecs_per_thread) {
        case 1: launch_for_width<kInputs, 1>(p, grid, block, access_bytes, use_pdl, stream); break;
        case 2: launch_for_width<kInputs, 2>(p, grid, block, access_bytes, use_pdl, stream); break;
        case 4: launch_for_width<kInputs, 4>(p, grid, block, access_bytes, use_pdl, stream); break;
        case 8: launch_for_width<kInputs, 8>(p, grid, block, access_bytes, use_pdl, stream); break;
        default:
            TORCH_CHECK(false, "concat_rows: vecs_per_thread must be 1, 2, 4 or 8");
    }
}

void launch_plan(const CopyPlan& p, dim3 grid, int block, int inputs,
                 int vecs_per_thread, int access_bytes, bool use_pdl,
                 cudaStream_t stream) {
#define CONCAT_CASE(N)                                                            \
    case N: launch_for_inputs<N>(p, grid, block, vecs_per_thread, access_bytes,    \
                                 use_pdl, stream); break;
    switch (inputs) {
        CONCAT_CASE(1) CONCAT_CASE(2) CONCAT_CASE(3) CONCAT_CASE(4)
        CONCAT_CASE(5) CONCAT_CASE(6) CONCAT_CASE(7) CONCAT_CASE(8)
        default:
            TORCH_CHECK(false, "concat_rows: input count must be 1 through 8");
    }
#undef CONCAT_CASE
    // Surface a launch failure here rather than in some later, unrelated sync.
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Every TORCH_CHECK below restates a condition the host-side predicate has
// already established. They are assertions, not a second predicate: if the
// predicate ever admits an input the kernel cannot handle, this raises instead
// of corrupting memory.
at::Tensor concat_rows_impl(const std::vector<at::Tensor>& xs, int64_t dim,
                            bool mismap_offsets) {
    const int64_t k = static_cast<int64_t>(xs.size());
    TORCH_CHECK(k >= 1 && k <= kMaxInputs, "concat_rows: bad input count ", k);
    const at::Tensor& first = xs[0];
    const int64_t ndim = first.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "concat_rows: dim ", dim, " out of range");
    TORCH_CHECK(first.is_cuda(), "concat_rows: expected CUDA tensors");
    const int64_t item = first.element_size();

    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= first.size(i);
    TORCH_CHECK(outer >= 1 && outer <= kMaxOuter,
                "concat_rows: outer extent ", outer, " out of range");

    // Before the allocation: without it both the output and the launch would go
    // to the current device rather than to the inputs' device.
    const at::cuda::CUDAGuard guard(first.device());

    int64_t row_bytes[kMaxInputs];
    int64_t row_bytes_total = 0;
    int64_t cat_extent = 0;
    for (int64_t j = 0; j < k; ++j) {
        const at::Tensor& t = xs[j];
        TORCH_CHECK(t.dim() == ndim && t.is_contiguous(),
                    "concat_rows: expected contiguous inputs of equal rank");
        TORCH_CHECK(t.device() == first.device() && t.scalar_type() == first.scalar_type(),
                    "concat_rows: expected one device and one dtype");
        const int64_t bytes = (t.numel() / outer) * item;
        TORCH_CHECK(bytes > 0 && bytes % kVectorBytes == 0,
                    "concat_rows: row of ", bytes, " bytes is not a multiple of 16");
        row_bytes[j] = bytes;
        row_bytes_total += bytes;
        cat_extent += t.size(dim);
    }

    std::vector<int64_t> out_sizes = first.sizes().vec();
    out_sizes[dim] = cat_extent;
    at::Tensor out = at::empty(out_sizes, first.options());

    CopyPlan p = {};
    uint8_t* out_base = reinterpret_cast<uint8_t*>(out.data_ptr());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out_base) % kVectorBytes == 0,
                "concat_rows: output is not 16-byte aligned");

    // Widen to 32-byte accesses only when every row length and every base pointer
    // allows it. The admission rule stays at 16 bytes so the claimed regime does
    // not shrink; this just goes faster when the shapes happen to cooperate, which
    // on the captured shapes they always do.
    int64_t access = g_max_access_bytes;
    if (access > kVectorBytes) {
        bool wide = (reinterpret_cast<uintptr_t>(out_base) % kWideVectorBytes) == 0;
        for (int64_t j = 0; j < k && wide; ++j) {
            wide = (row_bytes[j] % kWideVectorBytes == 0)
                   && (reinterpret_cast<uintptr_t>(xs[j].const_data_ptr())
                       % kWideVectorBytes == 0);
        }
        if (!wide) access = kVectorBytes;
    }

    p.dst = out_base;
    p.dst_row_vecs = row_bytes_total / access;

    int64_t byte_off = 0;
    for (int64_t j = 0; j < k; ++j) {
        const void* base = xs[j].const_data_ptr();
        TORCH_CHECK(reinterpret_cast<uintptr_t>(base) % access == 0,
                    "concat_rows: input ", j, " is not aligned to ", access);
        p.src[j] = reinterpret_cast<const uint8_t*>(base);
        p.src_row_vecs[j] = row_bytes[j] / access;
        p.dst_row_off[j] = byte_off / access;
        byte_off += row_bytes[j];
    }

    if (mismap_offsets) {
        // Deliberately wrong: write the slabs into the row in reverse order.
        // Equal row sizes keep the permutation inside the output and leave every
        // byte written, so the only thing wrong is the mapping -- which is
        // exactly what a byte-exactness comparison has to be able to catch.
        for (int64_t j = 1; j < k; ++j) {
            TORCH_CHECK(row_bytes[j] == row_bytes[0],
                        "concat_rows_mismapped: needs equal row sizes");
        }
        for (int64_t j = 0; j < k; ++j) {
            p.dst_row_off[j] = (k - 1 - j) * p.src_row_vecs[0];
        }
    }

    const int64_t block = g_block;
    const int64_t vecs_per_thread = g_vecs_per_thread;
    // Same helper the host predicate uses, called here with the live configuration.
    // Its 64-bit total is range-checked before being narrowed into the plan's 32-bit
    // prefix table, so a grid that would not fit raises rather than wrapping into a
    // wrong CTA-to-input mapping. The predicate declines such an input before this
    // function is reached, so reaching the check means the predicate has a hole.
    const GridPlan grid_plan =
        compute_grid_plan(row_bytes, k, block, vecs_per_thread, access);
    const int64_t total_ctas = grid_plan.total_ctas;
    TORCH_CHECK(total_ctas > 0 && total_ctas <= 2147483647,
                "concat_rows: grid.x ", total_ctas, " out of range");
    for (int64_t j = 0; j <= k; ++j) {
        p.cta_prefix[j] = static_cast<int32_t>(grid_plan.prefix[j]);
    }

    const dim3 grid(static_cast<unsigned>(total_ctas),
                    static_cast<unsigned>(outer), 1);
    const int64_t fields[kGeometryFields] = {
        k, outer, p.dst_row_vecs, total_ctas, block, vecs_per_thread,
        g_use_pdl ? 1 : 0, access};
    record_launch(fields);
    launch_plan(p, grid, static_cast<int>(block), static_cast<int>(k),
                static_cast<int>(vecs_per_thread), static_cast<int>(access),
                g_use_pdl, at::cuda::getCurrentCUDAStream());
    return out;
}

}  // namespace

at::Tensor concat_rows(std::vector<at::Tensor> xs, int64_t dim) {
    return concat_rows_impl(xs, dim, /*mismap_offsets=*/false);
}

// Test hook. Same plan builder and same launcher as concat_rows, with the output
// offsets deliberately permuted, so a test can show its byte comparison detects
// a mapping bug rather than merely passing. Never called by forward.
at::Tensor concat_rows_mismapped(std::vector<at::Tensor> xs, int64_t dim) {
    return concat_rows_impl(xs, dim, /*mismap_offsets=*/true);
}

// Read-only view of the launcher's own CTA arithmetic, so the host predicate can ask
// the same question the launcher will answer without a second implementation of it.
int64_t plan_grid_total(std::vector<int64_t> row_bytes, int64_t block,
                        int64_t vecs_per_thread, int64_t access_bytes) {
    return compute_grid_plan(row_bytes.data(),
                             static_cast<int64_t>(row_bytes.size()), block,
                             vecs_per_thread, access_bytes).total_ctas;
}

void set_launch_config(int64_t block, int64_t vecs_per_thread, int64_t use_pdl,
                       int64_t max_access_bytes) {
    TORCH_CHECK(max_access_bytes == 16 || max_access_bytes == 32,
                "set_launch_config: max_access_bytes must be 16 or 32");
    TORCH_CHECK(block > 0 && block <= 1024 && block % 32 == 0,
                "set_launch_config: block must be a positive multiple of 32 up to 1024");
    TORCH_CHECK(vecs_per_thread == 1 || vecs_per_thread == 2 ||
                vecs_per_thread == 4 || vecs_per_thread == 8,
                "set_launch_config: vecs_per_thread must be 1, 2, 4 or 8");
    if (use_pdl) {
        // The device-side dependency sync only compiles for sm_90 and above.
        // Requesting programmatic serialization without it would be a race.
        const auto* props = at::cuda::getCurrentDeviceProperties();
        TORCH_CHECK(props->major >= 9,
                    "set_launch_config: programmatic dependent launch needs sm_90+");
    }
    g_block = block;
    g_vecs_per_thread = vecs_per_thread;
    g_use_pdl = use_pdl != 0;
    g_max_access_bytes = max_access_bytes;
}

std::vector<int64_t> get_launch_config() {
    return {g_block, g_vecs_per_thread, g_use_pdl ? 1 : 0, g_max_access_bytes};
}

int64_t pdl_supported() {
    const auto* props = at::cuda::getCurrentDeviceProperties();
    return props->major >= 9 ? 1 : 0;
}

int64_t launch_count() { return g_launch_count; }

std::vector<std::vector<int64_t>> launch_geometries() {
    std::vector<std::vector<int64_t>> out;
    out.reserve(g_geometry_count);
    for (int i = 0; i < g_geometry_count; ++i) {
        out.emplace_back(g_geometry[i], g_geometry[i] + kGeometryFields);
    }
    return out;
}

void reset_counters() {
    g_launch_count = 0;
    g_geometry_count = 0;
}
"""

# Set by _load_extension(). A build failure leaves the handle None and the reason
# in _BUILD_ERROR, which is the only record of it -- import must not raise, or
# the operator is reported as a runtime error instead of falling back.
_EXTENSION = None
_BUILD_ERROR: str | None = None
_BUILD_DIR: str | None = None
_EXTENSION_NAME: str | None = None


def _build_identity() -> tuple[str, str]:
    """The extension name, and the identity string it hashes.

    Hashing the sources alone is not enough: the same source compiled with
    different flags, for a different architecture, or against a different torch /
    CUDA / Python ABI produces a different ``.so``, and reusing a stale one from
    the pinned directory would be worse than rebuilding.
    """
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    identity = "\x00".join([
        _CUDA_SOURCE,
        _CPP_SOURCE,
        repr(_CFLAGS),
        repr(_CUDA_CFLAGS),
        arch,
        torch.__version__,
        str(torch.version.cuda),
        f"{sys.version_info.major}.{sys.version_info.minor}",
        str(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "")),
    ])
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"yolov10_concat_sm{arch.replace('.', '')}_{digest}", identity


_CFLAGS = ["-O3"]
# No fast-math flag: this kernel never enters the float domain, so relaxing float
# semantics could only mislead a future reader into thinking it might.
_CUDA_CFLAGS = ["-O3"]


def _load_extension() -> None:
    """Compile at import time, into a pinned directory, without ever raising.

    Import time is the right moment. The benchmark snapshots its integrity before
    importing the candidate and fails a candidate whose thread count grows across
    the timed region, so ninja's workers must be long gone by then; a first-call
    or lazy build would put them inside the measured window.

    A build killed part-way leaves ``<build_dir>/lock`` behind and the next
    import will block on it. Deleting that file is the recovery.
    """
    global _EXTENSION, _BUILD_ERROR, _BUILD_DIR, _EXTENSION_NAME
    if not torch.cuda.is_available():
        _BUILD_ERROR = "no CUDA device available at import; using torch.cat"
        return
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        name, _ = _build_identity()
        build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / name
        build_dir.mkdir(parents=True, exist_ok=True)
        _BUILD_DIR = str(build_dir)
        _EXTENSION_NAME = name
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        _EXTENSION = load_inline(
            name=name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=[
                "concat_rows",
                "concat_rows_mismapped",
                "set_launch_config",
                "get_launch_config",
                "plan_grid_total",
                "launch_count",
                "launch_geometries",
                "reset_counters",
                "pdl_supported",
            ],
            extra_cflags=_CFLAGS,
            extra_cuda_cflags=_CUDA_CFLAGS,
            build_directory=str(build_dir),
            verbose=False,
        )
        _EXTENSION.set_launch_config(_BLOCK, _VECS_PER_THREAD,
                                     1 if _USE_PDL else 0, _MAX_ACCESS_BYTES)
    except Exception as exc:  # noqa: BLE001 - degrading to torch.cat beats not importing
        _EXTENSION = None
        _BUILD_ERROR = f"{type(exc).__name__}: {exc}"
    finally:
        # Leaving this mutated would change how unrelated extensions compile.
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list


_load_extension()


def _probe_cat_dtypes() -> frozenset:
    """Which dtypes ``torch.cat`` itself accepts on CUDA -- asked, not assumed.

    The byte-copy kernel is dtype-agnostic, and that is exactly the hazard: it will
    cheerfully concatenate a dtype ``torch.cat`` refuses, turning an exception into
    a silent success. On torch 2.11 ``cat_cuda`` is not implemented for ``int4``,
    ``uint1``, ``uint2`` or ``uint4``, while ``float4_e2m1fn_x2`` is fine -- not a
    distinction worth hard-coding. Probing at import ties the claim to what this
    build actually supports rather than to a list that goes stale.

    Returns an empty set if anything goes wrong, which makes the predicate decline
    everything and leaves ``torch.cat`` in charge.
    """
    if _EXTENSION is None:
        return frozenset()
    supported = set()
    try:
        # Some dtypes warn when they are merely allocated (ComplexHalf is
        # "experimental"). Importing this module must not emit that noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in dir(torch):
                dtype = getattr(torch, name, None)
                if not isinstance(dtype, torch.dtype):
                    continue
                try:
                    probe = torch.empty(2, dtype=dtype, device="cuda")
                    torch.cat([probe, probe], 0)
                except Exception:  # noqa: BLE001, PERF203 - refusal is the answer
                    continue
                supported.add(dtype)
    except Exception:  # noqa: BLE001 - a probe failure must not break import
        return frozenset()
    return frozenset(supported)


_CAT_DTYPES = _probe_cat_dtypes()

# Bound once so the predicate below does no attribute lookups on the hot path.
_len_function_stack = torch._C._len_torch_function_stack
_len_dispatch_stack = torch._C._len_torch_dispatch_stack
_peek_functorch_stack = torch._C._functorch.peek_interpreter_stack
_forward_ad = torch.autograd.forward_ad
_is_tracing = torch.jit.is_tracing
_is_compiling = torch.compiler.is_compiling
_Tensor = torch.Tensor
_strided = torch.strided


_INT32_MAX = 2147483647
# The smallest work-per-CTA any configuration set_launch_config will accept: block is a
# positive multiple of 32 and vectors-per-thread is at least 1. Bounding the grid against
# this rather than against the shipped configuration keeps the predicate's answer correct
# no matter what an experiment has set, since fewer bytes per CTA means more CTAs.
_MIN_VECTORS_PER_CTA = 32


def plan_grid(row_bytes, block: int = _MIN_VECTORS_PER_CTA,
              vecs_per_thread: int = 1,
              access_bytes: int = _VECTOR_BYTES) -> int:
    """Total CTAs the launcher will ask for, given each input's per-row byte count.

    A thin view of the extension's own ``compute_grid_plan``, which is the function the
    launcher itself calls. There is deliberately no Python implementation of the
    ceil-and-sum: an earlier version had one, and its docstring claimed the predicate and
    the launcher could not disagree while they were in fact two independent copies of the
    formula.
    """
    if _EXTENSION is None:
        raise RuntimeError("plan_grid needs the extension; it is the launcher's own "
                           "arithmetic, not a reimplementation of it")
    return _EXTENSION.plan_grid_total(list(row_bytes), block, vecs_per_thread,
                                      access_bytes)


def _grid_safe_total_bytes(limit: int = _INT32_MAX) -> int:
    """Total input bytes below which no accepted configuration can exceed *limit* CTAs.

    A sufficient condition, not a second copy of the planner. For each input,
    ``ceil(bytes_j / (access * per_cta)) <= bytes_j / (access * per_cta) + 1``, so with
    the narrowest access and the smallest work-per-CTA the total is at most
    ``sum_bytes / (16 * 32) + K``. Requiring that to stay under the limit is therefore
    strictly stronger than requiring the real total to.

    This exists so the hot path does not cross into the extension on every call: above
    this many bytes -- about 1.1 TB, which no captured or constructible input approaches
    -- the predicate asks the launcher's own helper for the exact answer. A self-check
    asserts the two never disagree over randomized row-byte vectors.
    """
    return (limit - _MAX_INPUTS) * _VECTOR_BYTES * _MIN_VECTORS_PER_CTA


def _classify(xs, dim: int) -> int:
    """Which path an input takes: ``PATH_CUSTOM``, or why it is declined.

    Every check is a host-side attribute read. Nothing here allocates, calls into
    CUDA or synchronises, and the cost is measured rather than assumed -- see
    ``docs/phase1-findings.md``. The benchmark's 253 MiB L2 flush sits inside the
    timing loop but outside the CUDA events, and nothing synchronises inside the
    loop, so the GPU carries tens of microseconds of queued work per iteration
    while the CPU runs ahead; that is the slack this cost lives in.
    """
    if _EXTENSION is None:
        return PATH_NO_EXTENSION
    if type(xs) is not list and type(xs) is not tuple:
        return PATH_NOT_A_SEQUENCE
    k = len(xs)
    if k < 1 or k > _MAX_INPUTS:
        return PATH_BAD_LENGTH

    # A live transform or mode stack would observe aten::cat from the baseline and
    # only aten::empty from here. The dual-level check is not redundant with the
    # requires_grad check below: a forward-mode dual tensor is exactly
    # torch.Tensor with requires_grad False.
    if (_len_function_stack() or _len_dispatch_stack()
            or _peek_functorch_stack() is not None
            or _forward_ad._current_level >= 0
            or _is_tracing() or _is_compiling()):
        return PATH_TRANSFORM_ACTIVE

    # torch.cat accepts a dimension *name* on a named tensor, and raises its own
    # TypeError for a float or a bool. Normalising a non-int here would raise a
    # different error than the baseline does, so anything but an int delegates.
    if dim.__class__ is not int:
        return PATH_NON_INT_DIM

    first = xs[0]
    if type(first) is not _Tensor:
        return PATH_NOT_PLAIN_TENSOR
    if first.dtype not in _CAT_DTYPES:
        return PATH_DTYPE_UNSUPPORTED
    ndim = first.dim()
    if ndim == 0:
        return PATH_ZERO_DIM
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        return PATH_DIM_OUT_OF_RANGE

    device = first.device
    dtype = first.dtype
    first_shape = first.shape
    head = first_shape[:dim]
    tail = first_shape[dim + 1:]

    outer = 1
    for extent in head:
        outer *= extent
    if outer < 1 or outer > 65535:
        return PATH_OUTER_TOO_LARGE

    item = first.element_size()
    check_format = ndim == 4 or ndim == 5
    row_bytes = []
    total_row_bytes = 0
    for t in xs:
        if type(t) is not _Tensor:
            return PATH_NOT_PLAIN_TENSOR
        if not t.is_cuda:
            return PATH_NOT_CUDA
        if t.device != device:
            return PATH_DEVICE_MISMATCH
        if t.dtype is not dtype:
            return PATH_DTYPE_MISMATCH
        if t.requires_grad:
            return PATH_REQUIRES_GRAD
        if t.layout is not _strided or t.is_nested or t.is_conj() or t.is_neg() \
                or t.has_names():
            return PATH_EXOTIC_TENSOR
        shape = t.shape
        if len(shape) == 0:
            return PATH_ZERO_DIM
        # Before the shape checks, because the case that matters most is the legacy
        # (0,) 1-D empty tensor, which torch.cat skips *without* applying its
        # shape-compatibility rule; reporting it as a rank disagreement instead
        # would name the wrong reason for declining.
        numel = t.numel()
        if numel == 0:
            return PATH_ZERO_NUMEL
        if len(shape) != ndim:
            return PATH_SHAPE_MISMATCH
        if shape[:dim] != head or shape[dim + 1:] != tail:
            return PATH_SHAPE_MISMATCH
        if not t.is_contiguous():
            return PATH_NOT_CONTIGUOUS
        # A 4-D tensor whose channel count is 1, or whose trailing extents
        # multiply to 1, is classically contiguous *and* channels_last contiguous
        # at the same time, and torch.cat's output stride metadata for such inputs
        # is not necessarily a plain contiguous layout. Checked per input because
        # the concat dimension is exactly the one that may differ between them.
        # The arithmetic form is cross-checked against
        # is_contiguous(memory_format=...) over a shape grid in the self-check.
        if check_format and (shape[1] == 1 or numel == shape[0] * shape[1]):
            return PATH_AMBIGUOUS_FORMAT
        # A tensor can be exactly torch.Tensor, CUDA, strided and contiguous and
        # still have no storage: an escaped functional tensor from
        # torch._to_functional_tensor has data_ptr() == 0, and zero passes an
        # alignment test. Reading from it would be an illegal access.
        pointer = t.data_ptr()
        if pointer == 0:
            return PATH_NO_STORAGE
        bytes_in_row = (numel // outer) * item
        if bytes_in_row % _VECTOR_BYTES or pointer % _VECTOR_BYTES:
            return PATH_MISALIGNED
        row_bytes.append(bytes_in_row)
        total_row_bytes += bytes_in_row

    # The launcher's prefix table and grid.x are 32-bit. Checking the bound here rather
    # than in the extension is what keeps the promise that a declined input reaches
    # torch.cat unchanged: the extension can only raise, and by the time it knows the
    # count it has already allocated the output.
    #
    # The byte comparison is a sufficient condition (see _grid_safe_total_bytes) that
    # keeps the common case out of the extension entirely; only above it does the
    # predicate ask the launcher's own helper for the exact total.
    if total_row_bytes > _grid_safe_total_bytes(_INT32_MAX) \
            and plan_grid(row_bytes) > _INT32_MAX:
        return PATH_GRID_TOO_LARGE
    return PATH_CUSTOM


def concat_path(xs, dim: int = 1) -> tuple[int, str]:
    """Report which path ``forward`` would take, and why, without doing any work.

    A test hook, and the only way to observe the choice: delegation produces
    output identical to the custom path, so a correctness check cannot tell them
    apart and an untested decline reason would look tested. It calls the same
    ``_classify`` ``forward`` calls, so what it reports is the real decision.
    ``forward`` must never call this.
    """
    code = _classify(xs, dim)
    return code, PATH_NAMES.get(code, f"unknown:{code}")


def build_report() -> dict[str, object]:
    """Inspectable build and launch state, so a silent fallback stays diagnosable.

    ``launch_count`` and ``launch_geometries`` are the evidence that the custom
    kernel actually ran. Each geometry row is
    ``(inputs, outer, output_row_vectors, grid_x, block, vectors_per_thread, pdl)``.
    ``forward`` never reads any of this.
    """
    report: dict[str, object] = {
        "extension_loaded": _EXTENSION is not None,
        "build_error": _BUILD_ERROR,
        "extension_name": _EXTENSION_NAME,
        "build_dir": _BUILD_DIR,
        "vector_bytes": _VECTOR_BYTES,
        "max_inputs": _MAX_INPUTS,
    }
    if _EXTENSION is not None:
        block, vecs_per_thread, use_pdl, max_access = _EXTENSION.get_launch_config()
        report.update({
            "block": block,
            "vecs_per_thread": vecs_per_thread,
            "use_pdl": bool(use_pdl),
            "max_access_bytes": max_access,
            "pdl_supported": bool(_EXTENSION.pdl_supported()),
            "launch_count": _EXTENSION.launch_count(),
            "launch_geometries": _EXTENSION.launch_geometries(),
        })
    return report


def _install_trace_dump() -> None:
    """Write ``build_report()`` at process exit when asked to by the environment.

    The benchmark runs each operator in a worker process of its own, so this is
    how a scored run can be made to prove the custom kernel ran on every case
    rather than having silently fallen back. Off unless the variable is set, and
    it registers no threads.
    """
    target = os.environ.get("FASTKERNELS_CONCAT_TRACE")
    if not target:
        return

    def dump() -> None:
        try:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), **build_report()}) + "\n")
        except Exception:  # noqa: BLE001 - diagnostics must never break a run
            pass

    atexit.register(dump)


_install_trace_dump()


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        d = self.d
        if _classify(xs, d) == PATH_CUSTOM:
            return _EXTENSION.concat_rows(xs, d if d >= 0 else d + xs[0].dim())
        return torch.cat(xs, d)
