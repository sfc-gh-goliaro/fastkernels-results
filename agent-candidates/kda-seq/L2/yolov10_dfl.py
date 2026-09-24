"""YOLOv10 Distribution Focal Loss layer, fused into a single CUDA kernel.

The reference is `F.softmax` over a transposed view followed by a `1x1` `F.conv2d`.
That is three kernels: `F.softmax` cannot reduce a strided non-final axis in place,
so it first materializes a contiguous copy, then runs the three-pass spatial softmax
over it, and the pointwise convolution then becomes a skinny `M=1, K=c1` GEMM.

Unfolding the index algebra shows none of that data movement is necessary. With
`row = b*4 + j`:

    x.view(b, 4, c1, a)[b, j, c, k] == x[b, j*c1 + c, k]        (input  base x + row*c1*a)
    m            = max_c x[b, j*c1+c, k]
    out[b, j, k] = sum_c w[c] * exp(x[b, j*c1+c, k] - m)
                   / sum_c exp(x[b, j*c1+c, k] - m)             (output base out + row*a)

The `view`/`transpose` is addressing, not movement, and the reduction is only `c1`
long, so one thread owns a whole reduction in registers: no cross-lane shuffle, no
shared-memory staging of the input, and one pass over HBM. Measured kernel counts per
`forward` call: 1 for this kernel, 2 for the frozen-L1 fallback composition, 3 for the
reference. HBM traffic is 4.57 MB against the fallback's 13.17 MB (b=4).

## What the profile actually said (`profile/dfl_v2_vec1/REPORT.md`)

The draft's cost model predicted a balanced kernel: 0.70 us of memory time against
0.6 us of `ex2.approx`. Both numbers are right and both are irrelevant, because the
kernel is neither bandwidth- nor MUFU-bound. NCU on the captured shapes: DRAM read at
4.8% of peak, MUFU 39.1%, SM speed-of-light well short of a limit -- nothing saturated --
with `long_scoreboard` dominating the stalls.

Reproducing the reference's probability rounding costs a second exponential per element,
which took MUFU from 24% to 39% and the kernel from 8.6 us to 11.7 us at b=4 -- and moved
the benched speedup not at all, because the window is floor-dominated (see below). That is
the correct trade: the fp32-throughout form was faster and wrong.

The limiter is that there is not enough work to fill the machine. `waves/SM` is 0.446
at b=4 and 0.111 at b=1: 134400 output elements over 148 SMs is 908 threads per SM
against a 2048-thread capacity, and b=1 has 227. That is a property of the operator at
these shapes, not of the kernel, and it is why the vector width -- which *sets* the
thread count at `rows*a/V` -- turned out to be the only knob that mattered. See
`select_max_width`, where the measured width table lives.

## Harness floors, which bound the achievable speedup from above

The bench times `_ShiftingPool.next()` -- a full device-to-device copy of the input --
*plus* the module call, between two CUDA events. Both terms are additive in
`baseline_ms` and `candidate_ms` alike and neither is ours. Measured with the bench's
own `_time_module` (`profile/probe_decompose.py`): a 7.66 us event/launch floor, and a
pool copy of 6.64 us (b=4) / 8.75 us (b=1).

So a module that returns a preallocated tensor and launches *nothing* scores 5.04x
(b=4) and 3.49x (b=1). That is the ceiling for any implementation of this operator
here. This one scores 2.43x and 2.33x, i.e. 62-67% of a ceiling no kernel work can
move; the two benched scenarios are the only ones scored, and they deduplicate from the
three occurrences `docs/shapes.md` lists because the bench keys cases on
`(init_args, forward_args)` and the two `[4, 64, 8400]` entries are byte-identical.

## Numerics

The kernel **reproduces the reference's arithmetic rather than improving on it**, and that is a
deliberate reversal of the original design.

The reference is two ops, and the boundary between them is observable: `F.softmax` writes its
probabilities out in the input dtype, and only then does `F.conv2d` contract them against the
weight. So the reference computes `sum_c w[c] * dtype(e_c/den)` with an fp32 accumulation, and
this kernel computes exactly that -- max in fp32, `expf` in fp32, the normalizer in fp32, then
each probability rounded to `scalar_t` and converted back before an fp32 FMA against the weight,
accumulated in channel order.

The original design instead kept everything in fp32 and divided once at the end, on the argument
that this is strictly more accurate than the reference and therefore safe. Both halves of that
argument were wrong:

  * You cannot bound the *difference* between two computations by the error of one of them. The
    honest coarse bound sums the intermediate rounding and both output roundings, ~`3u` relative,
    which closes comfortably for fp16 (`u = 2^-11`, 0.15% against a 1% `rtol`) but not for bf16
    (`u = 2^-8`, 1.17%, already over `rtol`).
  * More importantly, "more accurate" is not the contract. The gap between the two forms is not
    small-and-bounded but *unbounded*, because a caller may load any weight through
    `load_state_dict` and the module must still match the reference. With large alternating-sign
    weights the weighted sum cancels, so the reference's per-probability rounding is amplified
    instead of averaged away. Measured on the captured `[1, 64, 8400]` shape with the perfectly
    valid weight `[60000, -60000] * 8`: the fp32-throughout form matched only 98.4% of elements
    at `max_abs_error` 32.0 -- a scenario failure on an accepted input. Even `[1000, -1000] * 8`
    failed at 98.75%.

Guarding on weight values is not available: the harness fills the weight after construction, so
the route may not depend on them. Reproducing the rounding is the only correct fix, and it costs a
second exponential per element and no extra launch.

`expf` rather than `__expf`, also a reversal. The design called `__expf` load-bearing because the
`ex2.approx` count was projected to be the same order as the memory time. The profile falsified
the premise -- MUFU sits at 24% and the kernel is latency-bound, not throughput-bound -- and
`__expf`'s ~2 ulp is enough to flip a probability across an fp16 rounding boundary, which a large
weight then amplifies into a visible error. Measured with `[60000, -60000] * 8`: `__expf` left one
element at 141% of its bound, `expf` brings the worst to 6.8%. On the benched scenarios `expf`
makes the fused result **bit-identical to the reference** (`max_abs_error` exactly 0.0, down from
7.81e-03), and across every admitted dtype, `c1`, `b` and `a` with random weights the worst
element now uses 0.0% of its error bound. The remaining residual on pathological weights is fp32
accumulation *order* -- this kernel sums in channel order, the reference's GEMM does not -- which
is bounded by fp32 epsilon against the largest partial sum.

`NaN` cannot distinguish correct from incorrect here: the bench's comparator rejects `NaN` in
*either* output. `+inf` is the case worth knowing about, and the draft got it wrong -- for the
maximal element `x - m` is `inf - inf`, i.e. `NaN`, in both implementations, not `exp(0) = 1`.
Measured: both produce `NaN` in exactly the same places.

fp32 inputs are still not admitted. The reference's `F.conv2d` may reach TF32 through cuDNN
against a much tighter `(1e-5, 1e-3)` tolerance, so the reference can be the less accurate side in
a way this kernel cannot reproduce by matching rounding. No capture uses fp32.

## Alternatives, with what measuring them showed

Two of these were built as template flags so both sides could be measured from one
build, as the design intended. Both turned out to be non-wins, and both flags were then
removed rather than left as paths nothing exercises; the numbers are kept here and in
`profile/dfl_v1_fused/` and `profile/dfl_v3_smemw/` so a later phase need not redo them.

  * **Reload from L1 instead of staging in registers.** Re-reading `x` in the
    exponential pass trades load instructions for registers at unchanged HBM traffic
    (the block footprint stays inside L1). Measured difference: inside run-to-run noise
    at every width. Reopen if a profile shows staging spilling -- which is checked on
    every build by `profile/probe_ptxas.py`, currently 80 registers worst case and zero
    spills across every selectable instantiation.
  * **Per-thread weight reads instead of shared memory plus a barrier.** Removes the
    `barrier` stall entirely (0.85 per issue-active cycle at b=4, 1.34 at b=1) and is
    still slower: 8.74 us against 8.51 (b=4), 7.42 against 7.17 (b=1). The extra loads
    cost more than the barrier. Reopen if `barrier` ever dominates the stall histogram
    instead of trailing `long_scoreboard`.
  * **Splitting the `c1` reduction across lanes.** The knob that attacks the actual
    limiter: 2 lanes per output at b=4 or 4 at b=1 takes `waves/SM` from 0.446 to 0.892
    and from 0.111 to 0.446. It needs three shuffle reductions (the max and both sums),
    which may consume the gain -- expected 5-20% of kernel time but only 1-5% of the
    benched number. Excluded from this phase by the plan. Composes with the staging knob
    above: at 2-4 lanes each lane stages only 8-4 exponentials.
  * **256-bit loads** (`ld.global.v4.u64`, in use on sm_100). Implies `V = 16`, hence
    16x fewer threads -- the direction the width sweep says is wrong -- plus 128
    registers of staged data. Reopen only for a shape with work many times the SM
    capacity.
  * **`ld.global.nc.L1::no_allocate`** for the streaming input. L1 hit rate is already
    0.67%, so there is no reuse to protect and no thrash to remove; the KernelWiki
    result behind this knob came from a kernel with two streams of differing reuse.
    Reopen if `lg_throttle` or L1TEX queue pressure ever registers.
  * **`__launch_bounds__` with a min-blocks argument, or `-maxrregcount`.** At 32
    registers the shipped path is at 100% theoretical occupancy and the reported
    occupancy limit is `warps`, not `registers`, so there is nothing to win. The
    `__launch_bounds__(kMaxBlock)` that *is* present is load-bearing for a different
    reason -- see its comment.
  * **Online single-pass softmax.** The reference-order rounding this kernel now
    reproduces needs the completed normalizer before any probability can be rounded, so
    a single-pass formulation cannot express it at all. Reopened only if the contract
    ever stops requiring rounding parity.
  * **Storing the exponentials instead of recomputing them -- the top-ranked next knob.**
    The weighted pass re-evaluates `expf` rather than keeping the `C1*V` values live from
    the normalizer pass. Storing them halves the transcendental count, which is the
    largest reducible work component at XU 39%. An earlier version of this note said the
    register cost made it the wrong trade; the independent ranking in
    `profile/runs/task18_knob_ranking.md` shows that is wrong on two counts. The cost
    probably does not exist -- each staged fp16 value already occupies a full 32-bit
    register, so overwriting it with an fp32 exponential adds none:

        staged[c] = expf(staged[c] - m);   // the raw value is dead at this point
        den[t]   += staged[c];

    And even at a pessimistic 48 registers, an SM would hold 10 blocks while b=4 supplies
    7.1 and b=1 supplies 1.8, so both grids sit below the register-limited ceiling and the
    nominal occupancy drop is slack that cannot be reached. Expected 10-25% of kernel time
    at b=4, 2-6% of the benched window. Not taken here because it is a kernel change and
    its expected effect sits inside the benched ratio's measured run-to-run spread; judge
    it on a paired same-run CUDA-event A/B, not on an NCU duration.
  * **A grid-stride or persistent loop.** Reduces block count, i.e. concurrency, which
    is already the scarce thing.
  * **Triton as the implementation.** Reopen as a cross-check if the kernel ever
    underperforms its speed-of-light for a reason the profile cannot name. It currently
    underperforms for a reason the profile names precisely.
  * **CUDA graphs.** The harness calls the module directly, so there is no capture
    region available, and the in-window floor is the harness's own.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.conv2d import Conv2d
from ..L1.softmax import Softmax

_EXT_NAME = "fk_l2_yolov10_dfl"
_BUILD_DIR = Path(__file__).resolve().parent / ".torch_extensions"

# The fused kernel served the call.
ROUTE_FUSED = "fused"
# The kernel declined (or is unavailable): the frozen L1 Softmax/Conv2d composition,
# which is the baseline expression built from this workspace's frozen winners.
ROUTE_FROZEN = "frozen_fallback"
# Grad or autocast: `F.softmax` / `F.conv2d` directly, so autograd sees the
# reference's own ops and autocast performs its own casts and dtype promotion.
ROUTE_REFERENCE = "reference_fallback"

# Why the C++ side declined a call. Values mirror the enum in the CUDA source; the
# parity/guard script asserts against these rather than inferring the route from
# output values, because a guard that wrongly rejects everything still produces
# correct numbers.
DECLINE_NAMES = {
    0: "accepted",
    1: "accepted_empty",
    2: "not_cuda",
    3: "bad_rank",
    4: "not_contiguous",
    5: "bad_dtype",
    6: "lazy_neg_or_conj",
    7: "bad_channels",
    8: "unsupported_c1",
    9: "bad_weight",
    10: "too_many_rows",
    11: "offset_overflow",
    12: "zero_width",
}

PLAN_FIELDS = ("status", "c1", "vec", "block", "grid_x", "rows", "a")

_CPP_SOURCE = r"""
at::Tensor dfl_forward(const at::Tensor& x, const at::Tensor& w, int64_t c1);
at::Tensor dfl_forward_tuned(const at::Tensor& x, const at::Tensor& w, int64_t c1,
                             int64_t vec, int64_t block);
at::Tensor dfl_plan(const at::Tensor& x, const at::Tensor& w, int64_t c1);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>

namespace {

// Largest block this kernel is ever launched with, and the value nvcc budgets
// registers against. Without it nvcc assumes a 1024-thread block and allows only 64
// registers per thread, which is below the 80 the widest selectable instantiation
// needs -- the difference between no spills and spilling the staged input to local
// memory. 128 also measured fastest of {128, 64, 32} on both benched shapes.
constexpr int kMaxBlock = 128;

// A thread holds its `C1*V` loaded elements in registers across the two passes.
// Re-reading them from L1 in the second pass instead was built as a template flag and
// measured: the difference is inside run-to-run noise at every width, so the flag was
// removed rather than left as a path nothing exercises. What the measurement *did*
// establish is that ptxas does not pack two halves into one register, so staging costs
// `C1*V` registers and not `C1*V/2` -- see `select_max_width`, which is where that
// number actually changes the design.

// Decline reasons, mirrored by DECLINE_NAMES on the Python side.
constexpr int64_t kOk              = 0;
constexpr int64_t kOkEmpty         = 1;
constexpr int64_t kNotCuda         = 2;
constexpr int64_t kBadRank         = 3;
constexpr int64_t kNotContiguous   = 4;
constexpr int64_t kBadDtype        = 5;
constexpr int64_t kLazyFlag        = 6;
constexpr int64_t kBadChannels     = 7;
constexpr int64_t kUnsupportedC1   = 8;
constexpr int64_t kBadWeight       = 9;
constexpr int64_t kTooManyRows     = 10;
constexpr int64_t kOffsetOverflow  = 11;
constexpr int64_t kZeroWidth       = 12;

// A `dim3` y extent is 16 bits.
constexpr int64_t kMaxGridY = 65535;
// Within a row every offset is formed in 32-bit arithmetic, so the widest offset a
// row can carry, `(C1-1)*a + (a-1) = C1*a - 1`, has to fit. Row *bases* are formed
// in 64-bit, so this bounds `C1*a`, not the tensor.
constexpr int64_t kMaxInt32Offset = 2147483647LL;

template <typename scalar_t, int V>
struct alignas(sizeof(scalar_t) * V) AlignedVec {
  scalar_t val[V];
};

// One thread owns `V` consecutive outputs of one row and the whole length-`C1`
// reduction behind each of them.
//
// Alignment: the dispatcher admits a width only when the base pointer is a multiple
// of `V*sizeof(scalar_t)` and `a % V == 0`. That pair is *sufficient* for every
// access here, and no per-offset check is needed: `k0` is a multiple of `V` by
// construction, and `a % V == 0` makes `c*a` and `row*C1*a` multiples of `V` too, so
// every address below is the base plus a multiple of `V` elements. Weakening either
// half breaks the vector loads and stores.
template <typename scalar_t, int C1, int V>
__global__ __launch_bounds__(kMaxBlock) void dfl_fused(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ out,
    int a) {
  using vec_t = AlignedVec<scalar_t, V>;

  // `C1` weights, staged once per block and broadcast from shared memory afterwards.
  // The barrier is before any `return` so it is never divergent, and the dispatcher
  // guarantees `blockDim.x >= C1`, which is what makes this single-pass load cover
  // every weight.
  //
  // The barrier does show up in the stall histogram (0.85 per issue-active cycle at
  // b=4, 1.34 at b=1, where there is nothing else to overlap it with), so the
  // alternative -- every thread reading the `C1` weights from global and relying on
  // the broadcast -- was built and measured. It removes the barrier stall entirely and
  // is still *slower*: 8.74 us against 8.51 (b=4) and 7.42 against 7.17 (b=1), because
  // the extra loads cost more than the barrier, and at V=4 it also pushes registers
  // from 80 to 91. Reopen if a shape appears where `barrier` dominates the histogram
  // rather than trailing `long_scoreboard`.
  __shared__ float smem_w[C1];
  if (threadIdx.x < C1) {
    smem_w[threadIdx.x] = static_cast<float>(w[threadIdx.x]);
  }
  __syncthreads();

  const int64_t row = blockIdx.y;
  const scalar_t* __restrict__ xr = x + row * static_cast<int64_t>(C1)
                                        * static_cast<int64_t>(a);
  scalar_t* __restrict__ yr = out + row * static_cast<int64_t>(a);

  const int k0 = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * V;
  // The grid covers `ceil(a/V)` tiles rounded up to whole blocks, so the last block
  // holds lanes past the end of the row -- 8448 lanes for the 8400 tiles of the
  // captured shape at <16, V=1>, BLOCK = 128. Without this they would read and write
  // into the next row.
  if (k0 >= a) return;
  // `a % V == 0` and `k0 % V == 0` together give `k0 + V <= a` whenever `k0 < a`,
  // so an admitted width never leaves a partial vector at the end of a row.

  vec_t staged[C1];
  float m[V];
#pragma unroll
  for (int t = 0; t < V; ++t) m[t] = -INFINITY;

  // Pass 1: `C1` independent loads, reduced to the per-output maximum as they land.
  // Being independent is what gives a thread its memory-level parallelism -- `C1*V`
  // elements in flight at once -- which matters here because occupancy alone cannot
  // cover the latency at these shapes.
#pragma unroll
  for (int c = 0; c < C1; ++c) {
    const vec_t chunk = *reinterpret_cast<const vec_t*>(xr + c * a + k0);
    staged[c] = chunk;
#pragma unroll
    for (int t = 0; t < V; ++t) {
      m[t] = fmaxf(m[t], static_cast<float>(chunk.val[t]));
    }
  }

  float den[V];
#pragma unroll
  for (int t = 0; t < V; ++t) den[t] = 0.0f;

  // Pass 2: the normalizer alone, fp32.
#pragma unroll
  for (int c = 0; c < C1; ++c) {
#pragma unroll
    for (int t = 0; t < V; ++t) {
      den[t] += expf(static_cast<float>(staged[c].val[t]) - m[t]);
    }
  }

  float acc[V];
#pragma unroll
  for (int t = 0; t < V; ++t) acc[t] = 0.0f;

  // Pass 3: the weighted sum, deliberately reproducing the reference's *observable
  // intermediate rounding*. The reference is two ops, and the boundary between them is
  // visible: `F.softmax` writes its probabilities out in the input dtype, and only then
  // does `F.conv2d` contract them against the weight. So the reference computes
  // `sum_c w[c] * dtype(e_c/den)`, not `(sum_c w[c]*e_c)/den`.
  //
  // Those two agree to a few 1e-3 for a moderate non-negative weight such as the
  // `arange(c1)` this layer constructs, which is why keeping everything in fp32 looked
  // like the strictly-more-accurate choice. It is not equivalent, and the gap is
  // unbounded rather than small: a caller may load any weight through
  // `load_state_dict`, and with large alternating signs the weighted sum cancels, so
  // the reference's per-probability rounding is amplified instead of averaged out.
  // Measured with the valid weight `[60000, -60000] * 8` on the captured `[1,64,8400]`
  // shape: the fp32 form matched only 98.4% of elements at `max_abs_error` 32.0, i.e.
  // a scenario failure. Guarding on weight *values* is not an option -- the harness
  // fills the weight after construction, so the route may not depend on them -- so the
  // kernel reproduces the rounding instead, which costs a second exponential per
  // element and no extra launch.
#pragma unroll
  for (int c = 0; c < C1; ++c) {
    const float wc = smem_w[c];
#pragma unroll
    for (int t = 0; t < V; ++t) {
      const float e = expf(static_cast<float>(staged[c].val[t]) - m[t]);
      // Round to the storage dtype and back, exactly where the reference's softmax
      // stores its output, then accumulate in fp32 as its GEMM does.
      const float p = static_cast<float>(static_cast<scalar_t>(e / den[t]));
      acc[t] = fmaf(wc, p, acc[t]);
    }
  }

  vec_t o;
#pragma unroll
  for (int t = 0; t < V; ++t) o.val[t] = static_cast<scalar_t>(acc[t]);
  *reinterpret_cast<vec_t*>(yr + k0) = o;
}

struct Plan {
  int64_t status = kOk;
  int64_t c1 = 0;
  int64_t vec = 0;
  int64_t block = 0;
  int64_t grid_x = 0;
  int64_t rows = 0;
  int64_t a = 0;
};

// Widest width in {8, 4, 2, 1} that `a` divides and that *ptr* is aligned for,
// bounded by `max_v`.
int64_t select_width(const void* ptr, int64_t a, int64_t esize, int64_t max_v) {
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  int64_t v = max_v;
  while (v > 1 && (a % v != 0
                   || (addr % static_cast<uintptr_t>(v * esize)) != 0)) {
    v >>= 1;
  }
  return v;
}

// Widest vector width worth using, given how much work there is to spread.
//
// This is the one decision the profile actually turned on, so the reasoning is
// recorded here rather than left to a comment at the call site. Three facts, all
// measured on a B200 (`profile/dfl_v1_fused/`):
//
//  1. `-Xptxas -v`: ptxas gives each half its own 32-bit register instead of packing
//     two, so staging `C1*V` elements costs `C1*V` registers, not `C1*V/2`. At
//     C1*V = 128 that is 157 registers, which caps *theoretical* occupancy at 18.8%.
//     At C1*V <= 64 it is 80 registers (37.5%), at 32 it is 52 (56%), at 16 it is 32
//     (100%).
//  2. The width also *sets* the thread count: `rows*a/V` threads. At V = 8 the
//     captured `b=4` case is 16800 threads, 0.31 waves per SM -- the machine is not
//     merely under-occupied, it is under-supplied.
//  3. NCU on the captured shapes: 4.8% of peak DRAM read, 15.5% XU (MUFU), 9.4% SM
//     speed-of-light, and `long_scoreboard` plus `no_instruction` dominating the
//     stalls at 0.20 eligible warps per cycle. Nothing is saturated; the kernel is
//     latency-bound with too few warps to hide anything.
//
// So the width trades instruction count against thread count, and which side wins
// depends on whether there is enough work to fill the machine. Measured kernel
// durations, widest-to-narrowest at BLOCK = 128:
//
//     b=4  (0.13 M outputs):  V=8 11.0 us | V=4 9.0 | V=2 8.8 | V=1 8.4
//     b=1  (0.03 M outputs):  V=8 10.9 us | V=4 8.1 | V=2 7.7 | V=1 7.2
//     b=32 (1.08 M outputs):  V=8 21.6 us | V=4 19.2 | V=2 19.8 | V=1 21.8
//
// The two benched shapes are work-starved, so the narrowest width wins: it maximizes
// resident threads and minimizes registers, and a 2-byte load costs nothing in fetched
// bytes because consecutive lanes still read consecutive elements -- a warp asks for
// 32 x 2 B = 64 B and NCU reports 1.98 sectors per request against an ideal 2.00, i.e.
// ~100% sector efficiency. The narrow load buys threads and costs only instructions,
// and the LSU pipe is at 16%. Once the
// work exceeds what the SMs can hold, fewer wider loads win instead. V = 8 is never
// best on any shape measured -- its register cost outweighs the halved instruction
// count even in the work-rich case -- so `kMaxSelectableVec` excludes it outright.
// That means no ordinary call issues a 128-bit load: at 2-byte scalars V = 4 is a
// 64-bit load. The wide load is simply not what this kernel is short of, and the
// V = 8 instantiations are also the only ones ptxas spills (`profile/probe_ptxas.py`),
// so excluding them removes the spill as well. They stay compiled purely so
// `dfl_forward_tuned` can still force them for a future sweep.
constexpr int64_t kMaxSelectableVec = 4;

int64_t select_max_width(int64_t c1, int64_t rows, int64_t a, int64_t esize) {
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // sm_100 holds 2048 threads per SM.
  if (rows * a <= sms * 2048) return 1;
  return std::min<int64_t>(std::min<int64_t>(16 / esize, kMaxSelectableVec), 64 / c1);
}

// Every property the kernel assumes is checked here rather than in Python: it is a
// few nanoseconds each, and the alternative is a Python-side reimplementation of the
// same logic that can drift from what the kernel actually requires.
Plan make_plan(const at::Tensor& x, const at::Tensor& w, int64_t c1) {
  Plan p;
  auto decline = [&p](int64_t why) { p.status = why; return p; };

  if (!x.is_cuda()) return decline(kNotCuda);
  // The baseline's `view(b, 4, c1, a)` requires rank 3, and this kernel indexes as
  // though the layout were exactly that.
  if (x.dim() != 3) return decline(kBadRank);
  if (!x.is_contiguous()) return decline(kNotContiguous);
  const auto st = x.scalar_type();
  // fp32 is deliberately not admitted: its reference path is `F.conv2d`, which may
  // reach TF32 through cuDNN, so against the much tighter fp32 tolerance the
  // *reference* can be the less accurate side. No capture uses it.
  if (st != at::kHalf && st != at::kBFloat16) return decline(kBadDtype);
  // A `neg` or `conj` view carries a lazy flag ATen applies when it reads the
  // tensor. This kernel dereferences the storage below that flag, so it would see
  // sign-flipped values.
  if (x.is_neg() || x.is_conj()) return decline(kLazyFlag);
  if (c1 <= 0 || x.size(1) != 4 * c1) return decline(kBadChannels);
  if (c1 != 8 && c1 != 16 && c1 != 32) return decline(kUnsupportedC1);

  // The weight is read as raw memory, so a replacement parameter the reference would
  // reject must not reach the kernel. This is checked on the *tensor*, never on its
  // values: the harness fills the weight after construction.
  if (w.scalar_type() != st || w.device() != x.device()
      || w.dim() != 4 || w.size(0) != 1 || w.size(1) != c1
      || w.size(2) != 1 || w.size(3) != 1
      || !w.is_contiguous() || w.is_neg() || w.is_conj()) {
    return decline(kBadWeight);
  }

  p.c1 = c1;
  p.rows = x.size(0) * 4;
  p.a = x.size(2);

  // A zero-width row is *not* an empty-output case. `F.conv2d` rejects a zero-width
  // input ("Kernel size can't be greater than actual input size"), so the reference
  // raises -- measured -- and returning an empty tensor here would diverge from it.
  // Declining puts the call on the fallback, which raises the identical error.
  if (p.a == 0) return decline(kZeroWidth);
  // `b == 0` is different: the reference returns an empty result for it. Serve that
  // without a launch, because a zero-extent grid is a CUDA launch error.
  if (p.rows == 0) {
    p.status = kOkEmpty;
    return p;
  }
  if (p.rows > kMaxGridY) return decline(kTooManyRows);
  if (c1 * p.a > kMaxInt32Offset) return decline(kOffsetOverflow);

  const int64_t esize = x.element_size();
  p.vec = select_width(x.data_ptr(), p.a, esize,
                       select_max_width(c1, p.rows, p.a, esize));

  const int64_t tiles = (p.a + p.vec - 1) / p.vec;
  // A launch parameter, not a second kernel. Prefer the largest block that still
  // covers the SMs, and fall back to one warp per block for the small-`b` case where
  // no larger block reaches a full wave (36 blocks over 148 SMs at BLOCK = 128,
  // b = 1). `>= C1` is required by the single-pass weight load.
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  p.block = 32;
  for (const int64_t cand : {kMaxBlock, kMaxBlock / 2, kMaxBlock / 4}) {
    if (cand < c1) continue;
    if (((tiles + cand - 1) / cand) * p.rows >= sms) { p.block = cand; break; }
  }
  if (p.block < c1) p.block = c1;
  p.grid_x = (tiles + p.block - 1) / p.block;
  return p;
}

template <typename scalar_t, int C1, int V>
void maybe_launch(const Plan& p, const at::Tensor& x, const at::Tensor& w,
                  at::Tensor& out, cudaStream_t stream) {
  // A width wider than a 128-bit load is never selected, so skipping its
  // instantiation keeps the build small.
  if constexpr (V * static_cast<int>(sizeof(scalar_t)) <= 16) {
    const dim3 grid(static_cast<unsigned>(p.grid_x),
                    static_cast<unsigned>(p.rows), 1);
    const auto* xp = x.data_ptr<scalar_t>();
    const auto* wp = w.data_ptr<scalar_t>();
    auto* op = out.data_ptr<scalar_t>();
    const int ai = static_cast<int>(p.a);
    const unsigned block = static_cast<unsigned>(p.block);
    dfl_fused<scalar_t, C1, V><<<grid, block, 0, stream>>>(xp, wp, op, ai);
  }
}

template <typename scalar_t, int C1>
void launch_widths(const Plan& p, const at::Tensor& x, const at::Tensor& w,
                   at::Tensor& out, cudaStream_t stream) {
  switch (p.vec) {
    case 8: maybe_launch<scalar_t, C1, 8>(p, x, w, out, stream); break;
    case 4: maybe_launch<scalar_t, C1, 4>(p, x, w, out, stream); break;
    case 2: maybe_launch<scalar_t, C1, 2>(p, x, w, out, stream); break;
    default: maybe_launch<scalar_t, C1, 1>(p, x, w, out, stream); break;
  }
}

template <typename scalar_t>
void launch_c1(const Plan& p, const at::Tensor& x, const at::Tensor& w,
               at::Tensor& out, cudaStream_t stream) {
  switch (p.c1) {
    case 8: launch_widths<scalar_t, 8>(p, x, w, out, stream); break;
    case 16: launch_widths<scalar_t, 16>(p, x, w, out, stream); break;
    default: launch_widths<scalar_t, 32>(p, x, w, out, stream); break;
  }
}

}  // namespace

// `load_inline` does not translate C++ default arguments into pybind defaults, so the
// tuning knobs get their own entry point rather than three constants on every call.
namespace {

at::Tensor dfl_run(const at::Tensor& x, const at::Tensor& w, int64_t c1,
                   int64_t vec_override, int64_t block_override) {
  Plan p = make_plan(x, w, c1);
  if (p.status != kOk && p.status != kOkEmpty) return at::Tensor();

  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({x.size(0), 4, p.a}, x.options());
  if (p.status == kOkEmpty) return out;

  // An override the alignment or the `block >= C1` rule cannot honour is ignored
  // rather than silently producing a wrong result.
  if (vec_override > 0) {
    const int64_t esize = x.element_size();
    const int64_t v = select_width(x.data_ptr(), p.a, esize,
                                   std::min<int64_t>(vec_override, 16 / esize));
    if (v == vec_override) p.vec = v;
  }
  if (block_override >= c1 && block_override <= kMaxBlock) p.block = block_override;

  // The output comes from the caching allocator, which aligns to at least 512 B, so
  // this never narrows the width in practice. It is here so that a future change to
  // how the output is produced cannot silently break the vector store.
  p.vec = select_width(out.data_ptr(), p.a, x.element_size(), p.vec);
  const int64_t tiles = (p.a + p.vec - 1) / p.vec;
  p.grid_x = (tiles + p.block - 1) / p.block;

  auto stream = at::cuda::getCurrentCUDAStream();
  switch (x.scalar_type()) {
    case at::kHalf: launch_c1<at::Half>(p, x, w, out, stream); break;
    case at::kBFloat16: launch_c1<at::BFloat16>(p, x, w, out, stream); break;
    default: TORCH_CHECK(false, "unreachable dtype ", x.scalar_type());
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace

// Returns an undefined tensor -- Python `None` -- for anything the kernel cannot
// serve exactly, so the caller can fall back without the C++ side needing to know
// what the reference is.
at::Tensor dfl_forward(const at::Tensor& x, const at::Tensor& w, int64_t c1) {
  return dfl_run(x, w, c1, 0, 0);
}

// Same kernel with the launch shape forced, so `profile/probe_tune.py` can sweep the
// vector width and the block size from one build. Not on any hot path.
at::Tensor dfl_forward_tuned(const at::Tensor& x, const at::Tensor& w, int64_t c1,
                             int64_t vec, int64_t block) {
  return dfl_run(x, w, c1, vec, block);
}

// Reports the routing decision without running it, so a test can assert which path
// served a call instead of inferring it from the values.
at::Tensor dfl_plan(const at::Tensor& x, const at::Tensor& w, int64_t c1) {
  const Plan p = make_plan(x, w, c1);
  auto t = at::zeros({7}, at::TensorOptions().dtype(at::kLong));
  auto acc = t.accessor<int64_t, 1>();
  acc[0] = p.status;
  acc[1] = p.c1;
  acc[2] = p.vec;
  acc[3] = p.block;
  acc[4] = p.grid_x;
  acc[5] = p.rows;
  acc[6] = p.a;
  return t;
}
"""


def _load_extension():
    """Build (or reuse) the extension in a workspace-local, sm_100-only cache.

    A build failure must not escape. Correctness is available through the fallback
    either way, and an import error would cost every benched scenario.
    """
    from torch.utils.cpp_extension import load_inline

    # Test hook: injects a flag nvcc rejects, so the degradation path can be
    # exercised for real rather than argued for. Kept in its own build directory so
    # a poisoned build never becomes the cached good one.
    bad_flag = bool(os.environ.get("FK_DFL_BAD_FLAG"))
    build_dir = _BUILD_DIR / "badflag" if bad_flag else _BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)

    flags = ["-O3", "-std=c++17", "--generate-line-info", "--expt-relaxed-constexpr"]
    if bad_flag:
        flags.append("--this-flag-does-not-exist")

    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    try:
        return load_inline(
            name=_EXT_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["dfl_forward", "dfl_forward_tuned", "dfl_plan"],
            extra_cuda_cflags=flags,
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        # Restoring this matters beyond tidiness: the bench runs in one subprocess,
        # and a leaked value would change the arch list of whatever else it compiles.
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# At import, never lazily inside `forward`: a `ninja` invocation during a timed call
# would both pay the build cost in the measurement and raise the process thread
# count, which the harness reads as tampering.
if os.environ.get("FK_DFL_DISABLE_EXT"):
    _EXT, _EXT_ERROR = None, "disabled by FK_DFL_DISABLE_EXT"
else:
    try:
        _EXT, _EXT_ERROR = _load_extension(), None
    except Exception as exc:  # noqa: BLE001 - a build failure must degrade, not raise
        _EXT, _EXT_ERROR = None, f"{type(exc).__name__}: {exc}"

# Bound once so the fast path is an attribute-free call.
_dfl_forward = _EXT.dfl_forward if _EXT is not None else None


def extension_loaded() -> bool:
    """Whether the fused kernel is available in this process."""
    return _EXT is not None


def extension_error() -> str | None:
    """Why the extension is unavailable, or None if it loaded."""
    return _EXT_ERROR


def plan_for(x: torch.Tensor, w: torch.Tensor, c1: int) -> dict[str, int]:
    """The C++ admission decision for `(x, w, c1)`, without running it.

    `status` is a `DECLINE_NAMES` key. Raises if the extension is not loaded, so a
    caller cannot mistake a build failure for a routing answer. The `vec` reported
    here is selected from the input pointer alone; `dfl_forward` intersects it with
    the freshly allocated output's alignment, which the caching allocator's 512 B
    guarantee means never narrows it.
    """
    if _EXT is None:
        raise RuntimeError(f"extension not loaded: {_EXT_ERROR}")
    return dict(zip(PLAN_FIELDS, _EXT.dfl_plan(x, w, c1).tolist()))


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)
        # Every one of these is a plain attribute a caller can reassign after
        # construction. The fused route was chosen for one configuration and is only
        # valid while that configuration still holds, so all ten are cached and compared
        # as a single tuple -- the fast path can afford one compare, not ten branches.
        #
        # `in_channels`, `out_channels` and `kernel_size` are in here even though
        # `F.conv2d` takes its shapes from the weight rather than from them, and `c1` is
        # in here even though the kernel is handed the same `self.c1` the fallback's
        # `view` uses. Narrowing the tuple to "only what provably reaches the reference"
        # saves about 0.5 us per call and is the wrong trade: these are the module's
        # declared configuration, a caller who changes one has changed the module, and a
        # guard that is exhaustive over construction-time state is cheaper to reason
        # about than one that is exhaustive over a derivation someone has to re-verify.
        self._fused_config = self._config_now()

    def _config_now(self) -> tuple:
        # `_modules` / `_parameters` are read directly rather than through attribute
        # access: `nn.Module.__getattr__` misses `__dict__` for anything it registered
        # and walks three containers, which measured 0.23 us per submodule and 0.34 us
        # for `conv.weight` against a ~16 us window. The dict read is exactly
        # equivalent -- reassigning `m.conv` or `m.conv.weight` goes through
        # `Module.__setattr__`, which updates these same containers.
        conv = self._modules["conv"]
        return (self.c1, conv.in_channels, conv.out_channels, conv.kernel_size,
                conv.stride, conv.padding, conv.dilation, conv.groups,
                conv.bias is None, self._modules["_softmax"].dim)

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        """The baseline expression. Raises whatever the baseline would raise."""
        b, _, a = x.shape
        v = x.view(b, 4, self.c1, a).transpose(2, 1)
        if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
            # Under grad the graph has to be built, and under autocast the result
            # dtype is autocast's to choose -- the frozen L1 modules reach memory by
            # pointer, below the dispatcher that implements both. Going straight to
            # the functionals keeps the module's behaviour the reference's here.
            v = F.softmax(v, dim=self._softmax.dim)
            conv = self.conv
            return F.conv2d(v, conv.weight, conv.bias, stride=conv.stride,
                            padding=conv.padding, dilation=conv.dilation,
                            groups=conv.groups).view(b, 4, a)
        return self.conv(self._softmax(v)).view(b, 4, a)

    def route_for(self, x: torch.Tensor) -> str:
        """Which of `ROUTE_FUSED` / `ROUTE_FROZEN` / `ROUTE_REFERENCE` serves `x`.

        Mirrors `forward`'s decision without running it, so a test can assert the
        route rather than infer it: a guard that wrongly rejected everything would
        still return correct values.
        """
        if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
            return ROUTE_REFERENCE
        if _EXT is None or self._fused_config != self._config_now():
            return ROUTE_FROZEN
        status = plan_for(x, self.conv.weight, self.c1)["status"]
        return ROUTE_FUSED if status in (0, 1) else ROUTE_FROZEN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The config tuple is built inline rather than through `_config_now` to save
        # the call frame; `route_for` uses the helper, and a parity check asserts the
        # two agree so they cannot drift.
        if (_dfl_forward is not None
                and not torch.is_grad_enabled()
                and not torch.is_autocast_enabled("cuda")):
            conv = self._modules["conv"]
            if self._fused_config == (self.c1, conv.in_channels, conv.out_channels,
                                      conv.kernel_size, conv.stride, conv.padding,
                                      conv.dilation, conv.groups, conv.bias is None,
                                      self._modules["_softmax"].dim):
                out = _dfl_forward(x, conv._parameters["weight"], self.c1)
                # An undefined `at::Tensor` arrives here as None: the kernel declined
                # this input, and every property it checks is one the fallback handles.
                if out is not None:
                    return out
        return self._reference(x)
