"""T5 feed-forward dense layers with TP sharding (L2), fused gated-GELU-new.

Drop-in for ``baseline.py``. ``T5DenseActDense`` is carried over unchanged --
it is never benched, but ``L3/t5_block.py`` imports both classes from this
module, and a candidate file shadows the whole baseline module.

``T5DenseGatedActDense`` keeps both GEMMs and replaces only the activation. On
the captured shape (``bf16[1, 512, 4096]``, ``d_model = 4096``,
``d_ff = 10240``, ``dense_act_fn = "gelu_new"``) the baseline's 194.5 us splits
as ``wi`` 68.7 us at 1250 TFLOP/s, activation 91.2 us, ``wo`` 45.8 us at
937 TFLOP/s. Both GEMMs are nvjet and already fast; the activation is 47% of the
operator against a measured 13.2 us memory-traffic floor for the same 20 MiB read
+ 10 MiB written (``torch.mul`` of two contiguous ``[512, 10240]`` bf16 tensors,
the benchmark's own timing protocol). Two structural reasons, both removed by
fusing:

* ``NewGELUActivation`` is eight eager elementwise ops and the gate/up multiply
  is a ninth, so nine kernels each make a full round trip to HBM;
* ``gate_up.chunk(2, -1)`` hands four of them a strided view (span 10240, row
  stride 20480), which ATen cannot vectorize -- those degrade to
  ``elementwise_kernel<128, 4>`` at 11.5-13.8 us each against 4.4-5.4 us for the
  ``vectorized_elementwise_kernel<8>`` variants on contiguous operands.

The fused kernel reads the two halves as two independent, fully coalesced
streams 20 KiB apart within a row, and writes one. It costs 15.4 us -- 1.16x that
floor -- so the operator lands at 116 us, a **1.66-1.68x same-run speedup**, with
``matched_ratio`` exactly 1.0 and ``max_abs_error`` exactly 0.0 on every one of
eight ``validate.py`` runs. Three kernels launch instead of eleven.

Read the speedup rather than the microseconds: ``with_gpu.py`` leases whichever
B200 is free, and three of the four sit at an SM clock of 1155 MHz against one at
1965 MHz, which moves absolute latency by half while moving the same-run ratio by
1.5% (``profile/clock_state_note.txt``). The 116 us figure is from the 1965 MHz
regime, which is where the 193.5 us reference was measured.

Getting the activation there took one non-obvious step, and it is recorded next to
the code that does it (see ``widen_pair`` / ``rb2`` below): the cost of this kernel
is the *rounding*, not the arithmetic. Nine op boundaries plus ``pow``'s internal
one mean ten roundings, which a scalar formulation pays as nineteen
MIO-queued conversion instructions per element against nine arithmetic
operations. ncu measured ``mio_throttle`` at half the stall cycles with
``__float2bfloat16_rn`` the top stall site, while DRAM throughput sat at 9%.
Rounding pairs with ``cvt.rn.bf16x2.f32`` and widening with a shift cuts those
nineteen to five and took the activation from 21.4 us to 15.3 us, without changing
a single rounded value. ``profile/gated_gelu_new_v3_packed_cvt/REPORT.md`` has the
records; ``docs/results.md`` has the summary.

Why the kernel emulates an op chain instead of evaluating a formula
------------------------------------------------------------------

The benchmark compares only the *final* output, elementwise against
``atol + rtol*|ref|`` with ``atol = rtol = 1e-2`` for bf16, and needs 99% of
elements inside that bound. That sounds generous, and it is not: the baseline
rounds to bf16 (8 mantissa bits) after every one of its ops, so the reference
carries about 1% of its own rounding noise, and the ``wo`` GEMM does not average
it away. A *more accurate* activation is therefore rejected. Measured on the real
shape and weight distribution (``profile/probe_numerics.py``):

    same formula, fp32 throughout, one final round   matched_ratio 0.98299  FAIL
    torch.compile(baseline)         (111.8 us, 1.73x) matched_ratio 0.98290  FAIL
    op boundaries emulated, cube as fp32 g*g*g        matched_ratio 1.00000  pass
    op boundaries emulated, cube as bf16 (g*g)*g      matched_ratio 1.00000  pass,
                                                      and bit-exact (100.0000%)

So the kernel reproduces ATen's *op chain*, rounding to bf16 at each of the nine
boundaries below, with every constant entering as fp32 exactly as ATen passes a
CPU scalar for ``mul``/``add``. ``r()`` is round-to-nearest-even to bf16;
``g`` and ``u`` are the gate and up elements::

    f = r(0.5f * g)       exact in binary floating point anyway
    p = r(r(g*g) * g)     torch.pow(g, 3.0) -- two roundings, see below
    a = r(0.044715f * p)
    b = r(g + a)
    c = r(K * b)          K = 0.7978845608028654f, the fp32 value of sqrt(2/pi)
    d = r(tanhf(c))       accurate libdevice tanhf, the only tanh path here
    e = r(1.0f + d)
    h = r(f * e)
    y = r(h * u)

That is nine eager tensor ops but *ten* roundings, because ``pow`` rounds once
internally. The order is the order CPython evaluates the expression in, hence the
order ATen launches the kernels in; the branches are independent, so it does not
affect a value.

Each of those five facts was confirmed at machine level by disassembling
``libtorch_cuda.so`` for sm_100 in this environment, and is re-checked over all
65536 bf16 bit patterns -- against ATen on the contiguous *and* strided paths,
and against this kernel's own device code under the flags it is really built
with -- by ``profile/probe_aten_semantics.py``:

* ``torch.pow(x, 3.0)`` on bf16 is special-cased to a multiply chain evaluated in
  the scalar type: ``FMUL; F2F.BF16.F32; FMUL; F2FP.BF16.F32``, i.e.
  ``bf16(bf16(x*x) * x)``. Two roundings, not one.
* ``torch.tanh`` on bf16 widens to fp32, runs the accurate libdevice ``tanhf``
  (its ``0.6`` / ``9.010913848876953`` branch constants are visible; there is no
  ``MUFU.TANH`` anywhere in the library), and rounds once.
* bf16 elementwise ``mul``/``add`` compute in fp32 opmath and round once, and a
  CPU scalar enters as fp32, not bf16 (``div`` is the exception and is not used
  here). So ``K`` must be the fp32 literal and not a bf16-rounded one.
* ``c10::BFloat16(float)`` is ``cvt.rn.bf16.f32`` on sm_80 and later, which is
  what ``__float2bfloat16_rn`` emits.

The fp32-cube formulation is kept behind a compile-time define. It does not
depend on ``pow``'s internal decomposition, is 99.43% elementwise exact, and also
measures ``matched_ratio`` 1.00000 -- a validated alternative if a future ATen
changes that decomposition, at the cost of the bit-exactness claim.

Guards, each falling back to the eager chain in ``self.act``
-----------------------------------------------------------

Decided on the Python side, once at construction: ``dense_act_fn`` is not
``"gelu_new"`` (the fused kernel computes one specific activation, so a ``relu``
or ``silu`` configuration must not reach it), tensor parallelism is active, or
the extension did not build. Decided per call on the Python side: the input is
not exactly a ``torch.Tensor``, a ``__torch_dispatch__`` or
``__torch_function__`` mode is on the stack, the input carries a forward-mode
tangent, or its last dimension is not ``2 * d_ff``. Decided in C++ before any
launch: a dispatch key the flat traversal would drop (including ``Named``, whose
dimension names ``at::empty`` would not carry over), a device whose compute
capability is not the one this binary was compiled for, not CUDA, not strided, no
storage, empty, not contiguous, an odd last dimension, autograd-tracked with grad
mode on, dtype other than bf16, or a null data pointer. Alignment and row divisibility select
the scalar path rather than falling back.

Every decision is made *before* the launch on purpose: CUDA errors are
asynchronous, so catching a launch failure afterwards and retrying eagerly would
already have poisoned the context. A build or dispatch problem degrades to "slow
but right", never to "fast and wrong".

Environment switches, all sampled once at import
------------------------------------------------

``FK_T5_DENSE_FUSED=0``
    Never use the fused kernel; run the eager chain. Makes the fallback path
    testable without breaking the build.
``FK_T5_DENSE_FP32_CUBE=1``
    Build with the fp32-cube formulation instead of the bit-exact bf16 chain.
``FK_T5_DENSE_GEOMETRY=<block>,<unroll>``
    Force the launch geometry instead of letting ``choose_geometry`` pick, so the
    baked-in choice can be re-measured rather than trusted.
``FK_T5_DENSE_SCALAR=1``
    Take the scalar path even when the vector path is eligible, for A/B.
``FK_T5_DENSE_PACKED_CVT=0``
    Round one element at a time (``cvt.rn.bf16.f32``) and widen with
    ``cvt.f32.bf16``, instead of rounding pairs with ``cvt.rn.bf16x2.f32`` and
    widening with a shift. Identical values either way; the packed form exists
    because the conversions, not the arithmetic, are what this kernel spends its
    time on -- 21.5 us against 15.4 us on the captured shape.
``FK_T5_DENSE_CACHE_HINTS=<0..3>``
    Bit 0 adds ``ld.global.L1::no_allocate`` to the loads, bit 1 adds
    ``st.global.L1::evict_first`` to the stores; both on by default. Measured
    effect on this kernel: **none** (31.7 us at every setting). They are kept on
    because the accesses really are pure streams -- L1 hit rate 0%, L2 hit rate
    0.51% -- and cost nothing, not because they were measured to help.
``FK_T5_DENSE_MIN_BLOCKS=<n>``
    Compile with ``__launch_bounds__(block, n)`` to cap the register budget.
    Measured effect: **none** at n = 4, 6 or 8 (31.7 us at every setting), which
    is how we know the first ncu record's occupancy finding was an artefact of
    unroll 4 rather than the standing bottleneck.
``FK_T5_DENSE_BREAK_BUILD=1``
    Append a line that does not compile, to exercise the build-failure fallback.
``FK_T5_DENSE_BUILD_DIR=<path>``
    Override the build root. A path that resolves outside this workspace is
    **rejected**, not honoured: the guarantee is that this workspace owns its build
    tree, so a stale binary from elsewhere can never be picked up, and a switch is
    not allowed to void it.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import forward_ad as _forward_ad
from transformers import T5Config

from ....infra.tp import _tp_size
from ..L1.gelu import GELU
from ..L1.silu import SiLU
from .parallel_linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

import math


__targets__ = ["T5DenseGatedActDense", "T5DenseActDense"]


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <optional>
#include <vector>

std::optional<at::Tensor> gated_gelu_new(const at::Tensor& gate_up);
std::optional<at::Tensor> gated_gelu_new_with_geometry(const at::Tensor& gate_up,
                                                       int64_t block, int64_t unroll,
                                                       bool force_scalar);
std::vector<at::Tensor> gated_gelu_new_stages(const at::Tensor& gate,
                                              const at::Tensor& up);
bool gated_gelu_new_is_bit_exact_chain();
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <vector>

#ifndef FK_T5_FP32_CUBE
#define FK_T5_FP32_CUBE 0
#endif
#ifndef FK_T5_CACHE_HINTS
#define FK_T5_CACHE_HINTS 3
#endif
#ifndef FK_T5_MIN_BLOCKS
#define FK_T5_MIN_BLOCKS 0
#endif
#ifndef FK_T5_PACKED_CVT
#define FK_T5_PACKED_CVT 1
#endif
// The compute capability this translation unit was compiled for, times ten. The
// build targets a single local architecture, so a forward on a *different*
// architecture in the same process -- import while cuda:0 is current, call with a
// tensor on cuda:1 of another generation -- would launch a binary with no image
// for that device. That failure is asynchronous and would poison the context, so
// it has to be a predicate rather than an error path. 0 means "unknown", which
// disables the check.
#ifndef FK_T5_BUILD_SM
#define FK_T5_BUILD_SM 0
#endif

// A min-blocks-per-SM hint caps the register budget the compiler may spend. It
// is the lever the ncu record asked for: at 104 registers per thread only 2
// blocks fit per SM, so theoretical occupancy was 25% and the schedulers had no
// eligible warp 52.9% of the time.
#if FK_T5_MIN_BLOCKS > 0
#define FK_T5_BOUNDS(BLOCK) __launch_bounds__(BLOCK, FK_T5_MIN_BLOCKS)
#else
#define FK_T5_BOUNDS(BLOCK) __launch_bounds__(BLOCK)
#endif

namespace fk_t5_dense {

// ---------------------------------------------------------------------------
// The rounding schedule
// ---------------------------------------------------------------------------

// Round to nearest even, bf16, and come back to fp32 to keep computing. This is
// cvt.rn.bf16.f32, the same instruction c10::BFloat16(float) emits on sm_80 and
// later, so every intermediate below holds exactly the value the corresponding
// eager op would have materialised.
__device__ __forceinline__ float rb(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// The same schedule, two elements at a time.
//
// This is where the kernel's time actually goes, and it is the rounding itself
// rather than the arithmetic. The chain has ten rounding points (nine eager op
// boundaries, plus the one inside pow), so a scalar formulation issues ten
// cvt.rn.bf16.f32 and about nine cvt.f32.bf16 per element -- nineteen
// conversions against nine arithmetic operations. Those cvt instructions queue
// through MIO, and the ncu record put mio_throttle at 9.2 of the 18.4 cycles
// between issues, with __float2bfloat16_rn the top stall site by a wide margin.
// The arithmetic was never the problem: DRAM throughput 10.7%, ALU 31.3%.
//
// Two facts remove most of that traffic without changing a single rounded value:
//
// * ``cvt.rn.bf16x2.f32`` rounds *two* fp32 to bf16 in one instruction, with the
//   same round-to-nearest-even the scalar form uses. Ten conversions per element
//   become five.
// * Widening bf16 to fp32 needs no conversion instruction at all. bf16 is
//   literally the top sixteen bits of fp32 -- same sign, same 8-bit exponent,
//   same bias -- so a 16-bit left shift is an exact widening for *every*
//   pattern: normals, signed zeros, subnormals (the bias matches, so the
//   subnormal scale comes out right), infinities, and NaNs with their payload
//   preserved. That moves nine instructions per element off MIO and onto the
//   integer pipe.
//
// Nineteen conversions per element become five, and the exhaustive probe in
// profile/probe_aten_semantics.py re-checks every rounded value afterwards
// rather than taking the reasoning above on trust.
__device__ __forceinline__ float2 widen_pair(unsigned int w) {
  float2 out;
  out.x = __int_as_float(w << 16);            // low half  -> even element
  out.y = __int_as_float(w & 0xFFFF0000u);    // high half -> odd element
  return out;
}

__device__ __forceinline__ unsigned int narrow_pair(float lo, float hi) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  unsigned int w;
  __builtin_memcpy(&w, &h, sizeof(w));
  return w;
}

__device__ __forceinline__ float2 rb2(float lo, float hi) {
  return widen_pair(narrow_pair(lo, hi));
}

// The fp32 value of sqrt(2/pi). Written as a literal rather than computed as
// sqrtf(2.0f / M_PI): the baseline passes Python's float64
// math.sqrt(2.0 / math.pi) as a CPU scalar, ATen narrows it to fp32 once, and
// only the literal is guaranteed to be that same fp32 value. Likewise 0.044715f
// enters as fp32, not bf16.
__device__ constexpr float kBeta = 0.7978845608028654f;
__device__ constexpr float kKappa = 0.044715f;

// ATen's bf16 tanh widens to fp32 and calls the accurate libdevice tanhf, so that
// is what the chain calls. The hardware approximation is not an option here: the
// plan's path boundaries rule it out, and it is not present in this file.
__device__ __forceinline__ float tanh_for_chain(float v) { return ::tanhf(v); }

// NewGELUActivation(g) * u, evaluated as the nine eager op boundaries rather
// than as the algebraic formula. rb() after every step is what makes this
// bit-identical to the reference; it also blocks FMA contraction, since every
// operand is a conversion result.
__device__ __forceinline__ float gated_gelu_new_op(float g, float u) {
  // f first, because CPython evaluates 0.5 * input before the right-hand factor,
  // so that is the order ATen launches these in. The branches are independent, so
  // the order does not change a value -- it just keeps the code readable against
  // a trace.
  const float f = rb(0.5f * g);
#if FK_T5_FP32_CUBE
  // Does not rely on pow(x, 3.0) decomposing into (x*x)*x with a round between
  // the multiplies. 99.43% elementwise exact, matched_ratio still 1.00000.
  const float p = rb(g * g * g);
#else
  const float p = rb(rb(g * g) * g);
#endif
  const float a = rb(kKappa * p);
  const float b = rb(g + a);
  const float c = rb(kBeta * b);
  const float d = rb(tanh_for_chain(c));
  const float e = rb(1.0f + d);
  const float h = rb(f * e);
  return h * u;  // the caller rounds this, so the ninth boundary is the store
}

// The same nine boundaries on a packed pair. Every rounded value is identical to
// gated_gelu_new_op's by construction (cvt.rn.bf16x2.f32 rounds each component
// exactly as cvt.rn.bf16.f32 does), and identical by measurement: the probe
// compares this path, the scalar path and ATen over all 65536 patterns.
__device__ __forceinline__ unsigned int gated_gelu_new_pair(unsigned int wg,
                                                            unsigned int wu) {
  const float2 g = widen_pair(wg);
  const float2 u = widen_pair(wu);
  const float2 f = rb2(0.5f * g.x, 0.5f * g.y);
#if FK_T5_FP32_CUBE
  const float2 p = rb2(g.x * g.x * g.x, g.y * g.y * g.y);
#else
  const float2 sq = rb2(g.x * g.x, g.y * g.y);
  const float2 p = rb2(sq.x * g.x, sq.y * g.y);
#endif
  const float2 a = rb2(kKappa * p.x, kKappa * p.y);
  const float2 b = rb2(g.x + a.x, g.y + a.y);
  const float2 c = rb2(kBeta * b.x, kBeta * b.y);
  const float2 d = rb2(tanh_for_chain(c.x), tanh_for_chain(c.y));
  const float2 e = rb2(1.0f + d.x, 1.0f + d.y);
  const float2 h = rb2(f.x * e.x, f.y * e.y);
  return narrow_pair(h.x * u.x, h.y * u.y);
}

// ---------------------------------------------------------------------------
// Access unit
// ---------------------------------------------------------------------------

// One 128-bit access: 8 bf16, held as four 32-bit words because that is what the
// packed conversion below consumes. Words, not a union with a bf16 array: reading
// an inactive union member is undefined, and there is no need for one -- the
// inline asm writes the four words directly, and the element view is recovered
// with __ushort_as_bfloat16 where it is wanted.
struct alignas(16) Pack {
  static constexpr int kWidth = 8;
  static constexpr int kPairs = kWidth / 2;
  unsigned int w[kPairs];
};

// Both input halves and the output are pure streams: nothing is read twice, and
// the ncu record measured an L1 hit rate of 0% and an L2 hit rate of 0.51%, so
// allocating cache lines for them buys nothing and costs tag and fill work.
// FK_T5_CACHE_HINTS bit 0 keeps the loads out of L1, bit 1 marks the stores
// evict-first; both are on by default. Set it to 0 to re-run the comparison
// (profile/tune_geometry.py --stage hints).
__device__ __forceinline__ Pack load_pack(const Pack* p) {
  Pack out;
#if (FK_T5_CACHE_HINTS & 1)
  asm("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(out.w[0]), "=r"(out.w[1]), "=r"(out.w[2]), "=r"(out.w[3])
      : "l"(p));
#else
  out = *p;
#endif
  return out;
}

__device__ __forceinline__ void store_pack(Pack* p, const Pack& v) {
#if (FK_T5_CACHE_HINTS & 2)
  asm volatile("st.global.L1::evict_first.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(p), "r"(v.w[0]), "r"(v.w[1]), "r"(v.w[2]), "r"(v.w[3])
               : "memory");
#else
  *p = v;
#endif
}

// ---------------------------------------------------------------------------
// Kernels
//
// The output is [rows, half] and the input is [rows, 2*half], so within a row
// the gate half starts at 0 and the up half at `half`. Rows are carried on
// blockIdx.y (grid-strided, so a row count above the 65535 y-limit still works)
// and the packs of a row on blockIdx.x, which keeps the mapping free of the
// integer division by `half` that a single flat index would need. UNROLL packs
// are in flight per thread so the loads of one iteration overlap the math of the
// previous one.
// ---------------------------------------------------------------------------

template <int BLOCK, int UNROLL>
__global__ void FK_T5_BOUNDS(BLOCK) gated_gelu_new_vec_kernel(
    const Pack* __restrict__ in, Pack* __restrict__ out,
    long long rows, int packs_out_row, int packs_in_row) {
  // The in-row index is 32-bit on purpose: a row of 2^31 packs is 32 GiB, so the
  // narrow index cannot overflow, and it halves the address registers the unroll
  // needs. Only the row offset is computed in 64 bits, once per row.
  const int xstride = static_cast<int>(gridDim.x) * BLOCK;

  for (long long row = blockIdx.y; row < rows; row += gridDim.y) {
    const Pack* __restrict__ gate = in + row * packs_in_row;
    const Pack* __restrict__ up = gate + packs_out_row;
    Pack* __restrict__ dst = out + row * packs_out_row;

    for (int base = static_cast<int>(blockIdx.x) * BLOCK + threadIdx.x;
         base < packs_out_row; base += xstride * UNROLL) {
      Pack pg[UNROLL];
      Pack pu[UNROLL];

#pragma unroll
      for (int t = 0; t < UNROLL; ++t) {
        const int at = base + t * xstride;
        if (at < packs_out_row) {
          pg[t] = load_pack(gate + at);
          pu[t] = load_pack(up + at);
        }
      }

#pragma unroll
      for (int t = 0; t < UNROLL; ++t) {
        const int at = base + t * xstride;
        if (at >= packs_out_row) continue;
        Pack o;
#if FK_T5_PACKED_CVT
#pragma unroll
        for (int j = 0; j < Pack::kPairs; ++j) {
          o.w[j] = gated_gelu_new_pair(pg[t].w[j], pu[t].w[j]);
        }
#else
        // The scalar-conversion formulation, kept only so the packed one can be
        // A/B'd against it (profile/tune_variants.txt). One cvt per rounding.
#pragma unroll
        for (int j = 0; j < Pack::kPairs; ++j) {
          const unsigned int wg = pg[t].w[j];
          const unsigned int wu = pu[t].w[j];
          const __nv_bfloat16 lo = __float2bfloat16_rn(gated_gelu_new_op(
              __bfloat162float(__ushort_as_bfloat16((unsigned short)(wg & 0xFFFFu))),
              __bfloat162float(__ushort_as_bfloat16((unsigned short)(wu & 0xFFFFu)))));
          const __nv_bfloat16 hi = __float2bfloat16_rn(gated_gelu_new_op(
              __bfloat162float(__ushort_as_bfloat16((unsigned short)(wg >> 16))),
              __bfloat162float(__ushort_as_bfloat16((unsigned short)(wu >> 16)))));
          o.w[j] = (static_cast<unsigned int>(__bfloat16_as_ushort(hi)) << 16) |
                   static_cast<unsigned int>(__bfloat16_as_ushort(lo));
        }
#endif
        store_pack(dst + at, o);
      }
    }
  }
}

// Same mapping, one element at a time: for a row length that is not a multiple
// of the pack width, or a base pointer that is not 16-byte aligned.
template <int BLOCK>
__global__ void FK_T5_BOUNDS(BLOCK) gated_gelu_new_scalar_kernel(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    long long rows, int half, int width) {
  const int xstride = static_cast<int>(gridDim.x) * BLOCK;

  for (long long row = blockIdx.y; row < rows; row += gridDim.y) {
    const __nv_bfloat16* __restrict__ gate = in + row * width;
    const __nv_bfloat16* __restrict__ up = gate + half;
    __nv_bfloat16* __restrict__ dst = out + row * half;
    for (int i = static_cast<int>(blockIdx.x) * BLOCK + threadIdx.x; i < half;
         i += xstride) {
      dst[i] = __float2bfloat16_rn(
          gated_gelu_new_op(__bfloat162float(gate[i]), __bfloat162float(up[i])));
    }
  }
}

// Every intermediate of the chain, for the exhaustive semantics probe. Not a
// shipping path: one element per thread, no vectorisation, nine stores.
__global__ void gated_gelu_new_stages_kernel(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ o_p, __nv_bfloat16* __restrict__ o_a,
    __nv_bfloat16* __restrict__ o_b, __nv_bfloat16* __restrict__ o_c,
    __nv_bfloat16* __restrict__ o_d, __nv_bfloat16* __restrict__ o_e,
    __nv_bfloat16* __restrict__ o_f, __nv_bfloat16* __restrict__ o_h,
    __nv_bfloat16* __restrict__ o_y, long long n) {
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    const float f = rb(0.5f * g);
#if FK_T5_FP32_CUBE
    const float p = rb(g * g * g);
#else
    const float p = rb(rb(g * g) * g);
#endif
    const float a = rb(kKappa * p);
    const float b = rb(g + a);
    const float c = rb(kBeta * b);
    const float d = rb(tanh_for_chain(c));
    const float e = rb(1.0f + d);
    const float h = rb(f * e);
    o_p[i] = __float2bfloat16_rn(p);
    o_a[i] = __float2bfloat16_rn(a);
    o_b[i] = __float2bfloat16_rn(b);
    o_c[i] = __float2bfloat16_rn(c);
    o_d[i] = __float2bfloat16_rn(d);
    o_e[i] = __float2bfloat16_rn(e);
    o_f[i] = __float2bfloat16_rn(f);
    o_h[i] = __float2bfloat16_rn(h);
    o_y[i] = __float2bfloat16_rn(h * u);
  }
}

// ---------------------------------------------------------------------------
// Launch geometry
// ---------------------------------------------------------------------------

struct Geometry {
  int block;
  int unroll;
};

constexpr int kMaxGridX = 2147483647;
constexpr int kMaxGridY = 65535;

static inline unsigned grid_x_for(long long items, int block, int unroll) {
  const long long per_block = static_cast<long long>(block) * unroll;
  long long grid = (std::max<long long>(items, 1) + per_block - 1) / per_block;
  grid = std::min<long long>(std::max<long long>(grid, 1), kMaxGridX);
  return static_cast<unsigned>(grid);
}

// Measured, not assumed: profile/tune_geometry.txt sweeps block in {128, 256,
// 512} against unroll in {1, 2, 4, 8} in the benchmark's own timing loop, and
// profile/tune_variants.txt re-runs the top three geometries across build
// variants. One pack in flight per thread wins, and more packs in flight make
// this kernel monotonically *slower* -- at block 256 the activation goes
// 31.7 / 35.8 / 37.9 / 50.1 us for unroll 1 / 2 / 4 / 8. The payload registers
// are why: four packs in flight is 32 registers of payload alone, which is most
// of the 104 registers per thread the first ncu record measured, and that capped
// theoretical occupancy at 25%.
//
// Block size is inside run-to-run noise across 128 / 256 / 512 at unroll 1; 128
// is kept because it is the smallest grid granularity, so a short row spreads
// across more SMs.
//
// The unroll is not stepped up when the row is long, because it never helps; it
// stays at 1 unconditionally, and the loop still handles a row of any length.
static Geometry choose_geometry(long long packs_out_row, long long rows) {
  (void)packs_out_row;
  (void)rows;
  return Geometry{128, 1};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

#define FK_T5_VEC(BLOCK, UNROLL)                                              \
  gated_gelu_new_vec_kernel<BLOCK, UNROLL><<<grid, BLOCK, 0, stream>>>(        \
      reinterpret_cast<const Pack*>(in), reinterpret_cast<Pack*>(out), rows,   \
      packs_out_row, packs_in_row)

static void launch_vector(const void* in, void* out, long long rows,
                          int packs_out_row, int packs_in_row,
                          Geometry g, cudaStream_t stream) {
  const dim3 grid(grid_x_for(packs_out_row, g.block, g.unroll),
                  static_cast<unsigned>(std::min<long long>(rows, kMaxGridY)));
  switch (g.block) {
    case 128:
      switch (g.unroll) {
        case 8: FK_T5_VEC(128, 8); break;
        case 4: FK_T5_VEC(128, 4); break;
        case 2: FK_T5_VEC(128, 2); break;
        default: FK_T5_VEC(128, 1); break;
      }
      break;
    case 512:
      switch (g.unroll) {
        case 8: FK_T5_VEC(512, 8); break;
        case 4: FK_T5_VEC(512, 4); break;
        case 2: FK_T5_VEC(512, 2); break;
        default: FK_T5_VEC(512, 1); break;
      }
      break;
    default:
      switch (g.unroll) {
        case 8: FK_T5_VEC(256, 8); break;
        case 4: FK_T5_VEC(256, 4); break;
        case 2: FK_T5_VEC(256, 2); break;
        default: FK_T5_VEC(256, 1); break;
      }
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef FK_T5_VEC

static void launch_scalar(const void* in, void* out, long long rows, int half,
                          int width, Geometry g, cudaStream_t stream) {
  const dim3 grid(grid_x_for(half, g.block, 1),
                  static_cast<unsigned>(std::min<long long>(rows, kMaxGridY)));
  const auto* src = reinterpret_cast<const __nv_bfloat16*>(in);
  auto* dst = reinterpret_cast<__nv_bfloat16*>(out);
  switch (g.block) {
    case 128:
      gated_gelu_new_scalar_kernel<128><<<grid, 128, 0, stream>>>(src, dst, rows, half, width);
      break;
    case 512:
      gated_gelu_new_scalar_kernel<512><<<grid, 512, 0, stream>>>(src, dst, rows, half, width);
      break;
    default:
      gated_gelu_new_scalar_kernel<256><<<grid, 256, 0, stream>>>(src, dst, rows, half, width);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------

// A tensor carrying any of these keys must not reach the raw-pointer path. Some
// have no storage to point at, some have a null data pointer that would look
// 16-byte aligned, and for the conjugate and negative bits at::empty clears the
// key on the output, so a flat traversal would silently drop the transformation.
// Named is here for the same reason as those two: at::empty produces an unnamed
// result, so the fused path would quietly drop the dimension names the eager
// chain propagates.
static const c10::DispatchKeySet kUnsupportedKeys({
    c10::DispatchKey::Named,
    c10::DispatchKey::Conjugate,
    c10::DispatchKey::Negative,
    c10::DispatchKey::ZeroTensor,
    c10::DispatchKey::NestedTensor,
    c10::DispatchKey::BatchedNestedTensor,
    c10::DispatchKey::Python,
    c10::DispatchKey::PythonTLSSnapshot,
    c10::DispatchKey::Functionalize,
    c10::DispatchKey::FuncTorchBatched,
    c10::DispatchKey::FuncTorchGradWrapper,
    c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
    c10::DispatchKey::FuncTorchDynamicLayerBackMode,
});

// The key screen comes first: a wrapped tensor can throw from the very
// accessors the later checks use. requires_grad is excluded because this is not
// a dispatcher op -- a result produced by the raw kernel would carry no grad_fn
// and silently break the graph, where the eager chain records one.
// The part of the screen that is about the tensor being an ordinary dense bf16
// CUDA tensor with real storage. Shared with the diagnostic entry point, which
// needs exactly these properties before it may take a raw pointer.
bool eligible_operand(const at::Tensor& x) {
  if (!x.defined()) return false;
  if (x.key_set().has_any(kUnsupportedKeys)) return false;
  if (!x.is_cuda()) return false;
  if (x.layout() != at::kStrided) return false;
  if (!x.has_storage()) return false;
  if (x.scalar_type() != at::kBFloat16) return false;
  if (x.dim() < 1) return false;
  if (x.numel() == 0) return false;
  if (!x.is_contiguous()) return false;
  if (x.requires_grad() && at::GradMode::is_enabled()) return false;
  return x.const_data_ptr() != nullptr;
}

static bool eligible(const at::Tensor& x) {
  if (!eligible_operand(x)) return false;
  const long long width = x.size(-1);
  return width > 0 && (width % 2) == 0;
}

static inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// Printed once, on the first call the fused kernel actually serves, so a silent
// fallback cannot masquerade as a pass in the benchmark log. The benchmark runs
// its correctness rounds and its warmup before the timed region, so this lands
// outside every timed window.
static void announce_once(const char* path) {
  static std::atomic<bool> announced{false};
  if (!announced.exchange(true)) {
    std::printf("[t5_dense] fused gated-GELU-new kernel serving calls (%s, %s chain)\n",
                path, FK_T5_FP32_CUBE ? "fp32-cube" : "bit-exact bf16");
    std::fflush(stdout);
  }
}

static std::optional<at::Tensor> run(const at::Tensor& gate_up, Geometry g,
                                    bool auto_geometry, bool force_scalar) {
  if (!eligible(gate_up)) return std::nullopt;

  // Architecture check before anything else touches the device: see
  // FK_T5_BUILD_SM above.
#if FK_T5_BUILD_SM
  {
    const auto* props = at::cuda::getDeviceProperties(gate_up.device().index());
    if (props == nullptr || props->major * 10 + props->minor != FK_T5_BUILD_SM) {
      return std::nullopt;
    }
  }
#endif

  const c10::cuda::CUDAGuard guard(gate_up.device());

  const long long width = gate_up.size(-1);
  const long long half = width / 2;
  const long long rows = gate_up.numel() / width;

  std::vector<int64_t> shape(gate_up.sizes().begin(), gate_up.sizes().end());
  shape.back() = half;
  at::Tensor out = at::empty(shape, gate_up.options());

  const void* in = gate_up.const_data_ptr();
  void* dst = out.mutable_data_ptr();
  if (dst == nullptr) return std::nullopt;

  const bool vector_ok = !force_scalar && (half % Pack::kWidth) == 0 &&
                         (width % Pack::kWidth) == 0 && aligned16(in) && aligned16(dst);
  if (auto_geometry) {
    g = choose_geometry(vector_ok ? half / Pack::kWidth : half, rows);
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // The narrow in-row extents the kernels index with. A row wider than INT_MAX
  // elements cannot occur (it would be a 4 GiB row), but it is checked rather
  // than assumed, because the consequence would be silent corruption. The
  // breadcrumb comes after it, so it can never claim a call the kernel refused.
  if (width > std::numeric_limits<int>::max()) return std::nullopt;
  announce_once(vector_ok ? "vector" : "scalar");
  if (vector_ok) {
    launch_vector(in, dst, rows, static_cast<int>(half / Pack::kWidth),
                  static_cast<int>(width / Pack::kWidth), g, stream);
  } else {
    launch_scalar(in, dst, rows, static_cast<int>(half), static_cast<int>(width), g,
                  stream);
  }
  return out;
}

}  // namespace fk_t5_dense

std::optional<at::Tensor> gated_gelu_new(const at::Tensor& gate_up) {
  return fk_t5_dense::run(gate_up, fk_t5_dense::Geometry{128, 1}, true, false);
}

// Same kernel with the geometry forced, so choose_geometry can be re-measured
// instead of trusted.
std::optional<at::Tensor> gated_gelu_new_with_geometry(const at::Tensor& gate_up,
                                                       int64_t block, int64_t unroll,
                                                       bool force_scalar) {
  TORCH_CHECK(block == 128 || block == 256 || block == 512, "block must be 128, 256 or 512");
  TORCH_CHECK(unroll == 1 || unroll == 2 || unroll == 4 || unroll == 8,
              "unroll must be 1, 2, 4 or 8");
  const fk_t5_dense::Geometry g{static_cast<int>(block), static_cast<int>(unroll)};
  return fk_t5_dense::run(gate_up, g, false, force_scalar);
}

bool gated_gelu_new_is_bit_exact_chain() { return FK_T5_FP32_CUBE == 0; }

// Every intermediate of the chain, computed by the same device code the shipping
// kernel uses, so the exhaustive probe measures this build's flags and not a
// proxy.
std::vector<at::Tensor> gated_gelu_new_stages(const at::Tensor& gate,
                                              const at::Tensor& up) {
  // The same eligibility screen the shipping path applies, to *both* operands.
  // Without it this entry point reaches the raw-pointer launch for tensors that
  // have no storage to point at: a CUDA bf16 ZeroTensor passes every check below
  // -- right dtype, right device, right shape -- and has a null data pointer, so
  // the launch produces an asynchronous illegal access that surfaces at the next
  // synchronize and poisons the context. That was reproduced, not hypothesised.
  TORCH_CHECK(fk_t5_dense::eligible_operand(gate) && fk_t5_dense::eligible_operand(up),
              "stages: both operands must be ordinary dense CUDA bfloat16 tensors "
              "with storage and no unsupported dispatch key");
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "stages: expected CUDA tensors");
  // Both operands are read through raw pointers under a single device guard, so
  // they must live on the *same* device -- otherwise the launch would dereference
  // `up` from the wrong context and fail asynchronously. And the binary holds code
  // for one architecture, so a device of any other capability has no image for it.
  // Both are checked before .contiguous(), which would itself allocate and copy.
  TORCH_CHECK(gate.device() == up.device(),
              "stages: gate and up must be on the same device, got ", gate.device(),
              " and ", up.device());
  TORCH_CHECK(gate.scalar_type() == at::kBFloat16 && up.scalar_type() == at::kBFloat16,
              "stages: expected bfloat16");
  TORCH_CHECK(gate.sizes() == up.sizes(), "stages: shape mismatch");
#if FK_T5_BUILD_SM
  {
    const auto* props = at::cuda::getDeviceProperties(gate.device().index());
    TORCH_CHECK(props != nullptr &&
                    props->major * 10 + props->minor == FK_T5_BUILD_SM,
                "stages: this extension was built for sm_", FK_T5_BUILD_SM,
                " and cannot launch on this device");
  }
#endif
  const at::Tensor g = gate.contiguous();
  const at::Tensor u = up.contiguous();
  const c10::cuda::CUDAGuard guard(g.device());

  std::vector<at::Tensor> outs;
  outs.reserve(9);
  for (int i = 0; i < 9; ++i) outs.push_back(at::empty_like(g));

  const long long n = g.numel();
  if (n == 0) return outs;
  constexpr int kBlock = 256;
  const long long grid = std::min<long long>((n + kBlock - 1) / kBlock, 65535);
  fk_t5_dense::gated_gelu_new_stages_kernel<<<static_cast<unsigned>(grid), kBlock, 0,
                                              at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(g.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(u.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[0].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[1].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[2].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[3].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[4].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[5].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[6].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[7].mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(outs[8].mutable_data_ptr()), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outs;
}
"""

_ENTRY_POINTS = (
    "gated_gelu_new",
    "gated_gelu_new_with_geometry",
    "gated_gelu_new_stages",
    "gated_gelu_new_is_bit_exact_chain",
)


def _local_arch() -> str | None:
    """The single compute capability to build for.

    The ambient ``TORCH_CUDA_ARCH_LIST`` here names six architectures, which
    multiplies compile time by six for a kernel that only ever runs on one
    device. ``None`` means "leave the ambient value alone", which is right only
    when the capability cannot be determined at all. The device this process can
    see is asked first; ``nvidia-smi`` is a fallback only, because it reports
    every GPU on the host regardless of the lease, and on a mixed-architecture
    host it would answer for the wrong one.
    """
    cap = None
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            cap = f"{major}.{minor}"
    except Exception:
        cap = None
    if cap is None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=20)
            caps = sorted({line.strip() for line in out.splitlines() if line.strip()})
            cap = caps[0] if len(caps) == 1 else None
        except Exception:
            cap = None
    if cap is None:
        return None
    major = cap.split(".")[0]
    # Blackwell and Hopper want the architecture-specific variant.
    return f"{cap}a" if major in ("9", "10", "12") and not cap.endswith("a") else cap


def _workspace_root() -> Path:
    """The workspace this candidate file lives in.

    ``candidate/L2/t5_dense.py`` -> two levels up. Everything this module writes
    has to stay inside it.
    """
    return Path(__file__).resolve().parents[2]


def _build_root() -> Path:
    """A build directory this workspace owns.

    The default extension cache is shared by every workspace on the machine and
    keyed only by extension name, and its lock has no stale recovery, so a
    concurrent or previous build elsewhere could otherwise be imported in place
    of this one.

    There is deliberately no fall-back outside the workspace. An earlier version
    tried ``/tmp`` when the workspace was not writable; that trades one
    correctness property (this workspace owns its build tree, so a stale binary
    from somewhere else can never be picked up) for availability of a path that is
    only ever a performance optimisation. If the workspace is not writable, the
    right outcome is no extension and the eager chain.
    """
    workspace = _workspace_root()
    override = os.environ.get("FK_T5_DENSE_BUILD_DIR")
    root = Path(override) if override else workspace / ".torch_extensions"
    root.mkdir(parents=True, exist_ok=True)
    # Resolved *after* mkdir and for the default path too, not only for an
    # override: if `.torch_extensions` is itself a symlink out of the workspace,
    # an unresolved check would pass while every build landed elsewhere. What
    # remains is a TOCTOU window -- a path component replaced after this returns --
    # which a userspace check cannot close.
    resolved = root.resolve()
    if not resolved.is_relative_to(workspace):
        raise RuntimeError(
            f"the build root {root} resolves to {resolved}, outside the workspace "
            f"{workspace}; refusing to build there")
    # A per-process probe name: a fixed one lets two concurrent _build_root() calls
    # unlink each other's file and report the directory unwritable.
    probe = resolved / f".writable.{os.getpid()}"
    probe.touch()
    probe.unlink()
    return resolved


# The extension loader takes a lock file with no timeout and no stale recovery, so
# a build killed between creating that lock and releasing it would make every
# later import wait on it forever rather than fall back. Deleting the file on an
# age heuristic alone is not enough either: too eager and two live builders
# corrupt the directory, too lazy and the wait is unbounded.
#
# An advisory flock fixes both halves, because the kernel releases it when the
# holder dies. Whoever holds ours is the only live builder, so any loader lock
# still present at that moment is by construction orphaned and can be removed;
# and a builder that is still alive keeps the flock, so nobody removes its lock
# from under it.
#
# Both locks below **fail closed**. If the flock cannot be created or is not
# obtained before the deadline, this raises, the caller's except clause turns that
# into "no extension", and the module runs the eager chain. Building anyway would
# be the one outcome the guarantee forbids: two processes writing one build
# directory can produce a truncated or mixed .so that then loads and computes
# nonsense. Slow but right, never fast and wrong -- the same rule the dispatch
# guards follow.
_BUILD_LOCK_WAIT_SECONDS = 900
_BUILD_LOCK_POLL_SECONDS = 0.25

# TORCH_CUDA_ARCH_LIST is process-global, and so is the loader's own module
# registry, so architecture discovery, the environment mutation and load_inline
# have to be one critical section for the *process* -- not per build directory.
# Two build_extension() calls for different variants have different flock paths
# and would otherwise interleave their environment save/restore and leave the
# wrong value behind for whichever compiles next. RLock, because a nested call
# from the same thread must not deadlock.
_ARCH_ENV_LOCK = threading.RLock()


@contextlib.contextmanager
def _build_lock(build_dir: Path):
    """Hold an exclusive flock for the duration of a build, then clear stale locks.

    Raises rather than proceeding unsynchronised. See the note above.
    """
    lock_path = build_dir / ".fk_build.lock"
    fd = None
    held = False
    try:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as exc:
            raise RuntimeError(
                f"cannot create the build lock {lock_path} ({exc}); refusing to build "
                f"unsynchronised") from exc
        # Everything after the open is inside this try, so the descriptor is closed
        # even if the wait loop is interrupted.
        deadline = time.monotonic() + _BUILD_LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"waited {_BUILD_LOCK_WAIT_SECONDS}s for the build lock "
                        f"{lock_path} and did not get it; refusing to build "
                        f"unsynchronised")
                time.sleep(_BUILD_LOCK_POLL_SECONDS)
        # No other fk builder is alive, so a loader lock here is abandoned. It has
        # to go: load_inline waits on it with no deadline, so proceeding while it
        # exists is the unbounded hang this whole mechanism exists to prevent.
        # Failing to remove it therefore raises -- swallowing the error and
        # continuing would hand the process to that wait while still holding this
        # flock, the process-wide lock and a mutated TORCH_CUDA_ARCH_LIST.
        # (`lock` being a directory is the case that makes unlink() fail.)
        stray = build_dir / "lock"
        if stray.exists():
            print(f"[t5_dense] removing an abandoned build lock: {stray}", flush=True)
            try:
                stray.unlink()
            except OSError as exc:
                raise RuntimeError(
                    f"an abandoned loader lock {stray} could not be removed ({exc}); "
                    f"refusing to enter load_inline, which would wait on it forever"
                ) from exc
        yield
    finally:
        if fd is not None:
            try:
                if held:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def build_extension(defines: tuple[str, ...] = (), *, cuda_source: str | None = None,
                    verbose: bool | None = None):
    """Compile and import the embedded extension.

    The extension name carries a hash of the source, the build flags *and* the
    architecture, so a variant build -- the fp32-cube formulation, a deliberately
    broken source used by a test, or a binary compiled for another GPU -- can never
    be confused with the shipping one, in this process or a later one.

    Everything from architecture discovery to the end of ``load_inline`` runs under
    one process-wide lock. That is wider than it looks necessary: the reason is
    that ``TORCH_CUDA_ARCH_LIST`` and the loader's module registry are both
    process-global, so two ``build_extension`` calls for *different* variants --
    which have different build directories and therefore different file locks --
    would otherwise interleave their environment save/restore and leave the wrong
    architecture set for whichever one compiles next.

    Raises on any lock failure. The caller turns that into "no extension" and the
    module runs the eager chain.
    """
    with _ARCH_ENV_LOCK:
        return _build_locked(_CUDA_SOURCE if cuda_source is None else cuda_source,
                             defines, verbose)


def _build_locked(source: str, defines: tuple[str, ...], verbose: bool | None):
    """The body of build_extension, with _ARCH_ENV_LOCK already held."""
    from torch.utils.cpp_extension import load_inline

    arch = _local_arch()
    # Tell the kernel which capability it was built for, so dispatch can refuse a
    # device it has no image for instead of failing asynchronously at launch.
    sm = 0
    if arch:
        try:
            major, minor = arch.rstrip("a").split(".")
            sm = int(major) * 10 + int(minor)
        except ValueError:
            sm = 0
    # No fast-math anywhere: a redirected tanhf would break bit-exactness, and
    # denormal flushing would change the tails. The defaults already say this; the
    # flags say it explicitly so the hash records it and a future edit cannot
    # quietly drop it.
    cuda_flags = ["-O3", "-lineinfo", "--ftz=false", "--prec-div=true",
                  "--prec-sqrt=true", "--fmad=true", f"-DFK_T5_BUILD_SM={sm}", *defines]
    # When the capability cannot be determined the ambient list is what nvcc will
    # actually use, so it -- not the word "ambient" -- is what has to be in the
    # key, or two processes with different lists would share a name and the second
    # would import the first one's binary.
    arch_key = arch or f"ambient:{os.environ.get('TORCH_CUDA_ARCH_LIST', '')}"
    key = hashlib.sha256(
        "\x00".join([_CPP_SOURCE, source, *cuda_flags, arch_key]).encode()
    ).hexdigest()[:16]
    name = f"fk_t5_dense_{key}"

    build_dir = _build_root() / name
    build_dir.mkdir(parents=True, exist_ok=True)
    cold = not (build_dir / f"{name}.so").exists()
    if cold:
        # Keep the log moving: a silent compile can trip a no-output watchdog.
        print(f"[t5_dense] compiling {name} for arch {arch} (cold cache)", flush=True)

    with _build_lock(build_dir):
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        try:
            # Inside the try, so an interrupt between assigning and entering it
            # cannot leave the process-global value changed.
            if arch:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            return load_inline(
                name=name,
                cpp_sources=[_CPP_SOURCE],
                cuda_sources=[source],
                functions=list(_ENTRY_POINTS),
                extra_cflags=["-O3"],
                extra_cuda_cflags=cuda_flags,
                build_directory=str(build_dir),
                verbose=cold if verbose is None else verbose,
            )
        finally:
            if arch:
                if previous is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0")


def _defines_from_env() -> tuple[str, ...]:
    defines = []
    if _env_flag("FK_T5_DENSE_FP32_CUBE"):
        defines.append("-DFK_T5_FP32_CUBE=1")
    hints = os.environ.get("FK_T5_DENSE_CACHE_HINTS", "")
    if hints != "":
        defines.append(f"-DFK_T5_CACHE_HINTS={int(hints)}")
    minb = os.environ.get("FK_T5_DENSE_MIN_BLOCKS", "")
    if minb != "":
        defines.append(f"-DFK_T5_MIN_BLOCKS={int(minb)}")
    packed = os.environ.get("FK_T5_DENSE_PACKED_CVT", "")
    if packed != "":
        defines.append(f"-DFK_T5_PACKED_CVT={int(packed)}")
    return tuple(defines)


def _source_from_env() -> str | None:
    if _env_flag("FK_T5_DENSE_BREAK_BUILD"):
        return _CUDA_SOURCE + "\n#error deliberate build failure (FK_T5_DENSE_BREAK_BUILD)\n"
    return None


def _geometry_from_env() -> tuple[int, int] | None:
    spec = os.environ.get("FK_T5_DENSE_GEOMETRY", "")
    if not spec:
        return None
    try:
        block, unroll = (int(part) for part in spec.split(","))
    except ValueError:
        print(f"[t5_dense] ignoring malformed FK_T5_DENSE_GEOMETRY={spec!r}", flush=True)
        return None
    # Checked here rather than left to the C++ TORCH_CHECK, so an unsupported
    # value is a rejected switch at import instead of an exception from forward.
    if block not in (128, 256, 512) or unroll not in (1, 2, 4, 8):
        print(f"[t5_dense] ignoring unsupported FK_T5_DENSE_GEOMETRY={spec!r} "
              f"(block must be 128/256/512, unroll 1/2/4/8)", flush=True)
        return None
    return block, unroll


_EXT = None
_IMPORT_ERROR: str | None = None
if os.environ.get("FK_T5_DENSE_FUSED", "1") not in ("", "0"):
    try:
        _EXT = build_extension(_defines_from_env(), cuda_source=_source_from_env())
    except Exception as exc:  # noqa: BLE001 - a build failure must not break the module
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[t5_dense] extension unavailable, falling back to the eager chain "
              f"({_IMPORT_ERROR})", flush=True)
else:
    print("[t5_dense] fused kernel disabled by FK_T5_DENSE_FUSED=0", flush=True)


def _select_impl():
    """The callable that serves the activation, or None for the eager chain."""
    if _EXT is None:
        return None
    geometry = _geometry_from_env()
    force_scalar = _env_flag("FK_T5_DENSE_SCALAR")
    if geometry is None and not force_scalar:
        return _EXT.gated_gelu_new
    block, unroll = geometry if geometry is not None else (128, 1)
    print(f"[t5_dense] geometry forced to block={block} unroll={unroll} "
          f"scalar={force_scalar}", flush=True)

    def _impl(gate_up, _block=block, _unroll=unroll, _scalar=force_scalar):
        return _EXT.gated_gelu_new_with_geometry(gate_up, _block, _unroll, _scalar)

    return _impl


_FUSED_IMPL = _select_impl()


def _mode_active() -> bool:
    """Is a __torch_dispatch__ or __torch_function__ mode on the stack?

    Modes are thread-local state: they need not add a dispatch key or wrap the
    tensor in a subclass, so nothing else here would see them. A mode that
    rewrites or merely observes the activation's ops must see the eager chain,
    because the fused kernel does not go through the dispatcher at all -- and it
    would also miss the internal ``at::empty``. Two integer reads on a path whose
    floor is tens of microseconds.
    """
    try:
        return (torch._C._len_torch_dispatch_stack() > 0
                or torch._C._len_torch_function_stack() > 0)
    except AttributeError:  # pragma: no cover - older torch without the accessors
        return False


def _carries_forward_grad(x: torch.Tensor) -> bool:
    """Does x hold a forward-mode tangent?

    A dual tensor has exact type ``torch.Tensor`` and ``requires_grad=False``, so
    nothing else here would stop it, and the kernel writes into a fresh
    ``at::empty`` with no tangent attached -- the derivative would vanish
    silently. Checking the active dual level first makes this a single integer
    comparison when nobody is doing forward AD, which is always, in the
    benchmark.
    """
    if getattr(_forward_ad, "_current_level", 0) < 0:
        return False
    return _forward_ad.unpack_dual(x).tangent is not None


class NewGELUActivation(nn.Module):
    """GELU approximation matching HuggingFace's NewGELUActivation exactly."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def _get_act_fn(name: str) -> nn.Module:
    act_fns = {
        "relu": nn.ReLU(),
        "gelu": GELU(),
        "gelu_new": NewGELUActivation(),
        "silu": SiLU(),
    }
    if name in act_fns:
        return act_fns[name]
    raise ValueError(f"Unknown activation function: {name}")


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model, [config.d_ff, config.d_ff], bias=False,
        )
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)
        # The fused kernel computes one specific activation, so anything other
        # than "gelu_new" must keep the eager chain or the class would silently
        # compute the wrong function. Tensor parallelism is excluded because the
        # sharded path is never exercised here (the benchmark runs one GPU per
        # operator), not because the elementwise split would be wrong: the
        # merged weight stays laid out as [gate shard | up shard].
        self._fused = _FUSED_IMPL if (
            config.dense_act_fn == "gelu_new" and _tp_size() == 1) else None
        self._gate_up_width = 2 * (config.d_ff // _tp_size())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.wi(hidden_states)
        if (self._fused is not None and type(gate_up) is torch.Tensor
                and gate_up.size(-1) == self._gate_up_width
                and not _mode_active()
                and not _carries_forward_grad(gate_up)):
            fused = self._fused(gate_up)
            if fused is not None:
                return self.wo(fused)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
