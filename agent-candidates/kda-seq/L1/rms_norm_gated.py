"""Gated RMSNorm with an element-wise multiplicative gate (L1).

Computes ``out = RMSNorm(x, weight) * gate(z)`` (or ``RMSNorm(x * gate(z), weight)``
when ``norm_before_gate`` is false) in one fused kernel. It reproduces
``baseline.py::RMSNormGated`` to well inside the benchmark's comparison tolerance rather
than bit-exactly: every intermediate is fp32 and the result is rounded to the input dtype
exactly once on store, as in the baseline, but the reduction is differently parenthesized
and two fp32 primitives differ, so single-ULP disagreements do occur (up to 1.56e-2 on a
bf16 row of 128, against a bound of atol=rtol=1e-2 on 99% of elements).

Three implementations, chosen per call, widest-contract last:

* **Triton**, for the shape class the benchmark actually scores: bf16 ``x`` and ``z``,
  contiguous and 16-byte aligned, a power-of-two row length. Specializing the row length
  as a compile-time constant is worth ~8 us of 43 us on the largest captured shape, which
  is why this path exists at all — see the A/B in ``analysis/measure_speedup.py``.
* a **CUDA extension** built lazily via ``load_inline``, covering every other valid input:
  any row length, fp16/fp32, mismatched weight dtype, misaligned or non-contiguous
  storage. It has a vectorized path (``threads_per_row`` lanes per row, a register tile
  each, reduced with ``__shfl_xor_sync``) and a scalar path (one warp per row).
* an **eager PyTorch forward**, for what neither kernel covers — ``z=None``, an ``x``/``z``
  dtype pair that differs, float64 — and for a toolchain that cannot build the extension.

Only the extension build is guarded by ``try``. A kernel launch failure is allowed to
surface, because a swallowed asynchronous fault hides a real defect and a poisoned CUDA
context cannot be recovered from anyway.
"""

from __future__ import annotations

import os
import subprocess
import warnings

import torch
import torch.nn as nn
import triton
import triton.language as tl

__targets__ = ["RMSNormGated"]


# --- Gate selection -------------------------------------------------------------
# The vendored Triton kernel treats "swish" and "silu" as the same gate, "sigmoid" as
# the plain logistic, and *any other string* as no gate at all (its if/elif chain has
# no else, so ``y`` is stored unmultiplied). These codes reproduce that, including the
# silent no-gate case, so an unexpected activation name diverges nowhere.
_GATE_SWISH = 0
_GATE_SIGMOID = 1
_GATE_NONE = 2
_GATE_CODES = {"swish": _GATE_SWISH, "silu": _GATE_SWISH, "sigmoid": _GATE_SIGMOID}

# Dtypes the kernel is instantiated for. Anything else the baseline accepts -- float64, or
# an x/z dtype pair that differs -- goes to the eager fallback rather than being rejected,
# so the candidate never refuses an input the baseline would have computed.
_KERNEL_DTYPES = frozenset((torch.bfloat16, torch.float16, torch.float32))


# --- Tuning knobs ---------------------------------------------------------------
# Read once at import, not per call: these select a row mapping, not operator
# behaviour, so freezing them cannot make the module diverge from the baseline.
# 0 means "let the kernel pick the natural 128-bit width for the dtype".
_VALUES_PER_THREAD = int(os.environ.get("FK_RMSNORM_VALUES_PER_THREAD", "0"))
_THREADS_PER_BLOCK = int(os.environ.get("FK_RMSNORM_THREADS", "0"))
_MAX_BLOCKS = int(os.environ.get("FK_RMSNORM_MAX_BLOCKS", "0"))

# Which implementation may serve the scored shape class: "auto" (Triton when eligible),
# "cuda" (never Triton) or "triton". Exists so the backend A/B compares the shipped file
# against itself through `validate.py` rather than against a separate copy.
_BACKEND = os.environ.get("FK_RMSNORM_BACKEND", "auto").strip().lower()
_TRITON_ROWS_PER_BLOCK = int(os.environ.get("FK_RMSNORM_TRITON_ROWS", "16"))
_TRITON_NUM_WARPS = int(os.environ.get("FK_RMSNORM_TRITON_WARPS", "4"))

# Row lengths the Triton path will specialize for. Bounded and enumerated on purpose:
# Triton JIT-compiles once per (row length, gate, order) combination, and a combination
# first seen inside the harness's timing window would compile there, which the benchmark
# forbids. Every combination the benchmark reaches is compiled during its correctness
# rounds, which run before timing for each case.
_TRITON_ROW_LENGTHS = frozenset((32, 64, 128, 256, 512, 1024, 2048))
_TRITON_WEIGHT_DTYPES = frozenset((torch.bfloat16, torch.float32))

# Which implementation served the most recent call, for the correctness script to assert
# against. Diagnostic only.
_LAST_BACKEND = "none"


_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor rms_norm_gated(const at::Tensor& x, const at::Tensor& z,
                          const at::Tensor& weight, double eps,
                          int64_t gate, bool norm_before_gate);
void configure(int64_t values_per_thread, int64_t threads_per_block,
               int64_t max_blocks);
int64_t last_path();
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int GATE_SWISH = 0;
constexpr int GATE_SIGMOID = 1;
constexpr int GATE_NONE = 2;

// Row mapping, overridable from Python for tuning sweeps. 0 == pick the natural
// 128-bit-per-thread width for the element type.
int g_values_per_thread = 0;
int g_threads_per_block = 128;

// A positive value caps the vector launch's grid and makes it stride over rows instead,
// i.e. a persistent launch. 0 leaves one row-group per lane group, which is the default.
int g_max_blocks = 0;

// Which path the most recent call took: 0 for the scalar kernel, else the vector
// kernel's per-lane tile width. Host-side and diagnostic only -- the correctness
// script reads it so that "the vector path was taken" is observed rather than
// inferred. Single-threaded like the rest of the launch path.
int g_last_path = -1;

__device__ __forceinline__ float to_f32(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f32(const __half v) { return __half2float(v); }
__device__ __forceinline__ float to_f32(const float v) { return v; }

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ __forceinline__ __half from_f32<__half>(float v) {
  return __float2half(v);
}
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }

// One register tile of COUNT consecutive elements. Aligned to 16 B once the tile is
// that wide, so nvcc emits ld.global.v4.u32 / st.global.v4.u32 rather than a scalar
// burst; narrower tiles fall back to their natural width.
template <typename T, int COUNT>
struct alignas(sizeof(T) * COUNT < 16 ? sizeof(T) * COUNT : 16) Tile {
  T v[COUNT];
};

// The fp32 primitives below were chosen against the baseline kernel's own PTX, dumped and
// read rather than assumed -- it is kept at analysis/artifacts/ so the claims about it stay
// checkable -- and then against this kernel's SASS. Profiling says both
// kernels are ALU-issue-bound at the largest shape rather than bandwidth-bound (DRAM
// throughput 37%, ALU pipe 56%), so instruction count per element is what matters, and
// CUDA's accurate expf / div.rn are multi-instruction software sequences.
//
// Two deliberate departures from the baseline's arithmetic, both bounded well inside the
// comparison tolerance and both checked against an fp64 reference by
// analysis/check_correctness.py:
//
// * The gate's reciprocal uses rcp.approx.f32 where Triton emits div.full.f32, which
//   expands to a multi-instruction full-range sequence.
//   rcp.approx is a single instruction on a different pipe. Its accuracy is not asserted
//   from a specification here -- the PTX ISA documents it as a fast approximate
//   reciprocal -- but measured: analysis/check_correctness.py requires this kernel's worst
//   element-wise error against an fp64 reference to stay under
//   max(2 x the baseline's own error, one ULP of the storage dtype) on every case in the
//   matrix, so the candidate can never be materially less accurate than what it replaces.
// * exp uses the .ftz form. MUFU is flush-to-zero hardware, so ptxas wraps a non-ftz
//   approximation in a denormal fixup sequence; the non-ftz version of this kernel
//   compiled to 248 SASS instructions to process 8 elements, of which 9 FSETP.GT, 9
//   FSETP.GEU, 8 FSEL and much of its 76 FMULs were nothing but those fixups. Flushing is
//   safe for exp specifically: its result is only denormal when exp(-z) underflows, and
//   1.0f + denormal rounds to exactly 1.0f whether or not the denormal was flushed.
//
// The reciprocal is deliberately *not* .ftz, even though that costs the fixup sequence.
// sigmoid(z) is genuinely subnormal for z below about -88, and a large weight multiplies
// it back into range: at bf16 x=1, z=-88, fp32 weight=1e38 the baseline returns 0.605
// while a flushing reciprocal returns 0, failing the comparison on every element of the
// row. That input never occurs in the benched configuration -- the harness rewrites any
// parameter whose amax exceeds 1e4 -- but the operator's contract is not limited to the
// benched configuration, so the correct arithmetic wins over the cheaper one here.
__device__ __forceinline__ float fast_exp(float x) {
  float r;
  // 0f3FB8AA3B is log2(e), the same scale Triton applies before its own ex2.approx.
  asm("{ .reg .f32 t; mul.f32 t, %1, 0f3FB8AA3B; ex2.approx.ftz.f32 %0, t; }"
      : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float fast_rcp(float v) {
  float r;
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

// Kept as div.full.f32: this one runs once per row, not once per element, so matching the
// baseline's rounding costs nothing measurable.
__device__ __forceinline__ float fast_div(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float fast_rsqrt(float v) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ float sigmoid_of(float z) {
  return fast_rcp(1.0f + fast_exp(-z));
}

template <int GATE>
__device__ __forceinline__ float gate_of(float z) {
  if (GATE == GATE_SWISH) return z * sigmoid_of(z);
  if (GATE == GATE_SIGMOID) return sigmoid_of(z);
  return 1.0f;  // unrecognized activation: the baseline stores y unmultiplied
}

__device__ __forceinline__ float gate_of_runtime(float z, int gate) {
  if (gate == GATE_SWISH) return z * sigmoid_of(z);
  if (gate == GATE_SIGMOID) return sigmoid_of(z);
  return 1.0f;
}

// ---------------------------------------------------------------------------
// Vectorized path: threads_per_row lanes cooperate on one row, each holding
// VALUES_PER_THREAD elements. threads_per_row is a runtime power of two (it is
// row_length / VALUES_PER_THREAD, and the row length is not a compile-time
// constant) passed as a shift so the index math stays shifts and masks.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename weight_t, int VALUES_PER_THREAD, int GATE,
          bool NORM_BEFORE_GATE>
__global__ void rms_norm_gated_vector_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ z,
    const weight_t* __restrict__ weight, scalar_t* __restrict__ out,
    int64_t rows, int row_length, float eps, int log2_threads_per_row,
    int iterations) {
  using XTile = Tile<scalar_t, VALUES_PER_THREAD>;
  using WTile = Tile<weight_t, VALUES_PER_THREAD>;

  const int threads_per_row = 1 << log2_threads_per_row;
  const int lane_in_row = threadIdx.x & (threads_per_row - 1);
  const int rows_per_block = blockDim.x >> log2_threads_per_row;
  const int64_t first_row =
      static_cast<int64_t>(blockIdx.x) * rows_per_block + (threadIdx.x >> log2_threads_per_row);
  const int64_t row_stride = static_cast<int64_t>(gridDim.x) * rows_per_block;

  const int64_t column = static_cast<int64_t>(lane_in_row) * VALUES_PER_THREAD;

  // The weight tile is the same for every row this lane touches, so it is loaded once
  // outside the loop. With one iteration -- the default launch, where the grid covers
  // every row -- this is identical to loading it inline.
  const WTile wt = *reinterpret_cast<const WTile*>(weight + column);

  // Grid-stride, with a trip count computed on the host and therefore uniform across the
  // whole grid. Uniformity is load-bearing: every lane has to reach every shuffle below,
  // so the loop must not end early for some lane groups of a warp and not others.
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int64_t row = first_row + static_cast<int64_t>(iteration) * row_stride;
    const bool in_range = row < rows;
    const int64_t offset = row * row_length + column;

    // x and z are issued together rather than as two dependent memory phases. Worth
    // ~0.2 us of 43 us on the largest shape -- i.e. nothing, because nvcc already
    // hoisted the z load when it sat after the reduction; kept because it states the
    // intent directly. z stays in its storage dtype, since converting it lazily costs a
    // few ALU ops and saves half the registers an fp32 copy would need.
    float value[VALUES_PER_THREAD];
    XTile zt;
    float sum_of_squares = 0.0f;
    if (in_range) {
      const XTile xt = *reinterpret_cast<const XTile*>(x + offset);
      zt = *reinterpret_cast<const XTile*>(z + offset);
      if (!NORM_BEFORE_GATE) {
        // The gate is folded in before the reduction, so it changes the variance and
        // the same gated value is what gets normalized.
#pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
          const float v = to_f32(xt.v[i]) * gate_of<GATE>(to_f32(zt.v[i]));
          value[i] = v;
          sum_of_squares += v * v;
        }
      } else {
#pragma unroll
        for (int i = 0; i < VALUES_PER_THREAD; ++i) {
          const float v = to_f32(xt.v[i]);
          value[i] = v;
          sum_of_squares += v * v;
        }
      }
    }

    // Every lane reaches the shuffles, including lanes whose row is past the end --
    // they contribute zero. A full 0xffffffff mask with lanes that returned early is
    // undefined behaviour, and only row counts that are not a multiple of
    // rows_per_block reach this case, which the captured shapes never do.
    for (int offset_in_group = threads_per_row >> 1; offset_in_group > 0;
         offset_in_group >>= 1) {
      sum_of_squares += __shfl_xor_sync(0xffffffffu, sum_of_squares, offset_in_group);
    }
    if (!in_range) continue;

    // eps is added to the mean of squares, not to the sum.
    const float rstd =
        fast_rsqrt(fast_div(sum_of_squares, static_cast<float>(row_length)) + eps);

    XTile yt;
    if (NORM_BEFORE_GATE) {
#pragma unroll
      for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        float y = value[i] * rstd * to_f32(wt.v[i]);
        y *= gate_of<GATE>(to_f32(zt.v[i]));
        yt.v[i] = from_f32<scalar_t>(y);
      }
    } else {
#pragma unroll
      for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        yt.v[i] = from_f32<scalar_t>(value[i] * rstd * to_f32(wt.v[i]));
      }
    }
    *reinterpret_cast<XTile*>(out + offset) = yt;
  }
}

// ---------------------------------------------------------------------------
// Scalar path: one warp per row, lanes stride over columns. Handles any row
// length, any dtype pairing and any alignment. It reads the row twice, which is
// why it is not the path the captured shapes take.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename weight_t>
__global__ void rms_norm_gated_scalar_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ z,
    const weight_t* __restrict__ weight, scalar_t* __restrict__ out,
    int64_t rows, int row_length, float eps, int gate, bool norm_before_gate) {
  const int lane = threadIdx.x & 31;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  // Warp-uniform, so returning here cannot desynchronize the shuffles below.
  if (row >= rows) return;

  const scalar_t* x_row = x + row * row_length;
  const scalar_t* z_row = z + row * row_length;
  scalar_t* out_row = out + row * row_length;

  float sum_of_squares = 0.0f;
  for (int c = lane; c < row_length; c += 32) {
    float v = to_f32(x_row[c]);
    if (!norm_before_gate) v *= gate_of_runtime(to_f32(z_row[c]), gate);
    sum_of_squares += v * v;
  }
#pragma unroll
  for (int offset_in_group = 16; offset_in_group > 0; offset_in_group >>= 1) {
    sum_of_squares += __shfl_xor_sync(0xffffffffu, sum_of_squares, offset_in_group);
  }
  const float rstd = fast_rsqrt(fast_div(sum_of_squares, static_cast<float>(row_length)) + eps);

  for (int c = lane; c < row_length; c += 32) {
    float v = to_f32(x_row[c]);
    const float g = gate_of_runtime(to_f32(z_row[c]), gate);
    if (!norm_before_gate) v *= g;
    float y = v * rstd * to_f32(weight[c]);
    if (norm_before_gate) y *= g;
    out_row[c] = from_f32<scalar_t>(y);
  }
}

// ---------------------------------------------------------------------------
// Launch helpers.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename weight_t, int VALUES_PER_THREAD>
void launch_vector(int64_t rows, int row_length, float eps, int gate,
                   bool norm_before_gate, int threads, int log2_threads_per_row,
                   const scalar_t* x, const scalar_t* z, const weight_t* weight,
                   scalar_t* out, cudaStream_t stream) {
  const int rows_per_block = threads >> log2_threads_per_row;
  int64_t blocks = (rows + rows_per_block - 1) / rows_per_block;
  int iterations = 1;
  // A positive cap turns the launch persistent: fewer blocks, each striding over several
  // rows. Off by default -- the sweep in analysis/sweep_mapping.py found it never won,
  // since even the largest shape already runs 18 waves and has no tail worth closing.
  if (g_max_blocks > 0 && blocks > g_max_blocks) {
    blocks = g_max_blocks;
    const int64_t rows_per_wave = blocks * rows_per_block;
    iterations = static_cast<int>((rows + rows_per_wave - 1) / rows_per_wave);
  }
  TORCH_CHECK(blocks <= 2147483647L, "rms_norm_gated: grid too large");

#define FK_LAUNCH(GATE_CONST, NBG_CONST)                                            \
  rms_norm_gated_vector_kernel<scalar_t, weight_t, VALUES_PER_THREAD, GATE_CONST,   \
                               NBG_CONST>                                           \
      <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(                   \
          x, z, weight, out, rows, row_length, eps, log2_threads_per_row, iterations)
  if (norm_before_gate) {
    switch (gate) {
      case GATE_SWISH: FK_LAUNCH(GATE_SWISH, true); break;
      case GATE_SIGMOID: FK_LAUNCH(GATE_SIGMOID, true); break;
      default: FK_LAUNCH(GATE_NONE, true); break;
    }
  } else {
    switch (gate) {
      case GATE_SWISH: FK_LAUNCH(GATE_SWISH, false); break;
      case GATE_SIGMOID: FK_LAUNCH(GATE_SIGMOID, false); break;
      default: FK_LAUNCH(GATE_NONE, false); break;
    }
  }
#undef FK_LAUNCH
}

template <typename scalar_t, typename weight_t>
void launch_scalar(int64_t rows, int row_length, float eps, int gate,
                   bool norm_before_gate, int threads, const scalar_t* x,
                   const scalar_t* z, const weight_t* weight, scalar_t* out,
                   cudaStream_t stream) {
  const int rows_per_block = threads >> 5;
  const int64_t blocks = (rows + rows_per_block - 1) / rows_per_block;
  TORCH_CHECK(blocks <= 2147483647L, "rms_norm_gated: grid too large");
  rms_norm_gated_scalar_kernel<scalar_t, weight_t>
      <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
          x, z, weight, out, rows, row_length, eps, gate, norm_before_gate);
}

int log2_exact(int v) {
  int shift = 0;
  while ((1 << shift) < v) ++shift;
  return (1 << shift) == v ? shift : -1;
}

bool aligned(const void* p, int bytes) {
  return (reinterpret_cast<uintptr_t>(p) & static_cast<uintptr_t>(bytes - 1)) == 0;
}

// Can `values_per_thread` elements per lane cover this row with a power-of-two lane
// group that fits in a warp, at an alignment every vector access can rely on?
template <typename scalar_t, typename weight_t>
bool vector_path_ok(int values_per_thread, int row_length, int threads,
                    const void* x, const void* z, const void* weight, const void* out,
                    int* log2_threads_per_row) {
  if (values_per_thread <= 0 || row_length % values_per_thread != 0) return false;
  const int threads_per_row = row_length / values_per_thread;
  if (threads_per_row > 32) return false;
  const int shift = log2_exact(threads_per_row);
  if (shift < 0 || (threads >> shift) < 1 || threads % threads_per_row != 0) return false;

  const int x_bytes = std::min<int>(16, static_cast<int>(sizeof(scalar_t)) * values_per_thread);
  const int w_bytes = std::min<int>(16, static_cast<int>(sizeof(weight_t)) * values_per_thread);
  // Row starts are row_length elements apart; the tile alignment must survive that
  // stride as well as the base pointer.
  if ((static_cast<int64_t>(row_length) * sizeof(scalar_t)) % x_bytes != 0) return false;
  if (!aligned(x, x_bytes) || !aligned(z, x_bytes) || !aligned(out, x_bytes)) return false;
  if (!aligned(weight, w_bytes)) return false;
  *log2_threads_per_row = shift;
  return true;
}

template <typename scalar_t, typename weight_t>
void dispatch_typed(const at::Tensor& x, const at::Tensor& z, const at::Tensor& weight,
                    at::Tensor& out, int64_t rows, int row_length, float eps, int gate,
                    bool norm_before_gate, cudaStream_t stream) {
  const scalar_t* xp = reinterpret_cast<const scalar_t*>(x.const_data_ptr());
  const scalar_t* zp = reinterpret_cast<const scalar_t*>(z.const_data_ptr());
  const weight_t* wp = reinterpret_cast<const weight_t*>(weight.const_data_ptr());
  scalar_t* op = reinterpret_cast<scalar_t*>(out.data_ptr());

  const int threads = g_threads_per_block;
  constexpr int natural = 16 / static_cast<int>(sizeof(scalar_t));
  const int wanted = g_values_per_thread;  // 0 == choose automatically
  int shift = 0;

#define FK_TRY_TILE(WIDTH)                                                             \
  if ((wanted == 0 || wanted == (WIDTH)) &&                                            \
      vector_path_ok<scalar_t, weight_t>((WIDTH), row_length, threads, xp, zp, wp, op,  \
                                         &shift)) {                                    \
    g_last_path = (WIDTH);                                                             \
    launch_vector<scalar_t, weight_t, (WIDTH)>(rows, row_length, eps, gate,             \
                                               norm_before_gate, threads, shift, xp,    \
                                               zp, wp, op, stream);                    \
    return;                                                                            \
  }

  // Alternate tile widths are instantiated only for the bf16/bf16 pairing the captured
  // shapes use; every other pairing gets the natural 128-bit width. That keeps the sweep
  // in analysis/sweep_mapping.py measuring the shipped binary rather than a separate
  // build, without paying for three times the kernel instantiations.
  //
  // Widest first. A 256-bit tile halves the per-element instruction count of a 128-bit
  // one, and this kernel is issue-bound rather than bandwidth-bound, so wider wins: at
  // the largest shape a 16-element tile with 128 threads ran at 1.19x a torch.add
  // reference over the same bytes where an 8-element tile with 256 threads ran at 1.56x.
  // Trying 16 first also brings row lengths of 512 onto the vector path, which a
  // 128-bit tile cannot reach without a lane group wider than a warp.
  if constexpr (std::is_same<scalar_t, __nv_bfloat16>::value &&
                std::is_same<weight_t, __nv_bfloat16>::value) {
    FK_TRY_TILE(16)
    FK_TRY_TILE(8)
    FK_TRY_TILE(4)
  } else {
    FK_TRY_TILE(natural)
  }
#undef FK_TRY_TILE

  g_last_path = 0;
  launch_scalar<scalar_t, weight_t>(rows, row_length, eps, gate, norm_before_gate, threads,
                                    xp, zp, wp, op, stream);
}

}  // namespace

int64_t last_path() { return g_last_path; }

void configure(int64_t values_per_thread, int64_t threads_per_block,
               int64_t max_blocks) {
  if (values_per_thread > 0) g_values_per_thread = static_cast<int>(values_per_thread);
  if (threads_per_block > 0) {
    TORCH_CHECK(threads_per_block >= 32 && threads_per_block <= 1024 &&
                    threads_per_block % 32 == 0,
                "rms_norm_gated: threads_per_block must be a multiple of 32 in [32, 1024]");
    g_threads_per_block = static_cast<int>(threads_per_block);
  }
  g_max_blocks = max_blocks > 0 ? static_cast<int>(max_blocks) : 0;
}

at::Tensor rms_norm_gated(const at::Tensor& x, const at::Tensor& z,
                          const at::Tensor& weight, double eps, int64_t gate,
                          bool norm_before_gate) {
  TORCH_CHECK(x.is_cuda() && z.is_cuda() && weight.is_cuda(),
              "rms_norm_gated: x, z and weight must be CUDA tensors");
  TORCH_CHECK(x.dim() >= 1, "rms_norm_gated: x must have at least one dimension");
  TORCH_CHECK(z.sizes() == x.sizes(), "rms_norm_gated: z shape ", z.sizes(),
              " != x shape ", x.sizes());
  TORCH_CHECK(z.scalar_type() == x.scalar_type(),
              "rms_norm_gated: z dtype ", z.scalar_type(), " != x dtype ",
              x.scalar_type());

  const int64_t row_length64 = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == row_length64,
              "rms_norm_gated: weight must be 1-D of length ", row_length64, ", got ",
              weight.sizes());
  TORCH_CHECK(row_length64 <= 2147483647L, "rms_norm_gated: row too long");

  const at::cuda::CUDAGuard guard(x.device());

  // The baseline's input_guard contiguizes every tensor argument and
  // _layer_norm_fn_impl contiguizes weight again, so all three are contiguized here.
  const at::Tensor xc = x.contiguous();
  const at::Tensor zc = z.contiguous();
  at::Tensor wc = weight.contiguous();
  at::Tensor out = at::empty_like(xc);

  // Zero-element inputs would otherwise divide by a zero row length and launch an
  // empty grid, which CUDA rejects.
  if (xc.numel() == 0) return out;

  const int row_length = static_cast<int>(row_length64);
  const int64_t rows = xc.numel() / row_length64;
  const float eps_f = static_cast<float>(eps);
  const int gate_i = static_cast<int>(gate);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

  // A weight dtype outside the instantiated set is promoted to fp32 first. bf16 and
  // fp16 widen to fp32 exactly, so this changes no result -- and it keeps the benched
  // pairing (bf16 x with a bf16 weight) off the promotion path entirely.
  const at::ScalarType xt = xc.scalar_type();
  if (wc.scalar_type() != xt && wc.scalar_type() != at::kFloat) {
    wc = wc.to(at::kFloat);
  }
  const bool weight_is_x_dtype = wc.scalar_type() == xt;

  switch (xt) {
    case at::kBFloat16:
      if (weight_is_x_dtype) {
        dispatch_typed<__nv_bfloat16, __nv_bfloat16>(xc, zc, wc, out, rows, row_length,
                                                     eps_f, gate_i, norm_before_gate,
                                                     stream);
      } else {
        dispatch_typed<__nv_bfloat16, float>(xc, zc, wc, out, rows, row_length, eps_f,
                                             gate_i, norm_before_gate, stream);
      }
      break;
    case at::kHalf:
      if (weight_is_x_dtype) {
        dispatch_typed<__half, __half>(xc, zc, wc, out, rows, row_length, eps_f, gate_i,
                                       norm_before_gate, stream);
      } else {
        dispatch_typed<__half, float>(xc, zc, wc, out, rows, row_length, eps_f, gate_i,
                                      norm_before_gate, stream);
      }
      break;
    case at::kFloat:
      dispatch_typed<float, float>(xc, zc, wc, out, rows, row_length, eps_f, gate_i,
                                   norm_before_gate, stream);
      break;
    default:
      TORCH_CHECK(false, "rms_norm_gated: unsupported dtype ", xt);
  }
  return out;
}
"""


# --- Triton path: the shape class the benchmark scores ---------------------------
@triton.jit
def _triton_rms_norm_gated(X, Z, W, Y, rows, eps,
                           ROW_LENGTH: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr,
                           GATE: tl.constexpr, NORM_BEFORE_GATE: tl.constexpr):
    """One block per ROWS_PER_BLOCK rows; the row length is a compile-time constant.

    Structured to round exactly where the baseline rounds: fp32 throughout, eps added to
    the mean of squares, ``(x * rstd) * w`` then the gate, one store.
    """
    row_ids = tl.program_id(0) * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, ROW_LENGTH)
    mask = row_ids[:, None] < rows
    offsets = row_ids[:, None] * ROW_LENGTH + cols[None, :]

    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + offsets, mask=mask, other=0.0).to(tl.float32)
    if GATE == 0:  # swish / silu
        gate = z * tl.sigmoid(z)
    elif GATE == 1:  # sigmoid
        gate = tl.sigmoid(z)
    else:  # unrecognized activation: the baseline stores y unmultiplied
        gate = tl.full(z.shape, 1.0, tl.float32)

    if NORM_BEFORE_GATE:
        var = tl.sum(x * x, axis=1) / ROW_LENGTH
        rstd = tl.rsqrt(var + eps)
        w = tl.load(W + cols).to(tl.float32)
        y = x * rstd[:, None] * w[None, :] * gate
    else:
        # The gate is folded in before the reduction, so it changes the variance and the
        # same gated value is what gets normalized.
        x = x * gate
        var = tl.sum(x * x, axis=1) / ROW_LENGTH
        rstd = tl.rsqrt(var + eps)
        w = tl.load(W + cols).to(tl.float32)
        y = x * rstd[:, None] * w[None, :]

    tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


def _triton_eligible(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> bool:
    """Is this the shape class the Triton specialization is compiled for?

    Every condition here is a specialization boundary, not a preference. Alignment and
    contiguity are checked rather than assumed because Triton derives divisibility hints
    from the pointers, and a pointer class first seen during timing would compile there.
    """
    row_length = x.shape[-1]
    return (row_length in _TRITON_ROW_LENGTHS
            and x.dtype is torch.bfloat16
            and weight.dtype in _TRITON_WEIGHT_DTYPES
            # Shape agreement is a correctness precondition, not a preference: this kernel
            # indexes z at x's offsets and reads exactly row_length weights, so a shorter
            # weight or a differently shaped z would silently produce a wrong answer where
            # the baseline asserts. Failing the check falls through to the extension, whose
            # TORCH_CHECKs reject the same inputs the baseline rejects.
            and z.shape == x.shape
            and weight.dim() == 1 and weight.numel() == row_length
            and x.is_cuda and z.is_cuda and weight.is_cuda
            and x.is_contiguous() and z.is_contiguous() and weight.is_contiguous()
            # int32 offsets inside the kernel; leave anything larger to the extension.
            and x.numel() < 2147483648
            and x.data_ptr() % 16 == 0 and z.data_ptr() % 16 == 0
            and weight.data_ptr() % 16 == 0)


def _triton_forward(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float,
                    gate: int, norm_before_gate: bool) -> torch.Tensor:
    row_length = x.shape[-1]
    rows = x.numel() // row_length
    out = torch.empty_like(x)
    if rows:
        _triton_rms_norm_gated[(triton.cdiv(rows, _TRITON_ROWS_PER_BLOCK),)](
            x, z, weight, out, rows, eps,
            ROW_LENGTH=row_length, ROWS_PER_BLOCK=_TRITON_ROWS_PER_BLOCK,
            GATE=gate, NORM_BEFORE_GATE=norm_before_gate,
            num_warps=_TRITON_NUM_WARPS)
    return out


# --- Extension build ------------------------------------------------------------
_EXTENSION = None
_BUILD_FAILED = False


def _local_cuda_arch() -> str | None:
    """The local compute capability with Blackwell's architecture-specific suffix.

    Read from ``nvidia-smi`` rather than ``torch.cuda.get_device_capability()`` so
    resolving it does not force CUDA initialization, mirroring
    ``fastkernels/infra/cuda_ext.py``.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    mapped = []
    for cap in caps:
        major = cap.split(".")[0]
        mapped.append(f"{cap}a" if major in ("9", "10", "12") and not cap.endswith("a")
                      else cap)
    return " ".join(mapped) or None


# The image this runs inside exports this exact six-architecture list as a container
# default. It is inherited rather than chosen, and honouring it turns a ~36 s build into a
# multi-minute one for five architectures that cannot run this kernel. It is therefore
# recognized by value and narrowed to the local architecture. Any *other* pre-set value is
# respected verbatim, including a deliberate multi-architecture list: a value someone set
# on purpose is a build instruction, not an accident.
_INHERITED_ARCH_LIST = frozenset("7.5 8.0 8.6 9.0 10.0 12.0+PTX".split())


def _pin_build_arch() -> None:
    """Choose ``TORCH_CUDA_ARCH_LIST`` for the build.

    Precedence: an explicit ``FK_RMSNORM_CUDA_ARCH_LIST`` wins (empty string means "leave
    ``TORCH_CUDA_ARCH_LIST`` exactly as it is"); then a pre-set ``TORCH_CUDA_ARCH_LIST`` is
    respected, unless it is the inherited container default above; otherwise the local
    capability with Blackwell's architecture-specific suffix. ``FASTKERNELS_CUDA_ARCH_LIST``
    plays the same role for the in-repo loader ``fastkernels/infra/cuda_ext.py``.
    """
    override = os.environ.get("FK_RMSNORM_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    preset = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if preset and preset.strip():
        if frozenset(preset.split()) != _INHERITED_ARCH_LIST:
            return  # set on purpose: respected as-is
    arch = _local_cuda_arch()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch


def _build_extension():
    from torch.utils.cpp_extension import load_inline

    _pin_build_arch()
    return load_inline(
        name="fk_rms_norm_gated",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=["rms_norm_gated", "configure", "last_path"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            # torch's default nvcc flags define this; without it __nv_bfloat16 ->
            # float conversions do not compile.
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )


def _extension():
    """The compiled extension, or None if it could not be built.

    Only the build is guarded, and the failure is cached so a broken toolchain does
    not re-attempt a compile on every call.
    """
    global _EXTENSION, _BUILD_FAILED
    if _EXTENSION is not None or _BUILD_FAILED:
        return _EXTENSION
    try:
        module = _build_extension()
        module.configure(_VALUES_PER_THREAD, _THREADS_PER_BLOCK, _MAX_BLOCKS)
    except Exception as exc:  # noqa: BLE001 - any toolchain failure degrades, not raises
        _BUILD_FAILED = True
        warnings.warn(
            f"rms_norm_gated: CUDA extension unavailable ({exc!r}); "
            "falling back to eager PyTorch", RuntimeWarning, stacklevel=2)
        return None
    _EXTENSION = module
    return _EXTENSION


# --- Eager fallback -------------------------------------------------------------
def _eager_gate(z: torch.Tensor, gate: int) -> torch.Tensor | None:
    if gate == _GATE_SWISH:
        return z * torch.sigmoid(z)
    if gate == _GATE_SIGMOID:
        return torch.sigmoid(z)
    return None


def _eager_forward(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float,
                   norm_before_gate: bool, gate: int) -> torch.Tensor:
    row_length = x.shape[-1]
    if z.shape != x.shape:
        raise RuntimeError(f"rms_norm_gated: z shape {tuple(z.shape)} != x shape "
                           f"{tuple(x.shape)}")
    if weight.dim() != 1 or weight.numel() != row_length:
        raise RuntimeError(f"rms_norm_gated: weight must be 1-D of length {row_length}, "
                           f"got {tuple(weight.shape)}")
    value = x.float()
    gate_value = _eager_gate(z.float(), gate)
    if not norm_before_gate and gate_value is not None:
        value = value * gate_value
    var = value.pow(2).mean(dim=-1, keepdim=True)
    y = value * torch.rsqrt(var + eps) * weight.float()
    if norm_before_gate and gate_value is not None:
        y = y * gate_value
    return y.to(x.dtype)


class RMSNormGated(nn.Module):
    """Fused gated RMSNorm: ``out = activation(z) * RMSNorm(x, weight)``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 norm_before_gate: bool = True, activation: str = "swish"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # activation and norm_before_gate are read per call, not frozen in __init__:
        # the baseline reads them per forward too, so mutating them after
        # construction has to keep working. The row length comes from x, never from
        # weight.numel(), so a (rows, 64) input with a 128-element weight is rejected
        # rather than silently reinterpreted as one 128-wide row.
        global _LAST_BACKEND
        gate = _GATE_CODES.get(self.activation, _GATE_NONE)

        # Scored shape class first. Triton specializes the row length as a compile-time
        # constant, which a single CUDA instantiation serving every row length cannot do,
        # and that is worth ~8 us of 43 us on the largest captured shape.
        if (_BACKEND != "cuda" and z is not None and z.dtype is x.dtype
                and _triton_eligible(x, z, self.weight)):
            _LAST_BACKEND = "triton"
            return _triton_forward(x, z, self.weight, self.eps, gate,
                                   self.norm_before_gate)

        extension = _extension()
        # The extension covers one dtype for both x and z, out of three. The baseline is
        # more permissive: it accepts z=None (ungated), an x/z dtype pair that differs, and
        # float64. Those all route to the eager forward, which is slow but correct, so the
        # candidate's accepted set is never narrower than the baseline's.
        if (extension is None or z is None or z.dtype is not x.dtype
                or x.dtype not in _KERNEL_DTYPES):
            _LAST_BACKEND = "eager"
            if z is None:
                # The baseline's kernel takes HAS_Z=False and stores y ungated; passing x
                # as a stand-in gate operand is safe because GATE_NONE never reads it.
                return _eager_forward(x, x, self.weight, self.eps,
                                      self.norm_before_gate, _GATE_NONE)
            return _eager_forward(x, z, self.weight, self.eps, self.norm_before_gate,
                                  gate)
        # Raw pybind11 entry point rather than a torch.library.custom_op wrapper, a
        # deliberate departure from the in-repo precedent in
        # fastkernels/tasks/baseline/L1/gate_linear.py, which wraps its pybind11 functions
        # *because* Dynamo cannot trace them and fullgraph=True then fails.
        #
        # The cost of that wrapper is measured, not assumed (analysis/probe_custom_op.py):
        # it adds 7.9-8.6 us of host time per call, roughly doubling this path's host cost,
        # and it changes the harness's measured window by -0.16 to +0.03 us -- nothing. So
        # the wrapper is affordable and raw pybind11's advantage does not reach the score.
        # It is kept only because torch.compile traceability is out of scope for this phase;
        # a phase that needs Dynamo should wrap this without expecting to pay for it.
        _LAST_BACKEND = "cuda"
        return extension.rms_norm_gated(x, z, self.weight, self.eps, gate,
                                        self.norm_before_gate)
