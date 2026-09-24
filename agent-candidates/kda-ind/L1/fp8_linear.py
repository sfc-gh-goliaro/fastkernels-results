"""FP8 linear (block-scaled FP8 matrix multiply) with a register-resident
activation quantizer.

Same public surface as the baseline: ``Fp8Linear`` and ``PerTokenGroupQuantFp8``
with identical ``__init__`` / ``forward`` contracts, the same ``torch.compile``
opacity guarantees, and the same load-time weight transforms.  Two things change.

**The activation quantizer.**  The baseline's vendored CUDA kernel stages every
group through shared memory: global -> shared while reducing the absmax,
``__syncthreads()``, then shared -> global while quantizing.  With 16 threads per
128-element group and 8 elements per thread it spends ~54 thread-instructions per
element, which makes it issue-bound -- 84.6 % SM throughput against 12.6 % DRAM
throughput at M=16384, K=4096, where the 203 MB it moves implies a ~25 us
roofline against 114 us measured.  The kernel here keeps each thread's 16 or 32
elements in registers across the whole absmax -> scale -> quantize pipeline, so
there is no shared memory, no barrier, and no second read of the input.  The
absmax is reduced pairwise in bf16 (``__habs2`` / ``__hmax2``) and widened once,
and the four fp8 bytes of each 32-bit output word come from two
``cvt.rn.satfinite.e4m3x2.f32`` instructions.

**The host path.**  ``Fp8Linear.forward`` calls the extension directly instead of
hopping through ``torch.library`` dispatch, resolves the flags that cannot change
per call exactly once, and reuses the activation and scale scratch buffers.

Neither change is allowed to move a single bit.  The harness compares
``Fp8Linear`` output at bf16 tolerance (atol = rtol = 1e-2, 0.99 of elements),
but a systematically *different* activation rounding -- even an equally good one
-- shifts the output by about ``sqrt(K) * 0.03 * 0.02 ~= 0.04`` against a
``0.01 + 0.01 * 1.3 ~= 0.023`` bound.  So the quantizer reproduces the vendored
reference bit-for-bit, and ``tools/check_quant.py`` proves it by ``torch.equal``
on the fp8 bytes and the fp32 scale bits, including an exhaustive sweep of every
attainable UE8M0 scale.

The numerical reasoning behind each operation that *looks* like it could be
cheapened is recorded next to it in the CUDA source below; that reasoning is what
makes the kernel safe to modify later.
"""

import math
import os
import subprocess
import sys
import warnings

import torch
import torch.nn as nn

# The baseline is the numerical reference and the fallback for every input the
# fast path does not cover.  Importing it also gives us the load-time weight
# transforms, which the engine's weight loader resolves on whichever module
# implements the op, so they have to exist here too.
from fastkernels.tasks.baseline.L1.fp8_linear import (  # noqa: F401
    _C,
    _Fp8PrefillBufs,
    _alloc_colmajor_scale,
    _is_batch_invariant,
    _maybe_get_flashinfer_fp8_gemm,
    _per_token_group_quant_fp8,
    postprocess_fp8_weights,
    postprocess_fp8_weights_batched,
)
from fastkernels.tasks.baseline.L1.fp8_grouped_gemm_contiguous import (
    _is_deep_gemm_e8m0_used,
)

import deep_gemm

_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_GROUP_SIZE = 128

# vLLM's ``per_token_group_quant_fp8`` epsilon (fp8_utils.py:860).  It reaches the
# kernel twice: as the fp32 seed of the absmax reduction, and again as a floor on
# the raw scale before UE8M0 rounding.  Both matter -- an all-zero group ends up
# at 2^-33 because of the *second* floor, not the first.
_QUANT_EPS = 1e-10


# ---------------------------------------------------------------------------
# Embedded CUDA extension
# ---------------------------------------------------------------------------

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kGroupSize = 128;
constexpr int kMaxThreadsPerBlock = 256;

// ---------------------------------------------------------------------------
// float -> e4m3 conversion.
//
// One instruction per pair.  PTX puts the *first* source operand in the high
// byte, so the call is ordered (high, low) to return {byte0 = f(x0),
// byte1 = f(x1)}.
//
// `satfinite` clamps out-of-range magnitudes to +/-448 and would make the
// explicit clamp in the caller look redundant.  It is not: the reference clamps
// with fminf(fmaxf(v, -448), 448), and fmaxf(NaN, -448) returns -448, so a NaN
// quotient becomes byte 0xFE.  A bare saturating convert would return the e4m3
// NaN encoding 0x7F instead.  Every element is therefore clamped before it gets
// here, which also means this convert never actually saturates.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cvt_e4m3_pair(float x0, float x1) {
  uint16_t packed;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(packed)
      : "f"(x1), "f"(x0));
  return static_cast<uint32_t>(packed);
}

// ---------------------------------------------------------------------------
// Pairwise absmax in the input dtype.
//
// Safe because max only ever *selects* an operand that already exists as a
// 16-bit value, and widening to fp32 is exact and order-preserving on
// magnitudes -- so reducing narrow and widening once equals widening and
// reducing wide.  __hmax2 also matches fmaxf's NaN suppression (one NaN operand
// returns the other; two return a canonical NaN), which is what lets an all-NaN
// group fall through to the eps floor exactly as the reference does.
//
// The eps floor itself is *not* folded in here.  It is applied in fp32 after
// widening, because 1e-10 is not exactly representable in bf16 and rounding it
// first would change the value being compared.
// ---------------------------------------------------------------------------
template <typename T>
struct PairTraits;

template <>
struct PairTraits<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  static __device__ __forceinline__ Pair abs_of(Pair a) { return __habs2(a); }
  static __device__ __forceinline__ Pair max_of(Pair a, Pair b) {
    return __hmax2(a, b);
  }
  static __device__ __forceinline__ float widen_max(Pair a) {
    return fmaxf(__bfloat162float(a.x), __bfloat162float(a.y));
  }
  static __device__ __forceinline__ float widen(__nv_bfloat16 a) {
    return __bfloat162float(a);
  }
};

template <>
struct PairTraits<__half> {
  using Pair = __half2;
  static __device__ __forceinline__ Pair abs_of(Pair a) { return __habs2(a); }
  static __device__ __forceinline__ Pair max_of(Pair a, Pair b) {
    return __hmax2(a, b);
  }
  static __device__ __forceinline__ float widen_max(Pair a) {
    return fmaxf(__half2float(a.x), __half2float(a.y));
  }
  static __device__ __forceinline__ float widen(__half a) {
    return __half2float(a);
  }
};

// ---------------------------------------------------------------------------
// Register-resident per-token-group FP8 quantization, group size 128.
//
// Grid is 2-D: blockIdx.x tiles token rows (rows can exceed the 65535 cap on
// grid.y, so they go on x), blockIdx.y tiles k-groups.  A block covers
// ROWS_PER_BLOCK x KGROUPS_PER_BLOCK groups; which of the two is the fast axis
// *within* the block is chosen by scale layout, so the leader lanes of a warp
// write adjacent fp32 scales in either case:
//
//   mn-major scales -- consecutive flat groups are `mn` apart, so the fast axis
//                      is the row and a warp's leaders cover adjacent rows.
//   row-major scales -- consecutive flat groups are adjacent, so the fast axis
//                      is the k-group.
//
// Both tile counts are powers of two and arrive as (mask, log2) pairs so the
// split is two integer ops rather than a division.
// ---------------------------------------------------------------------------
template <typename T, bool kRowFastAxis, int kElemsPerThread>
__global__ void __launch_bounds__(kMaxThreadsPerBlock) quant_group128_kernel(
    const T* __restrict__ input, uint8_t* __restrict__ out_q,
    float* __restrict__ out_s, int mn, int groups_per_row, int rows_per_block,
    int rb_mask, int rb_log2, int kgroups_per_block, int kb_mask, int kb_log2,
    long long scale_stride_row, long long scale_stride_k, float eps,
    float qmin, float qmax) {
  constexpr int kThreadsPerGroup = kGroupSize / kElemsPerThread;
  static_assert(kGroupSize % kElemsPerThread == 0, "group must divide evenly");
  static_assert(32 % kThreadsPerGroup == 0,
                "threads per group must divide the warp so the shuffle mask "
                "covers exactly one subgroup");

  using Traits = PairTraits<T>;
  using Pair = typename Traits::Pair;
  constexpr int kPairsPerThread = kElemsPerThread / 2;
  constexpr int kElemsPerVec = 16 / static_cast<int>(sizeof(T));  // 8 for 16-bit T
  constexpr int kVecsPerThread = kElemsPerThread / kElemsPerVec;
  static_assert(kElemsPerThread % kElemsPerVec == 0, "need whole 16 B loads");

  const int local_group = threadIdx.x / kThreadsPerGroup;
  const int lane = threadIdx.x % kThreadsPerGroup;

  int row_local, k_local;
  if (kRowFastAxis) {
    row_local = local_group & rb_mask;
    k_local = local_group >> rb_log2;
  } else {
    k_local = local_group & kb_mask;
    row_local = local_group >> kb_log2;
  }
  const int row = blockIdx.x * rows_per_block + row_local;
  const int kgroup = blockIdx.y * kgroups_per_block + k_local;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  // Whole groups leave together -- all kThreadsPerGroup lanes of a group share
  // `row` and `kgroup` -- so the shuffle below never addresses an exited lane.
  // The tiling deliberately does not have to divide the group count; the tail is
  // handled here rather than by choosing a divisor of it.
  if (row >= mn || kgroup >= groups_per_row) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  // Each lane owns one contiguous run of kElemsPerThread elements, loaded as
  // consecutive 16 B vectors.
  //
  // The alternative -- striping so the lanes of a group cover consecutive 16 B
  // vectors within a single instruction -- was measured and rejected. It does
  // what it claims to: the profile shows 268 MB of L1 load sectors for 134 MB of
  // data with ~50 % of sectors flagged excessive, because with contiguous
  // ownership each lane's first 16 B load uses half of its own 32 B sector and a
  // later instruction fetches the other half. Striping makes every access a
  // whole number of fully-used sectors and removes that. It also makes no
  // difference to runtime: summed over the nine shapes and two layouts in
  // tools/tune_quant.py, striped is 137.4 us against 137.0 us contiguous, with
  // the two largest cases about 1 % worse. The "excess" sectors were L1 hits on
  // lines already fetched, which cost almost nothing. Kept contiguous because it
  // is the simpler mapping and marginally the faster one.
  const long long offset =
      static_cast<long long>(row) * groups_per_row * kGroupSize +
      static_cast<long long>(kgroup) * kGroupSize + lane * kElemsPerThread;

  alignas(16) T regs[kElemsPerThread];
  {
    const uint4* src = reinterpret_cast<const uint4*>(input + offset);
    uint4* dst = reinterpret_cast<uint4*>(&regs[0]);
#pragma unroll
    for (int v = 0; v < kVecsPerThread; ++v) {
      dst[v] = src[v];
    }
  }

  const Pair* pairs = reinterpret_cast<const Pair*>(&regs[0]);
  Pair acc = Traits::abs_of(pairs[0]);
#pragma unroll
  for (int i = 1; i < kPairsPerThread; ++i) {
    acc = Traits::max_of(acc, Traits::abs_of(pairs[i]));
  }
  // Widen once, then floor in fp32.  Equivalent to the reference's per-thread
  // `local_absmax = eps` seed because max is associative and eps is constant.
  float absmax = fmaxf(Traits::widen_max(acc), eps);

  // Subgroup reduce over the lanes sharing this group.  Order is irrelevant:
  // max selects an existing operand and performs no rounding.
  const unsigned mask = ((1u << kThreadsPerGroup) - 1u)
                        << (threadIdx.x & (31u & ~static_cast<unsigned>(
                                                    kThreadsPerGroup - 1)));
#pragma unroll
  for (int shift = kThreadsPerGroup / 2; shift >= 1; shift >>= 1) {
    absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, shift));
  }

  // A true fp32 division, once per group.  Replacing it with a multiply by
  // fl(1/448) perturbs the quotient by an ULP, and a single ULP on the high side
  // of an exact power of two flips ceil(log2(.)) and doubles the scale for all
  // 128 elements of the group.  It is cheap here (one division per 128 elements)
  // and it is not worth the risk.
  float raw_scale = absmax / qmax;
  // The reference's second floor, applied *before* UE8M0 rounding.  Without it
  // an all-zero group would land on 2^-42 instead of 2^-33, and the integer
  // exponent extraction below would stop agreeing with the transcendental form
  // for subnormal inputs.
  raw_scale = fmaxf(raw_scale, 1e-10f);

  // UE8M0: round the scale up to a power of two.  The reference writes this as
  // exp2f(ceilf(log2f(.))); for a positive normal float the same value is the
  // exponent field plus a carry from any set mantissa bit, which is four integer
  // ops instead of three transcendentals.  The two forms are equal over the
  // whole attainable domain -- absmax is always exactly a bf16 magnitude or the
  // eps floor, so there are only 2^15 + 1 reachable scales -- and
  // tools/check_quant.py enumerates all of them against the device's own
  // transcendental result rather than assuming it.
  const uint32_t bits = __float_as_uint(raw_scale);
  const uint32_t exponent =
      ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u);
  const float scale = __uint_as_float(exponent << 23);

  if (lane == 0) {
    out_s[static_cast<long long>(row) * scale_stride_row +
          static_cast<long long>(kgroup) * scale_stride_k] = scale;
  }

  // Exact: `scale` is a power of two, so its reciprocal is representable and
  // x * (1/scale) rounds to the same float as x / scale for every finite input.
  // The +Inf endpoint agrees too (1/Inf is +0, and finite * +0 is the same
  // signed zero as finite / Inf).  Computed as a division rather than by
  // negating the exponent field, which would send +Inf to -Inf and would break
  // wherever the reciprocal is subnormal.
  const float inv_scale = 1.0f / scale;

  // Four elements per 32-bit output word, two e4m3x2 converts each.  The output
  // for kElemsPerThread elements is half as many bytes as the input, so a
  // thread's whole run stores as kElemsPerThread/16 uint4s.
  constexpr int kOutWords = kElemsPerThread / 4;
  uint32_t words[kOutWords];
#pragma unroll
  for (int i = 0; i < kOutWords; ++i) {
    float q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[j] = fminf(fmaxf(Traits::widen(regs[4 * i + j]) * inv_scale, qmin),
                   qmax);
    }
    words[i] = cvt_e4m3_pair(q[0], q[1]) | (cvt_e4m3_pair(q[2], q[3]) << 16);
  }

  {
    uint4* dst = reinterpret_cast<uint4*>(out_q + offset);
    const uint4* src = reinterpret_cast<const uint4*>(&words[0]);
#pragma unroll
    for (int i = 0; i < kOutWords / 4; ++i) {
      dst[i] = src[i];
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------
// Launch geometry, chosen from measurement (tools/tune_quant.py).
//
// Start from a full 256-thread block and halve the groups per block until there
// are enough blocks to keep the SMs busy, with a floor of one warp per block.
// Then hand as many of those groups as possible to the k axis and the rest to
// the row axis.
//
// Preferring the k axis for *both* scale layouts is the measured result, and it
// is the opposite of what coalescing the scale stores alone would suggest: a
// row-heavy tile does make the leader lanes of a warp write adjacent mn-major
// scales, but it also scatters the input reads across as many rows, and the
// input is two orders of magnitude more traffic than the scales.  On
// M=16384, K=4096 mn-major, a 2x32 tile measures 30.4 us against 32.5 us for
// 64x1.  Where the k axis cannot absorb the block anyway -- K=384 leaves only
// three k-groups per row -- the rows take it and the coalescing follows for
// free, which is worth 14 % at M=5896, K=384.
//
// The k tile is capped at the largest power of two that *divides*
// groups_per_row, so no block is launched covering a k-slot that does not exist.
// Allowing a 2-wide k tile over three k-groups wastes a third of every second
// block, which cost 9 % on the K=384 shapes.
// ---------------------------------------------------------------------------
struct LaunchGeometry {
  int rows_per_block;
  int kgroups_per_block;
  int threads;
};

LaunchGeometry pick_geometry(long long mn, int groups_per_row,
                            int threads_per_group, int sm_count) {
  const int min_groups = 32 / threads_per_group;  // at least one warp
  int groups = kMaxThreadsPerBlock / threads_per_group;
  const long long total_groups = mn * static_cast<long long>(groups_per_row);
  // Two blocks per SM is enough to hide the tail; demanding four shrinks the
  // block past the point where it helps and costs 10 % at M=5896, K=384 and
  // M=1000, K=2048, which land just under the stricter bound.
  while (groups > min_groups &&
         total_groups < static_cast<long long>(groups) * sm_count * 2) {
    groups >>= 1;
  }

  const int kg_cap = groups_per_row & (-groups_per_row);
  const int kgroups = groups < kg_cap ? groups : kg_cap;
  const int rows = groups / kgroups;
  return {rows, kgroups, groups * threads_per_group};
}

// Elements per thread.  32 (four 16 B loads, four threads per group) amortizes
// the per-group fixed cost -- two divisions, the subgroup reduce, the scale
// store -- over twice as many elements, which is worth 10-40 % once the launch
// is large enough for that to be the dominant term.  Below that it loses to the
// wider 16-element variant's better latency hiding.  The threshold sits in the
// wide empty band between the largest shape that prefers 16 (32000 groups) and
// the smallest that prefers 32 (393216).
int pick_elems_per_thread(long long total_groups) {
  return total_groups >= 65536 ? 32 : 16;
}

int ilog2(int v) {
  int r = 0;
  while ((1 << r) < v) {
    ++r;
  }
  return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry point.  Mirrors _C.per_token_group_fp8_quant for group_size == 128,
// bf16/fp16 input, use_ue8m0 == True, and the two fp32 scale layouts the two
// scored classes use.  Everything else is rejected here and delegated in Python.
// ---------------------------------------------------------------------------
void quant_fp8_group128(const torch::Tensor& input, torch::Tensor& out_q,
                       torch::Tensor& out_s, double eps, double qmin,
                       double qmax, int64_t rows_override,
                       int64_t kgroups_override, int64_t elems_override) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  // Both outputs too, and on the same device: the launch below uses the input's
  // device guard and the current stream, so a CPU or cross-device output would
  // otherwise be handed to the kernel as a raw pointer it cannot address.
  TORCH_CHECK(out_q.is_cuda() && out_s.is_cuda(),
              "out_q and out_s must be CUDA tensors");
  TORCH_CHECK(out_q.device() == input.device() &&
                  out_s.device() == input.device(),
              "out_q and out_s must be on the same device as input");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out_q.is_contiguous(), "out_q must be contiguous");
  TORCH_CHECK(out_q.scalar_type() == at::kFloat8_e4m3fn,
              "out_q must be float8_e4m3fn");
  TORCH_CHECK(out_s.scalar_type() == at::kFloat, "out_s must be float32");
  TORCH_CHECK(out_s.dim() == 2, "out_s must be 2-D");

  const int64_t k = input.size(-1);
  TORCH_CHECK(k % kGroupSize == 0, "last dim must be a multiple of 128");
  const int64_t mn = input.numel() / k;
  const int64_t groups_per_row = k / kGroupSize;
  TORCH_CHECK(out_q.numel() == input.numel(), "out_q must match input numel");
  TORCH_CHECK(out_s.size(0) == mn && out_s.size(1) == groups_per_row,
              "out_s must be [mn, k/128]");
  // `row` is computed as blockIdx.x * rows_per_block + row_local *before* the
  // bounds check, so it can exceed mn by up to one block tile.  Leave that much
  // headroom below INT32_MAX rather than allowing the intermediate to overflow.
  TORCH_CHECK(mn <= INT32_MAX - kMaxThreadsPerBlock && k <= INT32_MAX,
              "shape too large");
  // Every load and store here is a 16 B vector, so both buffers have to be
  // 16 B-aligned.  Contiguity alone does not guarantee it -- a view can start
  // mid-allocation -- and the reference handles misalignment with a scalar
  // prologue that this kernel does not have.  Python rejects such a call before
  // it gets here and delegates; this is the backstop.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out_q.data_ptr()) % 16 == 0,
              "input and out_q must be 16-byte aligned");

  // Same test the reference uses to pick its scale addressing.  At mn == 1 both
  // strides are 1 so this is false, the row-major branch is taken, and the two
  // layouts coincide.
  const bool mn_major = out_s.stride(0) < out_s.stride(1);
  long long scale_stride_row, scale_stride_k;
  if (mn_major) {
    // Reference: offset(g) = (g % groups) * out_s.stride(1) + (g / groups).
    TORCH_CHECK(out_s.stride(0) == 1,
                "mn-major scales must have stride(0) == 1");
    scale_stride_row = 1;
    scale_stride_k = out_s.stride(1);
  } else {
    // Reference: offset(g) = g, i.e. flat, which is only the strided address if
    // the scale tensor really is contiguous.  Anything else is delegated so the
    // two implementations cannot disagree about where a scale belongs.
    TORCH_CHECK(out_s.stride(1) == 1 &&
                    (mn == 1 || out_s.stride(0) == groups_per_row),
                "row-major scales must be contiguous");
    scale_stride_row = groups_per_row;
    scale_stride_k = 1;
  }

  const int elems_per_thread =
      elems_override > 0 ? static_cast<int>(elems_override)
                         : pick_elems_per_thread(mn * groups_per_row);
  TORCH_CHECK(elems_per_thread == 16 || elems_per_thread == 32,
              "elems per thread must be 16 or 32");
  const int threads_per_group = kGroupSize / elems_per_thread;

  const c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  LaunchGeometry geo = pick_geometry(mn, static_cast<int>(groups_per_row),
                                    threads_per_group, sm_count);
  if (rows_override > 0 && kgroups_override > 0) {
    geo.rows_per_block = static_cast<int>(rows_override);
    geo.kgroups_per_block = static_cast<int>(kgroups_override);
    geo.threads =
        geo.rows_per_block * geo.kgroups_per_block * threads_per_group;
    // The kernel splits a block into its row x k-group tile with a mask and a
    // shift, which is only the same thing as a divmod when both tile extents are
    // powers of two.  A non-power-of-two override would address the wrong
    // groups silently, so it is refused rather than honoured.
    TORCH_CHECK((geo.rows_per_block & (geo.rows_per_block - 1)) == 0 &&
                    (geo.kgroups_per_block & (geo.kgroups_per_block - 1)) == 0,
                "overridden tile extents must be powers of two, got ",
                geo.rows_per_block, " x ", geo.kgroups_per_block);
  }
  TORCH_CHECK(geo.threads > 0 && geo.threads <= kMaxThreadsPerBlock &&
                  geo.threads % 32 == 0,
              "bad launch geometry: ", geo.threads, " threads");

  const int64_t row_blocks =
      (mn + geo.rows_per_block - 1) / geo.rows_per_block;
  const int64_t kgroup_blocks =
      (groups_per_row + geo.kgroups_per_block - 1) / geo.kgroups_per_block;
  TORCH_CHECK(kgroup_blocks <= 65535, "grid.y too large");

  const int rb_mask = geo.rows_per_block - 1;
  const int rb_log2 = ilog2(geo.rows_per_block);
  const int kb_mask = geo.kgroups_per_block - 1;
  const int kb_log2 = ilog2(geo.kgroups_per_block);

#define FK_LAUNCH(T, ROW_FAST, ELEMS)                                        \
  do {                                                                       \
    cudaLaunchConfig_t config = {};                                          \
    config.gridDim = dim3(static_cast<unsigned>(row_blocks),                 \
                          static_cast<unsigned>(kgroup_blocks));             \
    config.blockDim = dim3(static_cast<unsigned>(geo.threads));              \
    config.dynamicSmemBytes = 0;                                             \
    config.stream = stream;                                                  \
    cudaLaunchAttribute attrs[1];                                            \
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;        \
    attrs[0].val.programmaticStreamSerializationAllowed = 1;                 \
    config.numAttrs = 1;                                                     \
    config.attrs = attrs;                                                    \
    C10_CUDA_CHECK(cudaLaunchKernelEx(                                       \
        &config, quant_group128_kernel<T, ROW_FAST, ELEMS>,                  \
        static_cast<const T*>(input.const_data_ptr()),                       \
        static_cast<uint8_t*>(out_q.data_ptr()),                             \
        static_cast<float*>(out_s.data_ptr()), static_cast<int>(mn),         \
        static_cast<int>(groups_per_row), geo.rows_per_block, rb_mask,       \
        rb_log2, geo.kgroups_per_block, kb_mask, kb_log2, scale_stride_row,  \
        scale_stride_k, static_cast<float>(eps), static_cast<float>(qmin),   \
        static_cast<float>(qmax)));                                          \
  } while (0)

#define FK_DISPATCH_ELEMS(T, ROW_FAST)     \
  do {                                     \
    if (elems_per_thread == 16) {          \
      FK_LAUNCH(T, ROW_FAST, 16);          \
    } else {                               \
      FK_LAUNCH(T, ROW_FAST, 32);          \
    }                                      \
  } while (0)

#define FK_DISPATCH_LAYOUT(T)         \
  do {                                \
    if (mn_major) {                   \
      FK_DISPATCH_ELEMS(T, true);     \
    } else {                          \
      FK_DISPATCH_ELEMS(T, false);    \
    }                                 \
  } while (0)

  if (input.scalar_type() == at::kBFloat16) {
    FK_DISPATCH_LAYOUT(__nv_bfloat16);
  } else if (input.scalar_type() == at::kHalf) {
    FK_DISPATCH_LAYOUT(__half);
  } else {
    TORCH_CHECK(false, "input must be bfloat16 or float16");
  }

#undef FK_DISPATCH_LAYOUT
#undef FK_DISPATCH_ELEMS
#undef FK_LAUNCH
}

// ---------------------------------------------------------------------------
// Verification helpers.  These exist so tools/check_quant.py can settle two
// questions on the device instead of assuming them, and they are never on a
// timed path.
// ---------------------------------------------------------------------------

namespace {

// Both UE8M0 roundings, from the same starting absmax, using the device's own
// log2f/ceilf/exp2f rather than a host libm stand-in.
__global__ void ue8m0_pair_kernel(const float* __restrict__ absmax,
                                  float* __restrict__ transcendental,
                                  float* __restrict__ bitmath,
                                  float* __restrict__ mul_reciprocal, int n,
                                  float qmax) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float raw = fmaxf(absmax[i] / qmax, 1e-10f);
  transcendental[i] = exp2f(ceilf(log2f(fmaxf(fabsf(raw), 1e-10f))));
  const uint32_t bits = __float_as_uint(raw);
  bitmath[i] = __uint_as_float(
      (((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u)) << 23);

  // The tempting substitution: multiply by fl(1/448) instead of dividing.
  const float raw_mul = fmaxf(absmax[i] * (1.0f / qmax), 1e-10f);
  const uint32_t mbits = __float_as_uint(raw_mul);
  mul_reciprocal[i] = __uint_as_float(
      (((mbits >> 23) & 0xffu) + ((mbits & 0x7fffffu) != 0u ? 1u : 0u)) << 23);
}

// The hardware convert this kernel uses against the software convert the
// reference's fp8 type performs, over whatever float patterns the caller feeds.
__global__ void fp8_convert_pair_kernel(const float* __restrict__ x,
                                        uint8_t* __restrict__ hardware,
                                        uint8_t* __restrict__ software, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float v = x[i];
  hardware[i] = static_cast<uint8_t>(cvt_e4m3_pair(v, 0.0f) & 0xffu);
  const c10::Float8_e4m3fn ref(v);
  software[i] = *reinterpret_cast<const uint8_t*>(&ref);
}

}  // namespace

void ue8m0_rounding_probe(const torch::Tensor& absmax,
                         torch::Tensor& transcendental, torch::Tensor& bitmath,
                         torch::Tensor& mul_reciprocal, double qmax) {
  TORCH_CHECK(absmax.is_cuda() && absmax.is_contiguous());
  TORCH_CHECK(absmax.scalar_type() == at::kFloat);
  const int n = static_cast<int>(absmax.numel());
  const c10::cuda::CUDAGuard guard(absmax.device());
  const int threads = 256;
  ue8m0_pair_kernel<<<(n + threads - 1) / threads, threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      absmax.const_data_ptr<float>(), transcendental.data_ptr<float>(),
      bitmath.data_ptr<float>(), mul_reciprocal.data_ptr<float>(), n,
      static_cast<float>(qmax));
  C10_CUDA_CHECK(cudaGetLastError());
}

void fp8_convert_probe(const torch::Tensor& x, torch::Tensor& hardware,
                      torch::Tensor& software) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous());
  TORCH_CHECK(x.scalar_type() == at::kFloat);
  const int n = static_cast<int>(x.numel());
  const c10::cuda::CUDAGuard guard(x.device());
  const int threads = 256;
  fp8_convert_pair_kernel<<<(n + threads - 1) / threads, threads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      x.const_data_ptr<float>(), static_cast<uint8_t*>(hardware.data_ptr()),
      static_cast<uint8_t*>(software.data_ptr()), n);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

void quant_fp8_group128(const torch::Tensor& input, torch::Tensor& out_q,
                       torch::Tensor& out_s, double eps, double qmin,
                       double qmax, int64_t rows_override,
                       int64_t kgroups_override, int64_t elems_override);

void ue8m0_rounding_probe(const torch::Tensor& absmax,
                         torch::Tensor& transcendental, torch::Tensor& bitmath,
                         torch::Tensor& mul_reciprocal, double qmax);

void fp8_convert_probe(const torch::Tensor& x, torch::Tensor& hardware,
                      torch::Tensor& software);
"""


def _local_cuda_arch() -> str | None:
    """Local compute capability with Blackwell's ``a`` suffix, via nvidia-smi.

    Mirrors ``fastkernels.infra.cuda_ext._local_cuda_arch``.  Deliberately does
    not touch ``torch.cuda``, so importing this module does not initialize a
    CUDA context before the harness is ready for one.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except Exception:
        return None
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    mapped = []
    for cap in caps:
        major = cap.split(".")[0]
        if major in ("9", "10", "12") and not cap.endswith("a"):
            mapped.append(f"{cap}a")
        else:
            mapped.append(cap)
    return " ".join(mapped) or None


def _build_extension():
    """Compile the embedded CUDA source.

    Runs at import, so no build ever lands inside a timed region.  Build
    artifacts go to a workspace-local directory when one is writable, which
    keeps them out of the shared /tmp cache and away from its lock contention.

    ``TORCH_CUDA_ARCH_LIST`` is a process-global, so it is restored afterwards
    rather than left mutated for whatever imports next.  No fast-math and no
    ``-prec-div=false``: both would change the division and the transcendental
    results this kernel is required to reproduce exactly.
    """
    from torch.utils.cpp_extension import load_inline

    flags = [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]

    build_dir = None
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", ".torch_extensions")
    try:
        os.makedirs(root, exist_ok=True)
        build_dir = os.path.join(os.path.realpath(root), "fk_fp8_quant_cand")
        os.makedirs(build_dir, exist_ok=True)
    except OSError:
        build_dir = None

    arch = _local_cuda_arch()
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name="fk_fp8_quant_cand",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["quant_fp8_group128", "ue8m0_rounding_probe",
                       "fp8_convert_probe"],
            extra_cuda_cflags=flags,
            extra_cflags=["-O3", "-std=c++17"],
            build_directory=build_dir,
            verbose=False,
        )
    finally:
        if arch:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


_EXT = None
#: True when the register-resident kernel is available.  The validation tools
#: assert this, so a silent fall back to the baseline cannot be mistaken for a
#: working optimization.
_EXT_ACTIVE = False

try:
    _EXT = _build_extension()
    _EXT_ACTIVE = True
except Exception as exc:  # pragma: no cover - build environment dependent
    warnings.warn(
        f"fp8_linear candidate: CUDA extension build failed ({type(exc).__name__}: "
        f"{exc}); falling back to the baseline quantization path.",
        RuntimeWarning, stacklevel=2,
    )
    print(f"[fp8_linear candidate] extension build failed: {exc}",
          file=sys.stderr, flush=True)


def _parse_geometry_override():
    """``FK_FP8_QUANT_CFG=rows,kgroups,elems`` pins the launch geometry.

    Used by the tuning tools to sweep configurations without rebuilding; unset
    in normal operation, where the kernel picks its own geometry from the shape.
    """
    raw = os.environ.get("FK_FP8_QUANT_CFG", "")
    if not raw:
        return 0, 0, 0
    try:
        rows, kgroups, elems = (int(v) for v in raw.split(","))
        return rows, kgroups, elems
    except ValueError:
        warnings.warn(f"ignoring malformed FK_FP8_QUANT_CFG={raw!r}",
                      RuntimeWarning, stacklevel=2)
        return 0, 0, 0


_GEOM_ROWS, _GEOM_KGROUPS, _GEOM_ELEMS = _parse_geometry_override()


# ---------------------------------------------------------------------------
# Quantization entry point
# ---------------------------------------------------------------------------

def _fast_quant_ok(x: torch.Tensor, out_fp8: torch.Tensor,
                   out_scale: torch.Tensor) -> bool:
    """Whether the register-resident kernel covers this call exactly.

    Anything outside the covered domain is delegated to the baseline op, which is
    the reference -- so an unsupported input is answered identically rather than
    approximately.  In particular a ``K`` that is not a multiple of 128 is
    delegated: the reference partitions ``numel`` flat, letting groups straddle
    row boundaries, which this kernel's row-aligned mapping does not reproduce.
    """
    if not _EXT_ACTIVE:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if out_fp8.dtype is not torch.float8_e4m3fn:
        return False
    if out_scale.dtype is not torch.float32 or out_scale.dim() != 2:
        return False
    k = x.shape[-1]
    if k % _GROUP_SIZE != 0:
        return False
    if not x.is_contiguous() or not out_fp8.is_contiguous():
        return False
    if out_fp8.device != x.device or out_scale.device != x.device:
        return False
    # The kernel loads and stores 16 B vectors with no scalar prologue, so both
    # buffers must be 16 B-aligned.  Contiguity does not imply it: a view can
    # start mid-allocation.  Fresh allocations and row slices always are, so this
    # normally costs one comparison and never delegates.
    if x.data_ptr() % 16 or out_fp8.data_ptr() % 16:
        return False
    mn = x.numel() // k
    if out_scale.shape != (mn, k // _GROUP_SIZE):
        return False
    if out_scale.stride(0) < out_scale.stride(1):
        return out_scale.stride(0) == 1
    return out_scale.stride(1) == 1 and (
        mn == 1 or out_scale.stride(0) == k // _GROUP_SIZE)


def _quant_fp8(x: torch.Tensor, out_fp8: torch.Tensor,
               out_scale: torch.Tensor) -> None:
    """In-place per-token-group FP8 quantization, group size 128, UE8M0 scales.

    Bit-identical to ``_C.per_token_group_fp8_quant`` over the covered domain and
    delegated to it everywhere else.
    """
    if _fast_quant_ok(x, out_fp8, out_scale):
        _EXT.quant_fp8_group128(
            x, out_fp8, out_scale, _QUANT_EPS, _FP8_INFO.min, _FP8_INFO.max,
            _GEOM_ROWS, _GEOM_KGROUPS, _GEOM_ELEMS,
        )
        return
    _per_token_group_quant_fp8(
        x, out_fp8, out_scale, use_ue8m0=True,
        column_major_scales=out_scale.stride(0) < out_scale.stride(1),
    )


# ---------------------------------------------------------------------------
# Custom ops.  Registered under this module's own namespace: defining
# "fastkernels_fp8" a second time in a process that already imported the
# baseline would raise, and the baseline's registrations stay reachable for the
# delegating fallback.
# ---------------------------------------------------------------------------

_lib = torch.library.Library("fastkernels_fp8_cand", "DEF")

_lib.define(
    "per_token_group_quant_fp8(Tensor input, Tensor! output_fp8, "
    "Tensor! output_scale, bool column_major_scales=False) -> ()"
)


def _quant_op_impl(input, output_fp8, output_scale, column_major_scales=False):
    _quant_fp8(input, output_fp8, output_scale)


_lib.impl("per_token_group_quant_fp8", _quant_op_impl, "CUDA")


@torch.library.impl(_lib, "per_token_group_quant_fp8", "Meta")
def _quant_op_meta(input, output_fp8, output_scale, column_major_scales=False):
    pass


_lib.define(
    "flashinfer_blockscale_gemm(Tensor input_bf16, Tensor weight_fp8, "
    "Tensor weight_scale, Tensor! output) -> ()"
)


def _flashinfer_impl(input_bf16, weight_fp8, weight_scale, output):
    fn = _maybe_get_flashinfer_fp8_gemm()
    assert fn is not None, "FlashInfer FP8 blockscale GEMM not available"
    fn(input=input_bf16, weight=weight_fp8, input_scale=None,
       weight_scale=weight_scale, out=output, out_dtype=torch.bfloat16)


_lib.impl("flashinfer_blockscale_gemm", _flashinfer_impl, "CUDA")


@torch.library.impl(_lib, "flashinfer_blockscale_gemm", "Meta")
def _flashinfer_meta(input_bf16, weight_fp8, weight_scale, output):
    pass


# The M < 32 (FlashInfer) versus M >= 32 (DeepGEMM) choice has to be made on the
# runtime M even under capture: a Python ``if M < 32`` freezes the branch at
# trace time and silently drops the low-batch accuracy path.  Deciding inside an
# opaque custom op keeps it, exactly as the baseline does.
_lib.define(
    "blockscale_gemm_dispatch(Tensor input_2d, Tensor weight_fp8, "
    "Tensor weight_scale, bool flashinfer_ok) -> Tensor"
)

_FLASHINFER_M_THRESHOLD = 32


def _dispatch_impl(input_2d, weight_fp8, weight_scale, flashinfer_ok):
    n = weight_fp8.shape[0]
    m = input_2d.shape[0]
    output = torch.empty(m, n, dtype=torch.bfloat16, device=input_2d.device)
    if flashinfer_ok and m < _FLASHINFER_M_THRESHOLD:
        _flashinfer_impl(input_2d, weight_fp8, weight_scale, output)
        return output
    k = weight_fp8.shape[1]
    num_groups = (k + _GROUP_SIZE - 1) // _GROUP_SIZE
    q_input = torch.empty(m, k, dtype=torch.float8_e4m3fn,
                          device=input_2d.device)
    input_scale = _alloc_colmajor_scale(m, num_groups, input_2d.device)
    _quant_fp8(input_2d, q_input, input_scale)
    deep_gemm.fp8_gemm_nt((q_input, input_scale), (weight_fp8, weight_scale),
                          output, disable_ue8m0_cast=_disable_ue8m0_cast())
    return output


_lib.impl("blockscale_gemm_dispatch", _dispatch_impl, "CUDA")


@torch.library.impl(_lib, "blockscale_gemm_dispatch", "Meta")
def _dispatch_meta(input_2d, weight_fp8, weight_scale, flashinfer_ok):
    return input_2d.new_empty((input_2d.shape[0], weight_fp8.shape[0]),
                              dtype=torch.bfloat16)


class PerTokenGroupQuantFp8(nn.Module):
    """In-place per-token-group FP8 quantization (single CUDA kernel).

    Eager calls reach the kernel directly; under ``torch.compile`` the call is
    routed through this module's opaque custom op so tracing sees a single
    mutating operation rather than a pybind call it cannot handle.
    """

    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        if not x.is_contiguous():
            x = x.contiguous()
        if torch.compiler.is_compiling():
            torch.ops.fastkernels_fp8_cand.per_token_group_quant_fp8(
                x, out_fp8, out_scale,
                out_scale.stride(0) < out_scale.stride(1),
            )
            return
        _quant_fp8(x, out_fp8, out_scale)


# ---------------------------------------------------------------------------
# Fp8Linear
# ---------------------------------------------------------------------------

_UE8M0_CAST_DISABLED: bool | None = None


def _disable_ue8m0_cast() -> bool:
    """DeepGEMM's ``disable_ue8m0_cast``, resolved once.

    It is keyed off the same build/arch oracle the weight requantization uses, so
    it cannot change between calls in a process -- unlike
    ``_is_batch_invariant()``, which reads the environment and is therefore left
    per-call below.
    """
    global _UE8M0_CAST_DISABLED
    if _UE8M0_CAST_DISABLED is None:
        _UE8M0_CAST_DISABLED = not _is_deep_gemm_e8m0_used()
    return _UE8M0_CAST_DISABLED


_FLASHINFER_GATE: bool | None = None


def _flashinfer_available() -> bool:
    """Whether the FlashInfer swapAB kernel exists at all, resolved once.

    The gate requires compute capability major 9, so on Blackwell it is False and
    the M < 32 branch is statically dead -- but the branch is kept for H100
    parity with the baseline.
    """
    global _FLASHINFER_GATE
    if _FLASHINFER_GATE is None:
        _FLASHINFER_GATE = _maybe_get_flashinfer_fp8_gemm() is not None
    return _FLASHINFER_GATE


class _ScratchCache:
    """Reusable activation and scale scratch, keyed so reuse is provably safe.

    The activation buffer is one growable arena per ``(K, device)``; a ``[:M]``
    row slice of it keeps ``stride(0) == K``, which is all the quantizer needs.

    The scale buffer cannot be sliced the same way.  DeepGEMM asserts
    ``sf.stride(1) == 1 and sf.stride(2) == mn``, so an A-scale carved out of a
    larger arena -- k-stride ``max_M`` rather than ``M`` -- is rejected outright
    (``tools/probe_sf_stride.py`` demonstrates it).  Bucketing M upward does not
    help either: the k-stride would still be wrong, so it would mean running the
    GEMM at the bucketed M and slicing the output, paying extra compute.  Hence
    one exact-shaped entry per distinct ``(M, num_groups, device)``.

    Both are only reused when the quantizer overwrites every byte, which needs
    ``K % 128 == 0``; otherwise the trailing scale entries of each row would keep
    whatever the previous call left there.

    Growth is bounded.  The bound's size does not matter to the benchmark -- M is
    constant within a scored case, and a hit costs ~0.1 us against ~4 us for a
    fresh column-major allocation -- it is there so a real engine's M
    distribution cannot grow the cache without limit.
    """

    _MAX_SCALE_ENTRIES = 64

    __slots__ = ("_arenas", "_scales")

    def __init__(self):
        self._arenas: dict[tuple, torch.Tensor] = {}
        self._scales: dict[tuple, torch.Tensor] = {}

    def activations(self, m: int, k: int, device: torch.device) -> torch.Tensor:
        key = (k, device)
        arena = self._arenas.get(key)
        if arena is None or arena.shape[0] < m:
            arena = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=device)
            self._arenas[key] = arena
            return arena
        return arena[:m]

    def scales(self, m: int, num_groups: int,
               device: torch.device) -> torch.Tensor:
        key = (m, num_groups, device)
        scale = self._scales.get(key)
        if scale is None:
            if len(self._scales) >= self._MAX_SCALE_ENTRIES:
                self._scales.pop(next(iter(self._scales)))
            scale = _alloc_colmajor_scale(m, num_groups, device)
            self._scales[key] = scale
        return scale


_SCRATCH = _ScratchCache()


class Fp8Linear(nn.Module):
    """Block-scaled FP8 linear using ``deep_gemm.fp8_gemm_nt``.

    Weights are float8_e4m3fn with pre-processed UE8M0 block scales, transformed
    at load time.  Activations are quantized per-token-group (group 128) in place
    so the forward stays CUDA-graph compatible.
    """

    BLOCK_SIZE = 128

    # vLLM hard-codes the FlashInfer swapAB threshold to 32 (fp8_utils.py:308).
    _FLASHINFER_M_THRESHOLD = 32

    def __init__(self):
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf: _Fp8PrefillBufs | None = None

    def _ensure_buffers(self, max_tokens: int, K: int, N: int,
                        device: torch.device):
        """Pre-allocate activation FP8 buffers for CUDA graph capture.

        The scale buffer is column-major, matching what DeepGEMM's dense FP8 path
        expects (see ``_alloc_colmajor_scale``).
        """
        num_groups = math.ceil(K / self.BLOCK_SIZE)
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn,
                                  device=device)
        self._s_buf = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16,
                                  device=device)

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        N, K = weight_fp8.shape
        two_d = input_bf16.dim() == 2
        input_2d = input_bf16 if two_d else input_bf16.reshape(-1, K)
        M = input_2d.shape[0]

        # M-independent FlashInfer eligibility, same conditions as vLLM's
        # ``should_use_flashinfer_for_blockscale_fp8_gemm``.  The availability
        # gate is resolved once and comes first, so on Blackwell -- where it is
        # False and the branch is dead -- none of the rest is evaluated.
        # ``_is_batch_invariant()`` stays a per-call environment read, so
        # changing ``VLLM_BATCH_INVARIANT`` after import still takes effect.
        flashinfer_ok = (
            _flashinfer_available()
            and input_bf16.dtype == torch.bfloat16
            and weight_fp8.dtype == torch.float8_e4m3fn
            and N % 64 == 0
            and K % 128 == 0
            and not _is_batch_invariant()
        )

        if torch.compiler.is_compiling():
            output = torch.ops.fastkernels_fp8_cand.blockscale_gemm_dispatch(
                input_2d, weight_fp8, weight_scale_inv, flashinfer_ok,
            )
            if bias is not None:
                output = output + bias
            return output if two_d else output.view(*input_bf16.shape[:-1], N)

        if flashinfer_ok and M < self._FLASHINFER_M_THRESHOLD:
            output = torch.empty(M, N, dtype=torch.bfloat16,
                                 device=input_2d.device)
            _flashinfer_impl(input_2d, weight_fp8, weight_scale_inv, output)
            if bias is not None:
                output = output + bias
            return output if two_d else output.view(*input_bf16.shape[:-1], N)

        num_groups = (K + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        device = input_2d.device

        # Engine-provided buffers win, then the prefill set, then our own
        # scratch -- the baseline's precedence order, preserved.  Reusing our
        # scratch is only safe when the quantizer writes every byte of it, which
        # needs K to be a multiple of the block size.
        if self._a_buf is not None and M <= self._a_buf.shape[0]:
            q_input = self._a_buf[:M]
            output = self._o_buf[:M]
            input_scale = _SCRATCH.scales(M, num_groups, device)
        elif self._pf is not None and M <= self._pf.a.shape[0]:
            q_input = self._pf.a[:M]
            output = self._pf.o[:M]
            input_scale = _SCRATCH.scales(M, num_groups, device)
        elif K % self.BLOCK_SIZE == 0:
            q_input = _SCRATCH.activations(M, K, device)
            input_scale = _SCRATCH.scales(M, num_groups, device)
            # Fresh output every call: reusing it would hand back aliased
            # storage across calls for ~1.7 us of host time, and the baseline
            # only aliases when the engine explicitly pre-sizes buffers.
            output = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        else:
            q_input = torch.empty(M, K, dtype=torch.float8_e4m3fn,
                                  device=device)
            input_scale = _alloc_colmajor_scale(M, num_groups, device)
            output = torch.empty(M, N, dtype=torch.bfloat16, device=device)

        _quant_fp8(input_2d, q_input, input_scale)
        deep_gemm.fp8_gemm_nt(
            (q_input, input_scale), (weight_fp8, weight_scale_inv), output,
            disable_ue8m0_cast=_disable_ue8m0_cast(),
        )

        if bias is not None:
            output = output + bias

        return output if two_d else output.view(*input_bf16.shape[:-1], N)
