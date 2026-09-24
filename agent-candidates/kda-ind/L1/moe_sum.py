"""Top-k expert-output reduction with 128-bit gather-add loads.

``out[m, d] = sum_k input[m * topk + k, d]``.

Every input byte is read once and every output byte written once, so there is no
reuse to exploit and the operator is bounded by memory traffic and by how many
instructions it takes to move that traffic.  Two things follow, and they are the
whole design:

* Widen the access.  The reference kernel issues one 2-byte load per element per
  expert.  Loading 16 bytes at a time cuts the global-load instruction count by
  ``16 / sizeof(scalar_t)`` for exactly the same DRAM bytes.
* Shorten the host path.  At one token the kernel moves 72 kB and the call is
  dominated by dispatch, so shape derivation, allocation, the device guard, the
  stream lookup and the dtype dispatch all live on the C++ side and ``forward``
  is a single call into the extension.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Deliberately not "moe_sum": ``torch.utils.cpp_extension`` derives the build
# directory, the shared-object name and the ``sys.modules`` entry from this
# string, and the reference implementation already owns "moe_sum" in the same
# process.
_EXTENSION_NAME = "moe_sum_gather_add_v1"

_CPP_SOURCE = r"""
#include <tuple>

at::Tensor moe_sum_gather_add(const at::Tensor& input, int64_t topk);
at::Tensor moe_sum_gather_add_forced(const at::Tensor& input, int64_t topk,
                                     int64_t block, int64_t vectors_per_thread,
                                     bool wide);
std::tuple<bool, int64_t, int64_t, int64_t, int64_t> moe_sum_launch_plan(
    const at::Tensor& input, int64_t topk);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sum", &moe_sum_gather_add,
        "Reduce top-k expert outputs into one row per token (CUDA)");
  m.def("launch_plan", &moe_sum_launch_plan,
        "(wide_path, vector_elems, block, grid_x, grid_y) for this input");
  m.def("moe_sum_forced", &moe_sum_gather_add_forced,
        "Same reduction with the launch configuration pinned; for measuring "
        "configurations against each other, not for the hot path");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {

// 16 bytes is the widest single global access the SM issues (LDG.E.128 /
// STG.E.128).  One of these per thread per expert row is the point of the
// kernel.
constexpr int kWideBytes = 16;

// gridDim.y is capped at 65535, so tokens beyond that are picked up by a
// grid-stride loop rather than by a taller grid.
constexpr int64_t kMaxGridY = 65535;

// A 16-byte-aligned bundle of scalars.  The tensor storage holds live `scalar_t`
// objects, not `WideChunk` objects, so it is read and written with
// `__builtin_memcpy` through an `__builtin_assume_aligned` pointer rather than
// by dereferencing a cast: that is well defined under the object model and,
// because the host side has already proven the 16-byte alignment (see
// choose_plan), nvcc still folds each transfer into a single LDG.E.128 /
// STG.E.128.
//
// The alignment hint is load-bearing, not decoration.  Measured on sm_100a, per
// output vector:
//   dereference a reinterpret_cast    8 x LDG.E.128  + 1 x STG.E.128   (UB)
//   assume_aligned + memcpy           8 x LDG.E.128  + 1 x STG.E.128   (defined)
//   bare memcpy, no alignment hint  128 x LDG.E.U8   + 16 x STG.E.U8
// A plain memcpy from a `scalar_t*` carries only 2-byte alignment, so dropping
// the hint silently scalarizes the kernel and undoes the entire optimization.
template <typename scalar_t, int kElems>
struct alignas(kWideBytes) WideChunk {
  scalar_t v[kElems];
};

// 16 bytes from proven-aligned scalar storage into a local chunk, and back.
template <typename Chunk, typename scalar_t>
__device__ __forceinline__ Chunk load_wide(const scalar_t* p) {
  Chunk chunk;
  __builtin_memcpy(&chunk, __builtin_assume_aligned(p, kWideBytes), kWideBytes);
  return chunk;
}

template <typename Chunk, typename scalar_t>
__device__ __forceinline__ void store_wide(scalar_t* p, const Chunk& chunk) {
  __builtin_memcpy(__builtin_assume_aligned(p, kWideBytes), &chunk, kWideBytes);
}

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

struct LaunchPlan {
  bool wide;        // false selects the scalar fallback
  int vector_elems; // scalar_t elements per 16-byte access (1 on the fallback)
  int lanes;        // independent work items per token
  int block;
  int64_t grid_x;
  int64_t grid_y;
};

// Zero means "decide automatically".  Only the measurement entry point sets
// these; the hot path always runs the automatic heuristic.
struct ConfigOverride {
  int block = 0;
  int vectors_per_thread = 0;
  bool force_scalar = false;
};

// Vectors each thread walks in the automatic configuration.  One keeps the
// column loop a single trip for D = 4096 at 256 threads and leaves the
// memory-level parallelism to the topk independent load streams.
constexpr int kAutoVectorsPerThread = 1;

// One thread owns one 16-byte vector of output and reads the topk corresponding
// 16-byte vectors from the expert rows.  The k axis is entirely thread-private,
// so there is no shared memory, no shuffle and no __syncthreads anywhere: this
// is an n-way gather plus an add, not a parallel reduction.
template <typename scalar_t, int kTopkStatic>
__global__ void gather_add_wide_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    const int vecs_per_row,
    const int hidden,
    const int topk_dynamic,
    const int64_t tokens) {
  constexpr int kElems = kWideBytes / sizeof(scalar_t);
  using Chunk = WideChunk<scalar_t, kElems>;
  const int topk = kTopkStatic > 0 ? kTopkStatic : topk_dynamic;

  // blockIdx.y is the token and blockIdx.x the column tile, so no flat index
  // ever has to be divided by vecs_per_row -- a division would cost roughly as
  // much as the rest of the inner body.
  for (int64_t token = blockIdx.y; token < tokens; token += gridDim.y) {
    // The only 64-bit arithmetic in the kernel, and it runs once per token
    // instead of once per element.
    const scalar_t* __restrict__ row =
        in + token * static_cast<int64_t>(topk) * hidden;
    scalar_t* __restrict__ out_row = out + token * static_cast<int64_t>(hidden);

    // Column assignment is block-strided rather than thread-contiguous, so the
    // 32 lanes of a warp read 32 consecutive 16-byte vectors: one contiguous
    // 512-byte span per load instruction with every byte used.  Giving each
    // thread a contiguous run of vectors would move the same DRAM bytes with
    // worse sectors-per-request.
    //
    // The loop counter is 64-bit because `gridDim.x * blockDim.x` is an
    // unsigned product that can exceed INT_MAX for a very wide row, and
    // wrapping it into a signed int would produce a negative index that still
    // passes the bound test.  Only this bookkeeping is 64-bit; `col` and the
    // per-expert offsets below stay 32-bit, which is where the instruction
    // count actually matters.
    const int64_t column_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t c64 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         c64 < vecs_per_row; c64 += column_stride) {
      const int col = static_cast<int>(c64) * kElems;
      float acc[kElems];
#pragma unroll
      for (int j = 0; j < kElems; ++j) acc[j] = 0.0f;

      // Ascending k into an fp32 accumulator: the same reduction order and the
      // same accumulation width as the reference float-accumulating kernel, so
      // the two agree bit for bit.  The expert rows sit hidden*sizeof(scalar_t)
      // bytes apart, which makes these topk independent load streams and gives
      // memory-level parallelism for free.
      auto accumulate = [&](const int k) {
        const Chunk chunk = load_wide<Chunk>(row + k * hidden + col);
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          acc[j] += static_cast<float>(chunk.v[j]);
        }
      };
      if constexpr (kTopkStatic > 0) {
#pragma unroll
        for (int k = 0; k < kTopkStatic; ++k) accumulate(k);
      } else {
        for (int k = 0; k < topk; ++k) accumulate(k);
      }

      Chunk result;
#pragma unroll
      for (int j = 0; j < kElems; ++j) {
        result.v[j] = static_cast<scalar_t>(acc[j]);
      }
      store_wide<Chunk>(out_row + col, result);
    }
  }
}

// Fallback for hidden sizes that are not a whole number of 16-byte vectors, or
// for buffers the allocator did not hand out 16-byte aligned.  It only has to
// be correct, so it is not templated on topk and keeps the same fp32 ascending-k
// reduction as the wide path.
template <typename scalar_t>
__global__ void gather_add_scalar_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    const int hidden,
    const int topk,
    const int64_t tokens) {
  for (int64_t token = blockIdx.y; token < tokens; token += gridDim.y) {
    const scalar_t* __restrict__ row =
        in + token * static_cast<int64_t>(topk) * hidden;
    scalar_t* __restrict__ out_row = out + token * static_cast<int64_t>(hidden);
    // 64-bit loop bookkeeping for the same reason as the wide kernel.
    const int64_t column_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t d64 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         d64 < hidden; d64 += column_stride) {
      const int d = static_cast<int>(d64);
      float acc = 0.0f;
      for (int k = 0; k < topk; ++k) {
        acc += static_cast<float>(row[k * hidden + d]);
      }
      out_row[d] = static_cast<scalar_t>(acc);
    }
  }
}

template <typename scalar_t>
LaunchPlan choose_plan(
    const void* in_ptr,
    const void* out_ptr,
    const int hidden,
    const int64_t tokens,
    const int sm_count,
    const ConfigOverride ov = ConfigOverride{}) {
  constexpr int kElems = kWideBytes / sizeof(scalar_t);

  // Relying on the caching allocator's alignment would be relying on an
  // implementation detail, so the width is chosen from the pointers we actually
  // got.
  const bool wide = !ov.force_scalar &&
                    (hidden % kElems == 0) &&
                    (reinterpret_cast<uintptr_t>(in_ptr) % kWideBytes == 0) &&
                    (reinterpret_cast<uintptr_t>(out_ptr) % kWideBytes == 0);

  LaunchPlan plan{};
  plan.wide = wide;
  plan.vector_elems = wide ? kElems : 1;
  plan.lanes = wide ? hidden / kElems : hidden;

  const int64_t rows_in_grid = std::min<int64_t>(tokens, kMaxGridY);
  int block = ov.block > 0 ? ov.block : 256;
  int per_thread = ov.vectors_per_thread > 0 ? ov.vectors_per_thread
                                             : kAutoVectorsPerThread;

  if (ov.block <= 0) {
    // Prefer 256-thread blocks, but shrink toward 64 while the grid would leave
    // the device idle.  At one token the reference kernel launches a single
    // block on 148 SMs, and block count is the only parallelism available there.
    const int64_t want_blocks = 2LL * sm_count;
    while (block > 64) {
      if (ceil_div(plan.lanes, static_cast<int64_t>(block) * per_thread) *
              rows_in_grid >= want_blocks) {
        break;
      }
      block >>= 1;
    }
    // Never launch more threads per block than one tile has work for.
    if (plan.lanes < block) {
      block = static_cast<int>(std::max<int64_t>(32, ceil_div(plan.lanes, 32) * 32));
    }
  }
  block = std::min(block, 1024);

  plan.block = block;
  plan.grid_x = std::max<int64_t>(
      1, ceil_div(plan.lanes, static_cast<int64_t>(block) * per_thread));
  plan.grid_y = rows_in_grid;
  return plan;
}

template <typename scalar_t>
void launch_gather_add(
    at::Tensor& out,
    const at::Tensor& input,
    const int hidden,
    const int topk,
    const int64_t tokens,
    const int sm_count,
    const cudaStream_t stream,
    const ConfigOverride ov = ConfigOverride{}) {
  auto* out_ptr = out.data_ptr<scalar_t>();
  const auto* in_ptr = input.data_ptr<scalar_t>();
  const LaunchPlan plan =
      choose_plan<scalar_t>(in_ptr, out_ptr, hidden, tokens, sm_count, ov);

  const dim3 grid(static_cast<unsigned>(plan.grid_x),
                  static_cast<unsigned>(plan.grid_y));
  const dim3 block(static_cast<unsigned>(plan.block));

  if (!plan.wide) {
    gather_add_scalar_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_ptr, in_ptr, hidden, topk, tokens);
    return;
  }

#define MOE_SUM_LAUNCH_WIDE(TOPK_STATIC)                                  \
  gather_add_wide_kernel<scalar_t, TOPK_STATIC><<<grid, block, 0, stream>>>( \
      out_ptr, in_ptr, plan.lanes, hidden, topk, tokens)

  // The captured workload is entirely topk == 8; the other specializations exist
  // so the operator is not a single-configuration kernel, and topk values
  // outside the set fall through to a runtime-trip-count instantiation.
  switch (topk) {
    case 2: MOE_SUM_LAUNCH_WIDE(2); break;
    case 3: MOE_SUM_LAUNCH_WIDE(3); break;
    case 4: MOE_SUM_LAUNCH_WIDE(4); break;
    case 8: MOE_SUM_LAUNCH_WIDE(8); break;
    default: MOE_SUM_LAUNCH_WIDE(0); break;
  }

#undef MOE_SUM_LAUNCH_WIDE
}

struct Problem {
  int64_t tokens;
  int hidden;
  int topk;
};

Problem validate_and_shape(const at::Tensor& input, int64_t topk) {
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();

  TORCH_CHECK(input.is_cuda(), "moe_sum: input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "moe_sum: input must be 2-D [M*topk, D], got ",
              input.dim(), "-D");
  TORCH_CHECK(input.is_contiguous(), "moe_sum: input must be contiguous");
  TORCH_CHECK(topk > 0, "moe_sum: topk must be positive, got ", topk);
  // Checked before the product below so that forming the product cannot itself
  // overflow int64.
  TORCH_CHECK(topk <= kIntMax, "moe_sum: topk (", topk, ") is out of range");

  // Rejecting unsupported dtypes here rather than at the dispatch means an
  // empty input cannot slip past the check just because it returns early.
  const auto dtype = input.scalar_type();
  TORCH_CHECK(dtype == at::ScalarType::BFloat16 ||
                  dtype == at::ScalarType::Half ||
                  dtype == at::ScalarType::Float,
              "moe_sum: unsupported dtype ", dtype,
              "; expected bfloat16, float16 or float32");

  const int64_t rows = input.size(0);
  const int64_t hidden = input.size(1);
  TORCH_CHECK(rows % topk == 0, "moe_sum: input rows (", rows,
              ") must be divisible by topk (", topk, ")");
  TORCH_CHECK(hidden <= kIntMax / topk,
              "moe_sum: topk (", topk, ") * D (", hidden,
              ") exceeds the 32-bit range the kernel indexes rows with");

  Problem p{};
  p.tokens = rows / topk;
  p.hidden = static_cast<int>(hidden);
  p.topk = static_cast<int>(topk);
  return p;
}

at::Tensor reduce_experts(const at::Tensor& input, int64_t topk,
                          const ConfigOverride ov) {
  const Problem p = validate_and_shape(input, topk);
  at::Tensor out = at::empty({p.tokens, static_cast<int64_t>(p.hidden)},
                             input.options());
  if (p.tokens == 0 || p.hidden == 0) return out;

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_gather_add<at::BFloat16>(out, input, p.hidden, p.topk, p.tokens,
                                      sm_count, stream, ov);
      break;
    case at::ScalarType::Half:
      launch_gather_add<at::Half>(out, input, p.hidden, p.topk, p.tokens,
                                  sm_count, stream, ov);
      break;
    case at::ScalarType::Float:
      launch_gather_add<float>(out, input, p.hidden, p.topk, p.tokens, sm_count,
                               stream, ov);
      break;
    default:
      // Unreachable: validate_and_shape already rejected everything else.
      TORCH_CHECK(false, "moe_sum: unsupported dtype ", input.scalar_type());
  }
  return out;
}

}  // namespace

at::Tensor moe_sum_gather_add(const at::Tensor& input, int64_t topk) {
  return reduce_experts(input, topk, ConfigOverride{});
}

// Measurement-only entry point: pins the launch configuration so candidate
// configurations can be timed against each other instead of argued about.  The
// hot path never goes through here.
at::Tensor moe_sum_gather_add_forced(const at::Tensor& input, int64_t topk,
                                     int64_t block, int64_t vectors_per_thread,
                                     bool wide) {
  TORCH_CHECK(block > 0 && block <= 1024 && block % 32 == 0,
              "moe_sum_forced: block must be a positive multiple of 32 up to "
              "1024, got ", block);
  TORCH_CHECK(vectors_per_thread > 0 &&
                  vectors_per_thread <= std::numeric_limits<int>::max(),
              "moe_sum_forced: vectors_per_thread must be positive and fit in "
              "an int, got ", vectors_per_thread);
  ConfigOverride ov;
  ov.block = static_cast<int>(block);
  ov.vectors_per_thread = static_cast<int>(vectors_per_thread);
  ov.force_scalar = !wide;
  return reduce_experts(input, topk, ov);
}

// Reports the configuration ``moe_sum`` would pick for this input without
// launching anything, so the wide path can be confirmed rather than assumed.
std::tuple<bool, int64_t, int64_t, int64_t, int64_t> moe_sum_launch_plan(
    const at::Tensor& input, int64_t topk) {
  const Problem p = validate_and_shape(input, topk);
  at::Tensor out = at::empty({p.tokens, static_cast<int64_t>(p.hidden)},
                             input.options());
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const int sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  LaunchPlan plan{};
  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      plan = choose_plan<at::BFloat16>(input.data_ptr(), out.data_ptr(),
                                       p.hidden, p.tokens, sm_count);
      break;
    case at::ScalarType::Half:
      plan = choose_plan<at::Half>(input.data_ptr(), out.data_ptr(), p.hidden,
                                   p.tokens, sm_count);
      break;
    case at::ScalarType::Float:
      plan = choose_plan<float>(input.data_ptr(), out.data_ptr(), p.hidden,
                                p.tokens, sm_count);
      break;
    default:
      TORCH_CHECK(false, "moe_sum: unsupported dtype ", input.scalar_type(),
                  "; expected bfloat16, float16 or float32");
  }
  return std::make_tuple(plan.wide, static_cast<int64_t>(plan.vector_elems),
                         static_cast<int64_t>(plan.block), plan.grid_x,
                         plan.grid_y);
}
"""

_extension = None


def _local_arch() -> str:
    """The single architecture string to build for.

    Ambient ``TORCH_CUDA_ARCH_LIST`` values are routinely multi-architecture
    (a generic ``7.5 8.0 8.6 9.0 10.0 12.0+PTX`` is common), which makes the
    build both much slower and *unstable*: the nvcc command line then depends on
    whether something else in the process re-pinned the variable first, and any
    change to that command line makes ninja recompile from scratch.  Deriving a
    single arch from the device we are about to run on keeps the flags identical
    on every import, so the build directory stays a cache hit.

    Caveat: this reads the *current* device at build time, and ``__init__`` takes
    no device argument, so a process that constructs on one architecture and then
    feeds tensors from a different one would have built for the wrong target.
    That cannot happen on a single-architecture machine; a heterogeneous
    multi-GPU host would need the arch list widened to cover every device it may
    be handed.
    """
    major, minor = torch.cuda.get_device_capability()
    # Blackwell and Hopper need the arch-specific "a" variant for their
    # architecture-gated instructions; this matches what the surrounding
    # framework pins for the same device.
    suffix = "a" if major >= 9 else ""
    return f"{major}.{minor}{suffix}"


def _load_extension():
    """Build (or fetch the cached build of) the gather-add extension."""
    global _extension
    if _extension is None:
        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = _local_arch()
        try:
            _extension = _build_extension()
        finally:
            # Leave the process as we found it; other extensions in this process
            # have their own opinion about which architectures to target.
            if previous_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
    return _extension


def _build_extension():
    return load_inline(
        name=_EXTENSION_NAME,
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_cflags=["-O3"],
        # Deliberately no --use_fast_math: nvcc must not be allowed to
        # reassociate the fp32 add chain, or the reduction order stops matching
        # the reference kernel's.  -lineinfo is for ncu source correlation and
        # costs nothing at runtime.
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )


class MoeSum(nn.Module):
    """Reduce the top-k expert outputs of each token into a single row."""

    def __init__(self):
        super().__init__()
        self._kernel = _load_extension()

    def forward(
        self,
        input: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Sum over the topk dimension.

        Args:
            input: [M * topk, D] tensor
            topk: number of experts per token

        Returns:
            output: [M, D] tensor
        """
        return self._kernel.moe_sum(input, topk)
