"""Timestep and text projection embeddings for diffusion models (L2 composite).

Construction and call contract are identical to the baseline, including
``state_dict`` parameter names, so a diffusers checkpoint loads unchanged.

Every benched shape has batch 1, which makes each projection a matrix-vector
product: there is no tensor-core work to win and the baseline is limited by how
many kernels it launches, not by arithmetic.  The baseline turns one scalar into
a 3072-wide embedding through roughly thirty eager kernels.  All four classes
here are the same primitive -- some number of independent
(optional sinusoid, project, SiLU, project) chains whose results are summed --
so one CUDA kernel family serves all of them in two launches per call: one that
projects every chain's input and applies the activation, and one that projects
the activations and sums the chains.

Inputs outside the supported envelope take a plain torch path that reproduces the
baseline exactly: non-CUDA, non-bfloat16, batched, non-contiguous, misaligned,
oddly shaped, wider than a block's shared memory, or needing an autograd graph.
The kernels write their outputs directly and register no backward formula, so a
call that needs gradients falls back; inference under ``no_grad``, which is how
this operator is used, reaches the kernels.
"""

from __future__ import annotations

import math
import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "get_timestep_embedding",
    "Timesteps",
    "TimestepEmbedding",
    "CombinedTimestepTextProjEmbeddings",
    "CombinedTimestepGuidanceTextProjEmbeddings",
]

# Channel and width counts are held in signed ints inside the kernels, so a
# request anywhere near that limit takes the torch path instead of wrapping.
_MAX_CHANNELS = 1 << 24

# Largest shared-memory request a block will make: one staged sinusoid vector.
# Well inside the 48 KB a block gets without opting into the dynamic carveout, so
# no launch can fail on a shared-memory limit; anything larger takes the torch path.
_MAX_SMEM_BYTES = 40960

# Set FK_TIMESTEP_EMBEDDING_TRACE=1 to count which path each call took.  Off, the
# cost is one module-level bool test per call and nothing is recorded, so the
# counters cannot perturb a timed run.
_TRACE = os.environ.get("FK_TIMESTEP_EMBEDDING_TRACE", "") == "1"
# Counted per call site: a chain class that falls back still runs the sinusoid
# kernel, and a combined class that falls back still runs the single-chain kernel
# for each of its sub-embedders.  One pair of counters could not tell those apart,
# and a fast-path check built on it would pass vacuously.
path_counts: dict[str, int] = {
    "sinusoid_kernel": 0, "sinusoid_torch": 0,
    "mlp_kernel": 0, "mlp_torch": 0,
    "combined_kernel": 0, "combined_torch": 0,
}

# Set FK_TIMESTEP_EMBEDDING_DISABLE=1 to force every call down the torch path.
# Correctness alone cannot show the kernels ran, so the check that they did needs
# a configuration in which they demonstrably do not.
_DISABLED = os.environ.get("FK_TIMESTEP_EMBEDDING_DISABLE", "") == "1"


def _note(path: str) -> None:
    path_counts[path] += 1


def _needs_grad(*tensors: torch.Tensor) -> bool:
    """Whether this call has to build an autograd graph.

    The kernels write their outputs directly and register no backward formula, so
    a call that would need gradients takes the torch path.  Inference under
    ``no_grad`` -- which is how this operator is used and benchmarked -- still
    reaches the kernels.
    """
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxChains = 3;
// Channel and width counts are held in signed ints inside the kernels.
constexpr int64_t kMaxChannels = 1 << 24;
constexpr unsigned kFullMask = 0xffffffffu;

enum SrcKind : int { kSrcDirect = 0, kSrcSinusoid = 1 };

// Frequency schedule of the sinusoidal embedding.  Every constant is folded on
// the host in the same order and precision the reference builds it in: the
// period logarithm is taken in double and rounded once to float, and the
// divisor arrives as a reciprocal because torch's division by a python scalar
// multiplies by one.
struct SinusoidSpec {
  float neg_log_period;
  float inv_divisor;
  float scale;
  int half;
  int channels;
  int flip;
};

struct ChainDesc {
  const __nv_bfloat16* w1;
  const __nv_bfloat16* b1;
  const __nv_bfloat16* w2;
  const __nv_bfloat16* b2;
  const __nv_bfloat16* src;
  int k;
  int kind;
};

// Passed to the kernels by value: at ~150 bytes it fits the parameter space, so
// the descriptors need no device allocation and no extra launch to upload.
struct ChainPack {
  ChainDesc c[kMaxChains];
  int n_chain;
};

union BfVec8 {
  uint4 raw;
  __nv_bfloat16 h[8];
};

__device__ __forceinline__ BfVec8 load_shared8(const __nv_bfloat16* p) {
  BfVec8 v;
  v.raw = *reinterpret_cast<const uint4*>(p);
  return v;
}

// Vectors that every warp re-reads (the chain input, the activations) stay in L1
// on the read-only path.
__device__ __forceinline__ BfVec8 load_reuse8(const __nv_bfloat16* __restrict__ p) {
  BfVec8 v;
  v.raw = __ldg(reinterpret_cast<const uint4*>(p));
  return v;
}

// Weights are touched exactly once per call, so they skip L1 rather than evicting
// the vectors that are still needed there.
__device__ __forceinline__ BfVec8 load_stream8(const __nv_bfloat16* __restrict__ p) {
  BfVec8 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.raw.x), "=r"(v.raw.y), "=r"(v.raw.z), "=r"(v.raw.w)
               : "l"(p)
               : "memory");
  return v;
}

// Programmatic Dependent Launch. The second stage's dominant cost is streaming the
// second projection's weights, which do not depend on the first stage's output, so
// it can be resident and issuing those loads while the first stage is still
// running; only the activation reads have to wait. Both intrinsics need sm_90, and
// the wait is a no-op when the kernel was not launched with the dependency
// attribute, so an ordinary launch stays correct.
__device__ __forceinline__ void dependent_launch_signal() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ void dependent_launch_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(float v) { return v; }

// One element of the embedding: cos occupies the low half and sin the high half
// when flipped, and both halves share the same frequency index.
__device__ __forceinline__ float sinusoid_at(float t, const SinusoidSpec& s, int j) {
  const int i = (j < s.half) ? j : (j - s.half);
  const bool use_cos = s.flip ? (j < s.half) : (j >= s.half);
  const float exponent = (s.neg_log_period * static_cast<float>(i)) * s.inv_divisor;
  float v = t * expf(exponent);
  v = s.scale * v;
  return use_cos ? cosf(v) : sinf(v);
}

// Grid is (channels-worth of threads, one row per blockIdx.y) so the row index
// needs no division.
template <typename src_t>
__global__ void sinusoid_embedding_kernel(const src_t* __restrict__ timesteps,
                                         float* __restrict__ out, int64_t n_ts,
                                         SinusoidSpec spec) {
  // Strided over rows as well as channels: gridDim.y tops out at 65535, and a
  // rank-1 input longer than that is something the reference accepts.  The row
  // count is 64-bit throughout -- a narrower one wraps negative past 2^31 and the
  // loop then writes nothing at all, returning an uninitialized buffer.
  for (int64_t row = blockIdx.y; row < n_ts; row += gridDim.y) {
    const float t = to_float(timesteps[row]);
    float* __restrict__ dst = out + row * spec.channels;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < spec.channels;
         j += blockDim.x * gridDim.x)
      dst[j] = sinusoid_at(t, spec, j);
  }
}

// Project each chain's input and apply the activation, one output row per warp.
// Blocks are assigned per chain (grid.y), so a block only touches the one input
// it reads and a sinusoid input is regenerated by a third as many blocks as a
// flat chain-major row space would need.
//
// A sinusoid input is staged in shared memory: it costs arithmetic rather than a
// load, so computing it once per block and paying a barrier is cheaper than
// having every warp redo it.  A direct input is read per warp instead -- staging
// it would put a dependent global load in front of the barrier and stop the input
// and weight fetches from overlapping.
template <int kWarps, int kStaticK>
__global__ __launch_bounds__(kWarps* kWarpSize) void chain_project_act_kernel(
    ChainPack pack, int n_out, SinusoidSpec spec, __nv_bfloat16* __restrict__ act) {
  // Read through 128-bit loads below, so the alignment has to be stated.
  extern __shared__ __align__(16) __nv_bfloat16 s_x[];
  const ChainDesc ch = pack.c[blockIdx.y];
  const int k = (kStaticK > 0) ? kStaticK : ch.k;
  const bool staged = (ch.kind == kSrcSinusoid);

  if (staged) {
    const float t = to_float(ch.src[0]);
    for (int j = threadIdx.x; j < k; j += kWarps * kWarpSize)
      s_x[j] = __float2bfloat16(sinusoid_at(t, spec, j));
    __syncthreads();
  }

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarps + warp;
  if (row >= n_out) return;

  const __nv_bfloat16* __restrict__ w_row = ch.w1 + static_cast<size_t>(row) * k;
  float acc = 0.f;
#pragma unroll 4
  for (int base = lane * 8; base < k; base += kWarpSize * 8) {
    const BfVec8 w = load_stream8(w_row + base);
    const BfVec8 x = staged ? load_shared8(s_x + base) : load_reuse8(ch.src + base);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc = fmaf(to_float(w.h[e]), to_float(x.h[e]), acc);
  }
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1)
    acc += __shfl_xor_sync(kFullMask, acc, off);

  if (lane == 0) {
    // Rounded to bfloat16 after the bias, then again after the activation --
    // the two places the reference's dtype boundaries fall.
    const float pre = to_float(__float2bfloat16(acc + to_float(ch.b1[row])));
    act[static_cast<size_t>(blockIdx.y) * n_out + row] =
        __float2bfloat16(pre / (1.0f + expf(-pre)));
  }
  dependent_launch_signal();
}

// Project every chain's activations and sum them.  One output row is shared by
// kSplit warps, each covering a slice of the reduction: the row count alone caps
// the launch at N warps, which leaves the machine two thirds idle, so splitting
// the reduction is the only way to raise occupancy.  The kSplit partial sums meet
// in shared memory.
//
// The activations are broadcast data, small enough to sit in L1, and reading them
// through the read-only path rather than staging them keeps shared memory tiny --
// which matters once kSplit has multiplied the number of resident blocks.
template <int kWarps, int kStaticN, int kChains, int kSplit>
__global__ __launch_bounds__(kWarps* kWarpSize) void chain_project_sum_kernel(
    ChainPack pack, int n_out_rt, const __nv_bfloat16* __restrict__ act,
    __nv_bfloat16* __restrict__ out) {
  constexpr int kRowsPerBlock = kWarps / kSplit;
  __shared__ float s_part[kRowsPerBlock][kChains][kSplit];

  const int n_out = (kStaticN > 0) ? kStaticN : n_out_rt;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row_local = warp / kSplit;
  const int part = warp % kSplit;
  const int row = blockIdx.x * kRowsPerBlock + row_local;

  const int slice = n_out / kSplit;
  const int k_begin = part * slice;
  const int k_end = (part == kSplit - 1) ? n_out : k_begin + slice;

  dependent_launch_wait();

  if (row < n_out) {
    float acc[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) acc[c] = 0.f;

#pragma unroll 2
    for (int base = k_begin + lane * 8; base < k_end; base += kWarpSize * 8) {
#pragma unroll
      for (int c = 0; c < kChains; ++c) {
        const BfVec8 w =
            load_stream8(pack.c[c].w2 + static_cast<size_t>(row) * n_out + base);
        const BfVec8 h = load_reuse8(act + static_cast<size_t>(c) * n_out + base);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          acc[c] = fmaf(to_float(w.h[e]), to_float(h.h[e]), acc[c]);
      }
    }
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
#pragma unroll
      for (int c = 0; c < kChains; ++c)
        acc[c] += __shfl_xor_sync(kFullMask, acc[c], off);
    }
    if (lane == 0) {
#pragma unroll
      for (int c = 0; c < kChains; ++c) s_part[row_local][c][part] = acc[c];
    }
  }
  __syncthreads();

  if (threadIdx.x < kRowsPerBlock) {
    const int r_local = threadIdx.x;
    const int r = blockIdx.x * kRowsPerBlock + r_local;
    if (r < n_out) {
      // Each chain is rounded to bfloat16 on its own and the chains are then
      // added left to right, matching the reference's `a + b + c`.  Carrying one
      // fp32 accumulator across chains instead doubles the error.
      __nv_bfloat16 sum = __float2bfloat16(0.f);
#pragma unroll
      for (int c = 0; c < kChains; ++c) {
        float total = 0.f;
#pragma unroll
        for (int p = 0; p < kSplit; ++p) total += s_part[r_local][c][p];
        const __nv_bfloat16 y = __float2bfloat16(total + to_float(pack.c[c].b2[r]));
        sum = (c == 0) ? y : __float2bfloat16(to_float(sum) + to_float(y));
      }
      out[r] = sum;
    }
  }
}

constexpr int kActWarps = 8;
// One block, four warps, one output row: the reduction split is what raises the
// second stage from a third of a wave per multiprocessor to well over one, and
// spending it on smaller blocks rather than more rows per block measured better
// still (the combined case, 40.0 -> 36.0 us, at identical warp count -- 3072
// blocks distribute across 148 multiprocessors more evenly than 1536 do).
// Measured alternatives, all numerically identical: split 1 / 2 / 8 give
// 48.2 / 44.1 / 44.1 us, and 16 warps gives 40.1.
constexpr int kSumWarps = 4;
constexpr int kSumSplit = 4;

// Cached per device: whether the dependency attribute is usable at all. Racing
// threads compute the same answer, so no lock is needed.
bool dependent_launch_supported(int device) {
  constexpr int kMaxDevices = 16;
  static int cache[kMaxDevices] = {};
  if (device < 0 || device >= kMaxDevices) return false;
  if (cache[device] == 0) {
    int major = 0;
    cache[device] =
        (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) ==
             cudaSuccess &&
         major >= 9)
            ? 1
            : -1;
  }
  return cache[device] == 1;
}

template <int kWarps, int kStaticN, int kChains, int kSplit>
void launch_project_sum(dim3 grid, cudaStream_t stream, bool dependent, ChainPack pack,
                        int n_out, const __nv_bfloat16* act, __nv_bfloat16* out) {
  auto kernel = chain_project_sum_kernel<kWarps, kStaticN, kChains, kSplit>;
  if (dependent) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(kWarps * kWarpSize);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, pack, n_out, act, out));
  } else {
    kernel<<<grid, kWarps * kWarpSize, 0, stream>>>(pack, n_out, act, out);
  }
}

SinusoidSpec make_spec(int64_t channels, bool flip, double shift, double scale,
                       double max_period) {
  SinusoidSpec s;
  s.half = static_cast<int>(channels / 2);
  s.channels = static_cast<int>(channels);
  s.flip = flip ? 1 : 0;
  s.neg_log_period = static_cast<float>(-std::log(max_period));
  s.inv_divisor = 1.0f / static_cast<float>(static_cast<double>(s.half) - shift);
  s.scale = static_cast<float>(scale);
  return s;
}

#define DISPATCH_ACT(K_CONST)                                                  \
  chain_project_act_kernel<kActWarps, K_CONST>                                 \
      <<<act_grid, kActWarps * kWarpSize, smem_x, stream>>>(pack, n_out, spec,  \
                                                            act_ptr)

#define DISPATCH_SUM(N_CONST, C_CONST, SPLIT)                                  \
  launch_project_sum<kSumWarps, N_CONST, C_CONST, SPLIT>(                      \
      dim3((n_out + (kSumWarps / (SPLIT)) - 1) / (kSumWarps / (SPLIT))),        \
      stream, dependent, pack, n_out, act_ptr, out_ptr)

// The chain count and the reduction width are both compile-time here: the chain
// loop has to unroll for the accumulators to stay in registers, and the split
// factor sizes a shared array.
#define DISPATCH_SUM_SPLIT(N_CONST, SPLIT)                                     \
  do {                                                                         \
    if (n_chain == 1) {                                                        \
      DISPATCH_SUM(N_CONST, 1, SPLIT);                                         \
    } else if (n_chain == 2) {                                                  \
      DISPATCH_SUM(N_CONST, 2, SPLIT);                                         \
    } else {                                                                    \
      DISPATCH_SUM(N_CONST, 3, SPLIT);                                         \
    }                                                                          \
  } while (0)

#define DISPATCH_SUM_N(SPLIT)                                                  \
  do {                                                                         \
    if (n_out == 3072) {                                                        \
      DISPATCH_SUM_SPLIT(3072, SPLIT);                                          \
    } else {                                                                    \
      DISPATCH_SUM_SPLIT(0, SPLIT);                                             \
    }                                                                          \
  } while (0)

}  // namespace

at::Tensor timestep_sinusoid(const at::Tensor& timesteps, int64_t num_channels,
                             bool flip_sin_to_cos, double downscale_freq_shift,
                             double scale, double max_period) {
  TORCH_CHECK(timesteps.is_cuda(), "timesteps must be a CUDA tensor");
  TORCH_CHECK(timesteps.dim() == 1, "timesteps must be rank 1");
  TORCH_CHECK(num_channels > 0 && num_channels % 2 == 0,
              "num_channels must be positive and even");
  // The kernel indexes channels with a signed int; anything near that limit has
  // to stay on the torch path rather than silently wrap.
  TORCH_CHECK(num_channels <= kMaxChannels, "num_channels too large for the kernel");

  const c10::cuda::OptionalCUDAGuard guard(at::device_of(timesteps));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t n_ts = timesteps.numel();
  auto out = torch::empty({n_ts, num_channels},
                          timesteps.options().dtype(at::kFloat));
  if (n_ts == 0) return out;

  const SinusoidSpec spec =
      make_spec(num_channels, flip_sin_to_cos, downscale_freq_shift, scale, max_period);
  const int threads = 256;
  const dim3 grid(
      static_cast<unsigned>(std::min<int64_t>(
          (num_channels + threads - 1) / threads, 65535)),
      static_cast<unsigned>(std::min<int64_t>(n_ts, 65535)));

  switch (timesteps.scalar_type()) {
    case at::kBFloat16:
      sinusoid_embedding_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(timesteps.data_ptr()),
          out.data_ptr<float>(), n_ts, spec);
      break;
    case at::kHalf:
      sinusoid_embedding_kernel<__half><<<grid, threads, 0, stream>>>(
          reinterpret_cast<const __half*>(timesteps.data_ptr()),
          out.data_ptr<float>(), n_ts, spec);
      break;
    case at::kFloat:
      sinusoid_embedding_kernel<float><<<grid, threads, 0, stream>>>(
          timesteps.data_ptr<float>(), out.data_ptr<float>(), n_ts, spec);
      break;
    default:
      TORCH_CHECK(false, "unsupported timesteps dtype ", timesteps.scalar_type());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor chain_mlp_sum(const std::vector<at::Tensor>& srcs,
                         const std::vector<at::Tensor>& w1s,
                         const std::vector<at::Tensor>& b1s,
                         const std::vector<at::Tensor>& w2s,
                         const std::vector<at::Tensor>& b2s,
                         const std::vector<int64_t>& src_kinds,
                         int64_t sin_channels, bool flip_sin_to_cos,
                         double downscale_freq_shift, double scale,
                         double max_period) {
  const int n_chain = static_cast<int>(srcs.size());
  TORCH_CHECK(n_chain >= 1 && n_chain <= kMaxChains, "unsupported chain count ", n_chain);
  TORCH_CHECK(w1s.size() == srcs.size() && b1s.size() == srcs.size() &&
                  w2s.size() == srcs.size() && b2s.size() == srcs.size() &&
                  src_kinds.size() == srcs.size(),
              "chain argument lists must be the same length");

  const at::Tensor& ref = w1s[0];
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(ref));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t n_out64 = ref.size(0);
  TORCH_CHECK(n_out64 > 0 && n_out64 % 8 == 0, "output width must be a positive multiple of 8");
  const int n_out = static_cast<int>(n_out64);

  ChainPack pack{};
  pack.n_chain = n_chain;
  int max_staged_k = 0;
  bool uniform_k = true;
  for (int c = 0; c < n_chain; ++c) {
    TORCH_CHECK(w1s[c].size(0) == n_out64 && w2s[c].size(0) == n_out64 &&
                    w2s[c].size(1) == n_out64,
                "chain ", c, " has a mismatched output width");
    ChainDesc& d = pack.c[c];
    d.w1 = reinterpret_cast<const __nv_bfloat16*>(w1s[c].data_ptr());
    d.b1 = reinterpret_cast<const __nv_bfloat16*>(b1s[c].data_ptr());
    d.w2 = reinterpret_cast<const __nv_bfloat16*>(w2s[c].data_ptr());
    d.b2 = reinterpret_cast<const __nv_bfloat16*>(b2s[c].data_ptr());
    d.src = reinterpret_cast<const __nv_bfloat16*>(srcs[c].data_ptr());
    d.k = static_cast<int>(w1s[c].size(1));
    d.kind = static_cast<int>(src_kinds[c]);
    TORCH_CHECK(d.k > 0 && d.k % 8 == 0, "chain ", c, " input width must be a multiple of 8");
    if (c > 0 && d.k != pack.c[0].k) uniform_k = false;
    if (d.kind == kSrcSinusoid) max_staged_k = std::max(max_staged_k, d.k);
  }

  // Only a sinusoid input is staged, so a wide direct input costs no shared
  // memory and must not be turned away by this budget.
  const size_t smem_x = static_cast<size_t>(max_staged_k) * sizeof(__nv_bfloat16);
  TORCH_CHECK(smem_x <= 40960, "shared-memory request too large");

  auto act = torch::empty({static_cast<int64_t>(n_chain) * n_out64}, ref.options());
  auto out = torch::empty({n_out64}, ref.options());
  auto* act_ptr = reinterpret_cast<__nv_bfloat16*>(act.data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

  const SinusoidSpec spec =
      make_spec(sin_channels, flip_sin_to_cos, downscale_freq_shift, scale, max_period);

  const dim3 act_grid((n_out + kActWarps - 1) / kActWarps, n_chain);
  const bool dependent = dependent_launch_supported(ref.device().index());
  const dim3 sum_grid((n_out + kSumWarps - 1) / kSumWarps);

  // Specialize the two widths every chain in a diffusion transformer actually
  // uses; anything else runs the same code with a runtime trip count.
  const int k0 = pack.c[0].k;
  if (uniform_k && k0 == 256) {
    DISPATCH_ACT(256);
  } else if (uniform_k && k0 == 768) {
    DISPATCH_ACT(768);
  } else {
    DISPATCH_ACT(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Splitting the reduction needs each slice to stay 8-element aligned so the
  // 128-bit loads keep their alignment; otherwise one warp per row.
  if (n_out % (8 * kSumSplit) == 0) {
    DISPATCH_SUM_N(kSumSplit);
  } else {
    DISPATCH_SUM_N(1);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

at::Tensor timestep_sinusoid(const at::Tensor& timesteps, int64_t num_channels,
                             bool flip_sin_to_cos, double downscale_freq_shift,
                             double scale, double max_period);

at::Tensor chain_mlp_sum(const std::vector<at::Tensor>& srcs,
                         const std::vector<at::Tensor>& w1s,
                         const std::vector<at::Tensor>& b1s,
                         const std::vector<at::Tensor>& w2s,
                         const std::vector<at::Tensor>& b2s,
                         const std::vector<int64_t>& src_kinds,
                         int64_t sin_channels, bool flip_sin_to_cos,
                         double downscale_freq_shift, double scale,
                         double max_period);
"""

_ext_lock = threading.Lock()
_ext_module = None


def _ext():
    """JIT-compile the kernels on first use and memoize them for the process.

    Deliberately not done at import: importing this module on a machine without
    a GPU has to work, and the build is triggered by the first call that is
    eligible for the kernel path.
    """
    global _ext_module
    if _ext_module is not None:
        return _ext_module
    with _ext_lock:
        if _ext_module is None:
            from torch.utils.cpp_extension import load_inline

            if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                major, minor = torch.cuda.get_device_capability()
                suffix = "a" if major in (9, 10, 12) else ""
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"
            _ext_module = load_inline(
                name="fk_timestep_embedding",
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=["timestep_sinusoid", "chain_mlp_sum"],
                extra_cflags=["-O3"],
                # -lineinfo so a profiler can attribute SASS back to this source.
                # No fast-math: the sinusoid has to agree with torch's transcendentals.
                extra_cuda_cflags=[
                    "-O3",
                    "-lineinfo",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
    return _ext_module


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style)."""
    assert len(timesteps.shape) == 1

    if (
        not _DISABLED
        and timesteps.is_cuda
        and timesteps.is_contiguous()
        and timesteps.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and 0 < embedding_dim <= _MAX_CHANNELS
        and embedding_dim % 2 == 0
        and not _needs_grad(timesteps)
    ):
        if _TRACE:
            _note("sinusoid_kernel")
        return _ext().timestep_sinusoid(
            timesteps, embedding_dim, bool(flip_sin_to_cos),
            float(downscale_freq_shift), float(scale), float(max_period),
        )

    if _TRACE:
        _note("sinusoid_torch")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class _Linear(nn.Module):
    """Weight and bias only, so the parameter names match the baseline's Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class _SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


_DIRECT = 0
_SINUSOID = 1


def _chain_kernel_ok(chains, sin_channels: int | None) -> bool:
    """Whether the fused path can serve this call.

    Cheap attribute, shape and pointer tests only -- no tensor contents are read
    and nothing synchronizes, so a rejection costs a handful of python branches.
    """
    if _DISABLED:
        return False

    first = chains[0][2]
    device = first.weight.device
    if device.type != "cuda":
        return False

    n_out = first.weight.shape[0]
    # A zero width is something the reference handles (bias-only, or an empty
    # result); the kernels are not written for it.
    if n_out <= 0 or n_out > _MAX_CHANNELS or n_out % 8 != 0:
        return False

    staged_k = 0
    for kind, src, lin1, lin2 in chains:
        if lin1.bias is None or lin2.bias is None:
            return False
        if _needs_grad(src, lin1.weight, lin1.bias, lin2.weight, lin2.bias):
            return False
        for p in (lin1.weight, lin1.bias, lin2.weight, lin2.bias):
            if p.dtype is not torch.bfloat16 or p.device != device or not p.is_contiguous():
                return False
        # Both projections are read through 128-bit loads, so their rows have to
        # start on a 16-byte boundary: contiguity alone does not imply that.
        if lin1.weight.data_ptr() % 16 or lin2.weight.data_ptr() % 16:
            return False

        k = lin1.weight.shape[1]
        if k <= 0 or k > _MAX_CHANNELS or k % 8 != 0:
            return False
        if lin2.weight.shape != (n_out, n_out):
            return False
        if lin1.bias.shape != (n_out,) or lin2.bias.shape != (n_out,):
            return False

        if src.dtype is not torch.bfloat16 or src.device != device or not src.is_contiguous():
            return False
        if kind == _SINUSOID:
            # The reference feeds this through Timesteps, which asserts rank 1 and
            # yields [1, num_channels]; the broadcast shape above assumes that.
            if src.numel() != 1 or src.dim() != 1:
                return False
            if sin_channels is None or sin_channels != k or sin_channels % 2 != 0:
                return False
            # Only a staged input occupies shared memory.
            staged_k = max(staged_k, k)
        else:
            # Batch 1 only: the staged input vector is shared by every warp in a
            # block, so more than one row would need a different decomposition.
            if src.numel() != k or src.shape[-1] != k:
                return False
            if src.data_ptr() % 16:
                return False

    return staged_k * 2 <= _MAX_SMEM_BYTES


def _chain_kernel(chains, out_shape, sin_channels, flip, shift, scale, max_period):
    kinds = [kind for kind, _, _, _ in chains]
    srcs = [src for _, src, _, _ in chains]
    w1s = [lin1.weight for _, _, lin1, _ in chains]
    b1s = [lin1.bias for _, _, lin1, _ in chains]
    w2s = [lin2.weight for _, _, _, lin2 in chains]
    b2s = [lin2.bias for _, _, _, lin2 in chains]
    out = _ext().chain_mlp_sum(
        srcs, w1s, b1s, w2s, b2s, kinds,
        int(sin_channels or 0), bool(flip), float(shift), float(scale), float(max_period),
    )
    return out.view(out_shape)


class Timesteps(nn.Module):
    """Wraps get_timestep_embedding as an nn.Module."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = _Linear(in_channels, time_embed_dim, bias=True)
        self.act = _SiLU()
        self.linear_2 = _Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        chains = [(_DIRECT, sample, self.linear_1, self.linear_2)]
        if _chain_kernel_ok(chains, None):
            if _TRACE:
                _note("mlp_kernel")
            out_shape = sample.shape[:-1] + (self.linear_2.weight.shape[0],)
            return _chain_kernel(chains, out_shape, None, False, 0.0, 1.0, 10000.0)

        if _TRACE:
            _note("mlp_torch")
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep encoding with pooled text projection.

    Produces ``timestep_embedder`` + ``text_embedder`` weight names matching
    the diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        chains = [
            (_SINUSOID, timestep, self.timestep_embedder.linear_1, self.timestep_embedder.linear_2),
            (_DIRECT, pooled_projection, self.text_embedder.linear_1, self.text_embedder.linear_2),
        ]
        proj = self.time_proj
        if pooled_projection.dtype is torch.bfloat16 and _chain_kernel_ok(chains, proj.num_channels):
            if _TRACE:
                _note("combined_kernel")
            n = self.text_embedder.linear_2.weight.shape[0]
            # A sinusoid chain always yields [1, n]; the projection chain yields
            # its own leading dims. The sum broadcasts, so the result shape is the
            # broadcast of the two -- not the projection chain's shape alone.
            out_shape = torch.broadcast_shapes((1, n), pooled_projection.shape[:-1] + (n,))
            return _chain_kernel(
                chains, out_shape, proj.num_channels, proj.flip_sin_to_cos,
                proj.downscale_freq_shift, proj.scale, 10000.0,
            )

        if _TRACE:
            _note("combined_torch")
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep + guidance encoding with pooled text projection.

    Adds a ``guidance_embedder`` on top of
    :class:`CombinedTimestepTextProjEmbeddings`.  Weight names match the
    diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        chains = [
            (_SINUSOID, timestep, self.timestep_embedder.linear_1, self.timestep_embedder.linear_2),
            (_SINUSOID, guidance, self.guidance_embedder.linear_1, self.guidance_embedder.linear_2),
            (_DIRECT, pooled_projection, self.text_embedder.linear_1, self.text_embedder.linear_2),
        ]
        proj = self.time_proj
        if pooled_projection.dtype is torch.bfloat16 and _chain_kernel_ok(chains, proj.num_channels):
            if _TRACE:
                _note("combined_kernel")
            n = self.text_embedder.linear_2.weight.shape[0]
            # As above: the chain sum broadcasts, so take the broadcast shape.
            out_shape = torch.broadcast_shapes((1, n), pooled_projection.shape[:-1] + (n,))
            return _chain_kernel(
                chains, out_shape, proj.num_channels, proj.flip_sin_to_cos,
                proj.downscale_freq_shift, proj.scale, 10000.0,
            )

        if _TRACE:
            _note("combined_torch")
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections
