// Fused timestep-embedding kernels for B200 / sm_100.
//
// Three entry points, all reached only after the Python-side predicate in
// ``timestep_embedding.py`` has admitted the call:
//
//   sinusoid_forward        one kernel, fp32 out    -- Timesteps
//   fused_mlp_forward       two kernels, bf16 out   -- TimestepEmbedding
//   fused_combined2_forward two kernels, bf16 out   -- CombinedTimestepTextProj
//   fused_combined3_forward two kernels, bf16 out   -- ...GuidanceTextProj
//
// The MLP entry points differ only in arity: each builds a ``BranchSet`` and
// calls the shared ``run_branches`` helper, so there is one device
// implementation behind all of them. Fixed arity rather than a generic
// descriptor list because the whole budget for the composite is tens of
// microseconds and pybind marshalling is visible at that scale.
//
// Numerics track the baseline term for term. The baseline is
//
//   emb   = timesteps[:, None].float() * exp(exponent)[None, :]   # fp32
//   emb   = scale * emb                                          # fp32
//   proj  = cat([sin(emb), cos(emb)], -1)                        # fp32
//   proj  = cat([proj[:, half:], proj[:, :half]], -1)            # flip_sin_to_cos
//   x     = proj.to(bfloat16)
//   h1    = F.linear(x, W1, b1)                                  # bf16 out
//   h     = F.silu(h1)                                           # bf16 out
//   part  = F.linear(h, W2, b2)                                  # bf16 out
//   out   = (part_0 + part_1) + part_2                           # bf16 adds
//
// so every rounding the baseline performs is performed here, in the same place:
// ``(t * f) * scale`` in that association, one round to bf16 after the layer-1
// bias epilogue, a second after SiLU, one per layer-2 epilogue, and a
// left-associative bf16 sum across branches. The two deliberate departures are
// selectable at runtime (``trig_mode``, ``numeric_flags``) so the rejection
// experiments measure them rather than assume them.

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
// 16 bytes of bfloat16: the widest access the hardware offers, and what every
// K % 8 == 0 predicate in the Python guard exists to make legal.
constexpr int kElemsPerWord = 8;
constexpr int kMaxBranches = 3;
constexpr int kAlignBytes = 16;
// The widest branch input the first kernel will stage in shared memory. 8192
// bf16 is 16 KiB, which leaves room for the reduction scratch inside the 48 KiB
// a kernel gets without opting in to more, and covers every K these classes are
// constructed with (256, 320, 768, 1280, 2816, 3072).
constexpr int64_t kMaxStagedK = 8192;

// Widths are cast to int for the grid arithmetic, so anything that could wrap the
// cast is declined rather than trusted. Reaching either bound needs a tensor far
// larger than any device, but "needs an exabyte" is not a guard.
constexpr int64_t kMaxWidth = 1 << 30;

// Trig evaluation forms. kTrigSinCos is what ships; the other two exist so
// tests/test_numerics.py can measure them against the fp32 tolerance
// instead of the design resting on an assumption about them.
constexpr int kTrigSinCos = 0;  // sincosf -- one range reduction for both halves
constexpr int kTrigSeparate = 1;  // sinf + cosf, exactly what ATen calls
constexpr int kTrigApprox = 2;  // __sinf + __cosf, rejected on the fp32 bound

// Deliberate numerical departures from the baseline, off in the shipped path.
constexpr int kFlagNoPreRound = 1;   // skip the bf16 round between linear_1 and SiLU
constexpr int kFlagFp32Reduce = 2;   // sum branches in fp32 instead of bf16
constexpr int kFlagApproxSilu = 4;   // the frozen L1 form; see silu_approx below
// Keep the inter-layer h in fp32 instead of rounding it to bf16, which is the experiment
// docs/plan.md's AC-9 names. It is *not* the same as kFlagNoPreRound: that one changes
// whether the layer-1 linear result is rounded before SiLU sees it, and still stores h as
// bf16. This one preserves that rounding exactly and changes only h's storage, so layer 2
// consumes fp32 activations against bf16 weights. Test-only; never selected by forward.
constexpr int kFlagFp32H = 8;

// What ATen's silu_kernel evaluates for a bfloat16 input: widen to the opmath type,
// compute x / (1 + exp(-x)) with a precise expf and an IEEE divide, round once on
// store. Reproduced here rather than routed through the L1 extension because this
// kernel consumes the value in registers, and a launch plus a round trip through
// global memory would cost more than the whole margin.
//
// This is the *exact* form, not the frozen candidate/L1/silu.py approximation. Round 0
// reproduced the approximation for fidelity to the composed candidate, and the round-0
// review showed why that was the wrong call: __expf(-x) overflows to inf for
// x <= -87.5, so __fdividef returns -0 where ATen still resolves a ~1e-37 result, and a
// second-layer weight near the bfloat16 maximum amplifies that to a visible -296
// against 0. The activation is ~9,216 evaluations against 56 MiB of weight traffic, so
// the exact form is free -- and the plan's own "Cannot use" list rules out
// "approximations chosen for their own sake where the exact form is free".
__device__ __forceinline__ float silu_exact(const float x) {
  return x / (1.0f + expf(-x));
}

// The frozen candidate/L1/silu.py form, kept only so
// tests/test_numerics.py can measure it against the baseline instead of the
// difference resting on an argument. Never selected by ``forward``.
__device__ __forceinline__ float silu_approx(const float x) {
  return __fdividef(x, 1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_dispatch(const float x, const int numeric_flags) {
  return (numeric_flags & kFlagApproxSilu) ? silu_approx(x) : silu_exact(x);
}

__device__ __forceinline__ void trig_pair(const float arg, const int mode, float* s, float* c) {
  if (mode == kTrigSeparate) {
    *s = sinf(arg);
    *c = cosf(arg);
  } else if (mode == kTrigApprox) {
    *s = __sinf(arg);
    *c = __cosf(arg);
  } else {
    sincosf(arg, s, c);
  }
}

// The reinterpretation between a 16-byte access and its eight elements, made
// explicit rather than left to a pointer cast at each use.
union Word {
  uint4 raw;
  __nv_bfloat16 elem[kElemsPerWord];
};

// A strided dot product over 16-byte words, with kUnroll independent loads in
// flight before any of them is consumed.
//
// The unroll depth is the lever the NCU report pointed at
// (profile/p4_ncu_layer2_v2/). The evidence is this kernel's own stall counters:
// long_scoreboard at 9.194 cycles per issued instruction against 1.301 for wait, with
// DRAM at 28.4% of peak, 15.69/16 sectors per global-load request and zero register
// spills -- memory-*latency* bound, with nothing left to win on access pattern. The
// per-PC pass then put 39.8% of those stalls on this function's scalar tail while it did
// 33% of the work, which is why the depth is derived from the words a warp actually
// handles (see choose_unroll2) rather than fixed.
//
// An earlier version of this comment claimed the three-branch composite keeps three times
// as many loads outstanding as the single-branch case. It does not: mlp_layer2_kernel
// finishes each branch's dot *and* its shuffle reduction before starting the next, so the
// outstanding count within a branch is identical. The stall counter is the evidence, not a
// bandwidth ratio between shapes.
//
// Accumulation order changes with kUnroll. That is inside one fp32 dot product,
// which no association here can match cuBLAS on anyway; the bf16 rounding
// boundaries either side of it are untouched.
template <int kUnroll>
__device__ __forceinline__ float dot_words(const uint4* __restrict__ a,
                                           const uint4* __restrict__ b, const int words,
                                           const int start, const int stride) {
  float acc = 0.0f;
  int i = start;
  for (; i + (kUnroll - 1) * stride < words; i += kUnroll * stride) {
    Word wa[kUnroll], wb[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      wa[u].raw = a[i + u * stride];
      wb[u].raw = b[i + u * stride];
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
#pragma unroll
      for (int e = 0; e < kElemsPerWord; ++e) {
        acc = fmaf(__bfloat162float(wa[u].elem[e]), __bfloat162float(wb[u].elem[e]), acc);
      }
    }
  }
  // The tail, for the words the unrolled groups did not cover.
  for (; i < words; i += stride) {
    Word wa, wb;
    wa.raw = a[i];
    wb.raw = b[i];
#pragma unroll
    for (int e = 0; e < kElemsPerWord; ++e) {
      acc = fmaf(__bfloat162float(wa.elem[e]), __bfloat162float(wb.elem[e]), acc);
    }
  }
  return acc;
}

// One branch of the fused MLP: h_b = silu(W1_b . x_b + b1_b), then
// part_b = W2_b . h_b + b2_b.
//
// A branch is either *dense* -- ``x`` points at a bf16 activation the caller
// supplies -- or *sinusoid* -- ``x`` is null and the input is recomputed inside
// the kernel from ``freq`` and the device-resident scalar at ``tptr``. The
// scalar has to be read on the device: reading it on the host would mean a
// synchronisation, and the whole point of this file is to not pay for one.
struct Branch {
  const __nv_bfloat16* x;
  const __nv_bfloat16* tptr;
  const float* freq;
  const __nv_bfloat16* w1;
  const __nv_bfloat16* b1;
  const __nv_bfloat16* w2;
  const __nv_bfloat16* b2;
  int k;      // input width
  int half;   // k / 2; sinusoid branches only
  float scale;
  int flip;
};

// Kernel parameters cannot be a bare array, so the set travels as a struct.
struct BranchSet {
  Branch b[kMaxBranches];
};

// How the inter-layer scratch is stored and read back. bf16 is the shipped path -- it is
// what F.silu's store does; float exists only for the AC-9 rejection experiment.
template <typename HT>
struct StoreH;

template <>
struct StoreH<__nv_bfloat16> {
  __device__ __forceinline__ static __nv_bfloat16 from_f32(const float v) {
    return __float2bfloat16(v);
  }
  __device__ __forceinline__ static float to_f32(const __nv_bfloat16 v) {
    return __bfloat162float(v);
  }
};

template <>
struct StoreH<float> {
  __device__ __forceinline__ static float from_f32(const float v) { return v; }
  __device__ __forceinline__ static float to_f32(const float v) { return v; }
};

// ---------------------------------------------------------------------------
// Timesteps: one kernel, fp32 in and out.
// ---------------------------------------------------------------------------
// out[row, j] and out[row, half + j] are written by the same thread, so neither
// concatenation in the baseline's expression is materialised and the flip is a
// choice of which of the pair goes where. Getting that choice backwards yields a
// matched_ratio near 0.5, which looks plausible and fails.
__global__ void sinusoid_kernel(const __nv_bfloat16* __restrict__ t,
                                const float* __restrict__ freq,
                                float* __restrict__ out, const int64_t total, const int half,
                                const float scale, const int flip, const int trig_mode) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int64_t row = idx / half;
  const int j = static_cast<int>(idx - row * half);
  const float arg = (__bfloat162float(t[row]) * freq[j]) * scale;
  float s, c;
  trig_pair(arg, trig_mode, &s, &c);
  float* o = out + row * (2 * half);
  o[j] = flip ? c : s;
  o[half + j] = flip ? s : c;
}

// ---------------------------------------------------------------------------
// Fused MLP, layer 1: h[branch, row] = silu(W1 . x + b1), bf16.
// ---------------------------------------------------------------------------
// Staging the branch input in shared memory serves two purposes: a dense input
// is read once instead of once per row in the block, and a sinusoid input is
// *computed* once per block instead of once per row. For the composite that is
// ``half`` sincosf evaluations per block rather than ``half * rows_per_block``.
__device__ __forceinline__ void stage_branch_input(const Branch& br,
                                                   __nv_bfloat16* __restrict__ x_s,
                                                   const int trig_mode) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  if (br.x != nullptr) {
    const uint4* __restrict__ src = reinterpret_cast<const uint4*>(br.x);
    uint4* __restrict__ dst = reinterpret_cast<uint4*>(x_s);
    const int words = br.k / kElemsPerWord;
    for (int i = tid; i < words; i += nthreads) {
      dst[i] = src[i];
    }
    return;
  }
  // Every thread reads the same scalar: a broadcast, not ``nthreads`` loads.
  const float t = __bfloat162float(*br.tptr);
  const int half = br.half;
  for (int j = tid; j < half; j += nthreads) {
    const float arg = (t * br.freq[j]) * br.scale;
    float s, c;
    trig_pair(arg, trig_mode, &s, &c);
    // .to(bfloat16) on the fp32 projection, fused into where the value is born.
    x_s[j] = __float2bfloat16(br.flip ? c : s);
    x_s[half + j] = __float2bfloat16(br.flip ? s : c);
  }
}

template <typename HT>
__global__ void mlp_layer1_kernel(const BranchSet bs, HT* __restrict__ h,
                                  const int n_rows, const int wpr, const int rpb,
                                  const int trig_mode, const int numeric_flags) {
  const Branch br = bs.b[blockIdx.y];

  extern __shared__ char smem_raw[];
  __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  // k % 8 == 0 keeps this offset a multiple of 16, so the float array below is
  // aligned without padding.
  float* red = reinterpret_cast<float*>(smem_raw + static_cast<size_t>(br.k) * sizeof(__nv_bfloat16));

  stage_branch_input(br, x_s, trig_mode);
  __syncthreads();

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int row_local = warp / wpr;
  const int w = warp % wpr;
  const int row = blockIdx.x * rpb + row_local;
  const int words = br.k / kElemsPerWord;

  float acc = 0.0f;
  if (row < n_rows) {
    const uint4* __restrict__ wrow =
        reinterpret_cast<const uint4*>(br.w1 + static_cast<int64_t>(row) * br.k);
    const uint4* __restrict__ xw = reinterpret_cast<const uint4*>(x_s);
    // Each warp takes whole 512-byte runs, striding by wpr runs, so the reads
    // stay coalesced however the row is split.
    for (int i = w * kWarpSize + lane; i < words; i += wpr * kWarpSize) {
      Word a, b;
      a.raw = wrow[i];
      b.raw = xw[i];
#pragma unroll
      for (int e = 0; e < kElemsPerWord; ++e) {
        acc = fmaf(__bfloat162float(a.elem[e]), __bfloat162float(b.elem[e]), acc);
      }
    }
  }
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) {
    red[row_local * wpr + w] = acc;
  }
  __syncthreads();

  if (static_cast<int>(threadIdx.x) < rpb) {
    const int orow = blockIdx.x * rpb + static_cast<int>(threadIdx.x);
    if (orow < n_rows) {
      float s = 0.0f;
      for (int i = 0; i < wpr; ++i) {
        s += red[static_cast<int>(threadIdx.x) * wpr + i];
      }
      // F.linear widens the bf16 bias into its fp32 epilogue and rounds once on
      // store; SiLU then reads that bf16 value back. Keeping the intermediate in
      // fp32 instead would be arithmetically better and numerically wrong.
      s += __bfloat162float(br.b1[orow]);
      const float pre = (numeric_flags & kFlagNoPreRound)
                            ? s
                            : __bfloat162float(__float2bfloat16(s));
      const float activated = silu_dispatch(pre, numeric_flags);
      // The shipped path rounds here, which is what F.silu's bf16 store does. The fp32-h
      // experiment keeps the value wide instead, and changes nothing before this line.
      h[static_cast<int64_t>(blockIdx.y) * n_rows + orow] = StoreH<HT>::from_f32(activated);
    }
  }
}

// ---------------------------------------------------------------------------
// Prologue variant (task23). Not the shipped path.
// ---------------------------------------------------------------------------
// The alternative to recomputing a sinusoid branch's input inside every block of the
// first kernel: materialise it once here, as bf16 rows, and hand the result to the
// same two MLP kernels as an ordinary dense branch. That costs one extra launch and
// one extra round trip through global memory, and saves (blocks - 1) x half sincosf
// evaluations per sinusoid branch.
//
// docs/plan.md argues per-block recomputation should win -- roughly 300,000 sincosf
// against a guaranteed ~2.05 us for the launch -- but calls that an estimate and
// requires the comparison. This is the other arm of it; the measurement is in
// profile/p2_geometry/prologue_vs_recompute.log.
//
// One block per (row, branch): grid.y indexes the branch so the two projections of the
// guidance composite are written by one launch.
__global__ void sinusoid_prologue_kernel(const __nv_bfloat16* __restrict__ t0,
                                         const __nv_bfloat16* __restrict__ t1,
                                         const float* __restrict__ freq,
                                         __nv_bfloat16* __restrict__ out, const int half,
                                         const float scale, const int flip,
                                         const int trig_mode) {
  const __nv_bfloat16* src = (blockIdx.y == 0) ? t0 : t1;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= half) {
    return;
  }
  const float t = __bfloat162float(*src);
  const float arg = (t * freq[j]) * scale;
  float s, c;
  trig_pair(arg, trig_mode, &s, &c);
  __nv_bfloat16* row = out + static_cast<int64_t>(blockIdx.y) * (2 * half);
  row[j] = __float2bfloat16(flip ? c : s);
  row[half + j] = __float2bfloat16(flip ? s : c);
}

// ---------------------------------------------------------------------------
// Activation probe. Not on any scored path: it exists so
// tests/test_numerics.py can compare silu_dispatch against F.silu over every
// bfloat16 encoding directly, rather than inferring the activation's behaviour from
// a GEMV whose dot product has its own rounding.
// ---------------------------------------------------------------------------
__global__ void silu_probe_kernel(const __nv_bfloat16* __restrict__ in,
                                  __nv_bfloat16* __restrict__ out, const int64_t n,
                                  const int numeric_flags) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  // The same widen / evaluate / round-once sequence layer 1 performs.
  out[i] = __float2bfloat16(silu_dispatch(__bfloat162float(in[i]), numeric_flags));
}

// ---------------------------------------------------------------------------
// Fused MLP, layer 2: out[row] = sum_b bf16(W2_b . h_b + b2_b), bf16 adds.
// ---------------------------------------------------------------------------
// h is at most 18 KiB and was written by the kernel immediately before this
// one, so it is read from L2 rather than staged again: each warp touches only
// its own 1.5 KiB slice of it, and the traffic that matters here is W2.
// The fp32-h arm reads h element-wise rather than through a 16-byte word: eight fp32
// activations are 32 bytes, so the vectorised pairing with a bf16 weight word no longer
// holds. It is deliberately the slow path -- this exists to be measured and rejected, not
// to be fast.
template <int kUnroll, typename HT>
__device__ __forceinline__ float dot_row(const uint4* __restrict__ wrow,
                                         const HT* __restrict__ hv, const int words,
                                         const int start, const int stride) {
  if constexpr (sizeof(HT) == sizeof(__nv_bfloat16)) {
    return dot_words<kUnroll>(wrow, reinterpret_cast<const uint4*>(hv), words, start,
                              stride);
  } else {
    float acc = 0.0f;
    for (int i = start; i < words; i += stride) {
      Word wa;
      wa.raw = wrow[i];
#pragma unroll
      for (int e = 0; e < kElemsPerWord; ++e) {
        acc = fmaf(__bfloat162float(wa.elem[e]),
                   StoreH<HT>::to_f32(hv[i * kElemsPerWord + e]), acc);
      }
    }
    return acc;
  }
}

template <int kUnroll, typename HT>
__global__ void mlp_layer2_kernel(const BranchSet bs, const HT* __restrict__ h,
                                  __nv_bfloat16* __restrict__ out, const int n_rows,
                                  const int k2, const int n_branch, const int wpr,
                                  const int rpb, const int numeric_flags) {
  extern __shared__ char smem_raw[];
  float* red2 = reinterpret_cast<float*>(smem_raw);

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int row_local = warp / wpr;
  const int w = warp % wpr;
  const int row = blockIdx.x * rpb + row_local;
  const int words = k2 / kElemsPerWord;

  for (int b = 0; b < n_branch; ++b) {
    float acc = 0.0f;
    if (row < n_rows) {
      const uint4* __restrict__ wrow =
          reinterpret_cast<const uint4*>(bs.b[b].w2 + static_cast<int64_t>(row) * k2);
      const HT* __restrict__ hv = h + static_cast<int64_t>(b) * k2;
      acc = dot_row<kUnroll, HT>(wrow, hv, words, w * kWarpSize + lane, wpr * kWarpSize);
    }
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
      red2[(b * rpb + row_local) * wpr + w] = acc;
    }
  }
  __syncthreads();

  if (static_cast<int>(threadIdx.x) < rpb) {
    const int orow = blockIdx.x * rpb + static_cast<int>(threadIdx.x);
    if (orow < n_rows) {
      const int slot = static_cast<int>(threadIdx.x);
      if (numeric_flags & kFlagFp32Reduce) {
        float total = 0.0f;
        for (int b = 0; b < n_branch; ++b) {
          float acc = 0.0f;
          for (int i = 0; i < wpr; ++i) {
            acc += red2[(b * rpb + slot) * wpr + i];
          }
          total += acc + __bfloat162float(bs.b[b].b2[orow]);
        }
        out[orow] = __float2bfloat16(total);
        return;
      }
      // The baseline's association: each addend is already a bf16 tensor from
      // its own F.linear, and ``a + b + c`` groups left to right with each add
      // evaluated in fp32 and rounded back to bf16.
      __nv_bfloat16 total = __float2bfloat16(0.0f);
      for (int b = 0; b < n_branch; ++b) {
        float acc = 0.0f;
        for (int i = 0; i < wpr; ++i) {
          acc += red2[(b * rpb + slot) * wpr + i];
        }
        acc += __bfloat162float(bs.b[b].b2[orow]);
        const __nv_bfloat16 part = __float2bfloat16(acc);
        total = (b == 0) ? part
                         : __float2bfloat16(__bfloat162float(total) + __bfloat162float(part));
      }
      out[orow] = total;
    }
  }
}

// ---------------------------------------------------------------------------
// Host side: the admission predicate.
// ---------------------------------------------------------------------------
// This lives here rather than in Python because it is on the critical path. The
// TimestepEmbedding cases are DRAM-bound -- 26.6 us of cold device time against
// the baseline's 28.8 (profile/p2_geometry/cold_probe.log) -- so the entire
// margin is the host-visible part of the harness window, and an equivalent
// Python predicate measured 3.3 us of it
// (profile/p2_geometry/host_probe.log) against the baseline's 0.93 us of total
// host-visible excess. In C++ the same checks are free.
//
// Nothing below touches device memory or the CUDA API: only scalar_type, device,
// sizes, strides, numel, the data pointer *value*, and the thread-local grad
// mode. There is no launch, no allocation and no synchronisation, so the
// predicate cannot perturb what the harness is timing.
//
// Declining is reported in band: every entry point returns an undefined
// at::Tensor, which arrives in Python as None, and the caller runs the baseline
// body. A TORCH_CHECK would turn an unsupported layout into a failure instead of
// a fallback.
bool aligned16(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % kAlignBytes == 0;
}

// at::GradMode covers reverse mode only: torch.no_grad() does not disable
// forward-mode AD, so a dual tensor would otherwise be admitted and the kernel
// would return a primal with its tangent silently dropped. Raised by the task25
// review (docs/reviews/task25_codex_review.md).
bool no_dual(const at::Tensor& t) {
  return !t._fw_grad(/*level=*/0).defined();
}

// The four parameters of one branch, against what the kernels actually assume.
// On success writes the output width and the input width.
bool branch_ok(const at::Tensor& w1, const at::Tensor& b1, const at::Tensor& w2,
               const at::Tensor& b2, const at::Device& dev, int64_t* n_out, int64_t* k_out) {
  if (w1.scalar_type() != at::kBFloat16 || b1.scalar_type() != at::kBFloat16 ||
      w2.scalar_type() != at::kBFloat16 || b2.scalar_type() != at::kBFloat16) {
    return false;
  }
  if (w1.dim() != 2 || w2.dim() != 2 || b1.dim() != 1 || b2.dim() != 1) {
    return false;
  }
  const int64_t n = w1.size(0);
  const int64_t k = w1.size(1);
  if (k <= 0 || n <= 0 || k % kElemsPerWord != 0 || n % kElemsPerWord != 0 ||
      k > kMaxStagedK || n > kMaxWidth) {
    return false;
  }
  // The second layer must be square: its dot length is time_embed_dim, which is
  // also its row count, and the second kernel indexes h on that assumption, so a
  // non-square w2 would walk off the end of the scratch rather than produce a
  // wrong number.
  if (w2.size(0) != n || w2.size(1) != n) {
    return false;
  }
  if (w1.stride(1) != 1 || w1.stride(0) != k || w2.stride(1) != 1 || w2.stride(0) != n) {
    return false;
  }
  if (b1.size(0) != n || b2.size(0) != n || b1.stride(0) != 1 || b2.stride(0) != 1) {
    return false;
  }
  if (w1.device() != dev || w2.device() != dev || b1.device() != dev || b2.device() != dev) {
    return false;
  }
  // 16-byte vectorised loads need 16-byte-aligned row bases; k % 8 == 0 carries
  // that from the base to every row. The biases are read one element at a time
  // and so carry no alignment requirement.
  if (!aligned16(w1.const_data_ptr()) || !aligned16(w2.const_data_ptr())) {
    return false;
  }
  if (!no_dual(w1) || !no_dual(b1) || !no_dual(w2) || !no_dual(b2)) {
    return false;
  }
  *n_out = n;
  *k_out = k;
  return true;
}

// A contiguous bf16 activation of width k at batch one.
bool activation_ok(const at::Tensor& x, const int64_t k, const at::Device& dev,
                   const int min_dim) {
  // min_dim is 1 for a bare TimestepEmbedding, whose baseline returns [N] for a [K]
  // input, and 2 for the composites, whose baseline adds a [1, N] sinusoid result to
  // the [N] text result and broadcasts to [1, N] -- a shape the fused path, which
  // derives its output shape from the dense input alone, cannot produce. Raised by
  // the task25 review (docs/reviews/task25_codex_review.md).
  return x.scalar_type() == at::kBFloat16 && x.device() == dev && x.is_contiguous() &&
         x.dim() >= min_dim && x.numel() == k && x.size(-1) == k &&
         aligned16(x.const_data_ptr()) && no_dual(x);
}

// A single bf16 timestep, read on the device rather than the host: reading it
// here would mean a synchronisation, which is the one thing this file exists to
// avoid, so the batch is pinned to one.
bool scalar_ok(const at::Tensor& t, const at::Device& dev) {
  return t.scalar_type() == at::kBFloat16 && t.device() == dev && t.dim() == 1 &&
         t.numel() == 1 && t.is_contiguous() && aligned16(t.const_data_ptr()) && no_dual(t);
}

bool freq_ok(const at::Tensor& freq, const int64_t num_channels, const at::Device& dev) {
  return freq.defined() && freq.scalar_type() == at::kFloat && freq.device() == dev &&
         freq.dim() == 1 && freq.numel() == num_channels / 2 && freq.is_contiguous();
}

// Timesteps admits any batch, so the scalar check is relaxed to "1-D bf16" --
// the sinusoid kernel indexes the timestep per row and needs no host-side read.
bool scalar_ok_any(const at::Tensor& t) {
  return t.scalar_type() == at::kBFloat16 && t.dim() == 1 && t.is_contiguous() &&
         aligned16(t.const_data_ptr()) && no_dual(t);
}

bool sinusoid_ok(const at::Tensor& t, const at::Tensor& freq, const int64_t num_channels) {
  if (at::GradMode::is_enabled() || !t.is_cuda() || num_channels <= 0 ||
      num_channels % 2 != 0 || num_channels > kMaxWidth) {
    return false;
  }
  return scalar_ok_any(t) && freq_ok(freq, num_channels, t.device());
}

// ---------------------------------------------------------------------------
// Kernel geometry.
// ---------------------------------------------------------------------------
// (warps_per_row, rows_per_block) for each of the two MLP kernels. Winners of the
// interleaved paired comparison in profile/p2_geometry/ -- see that directory's
// REPORT.md for the measured spread. The branch count belongs in the key: the
// second kernel walks every branch inside one block, so a three-branch composite
// at (768, 3072) asks three times the bytes per block that a single-branch
// TimestepEmbedding at the same (k, n) does, and the two do not want the same
// split.
struct Geom {
  int wpr1, rpb1, wpr2, rpb2, unroll2;
};

// The unroll depths compiled into launch_layer2, ascending.
constexpr int kUnrollChoices[] = {1, 2, 3, 4, 6, 8};

// The deepest compiled unroll that still fits the work a single warp has to do, so the
// unrolled body runs at least once and the tail stays short.
//
// This is not a tuning constant, it is arithmetic: warp w handles
// ``ceil(words / (wpr2 * 32))`` sixteen-byte words, and dot_words only enters its
// unrolled body when ``i + (kUnroll - 1) * stride < words``. Ask for a depth deeper than
// the words available and the unrolled body never executes -- every word goes through
// the scalar tail, which has no instruction-level parallelism at all. At n = 3072 with
// wpr2 = 2 a warp gets 384 / 64 = 6 words, so 6 is both the deepest and the only
// tail-free choice; at n = 1280 it gets 2.5, so 6 would degenerate to a pure tail and 2
// is right. profile/p4_ncu_layer2_v2 measured the tail carrying 39.8% of the kernel's
// long-scoreboard stalls while being 33% of the work, which is what made this worth
// getting right rather than fixing at one value.
int choose_unroll2(const int64_t n, const int wpr2) {
  const int64_t words = n / kElemsPerWord;
  const int64_t per_warp = words / (static_cast<int64_t>(wpr2) * kWarpSize);
  int best = 1;
  for (const int u : kUnrollChoices) {
    if (u <= per_warp) {
      best = u;
    }
  }
  return best;
}

Geom choose_geometry(const int64_t k, const int64_t n, const int n_branch) {
  (void)k;
  (void)n_branch;
  // Three axes, all settled by the interleaved paired comparison in
  // profile/p2_geometry/, and every case and branch count agreed on the same (wpr, rpb)
  // point -- so there is no per-shape table for those. The same split ships everywhere.
  //
  // wpr2 = 2 is worth a full timer quantum over wpr2 = 1: paired_sweep_te.log has
  // [1,256] at 0.999x against 0.911x, and paired_sweep_comp.log has the composite at
  // 6.605x against 5.606x. Above 2 it falls away again as the per-warp slice stops being
  // long enough to keep loads in flight.
  //
  // rpb2 = 2 is the middle of a plateau that runs from 1 to 8 and costs a quantum at 16.
  // Same for the first kernel's (1, 8): wpr1 in {1,2} with rpb1 in {8,16} all measure
  // within a quantum (paired_sweep_k1.log), and wpr1 = 4 costs one.
  //
  // The unroll depth is derived, not tabulated; see choose_unroll2. At the scored shapes
  // that yields 6, which paired_sweep_unroll2.log measured a quantum ahead of 4 on the
  // composite (53.28 us against 55.30) and level with it elsewhere.
  constexpr int kWpr2 = 2;
  return Geom{1, 8, kWpr2, 2, choose_unroll2(n, kWpr2)};
}

// A non-positive override component means "use the measured table"; the sweep in
// profile/p2_geometry/ passes explicit values.
Geom resolve_geometry(const int64_t k, const int64_t n, const int n_branch, const int64_t wpr1,
                      const int64_t rpb1, const int64_t wpr2, const int64_t rpb2,
                      const int64_t unroll2) {
  if (wpr1 <= 0 || rpb1 <= 0 || wpr2 <= 0 || rpb2 <= 0 || unroll2 <= 0) {
    return choose_geometry(k, n, n_branch);
  }
  return Geom{static_cast<int>(wpr1), static_cast<int>(rpb1), static_cast<int>(wpr2),
              static_cast<int>(rpb2), static_cast<int>(unroll2)};
}

// ---------------------------------------------------------------------------
// Host side: launching.
// ---------------------------------------------------------------------------
const __nv_bfloat16* bf16_ptr(const at::Tensor& t) {
  return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());
}

Branch make_branch(const at::Tensor& w1, const at::Tensor& b1, const at::Tensor& w2,
                   const at::Tensor& b2) {
  Branch br{};
  br.x = nullptr;
  br.tptr = nullptr;
  br.freq = nullptr;
  br.w1 = bf16_ptr(w1);
  br.b1 = bf16_ptr(b1);
  br.w2 = bf16_ptr(w2);
  br.b2 = bf16_ptr(b2);
  br.k = static_cast<int>(w1.size(1));
  br.half = 0;
  br.scale = 1.0f;
  br.flip = 0;
  return br;
}

Branch make_dense_branch(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1,
                         const at::Tensor& w2, const at::Tensor& b2) {
  Branch br = make_branch(w1, b1, w2, b2);
  br.x = bf16_ptr(x);
  return br;
}

Branch make_sinusoid_branch(const at::Tensor& t, const at::Tensor& freq, const int num_channels,
                            const bool flip, const double scale, const at::Tensor& w1,
                            const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2) {
  Branch br = make_branch(w1, b1, w2, b2);
  br.tptr = bf16_ptr(t);
  br.freq = freq.const_data_ptr<float>();
  br.half = num_channels / 2;
  br.scale = static_cast<float>(scale);
  br.flip = flip ? 1 : 0;
  return br;
}

// Only these depths are compiled; anything else falls back to 1 rather than
// silently rounding to a neighbour.
template <int kUnroll, typename HT>
void launch_layer2(const BranchSet& bs, const at::Tensor& h, at::Tensor& out, const int n_rows,
                   const int k2, const int n_branch, const int wpr2, const int rpb2,
                   const int numeric_flags, cudaStream_t stream) {
  const int threads = kWarpSize * wpr2 * rpb2;
  const int blocks = (n_rows + rpb2 - 1) / rpb2;
  const size_t smem = static_cast<size_t>(n_branch) * rpb2 * wpr2 * sizeof(float);
  mlp_layer2_kernel<kUnroll, HT><<<static_cast<unsigned>(blocks), threads, smem, stream>>>(
      bs, static_cast<const HT*>(h.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n_rows, k2, n_branch, wpr2, rpb2,
      numeric_flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename HT>
void dispatch_layer2(const int unroll2, const BranchSet& bs, const at::Tensor& h,
                     at::Tensor& out, const int n_rows, const int k2, const int n_branch,
                     const int wpr2, const int rpb2, const int numeric_flags,
                     cudaStream_t stream) {
  switch (unroll2) {
    case 8:
      launch_layer2<8, HT>(bs, h, out, n_rows, k2, n_branch, wpr2, rpb2, numeric_flags,
                           stream);
      break;
    case 6:
      launch_layer2<6, HT>(bs, h, out, n_rows, k2, n_branch, wpr2, rpb2, numeric_flags,
                           stream);
      break;
    case 4:
      launch_layer2<4, HT>(bs, h, out, n_rows, k2, n_branch, wpr2, rpb2, numeric_flags,
                           stream);
      break;
    case 3:
      launch_layer2<3, HT>(bs, h, out, n_rows, k2, n_branch, wpr2, rpb2, numeric_flags,
                           stream);
      break;
    case 2:
      launch_layer2<2, HT>(bs, h, out, n_rows, k2, n_branch, wpr2, rpb2, numeric_flags,
                           stream);
      break;
    default:
      launch_layer2<1, HT>(bs, h, out, n_rows, k2, n_branch, wpr2, rpb2, numeric_flags,
                           stream);
      break;
  }
}

at::Tensor run_branches(const std::vector<Branch>& branches, const int n_rows, const int k2,
                        const std::vector<int64_t>& out_shape, const at::TensorOptions& opts,
                        const int wpr1, const int rpb1, const int wpr2, const int rpb2,
                        const int unroll2, const int trig_mode, const int numeric_flags,
                        const bool use_prologue) {
  const int n_branch = static_cast<int>(branches.size());
  TORCH_CHECK(n_branch >= 1 && n_branch <= kMaxBranches, "branch count out of range");
  TORCH_CHECK(n_rows == k2, "linear_2 must be square: n_rows ", n_rows, " != k2 ", k2);
  TORCH_CHECK(wpr1 >= 1 && rpb1 >= 1 && wpr2 >= 1 && rpb2 >= 1, "geometry must be positive");
  TORCH_CHECK(kWarpSize * wpr1 * rpb1 <= 1024 && kWarpSize * wpr2 * rpb2 <= 1024,
              "geometry exceeds the 1024-thread block limit");

  BranchSet bs{};
  int max_k = 0;
  for (int i = 0; i < n_branch; ++i) {
    bs.b[i] = branches[i];
    max_k = std::max(max_k, branches[i].k);
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  // Both allocations happen here rather than through a Python-level
  // torch.empty: the caching allocator makes them free of a cudaMalloc, and
  // keeping them on this side of pybind is what holds the launch count at two.
  // Their marginal cost is measured in profile/p1_calibration/alloc_calib.log.
  // bf16 h is the shipped path; the fp32 arm exists for the AC-9 experiment only.
  const bool fp32_h = (numeric_flags & kFlagFp32H) != 0;
  auto h = at::empty({n_branch, k2}, fp32_h ? opts.dtype(at::kFloat) : opts);
  auto out = at::empty(out_shape, opts);

  // The task23 prologue arm: materialise every sinusoid branch's input up front and
  // rewrite those branches as dense, so the first kernel stages a row instead of
  // recomputing one. Costs one launch. Never taken by the shipped path.
  at::Tensor proj;
  std::vector<Branch> effective = branches;
  const int n_sinusoid = n_branch - 1;
  if (use_prologue && n_sinusoid > 0) {
    const int half = branches[0].half;
    proj = at::empty({n_sinusoid, 2 * half}, opts);
    auto* proj_ptr = reinterpret_cast<__nv_bfloat16*>(proj.data_ptr());
    constexpr int kBlock = 128;
    const dim3 grid(static_cast<unsigned>((half + kBlock - 1) / kBlock),
                    static_cast<unsigned>(n_sinusoid));
    sinusoid_prologue_kernel<<<grid, kBlock, 0, stream>>>(
        branches[0].tptr, n_sinusoid > 1 ? branches[1].tptr : branches[0].tptr,
        branches[0].freq, proj_ptr, half, branches[0].scale, branches[0].flip,
        trig_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int b = 0; b < n_sinusoid; ++b) {
      effective[b].x = proj_ptr + static_cast<int64_t>(b) * (2 * half);
      effective[b].tptr = nullptr;
    }
    for (int i = 0; i < n_branch; ++i) {
      bs.b[i] = effective[i];
    }
  }

  {
    const int threads = kWarpSize * wpr1 * rpb1;
    const int blocks = (n_rows + rpb1 - 1) / rpb1;
    const size_t smem = static_cast<size_t>(max_k) * sizeof(__nv_bfloat16) +
                        static_cast<size_t>(rpb1) * wpr1 * sizeof(float);
    const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(n_branch));
    if (fp32_h) {
      mlp_layer1_kernel<float><<<grid, threads, smem, stream>>>(
          bs, h.data_ptr<float>(), n_rows, wpr1, rpb1, trig_mode, numeric_flags);
    } else {
      mlp_layer1_kernel<__nv_bfloat16><<<grid, threads, smem, stream>>>(
          bs, reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), n_rows, wpr1, rpb1, trig_mode,
          numeric_flags);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (fp32_h) {
    dispatch_layer2<float>(unroll2, bs, h, out, n_rows, k2, n_branch, wpr2, rpb2,
                           numeric_flags, stream);
  } else {
    dispatch_layer2<__nv_bfloat16>(unroll2, bs, h, out, n_rows, k2, n_branch, wpr2, rpb2,
                                   numeric_flags, stream);
  }
  return out;
}

std::vector<int64_t> replace_last(const at::Tensor& x, const int64_t n) {
  std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());
  shape.back() = n;
  return shape;
}

}  // namespace

at::Tensor sinusoid_forward(const at::Tensor& t, const at::Tensor& freq,
                            const int64_t num_channels, const bool flip, const double scale,
                            const int64_t block, const int64_t trig_mode) {
  if (!sinusoid_ok(t, freq, num_channels)) {
    return at::Tensor();
  }
  TORCH_CHECK(block > 0 && block <= 1024 && block % kWarpSize == 0,
              "block must be a positive multiple of 32 up to 1024, got ", block);
  const at::cuda::CUDAGuard guard(t.device());
  const int half = static_cast<int>(num_channels) / 2;
  const int64_t rows = t.numel();
  auto out = at::empty({rows, num_channels}, t.options().dtype(at::kFloat));
  const int64_t total = rows * half;
  if (total == 0) {
    return out;  // a zero-block grid is an invalid launch configuration
  }
  const int64_t blocks = (total + block - 1) / block;
  sinusoid_kernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(block), 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      bf16_ptr(t), freq.const_data_ptr<float>(), out.data_ptr<float>(), total, half,
      static_cast<float>(scale), flip ? 1 : 0, static_cast<int>(trig_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

bool sinusoid_admits(const at::Tensor& t, const at::Tensor& freq, const int64_t num_channels) {
  return sinusoid_ok(t, freq, num_channels);
}

namespace {

// Shared predicate for the three MLP entry points. ``groups`` holds
// (w1, b1, w2, b2) per branch in the baseline's summation order, sinusoid
// branches first, so the dense branch -- the one whose width the activation must
// match -- is always last and fixes n for the rest.
//
// Split out from the launch so ``*_admits`` can ask the question without running
// anything, which is what the guard tests need.
bool mlp_plan(const std::vector<const at::Tensor*>& groups, const at::Tensor& x,
              const at::Tensor* t, const at::Tensor* g, const at::Tensor* freq,
              const int64_t num_channels, int64_t* n_out, int64_t* k_out) {
  const int n_branch = static_cast<int>(groups.size()) / 4;
  if (n_branch < 1 || n_branch > kMaxBranches) {
    return false;
  }
  if (at::GradMode::is_enabled() || !x.is_cuda()) {
    return false;
  }
  const at::Device dev = x.device();

  const int dense = (n_branch - 1) * 4;
  int64_t n = 0, k_p = 0;
  if (!branch_ok(*groups[dense], *groups[dense + 1], *groups[dense + 2], *groups[dense + 3],
                 dev, &n, &k_p)) {
    return false;
  }
  if (!activation_ok(x, k_p, dev, n_branch > 1 ? 2 : 1)) {
    return false;
  }
  for (int b = 0; b < n_branch - 1; ++b) {
    int64_t nb = 0, kb = 0;
    if (!branch_ok(*groups[b * 4], *groups[b * 4 + 1], *groups[b * 4 + 2], *groups[b * 4 + 3],
                   dev, &nb, &kb)) {
      return false;
    }
    // Every embedder feeds the same sum, so they must agree on n; and a sinusoid
    // branch's input *is* the projection, so its in_channels must be
    // num_channels or the staged row would be the wrong length.
    if (nb != n || kb != num_channels) {
      return false;
    }
  }
  if (n_branch > 1) {
    if (num_channels <= 0 || num_channels % 2 != 0 || freq == nullptr ||
        !freq_ok(*freq, num_channels, dev)) {
      return false;
    }
    if (t == nullptr || !scalar_ok(*t, dev)) {
      return false;
    }
    if (n_branch > 2 && (g == nullptr || !scalar_ok(*g, dev))) {
      return false;
    }
  }
  *n_out = n;
  *k_out = k_p;
  return true;
}

at::Tensor mlp_dispatch(const std::vector<const at::Tensor*>& groups, const at::Tensor& x,
                        const at::Tensor* t, const at::Tensor* g, const at::Tensor* freq,
                        const int64_t num_channels, const bool flip, const double scale,
                        const int64_t wpr1, const int64_t rpb1, const int64_t wpr2,
                        const int64_t rpb2, const int64_t unroll2, const int64_t trig_mode,
                        const int64_t numeric_flags, const bool use_prologue) {
  int64_t n = 0, k_p = 0;
  if (!mlp_plan(groups, x, t, g, freq, num_channels, &n, &k_p)) {
    return at::Tensor();
  }
  const int n_branch = static_cast<int>(groups.size()) / 4;
  const int dense = (n_branch - 1) * 4;

  std::vector<Branch> branches;
  branches.reserve(n_branch);
  for (int b = 0; b < n_branch - 1; ++b) {
    branches.push_back(make_sinusoid_branch(*(b == 0 ? t : g), *freq,
                                            static_cast<int>(num_channels), flip, scale,
                                            *groups[b * 4], *groups[b * 4 + 1],
                                            *groups[b * 4 + 2], *groups[b * 4 + 3]));
  }
  branches.push_back(make_dense_branch(x, *groups[dense], *groups[dense + 1],
                                       *groups[dense + 2], *groups[dense + 3]));

  const Geom geom = resolve_geometry(k_p, n, n_branch, wpr1, rpb1, wpr2, rpb2, unroll2);
  const at::cuda::CUDAGuard guard(x.device());
  return run_branches(branches, static_cast<int>(n), static_cast<int>(n), replace_last(x, n),
                      x.options(), geom.wpr1, geom.rpb1, geom.wpr2, geom.rpb2, geom.unroll2,
                      static_cast<int>(trig_mode), static_cast<int>(numeric_flags),
                      use_prologue);
}

}  // namespace

at::Tensor fused_mlp_forward(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1,
                             const at::Tensor& w2, const at::Tensor& b2, const int64_t wpr1,
                             const int64_t rpb1, const int64_t wpr2, const int64_t rpb2,
                             const int64_t unroll2, const int64_t numeric_flags) {
  return mlp_dispatch({&w1, &b1, &w2, &b2}, x, nullptr, nullptr, nullptr, 0, false, 1.0, wpr1,
                      rpb1, wpr2, rpb2, unroll2, kTrigSinCos, numeric_flags,
                      /*use_prologue=*/false);
}

at::Tensor fused_combined2_forward(const at::Tensor& t, const at::Tensor& p,
                                   const at::Tensor& freq, const int64_t num_channels,
                                   const bool flip, const double scale, const at::Tensor& tw1,
                                   const at::Tensor& tb1, const at::Tensor& tw2,
                                   const at::Tensor& tb2, const at::Tensor& pw1,
                                   const at::Tensor& pb1, const at::Tensor& pw2,
                                   const at::Tensor& pb2, const int64_t wpr1, const int64_t rpb1,
                                   const int64_t wpr2, const int64_t rpb2, const int64_t unroll2,
                                   const int64_t trig_mode, const int64_t numeric_flags,
                                   const bool use_prologue) {
  return mlp_dispatch({&tw1, &tb1, &tw2, &tb2, &pw1, &pb1, &pw2, &pb2}, p, &t, nullptr, &freq,
                      num_channels, flip, scale, wpr1, rpb1, wpr2, rpb2, unroll2, trig_mode,
                      numeric_flags, use_prologue);
}

at::Tensor fused_combined3_forward(
    const at::Tensor& t, const at::Tensor& g, const at::Tensor& p, const at::Tensor& freq,
    const int64_t num_channels, const bool flip, const double scale, const at::Tensor& tw1,
    const at::Tensor& tb1, const at::Tensor& tw2, const at::Tensor& tb2, const at::Tensor& gw1,
    const at::Tensor& gb1, const at::Tensor& gw2, const at::Tensor& gb2, const at::Tensor& pw1,
    const at::Tensor& pb1, const at::Tensor& pw2, const at::Tensor& pb2, const int64_t wpr1,
    const int64_t rpb1, const int64_t wpr2, const int64_t rpb2, const int64_t unroll2,
    const int64_t trig_mode, const int64_t numeric_flags, const bool use_prologue) {
  return mlp_dispatch({&tw1, &tb1, &tw2, &tb2, &gw1, &gb1, &gw2, &gb2, &pw1, &pb1, &pw2, &pb2},
                      p, &t, &g, &freq, num_channels, flip, scale, wpr1, rpb1, wpr2, rpb2,
                      unroll2, trig_mode, numeric_flags, use_prologue);
}

bool mlp_admits(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1,
                const at::Tensor& w2, const at::Tensor& b2) {
  int64_t n = 0, k = 0;
  return mlp_plan({&w1, &b1, &w2, &b2}, x, nullptr, nullptr, nullptr, 0, &n, &k);
}

bool combined2_admits(const at::Tensor& t, const at::Tensor& p, const at::Tensor& freq,
                      const int64_t num_channels, const at::Tensor& tw1, const at::Tensor& tb1,
                      const at::Tensor& tw2, const at::Tensor& tb2, const at::Tensor& pw1,
                      const at::Tensor& pb1, const at::Tensor& pw2, const at::Tensor& pb2) {
  int64_t n = 0, k = 0;
  return mlp_plan({&tw1, &tb1, &tw2, &tb2, &pw1, &pb1, &pw2, &pb2}, p, &t, nullptr, &freq,
                  num_channels, &n, &k);
}

bool combined3_admits(const at::Tensor& t, const at::Tensor& g, const at::Tensor& p,
                      const at::Tensor& freq, const int64_t num_channels, const at::Tensor& tw1,
                      const at::Tensor& tb1, const at::Tensor& tw2, const at::Tensor& tb2,
                      const at::Tensor& gw1, const at::Tensor& gb1, const at::Tensor& gw2,
                      const at::Tensor& gb2, const at::Tensor& pw1, const at::Tensor& pb1,
                      const at::Tensor& pw2, const at::Tensor& pb2) {
  int64_t n = 0, k = 0;
  return mlp_plan({&tw1, &tb1, &tw2, &tb2, &gw1, &gb1, &gw2, &gb2, &pw1, &pb1, &pw2, &pb2}, p,
                  &t, &g, &freq, num_channels, &n, &k);
}

at::Tensor silu_probe(const at::Tensor& x, const int64_t numeric_flags) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_cuda() && x.is_contiguous(),
              "silu_probe expects a contiguous bfloat16 CUDA tensor");
  const at::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) {
    return out;
  }
  constexpr int kBlock = 256;
  silu_probe_kernel<<<static_cast<unsigned>((n + kBlock - 1) / kBlock), kBlock, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      bf16_ptr(x), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n,
      static_cast<int>(numeric_flags));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
