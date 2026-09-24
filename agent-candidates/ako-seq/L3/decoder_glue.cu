// Boundary kernels for a whole-decoder-layer CUDA graph.
//
// Two jobs, both of which exist only because the layer is replayed rather than
// re-dispatched.
//
// 1. THE GRAPH BOUNDARY, FUSED INTO THE FIRST NORM
// ------------------------------------------------
// A replayed layer reads its inputs from buffers fixed at capture time, and the
// benchmark's shifting memory pool moves the caller's ``hidden_states`` /
// ``residual`` / ``positions`` every iteration, so those three have to be
// copied in.  Its outputs have to land in memory the *caller* owns, which is a
// fresh allocation per call and so cannot be a baked graph operand.  Five
// ``Tensor.copy_`` calls is five dispatches and five launches for ~1.5 MB, which
// at the decode sizes this path exists for is the same order as the entire
// device cost of the layer's non-GEMM kernels.
//
// So the copies are not separate work here.  ``add_rms_norm`` /
// ``rms_norm_copy`` take *separate* input and output pointers, which makes the
// layer's first ``residual``-add + RMSNorm double as the whole inbound
// boundary: it reads the caller's tensors wherever they are, writes the
// normalized activation and the running residual into the graph's fixed
// buffers, and carries ``positions`` across in the same launch (one int64 per
// block -- one per token -- so it is free).  Aliasing the outputs onto the
// inputs turns the same kernel into the in-place form the reference uses, which
// is what the second norm (inside the graph) wants.
//
// The *outbound* half is the same trick pointed the other way.  The inbound
// kernel also publishes this call's two output addresses into a two-int64 device
// slot (``publish()``, one thread), and ``slot_copy_kernel`` -- a node **inside**
// the graph -- takes its destinations from that slot instead of from launch
// parameters.  So there is no eager launch after the replay at all, and the copy
// stops paying the 1.7-2.7 us the device spent starved at the graph's trailing
// edge.
//
// Net: the eager-side cost of a replayed layer is one launch in and one graph
// launch.  Both boundary calls are reached through integer entry points
// (``stage_raw`` / ``slot_copy``); see section 4 for why.
//
// 2. WHY THE NORM IS REIMPLEMENTED HERE AND NOT IMPORTED
// -----------------------------------------------------
// The frozen L1 RMSNorm winner reduces the variance with warp shuffles where
// the vendored vLLM kernel uses ``cub::BlockReduce``.  Both are correct and at
// L1 the difference is invisible -- a different fp32 summation order moves
// ``rsqrt(var/H + eps)`` by ~1e-7 relative, which flips a handful of output
// elements by one bf16 ULP.  At L3 that is not invisible: a single flipped
// element in row i of the *first* norm perturbs all 6144 qkv outputs of row i,
// and after attention, o_proj, the second norm (whose variance is recomputed
// over the perturbed row) and the MLP, the whole of row i differs at the 1e-2
// level.  One poisoned row out of 26 tokens is 3.8% of the output, and the
// harness allows 1%.  Measured on the captured shapes: 2 of 3 correctness
// rounds bit-identical, the third at 6.25e-2 with 4.2% of elements outside
// tolerance -- i.e. a fail whose appearance depends on the input draw.
//
// The layer therefore needs a norm that is *bit*-identical to the reference,
// not merely as accurate.  This one reproduces the vendored kernel stage by
// stage: the same 16-byte vector width, the same packed 16-bit residual add,
// the same per-thread square accumulation order (pairwise
// ``result += z.x*z.x + z.y*z.y`` for the fused form, sequential
// ``variance += x*x`` for the plain form -- the two vLLM kernels differ here),
// the same block size rule (``min(hidden, num_tokens < 256 ? 1024 : 256)`` for
// the fused form, ``min(hidden/8, ...)`` for the plain one) so each thread owns
// the same vectors, and the same block reduction shape as
// ``cub::BlockReduce<float, 1024>``'s warp-reductions specialization: a
// shuffle-down butterfly at offsets 1,2,4,8,16 inside each warp, then thread 0
// summing the warp aggregates *sequentially* in increasing warp order.  Warps
// beyond the ones holding data contribute exactly 0.0f in cub's full-tile case
// and are simply not visited here, which is the same sum.
//
// ``kernel.py`` verifies the bit-identity against the reference kernel on the
// first forward for each token count and falls back if it ever fails, so a
// compiler or cub change degrades to the reference instead of scoring wrong.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int kWidth = 8;          // 16-byte vector for a 16-bit dtype
constexpr int kWarp = 32;

// ---------------------------------------------------------------------------
// dtype plumbing: exactly the conversions the vendored kernel's _typeConvert
// uses, so a pair of 16-bit values becomes a float2 the same way.
// ---------------------------------------------------------------------------
template <typename T> struct Cvt;

template <> struct Cvt<__nv_bfloat16> {
  using T2 = __nv_bfloat162;
  __device__ static float2 to_f2(T2 x) { return __bfloat1622float2(x); }
  __device__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16(x); }
  __device__ static T2 add2(T2 a, T2 b) { return __hadd2(a, b); }
};

template <> struct Cvt<__half> {
  using T2 = __half2;
  __device__ static float2 to_f2(T2 x) { return __half22float2(x); }
  __device__ static float to_f(__half x) { return __half2float(x); }
  __device__ static __half from_f(float x) { return __float2half_rn(x); }
  __device__ static T2 add2(T2 a, T2 b) { return __hadd2(a, b); }
};

template <typename T>
struct alignas(16) Vec8 {
  T data[kWidth];
};

// Programmatic dependent launch.  Without the attribute the device-side
// ``cudaGridDependencySynchronize()`` degrades to a no-op, so every kernel here
// stays correct either way; with it, the block scheduler makes our blocks
// resident while the producer drains.  The frozen L1 SiluAndMul already does
// this, which is why its gap in ``dev/probe5.py`` is negative and ours were not.
template <typename F, typename... A>
inline void launch_pdl(F kern, dim3 grid, int threads, size_t smem,
                       cudaStream_t stream, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

// cub::BlockReduce<float, 1024>'s BLOCK_REDUCE_WARP_REDUCTIONS shape.
// ``n_warps`` is blockDim.x / 32; cub's remaining (data-free) warps add 0.0f,
// which is the identity, so they are skipped rather than materialized.
//
// ``smem`` is always ``kWarp`` floats wide, not ``n_warps``, so the aggregate
// loads below can be issued in fixed batches without reading past the
// allocation.  The sequential increasing-warp-order *sum* is part of the
// numerics and is preserved exactly; what is not part of the numerics is that
// cub reaches each aggregate through a dependent shared-memory load, which is
// ~960 cycles of thread 0 while the other 31 warps wait on ``__syncthreads``.
// The loads are issued in batches of 8 instead, leaving 31 dependent fp32 adds.
__device__ __forceinline__ float cub_block_sum(float v, float* smem,
                                              const int n_warps) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;
  // cub's ``warp_num_valid``: the last warp of a block that is not a whole
  // multiple of 32 is short, and only its live lanes may be shuffled.
  const int live = min(kWarp, static_cast<int>(blockDim.x) - (warp << 5));
  const unsigned mask =
      (live == kWarp) ? 0xffffffffu : ((1u << live) - 1u);
  const int last_lane = live - 1;
  // WarpReduceShfl::ReduceStep, offsets 1 << STEP for STEP = 0..4.
#pragma unroll
  for (int off = 1; off < kWarp; off <<= 1) {
    const float t = __shfl_down_sync(mask, v, off);
    if (lane + off <= last_lane) v = v + t;
  }
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    // ApplyWarpAggregates: sequential, increasing warp order.
#pragma unroll
    for (int base = 0; base < kWarp; base += 8) {
      if (base >= n_warps) break;
      float agg[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) agg[j] = smem[base + j];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (base + j >= 1 && base + j < n_warps) v = v + agg[j];
    }
    smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

// The other half of moving the outbound copy inside the graph: the inbound
// boundary publishes *this call's* two destination addresses into a device slot,
// and the copy node inside the graph reads them from there.  One int64 pair from
// one thread, in a kernel that is already crossing the boundary, so it is free;
// what it buys is one fewer eager launch and, with it, the 1.7-2.7 us the device
// spent starved between the graph's last kernel and the copy.
__device__ __forceinline__ void publish(int64_t* slot, int64_t a, int64_t b) {
  if (slot != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
    slot[0] = a;
    slot[1] = b;
  }
}

// ---------------------------------------------------------------------------
// residual-add + RMSNorm.  ``out``/``res_out`` may alias ``x``/``res_in``.
// ---------------------------------------------------------------------------
// ``MAXIT`` is a compile-time bound on the per-thread vector count
// (``ceil(vec_hidden / threads)``, which the launcher fixes), so the residual
// sum can live in registers and the epilogue does not have to read back the
// memory it just wrote.  Same values, same order, same rounding: bit-identical,
// and ``dev/tnorm.py`` proves it over dtype x hidden x token count.  ``MAXIT ==
// 0`` is the unbounded fallback for a hidden size too large to hold.
template <typename T, int MAXIT>
__global__ void add_rms_norm_kernel(
    T* out, T* res_out, const T* x, const T* res_in,
    const T* __restrict__ weight,
    int64_t* __restrict__ pos_out, const int64_t* __restrict__ pos_in,
    const float eps, const int hidden, const int n_warps, const int pos_rows,
    const int64_t pos_stride, const int n_tok, int64_t* __restrict__ slot,
    const int64_t dst_a, const int64_t dst_b) {
  using C = Cvt<T>;
  using T2 = typename C::T2;
  extern __shared__ float smem[];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Before the first global read: this kernel is the graph's inbound boundary
  // and its producer is the caller's own copy into the shifting pool.
  cudaGridDependencySynchronize();
#endif
  publish(slot, dst_a, dst_b);

  const int vec_hidden = hidden / kWidth;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * vec_hidden;
  auto* xv = reinterpret_cast<const Vec8<T>*>(x) + row;
  auto* rv_in = reinterpret_cast<const Vec8<T>*>(res_in) + row;
  auto* rv_out = reinterpret_cast<Vec8<T>*>(res_out) + row;
  auto* ov = reinterpret_cast<Vec8<T>*>(out) + row;
  auto* wv = reinterpret_cast<const Vec8<T>*>(weight);

  // The boundary crossing for ``positions``: one index per token, and this
  // block owns exactly one token.
  if (pos_out != nullptr && threadIdx.x == 0) {
    for (int r = 0; r < pos_rows; ++r) {
      pos_out[static_cast<int64_t>(r) * n_tok + blockIdx.x] =
          pos_in[static_cast<int64_t>(r) * pos_stride + blockIdx.x];
    }
  }

  // One vector of work, exactly as the vendored kernel does it: packed 16-bit
  // residual add (``_f16Vec::operator+=``), then a local fp32 accumulator over
  // element pairs (``_f16Vec::sum_squares()``) folded once into the running
  // variance.
  auto body = [&](int idx, float& variance) -> Vec8<T> {
    Vec8<T> t = xv[idx];
    const Vec8<T> r = rv_in[idx];
#pragma unroll
    for (int i = 0; i < kWidth; i += 2) {
      T2 acc = C::add2(T2{t.data[i], t.data[i + 1]},
                       T2{r.data[i], r.data[i + 1]});
      t.data[i] = acc.x;
      t.data[i + 1] = acc.y;
    }
    float result = 0.0f;
#pragma unroll
    for (int i = 0; i < kWidth; i += 2) {
      float2 z = C::to_f2(T2{t.data[i], t.data[i + 1]});
      result += z.x * z.x + z.y * z.y;
    }
    variance += result;
    rv_out[idx] = t;
    return t;
  };
  auto epilogue = [&](int idx, const Vec8<T>& res, const Vec8<T>& w,
                      float s_variance) {
    Vec8<T> o;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      const float xf = C::to_f(res.data[j]);
      const float wf = C::to_f(w.data[j]);
      o.data[j] = C::from_f(xf * s_variance * wf);
    }
    ov[idx] = o;
  };

  float variance = 0.0f;
  if constexpr (MAXIT == 0) {
    for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x)
      body(idx, variance);
    variance = cub_block_sum(variance, smem, n_warps);
    const float s_variance = rsqrtf(variance / hidden + eps);
    for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x)
      epilogue(idx, rv_out[idx], wv[idx], s_variance);
  } else {
    Vec8<T> keep[MAXIT];
    // The affine weight does not depend on the reduction, so its load is issued
    // here rather than in the epilogue: at one block per row the epilogue's read
    // of it was a whole HBM latency sitting behind two __syncthreads.
    Vec8<T> wk[MAXIT];
#pragma unroll
    for (int k = 0; k < MAXIT; ++k) {
      const int idx = threadIdx.x + k * static_cast<int>(blockDim.x);
      if (idx < vec_hidden) {
        wk[k] = wv[idx];
        keep[k] = body(idx, variance);
      }
    }
    variance = cub_block_sum(variance, smem, n_warps);
    const float s_variance = rsqrtf(variance / hidden + eps);
#pragma unroll
    for (int k = 0; k < MAXIT; ++k) {
      const int idx = threadIdx.x + k * static_cast<int>(blockDim.x);
      if (idx < vec_hidden) epilogue(idx, keep[k], wk[k], s_variance);
    }
  }
}

// ---------------------------------------------------------------------------
// plain RMSNorm (+ optional passthrough copy of the input, which is the
// residual the reference hands back when it was called with ``residual=None``).
// The vendored plain kernel accumulates its squares one element at a time, not
// in pairs, so this loop is deliberately not the one above.
// ---------------------------------------------------------------------------
template <typename T, int MAXIT>
__global__ void rms_norm_copy_kernel(
    T* out, T* res_out, const T* x,
    const T* __restrict__ weight, int64_t* __restrict__ pos_out,
    const int64_t* __restrict__ pos_in, const float eps, const int hidden,
    const int n_warps, const int pos_rows, const int64_t pos_stride,
    const int n_tok, int64_t* __restrict__ slot, const int64_t dst_a,
    const int64_t dst_b) {
  using C = Cvt<T>;
  extern __shared__ float smem[];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
  publish(slot, dst_a, dst_b);

  const int vec_hidden = hidden / kWidth;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * vec_hidden;
  auto* xv = reinterpret_cast<const Vec8<T>*>(x) + row;
  auto* ov = reinterpret_cast<Vec8<T>*>(out) + row;
  auto* rv = res_out ? reinterpret_cast<Vec8<T>*>(res_out) + row : nullptr;
  auto* wv = reinterpret_cast<const Vec8<T>*>(weight);

  if (pos_out != nullptr && threadIdx.x == 0) {
    for (int r = 0; r < pos_rows; ++r) {
      pos_out[static_cast<int64_t>(r) * n_tok + blockIdx.x] =
          pos_in[static_cast<int64_t>(r) * pos_stride + blockIdx.x];
    }
  }

  // The vendored plain kernel accumulates its squares one element at a time,
  // not in pairs; that is why this is not the loop above.
  auto body = [&](int idx, float& variance) -> Vec8<T> {
    const Vec8<T> v = xv[idx];
#pragma unroll
    for (int i = 0; i < kWidth; ++i) {
      const float f = C::to_f(v.data[i]);
      variance += f * f;
    }
    if (rv) rv[idx] = v;
    return v;
  };
  auto epilogue = [&](int idx, const Vec8<T>& src, const Vec8<T>& w,
                      float s_variance) {
    Vec8<T> o;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      const float xf = C::to_f(src.data[j]);
      const float wf = C::to_f(w.data[j]);
      o.data[j] = C::from_f(xf * s_variance * wf);
    }
    ov[idx] = o;
  };

  float variance = 0.0f;
  if constexpr (MAXIT == 0) {
    for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x)
      body(idx, variance);
    variance = cub_block_sum(variance, smem, n_warps);
    const float s_variance = rsqrtf(variance / hidden + eps);
    for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x)
      epilogue(idx, xv[idx], wv[idx], s_variance);
  } else {
    Vec8<T> keep[MAXIT];
    Vec8<T> wk[MAXIT];
#pragma unroll
    for (int k = 0; k < MAXIT; ++k) {
      const int idx = threadIdx.x + k * static_cast<int>(blockDim.x);
      if (idx < vec_hidden) {
        wk[k] = wv[idx];
        keep[k] = body(idx, variance);
      }
    }
    variance = cub_block_sum(variance, smem, n_warps);
    const float s_variance = rsqrtf(variance / hidden + eps);
#pragma unroll
    for (int k = 0; k < MAXIT; ++k) {
      const int idx = threadIdx.x + k * static_cast<int>(blockDim.x);
      if (idx < vec_hidden) epilogue(idx, keep[k], wk[k], s_variance);
    }
  }
}

// ---------------------------------------------------------------------------
// Outbound boundary: up to two equally shaped 16-bit blocks plus an optional
// int64 block, one launch.  One grid y-slice per array keeps every slice's
// grid-stride loop fully coalesced with no per-element branch.
// ---------------------------------------------------------------------------
constexpr int kCopyThreads = 256;
constexpr int kCopyMaxBlocks = 512;

__global__ void fused_copy_kernel(uint4* __restrict__ ad,
                                  const uint4* __restrict__ as,
                                  uint4* __restrict__ bd,
                                  const uint4* __restrict__ bs,
                                  int64_t* __restrict__ pd,
                                  const int64_t* __restrict__ ps,
                                  const int n_vec, const int n_pos) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int step = gridDim.x * blockDim.x;
  if (blockIdx.y == 0) {
    for (int i = tid; i < n_vec; i += step) ad[i] = as[i];
  } else if (blockIdx.y == 1) {
    for (int i = tid; i < n_vec; i += step) bd[i] = bs[i];
  } else {
    for (int i = tid; i < n_pos; i += step) pd[i] = ps[i];
  }
}

// ---------------------------------------------------------------------------
// 3. READ-ONLY L2 PREFETCH  --  DEV ONLY, NOT ON THE SHIPPED PATH
// ---------------------------------------------------------------------------
// Kept because it is the instrument that measured the round's central negative
// result, not because anything calls it.  ``dev/probe6.py`` drives it.
//
// The theory it was built to test: a decode call spends ~20 us in kernels that
// touch 1-2 MB (attention, the two norms, SiluAndMul, the frozen QKV glue)
// while the four projections around them read 436 MB of weights, so the next
// projection's weights could be pulled into L2 during those windows.  Measured
// on B200 it does not pay, and the numbers are in ITERATIONS.md: residency is
// real (prefetching ``gate_up``'s leading 64 MB with ``evict_last`` shortens its
// GEMM from 46.6 to 39.3 us in situ) but the window has no spare bandwidth --
// the same call's wall time goes from 136.0 to 143.4 us.  The window is idle in
// *SM* terms, not in memory terms.
//
//   mode 1  ``prefetch.global.L2``            -- one line hint per instruction,
//                                               droppable, no data path
//   mode 2  ``cp.async.bulk.prefetch.L2``     -- TMA bulk request, fire and
//                                               forget: the copy engine does
//                                               the fetch, the CTA exits
//   mode 3  ``ld.global.nc``                  -- a real load, discarded.  Fetches
//                                               for certain but holds SMs for
//                                               the whole transfer
//   mode 4  mode 2 + ``evict_last`` policy    -- ask L2 to keep the lines until
//                                               the consumer reads them
constexpr int kPfMaxRanges = 8;

struct PfArg {
  const char* base[kPfMaxRanges];
  uint32_t chunks[kPfMaxRanges];
  uint32_t* sink;
};

template <int MODE, int CHUNK>
__global__ void l2_prefetch_kernel(PfArg a) {
  const char* base = a.base[blockIdx.y];
  const uint32_t n_chunk = a.chunks[blockIdx.y];
  const uint32_t stride = gridDim.x * blockDim.x;
  uint32_t acc = 0;
  uint64_t pol = 0;
  if constexpr (MODE == 4) {
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
                 : "=l"(pol));
  }
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < n_chunk;
       c += stride) {
    const char* p = base + static_cast<size_t>(c) * CHUNK;
    if constexpr (MODE == 1) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p) : "memory");
    } else if constexpr (MODE == 2) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p),
                   "r"(static_cast<uint32_t>(CHUNK))
                   : "memory");
    } else if constexpr (MODE == 4) {
      asm volatile(
          "cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(
              p),
          "r"(static_cast<uint32_t>(CHUNK)), "l"(pol)
          : "memory");
    } else {
      const int4* q = reinterpret_cast<const int4*>(p);
      int4 v[CHUNK / 16];
#pragma unroll
      for (int i = 0; i < CHUNK / 16; ++i) v[i] = __ldg(q + i);
#pragma unroll
      for (int i = 0; i < CHUNK / 16; ++i)
        acc ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    }
  }
  if constexpr (MODE == 3) {
    // Never taken, and the compiler cannot prove it, so the loads above are not
    // dead.  ``sink`` is scratch owned by the layer; nothing reads it.
    if (acc == 0xDEADBEEFu) a.sink[0] = acc;
  }
}

// ---------------------------------------------------------------------------
// Raw launchers.  Both the checked Tensor entry points and the hot-path
// integer ones below land here, so there is exactly one place where launch
// geometry -- and therefore the fp32 accumulation order, which follows from it
// -- is decided.
// ---------------------------------------------------------------------------
#define NORM_ARGS(T)                                                          \
  reinterpret_cast<T*>(out), reinterpret_cast<T*>(res_out),                   \
      reinterpret_cast<const T*>(x), reinterpret_cast<const T*>(res_in),      \
      reinterpret_cast<const T*>(weight), pos_out, pos_in, eps, hidden,       \
      n_warps, pos_rows, pos_stride, n, slot, dst_a, dst_b

template <typename T>
inline void launch_add_norm_t(void* out, void* res_out, const void* x,
                             const void* res_in, const void* weight,
                             int64_t* pos_out, const int64_t* pos_in, float eps,
                             int hidden, int n, int pos_rows,
                             int64_t pos_stride, int64_t* slot, int64_t dst_a,
                             int64_t dst_b, cudaStream_t stream) {
  // vLLM ``fused_add_rms_norm``'s launch geometry, verbatim: the per-thread
  // vector assignment (and therefore the fp32 accumulation order) follows from
  // it, so it is part of the numerics and not a tuning knob.
  const int threads = std::min(hidden, (n < 256) ? 1024 : 256);
  const int n_warps = (threads + kWarp - 1) / kWarp;
  const size_t smem = kWarp * sizeof(float);
  const int it = (hidden / kWidth + threads - 1) / threads;
  if (it <= 1)
    launch_pdl(add_rms_norm_kernel<T, 1>, dim3(n), threads, smem, stream,
               NORM_ARGS(T));
  else if (it <= 2)
    launch_pdl(add_rms_norm_kernel<T, 2>, dim3(n), threads, smem, stream,
               NORM_ARGS(T));
  else if (it <= 4)
    launch_pdl(add_rms_norm_kernel<T, 4>, dim3(n), threads, smem, stream,
               NORM_ARGS(T));
  else
    launch_pdl(add_rms_norm_kernel<T, 0>, dim3(n), threads, smem, stream,
               NORM_ARGS(T));
}

#define COPY_ARGS(T)                                                          \
  reinterpret_cast<T*>(out), reinterpret_cast<T*>(res_out),                   \
      reinterpret_cast<const T*>(x), reinterpret_cast<const T*>(weight),      \
      pos_out, pos_in, eps, hidden, n_warps, pos_rows, pos_stride, n, slot,   \
      dst_a, dst_b

template <typename T>
inline void launch_plain_norm_t(void* out, void* res_out, const void* x,
                               const void* weight, int64_t* pos_out,
                               const int64_t* pos_in, float eps, int hidden,
                               int n, int pos_rows, int64_t pos_stride,
                               int64_t* slot, int64_t dst_a, int64_t dst_b,
                               cudaStream_t stream) {
  // vLLM ``rms_norm``'s geometry: note it divides by the vector width where the
  // fused form does not, so the two kernels put a different number of vectors on
  // each thread and their reductions are not interchangeable.
  const int threads = std::min(hidden / kWidth, (n < 256) ? 1024 : 256);
  const int n_warps = (threads + kWarp - 1) / kWarp;
  const size_t smem = kWarp * sizeof(float);
  const int it = (hidden / kWidth + threads - 1) / threads;
  if (it <= 1)
    launch_pdl(rms_norm_copy_kernel<T, 1>, dim3(n), threads, smem, stream,
               COPY_ARGS(T));
  else if (it <= 2)
    launch_pdl(rms_norm_copy_kernel<T, 2>, dim3(n), threads, smem, stream,
               COPY_ARGS(T));
  else if (it <= 4)
    launch_pdl(rms_norm_copy_kernel<T, 4>, dim3(n), threads, smem, stream,
               COPY_ARGS(T));
  else
    launch_pdl(rms_norm_copy_kernel<T, 0>, dim3(n), threads, smem, stream,
               COPY_ARGS(T));
}

inline void launch_norm(bool half, bool fused, void* out, void* res_out,
                        const void* x, const void* res_in, const void* weight,
                        int64_t* pos_out, const int64_t* pos_in, float eps,
                        int hidden, int n, int pos_rows, int64_t pos_stride,
                        int64_t* slot, int64_t dst_a, int64_t dst_b,
                        cudaStream_t stream) {
  if (fused) {
    if (half)
      launch_add_norm_t<__half>(out, res_out, x, res_in, weight, pos_out,
                                pos_in, eps, hidden, n, pos_rows, pos_stride,
                                slot, dst_a, dst_b, stream);
    else
      launch_add_norm_t<__nv_bfloat16>(out, res_out, x, res_in, weight, pos_out,
                                       pos_in, eps, hidden, n, pos_rows,
                                       pos_stride, slot, dst_a, dst_b, stream);
  } else if (half) {
    launch_plain_norm_t<__half>(out, res_out, x, weight, pos_out, pos_in, eps,
                                hidden, n, pos_rows, pos_stride, slot, dst_a,
                                dst_b, stream);
  } else {
    launch_plain_norm_t<__nv_bfloat16>(out, res_out, x, weight, pos_out, pos_in,
                                       eps, hidden, n, pos_rows, pos_stride,
                                       slot, dst_a, dst_b, stream);
  }
}

inline void launch_fused_copy(void* ad, const void* as, void* bd,
                              const void* bs, int64_t* pd, const int64_t* ps,
                              int n_vec, int n_pos, cudaStream_t stream) {
  const int lanes = std::max(n_vec, n_pos);
  const dim3 grid(
      std::min(kCopyMaxBlocks, (lanes + kCopyThreads - 1) / kCopyThreads),
      1u + (bd != nullptr ? 1u : 0u) + (pd != nullptr ? 1u : 0u));
  launch_pdl(fused_copy_kernel, grid, kCopyThreads, 0, stream,
             static_cast<uint4*>(ad), static_cast<const uint4*>(as),
             static_cast<uint4*>(bd), static_cast<const uint4*>(bs), pd, ps,
             n_vec, n_pos);
}

// Outbound boundary as a graph node: same body as ``fused_copy_kernel``, but
// the two destinations come from the slot the inbound kernel wrote rather than
// from baked launch parameters, which is what lets it be captured at all.
__global__ void slot_copy_kernel(int64_t* __restrict__ slot,
                                 const uint4* __restrict__ as,
                                 const uint4* __restrict__ bs,
                                 const int n_vec) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
  auto* dst = reinterpret_cast<uint4*>(blockIdx.y == 0 ? slot[0] : slot[1]);
  const uint4* src = blockIdx.y == 0 ? as : bs;
  const int step = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += step)
    dst[i] = src[i];
}

inline int64_t ptr_of(const torch::Tensor& t) {
  return reinterpret_cast<int64_t>(t.data_ptr());
}

inline void check_row(const torch::Tensor& t, int64_t n, int64_t hidden,
                      const char* what) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), what, ": cuda contiguous only");
  TORCH_CHECK(t.dim() == 2 && t.size(0) == n && t.size(1) == hidden,
              what, ": shape mismatch");
  TORCH_CHECK(ptr_of(t) % 16 == 0, what, ": needs 16B alignment");
}

}  // namespace

void add_rms_norm(torch::Tensor out, torch::Tensor res_out, torch::Tensor x,
                  torch::Tensor res_in, torch::Tensor weight, double eps,
                  c10::optional<torch::Tensor> pos_out,
                  c10::optional<torch::Tensor> pos_in) {
  const int64_t hidden = x.size(-1);
  const int64_t n = x.numel() / hidden;
  check_row(x, n, hidden, "add_rms_norm.x");
  check_row(res_in, n, hidden, "add_rms_norm.res_in");
  check_row(out, n, hidden, "add_rms_norm.out");
  check_row(res_out, n, hidden, "add_rms_norm.res_out");
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == hidden
                  && ptr_of(weight) % 16 == 0,
              "add_rms_norm: weight");
  TORCH_CHECK(hidden % kWidth == 0, "add_rms_norm: hidden % 8");
  TORCH_CHECK(res_out.scalar_type() == x.scalar_type()
                  && out.scalar_type() == x.scalar_type()
                  && weight.scalar_type() == x.scalar_type(),
              "add_rms_norm: dtype mismatch");
  if (n == 0) return;

  int pos_rows = 0;
  int64_t pos_stride = 0;
  int64_t* pd = nullptr;
  const int64_t* ps = nullptr;
  if (pos_out.has_value() && pos_in.has_value()) {
    const auto& po = *pos_out;
    const auto& pi = *pos_in;
    TORCH_CHECK(po.scalar_type() == at::kLong && pi.scalar_type() == at::kLong,
                "add_rms_norm: positions must be int64");
    TORCH_CHECK(po.numel() == pi.numel() && po.numel() % n == 0
                    && po.is_contiguous() && pi.stride(-1) == 1,
                "add_rms_norm: positions layout");
    pos_rows = static_cast<int>(po.numel() / n);
    pos_stride = pi.dim() >= 2 ? pi.stride(0) : n;
    pd = po.data_ptr<int64_t>();
    ps = pi.data_ptr<int64_t>();
  }

  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
              "add_rms_norm: bf16/fp16 only");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  launch_norm(x.scalar_type() == at::kHalf, /*fused=*/true, out.data_ptr(),
              res_out.data_ptr(), x.data_ptr(), res_in.data_ptr(),
              weight.data_ptr(), pd, ps, static_cast<float>(eps),
              static_cast<int>(hidden), static_cast<int>(n), pos_rows,
              pos_stride, nullptr, 0, 0, at::cuda::getCurrentCUDAStream());
}

void rms_norm_copy(torch::Tensor out, c10::optional<torch::Tensor> res_out,
                   torch::Tensor x, torch::Tensor weight, double eps,
                   c10::optional<torch::Tensor> pos_out,
                   c10::optional<torch::Tensor> pos_in) {
  const int64_t hidden = x.size(-1);
  const int64_t n = x.numel() / hidden;
  check_row(x, n, hidden, "rms_norm_copy.x");
  check_row(out, n, hidden, "rms_norm_copy.out");
  if (res_out.has_value()) check_row(*res_out, n, hidden, "rms_norm_copy.res");
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == hidden
                  && ptr_of(weight) % 16 == 0,
              "rms_norm_copy: weight");
  TORCH_CHECK(hidden % kWidth == 0, "rms_norm_copy: hidden % 8");
  TORCH_CHECK(out.scalar_type() == x.scalar_type()
                  && weight.scalar_type() == x.scalar_type(),
              "rms_norm_copy: dtype mismatch");
  if (n == 0) return;

  int pos_rows = 0;
  int64_t pos_stride = 0;
  int64_t* pd = nullptr;
  const int64_t* ps = nullptr;
  if (pos_out.has_value() && pos_in.has_value()) {
    const auto& po = *pos_out;
    const auto& pi = *pos_in;
    TORCH_CHECK(po.scalar_type() == at::kLong && pi.scalar_type() == at::kLong,
                "rms_norm_copy: positions must be int64");
    TORCH_CHECK(po.numel() == pi.numel() && po.numel() % n == 0
                    && po.is_contiguous() && pi.stride(-1) == 1,
                "rms_norm_copy: positions layout");
    pos_rows = static_cast<int>(po.numel() / n);
    pos_stride = pi.dim() >= 2 ? pi.stride(0) : n;
    pd = po.data_ptr<int64_t>();
    ps = pi.data_ptr<int64_t>();
  }

  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
              "rms_norm_copy: bf16/fp16 only");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  launch_norm(x.scalar_type() == at::kHalf, /*fused=*/false, out.data_ptr(),
              res_out.has_value() ? res_out->data_ptr() : nullptr, x.data_ptr(),
              nullptr, weight.data_ptr(), pd, ps, static_cast<float>(eps),
              static_cast<int>(hidden), static_cast<int>(n), pos_rows,
              pos_stride, nullptr, 0, 0, at::cuda::getCurrentCUDAStream());
}

void fused_copy(torch::Tensor a_dst, torch::Tensor a_src,
                c10::optional<torch::Tensor> b_dst,
                c10::optional<torch::Tensor> b_src,
                c10::optional<torch::Tensor> p_dst,
                c10::optional<torch::Tensor> p_src) {
  TORCH_CHECK(a_dst.is_cuda() && a_src.is_cuda(), "fused_copy: cuda only");
  TORCH_CHECK(a_dst.nbytes() == a_src.nbytes(), "fused_copy: size mismatch");
  const bool has_b = b_dst.has_value() && b_src.has_value();
  const bool has_p = p_dst.has_value() && p_src.has_value();
  if (has_b) {
    TORCH_CHECK(b_dst->nbytes() == b_src->nbytes()
                    && b_dst->nbytes() == a_dst.nbytes(),
                "fused_copy: b size mismatch");
  }
  const int n_vec = static_cast<int>(a_dst.nbytes() / 16);
  const int n_pos = has_p ? static_cast<int>(p_dst->numel()) : 0;
  if (n_vec == 0 && n_pos == 0) return;
  if (has_p) {
    TORCH_CHECK(p_dst->numel() == p_src->numel()
                    && p_dst->scalar_type() == at::kLong
                    && p_src->scalar_type() == at::kLong,
                "fused_copy: index block must be matched int64");
  }
  const at::cuda::OptionalCUDAGuard guard(a_dst.device());
  launch_fused_copy(a_dst.data_ptr(), a_src.data_ptr(),
                    has_b ? b_dst->data_ptr() : nullptr,
                    has_b ? b_src->data_ptr() : nullptr,
                    has_p ? p_dst->data_ptr<int64_t>() : nullptr,
                    has_p ? p_src->data_ptr<int64_t>() : nullptr, n_vec, n_pos,
                    at::cuda::getCurrentCUDAStream());
}

// Whether ``fused_copy`` may take this tensor as one of its 16-byte blocks.
bool copy_ok(torch::Tensor t) {
  return t.is_contiguous() && t.nbytes() % 16 == 0 && ptr_of(t) % 16 == 0;
}


// ``ptrs``/``nbytes`` describe read-only byte ranges (weight tensors).  Each
// range gets its own grid.y slice; the whole launch is one graph node.
void l2_prefetch(std::vector<int64_t> ptrs, std::vector<int64_t> nbytes,
                 int64_t mode, int64_t chunk, int64_t blocks, int64_t threads,
                 int64_t sink) {
  TORCH_CHECK(ptrs.size() == nbytes.size(), "l2_prefetch: ptr/size mismatch");
  TORCH_CHECK(!ptrs.empty() && ptrs.size() <= kPfMaxRanges,
              "l2_prefetch: 1..", kPfMaxRanges, " ranges");
  TORCH_CHECK(chunk >= 16 && (chunk & (chunk - 1)) == 0,
              "l2_prefetch: chunk must be a power of two >= 16");
  TORCH_CHECK(mode != 3 || sink != 0, "l2_prefetch: mode 3 needs a sink");
  PfArg a{};
  a.sink = reinterpret_cast<uint32_t*>(sink);
  uint32_t most = 0;
  for (size_t i = 0; i < ptrs.size(); ++i) {
    TORCH_CHECK(ptrs[i] != 0 && ptrs[i] % chunk == 0,
                "l2_prefetch: range ", i, " must be chunk-aligned");
    TORCH_CHECK(nbytes[i] >= 0, "l2_prefetch: negative size");
    // Truncate, never round up: a prefetch past the end of the tensor would
    // fault (mode 3) or fetch memory we do not own.
    const int64_t n = nbytes[i] / chunk;
    TORCH_CHECK(n <= 0xFFFFFFFFll, "l2_prefetch: range too large");
    a.base[i] = reinterpret_cast<const char*>(ptrs[i]);
    a.chunks[i] = static_cast<uint32_t>(n);
    most = std::max(most, a.chunks[i]);
  }
  if (most == 0) return;
  const int thr = static_cast<int>(threads);
  TORCH_CHECK(thr > 0 && thr <= 1024 && thr % 32 == 0,
              "l2_prefetch: threads must be a positive multiple of 32");
  const int bl = static_cast<int>(
      std::min<int64_t>(blocks, (most + thr - 1) / thr));
  TORCH_CHECK(bl > 0, "l2_prefetch: blocks must be positive");
  const dim3 grid(bl, static_cast<unsigned>(ptrs.size()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define PF_LAUNCH(M, C)                                                    \
  if (mode == (M) && chunk == (C)) {                                       \
    l2_prefetch_kernel<M, C><<<grid, thr, 0, stream>>>(a);                 \
    return;                                                                \
  }
  PF_LAUNCH(1, 64) PF_LAUNCH(1, 128) PF_LAUNCH(1, 256)
  PF_LAUNCH(2, 256) PF_LAUNCH(2, 512) PF_LAUNCH(2, 1024)
  PF_LAUNCH(2, 2048) PF_LAUNCH(2, 4096)
  PF_LAUNCH(3, 32) PF_LAUNCH(3, 64) PF_LAUNCH(3, 128) PF_LAUNCH(3, 256)
  PF_LAUNCH(4, 256) PF_LAUNCH(4, 512) PF_LAUNCH(4, 1024)
  PF_LAUNCH(4, 2048) PF_LAUNCH(4, 4096)
#undef PF_LAUNCH
  TORCH_CHECK(false, "l2_prefetch: no instantiation for mode ", mode,
              " chunk ", chunk);
}


// ---------------------------------------------------------------------------
// 4. THE HOT PATH, PRICED AS A LAUNCH
// ---------------------------------------------------------------------------
// ``kernel.py``'s ``forward`` has already established -- it has to, to decide it
// may replay at all -- that every tensor on this path is 2-D, contiguous,
// 16-byte aligned, of the plan's dtype and of the plan's shape.  Re-establishing
// all of it through six pybind ``Tensor`` conversions and a dozen
// ``TORCH_CHECK``s costs ~4.6 us of host time per call for ``add_rms_norm`` and
// ~5 us for ``fused_copy``, and ``dev/probe5.py`` measures the device sitting
// *starved* for 4.8 us between the staging launch and the graph's first kernel
// because the host has not got there yet.  So the replay path passes integers.
//
// These are internal entry points: the checked ``Tensor`` versions above are
// what the eager flat path, the reference-composition fallback and
// ``_verify_norm`` use, and they land on the same launchers, so there is one
// definition of the numerics and two ways in.
//
// ``kind``: bit 0 = fp16 rather than bf16, bit 1 = plain rather than fused.
void stage_raw(int64_t kind, int64_t out, int64_t res_out, int64_t x,
               int64_t res_in, int64_t weight, int64_t pos_out, int64_t pos_in,
               double eps, int64_t hidden, int64_t n, int64_t pos_rows,
               int64_t pos_stride, int64_t slot, int64_t dst_a, int64_t dst_b) {
  TORCH_CHECK(n > 0 && hidden > 0 && hidden % kWidth == 0,
              "stage_raw: bad shape");
  TORCH_CHECK(out && x && weight, "stage_raw: null pointer");
  TORCH_CHECK(!(kind & 2) || res_in == 0, "stage_raw: plain form takes no residual");
  TORCH_CHECK((kind & 2) || res_in, "stage_raw: fused form needs a residual");
  launch_norm(kind & 1, !(kind & 2), reinterpret_cast<void*>(out),
              reinterpret_cast<void*>(res_out), reinterpret_cast<const void*>(x),
              reinterpret_cast<const void*>(res_in),
              reinterpret_cast<const void*>(weight),
              reinterpret_cast<int64_t*>(pos_out),
              reinterpret_cast<const int64_t*>(pos_in),
              static_cast<float>(eps), static_cast<int>(hidden),
              static_cast<int>(n), static_cast<int>(pos_rows), pos_stride,
              reinterpret_cast<int64_t*>(slot), dst_a, dst_b,
              at::cuda::getCurrentCUDAStream());
}

// The outbound boundary, captured.  ``n_vec`` is fixed by the token count, the
// destinations are not, so they arrive through the slot.
void slot_copy(int64_t slot, int64_t a_src, int64_t b_src, int64_t n_vec) {
  TORCH_CHECK(slot && a_src && b_src, "slot_copy: null pointer");
  TORCH_CHECK(n_vec > 0 && n_vec <= 0x7FFFFFFFll, "slot_copy: bad extent");
  const dim3 grid(std::min<int64_t>(kCopyMaxBlocks,
                                    (n_vec + kCopyThreads - 1) / kCopyThreads),
                  2);
  launch_pdl(slot_copy_kernel, grid, kCopyThreads, 0,
             at::cuda::getCurrentCUDAStream(),
             reinterpret_cast<int64_t*>(slot),
             reinterpret_cast<const uint4*>(a_src),
             reinterpret_cast<const uint4*>(b_src), static_cast<int>(n_vec));
}

void fused_copy_raw(int64_t a_dst, int64_t a_src, int64_t b_dst, int64_t b_src,
                    int64_t p_dst, int64_t p_src, int64_t n_vec, int64_t n_pos) {
  TORCH_CHECK(a_dst && a_src, "fused_copy_raw: null pointer");
  TORCH_CHECK(n_vec >= 0 && n_pos >= 0 && n_vec <= 0x7FFFFFFFll
                  && n_pos <= 0x7FFFFFFFll,
              "fused_copy_raw: bad extent");
  if (n_vec == 0 && n_pos == 0) return;
  launch_fused_copy(reinterpret_cast<void*>(a_dst),
                    reinterpret_cast<const void*>(a_src),
                    reinterpret_cast<void*>(b_dst),
                    reinterpret_cast<const void*>(b_src),
                    reinterpret_cast<int64_t*>(p_dst),
                    reinterpret_cast<const int64_t*>(p_src),
                    static_cast<int>(n_vec), static_cast<int>(n_pos),
                    at::cuda::getCurrentCUDAStream());
}

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_rms_norm", &add_rms_norm,
        "vLLM-order fused residual-add + RMSNorm, out-of-place capable",
        py::arg("out"), py::arg("res_out"), py::arg("x"), py::arg("res_in"),
        py::arg("weight"), py::arg("eps"), py::arg("pos_out") = c10::nullopt,
        py::arg("pos_in") = c10::nullopt);
  m.def("rms_norm_copy", &rms_norm_copy,
        "vLLM-order RMSNorm with optional input passthrough", py::arg("out"),
        py::arg("res_out"), py::arg("x"), py::arg("weight"), py::arg("eps"),
        py::arg("pos_out") = c10::nullopt, py::arg("pos_in") = c10::nullopt);
  m.def("fused_copy", &fused_copy, "Fused multi-block device copy",
        py::arg("a_dst"), py::arg("a_src"), py::arg("b_dst") = c10::nullopt,
        py::arg("b_src") = c10::nullopt, py::arg("p_dst") = c10::nullopt,
        py::arg("p_src") = c10::nullopt);
  m.def("stage_raw", &stage_raw,
        "raw-pointer inbound boundary (norm + input staging)", py::arg("kind"),
        py::arg("out"), py::arg("res_out"), py::arg("x"), py::arg("res_in"),
        py::arg("weight"), py::arg("pos_out"), py::arg("pos_in"),
        py::arg("eps"), py::arg("hidden"), py::arg("n"), py::arg("pos_rows"),
        py::arg("pos_stride"), py::arg("slot") = 0, py::arg("dst_a") = 0,
        py::arg("dst_b") = 0);
  m.def("slot_copy", &slot_copy, "outbound boundary, destinations via a slot",
        py::arg("slot"), py::arg("a_src"), py::arg("b_src"), py::arg("n_vec"));
  m.def("fused_copy_raw", &fused_copy_raw, "raw-pointer outbound boundary",
        py::arg("a_dst"), py::arg("a_src"), py::arg("b_dst"), py::arg("b_src"),
        py::arg("p_dst"), py::arg("p_src"), py::arg("n_vec"), py::arg("n_pos"));
  m.def("copy_ok", &copy_ok, "16-byte vector eligibility", py::arg("t"));
  m.def("l2_prefetch", &l2_prefetch,
        "read-only L2 prefetch over weight byte ranges", py::arg("ptrs"),
        py::arg("nbytes"), py::arg("mode"), py::arg("chunk"),
        py::arg("blocks"), py::arg("threads"), py::arg("sink") = 0);
}
