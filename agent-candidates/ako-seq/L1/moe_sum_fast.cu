// Vectorized streaming MoE top-k reduction.
//
// The op is pure memory streaming: read M*topk*D, write M*D, one add chain per
// output element. Score is decided by request width, memory-level parallelism
// and launch-side latency, not by arithmetic. What this does that the scalar
// one-block-per-token reference does not:
//
//   1. 16-byte (uint4) loads/stores instead of one 2-byte bf16 at a time, with
//      all TOPK loads issued before any add (`STAGED`).
//   2. A flat 1-D index space over *output* vectors with a grid-stride loop,
//      so the grid is sized from the SM count and the actual work -- tiny
//      shapes (M=1 for the captured [8,4096]) spread over many SMs instead of
//      landing on a single block.
//   3. Float accumulation for every topk (matching the reference's topk=8
//      `facc` path) and a generic runtime-topk kernel, so there is no
//      `at::sum_out` fallback cliff.
//   4. Programmatic Dependent Launch. Every caller has a producer immediately
//      ahead of it on the stream (for the bench harness, the shifting pool's
//      input copy), so CTA dispatch, instruction fetch and kernel-param loads
//      can happen in that producer's shadow instead of serializing behind it.
//      Worth 2 us of stream time on every shape and 4 us at [8, 4096], where
//      the whole kernel fits in the shadow.
//      `griddepcontrol.wait` at kernel entry is what makes reading
//      producer-written memory safe: it is the barrier the implicit stream
//      dependency would otherwise have given us. It is not optional.
//   5. Descending traversal when the input is larger than L2, so the first
//      blocks scheduled read the freshest lines the producer wrote.
//
// Alignment is checked on the host; anything not 16-byte clean falls back to a
// flat scalar grid-stride kernel with the same float accumulation.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace {

// ---------------------------------------------------------------------------
// Per-dtype uint4 <-> float-lane packing.
// ---------------------------------------------------------------------------
template <typename T>
struct VecT;

template <>
struct VecT<nv_bfloat16> {
  static constexpr int LANES = 8;  // 16 B / 2 B
  __device__ __forceinline__ static void unpack_add(const uint4& w, float* acc) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(p[i]);
      acc[2 * i] += f.x;
      acc[2 * i + 1] += f.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* acc) {
    uint4 o;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    return o;
  }
};

template <>
struct VecT<nv_half> {
  static constexpr int LANES = 8;
  __device__ __forceinline__ static void unpack_add(const uint4& w, float* acc) {
    const __half2* p = reinterpret_cast<const __half2*>(&w);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __half22float2(p[i]);
      acc[2 * i] += f.x;
      acc[2 * i + 1] += f.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* acc) {
    uint4 o;
    __half2* p = reinterpret_cast<__half2*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __floats2half2_rn(acc[2 * i], acc[2 * i + 1]);
    return o;
  }
};

template <>
struct VecT<float> {
  static constexpr int LANES = 4;
  __device__ __forceinline__ static void unpack_add(const uint4& w, float* acc) {
    const float* p = reinterpret_cast<const float*>(&w);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] += p[i];
  }
  __device__ __forceinline__ static uint4 pack(const float* acc) {
    uint4 o;
    float* p = reinterpret_cast<float*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = acc[i];
    return o;
  }
};

// ---------------------------------------------------------------------------
// Reduce TOPK 16-byte vectors `kstride` vectors apart into one.
//
// STAGED issues every load before any add, so all TOPK requests are in flight
// at once instead of forming a load-add-load chain. TOPK_C == 0 means "runtime
// topk" (the generic path that replaces the at::sum_out cliff).
//
// Accumulation is in float, in the same order as the reference's
// moe_sum_facc_kernel, which is what makes topk=8 bit-exact against it.
// ---------------------------------------------------------------------------
template <typename T, int TOPK_C, bool STAGED>
__device__ __forceinline__ uint4 reduce_vec(const uint4* __restrict__ src, int64_t kstride,
                                            int topk_r) {
  constexpr int LANES = VecT<T>::LANES;
  float acc[LANES];
#pragma unroll
  for (int i = 0; i < LANES; ++i) acc[i] = 0.0f;

  if constexpr (TOPK_C > 0) {
    if constexpr (STAGED) {
      uint4 w[TOPK_C];
#pragma unroll
      for (int k = 0; k < TOPK_C; ++k) w[k] = src[static_cast<int64_t>(k) * kstride];
#pragma unroll
      for (int k = 0; k < TOPK_C; ++k) VecT<T>::unpack_add(w[k], acc);
    } else {
#pragma unroll
      for (int k = 0; k < TOPK_C; ++k)
        VecT<T>::unpack_add(src[static_cast<int64_t>(k) * kstride], acc);
    }
  } else {
    for (int k = 0; k < topk_r; ++k)
      VecT<T>::unpack_add(src[static_cast<int64_t>(k) * kstride], acc);
  }
  return VecT<T>::pack(acc);
}

// ---------------------------------------------------------------------------
// Flat reduction over output 16-byte vectors: one output vector per thread.
//
// `nvec` = M * vd output vectors, `vd` = D*sizeof(T)/16 vectors per row. For
// output vector j: token = j / vd, and the topk sources sit at
// j + token*(topk-1)*vd, stride vd.
//
// DESC picks the traversal direction. A "tile" is BLOCK consecutive output
// vectors; blocks are dispatched in increasing blockIdx.x, so walking tiles
// downwards makes the first-scheduled blocks touch the lines the producer
// wrote last, before our own misses evict them. Threads inside a tile always
// run ascending either way, so coalescing is identical. Only pays when the
// input exceeds L2 -- see `order_desc`.
//
// Sweep result: one vector per thread beats 2/4/8. More vectors per thread
// does add ILP, but staging TOPK*VPT uint4 costs 32*VPT registers -- VPT=4 is
// 58 regs (55% occupancy) vs VPT=1 at 40 (80%) -- and on a pure-streaming
// kernel the extra resident warps buy more outstanding loads than the extra
// per-thread ILP does. VPT is therefore gone.
//
// VD_POW2 is a template parameter rather than a runtime `vd_shift >= 0` test
// because the runtime form leaves a full 64-bit software integer division
// (MUFU.RCP + IMAD.HI chain) in the kernel body.
//
// Indexing stays 64-bit on purpose. Recasting to int32 (every captured shape
// addresses in 33.5 M vectors, so it is legal) *raised* the register count,
// 48 -> 56, and cost 2.0 us of span at [8000, 4096]. See ITERATIONS.md.
//
// `pdl` is a runtime argument rather than a template parameter so that one
// build can A/B it; the branch is grid-uniform and costs two instructions.
// ---------------------------------------------------------------------------
template <typename T, int TOPK_C, int BLOCK, bool VD_POW2, bool STAGED, bool DESC>
__global__ __launch_bounds__(BLOCK) void moe_sum_vec_kernel(
    uint4* __restrict__ out, const uint4* __restrict__ in, int64_t nvec, int ntiles, int vd,
    int vd_shift, int topk_r, int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Wait for the producer that wrote `in`. Free when there is nothing to wait
  // for; when there is, everything above (CTA setup, i-cache fill, param
  // loads) already happened in its shadow.
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int tk = TOPK_C > 0 ? TOPK_C : topk_r;
  const int64_t koff = static_cast<int64_t>(tk - 1) * vd;

  if constexpr (DESC) {
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
      const int64_t j = static_cast<int64_t>(ntiles - 1 - t) * BLOCK + threadIdx.x;
      if (j < nvec) {
        const int64_t tok = VD_POW2 ? (j >> vd_shift) : (j / vd);
        out[j] = reduce_vec<T, TOPK_C, STAGED>(in + tok * koff + j, vd, tk);
      }
    }
  } else {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t j = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x; j < nvec;
         j += stride) {
      const int64_t tok = VD_POW2 ? (j >> vd_shift) : (j / vd);
      out[j] = reduce_vec<T, TOPK_C, STAGED>(in + tok * koff + j, vd, tk);
    }
  }
}

// ---------------------------------------------------------------------------
// Scalar fallback: any dtype width / alignment / row stride, any topk.
// Float accumulation, so it is no less accurate than at::sum_out.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void moe_sum_scalar_kernel(T* __restrict__ out, const T* __restrict__ in, int64_t n,
                                      int d, int topk, int64_t out_row_stride, int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += stride) {
    const int64_t tok = i / d;
    const int64_t col = i - tok * d;
    const T* p = in + tok * static_cast<int64_t>(topk) * d + col;
    float x = 0.0f;
    for (int k = 0; k < topk; ++k) x += static_cast<float>(p[static_cast<int64_t>(k) * d]);
    out[tok * out_row_stride + col] = static_cast<T>(x);
  }
}

// ---------------------------------------------------------------------------
// Launch config.
// ---------------------------------------------------------------------------
inline const cudaDeviceProp& props() {
  static const cudaDeviceProp& p = *at::cuda::getCurrentDeviceProperties();
  return p;
}

inline int env_int(const char* name, int dflt) {
  const char* s = std::getenv(name);
  return (s && *s) ? std::atoi(s) : dflt;
}

struct Cfg {
  int block;   // threads per block; 128 measured best, 1024 is far worse
  int waves;   // cap on blocks per SM; beyond this the grid-stride loop covers it
  int staged;  // issue all TOPK loads before any add
  int pdl;     // programmatic dependent launch
  int desc;    // traversal direction; -1 = pick from the input size
};

inline Cfg read_cfg() {
  return Cfg{
      env_int("MOE_BLOCK", 128), env_int("MOE_WAVES", 256), env_int("MOE_STAGED", 1),
      env_int("MOE_PDL", 1),     env_int("MOE_DESC", -1),
  };
}

inline const Cfg& cfg() {
  static Cfg c = read_cfg();
  // MOE_DYNCFG=1 re-reads the env on every call so one process can sweep the
  // whole config space; costs a handful of getenv() per launch, so it stays
  // off by default.
  static const bool dyn = env_int("MOE_DYNCFG", 0) != 0;
  if (dyn) c = read_cfg();
  return c;
}

// Descending traversal only when the input is bigger than L2. Below that the
// producer's writes are either still resident -- measured 93% L2 hit rate at
// [2512, 4096] -- or unrecoverable whatever the order, and the loop form the
// descending path needs has measured as much as 1.8 us worse on the tiny
// [8, 4096] launch, where the flat grid-stride form sits exactly on the span
// floor.
inline bool order_desc(size_t in_bytes, int cfg_desc) {
  if (cfg_desc >= 0) return cfg_desc != 0;
  return in_bytes > static_cast<size_t>(props().l2CacheSize);
}

// ---------------------------------------------------------------------------
// One launch. Always through cudaLaunchKernelEx: PDL cannot be expressed with
// chevron syntax, and a single launch path (numAttrs = 0 when PDL is off or
// unsupported) keeps the two configurations comparable.
// ---------------------------------------------------------------------------
struct Params {
  void* out;
  const void* in;
  int64_t nvec;
  int ntiles;
  int vd;
  int vd_shift;
  int topk;
  int grid;
  int block;
  int pdl;
  bool desc;
  cudaStream_t stream;
};

template <typename K, typename... Args>
inline void launch_ex(K kern, int grid, int block, int pdl, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(grid < 1 ? 1 : grid, 1, 1);
  lc.blockDim = dim3(block, 1, 1);
  lc.dynamicSmemBytes = 0;
  lc.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = attr;
  lc.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, args...));
}

template <typename T, int TOPK_C, int BLOCK, bool VD_POW2, bool STAGED, bool DESC>
inline void launch_one(const Params& p) {
  launch_ex(moe_sum_vec_kernel<T, TOPK_C, BLOCK, VD_POW2, STAGED, DESC>, p.grid, BLOCK, p.pdl,
            p.stream, reinterpret_cast<uint4*>(p.out), reinterpret_cast<const uint4*>(p.in),
            p.nvec, p.ntiles, p.vd, p.vd_shift, p.topk, p.pdl);
}

// The one (dtype, topk, vd) combination every captured call takes. BLOCK keeps
// the two sizes worth re-checking -- 128 (the measured optimum, re-confirmed
// after PDL) and 256; 64/512/1024 were swept out in round 1 and are not
// carried. STAGED=0 stays reachable via MOE_STAGED=0 as a re-measure hook.
template <int BLOCK>
inline void launch_live_block(const Params& p, bool staged) {
  if (!staged) return launch_one<nv_bfloat16, 8, BLOCK, true, false, false>(p);
  if (p.desc) return launch_one<nv_bfloat16, 8, BLOCK, true, true, true>(p);
  return launch_one<nv_bfloat16, 8, BLOCK, true, true, false>(p);
}

inline void launch_live_bf16_topk8(const Params& p, bool staged) {
  if (p.block == 256) return launch_live_block<256>(p, staged);
  return launch_live_block<128>(p, staged);
}

// Paths no captured call takes: fixed BLOCK=128, ascending. Correctness and
// generality only.
template <typename T, bool VD_POW2>
inline void launch_generic(const Params& p) {
  switch (p.topk) {
    case 2:
      return launch_one<T, 2, 128, VD_POW2, true, false>(p);
    case 3:
      return launch_one<T, 3, 128, VD_POW2, true, false>(p);
    case 4:
      return launch_one<T, 4, 128, VD_POW2, true, false>(p);
    case 8:
      return launch_one<T, 8, 128, VD_POW2, true, false>(p);
    default:
      return launch_one<T, 0, 128, VD_POW2, true, false>(p);
  }
}

template <typename T>
void launch_vec(const Params& p, bool vd_pow2, bool staged) {
  if constexpr (std::is_same_v<T, nv_bfloat16>) {
    if (p.topk == 8 && vd_pow2) return launch_live_bf16_topk8(p, staged);
  }
  if (vd_pow2) return launch_generic<T, true>(p);
  return launch_generic<T, false>(p);
}

template <typename T>
void launch_scalar(const Params& p, int64_t n, int d, int64_t out_row_stride) {
  constexpr int BLOCK = 256;
  const int64_t nb = (n + BLOCK - 1) / BLOCK;
  const int64_t cap = static_cast<int64_t>(props().multiProcessorCount) * 16;
  const int grid = static_cast<int>(nb < cap ? nb : cap);
  launch_ex(moe_sum_scalar_kernel<T>, grid, BLOCK, p.pdl, p.stream, static_cast<T*>(p.out),
            static_cast<const T*>(p.in), n, d, static_cast<int>(p.topk), out_row_stride, p.pdl);
}

inline int log2_exact(int64_t v) {
  if (v <= 0 || (v & (v - 1)) != 0) return -1;
  int s = 0;
  while ((static_cast<int64_t>(1) << s) < v) ++s;
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry point.
//
// `input` is the raw 2-D [M*topk, D] tensor -- no host-side .view() needed.
// `output` is [M, D] and may be a row-slice of a larger cached buffer, so its
// row stride is honoured.
// ---------------------------------------------------------------------------
void moe_sum(const torch::Tensor& input_, const torch::Tensor& output, int64_t topk) {
  TORCH_CHECK(topk > 0, "moe_sum: topk must be positive, got ", topk);
  // Both kernels index the input as a dense [M, topk, D]; a non-contiguous
  // input has to be materialised first. Never taken by the captured shapes --
  // the check is one call that is already needed for `vec_ok`.
  const torch::Tensor input = input_.is_contiguous() ? input_ : input_.contiguous();

  const int64_t d = input.size(1);
  const int64_t m = input.size(0) / topk;
  const int64_t out_row_stride = output.stride(0);
  const Cfg& c = cfg();

  Params p = {};
  p.out = output.data_ptr();
  p.in = input.data_ptr();
  p.topk = static_cast<int>(topk);
  p.stream = at::cuda::getCurrentCUDAStream();
  p.desc = order_desc(static_cast<size_t>(input.numel()) * input.element_size(), c.desc);
  // PDL needs sm_90+ for griddepcontrol; without the wait the attribute is a
  // correctness bug rather than a slow path, so both sides are gated together.
  p.pdl = (c.pdl != 0 && props().major >= 9) ? 1 : 0;

  // uint4 fast path needs 16-byte-divisible rows (so row starts stay aligned
  // and the tensor is a flat uint4 array), 16-byte-aligned bases, and an
  // output whose row stride is exactly D. Every captured shape qualifies:
  // D=4096 bf16 is 8192 B per row, and torch allocations are 512-B aligned.
  const int64_t row_bytes = d * input.element_size();
  const bool vec_ok = (row_bytes % 16 == 0) && (out_row_stride == d) &&
                      (reinterpret_cast<uintptr_t>(p.in) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(p.out) % 16 == 0);

  if (vec_ok) {
    const int vd = static_cast<int>(row_bytes / 16);
    const int vd_shift = log2_exact(vd);
    const int64_t nvec = m * vd;
    const int64_t ntiles = (nvec + c.block - 1) / c.block;
    const int64_t cap = static_cast<int64_t>(props().multiProcessorCount) * c.waves;
    p.nvec = nvec;
    p.vd = vd;
    p.vd_shift = vd_shift;
    p.block = c.block;
    p.grid = static_cast<int>(ntiles < cap ? ntiles : cap);
    // The descending path indexes tiles in int; fall back to ascending in the
    // (unreachable for any real tensor) case that the tile count overflows.
    p.ntiles = static_cast<int>(ntiles);
    if (ntiles > INT32_MAX) p.desc = false;
    const bool staged = c.staged != 0;
    const bool vd_pow2 = vd_shift >= 0;
    switch (input.scalar_type()) {
      case at::ScalarType::BFloat16:
        launch_vec<nv_bfloat16>(p, vd_pow2, staged);
        return;
      case at::ScalarType::Half:
        launch_vec<nv_half>(p, vd_pow2, staged);
        return;
      case at::ScalarType::Float:
        launch_vec<float>(p, vd_pow2, staged);
        return;
      default:
        break;
    }
  }

  const int64_t n = m * d;
  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_scalar<nv_bfloat16>(p, n, static_cast<int>(d), out_row_stride);
      return;
    case at::ScalarType::Half:
      launch_scalar<nv_half>(p, n, static_cast<int>(d), out_row_stride);
      return;
    case at::ScalarType::Float:
      launch_scalar<float>(p, n, static_cast<int>(d), out_row_stride);
      return;
    default:
      TORCH_CHECK(false, "moe_sum: unsupported dtype ", input.scalar_type());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sum", &moe_sum, "MoE top-k sum reduction (vectorized streaming)");
}
