// Fused NeoX rotary embedding for YaRNRotaryEmbedding (GPT-OSS).
//
// One launch rotates *both* query and key in place. The cos/sin cache is
// pre-cast to the compute dtype at module __init__, so this kernel reads the
// cache in the same dtype as query/key -- no per-call conversion of the (1 GiB
// fp32 / 512 MiB bf16) scaled cache.
//
// Layout assumptions (checked on the host, with a scalar fallback):
//   positions : int64  [num_tokens]
//   query     : T      [num_tokens, num_heads    * head_size], row stride q_stride
//   key       : T      [num_tokens, num_kv_heads * head_size], row stride k_stride
//   cache     : T      [max_pos, rot_dim] contiguous, rot_dim = 2 * embed_dim
//
// NeoX rotation, per head, for r in [0, embed_dim):
//   x' = x*cos[r] - y*sin[r]      (x = arr[r], y = arr[r + embed_dim])
//   y' = y*cos[r] + x*sin[r]
// Arithmetic is done in fp32 and rounded once on store, matching the vendored
// vLLM kernel bit-for-bit.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

// --------------------------------------------------------------------------
// Programmatic dependent launch (PDL).
//
// This op is launch-latency bound for every decode-sized shape: the whole
// kernel moves a few hundred KB, so the measured cost is dominated by the
// fixed GPU-side cost of *starting* a grid after the producer of ``positions``
// (or of the fused QKV slice) retires. Launching with
// ``cudaLaunchAttributeProgrammaticStreamSerialization`` lets the grid be set
// up while that producer is still running; ``griddepcontrol.wait`` then blocks
// until it has retired, so the kernel never observes pre-producer data.
//
// The wait sits at the very top, above *every* global load, which is what makes
// this safe for an arbitrary producer. It is a no-op when the grid was not
// launched with the attribute (pre-sm_90, or the chevron fallback below).
// --------------------------------------------------------------------------
__device__ __forceinline__ void dep_wait() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// --------------------------------------------------------------------------
// dtype <-> float helpers
// --------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_f<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);  // round-to-nearest-even, as static_cast does
}
template <>
__device__ __forceinline__ __half from_f<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ float from_f<float>(float v) {
  return v;
}

// 16-byte vector of VEC elements of T.
template <typename T, int VEC>
struct alignas(16) Vec {
  T v[VEC];
};

// --------------------------------------------------------------------------
// Vectorized kernel.
//
// A "unit" is VEC (= 16 bytes worth) consecutive rotation pairs of one head of
// one token; a token owns ``nq_units`` query units followed by ``nk_units`` key
// units, so query and key rotate in a single pass. Threads walk the *flat*
// (token, unit) space grid-strided, which (a) leaves no idle lanes -- 144 units
// per token is not a multiple of the warp size -- and (b) lets the host cap the
// grid at a couple of resident waves instead of launching one block per token
// (measured ~6% faster at 16k tokens; identical at 1-60 tokens, where the
// measurement is quantized by the ~2.05 us scheduler tick the launch itself
// occupies and so cannot resolve any difference -- see ITERATIONS.md).
//
// Units of a token are contiguous in ``g``, so ``positions[token]`` and the
// cos/sin row are warp-broadcast L1 hits rather than per-lane gathers.
// --------------------------------------------------------------------------
template <typename T, int VEC, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rope_vec_kernel(
    const int64_t* __restrict__ positions, T* __restrict__ query,
    T* __restrict__ key, const T* __restrict__ cache, const int rot_dim,
    const int embed_dim, const int head_size, const int nq_units,
    const int nk_units, const int64_t q_stride, const int64_t k_stride,
    const int num_tokens) {
  using V = Vec<T, VEC>;
  dep_wait();

  const int units_per_token = nq_units + nk_units;
  const int64_t total = static_cast<int64_t>(num_tokens) * units_per_token;

  for (int64_t g = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
       g < total; g += static_cast<int64_t>(gridDim.x) * BLOCK) {
    const int token = static_cast<int>(g / units_per_token);
    int unit = static_cast<int>(g - static_cast<int64_t>(token) * units_per_token);

    const int64_t pos = positions[token];
    const T* __restrict__ crow = cache + pos * static_cast<int64_t>(rot_dim);

    T* row;
    if (unit < nq_units) {
      row = query + static_cast<int64_t>(token) * q_stride;
    } else {
      row = key + static_cast<int64_t>(token) * k_stride;
      unit -= nq_units;
    }

    const int p0 = unit * VEC;              // first rotation pair of this unit
    const int head = p0 / embed_dim;
    const int r0 = p0 - head * embed_dim;

    T* xp = row + static_cast<int64_t>(head) * head_size + r0;
    T* yp = xp + embed_dim;

    const V xv = *reinterpret_cast<const V*>(xp);
    const V yv = *reinterpret_cast<const V*>(yp);
    const V cv = *reinterpret_cast<const V*>(crow + r0);
    const V sv = *reinterpret_cast<const V*>(crow + embed_dim + r0);

    V xo, yo;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float x = to_f<T>(xv.v[j]);
      const float y = to_f<T>(yv.v[j]);
      const float c = to_f<T>(cv.v[j]);
      const float s = to_f<T>(sv.v[j]);
      xo.v[j] = from_f<T>(x * c - y * s);
      yo.v[j] = from_f<T>(y * c + x * s);
    }
    *reinterpret_cast<V*>(xp) = xo;
    *reinterpret_cast<V*>(yp) = yo;
  }
}

// --------------------------------------------------------------------------
// Scalar fallback (any alignment / divisibility). One pair per thread-step.
// --------------------------------------------------------------------------
template <typename T>
__global__ void rope_scalar_kernel(const int64_t* __restrict__ positions,
                                   T* __restrict__ query, T* __restrict__ key,
                                   const T* __restrict__ cache,
                                   const int rot_dim, const int embed_dim,
                                   const int head_size, const int nq,
                                   const int nk, const int64_t q_stride,
                                   const int64_t k_stride) {
  dep_wait();
  const int token = blockIdx.x;
  const int64_t pos = positions[token];
  const T* __restrict__ crow = cache + pos * static_cast<int64_t>(rot_dim);

  T* qrow = query + static_cast<int64_t>(token) * q_stride;
  T* krow = key != nullptr ? key + static_cast<int64_t>(token) * k_stride
                           : nullptr;

  const int total = nq + nk;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    T* row;
    int p;
    if (i < nq) {
      row = qrow;
      p = i;
    } else {
      row = krow;
      p = i - nq;
    }
    const int head = p / embed_dim;
    const int r = p - head * embed_dim;
    T* xp = row + static_cast<int64_t>(head) * head_size + r;
    T* yp = xp + embed_dim;
    const float x = to_f<T>(*xp);
    const float y = to_f<T>(*yp);
    const float c = to_f<T>(crow[r]);
    const float s = to_f<T>(crow[embed_dim + r]);
    *xp = from_f<T>(x * c - y * s);
    *yp = from_f<T>(y * c + x * s);
  }
}

inline int round_up_warp(int n, int lo, int hi) {
  int b = (n + 31) / 32 * 32;
  if (b < lo) b = lo;
  if (b > hi) b = hi;
  return b;
}

// Threads per block of the vectorized kernel, and the grid cap in blocks per SM
// (12-32 measured indistinguishable at 16k tokens; below the cap the grid is
// exactly as large as the work needs).
constexpr int kVecBlock = 256;
constexpr int kBlocksPerSM = 16;

inline int sm_count() {
  static const int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

// ``griddepcontrol`` needs sm_90+; on anything older the attribute is rejected,
// so fall back to a plain chevron launch (where ``dep_wait`` compiles away).
inline bool pdl_supported() {
  static const bool ok = at::cuda::getCurrentDeviceProperties()->major >= 9;
  return ok;
}

// Launch *kern* with programmatic stream serialization when supported.
template <typename K, typename... Args>
inline void launch_maybe_pdl(K kern, int grid, int block, cudaStream_t stream,
                             Args... args) {
  if (pdl_supported()) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid, 1, 1);
    cfg.blockDim = dim3(block, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
  } else {
    kern<<<grid, block, 0, stream>>>(args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename T>
void launch(const torch::Tensor& positions, torch::Tensor& query,
            torch::Tensor* key, const torch::Tensor& cache, int64_t head_size,
            int num_tokens, int num_heads, int num_kv_heads, int rot_dim,
            int64_t q_stride, int64_t k_stride, cudaStream_t stream) {
  const int embed_dim = rot_dim / 2;
  constexpr int VEC = 16 / static_cast<int>(sizeof(T));

  T* qp = reinterpret_cast<T*>(query.data_ptr());
  T* kp = key != nullptr ? reinterpret_cast<T*>(key->data_ptr()) : nullptr;
  const T* cp = reinterpret_cast<const T*>(cache.data_ptr());

  const bool aligned =
      (embed_dim % VEC == 0) && (head_size % VEC == 0) &&
      (rot_dim % VEC == 0) && (q_stride % VEC == 0) &&
      (kp == nullptr || k_stride % VEC == 0) &&
      (reinterpret_cast<uintptr_t>(qp) % 16 == 0) &&
      (kp == nullptr || reinterpret_cast<uintptr_t>(kp) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(cp) % 16 == 0);

  if (aligned) {
    const int nq_units = num_heads * embed_dim / VEC;
    const int nk_units = kp != nullptr ? num_kv_heads * embed_dim / VEC : 0;
    const int64_t total =
        static_cast<int64_t>(num_tokens) * (nq_units + nk_units);
    int64_t grid = (total + kVecBlock - 1) / kVecBlock;
    const int64_t cap = static_cast<int64_t>(sm_count()) * kBlocksPerSM;
    if (grid > cap) grid = cap;
    launch_maybe_pdl(rope_vec_kernel<T, VEC, kVecBlock>, static_cast<int>(grid),
                     kVecBlock, stream, positions.const_data_ptr<int64_t>(), qp,
                     kp, cp, rot_dim, embed_dim, static_cast<int>(head_size),
                     nq_units, nk_units, q_stride, k_stride, num_tokens);
  } else {
    const int nq = num_heads * embed_dim;
    const int nk = kp != nullptr ? num_kv_heads * embed_dim : 0;
    const int block = round_up_warp(nq + nk, 32, 512);
    launch_maybe_pdl(rope_scalar_kernel<T>, num_tokens, block, stream,
                     positions.const_data_ptr<int64_t>(), qp, kp, cp, rot_dim,
                     embed_dim, static_cast<int>(head_size), nq, nk, q_stride,
                     k_stride);
  }
}

}  // namespace

// Fused in-place NeoX rotary embedding. ``query`` / ``key`` are 2D
// [num_tokens, num_heads * head_size]; the cache dtype must match query.
void yarn_rope(torch::Tensor positions, torch::Tensor query,
               std::optional<torch::Tensor> key, torch::Tensor cos_sin_cache,
               int64_t head_size) {
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D");
  TORCH_CHECK(positions.scalar_type() == torch::kLong, "positions must be int64");
  TORCH_CHECK(query.dim() == 2, "query must be 2D");
  TORCH_CHECK(query.stride(1) == 1, "query must have contiguous rows");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be 2D contiguous");
  TORCH_CHECK(cos_sin_cache.scalar_type() == query.scalar_type(),
              "cos_sin_cache dtype must match query");

  const int num_tokens = static_cast<int>(positions.numel());
  TORCH_CHECK(query.size(0) == num_tokens, "query/positions token mismatch");
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  TORCH_CHECK(rot_dim % 2 == 0 && rot_dim == head_size,
              "this kernel handles full-head rotary only");
  const int query_hidden = static_cast<int>(query.size(1));
  TORCH_CHECK(query_hidden % head_size == 0, "query hidden % head_size != 0");
  const int num_heads = query_hidden / static_cast<int>(head_size);

  int num_kv_heads = num_heads;
  int64_t k_stride = 0;
  torch::Tensor* kptr = nullptr;
  if (key.has_value()) {
    TORCH_CHECK(key->dim() == 2, "key must be 2D");
    TORCH_CHECK(key->stride(1) == 1, "key must have contiguous rows");
    TORCH_CHECK(key->size(0) == num_tokens, "key/positions token mismatch");
    TORCH_CHECK(key->scalar_type() == query.scalar_type(),
                "key dtype must match query");
    const int key_hidden = static_cast<int>(key->size(1));
    TORCH_CHECK(key_hidden % head_size == 0, "key hidden % head_size != 0");
    num_kv_heads = key_hidden / static_cast<int>(head_size);
    k_stride = key->stride(0);
    kptr = &key.value();
  }
  if (num_tokens == 0) return;

  const int64_t q_stride = query.stride(0);
  const c10::cuda::CUDAGuard guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (query.scalar_type()) {
    case torch::kBFloat16:
      launch<__nv_bfloat16>(positions, query, kptr, cos_sin_cache, head_size,
                            num_tokens, num_heads, num_kv_heads, rot_dim,
                            q_stride, k_stride, stream);
      break;
    case torch::kHalf:
      launch<__half>(positions, query, kptr, cos_sin_cache, head_size,
                     num_tokens, num_heads, num_kv_heads, rot_dim, q_stride,
                     k_stride, stream);
      break;
    case torch::kFloat:
      launch<float>(positions, query, kptr, cos_sin_cache, head_size,
                    num_tokens, num_heads, num_kv_heads, rot_dim, q_stride,
                    k_stride, stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype for yarn_rope");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("yarn_rope", &yarn_rope, "Fused NeoX YaRN rotary embedding (CUDA)",
        py::arg("positions"), py::arg("query"), py::arg("key"),
        py::arg("cos_sin_cache"), py::arg("head_size"));
}
