// Online-softmax merge of two attention partitions (section 2.2 of
// https://www.arxiv.org/pdf/2501.01005), tuned for the captured shapes:
// [num_tokens, num_heads, head_size] bf16 with head_size = 128, num_heads = 16.
//
// The reference kernel is purely bandwidth-bound in principle, but spends a
// large slice of its time on index math: it derives (token, head, pack) with
// four integer divisions/modulos by *runtime* values, which ptxas expands into
// ~25 instructions each -- roughly 100 extra instructions per warp for 16 bytes
// of payload per lane. It also recomputes the softmax weights with two expf's
// and two divisions.
//
// The fast path here:
//   * decodes the flat 16-byte-pack index with shifts/masks when
//     packs-per-head and num_heads are powers of two (they are: 16 and 16),
//   * derives both merge weights from a single approximate exp plus one
//     reciprocal, using the overflow-safe 1/(1+exp(s-p)) form so no max
//     subtraction is needed,
//   * gives each lane PACKS_PER_LANE *adjacent* packs. Because that count
//     divides packs-per-head they all belong to the same (token, head), so one
//     LSE pair and one exp cover 32 bytes instead of 16. Two is the optimum:
//     the lanes of a warp then stride 32 B, which still lands inside the
//     sectors the paired instruction reads, while four or more spread a warp
//     over too many sectors and cost far more than the arithmetic saved,
//   * addresses the (contiguous) payload as a flat uint4 array, so the pack
//     index *is* the address, and
//   * loads the two payload streams with evict-first (streaming) hints: they
//     are never re-read, so keeping them out of L2 leaves the small LSE tables
//     resident instead. This is worth ~4 us on the 16384-token shape; the store
//     policy, in contrast, measured neutral, so stores stay plain write-back.
//
// Non-power-of-two head counts / packs-per-head still take this kernel, just
// with a divide instead of a shift. A generic kernel mirroring the reference
// semantics covers the rest: FP8 output, arbitrary head strides and
// prefill_tokens_with_context.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>
#include <optional>
#include <type_traits>

namespace {

// ---------------------------------------------------------------------------
// scalar <-> float conversion
// ---------------------------------------------------------------------------
template <typename T>
struct Cvt;
template <>
struct Cvt<float> {
  __device__ __forceinline__ static float to(float v) { return v; }
  __device__ __forceinline__ static float from(float v) { return v; }
};
template <>
struct Cvt<__half> {
  __device__ __forceinline__ static float to(__half v) { return __half2float(v); }
  __device__ __forceinline__ static __half from(float v) { return __float2half(v); }
};
template <>
struct Cvt<__nv_bfloat16> {
  __device__ __forceinline__ static float to(__nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ __forceinline__ static __nv_bfloat16 from(float v) {
    return __float2bfloat16(v);
  }
};

// ---------------------------------------------------------------------------
// Softmax merge weights.
//
// w_p = exp(p) / (exp(p) + exp(s)) = 1 / (1 + exp(s - p)), which needs no max
// subtraction: s - p = +inf saturates the reciprocal to 0 and s - p = -inf to
// 1. The reference folds an infinite LSE (either sign) to -inf first, and falls
// back to "emit the prefix" when both are -inf; both are reproduced here.
// ---------------------------------------------------------------------------
struct Weights {
  float p_lse, s_lse, w_p, w_s;
};

__device__ __forceinline__ Weights merge_weights(float p, float s) {
  if (fabsf(p) == CUDART_INF_F) p = -CUDART_INF_F;
  if (fabsf(s) == CUDART_INF_F) s = -CUDART_INF_F;
  float w = __frcp_rn(1.0f + __expf(s - p));
  if (fmaxf(p, s) == -CUDART_INF_F) w = 1.0f;  // both -inf: copy the prefix
  return {p, s, w, 1.0f - w};
}

__device__ __forceinline__ float merged_lse(const Weights& w) {
  const float m = fmaxf(w.p_lse, w.s_lse);
  if (m == -CUDART_INF_F) return -CUDART_INF_F;
  return m + log1pf(expf(-fabsf(w.s_lse - w.p_lse)));
}

// ---------------------------------------------------------------------------
// Fast path: contiguous [n, h, d], output dtype == input dtype, every token
// merged. PPL adjacent 16-byte packs per lane, all from one head.
// ---------------------------------------------------------------------------
template <typename scalar_t, bool WRITE_LSE, bool POW2, int NT, int PPL>
__global__ __launch_bounds__(NT) void merge_fast_kernel(
    uint4* __restrict__ output, float* __restrict__ output_lse,
    const uint4* __restrict__ prefix_output,
    const float* __restrict__ prefix_lse,
    const uint4* __restrict__ suffix_output,
    const float* __restrict__ suffix_lse, const uint32_t num_tokens,
    const uint32_t num_heads, const uint32_t heads_shift,
    const uint32_t packs_per_head, const uint32_t pph_shift,
    const uint32_t total_packs) {
  constexpr uint32_t kPack = 16u / sizeof(scalar_t);
  const uint32_t idx = PPL * (blockIdx.x * NT + threadIdx.x);
  // PPL divides packs_per_head, which divides total_packs, so one bound check
  // covers the whole group and the tail is exact.
  if (idx >= total_packs) return;

  uint32_t token_head, pack, token, head;
  if (POW2) {
    token_head = idx >> pph_shift;
    pack = idx & (packs_per_head - 1u);
    token = token_head >> heads_shift;
    head = token_head & (num_heads - 1u);
  } else {
    token_head = idx / packs_per_head;
    pack = idx - token_head * packs_per_head;
    token = token_head / num_heads;
    head = token_head - token * num_heads;
  }
  const uint32_t li = head * num_tokens + token;

  const Weights w = merge_weights(__ldg(prefix_lse + li), __ldg(suffix_lse + li));

  uint4 pv[PPL], sv[PPL];
#pragma unroll
  for (int k = 0; k < PPL; ++k) {
    pv[k] = __ldcs(prefix_output + idx + k);
    sv[k] = __ldcs(suffix_output + idx + k);
  }
#pragma unroll
  for (int k = 0; k < PPL; ++k) {
    uint4 ov;
    const scalar_t* pe = reinterpret_cast<const scalar_t*>(&pv[k]);
    const scalar_t* se = reinterpret_cast<const scalar_t*>(&sv[k]);
    scalar_t* oe = reinterpret_cast<scalar_t*>(&ov);
#pragma unroll
    for (uint32_t i = 0; i < kPack; ++i) {
      oe[i] = Cvt<scalar_t>::from(
          __fmaf_rn(Cvt<scalar_t>::to(pe[i]), w.w_p, Cvt<scalar_t>::to(se[i]) * w.w_s));
    }
    output[idx + k] = ov;
  }

  if (WRITE_LSE && pack == 0) output_lse[li] = merged_lse(w);
}

// ---------------------------------------------------------------------------
// Generic path: arbitrary head strides, FP8 output, prefill_tokens_with_context.
// One pack per thread, mirroring the reference kernel.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename output_t, bool USE_FP8>
__global__ void merge_generic_kernel(
    output_t* __restrict__ output, float* __restrict__ output_lse,
    const scalar_t* __restrict__ prefix_output,
    const float* __restrict__ prefix_lse,
    const scalar_t* __restrict__ suffix_output,
    const float* __restrict__ suffix_lse, const uint32_t num_tokens,
    const uint32_t num_heads, const uint32_t head_size,
    const uint32_t src_head_stride, const uint32_t dst_head_stride,
    const uint32_t prefix_num_tokens, const uint32_t total_threads,
    const float* __restrict__ output_scale, const float fp8_max,
    const bool write_lse) {
  constexpr uint32_t kPack = 16u / sizeof(scalar_t);
  using out_pack_t = std::conditional_t<
      USE_FP8, std::conditional_t<sizeof(scalar_t) == 4, uint32_t, uint2>, uint4>;

  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_threads) return;

  const uint32_t packs_per_head = head_size / kPack;
  const uint32_t token_head = gid / packs_per_head;
  const uint32_t pack = gid - token_head * packs_per_head;
  const uint32_t token = token_head / num_heads;
  const uint32_t head = token_head - token * num_heads;
  const uint32_t li = head * num_tokens + token;

  const scalar_t* pp =
      prefix_output + (token * num_heads + head) * src_head_stride + pack * kPack;
  const scalar_t* sp =
      suffix_output + (token * num_heads + head) * src_head_stride + pack * kPack;
  output_t* op =
      output + (token * num_heads + head) * dst_head_stride + pack * kPack;

  const float scale_inv = USE_FP8 ? 1.0f / *output_scale : 1.0f;

  float w_p, w_s, out_lse_val;
  if (token >= prefix_num_tokens) {  // suffix-only tail
    w_p = 0.0f;
    w_s = 1.0f;
    out_lse_val = suffix_lse[li];
  } else {
    const Weights w = merge_weights(prefix_lse[li], suffix_lse[li]);
    w_p = w.w_p;
    w_s = w.w_s;
    out_lse_val = merged_lse(w);
  }

  const uint4 pv = *reinterpret_cast<const uint4*>(pp);
  const uint4 sv = *reinterpret_cast<const uint4*>(sp);
  const scalar_t* pe = reinterpret_cast<const scalar_t*>(&pv);
  const scalar_t* se = reinterpret_cast<const scalar_t*>(&sv);

  out_pack_t ov;
  output_t* oe = reinterpret_cast<output_t*>(&ov);
#pragma unroll
  for (uint32_t i = 0; i < kPack; ++i) {
    // w_p == 0 means "copy the suffix": take it verbatim so a non-finite
    // prefix value cannot poison the result through 0 * inf.
    const float merged = (w_p == 0.0f)
                             ? Cvt<scalar_t>::to(se[i])
                             : __fmaf_rn(Cvt<scalar_t>::to(pe[i]), w_p,
                                         Cvt<scalar_t>::to(se[i]) * w_s);
    if (USE_FP8) {
      const float q = fmaxf(-fp8_max, fminf(merged * scale_inv, fp8_max));
      reinterpret_cast<__nv_fp8_storage_t*>(oe)[i] =
          __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    } else {
      reinterpret_cast<scalar_t*>(oe)[i] = Cvt<scalar_t>::from(merged);
    }
  }
  *reinterpret_cast<out_pack_t*>(op) = ov;

  if (write_lse && pack == 0) output_lse[li] = out_lse_val;
}

// ---------------------------------------------------------------------------
// Host launch
// ---------------------------------------------------------------------------
constexpr int kFastThreads = 256;
constexpr int kFastPacksPerLane = 2;

inline uint32_t pow2_shift(uint32_t v) {  // 0 when v is not a power of two
  return (v && (v & (v - 1u)) == 0u) ? static_cast<uint32_t>(__builtin_ctz(v)) : 0u;
}
inline bool is_pow2(uint32_t v) { return v && (v & (v - 1u)) == 0u; }

template <typename scalar_t>
void launch_fast(void* output, float* output_lse, const void* prefix_output,
                 const float* prefix_lse, const void* suffix_output,
                 const float* suffix_lse, uint32_t num_tokens,
                 uint32_t num_heads, uint32_t head_size, cudaStream_t stream) {
  constexpr uint32_t kPack = 16u / sizeof(scalar_t);
  const uint32_t packs_per_head = head_size / kPack;
  const uint32_t total_packs = num_tokens * num_heads * packs_per_head;
  const bool pow2 = is_pow2(packs_per_head) && is_pow2(num_heads);
  const uint32_t pph_shift = pow2_shift(packs_per_head);
  const uint32_t heads_shift = pow2_shift(num_heads);

  // Each lane takes PPL adjacent packs from one head; that only works when PPL
  // divides packs_per_head.
  const int ppl = (packs_per_head % kFastPacksPerLane == 0) ? kFastPacksPerLane : 1;

#define FK_LAUNCH_FAST(WLSE, POW2, PPL)                                       \
  merge_fast_kernel<scalar_t, WLSE, POW2, kFastThreads, PPL>                  \
      <<<dim3((total_packs / (PPL) + kFastThreads - 1) / kFastThreads),        \
         kFastThreads, 0, stream>>>(                                          \
          reinterpret_cast<uint4*>(output), output_lse,                       \
          reinterpret_cast<const uint4*>(prefix_output), prefix_lse,          \
          reinterpret_cast<const uint4*>(suffix_output), suffix_lse,          \
          num_tokens, num_heads, heads_shift, packs_per_head, pph_shift,      \
          total_packs)
#define FK_LAUNCH_FAST_PPL(WLSE, POW2)                                        \
  if (ppl == kFastPacksPerLane) {                                             \
    FK_LAUNCH_FAST(WLSE, POW2, kFastPacksPerLane);                            \
  } else {                                                                    \
    FK_LAUNCH_FAST(WLSE, POW2, 1);                                            \
  }

  if (output_lse != nullptr) {
    if (pow2) { FK_LAUNCH_FAST_PPL(true, true); }
    else      { FK_LAUNCH_FAST_PPL(true, false); }
  } else {
    if (pow2) { FK_LAUNCH_FAST_PPL(false, true); }
    else      { FK_LAUNCH_FAST_PPL(false, false); }
  }
#undef FK_LAUNCH_FAST_PPL
#undef FK_LAUNCH_FAST
}

template <typename scalar_t>
void launch_generic(void* output, float* output_lse, const void* prefix_output,
                    const float* prefix_lse, const void* suffix_output,
                    const float* suffix_lse, uint32_t num_tokens,
                    uint32_t num_heads, uint32_t head_size,
                    uint32_t src_head_stride, uint32_t dst_head_stride,
                    uint32_t prefix_num_tokens, const float* output_scale,
                    float fp8_max, bool use_fp8, cudaStream_t stream) {
  constexpr uint32_t kPack = 16u / sizeof(scalar_t);
  constexpr int kThreads = 128;
  const uint32_t total = num_tokens * num_heads * (head_size / kPack);
  const dim3 grid((total + kThreads - 1) / kThreads);
  const bool wlse = output_lse != nullptr;
  if (use_fp8) {
    merge_generic_kernel<scalar_t, uint8_t, true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<uint8_t*>(output), output_lse,
        reinterpret_cast<const scalar_t*>(prefix_output), prefix_lse,
        reinterpret_cast<const scalar_t*>(suffix_output), suffix_lse, num_tokens,
        num_heads, head_size, src_head_stride, dst_head_stride,
        prefix_num_tokens, total, output_scale, fp8_max, wlse);
  } else {
    merge_generic_kernel<scalar_t, scalar_t, false><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<scalar_t*>(output), output_lse,
        reinterpret_cast<const scalar_t*>(prefix_output), prefix_lse,
        reinterpret_cast<const scalar_t*>(suffix_output), suffix_lse, num_tokens,
        num_heads, head_size, src_head_stride, dst_head_stride,
        prefix_num_tokens, total, output_scale, fp8_max, wlse);
  }
}

bool natural_layout(const torch::Tensor& t, int64_t num_heads, int64_t head_size) {
  return t.stride(2) == 1 && t.stride(1) == head_size &&
         t.stride(0) == num_heads * head_size;
}

}  // namespace

void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse,
                       std::optional<int64_t> prefill_tokens_with_context,
                       const std::optional<torch::Tensor>& output_scale) {
  TORCH_CHECK(prefix_output.dim() == 3 && suffix_output.dim() == 3 &&
                  output.dim() == 3,
              "merge_attn_states expects [num_tokens, num_heads, head_size]");
  TORCH_CHECK(prefix_output.stride(1) == suffix_output.stride(1),
              "prefix_output and suffix_output must share the head stride");

  const auto in_dtype = prefix_output.scalar_type();
  const bool use_fp8 = output_scale.has_value();
  if (use_fp8) {
    TORCH_CHECK(output.scalar_type() == torch::kFloat8_e4m3fn,
                "output must be float8_e4m3fn when output_scale is provided");
  } else {
    TORCH_CHECK(output.scalar_type() == in_dtype,
                "output dtype must match prefix_output dtype");
  }

  const uint32_t num_tokens = static_cast<uint32_t>(output.size(0));
  const uint32_t num_heads = static_cast<uint32_t>(output.size(1));
  const uint32_t head_size = static_cast<uint32_t>(output.size(2));
  if (num_tokens == 0 || num_heads == 0 || head_size == 0) return;

  const uint32_t prefix_num_tokens =
      prefill_tokens_with_context.has_value()
          ? static_cast<uint32_t>(*prefill_tokens_with_context)
          : num_tokens;
  TORCH_CHECK(prefix_num_tokens <= num_tokens,
              "prefill_tokens_with_context must be <= num_tokens");

  float* output_lse_ptr =
      output_lse.has_value() ? output_lse->data_ptr<float>() : nullptr;
  const float* output_scale_ptr =
      use_fp8 ? output_scale->data_ptr<float>() : nullptr;

  const c10::cuda::CUDAGuard guard(prefix_output.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const uint32_t src_head_stride = static_cast<uint32_t>(prefix_output.stride(1));
  const uint32_t dst_head_stride = static_cast<uint32_t>(output.stride(1));
  const bool contiguous =
      !use_fp8 && prefix_num_tokens == num_tokens &&
      natural_layout(prefix_output, num_heads, head_size) &&
      natural_layout(suffix_output, num_heads, head_size) &&
      natural_layout(output, num_heads, head_size);

  const uint32_t pack =
      (in_dtype == torch::kFloat32) ? 4u : 8u;  // 16 bytes per load
  TORCH_CHECK(head_size % pack == 0, "head_size must be a multiple of ", pack);

#define FK_DISPATCH(scalar_t)                                                  \
  if (contiguous) {                                                            \
    launch_fast<scalar_t>(output.data_ptr(), output_lse_ptr,                    \
                          prefix_output.data_ptr(),                            \
                          prefix_lse.data_ptr<float>(),                        \
                          suffix_output.data_ptr(),                            \
                          suffix_lse.data_ptr<float>(), num_tokens, num_heads, \
                          head_size, stream);                                  \
  } else {                                                                     \
    launch_generic<scalar_t>(                                                  \
        output.data_ptr(), output_lse_ptr, prefix_output.data_ptr(),            \
        prefix_lse.data_ptr<float>(), suffix_output.data_ptr(),                \
        suffix_lse.data_ptr<float>(), num_tokens, num_heads, head_size,         \
        src_head_stride, dst_head_stride, prefix_num_tokens, output_scale_ptr,  \
        448.0f, use_fp8, stream);                                              \
  }

  switch (in_dtype) {
    case torch::kFloat32: { FK_DISPATCH(float); break; }
    case torch::kFloat16: { FK_DISPATCH(__half); break; }
    case torch::kBFloat16: { FK_DISPATCH(__nv_bfloat16); break; }
    default:
      TORCH_CHECK(false, "unsupported prefix_output dtype ", in_dtype);
  }
#undef FK_DISPATCH
}

// Minimal entry point used by the hot path: fewer arguments to unpack per call.
void merge(torch::Tensor& output, std::optional<torch::Tensor> output_lse,
           const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse,
           const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse) {
  merge_attn_states(output, std::move(output_lse), prefix_output, prefix_lse,
                    suffix_output, suffix_lse, std::nullopt, std::nullopt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge", &merge, "merge_attn_states (hot path)");
  m.def("merge_attn_states", &merge_attn_states, "merge_attn_states (full)",
        py::arg("output"), py::arg("output_lse"), py::arg("prefix_output"),
        py::arg("prefix_lse"), py::arg("suffix_output"), py::arg("suffix_lse"),
        py::arg("prefill_tokens_with_context") = std::nullopt,
        py::arg("output_scale") = std::nullopt);
}
