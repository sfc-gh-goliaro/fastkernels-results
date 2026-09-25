#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <optional>

namespace {

constexpr int kThreads = 128;
constexpr int kThreadsPerHead = 8;
constexpr int kVecsPerThread = 2;
constexpr unsigned kSubgroupMask = 0xffffffffu;

template <bool WriteLse>
__global__ __launch_bounds__(kThreads) void merge_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ prefix_output,
    const float* __restrict__ prefix_lse,
    const __nv_bfloat16* __restrict__ suffix_output,
    const float* __restrict__ suffix_lse,
    float* __restrict__ output_lse,
    int num_tokens) {
  const int global_thread = blockIdx.x * kThreads + threadIdx.x;
  const int total_threads = num_tokens * 16 * kThreadsPerHead;
  if (global_thread >= total_threads) return;
  const int token_head = global_thread / kThreadsPerHead;
  const int subgroup_lane = threadIdx.x & (kThreadsPerHead - 1);
  const int token = token_head >> 4;
  const int head = token_head & 15;

  float prefix_scale = 0.0f;
  float suffix_scale = 0.0f;
  float merged_lse = 0.0f;

  if (subgroup_lane == 0) {
    const int lse_idx = head * num_tokens + token;
    float p_lse = prefix_lse[lse_idx];
    float s_lse = suffix_lse[lse_idx];
    const float neg_inf = __int_as_float(0xff800000);
    if (isinf(p_lse)) p_lse = neg_inf;
    if (isinf(s_lse)) s_lse = neg_inf;
    const float max_lse = fmaxf(p_lse, s_lse);
    if (!isinf(max_lse)) {
      const float p_se = expf(p_lse - max_lse);
      const float s_se = expf(s_lse - max_lse);
      const float out_se = p_se + s_se;
      prefix_scale = p_se / out_se;
      suffix_scale = s_se / out_se;
      if constexpr (WriteLse) {
        merged_lse = logf(out_se) + max_lse;
      }
    } else {
      prefix_scale = -1.0f;
      if constexpr (WriteLse) {
        merged_lse = max_lse;
      }
    }
    if constexpr (WriteLse) {
      output_lse[lse_idx] = merged_lse;
    }
  }

  prefix_scale = __shfl_sync(kSubgroupMask, prefix_scale, 0, kThreadsPerHead);
  suffix_scale = __shfl_sync(kSubgroupMask, suffix_scale, 0, kThreadsPerHead);
  const __nv_bfloat162 prefix_scale2 =
      __float2bfloat162_rn(prefix_scale);
  const __nv_bfloat162 suffix_scale2 =
      __float2bfloat162_rn(suffix_scale);

#pragma unroll
  for (int vec = 0; vec < kVecsPerThread; ++vec) {
    const int pack_idx =
        token_head * 16 + vec * kThreadsPerHead + subgroup_lane;
    const uint4 p_pack =
        reinterpret_cast<const uint4*>(prefix_output)[pack_idx];
    if (prefix_scale < 0.0f) {
      reinterpret_cast<uint4*>(output)[pack_idx] = p_pack;
      continue;
    }
    const uint4 s_pack =
        reinterpret_cast<const uint4*>(suffix_output)[pack_idx];

    uint4 out_pack;
    const __nv_bfloat162* p =
        reinterpret_cast<const __nv_bfloat162*>(&p_pack);
    const __nv_bfloat162* s =
        reinterpret_cast<const __nv_bfloat162*>(&s_pack);
    __nv_bfloat162* out = reinterpret_cast<__nv_bfloat162*>(&out_pack);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[i] = __hfma2(p[i], prefix_scale2,
                       __hmul2(s[i], suffix_scale2));
    }
    reinterpret_cast<uint4*>(output)[pack_idx] = out_pack;
  }
}

void merge(
    at::Tensor& output,
    const at::Tensor& prefix_output,
    const at::Tensor& prefix_lse,
    const at::Tensor& suffix_output,
    const at::Tensor& suffix_lse,
    std::optional<at::Tensor> output_lse) {
  const int num_tokens = output.size(0);
  const int num_threads = num_tokens * 16 * kThreadsPerHead;
  const dim3 grid((num_threads + kThreads - 1) / kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  float* out_lse_ptr =
      output_lse ? output_lse->data_ptr<float>() : nullptr;

  if (out_lse_ptr) {
    merge_kernel<true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(prefix_output.data_ptr()),
        prefix_lse.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(suffix_output.data_ptr()),
        suffix_lse.data_ptr<float>(), out_lse_ptr, num_tokens);
  } else {
    merge_kernel<false><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(prefix_output.data_ptr()),
        prefix_lse.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(suffix_output.data_ptr()),
        suffix_lse.data_ptr<float>(), nullptr, num_tokens);
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge", &merge);
}
