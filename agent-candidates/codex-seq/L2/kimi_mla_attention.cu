#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int kHidden = 512;
constexpr int kVec = 8;
constexpr int kThreads = kHidden / kVec;

struct alignas(16) Bf16x8 {
  __nv_bfloat16 value[kVec];
};

__global__ void strided_rms_norm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int64_t row_stride,
    float eps) {
  const auto* input_vec = reinterpret_cast<const Bf16x8*>(
      input + static_cast<int64_t>(blockIdx.x) * row_stride);
  const auto* weight_vec = reinterpret_cast<const Bf16x8*>(weight);
  auto* output_vec = reinterpret_cast<Bf16x8*>(
      output + static_cast<int64_t>(blockIdx.x) * kHidden);

  const Bf16x8 x = input_vec[threadIdx.x];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float value = __bfloat162float(x.value[i]);
    sum += value * value;
  }

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  sum = BlockReduce(reduce_storage).Sum(sum);
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(sum / kHidden + eps);
  }
  __syncthreads();

  const Bf16x8 scale = weight_vec[threadIdx.x];
  Bf16x8 result;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float value = __bfloat162float(x.value[i]);
    const float w = __bfloat162float(scale.value[i]);
    result.value[i] = __float2bfloat16(value * inverse_rms * w);
  }
  output_vec[threadIdx.x] = result;
}

__global__ void pack_k_kernel(
    const __nv_bfloat16* __restrict__ k_nope,
    const __nv_bfloat16* __restrict__ k_pe,
    __nv_bfloat16* __restrict__ output,
    int heads,
    int heads_per_block,
    int64_t nope_row_stride,
    int64_t nope_head_stride,
    int64_t pe_row_stride) {
  constexpr int kPairsPerHead = 96;
  const int head_groups =
      (heads + heads_per_block - 1) / heads_per_block;
  const int row = blockIdx.x / head_groups;
  const int first_head = (blockIdx.x % head_groups) * heads_per_block;
  const auto* pe = reinterpret_cast<const __nv_bfloat162*>(
      k_pe + static_cast<int64_t>(row) * pe_row_stride);

  for (int item = threadIdx.x;
       item < heads_per_block * kPairsPerHead;
       item += blockDim.x) {
    const int head = first_head + item / kPairsPerHead;
    const int pair = item % kPairsPerHead;
    if (head < heads) {
      const auto* nope = reinterpret_cast<const __nv_bfloat162*>(
          k_nope + static_cast<int64_t>(row) * nope_row_stride +
          static_cast<int64_t>(head) * nope_head_stride);
      auto* out = reinterpret_cast<__nv_bfloat162*>(
          output + (static_cast<int64_t>(row) * heads + head) * 192);
      out[pair] = pair < 64 ? nope[pair] : pe[pair - 64];
    }
  }
}

torch::Tensor strided_rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda());
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHidden);
  TORCH_CHECK(input.stride(1) == 1);
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == kHidden);

  auto output = torch::empty(
      {input.size(0), kHidden}, input.options());
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  strided_rms_norm_kernel<<<input.size(0), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      input.stride(0),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor pack_k(
    const torch::Tensor& k_nope,
    const torch::Tensor& k_pe) {
  TORCH_CHECK(k_nope.is_cuda() && k_pe.is_cuda());
  TORCH_CHECK(k_nope.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(k_pe.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      k_nope.dim() == 3 && k_nope.size(2) == 128 &&
      k_pe.dim() == 3 && k_pe.size(1) == 1 && k_pe.size(2) == 64 &&
      k_nope.size(0) == k_pe.size(0));
  TORCH_CHECK(k_nope.stride(2) == 1 && k_pe.stride(2) == 1);

  const int64_t rows = k_nope.size(0);
  const int heads = k_nope.size(1);
  auto output = torch::empty({rows, heads, 192}, k_nope.options());
  const c10::cuda::CUDAGuard device_guard(k_nope.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int heads_per_block = rows >= 1024 ? 4 : 1;
  const int threads = heads_per_block == 4 ? 256 : 64;
  const int head_groups = (heads + heads_per_block - 1) / heads_per_block;
  pack_k_kernel<<<rows * head_groups, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_pe.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      heads,
      heads_per_block,
      k_nope.stride(0),
      k_nope.stride(1),
      k_pe.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("strided_rms_norm", &strided_rms_norm);
  module.def("pack_k", &pack_k);
}
