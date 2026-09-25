#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda/functional>
#include <torch/extension.h>

namespace {

struct alignas(16) BFloat16x8 {
  __nv_bfloat16 values[8];
};

__global__ void rms_norm_4096_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon) {
  constexpr int kHidden = 4096;
  constexpr int kVectorWidth = 8;
  constexpr int kVectors = kHidden / kVectorWidth;

  const auto* input_vec = reinterpret_cast<const BFloat16x8*>(
      input + static_cast<int64_t>(blockIdx.x) * kHidden);
  float variance = 0.0f;
  for (int index = threadIdx.x; index < kVectors; index += blockDim.x) {
    BFloat16x8 value = input_vec[index];
#pragma unroll
    for (int element = 0; element < kVectorWidth; ++element) {
      float x = __bfloat162float(value.values[element]);
      variance += x * x;
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  variance = BlockReduce(reduction_storage).Reduce(
      variance, cuda::std::plus<>(), blockDim.x);

  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(variance / kHidden + epsilon);
  }
  __syncthreads();

  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * kHidden;
  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    float x = __bfloat162float(input[row_offset + column]);
    float scale = __bfloat162float(weight[column]);
    output[row_offset + column] = __float2bfloat16(x * inverse_rms * scale);
  }
}

__global__ void fused_add_rms_norm_4096_kernel(
    __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon) {
  constexpr int kHidden = 4096;
  constexpr int kVectorWidth = 8;
  constexpr int kVectors = kHidden / kVectorWidth;
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * kHidden;
  auto* input_vec = reinterpret_cast<BFloat16x8*>(input + row_offset);
  auto* residual_vec = reinterpret_cast<BFloat16x8*>(residual + row_offset);

  float variance = 0.0f;
  for (int index = threadIdx.x; index < kVectors; index += blockDim.x) {
    BFloat16x8 x = input_vec[index];
    BFloat16x8 r = residual_vec[index];
#pragma unroll
    for (int element = 0; element < kVectorWidth; ++element) {
      float sum = __bfloat162float(x.values[element]) +
                  __bfloat162float(r.values[element]);
      r.values[element] = __float2bfloat16(sum);
      float rounded_sum = __bfloat162float(r.values[element]);
      variance += rounded_sum * rounded_sum;
    }
    residual_vec[index] = r;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  variance = BlockReduce(reduction_storage).Reduce(
      variance, cuda::std::plus<>(), blockDim.x);

  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(variance / kHidden + epsilon);
  }
  __syncthreads();

  for (int column = threadIdx.x; column < kHidden; column += blockDim.x) {
    float x = __bfloat162float(residual[row_offset + column]);
    float scale = __bfloat162float(weight[column]);
    input[row_offset + column] = __float2bfloat16(x * inverse_rms * scale);
  }
}

}  // namespace

torch::Tensor rms_norm_4096(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double epsilon) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::BFloat16 &&
          weight.scalar_type() == at::ScalarType::BFloat16,
      "expected bfloat16 tensors");
  TORCH_CHECK(input.size(-1) == 4096, "expected hidden size 4096");

  auto output = torch::empty_like(input);
  const int64_t rows = input.numel() / 4096;
  const c10::cuda::CUDAGuard device_guard(input.device());
  rms_norm_4096_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<float>(epsilon));
  return output;
}

void fused_add_rms_norm_4096(
    torch::Tensor input,
    torch::Tensor residual,
    const torch::Tensor& weight,
    double epsilon) {
  TORCH_CHECK(
      input.is_cuda() && residual.is_cuda() && weight.is_cuda(),
      "expected CUDA tensors");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::BFloat16 &&
          residual.scalar_type() == at::ScalarType::BFloat16 &&
          weight.scalar_type() == at::ScalarType::BFloat16,
      "expected bfloat16 tensors");
  TORCH_CHECK(
      input.size(-1) == 4096 && residual.size(-1) == 4096,
      "expected hidden size 4096");

  const int64_t rows = input.numel() / 4096;
  const c10::cuda::CUDAGuard device_guard(input.device());
  fused_add_rms_norm_4096_kernel<<<
      rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<float>(epsilon));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rms_norm_4096", &rms_norm_4096);
  module.def("fused_add_rms_norm_4096", &fused_add_rms_norm_4096);
}
