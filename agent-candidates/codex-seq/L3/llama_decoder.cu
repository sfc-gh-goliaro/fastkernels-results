#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

struct alignas(16) Bf16x8 {
  __nv_bfloat16 data[8];
};

__global__ void single_token_value_kernel(
    Bf16x8* __restrict__ output,
    const Bf16x8* __restrict__ value,
    int output_vectors,
    int head_vectors,
    int heads_per_kv) {
  for (int i = threadIdx.x; i < output_vectors; i += blockDim.x) {
    const int head = i / head_vectors;
    const int offset = i % head_vectors;
    const int kv_head = head / heads_per_kv;
    output[i] = value[kv_head * head_vectors + offset];
  }
}

template <int BlockSize>
__global__ void rmsnorm_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon,
    int rows) {
  constexpr int kHidden = 4096;
  constexpr int kWidth = 8;
  constexpr int kVectors = kHidden / kWidth;
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const auto* input_v = reinterpret_cast<const Bf16x8*>(
      input + static_cast<int64_t>(row) * kHidden);
  const auto* weight_v = reinterpret_cast<const Bf16x8*>(weight);
  auto* output_v =
      reinterpret_cast<Bf16x8*>(output + static_cast<int64_t>(row) * kHidden);

  float variance = 0.0f;
  for (int i = threadIdx.x; i < kVectors; i += BlockSize) {
    const Bf16x8 value = input_v[i];
#pragma unroll
    for (int j = 0; j < kWidth; j += 2) {
      const float2 f = __bfloat1622float2(
          __nv_bfloat162{value.data[j], value.data[j + 1]});
      variance += f.x * f.x + f.y * f.y;
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  variance = BlockReduce(reduce_storage).Reduce(
      variance, cuda::std::plus<>{}, BlockSize);

  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(variance / kHidden + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kVectors; i += BlockSize) {
    const Bf16x8 value = input_v[i];
    const Bf16x8 scale = weight_v[i];
    Bf16x8 normalized;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      const float x = __bfloat162float(value.data[j]);
      const float w = __bfloat162float(scale.data[j]);
      normalized.data[j] = __float2bfloat16(x * inverse_rms * w);
    }
    output_v[i] = normalized;
  }
}

template <int BlockSize>
__global__ void fused_add_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    float epsilon,
    int rows) {
  constexpr int kHidden = 4096;
  constexpr int kWidth = 8;
  constexpr int kVectors = kHidden / kWidth;
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  auto* input_v =
      reinterpret_cast<Bf16x8*>(input + static_cast<int64_t>(row) * kHidden);
  auto* residual_v = reinterpret_cast<Bf16x8*>(
      residual + static_cast<int64_t>(row) * kHidden);
  const auto* weight_v = reinterpret_cast<const Bf16x8*>(weight);

  float variance = 0.0f;
  for (int i = threadIdx.x; i < kVectors; i += BlockSize) {
    Bf16x8 value = input_v[i];
    const Bf16x8 skip = residual_v[i];
#pragma unroll
    for (int j = 0; j < kWidth; j += 2) {
      __nv_bfloat162 pair{value.data[j], value.data[j + 1]};
      pair += __nv_bfloat162{skip.data[j], skip.data[j + 1]};
      value.data[j] = pair.x;
      value.data[j + 1] = pair.y;
      const float2 f = __bfloat1622float2(pair);
      variance += f.x * f.x + f.y * f.y;
    }
    residual_v[i] = value;
  }

  using BlockReduce = cub::BlockReduce<float, BlockSize>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  variance = BlockReduce(reduce_storage).Reduce(
      variance, cuda::std::plus<>{}, BlockSize);

  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    inverse_rms = rsqrtf(variance / kHidden + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kVectors; i += BlockSize) {
    const Bf16x8 value = residual_v[i];
    const Bf16x8 scale = weight_v[i];
    Bf16x8 output;
#pragma unroll
    for (int j = 0; j < kWidth; ++j) {
      const float x = __bfloat162float(value.data[j]);
      const float w = __bfloat162float(scale.data[j]);
      output.data[j] = __float2bfloat16(x * inverse_rms * w);
    }
    input_v[i] = output;
  }
}

void fused_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double epsilon) {
  TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight.is_cuda());
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(residual.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous());
  TORCH_CHECK(weight.is_contiguous() && input.size(-1) == 4096);

  const int rows = input.numel() / 4096;
  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  auto* x = reinterpret_cast<__nv_bfloat16*>(input.data_ptr());
  auto* r = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr());
  const auto* w =
      reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr());
  if (rows < 256) {
    fused_add_rmsnorm_kernel<512><<<rows, 512, 0, stream>>>(
        x, r, w, static_cast<float>(epsilon), rows);
  } else {
    fused_add_rmsnorm_kernel<256><<<rows, 256, 0, stream>>>(
        x, r, w, static_cast<float>(epsilon), rows);
  }
}

torch::Tensor rmsnorm(
    torch::Tensor input,
    torch::Tensor weight,
    double epsilon) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda());
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(input.size(-1) == 4096);

  torch::Tensor output = torch::empty_like(input);
  const int rows = input.numel() / 4096;
  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const auto* x =
      reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr());
  const auto* w =
      reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr());
  auto* y = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  if (rows < 256) {
    rmsnorm_kernel<512><<<rows, 512, 0, stream>>>(
        y, x, w, static_cast<float>(epsilon), rows);
  } else {
    rmsnorm_kernel<256><<<rows, 256, 0, stream>>>(
        y, x, w, static_cast<float>(epsilon), rows);
  }
  return output;
}

torch::Tensor single_token_value(
    torch::Tensor value,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  TORCH_CHECK(value.is_cuda());
  TORCH_CHECK(value.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(head_dim % 8 == 0 && num_heads % num_kv_heads == 0);

  torch::Tensor output = torch::empty(
      {1, num_heads * head_dim}, value.options());
  const c10::cuda::CUDAGuard guard(value.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(value.get_device()).stream();
  const int output_vectors = num_heads * head_dim / 8;
  const int head_vectors = head_dim / 8;
  single_token_value_kernel<<<1, 256, 0, stream>>>(
      reinterpret_cast<Bf16x8*>(output.data_ptr()),
      reinterpret_cast<const Bf16x8*>(value.const_data_ptr()),
      output_vectors,
      head_vectors,
      num_heads / num_kv_heads);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rmsnorm", &rmsnorm);
  module.def("fused_add_rmsnorm", &fused_add_rmsnorm);
  module.def("single_token_value", &single_token_value);
}
