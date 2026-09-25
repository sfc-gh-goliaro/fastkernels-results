#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value);

template <>
__device__ __forceinline__ float to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value);

template <>
__device__ __forceinline__ __half from_float(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float silu_approx(float x) {
  const float x2 = x * x;
  const float even =
      x2 * fmaf(x2, fmaf(x2, 0.0004331403f, -0.0138038741f),
                0.2395166094f);
  const float central = fmaf(0.5f, x, even);
  return fmaxf(-0.28f, fminf(central, fmaxf(x, 0.0f)));
}

template <typename scalar_t>
struct alignas(16) Pack {
  scalar_t values[8];
};

template <typename scalar_t, int kPacksPerThread>
__global__ void silu_vector_kernel(const scalar_t* input, scalar_t* output,
                                   int64_t num_packs) {
  const auto* input_packs = reinterpret_cast<const Pack<scalar_t>*>(input);
  auto* output_packs = reinterpret_cast<Pack<scalar_t>*>(output);
  const int64_t first =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

#pragma unroll
  for (int item = 0; item < kPacksPerThread; ++item) {
    const int64_t index = first + item * stride;
    if (index >= num_packs) {
      continue;
    }
    Pack<scalar_t> values = input_packs[index];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      values.values[lane] =
          from_float<scalar_t>(silu_approx(to_float(values.values[lane])));
    }
    output_packs[index] = values;
  }
}

template <typename scalar_t>
__global__ void silu_scalar_kernel(const scalar_t* input, scalar_t* output,
                                   int64_t n) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = from_float<scalar_t>(silu_approx(to_float(input[index])));
  }
}

template <typename scalar_t>
void launch_silu(const torch::Tensor& input, torch::Tensor& output,
                 cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int64_t n = input.numel();
  const auto* input_ptr =
      reinterpret_cast<const scalar_t*>(input.const_data_ptr());
  auto* output_ptr = reinterpret_cast<scalar_t*>(output.mutable_data_ptr());

  if ((n & 7) == 0) {
    const int64_t num_packs = n / 8;
    if (n >= 4 * 1024 * 1024) {
      const int blocks =
          static_cast<int>((num_packs + kThreads * 8 - 1) / (kThreads * 8));
      silu_vector_kernel<scalar_t, 8>
          <<<blocks, kThreads, 0, stream>>>(input_ptr, output_ptr, num_packs);
    } else {
      const int blocks =
          static_cast<int>((num_packs + kThreads - 1) / kThreads);
      silu_vector_kernel<scalar_t, 1>
          <<<blocks, kThreads, 0, stream>>>(input_ptr, output_ptr, num_packs);
    }
  } else {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    silu_scalar_kernel<<<blocks, kThreads, 0, stream>>>(input_ptr, output_ptr,
                                                        n);
  }
}

torch::Tensor silu(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "only float16 and bfloat16 are supported");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  if (input.numel() == 0) {
    return output;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (input.scalar_type() == at::kHalf) {
    launch_silu<__half>(input, output, stream);
  } else {
    launch_silu<__nv_bfloat16>(input, output, stream);
  }
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("silu", &silu, "Vectorized SiLU approximation");
}
