#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

template <typename input_t>
__device__ __forceinline__ half load_position(const input_t* positions, int index);

template <>
__device__ __forceinline__ half load_position(
    const float* positions, int index) {
  return __float2half_rn(positions[index]);
}

template <>
__device__ __forceinline__ half load_position(
    const half* positions, int index) {
  return positions[index];
}

template <typename input_t, int num_freqs>
__global__ void forward_freqs_kernel(
    half2* __restrict__ output,
    const input_t* __restrict__ positions,
    const half* __restrict__ freqs) {
  const int index = threadIdx.y * num_freqs + threadIdx.x;
  const half position = load_position(positions, threadIdx.y);
  const half value = __hmul(position, freqs[threadIdx.x]);
  output[index] = __halves2half2(value, value);
}

torch::Tensor forward_freqs(
    const torch::Tensor& positions,
    const torch::Tensor& freqs) {
  auto output_shape = positions.sizes().vec();
  const int num_freqs = freqs.size(-1);
  output_shape.push_back(2 * num_freqs);
  torch::Tensor output = torch::empty(output_shape, freqs.options());

  const int num_positions = positions.numel();
  const dim3 threads(num_freqs, num_positions);
  const c10::cuda::CUDAGuard device_guard(positions.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  half2* output_ptr = reinterpret_cast<half2*>(output.data_ptr<at::Half>());
  const half* freqs_ptr =
      reinterpret_cast<const half*>(freqs.data_ptr<at::Half>());
  if (positions.scalar_type() == at::ScalarType::Float) {
    if (num_freqs == 32) {
      forward_freqs_kernel<float, 32><<<1, threads, 0, stream>>>(
          output_ptr, positions.data_ptr<float>(), freqs_ptr);
    } else {
      forward_freqs_kernel<float, 16><<<1, threads, 0, stream>>>(
          output_ptr, positions.data_ptr<float>(), freqs_ptr);
    }
  } else if (num_freqs == 32) {
    forward_freqs_kernel<half, 32><<<1, threads, 0, stream>>>(
        output_ptr,
        reinterpret_cast<const half*>(positions.data_ptr<at::Half>()),
        freqs_ptr);
  } else {
    forward_freqs_kernel<half, 16><<<1, threads, 0, stream>>>(
        output_ptr,
        reinterpret_cast<const half*>(positions.data_ptr<at::Half>()),
        freqs_ptr);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward_freqs", &forward_freqs, "Fused Oasis frequency expansion");
}
