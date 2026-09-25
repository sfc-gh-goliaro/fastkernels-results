#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

namespace {

constexpr int kFrequencySize = 256;

__global__ void timestep_embedding_kernel(
    const int64_t* __restrict__ timesteps,
    const float* __restrict__ frequencies,
    float* __restrict__ embedding,
    int batch) {
  const int frequency_index = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= batch) {
    return;
  }

  const float frequency = frequencies[frequency_index];
  const float argument = static_cast<float>(timesteps[row]) * frequency;
  float sine;
  float cosine;
  sincosf(argument, &sine, &cosine);
  embedding[row * kFrequencySize + frequency_index] = cosine;
  embedding[row * kFrequencySize + frequency_index + kFrequencySize / 2] =
      sine;
}

torch::Tensor embedding(
    const torch::Tensor& timesteps,
    const torch::Tensor& frequencies) {
  const c10::cuda::CUDAGuard device_guard(timesteps.device());
  const int batch = static_cast<int>(timesteps.numel());
  torch::Tensor output =
      torch::empty({batch, kFrequencySize}, frequencies.options());
  timestep_embedding_kernel<<<
      batch, kFrequencySize / 2, 0, at::cuda::getCurrentCUDAStream()>>>(
      timesteps.const_data_ptr<int64_t>(),
      frequencies.const_data_ptr<float>(),
      output.mutable_data_ptr<float>(),
      batch);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("embedding", &embedding, "Oasis timestep embedding");
}
