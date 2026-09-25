#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

namespace {

template <int kPacksPerThread>
__global__ __launch_bounds__(256)
void concat_half_packed(const uint4* __restrict__ x,
                        const uint4* __restrict__ y,
                        uint4* __restrict__ output,
                        int x_packs,
                        int y_packs) {
  const int out_packs = x_packs + y_packs;
  const int batch = blockIdx.y;
  const int first =
      blockIdx.x * blockDim.x * kPacksPerThread + threadIdx.x;

#pragma unroll
  for (int item = 0; item < kPacksPerThread; ++item) {
    const int inner = first + item * blockDim.x;
    if (inner >= out_packs) {
      continue;
    }
    const int output_index = batch * out_packs + inner;
    if (inner < x_packs) {
      output[output_index] = x[batch * x_packs + inner];
    } else {
      output[output_index] = y[batch * y_packs + inner - x_packs];
    }
  }
}

}  // namespace

torch::Tensor concat_half(const torch::Tensor& x, const torch::Tensor& y) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty(
      {x.size(0), x.size(1) + y.size(1), x.size(2), x.size(3)}, x.options());

  constexpr int kThreads = 256;
  constexpr int kHalfPerPack = sizeof(uint4) / sizeof(at::Half);
  const int batch = x.size(0);
  const int x_packs = x.numel() / batch / kHalfPerPack;
  const int y_packs = y.numel() / batch / kHalfPerPack;
  constexpr int kPacksPerThread = 1;
  const int packs_per_block = kThreads * kPacksPerThread;
  const int blocks = (x_packs + y_packs + packs_per_block - 1) /
                     packs_per_block;
  const dim3 grid(blocks, batch);

  concat_half_packed<kPacksPerThread>
      <<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(x.const_data_ptr()),
      reinterpret_cast<const uint4*>(y.const_data_ptr()),
      reinterpret_cast<uint4*>(output.mutable_data_ptr()), x_packs, y_packs);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("concat_half", &concat_half);
}
