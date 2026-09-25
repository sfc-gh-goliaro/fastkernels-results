#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

struct alignas(8) BFloat16x4 {
  __nv_bfloat162 xy;
  __nv_bfloat162 zw;
};

__global__ __launch_bounds__(128)
void moe_sum_bf16x4_kernel(
    const BFloat16x4* __restrict__ input,
    BFloat16x4* __restrict__ output) {
  constexpr int vectors_per_token = 4096 / 4;
  const int token = blockIdx.x >> 3;
  const int col = (blockIdx.x & 7) * blockDim.x + threadIdx.x;
  const int token_stride = 8 * vectors_per_token;
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
  for (int expert = 0; expert < 8; ++expert) {
    const BFloat16x4 value =
        input[token * token_stride + expert * vectors_per_token + col];
    sum.x += __bfloat162float(value.xy.x);
    sum.y += __bfloat162float(value.xy.y);
    sum.z += __bfloat162float(value.zw.x);
    sum.w += __bfloat162float(value.zw.y);
  }

  BFloat16x4 result;
  result.xy = __floats2bfloat162_rn(sum.x, sum.y);
  result.zw = __floats2bfloat162_rn(sum.z, sum.w);
  output[token * vectors_per_token + col] = result;
}

void moe_sum_bf16x4(const torch::Tensor& input, const torch::Tensor& output) {
  const int num_tokens = input.size(0) / 8;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  moe_sum_bf16x4_kernel<<<num_tokens * 8, 128, 0, stream>>>(
      reinterpret_cast<const BFloat16x4*>(input.data_ptr()),
      reinterpret_cast<BFloat16x4*>(output.data_ptr()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sum_bf16x4", &moe_sum_bf16x4);
}
