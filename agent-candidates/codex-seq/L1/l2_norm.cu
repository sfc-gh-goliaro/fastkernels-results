#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ __launch_bounds__(256)
void l2_norm_1024(const float* __restrict__ input,
                  float* __restrict__ output,
                  float eps,
                  int rows) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) {
    return;
  }
  const float4* src =
      reinterpret_cast<const float4*>(input + static_cast<int64_t>(row) * 1024);
  float4 values[8];
  float sum_x = 0.0f;
  float sum_y = 0.0f;
  float sum_z = 0.0f;
  float sum_w = 0.0f;

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = src[lane + i * 32];
    sum_x = fmaf(values[i].x, values[i].x, sum_x);
    sum_y = fmaf(values[i].y, values[i].y, sum_y);
    sum_z = fmaf(values[i].z, values[i].z, sum_z);
    sum_w = fmaf(values[i].w, values[i].w, sum_w);
  }

  float sum = (sum_x + sum_y) + (sum_z + sum_w);
  sum = warp_sum(sum);
  float scale = 0.0f;
  if (lane == 0) {
    scale = rsqrtf(fmaxf(sum, eps * eps));
  }
  scale = __shfl_sync(0xffffffff, scale, 0);
  float4* dst =
      reinterpret_cast<float4*>(output + static_cast<int64_t>(row) * 1024);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float4 value = values[i];
    value.x *= scale;
    value.y *= scale;
    value.z *= scale;
    value.w *= scale;
    dst[lane + i * 32] = value;
  }
}

}  // namespace

void l2_norm(torch::Tensor input, torch::Tensor output, double eps) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int rows = input.numel() / 1024;
  l2_norm_1024<<<(rows + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
      static_cast<float>(eps), rows);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("l2_norm", &l2_norm);
}
