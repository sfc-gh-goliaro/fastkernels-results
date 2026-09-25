#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cfloat>

namespace {

constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kGroupSize = 16;

template <int Warps>
__global__ __launch_bounds__(Warps * 32) void topk8_bf16_128_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ weights,
    int* __restrict__ indices,
    int rows) {
  constexpr int kRowsPerBlock = Warps * 2;
  const int lane = threadIdx.x & 31;
  const int group_lane = lane & (kGroupSize - 1);
  const int row = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5) * 2 +
                  (lane >> 4);
  if (row >= rows) {
    return;
  }

  const unsigned group_mask = 0xffffu << ((lane >> 4) * 16);
  const int first_col = group_lane * 8;
  const uint4 packed =
      *reinterpret_cast<const uint4*>(input + row * kExperts + first_col);
  const __nv_bfloat16* packed_bf16 =
      reinterpret_cast<const __nv_bfloat16*>(&packed);

  float values[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = __bfloat162float(packed_bf16[i]);
  }

  float local_max = values[0];
  int local_index = first_col;
#pragma unroll
  for (int i = 1; i < 8; ++i) {
    if (values[i] > local_max) {
      local_max = values[i];
      local_index = first_col + i;
    }
  }

  float selected_logit = 0.0f;
  int selected_index = 0;
  float first_max = 0.0f;
#pragma unroll
  for (int rank = 0; rank < kTopK; ++rank) {
    float best = local_max;
    int best_index = local_index;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      const float other =
          __shfl_xor_sync(group_mask, best, offset, kGroupSize);
      const int other_index =
          __shfl_xor_sync(group_mask, best_index, offset, kGroupSize);
      if (other > best || (other == best && other_index < best_index)) {
        best = other;
        best_index = other_index;
      }
    }

    if (rank == 0) {
      first_max = best;
    }
    if (group_lane == rank) {
      selected_logit = best;
      selected_index = best_index;
    }

    if (rank + 1 < kTopK && best_index >= first_col &&
        best_index < first_col + 8) {
      values[best_index - first_col] = -FLT_MAX;
      local_max = values[0];
      local_index = first_col;
#pragma unroll
      for (int i = 1; i < 8; ++i) {
        if (values[i] > local_max) {
          local_max = values[i];
          local_index = first_col + i;
        }
      }
    }
  }

  const float weight =
      group_lane < kTopK ? __expf(selected_logit - first_max) : 0.0f;
  float selected_sum = weight;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    selected_sum +=
        __shfl_xor_sync(group_mask, selected_sum, offset, kGroupSize);
  }
  const float inv_sum = 1.0f / selected_sum;
  if (group_lane < kTopK) {
    weights[row * kTopK + group_lane] = weight * inv_sum;
    indices[row * kTopK + group_lane] = selected_index;
  }
}

}  // namespace

void topk_softmax_cuda(
    const torch::Tensor& input,
    torch::Tensor& weights,
    torch::Tensor& indices) {
  const int rows = static_cast<int>(input.size(0));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto* input_ptr =
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  auto* weights_ptr = static_cast<float*>(weights.data_ptr());
  auto* indices_ptr = static_cast<int*>(indices.data_ptr());

  if (rows <= 2) {
    topk8_bf16_128_kernel<1><<<1, 32, 0, stream>>>(
        input_ptr, weights_ptr, indices_ptr, rows);
  } else if (rows <= 1024) {
    constexpr int kWarps = 2;
    constexpr int kRowsPerBlock = kWarps * 2;
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    topk8_bf16_128_kernel<kWarps><<<blocks, kWarps * 32, 0, stream>>>(
        input_ptr, weights_ptr, indices_ptr, rows);
  } else {
    constexpr int kWarps = 4;
    constexpr int kRowsPerBlock = kWarps * 2;
    const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    topk8_bf16_128_kernel<kWarps><<<blocks, kWarps * 32, 0, stream>>>(
        input_ptr, weights_ptr, indices_ptr, rows);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &topk_softmax_cuda);
}
