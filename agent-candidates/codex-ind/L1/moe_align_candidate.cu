#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

namespace {

constexpr int kExperts = 128;

__device__ __forceinline__ int warp_atomic_rank(int* ptr, int expert) {
  const unsigned active = __activemask();
  const unsigned peers = __match_any_sync(active, expert);
  const int leader = __ffs(peers) - 1;
  const int lane = threadIdx.x & 31;
  int base = 0;
  if (lane == leader) {
    base = atomicAdd(ptr, __popc(peers));
  }
  base = __shfl_sync(peers, base, leader);
  return base + __popc(peers & ((1u << lane) - 1));
}

__global__ __launch_bounds__(1024, 1) void prepare_128_kernel(
    const int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_pad,
    int* __restrict__ cursors,
    int numel,
    int block_size,
    int max_padded) {
  if (blockIdx.x == 1) {
    const int4 fill = make_int4(numel, numel, numel, numel);
    int4* sorted4 = reinterpret_cast<int4*>(sorted_token_ids);
    for (int i = threadIdx.x; i < max_padded / 4; i += blockDim.x) {
      sorted4[i] = fill;
    }
    return;
  }

  __shared__ int counts[kExperts];
  __shared__ int prefix[kExperts + 1];
  __shared__ int warp_sums[4];
  const int tid = threadIdx.x;

  if (tid < kExperts) {
    counts[tid] = 0;
  }
  __syncthreads();

  for (int i = tid; i < numel; i += blockDim.x) {
    atomicAdd(&counts[topk_ids[i]], 1);
  }
  __syncthreads();

  int padded = 0;
  int inclusive = 0;
  if (tid < kExperts) {
    padded = ((counts[tid] + block_size - 1) / block_size) * block_size;
    inclusive = padded;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if ((tid & 31) >= offset) {
        inclusive += other;
      }
    }
    if ((tid & 31) == 31) {
      warp_sums[tid >> 5] = inclusive;
    }
  }
  __syncthreads();

  if (tid < 32) {
    const int value = tid < 4 ? warp_sums[tid] : 0;
    int scan = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, scan, offset);
      if (tid >= offset) {
        scan += other;
      }
    }
    if (tid < 4) {
      warp_sums[tid] = scan - value;
    }
  }
  __syncthreads();

  if (tid < kExperts) {
    const int start = warp_sums[tid >> 5] + inclusive - padded;
    prefix[tid] = start;
    cursors[tid] = start;
    if (tid == kExperts - 1) {
      prefix[kExperts] = start + padded;
      *num_tokens_post_pad = start + padded;
    }
  }
  __syncthreads();

  if (tid < kExperts) {
    for (int i = prefix[tid]; i < prefix[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid;
    }
  }
}

constexpr int kSortThreads = 256;
constexpr int kItems = 4;
constexpr int kTile = kSortThreads * kItems;

__global__ __launch_bounds__(kSortThreads) void sort_128_kernel(
    const int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ cursors,
    int numel) {
  __shared__ int counts[kExperts];
  __shared__ int bases[kExperts];
  const int tid = threadIdx.x;
  int experts[kItems];
  bool valid[kItems];

  if (tid < kExperts) {
    counts[tid] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int index = blockIdx.x * kTile + item * kSortThreads + tid;
    valid[item] = index < numel;
    experts[item] = valid[item] ? topk_ids[index] : kExperts + tid;
    if (valid[item]) {
      warp_atomic_rank(&counts[experts[item]], experts[item]);
    }
  }
  __syncthreads();

  if (tid < kExperts) {
    bases[tid] = atomicAdd(&cursors[tid], counts[tid]);
    counts[tid] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int index = blockIdx.x * kTile + item * kSortThreads + tid;
    if (valid[item]) {
      const int rank = warp_atomic_rank(
          &counts[experts[item]], experts[item]);
      sorted_token_ids[bases[experts[item]] + rank] = index;
    }
  }
}

}  // namespace

void moe_align_128(
    torch::Tensor topk_ids,
    int64_t block_size,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor cursors) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int numel = static_cast<int>(topk_ids.numel());
  prepare_128_kernel<<<2, 1024, 0, stream>>>(
      topk_ids.data_ptr<int>(),
      sorted_token_ids.data_ptr<int>(),
      expert_ids.data_ptr<int>(),
      num_tokens_post_pad.data_ptr<int>(),
      cursors.data_ptr<int>(),
      numel,
      static_cast<int>(block_size),
      static_cast<int>(sorted_token_ids.numel()));
  sort_128_kernel<<<
      (numel + kTile - 1) / kTile, kSortThreads, 0, stream>>>(
      topk_ids.data_ptr<int>(),
      sorted_token_ids.data_ptr<int>(),
      cursors.data_ptr<int>(),
      numel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_128", &moe_align_128);
}
