#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

template <typename freq_t>
__device__ __forceinline__ float load_freq(const freq_t* values, int offset);

template <>
__device__ __forceinline__ float load_freq(
    const float* values, int offset) {
  return values[offset];
}

template <>
__device__ __forceinline__ float load_freq(
    const half* values, int offset) {
  return __half2float(values[offset]);
}

template <typename freq_t>
__global__ void rotary_inplace_kernel(
    half2* __restrict__ qkv,
    const freq_t* __restrict__ cos,
    const freq_t* __restrict__ sin,
    int seq_len,
    int heads,
    int head_dim,
    int pairs) {
  const int head_pair = threadIdx.x;
  const int head = head_pair / pairs;
  const int pair = head_pair - head * pairs;
  const int row = blockIdx.x;
  const int seq = row % seq_len;

  const int qkv_row_stride = 3 * heads * head_dim / 2;
  const int qk_stride = heads * head_dim / 2;
  const int freq_offset = seq * (2 * pairs) + 2 * pair;
  const float c = load_freq(cos, freq_offset);
  const float s = load_freq(sin, freq_offset);
  const int base =
      row * qkv_row_stride + head * (head_dim / 2) + pair;
  #pragma unroll
  for (int q_or_k = 0; q_or_k < 2; ++q_or_k) {
    const int offset = base + q_or_k * qk_stride;
    const float2 values = __half22float2(qkv[offset]);
    qkv[offset] = __floats2half2_rn(
        values.x * c - values.y * s,
        values.y * c + values.x * s);
  }
}

void rotary_inplace(
    torch::Tensor qkv,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t heads) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(qkv.is_contiguous());
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda());
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type());

  const int batch = qkv.size(0);
  const int seq_len = qkv.size(1);
  const int head_dim = qkv.size(2) / (3 * heads);
  const int pairs = cos.size(-1) / 2;
  const int threads = heads * pairs;
  TORCH_CHECK(threads <= 1024);

  const c10::cuda::CUDAGuard device_guard(qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* qkv_ptr = reinterpret_cast<half2*>(qkv.data_ptr<at::Half>());
  if (cos.scalar_type() == at::ScalarType::Float) {
    rotary_inplace_kernel<<<batch * seq_len, threads, 0, stream>>>(
        qkv_ptr,
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        seq_len,
        heads,
        head_dim,
        pairs);
  } else {
    TORCH_CHECK(cos.scalar_type() == at::ScalarType::Half);
    rotary_inplace_kernel<<<batch * seq_len, threads, 0, stream>>>(
        qkv_ptr,
        reinterpret_cast<const half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin.data_ptr<at::Half>()),
        seq_len,
        heads,
        head_dim,
        pairs);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rotary_inplace", &rotary_inplace);
}
