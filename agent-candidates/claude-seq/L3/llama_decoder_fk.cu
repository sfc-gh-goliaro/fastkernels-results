// Small kernels for the L3 llama_decoder fast path.
//
// Both are *bit-exact* re-implementations of the reference ops rather than
// faster-but-different ones: this layer is chaotic in its last bit, and
// perturbing a single one of the 245k elements of the first norm's output by one
// bf16 ulp already moves 1.5% of the layer's output past the scorer's 1%
// tolerance.  The goal here is the reference's exact arithmetic at a lower
// launch cost, not a different (even more accurate) formula.
//
// * ``fused_add_rmsnorm`` reproduces vLLM's ``fused_add_rms_norm_kernel``
//   width-8 specialization: bf16 residual add, per-thread sum of squares in the
//   same pairwise order, ``cub::BlockReduce<float, N>``'s reduction order
//   (per-warp ``shfl_down`` tree with offsets 1,2,4,8,16, then a serial sum over
//   the per-warp aggregates), the same ``rsqrtf(var/hidden + eps)`` and the same
//   ``(x * s) * w`` multiply order -- but keeps the row in registers instead of
//   re-reading the residual it just wrote, and picks the block size the
//   reference's launcher would have picked so the reduction tree matches.
// * ``rope`` reproduces ``rotary_embedding_kernel``'s NeoX path -- the same fp32
//   ``x*cos - y*sin`` over the same bf16-rounded cos/sin table, written as the
//   same expression so nvcc contracts it into the same FMA -- with one block per
//   (token, 4 heads) so each thread owns exactly one rotation pair, and consumes
//   Q and K in place inside the fused QKV buffer.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace fkdec {

constexpr unsigned kFull = 0xffffffffu;

// -- scalar / packed conversions, matching vLLM's _typeConvert ---------------
template <typename T>
struct Tr;
template <>
struct Tr<__nv_bfloat16> {
  using P = __nv_bfloat162;
  static __device__ __forceinline__ float2 tof(P v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ P fromf(float2 v) { return __float22bfloat162_rn(v); }
  static __device__ __forceinline__ float tof1(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 fromf1(float v) { return __float2bfloat16(v); }
  static __device__ __forceinline__ P add(P a, P b) { return __hadd2(a, b); }
};
template <>
struct Tr<__half> {
  using P = __half2;
  static __device__ __forceinline__ float2 tof(P v) { return __half22float2(v); }
  static __device__ __forceinline__ P fromf(float2 v) { return __float22half2_rn(v); }
  static __device__ __forceinline__ float tof1(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half fromf1(float v) { return __float2half(v); }
  static __device__ __forceinline__ P add(P a, P b) { return __hadd2(a, b); }
};

template <typename P>
struct alignas(16) Vec8 {
  P d[4];
};

// cub::BlockReduce<float, N, BLOCK_REDUCE_WARP_REDUCTIONS>::Reduce order.
__device__ __forceinline__ float block_sum(float v, float* smem, int nwarps) {
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) v += __shfl_down_sync(kFull, v, off);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = smem[0];
    for (int w = 1; w < nwarps; ++w) s += smem[w];
    smem[32] = s;
  }
  __syncthreads();
  return smem[32];
}

template <typename T, int CHUNKS>
__global__ void fused_add_rms_kernel(T* __restrict__ input,
                                     const int64_t vec_in_stride,
                                     T* __restrict__ residual,
                                     const T* __restrict__ weight,
                                     const float eps, const int hidden,
                                     const int vec_hidden) {
  using P = typename Tr<T>::P;
  using V = Vec8<P>;
  __shared__ float smem[33];
  V* __restrict__ iv = reinterpret_cast<V*>(input);
  V* __restrict__ rv = reinterpret_cast<V*>(residual);
  const V* __restrict__ wv = reinterpret_cast<const V*>(weight);
  const int64_t ibase = (int64_t)blockIdx.x * vec_in_stride;
  const int64_t rbase = (int64_t)blockIdx.x * vec_hidden;

  V keep[CHUNKS];
  float variance = 0.0f;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int idx = threadIdx.x + c * blockDim.x;
    if (idx < vec_hidden) {
      V t = iv[ibase + idx];
      const V r = rv[rbase + idx];
#pragma unroll
      for (int i = 0; i < 4; ++i) t.d[i] = Tr<T>::add(t.d[i], r.d[i]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 z = Tr<T>::tof(t.d[i]);
        variance += z.x * z.x + z.y * z.y;
      }
      rv[rbase + idx] = t;
      keep[c] = t;
    }
  }
  variance = block_sum(variance, smem, blockDim.x >> 5);
  const float s = rsqrtf(variance / hidden + eps);
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int idx = threadIdx.x + c * blockDim.x;
    if (idx < vec_hidden) {
      const V w = wv[idx];
      V o;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 x = Tr<T>::tof(keep[c].d[i]);
        const float2 wf = Tr<T>::tof(w.d[i]);
        float2 r;
        r.x = x.x * s * wf.x;
        r.y = x.y * s * wf.y;
        o.d[i] = Tr<T>::fromf(r);
      }
      iv[ibase + idx] = o;
    }
  }
}

#define FK_LAUNCH_NORM(T, CH)                                                 \
  fused_add_rms_kernel<T, CH><<<grid, block, 0, stream>>>(                    \
      reinterpret_cast<T*>(input.data_ptr()), in_stride / 8,                  \
      reinterpret_cast<T*>(residual.data_ptr()),                              \
      reinterpret_cast<const T*>(weight.data_ptr()),                          \
      static_cast<float>(eps), hidden, vec_hidden)

void fused_add_rmsnorm(at::Tensor input, at::Tensor residual, at::Tensor weight,
                       double eps) {
  TORCH_CHECK(input.is_contiguous() || input.stride(-1) == 1);
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(input.scalar_type() == residual.scalar_type() &&
              input.scalar_type() == weight.scalar_type(),
              "fused_add_rmsnorm: dtype mismatch");
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == input.size(-1));
  const int hidden = static_cast<int>(input.size(-1));
  const int64_t in_stride = input.stride(-2);
  const int num_tokens = static_cast<int>(input.numel() / hidden);
  TORCH_CHECK(hidden % 8 == 0 && in_stride % 8 == 0);
  const int vec_hidden = hidden / 8;
  // Same block size the reference launcher picks -- the reduction tree (and so
  // the last bits of the variance) depends on it.
  const int max_block = (num_tokens < 256) ? 1024 : 256;
  int nthreads = hidden < max_block ? hidden : max_block;
  // The reference launches ``min(hidden, max_block)`` threads but only
  // ``hidden/8`` of them ever hold a value; the rest contribute exact zeros to
  // the reduction.  Dropping those warps leaves the reduction tree -- and so
  // every rounded output element -- identical while halving the block.
  if (nthreads > vec_hidden) nthreads = ((vec_hidden + 31) / 32) * 32;
  if (nthreads < 32) nthreads = 32;
  dim3 grid(num_tokens);
  dim3 block(nthreads);
  const c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int chunks = (vec_hidden + nthreads - 1) / nthreads;
  TORCH_CHECK(chunks <= 8, "fused_add_rmsnorm: hidden size too large");
  const auto st = input.scalar_type();
  if (st == at::kBFloat16) {
    if (chunks == 1) {
      FK_LAUNCH_NORM(__nv_bfloat16, 1);
    } else if (chunks == 2) {
      FK_LAUNCH_NORM(__nv_bfloat16, 2);
    } else if (chunks <= 4) {
      FK_LAUNCH_NORM(__nv_bfloat16, 4);
    } else {
      FK_LAUNCH_NORM(__nv_bfloat16, 8);
    }
  } else {
    TORCH_CHECK(st == at::kHalf, "fused_add_rmsnorm: bf16/fp16 only");
    if (chunks == 1) {
      FK_LAUNCH_NORM(__half, 1);
    } else if (chunks == 2) {
      FK_LAUNCH_NORM(__half, 2);
    } else if (chunks <= 4) {
      FK_LAUNCH_NORM(__half, 4);
    } else {
      FK_LAUNCH_NORM(__half, 8);
    }
  }
}

// -- NeoX RoPE in place on the fused QKV buffer ------------------------------
template <typename T, int HALF, int HPB>
__global__ void rope_kernel(T* __restrict__ qkv, const int64_t row_stride,
                            const int64_t* __restrict__ pos,
                            const T* __restrict__ cache, const int nheads) {
  const int tok = blockIdx.x;
  const int h = blockIdx.y * HPB + (int)(threadIdx.x / HALF);
  if (h >= nheads) return;
  const int i = threadIdx.x % HALF;
  const T* cs = cache + pos[tok] * (2 * HALF);
  T* row = qkv + (int64_t)tok * row_stride + (int64_t)h * (2 * HALF);
  const float c = Tr<T>::tof1(cs[i]);
  const float s = Tr<T>::tof1(cs[HALF + i]);
  const float x = Tr<T>::tof1(row[i]);
  const float y = Tr<T>::tof1(row[HALF + i]);
  row[i] = Tr<T>::fromf1(x * c - y * s);
  row[HALF + i] = Tr<T>::fromf1(y * c + x * s);
}

void rope(at::Tensor qkv, at::Tensor positions, at::Tensor cache,
          int64_t num_heads) {
  const int m = static_cast<int>(qkv.size(0));
  const int hd = static_cast<int>(cache.size(-1));
  const int half = hd / 2;
  TORCH_CHECK(qkv.stride(1) == 1 && cache.is_contiguous());
  TORCH_CHECK(positions.scalar_type() == at::kLong);
  TORCH_CHECK(qkv.scalar_type() == cache.scalar_type());
  constexpr int kHPB = 4;
  const int nh = static_cast<int>(num_heads);
  dim3 grid(m, static_cast<unsigned>((nh + kHPB - 1) / kHPB));
  const c10::cuda::CUDAGuard guard(qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
#define FK_LAUNCH_ROPE(T, H)                                                  \
  rope_kernel<T, H, kHPB><<<grid, H * kHPB, 0, stream>>>(                     \
      reinterpret_cast<T*>(qkv.data_ptr()), qkv.stride(0),                    \
      positions.data_ptr<int64_t>(),                                          \
      reinterpret_cast<const T*>(cache.data_ptr()), nh)
  if (qkv.scalar_type() == at::kBFloat16) {
    if (half == 64) {
      FK_LAUNCH_ROPE(__nv_bfloat16, 64);
    } else {
      TORCH_CHECK(half == 32, "rope: head_dim 64 or 128");
      FK_LAUNCH_ROPE(__nv_bfloat16, 32);
    }
  } else {
    if (half == 64) {
      FK_LAUNCH_ROPE(__half, 64);
    } else {
      TORCH_CHECK(half == 32, "rope: head_dim 64 or 128");
      FK_LAUNCH_ROPE(__half, 32);
    }
  }
#undef FK_LAUNCH_ROPE
}

}  // namespace fkdec

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_add_rmsnorm", &fkdec::fused_add_rmsnorm,
        "fused add + RMSNorm (bit-exact with the reference)");
  m.def("rope", &fkdec::rope, "in-place NeoX RoPE on a fused QKV buffer");
}
