// In-place GELU(tanh) for the FLUX FFN's inner activation.
//
// The FFN's intermediate ``h = x W1^T + b1`` is a private temporary of
// ``FeedForward.forward``, so the activation can overwrite it instead of
// allocating a second [M, 4*dim] buffer.  Same traffic as an out-of-place pass
// but half the L2 footprint (the lines the GEMM just wrote are the lines the
// next GEMM reads back) and one allocation less per call.
//
// Numerics are bit-identical to the frozen L1 GELU: 0.5x(1 + tanh(P(x))) with
// P(x) = x(c1 + c3 x^2) evaluated in fp32 with one hardware tanh.approx.f32.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <torch/extension.h>

#define DEVI __device__ __forceinline__

DEVI float tanh_ap(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

DEVI float gelu_tanh16(float x) {
  const float c1 = 0.7978845608f;            // sqrt(2/pi)
  const float c3 = 0.0356774081f;            // 0.044715 * sqrt(2/pi)
  float p = __fmaf_rn(c3, x * x, c1) * x;
  float h = 0.5f * x;
  return __fmaf_rn(h, tanh_ap(p), h);
}

template <typename T> struct Pair;
template <> struct Pair<__half> { using type = __half2; };
template <> struct Pair<__nv_bfloat16> { using type = __nv_bfloat162; };
DEVI float2 cvt2f(const __half2 v) { return __half22float2(v); }
DEVI float2 cvt2f(const __nv_bfloat162 v) { return __bfloat1622float2(v); }
DEVI void cvt2t(__half2& d, const float2 v) { d = __float22half2_rn(v); }
DEVI void cvt2t(__nv_bfloat162& d, const float2 v) { d = __float22bfloat162_rn(v); }
DEVI float to_f(const __half v) { return __half2float(v); }
DEVI float to_f(const __nv_bfloat16 v) { return __bfloat162float(v); }
DEVI void from_f(__half& d, float v) { d = __float2half_rn(v); }
DEVI void from_f(__nv_bfloat16& d, float v) { d = __float2bfloat16_rn(v); }

// 32 B (16 values) per thread through a plain POD, as in the L1 GELU: an
// LDG.E.128 / STG.E.128 pair per access.  Anything that is not 32 B aligned
// falls back to a 16 B access, then to scalars.
struct alignas(32) Raw32 { unsigned w[8]; };
struct alignas(16) Raw16 { unsigned w[4]; };

template <typename T, typename Raw>
__global__ __launch_bounds__(256) void gelu_ip_vec(
    T* __restrict__ p, long long nvec, long long n) {
  using P = typename Pair<T>::type;
  constexpr int NP = sizeof(Raw) / sizeof(P);
  constexpr int VPT = 2 * NP;
  union Vec { Raw raw; P d[NP]; };
  char* __restrict__ b = reinterpret_cast<char*>(p);
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    Vec v;
    v.raw = *reinterpret_cast<const Raw*>(b + i * (long long)sizeof(Raw));
#pragma unroll
    for (int k = 0; k < NP; ++k) {
      float2 f = cvt2f(v.d[k]);
      f.x = gelu_tanh16(f.x);
      f.y = gelu_tanh16(f.y);
      cvt2t(v.d[k], f);
    }
    *reinterpret_cast<Raw*>(b + i * (long long)sizeof(Raw)) = v.raw;
  }
  const long long done = nvec * (long long)VPT;
  const long long rem = n - done;
  if (rem > 0 && blockIdx.x == 0 && threadIdx.x < rem) {
    const long long j = done + threadIdx.x;
    T r;
    from_f(r, gelu_tanh16(to_f(p[j])));
    p[j] = r;
  }
}

template <typename T>
__global__ void gelu_ip_scalar(T* __restrict__ p, long long n) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    T r;
    from_f(r, gelu_tanh16(to_f(p[i])));
    p[i] = r;
  }
}

#define FK_THREADS 256
#define FK_MAX_BLOCKS 9472   // 64 blocks/SM on a 148-SM B200

static inline int fk_blocks(long long work) {
  int b = (int)std::min<long long>((work + FK_THREADS - 1) / FK_THREADS, FK_MAX_BLOCKS);
  return b < 1 ? 1 : b;
}

template <typename T>
static void launch_ip(T* p, long long n, cudaStream_t s) {
  const uintptr_t bits = reinterpret_cast<uintptr_t>(p);
  if ((bits & 31u) == 0) {
    const long long nvec = n / 16;
    gelu_ip_vec<T, Raw32><<<fk_blocks(nvec), FK_THREADS, 0, s>>>(p, nvec, n);
  } else if ((bits & 15u) == 0) {
    const long long nvec = n / 8;
    gelu_ip_vec<T, Raw16><<<fk_blocks(nvec), FK_THREADS, 0, s>>>(p, nvec, n);
  } else {
    gelu_ip_scalar<T><<<fk_blocks(n), FK_THREADS, 0, s>>>(p, n);
  }
}

// Returns false when the tensor is outside what this kernel implements (the
// caller then uses the frozen out-of-place L1 GELU).
bool fk_ffn_gelu_tanh_(at::Tensor& x) {
  const auto st = x.scalar_type();
  if (!x.is_cuda() || !x.is_contiguous() ||
      (st != at::kHalf && st != at::kBFloat16)) {
    return false;
  }
  const long long n = x.numel();
  if (n == 0) return true;
  auto s = c10::cuda::getCurrentCUDAStream();
  if (st == at::kHalf) {
    launch_ip<__half>(reinterpret_cast<__half*>(x.data_ptr()), n, s);
  } else {
    launch_ip<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), n, s);
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu_tanh_", &fk_ffn_gelu_tanh_,
        "In-place GELU(tanh) for the FLUX FFN inner activation");
}
