// Router gate GEMM: C[M,N] = A[M,K] @ B[N,K]^T, bf16 in / fp32 out.
//
// Context (see kernel.py): with hidden_size=2304 vLLM's tier-1 DSV3 router
// kernel never fires, so every call -- decode-sized and prefill-sized alike --
// went through the tier-2 cuBLAS path. This file adds a K-parallel,
// register-blocked SIMT kernel for the small-token shapes and keeps cuBLAS for
// large M.
//
// Two properties of the benchmark's timing loop drive the design (both
// measured on B200):
//
//  1. A kernel occupies the stream in ~2.048us quanta: a launch cannot start
//     its grid setup until the previous kernel retires, and the reported time
//     moves in whole quanta. Programmatic dependent launch (PDL) overlaps that
//     setup with the preceding kernel, which removes one whole quantum -- an
//     empty PDL kernel measures +0.05us against +2.11us without it.
//  2. What is left is the kernel's own duration, and for these shapes that is
//     dominated by *redundant* L2 traffic, not by FLOPs: at M=64 the FP32 FMA
//     roofline is 1.06us and the total unique traffic is 1.5MB (~0.2us), but a
//     one-expert-per-warp kernel re-reads the 1.18MB weight once per token
//     block. Hence the register blocking below: each lane keeps BM*BNR
//     accumulators so every loaded A element feeds BNR FMAs and every B element
//     feeds BM, which is what brings the traffic down to where the kernel fits
//     in fewer quanta than cuBLAS.
//
// Numerics: bf16 widens to fp32 exactly (8-bit mantissa) and all accumulation
// is fp32, matching the accumulation class of the vLLM kernels this replaces.
// The grouped-topk router's near-tie expert selection depends on that.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

using bf16 = __nv_bfloat16;

inline int sm_version() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  return props->major * 10 + props->minor;
}

// BM tokens x BNR experts per block; K split over WK warps x 32 lanes x 8
// elements (one uint4 per lane per step). KIT = K/256 steps per lane-group.
template <int BM, int BNR, int WK, int KIT>
__global__ __launch_bounds__(32 * WK) void rg_simt(
    float* __restrict__ C, const bf16* __restrict__ A, const bf16* __restrict__ B,
    int M, int N) {
  constexpr int K = KIT * 256;
  constexpr int KPT = KIT / WK;  // uint4 per lane
  static_assert(KIT % WK == 0, "WK must divide K/256");

  const int lane = threadIdx.x & 31;
  const int wk = threadIdx.x >> 5;
  const int n0 = blockIdx.x * BNR;
  const int m0 = blockIdx.y * BM;
  const int mrows = min(BM, M - m0);
  const int kb = wk * (KPT * 256) + lane * 8;
  const bf16* bp = B + (size_t)n0 * K + kb;
  const bf16* ap = A + (size_t)m0 * K + kb;

  // Rows past M read a clamped (valid) row and are not stored: the inner loop
  // stays branch-free and never touches out-of-range memory.
  int aoff[BM];
#pragma unroll
  for (int m = 0; m < BM; ++m) aoff[m] = (m < mrows ? m : mrows - 1) * K;

  float acc[BM][BNR];
#pragma unroll
  for (int m = 0; m < BM; ++m)
#pragma unroll
    for (int j = 0; j < BNR; ++j) acc[m][j] = 0.f;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    uint4 bv[BNR], av[BM];
#pragma unroll
    for (int j = 0; j < BNR; ++j)
      bv[j] = *reinterpret_cast<const uint4*>(bp + (size_t)j * K + i * 256);
#pragma unroll
    for (int m = 0; m < BM; ++m)
      av[m] = *reinterpret_cast<const uint4*>(ap + aoff[m] + i * 256);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bf[BNR], af[BM];
#pragma unroll
      for (int j = 0; j < BNR; ++j)
        bf[j] = __bfloat162float(reinterpret_cast<const bf16*>(&bv[j])[k]);
#pragma unroll
      for (int m = 0; m < BM; ++m)
        af[m] = __bfloat162float(reinterpret_cast<const bf16*>(&av[m])[k]);
#pragma unroll
      for (int m = 0; m < BM; ++m)
#pragma unroll
        for (int j = 0; j < BNR; ++j) acc[m][j] = fmaf(af[m], bf[j], acc[m][j]);
    }
  }

  // Cross-lane reduction through shared memory. A per-accumulator butterfly
  // (BM*BNR x 5 shuffles) costs more than one transposed pass once the tile is
  // bigger than a few accumulators -- measured 3.6us vs 2.4us at BM=4,BNR=8.
  __shared__ float sh[WK][32][BM * BNR + 1];
#pragma unroll
  for (int m = 0; m < BM; ++m)
#pragma unroll
    for (int j = 0; j < BNR; ++j) sh[wk][lane][m * BNR + j] = acc[m][j];
  __syncwarp();
  for (int t = lane; t < BM * BNR; t += 32) {
    float s = 0.f;
#pragma unroll
    for (int l = 0; l < 32; ++l) s += sh[wk][l][t];
    sh[wk][1][t] = s;  // row 1 is free: it held lane 1's partials, now consumed
  }
  __syncthreads();
  for (int t = threadIdx.x; t < BM * BNR; t += 32 * WK) {
    const int m = t / BNR, j = t % BNR;
    if (m < mrows) {
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < WK; ++q) s += sh[q][1][t];
      C[(size_t)(m0 + m) * N + n0 + j] = s;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <int BM, int BNR, int WK, int KIT>
void launch_simt(float* c, const bf16* a, const bf16* b, int M, int N,
                 cudaStream_t s) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / BNR, (M + BM - 1) / BM, 1);
  cfg.blockDim = dim3(32 * WK, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.numAttrs = 1;
  cfg.attrs = attrs;
  cudaLaunchKernelEx(&cfg, rg_simt<BM, BNR, WK, KIT>, c, a, b, M, N);
}

void cublas_gemm(const at::Tensor& input, const at::Tensor& weight,
                 at::Tensor& out) {
  // Column-major cuBLAS on row-major tensors: C(N,M) = weight(N,K) @ input(K,M),
  // i.e. out[M,N] row-major. Same call the baseline's tier 2 makes.
  const int64_t M = input.size(0), N = weight.size(0), K = input.size(1);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));
  const float alpha = 1.0f, beta = 0.0f;
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
      static_cast<int>(K), &alpha, weight.data_ptr(), CUDA_R_16BF,
      static_cast<int>(K), input.data_ptr(), CUDA_R_16BF, static_cast<int>(K),
      &beta, out.data_ptr(), CUDA_R_32F, static_cast<int>(N), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

// Fast path only for the layout family it was tuned for: K=2304 (=9*256, so the
// uint4 lane stride tiles K exactly), N=256, 16B-aligned contiguous operands.
//
// The M bound is where this kernel measured faster than cuBLAS under the
// benchmark's timing loop (harness-shaped probe, median of 4 runs of 50 iters):
//   M=26  15.36us vs 15.38   M=64  15.39us vs 17.34
//   M=96  17.41us vs 17.38   M=128 19.43us vs 15.41
// Past ~64 tokens the weight re-read (ceil(M/BM) passes) crosses another launch
// quantum and cuBLAS's tiling wins, so hand those to cuBLAS.
constexpr int kFastMaxM = 64;

inline bool simt_eligible(const at::Tensor& x, const at::Tensor& w, int M, int N,
                          int K) {
  return K == 2304 && N == 256 && M >= 1 && M <= kFastMaxM &&
         x.is_contiguous() && w.is_contiguous() &&
         (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) &&
         (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0);
}

}  // namespace

at::Tensor router_gemm(const at::Tensor& x, const at::Tensor& weight) {
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "router_gemm: inputs must be 2-D");
  TORCH_CHECK(x.dtype() == at::kBFloat16 && weight.dtype() == at::kBFloat16,
              "router_gemm: inputs must be bfloat16");
  TORCH_CHECK(x.size(1) == weight.size(1), "router_gemm: K mismatch");

  const int M = static_cast<int>(x.size(0));
  const int N = static_cast<int>(weight.size(0));
  const int K = static_cast<int>(x.size(1));
  auto out = at::empty({x.size(0), weight.size(0)}, x.options().dtype(at::kFloat));

  static const int kSM = sm_version();
  if (kSM >= 90 && simt_eligible(x, weight, M, N, K)) {
    auto* c = out.data_ptr<float>();
    auto* a = reinterpret_cast<const bf16*>(x.data_ptr());
    auto* b = reinterpret_cast<const bf16*>(weight.data_ptr());
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    // Tile chosen per M bucket from measured kernel duration (globaltimer):
    // tiny M is latency-bound (keep the tile minimal), larger M is bound by
    // redundant weight traffic (needs the 4x8 register tile).
    if (M <= 4)
      launch_simt<1, 2, 1, 9>(c, a, b, M, N, s);
    else
      launch_simt<4, 8, 3, 9>(c, a, b, M, N, s);
    return out;
  }
  cublas_gemm(x, weight, out);
  return out;
}

at::Tensor router_gemm_cublas(const at::Tensor& x, const at::Tensor& weight) {
  auto out = at::empty({x.size(0), weight.size(0)}, x.options().dtype(at::kFloat));
  cublas_gemm(x, weight, out);
  return out;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("router_gemm", &router_gemm, "Router gate GEMM (bf16 x bf16 -> fp32)");
  m.def("router_gemm_cublas", &router_gemm_cublas, "cuBLAS-only router GEMM");
}
