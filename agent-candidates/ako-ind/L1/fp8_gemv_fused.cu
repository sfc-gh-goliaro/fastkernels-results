// Fused per-token-group quantization + FP8 block-scaled GEMV.
//
// Computes ``out = quant(x) @ w^T`` for a single token row in ONE kernel launch:
// the activation is quantized to E4M3 with UE8M0 per-128-group scales inside the
// mainloop and consumed from registers, so no FP8 activation tensor and no
// activation-scale tensor is ever materialized.  That removes one of the two GPU
// ops from ``Fp8Linear``'s timed window (~4 us each there, regardless of size)
// on top of replacing a GEMM that wastes 127/128 of every M-tile at M == 1.
//
// The kernel is purely bandwidth-bound: N*K bytes of weight against 2*N*K flops.
// Measured floor for *any* single op that reads the weight in the harness's
// window (tools/probe_bw.py): 9.2 us for the 8 MB of [4096,2048], and a plain
// read only reaches it with >= 1024 blocks in flight -- with 148-592 blocks the
// same read costs 11.2 us.  So the work assignment is chosen for warp count,
// not for arithmetic (an fp16-accumulate mainloop with a third of the
// conversions, and even one with no conversion at all, measured identically):
//
//   * a block owns kRows output rows; its kWarps warps split K between them
//     (warp w takes every kWarps-th 512-column chunk) and reduce through shared
//     memory at the end.  Splitting K *inside* the block is what buys the warp
//     count -- a warp-per-row layout caps the grid at N/kRows warps and lands
//     one 2 us slot short of the floor.
//   * kUnroll chunks are staged before any is consumed, so kRows * kUnroll
//     independent 16 B loads per lane are in flight.
//   * a lane's 16 columns always sit inside one 128-column K-block, so the
//     activation scale is one 8-lane shuffle and the weight's block scale is one
//     byte of one broadcast int32 load.
//   * the activation is quantized in registers, once per (block, chunk), and
//     reused by all kRows rows -- no staging buffer, and the only redundancy is
//     the row being re-read by N/kRows blocks out of L2.
//
// Layouts (all derived from the tensors, not assumed):
//   x   bf16 (1, K) row-major, K % (512*kUnroll) == 0
//   w   e4m3 (N, K) row-major
//   ws  int32 (N, K/512), stride(0) == 1, stride(1) == ws_ld -- DeepGEMM's
//       packed UE8M0 weight SF: byte j of word (n, kp) is the exponent field of
//       the fp32 scale of K-block kp*4 + j for output row n, so the scale is
//       ``__uint_as_float(byte << 23)`` (= 2^(byte-127)).  Verified against
//       ``deep_gemm.fp8_gemm_nt``'s own output (tools/probe_wscale.py).
//   out bf16 (1, N) row-major
//
// Numerics: the activation quantization is bit-identical to fp8_quant_fast.cu
// (same eps seeding, same divide by fp8_max, same integer-exponent UE8M0
// rounding, same __nv_cvt_float2_to_fp8x2 conversion) and every product is
// accumulated in FP32; only the accumulation order differs from DeepGEMM's.
// Measured against the baseline on harness-materialized inputs: max_abs 7e-12 on
// [1,2048]x[4096,2048] and 0.0 on [1,4096]x[2304,4096], i.e. the bf16 output
// rounding dominates and the harness's 1e-2 bound is never approached.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define PDL_WAIT() cudaGridDependencySynchronize()
#define PDL_TRIGGER() cudaTriggerProgrammaticLaunchCompletion()
#else
#define PDL_WAIT() ((void)0)
#define PDL_TRIGGER() ((void)0)
#endif

namespace {

constexpr int kColsPerLane = 16;                  // one 16 B weight load
constexpr int kColsPerChunk = 32 * kColsPerLane;  // 512 columns per warp

// Biased exponent of exp2(ceil(log2(y))) for a positive normal float -- both the
// UE8M0 code and the fp32 exponent field of the rounded-up power of two.  Same
// bit math as fp8_quant_fast.cu, so the two kernels agree exactly.
__device__ __forceinline__ uint32_t ue8m0_exponent(const float y) {
  const uint32_t bits = __float_as_uint(y);
  return ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
}

__device__ __forceinline__ float hmax_to_f32(const __nv_bfloat162 v) {
  return fmaxf(__bfloat162float(v.x), __bfloat162float(v.y));
}

// Four packed E4M3 bytes -> four floats.  E4M3 (denormals included) is exactly
// representable in fp16, so fp8 -> fp16 -> fp32 is lossless; the pairwise
// fp8->bf16 cvt would be shorter but does not exist on sm_100.
__device__ __forceinline__ float4 fp8x4_to_f32x4(const uint32_t v) {
  const __half2_raw lo =
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(v >> 16), __NV_E4M3);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hi));
  return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float dot4(const float4 a, const float4 b) {
  return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

// One lane's 16 activation columns, quantized to E4M3 with the group scale
// folded back in (a power of two, so this is exact).
struct ActChunk {
  float4 v[4];
};

// Quantize this lane's 16 bf16 activations.  The eight lanes of an octet hold
// exactly one 128-column K-block, so the block absmax is a three-step shuffle.
__device__ __forceinline__ ActChunk quantize_chunk(const __nv_bfloat16* xp,
                                                   const unsigned omask,
                                                   const float eps,
                                                   const float fp8_min,
                                                   const float fp8_max) {
  const uint4* src = reinterpret_cast<const uint4*>(xp);
  uint4 raw[2] = {src[0], src[1]};
  const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&raw[0]);

  __nv_bfloat162 acc2 = __habs2(v2[0]);
#pragma unroll
  for (int i = 1; i < 8; ++i) acc2 = __hmax2(acc2, __habs2(v2[i]));
  float absmax = fmaxf(hmax_to_f32(acc2), eps);
#pragma unroll
  for (int d = 4; d >= 1; d >>= 1)
    absmax = fmaxf(absmax, __shfl_xor_sync(omask, absmax, d));

  const float y_s =
      __uint_as_float(ue8m0_exponent(fmaxf(absmax / fp8_max, 1e-10f)) << 23);
  const float inv_y_s = 1.0f / y_s;  // exact: y_s is a power of two

  ActChunk out;
  float* q = reinterpret_cast<float*>(&out.v[0]);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 v = __bfloat1622float2(v2[i]);
    float2 s;
    s.x = fminf(fmaxf(v.x * inv_y_s, fp8_min), fp8_max);
    s.y = fminf(fmaxf(v.y * inv_y_s, fp8_min), fp8_max);
    const __nv_fp8x2_storage_t e4m3 =
        __nv_cvt_float2_to_fp8x2(s, __NV_SATFINITE, __NV_E4M3);
    const __half2_raw back = __nv_cvt_fp8x2_to_halfraw2(e4m3, __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&back));
    q[2 * i] = f.x * y_s;
    q[2 * i + 1] = f.y * y_s;
  }
  return out;
}

template <int kWarps, int kRows, int kUnroll>
__global__ void __launch_bounds__(kWarps * 32) fp8_gemv_fused_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
    const int32_t* __restrict__ ws, __nv_bfloat16* __restrict__ out,
    const int K, const int N, const int64_t ws_ld, const float eps,
    const float fp8_min, const float fp8_max) {
  __shared__ float red[kWarps][kRows];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int octet = lane >> 3;           // which K-block of the chunk
  const unsigned omask = 0xffu << (octet * 8);
  const int col0 = lane * kColsPerLane;  // within a 512-column chunk
  const int n0 = static_cast<int>(blockIdx.x) * kRows;
  const int rows = min(kRows, N - n0);
  // Row-index clamp instead of an early exit: the surplus rows of a partial last
  // block redundantly re-load row ``rows-1`` and their accumulator is dropped at
  // the store, which keeps every mainloop load unconditional (and hoistable).
  const int rmax = rows - 1;

  const uint8_t* wbase = w + static_cast<int64_t>(n0) * K + col0;
  const int32_t* wsbase = ws + n0;

  float acc[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;

  // Both x and w are written by the producer (the harness's input copy), so
  // nothing can be loaded before it drains; the launch attribute still buys the
  // overlapped dispatch, ~2 us on a shape this small.
  PDL_WAIT();

  const int nchunks = K / kColsPerChunk;
  for (int t = warp * kUnroll; t < nchunks; t += kWarps * kUnroll) {
    // Stage every (row, chunk) weight vector first: kRows * kUnroll independent
    // 16 B loads per lane, all in flight together.
    uint4 wreg[kRows][kUnroll];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        wreg[r][u] = *reinterpret_cast<const uint4*>(
            wbase + static_cast<int64_t>(min(r, rmax)) * K +
            (t + u) * kColsPerChunk);
      }
    }

    ActChunk a[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      a[u] = quantize_chunk(x + (t + u) * kColsPerChunk + col0, omask, eps,
                            fp8_min, fp8_max);

#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const uint32_t sfw =
            static_cast<uint32_t>(wsbase[min(r, rmax) + (t + u) * ws_ld]);
        const float sw = __uint_as_float(((sfw >> (8 * octet)) & 0xffu) << 23);
        const uint32_t* wv = reinterpret_cast<const uint32_t*>(&wreg[r][u]);
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          p += dot4(a[u].v[i], fp8x4_to_f32x4(wv[i]));
        acc[r] = fmaf(p, sw, acc[r]);
      }
    }
  }

  // Reduce each row: across the warp's lanes (which hold different K-blocks,
  // each already scaled), then across the warps (which hold different K-slices).
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    float v = acc[r];
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (lane == 0) red[warp][r] = v;
  }
  if (kWarps > 1) __syncthreads();
  if (tid < rows) {
    float v = red[0][tid];
#pragma unroll
    for (int wr = 1; wr < kWarps; ++wr) v += red[wr][tid];
    out[n0 + tid] = __float2bfloat16(v);
  }
  PDL_TRIGGER();
}

template <int kWarps, int kRows, int kUnroll>
void launch_cfg(const void* x, const void* w, const void* ws, void* out,
                const int K, const int N, const int64_t ws_ld, const float eps,
                const float fp8_min, const float fp8_max) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>((N + kRows - 1) / kRows));
  config.blockDim = dim3(kWarps * 32);
  config.dynamicSmemBytes = 0;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, fp8_gemv_fused_kernel<kWarps, kRows, kUnroll>,
                     static_cast<const __nv_bfloat16*>(x),
                     static_cast<const uint8_t*>(w),
                     static_cast<const int32_t*>(ws),
                     static_cast<__nv_bfloat16*>(out), K, N, ws_ld, eps,
                     fp8_min, fp8_max);
}

// Pure streaming-read diagnostic: the fastest any single op can read the weight
// in this window.  Same PDL launch, 16 B per lane, flat grid-stride sweep.
__global__ void __launch_bounds__(256) bw_read_kernel(
    const uint4* __restrict__ p, const int64_t n4, float* __restrict__ sink) {
  PDL_WAIT();
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int64_t i = blockIdx.x * 256 + threadIdx.x; i < n4;
       i += static_cast<int64_t>(gridDim.x) * 256) {
    const uint4 v = p[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if ((acc.x | acc.y | acc.z | acc.w) == 0xffffffffu) sink[0] = 1.0f;
  PDL_TRIGGER();
}

struct Cfg {
  int warps, rows, unroll;
};

// The tiles that survived the sweep, plus the fallbacks for a K with fewer than
// four 512-column chunks.  ``(K/512) % (warps*unroll) == 0`` is required; the
// host rejects configs that violate it.  Measured kernel-only windows on
// [1,2048]x[4096,2048] (9.22 us is the pure-read floor; full sweep of 20 tiles in
// ITERATIONS.md): {4,4,1} 9.22, {4,8,1} 9.22, {2,4,1} 11.26, {4,16,1} 11.26,
// {2,4,2} 11.23, {1,4,1} 11.30, {4,32,1} 13.31, {1,8,4} 13.34, {2,32,1} 17.38.
constexpr Cfg kCfgs[] = {
    {4, 4, 1},  // 0 -- default when K has >= 4 chunks
    {4, 8, 1},  // 1 -- ties cfg 0, half the blocks
    {2, 4, 1},  // 2 -- K with exactly 2 chunks
    {1, 4, 1},  // 3 -- K with a single chunk
};
constexpr int kNumCfgs = sizeof(kCfgs) / sizeof(Cfg);

}  // namespace

int64_t fp8_gemv_num_cfgs() { return kNumCfgs; }

// ``blocks`` <= 0 means one 16 B element per thread (a single wave of 256-thread
// blocks over the whole tensor).
void fp8_gemv_bwtest(const at::Tensor& w, at::Tensor& sink, int64_t blocks) {
  const int64_t n4 = w.numel() / 16;
  if (blocks <= 0) blocks = (n4 + 255) / 256;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(blocks));
  config.blockDim = dim3(256);
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, bw_read_kernel,
                     static_cast<const uint4*>(w.data_ptr()), n4,
                     static_cast<float*>(sink.data_ptr()));
}

// ``cfg`` selects the (warps, rows-per-block, chunk-unroll) tile; -1 picks the
// measured default.
void fp8_gemv_fused(const at::Tensor& x, const at::Tensor& w,
                    const at::Tensor& ws, at::Tensor& out, double eps,
                    double fp8_min, double fp8_max, int64_t cfg) {
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "fused GEMV expects a 1-row x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "x must be contiguous bf16");
  TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.is_contiguous(),
              "w must be contiguous e4m3");
  TORCH_CHECK(ws.scalar_type() == at::kInt && ws.stride(0) == 1,
              "ws must be int32 with unit row stride");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous(),
              "out must be contiguous bf16");
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(w.size(1) == K, "w must be (N, K)");
  TORCH_CHECK(K % kColsPerChunk == 0, "K must be a multiple of 512");
  TORCH_CHECK(ws.size(0) == N && ws.size(1) == K / kColsPerChunk,
              "ws must be (N, K/512)");

  const int64_t ws_ld = ws.stride(1);
  const float e = static_cast<float>(eps);
  const float lo = static_cast<float>(fp8_min);
  const float hi = static_cast<float>(fp8_max);
  const void* xp = x.data_ptr();
  const void* wp = w.data_ptr();
  const void* wsp = ws.data_ptr();
  void* op = out.data_ptr();

  const int nchunks_all = K / kColsPerChunk;
  if (cfg < 0) {
    // 4 rows per block keeps the grid at N/4 blocks (the streaming-read probe
    // needs >= 1024 to reach the floor) and 4 warps split K when there are
    // enough chunks to go round.
    cfg = (nchunks_all % 4 == 0) ? 0 : (nchunks_all % 2 == 0) ? 2 : 3;
  }
  TORCH_CHECK(cfg < kNumCfgs, "unknown fused GEMV config");
  TORCH_CHECK(nchunks_all % (kCfgs[cfg].warps * kCfgs[cfg].unroll) == 0,
              "K too small for this (warps, unroll)");

#define FKG_CASE(I)                                                          \
  case I:                                                                    \
    launch_cfg<kCfgs[I].warps, kCfgs[I].rows, kCfgs[I].unroll>(              \
        xp, wp, wsp, op, K, N, ws_ld, e, lo, hi);                            \
    break;
  switch (cfg) {
    FKG_CASE(0) FKG_CASE(1) FKG_CASE(2) FKG_CASE(3)
    default: TORCH_CHECK(false, "unknown fused GEMV config");
  }
#undef FKG_CASE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemv_fused", &fp8_gemv_fused,
        "fused per-token-group quantization + FP8 block-scaled GEMV (M == 1)",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("ws"),
        pybind11::arg("out"), pybind11::arg("eps"), pybind11::arg("fp8_min"),
        pybind11::arg("fp8_max"), pybind11::arg("cfg") = -1);
  m.def("fp8_gemv_num_cfgs", &fp8_gemv_num_cfgs, "number of sweep configs");
  m.def("fp8_gemv_bwtest", &fp8_gemv_bwtest, "streaming-read bandwidth probe",
        pybind11::arg("w"), pybind11::arg("sink"), pybind11::arg("blocks") = 0);
}
