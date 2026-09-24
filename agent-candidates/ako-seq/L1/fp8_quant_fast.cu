// Specialized per-token-group FP8 (E4M3) quantizer for group_size == 128.
//
// One CUDA thread owns a contiguous slice of a group and keeps it in registers
// for the whole absmax -> scale -> quantize -> store pipeline: 16 B vector
// loads in, 16 B vector stores out, a sub-warp shuffle reduction in between.
// No shared memory and no block-wide __syncthreads(), so a block never stalls
// waiting for its slowest group and the group data is read from DRAM exactly
// once.  The tile (4 threads per group -> 32 values = 64 B per thread, 32
// groups = 128 threads per block) was swept over every captured shape; see
// kTileThreadsPerGroup below.
//
// Numerics are bit-identical to the vendored vLLM per_token_group_quant_8bit
// kernel this replaces:
//   * per-thread absmax seeded with ``eps`` (1e-10), reduced with fmaxf
//   * y_s = absmax / fp8_max  (division, matching the reference)
//   * UE8M0 rounding exp2(ceil(log2(max(y_s, 1e-10)))) done in integer bit
//     math on the fp32 exponent field -- exact, and avoids the libdevice
//     log2f/exp2f round-trip
//   * the stored scale is an exact power of two, so quantizing with
//     ``v * (1/y_s)`` is exactly ``v / y_s``
//   * E4M3 conversion is round-to-nearest-even with saturation, matching
//     c10::Float8_e4m3fn's software conversion over the clamped range
//
// The scale tensor is addressed through its runtime strides and dtype, which
// covers the three layouts this workload uses:
//   * float32, row-major (M, G)                 -- PerTokenGroupQuantFp8
//   * float32, column-major (M, G) over (G, M)   -- vLLM's DeepGEMM SF layout
//   * int32, column-major (M, G/4) over
//     (G/4, align(M, 4)), one UE8M0 exponent
//     byte per group packed 4-per-word along K   -- what DeepGEMM's
//                                                   fp8_gemm_nt consumes with
//                                                   disable_ue8m0_cast=True
// Emitting the packed form directly removes DeepGEMM's internal SF-cast kernel
// from the Fp8Linear critical path; it is bit-identical because our scales are
// already exact powers of two, so the exponent byte is just their fp32
// exponent field.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define PDL_WAIT() cudaGridDependencySynchronize()
#define PDL_TRIGGER() cudaTriggerProgrammaticLaunchCompletion()
#else
#define PDL_WAIT() ((void)0)
#define PDL_TRIGGER() ((void)0)
#endif

namespace {

constexpr int kGroupSize = 128;

__device__ __forceinline__ float to_f32(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ float to_f32(const __half v) {
  return __half2float(v);
}

// |a| elementwise-max |b|, two lanes at a time, in the input's own precision.
// Widening to fp32 is exact and order preserving, so reducing in bf16/fp16 and
// converting once at the end gives exactly the fp32 reduction's answer -- at
// half the instruction count.
__device__ __forceinline__ __nv_bfloat162 absmax2(const __nv_bfloat162 acc,
                                                  const __nv_bfloat162 v) {
  return __hmax2(acc, __habs2(v));
}
__device__ __forceinline__ __half2 absmax2(const __half2 acc, const __half2 v) {
  return __hmax2(acc, __habs2(v));
}
__device__ __forceinline__ float hmax_to_f32(const __nv_bfloat162 v) {
  return fmaxf(__bfloat162float(v.x), __bfloat162float(v.y));
}
__device__ __forceinline__ float hmax_to_f32(const __half2 v) {
  return fmaxf(__half2float(v.x), __half2float(v.y));
}
template <typename T> struct Vec2;
template <> struct Vec2<__nv_bfloat16> { using type = __nv_bfloat162; };
template <> struct Vec2<__half> { using type = __half2; };

// Biased exponent of exp2(ceil(log2(y))) for a positive normal float, in
// exponent-field bit math.  y = 2^(e-127) * 1.m, so ceil(log2(y)) is (e-127)
// when m == 0 and (e-126) otherwise; the byte returned is that exponent
// re-biased, i.e. both the UE8M0 code and the fp32 exponent field of the
// rounded-up power of two.
__device__ __forceinline__ uint32_t ue8m0_exponent(const float y) {
  const uint32_t bits = __float_as_uint(y);
  return ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
}

// Scale-store addressing mode, chosen on the host from the scale tensor's
// dtype and strides.
enum ScaleMode {
  kScaleLinear = 0,   // fp32, row stride == groups_per_row * group stride
  kScaleStrided = 1,  // fp32, both strides applied (column-major)
  kScalePacked = 2,   // int32, one UE8M0 exponent byte per group
};

template <typename T, int kThreadsPerGroup, int kGroupsPerBlock, int kMode>
__global__ void __launch_bounds__(kGroupsPerBlock* kThreadsPerGroup)
per_token_group_quant_e4m3_kernel(
    const T* __restrict__ x, __nv_fp8_storage_t* __restrict__ out,
    void* __restrict__ scale, const int64_t num_groups,
    const int groups_per_row, const int64_t scale_stride_row,
    const int64_t scale_stride_group, const float eps, const float fp8_min,
    const float fp8_max) {
  constexpr int kValsPerThread = kGroupSize / kThreadsPerGroup;
  constexpr int kInVecs = kValsPerThread * sizeof(T) / 16;   // 16 B loads
  constexpr int kOutWords = kValsPerThread / 4;              // 4 B store words
  static_assert(kValsPerThread * sizeof(T) % 16 == 0, "load must be 16 B wide");
  static_assert(32 % kThreadsPerGroup == 0, "group must sit inside a warp");
  // The store below covers 8 B (one uint2) or whole 16 B vectors; both cases
  // are kept so the template stays valid across the swept tile sizes.
  static_assert(kOutWords == 2 || kOutWords % 4 == 0,
                "store must be 8 B or a whole number of 16 B vectors");

  // Programmatic dependent launch: this grid is scheduled while the producer
  // of ``x`` is still running, so the launch latency is hidden -- but the
  // loads below must not start before that producer has finished.
  PDL_WAIT();
  const int64_t group =
      static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + (threadIdx.x / kThreadsPerGroup);
  if (group >= num_groups) {  // whole groups exit together
    PDL_TRIGGER();
    return;
  }
  const int lane = threadIdx.x & (kThreadsPerGroup - 1);

  // ---- load the thread's slice of the group into registers ----------------
  const int64_t offset = group * kGroupSize + lane * kValsPerThread;
  alignas(16) T regs[kValsPerThread];
  {
    const uint4* src = reinterpret_cast<const uint4*>(x + offset);
    uint4* dst = reinterpret_cast<uint4*>(&regs[0]);
#pragma unroll
    for (int i = 0; i < kInVecs; ++i) dst[i] = src[i];
  }

  // ---- absmax: per-thread, then across the lanes of the group ------------
  using T2 = typename Vec2<T>::type;
  const T2* regs2 = reinterpret_cast<const T2*>(&regs[0]);
  T2 acc2 = __habs2(regs2[0]);
#pragma unroll
  for (int i = 1; i < kValsPerThread / 2; ++i) acc2 = absmax2(acc2, regs2[i]);
  // ``eps`` is folded in here rather than seeded into the half-precision
  // accumulator: it is below every representable non-zero input's magnitude
  // that matters, so max(eps, max|v|) is the same either way, and this keeps
  // the reduction exact.
  float absmax = fmaxf(hmax_to_f32(acc2), eps);
  // A group's lanes are one aligned sub-warp (kThreadsPerGroup divides 32), so
  // the mask is that sub-warp.
  const unsigned mask = ((1u << kThreadsPerGroup) - 1u)
                        << (threadIdx.x & (32u - kThreadsPerGroup));
#pragma unroll
  for (int d = kThreadsPerGroup >> 1; d >= 1; d >>= 1) {
    absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, d));
  }

  const uint32_t exp_byte = ue8m0_exponent(fmaxf(absmax / fp8_max, 1e-10f));
  const float y_s = __uint_as_float(exp_byte << 23);

  if (lane == 0) {
    if (kMode == kScaleLinear) {
      static_cast<float*>(scale)[group * scale_stride_group] = y_s;
    } else if (kMode == kScaleStrided) {
      const int64_t row = group / groups_per_row;
      static_cast<float*>(scale)[row * scale_stride_row +
                                 (group - row * groups_per_row) *
                                     scale_stride_group] = y_s;
    } else {
      // Packed: word (group_in_row / 4, row) of a (G/4, aligned_mn) int32
      // buffer, byte (group_in_row % 4).  ``scale_stride_group`` carries
      // aligned_mn.  The lanes of a warp that own one row's consecutive groups
      // write the bytes of a single word, so the stores coalesce.
      const int64_t row = group / groups_per_row;
      const int g = static_cast<int>(group - row * groups_per_row);
      const int64_t word = (g >> 2) * scale_stride_group + row;
      static_cast<uint8_t*>(scale)[word * 4 + (g & 3)] =
          static_cast<uint8_t>(exp_byte);
    }
  }

  // ---- quantize the registers and store ----------------------------------
  const float inv_y_s = 1.0f / y_s;  // exact: y_s is a power of two
  uint32_t words[kOutWords];
#pragma unroll
  for (int i = 0; i < kValsPerThread; i += 2) {
    float2 v;
    v.x = fminf(fmaxf(to_f32(regs[i]) * inv_y_s, fp8_min), fp8_max);
    v.y = fminf(fmaxf(to_f32(regs[i + 1]) * inv_y_s, fp8_min), fp8_max);
    const uint32_t pair =
        __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
    if ((i & 3) == 0) {
      words[i >> 2] = pair;
    } else {
      words[i >> 2] |= pair << 16;
    }
  }
  if constexpr (kOutWords == 2) {
    *reinterpret_cast<uint2*>(out + offset) = make_uint2(words[0], words[1]);
  } else {
#pragma unroll
    for (int i = 0; i < kOutWords / 4; ++i) {
      reinterpret_cast<uint4*>(out + offset)[i] =
          make_uint4(words[4 * i], words[4 * i + 1], words[4 * i + 2],
                     words[4 * i + 3]);
    }
  }
  PDL_TRIGGER();
}

// Thread/block mapping. Swept over every captured shape (see tools/sweep.py):
// 4 threads per group -- i.e. 32 values (64 B) per thread -- with 32 groups
// (128 threads) per block is fastest or within noise everywhere, from [8,384]
// to [131072,384].  Wider groups (8/16 threads) lose ~10-13% on the
// bandwidth-bound rows because each thread issues fewer independent 16 B
// loads; narrower ones (1/2 threads) lose because the grid runs out of
// threads.  A single fixed tile also means no per-call heuristic and no
// divisibility ladder: a partial last block just exits early.
constexpr int kTileThreadsPerGroup = 4;
constexpr int kTileGroupsPerBlock = 32;

template <typename T>
void launch(const at::Tensor& x, at::Tensor& out, at::Tensor& scale,
            const int64_t num_groups, const int groups_per_row, const int mode,
            const int64_t s_row, const int64_t s_group, double eps,
            double fp8_min, double fp8_max) {
  cudaLaunchConfig_t config = {};
  config.gridDim =
      dim3(static_cast<unsigned>((num_groups + kTileGroupsPerBlock - 1) /
                                 kTileGroupsPerBlock));
  config.blockDim = dim3(kTileGroupsPerBlock * kTileThreadsPerGroup);
  config.dynamicSmemBytes = 0;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.numAttrs = 1;
  config.attrs = attrs;

#define FKQ_LAUNCH(MODE)                                                     \
  cudaLaunchKernelEx(                                                        \
      &config,                                                               \
      per_token_group_quant_e4m3_kernel<T, kTileThreadsPerGroup,             \
                                        kTileGroupsPerBlock, MODE>,          \
      reinterpret_cast<const T*>(x.data_ptr()),                              \
      reinterpret_cast<__nv_fp8_storage_t*>(out.data_ptr()),                  \
      scale.data_ptr(), num_groups, groups_per_row, s_row, s_group,          \
      static_cast<float>(eps), static_cast<float>(fp8_min),                  \
      static_cast<float>(fp8_max))

  if (mode == kScaleLinear) {
    FKQ_LAUNCH(kScaleLinear);
  } else if (mode == kScaleStrided) {
    FKQ_LAUNCH(kScaleStrided);
  } else {
    FKQ_LAUNCH(kScalePacked);
  }
#undef FKQ_LAUNCH
}

}  // namespace

// ``scale`` selects its own layout:
//   float32 with row stride == G * group stride -> row-major (M, G)
//   float32 otherwise                          -> column-major (M, G)
//   int32                                      -> packed UE8M0, (M, G/4) over
//                                                 (G/4, align(M, 4)); needs
//                                                 G % 4 == 0
// ``x`` must be 2-D bf16/fp16 with a multiple-of-128 inner size.
void per_token_group_quant_e4m3(const at::Tensor& x_in, at::Tensor& out,
                                at::Tensor& scale, double eps, double fp8_min,
                                double fp8_max) {
  // Materializing a non-contiguous input here keeps the check off the Python
  // hot path (it is a no-op for the contiguous activations this sees).
  const at::Tensor x = x_in.is_contiguous() ? x_in : x_in.contiguous();
  TORCH_CHECK(x.dim() == 2 && scale.dim() == 2, "expected 2-D x and scale");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.scalar_type() == at::kFloat8_e4m3fn, "out must be e4m3fn");
  const int64_t K = x.size(1);
  TORCH_CHECK(K % kGroupSize == 0, "inner size must be a multiple of 128");
  const int groups_per_row = static_cast<int>(K / kGroupSize);
  const int64_t num_groups = x.size(0) * groups_per_row;
  if (num_groups == 0) return;

  int mode;
  int64_t s_row = scale.stride(0), s_group = scale.stride(1);
  if (scale.scalar_type() == at::kInt) {
    TORCH_CHECK(groups_per_row % 4 == 0,
                "packed UE8M0 scales need a multiple-of-4 group count");
    TORCH_CHECK(scale.size(1) == groups_per_row / 4 && s_row == 1,
                "packed scale must be (M, G/4) with unit row stride");
    mode = kScalePacked;  // s_group is aligned_mn in this mode
  } else {
    TORCH_CHECK(scale.scalar_type() == at::kFloat,
                "scale must be float32 or packed int32");
    mode = (s_row == static_cast<int64_t>(groups_per_row) * s_group)
               ? kScaleLinear
               : kScaleStrided;
  }

  if (x.scalar_type() == at::kBFloat16) {
    launch<__nv_bfloat16>(x, out, scale, num_groups, groups_per_row, mode,
                          s_row, s_group, eps, fp8_min, fp8_max);
  } else {
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be bf16 or fp16");
    launch<__half>(x, out, scale, num_groups, groups_per_row, mode, s_row,
                   s_group, eps, fp8_min, fp8_max);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("per_token_group_quant_e4m3", &per_token_group_quant_e4m3,
        "per-token-group FP8 E4M3 quantization (group_size=128, UE8M0 scales)");
}
