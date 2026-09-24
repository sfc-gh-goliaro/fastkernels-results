// QuickGELU (x * sigmoid(1.702 x)) as one fused elementwise kernel.
//
// The op is launch-latency bound, not bandwidth bound: the only captured shape
// is contiguous float32[1, 77, 3072] = 236,544 elements (~946 KB), so essential
// traffic is ~1.9 MB -- a fraction of a microsecond of B200 HBM. Eager runs
// three kernels (1.702*x, sigmoid, x*s) and pays three launches plus two extra
// HBM round trips. This is one launch, one read, one write, and the launch
// itself is hidden behind the preceding kernel with Programmatic Dependent
// Launch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

// 1.702 * log2(e), correctly rounded to fp32. Folding the sigmoid argument
// scale into the exp2 base makes the whole activation one EX2 + one divide:
//   x * sigmoid(1.702 x) == x / (1 + exp2(-K x)).
#define QG_K 2.4554670f

__device__ __forceinline__ float quickgelu1(float x) {
  float e, y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(-QG_K * x));
  // div.approx.f32 returns 0 once the denominator exceeds 2^126, which is
  // exactly the limit of x*sigmoid(1.702x) as x -> -inf, so this branch-free
  // form stays stable for large-magnitude negative x with no overflow.
  asm("div.approx.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(1.0f + e));
  return y;
}

__device__ __forceinline__ float4 quickgelu4(float4 v) {
  v.x = quickgelu1(v.x);
  v.y = quickgelu1(v.y);
  v.z = quickgelu1(v.z);
  v.w = quickgelu1(v.w);
  return v;
}

// PDL: the grid is dispatched onto the SMs while the producing kernel drains,
// so this kernel's launch latency overlaps that tail. The wait sits before the
// first load of producer-written data, which makes the ordering identical to a
// plain stream-ordered launch.
__device__ __forceinline__ void quickgelu_pdl_wait() {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

// 128-bit (float4) access: 4 elements per thread, one 16 B load + 16 B store.
// 236544 / 4 = 59136 float4s = exactly 231 blocks of 256 threads, so the
// captured shape has no ragged tail and the bounds test never fires.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void quickgelu_vec4_i32(
    const float4* __restrict__ in, float4* __restrict__ out, int n4) {
  const int i = blockIdx.x * BLOCK + threadIdx.x;
  quickgelu_pdl_wait();
  if (i < n4) out[i] = quickgelu4(in[i]);
}

// Element counts past 2^31 use a grid-stride loop with 64-bit indices.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void quickgelu_vec4_i64(
    const float4* __restrict__ in, float4* __restrict__ out, int64_t n4) {
  const int64_t stride = (int64_t)BLOCK * gridDim.x;
  int64_t i = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  quickgelu_pdl_wait();
  for (; i < n4; i += stride) out[i] = quickgelu4(in[i]);
}

// Fallbacks for element counts not divisible by 4 or sub-16 B base alignment.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void quickgelu_plain_i32(
    const float* __restrict__ in, float* __restrict__ out, int n) {
  const int i = blockIdx.x * BLOCK + threadIdx.x;
  quickgelu_pdl_wait();
  if (i < n) out[i] = quickgelu1(in[i]);
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void quickgelu_plain_i64(
    const float* __restrict__ in, float* __restrict__ out, int64_t n) {
  const int64_t stride = (int64_t)BLOCK * gridDim.x;
  int64_t i = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  quickgelu_pdl_wait();
  for (; i < n; i += stride) out[i] = quickgelu1(in[i]);
}

static bool quickgelu_pdl_probe() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)
      != cudaSuccess)
    return false;
  return major >= 9;  // PDL is Hopper (sm_90) and newer.
}

// Resolved once at module load so the hot path holds no guard variable.
static const bool kQuickgeluPdl = quickgelu_pdl_probe();

template <typename Kernel, typename... Args>
static inline void quickgelu_launch(Kernel kernel, int grid, int block,
                                    cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  if (kQuickgeluPdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(&cfg, kernel, args...);
}

static constexpr int kBlock = 256;
static constexpr int64_t kI32Max = 0x7fffffff;
static constexpr int64_t kMaxGrid = 1 << 20;

at::Tensor quickgelu(const at::Tensor& x) {
  at::Tensor out = at::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) return out;

  const float* ip = x.const_data_ptr<float>();
  float* op = out.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool vec4 =
      (n % 4 == 0) &&
      (((reinterpret_cast<uintptr_t>(ip) | reinterpret_cast<uintptr_t>(op)) & 0xF) == 0);

  if (vec4) {
    const int64_t n4 = n >> 2;
    if (n4 <= kI32Max) {
      quickgelu_launch(quickgelu_vec4_i32<kBlock>,
                       (int)((n4 + kBlock - 1) / kBlock), kBlock, stream,
                       reinterpret_cast<const float4*>(ip),
                       reinterpret_cast<float4*>(op), (int)n4);
    } else {
      quickgelu_launch(quickgelu_vec4_i64<kBlock>,
                       (int)std::min(kMaxGrid, (n4 + kBlock - 1) / kBlock), kBlock,
                       stream, reinterpret_cast<const float4*>(ip),
                       reinterpret_cast<float4*>(op), n4);
    }
  } else if (n <= kI32Max) {
    quickgelu_launch(quickgelu_plain_i32<kBlock>,
                     (int)((n + kBlock - 1) / kBlock), kBlock, stream, ip, op, (int)n);
  } else {
    quickgelu_launch(quickgelu_plain_i64<kBlock>,
                     (int)std::min(kMaxGrid, (n + kBlock - 1) / kBlock), kBlock,
                     stream, ip, op, n);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quickgelu", &quickgelu, "Fused QuickGELU (float32, CUDA)");
}
