// Hand-written sigmoid for FastKernels L1/sigmoid on B200 (sm_100).
//
// The captured call distribution is overwhelmingly tiny (7680/10251 calls are
// bfloat16[1,16,768] = 24 KiB), so per-call latency -- not bandwidth -- sets the
// score. Three things drive this kernel:
//
//  1. 128-bit vectorized load/store (8x bf16/fp16 per access), one vector per
//     thread, so a 12288-element input is a single wave of 6 blocks.
//  2. sigmoid via 0.5 + 0.5*tanh(x/2) using the native MUFU.TANH.{BF16,F16}
//     instruction. That is 1 MUFU + ~1 ALU per element, versus 2 MUFU
//     (ex2 + rcp) plus conversions for the 1/(1+exp(-x)) form in fp32.
//     On sm_100 the PTX `x2` forms decompose into two MUFU ops, so the win is
//     the halved MUFU count, not pair-packing.
//  3. Programmatic Dependent Launch: the grid is allowed to start before the
//     kernel that produced `x` retires, and `griddepcontrol.wait` re-imposes
//     the data dependency just after the (producer-independent) address math.
//     That hides this kernel's launch + CTA-dispatch latency inside the
//     producer's tail, which is the dominant per-call cost at these sizes.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kVec = 8;    // 16-bit elements per 128-bit access
constexpr int kBlock = 256;

// ---------------------------------------------------------------- math forms
// Packed pair of 16-bit values in a 32-bit word -> sigmoid, via native tanh.
__device__ __forceinline__ unsigned sig_pair(unsigned x, __nv_bfloat16) {
  const __nv_bfloat162 half = __float2bfloat162_rn(0.5f);
  __nv_bfloat162 h = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&x), half);
  unsigned t, a = *reinterpret_cast<unsigned*>(&h);
  asm("tanh.approx.bf16x2 %0, %1;" : "=r"(t) : "r"(a));
  __nv_bfloat162 y = __hfma2(*reinterpret_cast<__nv_bfloat162*>(&t), half, half);
  return *reinterpret_cast<unsigned*>(&y);
}

__device__ __forceinline__ unsigned sig_pair(unsigned x, __half) {
  const __half2 half = __float2half2_rn(0.5f);
  __half2 h = __hmul2(*reinterpret_cast<__half2*>(&x), half);
  unsigned t, a = *reinterpret_cast<unsigned*>(&h);
  asm("tanh.approx.f16x2 %0, %1;" : "=r"(t) : "r"(a));
  __half2 y = __hfma2(*reinterpret_cast<__half2*>(&t), half, half);
  return *reinterpret_cast<unsigned*>(&y);
}

// One element, through the same math (duplicate into a pair, keep the low half).
// Built with bit ops rather than by punning a T[2] array, whose 2-byte
// alignment would not satisfy a 4-byte `unsigned` access.
template <typename T>
__device__ __forceinline__ T sig_one(T x) {
  const unsigned short bits = *reinterpret_cast<const unsigned short*>(&x);
  unsigned w = static_cast<unsigned>(bits);
  w = sig_pair(w | (w << 16), T{});
  const unsigned short lo = static_cast<unsigned short>(w & 0xFFFFu);
  return *reinterpret_cast<const T*>(&lo);
}

template <typename T>
struct alignas(16) Vec {
  T v[kVec];
};

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// NOTAIL: caller guarantees n % kVec == 0. PDL: emit the dependency wait.
template <typename T, bool NOTAIL, bool PDL>
__launch_bounds__(kBlock) __global__ void sigmoid_vec_kernel(
    const T* __restrict__ in, T* __restrict__ out, int nvec, int n) {
  const int i = blockIdx.x * kBlock + threadIdx.x;
  // Address arithmetic is producer-independent; do it before the wait so it
  // overlaps the producer's tail.
  const Vec<T>* vin = reinterpret_cast<const Vec<T>*>(in) + i;
  Vec<T>* vout = reinterpret_cast<Vec<T>*>(out) + i;
  if (PDL) pdl_wait();
  if (i < nvec) {
    Vec<T> a = *vin;
    unsigned* w = reinterpret_cast<unsigned*>(&a);
#pragma unroll
    for (int k = 0; k < kVec / 2; ++k) w[k] = sig_pair(w[k], T{});
    *vout = a;
  } else if (!NOTAIL && i == nvec) {
    for (int k = nvec * kVec; k < n; ++k) out[k] = sig_one(in[k]);
  }
}

// Scalar path: unaligned pointers (any view whose offset is not 16B-aligned).
template <typename T, bool PDL>
__launch_bounds__(kBlock) __global__ void sigmoid_scalar_kernel(
    const T* __restrict__ in, T* __restrict__ out, int n) {
  const int i = blockIdx.x * kBlock + threadIdx.x;
  if (PDL) pdl_wait();
  if (i < n) out[i] = sig_one(in[i]);
}

// Strided path: an input whose trailing dims are contiguous but whose leading
// dims are a slice of a bigger buffer -- e.g. the captured
// float16[4,80,8400] with stride [1209600,8400,1], which is 4 contiguous runs
// of 672000 elements spaced 1209600 apart. torch handles this with a scalar
// offset-calculator kernel; we keep 128-bit accesses by mapping one contiguous
// run per blockIdx.y, so no thread's vector ever straddles a run boundary and
// no per-element index division is needed.
struct Lead {
  int size[4];
  long long stride[4];
  int n;
};

template <typename T, bool PDL>
__launch_bounds__(kBlock) __global__ void sigmoid_strided_kernel(
    const T* __restrict__ in, T* __restrict__ out, int inner_vec, Lead lead) {
  const int vi = blockIdx.x * kBlock + threadIdx.x;
  // Per-run base offsets: producer-independent, so hoist above the PDL wait.
  long long in_base = 0;
  int c = blockIdx.y;
#pragma unroll
  for (int d = 3; d >= 0; --d) {
    if (d < lead.n) {
      in_base += static_cast<long long>(c % lead.size[d]) * lead.stride[d];
      c /= lead.size[d];
    }
  }
  const long long out_base = static_cast<long long>(blockIdx.y) * inner_vec;
  if (PDL) pdl_wait();
  if (vi >= inner_vec) return;
  const Vec<T>* vin = reinterpret_cast<const Vec<T>*>(in + in_base) + vi;
  Vec<T>* vout = reinterpret_cast<Vec<T>*>(out) + (out_base + vi);
  Vec<T> a = *vin;
  unsigned* w = reinterpret_cast<unsigned*>(&a);
#pragma unroll
  for (int k = 0; k < kVec / 2; ++k) w[k] = sig_pair(w[k], T{});
  *vout = a;
}

// Decompose x into `outer` contiguous runs of `inner` elements. Returns false
// when the layout is not expressible that way (caller falls back to at::sigmoid).
bool analyze_strided(const at::Tensor& x, int64_t& inner, int64_t& outer, Lead& lead) {
  const int nd = x.dim();
  std::vector<int64_t> sz, st;
  for (int d = 0; d < nd; ++d) {
    if (x.size(d) == 1) continue;  // size-1 dims never constrain the layout
    sz.push_back(x.size(d));
    st.push_back(x.stride(d));
  }
  inner = 1;
  int d = static_cast<int>(sz.size()) - 1;
  int64_t expect = 1;
  for (; d >= 0; --d) {
    if (st[d] != expect) break;
    inner *= sz[d];
    expect *= sz[d];
  }
  const int nlead = d + 1;
  if (nlead > 4) return false;
  if (inner % kVec != 0) return false;
  outer = 1;
  lead.n = nlead;
  for (int i = 0; i < 4; ++i) {
    lead.size[i] = 1;
    lead.stride[i] = 0;
  }
  for (int i = 0; i < nlead; ++i) {
    // Every run must stay 16B aligned relative to the base pointer.
    if (st[i] % kVec != 0) return false;
    lead.size[i] = static_cast<int>(sz[i]);
    lead.stride[i] = st[i];
    outer *= sz[i];
  }
  if (outer > 65535) return false;  // gridDim.y limit
  const int64_t inner_vec = inner / kVec;
  if (inner_vec > 2147483647LL) return false;
  return true;
}

bool pdl_supported() {
  static const bool ok = [] {
    // FK_SIGMOID_NO_PDL=1 forces the plain launch, so the PDL contribution to a
    // measurement can be attributed without rebuilding.
    const char* off = std::getenv("FK_SIGMOID_NO_PDL");
    if (off && off[0] == '1') return false;
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)
        != cudaSuccess)
      return false;
    return major >= 9;
  }();
  return ok;
}

template <typename K, typename... Args>
void launch_pdl(K kernel, int blocks, cudaStream_t s, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(kBlock);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  TORCH_CHECK(err == cudaSuccess, "sigmoid: cudaLaunchKernelEx failed: ",
              cudaGetErrorString(err));
}

template <typename T>
void launch_strided(const T* in, T* out, int64_t inner, int64_t outer, Lead lead,
                    cudaStream_t s) {
  const int inner_vec = static_cast<int>(inner / kVec);
  const dim3 grid((inner_vec + kBlock - 1) / kBlock, static_cast<unsigned>(outer));
  if (pdl_supported()) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid;
    cfg.blockDim = dim3(kBlock);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t err = cudaLaunchKernelEx(&cfg, sigmoid_strided_kernel<T, true>, in,
                                         out, inner_vec, lead);
    TORCH_CHECK(err == cudaSuccess, "sigmoid: cudaLaunchKernelEx failed: ",
                cudaGetErrorString(err));
  } else {
    sigmoid_strided_kernel<T, false><<<grid, kBlock, 0, s>>>(in, out, inner_vec, lead);
  }
}

template <typename T>
void launch(const T* in, T* out, int64_t n64, cudaStream_t s) {
  const int n = static_cast<int>(n64);
  const bool aligned = (reinterpret_cast<uintptr_t>(in) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(out) % 16 == 0);
  const bool pdl = pdl_supported();
  if (!aligned) {
    const int blocks = (n + kBlock - 1) / kBlock;
    if (pdl)
      launch_pdl(sigmoid_scalar_kernel<T, true>, blocks, s, in, out, n);
    else
      sigmoid_scalar_kernel<T, false><<<blocks, kBlock, 0, s>>>(in, out, n);
    return;
  }
  const int nvec = n / kVec;
  const bool exact = (n % kVec) == 0;
  const int threads = nvec + (exact ? 0 : 1);
  const int blocks = threads > 0 ? (threads + kBlock - 1) / kBlock : 1;
  if (exact) {
    if (pdl)
      launch_pdl(sigmoid_vec_kernel<T, true, true>, blocks, s, in, out, nvec, n);
    else
      sigmoid_vec_kernel<T, true, false><<<blocks, kBlock, 0, s>>>(in, out, nvec, n);
  } else {
    if (pdl)
      launch_pdl(sigmoid_vec_kernel<T, false, true>, blocks, s, in, out, nvec, n);
    else
      sigmoid_vec_kernel<T, false, false><<<blocks, kBlock, 0, s>>>(in, out, nvec, n);
  }
}

}  // namespace

at::Tensor sigmoid(const at::Tensor& x) {
  const int64_t n = x.numel();
  const bool bf = x.scalar_type() == at::kBFloat16;
  TORCH_CHECK(bf || x.scalar_type() == at::kHalf, "sigmoid: expected bfloat16/float16");
  cudaStream_t s = at::cuda::getCurrentCUDAStream();

  if (x.is_contiguous()) {
    at::Tensor o = at::empty_like(x);
    if (n == 0) return o;
    if (bf)
      launch<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(x.const_data_ptr()),
                            static_cast<__nv_bfloat16*>(o.mutable_data_ptr()), n, s);
    else
      launch<__half>(static_cast<const __half*>(x.const_data_ptr()),
                     static_cast<__half*>(o.mutable_data_ptr()), n, s);
    return o;
  }

  // Non-contiguous: a run-decomposable slice keeps the vectorized path
  // (contiguous output); anything else goes back to ATen.
  int64_t inner = 0, outer = 0;
  Lead lead{};
  const bool aligned = reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0;
  if (n > 0 && aligned && analyze_strided(x, inner, outer, lead)) {
    at::Tensor o = at::empty(x.sizes(), x.options());
    if (bf)
      launch_strided<__nv_bfloat16>(
          static_cast<const __nv_bfloat16*>(x.const_data_ptr()),
          static_cast<__nv_bfloat16*>(o.mutable_data_ptr()), inner, outer, lead, s);
    else
      launch_strided<__half>(static_cast<const __half*>(x.const_data_ptr()),
                             static_cast<__half*>(o.mutable_data_ptr()), inner, outer,
                             lead, s);
    return o;
  }
  return at::sigmoid(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sigmoid", &sigmoid); }
