// Specialized dim-concat for contiguous tensors: one launch, vectorized copies,
// dispatched with Programmatic Dependent Launch.
//
// What the benchmark measures (this dictates the whole design).  The timed
// region is `l2.zero_(); start.record(); slot.copy_(src) x2; forward();
// end.record()`, so it already contains two input copies of its own, and the
// resulting CUDA-event latency is quantized into ~2.045 us steps: an iteration
// lands on the low step or the high one depending on whether our kernel's
// duration pushes the end event across a quantum boundary, and the median over
// 50 iterations picks whichever side won.  Two consequences:
//
//  1. The launch gap must come off the critical path -- PDL does that.  Measured
//     on B200: an empty kernel launched with
//     cudaLaunchAttributeProgrammaticStreamSerialization costs 11.25 us, exactly
//     the cost of launching nothing at all, while the same kernel without PDL
//     costs 13.31 us.
//  2. What is left is our own duration, and every nanosecond of it raises the
//     probability of crossing a boundary.  So the kernel is shaped for minimum
//     duration rather than for peak throughput: perfectly balanced work, one
//     load and one store per unit, no grid-stride loop, and almost no address
//     arithmetic ahead of the loads.
//
// The addressing trick that makes (2) possible: for a dim-`d` concat of
// contiguous tensors, input k's slice starts at the exclusive prefix sum of the
// slice lengths, so *the destination index of a unit equals its flat index in
// the output row*.  A thread that owns flat index `i` therefore stores to
// `out_row[i]` unconditionally and only has to select which input to load from,
// which is one compare and one pointer select over pre-biased base pointers.
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

namespace {

constexpr int MAX_IN = 8;
constexpr int MAX_PER_THREAD = 8;

template <typename V>
struct CatArgs {
  const V* src[MAX_IN];
  int inner[MAX_IN];  // V-units per output row contributed by input k
  int off[MAX_IN];    // V-unit offset of input k inside an output row
};

__device__ __forceinline__ void grid_dep_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

// Two-input fast path.  `p1` arrives pre-biased by -inner[0] so both inputs are
// indexed by the flat output index; UNROLL independent loads are issued before
// any store so one memory round trip covers them all.
//
// cudaGridDependencySynchronize() is called uniformly by every thread (PDL is a
// template constant and the call sits outside every divergent branch), which the
// intrinsic requires.
template <typename V, int PDL, int UNROLL>
__global__ void cat2_flat_kernel(const V* __restrict__ p0, const V* __restrict__ p1,
                                 V* __restrict__ out, int inner0, int inner1, int rowV) {
  const long long n = blockIdx.y;
  const V* s0 = p0 + n * (long long)inner0;
  const V* s1 = p1 + n * (long long)inner1;
  V* o = out + n * (long long)rowV;
  const int step = blockDim.x;
  const int g = blockIdx.x * step * UNROLL + threadIdx.x;
  if (PDL) grid_dep_sync();
  V v[UNROLL];
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    const int i = g + u * step;
    if (i < rowV) v[u] = (i < inner0 ? s0 : s1)[i];
  }
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    const int i = g + u * step;
    if (i < rowV) o[i] = v[u];
  }
}

// Generic path: any input count, grid-stride so it cannot run out of threads.
template <typename V, int PDL>
__global__ void catN_kernel(CatArgs<V> a, V* __restrict__ out, int rowV, int k) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const long long n = blockIdx.y;
  V* o = out + n * (long long)rowV;
  if (PDL) grid_dep_sync();
  for (int j = 0; j < k; ++j) {
    const V* s = a.src[j] + n * (long long)a.inner[j];
    V* d = o + a.off[j];
    const int m = a.inner[j];
    for (int i = t; i < m; i += stride) d[i] = s[i];
  }
}

int env_int(const char* name, int dflt) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : dflt;
}

struct Tune {
  int block = env_int("AKO_BLOCK", 192);
  int unroll = env_int("AKO_UNROLL", 2);
  int waves = env_int("AKO_WAVES", 4);
  int pdl = env_int("AKO_PDL", 1);
  int flat = env_int("AKO_FLAT", 1);
};
Tune g_tune;

int sm_count() {
  static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

// griddepcontrol is sm_90+; on older parts cudaLaunchKernelEx would reject the
// attribute and the kernel would never run.
bool pdl_ok() {
  static int ok = at::cuda::getCurrentDeviceProperties()->major >= 9 ? 1 : 0;
  return ok != 0;
}

int grid_x(int maxInner, int outer, int block, int waves) {
  long long gx = ((long long)sm_count() * waves + outer - 1) / outer;
  const long long need = ((long long)maxInner + (long long)block * MAX_PER_THREAD - 1) /
                         ((long long)block * MAX_PER_THREAD);
  if (gx < need) gx = need;
  const long long cap = ((long long)maxInner + block - 1) / block;
  if (gx > cap) gx = cap;
  if (gx < 1) gx = 1;
  return (int)gx;
}

template <typename K, typename... A>
void launch_pdl(K kern, dim3 grid, int block, cudaStream_t stream, A... args) {
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

template <typename V>
void launch(int K, const CatArgs<V>& a, void* out, int rowV, int outer, int maxInner) {
  const int block = g_tune.block;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  V* o = reinterpret_cast<V*>(out);
  const bool pdl = g_tune.pdl && pdl_ok();

  if (K == 2 && g_tune.flat) {
    const int U = g_tune.unroll == 1 ? 1 : (g_tune.unroll == 4 ? 4 : 2);
    const long long gx = ((long long)rowV + (long long)block * U - 1) / ((long long)block * U);
    const dim3 grid((unsigned)gx, outer);
    const V* p0 = a.src[0];
    const V* p1 = a.src[1] - a.inner[0];  // pre-biased: both indexed by flat index
    const int i0 = a.inner[0], i1 = a.inner[1];
#define CAT2_LAUNCH(UNROLLV)                                                    \
  do {                                                                                \
    if (pdl)                                                                          \
      launch_pdl(cat2_flat_kernel<V, 1, UNROLLV>, grid, block, stream, p0, p1, o, i0,  \
                 i1, rowV);                                                           \
    else                                                                              \
      cat2_flat_kernel<V, 0, UNROLLV><<<grid, block, 0, stream>>>(p0, p1, o, i0, i1,   \
                                                                  rowV);              \
  } while (0)
    if (U == 1) {
      CAT2_LAUNCH(1);
    } else if (U == 4) {
      CAT2_LAUNCH(4);
    } else {
      CAT2_LAUNCH(2);
    }
#undef CAT2_LAUNCH
    return;
  }

  const dim3 grid(grid_x(maxInner, outer, block, g_tune.waves), outer);
  if (pdl) {
    launch_pdl(catN_kernel<V, 1>, grid, block, stream, a, o, rowV, K);
  } else {
    catN_kernel<V, 0><<<grid, block, 0, stream>>>(a, o, rowV, K);
  }
}

template <typename V>
void fill_and_launch(int K, const std::vector<const void*>& srcs,
                     const std::vector<int>& inner, const std::vector<int>& off,
                     void* out, int rowV, int outer, int maxInner) {
  CatArgs<V> a;
  for (int k = 0; k < K; ++k) {
    a.src[k] = reinterpret_cast<const V*>(srcs[k]);
    a.inner[k] = inner[k];
    a.off[k] = off[k];
  }
  launch<V>(K, a, out, rowV, outer, maxInner);
}

}  // namespace

void set_tune(int64_t block, int64_t unroll, int64_t pdl, int64_t flat) {
  g_tune.block = (int)block;
  g_tune.unroll = (int)unroll;
  g_tune.pdl = (int)pdl;
  g_tune.flat = (int)flat;
}

// Returns nullopt (-> None in Python) when the fast path does not apply; the
// Python wrapper then defers to torch.cat.
std::optional<at::Tensor> cat_fast(const std::vector<at::Tensor>& xs, int64_t dim) {
  const int K = (int)xs.size();
  if (K < 1 || K > MAX_IN) return std::nullopt;
  const at::Tensor& x0 = xs[0];
  if (!x0.defined() || !x0.is_cuda()) return std::nullopt;
  const int64_t nd = x0.dim();
  if (nd < 1) return std::nullopt;
  const int64_t d = dim < 0 ? dim + nd : dim;
  if (d < 0 || d >= nd) return std::nullopt;
  const auto dtype = x0.scalar_type();
  const int64_t esz = x0.element_size();

  int64_t outer = 1;
  for (int64_t i = 0; i < d; ++i) outer *= x0.size(i);
  int64_t tail = 1;
  for (int64_t i = d + 1; i < nd; ++i) tail *= x0.size(i);
  if (outer < 1 || outer > 65535) return std::nullopt;

  int64_t catdim = 0;
  for (int k = 0; k < K; ++k) {
    const at::Tensor& t = xs[k];
    if (!t.defined() || !t.is_cuda() || t.scalar_type() != dtype || t.dim() != nd)
      return std::nullopt;
    if (t.device() != x0.device() || !t.is_contiguous()) return std::nullopt;
    for (int64_t i = 0; i < nd; ++i)
      if (i != d && t.size(i) != x0.size(i)) return std::nullopt;
    catdim += t.size(d);
  }

  auto sizes = x0.sizes().vec();
  sizes[d] = catdim;
  at::Tensor out = at::empty(sizes, x0.options());
  if (out.numel() == 0) return out;

  // Widest unit that divides every per-input row length and every base pointer.
  // 16 B for all captured shapes; narrower units keep unseen shapes correct.
  int64_t unit = 16;
  auto fits = [&](int64_t u) {
    if ((reinterpret_cast<uintptr_t>(out.data_ptr()) % (uintptr_t)u) != 0) return false;
    for (int k = 0; k < K; ++k) {
      if ((reinterpret_cast<uintptr_t>(xs[k].data_ptr()) % (uintptr_t)u) != 0) return false;
      if (((xs[k].size(d) * tail * esz) % u) != 0) return false;
    }
    return ((catdim * tail * esz) % u) == 0;
  };
  while (unit > 1 && !fits(unit)) unit >>= 1;

  std::vector<const void*> srcs(K);
  std::vector<int> inner(K), off(K);
  int64_t acc = 0, maxInner = 0;
  for (int k = 0; k < K; ++k) {
    const int64_t bytes = xs[k].size(d) * tail * esz;
    if (bytes / unit > INT32_MAX) return std::nullopt;
    srcs[k] = xs[k].data_ptr();
    inner[k] = (int)(bytes / unit);
    off[k] = (int)(acc / unit);
    maxInner = std::max<int64_t>(maxInner, inner[k]);
    acc += bytes;
  }
  if (acc / unit > INT32_MAX) return std::nullopt;
  const int rowV = (int)(acc / unit);

  void* op = out.data_ptr();
  const int o = (int)outer, mi = (int)maxInner;
  switch (unit) {
    case 16: fill_and_launch<uint4>(K, srcs, inner, off, op, rowV, o, mi); break;
    case 8: fill_and_launch<uint2>(K, srcs, inner, off, op, rowV, o, mi); break;
    case 4: fill_and_launch<uint32_t>(K, srcs, inner, off, op, rowV, o, mi); break;
    case 2: fill_and_launch<uint16_t>(K, srcs, inner, off, op, rowV, o, mi); break;
    default: fill_and_launch<uint8_t>(K, srcs, inner, off, op, rowV, o, mi); break;
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cat_fast", &cat_fast);
  m.def("set_tune", &set_tune);
}
