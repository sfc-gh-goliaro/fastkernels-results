// In-place RMSNorm over a strided row slice.
//
// This exists for one reason the frozen L1 normalization cannot serve: the
// packed MLA up-projection wants its input as `[normed latent | k_pe]`, one
// contiguous 576-element block per row. That block already exists inside the
// fused input projection's output -- columns [6144, 6720) of a [N, 6720] buffer.
// Normalizing the latent half *where it lies* makes the packed GEMM's input a
// plain slice of a tensor that is already allocated: no second buffer, no copy
// of the rope half, one launch. The frozen L1 kernel reads a strided view
// happily but allocates its own contiguous output, which cannot produce that
// adjacency.
//
// Numerics follow the vendored vLLM `rms_norm_kernel` exactly where it is a
// choice rather than an accident: the scale is `rsqrtf(sumsq / hidden + eps)`
// and each element is written as `(T)(x * scale * w)` -- fp32 throughout, a
// single rounding, and that multiplication order. The one thing that differs is
// the order in which the fp32 sum of squares is accumulated, which is not
// something a reduction can preserve across different block shapes.
//
// The row is read from global memory once, held in registers across the
// reduction, and written back to the same addresses. Each thread writes exactly
// the bundles it read, so the in-place write needs no ordering beyond the
// reduction that already separates them.
//
// Layouts this file does not claim are rejected by the host-side predicates,
// which report "not handled" in-band so the caller can route elsewhere. Nothing
// catches an exception around a launch: a launch that starts has to be a launch
// that is correct.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kWarp = 32;
constexpr int kMaxBlock = 1024;
// Bundles per thread. Four 128-bit bundles of bf16 is 64 elements, enough for a
// 2048-wide row on a single warp; beyond that the block flavor takes over.
constexpr int kMaxBundlesPerThread = 4;
constexpr int kWarpsPerBlock = 4;
// Both flavors grid-stride over rows, so clamping the grid costs nothing but
// keeps a row count above 2^31 from wrapping the dimension.
constexpr int64_t kMaxGrid = 2147483647;
// Grid ceiling in waves of the device's multiprocessors. One block per row
// launches a CTA per row and reloads the weight row in every one of them; a
// capped grid amortizes the weight registers over many rows instead.
constexpr int64_t kGridWavesCap = 256;

// ---------------------------------------------------------------------------
// Element traits. Only the two 16-bit floating types get a hand-written path:
// they are the only dtypes this operator's traffic uses and the ones where
// packed conversion pays. Anything else is reported as unclaimed.
// ---------------------------------------------------------------------------

template <typename T>
struct ElemTraits;

template <>
struct ElemTraits<__nv_bfloat16> {
  using pack_t = __nv_bfloat162;
  __device__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static float2 to_f2(pack_t x) { return __bfloat1622float2(x); }
  __device__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16(x); }
};

template <>
struct ElemTraits<__half> {
  using pack_t = __half2;
  __device__ static float to_f(__half x) { return __half2float(x); }
  __device__ static float2 to_f2(pack_t x) { return __half22float2(x); }
  __device__ static __half from_f(float x) { return __float2half_rn(x); }
};

// A VEC-wide bundle, aligned so the compiler emits one vector load/store per
// bundle. VEC is a power of two <= 8 over a 2-byte element, so the alignment is
// a power of two <= 16 -- a 128-bit access at the widest setting.
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Bundle {
  T e[VEC];
};

// Sum of squares of one bundle in fp32. Pairs convert together, matching the
// vendored `_f16Vec::sum_squares`.
template <typename T, int VEC>
__device__ __forceinline__ float sum_squares(const Bundle<T, VEC>& b) {
  using Tr = ElemTraits<T>;
  float acc = 0.0f;
  if constexpr (VEC % 2 == 0) {
#pragma unroll
    for (int i = 0; i < VEC; i += 2) {
      float2 z = Tr::to_f2(typename Tr::pack_t{b.e[i], b.e[i + 1]});
      acc += z.x * z.x + z.y * z.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      float z = Tr::to_f(b.e[i]);
      acc += z * z;
    }
  }
  return acc;
}

// out = (T)(x * scale * w), in fp32 and in that multiplication order. This is
// the vendored kernel's convention, reproduced so the only difference between
// the two is the reduction order of the sum of squares.
template <typename T, int VEC>
__device__ __forceinline__ Bundle<T, VEC> scale_and_weight(
    const Bundle<T, VEC>& x, float scale, const Bundle<T, VEC>& w) {
  using Tr = ElemTraits<T>;
  Bundle<T, VEC> out;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    out.e[i] = Tr::from_f(Tr::to_f(x.e[i]) * scale * Tr::to_f(w.e[i]));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Row addressing. The host collapses the leading dimensions to `lead`
// surviving (size, stride) pairs; `lead == 1` also carries the single-row case,
// where the index is always zero. Strides are 64-bit: a row stride that fits in
// int32 today does not stay that way when a fused projection widens.
// ---------------------------------------------------------------------------

struct RowMap {
  int64_t rows;
  int64_t inner;    // size of the inner surviving pair (lead == 2 only)
  int64_t stride0;
  int64_t stride1;  // stride of the inner surviving pair (lead == 2 only)
};

template <int LEAD>
__device__ __forceinline__ int64_t row_offset(const RowMap& m, int64_t row) {
  if constexpr (LEAD == 1) {
    return row * m.stride0;
  } else {
    return (row / m.inner) * m.stride0 + (row % m.inner) * m.stride1;
  }
}

// ---------------------------------------------------------------------------
// Warp-per-row flavor: one full warp owns one row, several warps per block,
// grid-striding over rows. The reduction is a shuffle butterfly over the whole
// warp, so there is no shared memory and no barrier -- which is what makes this
// the cheap flavor at the row counts that matter here.
//
// Requires `bundles == BPT * 32` exactly; the planner only selects it then.
// ---------------------------------------------------------------------------

template <typename T, int VEC, int LEAD, int BPT>
__global__ void norm_inplace_warp_rows(T* __restrict__ x,
                                       const T* __restrict__ weight,
                                       RowMap map, int hidden, float eps) {
  using B = Bundle<T, VEC>;

  const int lane = threadIdx.x & (kWarp - 1);
  const int warps = blockDim.x / kWarp;
  const int64_t row_stride = static_cast<int64_t>(gridDim.x) * warps;
  int64_t row =
      static_cast<int64_t>(blockIdx.x) * warps + (threadIdx.x / kWarp);

  // The weight row is loaded once for the lifetime of the thread, not once per
  // row: it is the same for every row this warp will visit.
  const B* __restrict__ wv = reinterpret_cast<const B*>(weight);
  B wheld[BPT];
#pragma unroll
  for (int j = 0; j < BPT; ++j) wheld[j] = wv[lane + j * kWarp];

  for (; row < map.rows; row += row_stride) {
    B* __restrict__ p =
        reinterpret_cast<B*>(x + row_offset<LEAD>(map, row));

    B held[BPT];
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      held[j] = p[lane + j * kWarp];
      acc += sum_squares(held[j]);
    }

#pragma unroll
    for (int off = kWarp >> 1; off; off >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    const float scale = rsqrtf(acc / hidden + eps);

#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      p[lane + j * kWarp] = scale_and_weight(held[j], scale, wheld[j]);
    }
  }
}

// ---------------------------------------------------------------------------
// Block-per-row flavor: the general case, for a bundle count that is not a
// whole multiple of the warp width or is too wide for one warp's registers.
// The block size is always a warp multiple and the tail lanes are predicated.
//
// Two barriers per row: one after the per-warp partials are published, and one
// at the end of the row so the next grid-stride iteration cannot overwrite
// those partials while a lagging warp is still reading them.
// ---------------------------------------------------------------------------

template <typename T, int VEC, int LEAD, int BPT>
__global__ void norm_inplace_block_rows(T* __restrict__ x,
                                        const T* __restrict__ weight,
                                        RowMap map, int hidden, int bundles,
                                        float eps) {
  using B = Bundle<T, VEC>;
  extern __shared__ float partials[];  // one entry per warp in the block

  const int warps = blockDim.x / kWarp;
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x / kWarp;

  const B* __restrict__ wv = reinterpret_cast<const B*>(weight);
  B wheld[BPT];
#pragma unroll
  for (int j = 0; j < BPT; ++j) {
    const int i = threadIdx.x + j * blockDim.x;
    if (i < bundles) wheld[j] = wv[i];
  }

  for (int64_t row = blockIdx.x; row < map.rows; row += gridDim.x) {
    B* __restrict__ p =
        reinterpret_cast<B*>(x + row_offset<LEAD>(map, row));

    B held[BPT];
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < bundles) {
        held[j] = p[i];
        acc += sum_squares(held[j]);
      }
    }

#pragma unroll
    for (int off = kWarp >> 1; off; off >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) partials[warp] = acc;
    __syncthreads();

    // Every warp reduces every partial: one barrier instead of two, and the
    // scale ends up in a register on the thread that needs it.
    float total = 0.0f;
    for (int j = 0; j < warps; ++j) total += partials[j];
    const float scale = rsqrtf(total / hidden + eps);

#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < bundles) p[i] = scale_and_weight(held[j], scale, wheld[j]);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host-side layout analysis
// ---------------------------------------------------------------------------

// Collapsing the leading dimensions. `ok == false` means the layout needs more
// than two surviving (size, stride) pairs, which no layout this operator
// produces does; such a tensor is reported unclaimed rather than indexed with a
// formula that does not describe it.
struct Layout {
  bool ok = false;
  int64_t rows = 1;
  int64_t sizes[2] = {1, 1};
  int64_t strides[2] = {0, 0};
  int lead = 1;
};

Layout collapse_leading(const at::Tensor& t) {
  Layout L;
  const int nd = static_cast<int>(t.dim());
  const int64_t hidden = t.size(-1);

  // Size-1 leading dims contribute nothing to row addressing and their strides
  // are unconstrained, so they are dropped.
  constexpr int kMaxDims = 8;
  int64_t sz[kMaxDims], st[kMaxDims];
  int n = 0;
  for (int d = 0; d < nd - 1; ++d) {
    if (t.size(d) == 1) continue;
    if (n == kMaxDims) return L;  // absurd rank; report unclaimed
    sz[n] = t.size(d);
    st[n] = t.stride(d);
    ++n;
  }

  // Merge adjacent pairs from the inner side while the outer stride is exactly
  // the inner extent. A fully contiguous tensor of any rank collapses to one
  // pair this way.
  bool merged = true;
  while (merged && n > 1) {
    merged = false;
    for (int i = n - 2; i >= 0; --i) {
      if (st[i] == sz[i + 1] * st[i + 1]) {
        sz[i] *= sz[i + 1];
        st[i] = st[i + 1];
        for (int j = i + 1; j < n - 1; ++j) {
          sz[j] = sz[j + 1];
          st[j] = st[j + 1];
        }
        --n;
        merged = true;
        break;
      }
    }
  }

  if (n > 2) return L;

  if (n == 0) {
    L.lead = 1;
    L.sizes[0] = 1;
    L.strides[0] = hidden;
    L.rows = 1;
  } else {
    L.lead = n;
    L.rows = 1;
    for (int i = 0; i < n; ++i) {
      L.sizes[i] = sz[i];
      L.strides[i] = st[i];
      L.rows *= sz[i];
    }
  }
  L.ok = true;
  return L;
}

bool dtype_supported(const at::Tensor& t) {
  return t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf;
}

// A weight we can index as a plain [hidden] bundle array on the device.
bool weight_ok(const at::Tensor& w, const at::Tensor& x, int64_t hidden) {
  return w.defined() && w.is_cuda() && w.is_contiguous() && w.dim() == 1 &&
         w.numel() == hidden && w.scalar_type() == x.scalar_type() &&
         w.device() == x.device();
}

bool aligned_for(const void* p, int vec, size_t elem_size) {
  const auto bytes = static_cast<std::uintptr_t>(vec) * elem_size;
  return reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

// Whether two distinct rows of the collapsed map can cover a shared element.
//
// This matters more here than it would for an out-of-place kernel. Every row is
// *written*, one warp or block per row, so overlapping rows are concurrent
// read-modify-writes of the same addresses by different warps -- a race, not
// merely a surprising answer. A row stride of zero is the extreme case and is
// not rejected by any of the other predicates: zero is divisible by every bundle
// width, and `rows == numel / hidden` still holds for a broadcast view.
//
// Rows are at `i * stride0 + j * stride1` over `i < sizes[0]`, `j < sizes[1]`,
// each covering `hidden` elements. They are pairwise disjoint exactly when each
// stride is at least the extent of everything nested inside it. One row is
// trivially disjoint from itself.
bool rows_overlap(const Layout& L, int64_t hidden) {
  if (L.rows <= 1) return false;
  if (L.lead == 1) return L.strides[0] < hidden;
  if (L.strides[1] < hidden) return true;
  return L.strides[0] < L.sizes[1] * L.strides[1];
}

// The byte range a strided row view can touch: from its base to the end of its
// last row. Used to establish that the weight is not aliased by the tensor being
// written, since both are declared `__restrict__`. Sound because `rows_overlap`
// has already established every stride is positive, so the base is the lowest
// address the view reaches.
bool weight_aliases_rows(const at::Tensor& x, const Layout& L, int64_t hidden,
                         const at::Tensor& w) {
  const size_t esz = x.element_size();
  int64_t last = 0;
  for (int i = 0; i < L.lead; ++i) last += (L.sizes[i] - 1) * L.strides[i];
  const char* xb = static_cast<const char*>(x.const_data_ptr());
  const char* xe = xb + (last + hidden) * esz;
  const char* wb = static_cast<const char*>(w.const_data_ptr());
  const char* we = wb + w.numel() * w.element_size();
  return xb < we && wb < xe;
}

// Widest bundle usable for this layout: the hidden size must divide evenly, both
// base pointers must be bundle-aligned, and *every surviving leading stride*
// must be a multiple of the bundle width. A base-pointer check alone would
// accept a layout whose second row starts mid-bundle -- which is exactly the
// hazard a strided in-place write introduces.
int choose_vec(const at::Tensor& x, const Layout& L, int64_t hidden,
               const at::Tensor& weight) {
  const size_t esz = x.element_size();
  const int max_vec = static_cast<int>(16 / esz);
  for (int v = max_vec; v >= 1; v >>= 1) {
    if (hidden % v != 0) continue;
    if (!aligned_for(x.const_data_ptr(), v, esz)) continue;
    if (!aligned_for(weight.const_data_ptr(), v, esz)) continue;
    bool strides_ok = true;
    for (int i = 0; i < L.lead; ++i) {
      if (L.strides[i] % v != 0) strides_ok = false;
    }
    if (!strides_ok) continue;
    return v;
  }
  return 0;
}

int sm_count() {
  static const int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

int64_t round_up_warp(int64_t v) { return ((v + kWarp - 1) / kWarp) * kWarp; }

// Launch geometry. `block == 0` means the shape is not handled.
struct Plan {
  bool warp_flavor = false;
  int block = 0;
  int bpt = 1;
  int64_t grid = 0;
};

Plan plan_launch(int bundles, int64_t rows) {
  Plan p;

  // A row that divides evenly across one warp, at a register cost one warp can
  // hold, takes the shuffle-only flavor.
  if (bundles % kWarp == 0 && bundles / kWarp <= kMaxBundlesPerThread) {
    p.warp_flavor = true;
    p.bpt = bundles / kWarp;
    const int64_t warps = std::min<int64_t>(kWarpsPerBlock, std::max<int64_t>(1, rows));
    p.block = static_cast<int>(warps) * kWarp;
    p.grid = std::min<int64_t>((rows + warps - 1) / warps,
                               kGridWavesCap * sm_count());
    if (p.grid > kMaxGrid) p.grid = kMaxGrid;
    return p;
  }

  for (int bpt = 1; bpt <= kMaxBundlesPerThread; ++bpt) {
    const int64_t block =
        round_up_warp((static_cast<int64_t>(bundles) + bpt - 1) / bpt);
    if (block <= kMaxBlock) {
      p.block = static_cast<int>(block);
      p.bpt = bpt;
      break;
    }
  }
  if (p.block == 0) return p;
  p.grid = std::min<int64_t>(rows, kGridWavesCap * sm_count());
  if (p.grid > kMaxGrid) p.grid = kMaxGrid;
  return p;
}

// ---------------------------------------------------------------------------
// Launch dispatch. The instantiation matrix is deliberately closed:
//   2 dtypes x 4 bundle widths x 2 addressings x 4 bundles-per-thread x
//   2 flavors = 128 kernels in one translation unit.
// ---------------------------------------------------------------------------

#define DISPATCH_VEC(v, ...) \
  switch (v) {               \
    case 8: {                \
      constexpr int VEC = 8; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
    case 4: {                \
      constexpr int VEC = 4; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
    case 2: {                \
      constexpr int VEC = 2; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
    default: {               \
      constexpr int VEC = 1; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
  }

#define DISPATCH_LEAD(l, ...) \
  if ((l) == 1) {             \
    constexpr int LEAD = 1;   \
    __VA_ARGS__;              \
  } else {                    \
    constexpr int LEAD = 2;   \
    __VA_ARGS__;              \
  }

#define DISPATCH_BPT(b, ...) \
  switch (b) {               \
    case 4: {                \
      constexpr int BPT = 4; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
    case 3: {                \
      constexpr int BPT = 3; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
    case 2: {                \
      constexpr int BPT = 2; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
    default: {               \
      constexpr int BPT = 1; \
      __VA_ARGS__;           \
      break;                 \
    }                        \
  }

template <typename T>
bool launch(T* x, const T* weight, const Layout& L, int hidden, int vec,
            float eps, cudaStream_t stream) {
  RowMap map;
  map.rows = L.rows;
  map.inner = L.sizes[1];
  map.stride0 = L.strides[0];
  map.stride1 = L.strides[1];

  const int bundles = hidden / vec;
  const Plan p = plan_launch(bundles, L.rows);
  if (p.block == 0 || p.grid <= 0) return false;

  if (p.warp_flavor) {
    DISPATCH_VEC(vec, DISPATCH_LEAD(L.lead, DISPATCH_BPT(p.bpt, {
                   norm_inplace_warp_rows<T, VEC, LEAD, BPT>
                       <<<static_cast<unsigned>(p.grid), p.block, 0, stream>>>(
                           x, weight, map, hidden, eps);
                 })));
    return true;
  }

  const size_t smem = static_cast<size_t>(p.block / kWarp) * sizeof(float);
  DISPATCH_VEC(vec, DISPATCH_LEAD(L.lead, DISPATCH_BPT(p.bpt, {
                 norm_inplace_block_rows<T, VEC, LEAD, BPT>
                     <<<static_cast<unsigned>(p.grid), p.block, smem, stream>>>(
                         x, weight, map, hidden, bundles, eps);
               })));
  return true;
}

bool norm_inplace_impl(at::Tensor& x, const at::Tensor& weight, double eps) {
  if (!x.is_cuda() || !dtype_supported(x) || x.dim() < 1) return false;
  const int64_t hidden = x.size(-1);
  if (hidden <= 0 || hidden > std::numeric_limits<int>::max()) return false;
  if (x.stride(-1) != 1) return false;
  if (!weight_ok(weight, x, hidden)) return false;

  const Layout L = collapse_leading(x);
  if (!L.ok || L.rows != x.numel() / hidden) return false;

  // Rows are written, one warp or block each, so overlapping rows would be a
  // race between warps rather than merely a different answer. Checked before the
  // weight-alias range calculation, which relies on every stride being positive.
  if (rows_overlap(L, hidden)) return false;

  // Both operands are `__restrict__` and one of them is written in place, so an
  // aliased pair would be undefined behaviour rather than merely a different
  // answer.
  if (weight_aliases_rows(x, L, hidden, weight)) return false;

  const int vec = choose_vec(x, L, hidden, weight);
  if (vec == 0) return false;
  if (L.rows == 0) return true;  // nothing to normalize, and correctly so

  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float epsf = static_cast<float>(eps);
  const int hid = static_cast<int>(hidden);

  if (x.scalar_type() == at::kBFloat16) {
    return launch<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr()), L, hid,
        vec, epsf, stream);
  }
  return launch<__half>(
      reinterpret_cast<__half*>(x.data_ptr()),
      reinterpret_cast<const __half*>(weight.const_data_ptr()), L, hid, vec,
      epsf, stream);
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry points
//
// `rms_norm_inplace` reports "not handled" in-band with a false return rather
// than by raising, so the caller routes to an out-of-place normalization for a
// layout this file does not claim.
// ---------------------------------------------------------------------------

bool rms_norm_inplace(at::Tensor x, const at::Tensor& weight, double eps) {
  return norm_inplace_impl(x, weight, eps);
}

// The layout decision without a launch: whether the layout is claimed, the
// surviving (size, stride) pairs, the row count, the chosen bundle width, and
// the launch geometry. Exposed so the tests can assert what the kernel claims
// instead of inferring it from results.
std::vector<int64_t> describe_layout(const at::Tensor& x,
                                     const at::Tensor& weight) {
  std::vector<int64_t> d;
  // Rank zero has no last dimension, so it must be answered before anything
  // indexes one. `collapse_leading` and `stride(-1)` both would, and a scalar
  // should get the same in-band "not claimed" answer the launch entry gives it
  // rather than an exception from an introspection call.
  if (x.dim() < 1) {
    d.assign(11, 0);
    return d;
  }
  const int64_t hidden = x.size(-1);
  const Layout L = collapse_leading(x);
  const bool basic = x.is_cuda() && dtype_supported(x) &&
                     hidden > 0 && x.stride(-1) == 1 &&
                     weight_ok(weight, x, hidden) && L.ok &&
                     L.rows == x.numel() / hidden &&
                     !rows_overlap(L, hidden) &&
                     !weight_aliases_rows(x, L, hidden, weight);
  const int vec = basic ? choose_vec(x, L, hidden, weight) : 0;
  const Plan p = vec ? plan_launch(static_cast<int>(hidden / vec), L.rows)
                     : Plan{};
  d.push_back(basic && vec != 0 && p.block != 0 ? 1 : 0);
  d.push_back(L.ok ? L.lead : 0);
  d.push_back(L.rows);
  for (int i = 0; i < 2; ++i) {
    d.push_back(L.sizes[i]);
    d.push_back(L.strides[i]);
  }
  d.push_back(vec);
  d.push_back(p.warp_flavor ? 1 : 0);
  d.push_back(p.block);
  d.push_back(p.bpt);
  d.push_back(p.grid);
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm_inplace", &rms_norm_inplace,
        "In-place RMSNorm over a strided row slice; false when the layout is "
        "not claimed");
  m.def("describe_layout", &describe_layout,
        "Layout decision and launch geometry, without launching");
}
