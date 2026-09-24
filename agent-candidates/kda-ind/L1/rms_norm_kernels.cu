// Single-pass RMSNorm kernels with stride-aware row addressing.
//
// Two things separate these kernels from the vendored vLLM ones they replace:
//
//   * The row is read from global memory exactly once. The vendored kernel
//     streams the row to accumulate the variance and then re-reads it for the
//     normalize pass; every hidden size we care about fits in registers, so the
//     row is held there across the reduction instead.
//   * Leading dimensions are analysed on the host and collapsed to at most two
//     (size, stride) pairs, so a strided view normalizes in place without the
//     caller having to materialize a contiguous copy first.
//
// Layouts this file does not claim are rejected by the host-side predicates
// below, which report "not handled" in-band so the caller can route to the
// vendored operator. There is deliberately no exception caught around a launch:
// a launch that starts has to be a launch that is correct.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
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
constexpr int kMaxBundlesPerThread = 2;
// Both flavors grid-stride over rows, so clamping the grid to the launch limit
// costs nothing but keeps a row count above 2^31 from wrapping the dimension.
constexpr int64_t kMaxGrid = 2147483647;
// Ceiling on the block-per-row grid, in waves of the device's multiprocessors.
// One block per row is the obvious choice and it is not the fast one: at ~10^6
// rows it launches ~10^6 CTAs and reloads the weight row in every one of them.
// Capping the grid makes each block grid-stride over many rows and amortize its
// weight registers across them, which measured 826 us -> 700 us on the widest
// captured shape. The curve is flat from about 96 waves upward; 256 sits well
// inside the plateau. Shapes with fewer rows than the cap are unaffected.
constexpr int64_t kGridWavesCap = 256;

// ---------------------------------------------------------------------------
// Element traits
//
// Only the two 16-bit floating types get a hand-written path. They are the only
// dtypes the captured traffic uses and the ones where packed arithmetic pays.
// Anything else routes to the vendored operator, which is correct for it.
// ---------------------------------------------------------------------------

template <typename T>
struct ElemTraits;

template <>
struct ElemTraits<__nv_bfloat16> {
  using pack_t = __nv_bfloat162;
  __device__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static float2 to_f2(pack_t x) { return __bfloat1622float2(x); }
  __device__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16(x); }
  __device__ static pack_t add(pack_t a, pack_t b) { return __hadd2(a, b); }
  __device__ static __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
  }
};

template <>
struct ElemTraits<__half> {
  using pack_t = __half2;
  __device__ static float to_f(__half x) { return __half2float(x); }
  __device__ static float2 to_f2(pack_t x) { return __half22float2(x); }
  __device__ static __half from_f(float x) { return __float2half_rn(x); }
  __device__ static pack_t add(pack_t a, pack_t b) { return __hadd2(a, b); }
  __device__ static __half add(__half a, __half b) { return __hadd(a, b); }
};

// A VEC-wide bundle of elements, aligned so the compiler emits one vector
// load/store per bundle. VEC is a power of two <= 8 over a 2-byte element, so
// the alignment is a power of two <= 16.
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Bundle {
  T e[VEC];
};

// Sum of squares of one bundle, accumulated in fp32 from float-converted lanes.
// Pairs convert together, matching the vendored `_f16Vec::sum_squares`.
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

// `dst += src` in the element dtype, pairwise-packed. This is the vendored
// `_f16Vec::operator+=`, reproduced so the residual handed back is bit-identical
// to the baseline's and the variance sees the same rounded values.
template <typename T, int VEC>
__device__ __forceinline__ void add_inplace(Bundle<T, VEC>& dst,
                                            const Bundle<T, VEC>& src) {
  using Tr = ElemTraits<T>;
  if constexpr (VEC % 2 == 0) {
#pragma unroll
    for (int i = 0; i < VEC; i += 2) {
      typename Tr::pack_t p =
          Tr::add(typename Tr::pack_t{dst.e[i], dst.e[i + 1]},
                  typename Tr::pack_t{src.e[i], src.e[i + 1]});
      dst.e[i] = p.x;
      dst.e[i + 1] = p.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) dst.e[i] = Tr::add(dst.e[i], src.e[i]);
  }
}

// out = (T)(x * scale * w), in fp32 and in that multiplication order.
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
// Row addressing
//
// The host collapses the leading dimensions to `lead` surviving (size, stride)
// pairs. `lead == 1` also carries the zero-surviving-pair case: there is exactly
// one row, so the index is always 0 and the stride never multiplies anything.
// That keeps the kernels down to two addressing specializations.
// ---------------------------------------------------------------------------

struct RowMap {
  int64_t rows;     // total number of rows
  int64_t inner;    // size of the inner surviving pair (lead == 2 only)
  int64_t stride0;  // stride of the outer surviving pair
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
// Warp-segment flavor: `lanes` adjacent lanes of one warp own one row.
//
// Used when a row is at most one warp wide in bundles, which is exactly where
// the vendored launcher would start a 16-thread block. The reduction is a
// shuffle butterfly inside the segment: no shared memory and no barrier.
//
// `lanes` is a runtime power of two <= 32, so the butterfly mask is the
// segment's own lanes. A segment's row index is identical across its lanes, so
// the `row < rows` test is segment-uniform and the masked shuffles stay
// converged even when a sibling segment in the same warp has run out of rows.
// The block is always a warp multiple, so a segment never straddles two warps.
// ---------------------------------------------------------------------------

template <typename T, int VEC, int LEAD>
__global__ void rms_norm_warp_rows(T* __restrict__ out, const T* __restrict__ in,
                                   const T* __restrict__ weight, RowMap map,
                                   int hidden, int lanes, float eps) {
  using B = Bundle<T, VEC>;

  const int lane_in_seg = threadIdx.x & (lanes - 1);
  const int segs_per_block = blockDim.x / lanes;
  const int64_t seg =
      static_cast<int64_t>(blockIdx.x) * segs_per_block + (threadIdx.x / lanes);
  const int64_t seg_stride = static_cast<int64_t>(gridDim.x) * segs_per_block;

  const unsigned mask =
      (lanes == kWarp)
          ? 0xffffffffu
          : (((1u << lanes) - 1u) << ((threadIdx.x & (kWarp - 1)) & ~(lanes - 1)));

  // One bundle of weight per lane covers the whole row, so it is loaded once
  // for the lifetime of the thread rather than once per row.
  const B w = reinterpret_cast<const B*>(weight)[lane_in_seg];

  for (int64_t row = seg; row < map.rows; row += seg_stride) {
    const B x =
        reinterpret_cast<const B*>(in + row_offset<LEAD>(map, row))[lane_in_seg];

    float acc = sum_squares(x);
    for (int off = lanes >> 1; off; off >>= 1) {
      acc += __shfl_xor_sync(mask, acc, off);
    }
    const float scale = rsqrtf(acc / hidden + eps);

    reinterpret_cast<B*>(out + row * hidden)[lane_in_seg] =
        scale_and_weight(x, scale, w);
  }
}

// ---------------------------------------------------------------------------
// Block-per-row flavor: one block owns one row, grid-striding over rows.
//
// The block size is always a warp multiple — deriving it from the bundle count
// would give 360 threads at hidden=2880, leaving a partial final warp — so the
// tail lanes are predicated instead. Each thread keeps up to BPT bundles of the
// row in registers across the reduction.
//
// Two barriers per row are required: one after the per-warp partials are
// published, and one at the end of the row so the next grid-stride iteration
// cannot overwrite those partials while a lagging warp is still reading them.
// ---------------------------------------------------------------------------

template <typename T, int VEC, int LEAD, int BPT>
__global__ void rms_norm_block_rows(T* __restrict__ out, const T* __restrict__ in,
                                    const T* __restrict__ weight, RowMap map,
                                    int hidden, int bundles, float eps) {
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
    const B* __restrict__ src =
        reinterpret_cast<const B*>(in + row_offset<LEAD>(map, row));

    B held[BPT];
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < bundles) {
        held[j] = src[i];
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

    B* __restrict__ dst = reinterpret_cast<B*>(out + row * hidden);
#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < bundles) dst[i] = scale_and_weight(held[j], scale, wheld[j]);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Fused residual add + normalize, one block per row.
//
// Both tensors are contiguous here (the host predicate requires it), which is
// also what the vendored `fused_add_rms_norm` requires of the residual. The sum
// is written back through `residual` and the normalized row through `x`, in
// place, matching the baseline's aliasing.
// ---------------------------------------------------------------------------

template <typename T, int VEC, int BPT>
__global__ void fused_add_rms_norm_rows(T* __restrict__ x, T* __restrict__ res,
                                        const T* __restrict__ weight,
                                        int64_t rows, int hidden, int bundles,
                                        float eps) {
  using B = Bundle<T, VEC>;
  extern __shared__ float partials[];

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

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = row * bundles;
    B* __restrict__ xv = reinterpret_cast<B*>(x) + base;
    B* __restrict__ rv = reinterpret_cast<B*>(res) + base;

    B held[BPT];
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < bundles) {
        held[j] = xv[i];
        add_inplace(held[j], rv[i]);
        acc += sum_squares(held[j]);
        rv[i] = held[j];
      }
    }

#pragma unroll
    for (int off = kWarp >> 1; off; off >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) partials[warp] = acc;
    __syncthreads();

    float total = 0.0f;
    for (int j = 0; j < warps; ++j) total += partials[j];
    const float scale = rsqrtf(total / hidden + eps);

#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < bundles) xv[i] = scale_and_weight(held[j], scale, wheld[j]);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host-side layout analysis
// ---------------------------------------------------------------------------

// Result of collapsing the leading dimensions. `ok == false` means the layout
// needs more than two surviving (size, stride) pairs, which no captured layout
// does; such a tensor goes to the vendored operator rather than being indexed
// with a formula that does not describe it.
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

  // Squeeze size-1 leading dims: they contribute nothing to row addressing and
  // their strides are unconstrained.
  constexpr int kMaxDims = 8;
  int64_t sz[kMaxDims], st[kMaxDims];
  int n = 0;
  for (int d = 0; d < nd - 1; ++d) {
    if (t.size(d) == 1) continue;
    if (n == kMaxDims) return L;  // absurd rank; let the vendored op have it
    sz[n] = t.size(d);
    st[n] = t.stride(d);
    ++n;
  }

  // Merge adjacent pairs from the inner side while the outer stride is exactly
  // the inner extent, repeating until no merge applies. A fully contiguous
  // tensor of any rank collapses to a single pair this way.
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
    // A single row: describe it as one pair whose index is always zero.
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

// Widest bundle usable for this layout: the hidden size must divide evenly, the
// base pointers must be bundle-aligned, and *every surviving leading stride*
// must be a multiple of the bundle width. A base-pointer check alone would
// accept a layout whose second row starts mid-bundle.
int choose_vec(const at::Tensor& x, const Layout& L, int64_t hidden,
               const at::Tensor& weight, const at::Tensor& extra) {
  const size_t esz = x.element_size();
  const int max_vec = static_cast<int>(16 / esz);
  for (int v = max_vec; v >= 1; v >>= 1) {
    if (hidden % v != 0) continue;
    if (!aligned_for(x.const_data_ptr(), v, esz)) continue;
    if (!aligned_for(weight.const_data_ptr(), v, esz)) continue;
    if (extra.defined() && !aligned_for(extra.const_data_ptr(), v, esz)) continue;
    bool strides_ok = true;
    for (int i = 0; i < L.lead; ++i) {
      if (L.strides[i] % v != 0) strides_ok = false;
    }
    if (!strides_ok) continue;
    return v;
  }
  return 0;
}

bool is_pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

// Multiprocessor count, read once per process. The grid cap is expressed in waves
// so it follows the device instead of being a magic block count.
int sm_count() {
  static const int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

// Whether two contiguous tensors cover intersecting bytes. Only meaningful once
// contiguity is established, which is where it is used.
bool ranges_overlap(const at::Tensor& a, const at::Tensor& b) {
  const char* pa = static_cast<const char*>(a.const_data_ptr());
  const char* pb = static_cast<const char*>(b.const_data_ptr());
  const char* ea = pa + a.numel() * a.element_size();
  const char* eb = pb + b.numel() * b.element_size();
  return pa < eb && pb < ea;
}

// Smallest warp multiple at least `v`, kept in 64-bit. Rounding a bundle count
// near INT_MAX in 32-bit wraps the result negative, and a negative block size
// then passes the `<= kMaxBlock` bound and reaches an invalid launch.
int64_t round_up_warp(int64_t v) { return ((v + kWarp - 1) / kWarp) * kWarp; }

// Launch geometry. `block == 0` means the shape is not handled.
struct Plan {
  bool warp_flavor = false;
  int block = 0;
  int bpt = 1;
  int lanes = 0;   // warp flavor only
  int64_t grid = 0;
};

// Overrides for the tuning entry point. Zero means "decide automatically".
struct Override {
  int block = 0;
  int bpt = 0;
  int64_t grid = 0;
};

Plan plan_launch(int bundles, int64_t rows, const Override& ov) {
  Plan p;

  // A row that fits inside one warp takes the shuffle-only flavor.
  if (bundles <= kWarp && is_pow2(bundles) && ov.block == 0 && ov.bpt == 0) {
    p.warp_flavor = true;
    p.lanes = bundles;
    // Enough threads to cover the rows, capped at 256 and rounded to a warp so
    // a sub-warp launch can never produce an invalid shuffle mask.
    const int64_t want = rows * p.lanes;
    p.block = (want < 256) ? static_cast<int>(round_up_warp(want)) : 256;
    if (p.block < kWarp) p.block = kWarp;
    const int segs = p.block / p.lanes;
    p.grid = ov.grid > 0 ? ov.grid : (rows + segs - 1) / segs;
    if (p.grid > kMaxGrid) p.grid = kMaxGrid;
    return p;
  }

  if (ov.block > 0) {
    if (ov.block % kWarp != 0 || ov.block > kMaxBlock) return p;
    const int bpt = ov.bpt > 0 ? ov.bpt : 1;
    if (bpt > kMaxBundlesPerThread) return p;
    if (bundles > ov.block * bpt) return p;
    p.block = ov.block;
    p.bpt = bpt;
  } else {
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
  }
  p.grid = ov.grid > 0
               ? std::min<int64_t>(ov.grid, rows)
               : std::min<int64_t>(rows, kGridWavesCap * sm_count());
  if (p.grid > kMaxGrid) p.grid = kMaxGrid;
  return p;
}

// ---------------------------------------------------------------------------
// Launch dispatch
//
// The instantiation matrix is deliberately closed and small:
//   warp flavor  : 2 dtypes x 4 bundle widths x 2 addressings      = 16
//   block flavor : 2 dtypes x 4 bundle widths x 2 addressings x 2  = 32
//   fused flavor : 2 dtypes x 4 bundle widths x 2                  = 16
// for 64 kernels in one translation unit.
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
  if ((b) == 1) {            \
    constexpr int BPT = 1;   \
    __VA_ARGS__;             \
  } else {                   \
    constexpr int BPT = 2;   \
    __VA_ARGS__;             \
  }

template <typename T>
bool launch_norm(T* out, const T* in, const T* weight, const Layout& L,
                 int hidden, int vec, float eps, const Override& ov,
                 cudaStream_t stream) {
  RowMap map;
  map.rows = L.rows;
  map.inner = L.sizes[1];
  map.stride0 = L.strides[0];
  map.stride1 = L.strides[1];

  const int bundles = hidden / vec;
  const Plan p = plan_launch(bundles, L.rows, ov);
  if (p.block == 0 || p.grid <= 0) return false;

  if (p.warp_flavor) {
    DISPATCH_VEC(vec, DISPATCH_LEAD(L.lead, {
                   rms_norm_warp_rows<T, VEC, LEAD>
                       <<<static_cast<unsigned>(p.grid), p.block, 0, stream>>>(
                           out, in, weight, map, hidden, p.lanes, eps);
                 }));
    return true;
  }

  const size_t smem = static_cast<size_t>(p.block / kWarp) * sizeof(float);
  DISPATCH_VEC(vec, DISPATCH_LEAD(L.lead, DISPATCH_BPT(p.bpt, {
                 rms_norm_block_rows<T, VEC, LEAD, BPT>
                     <<<static_cast<unsigned>(p.grid), p.block, smem, stream>>>(
                         out, in, weight, map, hidden, bundles, eps);
               })));
  return true;
}

template <typename T>
bool launch_fused(T* x, T* res, const T* weight, int64_t rows, int hidden,
                  int vec, float eps, const Override& ov, cudaStream_t stream) {
  const int bundles = hidden / vec;
  Override forced = ov;
  // The fused flavor has no warp-segment variant -- a residual row is at least
  // 512 elements wide in this workload -- so a row narrow enough that the
  // planner would pick one is pinned to a single-warp block instead. Wider rows
  // keep the automatic block / bundles-per-thread ladder, which is what lets a
  // row too wide for one block still be served.
  if (forced.block == 0 && bundles <= kWarp) forced.block = kWarp;
  const Plan p = plan_launch(bundles, rows, forced);
  if (p.block == 0 || p.grid <= 0 || p.warp_flavor) return false;

  const size_t smem = static_cast<size_t>(p.block / kWarp) * sizeof(float);
  DISPATCH_VEC(vec, DISPATCH_BPT(p.bpt, {
                 fused_add_rms_norm_rows<T, VEC, BPT>
                     <<<static_cast<unsigned>(p.grid), p.block, smem, stream>>>(
                         x, res, weight, rows, hidden, bundles, eps);
               }));
  return true;
}

// ---------------------------------------------------------------------------
// Shared front end for both entry points
// ---------------------------------------------------------------------------

at::Tensor norm_impl(const at::Tensor& x, const at::Tensor& weight, double eps,
                     const Override& ov) {
  if (!x.is_cuda() || !dtype_supported(x) || x.dim() < 1) return at::Tensor();
  const int64_t hidden = x.size(-1);
  if (hidden <= 0 || hidden > std::numeric_limits<int>::max()) return at::Tensor();
  if (x.stride(-1) != 1) return at::Tensor();
  if (!weight_ok(weight, x, hidden)) return at::Tensor();

  const Layout L = collapse_leading(x);
  if (!L.ok || L.rows != x.numel() / hidden) return at::Tensor();

  const int vec = choose_vec(x, L, hidden, weight, at::Tensor());
  if (vec == 0) return at::Tensor();

  const c10::cuda::CUDAGuard guard(x.device());
  // Allocated straight from the CUDA allocator rather than through the
  // `at::empty` dispatcher hop, which measured 0.26 us cheaper per call -- worth
  // having when a single-row shape spends its whole window on the host. Safe
  // because this path is never reached under tracing: `forward` routes to the
  // pure-PyTorch implementation whenever the compiler is active.
  at::Tensor out = at::detail::empty_cuda(x.sizes(), x.scalar_type(), x.device(),
                                          c10::MemoryFormat::Contiguous);
  if (L.rows == 0) return out;  // nothing to normalize; an empty grid is illegal

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float epsf = static_cast<float>(eps);
  const int hid = static_cast<int>(hidden);

  const bool launched =
      (x.scalar_type() == at::kBFloat16)
          ? launch_norm<__nv_bfloat16>(
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr()),
                L, hid, vec, epsf, ov, stream)
          : launch_norm<__half>(
                reinterpret_cast<__half*>(out.data_ptr()),
                reinterpret_cast<const __half*>(x.const_data_ptr()),
                reinterpret_cast<const __half*>(weight.const_data_ptr()), L, hid,
                vec, epsf, ov, stream);
  if (!launched) return at::Tensor();
  return out;
}

bool fused_impl(at::Tensor& x, at::Tensor& residual, const at::Tensor& weight,
                double eps, const Override& ov) {
  if (!x.is_cuda() || !dtype_supported(x) || x.dim() < 1) return false;
  if (!residual.defined() || residual.scalar_type() != x.scalar_type()) return false;
  if (residual.sizes() != x.sizes() || residual.device() != x.device()) return false;

  // The sum is written back through `residual` and the normalized row through
  // `x`, both by flat row index, so both must be contiguous. That is the only
  // layout the baseline ever reaches this path with, since it forces contiguity
  // on both arguments first.
  if (!x.is_contiguous() || !residual.is_contiguous()) return false;

  // Both arguments are written, and the kernel declares them `__restrict__`, so
  // a pair that shares bytes -- two overlapping views of one buffer -- would be
  // undefined behaviour rather than merely producing a different answer.
  if (ranges_overlap(x, residual)) return false;

  const int64_t hidden = x.size(-1);
  if (hidden <= 0 || hidden > std::numeric_limits<int>::max()) return false;
  if (!weight_ok(weight, x, hidden)) return false;

  Layout L;
  L.ok = true;
  L.lead = 1;
  L.rows = x.numel() / hidden;
  L.sizes[0] = L.rows;
  L.strides[0] = hidden;

  const int vec = choose_vec(x, L, hidden, weight, residual);
  if (vec == 0) return false;
  if (L.rows == 0) return true;  // nothing to do, and correctly so

  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float epsf = static_cast<float>(eps);
  const int hid = static_cast<int>(hidden);

  if (x.scalar_type() == at::kBFloat16) {
    return launch_fused<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.const_data_ptr()), L.rows,
        hid, vec, epsf, ov, stream);
  }
  return launch_fused<__half>(
      reinterpret_cast<__half*>(x.data_ptr()),
      reinterpret_cast<__half*>(residual.data_ptr()),
      reinterpret_cast<const __half*>(weight.const_data_ptr()), L.rows, hid, vec,
      epsf, ov, stream);
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry points
//
// Both report "not handled" in-band rather than by raising: an undefined tensor
// (which reaches Python as None) and a false return respectively. The caller
// uses that answer to route to the vendored operator.
// ---------------------------------------------------------------------------

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
  return norm_impl(x, weight, eps, Override{});
}

bool fused_add_rms_norm(at::Tensor& x, at::Tensor& residual,
                        const at::Tensor& weight, double eps) {
  return fused_impl(x, residual, weight, eps, Override{});
}

// Launch-geometry overrides, for the local configuration sweep only. The scored
// path never goes through these, so it never pays for the extra arguments.
at::Tensor rms_norm_tuned(const at::Tensor& x, const at::Tensor& weight,
                          double eps, int64_t block, int64_t bpt, int64_t grid) {
  Override ov;
  ov.block = static_cast<int>(block);
  ov.bpt = static_cast<int>(bpt);
  ov.grid = grid;
  return norm_impl(x, weight, eps, ov);
}

bool fused_add_rms_norm_tuned(at::Tensor& x, at::Tensor& residual,
                              const at::Tensor& weight, double eps,
                              int64_t block, int64_t bpt, int64_t grid) {
  Override ov;
  ov.block = static_cast<int>(block);
  ov.bpt = static_cast<int>(bpt);
  ov.grid = grid;
  return fused_impl(x, residual, weight, eps, ov);
}

// Layout analysis exposed for testing: the surviving (size, stride) pairs, the
// row count and the chosen bundle width, without launching anything.
std::vector<int64_t> describe_layout(const at::Tensor& x,
                                     const at::Tensor& weight) {
  const int64_t hidden = x.size(-1);
  const Layout L = collapse_leading(x);
  std::vector<int64_t> desc;
  desc.push_back(L.ok ? 1 : 0);
  if (!L.ok) return desc;
  desc.push_back(L.lead);
  desc.push_back(L.rows);
  for (int i = 0; i < 2; ++i) {
    desc.push_back(L.sizes[i]);
    desc.push_back(L.strides[i]);
  }
  desc.push_back(weight_ok(weight, x, hidden)
                     ? choose_vec(x, L, hidden, weight, at::Tensor())
                     : 0);
  return desc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm,
        "Single-pass RMSNorm; None when the layout is not claimed");
  m.def("fused_add_rms_norm", &fused_add_rms_norm,
        "In-place fused residual add + RMSNorm; false when not claimed");
  m.def("rms_norm_tuned", &rms_norm_tuned, "rms_norm with launch overrides");
  m.def("fused_add_rms_norm_tuned", &fused_add_rms_norm_tuned,
        "fused_add_rms_norm with launch overrides");
  m.def("describe_layout", &describe_layout,
        "Collapsed leading (size, stride) pairs and bundle width, for tests");
}
