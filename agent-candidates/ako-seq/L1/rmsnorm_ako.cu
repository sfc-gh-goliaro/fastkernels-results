// Hidden-size-specialized RMSNorm / fused-add-RMSNorm for 16-bit dtypes.
//
// Launch geometry is chosen for the *shape* instead of being inherited from
// hidden_size:
//   * kern A ("warp"):  LPR (<= 32) lanes cooperate on one row using 128-bit
//     vector loads and shuffle-only reductions -- no shared memory and no
//     __syncthreads.  RPB rows per block, grid-stride over rows, with RPB
//     picked so the grid still spreads across every SM.
//   * kern B ("block"): one row per block, several warps, grid == num_rows.
//     Chosen for wide rows, with the block width set by wave residency (see
//     pick_mode): the widest block whose whole grid still fits in one wave.
//   * kern G ("generic"): runtime hidden_size fallback, one row per block.
//
// The plain path reads its input through a RowMap, so a *strided* input (a
// captured shape is `[651,16,128]` with stride `[2304,128,1]`, i.e. the Q slice
// of a fused QKV projection) is gathered row-by-row by the norm kernel itself
// instead of being materialized by a preceding `.contiguous()` copy.  That copy
// is a general strided TensorIterator kernel and costs ~6.8 us on that shape --
// 2.7x the norm kernel it feeds, and two grid steps of the scored metric.
//
// Numerics mirror the vLLM reference: variance accumulated in fp32, the
// residual add performed in packed 16-bit arithmetic, and the output written
// as ((x * rsqrt(var/H + eps)) * w) with a single rounding.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <utility>
#include <cstdint>
#include <cstdlib>

namespace ako {

// --------------------------------------------------------------------------
// dtype traits
// --------------------------------------------------------------------------
template <typename D>
struct Tr;

template <>
struct Tr<__nv_bfloat16> {
  using pk = __nv_bfloat162;
  static __device__ __forceinline__ float2 to_f(pk v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ pk from_f(float2 v) { return __float22bfloat162_rn(v); }
  static __device__ __forceinline__ pk add(pk a, pk b) { return __hadd2(a, b); }
};

template <>
struct Tr<__half> {
  using pk = __half2;
  static __device__ __forceinline__ float2 to_f(pk v) { return __half22float2(v); }
  static __device__ __forceinline__ pk from_f(float2 v) { return __float22half2_rn(v); }
  static __device__ __forceinline__ pk add(pk a, pk b) { return __hadd2(a, b); }
};

// 128-bit vector of eight 16-bit elements.
template <typename D, int W>
struct alignas(16) Vec {
  D d[W];
};

template <typename D, int W>
__device__ __forceinline__ float sumsq(const Vec<D, W>& v) {
  using T = Tr<D>;
  using pk = typename T::pk;
  float r = 0.f;
#pragma unroll
  for (int i = 0; i < W; i += 2) {
    float2 z = T::to_f(pk{v.d[i], v.d[i + 1]});
    r += z.x * z.x + z.y * z.y;
  }
  return r;
}

template <typename D, int W>
__device__ __forceinline__ Vec<D, W> addv(const Vec<D, W>& a, const Vec<D, W>& b) {
  using T = Tr<D>;
  using pk = typename T::pk;
  Vec<D, W> o;
#pragma unroll
  for (int i = 0; i < W; i += 2) {
    pk p = T::add(pk{a.d[i], a.d[i + 1]}, pk{b.d[i], b.d[i + 1]});
    o.d[i] = p.x;
    o.d[i + 1] = p.y;
  }
  return o;
}

// out = (x * s) * w with a single rounding (reference expression order).
template <typename D, int W>
__device__ __forceinline__ Vec<D, W> scale_mul(const Vec<D, W>& v, float s,
                                               const Vec<D, W>& w) {
  using T = Tr<D>;
  using pk = typename T::pk;
  Vec<D, W> o;
#pragma unroll
  for (int i = 0; i < W; i += 2) {
    float2 z = T::to_f(pk{v.d[i], v.d[i + 1]});
    float2 g = T::to_f(pk{w.d[i], w.d[i + 1]});
    float2 r;
    r.x = z.x * s * g.x;
    r.y = z.y * s * g.y;
    pk p = T::from_f(r);
    o.d[i] = p.x;
    o.d[i + 1] = p.y;
  }
  return o;
}

// --------------------------------------------------------------------------
// Row addressing
//
// A row (the reduced, innermost dimension) is always contiguous; the *rows*
// need not be.  The leading dimensions are collapsed on the host into at most
// three (extent, stride) groups, so a row index maps to an offset with at most
// two integer divisions -- paid once per row, against a 256-byte load burst.
// STRIDED == false collapses all of this to `row * NV` at compile time.
// --------------------------------------------------------------------------
struct RowMap {
  int ext[3];      // group extents, outermost -> innermost
  int64_t es[3];   // element strides, outermost -> innermost
  int64_t vs[3];   // the same strides in 16-byte vector units
  int n;
};

template <bool STRIDED, typename OFF>
__device__ __forceinline__ int64_t row_off(const RowMap& m, int row, int span,
                                           const OFF* stride_tab) {
  if constexpr (!STRIDED) {
    return (int64_t)row * span;
  } else {
    if (m.n == 1) return (int64_t)row * stride_tab[0];
    if (m.n == 2) {
      const int i1 = row % m.ext[1];
      const int i0 = row / m.ext[1];
      return (int64_t)i0 * stride_tab[0] + (int64_t)i1 * stride_tab[1];
    }
    const int i2 = row % m.ext[2];
    const int t = row / m.ext[2];
    const int i1 = t % m.ext[1];
    const int i0 = t / m.ext[1];
    return (int64_t)i0 * stride_tab[0] + (int64_t)i1 * stride_tab[1] +
           (int64_t)i2 * stride_tab[2];
  }
}

template <bool STRIDED>
__device__ __forceinline__ int64_t vrow(const RowMap& m, int row, int nv) {
  return row_off<STRIDED>(m, row, nv, m.vs);
}

template <bool STRIDED>
__device__ __forceinline__ int64_t erow(const RowMap& m, int row, int h) {
  return row_off<STRIDED>(m, row, h, m.es);
}

// ==========================================================================
// kern A -- LPR lanes per row, shuffle-only reduction, RPB rows per block
// ==========================================================================
template <int LPR>
__device__ __forceinline__ float sub_warp_sum(float acc, unsigned mask) {
#pragma unroll
  for (int off = LPR / 2; off > 0; off >>= 1) acc += __shfl_xor_sync(mask, acc, off);
  return acc;
}

template <typename D, int H, int LPR, int RPB, bool STRIDED>
__global__ __launch_bounds__(LPR* RPB) void rmsn_warp(
    D* __restrict__ out, const D* __restrict__ in, const D* __restrict__ w,
    const float eps, const int num_rows, const RowMap map) {
  constexpr int VEC = 8;
  constexpr int VPT = H / (VEC * LPR);
  static_assert(H == VEC * LPR * VPT, "bad specialization");
  constexpr int NV = H / VEC;
  using V = Vec<D, VEC>;

  const int tid = threadIdx.x;
  const int lane = tid & (LPR - 1);
  const int grp = tid / LPR;
  constexpr unsigned LMASK = (LPR >= 32) ? 0xffffffffu : ((1u << LPR) - 1u);
  const unsigned mask = LMASK << ((tid & 31) - lane);

  // The weight is loop-invariant: load it once, at kernel entry, so the
  // scale-and-store pass never waits on a load issued *after* the reduction.
  const V* __restrict__ vw = reinterpret_cast<const V*>(w);
  V wv[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) wv[i] = vw[lane + i * LPR];

  const int stride = gridDim.x * RPB;
  for (int row = blockIdx.x * RPB + grp; row < num_rows; row += stride) {
    const V* __restrict__ vin =
        reinterpret_cast<const V*>(in) + vrow<STRIDED>(map, row, NV);
    V buf[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) buf[i] = vin[lane + i * LPR];

    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) acc += sumsq(buf[i]);
    acc = sub_warp_sum<LPR>(acc, mask);
    const float s = rsqrtf(acc / (float)H + eps);

    V* __restrict__ vout = reinterpret_cast<V*>(out) + (int64_t)row * NV;
#pragma unroll
    for (int i = 0; i < VPT; ++i) vout[lane + i * LPR] = scale_mul(buf[i], s, wv[i]);
  }
}

template <typename D, int H, int LPR, int RPB>
__global__ __launch_bounds__(LPR* RPB) void rmsn_add_warp(
    D* io, D* res, const D* __restrict__ w, const float eps, const int num_rows) {
  constexpr int VEC = 8;
  constexpr int VPT = H / (VEC * LPR);
  static_assert(H == VEC * LPR * VPT, "bad specialization");
  constexpr int NV = H / VEC;
  using V = Vec<D, VEC>;

  const int tid = threadIdx.x;
  const int lane = tid & (LPR - 1);
  const int grp = tid / LPR;
  constexpr unsigned LMASK = (LPR >= 32) ? 0xffffffffu : ((1u << LPR) - 1u);
  const unsigned mask = LMASK << ((tid & 31) - lane);

  // The weight is loop-invariant: load it once, at kernel entry, so the
  // scale-and-store pass never waits on a load issued *after* the reduction.
  const V* __restrict__ vw = reinterpret_cast<const V*>(w);
  V wv[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) wv[i] = vw[lane + i * LPR];

  const int stride = gridDim.x * RPB;
  for (int row = blockIdx.x * RPB + grp; row < num_rows; row += stride) {
    V* vio = reinterpret_cast<V*>(io) + (int64_t)row * NV;
    V* vres = reinterpret_cast<V*>(res) + (int64_t)row * NV;
    V t[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const V a = vio[lane + i * LPR];
      const V b = vres[lane + i * LPR];
      t[i] = addv(a, b);
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      vres[lane + i * LPR] = t[i];
      acc += sumsq(t[i]);
    }
    acc = sub_warp_sum<LPR>(acc, mask);
    const float s = rsqrtf(acc / (float)H + eps);
#pragma unroll
    for (int i = 0; i < VPT; ++i) vio[lane + i * LPR] = scale_mul(t[i], s, wv[i]);
  }
}

// ==========================================================================
// kern B -- one row per block, BLOCK threads (row-starved shapes)
// ==========================================================================
// REENTRANT == false drops the leading barrier, which exists only to keep a
// previous row iteration's readers of sm[] out of this row's writes.  The
// specialized block kernels are launched with grid == num_rows, so they own
// exactly one row and never need it.
template <int BLOCK, bool REENTRANT = true>
__device__ __forceinline__ float block_sum(float acc, float* sm, int tid) {
  constexpr int NW = BLOCK / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if constexpr (NW == 1) return acc;
  if constexpr (REENTRANT) __syncthreads();
  if ((tid & 31) == 0) sm[tid >> 5] = acc;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot += sm[i];
  return tot;
}

template <typename D, int H, int BLOCK, bool STRIDED>
__global__ __launch_bounds__(BLOCK) void rmsn_block(
    D* __restrict__ out, const D* __restrict__ in, const D* __restrict__ w,
    const float eps, const int num_rows, const RowMap map) {
  constexpr int VEC = 8;
  constexpr int VPT = H / (VEC * BLOCK);
  static_assert(H == VEC * BLOCK * VPT, "bad specialization");
  constexpr int NV = H / VEC;
  using V = Vec<D, VEC>;
  __shared__ float sm[BLOCK / 32];

  const int tid = threadIdx.x;
  const V* __restrict__ vw = reinterpret_cast<const V*>(w);
  // Issued before the first row's loads, so the store pass after the barrier
  // never stalls on the weight.
  V wv[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) wv[i] = vw[tid + i * BLOCK];

  const int row = blockIdx.x;  // grid == num_rows
  if (row >= num_rows) return;
  const V* __restrict__ vin =
      reinterpret_cast<const V*>(in) + vrow<STRIDED>(map, row, NV);
  V buf[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) buf[i] = vin[tid + i * BLOCK];
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) acc += sumsq(buf[i]);
  acc = block_sum<BLOCK, false>(acc, sm, tid);
  const float s = rsqrtf(acc / (float)H + eps);
  V* __restrict__ vout = reinterpret_cast<V*>(out) + (int64_t)row * NV;
#pragma unroll
  for (int i = 0; i < VPT; ++i)
    vout[tid + i * BLOCK] = scale_mul(buf[i], s, wv[i]);
}

template <typename D, int H, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rmsn_add_block(
    D* io, D* res, const D* __restrict__ w, const float eps, const int num_rows) {
  constexpr int VEC = 8;
  constexpr int VPT = H / (VEC * BLOCK);
  static_assert(H == VEC * BLOCK * VPT, "bad specialization");
  constexpr int NV = H / VEC;
  using V = Vec<D, VEC>;
  __shared__ float sm[BLOCK / 32];

  const int tid = threadIdx.x;
  const V* __restrict__ vw = reinterpret_cast<const V*>(w);
  V wv[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) wv[i] = vw[tid + i * BLOCK];

  const int row = blockIdx.x;  // grid == num_rows
  if (row >= num_rows) return;
  V* vio = reinterpret_cast<V*>(io) + (int64_t)row * NV;
  V* vres = reinterpret_cast<V*>(res) + (int64_t)row * NV;
  V t[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const V a = vio[tid + i * BLOCK];
    const V b = vres[tid + i * BLOCK];
    t[i] = addv(a, b);
  }
  float acc = 0.f;
  // The residual store is issued here, before the reduction, not after it:
  // res = x + r is known already, so the write overlaps the barrier instead of
  // queueing behind it.
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    vres[tid + i * BLOCK] = t[i];
    acc += sumsq(t[i]);
  }
  acc = block_sum<BLOCK, false>(acc, sm, tid);
  const float s = rsqrtf(acc / (float)H + eps);
#pragma unroll
  for (int i = 0; i < VPT; ++i)
    vio[tid + i * BLOCK] = scale_mul(t[i], s, wv[i]);
}

// ==========================================================================
// kern G -- runtime hidden size
// ==========================================================================
template <typename D, int BLOCK, bool STRIDED>
__global__ __launch_bounds__(BLOCK) void rmsn_generic_vec(
    D* __restrict__ out, const D* __restrict__ in, const D* __restrict__ w,
    const float eps, const int H, const int num_rows, const RowMap map) {
  constexpr int VEC = 8;
  using V = Vec<D, VEC>;
  __shared__ float sm[BLOCK / 32];
  const int tid = threadIdx.x;
  const int NV = H / VEC;
  const V* __restrict__ vw = reinterpret_cast<const V*>(w);
  for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {
    const V* __restrict__ vin =
        reinterpret_cast<const V*>(in) + vrow<STRIDED>(map, row, NV);
    float acc = 0.f;
    for (int i = tid; i < NV; i += BLOCK) acc += sumsq(vin[i]);
    acc = block_sum<BLOCK>(acc, sm, tid);
    const float s = rsqrtf(acc / (float)H + eps);
    V* __restrict__ vout = reinterpret_cast<V*>(out) + (int64_t)row * NV;
    for (int i = tid; i < NV; i += BLOCK) vout[i] = scale_mul(vin[i], s, vw[i]);
  }
}

template <typename D, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rmsn_add_generic_vec(
    D* io, D* res, const D* __restrict__ w, const float eps, const int H,
    const int num_rows) {
  constexpr int VEC = 8;
  using V = Vec<D, VEC>;
  __shared__ float sm[BLOCK / 32];
  const int tid = threadIdx.x;
  const int NV = H / VEC;
  const V* __restrict__ vw = reinterpret_cast<const V*>(w);
  for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {
    V* vio = reinterpret_cast<V*>(io) + (int64_t)row * NV;
    V* vres = reinterpret_cast<V*>(res) + (int64_t)row * NV;
    float acc = 0.f;
    for (int i = tid; i < NV; i += BLOCK) {
      V t = addv(vio[i], vres[i]);
      vres[i] = t;
      acc += sumsq(t);
    }
    acc = block_sum<BLOCK>(acc, sm, tid);
    const float s = rsqrtf(acc / (float)H + eps);
    for (int i = tid; i < NV; i += BLOCK) vio[i] = scale_mul(vres[i], s, vw[i]);
  }
}

template <typename D, int BLOCK, bool STRIDED>
__global__ __launch_bounds__(BLOCK) void rmsn_generic_scalar(
    D* __restrict__ out, const D* __restrict__ in, const D* __restrict__ w,
    const float eps, const int H, const int num_rows, const RowMap map) {
  __shared__ float sm[BLOCK / 32];
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {
    const D* rin = in + erow<STRIDED>(map, row, H);
    float acc = 0.f;
    for (int i = tid; i < H; i += BLOCK) {
      float x = (float)rin[i];
      acc += x * x;
    }
    acc = block_sum<BLOCK>(acc, sm, tid);
    const float s = rsqrtf(acc / (float)H + eps);
    D* rout = out + (int64_t)row * H;
    for (int i = tid; i < H; i += BLOCK)
      rout[i] = (D)((float)rin[i] * s * (float)w[i]);
  }
}

template <typename D, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rmsn_add_generic_scalar(
    D* io, D* res, const D* __restrict__ w, const float eps, const int H,
    const int num_rows) {
  __shared__ float sm[BLOCK / 32];
  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {
    D* rio = io + (int64_t)row * H;
    D* rres = res + (int64_t)row * H;
    float acc = 0.f;
    for (int i = tid; i < H; i += BLOCK) {
      D z = (D)((float)rio[i] + (float)rres[i]);
      rres[i] = z;
      float x = (float)z;
      acc += x * x;
    }
    acc = block_sum<BLOCK>(acc, sm, tid);
    const float s = rsqrtf(acc / (float)H + eps);
    for (int i = tid; i < H; i += BLOCK)
      rio[i] = (D)((float)rres[i] * s * (float)w[i]);
  }
}

// ==========================================================================
// host launch
// ==========================================================================
static int sm_count() {
  static int n = [] {
    int dev = 0, v = 148;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
    return v;
  }();
  return n;
}

static int threads_per_sm() {
  static int n = [] {
    int dev = 0, v = 2048;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    return v;
  }();
  return n;
}

static int env_int(const char* name, int fallback) {
  const char* s = std::getenv(name);
  if (!s || !*s) return fallback;
  int v = std::atoi(s);
  return v > 0 ? v : fallback;
}

// Largest RPB (power of two, <= MAXRPB) whose grid still covers TARGET blocks,
// so row-starved shapes keep spreading over the SMs instead of shrinking the
// grid.  Falls back to one row per block.
static int pick_rpb(int rows, int max_rpb) {
  const int target = env_int("AKO_RMSN_TARGET_BLOCKS", 2) * sm_count();
  int best = 1;
  for (int rpb = 1; rpb <= max_rpb; rpb <<= 1) {
    if ((rows + rpb - 1) / rpb >= target) best = rpb;
  }
  const int ov = env_int("AKO_RMSN_RPB", 0);
  if (ov > 0) best = std::min(ov, max_rpb);
  return best;
}

// Single launch point: keeps the explicit stream (never the legacy null
// stream, which is not capture-aware) in one place for every kernel below.
template <typename... ExpTypes, typename... ActTypes>
static inline void ako_launch(void (*kern)(ExpTypes...), int grid, int block,
                              cudaStream_t s, ActTypes&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cfg.attrs = nullptr;
  cfg.numAttrs = 0;
  cudaLaunchKernelEx(&cfg, kern, std::forward<ActTypes>(args)...);
}

static int grid_for(int rows, int rpb) { return (rows + rpb - 1) / rpb; }

// One rung of the rows-per-block ladder, for the plain (row-mapped) and the
// fused (contiguous-only) kernel families respectively.
#define AKO_WARP_RUNG_P(KERN, RPB)                                        \
  if constexpr (MAXRPB >= (RPB)) {                                        \
    if (rpb >= (RPB)) {                                                   \
      ako_launch(KERN<D, H, LPR, RPB, STRIDED>, grid_for(rows, RPB),       \
                 LPR*(RPB), s, out, in, w, eps, rows, m);                 \
      return;                                                             \
    }                                                                     \
  }

#define AKO_WARP_RUNG_F(KERN, RPB)                                        \
  if constexpr (MAXRPB >= (RPB)) {                                        \
    if (rpb >= (RPB)) {                                                   \
      ako_launch(KERN<D, H, LPR, RPB>, grid_for(rows, RPB), LPR*(RPB), s,  \
                 io, res, w, eps, rows);                                  \
      return;                                                             \
    }                                                                     \
  }

template <typename D, int H, int LPR, int MAXRPB, bool STRIDED>
static void warp_plain(D* out, const D* in, const D* w, float eps, int rows,
                       const RowMap& m, cudaStream_t s) {
  const int rpb = pick_rpb(rows, MAXRPB);
  AKO_WARP_RUNG_P(rmsn_warp, 16)
  AKO_WARP_RUNG_P(rmsn_warp, 8)
  AKO_WARP_RUNG_P(rmsn_warp, 4)
  AKO_WARP_RUNG_P(rmsn_warp, 2)
  ako_launch(rmsn_warp<D, H, LPR, 1, STRIDED>, grid_for(rows, 1), LPR, s, out,
             in, w, eps, rows, m);
}

template <typename D, int H, int LPR, int MAXRPB>
static void warp_fused(D* io, D* res, const D* w, float eps, int rows,
                       cudaStream_t s) {
  const int rpb = pick_rpb(rows, MAXRPB);
  AKO_WARP_RUNG_F(rmsn_add_warp, 16)
  AKO_WARP_RUNG_F(rmsn_add_warp, 8)
  AKO_WARP_RUNG_F(rmsn_add_warp, 4)
  AKO_WARP_RUNG_F(rmsn_add_warp, 2)
  ako_launch(rmsn_add_warp<D, H, LPR, 1>, grid_for(rows, 1), LPR, s, io, res, w,
             eps, rows);
}

// Modes: 1 = warp/sub-warp per row (VPT = H/(8*LPR)), 2/3/4 = one row per
// block with VPT = 1/2/4, 5 = runtime-H generic.
//
// For the one-row-per-block family the block width is set by *wave residency*,
// not by bytes in flight.  Widening the block does not increase the bytes one SM
// has outstanding -- that is (threads resident) x VPT x 16 either way -- it only
// buys warps to hide the reduction's two barriers behind.  So take the widest
// block whose entire grid is still resident in a single wave, and spend VPT only
// to get under that cap.  Measured on the fused [1000,4096] shape (6.8 blocks
// per SM, cap 2048 threads/SM => 292):
//
//     512 thr/block, VPT 1 : 5.95 us   (4 blocks/SM resident -> 2 waves)
//     256 thr/block, VPT 2 : 5.24 us   <- one wave, 8 warps/block
//     128 thr/block, VPT 4 : 5.63 us   (one wave, but only 4 warps/block)
//
// and that 0.39 us is worth a whole step of the scored metric (23.55 -> 21.47).
static int pick_mode(int H, int64_t rows) {
  const int ov = env_int("AKO_RMSN_MODE", 0);
  if (ov > 0) return ov;
  const int nv = H / 8;  // 128-bit vectors per row
  if (nv <= 32) return 1;
  if (nv % 32 != 0) return 5;

  const int64_t bps = (rows + sm_count() - 1) / sm_count();  // blocks per SM
  const int cap = env_int("AKO_RMSN_CAP", threads_per_sm()) / (int)(bps < 1 ? 1 : bps);
  int vpt = 1;
  while (vpt < 4 && nv / vpt > cap && (nv % (32 * vpt * 2)) == 0) vpt <<= 1;

  if (nv / 32 == vpt) return 1;  // warp-per-row already gives this VPT
  if (vpt == 1) return 2;
  if (vpt == 2) return 3;
  return 4;
}

#define AKO_BLOCK_MODE_P(KERN, MODE, BLK)                                 \
  if constexpr (((BLK) >= 32) && ((BLK) <= 1024) && ((BLK) % 32 == 0) &&  \
                (H % (8 * (BLK)) == 0)) {                                 \
    if (mode == (MODE)) {                                                 \
      ako_launch(KERN<D, H, BLK, STRIDED>, rows, BLK, s, out, in, w, eps,  \
                 rows, m);                                                \
      return;                                                             \
    }                                                                     \
  }

#define AKO_BLOCK_MODE_F(KERN, MODE, BLK)                                 \
  if constexpr (((BLK) >= 32) && ((BLK) <= 1024) && ((BLK) % 32 == 0) &&  \
                (H % (8 * (BLK)) == 0)) {                                 \
    if (mode == (MODE)) {                                                 \
      ako_launch(KERN<D, H, BLK>, rows, BLK, s, io, res, w, eps, rows);    \
      return;                                                             \
    }                                                                     \
  }

template <typename D, int H, bool STRIDED>
static void spec_plain(D* out, const D* in, const D* w, float eps, int rows,
                       int mode, const RowMap& m, cudaStream_t s) {
  constexpr int NV = H / 8;
  constexpr int LPR = NV < 32 ? NV : 32;
  constexpr int MAXRPB = (512 / LPR) < 16 ? (512 / LPR) : 16;
  AKO_BLOCK_MODE_P(rmsn_block, 2, NV)
  AKO_BLOCK_MODE_P(rmsn_block, 3, NV / 2)
  AKO_BLOCK_MODE_P(rmsn_block, 4, NV / 4)
  warp_plain<D, H, LPR, MAXRPB, STRIDED>(out, in, w, eps, rows, m, s);
}

template <typename D, int H>
static void spec_fused(D* io, D* res, const D* w, float eps, int rows, int mode,
                       cudaStream_t s) {
  constexpr int NV = H / 8;
  constexpr int LPR = NV < 32 ? NV : 32;
  constexpr int MAXRPB = (512 / LPR) < 16 ? (512 / LPR) : 16;
  AKO_BLOCK_MODE_F(rmsn_add_block, 2, NV)
  AKO_BLOCK_MODE_F(rmsn_add_block, 3, NV / 2)
  AKO_BLOCK_MODE_F(rmsn_add_block, 4, NV / 4)
  warp_fused<D, H, LPR, MAXRPB>(io, res, w, eps, rows, s);
}

#define AKO_SPEC_H(HV)                                                    \
  case HV:                                                                \
    if (strided) spec_plain<D, HV, true>(out, in, w, eps, rows, mode, m, s); \
    else spec_plain<D, HV, false>(out, in, w, eps, rows, mode, m, s);      \
    return;

template <typename D>
static void launch_plain(D* out, const D* in, const D* w, float eps, int H,
                         int rows, bool vec_ok, const RowMap& m, bool strided,
                         cudaStream_t s) {
  const int mode = vec_ok ? pick_mode(H, rows) : 5;
  if (mode != 5) {
    switch (H) {
      AKO_SPEC_H(128)
      AKO_SPEC_H(512)
      AKO_SPEC_H(2560)
      AKO_SPEC_H(4096)
      default: break;
    }
  }
  if (vec_ok) {
    if (strided)
      ako_launch(rmsn_generic_vec<D, 256, true>, rows, 256, s, out, in, w, eps, H, rows, m);
    else
      ako_launch(rmsn_generic_vec<D, 256, false>, rows, 256, s, out, in, w, eps, H, rows, m);
  } else if (strided) {
    ako_launch(rmsn_generic_scalar<D, 256, true>, rows, 256, s, out, in, w, eps, H, rows, m);
  } else {
    ako_launch(rmsn_generic_scalar<D, 256, false>, rows, 256, s, out, in, w, eps, H, rows, m);
  }
}

template <typename D>
static void launch_fused(D* io, D* res, const D* w, float eps, int H, int rows,
                         cudaStream_t s) {
  const int mode = pick_mode(H, rows);
  if (mode != 5) {
    switch (H) {
      case 128: spec_fused<D, 128>(io, res, w, eps, rows, mode, s); return;
      case 512: spec_fused<D, 512>(io, res, w, eps, rows, mode, s); return;
      case 2560: spec_fused<D, 2560>(io, res, w, eps, rows, mode, s); return;
      case 4096: spec_fused<D, 4096>(io, res, w, eps, rows, mode, s); return;
      default: break;
    }
  }
  if (H % 8 == 0) {
    ako_launch(rmsn_add_generic_vec<D, 256>, rows, 256, s, io, res, w, eps, H, rows);
  } else {
    ako_launch(rmsn_add_generic_scalar<D, 256>, rows, 256, s, io, res, w, eps, H, rows);
  }
}

static bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// Collapse the leading dimensions of `t` into at most three (extent, stride)
// groups enumerating rows in row-major order, so the kernel can address any row
// without a preceding materializing copy.  Extents of 1 are dropped and
// jointly-dense neighbours are merged, which means a merely *reshaped*
// contiguous tensor still comes out as one group with stride H (i.e. the
// STRIDED == false fast path).  Returns false when the layout needs a real copy.
static bool build_rowmap(const at::Tensor& t, int H, int rows, RowMap& m) {
  const int nd = (int)t.dim();
  if (nd < 1 || t.size(nd - 1) != H || t.stride(nd - 1) != 1) return false;

  int64_t sz[16], st[16];
  int k = 0;
  for (int i = 0; i < nd - 1; ++i) {
    if (t.size(i) == 1) continue;  // an extent of 1 never contributes an offset
    if (k == 16) return false;
    sz[k] = t.size(i);
    st[k] = t.stride(i);
    ++k;
  }
  if (k == 0) {  // a single row
    m.n = 1;
    m.ext[0] = 1;
    m.es[0] = H;
    m.vs[0] = H / 8;
    return rows == 1;
  }

  // Merge inner -> outer while the pair is jointly dense.
  int64_t gsz[16], gst[16];
  int g = 0;
  int64_t cur_sz = sz[k - 1], cur_st = st[k - 1];
  for (int i = k - 2; i >= 0; --i) {
    if (st[i] == cur_sz * cur_st) {
      cur_sz *= sz[i];
    } else {
      gsz[g] = cur_sz;
      gst[g] = cur_st;
      ++g;
      cur_sz = sz[i];
      cur_st = st[i];
    }
  }
  gsz[g] = cur_sz;
  gst[g] = cur_st;
  ++g;
  if (g > 3) return false;

  int64_t prod = 1;
  for (int i = 0; i < g; ++i) {  // reverse to outermost -> innermost
    const int j = g - 1 - i;
    m.ext[i] = (int)gsz[j];
    m.es[i] = gst[j];
    m.vs[i] = gst[j] / 8;
    prod *= gsz[j];
  }
  m.n = g;
  return prod == rows;
}

// Decide how the input is addressed.  `in` is replaced by a contiguous copy only
// when the layout genuinely cannot be walked (a non-unit innermost stride, or
// more than three collapsed groups).
static bool plan_input(at::Tensor& in, int H, int rows, RowMap& m, bool& strided,
                       bool& vec_ok) {
  strided = false;
  m.n = 1;
  m.ext[0] = rows;
  m.es[0] = H;
  m.vs[0] = H / 8;

  if (in.is_contiguous()) {
    vec_ok = (H % 8 == 0) && aligned16(in.data_ptr());
    return true;
  }
  if (build_rowmap(in, H, rows, m)) {
    bool vec = (H % 8 == 0) && aligned16(in.data_ptr());
    for (int i = 0; i < m.n; ++i) vec = vec && (m.es[i] % 8 == 0);
    // One group with stride H addresses exactly like a contiguous tensor.
    strided = !(m.n == 1 && m.es[0] == H);
    vec_ok = vec;
    return true;
  }
  in = in.contiguous();
  m.n = 1;
  m.ext[0] = rows;
  m.es[0] = H;
  m.vs[0] = H / 8;
  vec_ok = (H % 8 == 0) && aligned16(in.data_ptr());
  return true;
}

}  // namespace ako

void ako_rms_norm(at::Tensor out, at::Tensor input, at::Tensor weight, double eps) {
  TORCH_CHECK(out.is_contiguous() && weight.is_contiguous());
  TORCH_CHECK(input.scalar_type() == out.scalar_type());
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  const int H = (int)input.size(-1);
  TORCH_CHECK(weight.numel() == H);
  const int64_t rows64 = input.numel() / H;
  TORCH_CHECK(rows64 <= 0x7fffffffLL);
  const int rows = (int)rows64;
  if (rows == 0) return;

  const at::cuda::CUDAGuard guard(input.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();

  at::Tensor in = input;
  ako::RowMap map;
  bool strided = false, vec_ok = false;
  ako::plan_input(in, H, rows, map, strided, vec_ok);
  vec_ok = vec_ok && ako::aligned16(weight.data_ptr());

  if (in.scalar_type() == at::kBFloat16) {
    ako::launch_plain<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), (float)eps, H,
        rows, vec_ok, map, strided, s);
  } else if (in.scalar_type() == at::kHalf) {
    ako::launch_plain<__half>(reinterpret_cast<__half*>(out.data_ptr()),
                              reinterpret_cast<const __half*>(in.data_ptr()),
                              reinterpret_cast<const __half*>(weight.data_ptr()),
                              (float)eps, H, rows, vec_ok, map, strided, s);
  } else {
    TORCH_CHECK(false, "ako_rms_norm: unsupported dtype");
  }
}

void ako_fused_add_rms_norm(at::Tensor input, at::Tensor residual,
                            at::Tensor weight, double eps) {
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
              weight.is_contiguous());
  TORCH_CHECK(input.scalar_type() == residual.scalar_type());
  TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  const int H = (int)input.size(-1);
  TORCH_CHECK(weight.numel() == H);
  TORCH_CHECK(input.numel() == residual.numel());
  const int64_t rows64 = input.numel() / H;
  TORCH_CHECK(rows64 <= 0x7fffffffLL);
  const int rows = (int)rows64;
  if (rows == 0) return;

  const at::cuda::CUDAGuard guard(input.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::kBFloat16) {
    ako::launch_fused<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), (float)eps, H,
        rows, s);
  } else if (input.scalar_type() == at::kHalf) {
    ako::launch_fused<__half>(reinterpret_cast<__half*>(input.data_ptr()),
                              reinterpret_cast<__half*>(residual.data_ptr()),
                              reinterpret_cast<const __half*>(weight.data_ptr()),
                              (float)eps, H, rows, s);
  } else {
    TORCH_CHECK(false, "ako_fused_add_rms_norm: unsupported dtype");
  }
}

bool ako_supported(at::Tensor input, at::Tensor weight) {
  if (!input.is_cuda()) return false;
  if (input.scalar_type() != at::kBFloat16 && input.scalar_type() != at::kHalf)
    return false;
  const int H = (int)input.size(-1);
  if (H <= 0) return false;
  if (weight.numel() != H) return false;
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &ako_rms_norm, "RMSNorm (shape-specialized)");
  m.def("fused_add_rms_norm", &ako_fused_add_rms_norm,
        "Fused add + RMSNorm (shape-specialized)");
  m.def("supported", &ako_supported, "fast-path applicability");
}
