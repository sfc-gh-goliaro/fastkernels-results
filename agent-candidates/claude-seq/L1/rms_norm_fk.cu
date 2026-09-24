// Fast RMSNorm / fused-add-RMSNorm for the fastkernels L1 `rms_norm` op.
//
// Design notes
// ------------
// * Single pass over the input: a row is kept in registers between the
//   sum-of-squares reduction and the scaled store, so HBM sees exactly one read
//   and one write per element.  (The vLLM reference re-reads the row for the
//   normalize pass, and in the fused-add case re-reads the residual it has just
//   written.)
// * Strided inputs are consumed in place.  The captured q/k-norm shapes are
//   ``[B, H, D]`` slices of a fused QKV buffer -- stride ``[3*H*D, D, 1]``, so
//   not contiguous.  The reference stages them through ``.contiguous()``, which
//   costs an extra kernel launch plus a full read+write.  Here the leading dims
//   collapse to at most two strided levels and the outer one becomes ``grid.y``,
//   so a row base address is a multiply-add with no integer division.
// * Two row geometries, both reducing with ``__shfl_xor_sync``:
//     - `rms_warp`  : TPR threads (a power-of-two sub-warp, or a whole warp) per
//                     row, RPT row-groups per loop iteration so RPT*VPT 16 B
//                     loads are in flight per thread.  No shared memory, no
//                     barrier.  Used when hidden/8 <= 128.
//     - `rms_block` : one NW-warp block per row; warp shuffles, then a tiny
//                     shared staging array that every thread reads
//                     broadcast-style, so one barrier per row.
//   Rows stay in registers as the native 16-bit type rather than fp32, which
//   halves register pressure; the bf16<->f32 conversions are free next to the
//   memory traffic.
// * `rms_generic` covers whatever the vectorized paths reject: hidden not a
//   multiple of 8 elements, unaligned bases, hidden > 8192, fp32 input.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <type_traits>
#include <torch/extension.h>

namespace fkrms {

// ---------------------------------------------------------------------------
// Scalar / packed-pair helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ float from_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

template <typename T>
struct Pk;
template <>
struct Pk<__nv_bfloat16> {
  using p2 = __nv_bfloat162;
  static __device__ __forceinline__ float2 tof(p2 v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ p2 fromf(float a, float b) {
    return __floats2bfloat162_rn(a, b);
  }
};
template <>
struct Pk<__half> {
  using p2 = __half2;
  static __device__ __forceinline__ float2 tof(p2 v) { return __half22float2(v); }
  static __device__ __forceinline__ p2 fromf(float a, float b) {
    return __floats2half2_rn(a, b);
  }
};

// 16-byte row chunk (8 bf16/fp16 lanes).  Deliberately a POD of uint32_t: a
// struct whose members are __nv_bfloat162 gets copied member-wise and ptxas then
// emits four 32-bit accesses per chunk instead of one 128-bit one (measured: 4x
// the global load/store sectors, ~1.6x the kernel time).
template <typename T>
struct alignas(16) V8 {
  uint32_t w[4];
};

template <typename T>
__device__ __forceinline__ float2 unpack2(uint32_t v) {
  typename Pk<T>::p2 p;
  __builtin_memcpy(&p, &v, sizeof(p));
  return Pk<T>::tof(p);
}

template <typename T>
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  const typename Pk<T>::p2 p = Pk<T>::fromf(a, b);
  uint32_t v;
  __builtin_memcpy(&v, &p, sizeof(v));
  return v;
}

// Sum of squares of one 16-byte chunk, into four independent accumulators.
template <typename T>
__device__ __forceinline__ void accum4(const V8<T>& v, float& a0, float& a1,
                                       float& a2, float& a3) {
  const float2 f0 = unpack2<T>(v.w[0]);
  const float2 f1 = unpack2<T>(v.w[1]);
  const float2 f2 = unpack2<T>(v.w[2]);
  const float2 f3 = unpack2<T>(v.w[3]);
  a0 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, a0));
  a1 = fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, a1));
  a2 = fmaf(f2.x, f2.x, fmaf(f2.y, f2.y, a2));
  a3 = fmaf(f3.x, f3.x, fmaf(f3.y, f3.y, a3));
}

// dst = src * scale * weight (weight optional).
template <typename T>
__device__ __forceinline__ V8<T> scale_vec(const V8<T>& src, float s,
                                           const V8<T>& wv, bool has_w) {
  V8<T> o;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = unpack2<T>(src.w[j]);
    float x = f.x * s, y = f.y * s;
    if (has_w) {
      const float2 wf = unpack2<T>(wv.w[j]);
      x *= wf.x;
      y *= wf.y;
    }
    o.w[j] = pack2<T>(x, y);
  }
  return o;
}

// ---------------------------------------------------------------------------
// hidden/8 <= 128: TPR threads per row (TPR * VPT == hidden / 8 exactly),
// RPT row-groups retired per loop iteration.
// ---------------------------------------------------------------------------
template <typename T, int TPR, int VPT, int BLOCK, int RPT, int MINB, bool FUSED>
__global__ __launch_bounds__(BLOCK, MINB) void rms_warp(
    T* __restrict__ out,        // FUSED: unused, the input buffer is the output
    T* __restrict__ inp,        // [grid.y][nrows][hidden], strided
    T* __restrict__ res,        // FUSED only, row-contiguous
    const T* __restrict__ wgt,  // [hidden], or null for no affine scale
    float eps, float inv_h, int nrows, int n1, int64_t s1, int64_t s2,
    int64_t H) {
  using V = V8<T>;
  constexpr int RPB = BLOCK / TPR;  // rows resident in the block at once
  constexpr int RPI = RPB * RPT;    // rows retired per loop iteration
  const int lane = threadIdx.x & (TPR - 1);
  const int rib = threadIdx.x / TPR;

  V wv[VPT];
  const bool has_w = wgt != nullptr;
  if (has_w) {
    const V* __restrict__ wp = reinterpret_cast<const V*>(wgt);
#pragma unroll
    for (int k = 0; k < VPT; ++k) wv[k] = wp[lane + k * TPR];
  }

  const int64_t y_in = (int64_t)blockIdx.y * s2;
  const int64_t y_out = (int64_t)blockIdx.y * n1;

  for (int base = blockIdx.x * RPI; base < nrows; base += gridDim.x * RPI) {
    int row[RPT];
    bool act[RPT];
    // The non-fused input is read-only; a const-restrict pointer lets ptxas
    // emit non-coherent (ld.global.nc) loads.
    using LP = std::conditional_t<FUSED, V*, const V*>;
    LP __restrict__ ip[RPT];
    V zv[RPT][VPT];
#pragma unroll
    for (int t = 0; t < RPT; ++t) {
      row[t] = base + t * RPB + rib;
      act[t] = row[t] < nrows;
      ip[t] = reinterpret_cast<LP>(inp + y_in + (int64_t)row[t] * s1);
    }
    // All RPT * VPT loads are issued before any of the data is touched.
    V xv[RPT][VPT];
#pragma unroll
    for (int t = 0; t < RPT; ++t) {
      if (act[t]) {
#pragma unroll
        for (int k = 0; k < VPT; ++k) xv[t][k] = ip[t][lane + k * TPR];
      } else {
#pragma unroll
        for (int k = 0; k < VPT; ++k)
#pragma unroll
          for (int j = 0; j < 4; ++j) xv[t][k].w[j] = 0u;
      }
    }

    if constexpr (FUSED) {
      V* rp[RPT];
      V rvv[RPT][VPT];
#pragma unroll
      for (int t = 0; t < RPT; ++t)
        rp[t] = reinterpret_cast<V*>(res + (y_out + row[t]) * H);
#pragma unroll
      for (int t = 0; t < RPT; ++t) {
        if (act[t]) {
#pragma unroll
          for (int k = 0; k < VPT; ++k) rvv[t][k] = rp[t][lane + k * TPR];
        } else {
#pragma unroll
          for (int k = 0; k < VPT; ++k)
#pragma unroll
            for (int j = 0; j < 4; ++j) rvv[t][k].w[j] = 0u;
        }
      }
#pragma unroll
      for (int t = 0; t < RPT; ++t)
#pragma unroll
        for (int k = 0; k < VPT; ++k)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 xf = unpack2<T>(xv[t][k].w[j]);
            const float2 rf = unpack2<T>(rvv[t][k].w[j]);
            zv[t][k].w[j] = pack2<T>(xf.x + rf.x, xf.y + rf.y);
          }
#pragma unroll
      for (int t = 0; t < RPT; ++t)
        if (act[t]) {
#pragma unroll
          for (int k = 0; k < VPT; ++k) rp[t][lane + k * TPR] = zv[t][k];
        }
    } else {
#pragma unroll
      for (int t = 0; t < RPT; ++t)
#pragma unroll
        for (int k = 0; k < VPT; ++k) zv[t][k] = xv[t][k];
    }

    float s[RPT];
#pragma unroll
    for (int t = 0; t < RPT; ++t) {
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
      for (int k = 0; k < VPT; ++k) accum4<T>(zv[t][k], a0, a1, a2, a3);
      float acc = (a0 + a1) + (a2 + a3);
#pragma unroll
      for (int off = TPR / 2; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off, TPR);
      s[t] = rsqrtf(acc * inv_h + eps);
    }

#pragma unroll
    for (int t = 0; t < RPT; ++t) {
      if (!act[t]) continue;
      V* __restrict__ op;
      if constexpr (FUSED)
        op = const_cast<V*>((const V*)ip[t]);
      else
        op = reinterpret_cast<V*>(out + (y_out + row[t]) * H);
#pragma unroll
      for (int k = 0; k < VPT; ++k)
        op[lane + k * TPR] = scale_vec<T>(zv[t][k], s[t], wv[k], has_w);
    }
  }
}

// ---------------------------------------------------------------------------
// One NW-warp block per row, VPT chunks per thread, bounds-checked so any
// hidden/8 <= NW * 32 * VPT works.
// ---------------------------------------------------------------------------
template <typename T, int NW, int VPT, int MINB, bool FUSED>
__global__ __launch_bounds__(NW * 32, MINB) void rms_block(
    T* __restrict__ out, T* __restrict__ inp, T* __restrict__ res,
    const T* __restrict__ wgt, float eps, float inv_h, int nrows, int n1,
    int64_t s1, int64_t s2, int64_t H) {
  using V = V8<T>;
  constexpr int BLOCK = NW * 32;
  const int nvec = (int)(H >> 3);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // Double buffered so one barrier per row suffices even when the grid is
  // smaller than the row count.
  __shared__ float sm[2][NW];

  V wv[VPT];
  const bool has_w = wgt != nullptr;
  if (has_w) {
    const V* __restrict__ wp = reinterpret_cast<const V*>(wgt);
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int idx = tid + k * BLOCK;
      if (idx < nvec) wv[k] = wp[idx];
    }
  }

  const int64_t y_in = (int64_t)blockIdx.y * s2;
  const int64_t y_out = (int64_t)blockIdx.y * n1;
  int phase = 0;

  for (int row = blockIdx.x; row < nrows; row += gridDim.x, phase ^= 1) {
    using LP = std::conditional_t<FUSED, V*, const V*>;
    LP __restrict__ ip = reinterpret_cast<LP>(inp + y_in + (int64_t)row * s1);
    V zv[VPT];
    {
      V xv[VPT];
#pragma unroll
      for (int k = 0; k < VPT; ++k) {
        const int idx = tid + k * BLOCK;
        if (idx < nvec) xv[k] = ip[idx];
      }
      if constexpr (FUSED) {
        V* __restrict__ rp = reinterpret_cast<V*>(res + (y_out + row) * H);
        V rvv[VPT];
#pragma unroll
        for (int k = 0; k < VPT; ++k) {
          const int idx = tid + k * BLOCK;
          if (idx < nvec) rvv[k] = rp[idx];
        }
#pragma unroll
        for (int k = 0; k < VPT; ++k)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 xf = unpack2<T>(xv[k].w[j]);
            const float2 rf = unpack2<T>(rvv[k].w[j]);
            zv[k].w[j] = pack2<T>(xf.x + rf.x, xf.y + rf.y);
          }
#pragma unroll
        for (int k = 0; k < VPT; ++k) {
          const int idx = tid + k * BLOCK;
          if (idx < nvec) rp[idx] = zv[k];
        }
      } else {
#pragma unroll
        for (int k = 0; k < VPT; ++k) zv[k] = xv[k];
      }
    }
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int idx = tid + k * BLOCK;
      if (idx < nvec) accum4<T>(zv[k], a0, a1, a2, a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) sm[phase][warp] = acc;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) tot += sm[phase][i];
    const float s = rsqrtf(tot * inv_h + eps);

    V* __restrict__ op;
    if constexpr (FUSED)
      op = const_cast<V*>((const V*)ip);
    else
      op = reinterpret_cast<V*>(out + (y_out + row) * H);
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int idx = tid + k * BLOCK;
      if (idx < nvec) op[idx] = scale_vec<T>(zv[k], s, wv[k], has_w);
    }
  }
}

// ---------------------------------------------------------------------------
// Fallback: element-wise, two passes, any hidden size / alignment / dtype.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK, bool FUSED>
__global__ __launch_bounds__(BLOCK) void rms_generic(
    T* __restrict__ out, T* __restrict__ inp, T* __restrict__ res,
    const T* __restrict__ wgt, float eps, float inv_h, int nrows, int n1,
    int64_t s1, int64_t s2, int64_t H) {
  constexpr int NW = BLOCK / 32;
  __shared__ float sm[NW];
  const int tid = threadIdx.x;
  const bool has_w = wgt != nullptr;
  const int64_t y_in = (int64_t)blockIdx.y * s2;
  const int64_t y_out = (int64_t)blockIdx.y * n1;

  for (int row = blockIdx.x; row < nrows; row += gridDim.x) {
    T* __restrict__ ip = inp + y_in + (int64_t)row * s1;
    T* __restrict__ rp = nullptr;
    if constexpr (FUSED) rp = res + (y_out + row) * H;
    float acc = 0.f;
    for (int64_t i = tid; i < H; i += BLOCK) {
      float x = to_f(ip[i]);
      if constexpr (FUSED) {
        const T z = from_f<T>(x + to_f(rp[i]));
        rp[i] = z;
        x = to_f(z);
      }
      acc = fmaf(x, x, acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if ((tid & 31) == 0) sm[tid >> 5] = acc;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) tot += sm[i];
    const float s = rsqrtf(tot * inv_h + eps);
    T* __restrict__ op;
    if constexpr (FUSED)
      op = ip;
    else
      op = out + (y_out + row) * H;
    for (int64_t i = tid; i < H; i += BLOCK) {
      float x = to_f(FUSED ? rp[i] : ip[i]) * s;
      if (has_w) x *= to_f(wgt[i]);
      op[i] = from_f<T>(x);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
struct Geom {
  int64_t nrows = 1;  // rows along grid.x
  int64_t outer = 1;  // rows along grid.y
  int64_t s1 = 0;     // element stride between grid.x rows
  int64_t s2 = 0;     // element stride between grid.y rows
  bool ok = true;     // false -> caller must make the tensor contiguous
};

// Collapse the leading dims of `x` into at most two strided levels.
static Geom row_geometry(const at::Tensor& x) {
  Geom g;
  const int nd = x.dim();
  if (x.is_contiguous()) {  // cached flag; skips the stride walk below
    g.s1 = x.size(nd - 1);
    g.nrows = g.s1 ? x.numel() / g.s1 : 0;
    return g;
  }
  if (x.stride(nd - 1) != 1) {
    g.ok = false;
    return g;
  }
  int64_t n[2], s[2];
  int nl = 0;
  for (int d = nd - 2; d >= 0; --d) {
    const int64_t sz = x.size(d);
    if (sz == 1) continue;
    const int64_t st = x.stride(d);
    if (nl > 0 && st == s[nl - 1] * n[nl - 1]) {
      n[nl - 1] *= sz;
      continue;
    }
    if (nl == 2) {
      g.ok = false;
      return g;
    }
    n[nl] = sz;
    s[nl] = st;
    ++nl;
  }
  g.s1 = (nl > 0) ? s[0] : x.size(nd - 1);
  g.nrows = (nl > 0) ? n[0] : 1;
  if (nl == 2) {
    g.outer = n[1];
    g.s2 = s[1];
  }
  return g;
}

static inline int sm_count() {
  static const int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

// The fused kernel keeps three chunk arrays per row-group (input, residual and
// their sum) where the plain one keeps a single aliased array, so it needs a
// shallower unroll to stay inside the register budget MINB implies.
#define FK_RPT(RPTV) (FUSED && (RPTV) > 2 ? (RPTV) / 2 : (RPTV))
#define FK_RPI(TPRV, BLK, RPTV) (((BLK) / (TPRV)) * FK_RPT(RPTV))

#define FK_WARPG(TPRV, VPTV, BLK, RPTV, MINB, GRID)                           \
  do {                                                                        \
    rms_warp<T, TPRV, VPTV, BLK, FK_RPT(RPTV), MINB, FUSED>                   \
        <<<dim3((unsigned)(GRID), (unsigned)g.outer), BLK, 0, stream>>>(       \
            out, inp, res, wgt, eps, inv_h, (int)nrows, (int)nrows, g.s1,      \
            g.s2, H);                                                         \
    return;                                                                   \
  } while (0)

#define FK_WARP(TPRV, VPTV, BLK, RPTV, MINB)                                  \
  FK_WARPG(TPRV, VPTV, BLK, RPTV, MINB,                                       \
           (nrows + FK_RPI(TPRV, BLK, RPTV) - 1) / FK_RPI(TPRV, BLK, RPTV))

// Deep row-group unroll only when grid.x really has that many rows to chew
// (otherwise most row slots in the unroll sit idle) *and* the resulting grid
// still fills the machine; otherwise one row-group per iteration.
#define FK_WARP2(TPRV, VPTV, BLK, RPTV, MINB)                                 \
  do {                                                                        \
    constexpr int64_t RPI_ = FK_RPI(TPRV, BLK, RPTV);                         \
    if (nrows >= 4 * RPI_ && (nrows / RPI_) * g.outer >= 4 * sm_count())       \
      FK_WARP(TPRV, VPTV, BLK, RPTV, MINB);                                   \
    FK_WARP(TPRV, VPTV, BLK, 1, MINB);                                        \
  } while (0)

#define FK_BLOCKG(NWV, VPTV, MINB, GRID)                                      \
  do {                                                                        \
    rms_block<T, NWV, VPTV, MINB, FUSED>                                      \
        <<<dim3((unsigned)(GRID), (unsigned)g.outer), (NWV) * 32, 0, stream>>>( \
            out, inp, res, wgt, eps, inv_h, (int)nrows, (int)nrows, g.s1,      \
            g.s2, H);                                                         \
    return;                                                                   \
  } while (0)

// Capping grid.x lets each block chew several rows with the affine weight
// hoisted in registers, instead of re-reading it once per row.
#define FK_BLOCKW(NWV, VPTV, MINB, WAVES)                                     \
  FK_BLOCKG(NWV, VPTV, MINB,                                                  \
            std::min<int64_t>(nrows, (int64_t)(WAVES) * sm_count()))

template <typename T, bool FUSED>
static void launch(T* out, T* inp, T* res, const T* wgt, const Geom& g,
                   int64_t H, float eps, cudaStream_t stream) {
  const int64_t nrows = g.nrows;
  const float inv_h = 1.0f / (float)H;
  const int64_t nvec = H >> 3;

  const bool vec_ok =
      (H % 8 == 0) && (g.s1 % 8 == 0) && (g.s2 % 8 == 0) &&
      ((reinterpret_cast<uintptr_t>(inp) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(out) & 15) == 0) &&
      (!FUSED || (reinterpret_cast<uintptr_t>(res) & 15) == 0) &&
      (wgt == nullptr || (reinterpret_cast<uintptr_t>(wgt) & 15) == 0) &&
      nvec <= 1024;

  // Geometries are tuned on B200.  Two consistent findings drive them:
  //   * a row per sub-warp/warp with a deep row-group unroll wins while the row
  //     still fits in one warp's registers (more independent 16 B loads per
  //     thread); past that, a block per row with few threads and several chunks
  //     each beats a wide block with one chunk each;
  //   * the last argument is the __launch_bounds__ min-blocks-per-SM hint.  Left
  //     unconstrained, ptxas spends registers freely and occupancy collapses to
  //     ~20%; capping it around 112-128 registers per thread is worth 1.1-1.5x
  //     on the bandwidth-bound shapes and never costs anything.
  if (vec_ok) {
    switch (nvec) {
      case 1: FK_WARP(1, 1, 128, 1, 4);
      case 2: FK_WARP(2, 1, 128, 1, 4);
      case 4: FK_WARP(4, 1, 128, 1, 4);
      case 8: FK_WARP2(8, 1, 128, 6, 4);
      case 16: FK_WARP2(8, 2, 128, 6, 4);   // hidden 128 (q/k norm)
      case 32: FK_WARP2(32, 1, 128, 6, 4);
      case 64: FK_WARP2(32, 2, 128, 6, 4);  // hidden 512
      case 96: FK_WARP2(32, 3, 128, 4, 2);
      case 128: FK_WARP2(32, 4, 128, 4, 2);
      case 320: FK_BLOCKW(3, 4, 6, 8);      // hidden 2560
      case 512: FK_BLOCKW(4, 4, 4, 8);      // hidden 4096
      default: break;
    }
    if (nvec <= 128) FK_BLOCKW(2, 2, 8, 8);
    if (nvec <= 256) FK_BLOCKW(2, 4, 8, 8);
    if (nvec <= 384) FK_BLOCKW(3, 4, 6, 8);
    if (nvec <= 512) FK_BLOCKW(4, 4, 4, 8);
    if (nvec <= 768) FK_BLOCKW(6, 4, 3, 8);
    FK_BLOCKW(8, 4, 2, 8);
  }
  constexpr int GB = 256;
  rms_generic<T, GB, FUSED><<<dim3((unsigned)nrows, (unsigned)g.outer), GB, 0,
                              stream>>>(out, inp, res, wgt, eps, inv_h,
                                        (int)nrows, (int)nrows, g.s1, g.s2, H);
}

static inline void check_weight(const at::Tensor& w, const at::Tensor& x,
                                int64_t H) {
  TORCH_CHECK(w.scalar_type() == x.scalar_type(), "rms_norm: weight dtype");
  TORCH_CHECK(w.is_contiguous() && w.numel() == H, "rms_norm: weight shape");
}

#define FK_DISPATCH(FUSEDV, OUTP, RESP)                                      \
  switch (x.scalar_type()) {                                                 \
    case at::kBFloat16:                                                      \
      launch<__nv_bfloat16, FUSEDV>(                                         \
          (__nv_bfloat16*)(OUTP), (__nv_bfloat16*)x.data_ptr(),              \
          (__nv_bfloat16*)(RESP),                                            \
          w.defined() ? (const __nv_bfloat16*)w.data_ptr() : nullptr, g, H,  \
          e, stream);                                                        \
      break;                                                                 \
    case at::kHalf:                                                          \
      launch<__half, FUSEDV>(                                                \
          (__half*)(OUTP), (__half*)x.data_ptr(), (__half*)(RESP),           \
          w.defined() ? (const __half*)w.data_ptr() : nullptr, g, H, e,      \
          stream);                                                           \
      break;                                                                 \
    case at::kFloat:                                                         \
      rms_generic<float, 256, FUSEDV>                                        \
          <<<dim3((unsigned)g.nrows, (unsigned)g.outer), 256, 0, stream>>>(   \
              (float*)(OUTP), (float*)x.data_ptr(), (float*)(RESP),           \
              w.defined() ? (const float*)w.data_ptr() : nullptr, e,          \
              1.0f / (float)H, (int)g.nrows, (int)g.nrows, g.s1, g.s2, H);    \
      break;                                                                 \
    default:                                                                 \
      TORCH_CHECK(false, "rms_norm: unsupported dtype ", x.scalar_type());    \
  }

at::Tensor rmsnorm(at::Tensor x, at::Tensor weight, double eps) {
  const int64_t H = x.size(-1);
  Geom g = row_geometry(x);
  if (!g.ok) {
    x = x.contiguous();
    g = row_geometry(x);
  }
  // at::detail::empty_cuda bypasses the operator dispatcher; for the
  // launch-bound shapes the allocation is a measurable slice of the call.
  at::Tensor out(at::detail::empty_cuda(x.sizes(), x.scalar_type(), x.device(),
                                        std::nullopt));
  if (x.numel() == 0) return out;
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float e = (float)eps;
  const at::Tensor& w = weight;
  if (w.defined()) check_weight(w, x, H);
  FK_DISPATCH(false, out.data_ptr(), nullptr)
  return out;
}

std::tuple<at::Tensor, at::Tensor> fused_add_rmsnorm(
    at::Tensor x, at::Tensor residual, at::Tensor weight, double eps) {
  const int64_t H = x.size(-1);
  TORCH_CHECK(x.scalar_type() == residual.scalar_type(),
              "fused_add_rms_norm: dtype mismatch");
  TORCH_CHECK(x.sizes() == residual.sizes(),
              "fused_add_rms_norm: shape mismatch");
  Geom g = row_geometry(x);
  if (!g.ok) {
    x = x.contiguous();
    g = row_geometry(x);
  }
  if (!residual.is_contiguous()) residual = residual.contiguous();
  if (x.numel() == 0) return std::make_tuple(x, residual);
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float e = (float)eps;
  const at::Tensor& w = weight;
  if (w.defined()) check_weight(w, x, H);
  FK_DISPATCH(true, nullptr, residual.data_ptr())
  return std::make_tuple(x, residual);
}

}  // namespace fkrms

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &fkrms::rmsnorm, "RMSNorm (fastkernels candidate)");
  m.def("fused_add_rmsnorm", &fkrms::fused_add_rmsnorm,
        "Fused add + RMSNorm (fastkernels candidate)");
}
