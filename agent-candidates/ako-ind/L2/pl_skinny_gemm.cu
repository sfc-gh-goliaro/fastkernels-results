// Skinny-GEMM kernels for the L2 parallel-linear family.
//
// cuBLAS leaves 2-3x on the table for tiny-batch linear shapes: its 128x128
// tiling produces only a handful of CTAs on a 148-SM B200, so a decode-width
// GEMM streams the weight matrix through a fraction of the machine (and for
// narrow N it falls back to split-K plus a second reduction kernel).
//
// This kernel parallelises over *weight rows* instead.  A "group" of G threads
// (G in {32,64,128,256}, warp-aligned) owns one output column n, streams
// W[n, :] with 128-bit loads and accumulates M dot products in fp32.  Two
// details matter far more than the arithmetic:
//
//   * the K loop is unrolled 4x with the four 128-bit weight loads issued
//     before any of them is consumed.  Without that, each thread has exactly
//     one load in flight and the whole kernel runs at HBM latency rather than
//     HBM bandwidth (measured 4x difference at K=2048).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#define PL_BLOCK 256
#define PL_WARPS (PL_BLOCK / 32)

namespace {

template <typename T>
union Vec8 {
  uint4 raw;
  T v[8];
};

__device__ __forceinline__ float pl_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float pl_f(__half x) { return __half2float(x); }
__device__ __forceinline__ void pl_store(__nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}
__device__ __forceinline__ void pl_store(__half* p, float v) { *p = __float2half(v); }

// ---------------------------------------------------------------------------
// out[m, n] = sum_k x[m, k] * w[n, k] (+ bias[n]),  M <= MMAX <= 8
//
// Block covers R * rpw consecutive output columns, R = PL_BLOCK / G.
// ---------------------------------------------------------------------------
template <typename T, int MMAX, int G>
__global__ __launch_bounds__(PL_BLOCK) void pl_gemv_kernel(
    const T* __restrict__ X, const T* __restrict__ W, const T* __restrict__ Bias,
    T* __restrict__ Out, int M, int N, int K, int ldx, int ldw, int rpw) {
  constexpr int R = PL_BLOCK / G;
  constexpr int WPG = G / 32;  // warps per group

  __shared__ float red[PL_WARPS * MMAX];

  const int g = threadIdx.x / G;
  const int lane = threadIdx.x - g * G;
  const int warp = threadIdx.x >> 5;
  const int step = G * 8;
  const int n_base = blockIdx.x * (R * rpw) + g;

  for (int r = 0; r < rpw; ++r) {
    const int n = n_base + r * R;
    float acc[MMAX];
#pragma unroll
    for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;

    if (n < N) {
      const T* wrow = W + (size_t)n * ldw;
      int k = lane * 8;

#define PL_FMA(WV, KK)                                                       \
  {                                                                          \
    _Pragma("unroll") for (int m = 0; m < MMAX; ++m) {                        \
      if (m < M) {                                                           \
        Vec8<T> xv;                                                          \
        xv.raw = *reinterpret_cast<const uint4*>(X + (size_t)m * ldx + (KK)); \
        float s = acc[m];                                                    \
        _Pragma("unroll") for (int i = 0; i < 8; ++i)                         \
            s = fmaf(pl_f((WV).v[i]), pl_f(xv.v[i]), s);                      \
        acc[m] = s;                                                          \
      }                                                                      \
    }                                                                        \
  }

      // 4 independent weight loads in flight per thread.
      for (; k + 3 * step < K; k += 4 * step) {
        Vec8<T> a0, a1, a2, a3;
        a0.raw = *reinterpret_cast<const uint4*>(wrow + k);
        a1.raw = *reinterpret_cast<const uint4*>(wrow + k + step);
        a2.raw = *reinterpret_cast<const uint4*>(wrow + k + 2 * step);
        a3.raw = *reinterpret_cast<const uint4*>(wrow + k + 3 * step);
        PL_FMA(a0, k)
        PL_FMA(a1, k + step)
        PL_FMA(a2, k + 2 * step)
        PL_FMA(a3, k + 3 * step)
      }
      for (; k < K; k += step) {
        Vec8<T> a0;
        a0.raw = *reinterpret_cast<const uint4*>(wrow + k);
        PL_FMA(a0, k)
      }
#undef PL_FMA
    }

#pragma unroll
    for (int m = 0; m < MMAX; ++m) {
      if (m < M) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
      }
    }

    if (WPG == 1) {
      if (lane == 0 && n < N) {
#pragma unroll
        for (int m = 0; m < MMAX; ++m)
          if (m < M) {
            float v = acc[m];
            if (Bias) v += pl_f(Bias[n]);
            pl_store(Out + (size_t)m * N + n, v);
          }
      }
    } else {
      if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (int m = 0; m < MMAX; ++m)
          if (m < M) red[warp * MMAX + m] = acc[m];
      }
      __syncthreads();
      if (lane == 0 && n < N) {
        const int w0 = g * WPG;
#pragma unroll
        for (int m = 0; m < MMAX; ++m)
          if (m < M) {
            float v = 0.f;
#pragma unroll
            for (int i = 0; i < WPG; ++i) v += red[(w0 + i) * MMAX + m];
            if (Bias) v += pl_f(Bias[n]);
            pl_store(Out + (size_t)m * N + n, v);
          }
      }
      __syncthreads();
    }
  }
}

inline int pl_floor_pow2(int v) {
  int p = 1;
  while (p * 2 <= v) p *= 2;
  return p;
}

// Threads-per-column G and columns-per-group rpw.
//   * G large enough that a thread's K slice is at most 4 loads deep (so the
//     4x unroll keeps every thread's loads in flight),
//   * G small enough not to leave lanes with no K to chew on,
//   * grid capped near 512 CTAs -- the CWD dispatches roughly one block per
//     clock, and an empty 2048-CTA launch already costs 2us more than a
//     512-CTA one on B200.
inline void pl_pick(int N, int K, int* G_out, int* rpw_out) {
  int gmin = 32;
  while (gmin < 256 && (long)gmin * 32 < (long)K) gmin *= 2;
  int gmax = 32;
  while (gmax < 256 && (long)(gmax * 2) * (long)N <= 512L * PL_BLOCK) gmax *= 2;
  const int gk = pl_floor_pow2(K >> 3);
  if (gk > 32 && gk < gmax) gmax = gk;
  int G = gmin > gmax ? gmin : gmax;
  if (G > 256) G = 256;
  const int R = PL_BLOCK / G;
  const long blocks = ((long)N + R - 1) / R;
  int rpw = (int)((blocks + 511) / 512);
  if (rpw < 1) rpw = 1;
  *G_out = G;
  *rpw_out = rpw;
}

template <typename T, int MMAX>
void pl_gemv_launch(const T* X, const T* W, const T* Bias, T* Out, int M, int N,
                    int K, int ldx, int ldw, cudaStream_t stream, int Gf, int rf) {
  int G, rpw;
  pl_pick(N, K, &G, &rpw);
  if (Gf > 0) G = Gf;
  if (rf > 0) rpw = rf;
  const int R = PL_BLOCK / G;
  const int blocks = (int)(((long)N + (long)R * rpw - 1) / ((long)R * rpw));
#define PL_CASE(GG)                                                            \
  case GG:                                                                     \
    pl_gemv_kernel<T, MMAX, GG><<<blocks, PL_BLOCK, 0, stream>>>(              \
        X, W, Bias, Out, M, N, K, ldx, ldw, rpw);                              \
    break;
  switch (G) {
    PL_CASE(32)
    PL_CASE(64)
    PL_CASE(128)
    default:
      PL_CASE(256)
  }
#undef PL_CASE
}

}  // namespace

// ---------------------------------------------------------------------------
// out = x @ w.T (+ bias);  x [M, K] row stride ldx, w [N, K] row stride ldw.
// Caller guarantees: M <= 8, K % 8 == 0, 16B-aligned rows.
// ---------------------------------------------------------------------------
void pl_gemv(const at::Tensor& x, const at::Tensor& w,
             const c10::optional<at::Tensor>& bias, at::Tensor& out,
             int64_t Gf, int64_t rf) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)w.size(0);
  const int ldx = (int)x.stride(0);
  const int ldw = (int)w.stride(0);
  const void* bp = bias.has_value() ? bias->data_ptr() : nullptr;

  if (x.scalar_type() == at::kBFloat16) {
    using T = __nv_bfloat16;
    const T* X = (const T*)x.data_ptr();
    const T* W = (const T*)w.data_ptr();
    const T* B = (const T*)bp;
    T* O = (T*)out.data_ptr();
    if (M == 1) pl_gemv_launch<T, 1>(X, W, B, O, M, N, K, ldx, ldw, stream, (int)Gf, (int)rf);
    else pl_gemv_launch<T, 8>(X, W, B, O, M, N, K, ldx, ldw, stream, (int)Gf, (int)rf);
  } else {
    using T = __half;
    const T* X = (const T*)x.data_ptr();
    const T* W = (const T*)w.data_ptr();
    const T* B = (const T*)bp;
    T* O = (T*)out.data_ptr();
    if (M == 1) pl_gemv_launch<T, 1>(X, W, B, O, M, N, K, ldx, ldw, stream, (int)Gf, (int)rf);
    else pl_gemv_launch<T, 8>(X, W, B, O, M, N, K, ldx, ldw, stream, (int)Gf, (int)rf);
  }
}



// ===========================================================================
// Tensor-core skinny GEMM.  CORRECT (bit-exact vs cuBLAS) BUT NOT DISPATCHED --
// it measured 2-3x slower than cuBLAS on every captured shape, e.g.
// [60,2048]x[1024,2048] 31.7us vs 11.3us (best of a ks sweep) and
// [1000,4096]x[128,4096] 42.0us vs 17.5us.  Splitting K across the grid fixes
// the CTA count but not the real limiter: one staged chunk per CTA at a time
// keeps only ~64 KB of loads in flight, and these shapes are latency-bound.
// The missing piece is cp.async multistage pipelining (plus wider CTAs); kept
// here so the next round can add it to working machinery.  See ITERATIONS.md.
//
// Original rationale for the tiling below:
//
// cuBLAS picks 128x128 output tiles, so a shape like [1000, 4096] x [128, 4096]
// gets 8 CTAs and [60, 2048] x [1024, 2048] gets 8 -- on 148 SMs.  Here the
// output tile is 64x64 and, when that still is not enough CTAs, K is split
// across the grid with fp32 partials reduced by a second kernel.  Splitting is
// cheap in this harness: an extra launch costs ~0 as long as both grids stay
// under ~1024 CTAs.
//
// M is ragged in every captured shape (60, 379, 492, 673, 931, 1000), so the
// x tile is staged zero-padded through shared memory; K is required to be a
// multiple of 64 and N a multiple of 16 (true for every captured shape).
// ===========================================================================

#define TC_BM 64
#define TC_BN 64
#define TC_BK 64
#define TC_THREADS 128
#define TC_MT (TC_BM / 16)
#define TC_LD (TC_BK + 8)   // shared-tile row pitch, elements (16B-aligned rows)
#define TC_CLD 20           // fp32 epilogue tile pitch; wmma needs 16B-aligned rows

namespace {

using namespace nvcuda;

// out[m, n] = sum_k x[m, k] * w[n, k]; one CTA owns a 64x64 output tile and, if
// ks > 1, one of ks slices of K (fp32 partials go to Ws, reduced below).
//
// Both operands are staged through shared memory with 128-bit coalesced loads.
// Feeding wmma::load_matrix_sync a *global* col-major B pointer instead makes it
// emit a per-lane gather and costs ~4x (measured 109us vs cuBLAS 11us on
// [60,2048]x[1024,2048]).
template <typename T>
__global__ __launch_bounds__(TC_THREADS) void pl_tc_kernel(
    const T* __restrict__ X, const T* __restrict__ W, const T* __restrict__ Bias,
    T* __restrict__ Out, float* __restrict__ Ws, int M, int N, int K, int ldx,
    int ldw, int ks) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * TC_BM;
  const int nb = blockIdx.x * TC_BN;
  const int n0 = nb + warp * 16;
  const int s = blockIdx.z;

  const int chunks = K / TC_BK;
  const int c_lo = (int)((long)chunks * s / ks);
  const int c_hi = (int)((long)chunks * (s + 1) / ks);

  __shared__ T sa[TC_BM * TC_LD];
  __shared__ T sb[TC_BN * TC_LD];
  __shared__ float sc[(TC_THREADS / 32) * 16 * TC_CLD];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TC_MT];
#pragma unroll
  for (int mt = 0; mt < TC_MT; ++mt) wmma::fill_fragment(acc[mt], 0.f);

  const int mrows = (M - m0 < TC_BM) ? (M - m0) : TC_BM;
  const int nrows = (N - nb < TC_BN) ? (N - nb) : TC_BN;
  const bool have_n = n0 < N;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);

  for (int c = c_lo; c < c_hi; ++c) {
    const int k0 = c * TC_BK;
#pragma unroll
    for (int i = 0; i < (TC_BM * (TC_BK / 8)) / TC_THREADS; ++i) {
      const int t = i * TC_THREADS + threadIdx.x;
      const int r = t >> 3, j = t & 7;
      *reinterpret_cast<uint4*>(&sa[r * TC_LD + j * 8]) =
          (r < mrows) ? *reinterpret_cast<const uint4*>(X + (size_t)(m0 + r) * ldx + k0 + j * 8)
                      : zero;
    }
#pragma unroll
    for (int i = 0; i < (TC_BN * (TC_BK / 8)) / TC_THREADS; ++i) {
      const int t = i * TC_THREADS + threadIdx.x;
      const int r = t >> 3, j = t & 7;
      *reinterpret_cast<uint4*>(&sb[r * TC_LD + j * 8]) =
          (r < nrows) ? *reinterpret_cast<const uint4*>(W + (size_t)(nb + r) * ldw + k0 + j * 8)
                      : zero;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TC_BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b;
      wmma::load_matrix_sync(b, &sb[warp * 16 * TC_LD + kk * 16], TC_LD);
#pragma unroll
      for (int mt = 0; mt < TC_MT; ++mt) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
        wmma::load_matrix_sync(a, &sa[mt * 16 * TC_LD + kk * 16], TC_LD);
        wmma::mma_sync(acc[mt], a, b, acc[mt]);
      }
    }
    __syncthreads();
  }

  if (!have_n) return;
  float* scw = &sc[warp * 16 * TC_CLD];
  const size_t plane = (size_t)M * N;
#pragma unroll 1
  for (int mt = 0; mt < TC_MT; ++mt) {
    const int gm0 = m0 + mt * 16;
    if (gm0 >= M) break;
    wmma::store_matrix_sync(scw, acc[mt], TC_CLD, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 16 * 16; idx += 32) {
      const int r = idx >> 4, cc = idx & 15;
      const int gm = gm0 + r, gn = n0 + cc;
      if (gm < M && gn < N) {
        float v = scw[r * TC_CLD + cc];
        if (ks > 1) {
          Ws[(size_t)s * plane + (size_t)gm * N + gn] = v;
        } else {
          if (Bias) v += pl_f(Bias[gn]);
          pl_store(Out + (size_t)gm * N + gn, v);
        }
      }
    }
    __syncwarp();
  }
}

template <typename T>
__global__ __launch_bounds__(256) void pl_tc_reduce(const float* __restrict__ Ws,
                                                    const T* __restrict__ Bias,
                                                    T* __restrict__ Out, long total,
                                                    int N, int ks) {
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total;
       i += (long)gridDim.x * 256) {
    float v = 0.f;
    for (int s = 0; s < ks; ++s) v += Ws[(size_t)s * total + i];
    if (Bias) v += pl_f(Bias[(int)(i - (i / N) * N)]);
    pl_store(Out + i, v);
  }
}

template <typename T>
void pl_tc_launch(const T* X, const T* W, const T* Bias, T* Out, float* Ws, int M,
                  int N, int K, int ldx, int ldw, int ks, cudaStream_t stream) {
  dim3 grid((N + TC_BN - 1) / TC_BN, (M + TC_BM - 1) / TC_BM, ks);
  pl_tc_kernel<T><<<grid, TC_THREADS, 0, stream>>>(X, W, Bias, Out, Ws, M, N, K,
                                                   ldx, ldw, ks);
  if (ks > 1) {
    const long total = (long)M * N;
    int blocks = (int)((total + 2047) / 2048);
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;
    pl_tc_reduce<T><<<blocks, 256, 0, stream>>>(Ws, Bias, Out, total, N, ks);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// out = x @ w.T (+ bias) via tensor cores.  Caller guarantees K % 64 == 0,
// 16B-aligned rows, and (ks > 1) => ws holds ks * M * N floats.
// ---------------------------------------------------------------------------
void pl_tc_gemm(const at::Tensor& x, const at::Tensor& w,
                const c10::optional<at::Tensor>& bias, at::Tensor& out,
                const c10::optional<at::Tensor>& ws, int64_t ks) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)w.size(0);
  const int ldx = (int)x.stride(0);
  const int ldw = (int)w.stride(0);
  const void* bp = bias.has_value() ? bias->data_ptr() : nullptr;
  float* wsp = ws.has_value() ? (float*)ws->data_ptr() : nullptr;
  if (x.scalar_type() == at::kBFloat16) {
    using T = __nv_bfloat16;
    pl_tc_launch<T>((const T*)x.data_ptr(), (const T*)w.data_ptr(), (const T*)bp,
                    (T*)out.data_ptr(), wsp, M, N, K, ldx, ldw, (int)ks, stream);
  } else {
    using T = __half;
    pl_tc_launch<T>((const T*)x.data_ptr(), (const T*)w.data_ptr(), (const T*)bp,
                    (T*)out.data_ptr(), wsp, M, N, K, ldx, ldw, (int)ks, stream);
  }
}


// ===========================================================================
// FP8 block-scaled path (L1 ``Fp8Linear`` replacement pieces).
//
// On B200 the L1 op's FlashInfer swapAB fast path is dead -- its gate is
// ``get_device_capability()[0] == 9`` -- so *every* fp8 case, decode included,
// runs ``per_token_group_quant_fp8`` + ``deep_gemm.fp8_gemm_nt``.  Two
// structural costs fall out of that, both measured in-window:
//
//   * the baseline quant kernel is a two-pass shared-memory kernel at 16
//     threads per 128-element group and runs at 1.4 TB/s (10.2 us of QKV#4's
//     54.3 us window);
//   * it emits plain fp32 scales, so DeepGEMM runs its own
//     ``transform_sf_into_required_layout`` as an extra kernel -- worth 4.1-5.9
//     us per case.
//
// ``pl_fp8_group_quant`` fixes both: register-resident, one warp per (row, pack
// of 4 groups), and it writes the packed int32 mn-major SF layout DeepGEMM
// actually wants.  ``pl_fp8_gemv`` replaces quant+GEMM outright for a tiny
// batch, where the GEMM is a pure 37.75 MB weight stream that DeepGEMM runs at
// 2.3 TB/s from ceil(N/128) = 72 CTAs on 148 SMs.
//
// Scale layouts (verified against deep_gemm, not assumed -- see ITERATIONS.md):
//   weight: int32[N, ceil(K/512)], stride (1, N), one entry per weight *row*;
//           scale(n, kb) = bitcast_f32(((ws[n, kb/4] >> 8*(kb%4)) & 0xFF) << 23)
//   act:    int32[M, ceil(ng/4)], stride (1, align(M,4)), same byte packing.
// ===========================================================================

#include <cuda_fp8.h>

namespace {

constexpr float PL_FP8_MAX = 448.0f;
constexpr float PL_FP8_EPS = 1e-10f;
constexpr int PL_FP8_GROUP = 128;
#define PL_FP8_GEMV_BLOCK 256
// Activation staging budget: M*K floats of dynamic shared memory (32 KB).
#define PL_FP8_GEMV_MAX_XELEMS 8192

// Bit-exact with ``fp8_linear.cu:ComputeGroupScale`` (SCALE_UE8M0=true).  The
// build passes no --use_fast_math, so the identical expression is identical
// bits; do not "simplify" the fmaxf/fabsf away.
__device__ __forceinline__ float pl_ue8m0_scale(float absmax) {
  float y_s = absmax / PL_FP8_MAX;
  return exp2f(ceilf(log2f(fmaxf(fabsf(y_s), PL_FP8_EPS))));
}

__device__ __forceinline__ unsigned pl_ue8m0_byte(float scale) {
  // scale is exactly 2^e with a zero mantissa, so the biased exponent is the
  // whole payload -- same extraction as deep_gemm's pack_ue8m0_to_int.
  return (__float_as_uint(scale) >> 23) & 0xFFu;
}

// ---------------------------------------------------------------------------
// Per-token-group fp8 quantization, group size 128.
//
// One warp per (row m, pack p) where a pack is 4 consecutive groups = 512
// elements.  Lane t owns 16 contiguous elements, so lanes [8g, 8g+8) cover
// group g and the absmax reduction is a 3-step __shfl_xor inside 8 lanes; the
// four exponents then live in lanes 0/8/16/24 and lane 0 assembles the int32.
// ---------------------------------------------------------------------------
template <typename T, bool PACKED>
__global__ __launch_bounds__(256) void pl_fp8_quant_kernel(
    const T* __restrict__ X, __nv_fp8_storage_t* __restrict__ Q,
    void* __restrict__ S, int M, int K, int ng, int npack, int ldx, int lds) {
  const int warp = (blockIdx.x * 256 + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int m = warp / npack;
  if (m >= M) return;
  const int pack = warp - m * npack;

  const int sub = lane >> 3;              // which of the pack's 4 groups
  const int g = pack * 4 + sub;           // this lane's group index
  const bool active = g < ng;
  const int k0 = g * PL_FP8_GROUP + (lane & 7) * 16;

  // Every lane runs the reduction even when its group is past the end (a
  // divergent __shfl_xor_sync whose mask names absent lanes silently corrupts
  // the partial-pack case -- measured, see ITERATIONS.md).  Inactive lanes
  // carry eps, which is the identity for fmaxf here.
  float amax = PL_FP8_EPS;
  Vec8<T> a, b;
  if (active) {
    const T* src = X + (size_t)m * ldx + k0;
    a.raw = *reinterpret_cast<const uint4*>(src);
    b.raw = *reinterpret_cast<const uint4*>(src + 8);
    // four independent chains so the 16 fmaxf's are not one serial dependency
    float m0 = PL_FP8_EPS, m1 = 0.f, m2 = 0.f, m3 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      m0 = fmaxf(m0, fabsf(pl_f(a.v[i])));
      m1 = fmaxf(m1, fabsf(pl_f(a.v[4 + i])));
      m2 = fmaxf(m2, fabsf(pl_f(b.v[i])));
      m3 = fmaxf(m3, fabsf(pl_f(b.v[4 + i])));
    }
    amax = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
  }
  // lane ^ {4,2,1} stays inside the 8 lanes that share this group
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  const float y_s = pl_ue8m0_scale(amax);

  if (active) {
    // y_s is exactly 2^e, so 1/y_s is exact and x * (1/y_s) is bit-identical to
    // the baseline's x / y_s -- one divide per thread instead of sixteen.
    const float inv = 1.0f / y_s;
    uint4 out;
    __nv_fp8_storage_t* o = reinterpret_cast<__nv_fp8_storage_t*>(&out);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      o[i] = __nv_cvt_float_to_fp8(
          fminf(fmaxf(pl_f(a.v[i]) * inv, -PL_FP8_MAX), PL_FP8_MAX),
          __NV_SATFINITE, __NV_E4M3);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      o[8 + i] = __nv_cvt_float_to_fp8(
          fminf(fmaxf(pl_f(b.v[i]) * inv, -PL_FP8_MAX), PL_FP8_MAX),
          __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint4*>(Q + (size_t)m * K + k0) = out;
  }

  const unsigned e = active ? pl_ue8m0_byte(y_s) : 0u;
  if constexpr (PACKED) {
    // Gather the pack's four group exponents into lane 0 as one int32.
    const unsigned w1 = __shfl_sync(0xffffffffu, e, 8);
    const unsigned w2 = __shfl_sync(0xffffffffu, e, 16);
    const unsigned w3 = __shfl_sync(0xffffffffu, e, 24);
    if (lane == 0)
      reinterpret_cast<int*>(S)[m + (size_t)pack * lds] =
          (int)(e | (w1 << 8) | (w2 << 16) | (w3 << 24));
  } else {
    if ((lane & 7) == 0 && active)
      reinterpret_cast<float*>(S)[m + (size_t)g * lds] = y_s;
  }
}

// ---------------------------------------------------------------------------
// out[m, n] = sum_kb xs[m, kb] * ws[n, kb] * sum_{k in kb} xq[m, k] * wq[n, k]
//
// One warp per output row n.  Iteration i has lane t reading the uint4 at
// element (i*32 + t)*16 of the row, so the warp reads 512 contiguous bytes per
// iteration (a per-lane 128 B stride would waste half of every 32 B sector).
// Lane t's 16 elements sit entirely inside k-block b = i*4 + t/8, so the two
// block scales fold in per iteration; ws is a warp-wide broadcast because
// b/4 == i for every lane.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pl_fp8x16_to_f32(
    const uint4& raw, float (&out)[16]) {
  const __nv_fp8x2_storage_t* p = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p[j], __NV_E4M3);
    float2 f = __half22float2(*reinterpret_cast<__half2*>(&hr));
    out[2 * j] = f.x;
    out[2 * j + 1] = f.y;
  }
}

template <typename T, int MMAX, int UNROLL, bool READ_ONLY = false>
__global__ __launch_bounds__(PL_FP8_GEMV_BLOCK) void pl_fp8_gemv_kernel(
    const __nv_fp8_storage_t* __restrict__ Xq, const float* __restrict__ Xs,
    const __nv_fp8_storage_t* __restrict__ Wq, const int* __restrict__ Ws,
    const T* __restrict__ Bias, T* __restrict__ Out, int M, int N, int K,
    int ldxs, int ldws, int iters) {
  constexpr int WARPS = PL_FP8_GEMV_BLOCK / 32;
  const int lane = threadIdx.x & 31;
  const int sub = lane >> 3;                  // k-block inside the pack of 4

  // Dequantize the (tiny) activation ONCE into shared memory, with its
  // per-group scale already folded in.  Reading it as fp8 inside the row loop
  // instead costs one fp8->f32 convert per (row, element) -- 37.7M redundant
  // converts at N=9216, which measured 33.7 us against a 10.4 us floor.
  extern __shared__ float pl_xf[];
  const int ng = K / PL_FP8_GROUP;
  for (int i = threadIdx.x; i < M * K; i += PL_FP8_GEMV_BLOCK) {
    const int m = i / K, k = i - m * K;
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)Xq[i], __NV_E4M3);
    pl_xf[i] = __low2float(*reinterpret_cast<__half2*>(&hr)) *
               Xs[m + (size_t)(k / PL_FP8_GROUP) * ldxs];
  }
  __syncthreads();

  const int n = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (n >= N) return;

  float tot[MMAX];
#pragma unroll
  for (int m = 0; m < MMAX; ++m) tot[m] = 0.f;

  const __nv_fp8_storage_t* wrow = Wq + (size_t)n * K;
  const int* wsrow = Ws + n;

  // Iteration i has lane t read the uint4 at element (i*32 + t)*16, so a warp
  // reads 512 B contiguous (a per-lane 128 B stride wastes half of every 32 B
  // sector).  UNROLL weight loads are issued before any is consumed: with one
  // load in flight per thread the kernel runs at HBM *latency*, not bandwidth.
  for (int i0 = 0; i0 < iters; i0 += UNROLL) {
    uint4 wv[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      wv[u] = *reinterpret_cast<const uint4*>(wrow + ((i0 + u) * 32 + lane) * 16);
    unsigned wsw[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      wsw[u] = (unsigned)wsrow[(size_t)(i0 + u) * ldws];
    if (READ_ONLY) {   // floor reference: pay the loads, skip the math
#pragma unroll
      for (int u = 0; u < UNROLL; ++u)
        tot[0] += (float)(wv[u].x ^ wv[u].y ^ wv[u].z ^ wv[u].w) + (float)wsw[u];
      continue;
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int e = ((i0 + u) * 32 + lane) * 16;
      const float wsc = __uint_as_float(((wsw[u] >> (8 * sub)) & 0xFFu) << 23);
      float wf[16];
      pl_fp8x16_to_f32(wv[u], wf);
#pragma unroll
      for (int m = 0; m < MMAX; ++m) {
        if (m >= M) break;
        const float* xr = pl_xf + (size_t)m * K + e;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) acc = fmaf(xr[j], wf[j], acc);
        tot[m] = fmaf(acc, wsc, tot[m]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MMAX; ++m) {
    if (m >= M) break;
    float v = tot[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) {
      if (Bias) v += pl_f(Bias[n]);
      pl_store(Out + (size_t)m * N + n, v);
    }
  }
}

template <typename T, int MMAX, bool READ_ONLY>
void pl_fp8_gemv_dispatch(const void* xq, const float* xs, const void* wq,
                          const int* ws, const void* bias, void* out, int M,
                          int N, int K, int ldxs, int ldws, cudaStream_t stream) {
  constexpr int WARPS = PL_FP8_GEMV_BLOCK / 32;
  const int iters = K / 512;
  const dim3 grid((N + WARPS - 1) / WARPS);
  const size_t smem = (size_t)M * K * sizeof(float);
#define PL_GEMV_U(U)                                                          \
  pl_fp8_gemv_kernel<T, MMAX, U, READ_ONLY>                                   \
      <<<grid, PL_FP8_GEMV_BLOCK, smem, stream>>>(                            \
      (const __nv_fp8_storage_t*)xq, xs, (const __nv_fp8_storage_t*)wq, ws,   \
      (const T*)bias, (T*)out, M, N, K, ldxs, ldws, iters)
  if (iters % 4 == 0) {
    PL_GEMV_U(4);
  } else if (iters % 2 == 0) {
    PL_GEMV_U(2);
  } else {
    PL_GEMV_U(1);
  }
#undef PL_GEMV_U
}

template <typename T, bool READ_ONLY>
void pl_fp8_gemv_launch(const void* xq, const float* xs, const void* wq,
                        const int* ws, const void* bias, void* out, int M, int N,
                        int K, int ldxs, int ldws, cudaStream_t stream) {
  if (M == 1)
    pl_fp8_gemv_dispatch<T, 1, READ_ONLY>(xq, xs, wq, ws, bias, out, M, N, K, ldxs, ldws, stream);
  else if (M <= 2)
    pl_fp8_gemv_dispatch<T, 2, READ_ONLY>(xq, xs, wq, ws, bias, out, M, N, K, ldxs, ldws, stream);
  else if (M <= 4)
    pl_fp8_gemv_dispatch<T, 4, READ_ONLY>(xq, xs, wq, ws, bias, out, M, N, K, ldxs, ldws, stream);
  else
    pl_fp8_gemv_dispatch<T, 8, READ_ONLY>(xq, xs, wq, ws, bias, out, M, N, K, ldxs, ldws, stream);
}

}  // namespace

// ---------------------------------------------------------------------------
// x[M, K] (bf16/fp16, row pitch ldx) -> q[M, K] fp8 e4m3 + per-128-group UE8M0
// scales.  ``packed`` selects int32[M, ceil(ng/4)] (DeepGEMM's own SF layout,
// stride (1, lds)) over plain float32[M, ng] (stride (1, lds)).
// ---------------------------------------------------------------------------
void pl_fp8_group_quant(const at::Tensor& x, at::Tensor& q, at::Tensor& s,
                        bool packed) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int ng = K / PL_FP8_GROUP;
  // Warps always own a pack of 4 groups, packed SF layout or not.
  const int npack = (ng + 3) / 4;
  if (M <= 0 || npack <= 0) return;   // a zero-block grid is a launch error
  const int ldx = (int)x.stride(0);
  const int lds = (int)s.stride(1);
  const long warps = (long)M * npack;
  const int blocks = (int)((warps + 7) / 8);
  auto* qp = (__nv_fp8_storage_t*)q.data_ptr();
  if (x.scalar_type() == at::kBFloat16) {
    using T = __nv_bfloat16;
    if (packed)
      pl_fp8_quant_kernel<T, true><<<blocks, 256, 0, stream>>>(
          (const T*)x.data_ptr(), qp, s.data_ptr(), M, K, ng, npack, ldx, lds);
    else
      pl_fp8_quant_kernel<T, false><<<blocks, 256, 0, stream>>>(
          (const T*)x.data_ptr(), qp, s.data_ptr(), M, K, ng, npack, ldx, lds);
  } else {
    using T = __half;
    if (packed)
      pl_fp8_quant_kernel<T, true><<<blocks, 256, 0, stream>>>(
          (const T*)x.data_ptr(), qp, s.data_ptr(), M, K, ng, npack, ldx, lds);
    else
      pl_fp8_quant_kernel<T, false><<<blocks, 256, 0, stream>>>(
          (const T*)x.data_ptr(), qp, s.data_ptr(), M, K, ng, npack, ldx, lds);
  }
}

template <bool READ_ONLY>
static void pl_fp8_gemv_impl(const at::Tensor& xq, const at::Tensor& xs,
                             const at::Tensor& wq, const at::Tensor& ws,
                             const c10::optional<at::Tensor>& bias,
                             at::Tensor& out) {
  const at::cuda::OptionalCUDAGuard guard(device_of(xq));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int M = (int)xq.size(0);
  const int K = (int)xq.size(1);
  const int N = (int)wq.size(0);
  const void* bp = bias.has_value() ? bias->data_ptr() : nullptr;
  if (out.scalar_type() == at::kBFloat16)
    pl_fp8_gemv_launch<__nv_bfloat16, READ_ONLY>(
        xq.data_ptr(), (const float*)xs.data_ptr(), wq.data_ptr(),
        (const int*)ws.data_ptr(), bp, out.data_ptr(), M, N, K,
        (int)xs.stride(1), (int)ws.stride(1), stream);
  else
    pl_fp8_gemv_launch<__half, READ_ONLY>(
        xq.data_ptr(), (const float*)xs.data_ptr(), wq.data_ptr(),
        (const int*)ws.data_ptr(), bp, out.data_ptr(), M, N, K,
        (int)xs.stride(1), (int)ws.stride(1), stream);
}

// ---------------------------------------------------------------------------
// Block-scaled fp8 GEMV.  Caller guarantees M <= 8, K % 512 == 0, contiguous
// xq/wq, float32 activation scales with stride (1, ldxs), packed int32 weight
// scales with stride (1, ldws).
// ---------------------------------------------------------------------------
void pl_fp8_gemv(const at::Tensor& xq, const at::Tensor& xs,
                 const at::Tensor& wq, const at::Tensor& ws,
                 const c10::optional<at::Tensor>& bias, at::Tensor& out) {
  TORCH_CHECK(xq.size(0) * xq.size(1) <= PL_FP8_GEMV_MAX_XELEMS,
              "pl_fp8_gemv: M*K too large to stage the activation in shared "
              "memory; the caller must gate on PL_FP8_GEMV_MAX_XELEMS");
  TORCH_CHECK(xq.size(1) % 512 == 0, "pl_fp8_gemv: K must be a multiple of 512");
  const at::cuda::OptionalCUDAGuard guard(device_of(xq));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pl_fp8_gemv_impl<false>(xq, xs, wq, ws, bias, out);
}

void pl_fp8_gemv_read_only(const at::Tensor& xq, const at::Tensor& xs,
                           const at::Tensor& wq, const at::Tensor& ws,
                           const c10::optional<at::Tensor>& bias,
                           at::Tensor& out) {
  pl_fp8_gemv_impl<true>(xq, xs, wq, ws, bias, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pl_gemv", &pl_gemv, "tiny-batch skinny GEMV (CUDA)",
        py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("out"),
        py::arg("G") = 0, py::arg("rpw") = 0);
  m.def("pl_tc_gemm", &pl_tc_gemm, "tensor-core skinny GEMM (CUDA)",
        py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("out"),
        py::arg("ws"), py::arg("ks"));
  m.def("pl_fp8_group_quant", &pl_fp8_group_quant,
        "per-token-group fp8 quant, packed or plain UE8M0 scales (CUDA)",
        py::arg("x"), py::arg("q"), py::arg("s"), py::arg("packed"));
  m.def("pl_fp8_gemv", &pl_fp8_gemv, "block-scaled fp8 GEMV (CUDA)",
        py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"),
        py::arg("bias"), py::arg("out"));
  m.def("pl_fp8_gemv_read_only", &pl_fp8_gemv_read_only,
        "same loads, no math -- streaming floor reference (CUDA)",
        py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"),
        py::arg("bias"), py::arg("out"));
  m.attr("PL_FP8_GEMV_MAX_XELEMS") = (int64_t)PL_FP8_GEMV_MAX_XELEMS;
}
