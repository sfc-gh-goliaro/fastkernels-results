// Fused Oasis final DiT projection layer.
//
//   modulation = silu(c) @ Wa^T + ba          -> (shift, scale)
//   y          = layernorm(x) * (1 + scale) + shift
//   out        = y @ Wl^T + bl
//
// The eager reference is ~10 tiny launches (one of which promotes x to fp32 for
// the layernorm) over a problem holding only a few microseconds of real work,
// so it is dominated by fixed costs.  Two kernels here:
//
//   mod_kernel6     -- silu + the [2K,K] x [F,K] modulation GEMV, split over k,
//                      leaving NCH6 fp32 partial sums per row.
//   final_kernel_T2 -- joins those partials, then per 16-token tile does the
//                      layernorm (fp32 reduction), the modulate, and the
//                      K=1024 -> N=64 projection on tensor cores (m16n8k16).
//
// The captured activation is *frame-major*: x is [1, F, 9, 16, K] with strides
// [F*K*TOK, K*TOK, 16, 1, TOK] (TOK = 9*16), i.e. each frame is a dense
// [K][TOK] matrix whose contiguous axis is the token, not the hidden dim -- the
// transpose of what a projection kernel wants.  Anything else falls back to the
// eager path.
//
// Both kernels are latency-bound rather than bandwidth-bound at these sizes
// (ncu shows DRAM under 6% of peak), so the shapes here are chosen to maximize
// warps and in-flight loads; see the comments on each kernel.
//
// Rounding follows the eager reference: every intermediate PyTorch materializes
// as fp16 is rounded to fp16 here too.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int KDIM = 1024;        // hidden size
constexpr int NOUT = 64;          // out features (patch^2 * out_channels)
constexpr int _MAXF = 8;          // max modulation rows (frames) supported

// ---- modulation GEMV -------------------------------------------------------
constexpr int MR6 = 16;               // Wa rows per CTA
constexpr int KCH6 = 512;             // k per CTA
constexpr int NCH6 = KDIM / KCH6;     // partial sums per Wa row
constexpr int THREADS6 = 512;         // warps split MR6 rows
constexpr int RW6 = MR6 / (THREADS6 / 32);
constexpr int KV6 = KCH6 / 8;         // uint4 per Wa row within a chunk
constexpr int PSTRIDE = 2 * KDIM;     // floats per (chunk, frame) plane
constexpr int MGRID = 2 * KDIM / MR6; // modulation CTAs per k-chunk

// ---- projection ------------------------------------------------------------
constexpr int BMT = 16;           // tokens per CTA == one mma M tile
constexpr int ALDT = KDIM + 8;    // 1032: padded row stride, bank-conflict free
constexpr int THREADS_T2 = 1024;      // 32 warps
// NH = how many CTAs split the 64 output columns.  Staging Wl dominated the
// kernel, so a bigger NH stages less per CTA and grows the grid -- but only up
// to one wave: past multiProcessorCount CTAs it costs more than it saves, and
// the layernorm work is repeated by each of the NH CTAs.  Chosen per shape.
template <int NH> struct Cfg {
  static constexpr int NPC = NOUT / NH;             // output columns per CTA
  static constexpr int NGRP = NPC / 16;             // n-tile pairs per CTA
  static constexpr int KS2 = (THREADS_T2 / 32) / NGRP;   // k-slices
  static constexpr int KSPAN = KDIM / KS2;
};

// ---------------------------------------------------------------- primitives

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mma16816(float& d0, float& d1, float& d2, float& d3,
                                         unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One ldmatrix replaces the four (A) / two (B) scalar shared loads an mma
// fragment would otherwise need.
__device__ __forceinline__ void ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                       unsigned& r3, const __half* p) {
  unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(unsigned& r0, unsigned& r1,
                                       const __half* p) {
  unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(a));
}

// ------------------------------------------------------------- mod_kernel6
// Reading Wa (4 MiB, ~70% of the operator's traffic) is latency-bound, not
// bandwidth-bound: ncu shows DRAM under 6% of peak.  Two things follow.
//
// First, each CTA issues its *entire* Wa tile as one cp.async batch, so across
// the grid all 4 MiB is in flight at once rather than a couple of loads per
// warp.  Second, the grid has to be large -- with one CTA per SM there are not
// enough warps to cover the latency, and raising warps/CTA helped while cutting
// shared-memory traffic did not.
//
// The grid could not grow while every CTA needed u = silu(c) over all of k:
// silu is two quarter-rate MUFU ops per element and is recomputed per CTA, so
// its total cost scales with the CTA count.  Splitting k breaks the tie -- a
// CTA owning MR6 rows x KCH6 columns needs u only over its own KCH6, so total
// silu work depends on MR6 alone while the grid grows by NCH6.  The NCH6
// partial sums per row are joined by the consumer in final_kernel_T2.
template <int NF>
__global__ __launch_bounds__(THREADS6) void mod_kernel6(
    const __half* __restrict__ c, const __half* __restrict__ Wa,
    float* __restrict__ part) {
  __shared__ __half wsa[MR6 * KCH6];
  __shared__ __half u[NF * KCH6];

  const int tid = threadIdx.x;
  const int row0 = blockIdx.x * MR6;
  const int kc = blockIdx.y * KCH6;

#pragma unroll
  for (int it = 0; it < MR6 * (KCH6 / 8) / THREADS6; ++it) {
    const int v = tid + it * THREADS6;
    const int r = v / KV6;
    const int j = (v - r * KV6) * 8;
    cp_async16(&wsa[r * KCH6 + j], &Wa[(size_t)(row0 + r) * KDIM + kc + j]);
  }
  cp_commit();

  // silu(c) over this CTA's k slice, rounded to fp16 exactly as F.silu would.
  for (int v = tid; v < NF * KV6; v += THREADS6) {
    const int fi = v / KV6;
    const int j = (v - fi * KV6) * 8;
    uint4 g = *reinterpret_cast<const uint4*>(c + (size_t)fi * KDIM + kc + j);
    __half2 h[4];
    *reinterpret_cast<uint4*>(h) = g;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 ff = __half22float2(h[i]);
      ff.x = ff.x / (1.0f + __expf(-ff.x));
      ff.y = ff.y / (1.0f + __expf(-ff.y));
      h[i] = __float22half2_rn(ff);
    }
    *reinterpret_cast<uint4*>(&u[fi * KCH6 + j]) = *reinterpret_cast<uint4*>(h);
  }
  cp_wait<0>();
  __syncthreads();

  const int lane = tid & 31;
  const int warp = tid >> 5;

  float acc[RW6][NF];
#pragma unroll
  for (int r = 0; r < RW6; ++r)
#pragma unroll
    for (int f = 0; f < NF; ++f) acc[r][f] = 0.0f;

#pragma unroll
  for (int base = 0; base < KCH6; base += 256) {
    const int k = base + lane * 8;
    uint4 uv[NF];
#pragma unroll
    for (int f = 0; f < NF; ++f)
      uv[f] = *reinterpret_cast<const uint4*>(&u[f * KCH6 + k]);
#pragma unroll
    for (int r = 0; r < RW6; ++r) {
      uint4 wv =
          *reinterpret_cast<const uint4*>(&wsa[(warp * RW6 + r) * KCH6 + k]);
      const __half* wh = reinterpret_cast<const __half*>(&wv);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        const __half* uh = reinterpret_cast<const __half*>(&uv[f]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[r][f] = fmaf(__half2float(uh[j]), __half2float(wh[j]), acc[r][f]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RW6; ++r)
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        acc[r][f] += __shfl_xor_sync(0xffffffffu, acc[r][f], off);

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < RW6; ++r) {
      const int row = row0 + warp * RW6 + r;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        part[((size_t)blockIdx.y * _MAXF + f) * PSTRIDE + row] = acc[r][f];
    }
  }

}

// --------------------------------------------------------- final_kernel_T2
// One CTA per (16-token tile, frame).  Because the contiguous axis of x is the
// token, an 8x8 (k x p) block is loaded per thread and transposed into a [p][k]
// shared tile, which is the layout the m16n8k16 A-fragments want.
//
// Shaped by what ncu said limited it: 1024 threads (K split eight ways) so the
// single CTA/SM forced by 165 KiB of shared memory still fills half the warp
// slots; ldmatrix to cut inner-loop shared instructions; and two adjacent
// n-tiles per warp so an A fragment is fetched once and used twice.  The whole
// projection weight is staged with one cp.async batch waited on after the
// (independent) layernorm work, rather than a multi-stage pipeline whose
// serialized latency dominates at these sizes.
template <int NH>
__global__ __launch_bounds__(THREADS_T2) void final_kernel_T2(
    const __half* __restrict__ x, const float* __restrict__ part,
    const __half* __restrict__ ba, const __half* __restrict__ Wl,
    const __half* __restrict__ bl, __half* __restrict__ out, int tok,
    float eps) {
  constexpr int NPC = Cfg<NH>::NPC;
  constexpr int NGRP = Cfg<NH>::NGRP;
  constexpr int KS2 = Cfg<NH>::KS2;
  constexpr int KSPAN = Cfg<NH>::KSPAN;
  static_assert((KS2 - 1) * NGRP * 32 * 8 * sizeof(float) <=
                    BMT * ALDT * sizeof(__half),
                "reduction buffer overruns the x tile it aliases");
  extern __shared__ __align__(16) char smem[];
  __half* xs = reinterpret_cast<__half*>(smem);   // [BMT][ALDT]  x, then y
  __half* gs = xs + BMT * ALDT;                   // [KDIM]       1 + scale
  __half* ss = gs + KDIM;                         // [KDIM]       shift
  __half* wsm = ss + KDIM;                        // [NPC][ALDT]
  float* red = reinterpret_cast<float*>(smem);    // aliases xs after the GEMM
  __shared__ float mrs[2 * BMT];                  // mean, rstd per token

  const int tid = threadIdx.x;
  const int f = blockIdx.y;
  const int p0 = blockIdx.x * BMT;
  const int nbase = blockIdx.z * NPC;   // this CTA's slice of the output columns

  // (1) Stage this CTA's slice of Wl -- issued first so the transfer overlaps
  //     (2)-(5).  Staging all 64 rows was the single largest cost in the
  //     kernel (~3.5 us of 11.5); splitting the columns across NHALF CTAs cuts
  //     it proportionally and doubles the grid, which this kernel is short of.
#pragma unroll
  for (int it = 0; it < NPC * (KDIM / 8) / THREADS_T2; ++it) {
    const int v = tid + it * THREADS_T2;
    cp_async16(&wsm[(v >> 7) * ALDT + ((v & 127) << 3)],
               &Wl[(size_t)(nbase + (v >> 7)) * KDIM + ((v & 127) << 3)]);
  }
  cp_commit();

  // (2) Join this frame's modulation partials and add the bias.  scale is
  //     rounded to fp16 before the (1 + scale), matching eager.
  for (int n = tid; n < 2 * KDIM; n += THREADS_T2) {
    float a = 0.0f;
#pragma unroll
    for (int ch = 0; ch < NCH6; ++ch)
      a += part[((size_t)ch * _MAXF + f) * PSTRIDE + n];
    a += __half2float(ba[n]);
    if (n < KDIM) {
      ss[n] = __float2half(a);
    } else {
      __half sc = __float2half(a);
      gs[n - KDIM] = __float2half(1.0f + __half2float(sc));
    }
  }

  // (3) Two k's x eight p's per thread, transposed into xs[p][k].
  {
    const __half* xf = x + (size_t)f * KDIM * tok + p0;
    const int ph = tid & 1;
    const int kb = tid >> 1;          // 0..511 -> k = 2*kb, 2*kb + 1
    uint4 r0 =
        *reinterpret_cast<const uint4*>(xf + (size_t)(2 * kb) * tok + ph * 8);
    uint4 r1 =
        *reinterpret_cast<const uint4*>(xf + (size_t)(2 * kb + 1) * tok + ph * 8);
    const __half* h0 = reinterpret_cast<const __half*>(&r0);
    const __half* h1 = reinterpret_cast<const __half*>(&r1);
#pragma unroll
    for (int pp = 0; pp < 8; ++pp) {
      __half two[2] = {h0[pp], h1[pp]};
      *reinterpret_cast<unsigned*>(&xs[(ph * 8 + pp) * ALDT + 2 * kb]) =
          *reinterpret_cast<const unsigned*>(two);
    }
  }
  __syncthreads();

  // (4) LayerNorm statistics: one warp per token, fp32 reduction.
  if (tid < BMT * 32) {
    const int p = tid >> 5, j = tid & 31;
    const __half* row = &xs[p * ALDT + j * 32];
    float s1 = 0.0f, s2 = 0.0f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      uint4 v = *reinterpret_cast<const uint4*>(row + u * 8);
      const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        float fv = __half2float(h[q]);
        s1 += fv;
        s2 = fmaf(fv, fv, s2);
      }
    }
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      s2 += __shfl_xor_sync(0xffffffffu, s2, off);
    }
    if (j == 0) {
      const float mean = s1 * (1.0f / KDIM);
      mrs[p] = mean;
      mrs[BMT + p] = rsqrtf(fmaf(-mean, mean, s2 * (1.0f / KDIM)) + eps);
    }
  }
  __syncthreads();

  // (5) y = fp16(fp16(xhat) * gamma + shift), in place.  Every intermediate
  //     PyTorch materializes as fp16 is rounded to fp16 here too.
#pragma unroll
  for (int it = 0; it < BMT * (KDIM / 8) / THREADS_T2; ++it) {
    const int v = tid + it * THREADS_T2;
    const int p = v >> 7;
    const int j = (v & 127) << 3;
    const float mean = mrs[p], rstd = mrs[BMT + p];
    uint4 xv = *reinterpret_cast<const uint4*>(&xs[p * ALDT + j]);
    uint4 gv = *reinterpret_cast<const uint4*>(&gs[j]);
    uint4 sv = *reinterpret_cast<const uint4*>(&ss[j]);
    const __half* xh = reinterpret_cast<const __half*>(&xv);
    const __half* gh = reinterpret_cast<const __half*>(&gv);
    const __half* sh = reinterpret_cast<const __half*>(&sv);
    __half o[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      __half xhat = __float2half((__half2float(xh[q]) - mean) * rstd);
      __half m = __float2half(__half2float(xhat) * __half2float(gh[q]));
      o[q] = __float2half(__half2float(m) + __half2float(sh[q]));
    }
    *reinterpret_cast<uint4*>(&xs[p * ALDT + j]) =
        *reinterpret_cast<const uint4*>(o);
  }
  cp_wait<0>();
  __syncthreads();

  // (4) GEMM: warp -> (n-tile pair, k-slice).
  const int warp = tid >> 5, lane = tid & 31;
  const int ng = warp & (NGRP - 1);
  const int ks = warp / NGRP;
  const int n0 = ng * 16;

  // A: lane l gives row (l&7) of tile (l>>3); tile t -> m = (t&1)*8, k += (t>>1)*8
  const __half* apt = &xs[(size_t)(((lane >> 3) & 1) * 8 + (lane & 7)) * ALDT +
                          ((lane >> 4) & 1) * 8];
  // B: lane l gives row (l&7) of tile ((l>>3)&1); tile t -> n = n0, k += t*8
  const __half* bpt0 =
      &wsm[(size_t)(n0 + (lane & 7)) * ALDT + ((lane >> 3) & 1) * 8];
  const __half* bpt1 = bpt0 + 8 * ALDT;

  float d[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int kk = ks * KSPAN; kk < (ks + 1) * KSPAN; kk += 16) {
    unsigned a0, a1, a2, a3, b0, b1;
    ldm_x4(a0, a1, a2, a3, apt + kk);
    ldm_x2(b0, b1, bpt0 + kk);
    mma16816(d[0][0], d[0][1], d[0][2], d[0][3], a0, a1, a2, a3, b0, b1);
    ldm_x2(b0, b1, bpt1 + kk);
    mma16816(d[1][0], d[1][1], d[1][2], d[1][3], a0, a1, a2, a3, b0, b1);
  }

  // (5) Join the KS2 k-slice partials, then the ks == 0 warps write out.
  __syncthreads();                     // xs is dead; red may overwrite it
  if (ks > 0) {
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        red[(((ks - 1) * NGRP + ng) * 32 + lane) * 8 + t * 4 + q] = d[t][q];
  }
  __syncthreads();
  if (ks == 0) {
#pragma unroll
    for (int sl = 0; sl < KS2 - 1; ++sl)
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int q = 0; q < 4; ++q)
          d[t][q] += red[((sl * NGRP + ng) * 32 + lane) * 8 + t * 4 + q];

    __half* op = out + ((size_t)f * tok + p0) * NOUT;
    const int orow = lane >> 2;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int ocol = nbase + n0 + t * 8 + (lane & 3) * 2;
      const float bz = __half2float(bl[ocol]), bo = __half2float(bl[ocol + 1]);
      *reinterpret_cast<__half2*>(op + (size_t)orow * NOUT + ocol) =
          __halves2half2(__float2half(d[t][0] + bz), __float2half(d[t][1] + bo));
      *reinterpret_cast<__half2*>(op + (size_t)(orow + 8) * NOUT + ocol) =
          __halves2half2(__float2half(d[t][2] + bz), __float2half(d[t][3] + bo));
    }
  }
}

template <int NH>
int smem_bytes_T2() {
  return (BMT * ALDT + 2 * KDIM + Cfg<NH>::NPC * ALDT) * (int)sizeof(__half);
}

template <int NH>
void ensure_attr_T2() {
  static bool done = false;
  if (!done) {
    cudaError_t e = cudaFuncSetAttribute(
        final_kernel_T2<NH>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes_T2<NH>());
    TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute(", smem_bytes_T2<NH>(),
                ") failed: ", cudaGetErrorString(e));
    done = true;
  }
}

// Largest column split that still fits in a single wave of CTAs.
int pick_nhalf(int64_t tiles) {
  static int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (tiles * 4 <= sms) return 4;
  if (tiles * 2 <= sms) return 2;
  return 1;
}

// True for the captured frame-major activation described at the top.
bool is_frame_major(const at::Tensor& x, int64_t nmod, int64_t* tok_out) {
  if (x.dim() != 5 || x.size(0) != 1 || x.size(1) != nmod) return false;
  const int64_t tok = x.size(2) * x.size(3);
  if (tok % BMT != 0) return false;
  if (!(x.size(4) == KDIM && x.stride(4) == tok && x.stride(3) == 1 &&
        x.stride(2) == x.size(3) && x.stride(1) == (int64_t)KDIM * tok))
    return false;
  *tok_out = tok;
  return true;
}

}  // namespace

at::Tensor oasis_final(const at::Tensor& x, const at::Tensor& c, const at::Tensor& buf,
                       const at::Tensor& wa, const at::Tensor& ba,
                       const at::Tensor& wl, const at::Tensor& bl) {
  if (!(x.is_cuda() && x.scalar_type() == at::kHalf &&
        c.is_cuda() && c.scalar_type() == at::kHalf && c.is_contiguous() &&
        buf.device() == x.device() && wa.device() == x.device() &&
        buf.scalar_type() == at::kFloat &&
        buf.numel() >= (int64_t)NCH6 * _MAXF * PSTRIDE &&
        x.size(-1) == KDIM && c.size(-1) == KDIM &&
        wl.size(0) == NOUT && wl.size(1) == KDIM && wa.size(0) == 2 * KDIM))
    return at::Tensor();

  const int64_t nmod = c.numel() / KDIM;
  if (nmod < 1 || nmod > _MAXF) return at::Tensor();
  int64_t tok = 0;
  if (!is_frame_major(x, nmod, &tok)) return at::Tensor();
  const int64_t ntiles = tok / BMT * nmod;

  std::vector<int64_t> osz(x.sizes().begin(), x.sizes().end());
  osz.back() = NOUT;
  at::Tensor out = at::empty(osz, x.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  const __half* cp = reinterpret_cast<const __half*>(c.data_ptr());
  const __half* wap = reinterpret_cast<const __half*>(wa.data_ptr());
  const __half* bap = reinterpret_cast<const __half*>(ba.data_ptr());
  float* partp = buf.data_ptr<float>();

  const dim3 grid1(MGRID, NCH6);
#define LAUNCH_MOD(NF)                                              \
  mod_kernel6<NF><<<grid1, THREADS6, 0, stream>>>(cp, wap, partp);   \
  break
  switch (nmod) {
    case 1: LAUNCH_MOD(1);
    case 2: LAUNCH_MOD(2);
    case 3: LAUNCH_MOD(3);
    case 4: LAUNCH_MOD(4);
    case 5: LAUNCH_MOD(5);
    case 6: LAUNCH_MOD(6);
    case 7: LAUNCH_MOD(7);
    case 8: LAUNCH_MOD(8);
    default: return at::Tensor();
  }
#undef LAUNCH_MOD

#define LAUNCH_FIN(NH)                                                      \
  ensure_attr_T2<NH>();                                                     \
  final_kernel_T2<NH><<<dim3((unsigned)(tok / BMT), (unsigned)nmod, NH),     \
                        THREADS_T2, smem_bytes_T2<NH>(), stream>>>(          \
      reinterpret_cast<const __half*>(x.data_ptr()), partp, bap,             \
      reinterpret_cast<const __half*>(wl.data_ptr()),                        \
      reinterpret_cast<const __half*>(bl.data_ptr()),                        \
      reinterpret_cast<__half*>(out.data_ptr()), (int)tok, 1e-6f);            \
  break
  switch (pick_nhalf(ntiles)) {
    case 4: LAUNCH_FIN(4);
    case 2: LAUNCH_FIN(2);
    default: LAUNCH_FIN(1);
  }
#undef LAUNCH_FIN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---- debug entry points (used only by dev scripts) -------------------------
void mod_only(const at::Tensor& c, const at::Tensor& buf, const at::Tensor& wa) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t nmod = c.numel() / KDIM;
  const dim3 grid1(MGRID, NCH6);
  const __half* cp = reinterpret_cast<const __half*>(c.data_ptr());
  const __half* wap = reinterpret_cast<const __half*>(wa.data_ptr());
  float* partp = buf.data_ptr<float>();
#define LM(NF) \
  mod_kernel6<NF><<<grid1, THREADS6, 0, stream>>>(cp, wap, partp); break
  switch (nmod) {
    case 2: LM(2);
    case 3: LM(3);
    case 4: LM(4);
    case 5: LM(5);
    case 6: LM(6);
    default: TORCH_CHECK(false, "mod_only: nmod");
  }
#undef LM
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void final_only(const at::Tensor& x, const at::Tensor& buf, const at::Tensor& ba,
                const at::Tensor& wl, const at::Tensor& bl, at::Tensor& out,
                int64_t nmod, int64_t nh) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t tok = x.size(2) * x.size(3);
#define LF(NH)                                                             \
  ensure_attr_T2<NH>();                                                    \
  final_kernel_T2<NH><<<dim3((unsigned)(tok / BMT), (unsigned)nmod, NH),    \
                        THREADS_T2, smem_bytes_T2<NH>(), stream>>>(         \
      reinterpret_cast<const __half*>(x.data_ptr()), buf.data_ptr<float>(), \
      reinterpret_cast<const __half*>(ba.data_ptr()),                       \
      reinterpret_cast<const __half*>(wl.data_ptr()),                       \
      reinterpret_cast<const __half*>(bl.data_ptr()),                       \
      reinterpret_cast<__half*>(out.data_ptr()), (int)tok, 1e-6f);           \
  break
  switch (nh) {
    case 4: LF(4);
    case 2: LF(2);
    default: LF(1);
  }
#undef LF
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor alloc_only(const at::Tensor& x) {
  std::vector<int64_t> osz(x.sizes().begin(), x.sizes().end());
  osz.back() = NOUT;
  return at::empty(osz, x.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oasis_final", &oasis_final, "fused oasis final layer");
  m.def("mod_only", &mod_only, "debug: modulation kernel only");
  m.def("final_only", &final_only, "debug: projection kernel only");
  m.def("alloc_only", &alloc_only, "debug: output allocation only");
}
