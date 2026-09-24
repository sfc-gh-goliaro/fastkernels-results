// YOLOv10 detection head -- the whole head in one host call.
//
// The baseline spends ~1.7 ms on this operator with ~0.9 ms of actual kernel
// time: 24 convolutions, each split into conv + BatchNorm + SiLU, then an
// inference tail (DFL softmax, dist2bbox, sigmoid) and a v10 post-process whose
// two torch.topk calls alone cost ~145 us.  Nothing here is arithmetic-bound --
// the head is 1.9 GFLOP per image -- so the win is in launch count, layout and
// the top-k.
//
// This file keeps every activation in NHWC (channels last), which makes
//   * 1x1 convs a plain GEMM over the channel axis,
//   * 3x3 convs an implicit GEMM whose im2col gather is one contiguous
//     `Cin`-half run per (r,s) tap,
//   * depthwise 3x3 a fully coalesced 9-tap load, and
//   * the per-anchor tail (4x16 DFL softmax, 80-way class max) a contiguous
//     read of one row.
// BatchNorm folds into a per-channel scale/shift applied in the mma epilogue,
// with SiLU riding along; the intermediate roundings to fp16 that the baseline
// performs between conv, BN and SiLU are reproduced so the class logits land on
// the same fp16 grid (the post-process top-k is decided almost entirely by ties,
// see k_post).
//
// The six chains (3 feature levels x {cv2, cv3}) are independent, so they are
// forked onto side streams and joined before the post-process.  That whole fork
// is captured into a CUDA graph once and replayed afterwards: the conv kernels
// are individually cheap enough that ~5 us of host launch cost each was the
// binding constraint, so a call now costs three launches (the NCHW->NHWC
// transpose, the graph, the post-process).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>

namespace wm = nvcuda::wmma;

#define DIVUP(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// elementwise pieces, matching ATen's fp16 op-math (compute in float, round to
// half once per op)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_f(float x) {
  return x / (1.f + expf(-x));
}
__device__ __forceinline__ float sigmoid_f(float x) {
  return 1.f / (1.f + expf(-x));
}

// ---------------------------------------------------------------------------
// global -> shared staging.  `cp.async`'s src-size operand zero-fills the tail,
// which is exactly the im2col out-of-bounds tap, and decouples the load from the
// mma so the k-loop can run several tiles deep.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
  const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
               "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// act codes
#define ACT_BIAS 0   // y = half(acc + shift)                      (plain Conv2d)
#define ACT_SILU 1   // y = half(silu(half(scale*half(acc)+shift))) (YOLOConv)

// epilogue flavour
#define EPI_PLAIN 0  // write the activation
#define EPI_DFL 1    // cv2's last conv: DFL softmax + dist2bbox -> xywh
#define EPI_CLS 2    // cv3's last conv: sigmoid -> scores + per-anchor max

// ---------------------------------------------------------------------------
// NCHW -> NHWC
// ---------------------------------------------------------------------------
__device__ __forceinline__ void t_nchw2nhwc(const __half* __restrict__ xin,
                                            __half* __restrict__ yout, int C,
                                            int HW, int p0, int c0) {
  constexpr int TP = 64, TC = 64, LD = TC + 8;
  __shared__ __half s[TP * LD];
  const int tid = threadIdx.x;              // 256 threads
  for (int r = tid >> 3; r < TC; r += 32) {          // channel
    const int cc = (tid & 7) * 8;                    // pixel group
    const int p = p0 + cc;
    __half v[8];
    if (p + 7 < HW) {
      *(uint4*)v = *(const uint4*)(xin + (size_t)(c0 + r) * HW + p);
    } else {
#pragma unroll
      for (int u = 0; u < 8; ++u)
        v[u] = (p + u < HW) ? xin[(size_t)(c0 + r) * HW + p + u] : __half(0.f);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) s[(cc + u) * LD + r] = v[u];
  }
  __syncthreads();
  for (int r = tid >> 3; r < TP; r += 32) {          // pixel
    const int cc = (tid & 7) * 8;                    // channel group
    if (p0 + r < HW)
      *(uint4*)(yout + (size_t)(p0 + r) * C + c0 + cc) = *(const uint4*)(s + r * LD + cc);
  }
}

// All three levels in one launch: blockIdx.z selects (level, image), and blocks
// past a level's extent exit immediately.  One launch instead of three matters
// because everything downstream of here is replayed from a captured graph, so
// this kernel and the post-process are the only per-call launches left.
struct T3 {
  const __half* X[3];
  __half* Y[3];
  int C[3], HW[3], npx[3], ncy[3], cum[4];
};

__global__ __launch_bounds__(256) void k_t3(T3 d, int nimg) {
  int b = blockIdx.x, lv = 0;
  if (b >= d.cum[2]) lv = 2;
  else if (b >= d.cum[1]) lv = 1;
  b -= d.cum[lv];
  const int C = d.C[lv], HW = d.HW[lv], npx = d.npx[lv], ncy = d.ncy[lv];
  const int n = b / (npx * ncy);
  b -= n * npx * ncy;
  const int cy = b / npx, px = b - cy * npx;
  t_nchw2nhwc(d.X[lv] + (size_t)n * C * HW, d.Y[lv] + (size_t)n * HW * C, C, HW,
              px * 64, cy * 64);
}

// ---------------------------------------------------------------------------
// Dense conv (groups == 1), NHWC, fp16 in/out, fp32 accumulate, wmma.
//
//   GEMM:  Y[m, co] = sum_{t, ci} X[gather(m, t), ci] * W[co, t, ci]
//   m  = image * HW + pixel,  t = (r,s) tap (K3 ? 9 : 1)
//   W is [COUT][NTAP][CIN], contiguous along the GEMM's k.
// ---------------------------------------------------------------------------
template <int COUT, int CIN, int K3, int ACTK, int NWM, int NWN, int BM,
          int EPI, int NSTG>
__global__ __launch_bounds__(32 * NWM * NWN) void k_conv(
    const __half* __restrict__ X, const __half* __restrict__ W,
    const float* __restrict__ SC, const float* __restrict__ SH,
    __half* __restrict__ Y, int HW, int S, int xsi, int ysi,
    const __half* __restrict__ ANC, const __half* __restrict__ STR,
    const __half* __restrict__ DW, __half* __restrict__ O2, int aoff, int A) {
  constexpr int BK = (CIN % 64 == 0) ? 64 : CIN;
  constexpr int NKC = CIN / BK;
  constexpr int NTAP = K3 ? 9 : 1;
  constexpr int ITERS = NTAP * NKC;
  constexpr int NSTAGE = ITERS >= NSTG ? NSTG : ITERS;
  constexpr int LDA = BK + 8, LDB = BK + 8, LDC = COUT + 4;
  constexpr int NTHR = 32 * NWM * NWN;
  constexpr int WM = BM / NWM, WN = COUT / NWN;
  constexpr int WSTRIDE = NTAP * CIN;
  constexpr int NA = BM * LDA, NB = COUT * LDB;
  constexpr int SZ_STG = NSTAGE * (NA + NB) * (int)sizeof(__half);
  constexpr int SZ_C = BM * LDC * (int)sizeof(float);
  constexpr int OFF_META = ((SZ_STG > SZ_C ? SZ_STG : SZ_C) + 15) & ~15;
  constexpr int NLA = BM * BK / 8;          // 16B chunks of one A tile
  constexpr int NLB = COUT * BK / 8;

  extern __shared__ char sm[];
  __half* stg = (__half*)sm;
  float* Cs = (float*)sm;
  int* xoff = (int*)(sm + OFF_META);
  short* xh = (short*)(xoff + BM);
  short* xw = xh + BM;
  float* scs = (float*)(xw + BM);            // COUT
  float* shs = scs + COUT;                   // COUT
  float* pmx = shs + COUT;                   // BM*(COUT/8) partial maxima
  __half* dws = (__half*)(pmx + BM * (COUT / 8));   // 16 (EPI_DFL)

  const int tid = threadIdx.x;
  const int n = blockIdx.y;
  const int p0 = blockIdx.x * BM;
  const int xbase = n * xsi;

  for (int i = tid; i < BM; i += NTHR) {
    const int p = (p0 + i < HW) ? (p0 + i) : (HW - 1);
    const int h = p / S, w = p - h * S;
    xoff[i] = xbase + p * CIN;
    xh[i] = (short)h;
    xw[i] = (short)w;
  }
  for (int i = tid; i < COUT; i += NTHR) { scs[i] = SC[i]; shs[i] = SH[i]; }
  if (EPI == EPI_DFL)
    for (int i = tid; i < 16; i += NTHR) dws[i] = DW[i];
  __syncthreads();

  auto issue = [&](int it, int buf) {
    const int t = K3 ? (it / NKC) : 0;
    const int cc = NKC > 1 ? (it % NKC) : 0;
    const int dh = K3 ? (t / 3 - 1) : 0;
    const int dw = K3 ? (t % 3 - 1) : 0;
    __half* As = stg + buf * (NA + NB);
    __half* Bs = As + NA;
#pragma unroll
    for (int j = 0; j < (NLA + NTHR - 1) / NTHR; ++j) {
      const int idx = tid + j * NTHR;
      if ((NLA % NTHR) == 0 || idx < NLA) {
        const int r = idx / (BK / 8), seg = idx % (BK / 8);
        bool ok = true;
        if (K3) {
          const int ih = xh[r] + dh, iw = xw[r] + dw;
          ok = (ih >= 0 && ih < S && iw >= 0 && iw < S);
        }
        cp16(As + r * LDA + seg * 8,
             X + xoff[r] + (K3 ? (dh * S + dw) * CIN : 0) + cc * BK + seg * 8,
             ok);
      }
    }
#pragma unroll
    for (int j = 0; j < (NLB + NTHR - 1) / NTHR; ++j) {
      const int idx = tid + j * NTHR;
      if ((NLB % NTHR) == 0 || idx < NLB) {
        const int r = idx / (BK / 8), seg = idx % (BK / 8);
        cp16(Bs + r * LDB + seg * 8,
             W + r * WSTRIDE + t * CIN + cc * BK + seg * 8, true);
      }
    }
  };

#pragma unroll
  for (int st = 0; st < NSTAGE - 1; ++st) {
    issue(st, st);
    cp_commit();
  }

  wm::fragment<wm::accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; ++i)
#pragma unroll
    for (int j = 0; j < WN / 16; ++j) wm::fill_fragment(acc[i][j], 0.f);

  const int warp = tid >> 5;
  const int wm0 = (warp / NWN) * WM, wn0 = (warp % NWN) * WN;

  for (int it = 0; it < ITERS; ++it) {
    const int nxt = it + NSTAGE - 1;
    if (nxt < ITERS) issue(nxt, nxt % NSTAGE);
    cp_commit();
    cp_wait<NSTAGE - 1>();
    __syncthreads();
    const __half* As = stg + (it % NSTAGE) * (NA + NB);
    const __half* Bs = As + NA;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> af[WM / 16];
      wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> bf[WN / 16];
#pragma unroll
      for (int i = 0; i < WM / 16; ++i)
        wm::load_matrix_sync(af[i], As + (wm0 + i * 16) * LDA + kk * 16, LDA);
#pragma unroll
      for (int j = 0; j < WN / 16; ++j)
        wm::load_matrix_sync(bf[j], Bs + (wn0 + j * 16) * LDB + kk * 16, LDB);
#pragma unroll
      for (int i = 0; i < WM / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN / 16; ++j)
          wm::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }
  cp_wait<0>();
  __syncthreads();

#pragma unroll
  for (int i = 0; i < WM / 16; ++i)
#pragma unroll
    for (int j = 0; j < WN / 16; ++j)
      wm::store_matrix_sync(Cs + (wm0 + i * 16) * LDC + wn0 + j * 16, acc[i][j],
                            LDC, wm::mem_row_major);
  __syncthreads();

  __half* yrow = Y + n * ysi;
  if (EPI == EPI_DFL) {
    // cv2's last conv: the block holds all 4x16 distribution bins for its
    // pixels, so the DFL softmax, its arange dot and dist2bbox all happen here
    // and only the four box coordinates reach memory.  One thread per
    // (pixel, side) keeps all warps busy; the four sides of a pixel are
    // adjacent lanes, so the box assembly is a shuffle.
    for (int task = tid; task < BM * 4; task += NTHR) {
      const int row = task >> 2, j = task & 3;
      const int p = p0 + row;
      const float* crow = Cs + row * LDC + j * 16;
      const float* shj = shs + j * 16;
      float v[16], mx = -3.0e38f;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        v[k] = __half2float(__float2half(crow[k] + shj[k]));
        mx = fmaxf(mx, v[k]);
      }
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) { v[k] = __expf(v[k] - mx); sum += v[k]; }
      const float inv = 1.f / sum;
      float dot = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k)
        dot += __half2float(__float2half(v[k] * inv)) * __half2float(dws[k]);
      const float d = __half2float(__float2half(dot));
      const int a = aoff + (p < HW ? p : HW - 1);   // keep every lane in bounds
      const float an = __half2float(ANC[(j & 1) * A + a]);
      // lt sides subtract, rb sides add
      const float e = __half2float(__float2half(j < 2 ? an - d : an + d));
      const float lane0 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3));
      const float lane1 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3) | 1);
      const float lane2 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3) | 2);
      const float lane3 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3) | 3);
      if (j == 0 && p < HW) {
        const float st = __half2float(STR[a]);
        __half o[4];
        o[0] = __float2half(
            __half2float(__float2half(
                __half2float(__float2half(lane0 + lane2)) * 0.5f)) * st);
        o[1] = __float2half(
            __half2float(__float2half(
                __half2float(__float2half(lane1 + lane3)) * 0.5f)) * st);
        o[2] = __float2half(__half2float(__float2half(lane2 - lane0)) * st);
        o[3] = __float2half(__half2float(__float2half(lane3 - lane1)) * st);
        *(uint2*)(yrow + p * 4) = *(const uint2*)o;
      }
    }
    return;
  }

#pragma unroll
  for (int j = 0; j < (BM * COUT / 8 + NTHR - 1) / NTHR; ++j) {
    const int idx = tid + j * NTHR;
    if ((BM * COUT / 8) % NTHR != 0 && idx >= BM * COUT / 8) break;
    const int row = idx / (COUT / 8), g = idx % (COUT / 8);
    const int p = p0 + row;
    if (p >= HW) continue;
    __half o[8];
    float lmx = -3.0e38f;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int co = g * 8 + u;
      const float a = Cs[row * LDC + co];
      if (EPI == EPI_CLS) {
        const float lg = __half2float(__float2half(a + shs[co]));
        o[u] = __float2half(sigmoid_f(lg));
        lmx = fmaxf(lmx, __half2float(o[u]));
      } else if (ACTK == ACT_BIAS) {
        o[u] = __float2half(a + shs[co]);
      } else {
        const float c = __half2float(__float2half(a));
        const float bnv = __half2float(__float2half(scs[co] * c + shs[co]));
        o[u] = __float2half(silu_f(bnv));
      }
    }
    if (EPI == EPI_CLS) pmx[row * (COUT / 8) + g] = lmx;
    *(uint4*)(yrow + p * COUT + g * 8) = *(const uint4*)o;
  }
  if (EPI == EPI_CLS) {
    __syncthreads();
    for (int row = tid; row < BM; row += NTHR) {
      const int p = p0 + row;
      if (p >= HW) continue;
      float m = pmx[row * (COUT / 8)];
#pragma unroll
      for (int g = 1; g < COUT / 8; ++g) m = fmaxf(m, pmx[row * (COUT / 8) + g]);
      O2[n * A + aoff + p] = __float2half(m);
    }
  }
}

// ---------------------------------------------------------------------------
// Two convolutions in one kernel: (CIN -> C1, 3x3 or 1x1, BN+SiLU) followed by
// (C1 -> C2, 1x1) with a fused epilogue.
//
// A block already owns every output channel of the first conv for its pixel
// tile, so the second convolution's reduction is entirely inside the block:
// the intermediate never reaches memory and the chain loses one dependent
// launch.  Used for cv2's [3x3 64->64, 1x1 64->64] tail and cv3's
// [1x1 80->80, 1x1 80->80] tail.
// ---------------------------------------------------------------------------
template <int C1, int C2, int CIN, int K3, int EPI, int NWM1, int NWN1,
          int NWM2, int NWN2, int BM, int NSTG>
__global__ __launch_bounds__(32 * NWM1 * NWN1) void k_conv_pair(
    const __half* __restrict__ X, const __half* __restrict__ W1,
    const float* __restrict__ SC1, const float* __restrict__ SH1,
    const __half* __restrict__ W2, const float* __restrict__ SH2,
    __half* __restrict__ Y, int HW, int S, int xsi, int ysi,
    const __half* __restrict__ ANC, const __half* __restrict__ STR,
    const __half* __restrict__ DW, __half* __restrict__ O2, int aoff, int A) {
  constexpr int BK = (CIN % 64 == 0) ? 64 : CIN;
  constexpr int NKC = CIN / BK;
  constexpr int NTAP = K3 ? 9 : 1;
  constexpr int ITERS = NTAP * NKC;
  constexpr int NSTAGE = ITERS >= NSTG ? NSTG : ITERS;
  constexpr int LDA = BK + 8, LDB = BK + 8;
  constexpr int LDC1 = C1 + 4, LDC2 = C2 + 4;
  constexpr int LDA2 = C1 + 8, LDB2 = C1 + 8;
  constexpr int NTHR = 32 * NWM1 * NWN1;
  constexpr int WM1 = BM / NWM1, WN1 = C1 / NWN1;
  constexpr int WM2 = BM / NWM2, WN2 = C2 / NWN2;
  constexpr int WSTRIDE = NTAP * CIN;
  constexpr int NA = BM * LDA, NB = C1 * LDB;
  constexpr int NLA = BM * BK / 8, NLB = C1 * BK / 8;
  constexpr int SZ_STG = NSTAGE * (NA + NB) * (int)sizeof(__half);
  constexpr int SZ_C1 = BM * LDC1 * (int)sizeof(float);
  constexpr int SZ_C2 = BM * LDC2 * (int)sizeof(float);
  constexpr int SZ_R0 = (SZ_STG > SZ_C1 ? SZ_STG : SZ_C1) > SZ_C2
                            ? (SZ_STG > SZ_C1 ? SZ_STG : SZ_C1)
                            : SZ_C2;
  constexpr int OFF_A2 = (SZ_R0 + 15) & ~15;
  constexpr int OFF_B2 = OFF_A2 + BM * LDA2 * (int)sizeof(__half);
  constexpr int OFF_META =
      ((OFF_B2 + C2 * LDB2 * (int)sizeof(__half)) + 15) & ~15;

  extern __shared__ char sm[];
  __half* stg = (__half*)sm;
  float* Cs1 = (float*)sm;
  float* Cs2 = (float*)sm;
  __half* As2 = (__half*)(sm + OFF_A2);
  __half* Bs2 = (__half*)(sm + OFF_B2);
  int* xoff = (int*)(sm + OFF_META);
  short* xh = (short*)(xoff + BM);
  short* xw = xh + BM;
  float* scs = (float*)(xw + BM);             // C1
  float* shs = scs + C1;                      // C1
  float* sh2 = shs + C1;                      // C2
  float* pmx = sh2 + C2;                      // BM*(C2/8)
  __half* dws = (__half*)(pmx + BM * (C2 / 8));

  const int tid = threadIdx.x;
  const int n = blockIdx.y;
  const int p0 = blockIdx.x * BM;
  const int xbase = n * xsi;

  for (int i = tid; i < BM; i += NTHR) {
    const int p = (p0 + i < HW) ? (p0 + i) : (HW - 1);
    const int h = p / S, w = p - h * S;
    xoff[i] = xbase + p * CIN;
    xh[i] = (short)h;
    xw[i] = (short)w;
  }
  for (int i = tid; i < C1; i += NTHR) { scs[i] = SC1[i]; shs[i] = SH1[i]; }
  for (int i = tid; i < C2; i += NTHR) sh2[i] = SH2[i];
  if (EPI == EPI_DFL)
    for (int i = tid; i < 16; i += NTHR) dws[i] = DW[i];
  __syncthreads();

  auto issue = [&](int it, int buf) {
    const int t = K3 ? (it / NKC) : 0;
    const int cc = NKC > 1 ? (it % NKC) : 0;
    const int dh = K3 ? (t / 3 - 1) : 0;
    const int dw = K3 ? (t % 3 - 1) : 0;
    __half* As = stg + buf * (NA + NB);
    __half* Bs = As + NA;
#pragma unroll
    for (int j = 0; j < (NLA + NTHR - 1) / NTHR; ++j) {
      const int idx = tid + j * NTHR;
      if ((NLA % NTHR) == 0 || idx < NLA) {
        const int r = idx / (BK / 8), seg = idx % (BK / 8);
        bool ok = true;
        if (K3) {
          const int ih = xh[r] + dh, iw = xw[r] + dw;
          ok = (ih >= 0 && ih < S && iw >= 0 && iw < S);
        }
        cp16(As + r * LDA + seg * 8,
             X + xoff[r] + (K3 ? (dh * S + dw) * CIN : 0) + cc * BK + seg * 8,
             ok);
      }
    }
#pragma unroll
    for (int j = 0; j < (NLB + NTHR - 1) / NTHR; ++j) {
      const int idx = tid + j * NTHR;
      if ((NLB % NTHR) == 0 || idx < NLB) {
        const int r = idx / (BK / 8), seg = idx % (BK / 8);
        cp16(Bs + r * LDB + seg * 8,
             W1 + r * WSTRIDE + t * CIN + cc * BK + seg * 8, true);
      }
    }
  };

  // the second conv's weight rides along in the first staged group, so it is
  // resident long before the first mma finishes
  for (int idx = tid; idx < C2 * C1 / 8; idx += NTHR) {
    const int r = idx / (C1 / 8), seg = idx % (C1 / 8);
    cp16(Bs2 + r * LDB2 + seg * 8, W2 + r * C1 + seg * 8, true);
  }
#pragma unroll
  for (int st = 0; st < NSTAGE - 1; ++st) {
    issue(st, st);
    cp_commit();
  }

  wm::fragment<wm::accumulator, 16, 16, 16, float> acc[WM1 / 16][WN1 / 16];
#pragma unroll
  for (int i = 0; i < WM1 / 16; ++i)
#pragma unroll
    for (int j = 0; j < WN1 / 16; ++j) wm::fill_fragment(acc[i][j], 0.f);

  const int warp = tid >> 5;
  const int wm0 = (warp / NWN1) * WM1, wn0 = (warp % NWN1) * WN1;

  for (int it = 0; it < ITERS; ++it) {
    const int nxt = it + NSTAGE - 1;
    if (nxt < ITERS) issue(nxt, nxt % NSTAGE);
    cp_commit();
    cp_wait<NSTAGE - 1>();
    __syncthreads();
    const __half* As = stg + (it % NSTAGE) * (NA + NB);
    const __half* Bs = As + NA;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> af[WM1 / 16];
      wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> bf[WN1 / 16];
#pragma unroll
      for (int i = 0; i < WM1 / 16; ++i)
        wm::load_matrix_sync(af[i], As + (wm0 + i * 16) * LDA + kk * 16, LDA);
#pragma unroll
      for (int j = 0; j < WN1 / 16; ++j)
        wm::load_matrix_sync(bf[j], Bs + (wn0 + j * 16) * LDB + kk * 16, LDB);
#pragma unroll
      for (int i = 0; i < WM1 / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN1 / 16; ++j)
          wm::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }
  cp_wait<0>();
  __syncthreads();

#pragma unroll
  for (int i = 0; i < WM1 / 16; ++i)
#pragma unroll
    for (int j = 0; j < WN1 / 16; ++j)
      wm::store_matrix_sync(Cs1 + (wm0 + i * 16) * LDC1 + wn0 + j * 16,
                            acc[i][j], LDC1, wm::mem_row_major);
  __syncthreads();

  // BN + SiLU of conv 1, straight into the A operand of conv 2
#pragma unroll
  for (int j = 0; j < (BM * C1 / 8 + NTHR - 1) / NTHR; ++j) {
    const int idx = tid + j * NTHR;
    if ((BM * C1 / 8) % NTHR != 0 && idx >= BM * C1 / 8) break;
    const int row = idx / (C1 / 8), g = idx % (C1 / 8);
    __half o[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int co = g * 8 + u;
      const float c = __half2float(__float2half(Cs1[row * LDC1 + co]));
      const float bnv = __half2float(__float2half(scs[co] * c + shs[co]));
      o[u] = __float2half(silu_f(bnv));
    }
    *(uint4*)(As2 + row * LDA2 + g * 8) = *(const uint4*)o;
  }
  __syncthreads();

  wm::fragment<wm::accumulator, 16, 16, 16, float> ac2[WM2 / 16][WN2 / 16];
#pragma unroll
  for (int i = 0; i < WM2 / 16; ++i)
#pragma unroll
    for (int j = 0; j < WN2 / 16; ++j) wm::fill_fragment(ac2[i][j], 0.f);
  const int vm0 = (warp / NWN2) * WM2, vn0 = (warp % NWN2) * WN2;
#pragma unroll
  for (int kk = 0; kk < C1 / 16; ++kk) {
    wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> af[WM2 / 16];
    wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> bf[WN2 / 16];
#pragma unroll
    for (int i = 0; i < WM2 / 16; ++i)
      wm::load_matrix_sync(af[i], As2 + (vm0 + i * 16) * LDA2 + kk * 16, LDA2);
#pragma unroll
    for (int j = 0; j < WN2 / 16; ++j)
      wm::load_matrix_sync(bf[j], Bs2 + (vn0 + j * 16) * LDB2 + kk * 16, LDB2);
#pragma unroll
    for (int i = 0; i < WM2 / 16; ++i)
#pragma unroll
      for (int j = 0; j < WN2 / 16; ++j)
        wm::mma_sync(ac2[i][j], af[i], bf[j], ac2[i][j]);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < WM2 / 16; ++i)
#pragma unroll
    for (int j = 0; j < WN2 / 16; ++j)
      wm::store_matrix_sync(Cs2 + (vm0 + i * 16) * LDC2 + vn0 + j * 16,
                            ac2[i][j], LDC2, wm::mem_row_major);
  __syncthreads();

  __half* yrow = Y + n * ysi;
  if (EPI == EPI_DFL) {
    for (int task = tid; task < BM * 4; task += NTHR) {
      const int row = task >> 2, jj = task & 3;
      const int p = p0 + row;
      const float* crow = Cs2 + row * LDC2 + jj * 16;
      const float* shj = sh2 + jj * 16;
      float v[16], mx = -3.0e38f;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        v[k] = __half2float(__float2half(crow[k] + shj[k]));
        mx = fmaxf(mx, v[k]);
      }
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) { v[k] = __expf(v[k] - mx); sum += v[k]; }
      const float inv = 1.f / sum;
      float dot = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k)
        dot += __half2float(__float2half(v[k] * inv)) * __half2float(dws[k]);
      const float d = __half2float(__float2half(dot));
      const int a = aoff + (p < HW ? p : HW - 1);
      const float an = __half2float(ANC[(jj & 1) * A + a]);
      const float e = __half2float(__float2half(jj < 2 ? an - d : an + d));
      const float l0 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3));
      const float l1 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3) | 1);
      const float l2 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3) | 2);
      const float l3 = __shfl_sync(0xffffffffu, e, (threadIdx.x & ~3) | 3);
      if (jj == 0 && p < HW) {
        const float st = __half2float(STR[a]);
        __half o[4];
        o[0] = __float2half(
            __half2float(__float2half(__half2float(__float2half(l0 + l2)) * 0.5f)) * st);
        o[1] = __float2half(
            __half2float(__float2half(__half2float(__float2half(l1 + l3)) * 0.5f)) * st);
        o[2] = __float2half(__half2float(__float2half(l2 - l0)) * st);
        o[3] = __float2half(__half2float(__float2half(l3 - l1)) * st);
        *(uint2*)(yrow + p * 4) = *(const uint2*)o;
      }
    }
    return;
  }

#pragma unroll
  for (int j = 0; j < (BM * C2 / 8 + NTHR - 1) / NTHR; ++j) {
    const int idx = tid + j * NTHR;
    if ((BM * C2 / 8) % NTHR != 0 && idx >= BM * C2 / 8) break;
    const int row = idx / (C2 / 8), g = idx % (C2 / 8);
    const int p = p0 + row;
    if (p >= HW) continue;
    __half o[8];
    float lmx = -3.0e38f;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int co = g * 8 + u;
      const float a = Cs2[row * LDC2 + co];
      if (EPI == EPI_CLS) {
        o[u] = __float2half(sigmoid_f(__half2float(__float2half(a + sh2[co]))));
        lmx = fmaxf(lmx, __half2float(o[u]));
      } else {
        o[u] = __float2half(a + sh2[co]);
      }
    }
    if (EPI == EPI_CLS) pmx[row * (C2 / 8) + g] = lmx;
    *(uint4*)(yrow + p * C2 + g * 8) = *(const uint4*)o;
  }
  if (EPI == EPI_CLS) {
    __syncthreads();
    for (int row = tid; row < BM; row += NTHR) {
      const int p = p0 + row;
      if (p >= HW) continue;
      float m = pmx[row * (C2 / 8)];
#pragma unroll
      for (int g = 1; g < C2 / 8; ++g) m = fmaxf(m, pmx[row * (C2 / 8) + g]);
      O2[n * A + aoff + p] = __float2half(m);
    }
  }
}

// ---------------------------------------------------------------------------
// Depthwise 3x3 (stride 1, pad 1), NHWC, BN + SiLU folded in.
//   W is [9][C].
// ---------------------------------------------------------------------------
template <int C>
__global__ __launch_bounds__(256) void k_dw3(
    const __half* __restrict__ X, const __half* __restrict__ W,
    const float* __restrict__ SC, const float* __restrict__ SH,
    __half* __restrict__ Y, int HW, int S) {
  constexpr int GPP = C / 8;                 // 8-channel groups per pixel
  constexpr int PPB = 256 / GPP;             // pixels per block
  __shared__ __half ws[9 * C];
  __shared__ float scs[C], shs[C];
  const int tid = threadIdx.x;
  for (int i = tid; i < 9 * C; i += 256) ws[i] = W[i];
  for (int i = tid; i < C; i += 256) { scs[i] = SC[i]; shs[i] = SH[i]; }
  __syncthreads();

  if (tid / GPP >= PPB) return;
  const int p = blockIdx.x * PPB + tid / GPP;
  if (p >= HW) return;
  const int n = blockIdx.y;
  const int g = (tid % GPP) * 8;
  const int h = p / S, w = p - h * S;
  const __half* xin = X + (size_t)n * HW * C + (size_t)p * C + g;

  // All nine taps are loaded before any is consumed: an out-of-range tap reads
  // the centre instead of branching around the load, so the nine global loads
  // issue as one batch rather than as a chain of nine dependent round trips.
  uint4 vv[9];
  bool ok[9];
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    const int ih = h + t / 3 - 1, iw = w + t % 3 - 1;
    ok[t] = (ih >= 0 && ih < S && iw >= 0 && iw < S);
    const int off = ok[t] ? ((ih - h) * S + (iw - w)) * C : 0;
    vv[t] = *(const uint4*)(xin + off);
  }
  float a[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) a[u] = 0.f;
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    if (!ok[t]) continue;
    const __half* v = (const __half*)&vv[t];
    __half wv[8];
    *(uint4*)wv = *(const uint4*)(ws + t * C + g);
#pragma unroll
    for (int u = 0; u < 8; ++u) a[u] += __half2float(v[u]) * __half2float(wv[u]);
  }
  __half o[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const float c = __half2float(__float2half(a[u]));
    const float bnv = __half2float(__float2half(scs[g + u] * c + shs[g + u]));
    o[u] = __float2half(silu_f(bnv));
  }
  *(uint4*)(Y + (size_t)n * HW * C + (size_t)p * C + g) = *(const uint4*)o;
}

// ---------------------------------------------------------------------------
// v10 post-process, one block per image.
//
// Both top-k stages are "sort descending by value, ties by ascending index",
// which is what torch.topk(sorted=True) produces (verified against it for these
// sizes and tie densities).  That matters more than usual here: with random
// weights the class scores collapse onto a handful of fp16 values, so the
// selected 300 anchors are decided almost entirely by the tie order.
//
// Per stage: an exact two-level (high byte, then low byte) histogram of the
// order-preserving u16 key gives the k-th largest value; elements strictly above
// it are gathered and sorted, and the tie group is filled in index order.
// ---------------------------------------------------------------------------
#define POST_THR 1024
#define POST_CAP 8192
#define POST_WARPS (POST_THR / 32)

// Order-preserving u16 key for an fp16 (and back again, so the selected score
// can be recovered without a second read).
__device__ __forceinline__ unsigned short h2key_(unsigned short b) {
  return (b & 0x8000u) ? (unsigned short)(~b) : (unsigned short)(b | 0x8000u);
}
__device__ __forceinline__ __half key2h(unsigned short k) {
  return __ushort_as_half((k & 0x8000u) ? (unsigned short)(k & 0x7fffu)
                                        : (unsigned short)(~k));
}

// Warp-aggregated bump of a *per-warp private* histogram: `bin < 0` is a no-op
// lane.  Both levels of collapsing matter -- the scores here land on a handful
// of fp16 values, so a single shared histogram would serialise every lane of
// every warp onto one word.
__device__ __forceinline__ void hist_bump(int* h, int bin) {
  const unsigned int same = __match_any_sync(0xffffffffu, bin);
  if (bin >= 0 && (__ffs(same) - 1) == (threadIdx.x & 31))
    atomicAdd(&h[bin], __popc(same));
}

// Bitonic sort of `pow2` uint32 keys ascending, in shared memory.
__device__ void bitonic_u32(unsigned int* buf, int pow2) {
  for (int k = 2; k <= pow2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < pow2; i += POST_THR) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const bool up = ((i & k) == 0);
          const unsigned int a = buf[i], b = buf[ixj];
          if ((a > b) == up) { buf[i] = b; buf[ixj] = a; }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ int warp_iscan(int v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, v, d);
    if (lane >= d) v += y;
  }
  return v;
}

// Block-wide exclusive scan of one int per thread.
__device__ __forceinline__ int block_escan(int v, int* warpsum) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int x = warp_iscan(v);
  if (lane == 31) warpsum[warp] = x;
  __syncthreads();
  if (warp == 0) {
    const int t = (lane < POST_WARPS) ? warpsum[lane] : 0;
    int acc = warp_iscan(t);
    if (lane < POST_WARPS) warpsum[lane] = acc - t;
  }
  __syncthreads();
  const int mine = warpsum[warp] + x - v;
  __syncthreads();
  return mine;
}

// Reduce the per-warp histograms into `h`, then locate the 256-bin bucket that
// holds the K-th largest: the largest `b` whose suffix sum is still >= K.
// Writes {bucket, #strictly-above-bucket} to sh[0..1].
__device__ void hist_pick(int* hw, int* h, int K, int* sh) {
  const int tid = threadIdx.x;
  for (int b = tid; b < 256; b += POST_THR) {
    int acc = 0;
#pragma unroll
    for (int w = 0; w < POST_WARPS; ++w) acc += hw[w * 256 + b];
    h[b] = acc;
  }
  __syncthreads();
  if (tid < 32) {                       // one warp walks the 256 bins downward
    const int lane = tid;
    int local = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) local += h[255 - lane * 8 - j];
    const int inc = warp_iscan(local);
    int run = inc - local;              // suffix sum above this lane's 8 bins
    int fb = -1, fa = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int b = 255 - lane * 8 - j;
      const int nh = run + h[b];
      if (fb < 0 && nh >= K) { fb = b; fa = run; }
      run = nh;
    }
    int bestb = fb;
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      bestb = max(bestb, __shfl_xor_sync(0xffffffffu, bestb, d));
    if (fb == bestb) { sh[0] = fb; sh[1] = fa; }
  }
  __syncthreads();
}

// Select the top `K` of the `NV` u16 keys in `keys` under "value descending,
// ties by ascending index" -- torch.topk(sorted=True)'s rule -- and write the
// chosen indices, in that order, to `out`.
__device__ void topk_stage(const unsigned short* keys, int NV, int K, int* out,
                           unsigned int* sbuf, int* warpsum, int* sh, int* hw,
                           int* h, unsigned int* thr_out = nullptr) {
  const int tid = threadIdx.x;
  const int span = ((NV + POST_THR - 1) / POST_THR) | 1;  // contiguous, odd
  const int lo = tid * span;
  const int hi = lo + span < NV ? lo + span : NV;
  const int nit = (NV + POST_THR - 1) / POST_THR;
  int* myh = hw + (tid >> 5) * 256;

  for (int i = tid; i < POST_WARPS * 256; i += POST_THR) hw[i] = 0;
  __syncthreads();
  for (int j = 0; j < nit; ++j) {
    const int i = tid + j * POST_THR;
    const int k = (int)keys[i < NV ? i : 0];
    hist_bump(myh, i < NV ? (k >> 8) : -1);
  }
  __syncthreads();
  hist_pick(hw, h, K, sh);
  const int B = sh[0], aboveB = sh[1];

  for (int i = tid; i < POST_WARPS * 256; i += POST_THR) hw[i] = 0;
  __syncthreads();
  for (int j = 0; j < nit; ++j) {
    const int i = tid + j * POST_THR;
    const int k = (int)keys[i < NV ? i : 0];
    const bool in = (i < NV) && ((k >> 8) == B);
    hist_bump(myh, in ? (k & 0xff) : -1);
  }
  __syncthreads();
  hist_pick(hw, h, K - aboveB, sh);
  const unsigned int thr = ((unsigned)B << 8) | (unsigned)sh[0];
  const int nabove = aboveB + sh[1];
  const int nneed = K - nabove;

  // strictly above the threshold: gather, then sort by (value desc, index asc)
  if (nabove > 0) {
    int pow2 = 1;
    while (pow2 < nabove) pow2 <<= 1;
    for (int i = tid; i < pow2; i += POST_THR) sbuf[i] = 0xffffffffu;
    int c = 0;
    for (int i = lo; i < hi; ++i) c += ((unsigned)keys[i] > thr);
    __syncthreads();
    int w = block_escan(c, warpsum);
    for (int i = lo; i < hi; ++i)
      if ((unsigned)keys[i] > thr)
        sbuf[w++] = ((0xffffu ^ (unsigned)keys[i]) << 16) | (unsigned)i;
    __syncthreads();
    bitonic_u32(sbuf, pow2);
    for (int i = tid; i < nabove; i += POST_THR) out[i] = (int)(sbuf[i] & 0xffffu);
  }

  // the tie group fills the rest, in index order
  {
    int c = 0;
    for (int i = lo; i < hi; ++i) c += ((unsigned)keys[i] == thr);
    int w = block_escan(c, warpsum);
    if (w < nneed)
      for (int i = lo; i < hi && w < nneed; ++i)
        if ((unsigned)keys[i] == thr) out[nabove + w++] = i;
  }
  if (thr_out != nullptr && tid == 0) *thr_out = thr;
  __syncthreads();
}

__global__ __launch_bounds__(POST_THR) void k_post(
    const __half* __restrict__ XYWH, const __half* __restrict__ SCO,
    const __half* __restrict__ MAXS, __half* __restrict__ OUT, int A, int nc,
    int K, int POST_NKEY) {
  const int n = blockIdx.x;
  extern __shared__ char sm[];
  int* warpsum = (int*)sm;                   // POST_WARPS
  int* sh = warpsum + POST_WARPS;            // 8
  int* h = sh + 8;                           // 256
  int* hw = h + 256;                         // POST_WARPS * 256
  int* sel1 = hw + POST_WARPS * 256;         // K
  int* sel2 = sel1 + K;                      // K
  unsigned int* sbuf = (unsigned int*)(sel2 + K);          // 512
  unsigned int* thr1s = sbuf + 512;                        // 1
  unsigned short* keys = (unsigned short*)(thr1s + 4);      // max(A, K*nc)
  unsigned short* ck = keys + POST_NKEY;                    // POST_CAP
  unsigned short* ci = ck + POST_CAP;                       // POST_CAP

  for (int j = threadIdx.x; j < 2 * K; j += POST_THR) sel1[j] = 0;
  __syncthreads();
  // Every strided loop below clamps its index rather than relying on the loop
  // guard: the compiler is free to hoist the loads out of the guard, and a
  // hoisted `sel1[t / NG]` would index the tail of shared memory and turn into
  // an out-of-range global gather.
  const unsigned short* mrow = (const unsigned short*)(MAXS + (size_t)n * A);
  const int a8 = A >> 3;
  const int nit1 = (a8 + POST_THR - 1) / POST_THR;
  for (int j = 0; j < nit1; ++j) {
    const int t0 = threadIdx.x + j * POST_THR;
    const int t = t0 < a8 ? t0 : 0;
    unsigned short v[8];
    *(uint4*)v = *(const uint4*)(mrow + t * 8);
#pragma unroll
    for (int u = 0; u < 8; ++u) keys[t * 8 + u] = h2key_(v[u]);
  }
  for (int i = (a8 << 3) + threadIdx.x; i < A; i += POST_THR)
    keys[i] = h2key_(mrow[(a8 << 3) + threadIdx.x < A ? i : 0]);
  __syncthreads();
  topk_stage(keys, A, K, sel1, sbuf, warpsum, sh, hw, h, &thr1s[0]);
  const unsigned int thr1 = thr1s[0];

  const unsigned short* sbase = (const unsigned short*)(SCO + (size_t)n * A * nc);
  const int NG = nc >> 3;
  const int NV2 = K * nc;
  const int nit2 = (K * NG + POST_THR - 1) / POST_THR;
  for (int j = 0; j < nit2; ++j) {
    const int t0 = threadIdx.x + j * POST_THR;
    const int t = t0 < K * NG ? t0 : 0;
    const int i = t / NG, g = t - i * NG;
    unsigned short v[8];
    *(uint4*)v = *(const uint4*)(sbase + (size_t)sel1[i] * nc + g * 8);
    unsigned short* dst = keys + i * nc + g * 8;
#pragma unroll
    for (int u = 0; u < 8; ++u) dst[u] = h2key_(v[u]);
  }
  __syncthreads();

  // The k-th largest of the gathered scores is never below the stage-1
  // threshold -- the 300 selected anchors contribute 300 maxima that are all
  // >= it -- so everything below can be dropped before the second selection.
  // That usually leaves a few hundred candidates instead of K*nc.
  {
    const int spanf = ((NV2 + POST_THR - 1) / POST_THR) | 1;
    const int lo = threadIdx.x * spanf;
    const int hi = lo + spanf < NV2 ? lo + spanf : NV2;
    int c = 0;
    for (int i = lo; i < hi; ++i) c += ((unsigned)keys[i] >= thr1);
    int w = block_escan(c, warpsum);
    if (threadIdx.x == POST_THR - 1) sh[7] = w + c;
    for (int i = lo; i < hi && w < POST_CAP; ++i)
      if ((unsigned)keys[i] >= thr1) {
        ck[w] = keys[i];
        ci[w] = (unsigned short)i;
        ++w;
      }
    __syncthreads();
  }
  const int m = sh[7];
  if (m >= K && m <= POST_CAP) {
    topk_stage(ck, m, K, sel2, sbuf, warpsum, sh, hw, h);
    for (int j = threadIdx.x; j < K; j += POST_THR) {
      const int q = sel2[j];
      sel2[j] = ci[(unsigned)q < (unsigned)m ? q : 0];
    }
    __syncthreads();
  } else {
    topk_stage(keys, NV2, K, sel2, sbuf, warpsum, sh, hw, h);
  }
  for (int j0 = threadIdx.x; j0 < K; j0 += POST_THR) {
    const int j = j0 < K ? j0 : 0;
    const int f0 = sel2[j];
    const int f = (unsigned)f0 < (unsigned)NV2 ? f0 : 0;
    const int i = f / nc, c = f - i * nc;
    const int a = sel1[i];
    __half bx[4];
    *(uint2*)bx = *(const uint2*)(XYWH + ((size_t)n * A + a) * 4);
    const float x = __half2float(bx[0]), y = __half2float(bx[1]);
    const float w = __half2float(bx[2]), h4 = __half2float(bx[3]);
    const float hw2 = __half2float(__float2half(w * 0.5f));
    const float hh = __half2float(__float2half(h4 * 0.5f));
    __half* o = OUT + ((size_t)n * K + j0) * 6;
    o[0] = __float2half(x - hw2);
    o[1] = __float2half(y - hh);
    o[2] = __float2half(x + hw2);
    o[3] = __float2half(y + hh);
    o[4] = key2h(keys[f]);
    o[5] = __float2half((float)c);
  }
}

// ###########################################################################
// host side
// ###########################################################################
//
// plan[] (int64, CPU) -- built once on the python side:
//   per level lv (stride 40):
//     0..3   C, S, HW, anchor offset
//     4..11  packed-weight offsets (halves) for
//            [A1 3x3 C->64, A2 3x3 64->64, A3 1x1 64->64,
//             B1 dw3x3 C,   B2 1x1 C->80,  B3 dw3x3 80,
//             B4 1x1 80->80, B5 1x1 80->80]
//    12..19  per-channel scale offsets (floats)
//    20..27  per-channel shift offsets (floats)
//    28..33  workspace offsets (halves): xn t1 t2 u1 u2 u3
//   122..124 workspace offsets: XYWH SCO MAXS

namespace {

constexpr int LSTRIDE = 40;

struct Streams {
  cudaStream_t s[6];
  cudaStream_t cap;
  cudaEvent_t ev[7];
  bool init = false;
};
Streams g_st;

void ensure_streams() {
  if (g_st.init) return;
  for (int i = 0; i < 6; ++i)
    cudaStreamCreateWithFlags(&g_st.s[i], cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&g_st.cap, cudaStreamNonBlocking);
  for (int i = 0; i < 7; ++i)
    cudaEventCreateWithFlags(&g_st.ev[i], cudaEventDisableTiming);
  g_st.init = true;
}


template <int COUT, int CIN, int K3, int ACTK, int EPI, int NSTG, int BM>
void launch_conv_impl(const __half* X, const __half* W, const float* SC,
                      const float* SH, __half* Y, int HW, int S, int nimg,
                      int xsi, int ysi, cudaStream_t st, const __half* ANC,
                      const __half* STR, const __half* DW, __half* O2, int aoff,
                      int A) {
  constexpr int NWM = (COUT % 32 == 0) ? (BM / 32) : (BM / 16);
  constexpr int NWN = (COUT % 32 == 0) ? 2 : 1;
  constexpr int BK = (CIN % 64 == 0) ? 64 : CIN;
  constexpr int NKC = CIN / BK;
  constexpr int ITERS = (K3 ? 9 : 1) * NKC;
  constexpr int NSTAGE = ITERS >= NSTG ? NSTG : ITERS;
  constexpr int LDA = BK + 8, LDB = BK + 8, LDC = COUT + 4;
  constexpr int SZ_STG = NSTAGE * (BM * LDA + COUT * LDB) * (int)sizeof(__half);
  constexpr int SZ_C = BM * LDC * (int)sizeof(float);
  constexpr int OFF_META = ((SZ_STG > SZ_C ? SZ_STG : SZ_C) + 15) & ~15;
  constexpr int SMEM =
      OFF_META + BM * 8 + COUT * 8 + BM * (COUT / 8) * 4 + 32;
  auto fn = k_conv<COUT, CIN, K3, ACTK, NWM, NWN, BM, EPI, NSTG>;
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM);
    done = true;
  }
  fn<<<dim3(DIVUP(HW, BM), nimg), 32 * NWM * NWN, SMEM, st>>>(
      X, W, SC, SH, Y, HW, S, xsi, ysi, ANC, STR, DW, O2, aoff, A);
}

// The winning shape for every conv here: a 64-pixel tile with a four-deep k
// pipeline.  These kernels are latency-bound, not throughput-bound (a few
// hundred blocks at ~12% warp occupancy), so halving the tile to double the
// resident blocks and deepen the prefetch beat the larger tile measurably.
template <int COUT, int CIN, int K3, int ACTK, int EPI = EPI_PLAIN>
void launch_conv(const __half* X, const __half* W, const float* SC,
                 const float* SH, __half* Y, int HW, int S, int nimg, int xsi,
                 int ysi, cudaStream_t st, const __half* ANC = nullptr,
                 const __half* STR = nullptr, const __half* DW = nullptr,
                 __half* O2 = nullptr, int aoff = 0, int A = 0) {
  launch_conv_impl<COUT, CIN, K3, ACTK, EPI, 4, 64>(
      X, W, SC, SH, Y, HW, S, nimg, xsi, ysi, st, ANC, STR, DW, O2, aoff, A);
}

template <int C1, int C2, int CIN, int K3, int EPI, int NSTG>
void launch_pair_impl(const __half* X, const __half* W1, const float* SC1,
                      const float* SH1, const __half* W2, const float* SH2,
                      __half* Y, int HW, int S, int nimg, int xsi, int ysi,
                      cudaStream_t st, const __half* ANC, const __half* STR,
                      const __half* DW, __half* O2, int aoff, int A) {
  constexpr int BM = 128;
  constexpr int NWM1 = (C1 % 32 == 0) ? 4 : 8, NWN1 = (C1 % 32 == 0) ? 2 : 1;
  constexpr int NWM2 = (C2 % 32 == 0) ? 4 : 8, NWN2 = (C2 % 32 == 0) ? 2 : 1;
  constexpr int BK = (CIN % 64 == 0) ? 64 : CIN;
  constexpr int NKC = CIN / BK;
  constexpr int ITERS = (K3 ? 9 : 1) * NKC;
  constexpr int NSTAGE = ITERS >= NSTG ? NSTG : ITERS;
  constexpr int LDA = BK + 8, LDB = BK + 8;
  constexpr int LDC1 = C1 + 4, LDC2 = C2 + 4, LDA2 = C1 + 8, LDB2 = C1 + 8;
  constexpr int SZ_STG = NSTAGE * (BM * LDA + C1 * LDB) * (int)sizeof(__half);
  constexpr int SZ_C1 = BM * LDC1 * (int)sizeof(float);
  constexpr int SZ_C2 = BM * LDC2 * (int)sizeof(float);
  constexpr int M0 = SZ_STG > SZ_C1 ? SZ_STG : SZ_C1;
  constexpr int SZ_R0 = M0 > SZ_C2 ? M0 : SZ_C2;
  constexpr int OFF_A2 = (SZ_R0 + 15) & ~15;
  constexpr int OFF_B2 = OFF_A2 + BM * LDA2 * (int)sizeof(__half);
  constexpr int OFF_META =
      ((OFF_B2 + C2 * LDB2 * (int)sizeof(__half)) + 15) & ~15;
  constexpr int SMEM = OFF_META + BM * 8 + (2 * C1 + C2) * 4 +
                       BM * (C2 / 8) * 4 + 32;
  static_assert(NWM1 * NWN1 == NWM2 * NWN2, "warp count must match");
  auto fn = k_conv_pair<C1, C2, CIN, K3, EPI, NWM1, NWN1, NWM2, NWN2, BM, NSTG>;
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM);
    done = true;
  }
  fn<<<dim3(DIVUP(HW, BM), nimg), 32 * NWM1 * NWN1, SMEM, st>>>(
      X, W1, SC1, SH1, W2, SH2, Y, HW, S, xsi, ysi, ANC, STR, DW, O2, aoff, A);
}

template <int C1, int C2, int CIN, int K3, int EPI>
void launch_pair(const __half* X, const __half* W1, const float* SC1,
                 const float* SH1, const __half* W2, const float* SH2,
                 __half* Y, int HW, int S, int nimg, int xsi, int ysi,
                 cudaStream_t st, const __half* ANC = nullptr,
                 const __half* STR = nullptr, const __half* DW = nullptr,
                 __half* O2 = nullptr, int aoff = 0, int A = 0) {
  launch_pair_impl<C1, C2, CIN, K3, EPI, 4>(X, W1, SC1, SH1, W2, SH2, Y, HW, S,
                                            nimg, xsi, ysi, st, ANC, STR, DW,
                                            O2, aoff, A);
}

template <int C>
void launch_dw(const __half* X, const __half* W, const float* SC,
               const float* SH, __half* Y, int HW, int S, int nimg,
               cudaStream_t st) {
  constexpr int PPB = 256 / (C / 8);
  k_dw3<C><<<dim3(DIVUP(HW, PPB), nimg), 256, 0, st>>>(X, W, SC, SH, Y, HW, S);
}

}  // namespace

// The conv chains are replayed from a captured graph: the head is ~24 tiny
// kernels whose combined launch cost (~5us each on the host) exceeded their GPU
// time.  Only the NCHW->NHWC transpose (it reads the caller's tensors, whose
// addresses move every iteration) and the post-process (it writes the freshly
// allocated output) stay outside, so a call costs three launches.
namespace {

struct GraphCache {
  cudaGraphExec_t exec = nullptr;
  const void* ws = nullptr;
  const void* plan = nullptr;
  int64_t nimg = -1;
};
GraphCache g_gc;

void enqueue_convs(cudaStream_t root, const int64_t* pl, const __half* WB,
                   const float* PB, __half* W0, const __half* ANC,
                   const __half* STR, const __half* DWp, __half* XYWH,
                   __half* SCO, __half* MAXS, int64_t nimg, int64_t A,
                   int64_t nc) {
  cudaEventRecord(g_st.ev[6], root);
  for (int i = 0; i < 6; ++i) cudaStreamWaitEvent(g_st.s[i], g_st.ev[6], 0);

  for (int lv = 0; lv < 3; ++lv) {
    const int64_t* L = pl + lv * LSTRIDE;
    const int C = (int)L[0], S = (int)L[1], HW = (int)L[2], aoff = (int)L[3];
    __half* xn = W0 + L[28];
    __half* t1 = W0 + L[29];
    __half* t2 = W0 + L[30];
    __half* u1 = W0 + L[31];
    __half* u2 = W0 + L[32];
    __half* u3 = W0 + L[33];
    cudaStream_t sa = g_st.s[2 * lv], sb = g_st.s[2 * lv + 1];
    const int xsi = HW * C;
    const int ni = (int)nimg;
#define WOF(j) (WB + L[4 + (j)])
#define SCOF(j) (PB + L[12 + (j)])
#define SHOF(j) (PB + L[20 + (j)])
    // ---- cv2: 3x3 C->64, 3x3 64->64, 1x1 64->64 (+bias, DFL epilogue) ----
    {
      if (C == 64)
        launch_conv<64, 64, 1, ACT_SILU>(xn, WOF(0), SCOF(0), SHOF(0), t1, HW, S,
                                         ni, xsi, HW * 64, sa);
      else if (C == 128)
        launch_conv<64, 128, 1, ACT_SILU>(xn, WOF(0), SCOF(0), SHOF(0), t1, HW,
                                          S, ni, xsi, HW * 64, sa);
      else
        launch_conv<64, 256, 1, ACT_SILU>(xn, WOF(0), SCOF(0), SHOF(0), t1, HW,
                                          S, ni, xsi, HW * 64, sa);
      launch_pair<64, 64, 64, 1, EPI_DFL>(
          t1, WOF(1), SCOF(1), SHOF(1), WOF(2), SHOF(2),
          XYWH + (size_t)aoff * 4, HW, S, ni, HW * 64, (int)(A * 4), sa, ANC,
          STR, DWp, nullptr, aoff, (int)A);
    }
    // ---- cv3: dw3x3, 1x1 C->80, dw3x3, 1x1, 1x1 (+bias, sigmoid/max) ----
    {
      if (C == 64) {
        launch_dw<64>(xn, WOF(3), SCOF(3), SHOF(3), u1, HW, S, ni, sb);
        launch_conv<80, 64, 0, ACT_SILU>(u1, WOF(4), SCOF(4), SHOF(4), u2, HW, S,
                                         ni, xsi, HW * 80, sb);
      } else if (C == 128) {
        launch_dw<128>(xn, WOF(3), SCOF(3), SHOF(3), u1, HW, S, ni, sb);
        launch_conv<80, 128, 0, ACT_SILU>(u1, WOF(4), SCOF(4), SHOF(4), u2, HW,
                                          S, ni, xsi, HW * 80, sb);
      } else {
        launch_dw<256>(xn, WOF(3), SCOF(3), SHOF(3), u1, HW, S, ni, sb);
        launch_conv<80, 256, 0, ACT_SILU>(u1, WOF(4), SCOF(4), SHOF(4), u2, HW,
                                          S, ni, xsi, HW * 80, sb);
      }
      launch_dw<80>(u2, WOF(5), SCOF(5), SHOF(5), u3, HW, S, ni, sb);
      launch_pair<80, 80, 80, 0, EPI_CLS>(
          u3, WOF(6), SCOF(6), SHOF(6), WOF(7), SHOF(7),
          SCO + (size_t)aoff * nc, HW, S, ni, HW * 80, (int)(A * nc), sb,
          nullptr, nullptr, nullptr, MAXS, aoff, (int)A);
    }
#undef WOF
#undef SCOF
#undef SHOF
  }
  for (int i = 0; i < 6; ++i) {
    cudaEventRecord(g_st.ev[i], g_st.s[i]);
    cudaStreamWaitEvent(root, g_st.ev[i], 0);
  }
}

}  // namespace

at::Tensor head_forward(const at::Tensor& x0, const at::Tensor& x1,
                        const at::Tensor& x2, const at::Tensor& wbuf,
                        const at::Tensor& pbuf, const at::Tensor& anchors,
                        const at::Tensor& strides, const at::Tensor& dflw,
                        const at::Tensor& plan, const at::Tensor& ws,
                        int64_t nimg, int64_t A, int64_t nc, int64_t K) {
  ensure_streams();
  const int64_t* pl = plan.data_ptr<int64_t>();
  const __half* WB = (const __half*)wbuf.data_ptr();
  const float* PB = pbuf.data_ptr<float>();
  __half* W0 = (__half*)ws.data_ptr();
  __half* XYWH = W0 + pl[122];
  __half* SCO = W0 + pl[123];
  __half* MAXS = W0 + pl[124];
  const __half* ANC = (const __half*)anchors.data_ptr();
  const __half* STR = (const __half*)strides.data_ptr();
  const __half* DWp = (const __half*)dflw.data_ptr();

  auto out = at::empty({nimg, K, 6}, x0.options());
  cudaStream_t main = at::cuda::getCurrentCUDAStream();

  // ---- NCHW -> NHWC for all three levels (reads the caller's tensors) ----
  T3 td;
  const at::Tensor* xs[3] = {&x0, &x1, &x2};
  td.cum[0] = 0;
  for (int lv = 0; lv < 3; ++lv) {
    const int64_t* L = pl + lv * LSTRIDE;
    td.X[lv] = (const __half*)xs[lv]->data_ptr();
    td.Y[lv] = W0 + L[28];
    td.C[lv] = (int)L[0];
    td.HW[lv] = (int)L[2];
    td.npx[lv] = (int)DIVUP(L[2], 64);
    td.ncy[lv] = (int)DIVUP(L[0], 64);
    td.cum[lv + 1] = td.cum[lv] + td.npx[lv] * td.ncy[lv] * (int)nimg;
  }
  k_t3<<<td.cum[3], 256, 0, main>>>(td, (int)nimg);

  // ---- conv chains: captured once, replayed thereafter ----
  if (g_gc.exec == nullptr || g_gc.ws != (const void*)W0 ||
      g_gc.plan != (const void*)pl || g_gc.nimg != nimg) {
    if (g_gc.exec) {
      cudaGraphExecDestroy(g_gc.exec);
      g_gc.exec = nullptr;
    }
    // one eager pass first: it produces this call's result and forces every
    // one-time cudaFuncSetAttribute before the capture sees it
    enqueue_convs(main, pl, WB, PB, W0, ANC, STR, DWp, XYWH, SCO, MAXS, nimg, A,
                  nc);
    cudaStreamSynchronize(main);
    for (int i = 0; i < 6; ++i) cudaStreamSynchronize(g_st.s[i]);
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(g_st.cap, cudaStreamCaptureModeRelaxed) ==
        cudaSuccess) {
      enqueue_convs(g_st.cap, pl, WB, PB, W0, ANC, STR, DWp, XYWH, SCO, MAXS,
                    nimg, A, nc);
      if (cudaStreamEndCapture(g_st.cap, &graph) == cudaSuccess && graph) {
        cudaGraphExec_t exec = nullptr;
        if (cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess) {
          g_gc.exec = exec;
          g_gc.ws = (const void*)W0;
          g_gc.plan = (const void*)pl;
          g_gc.nimg = nimg;
        }
        cudaGraphDestroy(graph);
      }
    }
  } else {
    cudaGraphLaunch(g_gc.exec, main);
  }

  const int64_t nkey = (A > K * nc) ? A : K * nc;
  const int smem_post =
      (int)((POST_WARPS + 256 + POST_WARPS * 256 + 8) * 4 + 2 * K * 4 +
            (512 + 4) * 4 + (nkey + 2 * POST_CAP) * 2 + 64);
  static bool post_done = false;
  if (!post_done) {
    cudaFuncSetAttribute(k_post, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_post);
    post_done = true;
  }
  k_post<<<(unsigned)nimg, POST_THR, smem_post, main>>>(
      XYWH, SCO, MAXS, (__half*)out.data_ptr(), (int)A, (int)nc, (int)K,
      (int)nkey);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_forward", &head_forward, "YOLOv10 detection head (fused)");
}
