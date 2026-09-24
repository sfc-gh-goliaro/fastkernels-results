"""YOLOv10 bottleneck block -- fused tensor-core CUDA implementation.

The captured workload is always ``YOLOBottleneck(c, c, shortcut, 1, k=(3, 3),
e=1.0)`` on fp16 NCHW activations: two 3x3 same-channel convolutions, each
followed by BatchNorm (eval) + SiLU, plus an optional residual add.  Every
captured shape is tiny (0.2 - 3 MB of activations, ~0.5 GFLOP), so the baseline
-- ~13 kernels of nchw<->nhwc transposes, cudnn implicit GEMM, batch_norm, silu
and add -- spends nearly all of its time on launch overhead.

Both convolutions therefore run as *one* kernel launch: a cooperative grid
whose two phases are separated by a grid-wide barrier.  Each phase is an
implicit-GEMM (``mma.m16n8k16``, fp32 accumulate) over M = pixels,
N = out channels, K = 9 * C, with the BatchNorm affine folded into the epilogue
scale/bias and SiLU (+ residual) fused in.

Layout notes
------------
* A block owns a ``TR x (16*FW)`` pixel tile of one image and ``NT`` output
  channels.  The input patch (tile + 1 pixel halo, zero padded) is staged to
  shared memory *pixel-major* (``Xs[pixel][ci]``), which makes every 3x3 tap a
  plain pixel-index offset, so each A fragment is a single ``ldmatrix``.
* The weights are pre-reordered once to ``[9][cout][cin]`` so a tap's slice is
  contiguous, and are staged with ``cp.async`` and read as mma B fragments.
  Both the patch and the weight strides are padded by 8 halves, which is what
  keeps the fragment loads bank-conflict free.
* Activations may arrive strided (captured tensors are slices of a larger
  buffer), so the input strides are passed to the kernel rather than assumed.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .yolov10_conv import YOLOConv

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <type_traits>

#define DEVI static __device__ __forceinline__

DEVI unsigned smem_u32(const void *p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

DEVI void ldsm4(unsigned *a, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

DEVI void mma16816(float *d, const unsigned *a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

DEVI void cp_async16(unsigned dst, const void *src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(dst),
               "l"(src));
}
DEVI void cp_async_fence() { asm volatile("cp.async.commit_group;\n"); }
DEVI void cp_async_wait() { asm volatile("cp.async.wait_group 0;\n"); }

DEVI float silu(float v) { return v / (1.0f + __expf(-v)); }

// One 3x3 conv (C->C, stride 1, pad 1) + per-channel affine + SiLU (+ residual)
// as an implicit GEMM over M = pixels, N = out channels, K = 9 * C.
//   C  : channel count            TR : tile rows
//   FW : tile width / 16          NT : output channels per block
//   WM : warps along pixels       WN : warps along channels
//   VECW: halves per vector memory access (8 or 4)
template <int C, int TR, int FW, int NT, int WM, int WN, int VECW, bool ADD_RES>
__device__ __forceinline__ void conv_block(
    const __half *__restrict__ X, const __half *__restrict__ WT,
    const float *__restrict__ SCALE, const float *__restrict__ BIAS,
    const __half *__restrict__ RES, __half *__restrict__ OUT, int H, int W,
    int tiles_x, bool vec_stage, int sN, int sC, int sH, int xlim, int rsN,
    int rsC, int rsH) {
  using VT = typename std::conditional<VECW == 8, uint4, uint2>::type;
  constexpr int THREADS = 32 * WM * WN;
  constexpr int TW = 16 * FW;         // tile width in pixels
  constexpr int MT = TR * TW;         // output pixels per block
  constexpr int PW = TW + 2;          // padded patch row width
  constexpr int PR = TR + 2;          // patch rows
  constexpr int LDK = C + 8;          // shared patch channel stride
  constexpr int LDB = C + 8;          // shared weight channel stride
  constexpr int MF = TR * FW;         // m-fragments in the tile
  constexpr int NFT = NT / 8;         // n-fragments in the tile
  constexpr int MW = MF / WM;         // m-fragments per warp
  constexpr int NW = NFT / WN;        // n-fragments per warp
  constexpr int KS = C / 16;          // k-steps per tap
  constexpr int LDY = MT + 8;         // epilogue staging stride
  constexpr int XH = PR * PW * LDK;
  constexpr int YH = NT * LDY;
  constexpr int R1 = XH > YH ? XH : YH;
  constexpr int NV = NT * (C / 8);                  // uint4 per tap of B
  constexpr int NBI = (9 * NV + THREADS - 1) / THREADS;
  constexpr int NJ = (C * PW + THREADS - 1) / THREADS;
  constexpr int WL = ((TW + 2 * VECW) / VECW) * VECW;   // aligned load window
  constexpr int NCH = WL / VECW;
  constexpr int NPAIR = C * PR;
  constexpr int NPIT = (NPAIR + THREADS - 1) / THREADS;
  static_assert(MF % WM == 0 && NFT % WN == 0, "bad warp split");
  static_assert(WL >= TW + VECW + 1, "load window too small");

  extern __shared__ __align__(16) char smem[];
  __half *Xs = reinterpret_cast<__half *>(smem);
  __half *Ys = Xs;
  __half *Bs = reinterpret_cast<__half *>(smem + R1 * (int)sizeof(__half));

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp % WM;
  const int wn = warp / WM;

  const int tile = blockIdx.x;
  const int tx = tile % tiles_x;
  const int ty = tile / tiles_x;
  const int r0 = ty * TR;
  const int c0 = tx * TW;
  const int n0 = blockIdx.y * NT;
  const int HW = H * W;
  const __half *xb = X + (size_t)blockIdx.z * sN;
  const int xoff0 = blockIdx.z * sN;   // for the in-bounds guard on vector loads

  // ---- stage the weights (async): Bs[t][nn][ci] --------------------------
  {
    const uint4 *wsrc = reinterpret_cast<const uint4 *>(WT + (size_t)n0 * C);
    const unsigned bs0 = smem_u32(Bs);
    constexpr int CV = C / 8;
#pragma unroll
    for (int it = 0; it < NBI; ++it) {
      const int i = tid + it * THREADS;
      if (NBI * THREADS == 9 * NV || i < 9 * NV) {
        const int t = i / NV;
        const int r = i - t * NV;
        const int nn = r / CV;
        const int ch = r - nn * CV;
        cp_async16(bs0 + (unsigned)(((t * NT + nn) * LDB + ch * 8) * 2),
                   wsrc + (size_t)t * (C * CV) + nn * CV + ch);
      }
    }
    cp_async_fence();
  }

  // ---- stage the input patch: Xs[(r * PW + c)][ci] ------------------------
  if (vec_stage) {
    // Aligned window loads: a thread owns (ci, row) pairs with `ci` lane
    // consecutive, so the transposing shared scatter is conflict free.
    const __half zero = __float2half(0.0f);
    VT buf[NPIT][NCH];
#pragma unroll
    for (int it = 0; it < NPIT; ++it) {
      const int pidx = tid + it * THREADS;
      const int r = pidx / C;
      const int ci = pidx - r * C;
      const int gr = r0 - 1 + r;
      const bool rok = (unsigned)gr < (unsigned)H && pidx < NPAIR;
      const int o = ci * sC + gr * sH + c0 - VECW;
#pragma unroll
      for (int q = 0; q < NCH; ++q) {
        const int ad = o + q * VECW;
        if (rok && xoff0 + ad >= 0 && xoff0 + ad + VECW <= xlim)
          buf[it][q] = *reinterpret_cast<const VT *>(xb + ad);
        else
          memset(&buf[it][q], 0, sizeof(VT));
      }
    }
#pragma unroll
    for (int it = 0; it < NPIT; ++it) {
      const int pidx = tid + it * THREADS;
      if (NPIT * THREADS != NPAIR && pidx >= NPAIR) break;
      const int r = pidx / C;
      const int ci = pidx - r * C;
      const __half *v = reinterpret_cast<const __half *>(&buf[it][0]);
      __half *dst = Xs + r * PW * LDK + ci;
#pragma unroll
      for (int j = 0; j < PW; ++j) {
        const int gc = c0 - 1 + j;
        dst[j * LDK] = ((unsigned)gc < (unsigned)W) ? v[j + VECW - 1] : zero;
      }
    }
  } else {
    const __half zero = __float2half(0.0f);
    __half v[NJ][PR];
#pragma unroll
    for (int it = 0; it < NJ; ++it) {
      const int i = tid + it * THREADS;
      const int ci = i / PW;
      const int c = i - ci * PW;
      const int gc = c0 - 1 + c;
      const bool cok = (unsigned)gc < (unsigned)W && i < C * PW;
      const __half *src = xb + (size_t)ci * sC + (size_t)(r0 - 1) * sH + gc;
#pragma unroll
      for (int r = 0; r < PR; ++r) {
        const int gr = r0 - 1 + r;
        v[it][r] =
            (cok && (unsigned)gr < (unsigned)H) ? src[(size_t)r * sH] : zero;
      }
    }
#pragma unroll
    for (int it = 0; it < NJ; ++it) {
      const int i = tid + it * THREADS;
      if (NJ * THREADS == C * PW || i < C * PW) {
        const int ci = i / PW;
        const int c = i - ci * PW;
        __half *dst = Xs + c * LDK + ci;
#pragma unroll
        for (int r = 0; r < PR; ++r) dst[r * PW * LDK] = v[it][r];
      }
    }
  }

  // Epilogue addressing + residual prefetch, issued before the math so the
  // residual read latency overlaps the mma work.
  constexpr int CB = TW / VECW;
  constexpr int NBLK = NT * TR * CB;
  constexpr int NE = (NBLK + THREADS - 1) / THREADS;
  const size_t obase = (size_t)blockIdx.z * C * HW + (size_t)n0 * HW;
  const size_t rbase = (size_t)blockIdx.z * rsN + (size_t)n0 * rsC;
  int eb[NE];
  bool evec[NE];
  VT rbuf[NE];
#pragma unroll
  for (int e = 0; e < NE; ++e) {
    eb[e] = tid + e * THREADS;
    const int b = eb[e] < NBLK ? eb[e] : 0;
    const int cb = b % CB;
    const int rest = b / CB;
    const int tr = rest % TR;
    const int n = rest / TR;
    const int gr = r0 + tr;
    const int gc = c0 + cb * VECW;
    const size_t ro = rbase + (size_t)n * rsC + (size_t)gr * rsH + gc;
    evec[e] = eb[e] < NBLK && gr < H && gc + VECW <= W &&
              ((ro & (VECW - 1)) == 0);
    if (ADD_RES && evec[e])
      rbuf[e] = *reinterpret_cast<const VT *>(RES + ro);
    else
      memset(&rbuf[e], 0, sizeof(VT));
  }

  cp_async_wait();
  __syncthreads();

  // ---- main loop ---------------------------------------------------------
  float acc[MW][NW][4];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

  const unsigned abase =
      smem_u32(Xs + (size_t)(lane & 15) * LDK + (lane >> 4) * 8);
  const unsigned bbase =
      smem_u32(Bs + (size_t)(lane >> 2) * LDB + (lane & 3) * 2);

#pragma unroll
  for (int t = 0; t < 9; ++t) {
    constexpr int dyv[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    constexpr int dxv[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
#pragma unroll
    for (int k = 0; k < KS; ++k) {
      unsigned a[MW][4];
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        const int mf = wm * MW + i;
        const int tr = mf / FW;
        const int f = mf - tr * FW;
        const int pix = (tr + 1 + dyv[t]) * PW + f * 16 + dxv[t] + 1;
        ldsm4(a[i], abase + (unsigned)((pix * LDK + k * 16) * 2));
      }
      unsigned b0[NW], b1[NW];
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const unsigned off =
            (unsigned)(((t * NT + (wn * NW + j) * 8) * LDB + k * 16) * 2);
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(b0[j]) : "r"(bbase + off));
        asm volatile("ld.shared.b32 %0, [%1];\n"
                     : "=r"(b1[j])
                     : "r"(bbase + off + 16));
      }
#pragma unroll
      for (int j = 0; j < NW; ++j)
#pragma unroll
        for (int i = 0; i < MW; ++i) mma16816(acc[i][j], a[i], b0[j], b1[j]);
    }
  }

  // ---- epilogue: affine + SiLU into shared, then coalesced store ---------
  __syncthreads();
  {
    const int mrow = lane >> 2;
    const int nsub = (lane & 3) * 2;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const int n = (wn * NW + j) * 8 + nsub;
      const float s0 = SCALE[n0 + n], c0f = BIAS[n0 + n];
      const float s1 = SCALE[n0 + n + 1], c1f = BIAS[n0 + n + 1];
#pragma unroll
      for (int i = 0; i < MW; ++i) {
        const int m = (wm * MW + i) * 16 + mrow;
        Ys[n * LDY + m] = __float2half(silu(acc[i][j][0] * s0 + c0f));
        Ys[(n + 1) * LDY + m] = __float2half(silu(acc[i][j][1] * s1 + c1f));
        Ys[n * LDY + m + 8] = __float2half(silu(acc[i][j][2] * s0 + c0f));
        Ys[(n + 1) * LDY + m + 8] = __float2half(silu(acc[i][j][3] * s1 + c1f));
      }
    }
  }
  __syncthreads();
  {
#pragma unroll
    for (int e = 0; e < NE; ++e) {
      if (NE * THREADS != NBLK && eb[e] >= NBLK) break;
      const int cb = eb[e] % CB;
      const int rest = eb[e] / CB;
      const int tr = rest % TR;
      const int n = rest / TR;
      const int gr = r0 + tr;
      if (gr >= H) continue;
      const int col = cb * VECW;
      const int gc = c0 + col;
      const int m = tr * TW + col;
      const __half *ys = Ys + n * LDY + m;
      const size_t o = obase + (size_t)n * HW + (size_t)gr * W + gc;
      if (evec[e]) {
        VT val = *reinterpret_cast<const VT *>(ys);
        if (ADD_RES) {
          __half2 *aa = reinterpret_cast<__half2 *>(&val);
          const __half2 *bb = reinterpret_cast<const __half2 *>(&rbuf[e]);
#pragma unroll
          for (int q = 0; q < VECW / 2; ++q) aa[q] = __hadd2(aa[q], bb[q]);
        }
        *reinterpret_cast<VT *>(OUT + o) = val;
      } else {
        const size_t ro = rbase + (size_t)n * rsC + (size_t)gr * rsH + gc;
#pragma unroll
        for (int q = 0; q < VECW; ++q) {
          if (gc + q >= W) break;
          float fv = __half2float(ys[q]);
          if (ADD_RES) fv += __half2float(RES[ro + q]);
          OUT[o + q] = __float2half(fv);
        }
      }
    }
  }
}

struct Strides {
  int sN, sC, sH, lim;
};

// Single-conv kernel (fallback when a cooperative launch is not possible).
template <int C, int TR, int FW, int NT, int WM, int WN, int VECW, bool ADD_RES>
__global__ __launch_bounds__(32 * WM * WN) void conv3x3_kernel(
    const __half *__restrict__ X, const __half *__restrict__ WT,
    const float *__restrict__ SCALE, const float *__restrict__ BIAS,
    const __half *__restrict__ RES, __half *__restrict__ OUT, int H, int W,
    int tiles_x, bool vec_stage, int sN, int sC, int sH, int xlim, int rsN,
    int rsC, int rsH) {
  conv_block<C, TR, FW, NT, WM, WN, VECW, ADD_RES>(
      X, WT, SCALE, BIAS, RES, OUT, H, W, tiles_x, vec_stage, sN, sC, sH, xlim,
      rsN, rsC, rsH);
}

// Both convolutions in one launch: a grid-wide barrier separates them, which
// halves the (dominant) per-launch cost.
template <int C, int TR, int FW, int NT, int WM, int WN, int VECW, bool ADD_RES>
__global__ __launch_bounds__(32 * WM * WN) void bottleneck_kernel(
    const __half *__restrict__ X, const __half *__restrict__ WT1,
    const __half *__restrict__ WT2, const float *__restrict__ PRM,
    __half *__restrict__ Y1, __half *__restrict__ OUT, int H, int W,
    int tiles_x, bool vec_stage, bool vec_stage2, Strides xs, Strides ys) {
  conv_block<C, TR, FW, NT, WM, WN, VECW, false>(
      X, WT1, PRM, PRM + C, nullptr, Y1, H, W, tiles_x, vec_stage, xs.sN, xs.sC,
      xs.sH, xs.lim, ys.sN, ys.sC, ys.sH);
  cooperative_groups::this_grid().sync();
  conv_block<C, TR, FW, NT, WM, WN, VECW, ADD_RES>(
      Y1, WT2, PRM + 2 * C, PRM + 3 * C, X, OUT, H, W, tiles_x, vec_stage2,
      ys.sN, ys.sC, ys.sH, ys.lim, xs.sN, xs.sC, xs.sH);
}

#define IDIV_UP(a, b) (((a) + (b) - 1) / (b))

template <int C, int TR, int FW, int NT, int WM, int WN, int VECW, bool ADD_RES>
static void launch_one(const __half *X, const __half *WT, const float *SCALE,
                       const float *BIAS, const __half *RES, __half *OUT, int N,
                       int H, int W, bool vec_stage, const Strides &xs,
                       const Strides &rs, cudaStream_t stream) {
  constexpr int TW = 16 * FW;
  constexpr int MT = TR * TW;
  constexpr int LDK = C + 8;
  constexpr int XH = (TR + 2) * (TW + 2) * LDK;
  constexpr int YH = NT * (MT + 8);
  constexpr int R1 = XH > YH ? XH : YH;
  constexpr int SMEM = (R1 + 9 * NT * (C + 8)) * (int)sizeof(__half);
  auto kern = conv3x3_kernel<C, TR, FW, NT, WM, WN, VECW, ADD_RES>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM);
    configured = true;
  }
  const int tiles_x = IDIV_UP(W, TW);
  const int tiles_y = IDIV_UP(H, TR);
  dim3 grid(tiles_x * tiles_y, C / NT, N);
  kern<<<grid, 32 * WM * WN, SMEM, stream>>>(X, WT, SCALE, BIAS, RES, OUT, H, W,
                                             tiles_x, vec_stage, xs.sN, xs.sC,
                                             xs.sH, xs.lim, rs.sN, rs.sC,
                                             rs.sH);
}

// Returns true when the fused (cooperative) kernel was launched.
template <int C, int TR, int FW, int NT, int WM, int WN, int VECW, bool ADD_RES>
static bool launch_fused(const __half *X, const __half *WT1, const __half *WT2,
                         const float *PRM, __half *Y1, __half *OUT, int N,
                         int H, int W, bool vec_stage, bool vec_stage2,
                         const Strides &xs, const Strides &ys,
                         cudaStream_t stream) {
  constexpr int TW = 16 * FW;
  constexpr int MT = TR * TW;
  constexpr int LDK = C + 8;
  constexpr int XH = (TR + 2) * (TW + 2) * LDK;
  constexpr int YH = NT * (MT + 8);
  constexpr int R1 = XH > YH ? XH : YH;
  constexpr int SMEM = (R1 + 9 * NT * (C + 8)) * (int)sizeof(__half);
  constexpr int THREADS = 32 * WM * WN;
  auto kern = bottleneck_kernel<C, TR, FW, NT, WM, WN, VECW, ADD_RES>;
  static int max_blocks = -1;
  if (max_blocks < 0) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM);
    int per_sm = 0, nsm = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (void *)kern,
                                                  THREADS, SMEM);
    max_blocks = per_sm * nsm;
  }
  const int tiles_x = IDIV_UP(W, TW);
  const int tiles_y = IDIV_UP(H, TR);
  const int nblocks = tiles_x * tiles_y * (C / NT) * N;
  if (nblocks > max_blocks) return false;   // grid must be co-resident
  dim3 grid(tiles_x * tiles_y, C / NT, N);
  void *args[] = {(void *)&X,         (void *)&WT1,        (void *)&WT2,
                  (void *)&PRM,       (void *)&Y1,         (void *)&OUT,
                  (void *)&H,         (void *)&W,          (void *)&tiles_x,
                  (void *)&vec_stage, (void *)&vec_stage2, (void *)&xs,
                  (void *)&ys};
  cudaError_t err = cudaLaunchCooperativeKernel(
      (void *)kern, grid, dim3(THREADS), args, SMEM, stream);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

// Auto tile choice: keyed on channel count and total pixel count (N*H*W).
// Large problems favour bigger tiles (fewer shared round trips per output),
// small ones favour more blocks (the GPU is otherwise starved).
static int auto_cfg(int C, long px) {
  switch (C) {
    case 16:  return px >= 51200 ? 0 : 1;
    case 32:  return px >= 12800 ? 0 : 1;
    case 64:  return px >= 3200 ? 0 : 1;
    case 128: return px >= 800 ? 0 : 1;
    default:  return 0;
  }
}

template <int C>
static void launch_pair(int cfg, const __half *X, const __half *W1,
                        const float *S1, const float *B1, const __half *W2,
                        const float *S2, const float *B2, __half *Y1,
                        __half *OUT, bool add, int N, int H, int W, int vecw,
                        bool vstage, bool vstage2, const Strides &xs,
                        const Strides &ys, cudaStream_t stream, bool no_fuse);

#define RUN1(CC, TR, FW, NT, WM, WN, VE)                                       \
  do {                                                                         \
    bool fused = false;                                                        \
    if (!no_fuse) {                                                           \
      if (add)                                                                 \
        fused = launch_fused<CC, TR, FW, NT, WM, WN, VE, true>(                 \
            X, W1, W2, S1, Y1, OUT, N, H, W, vstage, vstage2, xs, ys, stream);   \
      else                                                                     \
        fused = launch_fused<CC, TR, FW, NT, WM, WN, VE, false>(                \
            X, W1, W2, S1, Y1, OUT, N, H, W, vstage, vstage2, xs, ys, stream);   \
    }                                                                          \
    if (fused) break;                                                          \
    launch_one<CC, TR, FW, NT, WM, WN, VE, false>(X, W1, S1, B1, nullptr, Y1,   \
                                                  N, H, W, vstage, xs, xs,     \
                                                  stream);                     \
    if (add)                                                                   \
      launch_one<CC, TR, FW, NT, WM, WN, VE, true>(Y1, W2, S2, B2, X, OUT, N,   \
                                                   H, W, vstage2, ys, xs,      \
                                                   stream);                    \
    else                                                                       \
      launch_one<CC, TR, FW, NT, WM, WN, VE, false>(Y1, W2, S2, B2, nullptr,    \
                                                    OUT, N, H, W, vstage2, ys, \
                                                    ys, stream);               \
  } while (0)

#define RUN(CC, TR, FW, NT, WM, WN)                                            \
  do {                                                                         \
    if (vecw == 8) {                                                           \
      RUN1(CC, TR, FW, NT, WM, WN, 8);                                         \
    } else {                                                                   \
      RUN1(CC, TR, FW, NT, WM, WN, 4);                                         \
    }                                                                          \
  } while (0)

#define PAIR_BODY(C, BODY)                                                    \
  template <>                                                                 \
  void launch_pair<C>(int cfg, const __half *X, const __half *W1,             \
                      const float *S1, const float *B1, const __half *W2,     \
                      const float *S2, const float *B2, __half *Y1,           \
                      __half *OUT, bool add, int N, int H, int W, int vecw,   \
                      bool vstage, bool vstage2, const Strides &xs,           \
                      const Strides &ys, cudaStream_t stream, bool no_fuse) {   \
    BODY                                                                      \
  }

PAIR_BODY(16, switch (cfg) {
  default: RUN(16, 8, 2, 16, 8, 1); break;
  case 1: RUN(16, 2, 2, 16, 4, 1); break;
})

PAIR_BODY(32, switch (cfg) {
  default: RUN(32, 4, 1, 32, 2, 2); break;
  case 1: RUN(32, 4, 1, 32, 4, 2); break;
})

PAIR_BODY(64, switch (cfg) {
  default: RUN(64, 4, 1, 64, 2, 2); break;
  case 1: RUN(64, 2, 1, 32, 2, 2); break;
})

PAIR_BODY(128, switch (cfg) {
  default: RUN(128, 8, 1, 32, 4, 2); break;
  case 1: RUN(128, 2, 1, 32, 2, 2); break;
})

// Scratch for the intermediate activation: reused across calls so the hot path
// performs a single allocation (the returned output).
static torch::Tensor &scratch_for(int64_t bytes, const torch::TensorOptions &opts) {
  static torch::Tensor buf;
  if (!buf.defined() || buf.numel() < bytes) {
    buf = torch::empty({bytes}, opts.dtype(torch::kUInt8));
  }
  return buf;
}

torch::Tensor bottleneck(const torch::Tensor &xin, const torch::Tensor &wt,
                         const torch::Tensor &prm, bool add, int64_t cfg) {
  const int N = xin.size(0), C = xin.size(1), H = xin.size(2), W = xin.size(3);
  // The inner dimension must be unit stride; anything else is materialized.
  const torch::Tensor x = xin.stride(3) == 1 ? xin : xin.contiguous();
  const int cfgi = cfg >= 0 ? (int)cfg : auto_cfg(C, (long)N * H * W);
  const int HW = H * W;
  // Offsets are 32-bit: YOLO activations are far below 2^31 elements.
  Strides xs{(int)x.stride(0), (int)x.stride(1), (int)x.stride(2), 0};
  xs.lim = (N - 1) * xs.sN + (C - 1) * xs.sC + (H - 1) * xs.sH + W;
  const Strides ys{C * HW, HW, W, N * C * HW};
  // Vector width for staging / epilogue: every stride touched must be aligned.
  const int alignmask = xs.sN | xs.sC | xs.sH | W;
  int vecw = 4;
  bool vstage = (alignmask & 3) == 0;
  if ((alignmask & 7) == 0) vecw = 8;
  // The intermediate is contiguous, so only W governs its access alignment.
  const bool vstage2 = (W & (vecw - 1)) == 0;
  auto out = torch::empty({N, C, H, W}, x.options());
  auto &scr = scratch_for(x.numel() * 2, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const __half *X = reinterpret_cast<const __half *>(x.data_ptr());
  const __half *W1 = reinterpret_cast<const __half *>(wt.data_ptr());
  const __half *W2 = W1 + (size_t)9 * C * C;
  __half *Y1 = reinterpret_cast<__half *>(scr.data_ptr());
  __half *O = reinterpret_cast<__half *>(out.data_ptr());
  const float *S1 = prm.data_ptr<float>();
  const float *B1 = S1 + C;
  const float *S2 = S1 + 2 * C;
  const float *B2 = S1 + 3 * C;

#define CASE(CC)                                                             \
  case CC:                                                                   \
    launch_pair<CC>(cfgi, X, W1, S1, B1, W2, S2, B2, Y1, O, add, N, H, W,     \
                    vecw, vstage, vstage2, xs, ys, stream, cfg == -2);        \
    break;
  switch (C) {
    CASE(16)
    CASE(32)
    CASE(64)
    CASE(128)
    default:
      TORCH_CHECK(false, "unsupported channel count ", C);
  }
#undef CASE
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bottleneck", &bottleneck, "fused YOLO bottleneck");
}
"""

_EXT = None
_SUPPORTED_C = (16, 32, 64, 128)


def _get_ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        suffix = "a" if major in (9, 10, 12) else ""
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"
        _EXT = load_inline(
            name="fk_yolo_bottleneck_mma",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def _fold(conv_mod):
    """Return (weight[9, co, ci] fp16, scale[co] fp32, bias[co] fp32) or None."""
    conv = conv_mod.conv
    w = conv.weight
    co, ci, kh, kw = w.shape
    if (kh, kw) != (3, 3) or conv.stride != (1, 1) or conv.padding != (1, 1):
        return None
    if conv.groups != 1 or conv.dilation != (1, 1) or co != ci:
        return None
    dev = w.device
    scale = torch.ones(co, device=dev, dtype=torch.float32)
    bias = torch.zeros(co, device=dev, dtype=torch.float32)
    if conv.bias is not None:
        bias = bias + conv.bias.detach().float()
    bn = getattr(conv_mod, "bn", None)
    if bn is not None:
        inv = torch.rsqrt(bn.running_var.detach().float() + bn.eps)
        g = bn.weight.detach().float() if bn.weight is not None else torch.ones_like(inv)
        beta = bn.bias.detach().float() if bn.bias is not None else torch.zeros_like(inv)
        s = g * inv
        scale = scale * s
        bias = bias * s + (beta - bn.running_mean.detach().float() * s)
    wt = w.detach().permute(2, 3, 0, 1).reshape(9, co, ci).contiguous().half()
    return wt, scale.contiguous(), bias.contiguous()


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self._fast = None

    def _build_fast(self, x: torch.Tensor) -> None:
        """Compile + fold weights on the first forward (weights are loaded by then)."""
        if x.dtype != torch.float16 or not x.is_cuda or x.dim() != 4:
            return  # leave the decision open: a later call may be supported
        self._fast = False
        try:
            c1 = self.cv1.conv.weight.shape[1]
            c_ = self.cv1.conv.weight.shape[0]
            c2 = self.cv2.conv.weight.shape[0]
            if not (c1 == c_ == c2 and c1 in _SUPPORTED_C):
                return
            for cv in (self.cv1, self.cv2):
                if type(cv.act).__name__ != "SiLU":
                    return
                if cv.conv.weight.dtype != torch.float16:
                    return
            f1 = _fold(self.cv1)
            f2 = _fold(self.cv2)
            if f1 is None or f2 is None:
                return
            ext = _get_ext()
            cfg = int(os.environ.get("FK_YOLO_BN_CFG", "-1"))
            wt = torch.stack((f1[0], f2[0])).contiguous()
            prm = torch.stack((f1[1], f1[2], f2[1], f2[2])).contiguous()
            self._fast = (ext.bottleneck, wt, prm, cfg)
        except Exception:
            self._fast = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._fast
        if f is None:
            self._build_fast(x)
            f = self._fast
        if f and x.dtype == torch.float16 and x.dim() == 4 and x.stride(3) == 1:
            return f[0](x, f[1], f[2], self.add, f[3])
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
