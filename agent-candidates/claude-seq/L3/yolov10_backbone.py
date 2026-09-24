"""YOLOv10 native backbone.

The frozen per-block winners already fuse each block down to a handful of
kernels, so what is left at this level is what no single block can see: the
layout at the seams, the two stem convolutions, and the cost of 30 launches.

*The seams.*  Every buffer between blocks is the block author's choice, so the
stages here hand each other **channel-last** tiles instead of NCHW.  That is what
lets the convolutions below gather their im2col operand with one ``ldmatrix`` out
of shared memory; ``stage2`` therefore writes channel-last, ``down3`` (a 3x3
stride-2 conv) consumes and produces channel-last, and only ``p3``/``p4``/``p5``
-- the tensors the caller sees -- are converted back to NCHW, by the epilogue of
the convolution that produces them.

*The kernels.*  Three templates cover the stem, every convolution inside a C2f
block, and the stage downsample, replacing 20 of the backbone's kernels:

  * ``stem1_kernel``/``stem2_kernel`` for the 3->16->32 stride-2 stem,
  * ``conv3k`` for 3x3 stride 1 and 2, and ``conv1k`` for 1x1 (cv1/cv2),

all of them staging the input tile in shared memory once, in a layout where the
16 halves an A-fragment wants for output pixel ``m`` begin exactly ``ldm * m``
into the tile -- so the im2col costs no address arithmetic and no bank conflicts
(the generic NCHW implicit GEMM they replace spends 30 M instructions and 0.74
issued instructions per scheduler cycle on the first stem layer alone, gathering
one 2-byte element per tap per channel).  BatchNorm is a per-channel affine in
the epilogue, in fp32, next to SiLU and the residual, so nothing about the
convolution arithmetic changes.  C2f's ``chunk``/``cat``/residual add are pure
addressing: ``cv1`` writes the head of a channel-last concat buffer, bottleneck
*i* writes slot ``2 + i`` and reads its residual from slot ``1 + i``.

*The launches.*  What remains is 30 kernels issued back to back, each waiting for
the previous one to drain.  So everything downstream of ``stem1`` is captured
once into a CUDA graph and replayed: ``stem1`` stays outside it because it is the
only op that reads the caller's tensor, whose address moves, and writing its
output into a buffer the graph owns is cheaper than copying the input in.

Which kernel case is used where was measured on a B200 at the captured shapes
(see ``_C2F_CASE``): at 40x40 and 20x20 there is not enough of a grid left to
hide a weight-staging latency, and the frozen whole-block winner is faster, so
those stages keep it.  Anything else the kernels do not cover -- a non-fp16 or
non-CUDA input, training mode, a fused ``YOLOConv``, a spatial extent the tiles
do not divide, a build or capture failure -- falls back to the plain composition.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from ..L2.yolov10_c2f import YOLOC2f
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_psa import YOLOPSA
from ..L2.yolov10_scdown import YOLOSCDown
from ..L2.yolov10_sppf import YOLOSPPF

_KEYS = ("p3_backbone", "p4_backbone", "p5_backbone")

_CUDA_SRC = r"""
// YOLOv10 stem: two 3x3 stride-2 convolutions (3->16->32), each with an
// eval-mode BatchNorm folded into an affine epilogue and SiLU.  NCHW fp16 in,
// NCHW fp16 out, sm_90+.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdlib>

using namespace nvcuda;
#define DIVUP(a, b) (((a) + (b) - 1) / (b))

__device__ __forceinline__ float fk_silu(float f) {
  return f / (1.f + __expf(-f));
}

__device__ __forceinline__ float fk_act(float f, int act) {
  return act ? fk_silu(f) : f;
}

// ---------------------------------------------------------------------------
// stem1: x[N,3,H,W] NCHW -> y[N,H/2,W/2,16] NHWC
// ---------------------------------------------------------------------------
template <int TY, int TX, int NW>
__global__ __launch_bounds__(32 * NW) void stem1_kernel(
    const __half* __restrict__ X, const __half* __restrict__ Wt,
    const float* __restrict__ SCv, const float* __restrict__ SHv,
    __half* __restrict__ Y, int H, int W_, int H2, int W2) {
  constexpr int SR = 2 * TY + 1;              // staged rows
  constexpr int SC_ = 2 * TX + 4;             // staged cols (even)
  constexpr int NT = 32 * NW;
  constexpr int MX = TX / 16;                 // m-tiles per row
  constexpr int MTOT = TY * MX;
  constexpr int MPW = MTOT / NW;              // m-tiles per warp
  constexpr int CHK = (SC_ + 6) / 8 + 1;      // aligned 8-col chunks per row
  constexpr int NTASK = SR * CHK;
  constexpr int PER = DIVUP(NTASK, NT);
  constexpr int ELD = 20;

  __shared__ __half sx[SR * SC_ * 4];
  __shared__ __half wb[3 * 256];
  __shared__ float ep[NW * 16 * ELD];
  __shared__ float sc[16], sh[16];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = blockIdx.z, ry0 = blockIdx.y * TY, cx0 = blockIdx.x * TX;
  const int base = 2 * cx0 - 1;               // global col of shared col 0
  const int r0 = 2 * ry0 - 1;

  {
    const uint4* wsrc = reinterpret_cast<const uint4*>(Wt);
    uint4* wdst = reinterpret_cast<uint4*>(wb);
    for (int i = tid; i < 3 * 256 / 8; i += NT) wdst[i] = wsrc[i];
  }
  if (tid < 16) { sc[tid] = SCv[tid]; sh[tid] = SHv[tid]; }

  {   // stage: 16 B per channel per task, all loads issued before any store
    const long plane = (long)H * W_;
    const __half* xp = X + (long)n * 3 * plane;
    uint4 v[PER][3];
    int row[PER], col[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int t = tid + i * NT;
      const int sr = t / CHK, ci = t - sr * CHK;
      row[i] = sr;
      col[i] = 8 * ci - 7;                    // shared col of this chunk
      const int gr = r0 + sr, gc = base + col[i];
      const bool ok = t < NTASK && (unsigned)gr < (unsigned)H;
      const __half* p = xp + (long)gr * W_ + gc;
      v[i][0] = v[i][1] = v[i][2] = make_uint4(0, 0, 0, 0);
      if (ok && gc >= 0 && gc + 8 <= W_) {
        v[i][0] = *reinterpret_cast<const uint4*>(p);
        v[i][1] = *reinterpret_cast<const uint4*>(p + plane);
        v[i][2] = *reinterpret_cast<const uint4*>(p + 2 * plane);
      } else if (ok) {                        // edge chunk: scalar, still rare
        unsigned short* a = reinterpret_cast<unsigned short*>(&v[i][0]);
        unsigned short* b = reinterpret_cast<unsigned short*>(&v[i][1]);
        unsigned short* c = reinterpret_cast<unsigned short*>(&v[i][2]);
#pragma unroll
        for (int t2 = 0; t2 < 8; ++t2) {
          const int g = gc + t2;
          if ((unsigned)g < (unsigned)W_) {
            a[t2] = __half_as_ushort(p[t2]);
            b[t2] = __half_as_ushort(p[t2 + plane]);
            c[t2] = __half_as_ushort(p[t2 + 2 * plane]);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int t = tid + i * NT;
      if (t >= NTASK) continue;
      const unsigned short* a = reinterpret_cast<const unsigned short*>(&v[i][0]);
      const unsigned short* b = reinterpret_cast<const unsigned short*>(&v[i][1]);
      const unsigned short* c = reinterpret_cast<const unsigned short*>(&v[i][2]);
#pragma unroll
      for (int t2 = 0; t2 < 8; ++t2) {
        const int s = col[i] + t2;
        if ((unsigned)s < (unsigned)SC_) {
          ushort4 q;
          q.x = a[t2]; q.y = b[t2]; q.z = c[t2]; q.w = 0;
          *reinterpret_cast<ushort4*>(&sx[(row[i] * SC_ + s) * 4]) = q;
        }
      }
    }
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MPW];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf[3];
#pragma unroll
  for (int kr = 0; kr < 3; ++kr) wmma::load_matrix_sync(bf[kr], &wb[kr * 256], 16);
#pragma unroll
  for (int u = 0; u < MPW; ++u) {
    const int mt = warp + u * NW;
    const int my = mt / MX, j = mt - my * MX;
    wmma::fill_fragment(acc[u], 0.f);
#pragma unroll
    for (int kr = 0; kr < 3; ++kr) {
      wmma::load_matrix_sync(af, &sx[((2 * my + kr) * SC_ + 32 * j) * 4], 8);
      wmma::mma_sync(acc[u], af, bf[kr], acc[u]);
    }
  }

  float* myep = &ep[warp * 16 * ELD];
  const int r = lane >> 1, ch = (lane & 1) * 8;
#pragma unroll
  for (int u = 0; u < MPW; ++u) {
    const int mt = warp + u * NW;
    const int my = mt / MX, j = mt - my * MX;
    __syncwarp();
    wmma::store_matrix_sync(myep, acc[u], ELD, wmma::mem_row_major);
    __syncwarp();
    const float4 q0 = *reinterpret_cast<const float4*>(myep + r * ELD + ch);
    const float4 q1 = *reinterpret_cast<const float4*>(myep + r * ELD + ch + 4);
    const float a[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    uint4 o;
    unsigned short* op = reinterpret_cast<unsigned short*>(&o);
#pragma unroll
    for (int t = 0; t < 8; ++t)
      op[t] = __half_as_ushort(__float2half(fk_silu(a[t] * sc[ch + t] + sh[ch + t])));
    *reinterpret_cast<uint4*>(
        Y + ((((long)n * H2 + ry0 + my) * W2) + cx0 + j * 16 + r) * 16 + ch) = o;
  }
}

// ---------------------------------------------------------------------------
// stem2: y1[N,H2,W2,16] NHWC -> out[N,COUT,H4,W4] NCHW
// ---------------------------------------------------------------------------
template <int TY, int TX, int NW, int CIN, int COUT>
__global__ __launch_bounds__(32 * NW) void stem2_kernel(
    const __half* __restrict__ X, const __half* __restrict__ Wt,
    const float* __restrict__ SCv, const float* __restrict__ SHv,
    __half* __restrict__ Y, int H2, int W2, int H4, int W4) {
  constexpr int SR = 2 * TY + 1;
  constexpr int SX = 2 * TX + 1;
  constexpr int XH = TX + 1;
  constexpr int PS = 24;                       // padded pixel stride (halves)
  constexpr int NT = 32 * NW;
  constexpr int MX = TX / 16;
  constexpr int MTOT = TY * MX;
  constexpr int MPW = MTOT / NW;
  constexpr int NN = COUT / 16;
  constexpr int BLD = 40;
  constexpr int ELD = 16 * NN + 4;
  constexpr int NV = SR * SX * 2;
  constexpr int PER = DIVUP(NV, NT);
  constexpr int SYH = 2 * SR * XH * PS;
  constexpr int EPF = NW * 16 * ELD;

  extern __shared__ char smem[];
  __half* sy = reinterpret_cast<__half*>(smem);
  float* ep = reinterpret_cast<float*>(smem);
  __half* wb = reinterpret_cast<__half*>(
      smem + ((size_t)SYH * 2 > (size_t)EPF * 4 ? (size_t)SYH * 2 : (size_t)EPF * 4));
  float* sc = reinterpret_cast<float*>(wb + 9 * CIN * BLD);
  float* sh = sc + COUT;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = blockIdx.z, ry0 = blockIdx.y * TY, cx0 = blockIdx.x * TX;
  const int r0 = 2 * ry0 - 1, c0 = 2 * cx0 - 1;

  {   // 16 B per lane: the B tiles are 11.5 kB, and copying them one half at a
      // time cost 23 loads + 23 stores per thread -- more memory instructions
      // than the input tile itself, paid by every block in the grid.
    const uint4* wsrc = reinterpret_cast<const uint4*>(Wt);
    uint4* wdst = reinterpret_cast<uint4*>(wb);
#pragma unroll
    for (int i = tid; i < 9 * CIN * BLD / 8; i += NT) wdst[i] = wsrc[i];
  }
  for (int i = tid; i < COUT; i += NT) { sc[i] = SCv[i]; sh[i] = SHv[i]; }

  {
    uint4 val[PER];
    int slot[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int v = tid + i * NT;
      const int sr = v / (SX * 2), rem = v - sr * (SX * 2);
      const int sx = rem >> 1, hi = rem & 1;
      const int gy = r0 + sr, gx = c0 + sx;
      const bool ok = v < NV && (unsigned)gy < (unsigned)H2 &&
                      (unsigned)gx < (unsigned)W2;
      val[i] = make_uint4(0, 0, 0, 0);
      if (ok)
        val[i] = *reinterpret_cast<const uint4*>(
            X + (((long)n * H2 + gy) * W2 + gx) * CIN + hi * 8);
      slot[i] = v < NV
                    ? (((sx & 1) * SR + sr) * XH + (sx >> 1)) * PS + hi * 8
                    : -1;
    }
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (slot[i] >= 0) *reinterpret_cast<uint4*>(sy + slot[i]) = val[i];
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MPW][NN];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf[NN];
#pragma unroll
  for (int u = 0; u < MPW; ++u)
#pragma unroll
    for (int t = 0; t < NN; ++t) wmma::fill_fragment(acc[u][t], 0.f);

#pragma unroll
  for (int tap = 0; tap < 9; ++tap) {
    const int kr = tap / 3, ks = tap - 3 * kr, par = ks & 1;
#pragma unroll
    for (int t = 0; t < NN; ++t)
      wmma::load_matrix_sync(bf[t], wb + tap * CIN * BLD + t * 16, BLD);
#pragma unroll
    for (int u = 0; u < MPW; ++u) {
      const int mt = warp + u * NW;
      const int my = mt / MX, j = mt - my * MX;
      wmma::load_matrix_sync(
          af, sy + ((par * SR + 2 * my + kr) * XH + j * 16 + (ks >> 1)) * PS, PS);
#pragma unroll
      for (int t = 0; t < NN; ++t) wmma::mma_sync(acc[u][t], af, bf[t], acc[u][t]);
    }
  }

  __syncthreads();                             // sy -> ep
  float* myep = ep + warp * 16 * ELD;
#pragma unroll
  for (int u = 0; u < MPW; ++u) {
    const int mt = warp + u * NW;
    const int my = mt / MX, j = mt - my * MX;
    __syncwarp();
#pragma unroll
    for (int t = 0; t < NN; ++t)
      wmma::store_matrix_sync(myep + t * 16, acc[u][t], ELD, wmma::mem_row_major);
    __syncwarp();
    if (lane < COUT) {
      const float s = sc[lane], b = sh[lane];
      uint4 o[2];
      unsigned short* op = reinterpret_cast<unsigned short*>(o);
#pragma unroll
      for (int m = 0; m < 16; ++m)
        op[m] = __half_as_ushort(__float2half(fk_silu(myep[m * ELD + lane] * s + b)));
      __half* yp = Y + (((long)n * COUT + lane) * H4 + ry0 + my) * W4 + cx0 + j * 16;
      *reinterpret_cast<uint4*>(yp) = o[0];
      *reinterpret_cast<uint4*>(yp + 8) = o[1];
    }
  }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
template <int TY, int TX, int NW>
static void run1(const __half* x, const __half* w, const float* s, const float* b,
                 __half* y, int N, int H, int W_, cudaStream_t st) {
  const int H2 = H / 2, W2 = W_ / 2;
  dim3 g(W2 / TX, H2 / TY, N);
  stem1_kernel<TY, TX, NW><<<g, 32 * NW, 0, st>>>(x, w, s, b, y, H, W_, H2, W2);
}

template <int TY, int TX, int NW, int COUT>
static void run2(const __half* x, const __half* w, const float* s, const float* b,
                 __half* y, int N, int H2, int W2, cudaStream_t st) {
  constexpr int SR = 2 * TY + 1;
  constexpr int SYH = 2 * SR * (TX + 1) * 24;
  constexpr int ELD = 16 * (COUT / 16) + 4;
  constexpr int EPF = 32 * TY * TX / 16 / 32 * 16 * ELD;  // NW*16*ELD
  constexpr size_t stage = (size_t)SYH * 2 > (size_t)EPF * 4 ? (size_t)SYH * 2
                                                             : (size_t)EPF * 4;
  const size_t smem = stage + (size_t)(9 * 16 * 40) * 2 + (size_t)COUT * 8;
  const int H4 = H2 / 2, W4 = W2 / 2;
  auto fn = stem2_kernel<TY, TX, NW, 16, COUT>;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 g(W4 / TX, H4 / TY, N);
  fn<<<g, 32 * NW, smem, st>>>(x, w, s, b, y, H2, W2, H4, W4);
}

// stem1 runs outside the CUDA graph (it is the only op that reads the caller's
// tensor, whose address the benchmark shifts every iteration) and writes into a
// caller-owned buffer, so the graph can start at stem2 and no 10 MB input copy
// is needed to feed it.
void stem1_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor s1,
                   torch::Tensor b1, torch::Tensor y1, int64_t cfg1) {
  const int N = x.size(0), H = x.size(2), W_ = x.size(3);
  const int H2 = H / 2, W2 = W_ / 2;
  auto st = at::cuda::getCurrentCUDAStream();
  const __half* xp = (const __half*)x.const_data_ptr();
  const __half* w1p = (const __half*)w1.const_data_ptr();
  __half* y1p = (__half*)y1.data_ptr();
  const float* s1p = s1.const_data_ptr<float>();
  const float* b1p = b1.const_data_ptr<float>();
  if (cfg1 < 0)
    cfg1 = ((long)N * (H2 / 16) * (W2 / 32) >= 300) ? 9 : 3;
  switch (cfg1) {
    case 2: run1<4, 64, 8>(xp, w1p, s1p, b1p, y1p, N, H, W_, st); break;
    case 3: run1<8, 32, 8>(xp, w1p, s1p, b1p, y1p, N, H, W_, st); break;
    case 7: run1<16, 64, 8>(xp, w1p, s1p, b1p, y1p, N, H, W_, st); break;
    case 9: run1<16, 32, 8>(xp, w1p, s1p, b1p, y1p, N, H, W_, st); break;
    default: TORCH_CHECK(false, "cfg1");
  }
}

torch::Tensor stem2_forward(torch::Tensor y1, torch::Tensor w2,
                            torch::Tensor s2, torch::Tensor b2, int64_t cfg2) {
  const int N = y1.size(0), H2 = y1.size(1), W2 = y1.size(2);
  const int COUT = b2.size(0);
  auto y2 = torch::empty({N, COUT, H2 / 2, W2 / 2}, y1.options());
  auto st = at::cuda::getCurrentCUDAStream();
  const __half* y1p = (const __half*)y1.const_data_ptr();
  const __half* w2p = (const __half*)w2.const_data_ptr();
  __half* y2p = (__half*)y2.data_ptr();
  const float* s2p = s2.const_data_ptr<float>();
  const float* b2p = b2.const_data_ptr<float>();
  TORCH_CHECK(COUT == 32, "stem2 COUT");
  if (cfg2 < 0)
    cfg2 = ((long)N * (H2 / 32) * (W2 / 32) >= 300) ? 7 : 2;
  switch (cfg2) {
    case 2: run2<4, 16, 4, 32>(y1p, w2p, s2p, b2p, y2p, N, H2, W2, st); break;
    case 3: run2<8, 32, 8, 32>(y1p, w2p, s2p, b2p, y2p, N, H2, W2, st); break;
    case 7: run2<16, 16, 8, 32>(y1p, w2p, s2p, b2p, y2p, N, H2, W2, st); break;
    case 9: run2<1, 32, 2, 32>(y1p, w2p, s2p, b2p, y2p, N, H2, W2, st); break;
    default: TORCH_CHECK(false, "cfg2");
  }
  return y2;
}

// ---------------------------------------------------------------------------
// The convolutions inside a C2f block, as two kernels: a 3x3 stride-1 one for
// the bottlenecks and a 1x1 one for cv1/cv2.  Both apply the eval-mode
// BatchNorm as a per-channel affine, then SiLU, then an optional residual, in
// the epilogue, and both address their source and destination as a *channel
// slice* of a larger channel-last buffer -- which is what makes C2f's chunk()
// and cat() disappear: cv1 writes the head of the concat buffer, bottleneck i
// writes slot 2+i, and cv2 reads the whole thing.
//
// The point of both is the same as in the stem: stage the input tile in shared
// memory once, in a layout where the 16 halves an A-fragment wants for output
// pixel m start exactly ldm*m into the tile, so the im2col is a single
// load_matrix_sync with no address arithmetic and no bank conflicts.  A
// channel-last tile with the pixel stride padded to CIN+8 halves does that for
// the 3x3 (48/72/136/264 B strides spread the 16 ldmatrix rows over all 32
// banks); for the 1x1 whose source is still NCHW, channel planes plus col-major
// A fragments do it with a pure vector copy for staging.
//
// Channel-last buffers are allocated with their rows padded out to a multiple of
// 16 pixels and the padding left at zero, so a block's 16-pixel m-tile never
// straddles a row and the 3x3 halo at the right edge reads the zeros it wants.
// ---------------------------------------------------------------------------

#define FK_MMA_ROUND(AFRAG)                                                  \
  {                                                                          \
    _Pragma("unroll") for (int t = 0; t < WN; ++t) wmma::mma_sync(            \
        acc[u][t], AFRAG, bf[t], acc[u][t]);                                  \
  }

// --- 3x3 stride 1, channel-last -> channel-last -----------------------------
template <int CIN, int COUT, int TY, int NW, int WM, int WN, int TAPC, int STR>
__global__ __launch_bounds__(32 * NW) void conv3k(
    const __half* __restrict__ X, const __half* __restrict__ Wt,
    const float* __restrict__ SCv, const float* __restrict__ SHv,
    const __half* __restrict__ RS, __half* __restrict__ Y, int Hi, int Wi,
    int H, int W_, int sW, int dW, int srcC, int dstC, int resC, int act) {
  // STR == 2 stages the tile split by input-x parity, so that for a fixed tap
  // the 16 pixels of an A-fragment are 16 *consecutive* slots again (a stride-2
  // m-step would otherwise put every ldmatrix row on the same pair of banks).
  constexpr int TX = 16;
  constexpr int SXP = STR == 1 ? TX + 2 : TX + 1;    // cols per (parity) plane
  constexpr int SRP = STR == 1 ? TY + 2 : 2 * TY + 1;
  constexpr int NPAR = STR == 1 ? 1 : 2;
  constexpr int PS = CIN + 8, BLD = COUT + 8;
  constexpr int NT = 32 * NW, MT = TY, NN = COUT / 16;
  constexpr int MW = MT / WM, ROUNDS = 9 / TAPC;
  static_assert(MT % WM == 0 && NN % WN == 0 && NW == MW * (NN / WN),
                "conv3k warp decomposition");
  static_assert(9 % TAPC == 0, "conv3k tap chunk");
  constexpr int ELD = WN * 16 + 4;
  constexpr int SAH = NPAR * SRP * SXP * PS, EPF = NW * 16 * ELD;
  constexpr size_t STAGE = (size_t)SAH * 2 > (size_t)EPF * 4 ? (size_t)SAH * 2
                                                             : (size_t)EPF * 4;
  extern __shared__ char smem[];
  __half* sy = reinterpret_cast<__half*>(smem);
  float* ep = reinterpret_cast<float*>(smem);
  __half* wb = reinterpret_cast<__half*>(smem + STAGE);
  float* sc = reinterpret_cast<float*>(wb + TAPC * CIN * BLD);
  float* sh = sc + COUT;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = blockIdx.z, ry0 = blockIdx.y * TY, cx0 = blockIdx.x * TX;
  const int mw = warp % MW, nw = warp / MW, coff = nw * WN * 16;
  const int iy0 = STR * ry0 - 1, ix0 = STR * cx0 - 1;   // staged tile origin

  for (int i = tid; i < COUT; i += NT) { sc[i] = SCv[i]; sh[i] = SHv[i]; }
  {   // input tile: every 16 B load issued before the first shared store
    constexpr int VPP = CIN / 8;
    constexpr int SXR = STR == 1 ? SXP : 2 * TX + 1;    // real staged cols
    constexpr int NV = SRP * SXR * VPP;
    constexpr int PER = DIVUP(NV, NT);
    uint4 v[PER];
    int slot[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int t = tid + i * NT;
      const int p = t / VPP, q = t - p * VPP;
      const int sr = p / SXR, sx = p - sr * SXR;
      const int gy = iy0 + sr, gx = ix0 + sx;
      v[i] = make_uint4(0, 0, 0, 0);
      const int pslot = STR == 1 ? p : ((sx & 1) * SRP + sr) * SXP + (sx >> 1);
      slot[i] = t < NV ? pslot * PS + q * 8 : -1;
      if (t < NV && (unsigned)gy < (unsigned)Hi && (unsigned)gx < (unsigned)Wi)
        v[i] = *reinterpret_cast<const uint4*>(
            X + (((long)n * Hi + gy) * sW + gx) * srcC + q * 8);
    }
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (slot[i] >= 0) *reinterpret_cast<uint4*>(sy + slot[i]) = v[i];
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM][WN];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf[WN];
#pragma unroll
  for (int u = 0; u < WM; ++u)
#pragma unroll
    for (int t = 0; t < WN; ++t) wmma::fill_fragment(acc[u][t], 0.f);

#pragma unroll 1
  for (int rd = 0; rd < ROUNDS; ++rd) {
    if (rd) __syncthreads();
    {   // B tiles for this round of taps: [tap][CIN][BLD], host-padded
      const uint4* ws = reinterpret_cast<const uint4*>(Wt) + rd * TAPC * CIN * BLD / 8;
      uint4* wd = reinterpret_cast<uint4*>(wb);
#pragma unroll 4
      for (int i = tid; i < TAPC * CIN * BLD / 8; i += NT) wd[i] = ws[i];
    }
    __syncthreads();
#pragma unroll
    for (int tt = 0; tt < TAPC; ++tt) {
      const int tap = rd * TAPC + tt;
      const int kr = tap / 3, ks = tap - kr * 3;
#pragma unroll 1
      for (int cg = 0; cg < CIN / 16; ++cg) {
#pragma unroll
        for (int t = 0; t < WN; ++t)
          wmma::load_matrix_sync(
              bf[t], wb + (tt * CIN + cg * 16) * BLD + coff + t * 16, BLD);
#pragma unroll
        for (int u = 0; u < WM; ++u) {
          const int my = mw * WM + u;
          const int base = STR == 1
                               ? ((my + kr) * SXP + ks)
                               : (((ks & 1) * SRP + STR * my + kr) * SXP
                                  + (ks >> 1));
          wmma::load_matrix_sync(af, sy + base * PS + cg * 16, PS);
          FK_MMA_ROUND(af)
        }
      }
    }
  }

  __syncthreads();
  float* myep = ep + warp * 16 * ELD;
  constexpr int CPL = WN * 2, PPP = 32 / CPL;
#pragma unroll
  for (int u = 0; u < WM; ++u) {
    const int my = mw * WM + u;
    if (ry0 + my >= H) break;
    __syncwarp();
#pragma unroll
    for (int t = 0; t < WN; ++t)
      wmma::store_matrix_sync(myep + t * 16, acc[u][t], ELD, wmma::mem_row_major);
    __syncwarp();
#pragma unroll
    for (int pass = 0; pass < 16 / PPP; ++pass) {
      const int idx = pass * 32 + lane;
      const int pm = idx / CPL, c8 = (idx - pm * CPL) * 8;
      const int ox = cx0 + pm;
      if (ox >= W_) continue;
      const float4 q0 = *reinterpret_cast<const float4*>(myep + pm * ELD + c8);
      const float4 q1 = *reinterpret_cast<const float4*>(myep + pm * ELD + c8 + 4);
      const float a[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
      const long pix = ((long)n * H + ry0 + my) * dW + ox;
      uint4 rv;
      const unsigned short* rp = reinterpret_cast<const unsigned short*>(&rv);
      if (RS != nullptr)
        rv = *reinterpret_cast<const uint4*>(RS + pix * resC + coff + c8);
      uint4 o;
      unsigned short* op = reinterpret_cast<unsigned short*>(&o);
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int cc = coff + c8 + t;
        float f = fk_act(a[t] * sc[cc] + sh[cc], act);
        if (RS != nullptr) f += __half2float(__ushort_as_half(rp[t]));
        op[t] = __half_as_ushort(__float2half(f));
      }
      *reinterpret_cast<uint4*>(Y + pix * dstC + coff + c8) = o;
    }
  }
}

// --- 1x1: channel-last or NCHW source -> channel-last or NCHW dest ----------
template <int CIN, int COUT, int TY, int NW, int WM, int WN, int KC, int INC,
          int OUTC>
__global__ __launch_bounds__(32 * NW) void conv1k(
    const __half* __restrict__ X, const __half* __restrict__ Wt,
    const float* __restrict__ SCv, const float* __restrict__ SHv,
    __half* __restrict__ Y, int H, int W_, int sW, int dW, int srcC, int dstC,
    int act) {
  constexpr int TX = 16, SRP = TY, SXP = TX;
  constexpr int PS = CIN + 8, PLS = SRP * SXP + 8, BLD = COUT + 8;
  constexpr int NT = 32 * NW, MT = TY, NN = COUT / 16;
  constexpr int MW = MT / WM, ROUNDS = CIN / KC;
  static_assert(MT % WM == 0 && NN % WN == 0 && NW == MW * (NN / WN),
                "conv1k warp decomposition");
  static_assert(CIN % KC == 0 && KC % 16 == 0, "conv1k k chunk");
  constexpr int ELD = WN * 16 + 4;
  constexpr int SAH = INC ? CIN * PLS : SRP * SXP * PS;
  constexpr int EPF = NW * 16 * ELD;
  constexpr size_t STAGE = (size_t)SAH * 2 > (size_t)EPF * 4 ? (size_t)SAH * 2
                                                             : (size_t)EPF * 4;
  extern __shared__ char smem[];
  __half* sy = reinterpret_cast<__half*>(smem);
  float* ep = reinterpret_cast<float*>(smem);
  __half* wb = reinterpret_cast<__half*>(smem + STAGE);
  float* sc = reinterpret_cast<float*>(wb + KC * BLD);
  float* sh = sc + COUT;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = blockIdx.z, ry0 = blockIdx.y * TY, cx0 = blockIdx.x * TX;
  const int mw = warp % MW, nw = warp / MW, coff = nw * WN * 16;

  for (int i = tid; i < COUT; i += NT) { sc[i] = SCv[i]; sh[i] = SHv[i]; }
  if (INC) {   // NCHW source -> channel planes (pixels already contiguous)
    constexpr int NV = CIN * SRP * SXP / 8;
    constexpr int PER = DIVUP(NV, NT);
    uint4 v[PER];
    int slot[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int t = tid + i * NT;
      const int c = t / (SRP * SXP / 8), rem = t - c * (SRP * SXP / 8);
      const int sr = rem / (SXP / 8), sx = (rem - sr * (SXP / 8)) * 8;
      const int gy = ry0 + sr, gx = cx0 + sx;
      v[i] = make_uint4(0, 0, 0, 0);
      slot[i] = t < NV ? c * PLS + sr * SXP + sx : -1;
      if (t < NV && gy < H) {
        const __half* p = X + (((long)n * srcC + c) * H + gy) * sW + gx;
        if (gx + 8 <= W_) {
          uint2* dst = reinterpret_cast<uint2*>(&v[i]);
          dst[0] = *reinterpret_cast<const uint2*>(p);
          dst[1] = *reinterpret_cast<const uint2*>(p + 4);
        } else {
          unsigned short* d = reinterpret_cast<unsigned short*>(&v[i]);
#pragma unroll
          for (int e = 0; e < 8; ++e)
            if (gx + e < W_) d[e] = __half_as_ushort(p[e]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (slot[i] >= 0) *reinterpret_cast<uint4*>(sy + slot[i]) = v[i];
  } else {     // channel-last source
    constexpr int VPP = CIN / 8, NV = SRP * SXP * VPP;
    constexpr int PER = DIVUP(NV, NT);
    uint4 v[PER];
    int slot[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int t = tid + i * NT;
      const int p = t / VPP, q = t - p * VPP;
      const int sr = p / SXP, sx = p - sr * SXP;
      const int gy = ry0 + sr, gx = cx0 + sx;
      v[i] = make_uint4(0, 0, 0, 0);
      slot[i] = t < NV ? p * PS + q * 8 : -1;
      if (t < NV && gy < H && gx < W_)
        v[i] = *reinterpret_cast<const uint4*>(
            X + (((long)n * H + gy) * sW + gx) * srcC + q * 8);
    }
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (slot[i] >= 0) *reinterpret_cast<uint4*>(sy + slot[i]) = v[i];
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM][WN];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> afr;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> afc;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf[WN];
#pragma unroll
  for (int u = 0; u < WM; ++u)
#pragma unroll
    for (int t = 0; t < WN; ++t) wmma::fill_fragment(acc[u][t], 0.f);

#pragma unroll 1
  for (int rd = 0; rd < ROUNDS; ++rd) {
    if (rd) __syncthreads();
    {
      const uint4* ws = reinterpret_cast<const uint4*>(Wt) + rd * KC * BLD / 8;
      uint4* wd = reinterpret_cast<uint4*>(wb);
#pragma unroll 4
      for (int i = tid; i < KC * BLD / 8; i += NT) wd[i] = ws[i];
    }
    __syncthreads();
#pragma unroll 1
    for (int cg = 0; cg < KC / 16; ++cg) {
      const int k0 = rd * KC + cg * 16;
#pragma unroll
      for (int t = 0; t < WN; ++t)
        wmma::load_matrix_sync(bf[t], wb + cg * 16 * BLD + coff + t * 16, BLD);
#pragma unroll
      for (int u = 0; u < WM; ++u) {
        const int my = mw * WM + u;
        if constexpr (INC) {
          wmma::load_matrix_sync(afc, sy + k0 * PLS + my * SXP, PLS);
          FK_MMA_ROUND(afc)
        } else {
          wmma::load_matrix_sync(afr, sy + my * SXP * PS + k0, PS);
          FK_MMA_ROUND(afr)
        }
      }
    }
  }

  __syncthreads();
  float* myep = ep + warp * 16 * ELD;
#pragma unroll
  for (int u = 0; u < WM; ++u) {
    const int my = mw * WM + u;
    if (ry0 + my >= H) break;
    __syncwarp();
#pragma unroll
    for (int t = 0; t < WN; ++t)
      wmma::store_matrix_sync(myep + t * 16, acc[u][t], ELD, wmma::mem_row_major);
    __syncwarp();
    if constexpr (OUTC) {          // NCHW dest: one lane per (channel, 8 pixels)
      const int c = lane >> 1, ph = (lane & 1) * 8;
      const int ox = cx0 + ph;
      const long ybase = (((long)n * dstC + coff) * H + ry0 + my) * dW + ox;
#pragma unroll
      for (int t = 0; t < WN; ++t) {
        const int cc = t * 16 + c;
        uint4 o;
        unsigned short* op = reinterpret_cast<unsigned short*>(&o);
#pragma unroll
        for (int m = 0; m < 8; ++m)
          op[m] = __half_as_ushort(__float2half(fk_act(
              myep[(ph + m) * ELD + cc] * sc[coff + cc] + sh[coff + cc], act)));
        __half* yp = Y + ybase + (long)cc * H * dW;
        if (ox + 8 <= W_) {
          // 8 B stores: an NCHW row stride is only guaranteed to be a multiple
          // of four halves (a 20-wide plane is not 16 B aligned).
          reinterpret_cast<uint2*>(yp)[0] = reinterpret_cast<const uint2*>(&o)[0];
          reinterpret_cast<uint2*>(yp)[1] = reinterpret_cast<const uint2*>(&o)[1];
        } else {
#pragma unroll
          for (int m = 0; m < 8; ++m)
            if (ox + m < W_) yp[m] = __ushort_as_half(op[m]);
        }
      }
    } else {                       // channel-last dest
      constexpr int CPL = WN * 2, PPP = 32 / CPL;
#pragma unroll
      for (int pass = 0; pass < 16 / PPP; ++pass) {
        const int idx = pass * 32 + lane;
        const int pm = idx / CPL, c8 = (idx - pm * CPL) * 8;
        const int ox = cx0 + pm;
        if (ox >= W_) continue;
        const float4 q0 = *reinterpret_cast<const float4*>(myep + pm * ELD + c8);
        const float4 q1 = *reinterpret_cast<const float4*>(myep + pm * ELD + c8 + 4);
        const float a[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        uint4 o;
        unsigned short* op = reinterpret_cast<unsigned short*>(&o);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const int cc = coff + c8 + t;
          op[t] = __half_as_ushort(
              __float2half(fk_act(a[t] * sc[cc] + sh[cc], act)));
        }
        *reinterpret_cast<uint4*>(
            Y + (((long)n * H + ry0 + my) * dW + ox) * dstC + coff + c8) = o;
      }
    }
  }
}

// --- host dispatch ----------------------------------------------------------
// One entry point per kernel family, selected by a small integer: the tile
// shape, channel counts and staging chunk of every convolution in the four C2f
// blocks are compile-time constants, and the Python plan picks the case.
struct FkArgs {
  const __half* x;
  const __half* w;
  const float* sc;
  const float* sh;
  const __half* res;
  __half* y;
  int N, Hi, Wi, H, W, sW, dW, srcC, dstC, resC, act;
};

template <int CIN, int COUT, int TY, int NW, int WM, int WN, int TAPC, int STR>
static void go3(const FkArgs& a, cudaStream_t st) {
  constexpr int SXP = STR == 1 ? 18 : 17;
  constexpr int SRP = STR == 1 ? TY + 2 : 2 * TY + 1;
  constexpr int NPAR = STR == 1 ? 1 : 2;
  constexpr int PS = CIN + 8, BLD = COUT + 8, ELD = WN * 16 + 4;
  constexpr size_t SAH = (size_t)NPAR * SRP * SXP * PS;
  constexpr size_t EPF = (size_t)NW * 16 * ELD;
  constexpr size_t STAGE = SAH * 2 > EPF * 4 ? SAH * 2 : EPF * 4;
  const size_t smem = STAGE + (size_t)TAPC * CIN * BLD * 2 + (size_t)COUT * 8;
  auto fn = conv3k<CIN, COUT, TY, NW, WM, WN, TAPC, STR>;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 g(DIVUP(a.W, 16), DIVUP(a.H, TY), a.N);
  fn<<<g, 32 * NW, smem, st>>>(a.x, a.w, a.sc, a.sh, a.res, a.y, a.Hi, a.Wi,
                               a.H, a.W, a.sW, a.dW, a.srcC, a.dstC, a.resC,
                               a.act);
}

template <int CIN, int COUT, int TY, int NW, int WM, int WN, int KC, int INC,
          int OUTC>
static void go1(const FkArgs& a, cudaStream_t st) {
  constexpr int PS = CIN + 8, PLS = TY * 16 + 8, BLD = COUT + 8;
  constexpr int ELD = WN * 16 + 4;
  constexpr size_t SAH = INC ? (size_t)CIN * PLS : (size_t)TY * 16 * PS;
  constexpr size_t EPF = (size_t)NW * 16 * ELD;
  constexpr size_t STAGE = SAH * 2 > EPF * 4 ? SAH * 2 : EPF * 4;
  const size_t smem = STAGE + (size_t)KC * BLD * 2 + (size_t)COUT * 8;
  auto fn = conv1k<CIN, COUT, TY, NW, WM, WN, KC, INC, OUTC>;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 g(DIVUP(a.W, 16), DIVUP(a.H, TY), a.N);
  fn<<<g, 32 * NW, smem, st>>>(a.x, a.w, a.sc, a.sh, a.y, a.H, a.W, a.sW, a.dW,
                               a.srcC, a.dstC, a.act);
}

void fk_conv(int64_t kind, torch::Tensor x, torch::Tensor w, torch::Tensor sc,
             torch::Tensor sh, c10::optional<torch::Tensor> res,
             torch::Tensor y, std::vector<int64_t> meta) {
  // meta: N Hi Wi Ho Wo sW dW srcC dstC resC srcOff dstOff resOff act
  FkArgs a;
  TORCH_CHECK(meta.size() == 14, "fk_conv meta");
  a.x = (const __half*)x.const_data_ptr() + meta[10];
  a.w = (const __half*)w.const_data_ptr();
  a.sc = sc.const_data_ptr<float>();
  a.sh = sh.const_data_ptr<float>();
  a.res = res.has_value()
              ? (const __half*)res->const_data_ptr() + meta[12]
              : nullptr;
  a.y = (__half*)y.data_ptr() + meta[11];
  a.N = (int)meta[0]; a.Hi = (int)meta[1]; a.Wi = (int)meta[2];
  a.H = (int)meta[3]; a.W = (int)meta[4];
  a.sW = (int)meta[5]; a.dW = (int)meta[6];
  a.srcC = (int)meta[7]; a.dstC = (int)meta[8]; a.resC = (int)meta[9];
  a.act = (int)meta[13];
  auto st = at::cuda::getCurrentCUDAStream();
  switch (kind) {
    // 3x3 stride 1, channel-last -> channel-last:  CIN, COUT, TY, NW, WM, WN, TAPC
    case 0: go3<16, 16, 16, 8, 2, 1, 9, 1>(a, st); break;
    case 102: go3<16, 16, 16, 16, 1, 1, 9, 1>(a, st); break;
    case 110: go3<32, 32, 4, 8, 1, 1, 9, 1>(a, st); break;
    case 122: go3<64, 64, 4, 16, 1, 1, 9, 1>(a, st); break;
    // 1x1, NCHW -> channel-last (cv1):  CIN, COUT, TY, NW, WM, WN, KC
    case 10: go1<32, 32, 16, 8, 2, 2, 32, 1, 0>(a, st); break;
    case 140: go1<32, 32, 8, 8, 1, 2, 32, 1, 0>(a, st); break;
    case 150: go1<64, 64, 4, 8, 1, 2, 64, 1, 0>(a, st); break;
    case 162: go1<128, 128, 4, 16, 1, 2, 64, 1, 0>(a, st); break;
    // 1x1, channel-last -> NCHW (cv2)
    case 22: go1<256, 128, 4, 8, 1, 4, 64, 0, 1>(a, st); break;
    case 190: go1<128, 64, 4, 8, 1, 2, 64, 0, 1>(a, st); break;
    // 3x3 stride 2, channel-last -> channel-last (the stage downsamples)
    case 300: go3<32, 64, 4, 8, 1, 2, 9, 2>(a, st); break;
    case 302: go3<32, 64, 2, 8, 1, 1, 9, 2>(a, st); break;
    // 1x1, channel-last -> channel-last
    case 410: go1<48, 32, 16, 8, 2, 2, 48, 0, 0>(a, st); break;
    case 411: go1<48, 32, 8, 8, 1, 2, 48, 0, 0>(a, st); break;
    case 420: go1<64, 64, 4, 8, 1, 2, 64, 0, 0>(a, st); break;
    default: TORCH_CHECK(false, "fk_conv kind ", kind);
  }
}
"""

_CPP_SRC = r"""
void stem1_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor s1,
                   torch::Tensor b1, torch::Tensor y1, int64_t cfg1);
torch::Tensor stem2_forward(torch::Tensor y1, torch::Tensor w2, torch::Tensor s2,
                            torch::Tensor b2, int64_t cfg2);
void fk_conv(int64_t kind, torch::Tensor x, torch::Tensor w, torch::Tensor sc,
             torch::Tensor sh, c10::optional<torch::Tensor> res,
             torch::Tensor y, std::vector<int64_t> meta);
"""

_EXT = None
_EXT_FAILED = False

# The stem kernels' grids divide exactly (no bounds checks in them), and the
# tile the host picks depends on the batch, so the input extent has to clear the
# largest of them: stem1 tiles 16x32 of the H/2 grid, stem2 16x16 of the H/4 one.
_STEM_MOD = 64


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    return f"{major}.{minor}" + ("a" if major >= 9 else "")


def _ext():
    """The compiled extension, or ``None`` if it cannot be built here."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline

            arch = _arch_list()
            if arch:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
            _EXT = load_inline(
                name=f"fk_yolo_backbone_stem_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["stem1_forward", "stem2_forward", "fk_conv"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
        except Exception:
            _EXT_FAILED = True
    return _EXT


def _bn_affine(bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Eval-mode BatchNorm as fp32 per-channel ``(scale, shift)``."""
    scale = bn.weight.detach().float() / torch.sqrt(
        bn.running_var.detach().float() + bn.eps)
    shift = bn.bias.detach().float() - bn.running_mean.detach().float() * scale
    return scale.contiguous(), shift.contiguous()


def _plain_3x3_s2(cv: YOLOConv, cin: int, cout: int) -> bool:
    """Whether *cv* is the stride-2 3x3 conv + eval BN + SiLU the kernels assume."""
    c, bn = cv.conv, getattr(cv, "bn", None)
    if cv._is_fused or bn is None or cv.training or not bn.track_running_stats:
        return False
    if type(cv.act).__name__ != "SiLU":
        return False
    if tuple(c.weight.shape) != (cout, cin, 3, 3) or c.bias is not None:
        return False
    if c.groups != 1 or tuple(c.stride) != (2, 2) or tuple(c.padding) != (1, 1):
        return False
    if tuple(c.dilation) != (1, 1) or c.weight.dtype is not torch.float16:
        return False
    return all(t is not None and t.dim() == 1 and t.shape[0] == cout
               for t in (bn.weight, bn.bias, bn.running_mean, bn.running_var))

# The kernel case per C2f width, measured on a B200 at the captured shapes: a
# (3x3, cv1, cv2) triple for a block whose pixel count clears the threshold and
# one for below it, plus the pixel count under which the frozen whole-block
# winner is still faster (its tiling amortizes the weight staging better once the
# spatial extent shrinks to 40x40 and 20x20, where there is not enough of a grid
# left to hide a staging latency).
_C2F_CASE = {
    # (c, channel-last in, channel-last out):
    #     (min pixels, (3x3, cv1, cv2) kinds when big, kinds when small)
    (16, False, True): (51200, (0, 10, 410), (102, 140, 411)),
    (32, True, False): (0, (110, 420, 190), (110, 420, 190)),
    (32, False, False): (0, (110, 150, 190), (110, 150, 190)),
    (64, False, False): (3200, (122, 162, 22), None),
    (128, False, False): (1 << 62, None, None),
}

# 3x3 stride-2 downsample (channel-last both sides), by pixel count of its input.
_DOWN3_CASE = (51200, 300, 302)


def _pack_c2f_1x1(cv):
    """[COUT, CIN, 1, 1] -> [CIN][COUT + 8]: k-major, n-minor, output padded so
    that the 16 rows of a B-fragment load spread over all 32 banks."""
    w = cv.conv.weight.detach()
    co, ci = w.shape[0], w.shape[1]
    t = torch.zeros(ci, co + 8, device=w.device, dtype=w.dtype)
    t[:, :co] = w.reshape(co, ci).t()
    s, b = _bn_affine(cv.bn)
    return t.contiguous(), s, b


def _pack_c2f_3x3(cv):
    """[COUT, CIN, 3, 3] -> [tap][CIN][COUT + 8]."""
    w = cv.conv.weight.detach()
    co, ci = w.shape[0], w.shape[1]
    t = torch.zeros(9, ci, co + 8, device=w.device, dtype=w.dtype)
    t[:, :, :co] = w.permute(2, 3, 1, 0).reshape(9, ci, co)
    s, b = _bn_affine(cv.bn)
    return t.contiguous(), s, b


def _plain_conv(cv, k: int) -> bool:
    c, bn = cv.conv, getattr(cv, "bn", None)
    if cv._is_fused or bn is None or cv.training or not bn.track_running_stats:
        return False
    if type(cv.act).__name__ != "SiLU" or c.bias is not None:
        return False
    if c.weight.dim() != 4 or c.weight.shape[2] != k or c.weight.shape[3] != k:
        return False
    if c.groups != 1 or tuple(c.stride) != (1, 1):
        return False
    if tuple(c.padding) != (k // 2, k // 2) or tuple(c.dilation) != (1, 1):
        return False
    return c.weight.dtype is torch.float16 and bn.weight is not None


class _C2FPlan:
    """One YOLOC2f block lowered to a fixed sequence of ``fk_conv`` launches.

    ``cv1`` writes the head of a channel-last concat buffer and bottleneck *i*
    writes slot ``2 + i`` with its residual read from slot ``1 + i``, so chunk(),
    cat() and the residual add are all addressing, not kernels.  The buffer's rows
    are padded out to a multiple of 16 pixels and left zero there, which keeps a
    16-pixel m-tile inside one row and gives the 3x3 halo the zeros it wants at
    the right edge.
    """

    def __init__(self, block, in_cl: bool = False, out_cl: bool = False):
        self.block = block
        self.in_cl, self.out_cl = in_cl, out_cl
        self.c = c = block.c
        self.n = n = len(block.m)
        self.cat_c = (2 + n) * c
        self.cin = block.cv1.conv.weight.shape[1]
        self.cout = block.cv2.conv.weight.shape[0]
        self.ws = {}
        case = _C2F_CASE.get((c, in_cl, out_cl))
        self.case = case
        self.ok = (case is not None and case[1] is not None
                   and _plain_conv(block.cv1, 1) and _plain_conv(block.cv2, 1)
                   and block.cv1.conv.weight.shape[0] == 2 * c
                   and all(_plain_conv(b.cv1, 3) and _plain_conv(b.cv2, 3)
                           and b.cv1.conv.weight.shape[0] == c
                           and b.cv2.conv.weight.shape[0] == c and b.add
                           for b in block.m))
        if not self.ok:
            return
        self.cv1 = _pack_c2f_1x1(block.cv1)
        self.cv2 = _pack_c2f_1x1(block.cv2)
        self.mid = [(_pack_c2f_3x3(b.cv1), _pack_c2f_3x3(b.cv2)) for b in block.m]

    def runnable(self, x: torch.Tensor, hw) -> bool:
        if not (self.ok and x.is_cuda and x.dtype is torch.float16
                and x.dim() == 4 and x.is_contiguous()):
            return False
        if x.size(3 if self.in_cl else 1) != self.cin:
            return False
        # an NCHW side is read/written 8 halves at a time, which needs its row
        # stride to be a multiple of four
        if hw[1] % 4:
            return False
        px = x.size(0) * hw[0] * hw[1]
        return px >= self.case[0] or self.case[2] is not None

    def _buf(self, key, shape, dev):
        t = self.ws.get(key)
        if t is None or tuple(t.shape) != shape:
            # zeros: the padding columns are never written and must stay zero
            t = self.ws[key] = torch.zeros(shape, device=dev, dtype=torch.float16)
        return t

    def run(self, x: torch.Tensor, hw) -> torch.Tensor:
        # hw is the *real* spatial extent: a channel-last buffer's rows are
        # padded out to a multiple of 16 pixels, so its shape cannot say.
        N, (H, W) = x.size(0), hw
        sW = x.size(2) if self.in_cl else W
        WP = -(-W // 16) * 16
        c, n, cc = self.c, self.n, self.cat_c
        k3, k1, k2 = (self.case[1] if N * H * W >= self.case[0]
                      else self.case[2])
        cat = self._buf("cat", (N, H, WP, cc), x.device)
        tmp = self._buf("tmp", (N, H, WP, c), x.device)
        if self.out_cl:
            out = self._buf("out", (N, H, WP, self.cout), x.device)
            oW = WP
        else:
            out = torch.empty((N, self.cout, H, W), device=x.device,
                              dtype=torch.float16)
            oW = W
        w, s, b = self.cv1
        # meta: N Hin Win Hout Wout srcRowStride dstRowStride srcC dstC resC
        #       srcOff dstOff resOff act
        _EXT.fk_conv(k1, x, w, s, b, None, cat,
                     [N, H, W, H, W, sW, WP, self.cin, cc, 0, 0, 0, 0, 1])
        for i, ((w1, s1, b1), (w2, s2, b2)) in enumerate(self.mid):
            _EXT.fk_conv(k3, cat, w1, s1, b1, None, tmp,
                         [N, H, W, H, W, WP, WP, cc, c, 0, (1 + i) * c, 0, 0, 1])
            _EXT.fk_conv(k3, tmp, w2, s2, b2, cat, cat,
                         [N, H, W, H, W, WP, WP, c, cc, cc, 0, (2 + i) * c,
                          (1 + i) * c, 1])
        w, s, b = self.cv2
        _EXT.fk_conv(k2, cat, w, s, b, None, out,
                     [N, H, W, H, W, WP, oW, cc, self.cout, 0, 0, 0, 0, 1])
        return out

class YOLOv10Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = YOLOConv(3, 16, 3, 2)
        self.stem2 = YOLOConv(16, 32, 3, 2)
        self.stage2 = YOLOC2f(32, 32, n=1, shortcut=True)
        self.down3 = YOLOConv(32, 64, 3, 2)
        self.stage3 = YOLOC2f(64, 64, n=2, shortcut=True)
        self.down4 = YOLOSCDown(64, 128, 3, 2)
        self.stage4 = YOLOC2f(128, 128, n=2, shortcut=True)
        self.down5 = YOLOSCDown(128, 256, 3, 2)
        self.stage5 = YOLOC2f(256, 256, n=1, shortcut=True)
        self.sppf = YOLOSPPF(256, 256, 5)
        self.psa = YOLOPSA(256, 256)
        self._pack = None       # packed stem weights, or False when ineligible
        self._y1 = None         # stem1's output, written outside the graph
        self._plans = None      # per-C2f-stage lowered plans
        self._d3 = None         # packed down3 weights
        self._ws = {}           # persistent channel-last workspaces
        self._graph = None      # (CUDAGraph, static_in, (p3, p4, p5))
        self._gkey = None
        self._gfail = False

    # -- fused stem ---------------------------------------------------------
    def _invalidate(self):
        self._pack = None
        self._y1 = None
        self._plans = None
        self._d3 = None
        self._ws = {}
        self._graph = None
        self._gkey = None

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate()
        return super()._load_from_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self._invalidate()
        return super().train(mode)

    def _build_pack(self):
        """Pack the two stems' weights into the layouts the kernels read.

        ``w1`` becomes ``[kr][ks * 4 + c][co]``: the staged input tile is
        channel-last padded to four, so one k-tile of 16 halves covers the three
        taps of kernel row ``kr`` (9 real values) plus one dead pixel, whose
        weights are the zeros in the padding.  ``w2`` becomes ``[tap][cin][cout]``
        with the output dimension padded to 40 so that the 16 rows of a
        ``load_matrix_sync`` spread over all 32 banks.
        """
        if not (_plain_3x3_s2(self.stem1, 3, 16)
                and _plain_3x3_s2(self.stem2, 16, 32)):
            return False
        if _ext() is None:
            return False
        w1 = self.stem1.conv.weight.detach()
        dev, dt = w1.device, w1.dtype
        t = torch.zeros(3, 4, 4, 16, device=dev, dtype=dt)
        t[:, :3, :3, :] = w1.permute(2, 3, 1, 0)
        s1, b1 = _bn_affine(self.stem1.bn)
        w2 = self.stem2.conv.weight.detach()
        u = torch.zeros(9, 16, 40, device=dev, dtype=dt)
        u[:, :, :32] = w2.permute(2, 3, 1, 0).reshape(9, 16, 32)
        s2, b2 = _bn_affine(self.stem2.bn)
        return (t.reshape(3, 16, 16).contiguous(), s1, b1,
                u.contiguous(), s2, b2)

    def _stem_ok(self, x: torch.Tensor) -> bool:
        if not (x.is_cuda and x.dim() == 4 and x.dtype is torch.float16
                and x.is_contiguous() and x.size(1) == 3):
            return False
        return not (x.size(2) % _STEM_MOD or x.size(3) % _STEM_MOD)

    def _y1buf(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.size(0), x.size(2) // 2, x.size(3) // 2, 16)
        t = self._y1
        if t is None or tuple(t.shape) != shape or t.device != x.device:
            t = self._y1 = torch.empty(shape, device=x.device, dtype=x.dtype)
        return t

    def _stem_pack(self, x: torch.Tensor):
        pack = self._pack
        if pack is None:
            pack = self._pack = self._build_pack()
        return pack if (pack is not False and self._stem_ok(x)) else None

    def _stem1(self, x: torch.Tensor):
        """stem1, outside any graph: the one op that reads the caller's tensor."""
        pack = self._stem_pack(x)
        if pack is None:
            return self.stem1(x)
        y1 = self._y1buf(x)
        _EXT.stem1_forward(x, pack[0], pack[1], pack[2], y1, -1)
        return y1

    def _stem2(self, y1: torch.Tensor) -> torch.Tensor:
        # y1 is stem1's channel-last output only when it *is* the buffer stem1
        # writes into; otherwise stem1 fell back and handed back NCHW.
        if y1 is not self._y1:
            return self.stem2(y1)
        pack = self._pack
        return _EXT.stem2_forward(y1, pack[3], pack[4], pack[5], -1)

    def _c2f(self, name: str, block, x, hw, in_cl=False, out_cl=False):
        plans = self._plans
        if plans is None:
            plans = self._plans = {}
        plan = plans.get(name)
        if plan is None:
            plan = plans[name] = (_C2FPlan(block, in_cl, out_cl)
                                  if _ext() is not None else False)
        if plan is not False and plan.runnable(x, hw):
            return plan.run(x, hw), True
        return block(x), False

    def _down3_pack(self):
        """down3's 3x3 stride-2 weights, in the same [tap][cin][cout + 8] layout
        the C2f 3x3 uses.  Its input is stage2's output, which the block above
        leaves channel-last, so the whole downsample is one more fk_conv."""
        cv = self.down3
        if not _plain_3x3_s2(cv, 32, 64):
            return False
        w = cv.conv.weight.detach()
        t = torch.zeros(9, 32, 72, device=w.device, dtype=w.dtype)
        t[:, :, :64] = w.permute(2, 3, 1, 0).reshape(9, 32, 64)
        s, b = _bn_affine(cv.bn)
        return t.contiguous(), s, b

    def _down3_run(self, p2, H, W):
        pack = self._d3
        if pack is None:
            pack = self._d3 = self._down3_pack()
        if pack is False or H % 2 or W % 2:
            return None
        N = p2.size(0)
        ho, wo = H // 2, W // 2
        wp = -(-wo // 16) * 16
        out = self._c2f_buf("d3", (N, ho, wp, 64), p2.device)
        kind = _DOWN3_CASE[1] if N * H * W >= _DOWN3_CASE[0] else _DOWN3_CASE[2]
        _EXT.fk_conv(kind, p2, pack[0], pack[1], pack[2], None, out,
                     [N, H, W, ho, wo, p2.size(2), wp, 32, 64, 0, 0, 0, 0, 1])
        return out

    def _c2f_buf(self, key, shape, dev):
        ws = self._ws
        t = ws.get(key)
        if t is None or tuple(t.shape) != shape:
            t = ws[key] = torch.zeros(shape, device=dev, dtype=torch.float16)
        return t

    # -- the plain composition (reference, warmup and fallback path) --------
    def _eager(self, y1: torch.Tensor):
        x = self._stem2(y1)
        h, w = x.size(2), x.size(3)
        p2, cl = self._c2f("stage2", self.stage2, x, (h, w), out_cl=True)
        d3 = self._down3_run(p2, h, w) if cl else None
        if d3 is None:
            if cl:
                p2 = p2[:, :, :w].permute(0, 3, 1, 2).contiguous()
            x = self.down3(p2)
            p3, _ = self._c2f("stage3n", self.stage3, x,
                              (x.size(2), x.size(3)))
        else:
            h, w = h // 2, w // 2
            p3, _ = self._c2f("stage3", self.stage3, d3, (h, w), in_cl=True)
        x = self.down4(p3)
        p4, _ = self._c2f("stage4", self.stage4, x, (x.size(2), x.size(3)))
        x = self.down5(p4)
        p5, _ = self._c2f("stage5", self.stage5, x, (x.size(2), x.size(3)))
        p5 = self.sppf(p5)
        p5 = self.psa(p5)
        return p3, p4, p5

    # -- graph capture ------------------------------------------------------
    def _key(self, x: torch.Tensor):
        return (tuple(x.shape), x.dtype, x.device)

    def _capture(self, x: torch.Tensor):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._eager(self._stem1(x))
        torch.cuda.current_stream().wait_stream(side)
        y1 = self._stem1(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = self._eager(y1)
        return graph, y1, tuple(outs)

    def _graph_ok(self, x: torch.Tensor) -> bool:
        return (not self.training and x.is_cuda and x.dim() == 4
                and x.is_contiguous() and not self._gfail
                and not torch.cuda.is_current_stream_capturing())

    def forward(self, x: torch.Tensor):
        g, y1 = self._graph, None
        if g is not None and self._gkey == self._key(x) and x.is_contiguous():
            y1 = self._stem1(x)          # writes the buffer the graph captured
            if y1 is g[1]:
                g[0].replay()
                return dict(zip(_KEYS, g[2]))
        if y1 is None and self._graph_ok(x) and self._stem_pack(x) is not None:
            with torch.no_grad():
                try:
                    g = self._capture(x)
                except Exception:
                    self._gfail = True
                else:
                    self._graph, self._gkey = g, self._key(x)
                    g[0].replay()
                    return dict(zip(_KEYS, g[2]))
        if y1 is None:
            y1 = self._stem1(x)
        return dict(zip(_KEYS, self._eager(y1)))
