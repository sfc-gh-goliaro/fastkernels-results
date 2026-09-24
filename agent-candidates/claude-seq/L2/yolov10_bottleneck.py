"""YOLOv10 bottleneck block -- the whole block in one hand-written CUDA kernel.

The captured shapes are tiny (the largest, 4x16x160x160, is ~1 GFLOP for the
whole block), so every case is latency bound: the baseline spends ~110 us of
*CPU* time enqueueing seven library ops (conv, batch-norm, SiLU, conv,
batch-norm, SiLU, add) while the GPU work is a few microseconds.  Three things
follow, and they shape this file:

* **One Python call, one C++ call, one launch.**  ``forward`` does no submodule
  calls, no ``F.*`` dispatches and no tensor bookkeeping beyond comparing a
  tuple of parameter versions; the weights are folded and registered with the
  extension once, and a handle plus the input is all the hot path passes.
* **Nothing but the convolutions survives as a kernel.**  BatchNorm in eval
  mode is a per-output-channel affine, so it folds into the convolution weights
  once; SiLU and the residual add become the convolutions' epilogues.
* **The phase joins are grid barriers, not launches.**  A launch costs ~2 us of
  CPU and ~3 us of GPU ramp here (a bare 400 KB elementwise kernel measures
  3.3 us), which is the same order as the work itself, so the kernel is
  persistent: repack, barrier, conv1, barrier, conv2.  The grid is sized from
  ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` so every block is resident
  and the barriers cannot deadlock.

Every captured init is 3x3, stride 1, pad 1, ``Cin == Cout == C`` with C in
{16, 32, 64, 128}, so one templated implicit GEMM serves both convolutions:

    out[p][co] = sum_{r,s,ci} x[h+r-1][w+s-1][ci] * wt[r][s][ci][co]

i.e. ``C[PQ, Cout] = A[PQ, 9*Cin] @ B[9*Cin, Cout]`` per image.  The activation
is the *A* operand, which is why phase 1 repacks it from NCHW into **zero-padded
NHWC** ([N][H+2][W+2][C]):

* The k axis (input channels) becomes the contiguous one, so a k-tile row is a
  16-byte-aligned run -- every staging load is a ``uint4`` handed to shared
  memory by ``cp.async``.  Gathering the same data from NCHW costs one 2-byte
  load per element: half of L1's per-cycle bytes wasted and ~10x the
  instructions (measured 22 us per convolution that way, versus ~2 us here).
* The 3x3 window shift ``(r-1, s-1)`` moves whole *rows* of A, so every address
  stays 16-byte aligned for any W, and the zero halo turns the shift into a
  plain pointer offset -- there is no bounds masking anywhere in the k loop.

The padded buffers are cached scratch, zeroed once when allocated; only pixel
interiors are ever written afterwards, so the halo stays zero.  Phase 3 writes
NCHW directly (``store_matrix_sync`` transposes the accumulator for free) and
folds in the residual with 16-byte loads.

A block owns every output channel of its pixel tile, so the free knobs are the
pixel tile, the pipeline depth, whether the weight matrix is resident in shared
memory for the block's lifetime, and how many blocks per SM the persistent grid
uses.  They trade weight traffic against SM coverage -- at 1x128x20x20 there are
only 400 pixels to spread over 148 SMs -- so the first call for a problem shape
measures the candidates and caches the winner.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from .yolov10_conv import YOLOConv

_CUDA_SRC = r"""
#include <torch/extension.h>

#include <array>
#include <cstdlib>
#include <map>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

#define DIVUP(a, b) (((a) + (b) - 1) / (b))

// Geometry of one bottleneck call.  Wp/Pp describe the zero-padded NHWC
// scratch layout: row stride W+2 pixels, image stride (H+2)*(W+2) pixels.
struct CP {
  int C, H, W, HW, Wp, Pp;
  unsigned Wm;      // magic multiplier: __umulhi(pixel, Wm) == pixel / W
  int nimg, G;      // images; resident blocks of the persistent grid
  int npack, ntile; // work items in the repack phase / per conv phase
  int nbx;          // pixel tiles per image (set by the launcher, needs BM)
  int WC;           // w-chunks per row in the repack phase
  // Pointers travel in the same struct: one kernel argument instead of
  // thirteen, which the launch path charges for individually.
  const __half* X;
  const __half* W1;
  const __half* W2;
  const float* B1;
  const float* B2;
  __half* Xp;
  __half* Mid;
  __half* Out;
  unsigned* bar;
  unsigned barbase;
  int add;
};

// pixel index -> (row, col) without an integer divide (pixels < 2^24 here).
__device__ __forceinline__ void rowcol(int pp, const CP& p, int& h, int& w) {
  h = (int)__umulhi((unsigned)pp, p.Wm);
  w = pp - h * p.W;
}

// silu(x) = x * sigmoid(x) = h * (1 + tanh(h)),  h = x/2  -- one MUFU op.
__device__ __forceinline__ float silu(float x) {
  const float h = 0.5f * x;
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return __fmaf_rn(h, t, h);
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}

// ---------------------------------------------------------------------------
// Grid barrier for the persistent grid.  The counter only ever increases (by G
// per barrier) and the comparison is a wrapped signed difference, so no sense
// reversal and no per-call reset is needed.  Safe because the launcher sizes the
// grid from cudaOccupancyMaxActiveBlocksPerMultiprocessor: every block is
// resident, so every block does reach the barrier.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void grid_barrier(unsigned* bar, int G, unsigned& tgt) {
  __syncthreads();                 // block's stores are ordered before the fence
  tgt += (unsigned)G;
  if (threadIdx.x == 0) {
    __threadfence();               // ... and pushed device-wide by the leader
    // Fire-and-forget arrival: a returning atomicAdd would stall the leader for
    // a full L2 round trip.  Polling uses volatile loads -- an acquire load
    // invalidates L1 on every poll, which throws away the activation reuse the
    // k loop depends on, and an atomic RMW per poll serialises all blocks.
    asm volatile("red.global.add.u32 [%0], 1;" ::"l"(bar) : "memory");
    unsigned v;
    do {
      __nanosleep(48);             // polling harder than this floods L2
      asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(bar));
    } while ((int)(v - tgt) < 0);
    __threadfence();               // acquire: later reads see the other blocks
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Phase 1 -- NCHW -> zero-padded NHWC repack.  One work item is (image, row,
// 32-column chunk); a warp owns one 8-channel group of 32 consecutive columns,
// so its eight loads are coalesced 64-byte lines and its store is one 16-byte
// chunk per pixel.  Going through shared memory instead would halve the store
// scatter but costs a second dependent round trip plus a block-wide barrier,
// and at these sizes the phase is pure latency.  Only interior pixels are
// written; the halo was zeroed when the scratch was allocated.
// ---------------------------------------------------------------------------
template <int NT, int C>
__device__ void pack_phase(const __half* __restrict__ X, __half* __restrict__ Y,
                           const CP& p) {
  constexpr int VC = C / 8;                      // 16B groups per pixel
  constexpr int PER = 32 * VC;                   // work units per (row, chunk)
  constexpr int PSH = VC == 2 ? 6 : (VC == 4 ? 7 : (VC == 8 ? 8 : 9));
  static_assert(PER == (1 << PSH), "PER must be a power of two");
  // Flat over (item, channel group, column) so every thread stays busy and all
  // the loads of a block are in flight together; a per-item loop would leave
  // 32*VC of the threads working and serialise one latency per item.
  const long total = (long)p.npack * PER;
  for (long u = (long)blockIdx.x * NT + threadIdx.x; u < total;
       u += (long)p.G * NT) {
    const int item = (int)(u >> PSH), r = (int)(u & (PER - 1));
    const int g = r >> 5, wl = r & 31;
    const int wc = item % p.WC, t = item / p.WC;
    const int h = t % p.H, img = t / p.H;
    const int w = wc * 32 + wl;
    if (w >= p.W) continue;
    const __half* sp = X + (long)img * C * p.HW + (long)h * p.W + w
                       + (long)g * 8 * p.HW;
    __half v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) v[e] = sp[e * p.HW];
    *reinterpret_cast<uint4*>(Y + (long)img * p.Pp * C
                              + ((long)(h + 1) * p.Wp + (w + 1)) * C + g * 8) =
        *reinterpret_cast<const uint4*>(v);
  }
}

// ---------------------------------------------------------------------------
// Phases 2/3 -- one 3x3 stride-1 pad-1 implicit-GEMM tile on padded NHWC,
// plus bias, SiLU and (optionally) the residual add.
//
//   C[BM pixels, BN=C channels] = A[BM, BK] @ B[BK, BN], accumulated over
//   NTILE = 9 * (C / BK) k-tiles.  A k-tile is one (r, s) and a block of input
//   channels: its A rows are contiguous 16B-aligned runs of the padded
//   activation and its B rows are contiguous rows of the [9*C][C] weight
//   matrix, so every staging load is a 16-byte cp.async and the k loop needs no
//   bounds masking at all (the halo supplies the zeros).
//
//   The loop is fully unrolled, which turns all shared-memory offsets and the
//   weight offset into immediates and leaves the activation offset as one of
//   three hoisted row strides.  With only 9-18 tiles and a handful of resident
//   warps per scheduler at these problem sizes, the pipeline is sized to keep
//   every tile in flight where shared memory allows.
//
//   omode 0: write padded NHWC (feeds the second convolution).
//   omode 1: write NCHW and add the residual -- store_matrix_sync transposes
//            the accumulator for free, so stores still run along pixels.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int WM, int WN, int STAGES, int WSH>
__device__ void conv_tile(const __half* __restrict__ A, const __half* __restrict__ Wt,
                          const float* __restrict__ Bias, const __half* __restrict__ Res,
                          __half* __restrict__ Out, const CP& p, int p0, int img,
                          int omode, __half* Ss, const __half* Wsh) {
  using namespace nvcuda::wmma;
  constexpr int NWM = BM / WM, NWN = BN / WN, NW = NWM * NWN, NT = 32 * NW;
  constexpr int LDA = BK + 8, LDB = BN + 8;
  constexpr int SA = BM * LDA, SB = WSH ? 0 : BK * LDB;
  constexpr int VA = BK / 8, VB = BN / 8;
  constexpr int TAV = BM * VA, TBV = BK * VB;      // 16B vectors per tile
  constexpr int NAV = DIVUP(TAV, NT), NBV = DIVUP(TBV, NT);
  constexpr int NCT = BN / BK, NTILE = 9 * NCT;
  static_assert(BK % 16 == 0 && WM % 16 == 0 && WN % 16 == 0, "mma shape");
  static_assert(BM % 16 == 0 && BN % 16 == 0, "block shape");
  static_assert(BN % BK == 0, "k tile must divide the channel count");
  static_assert(STAGES >= 2 && STAGES <= 10, "pipeline depth");
  static_assert(NTILE <= 18, "unrolled k loop covers at most 18 tiles");

  float* Ep = reinterpret_cast<float*>(Ss);        // epilogue reuses the staging
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int wm = warp / NWN, wn = warp % NWN;

  fragment<accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; i++)
#pragma unroll
    for (int j = 0; j < WN / 16; j++) fill_fragment(acc[i][j], 0.0f);

  // ---- staging addresses, hoisted out of the k loop -----------------------
  // A vector i of this thread: pixel row (idx / VA), 16B column (idx % VA).
  const __half* Ab = A + (long)img * p.Pp * p.C;
  const __half* apt[NAV];
  int adst[NAV];
#pragma unroll
  for (int i = 0; i < NAV; i++) {
    const int idx = tid + i * NT;
    const int m = idx / VA, v = idx - (idx / VA) * VA;
    const int pp = p0 + m < p.HW ? p0 + m : p.HW - 1;
    int h, w;
    rowcol(pp, p, h, w);
    apt[i] = Ab + (((h + 1) * p.Wp + (w + 1)) * BN + v * 8);
    adst[i] = m * LDA + v * 8;
    if (TAV % NT != 0 && idx >= TAV) apt[i] = nullptr;
  }
  const __half* bpt[NBV];
  int bdst[NBV];
  if (!WSH) {
#pragma unroll
    for (int i = 0; i < NBV; i++) {
      const int idx = tid + i * NT;
      const int k = idx / VB, v = idx - (idx / VB) * VB;
      bpt[i] = Wt + (k * BN + v * 8);
      bdst[i] = k * LDB + v * 8;
      if (TBV % NT != 0 && idx >= TBV) bpt[i] = nullptr;
    }
  }
  const int rowC = p.Wp * BN;                      // one padded row, in halves

#define FK_ISSUE(IT)                                                          \
  {                                                                           \
    constexpr int it_ = (IT);                                                 \
    constexpr int cit_ = it_ % NCT, trs_ = it_ / NCT;                         \
    constexpr int tr_ = trs_ / 3, ts_ = trs_ % 3;                             \
    constexpr int buf_ = it_ % STAGES;                                        \
    constexpr int boff_ = (trs_ * BN + cit_ * BK) * BN;                       \
    const int aoff_ = (tr_ - 1) * rowC + ((ts_ - 1) * BN + cit_ * BK);        \
    __half* sa_ = Ss + buf_ * (SA + SB);                                      \
    __half* sb_ = sa_ + SA;                                                   \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < NAV; i++)                                             \
      if (TAV % NT == 0 || apt[i] != nullptr)                                 \
        cp_async16(sa_ + adst[i], apt[i] + aoff_);                            \
    if (!WSH) {                                                               \
      _Pragma("unroll")                                                       \
      for (int i = 0; i < NBV; i++)                                           \
        if (TBV % NT == 0 || bpt[i] != nullptr)                               \
          cp_async16(sb_ + bdst[i], bpt[i] + boff_);                          \
    }                                                                         \
  }

#define FK_MMA(IT)                                                            \
  {                                                                           \
    constexpr int buf_ = (IT) % STAGES;                                       \
    constexpr int cit2_ = (IT) % NCT, trs2_ = (IT) / NCT;                     \
    const __half* sa = Ss + buf_ * (SA + SB);                                 \
    const __half* sb = WSH ? Wsh + (trs2_ * BN + cit2_ * BK) * LDB : sa + SA;  \
    _Pragma("unroll")                                                         \
    for (int kt = 0; kt < BK / 16; kt++) {                                    \
      fragment<matrix_a, 16, 16, 16, __half, row_major> af[WM / 16];          \
      fragment<matrix_b, 16, 16, 16, __half, row_major> bf[WN / 16];          \
      _Pragma("unroll")                                                       \
      for (int i = 0; i < WM / 16; i++)                                       \
        load_matrix_sync(af[i], sa + (wm * WM + i * 16) * LDA + kt * 16, LDA); \
      _Pragma("unroll")                                                       \
      for (int j = 0; j < WN / 16; j++)                                       \
        load_matrix_sync(bf[j], sb + (kt * 16) * LDB + wn * WN + j * 16, LDB); \
      _Pragma("unroll")                                                       \
      for (int i = 0; i < WM / 16; i++)                                       \
        _Pragma("unroll")                                                     \
        for (int j = 0; j < WN / 16; j++)                                     \
          mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);                       \
    }                                                                         \
  }

  // prologue: STAGES-1 groups committed (empty ones included, so the
  // wait_group accounting holds even when STAGES-1 exceeds the tile count).
#define FK_PRE(S)                                                             \
  {                                                                           \
    if ((S) < STAGES - 1) {                                                   \
      if ((S) < NTILE) { FK_ISSUE((S) % NTILE) }                              \
      asm volatile("cp.async.commit_group;\n" ::);                            \
    }                                                                         \
  }
  __syncthreads();                  // previous tile's epilogue is done reading
  FK_PRE(0) FK_PRE(1) FK_PRE(2) FK_PRE(3) FK_PRE(4)
  FK_PRE(5) FK_PRE(6) FK_PRE(7) FK_PRE(8)
#undef FK_PRE

#define FK_STEP(IT)                                                           \
  {                                                                           \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));              \
    __syncthreads();                                                          \
    if ((IT) + STAGES - 1 < NTILE) { FK_ISSUE(((IT) + STAGES - 1) % NTILE) }   \
    asm volatile("cp.async.commit_group;\n" ::);                              \
    FK_MMA(IT)                                                                \
  }
#define FK_STEP_IF(IT) { if ((IT) < NTILE) FK_STEP(IT) }
  FK_STEP_IF(0)  FK_STEP_IF(1)  FK_STEP_IF(2)  FK_STEP_IF(3)  FK_STEP_IF(4)
  FK_STEP_IF(5)  FK_STEP_IF(6)  FK_STEP_IF(7)  FK_STEP_IF(8)  FK_STEP_IF(9)
  FK_STEP_IF(10) FK_STEP_IF(11) FK_STEP_IF(12) FK_STEP_IF(13) FK_STEP_IF(14)
  FK_STEP_IF(15) FK_STEP_IF(16) FK_STEP_IF(17)
#undef FK_STEP_IF
#undef FK_STEP
#undef FK_MMA
#undef FK_ISSUE
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();

  // ---- epilogue: bias -> SiLU -> (+ residual) -> fp16 ---------------------
  float* ep = Ep + warp * 16 * WN;
  const bool vec = (p.HW & 7) == 0;
#pragma unroll
  for (int i = 0; i < WM / 16; i++) {
    const int m0 = wm * WM + i * 16;
    if (omode == 0) {
#pragma unroll
      for (int j = 0; j < WN / 16; j++)
        store_matrix_sync(ep + j * 16, acc[i][j], WN, mem_row_major);
      __syncwarp();
      __half* Ob = Out + (long)img * p.Pp * p.C;
      constexpr int RUNS = 16 * (WN / 8);
#pragma unroll
      for (int t = 0; t < DIVUP(RUNS, 32); t++) {
        const int run = lane + t * 32;
        if (RUNS % 32 != 0 && run >= RUNS) break;
        const int rr = run / (WN / 8), cc = (run - rr * (WN / 8)) * 8;
        const int pp = p0 + m0 + rr;
        if (pp >= p.HW) continue;
        int h, w;
        rowcol(pp, p, h, w);
        const float* sp = ep + rr * WN + cc;
        const float* bp = Bias + wn * WN + cc;
        __half v[8];
#pragma unroll
        for (int e = 0; e < 8; e++) v[e] = __float2half(silu(sp[e] + bp[e]));
        *reinterpret_cast<uint4*>(Ob + ((h + 1) * p.Wp + (w + 1)) * BN
                                  + wn * WN + cc) =
            *reinterpret_cast<const uint4*>(v);
      }
    } else {
#pragma unroll
      for (int j = 0; j < WN / 16; j++)
        store_matrix_sync(ep + j * 16 * 16, acc[i][j], 16, mem_col_major);
      __syncwarp();
      __half* Ob = Out + (long)img * p.C * p.HW;
      const __half* Rb = Res == nullptr ? nullptr : Res + (long)img * p.C * p.HW;
      constexpr int RUNS = WN * 2;                 // WN channels x 2 8-pixel runs
#pragma unroll
      for (int t = 0; t < DIVUP(RUNS, 32); t++) {
        const int run = lane + t * 32;
        if (RUNS % 32 != 0 && run >= RUNS) break;
        const int nn = run >> 1, g = (run & 1) * 8;
        const int co = wn * WN + nn;
        const int pp = p0 + m0 + g;
        const float* sp = ep + nn * 16 + g;
        const float bias = Bias[co];
        const long ob = (long)co * p.HW + pp;
        if (vec && pp + 8 <= p.HW) {
          __half v[8];
          if (Rb != nullptr) {
            const uint4 rv = *reinterpret_cast<const uint4*>(Rb + ob);
            const __half* rh = reinterpret_cast<const __half*>(&rv);
#pragma unroll
            for (int e = 0; e < 8; e++)
              v[e] = __float2half(silu(sp[e] + bias) + __half2float(rh[e]));
          } else {
#pragma unroll
            for (int e = 0; e < 8; e++) v[e] = __float2half(silu(sp[e] + bias));
          }
          *reinterpret_cast<uint4*>(Ob + ob) = *reinterpret_cast<const uint4*>(v);
        } else {
#pragma unroll
          for (int e = 0; e < 8; e++)
            if (pp + e < p.HW) {
              float f = silu(sp[e] + bias);
              if (Rb != nullptr) f += __half2float(Rb[ob + e]);
              Ob[ob + e] = __float2half(f);
            }
        }
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// The whole bottleneck in one launch: repack, barrier, conv1, barrier, conv2.
// Three launches cost ~3 us of GPU ramp each at these sizes (a bare 400 KB
// elementwise kernel measures 3.3 us) plus ~2 us of CPU each, which is the same
// order as the actual work -- so the grid is persistent and the phase joins are
// grid barriers instead of launches.
// ---------------------------------------------------------------------------
// Cooperative 16B copy of the [9*C][C] weight matrix into shared, padded to
// ld = C+8 so the mma's row addresses stay 16B aligned and spread over banks.
template <int BN, int NT>
__device__ __forceinline__ void issue_weights(const __half* __restrict__ Wt,
                                              __half* Wsh) {
  constexpr int VB = BN / 8, WV = 9 * BN * VB;
  for (int i = threadIdx.x; i < WV; i += NT) {
    const int row = i / VB, col = i - row * VB;
    cp_async16(Wsh + row * (BN + 8) + col * 8, Wt + row * BN + col * 8);
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES, int WSH>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN))
void bneck_kernel(CP p) {
  constexpr int NT = 32 * (BM / WM) * (BN / WN);
  constexpr int WSZ = WSH ? 9 * BN * (BN + 8) : 0;      // resident weight matrix
  extern __shared__ __align__(16) char raw[];
  __half* Wsh = reinterpret_cast<__half*>(raw);
  __half* Ss = Wsh + WSZ;                               // mma staging + epilogue
  unsigned tgt = p.barbase;

  // The weight load does not depend on the repack, so it is issued first and
  // waited for after the barrier: it costs nothing on the critical path.
  if (WSH) issue_weights<BN, NT>(p.W1, Wsh);
  pack_phase<NT, BN>(p.X, p.Xp, p);
  grid_barrier(p.bar, p.G, tgt);
  if (WSH) {
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
  }

  for (int item = blockIdx.x; item < p.ntile; item += p.G) {
    const int bx = item % p.nbx, img = item / p.nbx;
    conv_tile<BM, BN, BK, WM, WN, STAGES, WSH>(p.Xp, p.W1, p.B1, nullptr, p.Mid, p,
                                               bx * BM, img, 0, Ss, Wsh);
  }
  if (WSH) {                    // the second layer's weights load while the
    __syncthreads();            // barrier below drains
    issue_weights<BN, NT>(p.W2, Wsh);
  }
  grid_barrier(p.bar, p.G, tgt);
  if (WSH) {
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
  }

  for (int item = blockIdx.x; item < p.ntile; item += p.G) {
    const int bx = item % p.nbx, img = item / p.nbx;
    conv_tile<BM, BN, BK, WM, WN, STAGES, WSH>(p.Mid, p.W2, p.B2,
                                               p.add ? p.X : nullptr, p.Out, p,
                                               bx * BM, img, 1, Ss, Wsh);
  }
}

// ---------------------------------------------------------------------------
// Launch table.  BN = C, so a block owns every output channel of its pixel tile;
// the entries span the pixel tile BM (parallelism versus weight traffic), the
// warp tile (WM x WN), the k-tile BK, the pipeline depth and whether the weights
// stay resident -- all of which the shared-memory budget couples together, so
// the autotuner picks per problem shape.
// ---------------------------------------------------------------------------
//        id,   C,  BM,  BK, WM, WN, STAGES, WSH
// WSH keeps the whole 9*C x C weight matrix in shared memory for the block's
// lifetime, which takes the weights out of the k loop entirely; at C=128 they
// are 294 KB, so those configs stream them per k-tile instead.
#define FK_CFG_LIST                                                           \
  Y(0,   16, 128,  16, 16, 16, 10, 1)                                         \
  Y(1,   16,  64,  16, 16, 16, 10, 1)                                         \
  Y(2,   16,  32,  16, 16, 16, 10, 1)                                         \
  Y(3,   16,  16,  16, 16, 16, 10, 1)                                         \
  Y(4,   32, 128,  32, 16, 32, 6,  1)                                         \
  Y(5,   32,  64,  32, 16, 32, 10, 1)                                         \
  Y(6,   32,  32,  32, 16, 32, 10, 1)                                         \
  Y(7,   32,  16,  32, 16, 32, 10, 1)                                         \
  Y(8,   64, 128,  64, 32, 32, 4,  1)                                         \
  Y(9,   64,  64,  64, 32, 32, 4,  1)                                         \
  Y(10,  64,  32,  64, 16, 32, 6,  1)                                         \
  Y(11,  64,  16,  64, 16, 32, 8,  1)                                         \
  Y(12, 128, 128,  64, 32, 32, 3,  0)                                         \
  Y(13, 128,  64,  64, 32, 32, 4,  0)                                         \
  Y(14, 128,  32,  64, 16, 32, 5,  0)                                         \
  Y(15, 128,  16,  64, 16, 32, 6,  0)                                         \
  Y(16,  16, 128,  16, 16, 16, 4,  1)                                         \
  Y(17,  32,  32,  32, 16, 16, 10, 1)                                         \
  Y(18,  32,  64,  32, 16, 16, 10, 1)                                         \
  Y(19,  64,  32,  64, 16, 16, 3,  1)                                         \
  Y(20,  64,  64,  64, 32, 32, 2,  1)                                         \
  Y(21, 128,  16, 128, 16, 32, 4,  0)                                         \
  Y(22, 128,  32,  64, 16, 16, 5,  0)                                         \
  Y(23, 128,  16,  64, 16, 16, 6,  0)

template <int BM, int BN, int BK, int WM, int WN, int STAGES, int WSH>
static int conv_smem() {
  constexpr int NW = (BM / WM) * (BN / WN);
  constexpr int STAGE =
      (BM * (BK + 8) + (WSH ? 0 : BK * (BN + 8))) * (int)sizeof(__half);
  constexpr int EP = NW * 16 * WN * 4;
  constexpr int WSZ = (WSH ? 9 * BN * (BN + 8) : 0) * (int)sizeof(__half);
  return WSZ + (STAGES * STAGE > EP ? STAGES * STAGE : EP);
}

// Blocks that stay resident together (the grid barrier requires it), cached per
// config: the occupancy query itself costs more than a launch.
static int resident_blocks(int id, int nt, int sh, const void* fn) {
  static std::map<int, int> cache;
  auto it = cache.find(id);
  if (it != cache.end()) return it->second;
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, nt, (size_t)sh);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int g = nb > 0 ? nb * sms : 0;
  cache[id] = g;
  return g;
}

// Blocks per SM to cap the persistent grid at; index 0 means "as many as fit".
static const int kCapList[] = {0, 1, 2, 4};
constexpr int kNCap = 4;

static bool launch_bneck(int code, CP p, unsigned& barbase, cudaStream_t st) {
  const int id = code % 100;
  const int capb = kCapList[(code / 100) % kNCap];
  switch (id) {
#define Y(ID, CC, BM, BK, WM, WN, ST, WS)                                     \
  case ID: {                                                                  \
    if (p.C != CC) return false;                                              \
    constexpr int NT = 32 * (BM / WM) * (CC / WN);                            \
    const int sh = conv_smem<BM, CC, BK, WM, WN, ST, WS>();                   \
    const void* fn =                                                          \
        reinterpret_cast<const void*>(&bneck_kernel<BM,CC,BK,WM,WN,ST,WS>);   \
    if (sh > 48 * 1024) {                                                     \
      static bool once = false;                                               \
      if (!once) {                                                            \
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sh); \
        once = true;                                                          \
      }                                                                       \
    }                                                                         \
    int gmax = resident_blocks(ID, NT, sh, fn);                               \
    if (gmax <= 0) return false;                                              \
    if (capb > 0) {                                                           \
      const int lim = capb * at::cuda::getCurrentDeviceProperties()            \
                                 ->multiProcessorCount;                        \
      if (gmax > lim) gmax = lim;                                             \
    }                                                                         \
    p.nbx = DIVUP(p.HW, BM);                                                  \
    p.ntile = p.nbx * p.nimg;                                                 \
    const int work = p.ntile > p.npack ? p.ntile : p.npack;                   \
    p.G = gmax < work ? gmax : work;                                          \
    p.barbase = barbase;                                                      \
    cudaGetLastError();                                                       \
    bneck_kernel<BM, CC, BK, WM, WN, ST, WS><<<p.G, NT, sh, st>>>(p);          \
    /* a rejected launch performs no arrivals, so only count accepted ones */  \
    if (cudaPeekAtLastError() != cudaSuccess) return false;                    \
    barbase += 2u * (unsigned)p.G;          /* two barriers, G arrivals each */ \
    return true;                                                              \
  }
    FK_CFG_LIST
#undef Y
    default:
      return false;
  }
}

static std::vector<int> cfg_candidates(int C, int HW) {
  std::vector<int> v;
  for (int c = 0; c < kNCap; c++) {
#define Y(ID, CC, BM, BK, WM, WN, ST, WS)                                     \
  if (C == CC && (BM <= 16 || HW > BM / 2)) v.push_back(ID + 100 * c);
    FK_CFG_LIST
#undef Y
  }
  return v;
}

// ---------------------------------------------------------------------------
// Scratch per problem shape: two zero-padded NHWC buffers (repacked input and
// the first convolution's output) plus the barrier counter.  Allocated and
// zeroed once; only pixel interiors are written afterwards, so the halo stays
// zero, and the counter only ever counts up.
// ---------------------------------------------------------------------------
struct Scratch {
  at::Tensor xp, mid, bar;
  __half* xpp;
  __half* midp;
  unsigned* barp;
  unsigned barbase;      // arrivals the counter has already seen
};
static std::map<std::array<int, 5>, Scratch> g_scratch;

static Scratch& scratch_for(const at::Tensor& x, const CP& p) {
  const std::array<int, 5> key = {p.nimg, p.C, p.H, p.W, (int)x.device().index()};
  auto it = g_scratch.find(key);
  if (it != g_scratch.end()) return it->second;
  Scratch s;
  s.xp = at::zeros({(long)p.nimg * p.Pp * p.C}, x.options());
  s.mid = at::zeros({(long)p.nimg * p.Pp * p.C}, x.options());
  s.bar = at::zeros({4}, x.options().dtype(at::kInt));
  s.xpp = (__half*)s.xp.data_ptr();
  s.midp = (__half*)s.mid.data_ptr();
  s.barp = (unsigned*)s.bar.data_ptr();
  s.barbase = 0u;
  return g_scratch.emplace(key, std::move(s)).first->second;
}

// ---------------------------------------------------------------------------
// Registered weights.  The hot path passes a handle, so a call converts one
// tensor and one integer instead of five tensors -- at these latencies the
// pybind argument marshalling is a measurable part of the cost.
// ---------------------------------------------------------------------------
struct Plan {
  at::Tensor w1, b1, w2, b2;
  const __half* w1p;
  const __half* w2p;
  const float* b1p;
  const float* b2p;
  bool add;
  int cfg;                        // tuned config, -1 until the first call
  std::array<int, 5> cfg_key;
  CP cp;                          // cached geometry for cfg_key
  Scratch* sc;
};
static std::vector<Plan> g_plans;

int64_t bneck_register(const at::Tensor& w1, const at::Tensor& b1,
                       const at::Tensor& w2, const at::Tensor& b2, bool add) {
  Plan pl;
  pl.w1 = w1; pl.b1 = b1; pl.w2 = w2; pl.b2 = b2; pl.add = add;
  pl.w1p = (const __half*)w1.data_ptr();
  pl.w2p = (const __half*)w2.data_ptr();
  pl.b1p = b1.data_ptr<float>();
  pl.b2p = b2.data_ptr<float>();
  pl.cfg = -1;
  pl.cfg_key = {0, 0, 0, 0, 0};
  pl.sc = nullptr;
  g_plans.push_back(std::move(pl));
  return (int64_t)g_plans.size() - 1;
}

// ---------------------------------------------------------------------------
// Autotune: the pixel tile trades weight traffic against SM coverage and the
// captured shapes sit on both sides of that trade (400 to 25600 pixels per
// image), so the first call for a shape times the candidates -- with an L2
// flush, matching how the op is benchmarked -- and the winner is cached.
// ---------------------------------------------------------------------------
template <typename F>
static int autotune(const std::vector<int>& cands, F&& run, const at::Tensor& x,
                    cudaStream_t st) {
  if (cands.empty()) return -1;
  if (cands.size() == 1) return cands[0];
  static at::Tensor flush;
  static size_t nflush = 0;
  if (!flush.defined() || flush.device() != x.device()) {
    try {
      nflush = 2 * (size_t)at::cuda::getCurrentDeviceProperties()->l2CacheSize;
      flush = at::empty({(long)nflush}, x.options().dtype(at::kByte));
    } catch (...) {
      nflush = 0;
      flush = at::Tensor();
    }
  }
  cudaEvent_t e0, e1;
  if (cudaEventCreate(&e0) != cudaSuccess) return cands[0];
  if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); return cands[0]; }
  int best = cands[0];
  float bestms = 3.4e38f;
  for (int id : cands) {
    if (!run(id)) continue;
    if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      continue;
    // The event timer quantises a single launch of this size to ~2 us, which is
    // coarser than the gap between neighbouring configs, so each sample times a
    // batch of launches; the flush still leaves the first one cold, as in the
    // benchmark.
    constexpr int REPS = 13, BATCH = 4;
    float t[REPS];
    bool bad = false;
    for (int r = 0; r < REPS && !bad; r++) {
      if (nflush) cudaMemsetAsync(flush.data_ptr(), 0, nflush, st);
      cudaEventRecord(e0, st);
      for (int q = 0; q < BATCH; q++) run(id);
      cudaEventRecord(e1, st);
      if (cudaStreamSynchronize(st) != cudaSuccess) { bad = true; break; }
      if (cudaEventElapsedTime(&t[r], e0, e1) != cudaSuccess) { bad = true; break; }
    }
    if (bad) { cudaGetLastError(); continue; }
    for (int i = 1; i < REPS; i++)
      for (int j = i; j > 0 && t[j] < t[j - 1]; j--) {
        const float tmp = t[j]; t[j] = t[j - 1]; t[j - 1] = tmp;
      }
    float acc = 0.f;
    for (int i = REPS / 4; i < REPS - REPS / 4; i++) acc += t[i];
    acc /= (float)(REPS - 2 * (REPS / 4));
    if (acc < bestms) { bestms = acc; best = id; }
    cudaGetLastError();
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaGetLastError();
  return best;
}

// ---------------------------------------------------------------------------
// entry point: the whole bottleneck for a registered weight set.
// ---------------------------------------------------------------------------
at::Tensor bneck_forward(const at::Tensor& x, int64_t handle) {
  Plan& pl = g_plans[(size_t)handle];
  if (!x.is_contiguous()) return bneck_forward(x.contiguous(), handle);
  const std::array<int, 5> key = {(int)x.size(0), (int)x.size(1), (int)x.size(2),
                                  (int)x.size(3), (int)x.device().index()};
  if (pl.cfg_key != key) {
    TORCH_CHECK(x.dim() == 4, "bneck_forward: NCHW only");
    CP p;
    p.nimg = key[0];
    p.C = key[1];
    p.H = key[2];
    p.W = key[3];
    p.HW = p.H * p.W;
    p.Wp = p.W + 2;
    p.Pp = (p.H + 2) * p.Wp;
    p.Wm = (unsigned)((0x100000000ULL + p.W - 1) / (unsigned long long)p.W);
    p.WC = DIVUP(p.W, 32);
    p.npack = p.nimg * p.H * p.WC;
    p.nbx = 0;
    p.ntile = 0;
    p.G = 1;
    pl.sc = &scratch_for(x, p);
    p.W1 = pl.w1p;
    p.W2 = pl.w2p;
    p.B1 = pl.b1p;
    p.B2 = pl.b2p;
    p.Xp = pl.sc->xpp;
    p.Mid = pl.sc->midp;
    p.bar = pl.sc->barp;
    p.add = pl.add ? 1 : 0;
    p.barbase = 0;
    p.X = nullptr;
    p.Out = nullptr;
    pl.cp = p;
    pl.cfg_key = key;
    pl.cfg = -1;
  }

  at::Tensor out = at::empty_like(x);
  auto st = at::cuda::getCurrentCUDAStream();
  Scratch& sc = *pl.sc;
  CP p = pl.cp;
  p.X = (const __half*)x.data_ptr();
  p.Out = (__half*)out.data_ptr();
  auto run = [&](int id) -> bool {
    return launch_bneck(id, p, sc.barbase, st);
  };
  if (pl.cfg < 0) {
    const int id = autotune(cfg_candidates(pl.cp.C, pl.cp.HW), run, x, st);
    TORCH_CHECK(id >= 0, "bneck_forward: no usable config");
    pl.cfg = id;
  }
  TORCH_CHECK(run(pl.cfg), "bneck_forward: unsupported shape/config");
  return out;
}
"""

_CPP_SRC = r"""
int64_t bneck_register(const at::Tensor& w1, const at::Tensor& b1,
                       const at::Tensor& w2, const at::Tensor& b2, bool add);
at::Tensor bneck_forward(const at::Tensor& x_, int64_t handle);
"""

_EXT = None
_EXT_FAILED = False
# Channel counts the kernel table covers (BN = C).
_SUPPORTED_C = (16, 32, 64, 128)


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    try:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{'a' if major >= 9 else ''}"
    except Exception:
        pass
    tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
    return load_inline(
        name=f"fk_yolo_bneck_{tag}",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["bneck_register", "bneck_forward"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )


def _ext():
    """The compiled extension, or None if it cannot be built here (in which
    case the block falls back to the two ``YOLOConv`` submodules)."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            _EXT = _build_ext()
        except Exception:
            _EXT_FAILED = True
    return _EXT


@torch.no_grad()
def _fold(cv: YOLOConv) -> tuple[torch.Tensor, torch.Tensor]:
    """``cv``'s conv+BN as (weight permuted to [3][3][Cin][Cout], fp32 bias).

    BatchNorm in eval mode is a per-output-channel affine, so it folds into the
    weight exactly the way ``YOLOConv.fuse`` does -- done once here instead of
    two kernels per call.  An already-fused ``YOLOConv`` has no ``bn``.
    """
    conv = cv.conv
    w = conv.weight
    scale = None
    bias = conv.bias.float() if conv.bias is not None else None
    bn = getattr(cv, "bn", None)
    if bn is not None:
        scale = bn.weight.float() * (bn.running_var.float() + bn.eps).rsqrt()
        beta = bn.bias.float() - bn.running_mean.float() * scale
        bias = beta if bias is None else bias * scale + beta
    wf = w.float()
    if scale is not None:
        wf = wf * scale.view(-1, 1, 1, 1)
    # [Cout][Cin][3][3] -> [3][3][Cin][Cout]: k rows are (r, s, ci), Cout is the
    # contiguous (GEMM n) axis, so a k-tile's B rows are 16B-aligned runs.
    wp = wf.to(w.dtype).permute(2, 3, 1, 0).contiguous().view(-1, w.shape[0])
    if bias is None:
        bias = torch.zeros(w.shape[0], device=w.device, dtype=torch.float32)
    return wp, bias.contiguous()


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        # The fused path needs both 3x3 convs, no groups and Cin == Cout == C
        # with C in the table above; anything else runs the baseline modules.
        self._shape_ok = (
            c1 == c_ == c2
            and g == 1
            and c1 in _SUPPORTED_C
            and isinstance(k, (tuple, list))
            and tuple(k) == (3, 3)
        )
        self._handle = -1
        self._watch = ()
        self._key = None
        self._fwd = None
        self._register_load_state_dict_pre_hook(self._invalidate)

    def _invalidate(self, *args, **kwargs):
        self._key = None

    def _watch_list(self):
        """Tensors whose in-place mutation invalidates the folded weights."""
        ts = [self.cv1.conv.weight, self.cv2.conv.weight]
        for cv in (self.cv1, self.cv2):
            if cv.conv.bias is not None:
                ts.append(cv.conv.bias)
            bn = getattr(cv, "bn", None)
            if bn is not None:
                ts += [bn.weight, bn.bias, bn.running_mean, bn.running_var]
        return tuple(ts)

    def _build(self):
        ext = _ext()
        w1, b1 = _fold(self.cv1)
        w2, b2 = _fold(self.cv2)
        self._handle = ext.bneck_register(w1, b1, w2, b2, self.add)
        self._plan_tensors = (w1, b1, w2, b2)   # keep alive on the Python side
        self._fwd = ext.bneck_forward
        self._watch = self._watch_list()
        self._key = tuple(t._version for t in self._watch)

    def _fast_ok(self, x: torch.Tensor) -> bool:
        w = self.cv1.conv.weight
        return (
            self._shape_ok
            and x.is_cuda
            and x.dim() == 4
            and x.dtype is torch.float16
            and w.dtype is torch.float16
            and x.size(1) == w.size(0)
            and _ext() is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._key == tuple(t._version for t in self._watch):
            return self._fwd(x, self._handle)
        if not self._fast_ok(x):
            y = self.cv2(self.cv1(x))
            return x + y if self.add else y
        self._build()
        return self._fwd(x, self._handle)
