"""YOLOv10 C2f and C2fCIB blocks -- whole-block fused CUDA implementation.

The baseline evaluates each of these blocks as 15-25 eager launches (conv,
batch_norm, SiLU, the ``chunk`` views' copies, ``cat``, the residual add). At the
captured sizes that is bound on both ends: 100-250 us of eager dispatch and
60-130 us on the GPU for well under 2 GFLOP of real work, most of it spent in
``batch_norm_transform_input_kernel`` and the concat copy rather than in any
convolution.

So the block is lowered once, at the first forward, into a *plan*: a fixed
sequence of 4-7 fused kernel launches over a fixed buffer graph, issued from one
C++ entry point.

  * BatchNorm is not a kernel. In eval mode it is a per-channel affine map, so
    every conv carries a ``(scale, bias)`` pair its epilogue applies in fp32
    before SiLU. This is exact BN arithmetic -- the weights are left alone, so
    nothing is lost to an fp16 re-rounding of a folded weight.
  * ``chunk``/``cat`` are not kernels either. ``cv1`` writes its ``2c`` channels
    into the head of the ``(2+n)*c`` concat buffer and bottleneck *i* writes into
    slot ``2+i``, so ``cv2`` reads one already-contiguous tensor.
  * The bottleneck / CIB residual add rides in the last conv's epilogue.
  * RepVGGDW's 7x7 and 3x3 depthwise branches (each with its own BN) collapse
    into one fp32 7x7 depthwise weight plus one bias.

Anything the kernels do not cover -- a non-fp16 or non-CUDA input, a spatial
extent whose ``H*W`` is not a multiple of 16, grouped (but not depthwise) convs,
a build failure -- falls back to the eager baseline path in ``_eager``.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_bottleneck import YOLOBottleneck
from .yolov10_cib import YOLOCIB
from .yolov10_conv import YOLOConv

_CUDA_SRC = r"""
// Whole-block fused YOLOv10 C2f / C2fCIB for Blackwell (sm_100), NCHW fp16.
//
// The baseline runs 15-25 separate launches per block (conv, batch_norm, SiLU,
// the chunk copies, cat, the residual add) and is bound on both ends: 100-250 us
// of eager dispatch, plus 60-130 us on the GPU for well under 2 GFLOP of real
// work -- most of that in batch_norm_transform_input_kernel and the concat copy
// rather than in any convolution. So the whole block is lowered to one entry
// point that issues 4-7 kernels:
//
//   * BN is never a kernel. In eval mode it is an affine map, so each conv
//     carries a per-output-channel (scale, bias) pair that its epilogue applies
//     in fp32 before SiLU -- exact BN math, no weight rewriting.
//   * chunk() and cat() disappear. cv1 writes its 2c channels straight into the
//     head of the (2+n)*c concat buffer and bottleneck i writes into slot
//     (2+i), so cv2 reads one already-contiguous tensor.
//   * the bottleneck / CIB residual add is folded into the last conv's epilogue.
//
// Three kernels cover every conv here: a 1x1 batched GEMM, a general R x S
// implicit GEMM, and a depthwise R x R. The first two are tensor-core (wmma,
// fp32 accumulate); all three share the epilogue shape
//   y = silu(scale[co] * acc + bias[co]) [+ residual]
// and address input/output as a channel slice of a larger buffer.
//
// What these sizes reward, measured on a B200 (~0.7 instructions issued per
// scheduler cycle at ~1.1 GHz, i.e. a budget of ~4e5 warp-instructions per us):
//
//   * *Instruction count*, not FLOPs. Every operand reaches shared memory
//     through 128-bit vector loads and leaves it through ldmatrix; a scalar
//     staging loop costs 8x the instructions and dominated a first version.
//   * *Short k loops*. A block's staging stages are serialised by the barrier
//     between them, and one output tile here is only ever a few hundred warps
//     wide, so there is no occupancy to hide a long chain. Large BK (up to the
//     full channel count) trades registers for a shorter chain, and the tuner
//     gets to make that trade per problem.
//   * *Grid coverage*. A 14-block launch uses 9% of the machine no matter how
//     good its inner loop, so the tile list spans 16x32 to 128x128.
// No closed form picked the winner across those three pressures, so the first
// call measures every eligible tile per step and caches the result.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

#define DIVUP(a, b) (((a) + (b) - 1) / (b))

typedef __half h_t;

__device__ __forceinline__ float fk_act(float f, int act) {
  return act ? f / (1.f + __expf(-f)) : f;
}

// One 16x16 accumulator group is exactly 32 eight-element runs, so each lane
// owns one 16-byte store. ``ep`` is this warp's fp32 staging area with row
// stride ``LD``; ``co`` is the lane's output channel, already bounds-checked.
// ``LD`` is padded off a multiple of 32 floats: at exactly 128 B of stride every
// row of the accumulator tile lands on bank 0, and the store_matrix_sync that
// fills this area was most of the kernel's shared-memory bank conflicts.
#define FK_EPILOGUE(ep, LD, NGROUP, nn0, step8)                               \
  {                                                                           \
    const float sc = p.S[co], bb = p.B[co];                                   \
    _Pragma("unroll")                                                         \
    for (int j = 0; j < (NGROUP); j++) {                                      \
      const int nn = (nn0) + j * 16 + (step8);                                \
      if (nn < p.PQ) {                                                        \
        const float* sp_ = (ep) + (long)rr * (LD) + j * 16 + (step8);          \
        const float4 q0_ = *reinterpret_cast<const float4*>(sp_);              \
        const float4 q1_ = *reinterpret_cast<const float4*>(sp_ + 4);          \
        const float a_[8] = {q0_.x, q0_.y, q0_.z, q0_.w,                       \
                             q1_.x, q1_.y, q1_.z, q1_.w};                      \
        h_t rv_[8], v_[8];                                                    \
        if (p.hasres)                                                         \
          *reinterpret_cast<uint4*>(rv_) =                                    \
              *reinterpret_cast<const uint4*>(Rb + (long)co * p.PQ + nn);     \
        _Pragma("unroll")                                                     \
        for (int e = 0; e < 8; e++) {                                         \
          float f = fk_act(a_[e] * sc + bb, p.act);                            \
          if (p.hasres) f += __half2float(rv_[e]);                            \
          v_[e] = __float2half(f);                                            \
        }                                                                     \
        *reinterpret_cast<uint4*>(Yb + (long)co * p.PQ + nn) =                \
            *reinterpret_cast<const uint4*>(v_);                              \
      }                                                                       \
    }                                                                         \
  }

// ---------------------------------------------------------------------------
// 1x1 conv == per-image GEMM  C[Cout, PQ] = W[Cout, Cin] @ X[Cin, PQ].
// With no gather in the way both operands are row-major matrices in global
// memory, so both stage into shared with 16-byte vector loads and the mma
// operands come back out through ldmatrix.
// ---------------------------------------------------------------------------
struct G1P {
  const h_t* X;
  const h_t* W;
  const h_t* Res;
  h_t* Y;
  const float* S;
  const float* B;
  long ximg, yimg, rimg;
  int Cin, Cout, PQ, act, hasres;
};

template <int BM, int BN, int BK, int WM, int WN>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN)) void k_g1(G1P p) {
  using namespace nvcuda::wmma;
  constexpr int NWM = BM / WM, NWN = BN / WN, NW = NWM * NWN, NT = 32 * NW;
  constexpr int LDA = BK + 8, LDB = BN + 8;
  constexpr int SA = BM * LDA, SB = BK * LDB;
  constexpr int EPLD = WN + 4;
  constexpr int EPN = NW * 16 * EPLD;
  constexpr int STAGE = (int)(2 * (SA + SB) * sizeof(h_t));
  constexpr int RAW = STAGE > EPN * 4 ? STAGE : EPN * 4;
  constexpr int AVG = BK / 8, NAV = BM * AVG / NT, ARSTEP = NT / AVG;
  constexpr int BVG = BN / 8, NBV = BK * BVG / NT, BRSTEP = NT / BVG;
  static_assert(BM * AVG % NT == 0 && NT % AVG == 0, "bad A vector split");
  static_assert(BK * BVG % NT == 0 && NT % BVG == 0, "bad B vector split");
  static_assert(RAW <= 48 * 1024, "shared memory limit");

  __shared__ __align__(16) char raw[RAW];
  h_t* As0 = reinterpret_cast<h_t*>(raw);
  h_t* Bs0 = As0 + 2 * SA;
  float* Ep = reinterpret_cast<float*>(raw);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / NWN, wn = warp % NWN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, nb = blockIdx.z;

  fragment<accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; i++)
#pragma unroll
    for (int j = 0; j < WN / 16; j++) fill_fragment(acc[i][j], 0.0f);

  // Per-thread staging source pointers, advanced by one k-tile per iteration:
  // every address term that does not depend on k is computed once. A row past
  // Cout, or a column group past PQ, is clamped to a live one -- it lands in an
  // accumulator slot the epilogue never stores.
  const int a_c8 = (tid % AVG) * 8, a_r = tid / AVG;
  const int b_c8 = (tid % BVG) * 8, b_r = tid / BVG;
  const int bcol = n0 + b_c8 + 8 <= p.PQ ? n0 + b_c8 : p.PQ - 8;
  const h_t* Xb = p.X + (long)nb * p.ximg + bcol;
  const h_t* apt[NAV];
#pragma unroll
  for (int t = 0; t < NAV; t++) {
    const int gm = m0 + a_r + t * ARSTEP;
    apt[t] = p.W + (long)(gm < p.Cout ? gm : p.Cout - 1) * p.Cin + a_c8;
  }
  const h_t* bpt[NBV];
#pragma unroll
  for (int t = 0; t < NBV; t++) bpt[t] = Xb + (long)(b_r + t * BRSTEP) * p.PQ;
  const long bkpq = (long)BK * p.PQ;

  // Two k-tiles are in flight in registers at once: a tile is loaded one
  // iteration before the one in which it is written to shared, so its global
  // latency overlaps a whole iteration instead of only the mma phase between
  // the load and the store. That slack is what these grid sizes cannot buy with
  // occupancy.
  uint4 ra[2][NAV], rb[2][NBV];

#define FK1_LOAD(RS)                                                          \
  {                                                                           \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NAV; t++) {                                           \
      ra[RS][t] = *reinterpret_cast<const uint4*>(apt[t]);                    \
      apt[t] += BK;                                                           \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NBV; t++) {                                           \
      rb[RS][t] = *reinterpret_cast<const uint4*>(bpt[t]);                    \
      bpt[t] += bkpq;                                                         \
    }                                                                         \
  }

#define FK1_STORE(RS, BUF)                                                    \
  {                                                                           \
    h_t* as_ = As0 + (BUF)*SA;                                                \
    h_t* bs_ = Bs0 + (BUF)*SB;                                                \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NAV; t++)                                             \
      *reinterpret_cast<uint4*>(as_ + (a_r + t * ARSTEP) * LDA + a_c8) =      \
          ra[RS][t];                                                          \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NBV; t++)                                             \
      *reinterpret_cast<uint4*>(bs_ + (b_r + t * BRSTEP) * LDB + b_c8) =      \
          rb[RS][t];                                                          \
  }

  const int nstage = p.Cin / BK;
  FK1_LOAD(0)
  if (nstage > 1) FK1_LOAD(1)
  FK1_STORE(0, 0)
  __syncthreads();

#pragma unroll 2
  for (int it = 0; it < nstage; it++) {
    const int cur = it & 1;
    if (it + 2 < nstage) FK1_LOAD(cur)
    const h_t* as = As0 + cur * SA;
    const h_t* bs = Bs0 + cur * SB;
#pragma unroll
    for (int kt = 0; kt < BK / 16; kt++) {
      fragment<matrix_a, 16, 16, 16, h_t, row_major> af[WM / 16];
      fragment<matrix_b, 16, 16, 16, h_t, row_major> bf[WN / 16];
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
        load_matrix_sync(af[i], as + (wm * WM + i * 16) * LDA + kt * 16, LDA);
#pragma unroll
      for (int j = 0; j < WN / 16; j++)
        load_matrix_sync(bf[j], bs + (kt * 16) * LDB + wn * WN + j * 16, LDB);
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
#pragma unroll
        for (int j = 0; j < WN / 16; j++)
          mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    if (it + 1 < nstage) FK1_STORE(1 - cur, 1 - cur)
    __syncthreads();
  }
#undef FK1_LOAD
#undef FK1_STORE

  float* ep = Ep + warp * 16 * EPLD;
  const int rr = lane >> 1, cc = (lane & 1) * 8;
  h_t* Yb = p.Y + (long)nb * p.yimg;
  const h_t* Rb = p.Res + (long)nb * p.rimg;
#pragma unroll
  for (int i = 0; i < WM / 16; i++) {
#pragma unroll
    for (int j = 0; j < WN / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], EPLD, mem_row_major);
    __syncwarp();
    const int co = m0 + wm * WM + i * 16 + rr;
    if (co < p.Cout) FK_EPILOGUE(ep, EPLD, WN / 16, n0 + wn * WN, cc)
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// General R x S dense conv (stride 1, symmetric padding) as an implicit GEMM:
//   C[Cout, PQ] = A[Cout, R*S*Cin] @ B[R*S*Cin, PQ]
// A is the weight pre-permuted to [Cout][R][S][Cin], so a k-tile of it is
// contiguous and stages with 16-byte vector loads. B is im2col-gathered on the
// fly; the k loop walks (r, s) outside and channels inside, so within a k-tile
// the gather's row/column bounds are uniform and hoist out of the innermost
// statement, leaving a pointer bump plus a load. Tiles are double buffered
// through registers, so a tile's global loads overlap the previous tile's math.
// ---------------------------------------------------------------------------
struct CVP {
  const h_t* X;
  const h_t* W;
  const h_t* Res;
  h_t* Y;
  const float* S;
  const float* B;
  long ximg, yimg, rimg;
  int Cin, Cout, H, Wd, PQ, R, Sk, ph, pw, Kc, NCT, act, hasres;
};

template <int BM, int BN, int BK, int WM, int WN>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN)) void k_cv(CVP p) {
  using namespace nvcuda::wmma;
  constexpr int NWM = BM / WM, NWN = BN / WN, NW = NWM * NWN, NT = 32 * NW;
  constexpr int LDA = BK + 8, LDB = BN + 8;
  constexpr int SA = BM * LDA, SB = BK * LDB;
  constexpr int EPLD = WN + 4;
  constexpr int EPN = NW * 16 * EPLD;
  constexpr int STAGE = (int)(2 * (SA + SB) * sizeof(h_t));
  constexpr int RAW = STAGE > EPN * 4 ? STAGE : EPN * 4;
  constexpr int BNT = BN < NT ? BN : NT;
  constexpr int NJ = BN / BNT, KSTEP = NT / BNT, NK = BK / KSTEP;
  constexpr int AVG = BK / 8, NAV = BM * AVG / NT, ARSTEP = NT / AVG;
  static_assert(BM * AVG % NT == 0 && NT % AVG == 0, "bad A vector split");
  static_assert(NAV * ARSTEP == BM, "bad A row split");
  static_assert(BK % KSTEP == 0, "bad B tile split");
  static_assert(RAW <= 48 * 1024, "shared memory limit");

  __shared__ __align__(16) char raw[RAW];
  h_t* As0 = reinterpret_cast<h_t*>(raw);
  h_t* Bs0 = As0 + 2 * SA;
  float* Ep = reinterpret_cast<float*>(raw);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / NWN, wn = warp % NWN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, nb = blockIdx.z;

  fragment<accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; i++)
#pragma unroll
    for (int j = 0; j < WN / 16; j++) fill_fragment(acc[i][j], 0.0f);

  const h_t* Xb = p.X + (long)nb * p.ximg;
  const int jj = tid % BNT, kbase = tid / BNT;
  int hb[NJ], wb[NJ];
#pragma unroll
  for (int u = 0; u < NJ; u++) {
    const int nnr = n0 + u * BNT + jj;
    const int nn = nnr < p.PQ ? nnr : p.PQ - 1;
    const int pp = nn / p.Wd;
    hb[u] = pp - p.ph;
    wb[u] = (nn - pp * p.Wd) - p.pw;
  }
  const int a_c8 = (tid % AVG) * 8, a_r = tid / AVG;
  const h_t* apt[NAV];
#pragma unroll
  for (int t = 0; t < NAV; t++) {
    const int gm = m0 + a_r + t * ARSTEP;
    apt[t] = p.W + (long)(gm < p.Cout ? gm : p.Cout - 1) * p.Kc + a_c8;
  }
  const long hstride = (long)KSTEP * p.PQ;
  const int ntile = p.R * p.Sk * p.NCT;

  uint4 ra[NAV];
  h_t rb[NJ * NK];
  int cit = 0, trs = 0, tr = 0, ts = 0;

#define FK_ADV                                                                \
  {                                                                           \
    if (++cit == p.NCT) {                                                     \
      cit = 0;                                                                \
      trs++;                                                                  \
      if (++ts == p.Sk) { ts = 0; tr++; }                                     \
    }                                                                         \
  }

#define FK_LOAD                                                           \
  {                                                                           \
    const int ci0_ = cit * BK;                                                \
    const int koff_ = trs * p.Cin + ci0_;                                     \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NAV; t++)                                             \
      ra[t] = *reinterpret_cast<const uint4*>(apt[t] + koff_);                \
    _Pragma("unroll")                                                         \
    for (int u = 0; u < NJ; u++) {                                            \
      const int h_ = hb[u] + tr, w_ = wb[u] + ts;                             \
      const bool ok_ = (unsigned)h_ < (unsigned)p.H &&                        \
                       (unsigned)w_ < (unsigned)p.Wd;                         \
      if (!ok_) {                                                             \
        _Pragma("unroll")                                                     \
        for (int t = 0; t < NK; t++) rb[u * NK + t] = (h_t)0.f;               \
      } else {                                                                \
        const h_t* pq_ = Xb + ((long)(ci0_ + kbase) * p.PQ + h_ * p.Wd + w_); \
        _Pragma("unroll")                                                     \
        for (int t = 0; t < NK; t++) {                                        \
          rb[u * NK + t] = pq_[0];                                            \
          pq_ += hstride;                                                     \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }

#define FK_STORE(BUF)                                                     \
  {                                                                           \
    h_t* as_ = As0 + (BUF)*SA;                                                \
    h_t* bs_ = Bs0 + (BUF)*SB + jj;                                           \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NAV; t++)                                             \
      *reinterpret_cast<uint4*>(as_ + (a_r + t * ARSTEP) * LDA + a_c8) = ra[t]; \
    _Pragma("unroll")                                                         \
    for (int u = 0; u < NJ; u++)                                              \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < NK; t++)                                            \
        bs_[(kbase + t * KSTEP) * LDB + u * BNT] = rb[u * NK + t];            \
  }

  FK_LOAD
  FK_ADV
  FK_STORE(0)
  __syncthreads();

#pragma unroll 2
  for (int it = 0; it < ntile; it++) {
    const int cur = it & 1;
    if (it + 1 < ntile) { FK_LOAD FK_ADV }
    const h_t* as = As0 + cur * SA;
    const h_t* bs = Bs0 + cur * SB;
#pragma unroll
    for (int kt = 0; kt < BK / 16; kt++) {
      fragment<matrix_a, 16, 16, 16, h_t, row_major> af[WM / 16];
      fragment<matrix_b, 16, 16, 16, h_t, row_major> bf[WN / 16];
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
        load_matrix_sync(af[i], as + (wm * WM + i * 16) * LDA + kt * 16, LDA);
#pragma unroll
      for (int j = 0; j < WN / 16; j++)
        load_matrix_sync(bf[j], bs + (kt * 16) * LDB + wn * WN + j * 16, LDB);
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
#pragma unroll
        for (int j = 0; j < WN / 16; j++)
          mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    if (it + 1 < ntile) FK_STORE(1 - cur)
    __syncthreads();
  }
#undef FK_LOAD
#undef FK_ADV
#undef FK_STORE

  float* ep = Ep + warp * 16 * EPLD;
  const int rr = lane >> 1, cc = (lane & 1) * 8;
  h_t* Yb = p.Y + (long)nb * p.yimg;
  const h_t* Rb = p.Res + (long)nb * p.rimg;
#pragma unroll
  for (int i = 0; i < WM / 16; i++) {
#pragma unroll
    for (int j = 0; j < WN / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], EPLD, mem_row_major);
    __syncwarp();
    const int co = m0 + wm * WM + i * 16 + rr;
    if (co < p.Cout) FK_EPILOGUE(ep, EPLD, WN / 16, n0 + wn * WN, cc)
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// Depthwise R x R conv (stride 1, symmetric padding). A thread owns VW
// consecutive output pixels of one row, so the R+VW-1 input values it reads per
// kernel row serve all VW outputs; the taps live in registers rather than being
// re-read per multiply, which is what a naive one-pixel-per-thread version
// spends most of its instructions on. Weights are fp32: a depthwise tap count
// is tiny, and this is where RepVGGDW's pre-summed 7x7 + 3x3 branches land.
// ---------------------------------------------------------------------------
struct DWP {
  const h_t* X;
  const float* W;
  const h_t* Res;
  h_t* Y;
  const float* S;
  const float* B;
  long ximg, yimg, rimg;
  int C, H, Wd, PQ, ph, act, hasres;
};

template <int R, int VW>
__global__ __launch_bounds__(128) void k_dw(DWP p) {
  const int cn = blockIdx.y;
  const int nb = cn / p.C, c = cn - nb * p.C;
  float wr[R * R];
  const float* wp = p.W + (long)c * R * R;
#pragma unroll
  for (int i = 0; i < R * R; i++) wr[i] = wp[i];

  const int base = (blockIdx.x * 128 + threadIdx.x) * VW;
  if (base >= p.PQ) return;
  const int hh = base / p.Wd, ww = base - hh * p.Wd;
  const h_t* xb = p.X + (long)nb * p.ximg + (long)c * p.PQ;

  float acc[VW];
#pragma unroll
  for (int o = 0; o < VW; o++) acc[o] = 0.f;
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int ih = hh + r - p.ph;
    if ((unsigned)ih >= (unsigned)p.H) continue;
    const h_t* row = xb + (long)ih * p.Wd;
    float v[VW + R - 1];
#pragma unroll
    for (int e = 0; e < VW + R - 1; e++) {
      const int iw = ww - p.ph + e;
      v[e] = (unsigned)iw < (unsigned)p.Wd ? __half2float(row[iw]) : 0.f;
    }
#pragma unroll
    for (int o = 0; o < VW; o++)
#pragma unroll
      for (int s = 0; s < R; s++) acc[o] += wr[r * R + s] * v[o + s];
  }

  const float sc = p.S[c], bb = p.B[c];
  const long yo = (long)nb * p.yimg + (long)c * p.PQ + base;
  const long ro = (long)nb * p.rimg + (long)c * p.PQ + base;
#pragma unroll
  for (int o = 0; o < VW; o++) {
    float f = fk_act(acc[o] * sc + bb, p.act);
    if (p.hasres) f += __half2float(p.Res[ro + o]);
    p.Y[yo + o] = __float2half(f);
  }
}

// ###########################################################################
// host side: plan (a fixed launch sequence + its buffer graph) + autotune
// ###########################################################################

#define G1_LIST                                                               \
  X(0, 16, 32, 16, 16, 32) X(1, 32, 64, 16, 32, 32) \
  X(2, 64, 64, 16, 32, 32) X(3, 128, 128, 16, 32, 64) \
  X(4, 16, 32, 32, 16, 16) X(5, 16, 64, 32, 16, 32) \
  X(6, 32, 32, 32, 16, 16) X(7, 32, 64, 32, 16, 32) \
  X(8, 64, 64, 32, 16, 32) X(9, 64, 64, 32, 32, 32) \
  X(10, 32, 128, 32, 32, 64) X(11, 64, 128, 32, 32, 32) \
  X(12, 128, 64, 32, 32, 32) X(13, 128, 128, 32, 32, 64) \
  X(14, 16, 64, 64, 16, 16) X(15, 32, 32, 64, 16, 16) \
  X(16, 32, 64, 64, 16, 16) X(17, 32, 64, 64, 32, 32) \
  X(18, 64, 64, 64, 16, 32) X(19, 64, 64, 64, 32, 32) \
  X(20, 16, 128, 64, 16, 32) X(21, 32, 128, 64, 16, 32) \
  X(22, 16, 32, 64, 16, 32) X(23, 64, 64, 64, 16, 16) \
  X(24, 128, 128, 32, 32, 32) X(25, 64, 32, 64, 16, 16)

#define CV_LIST                                                               \
  X(0, 16, 32, 16, 16, 32) X(1, 16, 64, 16, 16, 64) \
  X(2, 32, 64, 16, 32, 32) X(3, 64, 64, 16, 32, 32) \
  X(4, 16, 32, 32, 16, 16) X(5, 16, 64, 32, 16, 32) \
  X(6, 32, 32, 32, 16, 16) X(7, 32, 64, 32, 16, 32) \
  X(8, 32, 64, 32, 32, 32) X(9, 64, 64, 32, 16, 32) \
  X(10, 64, 64, 32, 32, 32) X(11, 32, 128, 32, 32, 64) \
  X(12, 64, 128, 32, 32, 32) X(13, 128, 64, 32, 32, 32) \
  X(14, 16, 32, 64, 16, 32) X(15, 16, 64, 64, 16, 16) \
  X(16, 32, 32, 64, 16, 16) X(17, 32, 64, 64, 16, 16) \
  X(18, 32, 64, 64, 32, 32) X(19, 64, 64, 64, 16, 32) \
  X(20, 64, 64, 64, 32, 32) X(21, 32, 128, 64, 16, 32) \
  X(22, 16, 32, 128, 16, 16) X(23, 32, 32, 128, 16, 16) \
  X(24, 64, 64, 64, 16, 16) X(25, 128, 128, 32, 32, 32) \
  X(26, 64, 32, 64, 16, 16)

#define DW_LIST                                                               \
  X(0, 3, 1) X(1, 3, 2) X(2, 3, 4) X(3, 7, 1) X(4, 7, 2) X(5, 7, 4)         \
  X(6, 3, 8) X(7, 7, 8)

struct TileCfg { int bm, bn, bk, nt; };

static const TileCfg kG1[] = {
#define X(id, bm, bn, bk, wm, wn) {bm, bn, bk, 32 * (bm / wm) * (bn / wn)},
    G1_LIST
#undef X
};
static const TileCfg kCv[] = {
#define X(id, bm, bn, bk, wm, wn) {bm, bn, bk, 32 * (bm / wm) * (bn / wn)},
    CV_LIST
#undef X
};
struct DwCfg { int r, vw; };
static const DwCfg kDw[] = {
#define X(id, r, vw) {r, vw},
    DW_LIST
#undef X
};
static const int kNG1 = (int)(sizeof(kG1) / sizeof(TileCfg));
static const int kNCv = (int)(sizeof(kCv) / sizeof(TileCfg));
static const int kNDw = (int)(sizeof(kDw) / sizeof(DwCfg));

// Each kernel takes its whole parameter set as one by-value struct, so a launch
// is fully described by (entry point, grid, block, struct). That is what lets the
// same description feed both a direct launch and a graph node.
static void* g1_func(int cfg) {
  switch (cfg) {
#define X(id, bm, bn, bk, wm, wn) case id: return (void*)k_g1<bm, bn, bk, wm, wn>;
    G1_LIST
#undef X
    default: return nullptr;
  }
}

static void* cv_func(int cfg) {
  switch (cfg) {
#define X(id, bm, bn, bk, wm, wn) case id: return (void*)k_cv<bm, bn, bk, wm, wn>;
    CV_LIST
#undef X
    default: return nullptr;
  }
}

static void* dw_func(int cfg) {
  switch (cfg) {
#define X(id, r, vw) case id: return (void*)k_dw<r, vw>;
    DW_LIST
#undef X
    default: return nullptr;
  }
}

struct Step {
  int kind, Cin, Cout, R, Sk, ph, pw;
  int xbuf, xoff, ybuf, yoff, rbuf, roff, act, hasres;
  const void* w;
  const float* s;
  const float* b;
  int cfg;
};

// A launch, in the form both cudaLaunchKernel and cudaGraphAddKernelNode want.
struct SL {
  void* func;
  dim3 grid;
  dim3 block;
};

struct Plan {
  std::vector<at::Tensor> keep;
  std::vector<Step> steps;
  std::vector<int> bufC;
  int cin_total = 0, cout_total = 0;
  at::Tensor ws;
  long tuned_key = -1;
  // Replayed launch sequence. A launch costs ~3 us of driver time on this host,
  // which for a 4-7 kernel block is as much wall clock as the kernels
  // themselves, so the tuned sequence is instantiated once as a linear graph.
  // Only the steps that name the caller's input or output tensor need their
  // parameters refreshed when those addresses move.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t gexec = nullptr;
  std::vector<cudaGraphNode_t> gnode;
  std::vector<std::array<char, 192>> gpb;
  std::vector<void*> garg;
  std::vector<char> gupd;
  h_t *cap0 = nullptr, *cap1 = nullptr, *capws = nullptr;
  long graph_key = -1;
  bool graph_off = false;
};

static std::vector<Plan*> g_plans;
static std::mutex g_mu;

// (kind, Cin, Cout, R, PQ, Wd, N, act, hasres) -> winning config
using TKey = std::array<int, 9>;
static std::map<TKey, int> g_tune;

int64_t c2f_make_plan(std::vector<at::Tensor> tens, std::vector<int64_t> meta,
                      std::vector<int64_t> bufC) {
  const int nstep = (int)(meta.size() / 15);
  TORCH_CHECK((int)tens.size() == 3 * nstep, "plan: tensor/meta mismatch");
  Plan* P = new Plan();
  P->keep = tens;
  for (auto v : bufC) P->bufC.push_back((int)v);
  P->cin_total = P->bufC[0];
  P->cout_total = P->bufC[1];
  for (int i = 0; i < nstep; i++) {
    const int64_t* m = meta.data() + 15 * i;
    Step s;
    s.kind = (int)m[0]; s.Cin = (int)m[1]; s.Cout = (int)m[2];
    s.R = (int)m[3]; s.Sk = (int)m[4]; s.ph = (int)m[5]; s.pw = (int)m[6];
    s.xbuf = (int)m[7]; s.xoff = (int)m[8];
    s.ybuf = (int)m[9]; s.yoff = (int)m[10];
    s.rbuf = (int)m[11]; s.roff = (int)m[12];
    s.act = (int)m[13]; s.hasres = (int)m[14];
    s.w = tens[3 * i + 0].data_ptr();
    s.s = tens[3 * i + 1].data_ptr<float>();
    s.b = tens[3 * i + 2].data_ptr<float>();
    s.cfg = -1;
    P->steps.push_back(s);
  }
  std::lock_guard<std::mutex> lk(g_mu);
  g_plans.push_back(P);
  return (int64_t)g_plans.size() - 1;
}

namespace {

struct Ctx {
  h_t* base[10];
  long img[10];
  int N, H, Wd;
  long PQ;
};

// Writes this step's parameter struct into *pb* and its launch geometry into
// *sl*; shared by the direct-launch and graph-node paths.
inline bool prep_step(const Step& s, const Ctx& c, int cfg, void* pb, SL* sl) {
  const h_t* xp = c.base[s.xbuf] + (long)s.xoff * c.PQ;
  h_t* yp = c.base[s.ybuf] + (long)s.yoff * c.PQ;
  const h_t* rp = s.hasres ? c.base[s.rbuf] + (long)s.roff * c.PQ : yp;
  const long rimg = s.hasres ? c.img[s.rbuf] : 0;
  if (s.kind == 0) {
    if (cfg < 0 || cfg >= kNG1) return false;
    G1P p;
    p.X = xp; p.W = (const h_t*)s.w; p.Res = rp; p.Y = yp;
    p.S = s.s; p.B = s.b;
    p.ximg = c.img[s.xbuf]; p.yimg = c.img[s.ybuf]; p.rimg = rimg;
    p.Cin = s.Cin; p.Cout = s.Cout; p.PQ = (int)c.PQ;
    p.act = s.act; p.hasres = s.hasres;
    *reinterpret_cast<G1P*>(pb) = p;
    sl->func = g1_func(cfg);
    sl->grid = dim3(DIVUP(p.PQ, kG1[cfg].bn), DIVUP(p.Cout, kG1[cfg].bm), c.N);
    sl->block = dim3(kG1[cfg].nt);
  } else if (s.kind == 1) {
    if (cfg < 0 || cfg >= kNCv) return false;
    CVP p;
    p.X = xp; p.W = (const h_t*)s.w; p.Res = rp; p.Y = yp;
    p.S = s.s; p.B = s.b;
    p.ximg = c.img[s.xbuf]; p.yimg = c.img[s.ybuf]; p.rimg = rimg;
    p.Cin = s.Cin; p.Cout = s.Cout;
    p.H = c.H; p.Wd = c.Wd; p.PQ = (int)c.PQ;
    p.R = s.R; p.Sk = s.Sk; p.ph = s.ph; p.pw = s.pw;
    p.Kc = s.R * s.Sk * s.Cin; p.NCT = s.Cin / kCv[cfg].bk;
    p.act = s.act; p.hasres = s.hasres;
    *reinterpret_cast<CVP*>(pb) = p;
    sl->func = cv_func(cfg);
    sl->grid = dim3(DIVUP(p.PQ, kCv[cfg].bn), DIVUP(p.Cout, kCv[cfg].bm), c.N);
    sl->block = dim3(kCv[cfg].nt);
  } else {
    if (cfg < 0 || cfg >= kNDw || kDw[cfg].r != s.R) return false;
    DWP p;
    p.X = xp; p.W = (const float*)s.w; p.Res = rp; p.Y = yp;
    p.S = s.s; p.B = s.b;
    p.ximg = c.img[s.xbuf]; p.yimg = c.img[s.ybuf]; p.rimg = rimg;
    p.C = s.Cout; p.H = c.H; p.Wd = c.Wd; p.PQ = (int)c.PQ;
    p.ph = s.ph; p.act = s.act; p.hasres = s.hasres;
    *reinterpret_cast<DWP*>(pb) = p;
    sl->func = dw_func(cfg);
    sl->grid = dim3(DIVUP(p.PQ / kDw[cfg].vw, 128), p.C * c.N, 1);
    sl->block = dim3(128);
  }
  return sl->func != nullptr;
}

inline bool run_step(const Step& s, const Ctx& c, int cfg, cudaStream_t st) {
  alignas(16) char pb[192];
  SL sl;
  if (!prep_step(s, c, cfg, pb, &sl)) return false;
  void* args[1] = {pb};
  return cudaLaunchKernel(sl.func, sl.grid, sl.block, args, 0, st) == cudaSuccess;
}

std::vector<int> cfg_candidates(const Step& s, const Ctx& c) {
  std::vector<int> v;
  if (s.kind == 0) {
    for (int i = 0; i < kNG1; i++)
      if (s.Cin % kG1[i].bk == 0 && kG1[i].bm <= s.Cout &&
          kG1[i].bn <= 4 * c.PQ)
        v.push_back(i);
  } else if (s.kind == 1) {
    for (int i = 0; i < kNCv; i++)
      if (s.Cin % kCv[i].bk == 0 && kCv[i].bm <= s.Cout) v.push_back(i);
  } else {
    for (int i = 0; i < kNDw; i++)
      if (kDw[i].r == s.R && c.Wd % kDw[i].vw == 0) v.push_back(i);
  }
  return v;
}

int autotune(const Step& s, const Ctx& c, cudaStream_t st) {
  std::vector<int> cands = cfg_candidates(s, c);
  if (cands.empty()) return -1;
  if (cands.size() == 1) return run_step(s, c, cands[0], st) ? cands[0] : -1;
  cudaEvent_t e0, e1;
  if (cudaEventCreate(&e0) != cudaSuccess) return cands[0];
  if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); return cands[0]; }
  int best = -1;
  float bestms = 3.4e38f;
  for (int cfg : cands) {
    if (!run_step(s, c, cfg, st)) continue;
    if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      continue;
    // The event timer quantises these durations coarsely, so a plain median
    // leaves near-equal tiles tied; a trimmed mean resolves them without
    // letting one outlier decide.
    constexpr int REPS = 11;
    float t[REPS];
    bool bad = false;
    for (int r = 0; r < REPS; r++) {
      cudaEventRecord(e0, st);
      run_step(s, c, cfg, st);
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
    if (acc < bestms) { bestms = acc; best = cfg; }
    cudaGetLastError();
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaGetLastError();
  return best < 0 ? cands[0] : best;
}

void drop_graph(Plan& P) {
  if (P.gexec) cudaGraphExecDestroy(P.gexec);
  if (P.graph) cudaGraphDestroy(P.graph);
  P.gexec = nullptr;
  P.graph = nullptr;
  P.graph_key = -1;
}

// Instantiate the tuned step sequence as a linear graph. Every node's argument
// block is kept so the few nodes that reference the caller's tensors can be
// re-parameterised in place instead of forcing a rebuild.
bool build_graph(Plan& P, const Ctx& c) {
  const int n = (int)P.steps.size();
  P.gpb.assign(n, std::array<char, 192>{});
  P.garg.assign(n, nullptr);
  P.gnode.assign(n, nullptr);
  P.gupd.assign(n, 0);
  cudaGraph_t g = nullptr;
  if (cudaGraphCreate(&g, 0) != cudaSuccess) return false;
  cudaGraphNode_t prev = nullptr;
  for (int i = 0; i < n; i++) {
    const Step& s = P.steps[i];
    SL sl;
    if (!prep_step(s, c, s.cfg, P.gpb[i].data(), &sl)) { cudaGraphDestroy(g); return false; }
    P.garg[i] = P.gpb[i].data();
    cudaKernelNodeParams np = {};
    np.func = sl.func;
    np.gridDim = sl.grid;
    np.blockDim = sl.block;
    np.sharedMemBytes = 0;
    np.kernelParams = &P.garg[i];
    np.extra = nullptr;
    cudaGraphNode_t nd = nullptr;
    if (cudaGraphAddKernelNode(&nd, g, prev ? &prev : nullptr, prev ? 1 : 0, &np)
        != cudaSuccess) { cudaGraphDestroy(g); return false; }
    P.gnode[i] = nd;
    prev = nd;
    P.gupd[i] = (s.xbuf <= 1 || s.ybuf <= 1 || (s.hasres && s.rbuf <= 1)) ? 1 : 0;
  }
  cudaGraphExec_t ge = nullptr;
  if (cudaGraphInstantiateWithFlags(&ge, g, 0) != cudaSuccess) {
    cudaGraphDestroy(g);
    return false;
  }
  P.graph = g;
  P.gexec = ge;
  return true;
}

bool refresh_graph(Plan& P, const Ctx& c) {
  for (int i = 0; i < (int)P.steps.size(); i++) {
    if (!P.gupd[i]) continue;
    SL sl;
    if (!prep_step(P.steps[i], c, P.steps[i].cfg, P.gpb[i].data(), &sl)) return false;
    cudaKernelNodeParams np = {};
    np.func = sl.func;
    np.gridDim = sl.grid;
    np.blockDim = sl.block;
    np.sharedMemBytes = 0;
    np.kernelParams = &P.garg[i];
    np.extra = nullptr;
    if (cudaGraphExecKernelNodeSetParams(P.gexec, P.gnode[i], &np) != cudaSuccess)
      return false;
  }
  return true;
}

}  // namespace

static const bool kGraphOff = getenv("FK_C2F_NOGRAPH") != nullptr;

at::Tensor c2f_run(int64_t id, const at::Tensor& x_) {
  Plan* P;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    TORCH_CHECK(id >= 0 && id < (int64_t)g_plans.size(), "bad plan id");
    P = g_plans[id];
  }
  at::Tensor x = x_.is_contiguous() ? x_ : x_.contiguous();
  const int N = (int)x.size(0), H = (int)x.size(2), Wd = (int)x.size(3);
  const long PQ = (long)H * Wd;
  at::Tensor out = at::empty({N, P->cout_total, H, Wd}, x.options());

  const int nbuf = (int)P->bufC.size();
  TORCH_CHECK(nbuf <= 10, "too many buffers");
  long need = 0;
  long off[10];
  for (int i = 2; i < nbuf; i++) { off[i] = need; need += (long)P->bufC[i] * PQ * N; }
  if (!P->ws.defined() || P->ws.numel() < need || P->ws.device() != x.device())
    P->ws = at::empty({need}, x.options());

  Ctx c;
  c.N = N; c.H = H; c.Wd = Wd; c.PQ = PQ;
  c.base[0] = (h_t*)x.data_ptr();   c.img[0] = (long)P->cin_total * PQ;
  c.base[1] = (h_t*)out.data_ptr(); c.img[1] = (long)P->cout_total * PQ;
  h_t* wsp = (h_t*)P->ws.data_ptr();
  for (int i = 2; i < nbuf; i++) {
    c.base[i] = wsp + off[i];
    c.img[i] = (long)P->bufC[i] * PQ;
  }

  auto st = at::cuda::getCurrentCUDAStream();
  // Under stream capture neither the tuner (it synchronises) nor a graph launch
  // is legal, so a caller capturing this block into their own graph gets plain
  // launches of whatever tiles are already cached.
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  const bool capturing =
      cudaStreamIsCapturing(st, &cap) != cudaSuccess ||
      cap != cudaStreamCaptureStatusNone;
  const long key = ((long)N * 100003 + H) * 100003 + Wd;
  if (P->tuned_key != key) {
    for (auto& s : P->steps) {
      const TKey tk = {s.kind, s.Cin, s.Cout, s.R, (int)PQ, Wd, N, s.act, s.hasres};
      auto it = g_tune.find(tk);
      if (it != g_tune.end()) { s.cfg = it->second; continue; }
      if (capturing) {
        const std::vector<int> v = cfg_candidates(s, c);
        TORCH_CHECK(!v.empty(), "c2f_run: no usable config");
        s.cfg = v[0];
        continue;
      }
      const int cfg = autotune(s, c, st);
      TORCH_CHECK(cfg >= 0, "c2f_run: no usable config");
      s.cfg = cfg;
      g_tune[tk] = cfg;
    }
    if (!capturing) P->tuned_key = key;
  }

  if (!P->graph_off && !kGraphOff && !capturing) {
    if (P->gexec == nullptr || P->graph_key != key || P->capws != wsp) {
      drop_graph(*P);
      if (build_graph(*P, c)) {
        P->graph_key = key;
        P->capws = wsp;
        P->cap0 = c.base[0];
        P->cap1 = c.base[1];
      } else {
        drop_graph(*P);
        P->graph_off = true;
        cudaGetLastError();
      }
    } else if (P->cap0 != c.base[0] || P->cap1 != c.base[1]) {
      if (refresh_graph(*P, c)) {
        P->cap0 = c.base[0];
        P->cap1 = c.base[1];
      } else {
        drop_graph(*P);
        P->graph_off = true;
        cudaGetLastError();
      }
    }
    if (!P->graph_off) {
      if (cudaGraphLaunch(P->gexec, st) == cudaSuccess) return out;
      drop_graph(*P);
      P->graph_off = true;
      cudaGetLastError();
    }
  }

  for (const auto& s : P->steps)
    TORCH_CHECK(run_step(s, c, s.cfg, st), "c2f_run: launch failed");
  return out;
}
"""

_CPP_SRC = r"""
int64_t c2f_make_plan(std::vector<at::Tensor> tens, std::vector<int64_t> meta,
                      std::vector<int64_t> bufC);
at::Tensor c2f_run(int64_t id, const at::Tensor& x);
"""

_EXT = None
_EXT_FAILED = False


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    return f"{major}.{minor}{'a' if major >= 9 else ''}"


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    arch = _arch_list()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
    return load_inline(
        name=f"fk_yolo_c2f_{tag}",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["c2f_make_plan", "c2f_run"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
        ],
        verbose=False,
    )


def _ext():
    """The compiled extension, or None if it cannot be built on this machine.

    A build failure degrades to the eager baseline rather than breaking the op.
    """
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            _EXT = _build_ext()
        except Exception:
            _EXT_FAILED = True
    return _EXT


class _Unsupported(Exception):
    """This module/input shape is not covered by the fused kernels."""


# --- weight lowering --------------------------------------------------------
def _affine(m: nn.Module):
    """A YOLOConv's eval-time per-output-channel (scale, bias), in fp32.

    ``y = silu(bn(conv(x)))`` with frozen BN statistics is
    ``silu(conv(x) * scale + bias)``; computing the pair here keeps the
    convolution weight bit-exact and leaves one fused epilogue on the GPU.
    """
    conv = m.conv
    co = conv.weight.shape[0]
    dev, opt = conv.weight.device, dict(dtype=torch.float32)
    bn = getattr(m, "bn", None)
    if bn is None:
        scale = torch.ones(co, device=dev, **opt)
        bias = (conv.bias.detach().float() if conv.bias is not None
                else torch.zeros(co, device=dev, **opt))
        return scale, bias
    scale = bn.weight.detach().float() / torch.sqrt(
        bn.running_var.detach().float() + bn.eps)
    bias = bn.bias.detach().float() - bn.running_mean.detach().float() * scale
    if conv.bias is not None:
        bias = bias + conv.bias.detach().float() * scale
    return scale.contiguous(), bias.contiguous()


def _act_code(act: nn.Module) -> int:
    name = type(act).__name__
    if name == "SiLU":
        return 1
    if name == "Identity":
        return 0
    raise _Unsupported(f"activation {name}")


class _Plan:
    """Accumulates the (tensors, meta, buffer) triple the extension consumes."""

    KIND_GEMM1X1, KIND_CONV, KIND_DW = 0, 1, 2

    def __init__(self, cin: int, cout: int):
        self.tens: list[torch.Tensor] = []
        self.meta: list[int] = []
        self.bufC: list[int] = [cin, cout]

    def buffer(self, channels: int) -> int:
        self.bufC.append(channels)
        return len(self.bufC) - 1

    def step(self, kind, w, scale, bias, cin, cout, r, s, ph, pw,
             dst, src, res, act):
        self.tens += [w, scale, bias]
        rbuf, roff = res if res is not None else (0, 0)
        self.meta += [kind, cin, cout, r, s, ph, pw,
                      src[0], src[1], dst[0], dst[1], rbuf, roff,
                      act, 1 if res is not None else 0]

    # -- one YOLOConv ------------------------------------------------------
    def conv(self, m: nn.Module, dst, src, res=None, act=None):
        weight = m.conv.weight
        cout, cin_g, r, s = weight.shape
        conv = m.conv
        if tuple(conv.stride) != (1, 1) or tuple(conv.dilation) != (1, 1):
            raise _Unsupported("stride/dilation")
        if tuple(conv.padding) != (r // 2, s // 2):
            raise _Unsupported("padding")
        scale, bias = _affine(m)
        a = _act_code(m.act) if act is None else act
        if conv.groups == 1:
            if cin_g % 16 or cout % 16:
                raise _Unsupported("channel count")
            if r == 1 and s == 1:
                w = weight.detach().reshape(cout, cin_g).contiguous()
                kind, rr, ss = self.KIND_GEMM1X1, 1, 1
            else:
                w = weight.detach().permute(0, 2, 3, 1).contiguous()
                kind, rr, ss = self.KIND_CONV, r, s
            self.step(kind, w, scale, bias, cin_g, cout, rr, ss,
                      r // 2, s // 2, dst, src, res, a)
        elif conv.groups == cout and cin_g == 1:
            if r != s or r not in (3, 7):
                raise _Unsupported(f"depthwise {r}x{s}")
            w = weight.detach().float().reshape(cout, r, s).contiguous()
            self.step(self.KIND_DW, w, scale, bias, cout, cout, r, s,
                      r // 2, s // 2, dst, src, res, a)
        else:
            raise _Unsupported("grouped conv")

    # -- RepVGGDW: two depthwise branches + BN each -> one fp32 kernel -----
    def repvggdw(self, m: nn.Module, dst, src, res=None):
        big = m.conv
        if tuple(big.conv.stride) != (1, 1) or big.conv.groups != big.conv.weight.shape[0]:
            raise _Unsupported("repvggdw layout")
        c, _, r, s = big.conv.weight.shape
        if r != s or r not in (3, 7):
            raise _Unsupported(f"repvggdw {r}x{s}")
        scale, bias = _affine(big)
        w = big.conv.weight.detach().float().reshape(c, r, s) * scale.view(c, 1, 1)
        small = getattr(m, "conv1", None)
        if small is not None:
            s2, b2 = _affine(small)
            c2, _, r2, _ = small.conv.weight.shape
            if c2 != c or r2 > r:
                raise _Unsupported("repvggdw branch")
            t = small.conv.weight.detach().float().reshape(c, r2, r2) * s2.view(c, 1, 1)
            pad = (r - r2) // 2
            w = w + F.pad(t, (pad, pad, pad, pad))
            bias = bias + b2
        ones = torch.ones(c, device=w.device, dtype=torch.float32)
        self.step(self.KIND_DW, w.contiguous(), ones, bias.contiguous(),
                  c, c, r, s, r // 2, r // 2, dst, src, res, _act_code(m.act))

    def finish(self):
        return _ext().c2f_make_plan(self.tens, self.meta, self.bufC)


class YOLOC2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self._plan = -1          # -1 not built, -2 unsupported, else plan id
        self._run = None
        self._ph = self._pw = -1
        self._pcin = c1

    # -- eager reference (also the fallback) -------------------------------
    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    # -- the n middle blocks; overridden by the CIB variant ----------------
    def _lower_block(self, plan: _Plan, block: nn.Module, src, dst, res, tmp):
        plan.conv(block.cv1, dst=(tmp[0], 0), src=src)
        plan.conv(block.cv2, dst=dst, src=(tmp[0], 0), res=res)

    def _n_tmp(self) -> tuple[int, ...]:
        return (self.c,)

    def _build(self, x: torch.Tensor) -> int:
        if _ext() is None:
            raise _Unsupported("extension unavailable")
        if not (x.is_cuda and x.dim() == 4 and x.dtype is torch.float16):
            raise _Unsupported("input dtype/layout")
        if (x.size(2) * x.size(3)) % 16 != 0:
            raise _Unsupported("H*W not a multiple of 16")
        for p in self.parameters():
            if p.dtype is not torch.float16:
                raise _Unsupported("parameter dtype")
        c, n = self.c, len(self.m)
        plan = _Plan(self.cv1.conv.weight.shape[1], self.cv2.conv.weight.shape[0])
        cat = plan.buffer((2 + n) * c)
        tmp = tuple(plan.buffer(k) for k in self._n_tmp())
        plan.conv(self.cv1, dst=(cat, 0), src=(0, 0))
        for i, block in enumerate(self.m):
            src = (cat, (1 + i) * c)
            self._lower_block(plan, block, src=src, dst=(cat, (2 + i) * c),
                              res=src if block.add else None, tmp=tmp)
        plan.conv(self.cv2, dst=(1, 0), src=(cat, 0))
        return plan.finish()

    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        if self._plan == -2:
            return self._eager(x)
        if self._plan == -1:
            try:
                self._plan = self._build(x)
                self._run = _ext().c2f_run
            except Exception:
                self._plan = -2
                return self._eager(x)
        if (x.is_cuda and x.dim() == 4 and x.dtype is torch.float16
                and x.size(1) == self._pcin and (x.size(2) * x.size(3)) % 16 == 0):
            self._ph, self._pw = x.size(2), x.size(3)
            return self._run(self._plan, x)
        return self._eager(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Steady state: the plan is built and this input matches what it was
        # validated for, so go straight to the one extension call. ``_ph`` starts
        # at -1, so the first call (and any input the plan does not cover) takes
        # the slow path, which re-validates and falls back to ``_eager``.
        if (x.size(2) == self._ph and x.size(3) == self._pw
                and x.size(1) == self._pcin and x.dtype is torch.float16
                and x.is_cuda):
            return self._run(self._plan, x)
        return self._forward_slow(x)


class YOLOC2fCIB(YOLOC2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(YOLOCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

    def _n_tmp(self) -> tuple[int, ...]:
        return (self.c, 2 * self.c, 2 * self.c)

    def _lower_block(self, plan, block, src, dst, res, tmp):
        a, b, d = tmp
        seq = block.cv1
        if len(seq) != 5:
            raise _Unsupported("CIB depth")
        plan.conv(seq[0], dst=(a, 0), src=src)
        plan.conv(seq[1], dst=(b, 0), src=(a, 0))
        if type(seq[2]).__name__ == "YOLORepVGGDW":
            plan.repvggdw(seq[2], dst=(d, 0), src=(b, 0))
        else:
            plan.conv(seq[2], dst=(d, 0), src=(b, 0))
        plan.conv(seq[3], dst=(a, 0), src=(d, 0))
        plan.conv(seq[4], dst=dst, src=(a, 0), res=res)
