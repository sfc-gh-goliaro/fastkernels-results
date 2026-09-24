"""YOLOv10 detection head (L3 composite) -- fused CUDA implementation.

The eager baseline spends ~0.8 ms on ~1.9 GFLOP of work because the one2one
export path issues ~140 tiny kernels (conv / batch-norm / SiLU per layer, then
DFL + a two-stage top-k). This candidate keeps the module structure (so weights
load unchanged) but runs the whole head from a single host call that issues 11
kernels: one NCHW->NHWC transpose, eight fused convolutions (all three pyramid
levels per launch), one per-anchor class max and one select/decode kernel.

See ``_CUDA_SRC`` for the kernels. The epilogues round to fp16 exactly where the
eager chain rounds, which keeps the degenerate top-k tie-break (nearly every
anchor ties on this benchmark's weight distribution) aligned with the reference.
"""

from __future__ import annotations

import math
import copy
import os
import tempfile
import threading

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.sigmoid import Sigmoid
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_dfl import YOLODFL


def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1):
    lt, rb = distance.split([2, 2], dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def v10postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, dim=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    return boxes, scores, labels


# ===========================================================================
# Fused CUDA extension
# ===========================================================================
_CPP_SRC = r"""
#include <torch/extension.h>
void yolo_head_forward(const at::Tensor& x0, const at::Tensor& x1, const at::Tensor& x2,
                       const at::Tensor& wbuf, const at::Tensor& pbuf, const at::Tensor& work,
                       const at::Tensor& out, const at::Tensor& plan);
"""

_CUDA_SRC = r"""
// ---------------------------------------------------------------------------
// Fused YOLOv10 one2one detection head (export path) for NVIDIA B200 / sm_100.
//
// The eager baseline issues ~140 kernels (conv / batch-norm / SiLU per layer,
// then DFL + a two-stage top-k) for ~1.9 GFLOP of work, so it is launch bound.
// One host call here enqueues the whole head as 21 kernels:
//
//   k_trans    NCHW -> NHWC for the three feature maps
//   k_dense    8 fused convolutions per level; implicit GEMM on tensor cores,
//              batch-norm folded in, SiLU in the epilogue, and (FUSE_DW) the
//              separable blocks' depthwise 3x3 folded into the A-tile staging
//   k_clsmax   per-anchor max over classes -> fp16 sigmoid score
//   k_select   two-stage top-k + DFL decode + emit, one block per image
//
// The six (branch, level) chains are independent and none of them fills 148 SMs
// on its own, so each runs on its own stream; the fixed sequence is captured
// into a CUDA graph (see yolo_head_forward) to keep the host out of the way.
//
// Numerics are deliberately *rounding faithful*: every intermediate is rounded
// to fp16 exactly where the eager chain rounds (conv -> bn -> silu, and the
// accumulator before a conv bias). The benchmark's weights make the class
// logits tiny, so sigmoid collapses onto ~56 distinct fp16 values and the
// reference top-k is decided entirely by index-order tie-breaking: matching it
// requires the class logits to come out bit-identical, which they do.
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define MAXJ 3
#define DTHREADS 256

// ------------------------------- job records -------------------------------
struct TJob {            // NCHW -> NHWC
  __half* out;
  int slot;              // index into the device pointer table (graph friendly)
  int C, hw, B, npt, nct, tileBase, nTiles;
};
struct DJob {            // dense conv (1x1 / 3x3), NHWC
  const __half* in; const __half* w; const float* sc; const float* bs;
  const __half* bias; __half* out;
  int cin, H, W, npix, tileBase, nTiles;
  const __half* wdw; const float* scdw; const float* bsdw;   // fused depthwise pre-pass
};
struct CJob {            // per-anchor class max
  const __half* cls; __half* score;
  int npix, hw, anchorBase, tileBase, nTiles;
};
struct SJob {            // select / decode
  const __half* box; const __half* cls;
  int hw, W, H, anchorBase; float stride;
};
template<class J> struct Jobs { J j[MAXJ]; int n; };

__device__ __forceinline__ int pick_job(int blk, int n, const int* tileBase) {
  int ji = 0;
#pragma unroll
  for (int i = 1; i < MAXJ; ++i) if (i < n && blk >= tileBase[i]) ji = i;
  return ji;
}

// --------------------------- async shared-memory staging -------------------
// cp.async with a zero-fill src-size doubles as the 3x3 halo pad: an
// out-of-range tap writes 16 zero bytes without touching global memory.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const int sz = pred ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(sz));
}
#define CP_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define CP_WAIT(N)  asm volatile("cp.async.wait_group %0;\n" :: "n"(N))

// --------------------------- epilogue (rounding faithful) -------------------
// conv accumulator -> fp16 -> batch-norm (fp32 math on the fp16 value) -> fp16
// -> SiLU (fp32 math on the fp16 value) -> fp16.
template<bool SILU>
__device__ __forceinline__ __half epi_bn(float acc, float sc, float bs) {
  float x = __half2float(__float2half(acc));
  __half h = __float2half(x * sc + bs);
  if (SILU) {
    float f = __half2float(h);
    h = __float2half(f / (1.f + expf(-f)));
  }
  return h;
}
// cuDNN rounds the accumulator to fp16 *before* folding in the bias.
__device__ __forceinline__ __half epi_bias(float acc, float b) {
  return __float2half(__half2float(__float2half(acc)) + b);
}

// ------------------------------- NCHW -> NHWC ------------------------------
#define TP 64             // pixels per transpose tile
#define TC 32             // channels per transpose tile
__global__ __launch_bounds__(DTHREADS) void k_trans(Jobs<TJob> jobs, int tb0, int tb1, int tb2,
                                                   void* const* ptab) {
  const int tbs[MAXJ] = {tb0, tb1, tb2};
  const TJob J = jobs.j[pick_job(blockIdx.x, jobs.n, tbs)];
  const __half* const in = (const __half*)ptab[J.slot];
  int lt = blockIdx.x - J.tileBase;
  const int pt = lt % J.npt; lt /= J.npt;
  const int ct = lt % J.nct, n = lt / J.nct;

  __shared__ __half s[TC][TP + 8];
  const int tid = threadIdx.x;
  {                                   // read: 32 channels x 64 pixels, coalesced
    const int c = tid >> 3, g = (tid & 7) << 3;
    const int ch = ct * TC + c, p = pt * TP + g;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (ch < J.C && p + 7 < J.hw) {
      v = __ldg((const uint4*)(in + ((size_t)n * J.C + ch) * J.hw + p));
    } else if (ch < J.C && p < J.hw) {
      __half __align__(16) tmp[8];
#pragma unroll
      for (int e = 0; e < 8; ++e)
        tmp[e] = (p + e < J.hw) ? in[((size_t)n * J.C + ch) * J.hw + p + e] : __float2half(0.f);
      v = *(const uint4*)tmp;
    }
    *(uint4*)(&s[c][g]) = v;
  }
  __syncthreads();
  {                                   // write: 64 pixels x 32 channels
    const int p = tid >> 2, q = (tid & 3) << 3;
    const int pg = pt * TP + p, ch = ct * TC + q;
    if (pg < J.hw && ch < J.C) {
      __half __align__(16) o[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = s[q + e][p];
      if (ch + 7 < J.C) *(uint4*)(J.out + ((size_t)n * J.hw + pg) * J.C + ch) = *(const uint4*)o;
      else for (int e = 0; e < 8 && ch + e < J.C; ++e) J.out[((size_t)n * J.hw + pg) * J.C + ch + e] = o[e];
    }
  }
}

// ------------------------- dense conv, implicit GEMM -----------------------
// M = flattened pixels (MT per CTA), N = COUT (whole), K = KS*KS*cin.
//
// Tiling is chosen for *occupancy*, not for arithmetic intensity: the whole head
// is only ~3.7 GFLOP, so with big tiles the grid cannot even fill 148 SMs and the
// kernel just sits on memory latency. A CTA is NW = 2*(COUT/16) warps; each warp
// owns TPW 16x16 output tiles that share their column fragment, so a K step costs
// TPW A-fragments + 1 B-fragment.
//
// Both operands are staged in shared memory: the A tile is gathered per 3x3 tap
// (zero padded at the borders), the B tile is the tap's weight slice, contiguous
// in the packed [tap][cin][cout] layout. Letting each warp pull its own B
// fragments straight from global saturated L1 and cost ~20x.
// FUSE_DW folds the preceding depthwise 3x3 + BN + SiLU into the A-tile staging,
// so a separable block (dw3x3 -> 1x1) costs one kernel and one pass over the
// activations instead of two of each.
template<int COUT, int KS, int MT, int MODE, bool SILU, bool FUSE_DW>
__global__ __launch_bounds__(64 * (COUT / 16)) void k_dense(Jobs<DJob> jobs, int tb0, int tb1, int tb2) {
  constexpr int NCOL = COUT / 16;
  constexpr int NM = MT / 16;
  constexpr int NW = 2 * NCOL;
  constexpr int NTHR = 32 * NW;
  constexpr int TPW = (NM * NCOL) / NW;
  constexpr int LDB = COUT + 8;
  constexpr int NTAP = KS * KS;
  static_assert(NM * NCOL == NW * TPW, "tile/warp mapping must be exact");

  const int tbs[MAXJ] = {tb0, tb1, tb2};
  const DJob J = jobs.j[pick_job(blockIdx.x, jobs.n, tbs)];
  const int cin = J.cin, ldA = cin + 8;
  const int stage = MT * ldA + cin * LDB;            // halves per pipeline stage

  constexpr int NSTAGE = FUSE_DW ? 1 : 2;            // fused path stages A itself
  extern __shared__ char smem[];
  __half* S = (__half*)smem;
  int* meta = (int*)(S + NSTAGE * stage);
  float* Cs = (float*)S;                             // epilogue reuses stage 0

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int p0 = (blockIdx.x - J.tileBase) * MT;

  for (int i = tid; i < MT; i += NTHR) {
    int pix = p0 + i, hh = -1, ww = 0, nb = 0;
    if (pix < J.npix) {
      int q = pix / J.W; ww = pix - q * J.W;
      int n = q / J.H;   hh = q - n * J.H;
      nb = n * J.H * J.W;
    }
    meta[i] = hh; meta[MT + i] = ww; meta[2 * MT + i] = nb;
  }

  // staging maps: the divides are hoisted out of the tap loop
  const int vpr = cin >> 3, wpr = COUT >> 3;
  const int ar = tid / vpr, ac = (tid - ar * vpr) << 3, astep = NTHR / vpr;
  const bool afast = (astep * vpr == NTHR);
  const int bk = tid / wpr, bc = (tid - bk * wpr) << 3;   // NTHR/wpr == 32 by construction
  __syncthreads();

  // Issue tap `t` into pipeline stage t&1. Both operands are staged: the A tile
  // is the tap-shifted input patch, the B tile the tap's weight slice (contiguous
  // in the packed [tap][cin][cout] layout). Letting each warp pull its own wmma B
  // fragments from global instead saturated L1 and cost ~20x.
  auto issue = [&](int t) {
    if (t >= NTAP) { CP_COMMIT(); return; }
    __half* As = S + (t & 1) * stage;
    __half* Bs = As + MT * ldA;
    const int dy = (KS == 1) ? 0 : (t / KS - 1);
    const int dx = (KS == 1) ? 0 : (t % KS - 1);
    if (FUSE_DW) {                          // A comes from the depthwise pre-pass
      for (int k = bk; k < cin; k += 32)
        cp_async16(Bs + k * LDB + bc, J.w + (size_t)k * COUT + bc, true);
      CP_COMMIT();
      for (int r = ar; r < MT; r += astep) {
        const int hh = meta[r], ww = meta[MT + r], nb = meta[2 * MT + r];
        float a[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) a[e] = 0.f;
        if (hh >= 0) {
#pragma unroll
          for (int q = 0; q < 9; ++q) {
            const int h2 = hh + q / 3 - 1, w2 = ww + q % 3 - 1;
            if (h2 < 0 || h2 >= J.H || w2 < 0 || w2 >= J.W) continue;
            const uint4 xv = __ldg((const uint4*)(J.in + ((size_t)(nb + h2 * J.W + w2)) * cin + ac));
            const uint4 wv = __ldg((const uint4*)(J.wdw + (size_t)q * cin + ac));
            const __half* xh = (const __half*)&xv;
            const __half* wh = (const __half*)&wv;
#pragma unroll
            for (int e = 0; e < 8; ++e) a[e] += __half2float(xh[e]) * __half2float(wh[e]);
          }
        }
        __half __align__(16) o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = epi_bn<true>(a[e], J.scdw[ac + e], J.bsdw[ac + e]);
        *(uint4*)(As + r * ldA + ac) = *(const uint4*)o;
      }
      return;
    }
    if (afast) {
      for (int r = ar; r < MT; r += astep) {
        const int hh = meta[r], h2 = hh + dy, w2 = meta[MT + r] + dx;
        const bool ok = (hh >= 0 && h2 >= 0 && h2 < J.H && w2 >= 0 && w2 < J.W);
        const __half* src = ok ? J.in + ((size_t)(meta[2 * MT + r] + h2 * J.W + w2)) * cin + ac : J.in;
        cp_async16(As + r * ldA + ac, src, ok);
      }
    } else {
      for (int v = tid; v < MT * vpr; v += NTHR) {
        const int r = v / vpr, c8 = (v - r * vpr) << 3;
        const int hh = meta[r], h2 = hh + dy, w2 = meta[MT + r] + dx;
        const bool ok = (hh >= 0 && h2 >= 0 && h2 < J.H && w2 >= 0 && w2 < J.W);
        const __half* src = ok ? J.in + ((size_t)(meta[2 * MT + r] + h2 * J.W + w2)) * cin + c8 : J.in;
        cp_async16(As + r * ldA + c8, src, ok);
      }
    }
    const __half* wsrc = J.w + (size_t)t * cin * COUT + bc;
    for (int k = bk; k < cin; k += 32)
      cp_async16(Bs + k * LDB + bc, wsrc + (size_t)k * COUT, true);
    CP_COMMIT();
  };

  const int ncol = warp % NCOL;
  int mm[TPW];
#pragma unroll
  for (int k = 0; k < TPW; ++k) mm[k] = (warp + k * NW) / NCOL;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TPW];
#pragma unroll
  for (int k = 0; k < TPW; ++k) wmma::fill_fragment(acc[k], 0.f);

  issue(0);
  if (!FUSE_DW) issue(1);
  for (int t = 0; t < NTAP; ++t) {
    CP_WAIT(FUSE_DW ? 0 : 1);
    __syncthreads();
    const __half* As = S + (t & 1) * stage;
    const __half* Bs = As + MT * ldA;
    for (int kc = 0; kc < cin; kc += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
      wmma::load_matrix_sync(bf, Bs + (size_t)kc * LDB + ncol * 16, LDB);
#pragma unroll
      for (int k = 0; k < TPW; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
        wmma::load_matrix_sync(af, As + mm[k] * 16 * ldA + kc, ldA);
        wmma::mma_sync(acc[k], af, bf, acc[k]);
      }
    }
    __syncthreads();
    if (!FUSE_DW) issue(t + 2);
  }
  CP_WAIT(0);

  float* myCs = Cs + warp * 256;
  const int rr = lane >> 1, h8 = (lane & 1) << 3;
#pragma unroll
  for (int k = 0; k < TPW; ++k) {
    __syncwarp();
    wmma::store_matrix_sync(myCs, acc[k], 16, wmma::mem_row_major);
    __syncwarp();
    const int pix = p0 + mm[k] * 16 + rr;
    if (pix < J.npix) {
      __half __align__(16) o[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        int c = ncol * 16 + h8 + e;
        float v = myCs[rr * 16 + h8 + e];
        o[e] = (MODE == 0) ? epi_bn<SILU>(v, J.sc[c], J.bs[c])
                           : epi_bias(v, __half2float(J.bias[c]));
      }
      *(uint4*)(J.out + (size_t)pix * COUT + ncol * 16 + h8) = *(const uint4*)o;
    }
  }
}

// -------------------- per-anchor max over classes -> sigmoid ---------------
__global__ __launch_bounds__(DTHREADS) void k_clsmax(Jobs<CJob> jobs, int tb0, int tb1, int tb2,
                                                     int nc, int A) {
  const int tbs[MAXJ] = {tb0, tb1, tb2};
  const CJob J = jobs.j[pick_job(blockIdx.x, jobs.n, tbs)];
  int idx = (blockIdx.x - J.tileBase) * DTHREADS + threadIdx.x;
  if (idx >= J.npix) return;
  int n = idx / J.hw, p = idx - n * J.hw;
  const __half* c = J.cls + (size_t)idx * nc;
  float m = -1e30f;
  for (int i = 0; i < nc; i += 8) {
    uint4 v = *(const uint4*)(c + i);
    const __half* h = (const __half*)&v;
#pragma unroll
    for (int e = 0; e < 8; ++e) m = fmaxf(m, __half2float(h[e]));
  }
  J.score[(size_t)n * A + J.anchorBase + p] = __float2half(1.f / (1.f + expf(-m)));
}

// ------------------------------ select helpers -----------------------------
// Block-wide exclusive scan of one value per thread (3 barriers, used once).
__device__ __forceinline__ int block_excl_scan_val(int val, int* ws, int* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  int s = val;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int o = __shfl_up_sync(0xffffffffu, s, off);
    if (lane >= off) s += o;
  }
  if (lane == 31) ws[warp] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    int v2 = (threadIdx.x < nwarps) ? ws[threadIdx.x] : 0;
    int t = v2;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int o = __shfl_up_sync(0xffffffffu, t, off);
      if (lane >= off) t += o;
    }
    if (threadIdx.x < nwarps) ws[threadIdx.x] = t - v2;
    if (threadIdx.x == 31) ws[32] = t;
  }
  __syncthreads();
  *total = ws[32];
  const int r = ws[warp] + s - val;
  __syncthreads();
  return r;
}

// Two-value block reduction (sum, max) in two barriers. Shared-memory atomics
// are avoided on purpose: with 32 warps all hitting one address they serialise
// and dominated the whole select kernel.
__device__ __forceinline__ void block_reduce2(int csum, int cmax, int* ws, int* outc, int* outm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    csum += __shfl_down_sync(0xffffffffu, csum, off);
    cmax = max(cmax, __shfl_down_sync(0xffffffffu, cmax, off));
  }
  if (lane == 0) { ws[warp] = csum; ws[32 + warp] = cmax; }
  __syncthreads();
  if (threadIdx.x < 32) {
    int a = (threadIdx.x < nwarps) ? ws[threadIdx.x] : 0;
    int b = (threadIdx.x < nwarps) ? ws[32 + threadIdx.x] : 0;
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      a += __shfl_down_sync(0xffffffffu, a, off);
      b = max(b, __shfl_down_sync(0xffffffffu, b, off));
    }
    if (threadIdx.x == 0) { ws[64] = a; ws[65] = b; }
  }
  __syncthreads();
  *outc = ws[64]; *outm = ws[65];
}

__device__ __forceinline__ int block_count_ge(const unsigned short* v, int nv,
                                              unsigned thr, int* ws) {
  int c = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) c += ((unsigned)v[i] >= thr);
  int oc, om;
  block_reduce2(c, 0, ws, &oc, &om);
  return oc;
}

// One pass yielding both #(v[i] == cur) and max{v[i] : v[i] < cur}.
__device__ __forceinline__ void block_eq_next(const unsigned short* v, int nv, unsigned cur,
                                              int* ws, int* outc, unsigned* outm) {
  int c = 0, m = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const int x = (int)v[i];
    c += (x == (int)cur);
    if (x < (int)cur) m = max(m, x);
  }
  int oc, om;
  block_reduce2(c, m, ws, &oc, &om);
  *outc = oc; *outm = (unsigned)om;
}

__device__ __forceinline__ unsigned block_max(const unsigned short* v, int nv, int* ws) {
  int m = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) m = max(m, (int)v[i]);
  int oc, om;
  block_reduce2(0, m, ws, &oc, &om);
  return (unsigned)om;
}

// Select the K largest of v[0..nv) ordered by (value desc, index asc) -- exactly
// torch.topk's tie-break. sel[rank] = index.
//
// The threshold walk exploits how degenerate the benchmark's score distribution
// is: sigmoid of near-zero logits collapses onto a handful of fp16 values, so the
// largest value alone usually already covers all K slots and two passes suffice.
// Distinct-value walking is capped, with a key bisection as the fallback so a
// well-spread distribution stays bounded.
__device__ void select_topk(const unsigned short* v, int nv, int K, int* sel,
                            int* res, int* cand, int* ws) {
  if (threadIdx.x == 0) res[4] = 0;
  __syncthreads();

  unsigned T = 0;
  int cntGT = 0;
  bool done = false;
  {
    unsigned cur = block_max(v, nv, ws);
    int cnt = 0;
    for (int it = 0; it < 8; ++it) {
      int c; unsigned nxt;
      block_eq_next(v, nv, cur, ws, &c, &nxt);
      if (cnt + c >= K) { T = cur; cntGT = cnt; done = true; break; }
      cnt += c;
      cur = nxt;
    }
  }
  if (!done) {                                  // bisect the key space instead
    unsigned lo = 0, hi = 0xFFFFu;
    while (lo < hi) {
      const unsigned mid = (lo + hi + 1u) >> 1;
      if (block_count_ge(v, nv, mid, ws) >= K) lo = mid; else hi = mid - 1;
    }
    T = lo;
  }

  // one pass over a contiguous per-thread segment: compact the > T group (order
  // is recovered by rank-counting) and count the == T group in index order
  const int seg = (nv + (int)blockDim.x - 1) / (int)blockDim.x;
  const int i0 = (int)threadIdx.x * seg, i1 = min(nv, i0 + seg);
  int myeq = 0;
  for (int i = i0; i < i1; ++i) {
    const unsigned x = v[i];
    if (x > T) { int sl = atomicAdd(&res[4], 1); if (sl < K) cand[sl] = i; }
    else if (x == T) ++myeq;
  }
  int total;
  const int start = block_excl_scan_val(myeq, ws, &total);
  cntGT = res[4];

  const int ngt = min(cntGT, K);          // cntGT < K by construction; clamp anyway
  for (int a = threadIdx.x; a < ngt; a += blockDim.x) {
    const int ia = cand[a];
    const unsigned ka = ((65535u - (unsigned)v[ia]) << 16) | (unsigned)ia;
    int rank = 0;
    for (int b = 0; b < ngt; ++b) {
      const int ib = cand[b];
      rank += ((((65535u - (unsigned)v[ib]) << 16) | (unsigned)ib) < ka);
    }
    if (rank < K) sel[rank] = ia;
  }
  const int need = K - ngt;
  if (start < need) {
    int k = start;
    for (int i = i0; i < i1 && k < need; ++i)
      if ((unsigned)v[i] == T) sel[ngt + k++] = i;
  }
  __syncthreads();
}

// --------------------- select + DFL decode + emit --------------------------
// One CTA per image: stage-1 top-k over anchors, stage-2 top-k over the selected
// rows' classes, then DFL-decode only the 300 surviving boxes. RM (= reg_max) is
// a template parameter so the DFL gathers vectorise and stay in registers -- as a
// runtime bound the j-loops became a 192-deep dependent global load chain.
template<int RM>
__global__ __launch_bounds__(1024) void k_select(Jobs<SJob> jobs, const __half* score,
                                                void* const* ptab, int A, int nc, int reg_max, int K) {
  __half* const out = (__half*)ptab[3];
  const int n = blockIdx.x;
  const int nbox = 4 * RM;
  (void)reg_max;
  extern __shared__ char smem[];
  char* sm = smem;
  int* res = (int*)sm; sm += 8 * 4;
  int* ws = (int*)sm; sm += 72 * 4;
  int* cand = (int*)sm; sm += K * 4;
  int* sel = (int*)sm; sm += K * 4;
  int* fsel = (int*)sm; sm += K * 4;
  float* sxA = (float*)sm; sm += K * 4;
  float* syA = (float*)sm; sm += K * 4;
  float* stA = (float*)sm; sm += K * 4;
  float* dsm = (float*)sm; sm += K * 4 * 4;
  sm = (char*)(((uintptr_t)sm + 15) & ~(uintptr_t)15);
  const __half** clsR = (const __half**)sm; sm += K * 8;
  const __half** boxR = (const __half**)sm; sm += K * 8;
  unsigned short* sv = (unsigned short*)sm; sm += ((A * 2 + 15) & ~15);
  unsigned short* s2v = (unsigned short*)sm;

  const unsigned short* sc0 = (const unsigned short*)(score + (size_t)n * A);
  for (int i = threadIdx.x * 8; i + 8 <= A; i += blockDim.x * 8)
    *(uint4*)(sv + i) = __ldg((const uint4*)(sc0 + i));
  for (int i = (A & ~7) + threadIdx.x; i < A; i += blockDim.x) sv[i] = sc0[i];
  __syncthreads();

  select_topk(sv, A, K, sel, res, cand, ws);

  // per-row geometry + row pointers
  for (int r = threadIdx.x; r < K; r += blockDim.x) {
    const int a = sel[r];
    int L = 0;
#pragma unroll
    for (int i = 1; i < MAXJ; ++i) if (i < jobs.n && a >= jobs.j[i].anchorBase) L = i;
    const SJob J = jobs.j[L];
    const int p = a - J.anchorBase;
    const int hh = p / J.W, wwv = p - hh * J.W;
    sxA[r] = (float)wwv + 0.5f; syA[r] = (float)hh + 0.5f; stA[r] = J.stride;
    clsR[r] = J.cls + (size_t)((size_t)n * J.hw + p) * nc;
    boxR[r] = J.box + (size_t)((size_t)n * J.hw + p) * nbox;
  }
  __syncthreads();

  const int ncv = nc >> 3;
  for (int v = threadIdx.x; v < K * ncv; v += blockDim.x) {
    const int r = v / ncv, c8 = (v - r * ncv) << 3;
    uint4 q = *(const uint4*)(clsR[r] + c8);
    const __half* qh = (const __half*)&q;
    unsigned short __align__(16) tmp[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      __half sg = __float2half(1.f / (1.f + expf(-__half2float(qh[e]))));
      tmp[e] = *(const unsigned short*)&sg;
    }
    *(uint4*)(s2v + r * nc + c8) = *(const uint4*)tmp;
  }
  __syncthreads();

  select_topk(s2v, K * nc, K, fsel, res, cand, ws);

  // DFL: one thread per (row, distribution group)
  for (int idx = threadIdx.x; idx < K * 4; idx += blockDim.x) {
    const int r = idx >> 2, g = idx & 3;
    const __half* bx = boxR[fsel[r] / nc] + g * RM;
    float bv[RM];
#pragma unroll
    for (int j = 0; j < RM; j += 8) {
      uint4 q = *(const uint4*)(bx + j);
      const __half* qh = (const __half*)&q;
#pragma unroll
      for (int e = 0; e < 8; ++e) bv[j + e] = __half2float(qh[e]);
    }
    float mx = -1e30f;
#pragma unroll
    for (int j = 0; j < RM; ++j) mx = fmaxf(mx, bv[j]);
    float ev[RM], sum = 0.f;
#pragma unroll
    for (int j = 0; j < RM; ++j) { ev[j] = expf(bv[j] - mx); sum += ev[j]; }
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RM; ++j) acc += (float)j * __half2float(__float2half(ev[j] / sum));
    dsm[idx] = __half2float(__float2half(acc));
  }
  __syncthreads();

  for (int r = threadIdx.x; r < K; r += blockDim.x) {
    const int fi = fsel[r];
    const int row = fi / nc, cl = fi - row * nc;
    const float* d = dsm + r * 4;
    const float sx = sxA[row], sy = syA[row], st = stA[row];
    float x1 = __half2float(__float2half(sx - d[0]));
    float y1 = __half2float(__float2half(sy - d[1]));
    float x2 = __half2float(__float2half(sx + d[2]));
    float y2 = __half2float(__float2half(sy + d[3]));
    float cx = __half2float(__float2half(__half2float(__float2half(x1 + x2)) * 0.5f));
    float cy = __half2float(__float2half(__half2float(__float2half(y1 + y2)) * 0.5f));
    float bw = __half2float(__float2half(x2 - x1));
    float bh = __half2float(__float2half(y2 - y1));
    cx = __half2float(__float2half(cx * st)); cy = __half2float(__float2half(cy * st));
    bw = __half2float(__float2half(bw * st)); bh = __half2float(__float2half(bh * st));
    const float hw2 = __half2float(__float2half(bw * 0.5f));
    const float hh2 = __half2float(__float2half(bh * 0.5f));
    __half* o = out + ((size_t)n * K + r) * 6;
    o[0] = __float2half(cx - hw2);
    o[1] = __float2half(cy - hh2);
    o[2] = __float2half(cx + hw2);
    o[3] = __float2half(cy + hh2);
    o[4] = *(const __half*)&s2v[fi];
    o[5] = __float2half((float)cl);
  }
}

__global__ void k_setptrs(void** ptab, void* a, void* b, void* c, void* d) {
  ptab[0] = a; ptab[1] = b; ptab[2] = c; ptab[3] = d;
}

// ------------------------------- host launcher -----------------------------
// Plan layout (int64, CPU): 3 x F_NF per-level fields, then the globals.
#include <unordered_map>

enum {
  F_H = 0, F_W, F_NPIX, F_CH, F_IN,
  F_A0W, F_A0P, F_A0O,
  F_A1W, F_A1P, F_A1O,
  F_A2W, F_A2B, F_A2O,
  F_B0W, F_B0P, F_B0O,
  F_B1W, F_B1P, F_B1O,
  F_B2W, F_B2P, F_B2O,
  F_B3W, F_B3P, F_B3O,
  F_B4W, F_B4B, F_B4O,
  F_ABASE, F_STRIDE, F_NF
};

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int COUT, int KS, int MT_, int MODE, bool SILU, bool FDW>
static void launch_dense_t(Jobs<DJob>& jb, int tb, size_t smem, cudaStream_t st) {
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k_dense<COUT, KS, MT_, MODE, SILU, FDW>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 150 * 1024);
    once = true;
  }
  k_dense<COUT, KS, MT_, MODE, SILU, FDW><<<tb, 64 * (COUT / 16), smem, st>>>(
      jb, jb.j[0].tileBase, jb.j[1].tileBase, jb.j[2].tileBase);
}

#define DISPATCH_MT(CO, KS_, MODE, SILU, FDW)                                   \
  if (mt == 64) launch_dense_t<CO, KS_, 64, MODE, SILU, FDW>(jb, tb, smem, st); \
  else          launch_dense_t<CO, KS_, 32, MODE, SILU, FDW>(jb, tb, smem, st);
#define DISPATCH_KS(CO, MODE, SILU)                                             \
  if (ks == 1) {                                                                \
    if (fdw) { DISPATCH_MT(CO, 1, MODE, SILU, true) }                           \
    else     { DISPATCH_MT(CO, 1, MODE, SILU, false) }                          \
  } else     { DISPATCH_MT(CO, 3, MODE, SILU, false) }

static void launch_dense(int cout, int ks, int mode, bool silu, bool fdw, int mt,
                         Jobs<DJob>& jb, int tb, size_t smem, cudaStream_t st) {
  if (cout == 64) {
    if (mode == 0) { if (silu) { DISPATCH_KS(64, 0, true) } else { DISPATCH_KS(64, 0, false) } }
    else           { DISPATCH_KS(64, 1, false) }
  } else {
    if (mode == 0) { if (silu) { DISPATCH_KS(80, 0, true) } else { DISPATCH_KS(80, 0, false) } }
    else           { DISPATCH_KS(80, 1, false) }
  }
}

// The two branches (box regression and classification) are independent given the
// transposed input, and neither fills the GPU on its own -- every conv here runs
// at ~30% SM occupancy because the grids are small. Running them on two streams
// lets their blocks interleave.
#define NSTREAM 6                 // (level, branch) pairs
struct Side {
  cudaStream_t s[NSTREAM];
  cudaEvent_t ready, done[NSTREAM];
  Side() {
    cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
    for (int i = 0; i < NSTREAM; ++i) {
      cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
      cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming);
    }
  }
};
static Side& side() { static Side s; return s; }



static void enqueue_all(const int64_t* P, const __half* W, const float* PB, __half* WK,
                        void** ptab, cudaStream_t main) {
  Side& sd = side();
  cudaStream_t st = main;
  int lvl0 = 0, lvl1 = MAXJ;        // level range for the next stage helper call
  const int64_t* G = P + 3 * F_NF;
  const int B = (int)G[0], nc = (int)G[1], c2 = (int)G[2], c3 = (int)G[3];
  const int nbox = (int)G[4], A = (int)G[5], K = (int)G[7], reg_max = (int)G[8];
  __half* scoreBuf = WK + G[6];

  // ---- 0. NCHW -> NHWC for the three feature maps
  {
    Jobs<TJob> jb; memset(&jb, 0, sizeof(jb)); jb.n = MAXJ;
    int tb = 0;
    for (int i = 0; i < MAXJ; ++i) {
      const int64_t* L = P + i * F_NF;
      TJob& j = jb.j[i];
      j.slot = i; j.out = WK + L[F_IN];
      j.C = (int)L[F_CH]; j.hw = (int)(L[F_H] * L[F_W]); j.B = B;
      j.npt = cdiv(j.hw, TP); j.nct = cdiv(j.C, TC);
      j.tileBase = tb; j.nTiles = j.npt * j.nct * B; tb += j.nTiles;
    }
    k_trans<<<tb, DTHREADS, 0, st>>>(jb, jb.j[0].tileBase, jb.j[1].tileBase, jb.j[2].tileBase, ptab);
  }

  // ---- dense conv stage helper -------------------------------------------
  // Levels are packed into one launch, but only when their input channel count
  // matches: the shared-memory request is per launch, and making level 0 (which
  // owns ~80% of the tiles) pay level 2's 256-channel footprint would halve its
  // occupancy.
  auto dense_smem = [](int cin, int cout, int mt, bool fdw) {
    const size_t st = ((size_t)mt * (cin + 8) + (size_t)cin * (cout + 8)) * 2;
    // the fused-depthwise path stages A itself, so it needs a single stage only
    return std::max((fdw ? 1 : 2) * st, (size_t)(cout / 16) * 2 * 1024) + 3 * (size_t)mt * 4;
  };
  // fdwW/fdwP >= 0 folds a depthwise 3x3 (weights/bn at those plan fields) into
  // the A staging of this 1x1 convolution.
  auto dense_fused = [&](int cout, int ks, int mode, bool silu, int cinSel,
                         int fIn, int fW, int fPB, int fOut, int fdwW, int fdwP) {
    int done = 0;
    for (int g = lvl0; g < lvl1; ++g) {
      if (done >> g & 1) continue;
      int cin0 = (cinSel < 0) ? (int)P[g * F_NF + F_CH] : cinSel;
      Jobs<DJob> jb; memset(&jb, 0, sizeof(jb)); jb.n = 0;
      int tb = 0;
      for (int i = g; i < lvl1; ++i) {
        const int64_t* L = P + i * F_NF;
        int cin = (cinSel < 0) ? (int)L[F_CH] : cinSel;
        if (cin != cin0) continue;
        done |= 1 << i;
        DJob& j = jb.j[jb.n++];
        j.in = WK + L[fIn];
        j.w = W + L[fW];
        if (mode == 0) { j.sc = PB + L[fPB]; j.bs = PB + L[fPB] + cout; j.bias = nullptr; }
        else           { j.sc = nullptr; j.bs = nullptr; j.bias = W + L[fPB]; }
        j.out = WK + L[fOut];
        j.cin = cin;
        j.H = (int)L[F_H]; j.W = (int)L[F_W]; j.npix = (int)L[F_NPIX];
        if (fdwW >= 0) { j.wdw = W + L[fdwW]; j.scdw = PB + L[fdwP]; j.bsdw = PB + L[fdwP] + cin; }
        j.tileBase = tb; j.nTiles = j.npix; tb += j.nTiles;   // patched below
      }
      // Prefer the wide tile (2 output tiles per warp, fewer fragment loads) only
      // when it still yields ~2 CTAs per SM; otherwise halve it for more blocks.
      const int mt = (cdiv(tb, 64) >= 2 * 148) ? 64 : 32;
      tb = 0;
      for (int i = 0; i < jb.n; ++i) {
        jb.j[i].tileBase = tb;
        jb.j[i].nTiles = cdiv(jb.j[i].npix, mt);
        tb += jb.j[i].nTiles;
      }
      launch_dense(cout, ks, mode, silu, fdwW >= 0, mt, jb, tb,
                   dense_smem(cin0, cout, mt, fdwW >= 0), st);
    }
  };
  auto dense = [&](int cout, int ks, int mode, bool silu, int cinSel,
                   int fIn, int fW, int fPB, int fOut) {
    dense_fused(cout, ks, mode, silu, cinSel, fIn, fW, fPB, fOut, -1, -1);
  };
  // Six independent chains: {box, cls} x {level}. Each is launched on its own
  // stream so their (individually tiny) grids interleave on the SMs.
  cudaEventRecord(sd.ready, main);
  for (int i = 0; i < NSTREAM; ++i) cudaStreamWaitEvent(sd.s[i], sd.ready, 0);
  for (int L = 0; L < MAXJ; ++L) {
    lvl0 = L; lvl1 = L + 1;
    // box: 3x3 ch->c2, 3x3 c2->c2, 1x1 c2->4*reg_max
    st = sd.s[L];
    dense(c2,   3, 0, true,  -1, F_IN,   F_A0W, F_A0P, F_A0O);
    dense(c2,   3, 0, true,  c2, F_A0O,  F_A1W, F_A1P, F_A1O);
    dense(nbox, 1, 1, false, c2, F_A1O,  F_A2W, F_A2B, F_A2O);
    // cls: dw3x3 ch, 1x1 ch->c3, dw3x3 c3, 1x1 c3->c3, 1x1 c3->nc
    st = sd.s[MAXJ + L];
    dense_fused(c3, 1, 0, true, -1, F_IN,  F_B1W, F_B1P, F_B1O, F_B0W, F_B0P);
    dense_fused(c3, 1, 0, true, c3, F_B1O, F_B3W, F_B3P, F_B3O, F_B2W, F_B2P);
    dense(nc, 1, 1, false, c3, F_B3O, F_B4W, F_B4B, F_B4O);
  }
  for (int i = 0; i < NSTREAM; ++i) {
    cudaEventRecord(sd.done[i], sd.s[i]);
    cudaStreamWaitEvent(main, sd.done[i], 0);
  }
  lvl0 = 0; lvl1 = MAXJ;
  st = main;

  // ---- 9. per-anchor class max -> fp16 sigmoid score
  {
    Jobs<CJob> jb; memset(&jb, 0, sizeof(jb)); jb.n = MAXJ;
    int tb = 0;
    for (int i = 0; i < MAXJ; ++i) {
      const int64_t* L = P + i * F_NF;
      CJob& j = jb.j[i];
      j.cls = WK + L[F_B4O]; j.score = scoreBuf;
      j.npix = (int)L[F_NPIX]; j.hw = (int)(L[F_H] * L[F_W]); j.anchorBase = (int)L[F_ABASE];
      j.tileBase = tb; j.nTiles = cdiv(j.npix, DTHREADS); tb += j.nTiles;
    }
    k_clsmax<<<tb, DTHREADS, 0, st>>>(jb, jb.j[0].tileBase, jb.j[1].tileBase, jb.j[2].tileBase, nc, A);
  }

  // ---- 10. two-stage top-k (torch.topk tie-break) + DFL decode + emit
  {
    Jobs<SJob> jb; memset(&jb, 0, sizeof(jb)); jb.n = MAXJ;
    for (int i = 0; i < MAXJ; ++i) {
      const int64_t* L = P + i * F_NF;
      SJob& j = jb.j[i];
      j.box = WK + L[F_A2O]; j.cls = WK + L[F_B4O];
      j.hw = (int)(L[F_H] * L[F_W]); j.W = (int)L[F_W]; j.H = (int)L[F_H];
      j.anchorBase = (int)L[F_ABASE]; j.stride = (float)L[F_STRIDE];
    }
    size_t sz = 8 * 4 + 72 * 4 + (size_t)K * 4 * 10;
    sz = (sz + 15) & ~(size_t)15;
    sz += (size_t)K * 16;
    sz += ((A * 2 + 15) & ~15);
    sz += (size_t)K * nc * 2 + 16;
    static bool once = false;
    if (!once) {
      cudaFuncSetAttribute(k_select<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
      cudaFuncSetAttribute(k_select<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
      cudaFuncSetAttribute(k_select<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
      once = true;
    }
    if (reg_max == 16)      k_select<16><<<B, 1024, sz, st>>>(jb, scoreBuf, ptab, A, nc, reg_max, K);
    else if (reg_max == 8)  k_select<8><<<B, 1024, sz, st>>>(jb, scoreBuf, ptab, A, nc, reg_max, K);
    else                    k_select<32><<<B, 1024, sz, st>>>(jb, scoreBuf, ptab, A, nc, reg_max, K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// The head is 27 small kernels across 7 streams; at ~2 us of driver time each
// the host cannot keep the GPU fed (and on a busy host it becomes the limit
// outright). The sequence is fixed for a given shape, so it is captured once
// into a CUDA graph and replayed. Per call only the four tensor addresses
// change, and they are published into a device-side table by k_setptrs -- the
// graph itself never bakes in an input pointer.
struct GraphSlot { int state = 0; cudaGraphExec_t exec = nullptr; };
static std::unordered_map<long long, GraphSlot>& graph_cache() {
  static std::unordered_map<long long, GraphSlot> m;
  return m;
}
static cudaStream_t capture_stream() {
  static cudaStream_t s = nullptr;
  if (!s) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  return s;
}

void yolo_head_forward(const at::Tensor& x0, const at::Tensor& x1, const at::Tensor& x2,
                       const at::Tensor& wbuf, const at::Tensor& pbuf, const at::Tensor& work,
                       const at::Tensor& out, const at::Tensor& plan) {
  cudaStream_t main = at::cuda::getCurrentCUDAStream();
  const int64_t* P = plan.data_ptr<int64_t>();
  const int64_t* G = P + 3 * F_NF;
  const __half* W = (const __half*)wbuf.data_ptr<at::Half>();
  const float* PB = pbuf.data_ptr<float>();
  __half* WK = (__half*)work.data_ptr<at::Half>();
  void** ptab = (void**)(WK + G[11]);

  k_setptrs<<<1, 1, 0, main>>>(ptab, x0.data_ptr(), x1.data_ptr(), x2.data_ptr(), out.data_ptr());

  cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(main, &cap_status);
  const bool capturing = (cap_status != cudaStreamCaptureStatusNone);
  auto& cache = graph_cache();
  if (cache.size() > 8 && cache.find((long long)G[10]) == cache.end()) {
    for (auto& kv : cache) if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
    cache.clear();                       // keep a bound on retained graph execs
  }
  GraphSlot& gs = cache[(long long)G[10]];
  static const bool nograph = getenv("FK_YOLO_NOGRAPH") != nullptr;
  if (capturing || nograph) {
    enqueue_all(P, W, PB, WK, ptab, main);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (gs.state == 0) {                       // warm up outside capture
    enqueue_all(P, W, PB, WK, ptab, main);
    gs.state = 1;
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (gs.state == 1) {
    cudaStream_t cap = capture_stream();
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, main);
    cudaStreamWaitEvent(cap, ev, 0);
    cudaGraph_t g = nullptr;
    if (cudaStreamBeginCapture(cap, cudaStreamCaptureModeRelaxed) == cudaSuccess) {
      enqueue_all(P, W, PB, WK, ptab, cap);
      if (cudaStreamEndCapture(cap, &g) == cudaSuccess && g &&
          cudaGraphInstantiate(&gs.exec, g, nullptr, nullptr, 0) == cudaSuccess) {
        gs.state = 2;
        if (getenv("FK_YOLO_VERBOSE")) fprintf(stderr, "[fk_yolo] graph captured\n");
      } else {
        gs.state = 3;                        // capture failed: always go direct
        if (getenv("FK_YOLO_VERBOSE")) fprintf(stderr, "[fk_yolo] capture FAILED (end/instantiate)\n");
      }
      if (g) cudaGraphDestroy(g);
    } else {
      gs.state = 3;
      if (getenv("FK_YOLO_VERBOSE")) fprintf(stderr, "[fk_yolo] capture FAILED (begin)\n");
    }
    cudaEventDestroy(ev);
    if (gs.state != 2) {
      enqueue_all(P, W, PB, WK, ptab, main);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  if (gs.state == 2) cudaGraphLaunch(gs.exec, main);
  else enqueue_all(P, W, PB, WK, ptab, main);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_EXT = None
_EXT_TRIED = False
_EXT_LOCK = threading.Lock()


def _ext():
    """Build (once) and return the fused extension, or None if unavailable."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    with _EXT_LOCK:
        if _EXT_TRIED:
            return _EXT
        try:
            from torch.utils.cpp_extension import load_inline
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            flags = ["-O3", "-lineinfo",
                     f"-gencode=arch=compute_{arch},code=sm_{arch}"]
            build_dir = os.path.join(
                os.environ.get("FK_YOLO_BUILD_DIR", tempfile.gettempdir()),
                f"fk_yolov10_head_sm{arch}")
            os.makedirs(build_dir, exist_ok=True)
            _EXT = load_inline(
                name="fk_yolov10_head",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["yolo_head_forward"],
                extra_cuda_cflags=flags,
                extra_cflags=["-O3"],
                build_directory=build_dir,
                verbose=False,
            )
        except Exception:
            _EXT = None
        _EXT_TRIED = True
    return _EXT


class _Unsupported(Exception):
    pass


# plan field layout -- must match the enum in the CUDA source
_NF = 31
(F_H, F_W, F_NPIX, F_CH, F_IN,
 F_A0W, F_A0P, F_A0O, F_A1W, F_A1P, F_A1O, F_A2W, F_A2B, F_A2O,
 F_B0W, F_B0P, F_B0O, F_B1W, F_B1P, F_B1O, F_B2W, F_B2P, F_B2O,
 F_B3W, F_B3P, F_B3O, F_B4W, F_B4B, F_B4O, F_ABASE, F_STRIDE) = range(_NF)

_OK_COUT = (16, 32, 48, 64, 80)          # cout values the kernels are built for


_GEN = [0]


def _next_gen():
    """Unique id per built plan; keys the extension's CUDA-graph cache."""
    _GEN[0] += 1
    return _GEN[0]


class _Plan:
    __slots__ = ("plan", "wbuf", "pbuf", "work", "wver")


class YOLOv10DetectHead(nn.Module):
    dynamic = False
    export = True
    shape = None
    max_det = 300

    def __init__(self, nc: int = 80, ch: tuple[int, int, int] = (256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                YOLOConv(x, c2, 3),
                YOLOConv(c2, c2, 3),
                Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(YOLOConv(x, x, 3, g=x), YOLOConv(x, c3, 1)),
                nn.Sequential(YOLOConv(c3, c3, 3, g=c3), YOLOConv(c3, c3, 1)),
                Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = YOLODFL(self.reg_max)
        self._sigmoid = Sigmoid()
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))
        self._fk_cache = {}

    # ---------------------------------------------------------------- baseline
    def forward_feat(self, x: list[torch.Tensor], cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def inference(self, x: list[torch.Tensor]):
        b, _, h, w = x[0].shape
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in x], 2)
        spatial = (h, w)
        if self.dynamic or self.shape != spatial:
            anchors, strides = (t.transpose(0, 1).contiguous() for t in make_anchors(x, self.stride, 0.5))
            if self.anchors.numel() == anchors.numel() and self.strides.numel() == strides.numel():
                self.anchors.copy_(anchors)
                self.strides.copy_(strides)
            else:
                self.register_buffer("anchors", anchors)
                self.register_buffer("strides", strides)
            self.shape = spatial
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, self._sigmoid(cls)), 1)

    # ------------------------------------------------------------- fused path
    def _conv_specs(self):
        """Per-level (dense, depthwise) sub-module handles, validated."""
        out = []
        for i in range(self.nl):
            a = self.one2one_cv2[i]
            b = self.one2one_cv3[i]
            if len(a) != 3 or len(b) != 3 or len(b[0]) != 2 or len(b[1]) != 2:
                raise _Unsupported("unexpected head structure")
            out.append((a[0], a[1], a[2], b[0][0], b[0][1], b[1][0], b[1][1], b[2]))
        return out

    @staticmethod
    def _bn(m):
        bn = getattr(m, "bn", None)
        if bn is None or getattr(m, "_is_fused", False) or m.conv.bias is not None:
            raise _Unsupported("fused / biased YOLOConv")
        gamma = bn.weight.detach().float()
        invstd = torch.rsqrt(bn.running_var.detach().float() + bn.eps)
        scale = gamma * invstd
        bias = bn.bias.detach().float() - bn.running_mean.detach().float() * scale
        return scale, bias

    def _build(self, x):
        dev = x[0].device
        B = int(x[0].shape[0])
        nc, reg = int(self.nc), int(self.reg_max)
        nbox = 4 * reg
        specs = self._conv_specs()
        c2 = int(specs[0][0].conv.weight.shape[0])
        c3 = int(specs[0][4].conv.weight.shape[0])
        for v in (c2, c3, nc, nbox):
            if v not in _OK_COUT or v % 16:
                raise _Unsupported(f"unsupported channel width {v}")
        strides = [int(round(float(s))) for s in self.stride.tolist()]
        if any(abs(float(s) - r) > 1e-6 for s, r in zip(self.stride.tolist(), strides)):
            raise _Unsupported("non-integral stride")

        wparts, pparts = [], []
        woff = pparts_off = 0

        def addw(t):
            nonlocal woff
            t = t.detach().reshape(-1).to(torch.float16)
            pad = (-woff) % 8
            if pad:
                wparts.append(torch.zeros(pad, dtype=torch.float16, device=dev))
                woff += pad
            off = woff
            wparts.append(t)
            woff += t.numel()
            return off

        def addp(sc, bs):
            nonlocal pparts_off
            off = pparts_off
            pparts.append(sc.reshape(-1))
            pparts.append(bs.reshape(-1))
            pparts_off += sc.numel() + bs.numel()
            return off

        def dense_w(conv, cin, cout, ks, groups=1):
            w = conv.weight.detach()
            if tuple(w.shape) != (cout, cin // groups, ks, ks) or conv.groups != groups:
                raise _Unsupported("unexpected conv weight shape")
            if tuple(conv.stride) != (1, 1) or tuple(conv.padding) != (ks // 2, ks // 2):
                raise _Unsupported("unexpected conv stride/padding")
            if groups == 1:
                return addw(w.permute(2, 3, 1, 0).contiguous())
            return addw(w.reshape(cout, ks * ks).t().contiguous())

        plan = [0] * (3 * _NF + 12)
        abase = 0
        woff_work = 0
        for i, sp in enumerate(specs):
            a0, a1, a2, b0, b1, b2, b3, b4 = sp
            H, W = int(x[i].shape[2]), int(x[i].shape[3])
            ch = int(x[i].shape[1])
            if ch % 32 or int(a0.conv.weight.shape[1]) != ch:
                raise _Unsupported("channel count not supported")
            npix = B * H * W
            L = i * _NF
            plan[L + F_H], plan[L + F_W], plan[L + F_NPIX], plan[L + F_CH] = H, W, npix, ch
            plan[L + F_ABASE] = abase
            plan[L + F_STRIDE] = strides[i]
            abase += H * W

            plan[L + F_A0W] = dense_w(a0.conv, ch, c2, 3)
            plan[L + F_A1W] = dense_w(a1.conv, c2, c2, 3)
            plan[L + F_B0W] = dense_w(b0.conv, ch, ch, 3, groups=ch)
            plan[L + F_B1W] = dense_w(b1.conv, ch, c3, 1)
            plan[L + F_B2W] = dense_w(b2.conv, c3, c3, 3, groups=c3)
            plan[L + F_B3W] = dense_w(b3.conv, c3, c3, 1)
            plan[L + F_A2W] = dense_w(a2, c2, nbox, 1)
            plan[L + F_B4W] = dense_w(b4, c3, nc, 1)
            if a2.bias is None or b4.bias is None:
                raise _Unsupported("missing output bias")
            plan[L + F_A2B] = addw(a2.bias.detach())
            plan[L + F_B4B] = addw(b4.bias.detach())
            for fld, m in ((F_A0P, a0), (F_A1P, a1), (F_B0P, b0),
                           (F_B1P, b1), (F_B2P, b2), (F_B3P, b3)):
                plan[L + fld] = addp(*self._bn(m))

            # B0O/B2O are unused: the depthwise stages are fused into the 1x1
            # convolutions that consume them and never reach global memory.
            for fld, cnt in ((F_IN, ch), (F_A0O, c2), (F_A1O, c2), (F_A2O, nbox),
                             (F_B1O, c3), (F_B3O, c3), (F_B4O, nc)):
                plan[L + fld] = woff_work
                woff_work += npix * cnt

        A = abase
        if A > 65535 or self.max_det * nc > 65535 or self.max_det > A:
            raise _Unsupported("anchor / max_det range unsupported")
        score_off = woff_work
        woff_work += B * A + 8
        ptab_off = (woff_work + 7) & ~7
        woff_work = ptab_off + 32
        G = 3 * _NF
        plan[G + 0] = B
        plan[G + 1] = nc
        plan[G + 2] = c2
        plan[G + 3] = c3
        plan[G + 4] = nbox
        plan[G + 5] = A
        plan[G + 6] = score_off
        plan[G + 7] = int(self.max_det)
        plan[G + 8] = reg
        plan[G + 10] = _next_gen()
        plan[G + 11] = ptab_off

        st = _Plan()
        st.plan = torch.tensor(plan, dtype=torch.int64)
        st.wbuf = torch.cat(wparts)
        st.pbuf = torch.cat(pparts)
        st.work = torch.empty(woff_work, dtype=torch.float16, device=dev)
        st.wver = self._wver()
        return st

    def _wver(self):
        w = self.one2one_cv2[0][0].conv.weight
        b = self.one2one_cv3[0][2].weight
        return (w.data_ptr(), w._version, b.data_ptr(), b._version)

    def _fused(self, x):
        ext = _ext()
        if ext is None or len(x) != 3 or self.nl != 3:
            return None
        x0, x1, x2 = x
        if x0.dtype is not torch.float16 or not x0.is_cuda:
            return None
        if not (x0.is_contiguous() and x1.is_contiguous() and x2.is_contiguous()):
            return None
        if x0.dim() != 4 or x1.dim() != 4 or x2.dim() != 4:
            return None
        key = (x0.shape, x1.shape, x2.shape, x0.device.index)
        st = self._fk_cache.get(key)
        if st is None or st.wver != self._wver():
            try:
                st = self._build(x)
            except _Unsupported:
                self._fk_cache[key] = False
                return None
            except Exception:
                self._fk_cache[key] = False
                return None
            self._fk_cache[key] = st
        elif st is False:
            return None
        out = torch.empty((x0.shape[0], self.max_det, 6), dtype=torch.float16, device=x0.device)
        ext.yolo_head_forward(x0, x1, x2, st.wbuf, st.pbuf, st.work, out, st.plan)
        return out

    # ----------------------------------------------------------------- forward
    def forward(self, x: list[torch.Tensor]):
        if not self.training and self.export:
            fused = self._fused(x)
            if fused is not None:
                return fused
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)
            if self.export:
                boxes, scores, labels = v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([xywh2xyxy(boxes), scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)

        one2many = self.forward_feat(x, self.cv2, self.cv3)
        if self.training:
            return {"one2many": one2many, "one2one": one2one}
        one2many = self.inference(one2many)
        return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
