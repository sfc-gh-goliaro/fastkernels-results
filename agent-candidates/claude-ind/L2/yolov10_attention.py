"""YOLOv10 spatial attention block - fused single-kernel CUDA implementation.

The whole block (qkv 1x1 conv+BN, 2-head spatial attention, depthwise-3x3
positional encoding on V, and the output 1x1 conv+BN) runs in one kernel:

  phase 1   qkv projection with the BatchNorms folded into the conv weights,
            scattered into Q / K / V scratch buffers (wmma, smem-staged operands)
  barrier   grid-wide (all blocks resident, one block per SM)
  phase 2   per (batch, 16-query tile): flash-style attention for both heads
            (n=400, dk=32, dv=64), depthwise 3x3 pe(V) fused into the epilogue,
            then the 1x1 projection - all in shared memory / registers.

Falls back to the reference path for shapes / configurations the kernel does not
cover.
"""

from __future__ import annotations

import os
import subprocess

import torch
import torch.nn as nn

from ..L1.softmax import Softmax
from .yolov10_conv import YOLOConv

_CUDA_SRC = r"""
// Fused YOLOv10 spatial-attention block: dim=128, heads=2, attn_ratio=0.5, 20x20 grid.
//   phase 1 : qkv = fold(conv1x1+BN)(x)  ->  Q/K/V scratch
//   grid barrier
//   phase 2 : 2-head attention (dk=32, dv=64, n=400) + depthwise 3x3 pe(V) + 1x1 proj
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define NPOS   400
#define GW     20
#define GH     20
#define CDIM   128
#define NH     2
#define DK     32
#define DV     64
#define QKVC   256
#define NWARP  16
#define NTHREAD (NWARP * 32)
#define QT     16
#define NTILE  (NPOS / QT)
#define KSPLIT 4

#define LD_W   136              // staged weight rows (halves)
#define LD_V   408
#define LD_K   40
#define LD_QS  40
#define LD_S   404              // floats
#define LD_P   408
#define LD_O   68               // floats
#define LD_XT  88
#define LD_A1  88               // floats
#define LD_X   136

#define P1_PT  80
#define P1_NPT (NPOS / P1_PT)
#define P1_OC  64
#define P1_NI  (P1_NPT * (QKVC / P1_OC))

// ---- shared-memory map (bytes) ----------------------------------------------
#define OFF_WPJ  0
#define SZ_WPJ   (CDIM * LD_W * 2)              // 34816  persistent
#define OFF_WPE  (OFF_WPJ + SZ_WPJ)
#define SZ_WPE   (9 * CDIM * 2)                 // 2304   persistent
#define OFF_K    (OFF_WPE + SZ_WPE)
#define SZ_K     (NPOS * LD_K * 2)              // 25600
#define OFF_P    (OFF_K + SZ_K)
#define SZ_P     (QT * LD_P * 2)                // S and P share this buffer (fp16)
#define OFF_Q    (OFF_P + SZ_P)
#define SZ_Q     (NH * QT * LD_QS * 2)
#define OFF_INV  (OFF_Q + SZ_Q)
#define SZ_INV   (QT * 4)
#define OFF_O    (OFF_INV + SZ_INV)
#define SZ_O     (KSPLIT * QT * LD_O * 4)
#define OFF_X    (OFF_O + SZ_O)
#define SZ_X     (QT * LD_X * 2)
#define OFF_V    (OFF_X + SZ_X)
#define SZ_V     (CDIM * LD_V * 2)              // 102400
#define SMEM_TOT (OFF_V + SZ_V + 64)

#define OFF_OUT  OFF_P                          // proj staging, aliases S
#define OFF_WQ   OFF_V                          // phase 1, aliases V
#define OFF_XT   (OFF_WQ + P1_OC * LD_W * 2)
#define OFF_A1   (OFF_XT + CDIM * LD_XT * 2)

// ---- global scratch (halves, per batch) -------------------------------------
#define SC_Q 0
#define SC_K (NH * NPOS * LD_K)
#define SC_V (2 * NH * NPOS * LD_K)
#define SC_B (2 * NH * NPOS * LD_K + CDIM * LD_V)

namespace {

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void grid_barrier(unsigned* ctr, unsigned target, unsigned gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    unsigned old = atomicAdd(ctr, 1u);
    unsigned* flag = ctr + 32;
    if (old == target - 1) {
      atomicExch(flag, gen);
    } else {
      volatile unsigned* f = flag;
      while (*f < gen) { }
    }
  }
  __syncthreads();
}

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragBr = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using FragBc = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
using FragCh = wmma::fragment<wmma::accumulator, 16, 16, 16, __half>;

// --------------------------------------------------------------- phase 1 -----
__device__ void phase1(const __half* __restrict__ x, const __half* __restrict__ wqkv,
                       const float* __restrict__ bqkv, __half* __restrict__ scratch,
                       int B, char* sb) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  __half* sWq = (__half*)(sb + OFF_WQ);
  __half* sXt = (__half*)(sb + OFF_XT);
  float* sA1 = (float*)(sb + OFF_A1);

  const int wrow = tid >> 4;           // weight row base (16 threads per row)
  const int wu = (tid & 15) * 8;       // halves within a row
  const int xrow = tid >> 2;           // 0..127 (x channels)
  const int xu0 = tid & 3;

  for (int item = blockIdx.x; item < B * P1_NI; item += gridDim.x) {
    const int b = item / P1_NI;
    const int rem = item - b * P1_NI;
    const int oc0 = (rem / P1_NPT) * P1_OC;
    const int p0 = (rem - (rem / P1_NPT) * P1_NPT) * P1_PT;
    const __half* xb = x + (size_t)b * CDIM * NPOS;

    __syncthreads();
    for (int r = wrow; r < P1_OC; r += NTHREAD / 16)
      cp16(sWq + r * LD_W + wu, wqkv + (size_t)(oc0 + r) * CDIM + wu);
    for (int c = xrow; c < CDIM; c += NTHREAD / 4)
      for (int uu = xu0; uu < P1_PT / 8; uu += 4)
        cp16(sXt + c * LD_XT + uu * 8, xb + (size_t)c * NPOS + p0 + uu * 8);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // 4 m-tiles x P1_NPT n-tiles distributed over 16 warps
#pragma unroll 1
    for (int t = warp; t < 4 * P1_NPT; t += NWARP) {
      const int mt = t & 3;
      const int nt = t >> 2;
      FragA fa[CDIM / 16];
#pragma unroll
      for (int k = 0; k < CDIM / 16; ++k)
        wmma::load_matrix_sync(fa[k], sWq + mt * 16 * LD_W + k * 16, LD_W);
      {
        FragC acc0, acc1;
        wmma::fill_fragment(acc0, 0.f);
        wmma::fill_fragment(acc1, 0.f);
        FragBr fb[CDIM / 16];
#pragma unroll
        for (int k = 0; k < CDIM / 16; ++k)
          wmma::load_matrix_sync(fb[k], sXt + k * 16 * LD_XT + nt * 16, LD_XT);
#pragma unroll
        for (int k = 0; k < CDIM / 16; k += 2) {
          wmma::mma_sync(acc0, fa[k], fb[k], acc0);
          wmma::mma_sync(acc1, fa[k + 1], fb[k + 1], acc1);
        }
#pragma unroll
        for (int i = 0; i < acc0.num_elements; ++i) acc0.x[i] += acc1.x[i];
        wmma::store_matrix_sync(sA1 + mt * 16 * LD_A1 + nt * 16, acc0, LD_A1,
                                wmma::mem_row_major);
      }
    }
    __syncthreads();

    __half* scb = scratch + (size_t)b * SC_B;
    if ((oc0 & 127) == 0) {
      // rows 0..31 -> Q[h][p][dk], rows 32..63 -> K[h][p][dk]; 8 dk per store
      const int h = oc0 >> 7;
      const int dk8 = (tid & 3) * 8;          // 4 groups of 8 dk
      const int sel = (tid >> 2) & 1;
      const int pl0 = tid >> 3;
      __half* dst = scb + (sel ? SC_K : SC_Q) + ((size_t)h * NPOS + p0) * LD_K + dk8;
      const float* srcb = sA1 + (sel * DK + dk8) * LD_A1;
      const float* bb = bqkv + oc0 + sel * DK + dk8;
      for (int pl = pl0; pl < P1_PT; pl += NTHREAD / 8) {
        __align__(16) __half t[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) t[u] = __float2half(srcb[u * LD_A1 + pl] + bb[u]);
        *(uint4*)(dst + (size_t)pl * LD_K) = *(const uint4*)t;
      }
    } else {
      // rows are V of head h: 8 consecutive positions per store
      const int c0 = (oc0 >> 7) * DV;
      const int u0 = tid & 1;                 // 2 groups of 8 positions (P1_PT = 80)
      const int dv0 = tid >> 1;
      for (int dv = dv0; dv < DV; dv += NTHREAD / 2) {
        const float* srcb = sA1 + dv * LD_A1;
        const float bias = bqkv[oc0 + dv];
        __half* dst = scb + SC_V + (size_t)(c0 + dv) * LD_V + p0;
        for (int pl = u0 * 8; pl < P1_PT; pl += 16) {
          __align__(16) __half t[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) t[u] = __float2half(srcb[pl + u] + bias);
          *(uint4*)(dst + pl) = *(const uint4*)t;
        }
      }
    }
  }
}

// --------------------------------------------------------------- phase 2 -----
__device__ void phase2(const float* __restrict__ bpe, const float* __restrict__ bproj,
                       const __half* __restrict__ scratch, __half* __restrict__ out,
                       int B, char* sb) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  __half* sWpj = (__half*)(sb + OFF_WPJ);
  __half* sWpe = (__half*)(sb + OFF_WPE);
  __half* sK = (__half*)(sb + OFF_K);
  __half* sP = (__half*)(sb + OFF_P);
  __half* sQ = (__half*)(sb + OFF_Q);
  float* sInv = (float*)(sb + OFF_INV);
  float* sO = (float*)(sb + OFF_O);
  __half* sX = (__half*)(sb + OFF_X);
  __half* sV = (__half*)(sb + OFF_V);
  float* sOut = (float*)(sb + OFF_OUT);

  const int pe_t = (tid & 7) * 2;      // 0,2,..,14
  const int pe_dv = tid >> 3;          // 0..63
  const int pv_nt = warp & 3;
  const int pv_kg = warp >> 2;
  const int pv_kb = pv_kg * ((NPOS / 16 + KSPLIT - 1) / KSPLIT);
  const int pv_ke = min(pv_kb + (NPOS / 16 + KSPLIT - 1) / KSPLIT, NPOS / 16);

  for (int item = blockIdx.x; item < B * NTILE; item += gridDim.x) {
    const int b = item / NTILE;
    const int p0 = (item - b * NTILE) * QT;
    const __half* scb = scratch + (size_t)b * SC_B;

    __syncthreads();
    // group A: Q (both heads) + K (head 0)
    if (tid < QT * LD_QS / 8) {
#pragma unroll
      for (int hh = 0; hh < NH; ++hh)
        cp16(sQ + hh * QT * LD_QS + tid * 8,
             scb + SC_Q + ((size_t)hh * NPOS + p0) * LD_K + tid * 8);
    }
    for (int i = tid; i < NPOS * LD_K / 8; i += NTHREAD)
      cp16(sK + i * 8, scb + SC_K + i * 8);
    cp_commit();
    // group B: V (all channels)
    for (int i = tid; i < CDIM * LD_V / 8; i += NTHREAD)
      cp16(sV + i * 8, scb + SC_V + i * 8);
    cp_commit();
    cp_wait<1>();
    __syncthreads();

    for (int h = 0; h < NH; ++h) {
      if (h) { cp_wait<0>(); __syncthreads(); }
      // ---- S = Q K^T   (scale*log2e folded into Wq)
      {
        FragA fa[DK / 16];
#pragma unroll
        for (int k = 0; k < DK / 16; ++k)
          wmma::load_matrix_sync(fa[k], sQ + h * QT * LD_QS + k * 16, LD_QS);
#pragma unroll 1
        for (int nt = warp; nt < NPOS / 16; nt += NWARP) {
          FragCh acc;
          wmma::fill_fragment(acc, __float2half(0.f));
          FragBc fb[DK / 16];
#pragma unroll
          for (int k = 0; k < DK / 16; ++k)
            wmma::load_matrix_sync(fb[k], sK + nt * 16 * LD_K + k * 16, LD_K);
#pragma unroll
          for (int k = 0; k < DK / 16; ++k) wmma::mma_sync(acc, fa[k], fb[k], acc);
          wmma::store_matrix_sync(sP + nt * 16, acc, LD_P, wmma::mem_row_major);
        }
      }
      __syncthreads();
      if (h == 0) {   // prefetch head-1 K over head-0 softmax/PV
        for (int i = tid; i < NPOS * LD_K / 8; i += NTHREAD)
          cp16(sK + i * 8, scb + SC_K + NPOS * LD_K + i * 8);
        cp_commit();
      }

      // ---- row softmax over fp16 S, written back in place as P
      if (warp < QT) {
        __half* sr = sP + warp * LD_P;
        const bool has1 = lane < (NPOS - 256) / 8;
        uint4 a0 = *(const uint4*)(sr + lane * 8);
        uint4 a1;
        if (has1) a1 = *(const uint4*)(sr + 256 + lane * 8);
        else a1 = make_uint4(0xfbfffbffu, 0xfbfffbffu, 0xfbfffbffu, 0xfbfffbffu);
        __half2* p0 = (__half2*)&a0;
        __half2* p1 = (__half2*)&a1;
        __half2 mx = __hmax2(p0[0], p1[0]);
#pragma unroll
        for (int i = 1; i < 4; ++i) mx = __hmax2(mx, __hmax2(p0[i], p1[i]));
        float m = fmaxf(__low2float(mx), __high2float(mx));
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        const __half2 mh = __float2half2_rn(m);
        __half2 hs = __float2half2_rn(0.f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          p0[i] = h2exp2(__hsub2(p0[i], mh));
          hs = __hadd2(hs, p0[i]);
        }
        *(uint4*)(sr + lane * 8) = a0;
        if (has1) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            p1[i] = h2exp2(__hsub2(p1[i], mh));
            hs = __hadd2(hs, p1[i]);
          }
          *(uint4*)(sr + 256 + lane * 8) = a1;
        }
        float sum = __low2float(hs) + __high2float(hs);
#pragma unroll
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        if (lane == 0) sInv[warp] = 1.f / sum;
      }
      if (h == 0) cp_wait<1>();  // V (group B) resident before PV
      __syncthreads();

      // ---- O = P V
      if (warp < 4 * KSPLIT) {
        FragC acc0, acc1;
        wmma::fill_fragment(acc0, 0.f);
        wmma::fill_fragment(acc1, 0.f);
        const __half* vb = sV + (size_t)(h * DV + pv_nt * 16) * LD_V;
        int kt = pv_kb;
        for (; kt + 1 < pv_ke; kt += 2) {
          FragA fa0, fa1;
          FragBc fb0, fb1;
          wmma::load_matrix_sync(fa0, sP + kt * 16, LD_P);
          wmma::load_matrix_sync(fb0, vb + kt * 16, LD_V);
          wmma::load_matrix_sync(fa1, sP + (kt + 1) * 16, LD_P);
          wmma::load_matrix_sync(fb1, vb + (kt + 1) * 16, LD_V);
          wmma::mma_sync(acc0, fa0, fb0, acc0);
          wmma::mma_sync(acc1, fa1, fb1, acc1);
        }
        if (kt < pv_ke) {
          FragA fa0;
          FragBc fb0;
          wmma::load_matrix_sync(fa0, sP + kt * 16, LD_P);
          wmma::load_matrix_sync(fb0, vb + kt * 16, LD_V);
          wmma::mma_sync(acc0, fa0, fb0, acc0);
        }
#pragma unroll
        for (int i = 0; i < acc0.num_elements; ++i) acc0.x[i] += acc1.x[i];
        wmma::store_matrix_sync(sO + pv_kg * (QT * LD_O) + pv_nt * 16, acc0, LD_O,
                                wmma::mem_row_major);
      }
      __syncthreads();

      // ---- X = O/sum + pe(V) + bpe   (branch-free 3x3 depthwise)
      if (tid < 512) {
        const int c = h * DV + pe_dv;
        const int p = p0 + pe_t;
        const int y = p / GW;
        const int xx = p - y * GW;
        const __half* vrowc = sV + (size_t)c * LD_V;
        const __half* wr = sWpe + c;
        float o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int q = 0; q < KSPLIT; ++q) {
          const float* op = sO + q * (QT * LD_O) + pe_t * LD_O + pe_dv;
          o0 += op[0];
          o1 += op[LD_O];
        }
        float r0 = fmaf(o0, sInv[pe_t], bpe[c]);
        float r1 = fmaf(o1, sInv[pe_t + 1], bpe[c]);
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
          const int yy = y + ky - 1;
          const bool vy = (unsigned)yy < (unsigned)GH;
          const int base = (vy ? yy : y) * GW + xx;
          __half2 g0 = *(const __half2*)(vrowc + max(base - 2, 0));
          __half2 g1 = *(const __half2*)(vrowc + base);
          __half2 g2 = *(const __half2*)(vrowc + min(base + 2, NPOS - 2));
          const float h1 = __high2float(g0);
          const float h2 = __low2float(g1);
          const float h3 = __high2float(g1);
          const float h4 = __low2float(g2);
          const float w0 = vy ? __half2float(wr[(ky * 3 + 0) * CDIM]) : 0.f;
          const float w1 = vy ? __half2float(wr[(ky * 3 + 1) * CDIM]) : 0.f;
          const float w2 = vy ? __half2float(wr[(ky * 3 + 2) * CDIM]) : 0.f;
          r0 = fmaf(xx > 0 ? w0 : 0.f, h1, r0);
          r0 = fmaf(w1, h2, r0);
          r0 = fmaf(w2, h3, r0);
          r1 = fmaf(w0, h2, r1);
          r1 = fmaf(w1, h3, r1);
          r1 = fmaf(xx + 2 < GW ? w2 : 0.f, h4, r1);
        }
        sX[pe_t * LD_X + c] = __float2half(r0);
        sX[(pe_t + 1) * LD_X + c] = __float2half(r1);
      }
    }
    __syncthreads();

    // ---- proj
    if (warp < CDIM / 16) {
      FragC acc0, acc1;
      wmma::fill_fragment(acc0, 0.f);
      wmma::fill_fragment(acc1, 0.f);
      FragA fa[CDIM / 16];
      FragBc fb[CDIM / 16];
#pragma unroll
      for (int k = 0; k < CDIM / 16; ++k) {
        wmma::load_matrix_sync(fa[k], sX + k * 16, LD_X);
        wmma::load_matrix_sync(fb[k], sWpj + (size_t)warp * 16 * LD_W + k * 16, LD_W);
      }
#pragma unroll
      for (int k = 0; k < CDIM / 16; k += 2) {
        wmma::mma_sync(acc0, fa[k], fb[k], acc0);
        wmma::mma_sync(acc1, fa[k + 1], fb[k + 1], acc1);
      }
#pragma unroll
      for (int i = 0; i < acc0.num_elements; ++i) acc0.x[i] += acc1.x[i];
      wmma::store_matrix_sync(sOut + warp * 16, acc0, CDIM, wmma::mem_row_major);
    }
    __syncthreads();
    {
      __half* ob = out + (size_t)b * CDIM * NPOS + p0;
      const int oc = tid >> 1;
      const int t8 = (tid & 1) * 8;
      if (oc < CDIM) {
        const float bias = bproj[oc];
        __align__(16) __half tmp[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) tmp[u] = __float2half(sOut[(t8 + u) * CDIM + oc] + bias);
        *(uint4*)(ob + (size_t)oc * NPOS + t8) = *(const uint4*)tmp;
      }
    }
  }
}

__global__ __launch_bounds__(NTHREAD) void yolo_kernel(
    const __half* __restrict__ x, const __half* __restrict__ wqkv,
    const __half* __restrict__ wpe, const __half* __restrict__ wproj,
    const float* __restrict__ bqkv, const float* __restrict__ bpe,
    const float* __restrict__ bproj, __half* __restrict__ out,
    __half* __restrict__ scratch, unsigned* ctr, unsigned target, unsigned gen, int B) {
  extern __shared__ char sb[];
  const int tid = threadIdx.x;
  // persistent staged weights: wproj [128][LD_W], wpe [9][128]
  {
    __half* sWpj = (__half*)(sb + OFF_WPJ);
    const int u = (tid & 15) * 8;                // 16 threads per 128-half row
    for (int r = tid >> 4; r < CDIM; r += NTHREAD / 16)
      cp16(sWpj + r * LD_W + u, wproj + (size_t)r * CDIM + u);
    __half* sWpe = (__half*)(sb + OFF_WPE);
    if (tid < 9 * CDIM / 8) cp16(sWpe + tid * 8, wpe + tid * 8);
    cp_commit();
  }
  phase1(x, wqkv, bqkv, scratch, B, sb);
  grid_barrier(ctr, target, gen);
  phase2(bpe, bproj, scratch, out, B, sb);
}

at::Tensor ws_;
unsigned base_ = 0;
unsigned gen_ = 0;
bool init_ = false;

}  // namespace

at::Tensor yolo_attn(const at::Tensor& x, const at::Tensor& wpack, const at::Tensor& bpack) {
  const int B = x.size(0);
  auto out = at::empty_like(x);
  const int nblocks = B * NTILE;
  const size_t need = (size_t)B * SC_B * 2 + 256;
  if (!ws_.defined() || (size_t)ws_.numel() < need || ws_.device() != x.device()) {
    ws_ = at::zeros({(int64_t)need}, x.options().dtype(at::kByte));
    base_ = 0;
    gen_ = 0;
  }
  if (base_ > 0xF0000000u || gen_ > 0xF0000000u) { ws_.zero_(); base_ = 0; gen_ = 0; }
  if (!init_) {
    cudaFuncSetAttribute(yolo_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOT);
    init_ = true;
  }
  unsigned* ctr = (unsigned*)ws_.data_ptr();
  __half* scratch = (__half*)((char*)ws_.data_ptr() + 256);
  const __half* w = (const __half*)wpack.data_ptr();
  const float* bp = (const float*)bpack.data_ptr();
  base_ += (unsigned)nblocks;
  ++gen_;
  yolo_kernel<<<nblocks, NTHREAD, SMEM_TOT, c10::cuda::getCurrentCUDAStream()>>>(
      (const __half*)x.data_ptr(), w, w + QKVC * CDIM, w + QKVC * CDIM + 9 * CDIM, bp,
      bp + QKVC, bp + QKVC + CDIM, (__half*)out.data_ptr(), scratch, ctr, base_, gen_, B);
  return out;
}

int64_t smem_total() { return SMEM_TOT; }

"""

_CPP_SRC = r"""
#include <torch/extension.h>
at::Tensor yolo_attn(const at::Tensor& x, const at::Tensor& wpack, const at::Tensor& bpack);
int64_t smem_total();
"""

_EXT = None


def _arch_flag() -> str:
    cap = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
        caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
        if caps:
            cap = caps[0]
    except Exception:
        pass
    if cap is None:
        try:
            major, minor = torch.cuda.get_device_capability()
            cap = f"{major}.{minor}"
        except Exception:
            cap = "10.0"
    return f"{cap}a" if cap.split(".")[0] in ("9", "10", "12") else cap


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        os.environ["TORCH_CUDA_ARCH_LIST"] = _arch_flag()
        _EXT = load_inline(
            name="fk_yolov10_attention_fused",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["yolo_attn", "smem_total"],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            verbose=False,
        )
    return _EXT


def _fold(conv: nn.Module, bn: nn.Module | None):
    """Fold Conv(+bias) + eval-mode BatchNorm into (weight, bias) in fp32."""
    w = conv.weight.detach().float()
    b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(
        w.shape[0], dtype=torch.float32, device=w.device)
    if bn is None:
        return w, b
    g = bn.weight.detach().float()
    inv = g / torch.sqrt(bn.running_var.detach().float() + bn.eps)
    fw = w * inv.view(-1, *([1] * (w.dim() - 1)))
    fb = bn.bias.detach().float() + (b - bn.running_mean.detach().float()) * inv
    return fw, fb


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)
        self._softmax = Softmax(dim=-1)
        self._dim = dim
        self._fused_ok = (dim == 128 and num_heads == 2 and self.key_dim == 32
                          and self.head_dim == 64)
        self._packed = None

    # -- fused path ----------------------------------------------------------
    def _pack(self, device, dtype):
        import math
        wq, bq = _fold(self.qkv.conv, getattr(self.qkv, "bn", None))
        wpe, bpe = _fold(self.pe.conv, getattr(self.pe, "bn", None))
        wpj, bpj = _fold(self.proj.conv, getattr(self.proj, "bn", None))
        wq = wq.reshape(256, 128).clone()
        bq = bq.clone()
        s = self.scale * math.log2(math.e)     # fold scale + exp2 base into Q
        for hh in range(self.num_heads):
            r0 = hh * (2 * self.key_dim + self.head_dim)
            wq[r0:r0 + self.key_dim] *= s
            bq[r0:r0 + self.key_dim] *= s
        wpe = wpe.reshape(128, 9).t().contiguous()          # [9][128]
        wpack = torch.cat([wq.reshape(-1), wpe.reshape(-1),
                           wpj.reshape(128, 128).reshape(-1)]).to(device=device,
                                                                  dtype=torch.float16)
        bpack = torch.cat([bq, bpe, bpj]).to(device=device, dtype=torch.float32)
        self._packed = (wpack.contiguous(), bpack.contiguous())
        return self._packed

    # -- reference path ------------------------------------------------------
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = self._softmax(attn)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self._fused_ok and x.dtype == torch.float16 and x.is_cuda and x.dim() == 4
                and x.shape[1] == 128 and x.shape[2] == 20 and x.shape[3] == 20
                and x.shape[0] <= 5):
            p = self._packed
            if p is None:
                p = self._pack(x.device, x.dtype)
            return _ext().yolo_attn(x.contiguous(), p[0], p[1])
        return self._reference(x)
