"""YOLOv10 PSA (Partial Self-Attention) block -- fused CUDA implementation.

The reference block is ~26 separate aten launches (7 conv+bn pairs, 2 bmms,
softmax, silus, cat) over a tiny problem (<=1 GFLOP), so it is completely
launch-bound.  Here the whole block runs as two custom kernels:

  k1: cv1 (1x1, 256->256, +bn, silu) and qkv (1x1, 128->256, +bn)
  k2: 2-head attention, the 3x3 depthwise positional encoding, proj, the ffn,
      the concat and cv2

Every conv-bn pair is folded into a single weight/bias on the first call, the
1x1 convs become mma.m16n8k16 GEMMs over tiles of 16 tokens, and a tile stays
resident in shared memory for a whole kernel -- only q/k/v (which attention
needs across tiles) go back to global memory.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn

from .yolov10_attention import YOLOAttention
from .yolov10_conv import YOLOConv

_CUDA_SRC = r"""// YOLOv10 PSA block fused into two CUDA kernels.
//
//   k1: cv1 (1x1 256->256, bn, silu) -> a, bb ; qkv (1x1 128->256, bn) -> q,k,v
//   k2: 2-head attention + 3x3 depthwise pe + proj + ffn + concat + cv2
//
// The block is tiny (~1 GFLOP over 1600 tokens), so the reference -- 26 aten
// launches -- is almost pure launch overhead.  Here one CTA owns a tile of 16
// tokens and carries it through the whole chain in shared memory.  Each 1x1
// conv is an mma.m16n8k16 GEMM with M=tokens, N=out-channels, K=in-channels, so
// both operands keep K contiguous (activations stay token-major) and fragments
// come from ldmatrix.  Weights are consumed in chunks of 128 output channels,
// double buffered: the cp.async for the next chunk is issued before the mma of
// the current one, which is what hides the ~600ns L2 latency at this very low
// arithmetic intensity.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define P_N       400       // tokens per image
#define P_T       16        // tokens per CTA tile
#define P_NTILE   25        // tiles per image
#define P_NW      16        // warps per CTA
#define P_NTHR    (P_NW * 32)
#define P_SC      ((50 + P_NW - 1) / P_NW)
#define P_CN      128       // out-channels per weight chunk (= P_NW * 8)

// shared leading dimensions (halves); +8 padding keeps ldmatrix conflict free
#define P_LD256   264
#define P_LD128   136
#define P_LDP     408       // [16][400] probabilities
#define P_LDK     40        // [400][32] keys
#define P_LDV     408       // [128][400] values
#define P_KC      128       // K-halves per staged chunk
#define P_WBUF    (256 * (P_KC + 8))

// packed weight offsets (halves)
#define OFF_W1   0
#define OFF_WQ   65536
#define OFF_WPE  98304      // [128][16], 9 used
#define OFF_WP   100352
#define OFF_WF0  116736
#define OFF_WF1  149504
#define OFF_W2   182272
// packed bias offsets (floats)
#define OFF_B1   0
#define OFF_BQ   256
#define OFF_BPE  512
#define OFF_BP   640
#define OFF_BF0  768
#define OFF_BF1  1024
#define OFF_B2   1152

typedef unsigned int u32;

__device__ __forceinline__ void mma1688(float (&d)[4], const u32 (&a)[4], const u32 (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmA(u32 (&r)[4], u32 a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ u32 sma(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldmB(u32 (&r)[2], u32 a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ u32 pack2(float x, float y) {
  __half2 h = __floats2half2_rn(x, y);
  return *(u32*)&h;
}
__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ void cpa16(void* d, const void* s) {
  u32 dd = (u32)__cvta_generic_to_shared(d);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dd), "l"(s) : "memory");
}
#define CP_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
template <int N>
__device__ __forceinline__ void cpwait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// ldmatrix source-row offsets within a tile
#define LANE_A(ldm) ((lane & 15) * (ldm) + (lane >> 4) * 8)
#define LANE_B(ldm) ((lane & 7) * (ldm) + ((lane >> 3) & 1) * 8)

// Stage rows [r0, r0+NROW) x K-halves [koff, koff+KC) of Wg into `buf`, commit.
template <int NROW, int KC>
__device__ __forceinline__ void stage_w(const __half* __restrict__ Wg, int ldw, int r0,
                                        int koff, __half* buf, int tid) {
  constexpr int LDWS = KC + 8;
  constexpr int CPR = KC / 8;             // 16B copies per row
  constexpr int RPR = P_NTHR / CPR;       // rows per round
  constexpr int ROUNDS = NROW / RPR;
  const int row = tid / CPR, part = tid - row * CPR;
  const __half* gs = Wg + (size_t)(r0 + row) * ldw + koff + part * 8;
  __half* ds = buf + row * LDWS + part * 8;
#pragma unroll
  for (int r = 0; r < ROUNDS; r++)
    cpa16(ds + r * RPR * LDWS, gs + (size_t)r * RPR * ldw);
  CP_COMMIT();
}

// acc[NT] += As[16][KC] * buf[NT*8 rows][KC]^T for one staged K-chunk.
// NT==2 fetches both B fragments with a single ldmatrix.x4.
template <int NT, int KC>
__device__ __forceinline__ void accum_chunk(const __half* __restrict__ As, int lda, int koff,
                                            const __half* __restrict__ buf, float (&acc)[NT][4],
                                            int warp, int lane) {
  constexpr int LDWS = KC + 8;
  constexpr int KS = KC / 16;
  const u32 ap = sma(As + LANE_A(lda) + koff);
  const u32 ws = sma(buf + (warp * NT * 8) * LDWS + (NT == 2 ? LANE_A(LDWS) : LANE_B(LDWS)));
  // two-deep software pipeline: ldmatrix is volatile asm, so the fragments for
  // step ks+2 have to be issued ahead of the mma of step ks by hand.
  u32 a[3][4], bf[3][NT * 2];
#pragma unroll
  for (int p = 0; p < 2; p++) {
    ldmA(a[p], ap + p * 32);
    if (NT == 2) ldmA(*(u32(*)[4])bf[p], ws + p * 32);
    else
#pragma unroll
      for (int j = 0; j < NT; j++) ldmB(*(u32(*)[2])(bf[p] + 2 * j), ws + j * 16 * LDWS + p * 32);
  }
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    const int cur = ks % 3, nxt = (ks + 2) % 3;
    if (ks + 2 < KS) {
      ldmA(a[nxt], ap + (ks + 2) * 32);
      if (NT == 2) ldmA(*(u32(*)[4])bf[nxt], ws + (ks + 2) * 32);
      else
#pragma unroll
        for (int j = 0; j < NT; j++)
          ldmB(*(u32(*)[2])(bf[nxt] + 2 * j), ws + j * 16 * LDWS + (ks + 2) * 32);
    }
    if (NT == 2) {
      u32 b0[2] = {bf[cur][0], bf[cur][2]}, b1[2] = {bf[cur][1], bf[cur][3]};
      mma1688(acc[0], a[cur], b0);
      mma1688(acc[1], a[cur], b1);
    } else {
#pragma unroll
      for (int j = 0; j < NT; j++) {
        u32 b[2] = {bf[cur][2 * j], bf[cur][2 * j + 1]};
        mma1688(acc[j], a[cur], b);
      }
    }
  }
}

// bias/activation/residual epilogue for acc[NT]; warp w owns channels w*NT*8..+NT*8
template <int NT, int ACT, int HASADD>
__device__ __forceinline__ void epilogue(float (&acc)[NT][4], const float* __restrict__ bias,
                                         const __half* Add, int ldadd, __half* Out, int ldo,
                                         int lane) {
  const int gid = lane >> 2, t = lane & 3;
#pragma unroll
  for (int j = 0; j < NT; j++) {
    const int n0 = j * 8 + 2 * t;
    const float2 bv = *(const float2*)(bias + n0);
    float v0 = acc[j][0] + bv.x, v1 = acc[j][1] + bv.y;
    float v2 = acc[j][2] + bv.x, v3 = acc[j][3] + bv.y;
    if (HASADD) {
      u32 r0 = *(const u32*)(Add + gid * ldadd + n0);
      u32 r1 = *(const u32*)(Add + (gid + 8) * ldadd + n0);
      __half2 h0 = *(__half2*)&r0, h1 = *(__half2*)&r1;
      v0 += __low2float(h0); v1 += __high2float(h0);
      v2 += __low2float(h1); v3 += __high2float(h1);
    }
    if (ACT) { v0 = silu(v0); v1 = silu(v1); v2 = silu(v2); v3 = silu(v3); }
    *(u32*)(Out + gid * ldo + n0) = pack2(v0, v1);
    *(u32*)(Out + (gid + 8) * ldo + n0) = pack2(v2, v3);
  }
}
#define ZERO_ACC(a, NT) _Pragma("unroll") for (int _j = 0; _j < NT; _j++) { \
  a[_j][0] = 0.f; a[_j][1] = 0.f; a[_j][2] = 0.f; a[_j][3] = 0.f; }

// ---------------------------------------------------------------------------
// k1
// ---------------------------------------------------------------------------
#define K1_SM ((3 * P_T * P_LD256 + 2 * P_WBUF) * 2)

extern "C" __global__ __launch_bounds__(P_NTHR, 1) void psa_k1(
    const __half* __restrict__ x, const __half* __restrict__ W,
    const float* __restrict__ Bs, __half* __restrict__ Ab, __half* __restrict__ BBb,
    __half* __restrict__ Qb, __half* __restrict__ Kb, __half* __restrict__ Vb) {
  extern __shared__ __half sm[];
  __half* xs = sm;
  __half* us = xs + P_T * P_LD256;
  __half* qs = us + P_T * P_LD256;
  __half* wb0 = qs + P_T * P_LD256;
  __half* wb1 = wb0 + P_WBUF;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int blk = blockIdx.x, b = blk / P_NTILE, tile = blk - b * P_NTILE;
  const int tok0 = tile * P_T;

  stage_w<256, P_KC>(W + OFF_W1, 256, 0, 0, wb0, tid);
  stage_w<256, P_KC>(W + OFF_W1, 256, 0, P_KC, wb1, tid);

  {  // x tile -> token-major (transpose of the NCHW slice)
    const int ci = (tid >> 2) * 2, j0 = (tid & 3) * 4;
    const __half* p0 = x + (size_t)b * 256 * P_N + (size_t)ci * P_N + tok0 + j0;
    uint2 v0 = *(const uint2*)(p0);
    uint2 v1 = *(const uint2*)(p0 + P_N);
    const __half* h0 = (const __half*)&v0;
    const __half* h1 = (const __half*)&v1;
    __half* d = xs + ci;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      __half2 v = __halves2half2(h0[j], h1[j]);
      *(u32*)(d + (j0 + j) * P_LD256) = *(u32*)&v;
    }
  }
  float acc[2][4];
  ZERO_ACC(acc, 2)
  cpwait<1>();
  __syncthreads();
  accum_chunk<2, P_KC>(xs, P_LD256, 0, wb0, acc, warp, lane);
  __syncthreads();
  stage_w<256, P_KC>(W + OFF_WQ, 128, 0, 0, wb0, tid);
  cpwait<1>();
  __syncthreads();
  accum_chunk<2, P_KC>(xs, P_LD256, P_KC, wb1, acc, warp, lane);
  epilogue<2, 1, 0>(acc, Bs + OFF_B1 + warp * 16, nullptr, 0, us + warp * 16, P_LD256, lane);
  __syncthreads();

  if (tid < 256) {  // publish a / bb (token-major)
    const int i = tid >> 4, c8 = (tid & 15) * 8;
    const size_t o = ((size_t)b * P_N + tok0 + i) * 128 + c8;
    *(uint4*)(Ab + o) = *(const uint4*)(us + i * P_LD256 + c8);
    *(uint4*)(BBb + o) = *(const uint4*)(us + i * P_LD256 + 128 + c8);
  }
  float accq[2][4];
  ZERO_ACC(accq, 2)
  cpwait<0>();
  __syncthreads();
  accum_chunk<2, P_KC>(us + P_CN, P_LD256, 0, wb0, accq, warp, lane);
  epilogue<2, 0, 0>(accq, Bs + OFF_BQ + warp * 16, nullptr, 0, qs + warp * 16, P_LD256, lane);
  __syncthreads();

  if (tid < 256) {  // v -> channel-major
    const int c = tid >> 1, half8 = (tid & 1) * 8;
    const __half* src = qs + half8 * P_LD256 + ((c >> 6) * 128 + 64 + (c & 63));
    __align__(16) __half tmp[8];
#pragma unroll
    for (int i = 0; i < 8; i++) tmp[i] = src[i * P_LD256];
    *(uint4*)(Vb + ((size_t)b * 128 + c) * P_N + tok0 + half8) = *(const uint4*)(tmp);
  } else if (tid < 384) {  // q, k -> token-major
    const int u = tid - 256;
    const int h = u >> 6, r = u & 63, i = r >> 2, part = r & 3;
    const __half* s = qs + i * P_LD256 + h * 128 + part * 8;
    const size_t o = ((size_t)(b * 2 + h) * P_N + tok0 + i) * 32 + part * 8;
    *(uint4*)(Qb + o) = *(const uint4*)(s);
    *(uint4*)(Kb + o) = *(const uint4*)(s + 32);
  }
}

// ---------------------------------------------------------------------------
// k2
// ---------------------------------------------------------------------------
#define K2_ACT (2 * P_T * P_LDK + P_T * P_LDP + 3 * P_T * P_LD128 + 2 * P_T * P_LD256)
#define K2_KV  (2 * P_N * P_LDK + 128 * P_LDV)
#define K2_UNION (K2_KV > 2 * P_WBUF ? K2_KV : 2 * P_WBUF)
#define K2_SM ((K2_ACT + K2_UNION) * 2 + (P_NW * 16 + 32) * 4)

extern "C" __global__ __launch_bounds__(P_NTHR, 1) void psa_k2(
    const __half* __restrict__ Ab, const __half* __restrict__ BBb,
    const __half* __restrict__ Qb, const __half* __restrict__ Kb,
    const __half* __restrict__ Vb, const __half* __restrict__ W,
    const float* __restrict__ Bs, __half* __restrict__ out) {
  extern __shared__ __half sm[];
  __half* qsm = sm;                            // [2][16][P_LDK]
  __half* psm = qsm + 2 * P_T * P_LDK;         // [16][P_LDP]
  __half* zsm = psm + P_T * P_LDP;             // [16][P_LD128]
  __half* bbs = zsm + P_T * P_LD128;           // [16][P_LD128]
  __half* bsm = bbs + P_T * P_LD128;           // [16][P_LD128]
  __half* fsm = bsm + P_T * P_LD128;           // [16][P_LD256]
  __half* csm = fsm + P_T * P_LD256;           // [16][P_LD256]
  __half* un = csm + P_T * P_LD256;            // union: K/V, then weight buffers
  __half* ksm = un;                            // [2][400][P_LDK]
  __half* vsm = un + 2 * P_N * P_LDK;          // [128][P_LDV]
  __half* wb0 = un;
  __half* wb1 = un + P_WBUF;
  float* red = (float*)(un + K2_UNION);        // [P_NW][16]
  float* rmx = red + P_NW * 16;                // [16]
  float* rsm = rmx + 16;                       // [16]
  float* pvr = (float*)fsm;                    // split-k scratch (8*32*4 floats)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int gid = lane >> 2, t = lane & 3;
  const int blk = blockIdx.x, b = blk / P_NTILE, tile = blk - b * P_NTILE;
  const int tok0 = tile * P_T;
  const float scale = 0.17677669529663687f;

  // stage keys and values, one cp.async group per head
  {
    const int kr = tid >> 2, kp = tid & 3;          // 400 rows x 4 copies
    const int ve = tid >> 3, vp = tid & 7;          // 64 rows x 50 copies
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int row = kr + r * 128;
        if (row < P_N)
          cpa16(ksm + (h * P_N + row) * P_LDK + kp * 8,
                Kb + ((size_t)(b * 2 + h) * P_N + row) * 32 + kp * 8);
      }
#pragma unroll
      for (int p = 0; p < 7; p++) {
        const int col = vp + p * 8;
        if (col < 50)
          cpa16(vsm + (h * 64 + ve) * P_LDV + col * 8,
                Vb + ((size_t)b * 128 + h * 64 + ve) * P_N + col * 8);
      }
      CP_COMMIT();
    }
  }

  if (tid < 128) {  // q tile (pre-scaled by 1/sqrt(key_dim)), both heads
    const int h = tid >> 6, r = tid & 63, i = r >> 2, part = r & 3;
    const uint4 q = *(const uint4*)(Qb + ((size_t)(b * 2 + h) * P_N + tok0 + i) * 32 + part * 8);
    const __half2 sc2 = __float2half2_rn(scale);
    uint4 o;
    ((__half2*)&o)[0] = __hmul2(((const __half2*)&q)[0], sc2);
    ((__half2*)&o)[1] = __hmul2(((const __half2*)&q)[1], sc2);
    ((__half2*)&o)[2] = __hmul2(((const __half2*)&q)[2], sc2);
    ((__half2*)&o)[3] = __hmul2(((const __half2*)&q)[3], sc2);
    *(uint4*)(qsm + (h * P_T + i) * P_LDK + part * 8) = o;
  } else if (tid < 384) {  // cat[:, :128] = a ; bb
    const int u = tid - 128;
    const int i = u >> 4, c8 = (u & 15) * 8;
    const size_t o = ((size_t)b * P_N + tok0 + i) * 128 + c8;
    *(uint4*)(csm + i * P_LD256 + c8) = *(const uint4*)(Ab + o);
    *(uint4*)(bbs + i * P_LD128 + c8) = *(const uint4*)(BBb + o);
  }

  for (int h = 0; h < 2; h++) {
    if (h == 0) cpwait<1>(); else cpwait<0>();
    __syncthreads();

    float sc[P_SC][4];
    const u32 aq = sma(qsm + h * P_T * P_LDK + LANE_A(P_LDK));
    const u32 kp = sma(ksm + (h * P_N + warp * 8) * P_LDK + LANE_B(P_LDK));
    u32 qa[2][4];
    ldmA(qa[0], aq);
    ldmA(qa[1], aq + 32);
#pragma unroll
    for (int c = 0; c < P_SC; c++) {
      sc[c][0] = 0.f; sc[c][1] = 0.f; sc[c][2] = 0.f; sc[c][3] = 0.f;
      if (warp + c * P_NW < 50) {
        u32 b0[2], b1[2];
        ldmB(b0, kp + (c * P_NW * 8) * P_LDK * 2);
        ldmB(b1, kp + (c * P_NW * 8) * P_LDK * 2 + 32);
        mma1688(sc[c], qa[0], b0);
        mma1688(sc[c], qa[1], b1);
      }
    }
    float m0 = -1e30f, m1 = -1e30f;
#pragma unroll
    for (int c = 0; c < P_SC; c++)
      if (warp + c * P_NW < 50) {
        m0 = fmaxf(m0, fmaxf(sc[c][0], sc[c][1]));
        m1 = fmaxf(m1, fmaxf(sc[c][2], sc[c][3]));
      }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    if (t == 0) { red[warp * 16 + gid] = m0; red[warp * 16 + gid + 8] = m1; }
    __syncthreads();
    if (tid < 16) {
      float v = red[tid];
#pragma unroll
      for (int w = 1; w < P_NW; w++) v = fmaxf(v, red[w * 16 + tid]);
      rmx[tid] = v;
    }
    __syncthreads();
    const float mx0 = rmx[gid], mx1 = rmx[gid + 8];
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int c = 0; c < P_SC; c++) {
      const int nt = warp + c * P_NW;
      if (nt < 50) {
        float p0 = __expf(sc[c][0] - mx0), p1 = __expf(sc[c][1] - mx0);
        float p2 = __expf(sc[c][2] - mx1), p3 = __expf(sc[c][3] - mx1);
        s0 += p0 + p1; s1 += p2 + p3;
        *(u32*)(psm + gid * P_LDP + nt * 8 + 2 * t) = pack2(p0, p1);
        *(u32*)(psm + (gid + 8) * P_LDP + nt * 8 + 2 * t) = pack2(p2, p3);
      }
    }
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1); s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
    if (t == 0) { red[warp * 16 + gid] = s0; red[warp * 16 + gid + 8] = s1; }
    __syncthreads();
    if (tid < 16) {
      float v = red[tid];
#pragma unroll
      for (int w = 1; w < P_NW; w++) v += red[w * 16 + tid];
      rsm[tid] = 1.f / v;
    }
    __syncthreads();

    {  // o[16][64] = P[16][400] * V^T, keys split across two warp groups
      const int nt = warp & 7, kh = warp >> 3;
      const int ke = kh ? 25 : 13;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      const u32 pp = sma(psm + LANE_A(P_LDP));
      const u32 vp = sma(vsm + (h * 64 + nt * 8) * P_LDV + LANE_B(P_LDV));
      for (int ks = kh ? 13 : 0; ks < ke; ks++) {
        u32 a[4], bb[2];
        ldmA(a, pp + ks * 32);
        ldmB(bb, vp + ks * 32);
        mma1688(acc, a, bb);
      }
      if (kh) {
#pragma unroll
        for (int i = 0; i < 4; i++) pvr[(nt * 32 + lane) * 4 + i] = acc[i];
      }
      __syncthreads();
      if (!kh) {
        const float r0 = rsm[gid], r1 = rsm[gid + 8];
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i] += pvr[(nt * 32 + lane) * 4 + i];
        __half* o = zsm + h * 64 + nt * 8 + 2 * t;
        *(u32*)(o + gid * P_LD128) = pack2(acc[0] * r0, acc[1] * r0);
        *(u32*)(o + (gid + 8) * P_LD128) = pack2(acc[2] * r1, acc[3] * r1);
      }
      __syncthreads();
    }
  }

  {  // pe: 3x3 depthwise conv on v, folded into z.  4 lanes per channel, each
     // taking 4 of the 16 tokens, so consecutive lanes hit distinct banks.
    const int c = tid >> 2, i0 = (tid & 3) * 4;
    const uint4 w0 = *(const uint4*)(W + OFF_WPE + c * 16);
    const u32 w8 = *(const u32*)(W + OFF_WPE + c * 16 + 8);
    const __half* wpe = (const __half*)&w0;
    const __half w8h = *(const __half*)&w8;
    const float bp = Bs[OFF_BPE + c];
    const __half* vrow = vsm + c * P_LDV;
#pragma unroll
    for (int ii = 0; ii < 4; ii++) {
      const int i = i0 + ii;
      const int p = tok0 + i, r = p / 20, cc = p - r * 20;
      float a = bp;
#pragma unroll
      for (int dr = -1; dr <= 1; dr++) {
        if (r + dr < 0 || r + dr > 19) continue;
#pragma unroll
        for (int dc = -1; dc <= 1; dc++) {
          if (cc + dc < 0 || cc + dc > 19) continue;
          const int j = (dr + 1) * 3 + (dc + 1);
          a += __half2float(j == 8 ? w8h : wpe[j]) * __half2float(vrow[p + dr * 20 + dc]);
        }
      }
      __half* z = zsm + i * P_LD128 + c;
      *z = __float2half(__half2float(*z) + a);
    }
  }
  __syncthreads();

  // proj -> ffn -> concat -> cv2.  Each weight chunk is staged while the
  // previous chunk's mma runs, so only the very first load is exposed.
  stage_w<128, P_KC>(W + OFF_WP, 128, 0, 0, wb0, tid);
  stage_w<256, P_KC>(W + OFF_WF0, 128, 0, 0, wb1, tid);
  {
    float ap[1][4];
    ZERO_ACC(ap, 1)
    cpwait<1>();
    __syncthreads();
    accum_chunk<1, P_KC>(zsm, P_LD128, 0, wb0, ap, warp, lane);
    epilogue<1, 0, 1>(ap, Bs + OFF_BP + warp * 8, bbs + warp * 8, P_LD128,
                      bsm + warp * 8, P_LD128, lane);
  }
  __syncthreads();
  stage_w<128, P_KC>(W + OFF_WF1, 256, 0, 0, wb0, tid);
  {
    float af[2][4];
    ZERO_ACC(af, 2)
    cpwait<1>();
    __syncthreads();
    accum_chunk<2, P_KC>(bsm, P_LD128, 0, wb1, af, warp, lane);
    epilogue<2, 1, 0>(af, Bs + OFF_BF0 + warp * 16, nullptr, 0, fsm + warp * 16,
                      P_LD256, lane);
  }
  __syncthreads();
  stage_w<128, P_KC>(W + OFF_WF1, 256, 0, P_KC, wb1, tid);
  float ag[1][4];
  ZERO_ACC(ag, 1)
  cpwait<1>();
  __syncthreads();
  accum_chunk<1, P_KC>(fsm, P_LD256, 0, wb0, ag, warp, lane);
  __syncthreads();
  stage_w<256, P_KC>(W + OFF_W2, 256, 0, 0, wb0, tid);
  cpwait<1>();
  __syncthreads();
  accum_chunk<1, P_KC>(fsm, P_LD256, P_KC, wb1, ag, warp, lane);
  epilogue<1, 0, 1>(ag, Bs + OFF_BF1 + warp * 8, bsm + warp * 8, P_LD128,
                    csm + P_CN + warp * 8, P_LD256, lane);
  __syncthreads();
  stage_w<256, P_KC>(W + OFF_W2, 256, 0, P_KC, wb1, tid);
  float ac[2][4];
  ZERO_ACC(ac, 2)
  cpwait<1>();
  __syncthreads();
  accum_chunk<2, P_KC>(csm, P_LD256, 0, wb0, ac, warp, lane);
  cpwait<0>();
  __syncthreads();
  accum_chunk<2, P_KC>(csm, P_LD256, P_KC, wb1, ac, warp, lane);
  epilogue<2, 1, 0>(ac, Bs + OFF_B2 + warp * 16, nullptr, 0, fsm + warp * 16, P_LD256, lane);
  __syncthreads();

  if (tid < 256) {  // token-major -> NCHW
    const __half* s = fsm + (tid >> 1) + (tid & 1) * 128;
    __align__(16) __half tmp[P_T];
#pragma unroll
    for (int i = 0; i < P_T; i++) tmp[i] = s[i * P_LD256];
    __half* d = out + ((size_t)b * 256 + (tid >> 1) + (tid & 1) * 128) * P_N + tok0;
    *(uint4*)(d) = *(const uint4*)(tmp);
    *(uint4*)(d + 8) = *(const uint4*)(tmp + 8);
  }
}

// ---------------------------------------------------------------------------
// host entry
// ---------------------------------------------------------------------------
static void psa_init() {
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute((const void*)psa_k1, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)K1_SM);
    cudaFuncSetAttribute((const void*)psa_k2, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)K2_SM);
    done = true;
  }
}

at::Tensor psa_forward(at::Tensor x, at::Tensor W, at::Tensor Bs, at::Tensor ws) {
  const int B = (int)x.size(0);
  at::Tensor out = at::empty_like(x);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  __half* p = (__half*)ws.data_ptr();
  size_t off = 0;
  __half* Ab = p + off;  off += (size_t)B * P_N * 128;
  __half* BBb = p + off; off += (size_t)B * P_N * 128;
  __half* Qb = p + off;  off += (size_t)B * 2 * P_N * 32;
  __half* Kb = p + off;  off += (size_t)B * 2 * P_N * 32;
  __half* Vb = p + off;
  psa_init();
  const int grid = B * P_NTILE;
  const __half* xp = (const __half*)x.data_ptr();
  const __half* Wp = (const __half*)W.data_ptr();
  const float* Bp = (const float*)Bs.data_ptr();
  psa_k1<<<grid, P_NTHR, K1_SM, stream>>>(xp, Wp, Bp, Ab, BBb, Qb, Kb, Vb);
  psa_k2<<<grid, P_NTHR, K2_SM, stream>>>(Ab, BBb, Qb, Kb, Vb, Wp, Bp,
                                          (__half*)out.data_ptr());
  return out;
}

int64_t psa_ws_halves(int64_t B) { return B * (2 * P_N * 128 + 4 * P_N * 32 + 128 * P_N); }
"""

_CPP_SRC = """
#include <torch/extension.h>
at::Tensor psa_forward(at::Tensor x, at::Tensor W, at::Tensor Bs, at::Tensor ws);
int64_t psa_ws_halves(int64_t B);
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}" + ("a" if major in (9, 10, 12) else "")
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        try:
            _EXT = load_inline(
                name="fk_yolov10_psa_fused_v2",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["psa_forward", "psa_ws_halves"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
                verbose=False,
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _EXT


def _fold(conv, bn):
    """Fold conv(+bias) and eval-mode batch-norm into one weight/bias (fp32)."""
    w = conv.weight.detach().float()
    if bn is None:
        b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(
            w.shape[0], device=w.device)
        return w, b
    inv = torch.rsqrt(bn.running_var.detach().float() + bn.eps)
    g = bn.weight.detach().float() * inv
    wf = w * g.view(-1, *([1] * (w.dim() - 1)))
    bf = bn.bias.detach().float() - g * bn.running_mean.detach().float()
    if conv.bias is not None:
        bf = bf + g * conv.bias.detach().float()
    return wf, bf


_W_OFF = dict(w1=0, wq=65536, wpe=98304, wp=100352, wf0=116736, wf1=149504, w2=182272)
_W_TOTAL = 247808
_B_OFF = dict(b1=0, bq=256, bpe=512, bp=640, bf0=768, bf1=1024, b2=1152)
_B_TOTAL = 1408


class YOLOPSA(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )
        self._packed = None
        self._state = None
        self._ws = {}
        self._ok = (c1 == 256 and self.c == 128 and self.attn.num_heads == 2
                    and self.attn.key_dim == 32 and self.attn.head_dim == 64)

    # -- eager fallback (identical to the reference block) -------------------
    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

    @torch.no_grad()
    def _pack(self, device):
        W = torch.zeros(_W_TOTAL, dtype=torch.float16, device=device)
        Bs = torch.zeros(_B_TOTAL, dtype=torch.float32, device=device)
        specs = [
            ("w1", "b1", self.cv1.conv, getattr(self.cv1, "bn", None)),
            ("wq", "bq", self.attn.qkv.conv, getattr(self.attn.qkv, "bn", None)),
            ("wpe", "bpe", self.attn.pe.conv, getattr(self.attn.pe, "bn", None)),
            ("wp", "bp", self.attn.proj.conv, getattr(self.attn.proj, "bn", None)),
            ("wf0", "bf0", self.ffn[0].conv, getattr(self.ffn[0], "bn", None)),
            ("wf1", "bf1", self.ffn[1].conv, getattr(self.ffn[1], "bn", None)),
            ("w2", "b2", self.cv2.conv, getattr(self.cv2, "bn", None)),
        ]
        for wk, bk, conv, bn in specs:
            wf, bf = _fold(conv, bn)
            if wk == "wpe":                      # [128][3][3] -> [128][16]
                pad = torch.zeros(wf.shape[0], 16, device=wf.device)
                pad[:, :9] = wf.reshape(wf.shape[0], 9)
                wf = pad
            flat = wf.reshape(-1).to(torch.float16)
            W[_W_OFF[wk]:_W_OFF[wk] + flat.numel()] = flat
            Bs[_B_OFF[bk]:_B_OFF[bk] + bf.numel()] = bf
        self._packed = (W, Bs)

    # -- hot path: one tuple unpack, then straight into the extension --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        st = self._state
        if st is not None and x.shape == st[0] and x.dtype is st[1] and x.is_contiguous():
            return st[2](x, st[3], st[4], st[5])
        return self._setup(x)

    def _setup(self, x: torch.Tensor) -> torch.Tensor:
        if not (self._ok and x.is_cuda and x.dtype == torch.float16 and x.dim() == 4
                and x.shape[1] == 256 and x.shape[2] == 20 and x.shape[3] == 20
                and x.is_contiguous()):
            return self._eager(x)
        try:
            ext = _ext()
        except Exception:          # no nvcc / unsupported arch: stay correct
            self._ok = False
            return self._eager(x)
        if self._packed is None:
            self._pack(x.device)
        B = x.shape[0]
        ws = self._ws.get(B)
        if ws is None:
            ws = torch.empty(ext.psa_ws_halves(B), dtype=torch.float16, device=x.device)
            self._ws[B] = ws
        W, Bs = self._packed
        self._state = (x.shape, x.dtype, ext.psa_forward, W, Bs, ws)
        return ext.psa_forward(x, W, Bs, ws)
