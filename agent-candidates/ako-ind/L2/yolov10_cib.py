"""YOLOv10 CIB (Compact Inverted Block).

Inference-time reparameterization (BatchNorm folded into the preceding conv,
RepVGGDW 7x7/3x3 branches merged) plus a single cooperative CUDA kernel that
runs the whole block -- DW3x3 -> PW1x1 -> DW7x7 -> PW1x1 -> DW3x3 -> residual --
in one launch with two grid-wide barriers.  This shape is entirely launch-bound
(the baseline costs the same at batch 1 and batch 4), so collapsing ~20 eager
kernel launches into one is the whole win.
"""

from __future__ import annotations

import hashlib
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

_WP_HALVES = 81280          # packed-weight element count (must match the .cu)
_WS_PER_PIXEL = 512         # workspace halves per (batch, pixel)

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

#ifndef CIB_NWARP
#define CIB_NWARP 32
#endif
#ifndef CIB_R1
#define CIB_R1 1
#endif
#ifndef CIB_R3
#define CIB_R3 1
#endif
#ifndef CIB_CT
#define CIB_CT 4
#endif

#define CIB_C1   128
#define CIB_CM   256
#define CIB_CP1  64
#define CIB_CPM  128
#ifndef CIB_KU
#define CIB_KU 1
#endif
#ifndef CIB_COOP
#define CIB_COOP 1
#endif
#ifndef CIB_NOSILU
#define CIB_NOSILU 0
#endif
#ifndef CIB_NGROUP
#define CIB_NGROUP 1
#endif
#ifndef CIB_RG
#define CIB_RG 1
#endif
#define CIB_NTHREAD (CIB_NWARP * 32)
#define CIB_CHUNK 8
#define CIB_STR1(x) #x
#define CIB_STR(x) CIB_STR1(x)
#define CIB_UNROLL(n) _Pragma(CIB_STR(unroll n))

// Round a __half2 row pitch up so that (pitch % 32) is 8 or 24: the mma
// B-fragment read has lane -> (tig * pitch + gid), so pitch % 32 == 8 (or 24)
// spreads the four tig groups across all 32 shared-memory banks.
// Warp-group barrier.  Each group of GTHREAD threads runs an independent work
// item, so their memory latencies overlap on one SM without needing more blocks
// (the grid is smaller than the SM count, so extra blocks would not stack).
__device__ __forceinline__ void cib_bar(int id, int n) {
#if CIB_NGROUP == 1
    (void)id; (void)n;
    __syncthreads();
#else
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
#endif
}

__host__ __device__ constexpr int cib_ld(int n) {
    const int a = n + ((8 - n) & 31);
    const int b = n + ((24 - n) & 31);
    return a < b ? a : b;
}

// Packed-weight offsets, in __half2 units.
#define OFF_W2F  0
#define OFF_W4F  16384
#define OFF_W1   32768
#define OFF_W3   33344
#define OFF_W5   39616
#define OFF_B1   40192
#define OFF_B3   40256
#define OFF_B5   40384
#define OFF_B2   40448
#define OFF_B4   40576
#define CIB_WP_H2 40640

struct Acc { float x0, x1, x2, x3; };

__device__ __forceinline__ void mma16816(Acc &d, const uint32_t *a, const uint32_t *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d.x0), "+f"(d.x1), "+f"(d.x2), "+f"(d.x3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#if CIB_NOSILU
__device__ __forceinline__ float silu_f(float v) { return v * 0.5f; }
__device__ __forceinline__ __half2 silu_h2(__half2 v) { return __hmul2(v, __float2half2_rn(0.5f)); }
#else
__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + __expf(-v)); }

__device__ __forceinline__ __half2 silu_h2(__half2 v) {
    return __floats2half2_rn(silu_f(__low2float(v)), silu_f(__high2float(v)));
}
#endif

template <int WT>
__global__ void __launch_bounds__(CIB_NTHREAD, 1)
cib_kernel(const __half *__restrict__ xp, const __half2 *__restrict__ wp,
           __half2 *__restrict__ ws, __half *__restrict__ outp,
           int N, int H, int P, long sn, long sc) {
    constexpr int PT1 = CIB_R1 * WT;             // phase-1 output pixels / tile
    constexpr int NT1 = (PT1 + 7) / 8;
    constexpr int XR  = CIB_R1 + 2;              // x rows staged in phase 1
    constexpr int XPIX = XR * WT;
    constexpr int XLD = cib_ld(XPIX);
    constexpr int PT4 = (CIB_R3 + 2) * WT;       // phase-3 a4 pixel slots
    constexpr int NT4 = (PT4 + 7) / 8;
    constexpr int PO3 = CIB_R3 * WT;             // phase-3 output pixels / tile
    constexpr int LD3 = cib_ld(PT4);
    // GEMM warp partitioning: WM warps over M-tiles, WN warps over N-tiles.
    constexpr int MTOT1 = CIB_CM / 16, MTOT4 = CIB_C1 / 16;
    constexpr int GW = CIB_NWARP / CIB_NGROUP;
    constexpr int WM1 = MTOT1 < GW ? MTOT1 : GW;
    constexpr int WN1 = GW / WM1;
    constexpr int MT1 = MTOT1 / WM1;
    constexpr int NTW1 = (NT1 + WN1 - 1) / WN1;
    constexpr int WM4 = MTOT4 < GW ? MTOT4 : GW;
    constexpr int WN4 = GW / WM4;
    constexpr int MT4 = MTOT4 / WM4;
    constexpr int NTW4 = (NT4 + WN4 - 1) / WN4;
    constexpr int KT1 = CIB_C1 / 16, KT4 = CIB_CM / 16;
    constexpr int GWARP = CIB_NWARP / CIB_NGROUP;
    constexpr int GTHREAD = GWARP * 32;

    extern __shared__ __half2 smem[];
    const int lane = threadIdx.x & 31;
    const int grp = (int)(threadIdx.x >> 5) / GWARP;
    const int warp = (int)(threadIdx.x >> 5) - grp * GWARP;
    const int tid = (int)threadIdx.x - grp * GTHREAD;
    const int gid = lane >> 2, tig = lane & 3;
    const int nblk = gridDim.x;
    const __half2 zero2 = __float2half2_rn(0.0f);

    const __half2 *W2F = wp + OFF_W2F;
    const __half2 *W4F = wp + OFF_W4F;
    const __half2 *W1  = wp + OFF_W1;
    const __half2 *W3  = wp + OFF_W3;
    const __half2 *W5  = wp + OFF_W5;
    const __half2 *B1  = wp + OFF_B1;
    const __half2 *B3  = wp + OFF_B3;
    const __half2 *B5  = wp + OFF_B5;
    const __half  *B2  = (const __half *)(wp + OFF_B2);
    const __half  *B4  = (const __half *)(wp + OFF_B4);

    __half2 *A2 = ws;
    __half2 *A3 = A2 + (long)N * CIB_CPM * P;

    // ============ Phase 1 : DW3x3 -> SiLU -> PW(128->256) -> SiLU =========
    {
        const int tiles = (H + CIB_R1 - 1) / CIB_R1;
        const int nitem = N * tiles;
        __half2 *xs  = smem + grp * (CIB_CP1 * XLD + CIB_CP1 * PT1 + CIB_CPM * PT1);
        __half2 *a1s = xs + CIB_CP1 * XLD;          // [CP1][PT1]
        __half2 *o2s = a1s + CIB_CP1 * PT1;         // [CPM][PT1], half-interleaved
        __half  *o2h = (__half *)o2s;
        const int wm = warp % WM1, wn = warp / WM1;

        for (int item = blockIdx.x * CIB_NGROUP + grp; item < nitem;
             item += nblk * CIB_NGROUP) {
            const int n  = item / tiles;
            const int r0 = (item - n * tiles) * CIB_R1;
            const int nrow = min(CIB_R1, H - r0);
            const int npix = nrow * WT;
            const __half *xbase = xp + (long)n * sn;
            cib_bar(grp, GTHREAD);
            // --- stage x rows [r0-1, r0+R1], two pixels per load
            for (int i = tid; i < CIB_CP1 * (XPIX / 2); i += GTHREAD) {
                const int cp = i / (XPIX / 2);
                const int pp = (i - cp * (XPIX / 2)) * 2;
                const int ri = pp / WT, c2 = pp - ri * WT;
                const int rr = r0 - 1 + ri;
                __half2 va = zero2, vb = zero2;
                if (rr >= 0 && rr < H) {
                    const __half *b = xbase + (long)(2 * cp) * sc + (long)rr * WT + c2;
                    va = *(const __half2 *)b;
                    vb = *(const __half2 *)(b + sc);
                }
                __half2 *d = xs + cp * XLD + pp;
                d[0] = __halves2half2(__low2half(va), __low2half(vb));
                d[1] = __halves2half2(__high2half(va), __high2half(vb));
            }
            cib_bar(grp, GTHREAD);
            // --- a1 = SiLU(dw3x3(x) + b1), two pixels per thread
            for (int i = tid; i < CIB_CP1 * (PT1 / 2); i += GTHREAD) {
                const int cp = i / (PT1 / 2);
                const int pp = (i - cp * (PT1 / 2)) * 2;
                const int ri = pp / WT, c2 = pp - ri * WT;
                __half2 acc0 = B1[cp], acc1 = B1[cp];
                if (ri < nrow) {
                    const __half2 *xr = xs + cp * XLD + ri * WT;
#pragma unroll
                    for (int ky = 0; ky < 3; ++ky) {
                        __half2 v[4];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int cc = c2 - 1 + j;
                            v[j] = (cc >= 0 && cc < WT) ? xr[ky * WT + cc] : zero2;
                        }
#pragma unroll
                        for (int kx = 0; kx < 3; ++kx) {
                            const __half2 w = W1[(ky * 3 + kx) * CIB_CP1 + cp];
                            acc0 = __hfma2(v[kx], w, acc0);
                            acc1 = __hfma2(v[kx + 1], w, acc1);
                        }
                    }
                    acc0 = silu_h2(acc0);
                    acc1 = silu_h2(acc1);
                } else {
                    acc0 = zero2; acc1 = zero2;
                }
                __half2 *d = a1s + cp * PT1 + pp;
                d[0] = acc0; d[1] = acc1;
            }
            cib_bar(grp, GTHREAD);
            // --- a2 = SiLU(W2 @ a1 + b2)
            Acc acc[MT1][NTW1];
#pragma unroll
            for (int mt = 0; mt < MT1; ++mt)
#pragma unroll
                for (int nt = 0; nt < NTW1; ++nt) acc[mt][nt] = Acc{0.f, 0.f, 0.f, 0.f};
            CIB_UNROLL(CIB_KU)
            for (int kt = 0; kt < KT1; ++kt) {
                uint32_t bf[NTW1][2];
                const int cpb = kt * 8 + tig;
#pragma unroll
                for (int nt = 0; nt < NTW1; ++nt) {
                    const int p = (wn * NTW1 + nt) * 8 + gid;
                    const int q = p < PT1 ? p : PT1 - 1;
                    bf[nt][0] = *(const uint32_t *)&a1s[cpb * PT1 + q];
                    bf[nt][1] = *(const uint32_t *)&a1s[(cpb + 4) * PT1 + q];
                }
#pragma unroll
                for (int mt = 0; mt < MT1; ++mt) {
                    const int mtile = wm * MT1 + mt;
                    const int4 af = *(const int4 *)(W2F + (((long)mtile * KT1 + kt) * 32 + lane) * 4);
                    const uint32_t *a = (const uint32_t *)&af;
#pragma unroll
                    for (int nt = 0; nt < NTW1; ++nt) mma16816(acc[mt][nt], a, bf[nt]);
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT1; ++mt) {
                const int mtile = wm * MT1 + mt;
                const int coa = mtile * 16 + gid, cob = coa + 8;
                const float b2a = __half2float(B2[coa]), b2b = __half2float(B2[cob]);
                const int ha = (coa >> 1) * (PT1 * 2) + (coa & 1);
                const int hb = (cob >> 1) * (PT1 * 2) + (cob & 1);
#pragma unroll
                for (int nt = 0; nt < NTW1; ++nt) {
                    const int p = (wn * NTW1 + nt) * 8 + tig * 2;
                    const Acc &d = acc[mt][nt];
                    if (p < PT1) {
                        o2h[ha + p * 2] = __float2half(silu_f(d.x0 + b2a));
                        o2h[hb + p * 2] = __float2half(silu_f(d.x2 + b2b));
                    }
                    if (p + 1 < PT1) {
                        o2h[ha + (p + 1) * 2] = __float2half(silu_f(d.x1 + b2a));
                        o2h[hb + (p + 1) * 2] = __float2half(silu_f(d.x3 + b2b));
                    }
                }
            }
            cib_bar(grp, GTHREAD);
            for (int i = tid; i < CIB_CPM * npix; i += GTHREAD) {
                const int cp = i / npix, rem = i - cp * npix;
                A2[((long)(n * CIB_CPM) + cp) * P + r0 * WT + rem] = o2s[cp * PT1 + rem];
            }
        }
    }
    cg::this_grid().sync();

    // ============ Phase 2 : merged DW7x7 -> SiLU ==========================
    {
        const int cts = (CIB_CPM + CIB_CT - 1) / CIB_CT;
        const int nitem = N * cts;
        const int nchunk = (WT + CIB_CHUNK - 1) / CIB_CHUNK;
        const int rld = WT + 1;                 // row pitch: gcd(WT+1, 32) == 1
        const int cld = H * rld;
        __half2 *a2s = smem + grp * (CIB_CT * cld + CIB_CT * 49);
        __half2 *w3s = a2s + CIB_CT * cld;      // [CT][49]

        for (int item = blockIdx.x * CIB_NGROUP + grp; item < nitem;
             item += nblk * CIB_NGROUP) {
            const int n = item / cts;
            const int cp0 = (item - n * cts) * CIB_CT;
            const int ncp = min(CIB_CT, CIB_CPM - cp0);
            cib_bar(grp, GTHREAD);
            for (int i = tid; i < ncp * P; i += GTHREAD) {
                const int c = i / P, p = i - c * P;
                const int r = p / WT;
                a2s[c * cld + r * rld + (p - r * WT)] =
                    A2[((long)(n * CIB_CPM) + cp0 + c) * P + p];
            }
            for (int i = tid; i < ncp * 49; i += GTHREAD) {
                const int c = i / 49, k = i - c * 49;
                w3s[c * 49 + k] = W3[k * CIB_CPM + cp0 + c];
            }
            cib_bar(grp, GTHREAD);
            const int rgroups = (H + CIB_RG - 1) / CIB_RG;
            const int ntask = ncp * rgroups * nchunk;
            for (int task = tid; task < ntask; task += GTHREAD) {
                const int c = task / (rgroups * nchunk);
                const int rem = task - c * (rgroups * nchunk);
                const int rg = rem / nchunk;
                const int row0 = rg * CIB_RG;
                const int col0 = (rem - rg * nchunk) * CIB_CHUNK;
                // acc[u] holds output row row0+u; the 7-row vertical window of
                // consecutive output rows overlaps, so one staged row feeds
                // CIB_RG accumulators.
                __half2 acc[CIB_RG][CIB_CHUNK];
                const __half2 bias = B3[cp0 + c];
#pragma unroll
                for (int u = 0; u < CIB_RG; ++u)
#pragma unroll
                    for (int i = 0; i < CIB_CHUNK; ++i) acc[u][i] = bias;
#pragma unroll 1
                for (int ky = 0; ky < 6 + CIB_RG; ++ky) {
                    const int rr = row0 + ky - 3;
                    if (rr < 0 || rr >= H) continue;
                    const __half2 *src = a2s + c * cld + rr * rld;
                    __half2 v[CIB_CHUNK + 6];
#pragma unroll
                    for (int j = 0; j < CIB_CHUNK + 6; ++j) {
                        const int cc = col0 - 3 + j;
                        v[j] = (cc >= 0 && cc < WT) ? src[cc] : zero2;
                    }
#pragma unroll
                    for (int u = 0; u < CIB_RG; ++u) {
                        const int kyu = ky - u;
                        if (kyu < 0 || kyu > 6) continue;
#pragma unroll
                        for (int kx = 0; kx < 7; ++kx) {
                            const __half2 w = w3s[c * 49 + kyu * 7 + kx];
#pragma unroll
                            for (int i = 0; i < CIB_CHUNK; ++i)
                                acc[u][i] = __hfma2(v[i + kx], w, acc[u][i]);
                        }
                    }
                }
#pragma unroll
                for (int u = 0; u < CIB_RG; ++u) {
                    const int row = row0 + u;
                    if (row >= H) continue;
                    __half2 *dst = A3 + ((long)(n * CIB_CPM) + cp0 + c) * P + row * WT;
#pragma unroll
                    for (int i = 0; i < CIB_CHUNK; ++i)
                        if (col0 + i < WT) dst[col0 + i] = silu_h2(acc[u][i]);
                }
            }
        }
    }
    cg::this_grid().sync();

    // ============ Phase 3 : PW(256->128) -> SiLU -> DW3x3 -> +x ===========
    {
        const int tiles = (H + CIB_R3 - 1) / CIB_R3;
        const int nitem = N * tiles;
        __half2 *a3s = smem + grp * (CIB_CPM * LD3);   // [CPM][LD3]
        __half2 *a4s = a3s;                            // [CP1][LD3], aliases a3s
        __half  *a4h = (__half *)a4s;
        const int wm = warp % WM4, wn = warp / WM4;

        for (int item = blockIdx.x * CIB_NGROUP + grp; item < nitem;
             item += nblk * CIB_NGROUP) {
            const int n = item / tiles;
            const int r0 = (item - n * tiles) * CIB_R3;
            cib_bar(grp, GTHREAD);
            // stage the a3 rows this tile needs (zero outside the image)
            for (int i = tid; i < CIB_CPM * PT4; i += GTHREAD) {
                const int cp = i / PT4;
                const int slot = i - cp * PT4;
                const int ri = slot / WT, col = slot - ri * WT;
                const int rr = r0 - 1 + ri;
                a3s[cp * LD3 + slot] = (rr >= 0 && rr < H)
                    ? A3[((long)(n * CIB_CPM) + cp) * P + (long)rr * WT + col]
                    : zero2;
            }
            cib_bar(grp, GTHREAD);
            Acc acc[MT4][NTW4];
#pragma unroll
            for (int mt = 0; mt < MT4; ++mt)
#pragma unroll
                for (int nt = 0; nt < NTW4; ++nt) acc[mt][nt] = Acc{0.f, 0.f, 0.f, 0.f};
            CIB_UNROLL(CIB_KU)
            for (int kt = 0; kt < KT4; ++kt) {
                uint32_t bf[NTW4][2];
                const int cpb = kt * 8 + tig;
#pragma unroll
                for (int nt = 0; nt < NTW4; ++nt) {
                    const int i = (wn * NTW4 + nt) * 8 + gid;
                    const int q = i < PT4 ? i : PT4 - 1;
                    bf[nt][0] = *(const uint32_t *)&a3s[cpb * LD3 + q];
                    bf[nt][1] = *(const uint32_t *)&a3s[(cpb + 4) * LD3 + q];
                }
#pragma unroll
                for (int mt = 0; mt < MT4; ++mt) {
                    const int mtile = wm * MT4 + mt;
                    const int4 af = *(const int4 *)(W4F + (((long)mtile * KT4 + kt) * 32 + lane) * 4);
                    const uint32_t *a = (const uint32_t *)&af;
#pragma unroll
                    for (int nt = 0; nt < NTW4; ++nt) mma16816(acc[mt][nt], a, bf[nt]);
                }
            }
            cib_bar(grp, GTHREAD);
#pragma unroll
            for (int mt = 0; mt < MT4; ++mt) {
                const int mtile = wm * MT4 + mt;
                const int coa = mtile * 16 + gid, cob = coa + 8;
                const float b4a = __half2float(B4[coa]), b4b = __half2float(B4[cob]);
                const int ha = (coa >> 1) * (LD3 * 2) + (coa & 1);
                const int hb = (cob >> 1) * (LD3 * 2) + (cob & 1);
#pragma unroll
                for (int nt = 0; nt < NTW4; ++nt) {
                    const int p = (wn * NTW4 + nt) * 8 + tig * 2;
                    const Acc &d = acc[mt][nt];
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const int q = p + u;
                        if (q >= PT4) continue;
                        const int rr = r0 - 1 + q / WT;
                        const bool ok = (rr >= 0 && rr < H);
                        const float va = u ? d.x1 : d.x0;
                        const float vb = u ? d.x3 : d.x2;
                        a4h[ha + q * 2] = ok ? __float2half(silu_f(va + b4a)) : __float2half(0.f);
                        a4h[hb + q * 2] = ok ? __float2half(silu_f(vb + b4b)) : __float2half(0.f);
                    }
                }
            }
            cib_bar(grp, GTHREAD);
            // out = x + SiLU(dw3x3(a4) + b5), two pixels per thread
            const __half *xbase = xp + (long)n * sn;
            for (int i = tid; i < CIB_CP1 * (PO3 / 2); i += GTHREAD) {
                const int cp = i / (PO3 / 2);
                const int pp = (i - cp * (PO3 / 2)) * 2;
                const int ri = pp / WT, c2 = pp - ri * WT;
                const int rr = r0 + ri;
                if (rr >= H) continue;
                __half2 acc0 = B5[cp], acc1 = B5[cp];
                const __half2 *ar = a4s + cp * LD3 + ri * WT;
#pragma unroll
                for (int ky = 0; ky < 3; ++ky) {
                    __half2 v[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int cc = c2 - 1 + j;
                        v[j] = (cc >= 0 && cc < WT) ? ar[ky * WT + cc] : zero2;
                    }
#pragma unroll
                    for (int kx = 0; kx < 3; ++kx) {
                        const __half2 w = W5[(ky * 3 + kx) * CIB_CP1 + cp];
                        acc0 = __hfma2(v[kx], w, acc0);
                        acc1 = __hfma2(v[kx + 1], w, acc1);
                    }
                }
                acc0 = silu_h2(acc0);
                acc1 = silu_h2(acc1);
                const __half *xa = xbase + (long)(2 * cp) * sc + (long)rr * WT + c2;
                const __half2 xv0 = *(const __half2 *)xa;
                const __half2 xv1 = *(const __half2 *)(xa + sc);
                __half *d = outp + ((long)n * CIB_C1 + 2 * cp) * P + (long)rr * WT + c2;
                *(__half2 *)d = __floats2half2_rn(__low2float(xv0) + __low2float(acc0),
                                                  __high2float(xv0) + __low2float(acc1));
                *(__half2 *)(d + P) = __floats2half2_rn(__low2float(xv1) + __high2float(acc0),
                                                        __high2float(xv1) + __high2float(acc1));
            }
        }
    }
}

// ---------------------------------------------------------------- host side
template <int WT>
static bool cib_launch(const __half *xp, const __half2 *wp, __half2 *wsp, __half *outp,
                       int N, int H, int P, long sn, long sc) {
    constexpr int W_RT = WT;
    constexpr int PT1 = CIB_R1 * WT;
    constexpr int XLD = cib_ld((CIB_R1 + 2) * WT);
    constexpr int LD3 = cib_ld((CIB_R3 + 2) * WT);
    constexpr int S1 = CIB_NGROUP * (CIB_CP1 * XLD + CIB_CP1 * PT1 + CIB_CPM * PT1);
    constexpr int S3 = CIB_NGROUP * (CIB_CPM * LD3);
    const int s2 = CIB_NGROUP * (CIB_CT * H * (W_RT + 1) + CIB_CT * 49);
    int smem_h2 = S1 > S3 ? S1 : S3;
    if (s2 > smem_h2) smem_h2 = s2;
    const int smem = smem_h2 * (int)sizeof(__half2);

    void *fn = (void *)cib_kernel<WT>;
    // Driver queries (func attribute, occupancy, SM count) are several
    // microseconds of CPU each -- at this size that is a large slice of the
    // per-call budget, so do them once and cache.
    static int cached_smem = -1, cap = 0;
    if (cached_smem != smem) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        int sm_count = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, CIB_NTHREAD, smem);
        cap = per_sm * sm_count;
        cached_smem = smem;
    }
    if (cap <= 0) return false;

    const int tiles1 = (H + CIB_R1 - 1) / CIB_R1;
    const int tiles3 = (H + CIB_R3 - 1) / CIB_R3;
    const int cts = (CIB_CPM + CIB_CT - 1) / CIB_CT;
    int grid = (N * tiles1 + CIB_NGROUP - 1) / CIB_NGROUP;
    const int g2 = (N * cts + CIB_NGROUP - 1) / CIB_NGROUP;
    const int g3 = (N * tiles3 + CIB_NGROUP - 1) / CIB_NGROUP;
    if (g2 > grid) grid = g2;
    if (g3 > grid) grid = g3;
    if (grid > cap) grid = cap;

    void *args[] = {(void *)&xp, (void *)&wp, (void *)&wsp, (void *)&outp,
                    (void *)&N, (void *)&H, (void *)&P, (void *)&sn, (void *)&sc};
    return cudaLaunchCooperativeKernel(fn, dim3(grid), dim3(CIB_NTHREAD), args, smem,
                                       at::cuda::getCurrentCUDAStream()) == cudaSuccess;
}

torch::Tensor cib_forward(torch::Tensor x, torch::Tensor wp, torch::Tensor ws) {
    const int N = (int)x.size(0), H = (int)x.size(2), W = (int)x.size(3);
    const int P = H * W;
    TORCH_CHECK(x.dim() == 4 && x.size(1) == CIB_C1, "cib: bad shape");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "cib: dtype");
    TORCH_CHECK(x.stride(3) == 1 && x.stride(2) == W, "cib: layout");
    TORCH_CHECK((W & 1) == 0 && (x.stride(0) & 1) == 0 && (x.stride(1) & 1) == 0,
                "cib: odd stride");
    TORCH_CHECK((((uintptr_t)x.data_ptr()) & 3) == 0, "cib: alignment");
    TORCH_CHECK(wp.numel() == 2 * CIB_WP_H2, "cib: weight pack size");
    TORCH_CHECK(ws.numel() >= (long)N * P * 2 * 2 * CIB_CPM, "cib: workspace too small");
    auto out = torch::empty({N, CIB_C1, H, W}, x.options());
    bool ok = false;
    if (W == 20) {
        ok = cib_launch<20>((const __half *)x.data_ptr(), (const __half2 *)wp.data_ptr(),
                            (__half2 *)ws.data_ptr(), (__half *)out.data_ptr(),
                            N, H, P, (long)x.stride(0), (long)x.stride(1));
    }
    TORCH_CHECK(ok, "cib_forward: unsupported geometry or launch failure");
    return out;
}
'''

_CPP_SRC = r'''
#include <torch/extension.h>
torch::Tensor cib_forward(torch::Tensor x, torch::Tensor wp, torch::Tensor ws);
'''


_EXT = None
_EXT_TRIED = False
_EXT_LOCK = threading.Lock()


def _get_ext():
    """Compile (once) and return the CUDA extension, or None if unavailable."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    with _EXT_LOCK:
        if _EXT_TRIED:
            return _EXT
        _EXT_TRIED = True
        try:
            from torch.utils.cpp_extension import load_inline

            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            tag = hashlib.sha1((_CUDA_SRC + _CPP_SRC + arch).encode()).hexdigest()[:12]
            _EXT = load_inline(
                name=f"ako_yolov10_cib_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["cib_forward"],
                verbose=False,
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    f"-gencode=arch=compute_{arch},code=sm_{arch}",
                ],
            )
        except Exception:
            _EXT = None
    return _EXT


def _fold(block: nn.Module):
    """YOLOConv -> (weight, bias) in fp32 with BatchNorm folded in."""
    conv = block.conv
    w = conv.weight.detach().float()
    bias = conv.bias
    b = (torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
         if bias is None else bias.detach().float())
    bn = getattr(block, "bn", None)
    if bn is None:
        return w, b
    scale = bn.weight.detach().float() / torch.sqrt(
        bn.running_var.detach().float() + bn.eps)
    w = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
    b = (b - bn.running_mean.detach().float()) * scale + bn.bias.detach().float()
    return w, b


def _pack_mma_a(w: torch.Tensor) -> torch.Tensor:
    """[M, K] fp16 -> mma.m16n8k16 A-fragment order (16 bytes per lane per step)."""
    m_tiles, k_tiles = w.shape[0] // 16, w.shape[1] // 16
    lane = torch.arange(32, device=w.device)
    gid, tig = (lane >> 2).long(), (lane & 3).long()
    out = torch.empty(m_tiles, k_tiles, 32, 4, 2, dtype=w.dtype, device=w.device)
    for mt in range(m_tiles):
        for kt in range(k_tiles):
            r, c = mt * 16, kt * 16
            for reg, (dr, dc) in enumerate(((0, 0), (8, 0), (0, 8), (8, 8))):
                out[mt, kt, :, reg, 0] = w[r + gid + dr, c + tig * 2 + dc]
                out[mt, kt, :, reg, 1] = w[r + gid + dr, c + tig * 2 + dc + 1]
    return out.reshape(-1)


def _pack_dw(w: torch.Tensor) -> torch.Tensor:
    """[C, 1, k, k] fp16 -> [k*k][C/2][2] so a channel pair is one half2."""
    c = w.shape[0]
    t = w.reshape(c, -1)
    return t.reshape(c // 2, 2, t.shape[1]).permute(2, 0, 1).contiguous().reshape(-1)


def _pack_pairs(b: torch.Tensor) -> torch.Tensor:
    """[C] -> [C/2][2] (one half2 per channel pair)."""
    return b.reshape(-1, 2).contiguous().reshape(-1)


class YOLOCIB(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            YOLOConv(c1, c1, 3, g=c1),
            YOLOConv(c1, 2 * c_, 1),
            YOLOConv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else YOLORepVGGDW(2 * c_),
            YOLOConv(2 * c_, c2, 1),
            YOLOConv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2
        # Geometry the fused kernel is specialised for.
        self._supported = (c1 == 128 and c2 == 128 and int(c2 * e) == 128 and self.add)
        self._state = None      # (fn, packed_weights, workspace)
        self._packed = None
        self._ws = None
        self._ws_pixels = 0

    # ---------------------------------------------------------------- fast path
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # s = (kernel, packed weights, workspace, torch.is_grad_enabled).  The
        # fused kernel is inference-only (no autograd graph), so fall back to the
        # eager path whenever grad is on.
        s = self._state
        if s is not None and not s[3]():
            try:
                return s[0](x, s[1], s[2])
            except Exception:
                pass
        return self._slow(x)

    # ---------------------------------------------------------------- fallback
    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.add(x, y) if self.add else y

    def _slow(self, x: torch.Tensor) -> torch.Tensor:
        """Build (or grow) the fused state if this input is supported, else eager."""
        if not self._usable(x):
            return self._eager(x)
        ext = _get_ext()
        if ext is None:
            self._supported = False
            return self._eager(x)
        if self._packed is None:
            try:
                self._packed = self._pack(x.device)
            except Exception:
                self._supported = False
                return self._eager(x)
        pixels = x.shape[0] * x.shape[2] * x.shape[3]
        if self._ws is None or pixels > self._ws_pixels:
            self._ws = torch.empty(pixels * _WS_PER_PIXEL, dtype=torch.float16,
                                   device=x.device)
            self._ws_pixels = pixels
        self._state = (ext.cib_forward, self._packed, self._ws, torch.is_grad_enabled)
        try:
            return ext.cib_forward(x, self._packed, self._ws)
        except Exception:
            self._state = None
            self._supported = False
            return self._eager(x)

    def _usable(self, x: torch.Tensor) -> bool:
        return (self._supported and not torch.is_grad_enabled()
                and x.is_cuda and x.dtype is torch.float16
                and x.dim() == 4 and x.shape[1] == 128 and x.shape[3] == 20
                and x.stride(3) == 1 and x.stride(2) == x.shape[3])

    # ------------------------------------------------------- reparameterization
    @torch.no_grad()
    def _pack(self, device) -> torch.Tensor:
        cv = self.cv1
        w1, b1 = _fold(cv[0])
        w2, b2 = _fold(cv[1])
        mid = cv[2]
        if isinstance(mid, YOLORepVGGDW):
            w7, b7 = _fold(mid.conv)
            if hasattr(mid, "conv1"):
                w3s, b3s = _fold(mid.conv1)
                w7 = w7 + F.pad(w3s, [2, 2, 2, 2])
                b7 = b7 + b3s
            wm, bm = w7, b7
        else:
            wm, bm = _fold(mid)
            pad = (7 - wm.shape[-1]) // 2
            wm = F.pad(wm, [pad, pad, pad, pad])
        w4, b4 = _fold(cv[3])
        w5, b5 = _fold(cv[4])
        if wm.shape[-1] != 7 or wm.shape[0] != 256:
            raise ValueError("unsupported middle depthwise kernel")

        h = torch.float16
        parts = [
            _pack_mma_a(w2.reshape(256, 128).to(h)),
            _pack_mma_a(w4.reshape(128, 256).to(h)),
            _pack_dw(w1.to(h)),
            _pack_dw(wm.to(h)),
            _pack_dw(w5.to(h)),
            _pack_pairs(b1.to(h)),
            _pack_pairs(bm.to(h)),
            _pack_pairs(b5.to(h)),
            b2.to(h).contiguous(),
            b4.to(h).contiguous(),
        ]
        wp = torch.cat(parts).to(device=device, dtype=h).contiguous()
        if wp.numel() != _WP_HALVES:
            raise ValueError(f"packed weight size {wp.numel()} != {_WP_HALVES}")
        return wp
