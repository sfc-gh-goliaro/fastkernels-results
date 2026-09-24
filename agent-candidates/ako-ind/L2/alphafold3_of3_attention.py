"""Multi-head attention with bias list support for AlphaFold3 (L2).

Composes QKV projections + SDPA + gated output.

Reference: openfold3/core/model/primitives/attention.py Attention

Optimization notes
------------------
Every captured shape is tiny (largest activation ~50K elements) so the whole
module is overhead-bound, not FLOP-bound: the eager baseline spends ~200 us of
CPU issuing ~15 kernels whose total GPU time is ~40 us, and the harness's own
timed region has an ~11-15 us floor (measured directly with a forward that just
returns ``q_x``).  Everything here collapses the module into **one Python-level
call**, and then into as few kernels as the shape allows.

The binding resource is **CTAs of work**, not arithmetic: B*H is 16-64 and the
GPU has 148 SMs, so the same computation is fast or slow depending on how wide a
grid it can be spread over.  That is why there are two paths, chosen per call:

*Narrow c_q* (``c_hidden * (c_q + c_k) <= 12288``) -- **one fused kernel** per
(batch, head).  The head dimension partitions the *output columns* of every
projection weight (``linear_{q,k,v,g}`` all emit ``[.., H*c_hidden]``), so the
CTA that owns head h needs exactly rows ``[h*c_hidden, (h+1)*c_hidden)`` of each
weight and computes its own q/k/v/g in shared memory for the same total MACs as
a separate projection GEMM.  Nothing about q/k/v/g round-trips through global
memory, the bias tile is staged in the same phase and consumed by
``load_matrix_sync`` on the score accumulator (so the additive bias costs zero
instructions), and two of the three launches disappear.

*Wide c_q* -- **three PDL-chained kernels** (``proj2``, ``attn``, ``outproj``),
because there B*H is only 16 CTAs and one SM cannot stream a 294 KB weight slice
as fast as 192 CTAs can stream the same bytes between them.  Programmatic
Dependent Launch keeps the launch gaps off the critical path: producers call
``cudaTriggerProgrammaticLaunchCompletion()``, consumers
``cudaGridDependencySynchronize()`` immediately before their first read of
producer-written memory.

The GEMMs split their K reduction across the block's warps, so the *useful* warp
count is capped by K/16 -- 8 for c_q=128 but 24-48 for c_q=384/768.  The block
width is therefore chosen per call rather than fixed.

The concatenated projection weights (``[Wq|Wg]``, ``[Wk|Wv]``) are built lazily
on first forward and cached, invalidated by ``_apply`` / ``load_state_dict``
hooks plus a per-forward parameter ``_version`` check.  They carry 16 zero pad
rows so the fused kernel's ragged c_hidden column tile (c_hidden 24 -> a 32-wide
tile) can read past the last head without a predicate in the inner loop.

The generic torch path (identical to the baseline) is kept and is used for
anything the fast paths do not cover, so the forward contract still holds for
arbitrary bias lists, dtypes, layouts and batch shapes.

"""

from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.softmax import Softmax

# ---------------------------------------------------------------------------
# Fused CUDA kernels.
# ---------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <optional>

using namespace nvcuda;

// ==== KERNELS BEGIN ====
#define MAX_BIAS 4
#define MAX_BDIM 6

// Both kernels are latency-bound at 16-512 CTAs, so warps-per-SM is the lever
// that matters (ncu: 40% long-scoreboard stall at 4 warps/CTA).  The GEMMs want
// 8 warps -- that is an exact 8-way split of the 128-deep reductions in the
// captures -- while the attention wants 16, one warp per query row, which
// collapses every staging loop to a single iteration.  Unifying them (as a
// single cooperative kernel must) costs the GEMMs more than it saves in
// launches, so they stay separate kernels and the launch latency between them is
// hidden with PDL instead.
// Programmatic Dependent Launch: the consumer grid is launched while the
// producer drains, so the ~3.4 us end-of-kernel-to-start-of-next latency (three
// launches = ~7 us of gap on a ~12 us job here) is paid once instead of three
// times.  The wait sits immediately before the first read of producer-written
// memory.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define PDL_TRIGGER() cudaTriggerProgrammaticLaunchCompletion()
#define PDL_WAIT() cudaGridDependencySynchronize()
#else
#define PDL_TRIGGER()
#define PDL_WAIT()
#endif

#define GTHREADS 256
#define GTHREADS_W 512
#define KTWIDE 16
#define ATHREADS 512
#define UNROLL 4

#define ABQ 16
#define KCMAX 128
#define MAX_C 128
#define FTHREADS 1024
#define FQMAX 32
#define FKMAX 128

struct BiasOne {
    const void* ptr;
    long long bs[MAX_BDIM];
    long long sh, sq, sk;
};

struct BiasPack {
    BiasOne b[MAX_BIAS];
    int n;
};

struct BatchDims {
    int sizes[MAX_BDIM];
    int nd;
};

template <typename T> __device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }

template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }

// ---------------------------------------------------------------------------
// One CTA computes a 16x16 output tile of Out[m, n] = sum_k X[m, k] * W[n, k]
// (+ Bv[n]).  The K reduction is split across the block's warps and each warp
// keeps UNROLL wmma fragment pairs in flight, so a whole k-slice costs roughly
// one memory latency instead of one per 16-wide step.  Fragments are loaded
// straight from global memory -- no shared staging and no block-wide barrier in
// the main loop, which was worth ~4x over a staged BK=32 loop whose single
// exposed round trip per step dominated at these CTA counts.  Ragged edge tiles
// (M, N not multiples of 16, or K not a multiple of 16) take a scalar path.
// ---------------------------------------------------------------------------
template <typename T>
__device__ void gemm16_tile(const T* __restrict__ X, int ldx, const T* __restrict__ W, int ldw,
                            const T* __restrict__ Bv, T* __restrict__ Out, int ldo, int M, int N,
                            int K, int m0, int n0, float* red) {
    const int nthr = blockDim.x;
    const int nw = nthr >> 5;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;

    if (!(m0 + 16 <= M && n0 + 16 <= N && (K & 15) == 0)) {
        for (int idx = tid; idx < 256; idx += nthr) {
            const int r = idx >> 4;
            const int cn = idx & 15;
            const int gr = m0 + r;
            const int gc = n0 + cn;
            if (gr < M && gc < N) {
                const T* xr = X + (long long)gr * ldx;
                const T* wr = W + (long long)gc * ldw;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += to_f<T>(xr[k]) * to_f<T>(wr[k]);
                if (Bv != nullptr) acc += to_f<T>(Bv[gc]);
                Out[(long long)gr * ldo + gc] = from_f<T>(acc);
            }
        }
        return;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const T* Xb = X + (long long)m0 * ldx;
    const T* Wb = W + (long long)n0 * ldw;
    const int KT = K >> 4;
    const int nsteps = (KT - warp + nw - 1) / nw;  // k steps owned by this warp
    int kt = warp;
    int done = 0;
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa[UNROLL];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb[UNROLL];
        for (; done + UNROLL <= nsteps; done += UNROLL) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int off = (kt + u * nw) << 4;
                wmma::load_matrix_sync(fa[u], Xb + off, ldx);
                wmma::load_matrix_sync(fb[u], Wb + off, ldw);
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) wmma::mma_sync(acc, fa[u], fb[u], acc);
            kt += UNROLL * nw;
        }
    }
    for (; done < nsteps; ++done, kt += nw) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, Xb + (kt << 4), ldx);
        wmma::load_matrix_sync(fb, Wb + (kt << 4), ldw);
        wmma::mma_sync(acc, fa, fb, acc);
    }

    wmma::store_matrix_sync(red + warp * (16 * 20), acc, 20, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < 256; idx += nthr) {
        const int r = idx >> 4;
        const int cn = idx & 15;
        const int o = r * 20 + cn;
        float v = 0.0f;
        for (int w = 0; w < nw; ++w) v += red[w * (16 * 20) + o];
        if (Bv != nullptr) v += to_f<T>(Bv[n0 + cn]);
        Out[(long long)(m0 + r) * ldo + n0 + cn] = from_f<T>(v);
    }
}

// ---------------------------------------------------------------------------
// Fused attention: scaled QK^T + the additive bias list + softmax + PV +
// sigmoid gate, writing straight into [B, Q, H*C] layout.  One CTA per
// (batch, head, 16 queries).
//
// The captured shapes leave only B*H*ceil(Q/16) CTAs of work (16-96 here), so
// the binding constraint is instructions on the critical warp, not FLOPs.
// Consequences baked into this kernel:
//   * both matmuls go through wmma -- an FFMA loop over c costs ~3x the
//     instructions for the same result;
//   * every staging loop is warp-per-row / lane-over-column, so there is no
//     integer division by a runtime extent anywhere in the kernel and all
//     global accesses are contiguous in the fastest dimension;
//   * the bias list is folded into the softmax pass and held in registers
//     between the max and the exp, so scores are read once and no separate
//     block-wide pass (or barrier) exists for it;
//   * the gate is fetched together with Q so its latency overlaps K/V.
// Q/K/V are staged zero-padded up to a 16-multiple so c_hidden values like 24
// and 48 work.  Softmax accumulates in fp32 and P is rounded to the input dtype
// before PV, matching the baseline's `scores.to(value.dtype)`; the running
// max/sum make it correct for any key length.
// ---------------------------------------------------------------------------
template <typename T>
__device__ void attn_tile(const T* __restrict__ QG, const T* __restrict__ KV, T* __restrict__ O,
                          const BiasPack& bp, const BatchDims& bd, int Q, int Kn, int H, int C,
                          int CP, int D, int qgN, int kvN, int kc, int ldk, int gating,
                          float scale, int q0, int h, int b, char* smraw) {
    T* qsb = reinterpret_cast<T*>(smraw);                    // [ABQ][CP]
    T* kb = qsb + ABQ * CP;                                  // [kc][CP]   K as [j][d]
    T* vtb = kb + kc * CP;                                   // [CP][ldk]  V as [d][j]
    T* psb = vtb + CP * ldk;                                 // [ABQ][ldk] softmax probs
    float* scf = reinterpret_cast<float*>(psb + ABQ * ldk);   // [ABQ][ldk]
    float* oacc = scf + ABQ * ldk;                            // [ABQ][CP]
    float* ostg = oacc + ABQ * CP;                            // [ABQ][CP]
    float* gsb = ostg + ABQ * CP;                             // [ABQ][CP]
    float* mrow = gsb + ABQ * CP;
    float* lrow = mrow + ABQ;
    float* crow = lrow + ABQ;

    const int nthr = blockDim.x;
    const int nw = nthr >> 5;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nq = min(ABQ, Q - q0);
    const int KS = CP >> 4;
    const int DT = CP >> 4;
    const bool single = (Kn <= kc);

    long long boff[MAX_BIAS];
#pragma unroll
    for (int t = 0; t < MAX_BIAS; ++t) {
        long long off = 0;
        if (t < bp.n) {
            off = bp.b[t].sh * h;
            int rem = b;
            for (int d = bd.nd - 1; d >= 0; --d) {
                const int sz = bd.sizes[d];
                const int idx = rem % sz;
                rem /= sz;
                off += (long long)idx * bp.b[t].bs[d];
            }
        }
        boff[t] = off;
    }

    const long long qbase = (long long)b * Q * qgN + (long long)q0 * qgN + h * C;
    PDL_WAIT();
    for (int i = warp; i < ABQ; i += nw) {
        const T* p = QG + qbase + (long long)i * qgN;
        const bool ok = (i < nq);
        for (int d = lane; d < CP; d += 32) {
            float qv = 0.0f, g = 0.0f;
            if (ok && d < C) {
                qv = to_f<T>(p[d]) * scale;
                if (gating) g = to_f<T>(p[D + d]);
            }
            qsb[i * CP + d] = from_f<T>(qv);
            gsb[i * CP + d] = gating ? (1.0f / (1.0f + __expf(-g))) : 1.0f;
            if (!single) oacc[i * CP + d] = 0.0f;
        }
    }
    for (int i = tid; i < ABQ; i += nthr) {
        mrow[i] = -1e30f;
        lrow[i] = 0.0f;
    }

    for (int kc0 = 0; kc0 < Kn; kc0 += kc) {
        const int nk = min(kc, Kn - kc0);
        const int NT = (nk + 15) >> 4;
        const int N16 = NT << 4;
        const long long kbase = (long long)b * Kn * kvN + (long long)kc0 * kvN + h * C;

        __syncthreads();
        for (int j = warp; j < N16; j += nw) {
            const T* p = KV + kbase + (long long)j * kvN;
            const bool ok = (j < nk);
            for (int d = lane; d < CP; d += 32) {
                float kv = 0.0f, vv = 0.0f;
                if (ok && d < C) {
                    kv = to_f<T>(p[d]);
                    vv = to_f<T>(p[D + d]);
                }
                kb[j * CP + d] = from_f<T>(kv);
                vtb[d * ldk + j] = from_f<T>(vv);
            }
        }
        __syncthreads();

        for (int nt = warp; nt < NT; nt += nw) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int ks = 0; ks < KS; ++ks) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, qsb + (ks << 4), CP);
                wmma::load_matrix_sync(fb, kb + (nt << 4) * CP + (ks << 4), CP);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(scf + (nt << 4), acc, ldk, wmma::mem_row_major);
        }
        __syncthreads();

        // Bias + row softmax in one pass: scores are read once, biased in
        // registers, then exponentiated -- no separate block-wide bias pass.
        for (int i = warp; i < ABQ; i += nw) {
            if (i >= nq) {
                for (int j = lane; j < N16; j += 32) psb[i * ldk + j] = from_f<T>(0.0f);
                continue;
            }
            const T* brp[MAX_BIAS];
#pragma unroll
            for (int t = 0; t < MAX_BIAS; ++t)
                brp[t] = static_cast<const T*>(bp.b[t].ptr) + boff[t] +
                         bp.b[t].sq * (long long)(q0 + i);
            float v[KCMAX / 32];
            float mx = -1e30f;
            // N16 is warp-uniform; the <=32 form (every captured shape but one)
            // keeps one score per lane and skips the register array entirely.
            if (N16 <= 32) {
                float x = -1e30f;
                if (lane < nk) {
                    x = scf[i * ldk + lane];
#pragma unroll
                    for (int t = 0; t < MAX_BIAS; ++t) {
                        if (t < bp.n) {
                            x += to_f<T>(brp[t][bp.b[t].sk * (kc0 + lane)]);
                        }
                    }
                }
                v[0] = x;
                if (lane < N16) mx = x;
            } else {
#pragma unroll
                for (int u = 0; u < KCMAX / 32; ++u) {
                    const int j = lane + (u << 5);
                    float x = -1e30f;
                    if (j < nk) {
                        x = scf[i * ldk + j];
#pragma unroll
                        for (int t = 0; t < MAX_BIAS; ++t) {
                            if (t < bp.n) {
                                x += to_f<T>(brp[t][bp.b[t].sk * (kc0 + j)]);
                            }
                        }
                    }
                    v[u] = x;
                    if (j < N16) mx = fmaxf(mx, x);
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float mo = mrow[i];
            const float mn = fmaxf(mo, mx);
            float sp = 0.0f;
            if (N16 <= 32) {
                if (lane < N16) {
                    const float p = __expf(v[0] - mn);
                    psb[i * ldk + lane] = from_f<T>(p);
                    sp = p;
                }
            } else {
#pragma unroll
                for (int u = 0; u < KCMAX / 32; ++u) {
                    const int j = lane + (u << 5);
                    if (j < N16) {
                        const float p = __expf(v[u] - mn);
                        psb[i * ldk + j] = from_f<T>(p);
                        sp += p;
                    }
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, off);
            if (lane == 0) {
                crow[i] = __expf(mo - mn);
                mrow[i] = mn;
                lrow[i] = lrow[i] * crow[i] + sp;
            }
        }
        __syncthreads();

        float* dst = single ? oacc : ostg;
        for (int dt = warp; dt < DT; dt += nw) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int nt = 0; nt < NT; ++nt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, psb + (nt << 4), ldk);
                wmma::load_matrix_sync(fb, vtb + (dt << 4) * ldk + (nt << 4), ldk);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(dst + (dt << 4), acc, CP, wmma::mem_row_major);
        }
        if (!single) {
            __syncthreads();
            for (int i = warp; i < ABQ; i += nw)
                for (int d = lane; d < CP; d += 32)
                    oacc[i * CP + d] = oacc[i * CP + d] * crow[i] + ostg[i * CP + d];
        }
    }

    __syncthreads();
    for (int i = warp; i < nq; i += nw) {
        T* op = O + (long long)(b * Q + q0 + i) * D + h * C;
        const float inv = 1.0f / lrow[i];
        for (int d = lane; d < C; d += 32)
            op[d] = from_f<T>(oacc[i * CP + d] * gsb[i * CP + d] * inv);
    }
}
// ---------------------------------------------------------------------------
// Fully fused projection + attention.  One CTA owns (batch b, head h) and
// computes its own k/v/q/g, so nothing about q/k/v/g ever round-trips through
// global memory and two of the three launches disappear.
//
// This costs no redundant MACs because the head dimension partitions the
// *output columns* of every projection weight: linear_{k,v,q,g} all emit
// [.., H*C], so the CTA owning head h needs exactly columns [h*C, (h+1)*C) of
// each -- i.e. rows [h*C, (h+1)*C) of the row-major weight.  Only q_x / kv_x are
// re-read once per head, and at these sizes they are L2-resident.
//
// Traffic is minimal in both directions: the activation tiles are staged into
// shared memory once (each element read from global exactly once) and every
// wmma A-fragment comes from there, while the weights stream straight into
// B-fragments from global, each element also read exactly once.  Compare the
// separate `proj2`, which reads a 16x16 A tile per CTA and so re-reads the
// activations once per output column tile (measured: 19 MB of L1 traffic for a
// 4.8 MB problem on the 768-wide shape).
//
// The bias tile is staged during the same phase as the activations and is then
// consumed by `load_matrix_sync` on the score accumulator, so the additive bias
// costs zero instructions and its latency hides under the projection.
// ---------------------------------------------------------------------------
struct FArgs {
    int Q, Kn, H, C, CP, D, QP, KP, ldk;
    int CQ, CK, ldxq, ldxk, ldqx, ldkvx, ldwqg, ldwkv;
    int MTQ, MTK, CT, KSq, KSk, gating, ldc;
    float scale;
};

#define WSCLD 20

template <typename T>
__device__ void fattn_tile(const T* __restrict__ QX, const T* __restrict__ KVX,
                           const T* __restrict__ wqg, const T* __restrict__ bqg,
                           const T* __restrict__ wkv, T* __restrict__ O, const BiasPack& bp,
                           const BatchDims& bd, const FArgs& fa, int b, int h, char* smraw) {
    const int nthr = blockDim.x;
    const int nw = nthr >> 5;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int Q = fa.Q, Kn = fa.Kn, C = fa.C, CP = fa.CP, D = fa.D;
    const int QP = fa.QP, KP = fa.KP, ldk = fa.ldk;

    const int ldc = fa.ldc;
    T* kb = reinterpret_cast<T*>(smraw);            // [KP][ldc]  k
    T* vb = kb + KP * ldc;                          // [KP][ldc]  v (row major)
    T* qsb = vb + KP * ldc;                         // [QP][ldc]  q, pre-scaled
    float* gsb = reinterpret_cast<float*>(qsb + QP * ldc);  // [QP][ldc]  sigmoid gate
    float* scf = gsb + QP * ldc;                    // [QP][ldk]  bias, then scores
    char* ex = reinterpret_cast<char*>(scf + QP * ldk);
    // Projection-phase and attention-phase scratch never overlap in time.
    T* xkv = reinterpret_cast<T*>(ex);               // [KP][ldxk]
    T* xq = xkv + KP * fa.ldxk;                      // [QP][ldxq]
    float* wsc = reinterpret_cast<float*>(xq + QP * fa.ldxq);  // [nw][16*WSCLD]
    T* psb = reinterpret_cast<T*>(ex);                // [QP][ldk]
    float* oacc = reinterpret_cast<float*>(psb + QP * ldk);    // [QP][ldc]
    float* sr16 = oacc + QP * ldc;                    // [QP]  1/rowsum

    long long boff[MAX_BIAS];
#pragma unroll
    for (int t = 0; t < MAX_BIAS; ++t) {
        long long off = 0;
        if (t < bp.n) {
            off = bp.b[t].sh * h;
            int rem = b;
            for (int d = bd.nd - 1; d >= 0; --d) {
                const int sz = bd.sizes[d];
                off += (long long)(rem % sz) * bp.b[t].bs[d];
                rem /= sz;
            }
        }
        boff[t] = off;
    }

    PDL_WAIT();
    // ---- stage: bias tile + both activation tiles, all in flight together --
    for (int i = warp; i < QP; i += nw) {
        const T* brp[MAX_BIAS];
#pragma unroll
        for (int t = 0; t < MAX_BIAS; ++t)
            brp[t] = static_cast<const T*>(bp.b[t].ptr) + boff[t] + bp.b[t].sq * (long long)i;
        const bool ok = (i < Q);
        for (int j = lane; j < KP; j += 32) {
            float sv = 0.0f;
            if (ok && j < Kn) {
#pragma unroll
                for (int t = 0; t < MAX_BIAS; ++t)
                    if (t < bp.n) sv += to_f<T>(brp[t][bp.b[t].sk * j]);
            }
            scf[i * ldk + j] = sv;
        }
    }
    {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        const int CK = fa.CK, ldxk = fa.ldxk;
        for (int j = warp; j < KP; j += nw) {
            T* dst = xkv + j * ldxk;
            if (j < Kn) {
                const T* src = KVX + (long long)(b * Kn + j) * fa.ldkvx;
                for (int c = lane << 3; c < CK; c += 256)
                    *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(src + c);
            } else {
                for (int c = lane << 3; c < CK; c += 256)
                    *reinterpret_cast<uint4*>(dst + c) = z;
            }
        }
        const int CQ = fa.CQ, ldxq = fa.ldxq;
        for (int i = warp; i < QP; i += nw) {
            T* dst = xq + i * ldxq;
            if (i < Q) {
                const T* src = QX + (long long)(b * Q + i) * fa.ldqx;
                for (int c = lane << 3; c < CQ; c += 256)
                    *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(src + c);
            } else {
                for (int c = lane << 3; c < CQ; c += 256)
                    *reinterpret_cast<uint4*>(dst + c) = z;
            }
        }
    }
    __syncthreads();

    // ---- projection: 4 streams (k, v, q, g) of 16x16 output tiles ----------
    // Column tiles beyond C read the *next* head's weight rows; the values are
    // simply not stored (and the pad columns are zeroed), which is why the host
    // pads the concatenated weights with 16 zero rows -- head H-1's ragged tile
    // would otherwise read past the end of the tensor.
    {
        const int CT = fa.CT;
        const int nkv = fa.MTK * CT;
        const int nqg = fa.MTQ * CT;
        const int TT = 2 * nkv + 2 * nqg;
        float* ws = wsc + warp * (16 * WSCLD);
        for (int t = warp; t < TT; t += nw) {
            int stream, mt, nt, ks, ldx, ldw;
            const T* Xb;
            const T* Wb;
            if (t < 2 * nkv) {
                const int r = t - (t >= nkv ? nkv : 0);
                stream = (t >= nkv) ? 1 : 0;
                mt = r / CT;
                nt = r - mt * CT;
                ks = fa.KSk;
                ldx = fa.ldxk;
                ldw = fa.ldwkv;
                Xb = xkv + (long long)(mt << 4) * ldx;
                Wb = wkv + (long long)((stream ? D : 0) + h * C + (nt << 4)) * ldw;
            } else {
                const int t2 = t - 2 * nkv;
                const int r = t2 - (t2 >= nqg ? nqg : 0);
                stream = (t2 >= nqg) ? 3 : 2;
                mt = r / CT;
                nt = r - mt * CT;
                ks = fa.KSq;
                ldx = fa.ldxq;
                ldw = fa.ldwqg;
                Xb = xq + (long long)(mt << 4) * ldx;
                Wb = wqg + (long long)((stream == 3 ? D : 0) + h * C + (nt << 4)) * ldw;
            }
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            int k = 0;
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fx[UNROLL];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fw[UNROLL];
                for (; k + UNROLL <= ks; k += UNROLL) {
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        wmma::load_matrix_sync(fx[u], Xb + ((k + u) << 4), ldx);
                        wmma::load_matrix_sync(fw[u], Wb + ((k + u) << 4), ldw);
                    }
#pragma unroll
                    for (int u = 0; u < UNROLL; ++u) wmma::mma_sync(acc, fx[u], fw[u], acc);
                }
            }
            for (; k < ks; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fx;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fw;
                wmma::load_matrix_sync(fx, Xb + (k << 4), ldx);
                wmma::load_matrix_sync(fw, Wb + (k << 4), ldw);
                wmma::mma_sync(acc, fx, fw, acc);
            }
            wmma::store_matrix_sync(ws, acc, WSCLD, wmma::mem_row_major);
            const int m0 = mt << 4, n0 = nt << 4;
            const T* bqv = (stream == 2) ? bqg : nullptr;
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, c = e & 15;
                const int m = m0 + r, d = n0 + c;
                const bool live = (d < C);
                float v = live ? ws[r * WSCLD + c] : 0.0f;
                switch (stream) {
                    case 0: kb[m * ldc + d] = from_f<T>(v); break;
                    case 1: vb[m * ldc + d] = from_f<T>(v); break;
                    case 2:
                        if (live && bqv != nullptr) v += to_f<T>(bqv[h * C + d]);
                        qsb[m * ldc + d] = from_f<T>(v * fa.scale);
                        break;
                    default:
                        gsb[m * ldc + d] = fa.gating ? (1.0f / (1.0f + __expf(-v))) : 1.0f;
                        break;
                }
            }
        }
    }
    __syncthreads();

    // ---- attention: scores (bias-initialised accumulator) ------------------
    const int NT = (Kn + 15) >> 4;
    const int N16 = NT << 4;
    const int KS = CP >> 4;
    const int DT = CP >> 4;
    for (int t = warp; t < fa.MTQ * NT; t += nw) {
        const int mt = t / NT;
        const int nt = t - mt * NT;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::load_matrix_sync(acc, scf + (mt << 4) * ldk + (nt << 4), ldk, wmma::mem_row_major);
        for (int ks = 0; ks < KS; ++ks) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fq;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fk;
            wmma::load_matrix_sync(fq, qsb + (mt << 4) * ldc + (ks << 4), ldc);
            wmma::load_matrix_sync(fk, kb + (nt << 4) * ldc + (ks << 4), ldc);
            wmma::mma_sync(acc, fq, fk, acc);
        }
        wmma::store_matrix_sync(scf + (mt << 4) * ldk + (nt << 4), acc, ldk, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- softmax (one warp per query row, fp32) ----------------------------
    for (int i = warp; i < QP; i += nw) {
        if (i >= Q) {
            for (int j = lane; j < N16; j += 32) psb[i * ldk + j] = from_f<T>(0.0f);
            continue;
        }
        // N16 <= 32 (every captured shape but one) keeps a single score per lane
        // and skips the register array entirely.
        const float* sr = scf + i * ldk;
        T* pr = psb + i * ldk;
        float mx, sp = 0.0f;
        if (N16 <= 32) {
            const float x = (lane < Kn) ? sr[lane] : -1e30f;
            mx = x;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            if (lane < N16) {
                sp = (lane < Kn) ? __expf(x - mx) : 0.0f;
                pr[lane] = from_f<T>(sp);
            }
        } else {
            float v[FKMAX / 32];
            mx = -1e30f;
#pragma unroll
            for (int u = 0; u < FKMAX / 32; ++u) {
                const int j = lane + (u << 5);
                const float x = (j < Kn) ? sr[j] : -1e30f;
                v[u] = x;
                if (j < N16) mx = fmaxf(mx, x);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
#pragma unroll
            for (int u = 0; u < FKMAX / 32; ++u) {
                const int j = lane + (u << 5);
                if (j < N16) {
                    const float pv = (j < Kn) ? __expf(v[u] - mx) : 0.0f;
                    pr[j] = from_f<T>(pv);
                    sp += pv;
                }
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, off);
        if (lane == 0) sr16[i] = 1.0f / sp;
    }
    __syncthreads();

    // ---- PV, then gate + normalise straight into [.., Q, H*C] -------------
    for (int t = warp; t < fa.MTQ * DT; t += nw) {
        const int mt = t / DT;
        const int dt = t - mt * DT;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int nt = 0; nt < NT; ++nt) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fp;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> fv;
            wmma::load_matrix_sync(fp, psb + (mt << 4) * ldk + (nt << 4), ldk);
            wmma::load_matrix_sync(fv, vb + (nt << 4) * ldc + (dt << 4), ldc);
            wmma::mma_sync(acc, fp, fv, acc);
        }
        wmma::store_matrix_sync(oacc + (mt << 4) * ldc + (dt << 4), acc, ldc, wmma::mem_row_major);
    }
    __syncthreads();

    for (int i = warp; i < Q; i += nw) {
        T* op = O + (long long)(b * Q + i) * D + h * C;
        const float inv = sr16[i];
        for (int d = lane; d < C; d += 32)
            op[d] = from_f<T>(oacc[i * ldc + d] * gsb[i * ldc + d] * inv);
    }
}

template <typename T>
__global__ void __launch_bounds__(FTHREADS) fattn_kernel(
    const T* __restrict__ qx, const T* __restrict__ kvx, const T* __restrict__ wqg,
    const T* __restrict__ bqg, const T* __restrict__ wkv, T* __restrict__ ob, BiasPack bp,
    BatchDims bd, FArgs fa) {
    extern __shared__ char smraw[];
    const int h = blockIdx.x % fa.H;
    fattn_tile<T>(qx, kvx, wqg, bqg, wkv, ob, bp, bd, fa, blockIdx.x / fa.H, h, smraw);
    PDL_TRIGGER();
}

// ---------------------------------------------------------------------------
// Launch wrappers, one per phase.  A single cooperative kernel with grid syncs
// was measured and rejected: it forces one block shape on all three phases and
// the GEMMs lose more to that than the two saved launches gain (see
// ITERATIONS.md).  PDL closes the launch gap instead, without coupling the
// shapes.
// ---------------------------------------------------------------------------
struct ProjArgs {
    int ldqx, Mq, Kq, ldkvx, Mk, Kk, ldwqg, ldwkv, Nq, Nkv, nt0, ntiles0, ntiles1;
};
struct AttnArgs {
    int Q, Kn, H, C, CP, D, kc, ldk, gating, qt, nattn;
    float scale;
};
struct OutArgs {
    int ldwo, cq, ont, otiles, D, Mq;
};

template <typename T>
__device__ __forceinline__ void proj_phase(int t, const T* qx, const T* kvx, const T* wqg,
                                           const T* bqg, const T* wkv, T* qgo, T* kvo,
                                           const ProjArgs& pa, float* red) {
    if (t < pa.ntiles0) {
        const int mt = t / pa.nt0;
        gemm16_tile<T>(qx, pa.ldqx, wqg, pa.ldwqg, bqg, qgo, pa.Nq, pa.Mq, pa.Nq, pa.Kq, mt << 4,
                       (t - mt * pa.nt0) << 4, red);
    } else {
        const int t1 = t - pa.ntiles0;
        const int nt1 = (pa.Nkv + 15) >> 4;
        const int mt = t1 / nt1;
        gemm16_tile<T>(kvx, pa.ldkvx, wkv, pa.ldwkv, nullptr, kvo, pa.Nkv, pa.Mk, pa.Nkv, pa.Kk,
                       mt << 4, (t1 - mt * nt1) << 4, red);
    }
}

template <typename T>
__device__ __forceinline__ void attn_phase(int t, const T* qgo, const T* kvo, T* ob,
                                           const BiasPack& bp, const BatchDims& bd,
                                           const ProjArgs& pa, const AttnArgs& aa, char* sm) {
    const int qi = t % aa.qt;
    const int r = t / aa.qt;
    const int h = r % aa.H;
    attn_tile<T>(qgo, kvo, ob, bp, bd, aa.Q, aa.Kn, aa.H, aa.C, aa.CP, aa.D, pa.Nq, pa.Nkv, aa.kc,
                 aa.ldk, aa.gating, aa.scale, qi * ABQ, h, r / aa.H, sm);
}

template <typename T>
__device__ __forceinline__ void out_phase(int t, const T* ob, const T* wo, T* out,
                                          const OutArgs& oa, float* red) {
    const int mt = t / oa.ont;
    gemm16_tile<T>(ob, oa.D, wo, oa.ldwo, nullptr, out, oa.cq, oa.Mq, oa.cq, oa.D, mt << 4,
                   (t - mt * oa.ont) << 4, red);
}

template <typename T, int NTHR>
__global__ void __launch_bounds__(NTHR) proj2_kernel(
    const T* __restrict__ qx, const T* __restrict__ kvx, const T* __restrict__ wqg,
    const T* __restrict__ bqg, const T* __restrict__ wkv, T* __restrict__ qgo,
    T* __restrict__ kvo, ProjArgs pa) {
    extern __shared__ char smraw[];
    proj_phase<T>(blockIdx.x, qx, kvx, wqg, bqg, wkv, qgo, kvo, pa,
                  reinterpret_cast<float*>(smraw));
    PDL_TRIGGER();
}

template <typename T>
__global__ void __launch_bounds__(ATHREADS) attn_kernel(const T* __restrict__ qgo,
                                                       const T* __restrict__ kvo,
                                                       T* __restrict__ ob, BiasPack bp,
                                                       BatchDims bd, ProjArgs pa, AttnArgs aa) {
    extern __shared__ char smraw[];
    attn_phase<T>(blockIdx.x, qgo, kvo, ob, bp, bd, pa, aa, smraw);
    PDL_TRIGGER();
}

template <typename T, int NTHR>
__global__ void __launch_bounds__(NTHR) outproj_kernel(const T* __restrict__ ob,
                                                          const T* __restrict__ wo,
                                                          T* __restrict__ out, OutArgs oa) {
    extern __shared__ char smraw[];
    PDL_WAIT();
    out_phase<T>(blockIdx.x, ob, wo, out, oa, reinterpret_cast<float*>(smraw));
}

// ==== KERNELS END ====

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
static inline bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// Cached once per device: cudaDeviceGetAttribute on the per-call path costs
// about as much as a kernel launch.
static int optin_smem(int dev) {
    static int cache[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    if (dev < 0 || dev >= 16) {
        int v = 0;
        cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return v;
    }
    if (cache[dev] < 0)
        cudaDeviceGetAttribute(&cache[dev], cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return cache[dev];
}

// Fusing the projections into the attention CTA trades grid width for round
// trips: the CTA that owns head h must stream that head's whole weight slice,
// 4*C*(c_q + c_k) bytes, through one SM.  Measured on the captures: at
// C*(CQ+CK) = 8192 (c_q 128) the fused kernel is 8% faster end to end, at 18432
// (c_q 384) it is 14% slower and at 73728 (c_q 768) 30% slower, because there
// B*H is only 16 CTAs and the wider `proj2` grid streams the same weights from
// 12x as many SMs.  The threshold sits between the two measured families.
static bool fused_auto(int64_t Qn, int64_t Kn, int64_t H, int64_t C, int64_t CQ, int64_t CK) {
    return C * (CQ + CK) <= 12288;
}

static size_t smem_set[4] = {0, 0, 0, 0};
// Fill FArgs / the shared-memory budget for the fused path.  Returns 0 when the
// shapes fall outside it (the caller then uses the three-kernel path).
static inline size_t fused_plan(FArgs& fa, int64_t Qn, int64_t Kn, int64_t H, int64_t C,
                                int64_t D, int64_t CQ, int64_t CK, int nw) {
    if (Qn > FQMAX || Kn > FKMAX || (CQ & 15) || (CK & 15)) return 0;
    fa.Q = (int)Qn;
    fa.Kn = (int)Kn;
    fa.H = (int)H;
    fa.C = (int)C;
    fa.CP = (int)(((C + 15) / 16) * 16);
    fa.D = (int)D;
    fa.QP = (int)(((Qn + 15) / 16) * 16);
    fa.KP = (int)(((Kn + 15) / 16) * 16);
    fa.ldk = fa.KP + 8;
    fa.CQ = (int)CQ;
    fa.CK = (int)CK;
    fa.ldxq = (int)CQ + 8;
    fa.ldxk = (int)CK + 8;
    fa.MTQ = fa.QP >> 4;
    fa.MTK = fa.KP >> 4;
    fa.CT = fa.CP >> 4;
    fa.ldc = fa.CP + 8;
    fa.KSq = (int)(CQ >> 4);
    fa.KSk = (int)(CK >> 4);
    const size_t T2 = sizeof(__nv_bfloat16);
    const size_t keep = (size_t)(2 * fa.KP * fa.ldc + fa.QP * fa.ldc) * T2 +
                        (size_t)(fa.QP * fa.ldc + fa.QP * fa.ldk) * 4;
    const size_t proj = (size_t)(fa.KP * fa.ldxk + fa.QP * fa.ldxq) * T2 +
                        (size_t)nw * 16 * WSCLD * 4;
    const size_t attn = (size_t)(fa.QP * fa.ldk) * T2 + (size_t)(fa.QP * fa.ldc + fa.QP) * 4;
    return keep + (proj > attn ? proj : attn);
}

template <typename T>
static void launch_all(const at::Tensor& qx, const at::Tensor& kvx, const at::Tensor& wqg,
                       const c10::optional<at::Tensor>& bqg, const at::Tensor& wkv,
                       const at::Tensor& wo, at::Tensor& out, at::Tensor& ws, int64_t B,
                       int64_t Qn, int64_t Kn, int64_t H, int64_t C, int64_t D, int64_t Nq,
                       int64_t Nkv, int64_t cq, bool gating, const BiasPack& bp,
                       const BatchDims& bd, int slot, bool use_fused) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const T* pqx = static_cast<const T*>(qx.const_data_ptr());
    const T* pkvx = static_cast<const T*>(kvx.const_data_ptr());
    const T* pwqg = static_cast<const T*>(wqg.const_data_ptr());
    const T* pbqg = bqg.has_value() ? static_cast<const T*>(bqg->const_data_ptr()) : nullptr;
    const T* pwkv = static_cast<const T*>(wkv.const_data_ptr());
    const T* pwo = static_cast<const T*>(wo.const_data_ptr());
    T* pws = static_cast<T*>(ws.data_ptr());
    T* pout = static_cast<T*>(out.data_ptr());
    const int64_t Mq = B * Qn;
    const int64_t Mk = B * Kn;
    T* pqg = pws;
    T* pkv = pqg + Mq * Nq;
    T* pob = pkv + Mk * Nkv;

    ProjArgs pa;
    pa.ldqx = (int)qx.stride(-2);
    pa.Mq = (int)Mq;
    pa.Kq = (int)qx.size(-1);
    pa.ldkvx = (int)kvx.stride(-2);
    pa.Mk = (int)Mk;
    pa.Kk = (int)kvx.size(-1);
    pa.ldwqg = (int)wqg.stride(0);
    pa.ldwkv = (int)wkv.stride(0);
    pa.Nq = (int)Nq;
    pa.Nkv = (int)Nkv;
    pa.nt0 = (int)((Nq + 15) >> 4);
    pa.ntiles0 = (int)(((Mq + 15) >> 4) * pa.nt0);
    pa.ntiles1 = (int)(((Mk + 15) >> 4) * ((Nkv + 15) >> 4));

    AttnArgs aa;
    aa.Q = (int)Qn;
    aa.Kn = (int)Kn;
    aa.H = (int)H;
    aa.C = (int)C;
    aa.CP = (int)(((C + 15) / 16) * 16);
    aa.D = (int)D;
    aa.kc = (int)std::min<int64_t>(((Kn + 15) / 16) * 16, KCMAX);
    aa.ldk = aa.kc + 8;
    aa.gating = gating ? 1 : 0;
    aa.qt = (int)((Qn + ABQ - 1) / ABQ);
    aa.nattn = (int)(aa.qt * H * B);
    aa.scale = (float)(1.0 / std::sqrt((double)C));

    const size_t attn_smem =
        (size_t)(ABQ * aa.CP + aa.kc * aa.CP + aa.CP * aa.ldk + ABQ * aa.ldk) * sizeof(T) +
        (size_t)(ABQ * aa.ldk + 3 * ABQ * aa.CP + 3 * ABQ) * sizeof(float);
    OutArgs oa;
    oa.ldwo = (int)wo.stride(0);
    oa.cq = (int)cq;
    oa.ont = (int)((cq + 15) >> 4);
    oa.otiles = (int)(((Mq + 15) >> 4) * oa.ont);
    oa.D = (int)D;
    oa.Mq = (int)Mq;

    // K/16 steps available to split across warps, per GEMM.
    const int ktp = (int)std::min<int64_t>(pa.Kq, pa.Kk) >> 4;
    const int pthr = (ktp >= KTWIDE) ? GTHREADS_W : GTHREADS;
    const int othr = ((oa.D >> 4) >= KTWIDE) ? GTHREADS_W : GTHREADS;
    const size_t psmem = (size_t)(pthr / 32) * 16 * 20 * sizeof(float);
    const size_t osmem = (size_t)(othr / 32) * 16 * 20 * sizeof(float);

    // Fused single-kernel path: fold both projection GEMMs into the attention
    // CTA.  Selected on the host so a shape that prefers the wider-grid `proj2`
    // can keep it (see `use_fused`).
    if (use_fused) {
        FArgs fa;
        const size_t fsm = fused_plan(fa, Qn, Kn, H, C, D, qx.size(-1), kvx.size(-1),
                                      FTHREADS / 32);
        fa.ldqx = (int)qx.stride(-2);
        fa.ldkvx = (int)kvx.stride(-2);
        fa.ldwqg = (int)wqg.stride(0);
        fa.ldwkv = (int)wkv.stride(0);
        fa.gating = gating ? 1 : 0;
        fa.scale = (float)(1.0 / std::sqrt((double)C));
        const int fslot = slot + 2;
        if (fsm > 49152 && fsm > smem_set[fslot]) {
            cudaFuncSetAttribute((const void*)fattn_kernel<T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fsm);
            smem_set[fslot] = fsm;
        }
        cudaLaunchAttribute fpdl;
        fpdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        fpdl.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t f1 = {};
        f1.gridDim = dim3((unsigned)(B * H));
        f1.blockDim = dim3(FTHREADS);
        f1.dynamicSmemBytes = fsm;
        f1.stream = stream;
        cudaError_t fe = cudaLaunchKernelEx(&f1, fattn_kernel<T>, pqx, pkvx, pwqg, pbqg, pwkv,
                                           pob, bp, bd, fa);
        if (fe == cudaSuccess) {
            cudaLaunchConfig_t f2 = {};
            f2.gridDim = dim3((unsigned)oa.otiles);
            f2.blockDim = dim3(othr);
            f2.dynamicSmemBytes = osmem;
            f2.stream = stream;
            f2.attrs = &fpdl;
            f2.numAttrs = 1;
            fe = (othr == GTHREADS_W)
                     ? cudaLaunchKernelEx(&f2, outproj_kernel<T, GTHREADS_W>, pob, pwo, pout, oa)
                     : cudaLaunchKernelEx(&f2, outproj_kernel<T, GTHREADS>, pob, pwo, pout, oa);
        }
        if (fe != cudaSuccess) {
            cudaGetLastError();
            fattn_kernel<T><<<(unsigned)(B * H), FTHREADS, fsm, stream>>>(
                pqx, pkvx, pwqg, pbqg, pwkv, pob, bp, bd, fa);
            if (othr == GTHREADS_W)
                outproj_kernel<T, GTHREADS_W><<<oa.otiles, othr, osmem, stream>>>(pob, pwo, pout,
                                                                                  oa);
            else
                outproj_kernel<T, GTHREADS><<<oa.otiles, othr, osmem, stream>>>(pob, pwo, pout,
                                                                                oa);
        }
        return;
    }


    // Opt in past the 48 KB default only when needed and only up to what is
    // needed: asking for the device maximum shrinks the L1 side of the cache.
    if (attn_smem > 49152 && attn_smem > smem_set[slot]) {
        cudaFuncSetAttribute((const void*)attn_kernel<T>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)attn_smem);
        smem_set[slot] = attn_smem;
    }

    // Chain the three launches with PDL so each consumer grid is resident before
    // its producer drains; fall back to plain launches if the driver refuses.
    cudaLaunchAttribute pdl;
    pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t c1 = {};
    c1.gridDim = dim3((unsigned)(pa.ntiles0 + pa.ntiles1));
    c1.blockDim = dim3(pthr);
    c1.dynamicSmemBytes = psmem;
    c1.stream = stream;
    cudaError_t e = (pthr == GTHREADS_W)
                        ? cudaLaunchKernelEx(&c1, proj2_kernel<T, GTHREADS_W>, pqx, pkvx, pwqg,
                                             pbqg, pwkv, pqg, pkv, pa)
                        : cudaLaunchKernelEx(&c1, proj2_kernel<T, GTHREADS>, pqx, pkvx, pwqg,
                                             pbqg, pwkv, pqg, pkv, pa);
    if (e == cudaSuccess) {
        cudaLaunchConfig_t c2 = {};
        c2.gridDim = dim3((unsigned)aa.nattn);
        c2.blockDim = dim3(ATHREADS);
        c2.dynamicSmemBytes = attn_smem;
        c2.stream = stream;
        c2.attrs = &pdl;
        c2.numAttrs = 1;
        e = cudaLaunchKernelEx(&c2, attn_kernel<T>, pqg, pkv, pob, bp, bd, pa, aa);
    }
    if (e == cudaSuccess) {
        cudaLaunchConfig_t c3 = {};
        c3.gridDim = dim3((unsigned)oa.otiles);
        c3.blockDim = dim3(othr);
        c3.dynamicSmemBytes = osmem;
        c3.stream = stream;
        c3.attrs = &pdl;
        c3.numAttrs = 1;
        e = (othr == GTHREADS_W)
                ? cudaLaunchKernelEx(&c3, outproj_kernel<T, GTHREADS_W>, pob, pwo, pout, oa)
                : cudaLaunchKernelEx(&c3, outproj_kernel<T, GTHREADS>, pob, pwo, pout, oa);
    }
    if (e != cudaSuccess) {
        cudaGetLastError();
        const int ng = pa.ntiles0 + pa.ntiles1;
        if (pthr == GTHREADS_W)
            proj2_kernel<T, GTHREADS_W><<<ng, pthr, psmem, stream>>>(pqx, pkvx, pwqg, pbqg, pwkv,
                                                                     pqg, pkv, pa);
        else
            proj2_kernel<T, GTHREADS><<<ng, pthr, psmem, stream>>>(pqx, pkvx, pwqg, pbqg, pwkv,
                                                                   pqg, pkv, pa);
        attn_kernel<T><<<aa.nattn, ATHREADS, attn_smem, stream>>>(pqg, pkv, pob, bp, bd, pa, aa);
        if (othr == GTHREADS_W)
            outproj_kernel<T, GTHREADS_W><<<oa.otiles, othr, osmem, stream>>>(pob, pwo, pout, oa);
        else
            outproj_kernel<T, GTHREADS><<<oa.otiles, othr, osmem, stream>>>(pob, pwo, pout, oa);
    }
}

std::optional<at::Tensor> of3_forward(const at::Tensor& q_x, const at::Tensor& kv_x,
                                     const std::vector<at::Tensor>& biases,
                                     const at::Tensor& wqg, const c10::optional<at::Tensor>& bqg,
                                     const at::Tensor& wkv, const at::Tensor& wo, int64_t H,
                                     int64_t C, bool gating) {
    const int64_t nd = q_x.dim();
    if (nd < 2 || nd > MAX_BDIM + 5 || kv_x.dim() != nd) return std::nullopt;
    if (!q_x.is_cuda() || (int)biases.size() > MAX_BIAS) return std::nullopt;
    const auto dt = q_x.scalar_type();
    if (dt != at::kBFloat16 && dt != at::kHalf) return std::nullopt;
    if (kv_x.scalar_type() != dt || wqg.scalar_type() != dt || wkv.scalar_type() != dt ||
        wo.scalar_type() != dt)
        return std::nullopt;
    if (bqg.has_value() && bqg->scalar_type() != dt) return std::nullopt;
    if (C <= 0 || C > MAX_C || H <= 0) return std::nullopt;

    const int64_t D = H * C;
    const int64_t Nq = gating ? 2 * D : D;
    const int64_t Nkv = 2 * D;
    const int64_t cq = wo.size(0);
    // ``>=``: the fused path's ragged column tile reads up to 15 rows past the
    // last head, so the cached concatenation carries 16 zero pad rows.
    if (wqg.size(0) < Nq || wkv.size(0) < Nkv || wo.size(1) != D) return std::nullopt;

    // Batch dims must agree; collapse the size-1 ones away.
    BatchDims bd;
    bd.nd = 0;
    int64_t B = 1;
    int64_t full_batch[MAX_BDIM + 3];
    int64_t nfb = 0;
    if (nd - 2 > MAX_BDIM + 3) return std::nullopt;
    for (int64_t d = 0; d < nd - 2; ++d) {
        const int64_t s = q_x.size(d);
        if (kv_x.size(d) != s) return std::nullopt;
        full_batch[nfb++] = s;
        if (s == 1) continue;
        if (bd.nd >= MAX_BDIM) return std::nullopt;
        bd.sizes[bd.nd++] = (int)s;
        B *= s;
    }
    const int64_t Qn = q_x.size(nd - 2);
    const int64_t Kn = kv_x.size(nd - 2);
    if (Qn <= 0 || Kn <= 0) return std::nullopt;

    // Bias broadcast strides over (collapsed batch..., H, Q, K).
    const int64_t nsd = nfb + 3;
    BiasPack bp;
    bp.n = (int)biases.size();
    for (int t = 0; t < bp.n; ++t) {
        const at::Tensor& bt = biases[t];
        if (bt.scalar_type() != dt || !bt.is_cuda()) return std::nullopt;
        if (bt.dim() > nsd) return std::nullopt;
        const int64_t shift = nsd - bt.dim();
        int64_t st[MAX_BDIM + 3];
        for (int64_t a = 0; a < nsd; ++a) {
            const int64_t ba = a - shift;
            int64_t logical;
            if (a < nfb)
                logical = full_batch[a];
            else if (a == nsd - 3)
                logical = H;
            else if (a == nsd - 2)
                logical = Qn;
            else
                logical = Kn;
            if (ba < 0) {
                st[a] = 0;
            } else {
                const int64_t bsz = bt.size(ba);
                if (bsz == 1)
                    st[a] = 0;
                else if (bsz == logical)
                    st[a] = bt.stride(ba);
                else
                    return std::nullopt;
            }
        }
        int w = 0;
        for (int64_t a = 0; a < nfb; ++a) {
            if (full_batch[a] == 1) continue;
            bp.b[t].bs[w++] = st[a];
        }
        for (; w < MAX_BDIM; ++w) bp.b[t].bs[w] = 0;
        bp.b[t].sh = st[nsd - 3];
        bp.b[t].sq = st[nsd - 2];
        bp.b[t].sk = st[nsd - 1];
        bp.b[t].ptr = bt.const_data_ptr();
    }

    at::Tensor qc = q_x.contiguous();
    at::Tensor kc = kv_x.contiguous();
    const int64_t cqin = qc.size(nd - 1);
    const int64_t ckin = kc.size(nd - 1);
    if (wqg.size(1) != cqin || wkv.size(1) != ckin) return std::nullopt;
    if ((cqin & 7) || (ckin & 7) || (D & 7) || (Nq & 7) || (Nkv & 7) || (cq & 7))
        return std::nullopt;
    if (wqg.stride(1) != 1 || wkv.stride(1) != 1 || wo.stride(1) != 1) return std::nullopt;
    if ((wqg.stride(0) & 7) || (wkv.stride(0) & 7) || (wo.stride(0) & 7)) return std::nullopt;
    if (!aligned16(qc.const_data_ptr()) || !aligned16(kc.const_data_ptr()) ||
        !aligned16(wqg.const_data_ptr()) || !aligned16(wkv.const_data_ptr()) ||
        !aligned16(wo.const_data_ptr()))
        return std::nullopt;

    // Shared-memory budget for the fused attention kernel; bail to the generic
    // path rather than fail the launch if this device cannot host the tile.
    {
        const int64_t CP = ((C + 15) / 16) * 16;
        int64_t kcc = ((Kn + 15) / 16) * 16;
        if (kcc > KCMAX) kcc = KCMAX;
        const int64_t ldk = kcc + 8;
        int64_t need = (ABQ * CP + kcc * CP + CP * ldk + ABQ * ldk) * 2 +
                       (ABQ * ldk + 3 * ABQ * CP + 3 * ABQ) * 4;
        const int64_t gneed = (int64_t)(GTHREADS / 32) * 16 * 20 * 4;
        if (gneed > need) need = gneed;
        if (need > (int64_t)optin_smem((int)qc.device().index())) return std::nullopt;
    }

    // Fused-path eligibility: one CTA per (batch, head) must hold a whole
    // q-tile, all K rows and its weight slices' outputs in shared memory.
    bool use_fused = false;
    if (fused_auto(Qn, Kn, H, C, cqin, ckin)) {
        FArgs fp;
        const size_t fsm = fused_plan(fp, Qn, Kn, H, C, D, cqin, ckin, FTHREADS / 32);
        use_fused = (fsm != 0 && fsm <= (size_t)optin_smem((int)qc.device().index()));
    }

    const at::cuda::OptionalCUDAGuard guard(qc.device());
    int64_t oshape[MAX_BDIM + 5];
    for (int64_t d = 0; d < nd; ++d) oshape[d] = q_x.size(d);
    oshape[nd - 1] = cq;
    auto opts = qc.options();
    at::Tensor out = at::empty(at::IntArrayRef(oshape, (size_t)nd), opts);
    const int64_t Mq = B * Qn, Mk = B * Kn;
    at::Tensor ws = at::empty({Mq * Nq + Mk * Nkv + Mq * D}, opts);

    if (dt == at::kBFloat16)
        launch_all<__nv_bfloat16>(qc, kc, wqg, bqg, wkv, wo, out, ws, B, Qn, Kn, H, C, D, Nq,
                                  Nkv, cq, gating, bp, bd, 0, use_fused);
    else
        launch_all<__half>(qc, kc, wqg, bqg, wkv, wo, out, ws, B, Qn, Kn, H, C, D, Nq, Nkv, cq,
                           gating, bp, bd, 1, use_fused);
    return out;
}
"""

_CPP_DECL = r"""
#include <torch/extension.h>
#include <optional>
#include <vector>

std::optional<at::Tensor> of3_forward(const at::Tensor& q_x, const at::Tensor& kv_x,
                                     const std::vector<at::Tensor>& biases,
                                     const at::Tensor& wqg, const c10::optional<at::Tensor>& bqg,
                                     const at::Tensor& wkv, const at::Tensor& wo, int64_t H,
                                     int64_t C, bool gating);
"""

_EXT = None


def _build_ext():
    from torch.utils import cpp_extension

    tag = hashlib.sha1((_CPP_DECL + _CUDA_SRC).encode()).hexdigest()[:12]
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        return cpp_extension.load_inline(
            name=f"of3_attn_{tag}",
            cpp_sources=_CPP_DECL,
            cuda_sources=_CUDA_SRC,
            functions=["of3_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _EXT = _build_ext()
except Exception:  # pragma: no cover - fall back to the generic torch path
    _EXT = None

_NO_BIAS: list[torch.Tensor] = []


def _lsd_post_hook(module, incompatible_keys):  # noqa: ARG001
    """Drop the concatenated-weight cache when weights are (re)loaded."""
    module.__dict__["_fused"] = None


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

        self.__dict__["_fused"] = None
        # Cache invalidation: pointer-level changes (``.to()``, ``.half()``,
        # ``load_state_dict``) go through these hooks; in-place value changes are
        # caught by the ``_version`` sum checked on every forward.
        self.register_load_state_dict_post_hook(_lsd_post_hook)

    def _apply(self, *args, **kwargs):
        self.__dict__["_fused"] = None
        return super()._apply(*args, **kwargs)

    # -- fused-path weight cache -------------------------------------------
    def _params(self):
        """Exactly six entries (padded) so the per-forward version check is a
        fixed, unrolled expression rather than a Python loop."""
        wo = self.linear_o.weight
        return (self.linear_q.weight, self.linear_k.weight, self.linear_v.weight, wo,
                self.linear_g.weight if self.linear_g is not None else wo,
                self.linear_q.bias if self.linear_q.bias is not None else wo)

    def _build_fused(self):
        """Concatenate the projection weights once and cache them."""
        d = self.__dict__
        if _EXT is None or self.c_k != self.c_v:
            d["_fused"] = False  # structural: never eligible
            return False
        d["_fused"] = None
        ps = self._params()
        wq, wk, wv, wo = ps[0], ps[1], ps[2], ps[3]  # ps[4]/ps[5] may be padding
        wg = self.linear_g.weight if self.linear_g is not None else None
        bq = self.linear_q.bias
        if not wq.is_cuda or wq.dtype not in (torch.bfloat16, torch.float16):
            return None
        with torch.no_grad():
            # 16 zero pad rows: the fused kernel's ragged c_hidden column tile
            # (c_hidden 24 -> a 32-wide tile) reads up to 15 weight rows past the
            # last head.  Padding here keeps that read in bounds for free
            # instead of costing a predicate in the inner loop.
            wqg = torch.cat([wq, wg], 0) if wg is not None else wq
            wqg = F.pad(wqg, (0, 0, 0, 16)).contiguous()
            wkv = F.pad(torch.cat([wk, wv], 0), (0, 0, 0, 16)).contiguous()
            bqg = None
            if bq is not None:
                pad = wqg.shape[0] - bq.shape[0]
                bqg = (F.pad(bq, (0, pad)) if pad > 0 else bq).contiguous()
        vsum = (ps[0]._version + ps[1]._version + ps[2]._version + ps[3]._version
                + ps[4]._version + ps[5]._version)
        fused = (wqg, bqg, wkv, wo.contiguous(), self.no_heads, self.c_hidden, self.gating,
                 ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], vsum)
        d["_fused"] = fused
        return fused

    # -- generic (baseline-equivalent) path ---------------------------------
    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def _generic(self, q_x, kv_x, biases):
        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)
        return self._wrap_up(o, q_x)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = _NO_BIAS
        f = self.__dict__["_fused"]
        if f.__class__ is tuple:
            if (f[7]._version + f[8]._version + f[9]._version + f[10]._version + f[11]._version
                    + f[12]._version) != f[13]:
                f = self._build_fused()
        elif f is None:
            f = self._build_fused()
        if f.__class__ is tuple:
            out = _EXT.of3_forward(q_x, kv_x, biases, f[0], f[1], f[2], f[3],
                                   f[4], f[5], f[6])
            if out is not None:
                return out
        return self._generic(q_x, kv_x, biases)
