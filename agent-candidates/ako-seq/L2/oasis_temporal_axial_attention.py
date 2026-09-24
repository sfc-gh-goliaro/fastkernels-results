"""Oasis temporal axial attention -- one GEMM, one fused kernel, one GEMM.

The reference forward is 28 CUDA kernels and 199 ATen dispatches for an operator
whose entire arithmetic is two 1024-wide GEMMs and 2304 attention problems of
2-6 tokens.  Even standing on the L1 winners (which already delete the 69us
mis-tiled cuDNN kernel this shape otherwise lands on) it is 22 kernels, 127
dispatches and 231-293us of *device* time on B200 against ~80us of actual kernel
time: two thirds of the window is the GPU idling while Python and the dispatcher
catch up.  Nothing here
re-tunes the attention math -- ``candidate/L1/dense_attention.py`` already owns
these exact shapes -- the whole job is to delete the glue around it:

* **No layout materialization.**  ``to_qkv`` emits ``(b,t,h,w,3*heads*d)`` with
  ``(heads,d)`` already innermost and contiguous, so the reference's
  ``reshape -> permute(0,2,3,4,1,5) -> reshape -> transpose(1,2)`` chain is a
  pure index permutation.  Instead of three ``contiguous()`` copies, the fused
  kernel is handed the packed buffer plus the six strides that name the same
  elements, and it *writes* its result straight into a
  ``(b,t,h,w,heads*d)``-shaped buffer, so the output permute+reshape pair (two
  more copies) disappears as well and ``to_out`` sees a contiguous tensor.

* **No rotary glue.**  ``rotate_queries_or_keys`` is called twice per forward
  and each call redoes ``arange``, the freqs table, ``cos``, ``sin``,
  ``rotate_half``, two muls and an add -- ~20 launches for a table of at most
  6x64 elements.  Positions are ``0..T-1`` and ``freqs`` is a frozen parameter,
  so the (cos, sin) table is static per T: it is built once through the L1
  rotary kernel, interleaved to ``(T, d/2, 2)`` so that the frequency pairs one
  lane needs are one vector load -- the same access shape as its Q load -- and
  cached.  The rotation itself is folded into the fused kernel's Q/K tile load,
  so RoPE costs zero launches and zero table work per call.

* **No dispatch glue either.**  What is left -- qkv GEMM, the fused kernel, out
  GEMM -- is driven from C++ by a per-shape plan (``_HOST_CU``) in a single
  pybind11 call, with both GEMMs on cublasLt plans whose descriptor, layouts,
  algo and workspace are built once.  Every launch parameter of the fused kernel
  except its data pointers is a ``#define`` resolved at plan-build time
  (strides, grid, scale, sequence length, causality), so there are only five
  specializations to compile and the launch is one ``cuLaunchKernel``.

Per call, over the five captured shapes: 28 -> 3 CUDA kernels, 199 -> 2 ATen
dispatches, 394-490us -> 32-39us of host time, 352-449us -> 19.5-25.5us of
device time (13.9-18.6x on the harness).

The remaining window is *not* host time and not launch count -- neither a CUDA
graph nor a 17us host-time reduction moves it (see ITERATIONS.md) -- it is the
cost of streaming ``to_qkv``'s and ``to_out``'s 8 MiB of weights past the SMs,
which is where an operator this small should end up.

Everything degrades rather than fails: cublasLt declining falls back to
``at::mm_out``/``at::addmm``, an unbuildable extension to the same three launches
driven from Python, and a missing nvrtc (or a shape the fused kernel cannot
express -- partial rotary, ``d`` not a multiple of 4 or with ``d/4`` not a power
of two, fp32, T > 8) to ``_forward_ref``, the reference composition of the L1
winners.  All four paths are checked in ``dev/fallbacks.py``.

Numerics.  The harness casts every parameter to the input dtype, so ``freqs`` is
fp16 and the reference's RoPE is *fp16* arithmetic: ``h(h(a*c) - h(b*s))`` per
pair, with cos/sin themselves rounded to fp16.  The kernel reproduces that
rounding chain exactly (verified bit-identical against
``rotate_queries_or_keys`` for T=2..6), and the attention math -- fp32
accumulation, ``ex2`` softmax with log2(e) folded into the scale, ``rcp``
normalization -- is the L1 winner's.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding

_LOG2_E = 1.4426950408889634

# A/B switch for the two host GEMM paths (cublasLt plan vs at::mm_out/at::addmm).
_WANT_LT = not os.environ.get("OASIS_L2_NO_LT")

# Head-dimension slice one lane owns, in 32-bit words: a lane loads NW 32-bit
# words, i.e. DPL = 2*NW elements = NW whole rotary pairs, and D/DPL lanes
# cooperate on one row.  The cos/sin fetch is then exactly the shape of the Q
# fetch (one DPL-element row segment), which is what makes RoPE free.
#
# NW=2 (64-bit loads, 16 lanes per row) beats NW=4 (128-bit, 8 lanes) here even
# though it is the narrower load: this fork holds a whole (cos, sin) row per
# position on top of Q/K/V, so halving DPL halves the K/V register footprint.
# The T=4 window drops 23.5us -> 21.5us (reproduced 3x); T=2/3/5 unchanged, T=6
# marginally better.  See the sweep table in ITERATIONS.md.
_NW = 2
_WARPS = 4
_MINB = 2
# D / DPL lanes cooperate on one row, so D/DPL must be a power of two <= 32.
_LANE_GROUPS = (1, 2, 4, 8, 16, 32)
_MAX_SEQ = 8            # the grouped regime: every loop unrolls, S in registers
_DTYPES = (torch.float16, torch.bfloat16)


# ---------------------------------------------------------------------------
# The fused kernel.  Everything except the two data pointers is a #define, so
# the launch is one struct.pack_into plus one cuLaunchKernel.
#
# Lane assignment (from the L1 grouped kernel): lane = slot*LG + cl, so LG lanes
# each own a 16-byte slice of the head dimension for one (batch, h, w, head)
# problem and the 32 lanes of a warp cover PPW = 32/LG independent problems.
# All 3*S loads are issued before the first use -- one HBM round trip per
# problem instead of a dependent chain -- and QK^T's sum across the LG lanes of
# a slot is log2(LG) butterfly shuffles.  P*V needs no communication at all.
#
# RoPE rides along inside that: a lane's DPL elements are DPL/2 whole rotary
# pairs whose frequency indices are contiguous, so the (cos, sin) it needs is
# one more 16-byte load from the cached table and the rotation is register-local.
# ---------------------------------------------------------------------------
_CU_SRC = r"""
typedef unsigned short u16;
typedef unsigned int   u32;
typedef long long      i64;

__device__ __forceinline__ float bits2f(u32 x) {
    float f; asm("mov.b32 %0, %1;" : "=f"(f) : "r"(x)); return f;
}
__device__ __forceinline__ float h2f(u16 h) {
    float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ u16 f2h(float f) {
    u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ __forceinline__ float b2f(u16 h) { return bits2f(((u32)h) << 16); }
__device__ __forceinline__ u16 f2b(float f) {
    u16 h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ __forceinline__ float ex2(float x) {
    float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rcp(float x) {
    float r; asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float bfly(float v, int m) {
    float r;
    asm("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(r) : "f"(v), "r"(m));
    return r;
}
#if FP16
#define TOF(x)   h2f(x)
#define FROMF(x) f2h(x)
#else
#define TOF(x)   b2f(x)
#define FROMF(x) f2b(x)
#endif

#define DPL (2 * NW)
#define LG  (D / DPL)
#define PPW (32 / LG)

#if NW == 1
#define LDV(p, w) asm("ld.global.nc.b32 %0, [%1];" : "=r"(w[0]) : "l"(p))
#define STV(p, w) asm("st.global.b32 [%0], %1;" :: "l"(p), "r"(w[0]) : "memory")
#elif NW == 2
#define LDV(p, w) asm("ld.global.nc.v2.b32 {%0,%1}, [%2];" \
                      : "=r"(w[0]), "=r"(w[1]) : "l"(p))
#define STV(p, w) asm("st.global.v2.b32 [%0], {%1,%2};" \
                      :: "l"(p), "r"(w[0]), "r"(w[1]) : "memory")
#else
#define LDV(p, w) asm("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];" \
                      : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p))
#define STV(p, w) asm("st.global.v4.b32 [%0], {%1,%2,%3,%4};" \
                      :: "l"(p), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) \
                       : "memory")
#endif

/* One rotary pair, reproducing the reference's fp16 rounding chain exactly:
     out0 = h(h(a*c) - h(b*s))      out1 = h(h(b*c) + h(a*s))
   The inline-asm converts are also what stops ptxas from contracting the
   multiply and the add into an FMA, which would skip a rounding. */
#define ROPE_PAIR(dst0, dst1, a, b, c, s)                                 \
    do {                                                                  \
        float _p0 = TOF(FROMF((a) * (c))) - TOF(FROMF((b) * (s)));         \
        float _p1 = TOF(FROMF((b) * (c))) + TOF(FROMF((a) * (s)));         \
        dst0 = TOF(FROMF(_p0));                                           \
        dst1 = TOF(FROMF(_p1));                                           \
    } while (0)

extern "C" __global__ __launch_bounds__(WARPS * 32, MINB)
void rope_attn(const u16* __restrict__ QKV, u16* __restrict__ O,
               const u16* __restrict__ CS)
{
    const int lane = (int)(threadIdx.x & 31u);
    const int wid  = (int)(threadIdx.x >> 5);
    const int slot = lane / LG;
    const int cl   = lane - slot * LG;

    /* Out-of-range slots must not exit: they still take part in the butterfly
       shuffles below.  Clamp to problem 0 and drop only the store. */
    const int p0   = ((int)blockIdx.x * WARPS + wid) * PPW + slot;
    const bool live = p0 < NPROB;
    const int p    = live ? p0 : 0;
    /* (batch, hw, head) from one flat index; NH and HW are compile-time. */
    const int bi   = p / (HW * NH);
    const int r    = p - bi * (HW * NH);
    const int n    = r / NH;
    const int hd   = r - n * NH;

    const u16* base = QKV + (i64)bi * SBAT + (i64)n * SB + (i64)hd * SH
                    + cl * DPL;
    const u16* qp = base;
    const u16* kp = base + KOFF;
    const u16* vp = base + VOFF;
    const u16* cp = CS + cl * DPL;

    u32 qw[S][NW], kw[S][NW], vw[S][NW], cw[S][NW];
    #pragma unroll
    for (int i = 0; i < S; ++i) LDV(qp + (i64)i * SS, qw[i]);
    #pragma unroll
    for (int i = 0; i < S; ++i) LDV(kp + (i64)i * SS, kw[i]);
    #pragma unroll
    for (int i = 0; i < S; ++i) LDV(vp + (i64)i * SS, vw[i]);
    #pragma unroll
    for (int i = 0; i < S; ++i) LDV(cp + i * D, cw[i]);

    /* K is rotated on the way into registers; V is not rotated at all. */
    float kf[S][DPL], vf[S][DPL];
    #pragma unroll
    for (int j = 0; j < S; ++j) {
        #pragma unroll
        for (int t = 0; t < NW; ++t) {
            const float c  = TOF((u16)(cw[j][t] & 0xffffu));
            const float sn = TOF((u16)(cw[j][t] >> 16));
            const float a  = TOF((u16)(kw[j][t] & 0xffffu));
            const float b  = TOF((u16)(kw[j][t] >> 16));
            ROPE_PAIR(kf[j][2 * t], kf[j][2 * t + 1], a, b, c, sn);
            vf[j][2 * t]     = TOF((u16)(vw[j][t] & 0xffffu));
            vf[j][2 * t + 1] = TOF((u16)(vw[j][t] >> 16));
        }
    }

    u16* op = O + (i64)bi * SOBAT + (i64)n * SOB + (i64)hd * SOH + cl * DPL;
    #pragma unroll
    for (int i = 0; i < S; ++i) {
        float qf[DPL];
        #pragma unroll
        for (int t = 0; t < NW; ++t) {
            const float c  = TOF((u16)(cw[i][t] & 0xffffu));
            const float sn = TOF((u16)(cw[i][t] >> 16));
            const float a  = TOF((u16)(qw[i][t] & 0xffffu));
            const float b  = TOF((u16)(qw[i][t] >> 16));
            ROPE_PAIR(qf[2 * t], qf[2 * t + 1], a, b, c, sn);
        }
        /* Scores for this query row.  With CAUSAL the j > i half is dropped at
           compile time (both loop bounds are constants after unrolling). */
        float pr[S];
        float mx = -3.0e38f;
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            if (CAUSAL && j > i) { pr[j] = 0.0f; continue; }
            float a = 0.0f;
            #pragma unroll
            for (int t = 0; t < DPL; ++t) a += qf[t] * kf[j][t];
            #pragma unroll
            for (int m = LG >> 1; m > 0; m >>= 1) a += bfly(a, m);
            a *= SCALE;                 /* scale already folded with log2(e) */
            pr[j] = a;
            mx = a > mx ? a : mx;
        }
        float l = 0.0f;
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            if (CAUSAL && j > i) continue;
            pr[j] = ex2(pr[j] - mx);
            l += pr[j];
        }
        float acc[DPL];
        #pragma unroll
        for (int t = 0; t < DPL; ++t) acc[t] = 0.0f;
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            if (CAUSAL && j > i) continue;
            #pragma unroll
            for (int t = 0; t < DPL; ++t) acc[t] += pr[j] * vf[j][t];
        }
        const float rr = rcp(l);
        u32 ow[NW];
        #pragma unroll
        for (int t = 0; t < NW; ++t)
            ow[t] = (u32)FROMF(acc[2 * t] * rr)
                  | ((u32)FROMF(acc[2 * t + 1] * rr) << 16);
        if (live) { u16* o = op + (i64)i * SOS; STV(o, ow); }
    }
}
"""


# ---------------------------------------------------------------------------
# Host-side orchestration, in C++.
#
# With the glue gone the per-call cost is *entirely* the two cuBLAS calls: on
# this box `at::mm_out` costs 7.4us and `at::addmm` 9.4us of host time (measured
# in `dev/cpp_probe.py`), against 0.3us for the fused kernel's `cuLaunchKernel`.
# The harness times a window the host has to keep filled, so that host time is
# what is actually scored -- and ~10us of it is Python and dispatcher overhead
# plus cuBLAS re-deriving its heuristic.  Both go away here:
#
#   * one pybind11 call replaces `torch.matmul` + `cuLaunchKernel` + `F.linear`;
#   * each GEMM is a cublasLt plan (descriptor, layouts, algo and workspace)
#     built once per shape, so a call is one `cublasLtMatmul` -- 5.8us instead
#     of 7.4us for the qkv GEMM and 6.3us instead of 9.4us for the biased
#     output GEMM, with the algo the heuristic would have picked anyway (verified
#     bit-identical to `at::mm_out` / `at::addmm`).
#
# `at::mm_out`/`at::addmm` remain as the fallback for anything cublasLt declines.
# ---------------------------------------------------------------------------
_HOST_CPP = r"""
#include <torch/extension.h>
#include <vector>

/* The .cu side speaks only ATen + void*, so nvcc never sees pybind11 and this
   translation unit never needs `Plan` to be a complete type. */
void* oasis_plan_create(at::Tensor wq, at::Tensor wo, at::Tensor bo,
                        at::Tensor qkv, at::Tensor scr, at::Tensor cs,
                        int64_t fn, int64_t grid, int64_t block,
                        std::vector<int64_t> yshape, bool want_lt);
void oasis_plan_destroy(void* p);
at::Tensor oasis_plan_run(void* p, const at::Tensor& x);
bool oasis_plan_uses_lt(void* p);

static pybind11::capsule make_plan(at::Tensor wq, at::Tensor wo, at::Tensor bo,
                                   at::Tensor qkv, at::Tensor scr, at::Tensor cs,
                                   int64_t fn, int64_t grid, int64_t block,
                                   std::vector<int64_t> yshape, bool want_lt) {
  void* p = oasis_plan_create(wq, wo, bo, qkv, scr, cs, fn, grid, block,
                              std::move(yshape), want_lt);
  return pybind11::capsule(p, [](void* q) { oasis_plan_destroy(q); });
}

static at::Tensor run(pybind11::capsule cap, const at::Tensor& x) {
  return oasis_plan_run(cap.get_pointer(), x);
}

static bool uses_lt(pybind11::capsule cap) {
  return oasis_plan_uses_lt(cap.get_pointer());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_plan", &make_plan);
  m.def("run", &run);
  m.def("uses_lt", &uses_lt);
}
"""

_HOST_CU = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda.h>
#include <vector>

namespace {

/* One cublasLt plan: everything the call needs except the data pointers.
   Row-major C(m,n) = A(n,k) * B(m,k)^T is expressed as the column-major
   C^T(n,m) = A^T * B that cuBLAS wants natively, so `A` is the weight exactly
   as PyTorch stores it (n rows of k) and no transpose tensor is needed. */
struct LtPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  size_t ws = 0;
  at::Tensor wsbuf;
  bool ok = false;

  ~LtPlan() {
    if (op) cublasLtMatmulDescDestroy(op);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
  }

  bool build(cublasLtHandle_t h, int m, int n, int k, const void* bias,
             at::ScalarType st, const at::TensorOptions& opts) {
    cudaDataType_t dt = st == at::kHalf ? CUDA_R_16F : CUDA_R_16BF;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
      return false;
    cublasOperation_t t = CUBLAS_OP_T, n_ = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n_, sizeof(n_));
    if (cublasLtMatrixLayoutCreate(&la, dt, k, n, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&lb, dt, k, m, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&lc, dt, n, m, n) != CUBLAS_STATUS_SUCCESS)
      return false;
    if (bias) {
      cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
      /* the bias tensor never moves, so bind it once */
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                     &bias, sizeof(bias));
    }
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return false;
    size_t maxws = 32u * 1024u * 1024u;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxws, sizeof(maxws));
    cublasLtMatmulHeuristicResult_t res[1];
    int found = 0;
    bool got = cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, lc, pref, 1,
                                              res, &found) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulPreferenceDestroy(pref);
    if (!got || found <= 0) return false;
    algo = res[0].algo;
    ws = res[0].workspaceSize;
    if (ws) wsbuf = at::empty({(int64_t)ws}, opts.dtype(at::kByte));
    ok = true;
    return true;
  }

  cublasStatus_t run(cublasLtHandle_t h, const void* A, const void* B, void* C,
                     cudaStream_t stream) const {
    const float alpha = 1.f, beta = 0.f;
    return cublasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, C, lc, C, lc,
                          &algo, ws ? wsbuf.data_ptr() : nullptr, ws, stream);
  }
};

cublasLtHandle_t lt_handle() {
  /* Our own handle: `getCurrentCUDABlasHandle()` re-binds the stream and does
     workspace bookkeeping on every call, and cublasLtMatmul takes the stream
     as an argument anyway. */
  static cublasLtHandle_t h = nullptr;
  if (!h && cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
  return h;
}

}  // namespace

struct Plan {
  at::Tensor wq, wo, bo, qkv, scr, cs;   /* ownership */
  at::Tensor qkv2d, scr2d, wqt, wot;     /* views for the ATen fallback */
  std::vector<int64_t> yshape;
  at::TensorOptions opts;
  int64_t M = 0, K = 0, N = 0, Dm = 0;
  CUfunction fn = nullptr;
  unsigned gx = 0, bx = 0;
  void* p_qkv = nullptr;
  void* p_scr = nullptr;
  void* p_cs = nullptr;
  void* kargs[3] = {nullptr, nullptr, nullptr};
  bool use_lt = false;
  LtPlan lt_qkv, lt_out;
};

void* oasis_plan_create(at::Tensor wq, at::Tensor wo, at::Tensor bo,
                        at::Tensor qkv, at::Tensor scr, at::Tensor cs,
                        int64_t fn, int64_t grid, int64_t block,
                        std::vector<int64_t> yshape, bool want_lt) {
  Plan* p = new Plan();
  p->wq = wq; p->wo = wo; p->bo = bo; p->qkv = qkv; p->scr = scr; p->cs = cs;
  p->K = wq.size(1);
  p->N = wq.size(0);
  p->Dm = wo.size(0);
  p->M = scr.numel() / wo.size(1);
  p->qkv2d = qkv.view({p->M, p->N});
  p->scr2d = scr.view({p->M, wo.size(1)});
  p->wqt = wq.t();
  p->wot = wo.t();
  p->yshape = std::move(yshape);
  p->opts = scr.options();
  p->fn = (CUfunction)(uintptr_t)fn;
  p->gx = (unsigned)grid;
  p->bx = (unsigned)block;
  p->p_qkv = qkv.data_ptr();
  p->p_scr = scr.data_ptr();
  p->p_cs = cs.data_ptr();
  p->kargs[0] = &p->p_qkv;
  p->kargs[1] = &p->p_scr;
  p->kargs[2] = &p->p_cs;
  cublasLtHandle_t h = want_lt ? lt_handle() : nullptr;
  if (h && bo.defined() && bo.dim() == 1 && bo.size(0) == p->Dm) {
    bool a = p->lt_qkv.build(h, (int)p->M, (int)p->N, (int)p->K, nullptr,
                             wq.scalar_type(), p->opts);
    bool b = p->lt_out.build(h, (int)p->M, (int)p->Dm, (int)wo.size(1),
                             bo.data_ptr(), wo.scalar_type(), p->opts);
    p->use_lt = a && b;
  }
  return (void*)p;
}

void oasis_plan_destroy(void* q) { delete static_cast<Plan*>(q); }

bool oasis_plan_uses_lt(void* q) { return static_cast<Plan*>(q)->use_lt; }

at::Tensor oasis_plan_run(void* q, const at::Tensor& x) {
  Plan* p = static_cast<Plan*>(q);
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor y;
  if (p->use_lt) {
    cublasLtHandle_t h = lt_handle();
    /* qkv = x @ wq^T */
    p->lt_qkv.run(h, p->wq.const_data_ptr(), x.const_data_ptr(), p->p_qkv, stream);
    cuLaunchKernel(p->fn, p->gx, 1, 1, p->bx, 1, 1, 0, stream, p->kargs, nullptr);
    y = at::empty({p->M, p->Dm}, p->opts);
    /* y = attn @ wo^T + bo */
    p->lt_out.run(h, p->wo.const_data_ptr(), p->p_scr, y.data_ptr(), stream);
  } else {
    at::mm_out(p->qkv2d, x.view({p->M, p->K}), p->wqt);
    cuLaunchKernel(p->fn, p->gx, 1, 1, p->bx, 1, 1, 0, stream, p->kargs, nullptr);
    y = at::addmm(p->bo, p->scr2d, p->wot);
  }
  return y.view(p->yshape);
}
"""


def _host_ext():
    """Build (once) the host-side orchestration extension, or return None."""
    global _HOST
    if _HOST is not None:
        return _HOST or None
    _HOST = False
    try:
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.md5((_HOST_CPP + _HOST_CU).encode()).hexdigest()[:10]
        _HOST = load_inline(
            name=f"oasis_l2_host_{tag}",
            cpp_sources=_HOST_CPP,
            cuda_sources=_HOST_CU,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcublasLt", "-lcuda"],
            verbose=False,
        )
    except Exception:
        _HOST = False
    return _HOST or None


_HOST: object = None


# ---------------------------------------------------------------------------
# nvrtc bring-up + a pre-built cuLaunchKernel argument block.  Kept local (and
# every failure mode memoized) so the fused path can only ever be faster or
# absent: anything that goes wrong here returns None and the reference
# composition of the L1 winners runs instead.
# ---------------------------------------------------------------------------
_CU_STATE: dict | bool | None = None


def _cu_init():
    global _CU_STATE
    if _CU_STATE is not None:
        return _CU_STATE
    _CU_STATE = False
    try:
        if not torch.cuda.is_available():
            return _CU_STATE
        from cuda.bindings import driver as cud, nvrtc
        # torch's lazy init is what creates/binds the primary context; without a
        # current context cuModuleLoadData fails with INVALID_CONTEXT.
        torch.cuda.init()
        torch.empty(1, device="cuda")
        err, ctx = cud.cuCtxGetCurrent()
        if err != cud.CUresult.CUDA_SUCCESS or int(ctx) == 0:
            return _CU_STATE
        major, minor = torch.cuda.get_device_capability()
        if major < 8:                    # bf16 cvt / shfl.sync need sm_80+
            return _CU_STATE
        _CU_STATE = {"cud": cud, "nvrtc": nvrtc, "mods": [],
                     "arch": f"--gpu-architecture=sm_{major}{minor}".encode()}
    except Exception:
        _CU_STATE = False
    return _CU_STATE


_FN_CACHE: dict = {}


def _cu_function(defines):
    """Compile (once per define set) and return a CUfunction, or None."""
    key = tuple(sorted(defines.items()))
    if key in _FN_CACHE:
        return _FN_CACHE[key]
    _FN_CACHE[key] = None                # memoize failure by default
    st = _cu_init()
    if not st:
        return None
    try:
        cud, nvrtc = st["cud"], st["nvrtc"]
        src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + _CU_SRC
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"rope_attn.cu", 0, [], [])
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return None
        opts = [st["arch"], b"--std=c++17"]
        if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return None
        size = nvrtc.nvrtcGetCUBINSize(prog)[1]
        cubin = bytearray(size)
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, mod = cud.cuModuleLoadData(bytes(cubin))
        if err != cud.CUresult.CUDA_SUCCESS:
            return None
        err, fn = cud.cuModuleGetFunction(mod, b"rope_attn")
        if err != cud.CUresult.CUDA_SUCCESS:
            return None
        st["mods"].append(mod)           # CUfunction does not own the module
        _FN_CACHE[key] = fn
    except Exception:
        _FN_CACHE[key] = None
    return _FN_CACHE[key]


def _f32(x: float) -> str:
    """Round to fp32 and render as a C literal that parses back to it."""
    v = struct.unpack("<f", struct.pack("<f", x))[0]
    return repr(v) + "f"


class OasisTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
        *,
        is_causal: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")
        self._plans: dict = {}
        self._cs: dict = {}

    def _apply(self, *args, **kwargs):
        # A plan captures parameter tensors and a scratch buffer bound to one
        # device/dtype; `.to()` / `.half()` replace those, so drop the cache.
        self._plans.clear()
        self._cs.clear()
        return super()._apply(*args, **kwargs)

    # -- reference composition of the L1 winners (fallback) -----------------
    def _forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)

        q = q.reshape(bsz * height * width, self.heads, time, -1)
        k = k.reshape(bsz * height * width, self.heads, time, -1)
        v = v.reshape(bsz * height * width, self.heads, time, -1)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, causal=self.is_causal)
        out = out.reshape(bsz, height, width, time, self.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

    # -- (cos, sin) table, interleaved to (T, d/2, 2) and cached ------------
    def _cs_table(self, time: int, ref: torch.Tensor) -> torch.Tensor:
        """Laid out so a lane's DPL/2 frequency pairs are one DPL-element load at
        ``i*d + cl*DPL`` -- the same access shape as its Q load.  Built through
        the L1 rotary kernel, so the fp16 table is bit-identical to the
        reference's."""
        cs = self._cs.get(time)
        if cs is not None:
            return cs
        freqs = self.rotary_emb.freqs
        pos = torch.arange(time, device=ref.device, dtype=ref.dtype)
        tab = self.rotary_emb.forward(pos, freqs, seq_len=time)
        # freqs[..., 2i] == freqs[..., 2i+1]; keep one entry per rotary pair.
        cs = torch.stack((tab.cos()[:, 0::2], tab.sin()[:, 0::2]), dim=-1)
        cs = cs.reshape(time, -1).contiguous()
        self._cs[time] = cs
        return cs

    # -- plan build (once per shape) ---------------------------------------
    def _build_plan(self, x: torch.Tensor):
        heads = self.heads
        wq, wo = self.to_qkv.weight, self.to_out.weight
        bo = self.to_out.bias
        if x.dim() != 5 or x.dtype not in _DTYPES or not x.is_cuda:
            return None
        if wq.dtype is not x.dtype or wo.dtype is not x.dtype:
            return None
        if bo is not None and bo.dtype is not x.dtype:
            return None
        bsz, time, height, width, _ = x.shape
        d = wq.shape[0] // (3 * heads)
        if wq.shape[0] != 3 * heads * d or d % (2 * _NW) or d // (2 * _NW) not in _LANE_GROUPS:
            return None
        if not (0 < time <= _MAX_SEQ) or bsz * height * width * heads <= 0:
            return None
        freqs = self.rotary_emb.freqs
        # Partial rotary (rot_dim < d) needs the reference's cat; decline it.
        if freqs.dim() != 1 or 2 * freqs.numel() != d or freqs.dtype is not x.dtype:
            return None
        st = _cu_init()
        if not st:
            return None

        hw = height * width
        qkv_row = 3 * heads * d
        defines = {
            "S": time, "D": d, "NH": heads, "HW": hw,
            "NPROB": bsz * hw * heads,
            "CAUSAL": int(bool(self.is_causal)),
            "FP16": int(x.dtype is torch.float16),
            "NW": _NW, "WARPS": _WARPS, "MINB": _MINB,
            "SBAT": time * hw * qkv_row, "SS": hw * qkv_row,
            "SB": qkv_row, "SH": d,
            "KOFF": heads * d, "VOFF": 2 * heads * d,
            "SOBAT": time * hw * heads * d, "SOS": hw * heads * d,
            "SOB": heads * d, "SOH": d,
            "SCALE": _f32(d ** -0.5 * _LOG2_E),
        }
        fn = _cu_function(defines)
        if fn is None:
            return None

        # A qkv buffer whose (h, w) axes cannot be flattened into one stride, or
        # a non-contiguous GEMM result, would invalidate the strides above.
        probe = F.linear(x, wq)
        if not probe.is_contiguous() or probe.shape[-1] != qkv_row:
            return None
        del probe

        cs = self._cs_table(time, x)
        lg = d // (2 * _NW)
        per_cta = (32 // lg) * _WARPS
        grid = -(-(bsz * hw * heads) // per_cta)
        block = _WARPS * 32
        # Both intermediates are written whole and consumed within the same call,
        # so one buffer per shape is enough -- and a fixed address means the
        # kernel's whole argument block is packed once here instead of per call.
        qkv = torch.empty((bsz, time, height, width, qkv_row),
                          dtype=x.dtype, device=x.device)
        scratch = torch.empty((bsz, time, height, width, heads * d),
                              dtype=x.dtype, device=x.device)
        # `w.t()` is a view of the parameter, which is never written; caching it
        # keeps the GEMM a single `torch.matmul` with no per-call transpose.
        wqt = wq.t()

        cud = st["cud"]
        buf = ctypes.create_string_buffer(24)
        csize = ctypes.c_size_t(24)
        extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2,
                                     ctypes.addressof(csize), 0)
        struct.pack_into("<3Q", buf, 0, qkv.data_ptr(), scratch.data_ptr(),
                         cs.data_ptr())
        extra_addr = ctypes.addressof(extra)
        launch = cud.cuLaunchKernel
        raw_stream = torch._C._cuda_getCurrentRawStream
        dev_index = x.device.index if x.device.index is not None else 0

        # Preferred path: one pybind11 call does GEMM -> kernel -> GEMM, with
        # both GEMMs on cached cublasLt plans (see `_HOST_CU`).
        # `bo is None` (a bias-free `to_out`, which this operator's contract
        # never builds) would reach the host plan as an undefined tensor; keep it
        # on the Python path, which handles it natively.
        host = _host_ext() if bo is not None else None
        if host is not None:
            try:
                cap = host.make_plan(wq, wo, bo, qkv, scratch, cs, int(fn),
                                     grid, block,
                                     list(x.shape[:-1]) + [wo.shape[0]], _WANT_LT)

                def run(x, _run=host.run, _cap=cap):
                    return _run(_cap, x)

                return (x.dtype, run, [cap, cs, qkv, scratch, wq, wo, bo])
            except Exception:
                pass

        # Fallback: the same three launches driven from Python.
        def run(x, _mm=torch.matmul, _wqt=wqt, _qkv=qkv, _launch=launch,
                _fn=fn, _grid=grid, _block=block, _extra=extra_addr,
                _stream=raw_stream, _di=dev_index, _lin=F.linear,
                _wo=wo, _bo=bo, _scr=scratch):
            _mm(x, _wqt, out=_qkv)
            _launch(_fn, _grid, 1, 1, _block, 1, 1, 0, _stream(_di), 0, _extra)
            return _lin(_scr, _wo, _bo)

        # Keep the objects the launch block points into alive for the plan's
        # lifetime (ctypes buffers are not owned by the driver call).
        return (x.dtype, run, [buf, csize, extra, cs, qkv, scratch, wqt])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plans.get(x.shape)
        if plan is not None:
            # ``False`` memoizes "this shape is not for the fused path", so a
            # declined shape never re-runs the (nvrtc-compiling) plan builder.
            if plan is not False and x.dtype is plan[0] and x.is_contiguous():
                return plan[1](x)
            return self._forward_ref(x)
        try:
            plan = self._build_plan(x) if x.is_contiguous() else None
        except Exception:
            plan = None
        self._plans[x.shape] = plan if plan is not None else False
        return plan[1](x) if plan is not None else self._forward_ref(x)
