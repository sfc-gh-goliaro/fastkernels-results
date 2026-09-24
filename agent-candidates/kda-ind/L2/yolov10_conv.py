"""YOLOv10 Conv-BN-Act building block, collapsed into one fused CUDA kernel.

The benched block is launch-bound, not compute-bound: on four of the five cases
``fastkernels bench --target yolov10_conv`` selects, ~90% of the measured wall
latency is CPU launch/dispatch (see ``docs/draft.md`` section 3). The baseline pays
four kernel launches -- ``conv2d``, ``batch_norm_calc_invstd``,
``batch_norm_transform_input``, ``silu`` -- for as little as 4.8 us of real GPU work.

This candidate computes

    out = silu(scale[co] * conv(x, W)[n, co, ho, wo] + bias[co])

with one CUDA kernel launched by one pybind11 call. BatchNorm is folded into a
per-output-channel *fp32* ``(scale, bias)`` epilogue rather than into the fp16
weight: that keeps the fold out of fp16's range, costs nothing (the epilogue reads
two fp32 vectors of length ``Co`` either way), and serves the unfused and the
post-``fuse()`` states with the same kernel (then ``scale == 1``,
``bias == conv.bias``).

Tiers, most aggressive first; every tier has a proven fallback beneath it:

``SIMT``    one fused kernel, fp32 accumulate, implicit GEMM ``C[Co,P] = W[Co,K].X[K,P]``.
``FOLDED``  lazily BN-folded weight/bias, ``F.conv2d`` + activation (2 kernels).
``EXACT``   the literal baseline expression, for anything with different semantics
            (training, autograd, unrecognized activation, hooks on skipped submodules).

Everything weight-derived is built on the first ``forward``, never in ``__init__``:
the bench shares weights *after* construction (``_prepare_module`` ->
``_sanitize_float_params`` -> ``load_state_dict``), so precomputing in ``__init__``
would capture garbage. The cached plan is invalidated by a load-state-dict post
hook, an ``_apply`` override, ``train()``, ``fuse()``, and reassignment of
``act``/``conv``/``bn``.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU

__all__ = ["autopad", "_fuse_conv_bn", "fuse_module", "YOLOConv"]

TIER_EXACT = "EXACT"
TIER_FOLDED = "FOLDED"
TIER_SIMT = "SIMT"

_FOLDABLE_DTYPES = (torch.float16, torch.bfloat16)

_ACT_IDENTITY = 0
_ACT_SILU = 1

# ---------------------------------------------------------------------------
# The fused kernel.
#
# Convolution as an implicit GEMM ``C[Co, P] = W[Co, K] . X[K, P]`` with
# ``K = Ci*kh*kw`` and ``P = N*Ho*Wo``: the contiguous weight ``[Co, Ci, kh, kw]``
# *is* ``[Co, K]`` row-major, so no repacking is needed, and ``P`` (whose innermost
# axis is ``wo``, contiguous in NCHW) carries the coalescing.
#
# One thread owns ``CT`` output channels at one ``p``: consecutive threads walk ``p``,
# so a warp's loads of ``x`` are contiguous, and the ``CT`` accumulators both give
# instruction-level parallelism and cut the ``x`` traffic by ``CT`` (each loaded
# activation feeds ``CT`` FMAs). ``CT`` is chosen at launch to fill the machine:
# small problems want more threads, case B (P = 102400) wants the reuse.
# ---------------------------------------------------------------------------
_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace {

template <int CT, int PT, int KHC, int KWC>
__global__ void fused_conv_bn_act_simt(
    const __half* __restrict__ xp, const __half* __restrict__ wp,
    const float* __restrict__ scalep, const float* __restrict__ biasp,
    __half* __restrict__ outp,
    int Ci, int H, int W, int Co, int Ho, int Wo,
    int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW,
    long long sN, long long sC, long long sH, long long sW,
    int P, int Pimg, int K, int act) {
  // The CT weight rows this block needs, converted to fp32 once. Every lane reads
  // the same element, so these are shared-memory broadcasts.
  extern __shared__ float wsh[];
  const int co_base = blockIdx.y * CT;
#pragma unroll
  for (int t = 0; t < CT; ++t) {
    const int co = co_base + t;
    const long long src = (long long)co * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
      wsh[t * K + k] = (co < Co) ? __half2float(wp[src + k]) : 0.0f;
  }
  __syncthreads();

  const int tpb = blockDim.x;
  const int p0 = blockIdx.x * (tpb * PT) + threadIdx.x;

  int rq[PT], h0q[PT], w0q[PT];
  const __half* xnq[PT];
  bool live[PT];
#pragma unroll
  for (int q = 0; q < PT; ++q) {
    const int p = p0 + q * tpb;
    live[q] = (p < P);
    const int pp = live[q] ? p : 0;
    const int n = pp / Pimg;
    const int r = pp - n * Pimg;
    const int ho = r / Wo;
    const int wo = r - ho * Wo;
    rq[q] = r;
    h0q[q] = ho * SH - PH;
    w0q[q] = wo * SW - PW;
    xnq[q] = xp + (long long)n * sN;
  }

  float acc[PT][CT];
#pragma unroll
  for (int q = 0; q < PT; ++q)
#pragma unroll
    for (int t = 0; t < CT; ++t) acc[q][t] = 0.0f;

  const int kh = KHC ? KHC : KH;
  const int kw = KWC ? KWC : KW;
  const int khw = kh * kw;
  // Enough loads in flight to cover an L2 hit; capped so the unrolled body does not
  // spill (PT loads are already in flight per iteration).
  constexpr int UN = (PT >= 4) ? 4 : ((PT == 2) ? 8 : 16);

  for (int i = 0; i < kh; ++i) {
    for (int j = 0; j < kw; ++j) {
      const float* wr = wsh + (i * kw + j);
      const __half* xr[PT];
      bool ok[PT];
#pragma unroll
      for (int q = 0; q < PT; ++q) {
        const int h = h0q[q] + i * DH;
        const int w = w0q[q] + j * DW;
        ok[q] = live[q] && h >= 0 && h < H && w >= 0 && w < W;  // implicit zero pad
        // Only form the address when the tap is inside the tensor: forming a pointer
        // outside the allocation is undefined behaviour even if the load is predicated.
        xr[q] = ok[q] ? (xnq[q] + (long long)h * sH + (long long)w * sW) : xnq[q];
      }
      // Index from an invariant base rather than walking pointers: `xr[q][c*sC]` keeps
      // every load in the unrolled body independent, while `xc[q] += sC` chains them
      // through the address register and destroys the memory-level parallelism this
      // loop lives on (measured: case A 1.72x -> 0.91x, case B 1.27x -> 1.21x).
#pragma unroll UN
      for (int c = 0; c < Ci; ++c) {
        const long long off = (long long)c * sC;
        float xv[PT];
#pragma unroll
        for (int q = 0; q < PT; ++q) xv[q] = ok[q] ? __half2float(xr[q][off]) : 0.0f;
        const int koff = c * khw;
#pragma unroll
        for (int t = 0; t < CT; ++t) {
          const float wv = wr[t * K + koff];
#pragma unroll
          for (int q = 0; q < PT; ++q) acc[q][t] = fmaf(xv[q], wv, acc[q][t]);
        }
      }
    }
  }

  // fp32 BN affine + activation, one round to fp16 on store.
#pragma unroll
  for (int q = 0; q < PT; ++q) {
    if (live[q]) {
      const int p = p0 + q * tpb;
      const int n = p / Pimg;
      const long long obase = (long long)n * ((long long)Co * Pimg) + rq[q];
#pragma unroll
      for (int t = 0; t < CT; ++t) {
        const int co = co_base + t;
        if (co < Co) {
          float v = fmaf(acc[q][t], scalep[co], biasp[co]);
          if (act == 1) v = v / (1.0f + __expf(-v));   // SiLU
          outp[obase + (long long)co * Pimg] = __float2half(v);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 1x1, unit stride, no padding/dilation: the implicit GEMM's B matrix *is* x, so
// output position p is contiguous in memory and both the loads and the stores can be
// 8B/16B wide. That is the whole point -- the general kernel's 2B-per-lane loads are
// what makes it latency-bound (ncu: long_scoreboard 33.8 on case D), and three of the
// five benched cases land here.
// ---------------------------------------------------------------------------
template <int CT, int VEC>
__global__ void fused_conv1x1_bn_act_vec(
    const __half* __restrict__ xp, const __half* __restrict__ wp,
    const float* __restrict__ scalep, const float* __restrict__ biasp,
    __half* __restrict__ outp,
    int Ci, int Co, long long sN, long long sC,
    int P, int Pimg, int act) {
  extern __shared__ float wsh[];
  const int co_base = blockIdx.y * CT;
#pragma unroll
  for (int t = 0; t < CT; ++t) {
    const int co = co_base + t;
    const long long src = (long long)co * Ci;
    for (int k = threadIdx.x; k < Ci; k += blockDim.x)
      wsh[t * Ci + k] = (co < Co) ? __half2float(wp[src + k]) : 0.0f;
  }
  __syncthreads();

  const int p0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  if (p0 >= P) return;
  // Pimg % VEC == 0 is a launch precondition, so a VEC-run that starts inside an
  // image stays inside it and never runs past P.
  const int n = p0 / Pimg;
  const int r = p0 - n * Pimg;
  const __half* xr = xp + (long long)n * sN + r;

  float acc[VEC][CT];
#pragma unroll
  for (int q = 0; q < VEC; ++q)
#pragma unroll
    for (int t = 0; t < CT; ++t) acc[q][t] = 0.0f;

  // VEC=2 makes one warp-wide load cover exactly one 128 B line with no waste, which
  // is what the miss count (and therefore the MSHR-bound time) actually depends on.
  // Deep unroll: vectorizing shrinks the thread count, so the misses in flight per SM
  // (warps/SM x unroll) must be made up per thread or the kernel goes right back to
  // being MSHR-starved -- measured 40.9 us at unroll 4 vs 36.8 us for the scalar
  // kernel on case D, purely from losing memory-level parallelism.
  constexpr int UNV = 16;
#pragma unroll UNV
  for (int c = 0; c < Ci; ++c) {
    // VEC halves as VEC/2 __half2 values. 4 B (VEC=2) or 8 B (VEC=4) natural alignment of
    // every base address is a launch precondition -- see the `aligned` check in the
    // launcher -- and CUDA's vector types are the documented way to ask for a wide access.
    // The alternative the review suggested, a memcpy into an over-aligned local, measured
    // 25.6 us against 17.4 us on case C: the compiler can no longer prove alignment, so it
    // emits per-half accesses and the kernel loses the one-128 B-line-per-warp-load property
    // it exists for.
    const __half2* xr2 = reinterpret_cast<const __half2*>(xr + (long long)c * sC);
    float xv[VEC];
#pragma unroll
    for (int h = 0; h < VEC / 2; ++h) {
      const float2 f = __half22float2(xr2[h]);
      xv[2 * h] = f.x;
      xv[2 * h + 1] = f.y;
    }
#pragma unroll
    for (int t = 0; t < CT; ++t) {
      const float wv = wsh[t * Ci + c];
#pragma unroll
      for (int q = 0; q < VEC; ++q) acc[q][t] = fmaf(xv[q], wv, acc[q][t]);
    }
  }

  const long long obase = (long long)n * ((long long)Co * Pimg) + r;
#pragma unroll
  for (int t = 0; t < CT; ++t) {
    const int co = co_base + t;
    if (co < Co) {
      static_assert(VEC == 2, "only the 4 B (2-half) width is instantiated");
      __half res[VEC];
#pragma unroll
      for (int q = 0; q < VEC; ++q) {
        float v = fmaf(acc[q][t], scalep[co], biasp[co]);
        if (act == 1) v = v / (1.0f + __expf(-v));   // SiLU
        res[q] = __float2half(v);
      }
      __half2* o2 = reinterpret_cast<__half2*>(outp + obase + (long long)co * Pimg);
#pragma unroll
      for (int h = 0; h < VEC / 2; ++h)
        o2[h] = __halves2half2(res[2 * h], res[2 * h + 1]);
    }
  }
}

// ---------------------------------------------------------------------------
// Shared-memory-staged 1x1 kernel (unit stride, no padding/dilation), double buffered.
//
// The direct kernels read the activation once per output-channel tile, which is what made
// the deep-reduction 1x1 cases latency-bound (ncu: 0.25 waves/SM, 33.8 warps stalled on
// long_scoreboard per issue-active cycle). Here each block stages `W[BCO][BK]` and
// `X[BK][BP]` in shared memory, so every activation crosses the memory system once per
// *block*, the global loads are cooperative and coalesced, and the next K chunk is fetched
// while the current one is consumed. Each thread owns one P position and `BCO*BP/TPB` output
// channels; Co, Ci and P tails are zero-filled.
// ---------------------------------------------------------------------------
template <int BCO, int BP, int BK>
__device__ __forceinline__ void yc_stage_1x1(
    __half* __restrict__ Wd, __half* __restrict__ Xd,
    const __half* __restrict__ wp, const __half* __restrict__ xp,
    int c0, int Ci, int Co, int co_base, int p_base, int P, int Pimg,
    long long sN, long long sC, int tid, int tpb) {
  const __half kZero = __float2half(0.0f);
  for (int idx = tid; idx < BCO * BK; idx += tpb) {
    const int co = idx / BK, k = idx % BK, cc = c0 + k;
    Wd[co * BK + k] = (co_base + co < Co && cc < Ci)
                          ? wp[(long long)(co_base + co) * Ci + cc] : kZero;
  }
  for (int idx = tid; idx < BK * BP; idx += tpb) {
    const int k = idx / BP, pl = idx % BP, cc = c0 + k;
    const int p = p_base + pl;
    __half v = kZero;
    if (p < P && cc < Ci) {
      const int n = p / Pimg, r = p - n * Pimg;
      v = xp[(long long)n * sN + (long long)cc * sC + r];
    }
    Xd[k * BP + pl] = v;
  }
}

template <int BCO, int BP, int BK, int TPB>
__global__ void fused_conv1x1_staged(
    const __half* __restrict__ xp, const __half* __restrict__ wp,
    const float* __restrict__ scalep, const float* __restrict__ biasp,
    __half* __restrict__ outp,
    int Ci, int Co, long long sN, long long sC, int P, int Pimg, int act) {
  constexpr int CPT = BCO * BP / TPB;          // output channels per thread
  static_assert(BCO * BP == TPB * CPT, "tile must divide evenly among threads");
  __shared__ __half Ws[2][BCO * BK];
  __shared__ __half Xs[2][BK * BP];

  const int tid = threadIdx.x;
  const int p_local = tid % BP;
  const int co_lo = (tid / BP) * CPT;
  const int p_base = blockIdx.x * BP;
  const int co_base = blockIdx.y * BCO;

  float acc[CPT];
#pragma unroll
  for (int t = 0; t < CPT; ++t) acc[t] = 0.0f;

  int buf = 0;
  yc_stage_1x1<BCO, BP, BK>(Ws[0], Xs[0], wp, xp, 0, Ci, Co, co_base, p_base, P, Pimg,
                            sN, sC, tid, TPB);
  __syncthreads();

  for (int c0 = 0; c0 < Ci; c0 += BK) {
    const int nxt = c0 + BK;
    if (nxt < Ci)   // prefetch the next chunk into the other buffer while consuming this one
      yc_stage_1x1<BCO, BP, BK>(Ws[buf ^ 1], Xs[buf ^ 1], wp, xp, nxt, Ci, Co, co_base,
                                p_base, P, Pimg, sN, sC, tid, TPB);
    const __half* Wc = Ws[buf];
    const __half* Xc = Xs[buf];
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      const float xv = __half2float(Xc[k * BP + p_local]);   // warp-wide broadcast
#pragma unroll
      for (int t = 0; t < CPT; ++t)
        acc[t] = fmaf(xv, __half2float(Wc[(co_lo + t) * BK + k]), acc[t]);
    }
    __syncthreads();      // this chunk consumed, the prefetched one visible
    buf ^= 1;
  }

  const int p = p_base + p_local;
  if (p >= P) return;
  const int n = p / Pimg, r = p - n * Pimg;
  const long long obase = (long long)n * ((long long)Co * Pimg) + r;
#pragma unroll
  for (int t = 0; t < CPT; ++t) {
    const int co = co_base + co_lo + t;
    if (co < Co) {
      float v = fmaf(acc[t], scalep[co], biasp[co]);
      if (act == 1) v = v / (1.0f + __expf(-v));   // SiLU
      outp[obase + (long long)co * Pimg] = __float2half(v);
    }
  }
}

}  // namespace

#define YC_LAUNCH(CT, PT, KHC, KWC)                                            \
  fused_conv_bn_act_simt<CT, PT, KHC, KWC><<<grid, tpb, smem, stream>>>(        \
      xh, wh, sp, bp, oh, (int)Ci, (int)H, (int)W, (int)Co, (int)Ho, (int)Wo,   \
      (int)KH, (int)KW, (int)SH, (int)SW, (int)PH, (int)PW, (int)DH, (int)DW,   \
      sN, sC, sH, sWs, (int)P, (int)Pimg, (int)K, (int)act)

#define YC_DISPATCH_KHW(CT, PT)                                                \
  do {                                                                         \
    if (KH == 1 && KW == 1) { YC_LAUNCH(CT, PT, 1, 1); }                       \
    else if (KH == 3 && KW == 3) { YC_LAUNCH(CT, PT, 3, 3); }                  \
    else { YC_LAUNCH(CT, PT, 0, 0); }                                          \
  } while (0)

#define YC_DISPATCH_PT(CT)                                                     \
  do {                                                                         \
    if (pt >= 4) { YC_DISPATCH_KHW(CT, 4); }                                   \
    else if (pt == 2) { YC_DISPATCH_KHW(CT, 2); }                              \
    else { YC_DISPATCH_KHW(CT, 1); }                                           \
  } while (0)

// Returns false when the configuration cannot be launched safely (grid, shared memory,
// alignment); the caller then reports "unsupported" and the module falls back.
// Checked 64-bit arithmetic. Every product below feeds an int narrowing, a grid dimension
// or a shared-memory size, so it has to be proven representable *before* it is computed --
// comparing the result against a bound afterwards is already too late.
namespace ycmath {
constexpr int64_t kLim = 2147483647LL;          // what the kernels' int parameters hold

__host__ __device__ inline bool mul_ok(int64_t a, int64_t b, int64_t* out) {
  if (a < 0 || b < 0) return false;
  if (a != 0 && b > kLim / a) return false;
  *out = a * b;
  return *out <= kLim;
}
__host__ __device__ inline bool add_ok(int64_t a, int64_t b, int64_t* out) {
  if (a < 0 || b < 0 || a > kLim - b) return false;
  *out = a + b;
  return true;
}
__host__ __device__ inline bool in_int_range(int64_t v, int64_t lo, int64_t hi) {
  return v >= lo && v <= hi;
}
}  // namespace ycmath

bool yc_fused_launch(const at::Tensor& x, const at::Tensor& weight,
                     const at::Tensor& scale, const at::Tensor& bias,
                     at::Tensor& out,
                     int64_t N, int64_t Ci, int64_t H, int64_t W,
                     int64_t Co, int64_t Ho, int64_t Wo,
                     int64_t KH, int64_t KW, int64_t SH, int64_t SW,
                     int64_t PH, int64_t PW, int64_t DH, int64_t DW,
                     int64_t act, int64_t ct_in, int64_t pt_in, int64_t tpb_in,
                     int64_t force_kernel) {
  constexpr int64_t kMaxSmem = 32768;      // stay inside the 48 KB default, with headroom
  constexpr int64_t kMaxGridY = 65535;
  constexpr int64_t kMaxGridX = 2147483647LL;

  // Geometry fields arrive as int64 from Python and are handed to the kernels as int.
  if (!ycmath::in_int_range(KH, 1, 1024) || !ycmath::in_int_range(KW, 1, 1024)
      || !ycmath::in_int_range(SH, 1, 1024) || !ycmath::in_int_range(SW, 1, 1024)
      || !ycmath::in_int_range(PH, 0, 1 << 20) || !ycmath::in_int_range(PW, 0, 1 << 20)
      || !ycmath::in_int_range(DH, 1, 1 << 20) || !ycmath::in_int_range(DW, 1, 1 << 20)
      || !ycmath::in_int_range(N, 1, ycmath::kLim) || !ycmath::in_int_range(Ci, 1, ycmath::kLim)
      || !ycmath::in_int_range(Co, 1, ycmath::kLim) || !ycmath::in_int_range(H, 1, ycmath::kLim)
      || !ycmath::in_int_range(W, 1, ycmath::kLim) || !ycmath::in_int_range(Ho, 1, ycmath::kLim)
      || !ycmath::in_int_range(Wo, 1, ycmath::kLim)
      || !ycmath::in_int_range(ct_in, -1, 1024) || !ycmath::in_int_range(pt_in, -1, 1024)
      || !ycmath::in_int_range(tpb_in, -1, 1024))
    return false;

  int64_t Pimg = 0, P = 0, K = 0, khw = 0;
  if (!ycmath::mul_ok(Ho, Wo, &Pimg) || !ycmath::mul_ok(N, Pimg, &P)
      || !ycmath::mul_ok(KH, KW, &khw) || !ycmath::mul_ok(Ci, khw, &K))
    return false;

  const c10::cuda::OptionalCUDAGuard guard(at::device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();

  const __half* xh = reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>());
  const __half* wh = reinterpret_cast<const __half*>(weight.const_data_ptr<at::Half>());
  const float* sp = scale.const_data_ptr<float>();
  const float* bp = bias.const_data_ptr<float>();
  __half* oh = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

  const long long sN = x.stride(0), sC = x.stride(1), sH = x.stride(2), sWs = x.stride(3);

  // Is the vectorized 1x1 kernel usable? p is contiguous only for a unit-stride,
  // unpadded 1x1 convolution over an NCHW tensor whose rows are contiguous, and a
  // 4 B (2-half) load additionally needs every base address to be 4 B aligned.
  const bool vec_shape_geom =
      (KH == 1 && KW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0
       && sWs == 1 && sH == W);
  const bool vec_shape =
      (vec_shape_geom && Pimg % 2 == 0 && sC % 2 == 0 && sN % 2 == 0
       && reinterpret_cast<uintptr_t>(x.const_data_ptr<at::Half>()) % 4 == 0
       && reinterpret_cast<uintptr_t>(out.data_ptr<at::Half>()) % 4 == 0);

  int ct = (int)ct_in, pt = (int)pt_in, tpb = (int)tpb_in;
  // force_kernel selects the kernel; the shape gates still decide whether it is *legal*.
  const bool want_staged = (force_kernel == 3);
  // A forced kernel that cannot legally handle this shape is declined outright: silently
  // running a different kernel would make a "forced kernel" test prove nothing.
  if (force_kernel == 2 && !vec_shape) return false;
  const bool use_vec = vec_shape && force_kernel != 1 && force_kernel != 3
                       && (force_kernel == 2 || pt_in != 1);
  if (ct <= 0 || pt <= 0 || tpb <= 0) {
    // Measured on B200 over ct x pt x tpb = {1,2,4,8} x {1,2,4,8} x {64,128,256} on all
    // five benched cases (profile/fused_v1/analysis/tile_sweep.txt). Three regimes:
    //   * vectorized 1x1: one 4 B load per lane covers a full 128 B line per warp with
    //     no waste, which is what the miss count -- and so the MSHR-bound time -- keys
    //     off. 1.71x on case C vs 1.53x for the scalar tile.
    //   * tiny P on the general kernel: bound by L2 latency, so warps in flight are
    //     everything (ct=1, pt=1, tpb=256; ct=4 cost 12% on the 400-position cases).
    //   * large P (case B): bound by load and shared-load throughput, so maximum reuse
    //     in both directions (ct=8, pt=4) wins by 17%.
    int64_t outputs = 0;
    if (!ycmath::mul_ok(P, Co, &outputs)) return false;
    tpb = 256;
    if (use_vec) { pt = 2; ct = (outputs > (1 << 17)) ? 2 : 1; }
    else if (outputs <= (1 << 16)) { ct = 1; pt = 1; }
    else if (outputs <= (1 << 19)) { ct = 4; pt = 1; }
    else { ct = 8; pt = 4; }
    if (ct > Co) ct = 1;
    while (P < (int64_t)tpb * pt && tpb > 32) tpb >>= 1;
  }
  if (ct > 8) ct = 8;

  // --- tier 1: the shared-memory-staged 1x1 kernel -------------------------------------
  // Same 1x1 unit-stride geometry as the vector kernel but no alignment requirement (its
  // loads are scalar and cooperative). Reachable *only* through the explicit pt == 16
  // tuning override: measured on B200 it takes 39.9 us on case C against 17.4 us for the
  // vector kernel and 18.0 us for FOLDED, so staging the activation does not pay at these
  // problem sizes (profile/fused_v1/REPORT.md section 5). Kept as the evidence for that
  // conclusion, and as the starting point for a double-buffered version.
  if (want_staged) {
    if (!vec_shape_geom) return false;      // the staged kernel only handles 1x1 unit stride
    constexpr int BCO = 32, BP = 128, BK = 32, STPB = 256;
    const int64_t gx = (P + BP - 1) / BP, gy = (Co + BCO - 1) / BCO;
    if (gx > kMaxGridX || gy > kMaxGridY) return false;
    fused_conv1x1_staged<BCO, BP, BK, STPB><<<dim3((unsigned)gx, (unsigned)gy), STPB, 0,
                                              stream>>>(
        xh, wh, sp, bp, oh, (int)Ci, (int)Co, sN, sC, (int)P, (int)Pimg, (int)act);
    return true;
  }

  // Both remaining kernels stage ct weight rows (K = Ci for a 1x1) in shared memory.
  int64_t smem64 = 0;
  while (true) {
    if (!ycmath::mul_ok(ct, K * (int64_t)sizeof(float), &smem64)) {
      if (ct == 1) return false;
      ct >>= 1;
      continue;
    }
    if (smem64 <= kMaxSmem || ct == 1) break;
    ct >>= 1;
  }
  if (smem64 > kMaxSmem) return false;    // ct cannot go below 1: too deep a reduction

  // --- tier 2: the vectorized 1x1 kernel -----------------------------------------------
  // pt == 1 forces the general kernel instead (the tests use it to exercise that path on
  // 1x1 shapes too). pt is the vector width here, so it is read before the general
  // kernel's clamp.
  if (use_vec) {
    // Only the 2-half (4 B) width is instantiated. It is what dispatch selects everywhere --
    // one warp-wide load covers exactly one 128 B line with no waste, which is the property
    // the kernel exists for. The 4-half width was removed after tests/sanitize.py found it
    // wrong on shapes where Pimg is a multiple of 4 but not of 8 (max abs error 0.55 on a
    // 6x6 image); the 8-half width was removed earlier for a bad store. Neither was ever
    // chosen by dispatch, so both were failure surface with no upside.
    const int vec = 2;
    const bool aligned = Pimg % vec == 0 && sC % vec == 0 && sN % vec == 0
                         && (reinterpret_cast<uintptr_t>(x.const_data_ptr<at::Half>())
                             % (2 * (uintptr_t)vec) == 0)
                         && (reinterpret_cast<uintptr_t>(out.data_ptr<at::Half>())
                             % (2 * (uintptr_t)vec) == 0);
    if (aligned) {
      // Clamp before the grid: this kernel is instantiated for CT in {1,2,4}, and sizing
      // grid.y from an unclamped ct=8 would leave half the output channels uncomputed
      // (tests/sanitize.py caught exactly that: max abs error 0.52). Same failure shape as
      // the pt sentinel in the general path -- every clamp must precede the grid it feeds.
      if (ct > 4) ct = 4;
      int vtpb = tpb;
      while (P / vec < (int64_t)vtpb && vtpb > 32) vtpb >>= 1;
      const int64_t vgx = (P / vec + vtpb - 1) / vtpb;
      const int64_t vgy = (Co + ct - 1) / ct;
      int64_t vsmem64 = 0;
      if (!ycmath::mul_ok(ct, Ci * (int64_t)sizeof(float), &vsmem64)) return false;
      if (vgx > kMaxGridX || vgy > kMaxGridY || vsmem64 > kMaxSmem) return false;
      const dim3 vgrid((unsigned)vgx, (unsigned)vgy);
      const size_t vsmem = (size_t)vsmem64;
#define YC_LAUNCH_V(CTV, VECV)                                                  \
      fused_conv1x1_bn_act_vec<CTV, VECV><<<vgrid, vtpb, vsmem, stream>>>(       \
          xh, wh, sp, bp, oh, (int)Ci, (int)Co, sN, sC, (int)P, (int)Pimg, (int)act)
#define YC_DISPATCH_V(VECV)                                                     \
      do {                                                                      \
        if (ct >= 4) { YC_LAUNCH_V(4, VECV); }                                  \
        else if (ct == 2) { YC_LAUNCH_V(2, VECV); }                             \
        else { YC_LAUNCH_V(1, VECV); }                                          \
      } while (0)
      YC_DISPATCH_V(2);
#undef YC_DISPATCH_V
#undef YC_LAUNCH_V
      return true;
    }
  }

  // --- tier 3: the general kernel -------------------------------------------------------
  if (pt > 4) pt = 4;            // the widest instantiated P tile
  if (tpb <= 0 || tpb > 1024) return false;
  int64_t per_block = 0;
  if (!ycmath::mul_ok(tpb, pt, &per_block) || per_block <= 0) return false;
  const int64_t gx = (P + per_block - 1) / per_block;
  const int64_t gy = (Co + ct - 1) / ct;
  if (gx > kMaxGridX || gy > kMaxGridY) return false;
  const dim3 grid((unsigned)gx, (unsigned)gy);
  const size_t smem = (size_t)ct * (size_t)K * sizeof(float);

  if (ct >= 8) { YC_DISPATCH_PT(8); }
  else if (ct >= 4) { YC_DISPATCH_PT(4); }
  else if (ct == 2) { YC_DISPATCH_PT(2); }
  else { YC_DISPATCH_PT(1); }
  return true;
}
"""

# ---------------------------------------------------------------------------
# Host side: a ``Plan`` holding the weight/scale/bias tensors and every scalar, so
# the hot path is one two-argument pybind call. Everything input-dependent
# (``N, H, W, Ho, Wo``, the launch config) is derived here in C++ from
# ``x.sizes()``/``x.strides()``. ``run`` returns ``None`` -- not an exception --
# for an input the kernel does not claim, so Python falls back instead of raising.
# ---------------------------------------------------------------------------
_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <optional>

// Returns false when the configuration cannot be launched safely (grid, shared memory,
// alignment); the caller then reports "unsupported" and the module falls back.
bool yc_fused_launch(const at::Tensor& x, const at::Tensor& weight,
                     const at::Tensor& scale, const at::Tensor& bias,
                     at::Tensor& out,
                     int64_t N, int64_t Ci, int64_t H, int64_t W,
                     int64_t Co, int64_t Ho, int64_t Wo,
                     int64_t KH, int64_t KW, int64_t SH, int64_t SW,
                     int64_t PH, int64_t PW, int64_t DH, int64_t DW,
                     int64_t act, int64_t ct, int64_t pt, int64_t tpb,
                     int64_t force_kernel);

// Checked 64-bit arithmetic. Every product below feeds an int narrowing, a grid dimension
// or a shared-memory size, so it has to be proven representable *before* it is computed --
// comparing the result against a bound afterwards is already too late.
namespace ycmath {
constexpr int64_t kLim = 2147483647LL;          // what the kernels' int parameters hold

__host__ __device__ inline bool mul_ok(int64_t a, int64_t b, int64_t* out) {
  if (a < 0 || b < 0) return false;
  if (a != 0 && b > kLim / a) return false;
  *out = a * b;
  return *out <= kLim;
}
__host__ __device__ inline bool add_ok(int64_t a, int64_t b, int64_t* out) {
  if (a < 0 || b < 0 || a > kLim - b) return false;
  *out = a + b;
  return true;
}
__host__ __device__ inline bool in_int_range(int64_t v, int64_t lo, int64_t hi) {
  return v >= lo && v <= hi;
}
}  // namespace ycmath

struct Plan {
  at::Tensor weight, scale, bias;
  int64_t Co = 0, Ci = 0, KH = 0, KW = 0;
  int64_t SH = 1, SW = 1, PH = 0, PW = 0, DH = 1, DW = 1;
  int64_t act = 0;
  int64_t tier = 0;
  int64_t device_index = -1;
  int64_t ct = -1, pt = -1, tpb = -1;   // -1 = pick the tile at launch
  // 0 = let the heuristics choose; 1 = general, 2 = vectorized 1x1, 3 = staged 1x1. A forced
  // kernel bypasses *performance* dispatch only -- every semantic and safety gate still
  // applies, and an ineligible shape is still declined.
  int64_t force_kernel = 0;
};

Plan make_plan(at::Tensor weight, at::Tensor scale, at::Tensor bias,
               int64_t SH, int64_t SW, int64_t PH, int64_t PW,
               int64_t DH, int64_t DW, int64_t act, int64_t tier,
               int64_t ct, int64_t pt, int64_t tpb, int64_t force_kernel) {
  TORCH_CHECK(weight.dim() == 4, "weight must be 4-D");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be fp16");
  TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
  TORCH_CHECK(scale.dtype() == at::kFloat && bias.dtype() == at::kFloat,
              "scale/bias must be fp32");
  TORCH_CHECK(scale.is_contiguous() && bias.is_contiguous(),
              "scale/bias must be contiguous");
  TORCH_CHECK(scale.numel() == weight.size(0) && bias.numel() == weight.size(0),
              "scale/bias must have Co elements");
  TORCH_CHECK(scale.device() == weight.device() && bias.device() == weight.device(),
              "scale/bias must share the weight's device");
  TORCH_CHECK(SH > 0 && SW > 0 && DH > 0 && DW > 0 && PH >= 0 && PW >= 0,
              "bad conv geometry");
  Plan p;
  p.weight = std::move(weight);
  p.scale = std::move(scale);
  p.bias = std::move(bias);
  p.Co = p.weight.size(0);
  p.Ci = p.weight.size(1);
  p.KH = p.weight.size(2);
  p.KW = p.weight.size(3);
  p.SH = SH; p.SW = SW; p.PH = PH; p.PW = PW; p.DH = DH; p.DW = DW;
  p.act = act;
  p.tier = tier;
  p.ct = ct; p.pt = pt; p.tpb = tpb;
  TORCH_CHECK(force_kernel >= 0 && force_kernel <= 3, "force_kernel must be 0..3");
  p.force_kernel = force_kernel;
  p.device_index = p.weight.device().index();
  return p;
}

std::optional<at::Tensor> run(const Plan& p, const at::Tensor& x) {
  // Everything the fused kernel requires of its input. A rejection returns
  // nullopt (-> None) so the module can fall back to a slower but correct tier.
  if (x.dim() != 4 || x.scalar_type() != at::kHalf || !x.is_cuda()) return std::nullopt;
  if (x.device().index() != p.device_index) return std::nullopt;
  if (x.stride(3) != 1 || x.stride(2) != x.size(3)) return std::nullopt;
  if (x.size(1) != p.Ci) return std::nullopt;

  const int64_t N = x.size(0), H = x.size(2), W = x.size(3);
  // Effective kernel extents and the padded input first, each checked, so nothing overflows
  // before it is compared: an adversarial padding or dilation must be refused, not wrapped.
  int64_t ext_h = 0, ext_w = 0, padded_h = 0, padded_w = 0, twoPH = 0, twoPW = 0;
  if (!ycmath::mul_ok(p.DH, p.KH - 1, &ext_h) || !ycmath::mul_ok(p.DW, p.KW - 1, &ext_w)
      || !ycmath::mul_ok(2, p.PH, &twoPH) || !ycmath::mul_ok(2, p.PW, &twoPW)
      || !ycmath::add_ok(H, twoPH, &padded_h) || !ycmath::add_ok(W, twoPW, &padded_w))
    return std::nullopt;
  if (padded_h - ext_h - 1 < 0 || padded_w - ext_w - 1 < 0) return std::nullopt;
  const int64_t Ho = (padded_h - ext_h - 1) / p.SH + 1;
  const int64_t Wo = (padded_w - ext_w - 1) / p.SW + 1;
  if (N <= 0 || Ho <= 0 || Wo <= 0 || x.numel() == 0) return std::nullopt;

  // Measured dispatch (the one place shape-dependent tier choice lives). The fused
  // kernel re-reads the activation once per output-channel tile, while cuDNN + SiLU
  // read it once each across two launches -- so on a 1x1 unit-stride convolution the
  // fused kernel only wins when there are many output positions per unit of reduction
  // depth. Measured on B200: case C (P=6400, K=96) 1.71x fused vs 1.62x folded, but
  // case D (P=400, K=256) 1.01x fused vs 1.36x folded and case E (P=1600, K=256)
  // 1.09x vs 1.36x. P >= 16*K separates them; a 3x3 always goes fused, because there
  // cuDNN additionally pays NCHW->NHWC->NCHW layout transforms (1.64x vs 1.17x on
  // case A). Returning nullopt lets forward() fall back rather than guess.
  // Everything the kernels receive as int, computed with checked arithmetic.
  int64_t khw = 0, K = 0, Pimg = 0, P = 0, thresh = 0;
  if (!ycmath::mul_ok(p.KH, p.KW, &khw) || !ycmath::mul_ok(p.Ci, khw, &K)
      || !ycmath::mul_ok(Ho, Wo, &Pimg) || !ycmath::mul_ok(N, Pimg, &P)
      || !ycmath::mul_ok(16, K, &thresh))
    return std::nullopt;
  if (x.numel() > ycmath::kLim || p.Co > ycmath::kLim
      || H > ycmath::kLim || W > ycmath::kLim)
    return std::nullopt;

  // Performance dispatch (not a safety gate): see the comment below. `force_kernel` lets a
  // test exercise a kernel this heuristic would route elsewhere.
  if (p.force_kernel == 0 && p.KH == 1 && p.KW == 1 && p.SH == 1 && p.SW == 1 && P < thresh)
    return std::nullopt;

  at::Tensor out = at::empty({N, p.Co, Ho, Wo}, x.options());
  if (!yc_fused_launch(x, p.weight, p.scale, p.bias, out, N, p.Ci, H, W, p.Co, Ho, Wo,
                       p.KH, p.KW, p.SH, p.SW, p.PH, p.PW, p.DH, p.DW, p.act,
                       p.ct, p.pt, p.tpb, p.force_kernel))
    return std::nullopt;          // configuration refused before any launch
  // Check the launch rather than letting a bad configuration corrupt the stream. A launch
  // failure means nothing ran, so reporting "unsupported" and falling back is correct --
  // and keeps the promise that forward() never raises.
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_WARN_ONCE("yolov10_conv: fused launch failed (", cudaGetErrorString(err),
                    "); falling back");
    return std::nullopt;
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Plan>(m, "Plan")
      .def_readonly("tier", &Plan::tier)
      .def_readonly("act", &Plan::act)
      .def_readonly("Co", &Plan::Co)
      .def_readonly("Ci", &Plan::Ci)
      .def_readonly("KH", &Plan::KH)
      .def_readonly("KW", &Plan::KW)
      .def_readonly("ct", &Plan::ct)
      .def_readonly("pt", &Plan::pt)
      .def_readonly("tpb", &Plan::tpb)
      .def_readonly("force_kernel", &Plan::force_kernel)
      .def_readonly("scale", &Plan::scale)
      .def_readonly("bias", &Plan::bias);
  m.def("make_plan", &make_plan, "build a fused conv-bn-act plan",
        pybind11::arg("weight"), pybind11::arg("scale"), pybind11::arg("bias"),
        pybind11::arg("SH"), pybind11::arg("SW"), pybind11::arg("PH"),
        pybind11::arg("PW"), pybind11::arg("DH"), pybind11::arg("DW"),
        pybind11::arg("act"), pybind11::arg("tier"),
        pybind11::arg("ct") = -1, pybind11::arg("pt") = -1,
        pybind11::arg("tpb") = -1, pybind11::arg("force_kernel") = 0);
  m.def("run", &run, "run the fused conv-bn-act kernel (None if unsupported)");
}
"""

_EXT = None
_EXT_ERROR: str | None = None
_RUN = None
_MIN_CAPABILITY = (10, 0)  # compiled for sm_100 only


def _build_extension():
    """Compile the fused kernel. Called once at *import*, never from ``forward``.

    ``fastkernels.bench._check_threads`` fails a candidate whose background-thread
    count rises during the timed region, and ninja spawns threads -- so the build
    has to happen here. The name is a function of the source text so concurrent
    bench workers share one build directory and serialize on torch's file baton
    instead of each compiling a private copy, and exactly one gencode target is
    passed (the environment's ``TORCH_CUDA_ARCH_LIST`` expands to seven, which
    inflates import time by minutes).
    """
    if os.environ.get("YOLOV10_CONV_FORCE_BUILD_FAILURE") == "1":
        raise RuntimeError("build failure forced by YOLOV10_CONV_FORCE_BUILD_FAILURE=1")
    from torch.utils.cpp_extension import load_inline

    digest = hashlib.sha1((_CPP_SOURCE + _CUDA_SOURCE).encode()).hexdigest()[:8]
    cuda_flags = ["-O3", "-lineinfo", "-gencode=arch=compute_100,code=sm_100"]
    assert sum("arch" in f for f in cuda_flags) == 1, "exactly one gencode target"
    return load_inline(
        name=f"yolov10_conv_fused_{digest}",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        verbose=False,
    )


if os.environ.get("YOLOV10_CONV_DISABLE_EXT") != "1":
    try:
        _EXT = _build_extension()
        _RUN = _EXT.run
    except Exception as exc:  # noqa: BLE001 - a build failure must degrade, not raise
        _EXT, _RUN = None, None
        _EXT_ERROR = f"{type(exc).__name__}: {exc}"

_FORCE_TIER = os.environ.get("YOLOV10_CONV_FORCE_TIER") or None


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False
        # Nothing weight-derived here: the bench shares weights after __init__.
        self._plan = None
        self._plan_built = False
        self._plan_fp = None
        self._folded = None
        self._state_fp = None
        self._tier = None
        self._force_tier = _FORCE_TIER
        # Testing overrides: a forced kernel/tile bypasses performance dispatch only.
        self._force_kernel = 0
        self._force_ct = -1
        self._force_pt = -1
        self._force_tpb = -1
        self.register_load_state_dict_post_hook(YOLOConv._load_state_dict_post_hook)

    # -- hot path ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if plan is not None:
            # Submodule references come from __dict__ (not nn.Module.__getattr__) so these
            # are plain attribute reads. A tier that skips self.conv/self.bn/self.act must
            # not skip a hook someone registered on them, and BatchNorm switching to batch
            # statistics on its own (bn.train(), or track_running_stats cleared) changes
            # what the correct answer *is* -- both are checked per call, not per plan.
            act, conv, bn = self._fast_act, self._fast_conv, self._fast_bn
            if (not self.training
                    and not torch.is_grad_enabled()
                    and not self._forward_hooks
                    and not self._forward_pre_hooks
                    and not act._forward_hooks
                    and not act._forward_pre_hooks
                    and not conv._forward_hooks
                    and not conv._forward_pre_hooks
                    and (bn is None
                         or (not bn._forward_hooks and not bn._forward_pre_hooks
                             and not bn.training and bn.track_running_stats))
                    and self._fingerprint() == self._plan_fp):
                out = _RUN(plan, x)   # None when the kernel does not claim x
                if out is not None:
                    self.__dict__["_tier"] = TIER_SIMT
                    return out
        return self._forward_slow(x)

    @staticmethod
    def _tref(t):
        """Identity, version and storage of a tensor, or None.

        Identity catches `conv.weight = Parameter(...)`, `_version` catches ordinary
        in-place updates (an optimizer step, `copy_`, `add_`), and `data_ptr()` catches a
        child-only `_apply` (`conv.half()`), which swaps `p.data` without touching either.
        """
        return None if t is None else (id(t), t._version, t.data_ptr())

    def _fingerprint(self):
        """Everything the cached plan and folded cache were derived from.

        Read through `__dict__`/`_parameters`/`_buffers` to stay off
        `nn.Module.__getattr__`; measured at ~1.3 us per call, against ~10 us of launch
        overhead the fused tier saves. A mismatch means the cache is stale, so `forward`
        falls through to `_forward_slow`, which invalidates and rebuilds.
        """
        conv = self._fast_conv
        cd = conv.__dict__
        cp = cd["_parameters"]
        tref = YOLOConv._tref
        bn = self._fast_bn
        # `_is_fused` decides whether BN is folded in at all, so a plan built on one side of
        # a fuse() must not answer on the other (a stale plan measured max abs error 2.30).
        # Child identity *and* exact type: a replaced subclass with an overridden forward
        # must not be skipped just because its tensor fields look familiar.
        base = (self.__dict__.get("_is_fused"),
                id(conv), type(conv), id(bn), type(bn),
                tref(cp.get("weight")), tref(cp.get("bias")),
                cd.get("stride"), cd.get("padding"), cd.get("dilation"), cd.get("groups"),
                id(self._fast_act), type(self._fast_act),
                self.__dict__.get("_force_tier"), self.__dict__.get("_force_kernel"),
                self.__dict__.get("_force_ct"), self.__dict__.get("_force_pt"),
                self.__dict__.get("_force_tpb"))
        if bn is None:
            return base
        bd = bn.__dict__
        bp = bd["_parameters"]
        bb = bd["_buffers"]
        return base + (tref(bp.get("weight")), tref(bp.get("bias")),
                       tref(bb.get("running_mean")), tref(bb.get("running_var")),
                       bd.get("eps"), bd.get("training"), bd.get("track_running_stats"))

    # -- fallback tiers ----------------------------------------------------
    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        # A stale cache is rebuilt, not merely bypassed: otherwise a module whose weights
        # were updated in place would sit on the slowest tier for the rest of its life.
        # `_state_fp` is the state at the last build *attempt*, so a state change also
        # retries an attempt that previously declined (e.g. bn.train() then bn.eval()).
        fp = self._fingerprint()
        if fp != self._state_fp:
            self._invalidate()
            self.__dict__["_state_fp"] = fp
        if not self._plan_built:
            self._plan_built = True
            self._plan = self._build_plan()
            if self._plan is not None:
                self.__dict__["_plan_fp"] = self._fingerprint()
                self._tier = TIER_SIMT
                return self.forward(x)

        if (self.training
                or torch.is_grad_enabled()
                or self._force_tier == TIER_EXACT
                or not self._folded_ok(x)):
            self._tier = TIER_EXACT
            return self._forward_exact(x)

        folded = self._folded
        if folded is None:
            folded = self._build_folded()
            if folded is None:
                self._tier = TIER_EXACT
                return self._forward_exact(x)
            self._folded = folded
        self._tier = TIER_FOLDED
        weight, bias = folded
        conv = self.conv
        y = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return self.act(y)

    def _forward_exact(self, x: torch.Tensor) -> torch.Tensor:
        """The literal baseline expression -- identical semantics, side effects included."""
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def _folded_ok(self, x: torch.Tensor) -> bool:
        """Can BN be folded away for this call without changing what is computed?"""
        conv = self.conv
        if self._is_fused:
            return False           # already a plain conv + act; EXACT *is* that path
        if not isinstance(x, torch.Tensor) or x.dtype is not conv.weight.dtype:
            return False
        if x.dim() != 4:
            return False
        if x.dtype not in _FOLDABLE_DTYPES:
            # Folding perturbs the conv by ~1e-3 relative (TF32 rounds the scaled
            # weight differently), which is inside the bench's fp16/bf16 tolerance
            # (measured max abs err <= 2e-3) but outside its fp32 one
            # (atol 1e-5, rtol 1e-3). High precision in, EXACT out.
            return False
        if conv._forward_hooks or conv._forward_pre_hooks:
            return False           # this tier does not call self.conv
        bn = self._fast_bn
        if bn is None or bn.training or not bn.track_running_stats:
            return False   # batch statistics: only EXACT reproduces them
        if bn._forward_hooks or bn._forward_pre_hooks:
            return False           # ... nor self.bn
        return True

    # -- lazily built state ------------------------------------------------
    def _act_code(self) -> int | None:
        cls = type(self.act)
        if cls is nn.Identity:
            return _ACT_IDENTITY
        if cls is SiLU or cls is nn.SiLU:
            return _ACT_SILU
        return None

    def _bn_affine(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The BN fold as fp32 per-output-channel ``(scale, bias)``.

        ``scale = gamma / sqrt(running_var + eps)``,
        ``bias = beta - scale * running_mean (+ scale * conv.bias)``, all in fp32:
        ``running_mean``/``running_var`` are buffers the bench leaves in fp32 while
        ``gamma``/``beta`` are parameters it casts to fp16.
        """
        conv = self.conv
        co = conv.weight.shape[0]
        device = conv.weight.device
        bn = getattr(self, "bn", None)
        if self._is_fused or bn is None:
            scale = torch.ones(co, dtype=torch.float32, device=device)
            bias = torch.zeros(co, dtype=torch.float32, device=device)
        else:
            if not isinstance(bn, (BatchNorm2d, nn.BatchNorm2d)):
                return None
            if bn.training or not bn.track_running_stats:
                return None
            if bn.weight is None or bn.bias is None:
                return None       # affine=False is not this fold
            if bn.running_mean is None or bn.running_var is None:
                return None
            if bn.running_var.numel() != co:
                return None
            scale = bn.weight.detach().float() / torch.sqrt(
                bn.running_var.detach().float() + bn.eps)
            bias = bn.bias.detach().float() - scale * bn.running_mean.detach().float()
        if conv.bias is not None:
            bias = bias + scale * conv.bias.detach().float()
        return scale.contiguous(), bias.contiguous()

    def _build_plan(self):
        """Build the fused-kernel plan, or return None to leave it to a lower tier."""
        if _RUN is None or self._force_tier in (TIER_EXACT, TIER_FOLDED):
            return None
        conv = self.conv
        weight = conv.weight
        if (conv.groups != 1
                or weight.dtype is not torch.float16
                or not weight.is_cuda
                or not weight.is_contiguous()
                or weight.dim() != 4):
            return None
        if torch.cuda.get_device_capability(weight.device) != _MIN_CAPABILITY:
            return None
        act_code = self._act_code()
        if act_code is None:
            return None
        affine = self._bn_affine()
        if affine is None:
            return None
        scale, bias = affine
        sh, sw = conv.stride
        ph, pw = conv.padding
        dh, dw = conv.dilation
        try:
            return _EXT.make_plan(weight.detach(), scale, bias,
                                  sh, sw, ph, pw, dh, dw, act_code, 0,
                                  ct=self._force_ct, pt=self._force_pt, tpb=self._force_tpb,
                                  force_kernel=self._force_kernel)
        except Exception:  # noqa: BLE001 - an unbuildable plan is a fallback, not an error
            return None

    def _build_folded(self):
        """A BN-folded weight/bias *copy* -- ``self.conv.weight`` is never mutated,
        so a later ``fuse()`` is not a double fold."""
        affine = self._bn_affine()
        if affine is None:
            return None
        scale, bias = affine
        weight = self.conv.weight
        dtype = weight.dtype
        if not dtype.is_floating_point:
            return None
        folded_w = (weight.detach().float() * scale.view(-1, 1, 1, 1)).to(dtype)
        return folded_w, bias.to(dtype)

    # -- invalidation ------------------------------------------------------
    def _invalidate(self, *_args) -> None:
        d = self.__dict__
        d["_plan"] = None
        d["_plan_built"] = False
        d["_plan_fp"] = None
        d["_folded"] = None
        d["_state_fp"] = None
        d["_tier"] = None

    @staticmethod
    def _load_state_dict_post_hook(module, _incompatible_keys) -> None:
        module._invalidate()

    def __setattr__(self, name, value):
        # `_is_fused` and `_force_tier` change which tier is *allowed*, so they invalidate
        # like a submodule swap does; `_force_tier` is also in the fingerprint, so setting it
        # after a plan exists takes effect on the very next call.
        if name in ("act", "conv", "bn", "_is_fused", "_force_tier",
                    "_force_kernel", "_force_ct", "_force_pt", "_force_tpb"):
            self._invalidate()
        super().__setattr__(name, value)
        if name in ("act", "conv", "bn"):
            # Direct __dict__ writes: plain attributes, so they neither register duplicate
            # submodules nor add state_dict keys, and reading them in the hot path costs a
            # dict lookup instead of nn.Module.__getattr__.
            if name == "act" and not isinstance(value, nn.Module):
                value = nn.Identity()
            self.__dict__[f"_fast_{name}"] = value

    def __delattr__(self, name):
        # fuse() does `delattr(self, "bn")`; the cached reference has to go with it.
        if name in ("act", "conv", "bn"):
            self._invalidate()
        super().__delattr__(name)
        if name in ("act", "conv", "bn"):
            self.__dict__[f"_fast_{name}"] = None

    def _apply(self, fn, recurse=True):
        self._invalidate()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True):
        # Training mutates bn.running_mean/var in place, so any cached fold is stale.
        self._invalidate()
        return super().train(mode)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        self._invalidate()
        return self


def fuse_module(module: nn.Module) -> nn.Module:
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
