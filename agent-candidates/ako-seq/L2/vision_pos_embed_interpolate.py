"""Bilinear interpolation of learned 2D position embeddings (Qwen3-VL).

Owns a learned embedding weight of (num_grid_per_side^2, hidden_size).
forward() interpolates these onto arbitrary (h, w) grids using bilinear
weights, then reshuffles by spatial_merge_size for the vision encoder.

Why this is not the reference implementation
--------------------------------------------
``forward`` takes **no tensor arguments**: ``grid_thw_list`` is a Python list of
ints, so every tap index, every bilinear weight, the spatial-merge permutation
and each entry's row offset in the concatenated result are pure functions of
``(grid_thw_list, num_grid_per_side, spatial_merge_size)``.  None of them depend
on the embedding table.  The reference rebuilds all of it on the GPU on every
call (~25 tiny kernels per list entry: linspace / meshgrid / clamp / stack /
arithmetic), materializes a ``(4, h*w, hidden)`` gather -- 4x the output bytes,
written once and then re-read twice for the multiply and the sum -- pays a copy
for ``permute().reshape()``, another for ``expand(t).reshape()``, and finally a
``torch.cat`` over every entry.

Here **one** kernel launch per forward writes the final concatenated buffer
directly, and the geometry never touches memory at all: a per-entry descriptor
(<= 16 entries x 9 words, ~640 B) travels *by value* in the kernel's parameter
block, so it arrives through the constant bank with no HBM traffic, no H2D copy
and -- the point -- no load that the table pointers have to wait for.  Each
program owns one *distinct* output row and derives from its row index:

* the source position, by inverting the spatial-merge permutation
  (``row = ((a*wm + b)*m + c)*m + d`` -> ``(i, j) = (a*m + c, b*m + d)``);
* the two fp32 sample coordinates, in closed form because ``linspace`` is affine;
* floor / clamped ceil / fractions -> the 4 table rows and the 4 weights;
* its destination row, and the ``t`` frame copies (bit-identical, so they are
  stored from the same registers rather than recomputed).

An earlier revision instead built three device-resident metadata arrays (taps,
weights, destination) once per distinct call and read 40 B/row from them.  That
cost a *cold* dependent miss round in the prologue -- the bench flushes L2 before
every timed call, and the table pointers were data-dependent on ``taps[row]`` --
re-paid once per ``gridDim.y`` column slice.  Deriving the geometry instead is
~65 ALU instructions per row and measured a full harness step (11.26 -> 9.22 us)
on the image case, where the column split had been fetching each row's metadata
six times.  That path is *kept*, in ``_build_meta``, for the shapes the closed
form does not cover (h or w of 1, more than 16 entries, h*w >= 2^22) and as the
fallback if the linspace verification below ever disagrees.

Bit-exactness (``max_abs_error == 0.00e+00`` on every benched case and on
fp16/fp32, not merely within tolerance) is what constrains the arithmetic:

* ATen's CUDA linspace is ``ind < steps/2 ? start + step*ind : end -
  step*(steps-1-ind)`` with an fp32 step -- and its build **contracts the second
  form into an FMA**.  Verified against ``torch.linspace`` for every n in 1..259
  plus a spread to 4096 and 5 values of num_grid: the unfused form differs on
  1110 of those, the fused form on none.  So ``fk_lin`` uses an explicit
  ``__fmaf_rn``, ``step`` is computed host-side in fp32, everything else stays
  unfused (``-fmad=false``), and ``_lin_ok`` re-checks the closed form against
  torch per distinct n before the descriptor path is used at all.
* the weights keep the reference's expression order (``w11 = dh*dw``,
  ``w10 = dh - w11``, ``w01 = dw - w11``, ``w00 = (1 - dh) - w01``) and its
  ``.to(dtype)`` rounding, and each product is rounded to ``dtype`` before
  entering the fp32 accumulator, because the reference multiplies a ``dtype``
  weight by a ``dtype`` table row and only then sums.  ``__hmul2`` gives exactly
  that, two elements at a time.

Where the 15.4 us of the hottest case goes -- every term measured in the bench's
own timing loop (dev/floor.py, dev/launch.py, dev/roof.py):

    2.85  harness floor: a forward that returns a cached tensor measures this
    2.30  the single kernel launch (~2.05 us each; flat in grid size and in
          parameter-block size, and a CUDA-graph replay is *worse*)
    6.14  writing 32.4 MB -- roofline: cudaMemsetAsync of the same bytes in the
          same loop measures identically (5.3 TB/s)
    2.5   instruction issue, 76 ops per 16 B vector, minimal for this arithmetic
    1.5   the forced cold fetch of the 5.3 MB table (L2 is flushed every call)

Anything outside the fused path's regime (non-CUDA device, table dtype !=
requested dtype, h or w not divisible by spatial_merge_size, empty list, a row
pitch that is not 16 B-vectorizable) falls back to the reference implementation,
which is kept verbatim below.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

from ..L1.embedding import Embedding

_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>

namespace {

// Host-side float from its bits (the descriptor carries fp32 values as int32).
inline float __int_as_float_h(int x) {
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

// ---------------------------------------------------------------------------
// Arithmetic.  The reference computes ``dtype(w) * dtype(table_row)`` -- so each
// product is rounded to ``dtype`` -- and only then sums the four terms, in a
// float accumulator (``sum``'s acc_type for a low-precision input).  For the
// 2-byte dtypes ``__hmul2`` gives exactly that product rounding two elements at
// a time: the exact bf16xbf16 product fits in fp32, so PyTorch's
// "fp32 multiply, then cast" and the hardware's single-rounding ``mul.rn`` agree
// bit for bit.  Doing it pairwise is what makes this cheap -- the scalar form
// (cvt up, fp32 mul, cvt down, cvt up) costs ~2x the instructions and measured
// 17.3 us vs 15.3 us on the hottest case.
// ---------------------------------------------------------------------------
template <typename T>
struct P2;

template <>
struct P2<__nv_bfloat16> {
  using type = __nv_bfloat162;
  __device__ __forceinline__ static type mul(type a, type b) { return __hmul2(a, b); }
  __device__ __forceinline__ static float2 up(type a) { return __bfloat1622float2(a); }
  __device__ __forceinline__ static type pack(float x, float y) {
    return __floats2bfloat162_rn(x, y);
  }
  __device__ __forceinline__ static type bcast(float x) {
    return __bfloat162bfloat162(__float2bfloat16(x));
  }
};

template <>
struct P2<__half> {
  using type = __half2;
  __device__ __forceinline__ static type mul(type a, type b) { return __hmul2(a, b); }
  __device__ __forceinline__ static float2 up(type a) { return __half22float2(a); }
  __device__ __forceinline__ static type pack(float x, float y) {
    return __floats2half2_rn(x, y);
  }
  __device__ __forceinline__ static type bcast(float x) {
    return __half2half2(__float2half(x));
  }
};

// NTAP taps of one 16 B slice.  ``wp`` holds the (already dtype-rounded) weights,
// one per tap, broadcast into both halves of a pair.
template <typename T, int NTAP>
__device__ __forceinline__ uint4 blend_pair(uint4 wp, uint4 r0, uint4 r1,
                                            uint4 r2, uint4 r3) {
  using PT = typename P2<T>::type;
  union U {
    uint4 u;
    PT h[4];
  };
  U w, a, b, c, d, o;
  w.u = wp;
  a.u = r0;
  if (NTAP >= 2) b.u = r1;
  if (NTAP >= 4) {
    c.u = r2;
    d.u = r3;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = P2<T>::up(P2<T>::mul(w.h[0], a.h[i]));
    if (NTAP >= 2) {
      float2 g = P2<T>::up(P2<T>::mul(w.h[1], b.h[i]));
      f.x += g.x;
      f.y += g.y;
    }
    if (NTAP >= 4) {
      float2 g = P2<T>::up(P2<T>::mul(w.h[2], c.h[i]));
      float2 h = P2<T>::up(P2<T>::mul(w.h[3], d.h[i]));
      f.x += g.x;
      f.y += g.y;
      f.x += h.x;
      f.y += h.y;
    }
    o.h[i] = P2<T>::pack(f.x, f.y);
  }
  return o.u;
}

// fp32 output: 4 elements per 16 B, weights are plain floats, and the reference's
// products need no rounding step at all.
template <int NTAP>
__device__ __forceinline__ uint4 blend_f32(uint4 wp, uint4 r0, uint4 r1,
                                           uint4 r2, uint4 r3) {
  union U {
    uint4 u;
    float f[4];
  };
  U w, a, b, c, d, o;
  w.u = wp;
  a.u = r0;
  if (NTAP >= 2) b.u = r1;
  if (NTAP >= 4) {
    c.u = r2;
    d.u = r3;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float acc = w.f[0] * a.f[i];
    if (NTAP >= 2) acc += w.f[1] * b.f[i];
    if (NTAP >= 4) {
      acc += w.f[2] * c.f[i];
      acc += w.f[3] * d.f[i];
    }
    o.f[i] = acc;
  }
  return o.u;
}

template <typename T, int NTAP>
struct Blend {
  __device__ __forceinline__ static uint4 go(uint4 w, uint4 a, uint4 b, uint4 c,
                                             uint4 d) {
    return blend_pair<T, NTAP>(w, a, b, c, d);
  }
};

template <int NTAP>
struct Blend<float, NTAP> {
  __device__ __forceinline__ static uint4 go(uint4 w, uint4 a, uint4 b, uint4 c,
                                             uint4 d) {
    return blend_f32<NTAP>(w, a, b, c, d);
  }
};

// ---------------------------------------------------------------------------
// One program per *distinct* output row.  LANES threads cooperate on a row and
// ROWS rows share a block; a row's 16 B vectors are walked LANES at a time (and
// gridDim.y-strided when the row count alone does not fill the machine).
// Stores use __stwt: the result is never read back, so write-through avoids
// keeping 32 MB of dead lines in L2 (measured ~5% on the hottest case).
// ---------------------------------------------------------------------------
template <typename T, int LANES, int ROWS, int NTAP>
__global__ __launch_bounds__(LANES* ROWS) void fk_vpe(
    const uint4* __restrict__ table, const int4* __restrict__ taps,
    const uint4* __restrict__ wts, const int2* __restrict__ dsti,
    uint4* __restrict__ out, int nvec, long long wrowvec, int nrows) {
  const int rin = (int)(threadIdx.x / LANES);
  const int c = (int)(threadIdx.x - rin * LANES);
  const int row = (int)blockIdx.x * ROWS + rin;
  if (row >= nrows) return;

  // All LANES threads of a group read the same metadata -> L1 broadcast.
  const int4 tp = taps[row];
  const uint4 wv = wts[row];
  const int2 di = dsti[row];

  const uint4* __restrict__ p0 = table + (long long)tp.x * wrowvec;
  const uint4* __restrict__ p1 = table + (long long)tp.y * wrowvec;
  const uint4* __restrict__ p2 = table + (long long)tp.z * wrowvec;
  const uint4* __restrict__ p3 = table + (long long)tp.w * wrowvec;

  const int hw = di.y & 0x00FFFFFF;
  const int t = (int)((unsigned)di.y >> 24);
  uint4* __restrict__ q = out + (long long)di.x * nvec;
  const long long fstride = (long long)hw * nvec;

  const int j0 = c + (int)blockIdx.y * LANES;
  const int jstep = LANES * (int)gridDim.y;
  for (int j = j0; j < nvec; j += jstep) {
    const uint4 res = Blend<T, NTAP>::go(
        wv, p0[j], NTAP >= 2 ? p1[j] : wv, NTAP >= 4 ? p2[j] : wv,
        NTAP >= 4 ? p3[j] : wv);
    // The t frames of an entry are bit-identical: same registers, t stores.
    uint4* __restrict__ w = q + j;
    for (int k = 0; k < t; ++k) {
      __stwt(w, res);
      w += fstride;
    }
  }
}

// ---------------------------------------------------------------------------
// Descriptor path: the same kernel with the metadata *derived* instead of read.
//
// r1 opened with three independent cold loads per row (taps / weights /
// destination, 40 B from three arrays) and the four table pointers were
// data-dependent on the first of them -- two serialized cold-miss rounds in the
// prologue of a 6-10 us kernel, re-paid once per gridDim.y slice, with the
// bench flushing L2 before every call so they are always cold.  None of it was
// information: ``torch.linspace(0, G-1, n)`` is affine, so every tap, every
// weight and the spatial-merge permutation are closed forms of the row index.
// Here a per-entry descriptor (<= 16 entries x 9 words) travels by value in the
// kernel's parameter block -- constant bank, no HBM, no H2D, no dependent load
// -- and the table loads issue in the kernel's first instructions.
//
// Bit-exactness is the constraint.  ATen's CUDA linspace is
//     ind <  steps/2 : start + step*ind
//     ind >= steps/2 : end   - step*(steps-1-ind)
// with ``step`` in fp32 -- and its build *contracts* the second form into an
// FMA (verified against torch for every n in 1..259 plus a spread to 4096 and
// G in {2,16,32,48,64,2304}: the non-fused form differs on 1110 of them, the
// fused form on none).  So ``fk_lin`` uses an explicit ``__fmaf_rn`` while
// everything else stays unfused (-fmad=false), ``step`` is computed on the host
// in fp32 so the device never re-derives it, and the host re-verifies the
// closed form against ``torch.linspace`` per distinct n before using this path.
// ---------------------------------------------------------------------------

#define FK_MAXENT 16
#define FK_NW 9
#define FK_HDR 16

// Per-entry descriptor words (struct-of-arrays, indexed by blockIdx.z).
enum {
  FE_HW = 0,    // h*w: distinct rows in this entry (also the frame stride)
  FE_DST,       // destination row of this entry's local row 0
  FE_T,         // frames: t bit-identical copies, hw rows apart
  FE_H,
  FE_W,
  FE_WM,        // w / spatial_merge_size
  FE_HSTEP,     // fp32 bits of (G-1)/(h-1), computed host-side
  FE_WSTEP,
  FE_RWM,       // fp32 bits of 1/wm
};

// Header words of the packed descriptor (host -> kernel args, not by-value).
enum {
  FH_ROWS = 0,
  FH_NENT,
  FH_MODE,
  FH_M,
  FH_MM,
  FH_RM,
  FH_RMM,
  FH_ENDV,
  FH_GNUM,
};

// Tap modes.  ``linspace`` landing on integers along a dimension makes two of
// the four weights exactly zero, and the reference's dropped addend is
// bf16(0*x) = +/-0.0, which no finite sum notices -- so those taps can go.
enum { FK_M4 = 0, FK_M_H = 1, FK_M_W = 2, FK_M_1 = 3 };

struct FkDesc {
  int v[FK_NW * FK_MAXENT];
};

// Exact (n/d, n%d) for 0 <= n < 2^22 and 1 <= d, given rd = 1/d in fp32.
// fl(n*fl(1/d)) differs from n/d by at most (n/d)*2^-22.5 < 1 over that range,
// so truncation lands within one of the true quotient and a *single* correction
// in each direction is exact -- and stays branchless.  A ``while`` here instead
// of an ``if`` costs ~40 instructions: nvcc cannot bound the trip count, so it
// emits a full integer division (MUFU.RCP + IMAD.HI chain) to compute it.
__device__ __forceinline__ int2 fk_divu(int n, int d, float rd) {
  int q = (int)((float)n * rd);
  int r = n - q * d;
  if (r < 0) {
    --q;
    r += d;
  } else if (r >= d) {
    ++q;
    r -= d;
  }
  return make_int2(q, r);
}

// torch.linspace(0, endv, n)[i], bit for bit (see the note above).
__device__ __forceinline__ float fk_lin(int i, int n, float step, float endv) {
  return (i < (n >> 1)) ? step * (float)i
                        : __fmaf_rn(-step, (float)(n - 1 - i), endv);
}

template <typename T>
struct PackW {
  __device__ __forceinline__ static void set(uint4& q, int k, float v) {
    using PT = typename P2<T>::type;
    union U {
      uint4 u;
      PT h[4];
    };
    U t;
    t.u = q;
    t.h[k] = P2<T>::bcast(v);
    q = t.u;
  }
};

template <>
struct PackW<float> {
  __device__ __forceinline__ static void set(uint4& q, int k, float v) {
    union U {
      uint4 u;
      float f[4];
    };
    U t;
    t.u = q;
    t.f[k] = v;
    q = t.u;
  }
};

template <typename T, int LANES, int ROWS, int MODE>
__global__ __launch_bounds__(LANES* ROWS) void fk_vpe2(
    const uint4* __restrict__ table, uint4* __restrict__ out, const FkDesc desc,
    int nvec, int wrowvec, float endv, int gnum, int m, int mm, float rm,
    float rmm) {
  const int e = blockIdx.z;
  const int hw = desc.v[FE_HW * FK_MAXENT + e];
  const int rin = (int)(threadIdx.x / LANES);
  const int lane = (int)(threadIdx.x - rin * LANES);
  const int row = (int)blockIdx.x * ROWS + rin;
  if (row >= hw) return;

  // Invert the spatial-merge permutation: the destination row is
  // ((a*wm + b)*m + c)*m + d for source position (i, j) = (a*m + c, b*m + d).
  const int wm = desc.v[FE_WM * FK_MAXENT + e];
  const float rwm = __int_as_float(desc.v[FE_RWM * FK_MAXENT + e]);
  const int2 qs = fk_divu(row, mm, rmm);   // q = a*wm + b, rem = c*m + d
  const int2 cd = fk_divu(qs.y, m, rm);    // c, d
  const int2 ab = fk_divu(qs.x, wm, rwm);  // a, b
  const int i = ab.x * m + cd.x;
  const int j = ab.y * m + cd.y;

  const float hidx = fk_lin(i, desc.v[FE_H * FK_MAXENT + e],
                            __int_as_float(desc.v[FE_HSTEP * FK_MAXENT + e]), endv);
  const float widx = fk_lin(j, desc.v[FE_W * FK_MAXENT + e],
                            __int_as_float(desc.v[FE_WSTEP * FK_MAXENT + e]), endv);
  const int hf = (int)hidx;  // .long() truncates; hidx >= 0
  const int wf = (int)widx;
  const int hc = min(hf + 1, gnum - 1);
  const int wc = min(wf + 1, gnum - 1);
  const float dh = hidx - (float)hf;
  const float dw = widx - (float)wf;

  // The reference's expression order and rounding, exactly (-fmad=false keeps
  // the subtractions from contracting into the products).
  const float w11 = dh * dw;
  const float w10 = dh - w11;
  const float w01 = dw - w11;
  const float w00 = (1.0f - dh) - w01;

  const int hfb = hf * gnum;
  const int hcb = hc * gnum;
  const uint4* __restrict__ p0 = table + (long long)(hfb + wf) * wrowvec;
  const uint4* __restrict__ p1 = table;
  const uint4* __restrict__ p2 = table;
  const uint4* __restrict__ p3 = table;
  uint4 wq = make_uint4(0, 0, 0, 0);
  PackW<T>::set(wq, 0, w00);
  if (MODE == FK_M4) {
    PackW<T>::set(wq, 1, w01);
    PackW<T>::set(wq, 2, w10);
    PackW<T>::set(wq, 3, w11);
    p1 = table + (long long)(hfb + wc) * wrowvec;
    p2 = table + (long long)(hcb + wf) * wrowvec;
    p3 = table + (long long)(hcb + wc) * wrowvec;
  } else if (MODE == FK_M_H) {  // dh == 0: w10 == w11 == 0
    PackW<T>::set(wq, 1, w01);
    p1 = table + (long long)(hfb + wc) * wrowvec;
  } else if (MODE == FK_M_W) {  // dw == 0: w01 == w11 == 0
    PackW<T>::set(wq, 1, w10);
    p1 = table + (long long)(hcb + wf) * wrowvec;
  }
  constexpr int NTAP = (MODE == FK_M4) ? 4 : ((MODE == FK_M_1) ? 1 : 2);

  const int t = desc.v[FE_T * FK_MAXENT + e];
  uint4* __restrict__ qout =
      out + (long long)(desc.v[FE_DST * FK_MAXENT + e] + row) * nvec;
  const long long fstride = (long long)hw * nvec;

  const int j0 = lane + (int)blockIdx.y * LANES;
  const int jstep = LANES * (int)gridDim.y;
  for (int v = j0; v < nvec; v += jstep) {
    const uint4 res = Blend<T, NTAP>::go(wq, p0[v], NTAP >= 2 ? p1[v] : wq,
                                         NTAP >= 4 ? p2[v] : wq,
                                         NTAP >= 4 ? p3[v] : wq);
    uint4* __restrict__ o = qout + v;
    for (int k = 0; k < t; ++k) {
      __stwt(o, res);
      o += fstride;
    }
  }
}

// Swept on B200 across LANES in {8,16,32,48,72,144} x threads/block in
// {128..576} x block budget in {1..32} waves, scored by geomean over the five
// benched cases.  LANES=144 (one thread per vector, a whole row per block) ties
// on the big video cases and wins the mixed-w one, but costs a step on the small
// image case, where half a warp per 144-thread block goes to waste; 16/256/8
// wins overall.  kWaves is the block budget before the column loop is split over
// gridDim.y -- below ~8 waves the machine runs short of warps to hide L2 latency.
constexpr int kLanes = 16;
constexpr int kTpb = 256;
constexpr int kRows = kTpb / kLanes;
constexpr int kWaves = 8;

int sm_count() {
  static int n = [] {
    int dev = 0, sms = 148;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms;
  }();
  return n;
}

}  // namespace

at::Tensor fk_vpe_forward(const at::Tensor& table, const at::Tensor& taps,
                          const at::Tensor& wts, const at::Tensor& dsti,
                          int64_t total_rows, int64_t ntap) {
  const c10::cuda::CUDAGuard guard(table.device());
  const int64_t D = table.size(1);
  const int64_t esz = table.element_size();
  const int64_t rowbytes = D * esz;

  at::Tensor out = at::empty({total_rows, D}, table.options());
  const int nrows = (int)taps.size(0);
  if (nrows == 0 || total_rows == 0 || D == 0) return out;

  TORCH_CHECK(rowbytes % 16 == 0, "row pitch must be 16 B-vectorizable");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(table.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "16 B alignment required");

  const int nvec = (int)(rowbytes / 16);
  const long long wrowvec = (long long)(table.stride(0) * esz / 16);

  int gx = (nrows + kRows - 1) / kRows;
  int gy = 1;
  const int budget = sm_count() * kWaves;
  if (gx < budget) {
    const int colblocks = (nvec + kLanes - 1) / kLanes;
    gy = std::max(1, std::min(colblocks, budget / gx));
  }
  const dim3 grid(gx, gy, 1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const uint4* tp = (const uint4*)table.const_data_ptr();
  const int4* ip = (const int4*)taps.const_data_ptr();
  const uint4* wp = (const uint4*)wts.const_data_ptr();
  const int2* dp = (const int2*)dsti.const_data_ptr();
  uint4* op = (uint4*)out.data_ptr();

#define FK_GO(T, NT)                                                     \
  fk_vpe<T, kLanes, kRows, NT><<<grid, kTpb, 0, stream>>>(                \
      tp, ip, wp, dp, op, nvec, wrowvec, nrows)
#define FK_TAPS(T)                                                       \
  if (ntap == 4) {                                                       \
    FK_GO(T, 4);                                                         \
  } else if (ntap == 2) {                                                \
    FK_GO(T, 2);                                                         \
  } else {                                                               \
    FK_GO(T, 1);                                                         \
  }                                                                      \
  break

  switch (table.scalar_type()) {
    case at::kBFloat16: FK_TAPS(__nv_bfloat16);
    case at::kHalf: FK_TAPS(__half);
    case at::kFloat: FK_TAPS(float);
    default: TORCH_CHECK(false, "unsupported dtype ", table.scalar_type());
  }
#undef FK_TAPS
#undef FK_GO
  return out;
}

// Descriptor launch.  ``desc`` is one small CPU int32 buffer, built once per
// distinct (grid_thw_list, dtype, device) and handed back by the host memo: a
// header of scalars plus the per-entry struct-of-arrays, which is copied into
// the by-value kernel parameter and so reaches the kernel through the constant
// bank.  Nothing about the geometry lives in device memory any more.
at::Tensor fk_vpe2_forward(const at::Tensor& table, const at::Tensor& desc) {
  TORCH_CHECK(desc.is_cpu() && desc.scalar_type() == at::kInt && desc.is_contiguous() &&
                  desc.numel() == FK_HDR + FK_NW * FK_MAXENT,
              "malformed descriptor");
  const int* p = desc.const_data_ptr<int>();
  const int64_t total_rows = (int64_t)p[FH_ROWS];
  const int nent = p[FH_NENT];
  const int mode = p[FH_MODE];

  const c10::cuda::CUDAGuard guard(table.device());
  const int64_t D = table.size(1);
  const int64_t esz = table.element_size();
  const int64_t rowbytes = D * esz;

  at::Tensor out = at::empty({total_rows, D}, table.options());
  if (nent == 0 || total_rows == 0 || D == 0) return out;
  TORCH_CHECK(rowbytes % 16 == 0, "row pitch must be 16 B-vectorizable");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(table.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "16 B alignment required");

  FkDesc dd;
  std::memcpy(dd.v, p + FK_HDR, sizeof(dd.v));

  const int nvec = (int)(rowbytes / 16);
  const int wrowvec = (int)(table.stride(0) * esz / 16);

  int gx = 1;
  for (int e = 0; e < nent; ++e)
    gx = std::max(gx, (dd.v[FE_HW * FK_MAXENT + e] + kRows - 1) / kRows);
  int gy = 1;
  const int budget = sm_count() * kWaves;
  const int blocks = gx * nent;
  if (blocks < budget) {
    const int colblocks = (nvec + kLanes - 1) / kLanes;
    gy = std::max(1, std::min(colblocks, budget / blocks));
  }
  const dim3 grid(gx, gy, nent);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define FK_GO2(T, MD)                                                        \
  fk_vpe2<T, kLanes, kRows, MD><<<grid, kTpb, 0, stream>>>(                   \
      (const uint4*)table.const_data_ptr(), (uint4*)out.data_ptr(), dd, nvec, \
      wrowvec, __int_as_float_h(p[FH_ENDV]), p[FH_GNUM], p[FH_M], p[FH_MM],   \
      __int_as_float_h(p[FH_RM]), __int_as_float_h(p[FH_RMM]))
#define FK_MODES(T)                                                          \
  if (mode == FK_M4) {                                                       \
    FK_GO2(T, FK_M4);                                                        \
  } else if (mode == FK_M_H) {                                               \
    FK_GO2(T, FK_M_H);                                                       \
  } else if (mode == FK_M_W) {                                               \
    FK_GO2(T, FK_M_W);                                                       \
  } else {                                                                   \
    FK_GO2(T, FK_M_1);                                                       \
  }                                                                          \
  break

  switch (table.scalar_type()) {
    case at::kBFloat16: FK_MODES(__nv_bfloat16);
    case at::kHalf: FK_MODES(__half);
    case at::kFloat: FK_MODES(float);
    default: TORCH_CHECK(false, "unsupported dtype ", table.scalar_type());
  }
#undef FK_MODES
#undef FK_GO2
  return out;
}
"""

_DECL = (
    "#include <torch/extension.h>\n"
    "at::Tensor fk_vpe_forward(const at::Tensor&, const at::Tensor&,\n"
    "                          const at::Tensor&, const at::Tensor&, int64_t,\n"
    "                          int64_t);\n"
    "at::Tensor fk_vpe2_forward(const at::Tensor&, const at::Tensor&);\n"
)

_NAME = "fk_vpe_" + hashlib.sha1(_CUDA.encode()).hexdigest()[:12]

# Build for the GPU actually present: the default arch list is 7 targets
# (~90 s of nvcc) versus ~12 s for one, and this compiles inside the bench
# worker on a cold cache.
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    try:
        _cc = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"
    except Exception:  # noqa: BLE001 - fall back to torch's default list
        pass

_EXT = load_inline(
    name=_NAME,
    cpp_sources=_DECL,
    cuda_sources=_CUDA,
    functions=["fk_vpe_forward", "fk_vpe2_forward"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-fmad=false"],
    verbose=False,
)

_RUN = _EXT.fk_vpe_forward
_RUN2 = _EXT.fk_vpe2_forward

# Descriptor layout, mirroring the enums in the CUDA source above.
_MAXENT, _NW, _HDR = 16, 9, 16
(_FE_HW, _FE_DST, _FE_T, _FE_H, _FE_W, _FE_WM, _FE_HSTEP, _FE_WSTEP,
 _FE_RWM) = range(_NW)
(_FH_ROWS, _FH_NENT, _FH_MODE, _FH_M, _FH_MM, _FH_RM, _FH_RMM, _FH_ENDV,
 _FH_GNUM) = range(9)
_M4, _M_H, _M_W, _M_1 = 0, 1, 2, 3


def _fbits(x) -> int:
    return int(np.float32(x).view(np.int32))


def _lin_np(n: int, endv: int) -> np.ndarray:
    """``torch.linspace(0, endv, n, dtype=float32, device='cuda')`` in closed
    form -- the same expression the kernel evaluates, including ATen's
    fma-contracted upper half (emulated here in float64, which is exact for a
    product of two fp32 values plus an fp32 addend)."""
    e = np.float32(endv)
    if n == 1:
        return np.zeros(1, dtype=np.float32)
    step = np.float32(e / np.float32(n - 1))
    i = np.arange(n, dtype=np.int64)
    lo = (step * i.astype(np.float32)).astype(np.float32)
    k = (n - 1 - i).astype(np.float32)
    hi = (np.float64(-step) * k.astype(np.float64) + np.float64(e)).astype(np.float32)
    return np.where(i < n // 2, lo, hi).astype(np.float32)


# (n, endv) -> fractional parts; (n, endv, device) -> does the closed form match
# torch bit for bit.  A numerics surprise on some n costs the descriptor path,
# not correctness: the caller falls back to the memoized-metadata kernel.
_FRAC: dict = {}
_LIN_OK: dict = {}


def _frac(n: int, endv: int) -> np.ndarray:
    key = (n, endv)
    f = _FRAC.get(key)
    if f is None:
        v = _lin_np(n, endv)
        f = _FRAC[key] = (v - np.floor(v)).astype(np.float32)
    return f


def _lin_ok(n: int, endv: int, device) -> bool:
    key = (n, endv, device.type, device.index)
    ok = _LIN_OK.get(key)
    if ok is None:
        ref = torch.linspace(0, endv, n, dtype=torch.float32,
                             device=device).cpu().numpy()
        ok = _LIN_OK[key] = bool(np.array_equal(ref.view(np.int32),
                                                _lin_np(n, endv).view(np.int32)))
    return ok



class VisionPosEmbedInterpolate(nn.Module):
    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        self._embed = Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size
        # Plain dict entries (not buffers): metadata is derived, never trained,
        # and must not leak into state_dict.
        self.__dict__["_meta"] = {}
        self.__dict__["_w"] = self._embed.emb.weight

    # ``.to()`` / ``.cuda()`` / ``.half()`` replace the Parameter object and can
    # move the table to another device, which invalidates every cached (device
    # resident) metadata bundle.
    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self.__dict__["_w"] = self._embed.emb.weight
        self.__dict__["_meta"] = {}
        return out

    # ``load_state_dict(..., assign=True)`` swaps the Parameter object, and the
    # memo's payload holds a direct reference to it -- so drop the memo too.
    def load_state_dict(self, *args, **kwargs):
        out = super().load_state_dict(*args, **kwargs)
        self.__dict__["_w"] = self._embed.emb.weight
        self.__dict__["_meta"] = {}
        return out

    # ------------------------------------------------------------------
    # Host-side metadata: one bundle per distinct (grid_thw_list, dtype, device)
    # ------------------------------------------------------------------
    def _build_meta(self, grid_thw_list, dtype, device):
        """Return (taps, wts, dsti, total_rows, ntap) or ``False`` if the fused
        path does not apply to this call."""
        table = self._w
        if (table.dim() != 2 or not table.is_cuda or table.dtype != dtype
                or not table.is_contiguous() or table.data_ptr() % 16
                or table.size(1) == 0 or (table.size(1) * table.element_size()) % 16
                or dtype not in (torch.bfloat16, torch.float16, torch.float32)):
            return False
        if getattr(device, "type", None) != "cuda" or not grid_thw_list:
            return False
        # An index-less ``torch.device("cuda")`` means the current device, which
        # is the table's device in every sane setup -- resolve it rather than
        # failing the identity test and silently dropping to the slow path.
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        if idx != table.device.index:
            return False

        num_grid = self.num_grid_per_side
        m = self.spatial_merge_size
        if num_grid < 1 or m < 1 or num_grid * num_grid > table.size(0):
            return False

        entries = []
        total = 0
        for ent in grid_thw_list:
            if len(ent) != 3:
                return False
            t, h, w = int(ent[0]), int(ent[1]), int(ent[2])
            # Outside this regime the reference either raises (a reshape that
            # does not divide) or needs a different layout; hand it back.
            if t < 1 or t > 127 or h < 1 or w < 1 or h % m or w % m:
                return False
            if h * w >= (1 << 24):
                return False
            entries.append((t, h, w))
            total += t * h * w
        if total >= (1 << 31):
            return False

        taps_l, wts_l, dst_l = [], [], []
        off = 0
        for t, h, w in entries:
            # Same ops, same device as the reference -> bit-identical indices.
            h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32,
                                    device=device)
            w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32,
                                    device=device)
            h_floor = h_idxs.long()
            w_floor = w_idxs.long()
            h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)

            dh = (h_idxs - h_floor).unsqueeze(1)          # (h, 1)
            dw = (w_idxs - w_floor).unsqueeze(0)          # (1, w)
            w11 = dh * dw                                 # (h, w)
            w10 = dh - w11
            w01 = dw - w11
            w00 = 1 - dh - w01

            hf = (h_floor * num_grid).unsqueeze(1)
            hc = (h_ceil * num_grid).unsqueeze(1)
            wf = w_floor.unsqueeze(0)
            wc = w_ceil.unsqueeze(0)
            taps = torch.stack([hf + wf, hf + wc, hc + wf, hc + wc], dim=-1)
            # Rounded to dtype exactly as the reference's ``.to(dtype)`` does, so
            # the kernel's weight is the reference's weight bit for bit.
            wts = torch.stack([w00, w01, w10, w11], dim=-1).to(dtype)

            # The spatial-merge reshuffle, applied to the *metadata* (h*w*4
            # values) instead of the embeddings (h*w*hidden values).
            hm, wm = h // m, w // m
            perm = (0, 2, 1, 3, 4)
            taps = (taps.reshape(hm, m, wm, m, 4).permute(*perm)
                    .reshape(h * w, 4).to(torch.int32))
            wts = wts.reshape(hm, m, wm, m, 4).permute(*perm).reshape(h * w, 4)

            n = h * w
            rows = torch.arange(off, off + n, dtype=torch.int32, device=device)
            dst = torch.stack([rows, torch.full_like(rows, (t << 24) | n)], dim=1)

            taps_l.append(taps)
            wts_l.append(wts)
            dst_l.append(dst)
            off += t * n

        taps = taps_l[0] if len(taps_l) == 1 else torch.cat(taps_l, 0)
        wts = wts_l[0] if len(wts_l) == 1 else torch.cat(wts_l, 0)
        dst = dst_l[0] if len(dst_l) == 1 else torch.cat(dst_l, 0)

        # Tap compaction.  When linspace lands exactly on integers along a
        # dimension its fractional part is 0 for every row, so two of the four
        # weights are exactly zero -- e.g. h == num_grid == 48 for the
        # ``[[1,48,64]]`` capture.  The reference still adds those terms, but the
        # addend is bf16(0 * x) = +/-0.0 and ``y + (+/-0.0) == y`` for every finite
        # y (including y == 0, where only the sign of the zero can differ, which
        # compares equal), so dropping them is bit-safe -- and it halves the
        # table reads.  Column order is preserved so the surviving terms still
        # accumulate in the reference's order.
        z1 = not bool(wts[:, 1].any())
        z2 = not bool(wts[:, 2].any())
        z3 = not bool(wts[:, 3].any())
        if z1 and z2 and z3:
            ntap, order = 1, (0, 1, 2, 3)
        elif z2 and z3:
            ntap, order = 2, (0, 1, 2, 3)
        elif z1 and z3:
            ntap, order = 2, (0, 2, 1, 3)
        else:
            ntap, order = 4, (0, 1, 2, 3)
        if order != (0, 1, 2, 3):
            idx = torch.tensor(order, device=device)
            taps = taps.index_select(1, idx)
            wts = wts.index_select(1, idx)

        # Weight layout, 16 B per row either way: for the 2-byte dtypes each
        # weight is broadcast into both halves of a pair so the kernel can use a
        # packed multiply; fp32 needs no packing.
        if dtype is not torch.float32:
            r = wts.size(0)
            wts = wts.unsqueeze(-1).expand(r, 4, 2).reshape(r, 8)

        return (taps.contiguous(), wts.contiguous(), dst.contiguous(), total, ntap)

    # ------------------------------------------------------------------
    # Descriptor: the whole geometry as ~150 B of kernel *parameters*
    # ------------------------------------------------------------------
    def _build_desc(self, grid_thw_list, dtype, device):
        """One small CPU int32 buffer -- header scalars plus a per-entry
        struct-of-arrays -- or ``None`` when the arithmetic path does not apply
        and the memoized-metadata kernel should be used instead."""
        table = self._w
        if (table.dim() != 2 or not table.is_cuda or table.dtype != dtype
                or not table.is_contiguous() or table.data_ptr() % 16
                or table.size(1) == 0 or (table.size(1) * table.element_size()) % 16
                or dtype not in (torch.bfloat16, torch.float16, torch.float32)):
            return None
        if getattr(device, "type", None) != "cuda" or not grid_thw_list:
            return None
        if len(grid_thw_list) > _MAXENT:
            return None
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        if idx != table.device.index:
            return None

        num_grid = self.num_grid_per_side
        m = self.spatial_merge_size
        # num_grid < 2 would make the linspace step degenerate (n == 1 too, so
        # h and w of 1 go to the metadata path, which builds them with torch).
        if num_grid < 2 or m < 1 or num_grid * num_grid > table.size(0):
            return None

        entries, total = [], 0
        for ent in grid_thw_list:
            if len(ent) != 3:
                return None
            t, h, w = int(ent[0]), int(ent[1]), int(ent[2])
            if t < 1 or h < 2 or w < 2 or h % m or w % m:
                return None
            # fk_divu's float quotient is exact to +/-1 only below 2^22.
            if h * w >= (1 << 22):
                return None
            entries.append((t, h, w))
            total += t * h * w
        if total >= (1 << 31):
            return None

        endv = num_grid - 1
        zh = zw = True
        for t, h, w in entries:
            if not _lin_ok(h, endv, device) or not _lin_ok(w, endv, device):
                return None
            zh = zh and not _frac(h, endv).any()
            zw = zw and not _frac(w, endv).any()
        # Tap compaction: an integral linspace along a dimension zeroes two of
        # the four weights, and the reference's dropped addend is bf16(0*x) =
        # +/-0.0, which no finite sum notices.  Only fires when it holds for
        # *every* entry; the surviving terms keep the reference's order.
        mode = _M_1 if (zh and zw) else _M_H if zh else _M_W if zw else _M4

        d = np.zeros(_HDR + _NW * _MAXENT, dtype=np.int32)
        d[_FH_ROWS] = total
        d[_FH_NENT] = len(entries)
        d[_FH_MODE] = mode
        d[_FH_M] = m
        d[_FH_MM] = m * m
        d[_FH_RM] = _fbits(np.float32(1.0) / np.float32(m))
        d[_FH_RMM] = _fbits(np.float32(1.0) / np.float32(m * m))
        d[_FH_ENDV] = _fbits(endv)
        d[_FH_GNUM] = num_grid
        e = d[_HDR:].reshape(_NW, _MAXENT)
        off = 0
        for k, (t, h, w) in enumerate(entries):
            e[_FE_HW, k] = h * w
            e[_FE_DST, k] = off
            e[_FE_T, k] = t
            e[_FE_H, k] = h
            e[_FE_W, k] = w
            e[_FE_WM, k] = w // m
            e[_FE_HSTEP, k] = _fbits(np.float32(endv) / np.float32(h - 1))
            e[_FE_WSTEP, k] = _fbits(np.float32(endv) / np.float32(w - 1))
            e[_FE_RWM, k] = _fbits(np.float32(1.0) / np.float32(w // m))
            off += t * h * w
        return torch.from_numpy(d)

    def _prepare(self, grid_thw_list, dtype, device):
        """The memo's payload: a callable plus its complete argument tuple, so a
        warm call is one dict lookup and one extension call.  ``False`` means
        the reference path."""
        desc = self._build_desc(grid_thw_list, dtype, device)
        if desc is not None:
            return (_RUN2, (self._w, desc))
        meta = self._build_meta(grid_thw_list, dtype, device)
        if meta is False:
            return False
        return (_RUN, (self._w,) + meta)

    def forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        meta = self._meta
        # The table dtype is part of the key: ``_prepare_module``-style casts
        # (``p.data = p.data.to(...)``) replace the storage without going through
        # ``_apply``, so a cached verdict must not outlive them.
        key = (tuple(map(tuple, grid_thw_list)), dtype, device, self._w.dtype)
        ent = meta.get(key)
        if ent is None:
            if len(meta) > 512:
                meta.clear()
            ent = meta[key] = self._prepare(grid_thw_list, dtype, device)
        if ent is not False:
            return ent[0](*ent[1])
        return self._forward_reference(grid_thw_list, dtype, device)

    # ------------------------------------------------------------------
    # Reference path, verbatim: the fused kernel's regime is a subset.
    # ------------------------------------------------------------------
    def _forward_reference(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_grid = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.hidden_size

        outputs = []
        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32, device=device)

            h_floor = h_idxs.long()
            w_floor = w_idxs.long()
            h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            indices = (h_grid * num_grid + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)

            embeds = self._embed(indices) * weights
            combined = embeds.sum(dim=0)
            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            ).permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)
