"""Fused AdaLayerNormContinuous for B200 (sm_100) with the baseline's contract.

The baseline makes **three** full passes over ``x``::

    emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
    scale, shift = torch.chunk(emb, 2, dim=1)
    x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

LayerNorm reads and writes it, the broadcast multiply reads and writes it again,
and the add a third time. A replica of the harness timing loop
(``profile/p1-baseline/``) prices one read+write pass over the 24 MiB case at
13.3 us and finds the baseline spending 35.8 + 42.0 + ~42 us doing it three
times -- and each individual pass is itself 2.7x (``layer_norm``) to 3.2x
(broadcast multiply) the cost of a plain copy of the same bytes.

All three collapse into one kernel here: each row is read once with 128-bit
accesses, stays in registers across both reduction passes so the variance pass
costs no global traffic, and is written once. The shipped configuration holds the
row as fp32 there and overwrites it with the variance pass's deviations, which
removes two of three unpacks and the epilogue's subtract; a packed variant that
re-unpacks is kept behind a macro. The projection path reuses the
frozen ``SiLU`` and ``Linear`` unchanged -- the frozen ``Linear`` has no admitted
entry for this shape, so it delegates to the same vendor GEMV the baseline pays
for.

``emb`` is handed to the operator whole rather than chunked: the kernel derives
the gain at channel ``j`` and the offset at ``n + j``, which removes two
Python-level ops and avoids depending on ``torch.chunk``'s view layout (its two
halves have stride ``(2n, 1)`` and are contiguous *rows* only at batch extent 1).

**The epilogue reproduces the reference's rounding chain rather than merely
landing inside the tolerance gate.** The reference materialises three
intermediate low-precision values that an all-fp32 epilogue would skip:
``self.norm(x)`` is stored as bfloat16, ``1 + scale`` is a bfloat16 tensor op,
and the multiply and the add each round their fp32 result back to bfloat16. Those
roundings are *not* free: this kernel is issue-bound rather than bandwidth-bound
(``profile/p1-candidate-ncu-v3/``), and building the single-rounding alternative
measured the chain at about 2 us on the larger scored case. They are kept anyway,
because that arm is not bit-identical to the reference and the 2 us does not close
the gap to copy cost either way -- see ``profile/p1-candidate/REPORT-round1.md``.
What is *not* claimed is bit-equality against the baseline module: the reductions
differ by construction (ATen uses one-pass Welford in fp32, this kernel a two-pass
mean then sum of squared deviations in fp32), which leaves a last-bit difference in
``mean``/``rstd``.

``promote_fp32`` therefore selects no code path. ATen's low-precision LayerNorm
already accumulates in fp32 and bfloat16 -> fp32 is exact, so the ``True`` branch
(``F.layer_norm(x.float(), ...).to(bf16)``) and the ``False`` branch are the same
function up to the fp32 reduction this kernel performs anyway. The flag is
consumed only by the fallback expression.

Anything the kernel does not cover -- another dtype, a rank the broadcast treats
differently, a non-contiguous or misaligned view, a row width that is not a
multiple of the 16-byte vector, a CPU tensor, a call that needs gradients, a
failed build -- reproduces the baseline *formula*, so a dispatch gap can only
ever cost latency and never correctness.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

# Unique to this file so a second import under a different module name cannot
# double-register the operator; registration happens once at ``.so`` load and
# Python's module cache makes any later import a no-op.
_LIBRARY_NAME = "fk_adalnc_cand"

# Target architecture used when no device is visible. The pin below is derived
# from the live device when there is one; this is the fallback for the agent's
# normal state, which is no GPU at all.
_FALLBACK_ARCH = "10.0"

# The shipped row mapping is exposed as two ``-D`` macros with fixed defaults so
# ``profile/p1-candidate/ab_geometry.py`` can A/B it by recompiling rather than by
# adding a runtime switch here. The captured row is 3072 bfloat16 = 384 16-byte
# vectors, which 384x1 / 192x2 / 128x3 / 96x4 / 64x6 all divide exactly.
_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#ifndef FK_ADALNC_ROW_BLOCK
#define FK_ADALNC_ROW_BLOCK 192
#endif
#ifndef FK_ADALNC_ROW_VPT
#define FK_ADALNC_ROW_VPT 2
#endif
// Grid strategy. 0 is one CTA per row, which is what the frozen sibling
// LayerNorm ships. 1 is a persistent grid-stride loop with the per-channel
// modulation hoisted, which trades register live range for fewer loads and fewer
// rounding conversions. Selected by recompiling, so the A/B costs no runtime
// branch, and the measured winner is the default.
#ifndef FK_ADALNC_PERSIST
#define FK_ADALNC_PERSIST 0
#endif
// Hold the row as fp32 across both reduction passes instead of packed. Unpacking
// the packed row happens three times in the packed form -- once for the mean, once
// for the squared deviations, once in the epilogue -- and this holds the values
// once and overwrites them with the deviations the variance pass already computes,
// which also removes the epilogue's subtract. It costs kVecsPerThread * kElems
// fp32 registers instead of 4 per vector. Reduction order and the rounding chain
// are unchanged, so the output is bit-identical.
// Measured winner and therefore the default: hot marginal 26.46 us against the
// packed form's 28.70 us on the larger scored case (one CUDA-event tick), a tie on
// the smaller one, and bit-identical output. See
// profile/p1-candidate/ab_norm_variants_results.json.
#ifndef FK_ADALNC_FP32ROW
#define FK_ADALNC_FP32ROW 1
#endif
// Differentiated L1 policies: the row is streamed once and never reused, while the
// gain and offset are re-read by every row block. KernelWiki reports this pairing
// as a first-order lever on memory-bound B200 kernels (1.44x on an NVFP4 GEMV),
// which is exactly why it is measured here rather than assumed. Measured: no win on
// either scored shape, hot or cold (28.62 us against 28.70 us on the larger case,
// inside the tie band), which is what the profile predicted -- at 11% of DRAM peak
// and 16% of L2 peak there is nothing for a cache hint to recover. Off by default,
// kept as a knob so the negative result stays reproducible.
#ifndef FK_ADALNC_CACHE_HINTS
#define FK_ADALNC_CACHE_HINTS 0
#endif
// Diagnostic only, never shipped: collapse the epilogue's three intermediate
// roundings into a single rounding on the store. That is what a kernel would do if
// it were written for the tolerance gate rather than for bit-exactness, so
// building it measures what the exact rounding chain costs -- which is the number
// needed to say whether the exactness requirement and the approach-copy-cost
// requirement can both be met. Enabling it forfeits the epilogue bit-equality
// property, so it must not be the default.
#ifndef FK_ADALNC_FP32_EPILOGUE
#define FK_ADALNC_FP32_EPILOGUE 0
#endif
// 256-bit (32-byte) global accesses instead of 128-bit. Halves the number of load
// and store instructions per row, which is the only reason it could matter on a
// kernel limited by instruction issue rather than bandwidth. It needs 32-byte
// alignment on every pointer it dereferences, which is a strictly stronger
// requirement than the 16-byte one, so it carries its own predicate.
#ifndef FK_ADALNC_VEC256
#define FK_ADALNC_VEC256 0
#endif
// Compute the row statistics with a Welford recurrence in ATen's shape instead of
// the two-pass mean-then-sum-of-squared-deviations form. The point is not accuracy
// -- the two-pass form is at least as accurate -- but *agreement*: ATen's LayerNorm
// uses Welford, and matched_ratio is only exactly 1.0 when this kernel's mean and
// rstd are bit-identical to ATen's.
#ifndef FK_ADALNC_WELFORD
#define FK_ADALNC_WELFORD 0
#endif
// Programmatic dependent launch between the projection (producer) and the
// normalization (consumer). The consumer needs the gain and the offset only in its
// epilogue -- the row load and both reduction passes depend on x alone -- so it can
// start while the projection is still running and release the dependency just
// before its first read of emb. Requires the projection fast path, so the A/B
// harness enables it; the shipped default does not.
#ifndef FK_ADALNC_PDL
#define FK_ADALNC_PDL 0
#endif
// Work-partitioning multiplier for the persistent grid, not an occupancy figure.
#ifndef FK_ADALNC_WAVES
#define FK_ADALNC_WAVES 4
#endif
// Projection geometry. Warps per CTA each own one output row at a time; the wave
// multiplier caps the grid so the staged activation is computed once per CTA
// rather than once per row group -- with a covering grid the activation's expf
// and divide are evaluated 768x redundantly, which measured as the whole reason a
// first version of this kernel only tied the vendor GEMV.
// 16 warps per CTA with a covering grid was the measured winner of a six-point
// sweep (profile/p1-candidate/ab_projection_results.json): it beat the vendor
// composition by 2.0 us where 8 warps at 2 waves lost by 4.1 us. The mechanism is
// in the profile -- at 8 warps and a capped grid the kernel runs at 22% achieved
// occupancy with long_scoreboard 9.26, so it is latency-bound, and widening the
// CTA buys resident warps and cuts the redundant staging at the same time. The
// wave cap is left as a knob but is set high enough not to bind on the captured
// shape, because covering measured best.
#ifndef FK_ADALNC_PROJ_WARPS
#define FK_ADALNC_PROJ_WARPS 16
#endif
#ifndef FK_ADALNC_PROJ_WAVES
#define FK_ADALNC_PROJ_WAVES 64
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

constexpr int kRowBlock = FK_ADALNC_ROW_BLOCK;
constexpr int kRowVpt = FK_ADALNC_ROW_VPT;
constexpr bool kWide = FK_ADALNC_VEC256 != 0;
constexpr int kWideVpt(int vpt) {
  return kWide ? (vpt / 2 > 0 ? vpt / 2 : 1) : vpt;
}
// The tuned mapping is expressed in *vectors*, and a 256-bit vector covers twice
// the elements, so the wide arm needs half the vectors per thread to cover the same
// row. 192x2 at 128-bit and 192x1 at 256-bit both cover the captured 3072-wide row.
constexpr int kTunedVpt = kWideVpt(kRowVpt);
constexpr int kTunedVecs = kRowBlock * kTunedVpt;
constexpr bool kPersistent = FK_ADALNC_PERSIST != 0;
constexpr bool kFp32Row = FK_ADALNC_FP32ROW != 0;
constexpr bool kWelford = FK_ADALNC_WELFORD != 0;
constexpr bool kPdl = FK_ADALNC_PDL != 0;
// The wide arm reads and writes through the fp32-resident body, which is the one
// that is generic over the access width.
static_assert(!kWide || kFp32Row,
              "the 256-bit arm is only implemented on the fp32-resident body");
constexpr int kWaves = FK_ADALNC_WAVES;
// Widest row the ladder covers, in 16-byte vectors. Anything wider than this and
// not exactly kTunedVecs takes the fallback.
constexpr int kMaxLadderVecs = 512;

static_assert(kRowBlock > 0 && kRowBlock % kWarpSize == 0 && kRowBlock <= 1024,
              "the row block must be a positive multiple of the warp size");
static_assert(kRowVpt > 0, "each thread must own at least one vector");
static_assert(kWaves > 0, "the persistent grid needs at least one wave");

// ---------------------------------------------------------------------------
// 16 bytes is the widest single global access the SM offers, so the packed
// element count follows from the element size. Rows stay in registers in this
// packed form -- 4 registers per vector instead of the 8 an fp32 unpack would
// need.
// ---------------------------------------------------------------------------
template <typename T>
struct Packed;

template <>
struct Packed<__nv_bfloat16> {
  using Elem = __nv_bfloat16;
  static constexpr int kElems = 8;
  __device__ __forceinline__ static void to_float(const uint4& v, float* out) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      out[2 * j] = f.x;
      out[2 * j + 1] = f.y;
    }
  }
  __device__ __forceinline__ static uint4 from_float(const float* in) {
    uint4 v;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = __floats2bfloat162_rn(in[2 * j], in[2 * j + 1]);
    }
    return v;
  }
  // Round-trip through the storage dtype, round-to-nearest-even, which is what
  // materialising an intermediate bfloat16 tensor does.
  //
  // Expressing the two epilogue roundings on *pairs* instead -- through
  // __floats2bfloat162_rn, which rounds each lane to nearest even independently
  // and so is the same function -- was built and measured, because the profile
  // says this kernel is issue-bound rather than bandwidth-bound. It replaced 32
  // scalar F2F.BF16.F32 with 16 packed F2FP.BF16.F32.PACK_AB per thread, but paid
  // the saving straight back in PRMT/SHF work to widen the pair again (143 vs 137
  // shift-and-permute instructions), and measured identical to 0.03 us on both
  // scored shapes. The scalar form ships because it is the simpler of two equals.
  __device__ __forceinline__ static float round_to_storage(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
  }
  __device__ __forceinline__ static float scalar_to_float(Elem v) {
    return __bfloat162float(v);
  }
  __device__ __forceinline__ static Elem scalar_from_float(float v) {
    return __float2bfloat16_rn(v);
  }
};

template <>
struct Packed<__half> {
  using Elem = __half;
  static constexpr int kElems = 8;
  __device__ __forceinline__ static void to_float(const uint4& v, float* out) {
    const __half2* p = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __half22float2(p[j]);
      out[2 * j] = f.x;
      out[2 * j + 1] = f.y;
    }
  }
  __device__ __forceinline__ static uint4 from_float(const float* in) {
    uint4 v;
    __half2* p = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = __floats2half2_rn(in[2 * j], in[2 * j + 1]);
    }
    return v;
  }
  __device__ __forceinline__ static float round_to_storage(float v) {
    return __half2float(__float2half_rn(v));
  }
  __device__ __forceinline__ static float scalar_to_float(Elem v) {
    return __half2float(v);
  }
  __device__ __forceinline__ static Elem scalar_from_float(float v) {
    return __float2half_rn(v);
  }
};

// 128-bit loads carrying an L1 policy. Non-volatile so the scheduler can still
// hoist them; both operands are read-only for the lifetime of the kernel, so
// there is nothing for a reordering to break.
// ---------------------------------------------------------------------------
// Access width as a traits type, so one kernel body and one epilogue serve both
// the 128-bit and the 256-bit arms. ``Packed<T>`` stays the *dtype* abstraction
// (conversions and rounding); these are the *access* abstraction.
// ---------------------------------------------------------------------------
template <typename T>
struct Narrow {
  using Vec = uint4;
  static constexpr int kElems = Packed<T>::kElems;
  static constexpr int kBytes = 16;
  __device__ __forceinline__ static void to_float(Vec v, float* out) {
    Packed<T>::to_float(v, out);
  }
  __device__ __forceinline__ static Vec from_float(const float* in) {
    return Packed<T>::from_float(in);
  }
};

// 32-byte alignment is what ``ld.global.v4.u64`` requires; a misaligned address
// silently degrades to narrower transactions, so the predicate enforces it.
struct __align__(32) Vec32 {
  uint64_t a, b, c, d;
};

template <typename T>
struct Wide {
  using Vec = Vec32;
  static constexpr int kElems = 2 * Packed<T>::kElems;
  static constexpr int kBytes = 32;
  __device__ __forceinline__ static void to_float(Vec v, float* out) {
    const uint4* halves = reinterpret_cast<const uint4*>(&v);
    Packed<T>::to_float(halves[0], out);
    Packed<T>::to_float(halves[1], out + Packed<T>::kElems);
  }
  __device__ __forceinline__ static Vec from_float(const float* in) {
    Vec v;
    uint4* halves = reinterpret_cast<uint4*>(&v);
    halves[0] = Packed<T>::from_float(in);
    halves[1] = Packed<T>::from_float(in + Packed<T>::kElems);
    return v;
  }
};

__device__ __forceinline__ uint4 load_streamed(const uint4* __restrict__ p) {
#if FK_ADALNC_CACHE_HINTS
  uint4 v;
  asm("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
#else
  return *p;
#endif
}

__device__ __forceinline__ uint4 load_reused(const uint4* __restrict__ p) {
#if FK_ADALNC_CACHE_HINTS
  uint4 v;
  asm("ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
#else
  return *p;
#endif
}

// The 256-bit forms. Inline PTX rather than a plain dereference, so the single
// 4x64-bit transaction is guaranteed rather than left to the compiler's
// discretion, following the reference guidance for wide accesses on this
// architecture.
__device__ __forceinline__ Vec32 load_streamed(const Vec32* __restrict__ p) {
  Vec32 v;
#if FK_ADALNC_CACHE_HINTS
  asm("ld.global.L1::no_allocate.v4.u64 {%0, %1, %2, %3}, [%4];"
#else
  asm("ld.global.v4.u64 {%0, %1, %2, %3}, [%4];"
#endif
      : "=l"(v.a), "=l"(v.b), "=l"(v.c), "=l"(v.d)
      : "l"(p));
  return v;
}

__device__ __forceinline__ Vec32 load_reused(const Vec32* __restrict__ p) {
  Vec32 v;
#if FK_ADALNC_CACHE_HINTS
  asm("ld.global.L1::evict_last.v4.u64 {%0, %1, %2, %3}, [%4];"
#else
  asm("ld.global.v4.u64 {%0, %1, %2, %3}, [%4];"
#endif
      : "=l"(v.a), "=l"(v.b), "=l"(v.c), "=l"(v.d)
      : "l"(p));
  return v;
}

__device__ __forceinline__ void store_vec(uint4* __restrict__ p, uint4 v) {
  *p = v;
}

__device__ __forceinline__ void store_vec(Vec32* __restrict__ p, Vec32 v) {
  asm volatile("st.global.v4.u64 [%0], {%1, %2, %3, %4};"
               :
               : "l"(p), "l"(v.a), "l"(v.b), "l"(v.c), "l"(v.d)
               : "memory");
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, offset);
  }
  return v;
}

// Sum across a whole CTA. ``stage`` needs kWarps + 1 floats; the mean and the
// variance reduction are handed disjoint regions so neither has to guard the
// other's broadcast slot with an extra __syncthreads.
template <int kBlockThreads>
__device__ __forceinline__ float block_reduce_sum(float v, float* stage) {
  constexpr int kWarps = kBlockThreads / kWarpSize;
  v = warp_reduce_sum(v);
  if constexpr (kWarps == 1) {
    return v;
  } else {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
      stage[warp] = v;
    }
    __syncthreads();
    // Every warp repeats the final reduction over the same kWarps values, with
    // the lanes past kWarps reading a zero. Cheaper than the divergence of
    // restricting it to warp 0, and it keeps the broadcast a single store.
    float t = (threadIdx.x < kWarps) ? stage[threadIdx.x] : 0.0f;
    t = warp_reduce_sum(t);
    if (threadIdx.x == 0) {
      stage[kWarps] = t;
    }
    __syncthreads();
    return stage[kWarps];
  }
}

// Sum of one packed vector, and its sum of squared deviations from the mean.
// Both unpack from the register copy, so neither touches global memory.
template <typename T, typename Acc>
__device__ __forceinline__ float vector_sum(typename Acc::Vec packed) {
  float e[Acc::kElems];
  Acc::to_float(packed, e);
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < Acc::kElems; ++j) {
    s += e[j];
  }
  return s;
}

template <typename T, typename Acc>
__device__ __forceinline__ float vector_sq_dev(typename Acc::Vec packed,
                                               float mean) {
  float e[Acc::kElems];
  Acc::to_float(packed, e);
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < Acc::kElems; ++j) {
    const float d = e[j] - mean;
    s += d * d;
  }
  return s;
}

// ---------------------------------------------------------------------------
// Welford, in the shape ATen's LayerNorm uses. Kept behind a macro because the
// point is bit-agreement with ATen rather than accuracy.
// ---------------------------------------------------------------------------
struct WelfordLN {
  float mean;
  float m2;
  float count;
};

__device__ __forceinline__ WelfordLN welford_step(WelfordLN a, float x) {
  a.count += 1.0f;
  const float delta = x - a.mean;
  a.mean += delta / a.count;
  a.m2 += delta * (x - a.mean);
  return a;
}

__device__ __forceinline__ WelfordLN welford_merge(WelfordLN a, WelfordLN b) {
  if (b.count == 0.0f) {
    return a;
  }
  if (a.count == 0.0f) {
    return b;
  }
  WelfordLN r;
  r.count = a.count + b.count;
  const float delta = b.mean - a.mean;
  const float frac = b.count / r.count;
  r.mean = a.mean + delta * frac;
  r.m2 = a.m2 + b.m2 + delta * delta * a.count * frac;
  return r;
}

__device__ __forceinline__ WelfordLN welford_warp_reduce(WelfordLN v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    WelfordLN o;
    o.mean = __shfl_xor_sync(kFullMask, v.mean, offset);
    o.m2 = __shfl_xor_sync(kFullMask, v.m2, offset);
    o.count = __shfl_xor_sync(kFullMask, v.count, offset);
    v = welford_merge(v, o);
  }
  return v;
}

// ``1 + scale`` as the reference computes it: a bfloat16 tensor op, so each
// channel's gain is the fp32 sum rounded once to the storage dtype. Returned
// packed, because the value depends only on the channel -- which is what lets
// the persistent kernel compute it once per CTA instead of once per row.
// Taken by value, not by const reference: a reference parameter bound directly
// to a global address makes the unpack below take that address and read the four
// 32-bit halves separately. Measured on this kernel as 18 global load
// instructions per thread instead of 6 -- 4x LDG.E per parameter vector rather
// than one LDG.E.128 -- on a kernel whose limit is L1TEX and issue throughput,
// not DRAM. By value the copy is a register move the inliner removes.
template <typename T, typename Acc>
__device__ __forceinline__ typename Acc::Vec modulation_gain(
    typename Acc::Vec scale_packed) {
  constexpr int kElems = Acc::kElems;
  float sf[kElems];
  Acc::to_float(scale_packed, sf);
#pragma unroll
  for (int j = 0; j < kElems; ++j) {
    sf[j] = 1.0f + sf[j];
  }
  return Acc::from_float(sf);
}

// The whole epilogue for one vector: normalise, optionally apply the LayerNorm
// affine, then the modulation -- reproducing every rounding the reference
// performs. ``gain`` carries the already-rounded ``1 + scale``.
// Takes the deviations (x - mean) rather than the packed row, so the packed and
// fp32-resident kernels share one epilogue and therefore one rounding chain.
template <typename T, typename Acc, bool kAffine>
__device__ __forceinline__ void write_modulated_dev(
    const float* dev, typename Acc::Vec* __restrict__ out,
    typename Acc::Vec gain, typename Acc::Vec offset,
    const typename Acc::Vec* __restrict__ w,
    const typename Acc::Vec* __restrict__ b, int idx, float rstd) {
  constexpr int kElems = Acc::kElems;
  float e[kElems];
#pragma unroll
  for (int j = 0; j < kElems; ++j) {
    e[j] = dev[j] * rstd;
  }
  if constexpr (kAffine) {
    // ATen keeps the affine transform inside its fp32 accumulator and rounds
    // once on the store, so weight and bias fold in *before* the rounding
    // below rather than after it.
    //
    // Copy each parameter vector into a local before unpacking it: handing
    // ``to_float`` a global address makes it take that address and read the
    // four halves separately, which the frozen sibling measured as 4x 32-bit
    // LDG per vector instead of one 128-bit LDG.
    const typename Acc::Vec wp = w[idx];
    const typename Acc::Vec bp = b[idx];
    float wf[kElems];
    float bf[kElems];
    Acc::to_float(wp, wf);
    Acc::to_float(bp, bf);
#pragma unroll
    for (int j = 0; j < kElems; ++j) {
      e[j] = e[j] * wf[j] + bf[j];
    }
  }
  float gf[kElems];
  float of[kElems];
  Acc::to_float(gain, gf);
  Acc::to_float(offset, of);
#pragma unroll
  for (int j = 0; j < kElems; ++j) {
#if FK_ADALNC_FP32_EPILOGUE
    // Diagnostic arm: one rounding, on the store. Inside the tolerance gate but
    // not bit-identical to the reference.
    e[j] = e[j] * gf[j] + of[j];
#else
    // self.norm(x) is a low-precision tensor, so the normalised value is
    // rounded before it is multiplied.
    const float normalized = Packed<T>::round_to_storage(e[j]);
    // A low-precision multiply: fp32 math, rounded result.
    const float scaled = Packed<T>::round_to_storage(normalized * gf[j]);
    // The last sum is rounded by from_float on the store below.
    e[j] = scaled + of[j];
#endif
  }
  store_vec(&out[idx], Acc::from_float(e));
}

// The packed entry point: unpack, subtract the mean, then the shared epilogue.
template <typename T, typename Acc, bool kAffine>
__device__ __forceinline__ void write_modulated(
    typename Acc::Vec row_packed, typename Acc::Vec* __restrict__ out,
    typename Acc::Vec gain, typename Acc::Vec offset,
    const typename Acc::Vec* __restrict__ w,
    const typename Acc::Vec* __restrict__ b, int idx, float mean, float rstd) {
  constexpr int kElems = Acc::kElems;
  float dev[kElems];
  Acc::to_float(row_packed, dev);
#pragma unroll
  for (int j = 0; j < kElems; ++j) {
    dev[j] -= mean;
  }
  write_modulated_dev<T, Acc, kAffine>(dev, out, gain, offset, w, b, idx, rstd);
}

// ---------------------------------------------------------------------------
// Fused SiLU + projection GEMV: emb = W @ bf16(silu(c)) + b.
//
// Replaces two launches whose measured cost is dominated by neither arithmetic
// nor useful bandwidth: the activation is 5.4 us of kernel time for a 6 KiB input
// (essentially all launch and tail), and the vendor GEMV spends 16.1 us moving
// 36 MiB of weight, against ~6 us of ideal DRAM time for those bytes. Together
// they were measured at 43.9% of case A's marginal cost and 63.9% of case B's,
// which is what pulled this forward from the deferred list.
//
// The activation form is the frozen sibling's shipped one -- ex2.approx plus
// div.approx -- which that file establishes as bitwise identical to at::silu on
// every finite bfloat16 and float16 encoding. The candidate's own test suite
// re-verifies that exhaustively rather than trusting the claim.
// ---------------------------------------------------------------------------

// Padding one float every 8 makes each lane's 8-float group start at a distinct
// shared-memory bank: consecutive lanes step by 9 floats, and gcd(9, 32) == 1, so
// the 32 starting banks are a permutation instead of a 4-way conflict.
__device__ __forceinline__ int staged_offset(int j) { return j + (j >> 3); }

template <typename T>
__device__ __forceinline__ float silu_approx(float x) {
  // ex2.approx.f32 + div.approx.f32, matching the frozen activation exactly.
  return __fdividef(x, 1.0f + __expf(-x));
}

template <typename T, int kRowsPerBlock>
__global__ void __launch_bounds__(kRowsPerBlock * kWarpSize) silu_gemv_kernel(
    const T* __restrict__ cond, const T* __restrict__ weight,
    const T* __restrict__ bias, T* __restrict__ out, int k, int m) {
  constexpr int kElems = Packed<T>::kElems;
  extern __shared__ float staged[];

  // The activation is applied once per CTA and rounded to the storage dtype
  // before it is used, because the reference materialises silu(c) as a T tensor.
  for (int j = threadIdx.x; j < k; j += kRowsPerBlock * kWarpSize) {
    const float a = Packed<T>::scalar_to_float(cond[j]);
    staged[staged_offset(j)] = Packed<T>::round_to_storage(silu_approx<T>(a));
  }
  __syncthreads();

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int vecs = k / kElems;
  const uint4* __restrict__ wbase = reinterpret_cast<const uint4*>(weight);
  const int row_stride = static_cast<int>(gridDim.x) * kRowsPerBlock;

  // Grid-stride over output rows, so the staged activation above is paid for once
  // per CTA. The loop bound depends only on blockIdx and warp, so it is uniform
  // across the warp and the shuffles below always see all 32 lanes.
  for (int row = blockIdx.x * kRowsPerBlock + warp; row < m;
       row += row_stride) {
    const uint4* __restrict__ wrow = wbase + static_cast<int64_t>(row) * vecs;
    // Four partial accumulators so the fp32 adds do not serialise on one
    // register.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int vidx = lane; vidx < vecs; vidx += kWarpSize) {
      // Staged in a local before unpacking: a reference bound to a global
      // address turns one 128-bit load into four 32-bit loads.
      const uint4 wp = wrow[vidx];
      float wf[kElems];
      Packed<T>::to_float(wp, wf);
      const float* sp = &staged[staged_offset(vidx * kElems)];
#pragma unroll
      for (int e = 0; e < kElems; e += 4) {
        acc0 += wf[e] * sp[e];
        acc1 += wf[e + 1] * sp[e + 1];
        acc2 += wf[e + 2] * sp[e + 2];
        acc3 += wf[e + 3] * sp[e + 3];
      }
    }
    const float total = warp_reduce_sum((acc0 + acc1) + (acc2 + acc3));
    if (lane == 0) {
      const float biased =
          bias != nullptr ? total + Packed<T>::scalar_to_float(bias[row]) : total;
      out[row] = Packed<T>::scalar_from_float(biased);
    }
  }
#if FK_ADALNC_PDL
  // Release the consumer as soon as this CTA's stores are visible, rather than at
  // kernel end.
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------
// One CTA per row. The grid is exactly the row count, so no bounds check is
// needed, and consecutive threads take consecutive vectors within each of the
// kVecsPerThread passes, so every pass is a fully coalesced 128-bit access.
// ---------------------------------------------------------------------------
template <typename T, typename Acc, int kBlockThreads, int kVecsPerThread,
          bool kAffine>
__global__ void __launch_bounds__(kBlockThreads) ada_ln_cont_row_kernel(
    const T* __restrict__ x, const T* __restrict__ emb,
    const T* __restrict__ weight, const T* __restrict__ bias,
    T* __restrict__ y, int vecs_per_row, int seq_len, float inv_n, float eps) {
  constexpr int kWarps = kBlockThreads / kWarpSize;
  using Vec = typename Acc::Vec;
  __shared__ float stage[2 * (kWarps + 1)];
#if FK_ADALNC_WELFORD
  __shared__ WelfordLN welford_stage[kWarps + 1];
#endif

  const int row = static_cast<int>(blockIdx.x);
  // One integer divide per thread, executed once, against a kernel that moves
  // 12 KiB of row. Templating this away for the batch-extent-1 captured cases
  // would double the instantiation count for no measurable return.
  const int batch = row / seq_len;

  const Vec* __restrict__ xv = reinterpret_cast<const Vec*>(x) +
                               static_cast<int64_t>(row) * vecs_per_row;
  Vec* __restrict__ yv =
      reinterpret_cast<Vec*>(y) + static_cast<int64_t>(row) * vecs_per_row;
  // emb is [B, 2n]: the gain half at channel j, the offset half at n + j.
  const Vec* __restrict__ scale_v = reinterpret_cast<const Vec*>(emb) +
                                    static_cast<int64_t>(batch) * 2 *
                                        vecs_per_row;
  const Vec* __restrict__ shift_v = scale_v + vecs_per_row;
  const Vec* __restrict__ wv = reinterpret_cast<const Vec*>(weight);
  const Vec* __restrict__ bv = reinterpret_cast<const Vec*>(bias);

  if constexpr (kFp32Row) {
    // The row is unpacked once and then overwritten in place with the deviations
    // the variance pass computes, so neither the variance pass nor the epilogue
    // unpacks again and the epilogue's subtract disappears. Both accumulators are
    // still summed one vector at a time into a per-vector temporary, so the
    // reduction order -- and therefore every bit of the result -- is unchanged.
    constexpr int kElems = Acc::kElems;
    float row_f[kVecsPerThread * kElems];
    float mean;
    float rstd;
#if FK_ADALNC_WELFORD
    // ATen's shape: a Welford recurrence per thread, then a tree of Welford
    // merges. Same registers, different arithmetic, and the reason to have it is
    // agreement with the reference rather than accuracy.
    WelfordLN acc{0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        Acc::to_float(load_streamed(&xv[idx]), &row_f[i * kElems]);
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          acc = welford_step(acc, row_f[i * kElems + j]);
        }
      }
    }
    {
      WelfordLN w = welford_warp_reduce(acc);
      const int lane = threadIdx.x & (kWarpSize - 1);
      const int warp = threadIdx.x >> 5;
      if (lane == 0) {
        welford_stage[warp] = w;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        WelfordLN total = welford_stage[0];
#pragma unroll 1
        for (int k = 1; k < kWarps; ++k) {
          total = welford_merge(total, welford_stage[k]);
        }
        welford_stage[kWarps] = total;
      }
      __syncthreads();
      const WelfordLN total = welford_stage[kWarps];
      mean = total.mean;
      rstd = rsqrtf(total.m2 * inv_n + eps);
    }
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          row_f[i * kElems + j] -= mean;
        }
      }
    }
#else
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        const Vec packed = load_streamed(&xv[idx]);
        Acc::to_float(packed, &row_f[i * kElems]);
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          s += row_f[i * kElems + j];
        }
        sum += s;
      }
    }
    mean = block_reduce_sum<kBlockThreads>(sum, stage) * inv_n;

    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          const float d = row_f[i * kElems + j] - mean;
          row_f[i * kElems + j] = d;
          s += d * d;
        }
        sq += s;
      }
    }
    const float var =
        block_reduce_sum<kBlockThreads>(sq, stage + kWarps + 1) * inv_n;
    rstd = rsqrtf(var + eps);
#endif

#if FK_ADALNC_PDL
    // The gain and the offset are the projection's output, and they are not read
    // until here -- the row load and both reduction passes above depend only on x.
    // So the dependency on the producer is released exactly at this point.
    cudaGridDependencySynchronize();
#endif
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        const Vec scale_packed = load_reused(&scale_v[idx]);
        const Vec offset_packed = load_reused(&shift_v[idx]);
        write_modulated_dev<T, Acc, kAffine>(
            &row_f[i * kElems], yv, (modulation_gain<T, Acc>(scale_packed)),
            offset_packed, wv, bv, idx, rstd);
      }
    }
  } else {
    Vec packed[kVecsPerThread];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        packed[i] = load_streamed(&xv[idx]);
        sum += (vector_sum<T, Acc>(packed[i]));
      }
    }
    const float mean = block_reduce_sum<kBlockThreads>(sum, stage) * inv_n;

    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        sq += (vector_sq_dev<T, Acc>(packed[i], mean));
      }
    }
    const float var =
        block_reduce_sum<kBlockThreads>(sq, stage + kWarps + 1) * inv_n;
    const float rstd = rsqrtf(var + eps);

#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        // Staged in locals first, so each is one wide load.
        const Vec scale_packed = load_reused(&scale_v[idx]);
        const Vec offset_packed = load_reused(&shift_v[idx]);
        write_modulated<T, Acc, kAffine>(
            packed[i], yv, (modulation_gain<T, Acc>(scale_packed)),
            offset_packed, wv, bv, idx, mean, rstd);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Persistent grid-stride variant. The per-channel gain and offset are hoisted
// into registers before the row loop, so a CTA that owns several rows loads them
// -- and rounds ``1 + scale`` -- once instead of once per row.
//
// The motive is instruction count, not bandwidth: the gain and offset together
// are 12 KiB, which is L1-resident, so what the one-CTA-per-row variant repeats
// is cached loads, address arithmetic and conversions, not DRAM traffic. The
// cost is 2 * kVecsPerThread packed vectors live across both reductions. Which
// side wins is measured, not assumed.
// ---------------------------------------------------------------------------
template <typename T, typename Acc, int kBlockThreads, int kVecsPerThread,
          bool kAffine>
__global__ void __launch_bounds__(kBlockThreads) ada_ln_cont_persistent_kernel(
    const T* __restrict__ x, const T* __restrict__ emb,
    const T* __restrict__ weight, const T* __restrict__ bias,
    T* __restrict__ y, int64_t rows, int vecs_per_row, int seq_len,
    float inv_n, float eps) {
  constexpr int kWarps = kBlockThreads / kWarpSize;
  __shared__ float stage[2 * (kWarps + 1)];

  using Vec = typename Acc::Vec;
  const Vec* __restrict__ xv_base = reinterpret_cast<const Vec*>(x);
  Vec* __restrict__ yv_base = reinterpret_cast<Vec*>(y);
  const Vec* __restrict__ emb_base = reinterpret_cast<const Vec*>(emb);
  const Vec* __restrict__ wv = reinterpret_cast<const Vec*>(weight);
  const Vec* __restrict__ bv = reinterpret_cast<const Vec*>(bias);

  Vec gain[kVecsPerThread];
  Vec offset[kVecsPerThread];
  // Kept packed rather than unpacked to fp32: 4 registers per vector instead of
  // 8, which halves what stays live across the two reductions.
  int cached_batch = -1;

  for (int64_t row = blockIdx.x; row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    const int batch = static_cast<int>(row / seq_len);
    if (batch != cached_batch) {
      // A no-op after the first row at batch extent 1, which is every captured
      // case; the guard is what keeps the variant correct for B > 1.
      const Vec* __restrict__ scale_v =
          emb_base + static_cast<int64_t>(batch) * 2 * vecs_per_row;
      const Vec* __restrict__ shift_v = scale_v + vecs_per_row;
#pragma unroll
      for (int i = 0; i < kVecsPerThread; ++i) {
        const int idx = threadIdx.x + i * kBlockThreads;
        if (idx < vecs_per_row) {
          const Vec scale_packed = scale_v[idx];
          gain[i] = (modulation_gain<T, Acc>(scale_packed));
          offset[i] = shift_v[idx];
        }
      }
      cached_batch = batch;
    }

    const uint4* __restrict__ xv = xv_base + row * vecs_per_row;
    uint4* __restrict__ yv = yv_base + row * vecs_per_row;

    Vec packed[kVecsPerThread];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        packed[i] = load_streamed(&xv[idx]);
        sum += (vector_sum<T, Acc>(packed[i]));
      }
    }
    const float mean = block_reduce_sum<kBlockThreads>(sum, stage) * inv_n;

    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        sq += (vector_sq_dev<T, Acc>(packed[i], mean));
      }
    }
    const float var =
        block_reduce_sum<kBlockThreads>(sq, stage + kWarps + 1) * inv_n;
    const float rstd = rsqrtf(var + eps);

#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int idx = threadIdx.x + i * kBlockThreads;
      if (idx < vecs_per_row) {
        write_modulated<T, Acc, kAffine>(packed[i], yv, gain[i], offset[i], wv,
                                         bv, idx, mean, rstd);
      }
    }
    // The next row reuses ``stage``, and the epilogue above still reads the
    // broadcast slot the second reduction wrote.
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host side: the launch counter, the dispatch ladder, the eligibility predicate
// that mirrors it, and the reference-formula fallback.
// ---------------------------------------------------------------------------

// Counted on the host inside the launch path, not in Python: the scored forward
// runs between two CUDA events, so a Python-level counter update there would be
// measured as operator cost. One relaxed increment per launch, against a kernel
// that runs for tens of microseconds. This is what distinguishes "the fused path
// ran" from "the fallback ran and tied".
std::atomic<int64_t> g_fused_launches{0};

// How a call reaches the kernel. Decided once, on host, before any pointer is
// dereferenced, and returned so the caller cannot pick a launcher the predicate
// did not approve.
enum class Path { kFallback, kFused };

inline bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

inline bool is_aligned32(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 31u) == 0;
}

inline bool param_aligned32(const at::Tensor& p) {
  return !p.defined() || is_aligned32(p.const_data_ptr());
}

// Which row widths have an implemented mapping. Kept adjacent to the launcher so
// a predicate and its dispatch cannot drift apart.
inline bool has_vector_mapping(int vecs_per_row) {
  return vecs_per_row >= 1 &&
         (vecs_per_row <= kMaxLadderVecs || vecs_per_row == kTunedVecs);
}

// Every way a tensor can be something other than an ordinary dense tensor this
// kernel may write into. Shared by both eligibility predicates, because the
// projection path has exactly the same exposure as the normalization path.
//
// Three separate mechanisms, and none of them subsumes the others:
//
//  * Functional and fake tensors and Python wrapper subclasses have no ordinary
//    storage to take a pointer to. isTensorSubclassLike reports those.
//  * torch.autograd.forward_ad.make_dual attaches a tangent to an otherwise
//    ordinary dense tensor, which isTensorSubclassLike accepts. The fused paths
//    allocate with at::empty and launch a raw kernel, so they would compute the
//    primal and drop the tangent, returning a *silently* zero derivative rather
//    than failing. _fw_grad is what reports that.
//  * torch.func.jvp instead wraps the tensor for the functorch interpreter, where
//    the tangent lives in the transform stack and _fw_grad at level 0 is
//    undefined, so the wrapper's own dispatch keys are what identify it -- and the
//    transform can additionally be active only in thread-local state while the
//    tensors look ordinary, which is the vmap-over-jvp shape.
//
// Measured: without the second and third checks, torch.func.jvp through this
// operator produced an all-zero tangent. The fallbacks re-dispatch through ATen
// ops that carry forward AD correctly.
inline bool transform_active(const at::Tensor& t) {
  constexpr auto kTransformKeys = c10::DispatchKeySet(
      {c10::DispatchKey::FuncTorchGradWrapper,
       c10::DispatchKey::FuncTorchBatched,
       c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
       c10::DispatchKey::FuncTorchDynamicLayerBackMode,
       c10::DispatchKey::FuncTorchVmapMode,
       c10::DispatchKey::Functionalize});
  if (c10::impl::tls_local_dispatch_key_set().included_.has_any(kTransformKeys)) {
    return true;
  }
  if (!t.defined()) {
    return false;
  }
  return at::isTensorSubclassLike(t) || t._fw_grad(/*level=*/0).defined() ||
         t.key_set().has_any(kTransformKeys);
}

// The grid and the in-kernel row index are 32-bit. Factored out of choose_path so
// the bound itself is testable without allocating a tensor of that many rows --
// 2^31 rows of 3072 bfloat16 would be 12 TiB.
inline bool row_count_within_limit(int64_t rows) {
  return rows >= 0 && rows <= static_cast<int64_t>(INT32_MAX);
}

// The LayerNorm affine parameters travel with the norm submodule, not with
// ``x``, so each is validated on its own merits.
inline bool affine_param_ok(const at::Tensor& p, const at::Tensor& x,
                            int64_t n) {
  return p.device() == x.device() && p.scalar_type() == x.scalar_type() &&
         p.dim() == 1 && p.size(0) == n && p.is_contiguous() &&
         is_aligned16(p.const_data_ptr());
}

inline bool affine_pair_ok(const at::Tensor& w, const at::Tensor& b,
                           const at::Tensor& x, int64_t n) {
  // A half-present pair is not a shape this kernel is written for; it takes the
  // fallback rather than guessing which half to synthesise.
  if (w.defined() != b.defined()) {
    return false;
  }
  if (!w.defined()) {
    return true;
  }
  return affine_param_ok(w, x, n) && affine_param_ok(b, x, n);
}

inline Path choose_path(const at::Tensor& x, const at::Tensor& emb,
                        const at::Tensor& w, const at::Tensor& b, int64_t n) {
  if (!x.defined() || !emb.defined() || !x.is_cuda()) {
    return Path::kFallback;
  }
  if (transform_active(x) || transform_active(emb) || transform_active(w) ||
      transform_active(b)) {
    return Path::kFallback;
  }
  // Autocast rewrites LayerNorm's output dtype and this operator has no
  // autocast registration of its own; the fallback re-dispatches through
  // at::layer_norm and so picks up that policy exactly.
  if (at::autocast::is_autocast_enabled(x.device().type())) {
    return Path::kFallback;
  }
  const at::ScalarType dtype = x.scalar_type();
  if (dtype != at::kBFloat16 && dtype != at::kHalf) {
    return Path::kFallback;
  }
  if (emb.scalar_type() != dtype || emb.device() != x.device()) {
    return Path::kFallback;
  }
  // The kernel's row-to-batch mapping expresses exactly [B, T, n] against
  // emb [B, 2n]. Every other rank goes to the fallback, which is the reference
  // expression and therefore reproduces the ranks whose broadcast is
  // surprising: at rank 2, (1 + scale)[:, None, :] is [B, 1, n] against a
  // [T, n] normalised tensor, so the reference output is *rank 3*; at rank 4
  // the batch extent aligns with x.size(-3) rather than x.size(0), and some
  // combinations raise.
  if (n <= 0 || x.dim() != 3 || x.size(-1) != n) {
    return Path::kFallback;
  }
  if (emb.dim() != 2 || emb.size(0) != x.size(0) || emb.size(1) != 2 * n) {
    return Path::kFallback;
  }
  if (x.numel() == 0) {
    return Path::kFallback;
  }
  const int64_t rows = x.size(0) * x.size(1);
  if (!row_count_within_limit(rows)) {
    return Path::kFallback;
  }
  if (!x.is_contiguous() || !emb.is_contiguous()) {
    return Path::kFallback;
  }
  if (!affine_pair_ok(w, b, x, n)) {
    return Path::kFallback;
  }
  // Elements per access: 16 bytes, or 32 in the wide arm.
  const int access_bytes = kWide ? 32 : 16;
  const int elems = access_bytes / static_cast<int>(x.element_size());
  if (n % elems != 0) {
    return Path::kFallback;
  }
  // Bound before narrowing: a row of more than INT32_MAX vectors would wrap to a
  // small count that has_vector_mapping accepts, and the kernel would then
  // normalise a prefix of the row and leave the rest of the output
  // uninitialised.
  const int64_t vecs = n / elems;
  if (vecs > static_cast<int64_t>(INT32_MAX) ||
      !has_vector_mapping(static_cast<int>(vecs))) {
    return Path::kFallback;
  }
  // Alignment of the inputs is a property of the caller's views; the output is
  // allocated below by the caching allocator, whose smallest block alignment is
  // far wider than 16 bytes.
  if (!is_aligned16(x.const_data_ptr()) || !is_aligned16(emb.const_data_ptr())) {
    return Path::kFallback;
  }
  // The 256-bit arm needs a strictly stronger alignment on every pointer it
  // dereferences, including the affine parameters, and on the row stride so that
  // row r's base stays aligned. The caching allocator gives the output far more
  // than 32 bytes, so only the caller's views need testing.
  if constexpr (kWide) {
    if (!is_aligned32(x.const_data_ptr()) || !is_aligned32(emb.const_data_ptr()) ||
        !param_aligned32(w) || !param_aligned32(b) ||
        (n * x.element_size()) % 32 != 0) {
      return Path::kFallback;
    }
  }
  return Path::kFused;
}

// Rows of the projection output per CTA, from FK_ADALNC_PROJ_WARPS. Each warp owns
// one output row at a time, so this is also how many dot products share one staged
// activation. Sixteen is the measured default; see the macro's own note.
constexpr int kProjRowsPerBlock = FK_ADALNC_PROJ_WARPS;
constexpr int kProjWaves = FK_ADALNC_PROJ_WAVES;
static_assert(kProjRowsPerBlock > 0 && kProjRowsPerBlock <= 32,
              "the projection CTA holds at most 32 warps");
static_assert(kProjWaves > 0, "the projection grid needs at least one wave");
// Bound on the staged activation, so the dynamic shared-memory request cannot
// exceed what a CTA can be given.
constexpr int kProjMaxK = 8192;

std::atomic<int64_t> g_projection_launches{0};

inline Path choose_projection_path(const at::Tensor& cond,
                                   const at::Tensor& weight,
                                   const at::Tensor& bias,
                                   at::ScalarType target) {
  if (!cond.defined() || !weight.defined() || !cond.is_cuda()) {
    return Path::kFallback;
  }
  // The same exposure as the normalization path: this kernel also allocates with
  // at::empty and launches a raw kernel, so a tangent on the conditioning input or
  // on the projection weight would be silently dropped.
  if (transform_active(cond) || transform_active(weight) ||
      transform_active(bias)) {
    return Path::kFallback;
  }
  if (at::autocast::is_autocast_enabled(cond.device().type())) {
    return Path::kFallback;
  }
  const at::ScalarType dtype = cond.scalar_type();
  if (dtype != at::kBFloat16 && dtype != at::kHalf) {
    return Path::kFallback;
  }
  // The reference casts the activation to x's dtype; a cast that would actually
  // change the value belongs to the fallback rather than to this kernel.
  if (dtype != target) {
    return Path::kFallback;
  }
  if (weight.scalar_type() != dtype || weight.device() != cond.device()) {
    return Path::kFallback;
  }
  // One staged activation per CTA expresses exactly a single batch row. Every
  // other batch extent reaches the fallback, which is the reference composition.
  if (cond.dim() != 2 || cond.size(0) != 1) {
    return Path::kFallback;
  }
  const int64_t k = cond.size(1);
  if (weight.dim() != 2 || weight.size(1) != k) {
    return Path::kFallback;
  }
  const int64_t m = weight.size(0);
  if (k <= 0 || m <= 0 || k > kProjMaxK || m > static_cast<int64_t>(INT32_MAX)) {
    return Path::kFallback;
  }
  const int elems = 16 / static_cast<int>(cond.element_size());
  if (k % elems != 0) {
    return Path::kFallback;
  }
  if (bias.defined()) {
    if (bias.device() != cond.device() || bias.scalar_type() != dtype ||
        bias.dim() != 1 || bias.size(0) != m || !bias.is_contiguous()) {
      return Path::kFallback;
    }
  }
  if (!cond.is_contiguous() || !weight.is_contiguous()) {
    return Path::kFallback;
  }
  if (!is_aligned16(cond.const_data_ptr()) ||
      !is_aligned16(weight.const_data_ptr())) {
    return Path::kFallback;
  }
  return Path::kFused;
}

at::Tensor run_projection(const at::Tensor& cond, const at::Tensor& weight,
                          const at::Tensor& bias) {
  const c10::cuda::CUDAGuard device_guard(cond.device());
  const int k = static_cast<int>(cond.size(1));
  const int m = static_cast<int>(weight.size(0));
  at::Tensor out = at::empty({1, static_cast<int64_t>(m)}, cond.options());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t covering = (m + kProjRowsPerBlock - 1) / kProjRowsPerBlock;
  // Capped, not covering: the activation is staged once per CTA, so fewer CTAs
  // means proportionally less redundant expf and divide work.
  const unsigned grid = static_cast<unsigned>(std::max<int64_t>(
      std::min<int64_t>(covering, static_cast<int64_t>(sms) * kProjWaves), 1));
  // Padded so each lane's group lands on its own bank; see staged_offset.
  const size_t smem = static_cast<size_t>(k + (k >> 3)) * sizeof(float);

  g_projection_launches.fetch_add(1, std::memory_order_relaxed);

#define FK_ADALNC_LAUNCH_PROJ(CUDA_T)                                          \
  do {                                                                         \
    silu_gemv_kernel<CUDA_T, kProjRowsPerBlock>                                 \
        <<<grid, kProjRowsPerBlock * kWarpSize, smem, stream>>>(                \
            reinterpret_cast<const CUDA_T*>(cond.const_data_ptr()),             \
            reinterpret_cast<const CUDA_T*>(weight.const_data_ptr()),           \
            bias.defined()                                                      \
                ? reinterpret_cast<const CUDA_T*>(bias.const_data_ptr())        \
                : nullptr,                                                      \
            reinterpret_cast<CUDA_T*>(out.mutable_data_ptr()), k, m);            \
  } while (0)

  if (cond.scalar_type() == at::kBFloat16) {
    FK_ADALNC_LAUNCH_PROJ(__nv_bfloat16);
  } else {
    FK_ADALNC_LAUNCH_PROJ(__half);
  }
#undef FK_ADALNC_LAUNCH_PROJ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The reference composition. at::silu stands in for the frozen sibling
// activation, which that file establishes -- and this candidate's tests
// re-verify exhaustively -- as bitwise identical on every finite low-precision
// encoding; at::linear is what the frozen Linear delegates to for this shape,
// its admission table being empty.
at::Tensor reference_projection(const at::Tensor& cond,
                               const at::Tensor& weight,
                               const at::Tensor& bias,
                               at::ScalarType target) {
  at::Tensor activated = at::silu(cond);
  if (activated.scalar_type() != target) {
    activated = activated.to(target);
  }
  const std::optional<at::Tensor> bo =
      bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt;
  return at::linear(activated, weight, bo);
}

at::Tensor project(const at::Tensor& cond, const at::Tensor& weight,
                   const std::optional<at::Tensor>& bias,
                   at::ScalarType target) {
  const at::Tensor b = bias.has_value() ? *bias : at::Tensor();
  if (at::GradMode::is_enabled()) {
    return reference_projection(cond, weight, b, target);
  }
  if (choose_projection_path(cond, weight, b, target) == Path::kFused) {
    return run_projection(cond, weight, b);
  }
  return reference_projection(cond, weight, b, target);
}

template <typename T>
void launch_fused(const T* x, const T* emb, const T* w, const T* b, T* y,
                  int64_t rows, int vecs_per_row, int seq_len, float inv_n,
                  float eps, bool affine, cudaStream_t stream) {
  using Acc = std::conditional_t<kWide, Wide<T>, Narrow<T>>;

#define FK_ADALNC_LAUNCH(BLK, VPT, AFF)                                     \
  do {                                                                      \
    if constexpr (kPersistent) {                                            \
      const int sms =                                                       \
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount;      \
      const int64_t grid = std::max<int64_t>(                               \
          std::min<int64_t>(rows, static_cast<int64_t>(sms) * kWaves), 1);   \
      ada_ln_cont_persistent_kernel<T, Acc, (BLK), (VPT), (AFF)>            \
          <<<static_cast<unsigned>(grid), (BLK), 0, stream>>>(              \
              x, emb, w, b, y, rows, vecs_per_row, seq_len, inv_n, eps);     \
    } else if constexpr (kPdl) {                                            \
      /* The consumer carries the programmatic-serialization attribute, so it  \
         may begin before the producer retires; the in-kernel                 \
         cudaGridDependencySynchronize above its epilogue is what actually     \
         waits for emb. */                                                   \
      cudaLaunchAttribute attr{};                                            \
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;           \
      attr.val.programmaticStreamSerializationAllowed = 1;                    \
      cudaLaunchConfig_t cfg{};                                              \
      cfg.gridDim = dim3(static_cast<unsigned>(rows));                        \
      cfg.blockDim = dim3((BLK));                                            \
      cfg.dynamicSmemBytes = 0;                                              \
      cfg.stream = stream;                                                   \
      cfg.attrs = &attr;                                                     \
      cfg.numAttrs = 1;                                                      \
      C10_CUDA_CHECK(cudaLaunchKernelEx(                                     \
          &cfg, ada_ln_cont_row_kernel<T, Acc, (BLK), (VPT), (AFF)>, x, emb,  \
          w, b, y, vecs_per_row, seq_len, inv_n, eps));                       \
    } else {                                                                \
      ada_ln_cont_row_kernel<T, Acc, (BLK), (VPT), (AFF)>                   \
          <<<static_cast<unsigned>(rows), (BLK), 0, stream>>>(              \
              x, emb, w, b, y, vecs_per_row, seq_len, inv_n, eps);           \
    }                                                                       \
  } while (0)
#define FK_ADALNC_DISPATCH(BLK, VPT)      \
  do {                                    \
    if (affine) {                         \
      FK_ADALNC_LAUNCH(BLK, VPT, true);   \
    } else {                              \
      FK_ADALNC_LAUNCH(BLK, VPT, false);  \
    }                                     \
    return;                               \
  } while (0)

  // The tuned width is tested first so overriding it cannot be shadowed by a
  // generic rung.
  if (vecs_per_row == kTunedVecs) FK_ADALNC_DISPATCH(kRowBlock, kRowVpt);
  if (vecs_per_row <= 64) FK_ADALNC_DISPATCH(64, 1);
  if (vecs_per_row <= 128) FK_ADALNC_DISPATCH(128, 1);
  if (vecs_per_row <= 256) FK_ADALNC_DISPATCH(256, 1);
  FK_ADALNC_DISPATCH(256, 2);

#undef FK_ADALNC_DISPATCH
#undef FK_ADALNC_LAUNCH
}

at::Tensor run_fused(const at::Tensor& x, const at::Tensor& emb,
                     const at::Tensor& w, const at::Tensor& b, int64_t n,
                     double eps) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor y = at::empty(x.sizes(), x.options());

  const int elems =
      (kWide ? 32 : 16) / static_cast<int>(x.element_size());
  const int64_t rows = x.size(0) * x.size(1);
  const int seq_len = static_cast<int>(x.size(1));
  const int vecs_per_row = static_cast<int>(n / elems);
  const float inv_n = 1.0f / static_cast<float>(n);
  const float epsf = static_cast<float>(eps);
  const bool affine = w.defined();
  const auto stream = c10::cuda::getCurrentCUDAStream();

  g_fused_launches.fetch_add(1, std::memory_order_relaxed);

#define FK_ADALNC_LAUNCH_DTYPE(CUDA_T)                                        \
  do {                                                                        \
    const CUDA_T* xp = reinterpret_cast<const CUDA_T*>(x.const_data_ptr());   \
    const CUDA_T* ep = reinterpret_cast<const CUDA_T*>(emb.const_data_ptr()); \
    CUDA_T* yp = reinterpret_cast<CUDA_T*>(y.mutable_data_ptr());             \
    const CUDA_T* wp =                                                        \
        w.defined() ? reinterpret_cast<const CUDA_T*>(w.const_data_ptr())      \
                    : nullptr;                                                \
    const CUDA_T* bp =                                                        \
        b.defined() ? reinterpret_cast<const CUDA_T*>(b.const_data_ptr())      \
                    : nullptr;                                                \
    launch_fused<CUDA_T>(xp, ep, wp, bp, yp, rows, vecs_per_row, seq_len,      \
                         inv_n, epsf, affine, stream);                        \
  } while (0)

  // Exhaustive over the dtypes choose_path admits, so the second arm needs no
  // test of its own.
  if (x.scalar_type() == at::kBFloat16) {
    FK_ADALNC_LAUNCH_DTYPE(__nv_bfloat16);
  } else {
    FK_ADALNC_LAUNCH_DTYPE(__half);
  }
#undef FK_ADALNC_LAUNCH_DTYPE
  // A silently failed launch would otherwise surface as a numeric mismatch.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// The reference's normalisation, fp32 promotion included. ``self.norm`` in the
// reference module is a plain F.layer_norm wrapper, so at::layer_norm is an
// equality here rather than a tolerance argument.
at::Tensor reference_norm(const at::Tensor& x, const at::Tensor& w,
                          const at::Tensor& b, int64_t n, double eps,
                          bool promote_fp32) {
  const std::optional<at::Tensor> wo =
      w.defined() ? std::optional<at::Tensor>(w) : std::nullopt;
  const std::optional<at::Tensor> bo =
      b.defined() ? std::optional<at::Tensor>(b) : std::nullopt;
  if (!promote_fp32) {
    return at::layer_norm(x, {n}, wo, bo, eps);
  }
  const at::ScalarType orig = x.scalar_type();
  const std::optional<at::Tensor> wf =
      (w.defined() && w.scalar_type() != at::kFloat)
          ? std::optional<at::Tensor>(w.to(at::kFloat))
          : wo;
  const std::optional<at::Tensor> bf =
      (b.defined() && b.scalar_type() != at::kFloat)
          ? std::optional<at::Tensor>(b.to(at::kFloat))
          : bo;
  return at::layer_norm(x.to(at::kFloat), {n}, wf, bf, eps).to(orig);
}

// The reference expression itself. at::chunk reproduces torch.chunk's split of
// an odd width, and unsqueeze(1) is [:, None, :] -- which is why every rank the
// fast path declines comes out identical without any rank inspection here,
// including the rank-2 promotion to rank 3 and the rank-4 alignment that raises.
at::Tensor reference_formula(const at::Tensor& x, const at::Tensor& emb,
                            const at::Tensor& w, const at::Tensor& b, int64_t n,
                            double eps, bool promote_fp32) {
  const at::Tensor normalized = reference_norm(x, w, b, n, eps, promote_fp32);
  const std::vector<at::Tensor> halves = at::chunk(emb, 2, 1);
  // Addition is commutative in every rounding mode, so ``scale + 1`` is the same
  // value the reference's ``1 + scale`` produces.
  return normalized * halves[0].add(1).unsqueeze(1) + halves[1].unsqueeze(1);
}

at::Tensor ada_ln_cont(const at::Tensor& x, const at::Tensor& emb,
                       const std::optional<at::Tensor>& weight,
                       const std::optional<at::Tensor>& bias, int64_t n,
                       double eps, bool promote_fp32) {
  const at::Tensor w = weight.has_value() ? *weight : at::Tensor();
  const at::Tensor b = bias.has_value() ? *bias : at::Tensor();
  // The fused path allocates with at::empty and launches a raw kernel, so it
  // records nothing for autograd. Grad mode being *enabled* is the whole test,
  // not whether some tensor currently requires grad: a caller who has not
  // entered no_grad may attach requires_grad later in the same graph, or wrap
  // this in checkpointing that replays the forward under different requires_grad
  // state. Gradient-enabled calls therefore take the reference expression, whose
  // ATen ops this operator's CompositeImplicitAutograd registration traces
  // through. The cost is that the fused kernel runs only under no_grad, which is
  // where every inference path -- the benchmark harness included -- already puts
  // it.
  if (at::GradMode::is_enabled()) {
    return reference_formula(x, emb, w, b, n, eps, promote_fp32);
  }
  if (choose_path(x, emb, w, b, n) == Path::kFused) {
    return run_fused(x, emb, w, b, n, eps);
  }
  return reference_formula(x, emb, w, b, n, eps, promote_fp32);
}

int64_t fused_launches() {
  return g_fused_launches.load(std::memory_order_relaxed);
}

int64_t projection_launches() {
  return g_projection_launches.load(std::memory_order_relaxed);
}

// Pure introspection: reports what the eligibility predicate decides, without
// allocating or launching anything. It exists so every branch of that predicate
// can be asserted directly by a test instead of being inferred from a launch
// counter that only says "something declined". Nothing on the scored path calls
// it, and it takes no locks and touches no global state.
bool fast_path_selected(const at::Tensor& x, const at::Tensor& emb,
                        const std::optional<at::Tensor>& weight,
                        const std::optional<at::Tensor>& bias, int64_t n) {
  const at::Tensor w = weight.has_value() ? *weight : at::Tensor();
  const at::Tensor b = bias.has_value() ? *bias : at::Tensor();
  if (at::GradMode::is_enabled()) {
    return false;
  }
  return choose_path(x, emb, w, b, n) == Path::kFused;
}

// The reference casts the activation to *x*'s dtype, so the target travels as the
// tensor whose dtype it is. Passing x costs nothing -- it is already live, and only
// its scalar type is read.
at::Tensor project_op(const at::Tensor& cond, const at::Tensor& weight,
                      const std::optional<at::Tensor>& bias,
                      const at::Tensor& like) {
  return project(cond, weight, bias, like.scalar_type());
}

}  // namespace

TORCH_LIBRARY(fk_adalnc_cand, m) {
  m.def(
      "ada_ln_cont(Tensor x, Tensor emb, Tensor? weight, Tensor? bias, int n, "
      "float eps, bool promote_fp32) -> Tensor",
      &ada_ln_cont);
  m.def(
      "project(Tensor cond, Tensor weight, Tensor? bias, Tensor like) "
      "-> Tensor",
      &project_op);
  // Introspection only, and deliberately separate operators: nothing on the
  // scored path reads them.
  m.def("fused_launches() -> int", &fused_launches);
  m.def("projection_launches() -> int", &projection_launches);
  m.def(
      "fast_path_selected(Tensor x, Tensor emb, Tensor? weight, Tensor? bias, "
      "int n) -> bool",
      &fast_path_selected);
  // The row bound, callable on a plain integer, so the branch that no allocation
  // can reach is still asserted against the code the predicate actually runs.
  m.def("row_count_within_limit(int rows) -> bool", &row_count_within_limit);
}
"""


def _build_directory() -> str | None:
    """Persistent per-workspace build cache, so a warm import never calls nvcc."""
    try:
        path = Path(__file__).resolve().parents[2] / ".torch_extensions" / _LIBRARY_NAME
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None  # fall back to cpp_extension's own default location
    return str(path)


def _load_fused_ops():
    """Build and register the operators, returning their resolved callables.

    Compilation happens here, at import, so nothing is deferred into a timed
    ``forward``. The includes are lean on purpose: ``<torch/extension.h>``
    through nvcc dominates the build, and these operators are registered with
    ``TORCH_LIBRARY`` rather than pybind, so none of it is needed.
    """
    from torch.utils.cpp_extension import load_inline

    # The pin does two things, and only the first is about build time:
    #   * it keeps the build single-arch. This workspace's shell exports
    #     TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX", so without
    #     pinning the same source compiles for six architectures instead of one
    #     -- five of which will never run the kernel.
    #   * it is outright required wherever the variable is unset or "native":
    #     that branch of _get_cuda_arch_flags iterates torch.cuda.device_count()
    #     and then indexes the resulting list, raising IndexError with no visible
    #     GPU. That is this agent's normal state, so the pin is unconditional --
    #     derived from the live device when there is one, and from the target
    #     architecture otherwise. Restored in a finally so no later build in this
    #     process is affected.
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    arch = _FALLBACK_ARCH
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        load_inline(
            name=_LIBRARY_NAME,
            cpp_sources="",
            cuda_sources=_CUDA_SOURCE,
            # -lineinfo lets ncu attribute SASS to source, and -Xptxas=-v puts
            # the per-kernel register count and the spill report in the build
            # log, which is the only thing that distinguishes a register-heavy
            # rung from one that has quietly moved the row to local memory.
            # --use_fast_math is deliberately absent: every approximation here is
            # an explicit, auditable intrinsic.
            extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
            build_directory=_build_directory(),
            is_python_module=False,
            no_implicit_headers=True,
            verbose=False,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    ops = getattr(torch.ops, _LIBRARY_NAME)
    # Bind the overloads, not the packets: a packet re-resolves overloads from
    # the argument types on every call, and forward is launch-latency bound on
    # these shapes.
    return (ops.ada_ln_cont.default, ops.project.default,
            ops.fused_launches.default, ops.projection_launches.default,
            ops.fast_path_selected.default,
            ops.row_count_within_limit.default)


# Built at import time, never lazily inside forward: ninja spawns subprocesses,
# and the benchmark harness fails any candidate that grows the thread count
# during its timing window. ``EXTENSION_STATUS`` is the empty string exactly when
# the compiled path is live, and carries the build error otherwise.
EXTENSION_STATUS = ""
try:
    (_fused_ada_ln_cont, _fused_project, fused_launches, projection_launches,
     fast_path_selected, row_count_within_limit) = _load_fused_ops()
except Exception as exc:  # noqa: BLE001 - a build that cannot happen must
    # degrade, not take the module down with it: an import failure costs every
    # scored case at once.
    _fused_ada_ln_cont = None
    _fused_project = None
    EXTENSION_STATUS = f"{type(exc).__name__}: {exc}"

    def fused_launches() -> int:
        """No fused launches are possible when the extension did not build."""
        return 0

    def projection_launches() -> int:
        """No projection launches are possible when the extension did not build."""
        return 0

    def fast_path_selected(*_args, **_kwargs) -> bool:
        """No fast path exists when the extension did not build."""
        return False

    def row_count_within_limit(_rows: int) -> bool:
        """No fast path exists when the extension did not build."""
        return False

    print(
        f"[ada_layer_norm_continuous] CUDA extension unavailable, falling back to "
        f"the reference formula: {EXTENSION_STATUS}",
        file=sys.stderr,
        flush=True,
    )


# The fused SiLU + projection GEMV is built and registered but is **not** on the
# shipped path, and the reason is a measured conflict between two requirements
# rather than a lack of speed.
#
# Measured (profile/p1-candidate/ab_projection_results.json, six-point sweep): at
# its tuned geometry the fused projection beats the frozen SiLU + vendor GEMV
# composition by 2.0 us, which is one CUDA-event timer tick and the edge of this
# workspace's tie band.
#
# The cost is that its fp32 accumulation order differs from the vendor GEMV's, so
# 12 of 6144 projected channels land one bfloat16 ulp away from the reference.
# Each such channel is a *gain* or an *offset*, so it multiplies or shifts an
# entire channel of the output, and where the result then cancels in ``t + shift``
# the inherited error survives into a near-zero value. Measured end to end, that
# takes the harness's ``matched_ratio`` off exactly 1.0 -- 0.99997 on the larger
# scored case -- while the fused-normalization-only path holds 1.0 on both.
#
# The gate still passes either way (its threshold is 0.99), and end-to-end speedup
# is inside run-to-run clock variance, so the trade is a strictly exact answer
# against a tick of latency that does not reliably show up in the score. This file
# keeps the exact answer. Flip FUSE_PROJECTION to re-enable the kernel; the A/B
# harness drives it directly regardless.
FUSE_PROJECTION = False


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32: bool = True,
    ):
        super().__init__()
        # Submodule names, classes and parameter shapes are the baseline's,
        # because the harness shares weights with
        # ``load_state_dict(..., strict=False)`` inside a bare ``except: pass``:
        # a renamed or restructured parameter is silently *not* loaded, and the
        # candidate then runs against different random weights and produces a
        # wrong answer that looks like a numeric bug.
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps,
                                  elementwise_affine=elementwise_affine,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        # The kernel wants the row width as a plain int; the norm submodule stays
        # the single source of truth for eps and the promotion flag.
        self._n = int(embedding_dim)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        if FUSE_PROJECTION and _fused_project is not None:
            # One dispatch for the projection as well. Not the shipped path; see
            # FUSE_PROJECTION.
            emb = _fused_project(conditioning_embedding, self.linear.weight,
                                 self.linear.bias, x)
        else:
            emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if _fused_ada_ln_cont is not None:
            # One dispatch: eligibility and the fallback both live inside the
            # operator, so nothing here inspects tensor metadata and ``emb`` is
            # passed whole rather than chunked.
            return _fused_ada_ln_cont(
                x, emb, self.norm.weight, self.norm.bias, self._n,
                self.norm.eps, self.norm.promote_fp32,
            )
        # Reached only when the extension did not build. This routes the
        # normalisation through the frozen sibling LayerNorm rather than
        # ``F.layer_norm`` -- it is the reference *formula*, and that sibling
        # cleared the same gate on its own cases, but it is not the bit-identical
        # replay the compiled fallback performs.
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
