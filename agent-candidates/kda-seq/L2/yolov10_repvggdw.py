"""YOLOv10 RepVGG depthwise block as one depthwise 7x7 launch.

The scored graph is ``SiLU(BN7(dwconv7x7(x, W7)) + BN3(dwconv3x3(x, W3)))``, which the baseline
evaluates as six kernel launches behind eight ``nn.Module.__call__`` frames. Both branches are
depthwise, both BatchNorms are per-channel affine in eval mode, and the branch sum is linear, so
the whole block collapses -- exactly, not approximately -- into one 7x7 depthwise convolution with
a per-channel bias followed by SiLU. Per channel ``c``, with ``eps`` read from the live BatchNorm::

    s7 = bn7.weight / sqrt(bn7.running_var + eps)   b7 = bn7.bias - s7 * bn7.running_mean
    s3 = bn3.weight / sqrt(bn3.running_var + eps)   b3 = bn3.bias - s3 * bn3.running_mean

    W_f[c]             = s7[c] * W7[c]              # 7x7
    W_f[c][2:5, 2:5]  += s3[c] * W3[c]              # 3x3 centred, 2 per side
    b_f[c]             = b7[c] + b3[c]

    y = SiLU(dwconv7x7(x, W_f, pad 3) + b_f)

That is the same algebra ``fuse()`` performs, computed non-destructively in fp32 from whatever
weights are live, and cached.

Why one launch and not a faster convolution: at the captured shapes the operator carries ~40 MFLOP
and ~1.6 MB of traffic, which is ~0.7 us of FFMA and ~0.2 us of DRAM on a B200. It is overhead-
bound. Measurement through the harness's own timing loop (``profile/phase1_floor/``) puts the
per-iteration floor for a single-launch module at ~9.2 us, decomposing into 2.91 us of CUDA event
bracket, 4.26 us for the harness's own per-iteration input copy, and 2.05 us for one launch, with
kernel duration entering the score at ~0.89 us per us. Two consequences shaped this file: the
launch count is the whole optimization, and the two-launch ``F.silu(F.conv2d(...))`` form measures
2.36x geomean -- useful as a fallback, not fast enough to be the fast path. One forward issues
exactly one CUDA kernel against the baseline's ten, for a measured 4.4-5.0x geomean speedup
depending on how loaded the device is.

The mapping is measured, not argued: outputs per thread, shared row stride, planes per block and
whether the taps live in registers or shared memory are compile-time parameters, and the shipped
combination won a 13-entry sweep run through ``bench._time_module`` (``profile/phase1_mapping/``).
Shared-memory row stride turned out not to matter at all (26, 32, 33 and 34 land within 0.2%),
despite a real 1.71x excess of shared-load wavefronts over ideal on the window read, because the
shared pipe runs at 21% of peak. Staging the taps in shared memory did matter: it drops the kernel
from 64 to 32 registers, doubles theoretical occupancy, and makes the tap reads exactly
conflict-free (1.00x actual/ideal, measured per source line).

The grid is smaller than the device's resident capacity -- 6.9 blocks per SM against a 16-block
register-limited ceiling, with each SM active 43-50% of the SM-elapsed cycles -- and the obvious
remedy of splitting each plane across several blocks was implemented and measured over five
interleaved repetitions per cell. It **loses** 5.0% of weighted geomean, and the deciding gap is at
N=1 (+14.3%) rather than N=4 (+0.7%): doubling the grid moved achieved occupancy only 35.3% to 36.7%
while stripe staging redoes 60-120% of the input loads. Both kernels are in this file; the striped
one is reachable only through the sweep. ``profile/p1_dw7x7_v2/REPORT.md`` has the per-line
evidence.

The fused parameters are cached because the alternative -- reading the live parameters and folding
per call -- costs several Python attribute reads on a call whose entire budget is ~13 us. The cost
of caching is invalidation, and getting that wrong is silent: it returns confidently wrong numbers.
Four mechanisms cover it.

Three structural hooks drop the cache outright: ``load_state_dict`` (post-hook), ``_apply`` (so
``.to`` / ``.half`` / ``.float`` / ``.cuda`` are covered), and ``fuse()``.

Those hooks are not sufficient. They fire on *rebinding*, so they observe neither an ordinary
in-place update (``conv.weight.mul_(4)`` under ``no_grad``, or an optimizer step) nor training-mode
BatchNorm rewriting ``running_mean`` and ``running_var``. So every eval call also re-checks a
**freshness signature**, and every training-mode forward clears the cache before delegating.

The signature is read from the **live** structure on each call -- ``self.conv``, an optional
``conv1``, each branch's current ``bn`` and current convolution bias -- and carries structural
markers plus, per source tensor, its ``id``, ``data_ptr``, ``dtype``, ``device`` and ``_version``.
Every element of that is load-bearing, and two earlier revisions of this file got it wrong in ways
worth naming, because the plausible cheap versions are the broken ones:

* Trusting the three hooks alone missed in-place mutation and training statistics: ``max_abs`` 2.24
  and 1.44, with under 5% of elements inside tolerance.
* Re-summing ``_version`` over the tensors *stored at build time* missed anything that replaced an
  object rather than mutating it. A fresh ``nn.Parameter`` starts at version 0 while the cached
  tensor stays alive with its version frozen, so the sum says "fresh" forever: rebinding
  ``conv.conv.weight`` gave ``max_abs`` 2.61, and a cached forward followed by a direct
  ``YOLOConv.fuse()`` on the children gave ``max_abs`` 2.08 with *nothing* in tolerance. Child
  fusion is the sharp case: it writes the folded weight through ``.data.copy_()``, which does not
  move ``_version``, and it binds a brand-new bias ``Parameter`` that the old source list never
  contained.

Comparing ``id`` is sound here only because the cache keeps references to the tensors it was built
from, so those objects stay alive and their addresses cannot be recycled by a replacement while the
cache exists.

The signature costs 5.85 us of host time and a measured **0.000 us of score** at both captured
shapes, over nine interleaved repetitions of the harness's own timing loop
(``profile/phase1_cache_guard/guard_cost.json``). That is not a rounding: the harness enqueues a
~110 us L2 flush before its start event, so the host runs far ahead of the device and host-side work
inside the timed window is hidden entirely. An earlier estimate priced a per-call guard at several
percent of the score and declined it on that basis; the estimate was wrong, and wrong for the same
reason an earlier fallback-speedup estimate was.

Cache construction reads the *observable submodule structure* rather than ``_is_fused``: for each
branch, a ``bn`` child means BN still has to be folded, and its absence means the branch weight is
already folded and ``conv.bias`` carries the branch bias. That is what makes the state a bare
``yolov10_conv.fuse_module(self)`` leaves behind -- children fused, root still flagged unfused --
work instead of raising on a missing ``conv.bn``.

**The one remaining limitation**: mutating a parameter or buffer through ``.data``
(``weight.data.mul_(2)``) leaves the object, its storage pointer, its dtype, its device and its
``_version`` all unchanged, so nothing in the signature can see it. Call
``refresh_fused_weights()`` after doing that. It is the only staleness path left, and that claim is
now backed by regressions for in-place mutation, training-statistic updates, Parameter rebinding,
buffer rebinding and cached-forward-then-child-fusion -- the two previous revisions each asserted
this and were each wrong, so it is stated only as far as the tests reach.

Routing, in the order ``forward`` applies it. Training mode and grad-enabled calls go to the
inherited baseline graph: training-mode BatchNorm uses batch statistics, which are not foldable from
running statistics, and a grad-enabled call has to build a graph the kernel cannot. Next, anything
that is not a 4-D CUDA tensor whose dtype and device match the live convolution parameters *also*
goes to the baseline graph -- and that is a deliberate restriction, not an oversight. Folding
BatchNorm away removes the operation that rejects a CPU float16 tensor, a float64 tensor against
float32 running statistics, a 3-D tensor, or a mismatched dtype; left alone this module would quietly
*accept* all of them, and a caller that dispatches on the baseline raising would silently change
behaviour. Being more permissive than the thing you replace is still a behavioural difference.

What remains -- a 4-D CUDA tensor matching the parameters -- is the regime the algebraic fold is
equivalent over. There the custom kernel runs when the layout and plane geometry are instantiated,
and the two-launch convolution with per-dtype cached weights covers the rest (non-contiguous,
channels-last, zero-element, and geometries outside the compiled set).
``EXTENSION_STATUS`` is the empty string exactly when the compiled path is live.

On numerics: the fused weights are derived in fp32 and the 49 taps are accumulated in a single
fp32 accumulator, rounded to the output dtype once. The reference rounds to float16 three times, so
this is not the same arithmetic, and closer to the real-valued expression is not automatically
closer to the reference. Across both captured shapes, N in {1,2,4,8}, unfused and fused states,
float16/bfloat16/float32 and weight scales {0.02, 0.2, 1.0}, agreement is total, with max_abs
4.9e-04 at the scale the harness generates -- about 20x inside ``atol``. A construction where the
two branches cancel exactly does push 1-2 elements per 100,000 past the elementwise bound, against
the 1,000 per 100,000 the harness allows; that residual is the *reference's* one-ulp float16
cancellation noise, not this kernel's, and ``tests/codex_numerics_decision.md`` records why
replicating the staged rounding would imitate it rather than remove it.

What is not established: the internal origin of the 2.91 us event floor, the mapping between the
~2.30 us inter-command gaps Nsight reports and CUDA event execution, and the kernel's absolute
duration, for which three measurement routes disagree by 1.6x. See
``profile/phase1_floor/ATTRIBUTION.md`` and ``profile/p1_dw7x7_v1/REPORT.md`` for what was measured
and what was not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Subclassing the baseline block is what makes ``yolov10_conv.fuse_module``'s ``isinstance`` check
# succeed on a bare instance, and it inherits ``__init__`` (hence byte-identical state_dict keys)
# and ``fuse()``'s exact semantics rather than restating them. The import is absolute because the
# relative name ``.yolov10_repvggdw`` is *this* module. Note that ``list.apply_candidates`` rebinds
# every module-level reference to the baseline class -- including this alias -- to the candidate
# class, so nothing below may use the alias at runtime; ``super()`` resolves through the MRO and is
# unaffected.
from fastkernels.tasks.baseline.L2.yolov10_repvggdw import YOLORepVGGDW as _BaselineBlock

# Shipped launch mapping. ``_VARIANT`` indexes the compile-time (outputs-per-thread, shared row
# stride, planes-per-block) set enumerated in the CUDA source; it is a measured knob, settled by
# profile/phase1_mapping/ rather than by argument.
_VARIANT = 8

_EXTENSION_NAME = "fk_yolov10_repvggdw_dw7x7_v1"
_CUDA_ARCH = "10.0"

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>

namespace {

constexpr int kK = 7;                  // 7x7 after the 3x3 branch is folded in
constexpr int kPad = kK / 2;           // 3
// 49 floats rounded up to a float4 multiple, so a channel's taps are 208 bytes and every
// channel base stays 16-byte aligned for the float4 pulls below.
constexpr int kTapStride = 52;

// The shipped mapping. Spelled out as constants that both the variant switch and the
// non-float16 dispatch use, so the shipped configuration and its numbered sweep entry cannot
// drift apart. Winner of profile/phase1_mapping/: 17.30 us against 18.46 us for the same mapping
// with the taps in registers, at N=4; every mapping ties at N=1, where the score is all floor.
constexpr int kDefaultVariant = 8;
constexpr int kDefaultOut = 4;
constexpr int kDefaultStride = 26;
constexpr int kDefaultPlanes = 1;
constexpr bool kDefaultSharedTaps = true;

// load_inline compiles with -D__CUDA_NO_HALF_CONVERSIONS__ and
// -D__CUDA_NO_BFLOAT16_CONVERSIONS__, so every conversion here is an explicit
// round-to-nearest-even intrinsic, matching ATen's static_cast out of its fp32 accumulator.
template <typename T>
struct Convert;

template <>
struct Convert<__half> {
  __device__ __forceinline__ static float to_f32(const __half v) { return __half2float(v); }
  __device__ __forceinline__ static __half from_f32(const float v) { return __float2half(v); }
};

template <>
struct Convert<__nv_bfloat16> {
  __device__ __forceinline__ static float to_f32(const __nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ __forceinline__ static __nv_bfloat16 from_f32(const float v) {
    return __float2bfloat16(v);
  }
};

template <>
struct Convert<float> {
  __device__ __forceinline__ static float to_f32(const float v) { return v; }
  __device__ __forceinline__ static float from_f32(const float v) { return v; }
};

// Access words for the global loads and stores. Only widths 2 and 4 are needed: the input is
// always read four elements at a time, and a thread stores its whole output run at once.
template <typename T, int kElems>
struct AccessWord;

template <> struct AccessWord<__half, 2> { using type = uint32_t; };
template <> struct AccessWord<__half, 4> { using type = uint2; };
template <> struct AccessWord<__nv_bfloat16, 2> { using type = uint32_t; };
template <> struct AccessWord<__nv_bfloat16, 4> { using type = uint2; };
template <> struct AccessWord<float, 2> { using type = float2; };
template <> struct AccessWord<float, 4> { using type = float4; };

template <typename T, int kElems>
union Pack {
  typename AccessWord<T, kElems>::type raw;
  T elem[kElems];
};

// x[n,c,y,x] -> out[n,c,y,x] = silu(b_f[c] + sum_{ky,kx} W_f[c,ky,kx] * x[n,c,y+ky-3,x+kx-3]).
//
// One (n,c) plane per thread group. The plane is staged into shared memory *as fp32* with its
// 3-wide halo zero-filled, which removes every bounds test and every per-use conversion from the
// inner loop; the 49 taps live in registers, read from an address that is uniform across the group
// so the loads broadcast; accumulators are fp32, seeded with the bias.
//
// kOut, kStride, kPlanes and kTapsInShared are the measured mapping knobs. kStride is the shared
// row stride in floats -- 26 is the minimum for a 20-wide plane with the halo, and larger values
// trade shared memory for a different bank pattern on the window reads. kTapsInShared trades 49
// registers of tap storage for one broadcast shared read per tap: profiling the register variant
// showed 64 registers capping residency at 8 blocks/SM with only 0.92 eligible warps per cycle, so
// on a latency-bound grid the extra warps can be worth more than the extra loads.
template <typename T, int H, int W, int kOut, int kStride, int kPlanes, bool kTapsInShared>
__global__ __launch_bounds__(kPlanes * (H * W / kOut)) void dw7x7_bias_silu_kernel(
    const T* __restrict__ x, const float* __restrict__ taps, const float* __restrict__ bias,
    T* __restrict__ out, const int channels, const int planes_total) {
  constexpr int kPlaneElems = H * W;
  constexpr int kThreadsPerPlane = kPlaneElems / kOut;
  constexpr int kTileRows = H + kK - 1;
  constexpr int kTileElems = kTileRows * kStride;
  constexpr int kWindow = kOut + kK - 1;
  constexpr int kLoadElems = 4;

  __shared__ float tile[kPlanes * kTileElems];
  // Sized 1 rather than 0 in the register variant: a zero-length array is not valid here.
  // __align__(16) is load-bearing: the staging store below is a float4, and a plain __shared__
  // float array is only 4-byte aligned at an offset that depends on the tile size preceding it.
  __align__(16) __shared__ float tap_cache[kTapsInShared ? kPlanes * kTapStride : 1];

  const int sub = (kPlanes == 1) ? 0 : static_cast<int>(threadIdx.x) / kThreadsPerPlane;
  const int t = static_cast<int>(threadIdx.x) - sub * kThreadsPerPlane;
  const int plane = static_cast<int>(blockIdx.x) * kPlanes + sub;
  const bool live = plane < planes_total;

  float* __restrict__ my_tile = tile + sub * kTileElems;

  float w[kTapsInShared ? 1 : kTapStride];
  float acc[kOut];

  if (live) {
    // Halo only. The interior is written by the cooperative load below and the two regions are
    // disjoint, so one barrier is enough for both.
    for (int i = t; i < kTileElems; i += kThreadsPerPlane) {
      const int r = i / kStride;
      const int col = i - r * kStride;
      if (r < kPad || r >= H + kPad || col < kPad || col >= W + kPad) {
        my_tile[i] = 0.0f;
      }
    }

    // Four contiguous input elements per thread per step. W % 4 == 0 means such a run never
    // crosses a row, so the destination is one contiguous span of the tile interior.
    const T* __restrict__ src = x + static_cast<int64_t>(plane) * kPlaneElems;
    for (int i = t; i < kPlaneElems / kLoadElems; i += kThreadsPerPlane) {
      Pack<T, kLoadElems> word;
      word.raw = *reinterpret_cast<const typename AccessWord<T, kLoadElems>::type*>(
          src + kLoadElems * i);
      const int off = kLoadElems * i;
      const int r = off / W;
      float* __restrict__ dst = my_tile + (r + kPad) * kStride + (off - r * W) + kPad;
#pragma unroll
      for (int k = 0; k < kLoadElems; ++k) {
        dst[k] = Convert<T>::to_f32(word.elem[k]);
      }
    }

    const int c = plane % channels;
    const float4* __restrict__ tap_words =
        reinterpret_cast<const float4*>(taps + static_cast<int64_t>(c) * kTapStride);
    if constexpr (kTapsInShared) {
      // 13 float4 per plane, one per thread; the barrier below already covers this write.
      if (t < kTapStride / 4) {
        reinterpret_cast<float4*>(tap_cache + sub * kTapStride)[t] = __ldg(tap_words + t);
      }
    } else {
#pragma unroll
      for (int k = 0; k < kTapStride / 4; ++k) {
        const float4 v = __ldg(tap_words + k);
        w[4 * k + 0] = v.x;
        w[4 * k + 1] = v.y;
        w[4 * k + 2] = v.z;
        w[4 * k + 3] = v.w;
      }
    }
    const float b = __ldg(bias + c);
#pragma unroll
    for (int k = 0; k < kOut; ++k) {
      acc[k] = b;
    }
  }

  __syncthreads();
  if (!live) {
    return;
  }

  // This thread owns kOut adjacent outputs in one row. Output (oy, ox) reads tile rows
  // oy..oy+6 and tile columns ox..ox+6, so the run needs a kOut+6 wide window per row.
  const int oy = t / (W / kOut);
  const int ox = (t - oy * (W / kOut)) * kOut;

#pragma unroll
  for (int ky = 0; ky < kK; ++ky) {
    const float* __restrict__ row = my_tile + (oy + ky) * kStride + ox;
    float win[kWindow];
#pragma unroll
    for (int k = 0; k < kWindow; ++k) {
      win[k] = row[k];
    }
    // One tap row live at a time. Every thread of a group reads the same address, so these are
    // broadcasts rather than conflicting accesses.
    float tap_row[kK];
#pragma unroll
    for (int kx = 0; kx < kK; ++kx) {
      tap_row[kx] = kTapsInShared ? tap_cache[sub * kTapStride + ky * kK + kx] : w[ky * kK + kx];
    }
#pragma unroll
    for (int i = 0; i < kOut; ++i) {
#pragma unroll
      for (int kx = 0; kx < kK; ++kx) {
        acc[i] = fmaf(tap_row[kx], win[i + kx], acc[i]);
      }
    }
  }

  // silu(v) = v / (1 + exp(-v)). ex2.approx.f32 + div.approx.f32: the form the frozen L1 SiLU
  // winner validated as bitwise identical to ATen over every finite fp16 and bf16 encoding, and
  // it saturates correctly at both ends.
  T* __restrict__ dst = out + static_cast<int64_t>(plane) * kPlaneElems + oy * W + ox;
#pragma unroll
  for (int base = 0; base < kOut; base += (kOut % 4 == 0) ? 4 : 2) {
    constexpr int kStoreElems = (kOut % 4 == 0) ? 4 : 2;
    Pack<T, kStoreElems> word;
#pragma unroll
    for (int k = 0; k < kStoreElems; ++k) {
      const float v = acc[base + k];
      word.elem[k] = Convert<T>::from_f32(__fdividef(v, 1.0f + __expf(-v)));
    }
    *reinterpret_cast<typename AccessWord<T, kStoreElems>::type*>(dst + base) = word.raw;
  }
}

template <typename T, int kOut, int kStride, int kPlanes, bool kTapsInShared>
void launch_20x20(const at::Tensor& x, const at::Tensor& taps, const at::Tensor& bias,
                  at::Tensor& out) {
  constexpr int H = 20;
  constexpr int W = 20;
  const int channels = static_cast<int>(x.size(1));
  const int planes_total = static_cast<int>(x.size(0)) * channels;
  const int blocks = (planes_total + kPlanes - 1) / kPlanes;
  dw7x7_bias_silu_kernel<T, H, W, kOut, kStride, kPlanes, kTapsInShared>
      <<<blocks, kPlanes * (H * W / kOut), 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<const T*>(x.const_data_ptr()), taps.const_data_ptr<float>(),
          bias.const_data_ptr<float>(), static_cast<T*>(out.data_ptr()), channels, planes_total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Spatial split: one block per (plane, stripe) rather than one per plane, so the grid grows by
// kStripes and can exceed the device's resident-block ceiling. The trade is halo duplication --
// a kStripeRows-tall stripe stages kStripeRows+6 input rows -- and worse tap reuse per shared
// load, because a thread owning fewer outputs amortizes its window over fewer FFMAs. Which side
// wins is measured, not argued; see profile/phase1_mapping/.
//
// kThreads is the block size and is deliberately allowed to exceed the number of compute threads
// (kStripeRows * W / kOut): every thread helps stage the tile and the taps, then the surplus
// threads sit out the compute phase, which keeps the block a whole number of warps.
template <typename T, int H, int W, int kStripeRows, int kOut, int kStride, int kThreads>
__global__ __launch_bounds__(kThreads) void dw7x7_bias_silu_striped_kernel(
    const T* __restrict__ x, const float* __restrict__ taps, const float* __restrict__ bias,
    T* __restrict__ out, const int channels, const int planes_total) {
  constexpr int kPlaneElems = H * W;
  constexpr int kStripes = H / kStripeRows;
  constexpr int kTileRows = kStripeRows + kK - 1;
  constexpr int kTileElems = kTileRows * kStride;
  constexpr int kWindow = kOut + kK - 1;
  constexpr int kComputeThreads = kStripeRows * W / kOut;
  constexpr int kLoadElems = 4;
  constexpr int kQuadsPerRow = W / kLoadElems;

  __shared__ float tile[kTileElems];
  __align__(16) __shared__ float tap_cache[kTapStride];  // float4 staging store; see above

  const int plane = static_cast<int>(blockIdx.x) / kStripes;
  const int stripe = static_cast<int>(blockIdx.x) - plane * kStripes;
  const int t = static_cast<int>(threadIdx.x);
  if (plane >= planes_total) {
    return;  // whole block is out of range, so no barrier is reached
  }

  const int c = plane % channels;
  const T* __restrict__ src = x + static_cast<int64_t>(plane) * kPlaneElems;

  // Column halo. Unlike the full-plane mapping the *row* halo is not always zero: interior
  // stripes read their neighbours' rows, so row clipping happens in the staging loop below.
  for (int i = t; i < kTileElems; i += kThreads) {
    const int r = i / kStride;
    const int col = i - r * kStride;
    if (col < kPad || col >= W + kPad) {
      tile[i] = 0.0f;
    }
  }

  for (int q = t; q < kTileRows * kQuadsPerRow; q += kThreads) {
    const int r = q / kQuadsPerRow;
    const int col = (q - r * kQuadsPerRow) * kLoadElems;
    const int in_row = stripe * kStripeRows + r - kPad;
    float* __restrict__ dst = tile + r * kStride + col + kPad;
    if (in_row < 0 || in_row >= H) {
      // Above the first row or below the last: this is the genuine zero halo.
#pragma unroll
      for (int k = 0; k < kLoadElems; ++k) {
        dst[k] = 0.0f;
      }
    } else {
      Pack<T, kLoadElems> word;
      word.raw = *reinterpret_cast<const typename AccessWord<T, kLoadElems>::type*>(
          src + in_row * W + col);
#pragma unroll
      for (int k = 0; k < kLoadElems; ++k) {
        dst[k] = Convert<T>::to_f32(word.elem[k]);
      }
    }
  }

  if (t < kTapStride / 4) {
    reinterpret_cast<float4*>(tap_cache)[t] =
        __ldg(reinterpret_cast<const float4*>(taps + static_cast<int64_t>(c) * kTapStride) + t);
  }

  const float b = __ldg(bias + c);
  __syncthreads();

  if (t >= kComputeThreads) {
    return;
  }
  const int oy = t / (W / kOut);
  const int ox = (t - oy * (W / kOut)) * kOut;

  float acc[kOut];
#pragma unroll
  for (int k = 0; k < kOut; ++k) {
    acc[k] = b;
  }
#pragma unroll
  for (int ky = 0; ky < kK; ++ky) {
    const float* __restrict__ row = tile + (oy + ky) * kStride + ox;
    float win[kWindow];
#pragma unroll
    for (int k = 0; k < kWindow; ++k) {
      win[k] = row[k];
    }
#pragma unroll
    for (int i = 0; i < kOut; ++i) {
#pragma unroll
      for (int kx = 0; kx < kK; ++kx) {
        acc[i] = fmaf(tap_cache[ky * kK + kx], win[i + kx], acc[i]);
      }
    }
  }

  T* __restrict__ dst = out + static_cast<int64_t>(plane) * kPlaneElems +
                        (stripe * kStripeRows + oy) * W + ox;
#pragma unroll
  for (int base = 0; base < kOut; base += (kOut % 4 == 0) ? 4 : 2) {
    constexpr int kStoreElems = (kOut % 4 == 0) ? 4 : 2;
    Pack<T, kStoreElems> word;
#pragma unroll
    for (int k = 0; k < kStoreElems; ++k) {
      const float v = acc[base + k];
      word.elem[k] = Convert<T>::from_f32(__fdividef(v, 1.0f + __expf(-v)));
    }
    *reinterpret_cast<typename AccessWord<T, kStoreElems>::type*>(dst + base) = word.raw;
  }
}

template <typename T, int kStripeRows, int kOut, int kStride, int kThreads>
void launch_striped_20x20(const at::Tensor& x, const at::Tensor& taps, const at::Tensor& bias,
                          at::Tensor& out) {
  constexpr int H = 20;
  constexpr int W = 20;
  const int channels = static_cast<int>(x.size(1));
  const int planes_total = static_cast<int>(x.size(0)) * channels;
  const int blocks = planes_total * (H / kStripeRows);
  dw7x7_bias_silu_striped_kernel<T, H, W, kStripeRows, kOut, kStride, kThreads>
      <<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<const T*>(x.const_data_ptr()), taps.const_data_ptr<float>(),
          bias.const_data_ptr<float>(), static_cast<T*>(out.data_ptr()), channels, planes_total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The enumerated mapping set, measured through the real harness timing loop in
// profile/phase1_mapping/mapping_sweep.py. Only float16 carries the whole set, because float16 is
// the only dtype the captured cases score and each entry is a separate fully unrolled 196-FFMA
// instantiation.
template <typename T, bool kAllVariants>
bool dispatch_20x20(const int variant, const at::Tensor& x, const at::Tensor& taps,
                    const at::Tensor& bias, at::Tensor& out) {
  if constexpr (kAllVariants) {
    switch (variant) {
      case 0: launch_20x20<T, 4, 26, 1, false>(x, taps, bias, out); return true;
      case 1: launch_20x20<T, 4, 32, 1, false>(x, taps, bias, out); return true;
      case 2: launch_20x20<T, 4, 33, 1, false>(x, taps, bias, out); return true;
      case 3: launch_20x20<T, 4, 34, 1, false>(x, taps, bias, out); return true;
      case 4: launch_20x20<T, 2, 26, 1, false>(x, taps, bias, out); return true;
      case 5: launch_20x20<T, 2, 32, 1, false>(x, taps, bias, out); return true;
      case 6: launch_20x20<T, 4, 26, 2, false>(x, taps, bias, out); return true;
      case 7: launch_20x20<T, 4, 32, 2, false>(x, taps, bias, out); return true;
      case kDefaultVariant:
        launch_20x20<T, kDefaultOut, kDefaultStride, kDefaultPlanes, kDefaultSharedTaps>(
            x, taps, bias, out);
        return true;
      case 9: launch_20x20<T, 4, 32, 1, true>(x, taps, bias, out); return true;
      case 10: launch_20x20<T, 2, 26, 1, true>(x, taps, bias, out); return true;
      case 11: launch_20x20<T, 2, 32, 1, true>(x, taps, bias, out); return true;
      case 12: launch_20x20<T, 4, 26, 2, true>(x, taps, bias, out); return true;
      // Spatial split: (plane, stripe) blocks. 13 is the design the round-0 review specified --
      // four 5-row stripes, 64 threads, two adjacent x outputs per compute thread. 14 halves the
      // stripe count so the same idea can be read at a second point rather than one.
      case 13: launch_striped_20x20<T, 5, 2, 26, 64>(x, taps, bias, out); return true;
      case 14: launch_striped_20x20<T, 10, 2, 26, 128>(x, taps, bias, out); return true;
      case 15: launch_striped_20x20<T, 5, 2, 32, 64>(x, taps, bias, out); return true;
      // 16 and 17 exist to separate the split from its confound: variants 13-15 all use two
      // outputs per thread, and every out=2 full-plane mapping also loses ~15% at N=4, so
      // without an out=4 split the negative result would be about kOut, not about splitting.
      case 16: launch_striped_20x20<T, 10, 4, 26, 64>(x, taps, bias, out); return true;
      case 17: launch_striped_20x20<T, 5, 4, 26, 32>(x, taps, bias, out); return true;
      default: return false;
    }
  } else {
    if (variant != kDefaultVariant) {
      return false;
    }
    launch_20x20<T, kDefaultOut, kDefaultStride, kDefaultPlanes, kDefaultSharedTaps>(
        x, taps, bias, out);
    return true;
  }
}

bool is_aligned(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

}  // namespace

// Returns nullopt rather than throwing whenever the fast path does not apply, so the caller can
// fall back without the cost of an exception and without this file having to reimplement the
// generic convolution.
std::optional<at::Tensor> dw7x7_bias_silu(const at::Tensor& x, const at::Tensor& taps,
                                          const at::Tensor& bias, const int64_t variant) {
  // A grad-enabled call has to produce a graph, which this kernel cannot; the caller routes those
  // to the baseline computation instead.
  if (at::GradMode::is_enabled()) {
    return std::nullopt;
  }
  if (!x.is_cuda() || x.dim() != 4 || !x.is_contiguous() || x.numel() == 0) {
    return std::nullopt;
  }
  const int64_t H = x.size(2);
  const int64_t W = x.size(3);
  // The guards the mapping needs in general: a four-element run must not cross a row, and the
  // plane plus its halo must fit a static tile. The instantiated geometry is currently {20x20}
  // alone, so everything else is declined here and served by the fallback.
  if (W % 4 != 0 || H > 32 || W > 32 || H != 20 || W != 20) {
    return std::nullopt;
  }
  if (taps.dim() != 2 || taps.size(1) != kTapStride || taps.scalar_type() != at::kFloat ||
      !taps.is_contiguous() || taps.size(0) != x.size(1)) {
    return std::nullopt;
  }
  if (bias.scalar_type() != at::kFloat || !bias.is_contiguous() || bias.numel() != x.size(1)) {
    return std::nullopt;
  }
  if (taps.device() != x.device() || bias.device() != x.device()) {
    return std::nullopt;
  }
  if (!is_aligned(x.const_data_ptr()) || !is_aligned(taps.const_data_ptr())) {
    return std::nullopt;
  }

  const at::cuda::CUDAGuard device_guard(x.device());
  auto out = at::empty_like(x);
  const int v = static_cast<int>(variant);
  switch (x.scalar_type()) {
    case at::kHalf:
      if (!dispatch_20x20<__half, true>(v, x, taps, bias, out)) return std::nullopt;
      break;
    case at::kBFloat16:
      if (!dispatch_20x20<__nv_bfloat16, false>(v, x, taps, bias, out)) return std::nullopt;
      break;
    case at::kFloat:
      if (!dispatch_20x20<float, false>(v, x, taps, bias, out)) return std::nullopt;
      break;
    default:
      return std::nullopt;
  }
  return out;
}
"""

_CPP_SOURCE = r"""
#include <pybind11/stl.h>

#include <optional>

std::optional<at::Tensor> dw7x7_bias_silu(const at::Tensor& x, const at::Tensor& taps,
                                          const at::Tensor& bias, int64_t variant);
"""


def _build_directory() -> str | None:
    """Persistent per-workspace build cache, so a warm import never calls nvcc."""
    try:
        path = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXTENSION_NAME
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None  # fall back to cpp_extension's own default location
    return str(path)


def _load_extension(arch: str | None = _CUDA_ARCH):
    # Pinning the arch list keeps the build single-arch -- this workspace's shell exports six
    # targets -- and is outright required when no GPU is visible, because that branch of
    # _get_cuda_arch_flags indexes a list built from torch.cuda.device_count().
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is None:
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["dw7x7_bias_silu"],
            # -lineinfo so ncu can attribute SASS to source without profiling a different build
            # than the one that ships. --use_fast_math is deliberately absent: it would also relax
            # the fp32 accumulation the numerical agreement depends on.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=_build_directory(),
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# Built at import, never lazily inside forward: ninja spawns subprocesses, and the harness fails
# any candidate whose thread count grows during its timing window. Import must survive a build
# failure, so the module degrades to the convolution fallback and says so on stderr rather than
# becoming unimportable.
EXTENSION_STATUS = ""
try:
    _EXT = _load_extension()
    _FAST = _EXT.dw7x7_bias_silu
except Exception as exc:  # noqa: BLE001 - a build failure must stay importable
    _EXT = None
    _FAST = None
    EXTENSION_STATUS = f"{type(exc).__name__}: {exc}"
    print(
        f"[yolov10_repvggdw] CUDA extension unavailable, falling back to F.conv2d: "
        f"{EXTENSION_STATUS}",
        file=sys.stderr,
        flush=True,
    )

_TAP_STRIDE = 52  # mirrors kTapStride in the CUDA source

# Cache tuple layout. Indexed positionally in ``forward`` so the scored path costs one attribute
# read rather than six.
_TAPS, _BIAS, _BY_DTYPE, _NCHW, _SOURCES, _SIGNATURE = range(6)


def _tensor_tag(t: torch.Tensor) -> tuple:
    """Everything about a source tensor that would invalidate parameters derived from it.

    ``id`` catches rebinding, ``data_ptr`` catches a swapped storage, ``dtype``/``device`` catch a
    cast or a move, and ``_version`` catches in-place mutation. ``id`` is sound here only because
    the cache keeps a reference to the tensor it was built from, so that object stays alive and its
    address cannot be recycled by a replacement while the cache exists.
    """
    return (id(t), t.data_ptr(), t.dtype, t.device, t._version)


def _branch_signature(block) -> tuple:
    """One branch's structure and sources, or ``None`` markers for the parts that are absent."""
    conv = block.conv
    bias = conv.bias
    bn = getattr(block, "bn", None)
    if bn is None:
        return (_tensor_tag(conv.weight), None if bias is None else _tensor_tag(bias), None)
    return (_tensor_tag(conv.weight),
            None if bias is None else _tensor_tag(bias),
            (_tensor_tag(bn.weight), _tensor_tag(bn.bias),
             _tensor_tag(bn.running_mean), _tensor_tag(bn.running_var), bn.eps))


def _live_signature(module) -> tuple:
    """Read the *live* structure, not whatever the cache happens to be holding.

    This is the whole point, and an earlier revision got it wrong: it stored the source tensors at
    build time and re-checked their ``_version``, so rebinding a Parameter left the cache watching a
    dead object whose version could never move again, and a direct ``YOLOConv.fuse()`` on a child --
    which writes through ``.data.copy_()`` and binds a *new* bias Parameter -- was invisible on both
    counts. Reconstructing from ``self`` each call is what makes those observable.
    """
    branch3 = getattr(module, "conv1", None)
    return (_branch_signature(module.conv),
            None if branch3 is None else _branch_signature(branch3))


def _branch_sources(block) -> tuple:
    """The tensors ``_fold_branch`` reads. Kept alive by the cache so their ids cannot be reused."""
    conv = block.conv
    sources = [conv.weight] if conv.bias is None else [conv.weight, conv.bias]
    bn = getattr(block, "bn", None)
    if bn is not None:
        sources += [bn.weight, bn.bias, bn.running_mean, bn.running_var]
    return tuple(sources)


def _fold_branch(block) -> tuple[torch.Tensor, torch.Tensor]:
    """One branch's weight and bias, in fp32, folded if it still has a BatchNorm.

    Reads the structure rather than any ``_is_fused`` flag: a ``bn`` child means the fold is still
    outstanding, its absence means ``conv.weight`` already carries it and ``conv.bias`` is the
    branch bias. That is the only reason a bare ``fuse_module(self)`` -- which fuses the children
    but leaves the root flagged unfused -- does not raise here.
    """
    conv = block.conv
    # ``.float()`` is a no-op view on an fp32 parameter, so clone: this weight is added into in
    # place below, and the cache must not alias anything the module can later mutate.
    weight = conv.weight.detach().float().clone()
    bias = conv.bias
    fused_bias = (bias.detach().float().clone() if bias is not None
                  else weight.new_zeros(weight.shape[0]))
    bn = getattr(block, "bn", None)
    if bn is not None:
        scale = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
        fused_bias = fused_bias * scale + bn.bias.detach().float() - scale * bn.running_mean.detach().float()
        weight = weight * scale.view(-1, 1, 1, 1)
    return weight, fused_bias


class YOLORepVGGDW(_BaselineBlock):
    """Drop-in replacement for the baseline block; see the module docstring."""

    def __init__(self, ed: int):
        super().__init__(ed)
        # Never a buffer and never populated here: the harness shares weights via
        # ``load_state_dict`` *after* construction, so anything derived in __init__ is stale.
        self._fused: tuple | None = None
        self.register_load_state_dict_post_hook(_drop_cache_after_load)

    # --- fused-parameter cache ---------------------------------------------------------------
    def _build_fused(self) -> tuple:
        weight, bias = _fold_branch(self.conv)
        branch3 = getattr(self, "conv1", None)
        if branch3 is not None:
            weight3, bias3 = _fold_branch(branch3)
            offset = (weight.shape[-1] - weight3.shape[-1]) // 2
            end = offset + weight3.shape[-1]
            weight[:, :, offset:end, offset:end] += weight3
            bias = bias + bias3
        channels = weight.shape[0]
        # [C, 52] rather than [C, 49] so each channel's taps start 16-byte aligned and can be
        # pulled with float4 loads; the three trailing floats are never read.
        taps = weight.new_zeros(channels, _TAP_STRIDE)
        taps[:, :weight.shape[-1] * weight.shape[-2]] = weight.reshape(channels, -1)
        sources = _branch_sources(self.conv)
        if branch3 is not None:
            sources += _branch_sources(branch3)
        return (taps.contiguous(), bias.contiguous(), {}, weight.contiguous(),
                sources, _live_signature(self))

    def refresh_fused_weights(self) -> None:
        """Drop the cache. Needed only after mutating a parameter or buffer through ``.data``,
        which is the one path neither the hooks nor the version fingerprint can see."""
        self._fused = None

    def _apply(self, *args, **kwargs):
        # Covers .to() / .half() / .float() / .cuda(): the cache holds tensors that _apply does
        # not walk, so it has to be rebuilt at the new device and dtype.
        self._fused = None
        return super()._apply(*args, **kwargs)

    @torch.no_grad()
    def fuse(self):
        result = super().fuse()
        self._fused = None
        return result

    # --- forward -----------------------------------------------------------------------------
    def _convolution_fallback(self, x: torch.Tensor, fused: tuple) -> torch.Tensor:
        """Two launches, for eval-mode inputs the kernel declines.

        The dtype-matched weight and bias are cached per dtype; calling ``.to(x.dtype)`` per call
        would allocate and launch a conversion kernel every time, which is what makes the naive
        form three launches rather than two.
        """
        matched = fused[_BY_DTYPE].get(x.dtype)
        if matched is None:
            matched = fused[_BY_DTYPE][x.dtype] = (fused[_NCHW].to(x.dtype),
                                                  fused[_BIAS].to(x.dtype))
        weight, bias = matched
        return F.silu(F.conv2d(x, weight, bias, padding=weight.shape[-1] // 2,
                               groups=weight.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Training-mode BatchNorm uses batch statistics, which are not foldable from running
            # statistics -- and it also *updates* the running statistics, so anything cached from
            # an earlier eval call is about to be wrong. Drop it before delegating.
            self._fused = None
            return super().forward(x)
        if torch.is_grad_enabled():
            # A grad-enabled call has to build a graph, which the kernel cannot. This is exactly
            # the inherited baseline computation.
            return super().forward(x)
        weight = self.conv.conv.weight
        if (type(x) is not torch.Tensor or x.dim() != 4 or not x.is_cuda
                or x.dtype is not weight.dtype or x.device != weight.device):
            # Anything outside the regime the folded form is equivalent over goes to the baseline
            # graph, so the *exceptions* match too and not just the outputs. Folding BatchNorm away
            # would otherwise make this module quietly accept inputs the baseline rejects -- a CPU
            # or float64 tensor against float32 running statistics, or a 3-D tensor -- and a caller
            # that dispatches on the baseline raising would silently change behaviour.
            return super().forward(x)
        fused = self._fused
        if fused is None or fused[_SIGNATURE] != _live_signature(self):
            fused = self._fused = self._build_fused()
        if _FAST is not None:
            out = _FAST(x, fused[_TAPS], fused[_BIAS], _VARIANT)
            if out is not None:
                return out
        return self._convolution_fallback(x, fused)


def _drop_cache_after_load(module: YOLORepVGGDW, incompatible_keys) -> None:  # noqa: ARG001
    module._fused = None
