"""YOLOv10 CIB (Compact Inverted Block) as one pybind crossing over folded weights.

The captured block is ``c1=c2=128, e=1.0, lk=True`` on ``float16[N,128,20,20]`` with
``N`` in {1, 4}. Reference-side that is a five-``YOLOConv`` sequence whose middle stage
is a ``YOLORepVGGDW``, so the eager form issues 31 device kernels: six convolutions,
six batch norms, six SiLUs, a branch add, the residual add, and the elementwise traffic
between them. At 20x20 none of those kernels is doing enough arithmetic to matter --
the whole block is 257 MFLOP against 157 KB of weights -- so what the reference
actually spends is dispatch and launch.

Two exact rewrites collapse the structure, and neither is ours to invent: they are the
identities the reference's own ``YOLOConv.fuse()`` and ``YOLORepVGGDW.fuse()``
implement, valid because the harness benches in ``eval`` mode.

* **BatchNorm folding.** With ``s = weight / sqrt(running_var + eps)`` (``eps`` is
  ``1e-3`` here, not PyTorch's default), a convolution followed by its norm is one
  biased convolution: ``w' = w * s[:,None,None,None]``, ``b' = bias - running_mean*s``.
* **RepVGG merge.** The middle stage computes ``SiLU(conv7(t) + conv3(t))`` with both
  branches depthwise and activation-free, so once each branch is folded the 3x3 kernel
  zero-pads into the 7x7 support and the two collapse: ``w'' = w7' + pad(w3',[2,2,2,2])``,
  ``b'' = b7' + b3'``.

What is left is five biased convolutions with bias, SiLU and the residual fused into
their epilogues::

    dw3x3 g=c1        -> SiLU
    pw1x1 c1  -> cmid -> SiLU
    dw7x7 g=cmid      -> SiLU
    pw1x1 cmid -> c2  -> SiLU
    dw3x3 g=c2        -> SiLU -> + x

A three-launch form of the same five stages also exists, as ``cib_forward3``, fusing each
pointwise stage into the depthwise stage beside it::

    launch 1   dw3x3 -> SiLU -> pw1x1 (wmma) -> SiLU
    launch 2   dw7x7 -> SiLU
    launch 3   pw1x1 (wmma) -> SiLU -> dw3x3 -> SiLU -> + x

It is correct -- it agrees with the five-launch form to `max_abs` 0.00098 on both scored
shapes -- and it is *slower*, by a factor of two. Two fewer launches is worth about 6 us;
hand-writing the two pointwise stages instead of calling cuBLAS costs about 51. That is
the measurement, not a prediction, and ``PREFER_THREE_LAUNCH`` below records the decision
it forces.

Three measured costs shape the rest of this file. All were taken on a B200, the first
under the harness's own timing loop so that CPU dispatch sits inside the window exactly
as it does when scored.

* **9.2 us of every measured window is harness overhead** -- the shifting input pool,
  the argument-tree rebuild, ``Module.__call__``. An identity module measures 9.2 us.
  It is the same for reference and candidate and cannot be removed.
* **A kernel launch costs 3-4 us before it does anything.** An empty one-block kernel
  launched back to back in a C++ loop measures 2.9-3.0 us; a 640-block empty kernel
  measures 2.9-4.1 us depending on clocks. So five launches cost 15-20 us whatever they
  compute, which is why all five stages are issued from a single entry point, why every
  epilogue is fused into a stage that was going to run anyway, and why a separate
  elementwise pass is never worth it here.
* **Device arithmetic is *not* below that floor**, which is where an earlier version of
  this design went wrong. Its five hand-written kernels measured 102 us of device time,
  and the two pointwise stages were 76 us of it: 7 million instructions each, one
  shared-memory load per FMA, at 8% issue efficiency. The block is only 257 MFLOP, but
  at 20x20 over 148 SMs the grid is far too small to hide load latency with plain FFMA.

That last measurement is what puts cuBLAS in the pointwise stages. Ten hand-written
register-blocked half2 formulations were swept; the best reached 17.4 us on the 128->256
stage where cuBLAS reaches 5.7 us against a 4.1 us launch floor -- about 1.6 us of
arithmetic against 13. The difference is tensor cores, and it is not a difference a
better tile closes. Register blocking, half2 pairing and shared-memory staging were all
measured rather than assumed, and the numbers are in
``profile/cib_v2_kernel_sweep/reports/``.

A pybind crossing and a tensor allocation remain free at this scale (9.18-9.22 us for
identity, passthrough, and passthrough plus one ``empty_like``), so the entry point does
not economize on either.

Layout is NCHW end to end. That is not a default but a measurement: the folded chain
under ``channels_last`` ran at 1.40x against plain NCHW's 1.90x. For a *fixed* image
NCHW already gives the pointwise stages the K-major operand they want -- ``A[ic, m]``
with row stride ``H*W`` and ``m`` contiguous. Across images there is a batch stride, so
the pointwise stages are a **batch** of ``N`` independent
``(C_out x C_in) x (C_in x HW)`` products; flattening NCHW into one ``C x (N*H*W)``
matrix would mix images and be wrong for every ``N > 1``.

That batch stride is *not* ``C*H*W``, and assuming it was cost this design a 1.0x
result before it was caught. The captured 4-image case records a stride of
``(102400, 400, 20, 1)`` on a ``[4, 128, 20, 20]`` tensor -- 102400 elements between
images against 51200 per image -- because in the model this input is a slice of a wider
tensor, and the harness reproduces the recorded layout with ``empty_strided``. So
``x.is_contiguous()`` is **False** for the larger of the two scored cases, and an
eligibility check that demanded contiguity sent it to the reference path and scored
1.0x while the 1-image case scored 4.5x. What the kernels actually need is that each
*image* be internally contiguous; the batch stride is passed in and honoured, by the
depthwise stages and by the GEMM's ``strideA`` alike.

The folded parameters are derived once, lazily, on the first eligible call. Lazily
because at ``__init__`` the weights are still ``torch.empty`` garbage -- they only become
real after the harness's ``_prepare_module`` -> ``_sanitize_float_params`` ->
``load_state_dict`` sequence. Once, because the alternative was measured and rejected:
a ``(_version, data_ptr)`` signature over the 36 source tensors costs 4.17 us per call,
14-28% of the entire budget, and is *unsound* for the case it was written for -- a
direct ``p.data.mul_(1.5)`` leaves it unchanged, because ``.data`` hands out a tensor
carrying its own fresh version counter. Instead the cache is invalidated through the
lifecycle routes a caller actually takes: ``load_state_dict``, ``_apply`` (and so
``.to()``, ``.half()``, ``.cuda()``), and a train/eval transition. A direct in-place
parameter edit between forwards is not one of those, and is not supported;
:meth:`YOLOCIB.clear_folded_cache` makes that boundary usable. This is the same
boundary the reference's own ``fuse()`` draws -- after it, the norm is deleted and later
weight edits are equally unreflected.

Anything the eligibility conjunction rejects runs the reference block body verbatim, so
whatever the reference would return or raise is what happens. Two of its terms exist
specifically because this path reaches memory by pointer rather than through the
dispatcher. ``x.is_neg()`` and ``x.is_conj()`` are *lazy* flags that ATen applies when it
reads a tensor, so a negative view's logical values are not the bytes in its storage --
admitting one produced a result at `max_abs = 9.44` with 0.4% of elements inside
tolerance. And the fold is fp16-only and device-pinned, so a module converted with
``.float()`` (whose reference raises on an fp16 input) or a fold cached on another GPU is
rejected rather than silently rounded or handed foreign pointers. That guarantee covers
rejected calls only: an accepted call can still be wrong through a stale cache or a bad
kernel, which is what the per-stage tests are for. ``FAST_PATH_AVAILABLE`` and
:meth:`YOLOCIB.tier_counts` exist so a build failure or a silent fallback cannot
masquerade as a pass -- extension availability alone does not prove the fast path ran.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

_EXTENSION_NAME = "fk_l2_yolov10_cib_fused"

# Depthwise mapping: one block per (image, channel) plane, the plane staged in shared
# memory, each thread owning a run of _DW_PIX consecutive output columns of one row. 4
# columns divides the captured W=20 exactly, and on the closely related global-gather
# mapping that this one grew out of it was the fastest of {2, 4, 5, 10, 20} on the 7x7
# stage and tied on the 3x3 (profile/cib_v2_kernel_sweep/reports/sweep3.txt, variant B).
# That sweep was not repeated against the shared-memory staging below, so treat 4 as
# measured-adjacent rather than measured here.
_DW_PIX = 4
_DW_THREADS = 128

# Largest plane the depthwise staging buffer holds. 4096 halves is 8 KB of shared memory
# per block, which does not constrain occupancy at any block count this operator reaches,
# and covers every spatial extent up to 64x64. Wider planes take the reference path.
_MAX_PLANE_ELEMS = 4096

_CPP_SOURCE = """
#include <torch/extension.h>

#include <optional>

// The generated binding translation unit does not see the .cu source, so the entry
// points have to be declared here for it to compile.
at::Tensor cib_forward(const at::Tensor& x, const at::Tensor& packed_weight,
                       const at::Tensor& packed_bias, int64_t mid_channels,
                       int64_t out_channels, bool add_residual);
at::Tensor cib_depthwise_stage(const at::Tensor& x, const at::Tensor& weight,
                               const at::Tensor& bias,
                               const std::optional<at::Tensor>& residual,
                               const std::optional<at::Tensor>& source_bias);
at::Tensor cib_pointwise_stage(const at::Tensor& x, const at::Tensor& weight,
                               const at::Tensor& bias);
at::Tensor cib_forward3(const at::Tensor& x, const at::Tensor& packed_weight,
                        const at::Tensor& packed_bias, int64_t mid_channels,
                        int64_t out_channels, bool add_residual, bool use_pdl);
bool cib_geometry_supported3(int64_t in_channels, int64_t mid_channels,
                             int64_t out_channels, int64_t H, int64_t W);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <limits>
#include <optional>

namespace {

constexpr int kDwPix = __DW_PIX__;
constexpr int kDwThreads = __DW_THREADS__;
constexpr int kEpilogueThreads = 256;
// A plane must fit the staging buffer; the eligibility check enforces the same bound.
constexpr int kMaxPlaneElems = __MAX_PLANE__;

static_assert(kDwThreads % 32 == 0, "a block must be whole warps");
static_assert(kDwPix >= 1, "each thread must own at least one output column");

// SiLU. The fast intrinsics are used deliberately: the harness compares fp16 at
// atol = rtol = 1e-2, which is an order of magnitude wider than the ~1e-3 the fp16
// store already costs, and there are about 1.4 million activations per call.
// Overflow of __expf() for very negative v saturates the quotient to -0.0f, the limit,
// so there is no NaN to guard against.
__device__ __forceinline__ float silu(float v) {
  return __fdividef(v, 1.0f + __expf(-v));
}

// Launch a kernel with programmatic dependent launch opted in, so it may begin during the
// tail of the grid before it. Measured on this device at 12.31 -> 8.33 us over a
// three-stage chain (profile/cib_v5_pdl_coop/reports/probe.txt); the same probe confirms
// the protocol is wired correctly by showing a 1.14x win on a chain with a deliberately
// long producer tail.
template <typename Kernel, typename... Args>
void launch_pdl(Kernel kernel, dim3 grid, dim3 block, size_t shared, cudaStream_t stream,
                Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = 1;
  AT_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

// One block per (image, channel) plane of a depthwise KxK convolution with SAME zero
// padding, bias and SiLU in the epilogue, and optionally the block input added.
//
// The plane is staged in shared memory once, up front. That is what makes this cheap:
// a plane is H*W halves (800 bytes at 20x20), every one of the K*K taps then comes out
// of shared rather than L1, and the *source* epilogue -- the bias and SiLU belonging to
// a preceding pointwise stage that cuBLAS computed without one -- is applied exactly
// once per element on the way in, rather than once per tap that reads it. A formulation
// that applied it per tap would evaluate SiLU K*K times per element.
//
// After the barrier each thread owns a run of kDwPix consecutive output columns of one
// row, so its kDwPix outputs share their tap loads: a run needs kDwPix + K - 1 inputs
// per tap row instead of kDwPix * K. Padding is a bounds test against the staged plane,
// so no boundary case can reach outside the plane it belongs to.
template <int K, int PIX, int THREADS, bool RESIDUAL, bool SOURCE_EPILOGUE>
__global__ __launch_bounds__(THREADS) void dw_tile(
    const __half* __restrict__ src, const __half* __restrict__ source_bias,
    const __half* __restrict__ weight, const __half* __restrict__ bias,
    const __half* __restrict__ residual, __half* __restrict__ dst, int channels, int H,
    int W, int src_batch_stride, int res_batch_stride, bool use_pdl) {
  constexpr int R = K / 2, SPAN = PIX + K - 1;
  static_assert(K % 2 == 1, "an even kernel extent has no centre tap to pad around");

  extern __shared__ __half plane[];
  const int p = blockIdx.x;
  const int n = p / channels;
  const int c = p - n * channels;
  const int HW = H * W;
  // The source may be a slice of a wider tensor: images are ``src_batch_stride`` apart
  // rather than ``channels * HW``. Everything inside an image is contiguous, which the
  // entry point verifies.
  const __half* psrc = src + (long long)n * src_batch_stride + (long long)c * HW;

  if (use_pdl) {
    cudaGridDependencySynchronize();
  }

  if (SOURCE_EPILOGUE) {
    const float sb = __half2float(source_bias[c]);
    for (int i = threadIdx.x; i < HW; i += THREADS) {
      plane[i] = __float2half(silu(__half2float(psrc[i]) + sb));
    }
  } else {
    for (int i = threadIdx.x; i < HW; i += THREADS) {
      plane[i] = psrc[i];
    }
  }
  __syncthreads();

  float taps[K * K];
  const __half* pw = weight + (long long)c * K * K;
#pragma unroll
  for (int i = 0; i < K * K; ++i) {
    taps[i] = __half2float(pw[i]);
  }
  const float bias_v = __half2float(bias[c]);

  __half* pdst = dst + (long long)p * HW;
  const __half* pres =
      RESIDUAL ? residual + (long long)n * res_batch_stride + (long long)c * HW : nullptr;
  const int groups = (W + PIX - 1) / PIX;

  for (int task = threadIdx.x; task < H * groups; task += THREADS) {
    const int row = task / groups;
    const int w0 = (task - row * groups) * PIX;
    float acc[PIX];
#pragma unroll
    for (int q = 0; q < PIX; ++q) {
      acc[q] = bias_v;
    }
#pragma unroll
    for (int kh = 0; kh < K; ++kh) {
      const int r = row + kh - R;
      if (r < 0 || r >= H) {
        continue;
      }
      const __half* prow = plane + r * W;
      float v[SPAN];
#pragma unroll
      for (int i = 0; i < SPAN; ++i) {
        const int col = w0 - R + i;
        v[i] = (col >= 0 && col < W) ? __half2float(prow[col]) : 0.0f;
      }
#pragma unroll
      for (int q = 0; q < PIX; ++q) {
#pragma unroll
        for (int kw = 0; kw < K; ++kw) {
          acc[q] = fmaf(taps[kh * K + kw], v[q + kw], acc[q]);
        }
      }
    }
#pragma unroll
    for (int q = 0; q < PIX; ++q) {
      const int w = w0 + q;
      if (w >= W) {
        continue;
      }
      float out = silu(acc[q]);
      if (RESIDUAL) {
        out += __half2float(pres[row * W + w]);
      }
      pdst[row * W + w] = __float2half(out);
    }
  }
  if (use_pdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Bias and SiLU over a raw GEMM result. Only the single-stage test entry point uses
// this: on the fused path the same epilogue is folded into the next depthwise stage's
// stage-in, which costs nothing extra.
__global__ __launch_bounds__(kEpilogueThreads) void bias_silu(
    const __half* __restrict__ src, const __half* __restrict__ bias,
    __half* __restrict__ dst, int channels, int HW, long long total) {
  for (long long i = (long long)blockIdx.x * kEpilogueThreads + threadIdx.x; i < total;
       i += (long long)gridDim.x * kEpilogueThreads) {
    const int c = (int)((i / HW) % channels);
    dst[i] = __float2half(silu(__half2float(src[i]) + __half2float(bias[c])));
  }
}

template <int K, bool RESIDUAL, bool SOURCE_EPILOGUE>
void launch_dw(const __half* src, const __half* source_bias, const __half* weight,
               const __half* bias, const __half* residual, __half* dst, int planes,
               int channels, int H, int W, int src_batch_stride, int res_batch_stride,
               cudaStream_t stream, bool use_pdl = false) {
  const size_t shared = (size_t)H * W * sizeof(__half);
  if (use_pdl) {
    launch_pdl(dw_tile<K, kDwPix, kDwThreads, RESIDUAL, SOURCE_EPILOGUE>, dim3(planes),
               dim3(kDwThreads), shared, stream, src, source_bias, weight, bias, residual,
               dst, channels, H, W, src_batch_stride, res_batch_stride, true);
    return;
  }
  dw_tile<K, kDwPix, kDwThreads, RESIDUAL, SOURCE_EPILOGUE>
      <<<planes, kDwThreads, shared, stream>>>(src, source_bias, weight, bias, residual,
                                               dst, channels, H, W, src_batch_stride,
                                               res_batch_stride, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// A 1x1 convolution over NCHW is a batch of N independent (OC x IC) x (IC x HW)
// products: for one image ``src[ic, m]`` is exactly the K-major operand a GEMM wants,
// with ``m`` contiguous and row stride HW, but consecutive images are C*H*W apart. So
// the batch is a batch, with stride ``IC*HW`` on the activation and stride 0 on the
// shared weight -- never folded into M, which would mix images and be wrong for every
// N > 1.
//
// cuBLAS rather than a hand-written tile because the difference was measured, not
// assumed: the best of ten hand-written register-blocked half2 formulations reached
// 17.4 us on the 128->256 stage against cuBLAS's 5.7 us, against a per-launch floor of
// 4.1 us -- so cuBLAS does the arithmetic in about 1.6 us and the hand-written kernel
// in about 13. The reason is tensor cores: at 52 MMAC over a 148-SM device the grid is
// too small to hide load latency with plain FFMA, and cuBLAS issues HMMA.
//
// cuBLAS is column-major, and a row-major (R x C) matrix with leading dimension C is
// the same bytes as a column-major (C x R) matrix with the same leading dimension. So
// with no explicit transposes, ``C_col = A_col * B_col`` where A_col is the activation
// read as (HW x IC), B_col the weight read as (IC x OC), and C_col the result as
// (HW x OC) -- which, read back as row-major, is exactly ``dst[oc, m] = sum_ic
// w[oc, ic] * src[ic, m]``.
void launch_gemm(const __half* src, const __half* weight, __half* dst, int N,
                 int in_channels, int out_channels, int HW, int src_batch_stride,
                 cudaStream_t stream) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  const float alpha = 1.0f, beta = 0.0f;
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, HW, out_channels, in_channels, &alpha, src,
      CUDA_R_16F, HW, (long long)src_batch_stride, weight, CUDA_R_16F, in_channels, 0,
      &beta, dst, CUDA_R_16F, HW, (long long)out_channels * HW, N, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

// ---------------------------------------------------------------------------
// The three-launch path.
//
// Five launches cost five times a 3-4 us floor, so the plan asked for the count to come
// down. Doing that means fusing each pointwise stage into the depthwise stage next to it,
// which in turn means the pointwise stages can no longer be cuBLAS calls -- so they are
// written here with `wmma`, which is the only way a hand-written version gets within
// reach of cuBLAS's 1.6 us of arithmetic (the best of ten FFMA/half2 tilings needed 13).
//
// The ownership rule these two kernels obey is stronger than the halo argument the plan
// asked for. A halo argument is needed when a block reads an intermediate row that a
// *different* block wrote, which within one launch has no ordering guarantee: the
// neighbour may not have started, may be mid-write, or may be unable to start because
// the reader holds the residency the scheduler would have given it, and `__syncthreads()`
// does not reach across blocks. Here **no intermediate is ever written to global memory
// at all** -- the fused stage lives only in the writing block's shared memory, and each
// block recomputes whatever halo it needs from the previous *launch's* output, which is
// globally visible because a launch boundary orders it. So there is no cross-block read
// to reason about, and `cib_forward3` allocates no buffer for either intermediate.
//
// The cost of that is recomputation, and it is bounded: `pw_dw3_fused` computes
// `ROWS + 2` rows of the pointwise stage for every `ROWS` rows it owns, so at ROWS=4 it
// does 1.5x the pointwise arithmetic. `dw3_pw_fused` recomputes nothing, because a
// pointwise stage needs its input only at the position it is writing.
namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kFusedWarps = 8;
constexpr int kFusedThreads = 32 * kFusedWarps;
// Spatial positions per block in the first fused kernel: one wmma N-tile.
constexpr int kMTile = kWmmaN;
// Output rows per block in the second fused kernel, and the padded position count its
// shared buffers hold: (ROWS + 2) halo rows times the row width must fit.
constexpr int kFusedRows = 4;
constexpr int kPosPad = 128;
// Output channels one block of `dw3_pw_fused` produces. Splitting them over blockIdx.y is
// what keeps the grid large enough to fill the device: all 256 in one block gives 100
// blocks over 148 SMs, which measured latency-bound at 8% issue efficiency. The cost is
// that the depthwise stage is recomputed once per group, and that stage is 1.8 MMAC of
// the block's 128.
constexpr int kOcGroup = 128;
// Input channels staged per pass in `pw_dw3_fused`. One wmma K-step per pass would mean
// two barriers per step, 32 for a 256-channel contraction; staging four steps at a time
// cuts that to eight.
constexpr int kStageK = 64;

// Stages 0 and 1: depthwise 3x3 + bias + SiLU, then the pointwise expansion + bias +
// SiLU, in one launch. One block owns one image and kMTile spatial positions, and
// produces *all* mid_channels outputs for them.
__global__ __launch_bounds__(kFusedThreads) void dw3_pw_fused(
    const __half* __restrict__ x, int x_batch_stride, const __half* __restrict__ w_dw,
    const __half* __restrict__ b_dw, const __half* __restrict__ w_pw,
    const __half* __restrict__ b_pw, __half* __restrict__ out, int in_channels,
    int mid_channels, int H, int W, int HW, bool use_pdl) {
  extern __shared__ char fused_smem[];
  __half* act = reinterpret_cast<__half*>(fused_smem);          // [in_channels][kMTile]
  float* tile = reinterpret_cast<float*>(act + in_channels * kMTile);  // [warps][16*16]

  const int m0 = blockIdx.x * kMTile;
  const int oc_base = blockIdx.y * kOcGroup;
  const int image = blockIdx.z;
  const __half* xn = x + (long long)image * x_batch_stride;

  if (use_pdl) {
    // Nothing before this reads the previous launch's output, so the wait costs nothing
    // beyond the dependency itself.
    cudaGridDependencySynchronize();
  }

  // Stage 0, straight from the input, into shared memory. Each (channel, position) pair
  // gathers its own 3x3 window with the borders clipped, so the zero padding is a bounds
  // test and no thread can reach outside the (image, channel) plane it belongs to.
  for (int idx = threadIdx.x; idx < in_channels * kMTile; idx += kFusedThreads) {
    const int c = idx / kMTile;
    const int p = idx - c * kMTile;
    const int m = m0 + p;
    float acc = 0.0f;
    if (m < HW) {
      const int row = m / W;
      const int col = m - row * W;
      const __half* plane = xn + (long long)c * HW;
      const __half* taps = w_dw + (long long)c * 9;
#pragma unroll
      for (int kh = 0; kh < 3; ++kh) {
        const int r = row + kh - 1;
        if (r < 0 || r >= H) {
          continue;
        }
#pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
          const int cc = col + kw - 1;
          if (cc < 0 || cc >= W) {
            continue;
          }
          acc = fmaf(__half2float(taps[kh * 3 + kw]),
                     __half2float(plane[(long long)r * W + cc]), acc);
        }
      }
      acc = silu(acc + __half2float(b_dw[c]));
    }
    act[(long long)c * kMTile + p] = __float2half(m < HW ? acc : 0.0f);
  }
  __syncthreads();

  // Stage 1 as a wmma GEMM: C[16 out-channels x kMTile positions] += A[16 x 16] * B[16 x
  // kMTile] over the input-channel axis. A is the pointwise weight read straight from
  // global (row-major, leading dimension in_channels); B is the staged activation.
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int oc_tiles = kOcGroup / kWmmaM;
  for (int t = warp; t < oc_tiles; t += kFusedWarps) {
    const int oc_tile_base = oc_base + t * kWmmaM;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    for (int k0 = 0; k0 < in_channels; k0 += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> b;
      wmma::load_matrix_sync(a, w_pw + (long long)oc_tile_base * in_channels + k0,
                             in_channels);
      wmma::load_matrix_sync(b, act + (long long)k0 * kMTile, kMTile);
      wmma::mma_sync(acc_frag, a, b, acc_frag);
    }
    float* mine = tile + (long long)warp * kWmmaM * kWmmaN;
    wmma::store_matrix_sync(mine, acc_frag, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < kWmmaM * kWmmaN; i += 32) {
      const int oc = oc_tile_base + i / kWmmaN;
      const int m = m0 + (i % kWmmaN);
      if (m < HW) {
        out[(long long)image * mid_channels * HW + (long long)oc * HW + m] =
            __float2half(silu(mine[i] + __half2float(b_pw[oc])));
      }
    }
    __syncwarp();
  }
  if (use_pdl) {
    // Every store this grid's consumer reads has retired by here.
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Stages 3 and 4: the pointwise contraction + bias + SiLU, then the depthwise 3x3 +
// bias + SiLU + residual, in one launch. One block owns one image, kWmmaM output
// channels, and kFusedRows output rows. Because the depthwise stage reads one row above
// and below, the block computes the pointwise stage for its rows *plus* those two, from
// the previous launch's globally visible output -- never from a neighbour's.
__global__ __launch_bounds__(kFusedThreads) void pw_dw3_fused(
    const __half* __restrict__ mid, const __half* __restrict__ w_pw,
    const __half* __restrict__ b_pw, const __half* __restrict__ w_dw,
    const __half* __restrict__ b_dw, const __half* __restrict__ residual,
    int res_batch_stride, __half* __restrict__ out, int mid_channels, int out_channels,
    int H, int W, int HW, bool add_residual, bool use_pdl) {
  __shared__ __half staged[kStageK][kPosPad];     // several K-slices of the previous output
  __shared__ __half activated[kWmmaM][kPosPad];   // this block's pointwise result
  __shared__ float tile[kFusedWarps][kWmmaM * kWmmaN];

  const int row0 = blockIdx.x * kFusedRows;
  const int oc0 = blockIdx.y * kWmmaM;
  const int image = blockIdx.z;

  // The rows this block must produce the pointwise stage for: its own, plus one above and
  // one below, clipped to the image. Rows outside the image do not exist in the input
  // either, and the depthwise tap loop below treats them as the zero padding they are.
  const int row_lo = row0 > 0 ? row0 - 1 : 0;
  int row_hi = row0 + kFusedRows + 1;
  if (row_hi > H) {
    row_hi = H;
  }
  const int m_lo = row_lo * W;
  const int npos = (row_hi - row_lo) * W;
  const int pos_tiles = (npos + kWmmaN - 1) / kWmmaN;

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const __half* midn = mid + (long long)image * mid_channels * HW;

  if (use_pdl) {
    cudaGridDependencySynchronize();
  }

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);
  for (int kbase = 0; kbase < mid_channels; kbase += kStageK) {
    // Stage several K-slices into shared memory rather than pointing wmma at global: the
    // global row start is m_lo = row_lo * W, which for W = 20 is not a multiple of the
    // eight halves `load_matrix_sync` requires the pointer to be aligned to. Staging
    // kStageK at a time amortizes the two barriers a pass costs over kStageK/kWmmaK
    // wmma steps.
    for (int i = threadIdx.x; i < kStageK * kPosPad; i += kFusedThreads) {
      const int kk = i / kPosPad;
      const int pp = i - kk * kPosPad;
      staged[kk][pp] = (pp < npos && kbase + kk < mid_channels)
                           ? midn[(long long)(kbase + kk) * HW + m_lo + pp]
                           : __float2half(0.0f);
    }
    __syncthreads();
    if (warp < pos_tiles) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> b;
#pragma unroll
      for (int ks = 0; ks < kStageK; ks += kWmmaK) {
        wmma::load_matrix_sync(a, w_pw + (long long)oc0 * mid_channels + kbase + ks,
                               mid_channels);
        wmma::load_matrix_sync(b, &staged[ks][warp * kWmmaN], kPosPad);
        wmma::mma_sync(acc_frag, a, b, acc_frag);
      }
    }
    __syncthreads();
  }
  if (warp < pos_tiles) {
    wmma::store_matrix_sync(tile[warp], acc_frag, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < kWmmaM * kWmmaN; i += 32) {
      const int oc_local = i / kWmmaN;
      const int p = warp * kWmmaN + (i % kWmmaN);
      if (p < npos) {
        activated[oc_local][p] =
            __float2half(silu(tile[warp][i] + __half2float(b_pw[oc0 + oc_local])));
      }
    }
  }
  __syncthreads();

  // Stage 4, entirely out of this block's own shared memory.
  __half* outn = out + (long long)image * out_channels * HW;
  const __half* resn =
      add_residual ? residual + (long long)image * res_batch_stride : nullptr;
  for (int idx = threadIdx.x; idx < kWmmaM * kFusedRows * W; idx += kFusedThreads) {
    const int oc_local = idx / (kFusedRows * W);
    const int rem = idx - oc_local * (kFusedRows * W);
    const int rr = rem / W;
    const int col = rem - rr * W;
    const int row = row0 + rr;
    if (row >= H) {
      continue;
    }
    const int oc = oc0 + oc_local;
    const __half* taps = w_dw + (long long)oc * 9;
    float acc = __half2float(b_dw[oc]);
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
      const int r = row + kh - 1;
      if (r < 0 || r >= H) {
        continue;
      }
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        const int cc = col + kw - 1;
        if (cc < 0 || cc >= W) {
          continue;
        }
        acc = fmaf(__half2float(taps[kh * 3 + kw]),
                   __half2float(activated[oc_local][(r - row_lo) * W + cc]), acc);
      }
    }
    float value = silu(acc);
    if (add_residual) {
      value += __half2float(resn[(long long)oc * HW + row * W + col]);
    }
    outn[(long long)oc * HW + (long long)row * W + col] = __float2half(value);
  }
  if (use_pdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Whether the three-launch kernels cover a geometry. Deliberately narrow: the wmma tiles
// need channel counts that are multiples of 16, and the second kernel's shared buffers
// hold (kFusedRows + 2) rows of width W.
bool geometry_supported3(int in_channels, int mid_channels, int out_channels, int H,
                         int W) {
  return in_channels > 0 && mid_channels > 0 && out_channels > 0 && H > 0 && W > 0 &&
         in_channels % kWmmaK == 0 && mid_channels % kOcGroup == 0 &&
         mid_channels % kStageK == 0 && out_channels % kWmmaM == 0 &&
         (kFusedRows + 2) * W <= kPosPad && H * W <= kMaxPlaneElems;
}

// Whether every image of an NCHW tensor is internally contiguous. The batch stride is
// deliberately not constrained: the scored 4-image case is a slice of a wider tensor,
// with images 102400 elements apart against 51200 per image, so requiring full
// contiguity here would reject it.
bool images_are_contiguous(const at::Tensor& t) {
  return t.dim() == 4 && t.stride(3) == 1 && t.stride(2) == t.size(3) &&
         t.stride(1) == (long long)t.size(2) * t.size(3);
}

}  // namespace

// Issues the whole block on the current stream inside one crossing.
//
// ``packed_weight`` and ``packed_bias`` are the five folded stages laid end to end in
// forward order, exactly as ``YOLOCIB._derive_folded`` packs them; the offsets below are
// that layout read back. Packing keeps the crossing to six arguments and gives the
// weights one contiguous residency instead of ten.
//
// The two pointwise biases are *not* consumed where their GEMM runs -- cuBLAS has no
// epilogue -- but on stage-in of the depthwise kernel that reads the GEMM's output. So
// every one of the five folded biases and both SiLUs are still applied exactly once,
// and no separate elementwise pass exists.
//
// Scratch is one allocation carved by offset into four disjoint stage buffers, so every
// element any kernel reads was written by an earlier kernel of this call, or is a
// compile-time zero of a padded stencil. The buffers are sized from the stage geometry,
// and the packed tensors are checked against the element counts that geometry implies
// before any pointer is formed.
at::Tensor cib_forward(const at::Tensor& x, const at::Tensor& packed_weight,
                       const at::Tensor& packed_bias, int64_t mid_channels,
                       int64_t out_channels, bool add_residual) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 4, "input must be NCHW");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(images_are_contiguous(x), "each image of the input must be contiguous");
  TORCH_CHECK(packed_weight.is_cuda() && packed_weight.is_contiguous() &&
                  packed_weight.scalar_type() == at::kHalf,
              "packed weights must be contiguous CUDA float16");
  TORCH_CHECK(packed_bias.is_cuda() && packed_bias.is_contiguous() &&
                  packed_bias.scalar_type() == at::kHalf,
              "packed biases must be contiguous CUDA float16");
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "the fused path records no autograd node; call it under no_grad");
  // Defense in depth for the Python guard: every pointer this call forms must belong to
  // the device the guard below installs, and none of them may carry a lazy neg/conj flag
  // that only ATen would apply.
  TORCH_CHECK(packed_weight.device() == x.device() && packed_bias.device() == x.device(),
              "folded parameters live on ", packed_weight.device(), " but the input is on ",
              x.device());
  TORCH_CHECK(!x.is_neg() && !x.is_conj() && !packed_weight.is_neg() &&
                  !packed_weight.is_conj() && !packed_bias.is_neg() &&
                  !packed_bias.is_conj(),
              "this path reads storage directly and cannot honour a lazy neg/conj view");

  const int N = (int)x.size(0);
  const int c1 = (int)x.size(1);
  const int H = (int)x.size(2);
  const int W = (int)x.size(3);
  const int cmid = (int)mid_channels;
  const int c2 = (int)out_channels;
  TORCH_CHECK(N >= 1 && c1 >= 1 && H >= 1 && W >= 1, "empty input");
  TORCH_CHECK(cmid >= 1 && c2 >= 1, "channel counts must be positive");
  TORCH_CHECK(H * W <= kMaxPlaneElems, "a plane must fit the depthwise staging buffer");
  TORCH_CHECK(!add_residual || c1 == c2,
              "the residual requires matching input and output channel counts");

  const int HW = H * W;
  // Stage weight and bias counts, in forward order.
  const long long wn[5] = {(long long)c1 * 9, (long long)cmid * c1, (long long)cmid * 49,
                           (long long)c2 * cmid, (long long)c2 * 9};
  const long long bn[5] = {c1, cmid, cmid, c2, c2};
  long long wo[5], bo[5], w_total = 0, b_total = 0;
  for (int i = 0; i < 5; ++i) {
    wo[i] = w_total;
    bo[i] = b_total;
    w_total += wn[i];
    b_total += bn[i];
  }
  TORCH_CHECK(packed_weight.numel() == w_total, "packed weights have ",
              packed_weight.numel(), " elements, expected ", w_total);
  TORCH_CHECK(packed_bias.numel() == b_total, "packed biases have ", packed_bias.numel(),
              " elements, expected ", b_total);
  TORCH_CHECK((long long)N * cmid * HW <= (long long)std::numeric_limits<int>::max(),
              "activation too large for this kernel");

  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto y = at::empty({N, c2, H, W}, x.options());
  // t0 (c1) -> t1 (cmid, raw GEMM) -> t2 (cmid) -> t3 (c2, raw GEMM); disjoint, one
  // allocation.
  const long long n0 = (long long)N * c1 * HW;
  const long long n1 = (long long)N * cmid * HW;
  const long long n2 = n1;
  const long long n3 = (long long)N * c2 * HW;
  auto scratch = at::empty({n0 + n1 + n2 + n3}, x.options());

  const __half* xp = reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>());
  const __half* wp =
      reinterpret_cast<const __half*>(packed_weight.const_data_ptr<at::Half>());
  const __half* bp =
      reinterpret_cast<const __half*>(packed_bias.const_data_ptr<at::Half>());
  __half* sp = reinterpret_cast<__half*>(scratch.data_ptr<at::Half>());
  __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

  __half* t0 = sp;
  __half* t1 = t0 + n0;
  __half* t2 = t1 + n1;
  __half* t3 = t2 + n2;

  // The input's own batch stride; every scratch buffer is fully contiguous.
  const int xs = (int)x.stride(0);
  const int s1 = c1 * HW, sm = cmid * HW, s2 = c2 * HW;

  launch_dw<3, false, false>(xp, nullptr, wp + wo[0], bp + bo[0], nullptr, t0, N * c1, c1,
                             H, W, xs, 0, stream);
  launch_gemm(t0, wp + wo[1], t1, N, c1, cmid, HW, s1, stream);
  launch_dw<7, false, true>(t1, bp + bo[1], wp + wo[2], bp + bo[2], nullptr, t2, N * cmid,
                            cmid, H, W, sm, 0, stream);
  launch_gemm(t2, wp + wo[3], t3, N, cmid, c2, HW, sm, stream);
  if (add_residual) {
    launch_dw<3, true, true>(t3, bp + bo[3], wp + wo[4], bp + bo[4], xp, yp, N * c2, c2, H,
                             W, s2, xs, stream);
  } else {
    launch_dw<3, false, true>(t3, bp + bo[3], wp + wo[4], bp + bo[4], nullptr, yp, N * c2,
                              c2, H, W, s2, 0, stream);
  }
  return y;
}

// Whether the three-launch kernels cover a geometry, for the Python guard to ask once.
bool cib_geometry_supported3(int64_t in_channels, int64_t mid_channels,
                             int64_t out_channels, int64_t H, int64_t W) {
  return geometry_supported3((int)in_channels, (int)mid_channels, (int)out_channels, (int)H,
                             (int)W);
}

// The scored path: the whole block in three launches, on the current stream, inside one
// crossing.
//
// The middle intermediate is the only global buffer, because the two fused kernels keep
// their fused stage in shared memory. So this allocates one scratch tensor rather than
// four, and there is no cross-block intermediate for a halo argument to be about.
//
// `use_pdl` opts each launch into programmatic dependent launch, letting a kernel begin
// during the tail of the one before it. It is a parameter rather than a constant so the
// two can be measured against each other and differential-tested against the five-launch
// path both ways.
at::Tensor cib_forward3(const at::Tensor& x, const at::Tensor& packed_weight,
                        const at::Tensor& packed_bias, int64_t mid_channels,
                        int64_t out_channels, bool add_residual, bool use_pdl) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 4, "input must be NCHW");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(images_are_contiguous(x), "each image of the input must be contiguous");
  TORCH_CHECK(packed_weight.is_cuda() && packed_weight.is_contiguous() &&
                  packed_weight.scalar_type() == at::kHalf,
              "packed weights must be contiguous CUDA float16");
  TORCH_CHECK(packed_bias.is_cuda() && packed_bias.is_contiguous() &&
                  packed_bias.scalar_type() == at::kHalf,
              "packed biases must be contiguous CUDA float16");
  TORCH_CHECK(!at::GradMode::is_enabled(),
              "the fused path records no autograd node; call it under no_grad");
  TORCH_CHECK(packed_weight.device() == x.device() && packed_bias.device() == x.device(),
              "folded parameters live on ", packed_weight.device(), " but the input is on ",
              x.device());
  TORCH_CHECK(!x.is_neg() && !x.is_conj() && !packed_weight.is_neg() &&
                  !packed_weight.is_conj() && !packed_bias.is_neg() &&
                  !packed_bias.is_conj(),
              "this path reads storage directly and cannot honour a lazy neg/conj view");

  const int N = (int)x.size(0);
  const int c1 = (int)x.size(1);
  const int H = (int)x.size(2);
  const int W = (int)x.size(3);
  const int cmid = (int)mid_channels;
  const int c2 = (int)out_channels;
  const int HW = H * W;
  TORCH_CHECK(N >= 1, "empty input");
  TORCH_CHECK(geometry_supported3(c1, cmid, c2, H, W),
              "this geometry is outside what the three-launch kernels implement");
  TORCH_CHECK(!add_residual || c1 == c2,
              "the residual requires matching input and output channel counts");

  const long long wn[5] = {(long long)c1 * 9, (long long)cmid * c1, (long long)cmid * 49,
                           (long long)c2 * cmid, (long long)c2 * 9};
  const long long bn[5] = {c1, cmid, cmid, c2, c2};
  long long wo[5], bo[5], w_total = 0, b_total = 0;
  for (int i = 0; i < 5; ++i) {
    wo[i] = w_total;
    bo[i] = b_total;
    w_total += wn[i];
    b_total += bn[i];
  }
  TORCH_CHECK(packed_weight.numel() == w_total, "packed weights have ",
              packed_weight.numel(), " elements, expected ", w_total);
  TORCH_CHECK(packed_bias.numel() == b_total, "packed biases have ", packed_bias.numel(),
              " elements, expected ", b_total);

  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto y = at::empty({N, c2, H, W}, x.options());
  // Two buffers, both mid_channels wide: the expansion's output and the merged 7x7's.
  const long long nmid = (long long)N * cmid * HW;
  auto scratch = at::empty({2 * nmid}, x.options());

  const __half* xp = reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>());
  const __half* wp =
      reinterpret_cast<const __half*>(packed_weight.const_data_ptr<at::Half>());
  const __half* bp =
      reinterpret_cast<const __half*>(packed_bias.const_data_ptr<at::Half>());
  __half* t1 = reinterpret_cast<__half*>(scratch.data_ptr<at::Half>());
  __half* t2 = t1 + nmid;
  __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

  const int xs = (int)x.stride(0);
  const int m_tiles = (HW + kMTile - 1) / kMTile;
  const size_t shared0 =
      (size_t)c1 * kMTile * sizeof(__half) + (size_t)kFusedWarps * kWmmaM * kWmmaN * sizeof(float);
  const dim3 grid0(m_tiles, cmid / kOcGroup, N);
  const dim3 grid2((H + kFusedRows - 1) / kFusedRows, c2 / kWmmaM, N);

  if (use_pdl) {
    launch_pdl(dw3_pw_fused, grid0, dim3(kFusedThreads), shared0, stream, xp, xs,
               wp + wo[0], bp + bo[0], wp + wo[1], bp + bo[1], t1, c1, cmid, H, W, HW,
               true);
  } else {
    dw3_pw_fused<<<grid0, kFusedThreads, shared0, stream>>>(
        xp, xs, wp + wo[0], bp + bo[0], wp + wo[1], bp + bo[1], t1, c1, cmid, H, W, HW,
        false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // The merged depthwise 7x7 is unchanged; its input already carries its epilogue.
  launch_dw<7, false, false>(t1, nullptr, wp + wo[2], bp + bo[2], nullptr, t2, N * cmid,
                             cmid, H, W, cmid * HW, 0, stream, use_pdl);

  if (use_pdl) {
    launch_pdl(pw_dw3_fused, grid2, dim3(kFusedThreads), 0, stream, t2, wp + wo[3],
               bp + bo[3], wp + wo[4], bp + bo[4], add_residual ? xp : nullptr, xs, yp,
               cmid, c2, H, W, HW, add_residual, true);
  } else {
    pw_dw3_fused<<<grid2, kFusedThreads, 0, stream>>>(
        t2, wp + wo[3], bp + bo[3], wp + wo[4], bp + bo[4], add_residual ? xp : nullptr,
        xs, yp, cmid, c2, H, W, HW, add_residual, false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}

// One depthwise stage on its own, so each instantiation can be tested against
// ``F.conv2d(..., bias) -> F.silu`` at the shape it actually sees. ``source_bias``, when
// given, exercises the stage-in epilogue the fused path relies on. Not on the scored
// path; ``cib_forward`` calls the launchers directly.
at::Tensor cib_depthwise_stage(const at::Tensor& x, const at::Tensor& weight,
                               const at::Tensor& bias,
                               const std::optional<at::Tensor>& residual,
                               const std::optional<at::Tensor>& source_bias) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.scalar_type() == at::kHalf &&
                  images_are_contiguous(x),
              "input must be a 4-D CUDA float16 tensor with contiguous images");
  TORCH_CHECK(weight.is_cuda() && weight.dim() == 4 && weight.size(1) == 1 &&
                  weight.scalar_type() == at::kHalf && weight.is_contiguous(),
              "weight must be a contiguous depthwise CUDA float16 tensor");
  const int K = (int)weight.size(2);
  TORCH_CHECK(weight.size(3) == K, "only square kernels are instantiated");
  TORCH_CHECK(K == 3 || K == 7, "only the 3x3 and 7x7 instantiations exist");
  const int N = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2),
            W = (int)x.size(3);
  TORCH_CHECK(weight.size(0) == C, "one filter per input channel");
  TORCH_CHECK(bias.numel() == C, "one bias per channel");
  TORCH_CHECK(N >= 1 && C >= 1 && H >= 1 && W >= 1, "empty input");
  TORCH_CHECK(H * W <= kMaxPlaneElems, "a plane must fit the depthwise staging buffer");
  TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device(),
              "weight and bias must live on the input's device");
  TORCH_CHECK(!x.is_neg() && !x.is_conj() && !weight.is_neg() && !weight.is_conj(),
              "this path reads storage directly and cannot honour a lazy neg/conj view");

  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto y = at::empty(x.sizes(), x.options());
  const __half* xp = reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>());
  const __half* wp = reinterpret_cast<const __half*>(weight.const_data_ptr<at::Half>());
  const __half* bp = reinterpret_cast<const __half*>(bias.const_data_ptr<at::Half>());
  __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

  const __half* rp = nullptr;
  int rs = 0;
  if (residual.has_value()) {
    const at::Tensor& r = residual.value();
    TORCH_CHECK(r.is_cuda() && images_are_contiguous(r) && r.scalar_type() == at::kHalf &&
                    r.sizes() == x.sizes(),
                "residual must match the input");
    rp = reinterpret_cast<const __half*>(r.const_data_ptr<at::Half>());
    rs = (int)r.stride(0);
  }
  const __half* sbp = nullptr;
  if (source_bias.has_value()) {
    const at::Tensor& sb = source_bias.value();
    TORCH_CHECK(sb.is_cuda() && sb.is_contiguous() && sb.scalar_type() == at::kHalf &&
                    sb.numel() == C,
                "source bias must be one contiguous CUDA float16 value per channel");
    sbp = reinterpret_cast<const __half*>(sb.const_data_ptr<at::Half>());
  }

  const int planes = N * C;
  const int xs = (int)x.stride(0);
#define CIB_DW_CASE(KK, RES, EPI)                                                     \
  launch_dw<KK, RES, EPI>(xp, sbp, wp, bp, rp, yp, planes, C, H, W, xs, rs, stream)
  if (K == 3) {
    if (rp && sbp) {
      CIB_DW_CASE(3, true, true);
    } else if (rp) {
      CIB_DW_CASE(3, true, false);
    } else if (sbp) {
      CIB_DW_CASE(3, false, true);
    } else {
      CIB_DW_CASE(3, false, false);
    }
  } else {
    if (rp && sbp) {
      CIB_DW_CASE(7, true, true);
    } else if (rp) {
      CIB_DW_CASE(7, true, false);
    } else if (sbp) {
      CIB_DW_CASE(7, false, true);
    } else {
      CIB_DW_CASE(7, false, false);
    }
  }
#undef CIB_DW_CASE
  return y;
}

// One pointwise stage on its own: the batched GEMM followed by the epilogue that the
// fused path folds into the next depthwise stage. The batch strides are what this
// exists to test -- a formulation that flattened NCHW into one C x (N*H*W) matrix would
// agree here at N == 1 and disagree at N > 1.
at::Tensor cib_pointwise_stage(const at::Tensor& x, const at::Tensor& weight,
                               const at::Tensor& bias) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.scalar_type() == at::kHalf &&
                  images_are_contiguous(x),
              "input must be a 4-D CUDA float16 tensor with contiguous images");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kHalf &&
                  weight.is_contiguous(),
              "weight must be contiguous CUDA float16");
  TORCH_CHECK(weight.dim() == 4 && weight.size(2) == 1 && weight.size(3) == 1,
              "the pointwise stage takes a 1x1 kernel");
  const int N = (int)x.size(0), IC = (int)x.size(1), H = (int)x.size(2),
            W = (int)x.size(3);
  const int OC = (int)weight.size(0);
  TORCH_CHECK(weight.size(1) == IC, "weight input channels must match the activation");
  TORCH_CHECK(bias.numel() == OC, "one bias per output channel");
  TORCH_CHECK(N >= 1 && IC >= 1 && OC >= 1 && H >= 1 && W >= 1, "empty input");
  TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device(),
              "weight and bias must live on the input's device");
  TORCH_CHECK(!x.is_neg() && !x.is_conj() && !weight.is_neg() && !weight.is_conj(),
              "this path reads storage directly and cannot honour a lazy neg/conj view");

  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int HW = H * W;
  auto raw = at::empty({N, OC, H, W}, x.options());
  auto y = at::empty({N, OC, H, W}, x.options());
  launch_gemm(reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>()),
              reinterpret_cast<const __half*>(weight.const_data_ptr<at::Half>()),
              reinterpret_cast<__half*>(raw.data_ptr<at::Half>()), N, IC, OC, HW,
              (int)x.stride(0), stream);
  const long long total = (long long)N * OC * HW;
  int blocks = (int)((total + kEpilogueThreads - 1) / kEpilogueThreads);
  if (blocks > 8192) {
    blocks = 8192;
  }
  bias_silu<<<blocks, kEpilogueThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(raw.const_data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(bias.const_data_ptr<at::Half>()),
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), OC, HW, total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""


def _workspace_root() -> Path:
    """Directory to anchor build products in.

    The harness imports this file in place, so ``__file__`` is the real path inside the
    operator workspace. Keeping build products here rather than in the shared
    ``~/.cache/torch_extensions`` is what stops the concurrently running sibling
    operator workspaces from contending over one ninja lock or loading each other's
    ``.so``.
    """
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "validate.py").is_file():
            return parent
    return here


def _target_arch() -> str:
    """The single architecture to compile for.

    This workspace's shell exports a six-architecture ``TORCH_CUDA_ARCH_LIST``, which
    would multiply a cold build sixfold. Pinning is also outright required when the
    variable is unset or ``native`` with no visible GPU, where PyTorch's
    ``_get_cuda_arch_flags`` indexes an empty device list.
    """
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}.{minor}"
    except Exception:
        pass
    return "10.0"


def _one_line(text: str, limit: int = 800) -> str:
    """Collapse a message to one physical line, truncating in the middle if long.

    Compiler and ninja errors are multi-line by nature; the middle is dropped rather than
    the tail because the last lines are usually the ones that name the failure.
    """
    flat = " | ".join(part.strip() for part in str(text).splitlines() if part.strip())
    if len(flat) <= limit:
        return flat
    head = limit // 2
    return f"{flat[:head]} ...[{len(flat) - limit} chars elided]... {flat[-(limit - head):]}"


def _build_extension():
    from torch.utils.cpp_extension import load_inline

    build_dir = _workspace_root() / ".torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    source = (
        _CUDA_SOURCE.replace("__DW_PIX__", str(_DW_PIX))
        .replace("__DW_THREADS__", str(_DW_THREADS))
        .replace("__MAX_PLANE__", str(_MAX_PLANE_ELEMS))
    )

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _target_arch()
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=source,
            functions=[
                "cib_forward",
                "cib_forward3",
                "cib_geometry_supported3",
                "cib_depthwise_stage",
                "cib_pointwise_stage",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            # The pointwise stages go through cuBLAS; torch already links it, but the
            # symbol has to be resolvable from this translation unit.
            extra_ldflags=["-lcublas"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


#: Set when the extension compiled and loaded. When False the module still works: an
#: otherwise-eligible input takes the folded eager chain, and everything else takes the
#: reference block body.
FAST_PATH_AVAILABLE = False
#: Populated with the build failure when ``FAST_PATH_AVAILABLE`` is False.
BUILD_ERROR: str | None = None

_extension = None
try:
    _extension = _build_extension()
    FAST_PATH_AVAILABLE = True
except Exception as exc:  # a build failure must not make this module unimportable
    # nvcc and ninja errors run to dozens of lines. Collapse to one physical line so the
    # promise of a single diagnostic is kept literally: a workspace that shares a machine
    # with a dozen sibling operator builds should not have its log buried.
    BUILD_ERROR = _one_line(f"{type(exc).__name__}: {exc}")
    print(
        f"[{_EXTENSION_NAME}] CUDA extension unavailable, falling back to the folded "
        f"eager chain: {BUILD_ERROR}",
        file=sys.stderr,
        flush=True,
    )


#: Indices into the per-instance tier record. ``FUSED3`` is the scored path -- three
#: launches, both pointwise stages fused into a neighbouring depthwise stage with `wmma`.
#: ``FUSED5`` is the five-launch path with cuBLAS pointwise stages, kept as the
#: differential reference and used for geometries the fused kernels do not implement.
#: ``FOLDED`` is the same arithmetic in eager ATen, taken when the extension did not
#: build; ``REFERENCE`` is the literal reference block body.
TIER_FUSED3 = 0
TIER_FUSED5 = 1
TIER_FOLDED = 2
TIER_REFERENCE = 3

#: Route eligible calls through the three-launch kernels when their geometry is covered.
#:
#: False, by measurement. Fusing each pointwise stage into a neighbouring depthwise stage
#: removes two launches, worth about 6 us at a 3 us floor, and costs about 51 us of
#: arithmetic and occupancy: paired in one clock block under the harness's own timing loop,
#: three launches measure 94.2 us against five launches' 49.1 us at N=4 (85.0 against 43.0
#: at N=1). NCU says why. ``pw_dw3_fused`` runs 4.08 M instructions at 10.8% issue
#: efficiency over 160 blocks, where cuBLAS's ``nvjet`` kernel does the same contraction in
#: 0.15 M -- a 16x16x16 ``wmma`` tile per warp with a shared staging pass around it is not a
#: competitive GEMM, and closing that would mean a real Blackwell GEMM (tcgen05/TMEM, 2-SM
#: cooperative, software-pipelined) fused with the depthwise stages.
#:
#: The three-launch path is kept, correct and differential-tested, because it is the
#: experiment that establishes the above rather than predicting it; flipping this constant
#: reproduces the measurement. See ``profile/cib_v6_three_launch/REPORT.md``.
PREFER_THREE_LAUNCH = False
#: Opt the three-launch path's kernels into programmatic dependent launch, so each may
#: begin during the tail of the one before it. Measured at 12.31 -> 8.33 us over a
#: three-stage chain; see ``profile/cib_v5_pdl_coop/``.
USE_PDL = True


def _fold_conv_bn(conv: nn.Module, bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """One convolution and its batch norm as a single biased convolution.

    All of the arithmetic is fp32. ``bn.weight`` and ``bn.bias`` are parameters and so
    have been cast to the run dtype by the harness; ``running_mean`` and ``running_var``
    are buffers and remain fp32, since the harness casts parameters only. ``eps`` is
    read from the module because this block uses ``1e-3``, not PyTorch's ``1e-5``.
    """
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    weight = conv.weight.float() * scale.view(-1, *([1] * (conv.weight.dim() - 1)))
    bias = bn.bias.float() - bn.running_mean.float() * scale
    if conv.bias is not None:
        bias = bias + conv.bias.float() * scale
    return weight, bias


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

        # Geometry the fused path needs, resolved once. These are plain attributes, not
        # buffers, so they cannot reach ``state_dict``.
        self._cib_c1 = c1
        self._cib_mid = 2 * c_
        self._cib_c2 = c2
        # The structural verdict: does this configuration have the shape the kernels
        # implement? Answered once here, from module types rather than constructor
        # arguments, so a hand-assembled tree cannot slip past it.
        self._cib_structural = self._structurally_supported()
        # None until the first eligible call derives it; see the module docstring for
        # why this is lazy and why nothing checks it per call.
        self._cib_folded: tuple[torch.Tensor, torch.Tensor, list] | None = None
        # Per-tier call counts, indexed by TIER_*. A list so the increment in
        # ``forward`` is an item assignment rather than an ``nn.Module.__setattr__``.
        self._cib_tiers = [0, 0, 0, 0]
        # Whether the three-launch kernels cover a given (H, W); the channel counts are
        # fixed at construction, so only the spatial extent can change between calls.
        # Asked of the extension once per distinct extent, then cached.
        self._cib_geom3: dict[tuple[int, int], bool] = {}
        self.register_load_state_dict_post_hook(self._cib_invalidate_after_load)

    # -- structure and lifecycle ------------------------------------------------

    def _structurally_supported(self) -> bool:
        """Whether the tree is the five-stage form with a merged RepVGG middle stage.

        The kernels implement exactly that shape: a depthwise 3x3, a pointwise, a
        depthwise 7x7 that is the RepVGG merge of a 7x7 and a 3x3 branch, a pointwise,
        and a depthwise 3x3. ``lk=False`` builds a plain depthwise 3x3 in the middle
        instead, which this does not cover, so it takes the reference path.

        A stage that has already been through the reference's own ``fuse()`` has had its
        norm deleted, so there is nothing left to fold; such a tree is also rejected.
        """
        cv = self.cv1
        if not isinstance(cv, nn.Sequential) or len(cv) != 5:
            return False
        if not isinstance(cv[2], YOLORepVGGDW):
            return False
        plain = (cv[0], cv[1], cv[3], cv[4])
        if not all(isinstance(m, YOLOConv) for m in plain):
            return False
        if any(getattr(m, "_is_fused", False) for m in plain):
            return False
        rep = cv[2]
        if getattr(rep, "_is_fused", False):
            return False
        if not (isinstance(rep.conv, YOLOConv) and isinstance(rep.conv1, YOLOConv)):
            return False
        if any(getattr(m, "_is_fused", False) for m in (rep.conv, rep.conv1)):
            return False
        geometry = (
            (cv[0].conv, (self._cib_c1, 1, 3, 3), self._cib_c1, (1, 1)),
            (cv[1].conv, (self._cib_mid, self._cib_c1, 1, 1), 1, (0, 0)),
            (rep.conv.conv, (self._cib_mid, 1, 7, 7), self._cib_mid, (3, 3)),
            (rep.conv1.conv, (self._cib_mid, 1, 3, 3), self._cib_mid, (1, 1)),
            (cv[3].conv, (self._cib_c2, self._cib_mid, 1, 1), 1, (0, 0)),
            (cv[4].conv, (self._cib_c2, 1, 3, 3), self._cib_c2, (1, 1)),
        )
        for conv, shape, groups, pad in geometry:
            if tuple(conv.weight.shape) != shape or conv.groups != groups:
                return False
            if tuple(conv.padding) != pad:
                return False
            if tuple(conv.stride) != (1, 1) or tuple(conv.dilation) != (1, 1):
                return False
        return True

    def _cib_invalidate_after_load(self, module, incompatible_keys) -> None:  # noqa: ARG002
        self._cib_folded = None

    def _apply(self, *args, **kwargs):
        # Covers ``.to()``, ``.half()``, ``.cuda()`` and anything else that rewrites the
        # parameters in place through the standard route.
        result = super()._apply(*args, **kwargs)
        self._cib_folded = None
        return result

    def train(self, mode: bool = True):
        # The fold is only valid in eval, and a training forward would also update the
        # running statistics it was derived from.
        self._cib_folded = None
        return super().train(mode)

    def clear_folded_cache(self) -> None:
        """Discard the cached fold, so the next eligible call re-derives it.

        The cache is invalidated automatically by ``load_state_dict``, ``_apply`` and a
        train/eval transition. A direct in-place parameter edit -- ``p.data.mul_(1.5)``
        between two forwards -- is not one of those routes and is not detected: a
        per-call signature over the source tensors was measured at 4.17 us, a sixth of
        the entire latency budget, and still missed that exact case because ``.data``
        hands out a tensor with its own version counter. This method is the supported
        way to say so explicitly.
        """
        self._cib_folded = None

    def tier_counts(self) -> dict[str, int]:
        """How many forwards took each path, so a silent fallback is visible.

        ``FAST_PATH_AVAILABLE`` says the extension built; this says the fused path
        actually ran. A performance claim needs both.
        """
        t = self._cib_tiers
        return {
            "fused3": t[TIER_FUSED3],
            "fused5": t[TIER_FUSED5],
            # Either custom-CUDA path; what most callers mean by "the fast path ran".
            "fused": t[TIER_FUSED3] + t[TIER_FUSED5],
            "folded_eager": t[TIER_FOLDED],
            "reference": t[TIER_REFERENCE],
        }

    # -- the fold ---------------------------------------------------------------

    def _source_state_supported(self) -> bool:
        """Whether the module's own tensors are in a state the fold can represent.

        The fold rounds to fp16 and the kernels are fp16-only, so a module whose
        parameters have been converted to fp32 or bf16 -- a perfectly ordinary
        ``.float()`` -- must not be served by the fast path: it would silently round the
        weights back down and answer a call the reference rejects outright. Likewise a
        module left on the CPU has nothing to hand a CUDA kernel. Checked once, when the
        cache is derived, not per call.
        """
        try:
            params = [p for p in self.parameters() if p.is_floating_point()]
            buffers = [
                b for b in self.buffers() if b is not None and b.is_floating_point()
            ]
        except Exception:  # noqa: BLE001 - a hand-mangled tree is simply unsupported
            return False
        if not params:
            return False
        return (
            all(p.dtype is torch.float16 for p in params)
            and all(p.is_cuda for p in params)
            and all(b.is_cuda for b in buffers)
        )

    def _derive_folded(self):
        """Fold every norm into its convolution and merge the RepVGG branches.

        Returns ``(packed_weight, packed_bias, stages, device)`` or ``None`` when the
        module's tensors are not in a state the fold can represent. The five stages are
        computed in fp32 and rounded to fp16 exactly once; ``packed_weight`` and
        ``packed_bias`` are then fp16 copies laid end to end in forward order, which is
        the layout ``cib_forward`` reads back. ``stages`` keeps the same tensors in
        per-stage form for the eager path, and ``device`` is recorded so a later call on
        another GPU is rejected rather than handed foreign pointers.
        """
        if not self._source_state_supported():
            return None
        cv = self.cv1
        rep = cv[2]
        with torch.no_grad():
            w0, b0 = _fold_conv_bn(cv[0].conv, cv[0].bn)
            w1, b1 = _fold_conv_bn(cv[1].conv, cv[1].bn)
            # Both RepVGG branches are depthwise and activation-free, so after folding
            # the 3x3 support zero-pads into the 7x7 and the two convolutions add.
            w7, b7 = _fold_conv_bn(rep.conv.conv, rep.conv.bn)
            w3, b3 = _fold_conv_bn(rep.conv1.conv, rep.conv1.bn)
            w2, b2 = w7 + F.pad(w3, [2, 2, 2, 2]), b7 + b3
            w3s, b3s = _fold_conv_bn(cv[3].conv, cv[3].bn)
            w4, b4 = _fold_conv_bn(cv[4].conv, cv[4].bn)

            groups_pad = (
                (self._cib_c1, 1),
                (1, 0),
                (self._cib_mid, 3),
                (1, 0),
                (self._cib_c2, 1),
            )
            stages = [
                (w.half(), b.half(), g, p)
                for (w, b), (g, p) in zip(
                    ((w0, b0), (w1, b1), (w2, b2), (w3s, b3s), (w4, b4)), groups_pad
                )
            ]
            packed_weight = torch.cat([w.reshape(-1) for w, _, _, _ in stages])
            packed_bias = torch.cat([b for _, b, _, _ in stages])
        return packed_weight, packed_bias, stages, packed_weight.device

    def _folded_eager(self, x: torch.Tensor, stages: list) -> torch.Tensor:
        """The folded chain in eager ATen: the same arithmetic, eleven dispatches.

        Kept as the middle tier so a machine whose nvcc hiccups still gets the
        reparameterization's roughly 2x rather than the reference's 1x.
        """
        y = x
        for weight, bias, groups, pad in stages:
            y = F.silu(F.conv2d(y, weight, bias, padding=pad, groups=groups))
        return x + y if self.add else y

    # -- forward ----------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A flat conjunction, inlined rather than delegated: ten similar terms measured
        # 0.79 us against a budget of roughly 30, and a method call would add to that.
        # (These are thirteen terms, three of them ``stride`` calls, so read that figure
        # as the order of magnitude rather than as this conjunction's cost.) Each term is
        # here because the reference reads its attributes and parameters on every call and
        # dispatches through ATen, whereas the fused path reaches memory by pointer and
        # records no autograd node.
        if (
            self._cib_structural
            and x.is_cuda
            and x.dim() == 4
            and x.dtype is torch.float16
            and x.stride(3) == 1
            and x.stride(2) == x.shape[3]
            and x.stride(1) == x.shape[2] * x.shape[3]
            and x.shape[1] == self._cib_c1
            and x.shape[0] >= 1
            and x.shape[2] >= 1
            and x.shape[3] >= 1
            and x.shape[2] * x.shape[3] <= _MAX_PLANE_ELEMS
            # ``neg`` and ``conj`` are lazy flags that ATen applies when it *reads* a
            # tensor, so the values PyTorch sees are not the values in storage. Reaching
            # memory by pointer would silently use the unnegated bytes: measurably wrong
            # rather than approximately wrong.
            and not x.is_neg()
            and not x.is_conj()
            and not self.training
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cuda")
        ):
            folded = self._cib_folded
            if folded is None:
                folded = self._cib_folded = self._derive_folded()
            # ``_derive_folded`` returns None when the module's own tensors are not in a
            # state the fold can represent -- fp32 parameters after ``.float()``, or a
            # CPU tree. And a fold cached on one GPU must not be handed to a kernel
            # launched for a tensor on another, which is a device compare, not a fold.
            if folded is not None and x.device == folded[3]:
                tiers = self._cib_tiers
                if _extension is not None:
                    if PREFER_THREE_LAUNCH:
                        hw = (x.shape[2], x.shape[3])
                        covered = self._cib_geom3.get(hw)
                        if covered is None:
                            covered = self._cib_geom3[hw] = bool(
                                _extension.cib_geometry_supported3(
                                    self._cib_c1, self._cib_mid, self._cib_c2, hw[0], hw[1]
                                )
                            )
                        if covered:
                            tiers[TIER_FUSED3] += 1
                            return _extension.cib_forward3(
                                x, folded[0], folded[1], self._cib_mid, self._cib_c2,
                                self.add, USE_PDL,
                            )
                    tiers[TIER_FUSED5] += 1
                    return _extension.cib_forward(
                        x, folded[0], folded[1], self._cib_mid, self._cib_c2, self.add
                    )
                tiers[TIER_FOLDED] += 1
                return self._folded_eager(x, folded[2])

        self._cib_tiers[TIER_REFERENCE] += 1
        y = self.cv1(x)
        return x + y if self.add else y
