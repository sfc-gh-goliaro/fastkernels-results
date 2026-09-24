"""Attention with pair bias for AlphaFold3, fused to one dispatch per forward.

Same contract as ``baseline.py``: the two classes, their ``__init__``/``forward``
signatures and their submodule trees are identical, so weights still load by name
and the frozen L1 winners still back the fallback.

Why one dispatch
----------------
At the captured shapes this operator does almost no arithmetic -- 12.5 / 62.3 /
153.9 MMAC for the three configurations -- and spends its time issuing work. One
baseline forward makes 27 / 38 / 120 kernel-launching ATen calls, and a profile of
the resulting timeline (``tools/timeline.py``) shows 533 / 668 / 2420 us of *gaps*
against only 88 / 135 / 415 us of kernel time: the host cannot keep the device fed.
Measured in this workspace with the harness's own timing recipe
(``tools/skeleton_bench.py``):

* the scored window costs 13.3 / 19.4 / 17.4 us before any work at all -- that is
  the shifting pool's input copies plus one dispatch, and it is the floor;
* one extra back-to-back kernel on the same stream costs 2.05 us, flat from one to
  six launches;
* an ``at::empty`` costs 0.03-0.06 us, whether it is the output or a workspace, so
  caching a workspace across calls saves nothing (measured at -0.13 to +0.03 us)
  and is deliberately not done;
* the baseline modules take 346 / 502 / 1482 us.

So the design objective is one Python-level dispatch and as few kernels as the
dependency chain allows, and arithmetic efficiency is a second-order concern:
plain fp32 FFMA loops with the activation tile staged in shared memory are enough.

The reformulation
-----------------
``AttentionPairBias`` is per-token already. ``CrossAttentionPairBias`` is not: the
baseline materialises windowed ``[N_blocks, n_key, C]`` key blocks and normalises
them, recomputing ``K``/``V`` four times over. Every per-side transform is
row-wise, so it collapses to a per-atom form -- ``ahat`` computed once and shared
by the query and key sides (``AdaLN.layer_norm_a`` has neither scale nor offset),
``K``/``V`` computed once per atom and gathered. Verified as an identity, not an
approximation: an fp32 evaluation of exactly these formulas reproduces the
baseline modules to 1.0e-5 (A), 6.1e-6 (B) and 3.7e-7 (C), which is the same
distance the baseline itself sits from an fp64 evaluation.

The collapse needs the invalid-slot semantics stated explicitly. The baseline
zeroes a gathered key row whose index is out of range *before* normalising it, and
for ``use_ada_layer_norm=True`` that propagates to ``K = V = 0`` exactly. Leaning
on the ``-inf`` bias instead is not equivalent: a query row with no valid slot
softmaxes to uniform and then averages whatever rows were gathered. So the fused
kernel gates the gathered ``K``/``V`` by the validity indicator, and
``use_ada_layer_norm=False`` is refused for this class, because there a zeroed row
normalises to the offset vector rather than to zero.

The gather indices round through bfloat16
-----------------------------------------
``_get_block_key_indices`` derives its window from ``n_real = atom_mask.sum(-1)``,
which is bf16 because the mask is, and every later operation mixes it with an
int32 ``initial`` -- so PyTorch promotes to bf16, *including the cast of the int32
operand*. bf16 carries 8 significant bits, so integers >= 256 are representable
only when even and an odd one lands on the nearest multiple of 4 (ties-to-even):
``bf16(367) = 368``, ``bf16(399) = 400``, ``bf16(431) = 432``, ``bf16(257) = 256``,
``bf16(259) = 260``.

The windows are therefore *not* contiguous ranges. For the benched 368-atom mask
the last slot of blocks 9-11 is out of range, blocks 10 and 11 read an identical
key set, and above index 256 odd slots collapse onto multiples of 4. Using the
intended ``start + j`` window would change roughly half the keys of the upper
blocks. The kernel emulates the rounding with ``__float2bfloat16_rn`` round trips
rather than a bit trick, so it stays correct for other ``n_real``, and
``tools/index_oracle.py`` asserts the device tables bit-equal to the helper.
Because the rule is a consequence of the mask *dtype*, a mask that is not bf16 is
refused. The reduction that produces ``n_real`` is exact for any 0/1 mask
regardless of summation order -- integers are exactly representable in fp32 up to
2^24 and the admitted atom count is bounded at 2^20 -- which is what makes a
device-side reduction safe here; the table is computed on the device and never
read back to the host.

That exactness argument is the reason a mask is contractually 0/1 here. A mask
carrying fractional values would still gather correctly, but its ``n_real`` is a
sum of inexact addends, so this kernel's reduction order and ATen's could round to
different bf16 values and select a different gather table. Detecting that would
need a device reduction read back to the host, which is precisely what this
operator must not do, so it is stated as a limit rather than checked: with a
non-binary mask the fused path is not guaranteed to match the baseline. Every
input the benchmark and the reference presents is 0/1.

Rounding-point policy
---------------------
Every reduction accumulates in fp32, and the result is rounded to bf16 at exactly
the places the baseline rounds. That is a fidelity choice, not an accuracy one,
and it is deliberate: a single bf16 ULP of difference in a pre-softmax logit
(~0.016 at these magnitudes) becomes a ~1.6 % difference in a probability and
~3 ULP of the output peak, which would breach the agreement bound the checker
holds the candidate to. Concretely, and each verified against ATen:

* LayerNorm: two-pass fp32 mean/variance, scale and offset applied in fp32, one
  rounding on the store -- matching ``F.layer_norm(x.float(), ...).to(bf16)``.
* every linear: fp32 accumulation, bias added in fp32, one rounding. ``F.linear``
  on bf16 does exactly this (it lowers to ``addmm``; a two-rounding
  ``bf16(x@W.T) + b`` differs on a quarter of all elements).
* ``q / sqrt(c_hidden)``: an fp32 divide of the bf16 value, rounded to bf16 --
  the baseline's elementwise division, division not reciprocal-multiply.
* scores: fp32 dot rounded to bf16, then the mask bias and the pair bias added
  with a rounding after each, matching the baseline's chain of bf16 adds.
* softmax: fp32 max and fp32 sum of exponentials, each probability rounded to
  bf16 -- which is bit-exactly what ATen's bf16 softmax produces.
* ``P V``: fp32 accumulation rounded to bf16; gates are fp32 sigmoids rounded to
  bf16 and applied as a bf16 product.

The mask bias keeps the literal ``inf * (mask - 1)`` form in fp32, so a genuinely
masked query row degenerates to a uniform softmax exactly as the baseline's does
rather than producing NaN.

What the fused path refuses
---------------------------
An explicit conjunction of *sufficient* conditions gates it, evaluated in the
operator itself; a refusal is signalled back as a zero-element tensor and the
Python ``forward`` then runs the baseline formula through the same submodules, so
its answer is bit-identical to ``baseline.py``'s. Refused: any dtype other than
bf16, a real batch dimension, non-contiguous or tensor-subclass inputs, gradients
enabled, active autocast, ``use_ada_layer_norm=False`` on
``CrossAttentionPairBias``, a ``LayerNorm`` that is not ``promote_fp32`` or whose
``eps`` disagrees with its siblings, shapes past the bounds the kernels are
written for, and a build that did not happen at all.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN
from .alphafold3_of3_attention import OF3Attention

# Unique to this file so a second import under a different module name cannot
# double-register the operators.
_LIBRARY_NAME = "fk_af3_apb_cand"


def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

using bf16_t = __nv_bfloat16;

// Rows of the activation tile a warp dots against one weight row, and the K
// chunk staged in shared memory. 16 x 256 bf16 is 8 KB, which leaves room for
// everything else at any admitted shape.
constexpr int kTileRows = 16;
constexpr int kChunk = 256;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kSmemLimit = 48 * 1024;

// Query rows one attention CTA walks before it gives up its gathered keys. The
// rows of a block share an index table, so this is pure reuse.
constexpr int kQueryRows = 8;

__device__ __forceinline__ float ld(const bf16_t* p) { return __bfloat162float(*p); }
__device__ __forceinline__ bf16_t st(float x) { return __float2bfloat16_rn(x); }

// One bf16 rounding step with the arithmetic left in fp32 -- the primitive the
// whole rounding-point policy is expressed in.
__device__ __forceinline__ float rbf(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// Block-wide sum. Every thread of the CTA must call it, and every thread gets
// the answer.
// ---------------------------------------------------------------------------
__device__ float block_sum(float v, float* red) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nwarps = static_cast<int>(blockDim.x) >> 5;
  if (lane == 0) red[wid] = v;
  __syncthreads();
  float t = (static_cast<int>(threadIdx.x) < nwarps) ? red[threadIdx.x] : 0.0f;
#pragma unroll
  for (int off = 16; off; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
  if (threadIdx.x == 0) red[nwarps] = t;
  __syncthreads();
  const float total = red[nwarps];
  __syncthreads();
  return total;
}

// ---------------------------------------------------------------------------
// Row LayerNorm statistics, fp32 two-pass. ``src == nullptr`` means the row is
// one of the zero-padded ones, whose normalisation is zero.
// ---------------------------------------------------------------------------
__device__ void ln_row(const bf16_t* __restrict__ src, int n, float eps,
                       float* row, float* red, float* mean_out, float* rstd_out) {
  float s = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = src ? ld(src + i) : 0.0f;
    row[i] = v;
    s += v;
  }
  const float mean = block_sum(s, red) / static_cast<float>(n);
  float v2 = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float d = row[i] - mean;
    v2 += d * d;
  }
  const float var = block_sum(v2, red) / static_cast<float>(n);
  *mean_out = mean;
  *rstd_out = rsqrtf(var + eps);
}

// ---------------------------------------------------------------------------
// acc[m] = sum_k A[(m0 + m) * lda + k] * W[wrow * ldw + k], for m in
// [0, kTileRows). Rows past ``M`` contribute zero. Cooperative: every thread of
// the CTA must call this, including the warps whose ``wrow`` is out of range,
// because the staging loads and the barriers are shared.
// ---------------------------------------------------------------------------
__device__ void cta_col_dots(const bf16_t* __restrict__ A, int64_t lda, int m0,
                             int M, int K, const bf16_t* __restrict__ W,
                             int64_t ldw, int wrow, bool active, bf16_t* Asm,
                             float* acc) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 0; m < kTileRows; ++m) acc[m] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kChunk) {
    const int kn = min(kChunk, K - k0);
    __syncthreads();
    // A flat strided loop with a *compile-time* divisor. Two other shapes of this
    // loop were written and measured, and both are worse:
    //
    //  * bounding it by ``kTileRows * kn`` to skip the padding replaces the shift
    //    with an integer division by a runtime value on every one of a thread's
    //    sixteen staging steps -- 25-45 % slower on every projection, more than
    //    the padding stores it saves;
    //  * hoisting the row to an outer loop removes the division entirely but
    //    leaves one dependent global load per iteration instead of sixteen
    //    independent ones, and is slower again.
    //
    // These loops are short of memory-level parallelism, not of instructions.
    for (int idx = threadIdx.x; idx < kTileRows * kChunk; idx += blockDim.x) {
      const int m = idx / kChunk;
      const int k = idx - m * kChunk;
      const int gm = m0 + m;
      Asm[idx] = (k < kn && gm < M)
                     ? A[static_cast<int64_t>(gm) * lda + k0 + k]
                     : st(0.0f);
    }
    __syncthreads();
    if (active) {
      // Scalar, deliberately, after three measured alternatives.
      //
      // NCU (profile/p1-fused-candidate) says this loop is bound on shared-memory
      // throughput -- DRAM at 0.14-0.73 % of peak against L1TEX at 29-47 % -- and
      // that a scalar bf16 read has adjacent lanes sharing a 4-byte bank. Both
      // obvious remedies were tried on a pinned GPU:
      //
      //  * a ``uint4`` form pulling eight weights per lane: 30-45 % slower,
      //    because at a 256-wide chunk it leaves each lane with a single
      //    outstanding load where the scalar form has eight, and widening the
      //    chunk to restore that costs 4x the shared memory;
      //  * a ``__nv_bfloat162`` form pulling two: it does help where the reduction
      //    is narrow (case C's 128-wide conditioning kernel, 33.5 -> 30.4 us) but
      //    costs more than that back on the wide projections (case B's, 17.3 ->
      //    19.9 us), for no net gain across the four benched cases.
      //
      // So the shared-memory bound is real but not reachable by widening the
      // access; converting it needs the arithmetic itself to move off scalar FFMA,
      // which is the phase-2 tensor-core rung. (Any such A/B on this node has to
      // pin the GPU: two of its four B200s are clocked at 1155 MHz against 1965,
      // which silently inverts a comparison that changes device.)
      const bf16_t* wp = W + static_cast<int64_t>(wrow) * ldw + k0;
      for (int k = lane; k < kn; k += 32) {
        const float w = ld(wp + k);
#pragma unroll
        for (int m = 0; m < kTileRows; ++m)
          acc[m] = fmaf(ld(Asm + m * kChunk + k), w, acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < kTileRows; ++m) {
    float v = acc[m];
#pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    acc[m] = v;
  }
}

// ###########################################################################
// AttentionPairBias
// ###########################################################################
struct ApbParams {
  const bf16_t* a;
  const bf16_t* z;
  const bf16_t* s;
  const bf16_t* mask;
  const bf16_t* ln_a_w;
  const bf16_t* ln_a_b;
  const bf16_t* ada_s_norm_w;
  const bf16_t* ada_g_w;
  const bf16_t* ada_g_b;
  const bf16_t* ada_s_w;
  const bf16_t* ln_z_w;
  const bf16_t* ln_z_b;
  const bf16_t* w_z;
  const bf16_t* w_q;
  const bf16_t* b_q;
  const bf16_t* w_k;
  const bf16_t* w_v;
  const bf16_t* w_g;
  const bf16_t* w_o;
  const bf16_t* w_ada_out;
  const bf16_t* b_ada_out;
  bf16_t* zb;
  bf16_t* ax;
  bf16_t* ahat;
  bf16_t* sx;
  bf16_t* gout;
  bf16_t* qkvg;
  bf16_t* o;
  bf16_t* out;
  int N, C, cs, cz, H, D, HD;
  int row_smem;
  int use_ada;
  float inf, eps, sqrt_d;
};

// Per-row normalisations, plus the pair bias. One CTA per z row, per a row and
// (for the conditioned variant) per s row.
__global__ void apb_norms(ApbParams p) {
  extern __shared__ float sm[];
  float* row = sm;
  float* red = sm + p.row_smem;
  const int bid = static_cast<int>(blockIdx.x);
  const int npair = p.N * p.N;

  if (bid < npair) {
    const int qi = bid / p.N;
    const int ki = bid - qi * p.N;
    float mean, rstd;
    ln_row(p.z + static_cast<int64_t>(bid) * p.cz, p.cz, p.eps, row, red, &mean,
           &rstd);
    for (int i = threadIdx.x; i < p.cz; i += blockDim.x) {
      float v = (row[i] - mean) * rstd * ld(p.ln_z_w + i);
      if (p.ln_z_b) v += ld(p.ln_z_b + i);
      row[i] = rbf(v);
    }
    __syncthreads();
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int h = wid; h < p.H; h += kWarps) {
      float acc = 0.0f;
      for (int i = lane; i < p.cz; i += 32)
        acc = fmaf(row[i], ld(p.w_z + static_cast<int64_t>(h) * p.cz + i), acc);
#pragma unroll
      for (int off = 16; off; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
      if (lane == 0)
        p.zb[(static_cast<int64_t>(h) * p.N + qi) * p.N + ki] = st(acc);
    }
    return;
  }

  if (bid < npair + p.N) {
    const int i = bid - npair;
    float mean, rstd;
    ln_row(p.a + static_cast<int64_t>(i) * p.C, p.C, p.eps, row, red, &mean,
           &rstd);
    if (p.use_ada) {
      for (int j = threadIdx.x; j < p.C; j += blockDim.x)
        p.ahat[static_cast<int64_t>(i) * p.C + j] = st((row[j] - mean) * rstd);
    } else {
      for (int j = threadIdx.x; j < p.C; j += blockDim.x) {
        float v = (row[j] - mean) * rstd * ld(p.ln_a_w + j);
        if (p.ln_a_b) v += ld(p.ln_a_b + j);
        p.ax[static_cast<int64_t>(i) * p.C + j] = st(v);
      }
    }
    return;
  }

  const int i = bid - npair - p.N;
  float mean, rstd;
  ln_row(p.s + static_cast<int64_t>(i) * p.cs, p.cs, p.eps, row, red, &mean,
         &rstd);
  for (int j = threadIdx.x; j < p.cs; j += blockDim.x)
    p.sx[static_cast<int64_t>(i) * p.cs + j] =
        st((row[j] - mean) * rstd * ld(p.ada_s_norm_w + j));
}

// The conditioned variant's two column-wise products: the AdaLN gate applied to
// ``ahat``, and the output gate read off the raw ``s``.
__global__ void apb_conditioning(ApbParams p, int ntile) {
  extern __shared__ char smraw[];
  bf16_t* Asm = reinterpret_cast<bf16_t*>(smraw);
  const int m0 = static_cast<int>(blockIdx.y) * kTileRows;
  const bool out_gate = static_cast<int>(blockIdx.x) >= ntile;
  const int tile = out_gate ? static_cast<int>(blockIdx.x) - ntile
                            : static_cast<int>(blockIdx.x);
  const int col = tile * kWarps + static_cast<int>(threadIdx.x >> 5);
  const bool active = col < p.C;
  const bool lead = (threadIdx.x & 31) == 0;
  float acc_g[kTileRows];

  if (!out_gate) {
    float acc_s[kTileRows];
    cta_col_dots(p.sx, p.cs, m0, p.N, p.cs, p.ada_g_w, p.cs, col, active, Asm,
                 acc_g);
    cta_col_dots(p.sx, p.cs, m0, p.N, p.cs, p.ada_s_w, p.cs, col, active, Asm,
                 acc_s);
    if (active && lead) {
#pragma unroll
      for (int m = 0; m < kTileRows; ++m) {
        const int i = m0 + m;
        if (i >= p.N) continue;
        const float gate =
            rbf(sigmoidf(rbf(acc_g[m] + ld(p.ada_g_b + col))));
        const float sum = rbf(ld(p.ahat + static_cast<int64_t>(i) * p.C + col) +
                              rbf(acc_s[m]));
        p.ax[static_cast<int64_t>(i) * p.C + col] = st(gate * sum);
      }
    }
    return;
  }

  cta_col_dots(p.s, p.cs, m0, p.N, p.cs, p.w_ada_out, p.cs, col, active, Asm,
               acc_g);
  if (active && lead) {
#pragma unroll
    for (int m = 0; m < kTileRows; ++m) {
      const int i = m0 + m;
      if (i >= p.N) continue;
      p.gout[static_cast<int64_t>(i) * p.C + col] =
          st(sigmoidf(rbf(acc_g[m] + ld(p.b_ada_out + col))));
    }
  }
}

// Q, K, V and the query-side gate. The grid is blocked *per projection* rather
// than over one flat 4*HD column range, so that every warp of a CTA shares the
// same activation matrix -- which it must, because the staging into shared memory
// is cooperative.
__global__ void apb_project(ApbParams p, int per) {
  extern __shared__ char smraw[];
  bf16_t* Asm = reinterpret_cast<bf16_t*>(smraw);
  const int m0 = static_cast<int>(blockIdx.y) * kTileRows;
  const int which = static_cast<int>(blockIdx.x) / per;
  const int tile = static_cast<int>(blockIdx.x) - which * per;
  const int n = tile * kWarps + static_cast<int>(threadIdx.x >> 5);
  const bf16_t* W = nullptr;
  const bf16_t* bias = nullptr;
  switch (which) {
    case 0: W = p.w_q; bias = p.b_q; break;
    case 1: W = p.w_k; break;
    case 2: W = p.w_v; break;
    default: W = p.w_g; break;
  }
  const bool active = n < p.HD && W != nullptr;
  float acc[kTileRows];
  cta_col_dots(p.ax, p.C, m0, p.N, p.C, active ? W : p.w_q, p.C, active ? n : 0,
               active, Asm, acc);
  if (active && (threadIdx.x & 31) == 0) {
    bf16_t* dst = p.qkvg + static_cast<int64_t>(which) * p.N * p.HD;
#pragma unroll
    for (int m = 0; m < kTileRows; ++m) {
      const int i = m0 + m;
      if (i >= p.N) continue;
      float v = acc[m];
      if (bias) v += ld(bias + n);
      dst[static_cast<int64_t>(i) * p.HD + n] = st(v);
    }
  }
}

// One CTA per head: scores, softmax, P V, and the query-side gate.
//
// K and V are staged in shared memory rather than read from global inside the
// innermost loops. That is the whole difference between 12 us and about 1 us here:
// the grid is only ``no_heads`` CTAs, so there are nowhere near enough warps
// resident to hide an L2 round trip per multiply-accumulate. K is staged
// transposed so that lanes walking adjacent keys read adjacent shared-memory
// addresses instead of colliding on one bank.
__global__ void apb_attend(ApbParams p) {
  extern __shared__ char smraw[];
  const int h = static_cast<int>(blockIdx.x);
  float* qs = reinterpret_cast<float*>(smraw);   // [N][D]
  float* sc = qs + p.N * p.D;                    // [N][N]
  float* mb = sc + p.N * p.N;                    // [N]
  bf16_t* ksm = reinterpret_cast<bf16_t*>(mb + p.N);   // [D][N], transposed
  bf16_t* vsm = ksm + p.N * p.D;                       // [N][D]

  const int64_t plane = static_cast<int64_t>(p.N) * p.HD;
  const bf16_t* Q = p.qkvg;
  const bf16_t* K = p.qkvg + plane;
  const bf16_t* V = p.qkvg + 2 * plane;
  const bf16_t* G = p.qkvg + 3 * plane;
  const int64_t hoff = static_cast<int64_t>(h) * p.D;

  for (int t = threadIdx.x; t < p.N * p.D; t += blockDim.x) {
    const int i = t / p.D;
    const int d = t - i * p.D;
    const int64_t row = static_cast<int64_t>(i) * p.HD + hoff;
    qs[t] = rbf(ld(Q + row + d) / p.sqrt_d);
    ksm[d * p.N + i] = K[row + d];
    vsm[t] = V[row + d];
  }
  for (int t = threadIdx.x; t < p.N; t += blockDim.x) {
    const float m = p.mask ? ld(p.mask + t) : 1.0f;
    mb[t] = rbf(p.inf * rbf(m - 1.0f));
  }
  __syncthreads();

  for (int t = threadIdx.x; t < p.N * p.N; t += blockDim.x) {
    const int qi = t / p.N;
    const int ki = t - qi * p.N;
    float acc = 0.0f;
    for (int d = 0; d < p.D; ++d)
      acc = fmaf(qs[qi * p.D + d], ld(ksm + d * p.N + ki), acc);
    float v = rbf(acc);
    v = rbf(v + mb[ki]);
    v = rbf(v + ld(p.zb + (static_cast<int64_t>(h) * p.N + qi) * p.N + ki));
    sc[t] = v;
  }
  __syncthreads();

  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int qi = wid; qi < p.N; qi += kWarps) {
    float* rowsc = sc + static_cast<int64_t>(qi) * p.N;
    float mx = -INFINITY;
    for (int k = lane; k < p.N; k += 32) mx = fmaxf(mx, rowsc[k]);
#pragma unroll
    for (int off = 16; off; off >>= 1)
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    float sum = 0.0f;
    for (int k = lane; k < p.N; k += 32) sum += expf(rowsc[k] - mx);
#pragma unroll
    for (int off = 16; off; off >>= 1)
      sum += __shfl_xor_sync(0xffffffffu, sum, off);
    for (int k = lane; k < p.N; k += 32) rowsc[k] = rbf(expf(rowsc[k] - mx) / sum);
  }
  __syncthreads();

  for (int t = threadIdx.x; t < p.N * p.D; t += blockDim.x) {
    const int qi = t / p.D;
    const int d = t - qi * p.D;
    const float* rowsc = sc + static_cast<int64_t>(qi) * p.N;
    float acc = 0.0f;
    for (int k = 0; k < p.N; ++k)
      acc = fmaf(rowsc[k], ld(vsm + static_cast<int64_t>(k) * p.D + d), acc);
    float v = rbf(acc);
    if (p.w_g)
      v *= rbf(sigmoidf(ld(G + static_cast<int64_t>(qi) * p.HD + hoff + d)));
    p.o[static_cast<int64_t>(qi) * p.HD + hoff + d] = st(v);
  }
}

__global__ void apb_output(ApbParams p) {
  extern __shared__ char smraw[];
  bf16_t* Asm = reinterpret_cast<bf16_t*>(smraw);
  const int m0 = static_cast<int>(blockIdx.y) * kTileRows;
  const int col = static_cast<int>(blockIdx.x) * kWarps +
                  static_cast<int>(threadIdx.x >> 5);
  const bool active = col < p.C;
  float acc[kTileRows];
  cta_col_dots(p.o, p.HD, m0, p.N, p.HD, p.w_o, p.HD, active ? col : 0, active,
               Asm, acc);
  if (active && (threadIdx.x & 31) == 0) {
#pragma unroll
    for (int m = 0; m < kTileRows; ++m) {
      const int i = m0 + m;
      if (i >= p.N) continue;
      float v = rbf(acc[m]);
      if (p.use_ada) v *= ld(p.gout + static_cast<int64_t>(i) * p.C + col);
      p.out[static_cast<int64_t>(i) * p.C + col] = st(v);
    }
  }
}

// ###########################################################################
// CrossAttentionPairBias
// ###########################################################################
struct CapbParams {
  const bf16_t* a;
  const bf16_t* z;
  const bf16_t* s;
  const bf16_t* mask;
  const bf16_t* q_norm_w;
  const bf16_t* q_g_w;
  const bf16_t* q_g_b;
  const bf16_t* q_s_w;
  const bf16_t* k_norm_w;
  const bf16_t* k_g_w;
  const bf16_t* k_g_b;
  const bf16_t* k_s_w;
  const bf16_t* w_z;
  const bf16_t* w_q;
  const bf16_t* b_q;
  const bf16_t* w_k;
  const bf16_t* w_v;
  const bf16_t* w_g;
  const bf16_t* w_o;
  const bf16_t* w_ada_out;
  const bf16_t* b_ada_out;
  int32_t* idx;
  float* kvalid;   // 1 where the gather index is in range
  float* kmask;    // that, times the padded atom mask at the gathered index
  bf16_t* ahat;
  bf16_t* sxq;
  bf16_t* sxk;
  bf16_t* aq;
  bf16_t* ak;
  bf16_t* gout;
  bf16_t* qg;
  bf16_t* kv;
  bf16_t* o;
  bf16_t* out;
  int n_atom, apad, nb, nq, nk, C, cs, cz, H, D, HD;
  int row_smem;
  float inf, eps, sqrt_d;
};

// The bf16-rounded gather table, one CTA. Written once and shared verbatim
// between the fused path and the oracle operator below, so the test that asserts
// it bit-equal to ``_get_block_key_indices`` cannot drift from the code it tests.
//
// Every step reproduces one operation of the helper, including which operand gets
// cast: ``n_real`` is a bf16 sum of the padded mask; ``nm1`` rounds again (367 ->
// 368); ``over`` rounds the int32 ``initial[..., -1]`` on the way into the
// subtraction (431 -> 432, so the shift is 64 and not 63); ``under`` is exact in
// int32 but is rounded when ``torch.where`` promotes it against ``-over``; and
// ``final`` rounds the int32 index before and after the shift is added.
__device__ void emit_key_table(const bf16_t* __restrict__ mask, int n_atom,
                               int apad, int nb, int nq, int nk, float* red,
                               int32_t* idx, float* kvalid, float* kmask) {
  // Exact for any 0/1 mask whatever the summation order, which is what lets this
  // be a device reduction rather than a host readback.
  float part = 0.0f;
  for (int i = threadIdx.x; i < apad; i += blockDim.x)
    part += (i < n_atom) ? ld(mask + i) : 0.0f;
  const float n_real = rbf(block_sum(part, red));
  const float nm1 = rbf(n_real - 1.0f);
  const float hi = fmaxf(nm1, 0.0f);
  const int lo = -((nk + 1) / 2);
  const int total = nb * nk;
  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    const int b = t / nk;
    const int j = t - b * nk;
    const int init_first = nq / 2 + b * nq + lo;
    const int init_last = init_first + nk - 1;
    const int under = max(-init_first, 0);
    const float over =
        fmaxf(rbf(rbf(static_cast<float>(init_last)) - nm1), 0.0f);
    const float shift = (under > 0) ? rbf(static_cast<float>(under)) : -over;
    const float fin = rbf(rbf(static_cast<float>(init_first + j)) + shift);
    const bool invalid = (fin < 0.0f) || (fin >= n_real);
    const int safe = static_cast<int>(fminf(fmaxf(fin, 0.0f), hi));
    idx[t] = safe;
    kvalid[t] = invalid ? 0.0f : 1.0f;
    kmask[t] = (invalid || safe >= n_atom) ? 0.0f : ld(mask + safe);
  }
}

__global__ void capb_key_table(const bf16_t* __restrict__ mask, int n_atom,
                               int apad, int nb, int nq, int nk, int32_t* idx,
                               float* kvalid, float* kmask) {
  extern __shared__ float red[];
  emit_key_table(mask, n_atom, apad, nb, nq, nk, red, idx, kvalid, kmask);
}

// The gather table, and the per-atom normalisations the two sides share.
__global__ void capb_prepare(CapbParams p) {
  extern __shared__ float sm[];
  float* row = sm;
  float* red = sm + p.row_smem;
  const int bid = static_cast<int>(blockIdx.x);

  if (bid == 0) {
    emit_key_table(p.mask, p.n_atom, p.apad, p.nb, p.nq, p.nk, red, p.idx,
                   p.kvalid, p.kmask);
    return;
  }

  const int i = bid - 1;
  const bool real = i < p.n_atom;
  float mean, rstd;
  ln_row(real ? p.a + static_cast<int64_t>(i) * p.C : nullptr, p.C, p.eps, row,
         red, &mean, &rstd);
  for (int j = threadIdx.x; j < p.C; j += blockDim.x)
    p.ahat[static_cast<int64_t>(i) * p.C + j] = st((row[j] - mean) * rstd);
  __syncthreads();
  ln_row(real ? p.s + static_cast<int64_t>(i) * p.cs : nullptr, p.cs, p.eps, row,
         red, &mean, &rstd);
  for (int j = threadIdx.x; j < p.cs; j += blockDim.x) {
    const float v = (row[j] - mean) * rstd;
    p.sxq[static_cast<int64_t>(i) * p.cs + j] = st(v * ld(p.q_norm_w + j));
    p.sxk[static_cast<int64_t>(i) * p.cs + j] = st(v * ld(p.k_norm_w + j));
  }
}

// The two conditioned activations and the output gate, per atom.
__global__ void capb_conditioning(CapbParams p, int ntile) {
  extern __shared__ char smraw[];
  bf16_t* Asm = reinterpret_cast<bf16_t*>(smraw);
  const int m0 = static_cast<int>(blockIdx.y) * kTileRows;
  const int role = static_cast<int>(blockIdx.x) / ntile;
  const int tile = static_cast<int>(blockIdx.x) - role * ntile;
  const int col = tile * kWarps + static_cast<int>(threadIdx.x >> 5);
  const bool active = col < p.C;
  const bool lead = (threadIdx.x & 31) == 0;
  float acc_g[kTileRows];

  if (role == 2) {
    cta_col_dots(p.s, p.cs, m0, p.n_atom, p.cs, p.w_ada_out, p.cs, col, active,
                 Asm, acc_g);
    if (active && lead) {
#pragma unroll
      for (int m = 0; m < kTileRows; ++m) {
        const int i = m0 + m;
        if (i >= p.n_atom) continue;
        p.gout[static_cast<int64_t>(i) * p.C + col] =
            st(sigmoidf(rbf(acc_g[m] + ld(p.b_ada_out + col))));
      }
    }
    return;
  }

  const bf16_t* sx = role ? p.sxk : p.sxq;
  const bf16_t* wg = role ? p.k_g_w : p.q_g_w;
  const bf16_t* bg = role ? p.k_g_b : p.q_g_b;
  const bf16_t* ws = role ? p.k_s_w : p.q_s_w;
  bf16_t* dst = role ? p.ak : p.aq;
  float acc_s[kTileRows];
  cta_col_dots(sx, p.cs, m0, p.apad, p.cs, wg, p.cs, col, active, Asm, acc_g);
  cta_col_dots(sx, p.cs, m0, p.apad, p.cs, ws, p.cs, col, active, Asm, acc_s);
  if (active && lead) {
#pragma unroll
    for (int m = 0; m < kTileRows; ++m) {
      const int i = m0 + m;
      if (i >= p.apad) continue;
      const float gate = rbf(sigmoidf(rbf(acc_g[m] + ld(bg + col))));
      const float sum = rbf(ld(p.ahat + static_cast<int64_t>(i) * p.C + col) +
                            rbf(acc_s[m]));
      dst[static_cast<int64_t>(i) * p.C + col] = st(gate * sum);
    }
  }
}

// Q and the query gate from the query-side activation; K and V from the key-side
// one -- once per atom, where the baseline recomputes them once per key slot.
__global__ void capb_project(CapbParams p, int per) {
  extern __shared__ char smraw[];
  bf16_t* Asm = reinterpret_cast<bf16_t*>(smraw);
  const int m0 = static_cast<int>(blockIdx.y) * kTileRows;
  const int which = static_cast<int>(blockIdx.x) / per;
  const int tile = static_cast<int>(blockIdx.x) - which * per;
  const int n = tile * kWarps + static_cast<int>(threadIdx.x >> 5);
  // Blocked per projection so ``A`` is uniform across the CTA: Q and the gate
  // read the query-side activation, K and V the key-side one, and the staging
  // into shared memory is a whole-CTA operation.
  const bf16_t* A = which < 2 ? p.aq : p.ak;
  const bf16_t* W = nullptr;
  const bf16_t* bias = nullptr;
  bf16_t* dst = nullptr;
  switch (which) {
    case 0: W = p.w_q; bias = p.b_q; dst = p.qg; break;
    case 1: W = p.w_g; dst = p.qg + static_cast<int64_t>(p.apad) * p.HD; break;
    case 2: W = p.w_k; dst = p.kv; break;
    default: W = p.w_v; dst = p.kv + static_cast<int64_t>(p.apad) * p.HD; break;
  }
  const bool active = n < p.HD && W != nullptr;
  float acc[kTileRows];
  cta_col_dots(active ? A : p.aq, p.C, m0, p.apad, p.C, active ? W : p.w_q, p.C,
               active ? n : 0, active, Asm, acc);
  if (active && (threadIdx.x & 31) == 0) {
#pragma unroll
    for (int m = 0; m < kTileRows; ++m) {
      const int i = m0 + m;
      if (i >= p.apad) continue;
      float v = acc[m];
      if (bias) v += ld(bias + n);
      dst[static_cast<int64_t>(i) * p.HD + n] = st(v);
    }
  }
}

// Blocked attention. One CTA per (block, head, query tile).
//
// This kernel was 153 us of the 218 us this configuration spent on the device, for
// about a microsecond of arithmetic, and both causes were structural. It ran one
// CTA per (block, query tile) -- 48 CTAs on 148 SMs, eight warps each, so roughly
// an eighth of the machine with no way to overlap a memory stall. And it gathered
// K and V from global memory inside the innermost loop, one scattered ~600-cycle
// access per multiply-accumulate.
//
// Splitting the head dimension into the grid gives four times the CTAs, and the
// gathered K/V for a block are staged once into shared memory and reused by every
// query row and by both the score and the P V phase. The validity gating is applied
// during that staging -- an invalid slot contributes a literal zero row, which is
// what the baseline produces by zeroing the gathered activation before normalising
// it, and is not the same thing as relying on the -inf bias.
//
// K is staged transposed and V is not, so that in each loop the lanes walk
// consecutive shared-memory addresses: adjacent lanes hold adjacent keys when
// computing scores, and adjacent channels when accumulating P V.
//
// ``z`` is read once per head rather than once overall, which is a deliberate
// trade: 6.3 MB instead of 1.6 MB, against four times the parallelism.
__global__ void capb_attend(CapbParams p, int tq) {
  extern __shared__ char smraw[];
  const int b = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int r0 = static_cast<int>(blockIdx.z) * tq;

  float* wz = reinterpret_cast<float*>(smraw);   // [cz]
  float* kmsk = wz + p.cz;                       // [nk]
  float* qs = kmsk + p.nk;                       // [tq][D]
  float* sc = qs + tq * p.D;                     // [tq][nk]
  bf16_t* ksm = reinterpret_cast<bf16_t*>(sc + tq * p.nk);   // [D][nk]
  bf16_t* vsm = ksm + p.nk * p.D;                            // [nk][D]

  const int32_t* idx = p.idx + static_cast<int64_t>(b) * p.nk;
  const float* kvalid = p.kvalid + static_cast<int64_t>(b) * p.nk;
  const bf16_t* Q = p.qg;
  const bf16_t* G = p.qg + static_cast<int64_t>(p.apad) * p.HD;
  const bf16_t* K = p.kv;
  const bf16_t* V = p.kv + static_cast<int64_t>(p.apad) * p.HD;
  const int64_t hoff = static_cast<int64_t>(h) * p.D;

  for (int t = threadIdx.x; t < p.cz; t += blockDim.x)
    wz[t] = ld(p.w_z + static_cast<int64_t>(h) * p.cz + t);
  for (int t = threadIdx.x; t < p.nk; t += blockDim.x)
    kmsk[t] = p.kmask[static_cast<int64_t>(b) * p.nk + t];
  for (int t = threadIdx.x; t < p.nk * p.D; t += blockDim.x) {
    const int j = t / p.D;
    const int d = t - j * p.D;
    // The gate is exactly 0 or 1, so the product is exact and the store is a
    // no-op rounding of a value that is already representable.
    const float gate = kvalid[j];
    const int64_t row = static_cast<int64_t>(idx[j]) * p.HD + hoff;
    ksm[d * p.nk + j] = st(ld(K + row + d) * gate);
    vsm[t] = st(ld(V + row + d) * gate);
  }
  for (int t = threadIdx.x; t < tq * p.D; t += blockDim.x) {
    const int rl = t / p.D;
    const int d = t - rl * p.D;
    const int r = r0 + rl;
    qs[t] = (r < p.nq)
                ? rbf(ld(Q + static_cast<int64_t>(b * p.nq + r) * p.HD + hoff + d)
                      / p.sqrt_d)
                : 0.0f;
  }
  __syncthreads();

  for (int t = threadIdx.x; t < tq * p.nk; t += blockDim.x) {
    const int rl = t / p.nk;
    const int j = t - rl * p.nk;
    const int r = r0 + rl;
    if (r >= p.nq) {
      sc[t] = 0.0f;
      continue;
    }
    const int i = b * p.nq + r;
    float acc = 0.0f;
    for (int d = 0; d < p.D; ++d)
      acc = fmaf(qs[rl * p.D + d], ld(ksm + d * p.nk + j), acc);
    float v = rbf(acc);
    const float mq = (i < p.n_atom) ? ld(p.mask + i) : 0.0f;
    const float bm = rbf(mq * kmsk[j]);
    v = rbf(v + rbf(p.inf * rbf(bm - 1.0f)));
    const bf16_t* zp =
        p.z + ((static_cast<int64_t>(b) * p.nq + r) * p.nk + j) * p.cz;
    float zb = 0.0f;
    for (int c = 0; c < p.cz; ++c) zb = fmaf(ld(zp + c), wz[c], zb);
    sc[t] = rbf(v + rbf(zb));
  }
  __syncthreads();

  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int rl = wid; rl < tq; rl += kWarps) {
    if (r0 + rl >= p.nq) continue;
    float* rowsc = sc + static_cast<int64_t>(rl) * p.nk;
    float mx = -INFINITY;
    for (int j = lane; j < p.nk; j += 32) mx = fmaxf(mx, rowsc[j]);
#pragma unroll
    for (int off = 16; off; off >>= 1)
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    float sum = 0.0f;
    for (int j = lane; j < p.nk; j += 32) sum += expf(rowsc[j] - mx);
#pragma unroll
    for (int off = 16; off; off >>= 1)
      sum += __shfl_xor_sync(0xffffffffu, sum, off);
    for (int j = lane; j < p.nk; j += 32)
      rowsc[j] = rbf(expf(rowsc[j] - mx) / sum);
  }
  __syncthreads();

  for (int t = threadIdx.x; t < tq * p.D; t += blockDim.x) {
    const int rl = t / p.D;
    const int d = t - rl * p.D;
    const int r = r0 + rl;
    if (r >= p.nq) continue;
    const int i = b * p.nq + r;
    const float* rowsc = sc + static_cast<int64_t>(rl) * p.nk;
    float acc = 0.0f;
    for (int j = 0; j < p.nk; ++j)
      acc = fmaf(rowsc[j], ld(vsm + static_cast<int64_t>(j) * p.D + d), acc);
    float v = rbf(acc);
    if (p.w_g)
      v *= rbf(sigmoidf(ld(G + static_cast<int64_t>(i) * p.HD + hoff + d)));
    p.o[static_cast<int64_t>(i) * p.HD + hoff + d] = st(v);
  }
}

__global__ void capb_output(CapbParams p) {
  extern __shared__ char smraw[];
  bf16_t* Asm = reinterpret_cast<bf16_t*>(smraw);
  const int m0 = static_cast<int>(blockIdx.y) * kTileRows;
  const int col = static_cast<int>(blockIdx.x) * kWarps +
                  static_cast<int>(threadIdx.x >> 5);
  const bool active = col < p.C;
  float acc[kTileRows];
  cta_col_dots(p.o, p.HD, m0, p.n_atom, p.HD, p.w_o, p.HD, active ? col : 0,
               active, Asm, acc);
  if (active && (threadIdx.x & 31) == 0) {
#pragma unroll
    for (int m = 0; m < kTileRows; ++m) {
      const int i = m0 + m;
      if (i >= p.n_atom) continue;
      const float v = rbf(acc[m]) * ld(p.gout + static_cast<int64_t>(i) * p.C + col);
      p.out[static_cast<int64_t>(i) * p.C + col] = st(v);
    }
  }
}

// ###########################################################################
// Host side: eligibility, workspace, launches.
// ###########################################################################

// Sufficient conditions only. Anything not admitted here is signalled back to
// Python, which then runs the baseline formula through the same submodules.
constexpr int kMaxTokens = 64;
constexpr int kMaxNQuery = 64;
constexpr int kMaxNKey = 256;
constexpr int kMaxD = 64;
constexpr int kMaxCz = 256;
constexpr int kMaxWidth = 2048;
constexpr int kMaxHD = 2048;
constexpr int kMaxHeads = 64;
constexpr int kMaxAtoms = 1 << 20;

at::Tensor refuse(const at::Tensor& a) {
  return at::empty({0}, a.options());
}

// Deliberately the per-tensor half of ``at::isTensorSubclassLike`` and not that
// function itself: it short-circuits to true whenever *any* dispatch mode is
// active, and a plain counting or logging ``TorchDispatchMode`` over ordinary CUDA
// tensors is exactly the situation in which the fused path has to remain
// observable. The key-set test is the precise condition -- a FakeTensor, a
// functorch wrapper or a ``__torch_dispatch__`` subclass all carry one of these
// keys and are refused, while a real tensor seen through a mode is not.
bool subclass_like(const at::Tensor& t) {
  return !(t.unsafeGetTensorImpl()->key_set() & at::kTensorSubclassLike).empty();
}

bool plain(const at::Tensor& t) {
  return t.defined() && t.is_cuda() && t.is_contiguous() &&
         t.scalar_type() == at::kBFloat16 && !subclass_like(t) &&
         !t.is_conj() && !t.is_neg();
}

bool plain_opt(const std::optional<at::Tensor>& t) {
  return !t.has_value() || plain(*t);
}

const bf16_t* raw(const at::Tensor& t) {
  return t.defined() ? reinterpret_cast<const bf16_t*>(t.const_data_ptr())
                     : nullptr;
}

const bf16_t* raw_opt(const std::optional<at::Tensor>& t) {
  return (t.has_value() && t->defined()) ? raw(*t) : nullptr;
}

at::Tensor get(const std::optional<at::Tensor>& t) {
  return t.has_value() ? *t : at::Tensor();
}

// prod(sizes[:-trailing]) -- the batch extent the fused path requires to be 1.
int64_t lead_extent(const at::Tensor& t, int trailing) {
  int64_t n = 1;
  for (int64_t i = 0; i + trailing < t.dim(); ++i) n *= t.size(i);
  return n;
}

// Byte offsets into one allocation, each 256-byte aligned so every typed view is
// naturally aligned and no offset can be produced by an overflowing product.
struct Arena {
  int64_t total = 0;
  bool overflow = false;

  int64_t take(int64_t count, int64_t elem) {
    if (count < 0 || elem <= 0 || count > (int64_t{1} << 40)) {
      overflow = true;
      return 0;
    }
    const int64_t off = total;
    const int64_t bytes = count * elem;
    total = off + ((bytes + 255) / 256) * 256;
    if (total < off) overflow = true;
    return off;
  }
};

std::vector<int64_t> broadcast_out_shape(const at::Tensor& a,
                                         const at::Tensor& s, int64_t rows,
                                         int64_t width) {
  std::vector<int64_t> la(a.sizes().begin(), a.sizes().end() - 2);
  std::vector<int64_t> ls;
  if (s.defined()) ls.assign(s.sizes().begin(), s.sizes().end() - 2);
  const size_t rank = std::max(la.size(), ls.size());
  std::vector<int64_t> shape(rank + 2, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = (i + la.size() >= rank) ? la[i + la.size() - rank] : 1;
    const int64_t ds = (i + ls.size() >= rank) ? ls[i + ls.size() - rank] : 1;
    shape[i] = std::max(da, ds);
  }
  shape[rank] = rows;
  shape[rank + 1] = width;
  return shape;
}

at::Tensor attn_pair_bias(
    const at::Tensor& a, const at::Tensor& z, const std::optional<at::Tensor>& s,
    const std::optional<at::Tensor>& mask,
    const std::optional<at::Tensor>& ln_a_w,
    const std::optional<at::Tensor>& ln_a_b,
    const std::optional<at::Tensor>& ada_s_norm_w,
    const std::optional<at::Tensor>& ada_g_w,
    const std::optional<at::Tensor>& ada_g_b,
    const std::optional<at::Tensor>& ada_s_w, const at::Tensor& ln_z_w,
    const std::optional<at::Tensor>& ln_z_b, const at::Tensor& w_z,
    const at::Tensor& w_q, const std::optional<at::Tensor>& b_q,
    const at::Tensor& w_k, const at::Tensor& w_v,
    const std::optional<at::Tensor>& w_g, const at::Tensor& w_o,
    const std::optional<at::Tensor>& w_ada_out,
    const std::optional<at::Tensor>& b_ada_out, int64_t no_heads,
    int64_t c_hidden, double inf, double eps, bool use_ada) {
  if (at::GradMode::is_enabled() ||
      at::autocast::is_autocast_enabled(at::kCUDA)) {
    return refuse(a);
  }
  if (!(plain(a) && plain(z) && plain(ln_z_w) && plain(w_z) && plain(w_q) &&
        plain(w_k) && plain(w_v) && plain(w_o))) {
    return refuse(a);
  }
  if (!(plain_opt(s) && plain_opt(mask) && plain_opt(ln_a_w) &&
        plain_opt(ln_a_b) && plain_opt(ada_s_norm_w) && plain_opt(ada_g_w) &&
        plain_opt(ada_g_b) && plain_opt(ada_s_w) && plain_opt(ln_z_b) &&
        plain_opt(b_q) && plain_opt(w_g) && plain_opt(w_ada_out) &&
        plain_opt(b_ada_out))) {
    return refuse(a);
  }
  if (!std::isfinite(inf) || !(eps > 0.0) || !std::isfinite(eps)) {
    return refuse(a);
  }
  if (a.dim() < 2 || z.dim() < 3 || no_heads <= 0 || c_hidden <= 0) {
    return refuse(a);
  }

  const int64_t N = a.size(-2);
  const int64_t C = a.size(-1);
  const int64_t cz = z.size(-1);
  const int64_t H = no_heads;
  const int64_t D = c_hidden;
  const int64_t HD = H * D;
  if (N <= 0 || C <= 0 || cz <= 0) return refuse(a);
  if (N > kMaxTokens || C > kMaxWidth || cz > kMaxCz || D > kMaxD ||
      H > kMaxHeads || HD > kMaxHD) {
    return refuse(a);
  }
  if (lead_extent(a, 2) != 1) return refuse(a);
  if (z.size(-2) != N || z.size(-3) != N || lead_extent(z, 3) != 1) {
    return refuse(a);
  }
  if (w_q.dim() != 2 || w_q.size(0) != HD || w_q.size(1) != C) return refuse(a);
  if (w_k.dim() != 2 || w_k.size(0) != HD || w_k.size(1) != C) return refuse(a);
  if (w_v.dim() != 2 || w_v.size(0) != HD || w_v.size(1) != C) return refuse(a);
  if (w_o.dim() != 2 || w_o.size(0) != C || w_o.size(1) != HD) return refuse(a);
  if (w_z.dim() != 2 || w_z.size(0) != H || w_z.size(1) != cz) return refuse(a);
  if (ln_z_w.dim() != 1 || ln_z_w.size(0) != cz) return refuse(a);
  if (ln_z_b.has_value() &&
      (ln_z_b->dim() != 1 || ln_z_b->size(0) != cz)) {
    return refuse(a);
  }
  if (b_q.has_value() && (b_q->dim() != 1 || b_q->size(0) != HD)) {
    return refuse(a);
  }
  if (w_g.has_value() &&
      (w_g->dim() != 2 || w_g->size(0) != HD || w_g->size(1) != C)) {
    return refuse(a);
  }
  // Extent and prefix, not just ``numel``: a [4, 4] mask has sixteen elements
  // like a 16-token mask does, and the baseline's ``expand`` over the batch dims
  // would do something else with it -- or raise.
  if (mask.has_value() && (mask->dim() < 1 || mask->size(-1) != N ||
                           lead_extent(*mask, 1) != 1)) {
    return refuse(a);
  }

  int64_t cs = 0;
  if (use_ada) {
    if (!s.has_value() || !s->defined() || s->dim() < 2) return refuse(a);
    cs = s->size(-1);
    if (cs <= 0 || cs > kMaxWidth) return refuse(a);
    if (s->size(-2) != N || lead_extent(*s, 2) != 1) return refuse(a);
    if (!(ada_s_norm_w.has_value() && ada_g_w.has_value() &&
          ada_g_b.has_value() && ada_s_w.has_value() && w_ada_out.has_value() &&
          b_ada_out.has_value())) {
      return refuse(a);
    }
    if (ada_s_norm_w->dim() != 1 || ada_s_norm_w->size(0) != cs) return refuse(a);
    if (ada_g_w->dim() != 2 || ada_g_w->size(0) != C || ada_g_w->size(1) != cs) {
      return refuse(a);
    }
    if (ada_g_b->dim() != 1 || ada_g_b->size(0) != C) return refuse(a);
    if (ada_s_w->dim() != 2 || ada_s_w->size(0) != C || ada_s_w->size(1) != cs) {
      return refuse(a);
    }
    if (w_ada_out->dim() != 2 || w_ada_out->size(0) != C ||
        w_ada_out->size(1) != cs) {
      return refuse(a);
    }
    if (b_ada_out->dim() != 1 || b_ada_out->size(0) != C) return refuse(a);
    if (ln_z_b.has_value()) return refuse(a);   // conditioned LN_z has no offset
  } else {
    if (!(ln_a_w.has_value() && ln_a_b.has_value())) return refuse(a);
    if (ln_a_w->dim() != 1 || ln_a_w->size(0) != C) return refuse(a);
    if (ln_a_b->dim() != 1 || ln_a_b->size(0) != C) return refuse(a);
    if (!ln_z_b.has_value()) return refuse(a);
  }

  // Shared-memory budgets, checked rather than assumed.
  const int64_t row_smem = std::max(std::max(C, cz), cs);
  if ((row_smem + kWarps + 1) * 4 > kSmemLimit) return refuse(a);
  // qs + scores + mask bias in fp32, then K (transposed) and V in bf16.
  const int64_t attend_smem = (N * D + N * N + N) * 4 + 2 * N * D * 2;
  if (attend_smem > kSmemLimit) return refuse(a);

  const c10::cuda::CUDAGuard guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();

  Arena arena;
  const int64_t off_zb = arena.take(H * N * N, 2);
  const int64_t off_ax = arena.take(N * C, 2);
  const int64_t off_ahat = use_ada ? arena.take(N * C, 2) : 0;
  const int64_t off_sx = use_ada ? arena.take(N * cs, 2) : 0;
  const int64_t off_gout = use_ada ? arena.take(N * C, 2) : 0;
  const int64_t off_qkvg = arena.take(4 * N * HD, 2);
  const int64_t off_o = arena.take(N * HD, 2);
  if (arena.overflow) return refuse(a);

  at::Tensor ws = at::empty({arena.total}, a.options().dtype(at::kByte));
  auto* base = static_cast<char*>(ws.mutable_data_ptr());
  // Only the conditioned variant multiplies by a gate derived from ``s``, so
  // only there does ``s``'s batch rank take part in the output's shape.
  const at::Tensor s_t = use_ada ? get(s) : at::Tensor();
  at::Tensor out = at::empty(broadcast_out_shape(a, s_t, N, C), a.options());

  ApbParams p{};
  p.a = raw(a);
  p.z = raw(z);
  p.s = raw_opt(s);
  p.mask = raw_opt(mask);
  p.ln_a_w = raw_opt(ln_a_w);
  p.ln_a_b = raw_opt(ln_a_b);
  p.ada_s_norm_w = raw_opt(ada_s_norm_w);
  p.ada_g_w = raw_opt(ada_g_w);
  p.ada_g_b = raw_opt(ada_g_b);
  p.ada_s_w = raw_opt(ada_s_w);
  p.ln_z_w = raw(ln_z_w);
  p.ln_z_b = raw_opt(ln_z_b);
  p.w_z = raw(w_z);
  p.w_q = raw(w_q);
  p.b_q = raw_opt(b_q);
  p.w_k = raw(w_k);
  p.w_v = raw(w_v);
  p.w_g = raw_opt(w_g);
  p.w_o = raw(w_o);
  p.w_ada_out = raw_opt(w_ada_out);
  p.b_ada_out = raw_opt(b_ada_out);
  p.zb = reinterpret_cast<bf16_t*>(base + off_zb);
  p.ax = reinterpret_cast<bf16_t*>(base + off_ax);
  p.ahat = use_ada ? reinterpret_cast<bf16_t*>(base + off_ahat) : nullptr;
  p.sx = use_ada ? reinterpret_cast<bf16_t*>(base + off_sx) : nullptr;
  p.gout = use_ada ? reinterpret_cast<bf16_t*>(base + off_gout) : nullptr;
  p.qkvg = reinterpret_cast<bf16_t*>(base + off_qkvg);
  p.o = reinterpret_cast<bf16_t*>(base + off_o);
  p.out = reinterpret_cast<bf16_t*>(out.mutable_data_ptr());
  p.N = static_cast<int>(N);
  p.C = static_cast<int>(C);
  p.cs = static_cast<int>(cs);
  p.cz = static_cast<int>(cz);
  p.H = static_cast<int>(H);
  p.D = static_cast<int>(D);
  p.HD = static_cast<int>(HD);
  p.row_smem = static_cast<int>(row_smem);
  p.use_ada = use_ada ? 1 : 0;
  p.inf = static_cast<float>(inf);
  p.eps = static_cast<float>(eps);
  p.sqrt_d = static_cast<float>(std::sqrt(static_cast<double>(D)));

  const int rowtiles = static_cast<int>((N + kTileRows - 1) / kTileRows);
  const int ctiles = static_cast<int>((C + kWarps - 1) / kWarps);
  const size_t tile_smem = sizeof(bf16_t) * kTileRows * kChunk;

  apb_norms<<<static_cast<unsigned>(N * N + N + (use_ada ? N : 0)), kThreads,
              (row_smem + kWarps + 1) * sizeof(float), stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (use_ada) {
    apb_conditioning<<<dim3(2 * ctiles, rowtiles), kThreads, tile_smem,
                       stream>>>(p, ctiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int per = static_cast<int>((HD + kWarps - 1) / kWarps);
  apb_project<<<dim3(4 * per, rowtiles), kThreads, tile_smem, stream>>>(p, per);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  apb_attend<<<static_cast<unsigned>(H), kThreads, attend_smem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  apb_output<<<dim3(ctiles, rowtiles), kThreads, tile_smem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cross_attn_pair_bias(
    const at::Tensor& a, const at::Tensor& z, const std::optional<at::Tensor>& s,
    const std::optional<at::Tensor>& mask, const at::Tensor& q_norm_w,
    const at::Tensor& q_g_w, const at::Tensor& q_g_b, const at::Tensor& q_s_w,
    const at::Tensor& k_norm_w, const at::Tensor& k_g_w, const at::Tensor& k_g_b,
    const at::Tensor& k_s_w, const at::Tensor& w_z, const at::Tensor& w_q,
    const std::optional<at::Tensor>& b_q, const at::Tensor& w_k,
    const at::Tensor& w_v, const std::optional<at::Tensor>& w_g,
    const at::Tensor& w_o, const at::Tensor& w_ada_out,
    const at::Tensor& b_ada_out, int64_t no_heads, int64_t c_hidden,
    int64_t n_query, int64_t n_key, double inf, double eps) {
  if (at::GradMode::is_enabled() ||
      at::autocast::is_autocast_enabled(at::kCUDA)) {
    return refuse(a);
  }
  if (!(plain(a) && plain(z) && plain(q_norm_w) && plain(q_g_w) &&
        plain(q_g_b) && plain(q_s_w) && plain(k_norm_w) && plain(k_g_w) &&
        plain(k_g_b) && plain(k_s_w) && plain(w_z) && plain(w_q) && plain(w_k) &&
        plain(w_v) && plain(w_o) && plain(w_ada_out) && plain(b_ada_out))) {
    return refuse(a);
  }
  if (!(plain_opt(s) && plain_opt(mask) && plain_opt(b_q) && plain_opt(w_g))) {
    return refuse(a);
  }
  if (!std::isfinite(inf) || !(eps > 0.0) || !std::isfinite(eps)) {
    return refuse(a);
  }
  if (!s.has_value() || !s->defined()) return refuse(a);
  if (a.dim() < 2 || z.dim() < 4 || s->dim() < 2) return refuse(a);
  if (n_query <= 0 || n_key <= 0 || no_heads <= 0 || c_hidden <= 0) {
    return refuse(a);
  }

  const int64_t n_atom = a.size(-2);
  const int64_t C = a.size(-1);
  const int64_t cs = s->size(-1);
  const int64_t cz = z.size(-1);
  const int64_t H = no_heads;
  const int64_t D = c_hidden;
  const int64_t HD = H * D;
  if (n_atom <= 0 || C <= 0 || cs <= 0 || cz <= 0) return refuse(a);
  const int64_t nb = (n_atom + n_query - 1) / n_query;
  const int64_t apad = nb * n_query;
  if (n_atom > kMaxAtoms || C > kMaxWidth || cs > kMaxWidth || cz > kMaxCz ||
      D > kMaxD || H > kMaxHeads || HD > kMaxHD || n_query > kMaxNQuery ||
      n_key > kMaxNKey) {
    return refuse(a);
  }
  if (lead_extent(a, 2) != 1 || lead_extent(*s, 2) != 1) return refuse(a);
  if (s->size(-2) != n_atom) return refuse(a);
  if (z.size(-2) != n_key || z.size(-3) != n_query || z.size(-4) != nb ||
      lead_extent(z, 4) != 1) {
    return refuse(a);
  }
  // ``mask=None`` makes the module fabricate ``a.new_ones(a.shape[:-1])``, which
  // the baseline can only expand over ``s`` when the two ranks agree -- otherwise
  // it raises, and the fallback has to be the one that raises.
  if (!mask.has_value() || !mask->defined()) {
    if (a.dim() != s->dim()) return refuse(a);
  } else if (mask->dim() < 1 || mask->size(-1) != n_atom ||
             lead_extent(*mask, 1) != 1) {
    return refuse(a);
  }
  if (w_q.dim() != 2 || w_q.size(0) != HD || w_q.size(1) != C) return refuse(a);
  if (w_k.dim() != 2 || w_k.size(0) != HD || w_k.size(1) != C) return refuse(a);
  if (w_v.dim() != 2 || w_v.size(0) != HD || w_v.size(1) != C) return refuse(a);
  if (w_o.dim() != 2 || w_o.size(0) != C || w_o.size(1) != HD) return refuse(a);
  if (w_z.dim() != 2 || w_z.size(0) != H || w_z.size(1) != cz) return refuse(a);
  if (b_q.has_value() && (b_q->dim() != 1 || b_q->size(0) != HD)) {
    return refuse(a);
  }
  if (w_g.has_value() &&
      (w_g->dim() != 2 || w_g->size(0) != HD || w_g->size(1) != C)) {
    return refuse(a);
  }
  for (const at::Tensor* t : {&q_norm_w, &k_norm_w}) {
    if (t->dim() != 1 || t->size(0) != cs) return refuse(a);
  }
  for (const at::Tensor* t : {&q_g_w, &q_s_w, &k_g_w, &k_s_w, &w_ada_out}) {
    if (t->dim() != 2 || t->size(0) != C || t->size(1) != cs) return refuse(a);
  }
  for (const at::Tensor* t : {&q_g_b, &k_g_b, &b_ada_out}) {
    if (t->dim() != 1 || t->size(0) != C) return refuse(a);
  }

  const int64_t row_smem = std::max(C, cs);
  if ((row_smem + kWarps + 1) * 4 > kSmemLimit) return refuse(a);
  // One head and kQueryRows query rows per CTA: the pair-bias weights, the
  // gathered mask, the scaled queries and the scores in fp32, plus the gathered
  // K and V in bf16.
  const int64_t attend_smem = (cz + n_key + kQueryRows * D + kQueryRows * n_key) * 4
                              + 2 * n_key * D * 2;
  if (attend_smem > kSmemLimit) return refuse(a);

  const c10::cuda::CUDAGuard guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();

  Arena arena;
  const int64_t off_idx = arena.take(nb * n_key, 4);
  const int64_t off_kvalid = arena.take(nb * n_key, 4);
  const int64_t off_kmask = arena.take(nb * n_key, 4);
  const int64_t off_ahat = arena.take(apad * C, 2);
  const int64_t off_sxq = arena.take(apad * cs, 2);
  const int64_t off_sxk = arena.take(apad * cs, 2);
  const int64_t off_aq = arena.take(apad * C, 2);
  const int64_t off_ak = arena.take(apad * C, 2);
  const int64_t off_gout = arena.take(n_atom * C, 2);
  const int64_t off_qg = arena.take(2 * apad * HD, 2);
  const int64_t off_kv = arena.take(2 * apad * HD, 2);
  const int64_t off_o = arena.take(apad * HD, 2);
  if (arena.overflow) return refuse(a);

  at::Tensor ws = at::empty({arena.total}, a.options().dtype(at::kByte));
  auto* base = static_cast<char*>(ws.mutable_data_ptr());
  at::Tensor ones;
  const bf16_t* mask_ptr = raw_opt(mask);
  if (mask_ptr == nullptr) {
    // Only reachable on the no-mask path the predicate admitted; materialising
    // ones here keeps the kernels free of a second mask convention. This costs
    // one extra ATen call, but only on a path the benchmark never takes.
    ones = at::ones({n_atom}, a.options());
    mask_ptr = raw(ones);
  }

  at::Tensor out = at::empty(broadcast_out_shape(a, *s, n_atom, C), a.options());

  CapbParams p{};
  p.a = raw(a);
  p.z = raw(z);
  p.s = raw(*s);
  p.mask = mask_ptr;
  p.q_norm_w = raw(q_norm_w);
  p.q_g_w = raw(q_g_w);
  p.q_g_b = raw(q_g_b);
  p.q_s_w = raw(q_s_w);
  p.k_norm_w = raw(k_norm_w);
  p.k_g_w = raw(k_g_w);
  p.k_g_b = raw(k_g_b);
  p.k_s_w = raw(k_s_w);
  p.w_z = raw(w_z);
  p.w_q = raw(w_q);
  p.b_q = raw_opt(b_q);
  p.w_k = raw(w_k);
  p.w_v = raw(w_v);
  p.w_g = raw_opt(w_g);
  p.w_o = raw(w_o);
  p.w_ada_out = raw(w_ada_out);
  p.b_ada_out = raw(b_ada_out);
  p.idx = reinterpret_cast<int32_t*>(base + off_idx);
  p.kvalid = reinterpret_cast<float*>(base + off_kvalid);
  p.kmask = reinterpret_cast<float*>(base + off_kmask);
  p.ahat = reinterpret_cast<bf16_t*>(base + off_ahat);
  p.sxq = reinterpret_cast<bf16_t*>(base + off_sxq);
  p.sxk = reinterpret_cast<bf16_t*>(base + off_sxk);
  p.aq = reinterpret_cast<bf16_t*>(base + off_aq);
  p.ak = reinterpret_cast<bf16_t*>(base + off_ak);
  p.gout = reinterpret_cast<bf16_t*>(base + off_gout);
  p.qg = reinterpret_cast<bf16_t*>(base + off_qg);
  p.kv = reinterpret_cast<bf16_t*>(base + off_kv);
  p.o = reinterpret_cast<bf16_t*>(base + off_o);
  p.out = reinterpret_cast<bf16_t*>(out.mutable_data_ptr());
  p.n_atom = static_cast<int>(n_atom);
  p.apad = static_cast<int>(apad);
  p.nb = static_cast<int>(nb);
  p.nq = static_cast<int>(n_query);
  p.nk = static_cast<int>(n_key);
  p.C = static_cast<int>(C);
  p.cs = static_cast<int>(cs);
  p.cz = static_cast<int>(cz);
  p.H = static_cast<int>(H);
  p.D = static_cast<int>(D);
  p.HD = static_cast<int>(HD);
  p.row_smem = static_cast<int>(row_smem);
  p.inf = static_cast<float>(inf);
  p.eps = static_cast<float>(eps);
  p.sqrt_d = static_cast<float>(std::sqrt(static_cast<double>(D)));

  const int ctiles = static_cast<int>((C + kWarps - 1) / kWarps);
  const int atiles = static_cast<int>((apad + kTileRows - 1) / kTileRows);
  const int otiles = static_cast<int>((n_atom + kTileRows - 1) / kTileRows);
  const size_t tile_smem = sizeof(bf16_t) * kTileRows * kChunk;

  capb_prepare<<<static_cast<unsigned>(apad + 1), kThreads,
                 (row_smem + kWarps + 1) * sizeof(float), stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  capb_conditioning<<<dim3(3 * ctiles, atiles), kThreads, tile_smem, stream>>>(
      p, ctiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int per = static_cast<int>((HD + kWarps - 1) / kWarps);
  capb_project<<<dim3(4 * per, atiles), kThreads, tile_smem, stream>>>(p, per);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  capb_attend<<<dim3(static_cast<unsigned>(nb), static_cast<unsigned>(H),
                     static_cast<unsigned>((n_query + kQueryRows - 1) /
                                           kQueryRows)),
                kThreads, attend_smem, stream>>>(p, kQueryRows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  capb_output<<<dim3(ctiles, otiles), kThreads, tile_smem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// The oracle entry point for the gather table. Never called by ``forward``; it
// exists so a test can compare the device tables the fused path would build
// against ``_get_block_key_indices`` on device tensors, without reimplementing
// the rule inside the assertion.
std::tuple<at::Tensor, at::Tensor, at::Tensor> key_table(const at::Tensor& mask,
                                                        int64_t n_query,
                                                        int64_t n_key) {
  TORCH_CHECK(plain(mask) && mask.dim() == 1,
              "key_table expects a contiguous 1-D bf16 CUDA mask");
  TORCH_CHECK(n_query > 0 && n_key > 0, "key_table needs positive block sizes");
  const int64_t n_atom = mask.size(0);
  const int64_t nb = (n_atom + n_query - 1) / n_query;
  const int64_t apad = nb * n_query;
  const c10::cuda::CUDAGuard guard(mask.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  at::Tensor idx = at::empty({nb, n_key}, mask.options().dtype(at::kInt));
  at::Tensor kvalid = at::empty({nb, n_key}, mask.options().dtype(at::kFloat));
  at::Tensor kmask = at::empty({nb, n_key}, mask.options().dtype(at::kFloat));
  capb_key_table<<<1, kThreads, (kWarps + 1) * sizeof(float), stream>>>(
      raw(mask), static_cast<int>(n_atom), static_cast<int>(apad),
      static_cast<int>(nb), static_cast<int>(n_query), static_cast<int>(n_key),
      static_cast<int32_t*>(idx.mutable_data_ptr()),
      static_cast<float*>(kvalid.mutable_data_ptr()),
      static_cast<float*>(kmask.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {idx, kvalid, kmask};
}

}  // namespace

// Schema and implementation are registered separately, and the implementation
// lands on the CUDA key rather than as a catch-all. That is not cosmetic: a
// catch-all kernel is registered above the dispatcher's Python key, so the op is
// entered before a ``TorchDispatchMode`` can see it and a mode observes only the
// ATen calls made *inside* it. Keyed to CUDA, one call shows up as exactly one
// dispatch, which is the property the dispatch-count test needs to be able to
// check. The Python side only reaches here for CUDA tensors with grad off, and
// the predicate below re-checks both.
TORCH_LIBRARY(fk_af3_apb_cand, m) {
  m.def(
      "attn_pair_bias(Tensor a, Tensor z, Tensor? s, Tensor? mask, "
      "Tensor? ln_a_w, Tensor? ln_a_b, Tensor? ada_s_norm_w, Tensor? ada_g_w, "
      "Tensor? ada_g_b, Tensor? ada_s_w, Tensor ln_z_w, Tensor? ln_z_b, "
      "Tensor w_z, Tensor w_q, Tensor? b_q, Tensor w_k, Tensor w_v, "
      "Tensor? w_g, Tensor w_o, Tensor? w_ada_out, Tensor? b_ada_out, "
      "int no_heads, int c_hidden, float inf, float eps, bool use_ada"
      ") -> Tensor");
  m.def(
      "cross_attn_pair_bias(Tensor a, Tensor z, Tensor? s, Tensor? mask, "
      "Tensor q_norm_w, Tensor q_g_w, Tensor q_g_b, Tensor q_s_w, "
      "Tensor k_norm_w, Tensor k_g_w, Tensor k_g_b, Tensor k_s_w, "
      "Tensor w_z, Tensor w_q, Tensor? b_q, Tensor w_k, Tensor w_v, "
      "Tensor? w_g, Tensor w_o, Tensor w_ada_out, Tensor b_ada_out, "
      "int no_heads, int c_hidden, int n_query, int n_key, float inf, float eps"
      ") -> Tensor");
  m.def("key_table(Tensor mask, int n_query, int n_key) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fk_af3_apb_cand, CUDA, m) {
  m.impl("attn_pair_bias", TORCH_FN(attn_pair_bias));
  m.impl("cross_attn_pair_bias", TORCH_FN(cross_attn_pair_bias));
  m.impl("key_table", TORCH_FN(key_table));
}
"""


def _load_fused_ops():
    """Build and register both operators at import, returning their callables.

    At import rather than on first use: the harness's guards aside, compiling
    inside a timed ``forward`` would put nvcc in the scored window, and building
    here happens while the worker is still producing output, clear of the
    no-output stall watchdog. The includes are lean because ``TORCH_LIBRARY``
    needs none of ``<torch/extension.h>``, which is what dominates the build.
    """
    from torch.utils.cpp_extension import load_inline

    build_dir = os.environ.get("FK_AF3_APB_BUILD_DIR")
    if build_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        build_dir = os.path.join(os.path.dirname(root), ".torch_extensions")
    os.makedirs(build_dir, exist_ok=True)

    # The ambient arch list names every architecture the environment might ever
    # target, which is one nvcc pass each. Narrowed to the live device -- derived,
    # never hardcoded -- and restored so no later build in this process is
    # affected.
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        load_inline(
            name=_LIBRARY_NAME,
            cpp_sources="",
            cuda_sources=_CUDA_SOURCE,
            extra_cuda_cflags=["-O3", "-lineinfo"],
            is_python_module=False,
            no_implicit_headers=True,
            build_directory=build_dir,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list

    lib = getattr(torch.ops, _LIBRARY_NAME)
    # The concrete overloads, not the packets: a packet re-resolves overloads from
    # the argument types on every call, and forward is latency-bound.
    return lib.attn_pair_bias.default, lib.cross_attn_pair_bias.default


# Why the build outcome is recorded rather than discarded: degrading silently is
# the right runtime behaviour and a terrible diagnostic. A broken build makes every
# call fall back, which is *correct* and also indistinguishable from a kernel that
# simply is not being reached -- so the status is published for the checker to
# assert on, and the checker fails the run if the fused path was not taken.
_BUILD_STATUS = "disabled by FK_AF3_APB_DISABLE"
if os.environ.get("FK_AF3_APB_DISABLE"):
    # The switch the contract tests use to exercise the degradation path without
    # having to break the toolchain.
    _fused_apb = None
    _fused_capb = None
else:
    try:
        _fused_apb, _fused_capb = _load_fused_ops()
        _BUILD_STATUS = "ok"
    except Exception as exc:  # noqa: BLE001 - a build that cannot happen must
        # degrade, not take the module down with it: an import failure costs every
        # case at once, where a fallback costs only the speedup.
        _fused_apb = None
        _fused_capb = None
        _BUILD_STATUS = f"{type(exc).__name__}: {exc}"[:4000]


def _norm_ok(ln, eps) -> bool:
    """A LayerNorm the kernel's formula actually reproduces."""
    return (getattr(ln, "promote_fp32", False)
            and float(getattr(ln, "eps", -1.0)) == eps
            and getattr(ln, "elementwise_affine", False))


class AttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Attention with pair bias.

    When use_ada_layer_norm is True, uses two separate AdaLN instances
    (layer_norm_a_q, layer_norm_a_k) for query and key normalization,
    plus a linear_ada_out for output gating.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        # Whether the *configuration* -- never a weight value -- is one the fused
        # kernels implement. Derived from the submodules just constructed, so it
        # cannot go stale against the weights the harness loads afterwards.
        self._eps = float(self.layer_norm_z.eps)
        norms = [self.layer_norm_z]
        if use_ada_layer_norm:
            norms += [self.layer_norm_a.layer_norm_a, self.layer_norm_a.layer_norm_s]
        else:
            norms.append(self.layer_norm_a)
        self._fused_ok = (
            c_k == c_q and c_v == c_q
            and all(_norm_ok(n, self._eps) for n in norms)
            and (not use_ada_layer_norm or c_s > 0)
        )

    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = _permute_final_dims(z, [2, 0, 1])
        biases.append(z)

        return biases

    def _module_forward(
        self, a: torch.Tensor, z: torch.Tensor, s: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """The baseline formula, through the same submodules, for refused inputs."""
        biases = self._prep_bias(a=a, z=z, mask=mask)
        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)
        a = self.mha(q_x=a, kv_x=a, biases=biases)
        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a
        return a

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        if (_fused_apb is not None and self._fused_ok and a.is_cuda
                and not torch.is_grad_enabled()):
            # Metadata reads and a TLS flag: no dispatch, no kernel. They are here
            # rather than only in C++ because the operator is registered on the
            # CUDA key alone, so a CPU tensor would find no kernel instead of
            # taking the fallback, and a grad-enabled call would route through the
            # autograd fallback rather than this code.
            mha = self.mha
            ada = self.layer_norm_a if self.use_ada_layer_norm else None
            # The live eps, not the one snapshotted in __init__: cheap enough
            # (one attribute read and one compare) and it means a caller who
            # retunes a LayerNorm after construction gets the fallback rather
            # than a kernel silently using the old value.
            eps = float(self.layer_norm_z.eps)
            if eps != self._eps:
                return self._module_forward(a, z, s, mask)
            out = _fused_apb(
                a, z, s, mask,
                None if ada is not None else self.layer_norm_a.weight,
                None if ada is not None else self.layer_norm_a.bias,
                None if ada is None else ada.layer_norm_s.weight,
                None if ada is None else ada.linear_g.weight,
                None if ada is None else ada.linear_g.bias,
                None if ada is None else ada.linear_s.weight,
                self.layer_norm_z.weight, self.layer_norm_z.bias,
                self.linear_z.weight,
                mha.linear_q.weight, mha.linear_q.bias,
                mha.linear_k.weight, mha.linear_v.weight,
                None if mha.linear_g is None else mha.linear_g.weight,
                mha.linear_o.weight,
                None if ada is None else self.linear_ada_out.weight,
                None if ada is None else self.linear_ada_out.bias,
                mha.no_heads, mha.c_hidden, self.inf, eps,
                self.use_ada_layer_norm,
            )
            # A zero-element result is the operator's refusal signal; reading
            # ``numel`` is metadata, not a dispatch, so the admitted path stays at
            # exactly one.
            if out.numel() != 0:
                return out
        return self._module_forward(a, z, s, mask)


class CrossAttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Uses separate layer_norm_a_q and layer_norm_a_k for query/key, and
    does NOT apply layer_norm_z (pair bias goes through linear_z directly).
    Handles sequence-local blocked inputs.

    Reference: openfold3/core/model/layers/attention_pair_bias.py CrossAttentionPairBias

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        # ``use_ada_layer_norm=False`` is refused outright: there a zeroed
        # invalid key row normalises to the LayerNorm offset rather than to zero,
        # so the per-atom form is no longer an identity.
        self._eps = 0.0
        self._fused_ok = False
        if use_ada_layer_norm and c_s > 0 and c_k == c_q and c_v == c_q:
            self._eps = float(self.layer_norm_a_q.layer_norm_a.eps)
            norms = [self.layer_norm_a_q.layer_norm_a,
                     self.layer_norm_a_q.layer_norm_s,
                     self.layer_norm_a_k.layer_norm_a,
                     self.layer_norm_a_k.layer_norm_s]
            self._fused_ok = (
                all(_norm_ok(n, self._eps) for n in norms)
                and isinstance(n_query, int) and isinstance(n_key, int)
                and n_query > 0 and n_key > 0
            )

    def _module_forward(
        self, a: torch.Tensor, z: torch.Tensor, s: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """The baseline formula, through the same submodules, for refused inputs."""
        from .alphafold3_atom_attention import (
            _convert_single_rep_to_blocks, _apply_block_indices,
        )

        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        a_query, a_key, block_mask = _convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
        )

        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        z_bias = self.linear_z(z)
        z_bias = _permute_final_dims(z_bias, [2, 0, 1])
        biases.append(z_bias)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = _apply_block_indices(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
            )
            a_q = self.layer_norm_a_q(a_query, s_q)
            a_k = self.layer_norm_a_k(a_key, s_k)
        else:
            a_q = self.layer_norm_a_q(a_query)
            a_k = self.layer_norm_a_k(a_key)

        a_out = self.mha(q_x=a_q, kv_x=a_k, biases=biases)

        a_out = a_out.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a_out = self.sigmoid(self.linear_ada_out(s)) * a_out

        return a_out

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """
        if (_fused_capb is not None and self._fused_ok and s is not None
                and a.is_cuda and not torch.is_grad_enabled()):
            # See the note in AttentionPairBias.forward: the operator is keyed to
            # CUDA, so these two cheap metadata checks have to precede the call.
            mha = self.mha
            q, k = self.layer_norm_a_q, self.layer_norm_a_k
            eps = float(q.layer_norm_a.eps)
            if eps != self._eps:
                return self._module_forward(a, z, s, mask)
            out = _fused_capb(
                a, z, s, mask,
                q.layer_norm_s.weight, q.linear_g.weight, q.linear_g.bias,
                q.linear_s.weight,
                k.layer_norm_s.weight, k.linear_g.weight, k.linear_g.bias,
                k.linear_s.weight,
                self.linear_z.weight,
                mha.linear_q.weight, mha.linear_q.bias,
                mha.linear_k.weight, mha.linear_v.weight,
                None if mha.linear_g is None else mha.linear_g.weight,
                mha.linear_o.weight,
                self.linear_ada_out.weight, self.linear_ada_out.bias,
                mha.no_heads, mha.c_hidden, self.n_query, self.n_key,
                self.inf, eps,
            )
            if out.numel() != 0:
                return out
        return self._module_forward(a, z, s, mask)
