"""MSA pair-weighted averaging for AlphaFold3 (Algorithm 10), fused into one CUDA kernel.

Weighted averaging over the MSA representation using pair activations,
NOT key-query self-attention.

The reference implementation spends roughly twenty kernel launches on ~2 M MAC of
arithmetic, so it is bound by per-launch fixed cost rather than by math. This
module keeps the reference module contract verbatim and services the captured
configuration with a single kernel; everything outside that configuration takes
the reference path.

Reference: openfold3/core/model/layers/msa.py MSAPairWeightedAveraging
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.softmax import Softmax
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# --------------------------------------------------------------------------- #
# Fused kernel
# --------------------------------------------------------------------------- #
# One CTA per output row (batch, sequence, residue); 256 threads == 8 warps, one
# warp per head and two residues per warp. Every reduction in the operator is
# short (16, 64 or 128 elements), so all of them fit inside a CTA and no second
# kernel or cross-CTA communication is needed. The pair-side normalization and
# the row softmax are recomputed by the CTAs that share a residue, and the MSA
# normalization by the CTAs that share a sequence; that redundancy is a few
# thousand MAC and buys a single launch.
#
# The one algebraic change against the reference op order is to average before
# projecting. The residue average is linear, the softmax weights do not depend on
# the sequence index, and the value weight is head-blocked (channel h*C_H+c
# belongs to head h), so
#
#     o[h,c] = sum_k w[h,k] * sum_j mln[k,j] * Wv[h*C_H+c, j]
#            = sum_j Wv[h*C_H+c, j] * u[h,j],   u[h,j] = sum_k w[h,k] * mln[k,j]
#
# which replaces 66 560 MAC per CTA with 12 288. It is exact over the reals but
# not under the reference's rounding, which rounds the value projection to
# bfloat16 before the residue sum. Accumulation here is fp32 throughout, and
# bfloat16 rounding is applied at exactly the points where the reference casts,
# so the deviation stays dominated by the final store rounding.

_FUSED_RESIDUES = 16  # N_res
_FUSED_PAIR_CHANS = 128  # C_z
_FUSED_MSA_CHANS = 64  # C_m
_FUSED_HEADS = 8  # no_heads
_FUSED_HEAD_CHANS = 8  # c_hidden
_FUSED_EXT_NAME = "fk_af3_msa_pair_weighted_avg"
# The kernel is compiled for exactly one architecture, so any other device has no
# compatible image and must take the reference path instead of failing to launch.
# The match must be exact rather than on the major version: CUDA 13 exposes other
# 10.x targets (sm_103 and beyond) whose code is not compatible with an sm_100a
# binary. Keyed by device index; hardware capability cannot change under a live
# process, so caching it is safe.
_FUSED_ARCH = (10, 0)  # Blackwell / sm_100a
_ARCH_SUPPORTED: dict[int, bool] = {}


def _arch_supported(device: torch.device) -> bool:
    index = device.index if device.index is not None else torch.cuda.current_device()
    supported = _ARCH_SUPPORTED.get(index)
    if supported is None:
        try:
            supported = torch.cuda.get_device_capability(index) == _FUSED_ARCH
        except Exception:  # noqa: BLE001 - treat an unreadable device as unsupported
            supported = False
        _ARCH_SUPPORTED[index] = supported
    return supported


# Largest batch the kernel can launch: the batch indexes grid.z.
_MAX_FUSED_BATCH = 65535

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <optional>

torch::Tensor msa_pair_weighted_avg(
    torch::Tensor msa,
    torch::Tensor pair,
    std::optional<torch::Tensor> mask,
    torch::Tensor msa_ln_weight,
    torch::Tensor msa_ln_bias,
    torch::Tensor pair_ln_weight,
    torch::Tensor pair_ln_bias,
    torch::Tensor pair_proj_weight,
    torch::Tensor value_weight,
    torch::Tensor gate_weight,
    torch::Tensor out_weight,
    double mask_inf,
    double msa_eps,
    double pair_eps);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>

namespace {

constexpr int kResidues = 16;    // N_res
constexpr int kPairChans = 128;  // C_z
constexpr int kMsaChans = 64;    // C_m
constexpr int kHeads = 8;        // no_heads
constexpr int kHeadChans = 8;    // c_hidden, with kHeads * kHeadChans == kMsaChans
constexpr int kWarps = kHeads;   // one warp per head, two residues per warp
constexpr int kThreads = kWarps * 32;
constexpr int kPairPerLane = kPairChans / 32;  // 4
constexpr int kMsaPerLane = kMsaChans / 32;    // 2

static_assert(kHeads * kHeadChans == kMsaChans, "head blocking assumption");
static_assert(kResidues == 2 * kWarps, "two residues per warp");

// Round through bfloat16 wherever the reference casts back to the storage dtype,
// so the fused result tracks the reference's rounding rather than an exact
// fp32 evaluation.
__device__ __forceinline__ float to_bf16_and_back(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  }
  return v;
}

__device__ __forceinline__ __nv_bfloat162 load_bf162(const __nv_bfloat16* p) {
  return *reinterpret_cast<const __nv_bfloat162*>(p);
}

__global__ __launch_bounds__(kThreads) void msa_pair_weighted_avg_kernel(
    const __nv_bfloat16* __restrict__ msa,          // [B, N_seq, N_res, C_m]
    const __nv_bfloat16* __restrict__ pair,         // [B, N_res, N_res, C_z]
    const __nv_bfloat16* __restrict__ mask,         // [B, N_res, N_res] or null
    const __nv_bfloat16* __restrict__ msa_ln_w,     // [C_m]
    const __nv_bfloat16* __restrict__ msa_ln_b,     // [C_m]
    const __nv_bfloat16* __restrict__ pair_ln_w,    // [C_z]
    const __nv_bfloat16* __restrict__ pair_ln_b,    // [C_z]
    const __nv_bfloat16* __restrict__ pair_proj_w,  // [no_heads, C_z]
    const __nv_bfloat16* __restrict__ value_w,      // [C_m, C_m]
    const __nv_bfloat16* __restrict__ gate_w,       // [C_m, C_m]
    const __nv_bfloat16* __restrict__ out_w,        // [C_m, C_m]
    __nv_bfloat16* __restrict__ out,                // [B, N_seq, N_res, C_m]
    const float mask_inf,
    const float msa_eps,
    const float pair_eps) {
  // Pair-side staging is shared by every CTA that owns this residue, so put the
  // only weight with intra-CTA reuse into shared memory and keep the rest
  // streaming from global.
  // Explicitly over-aligned: the bfloat16 arrays are read and written through
  // __nv_bfloat162, and the normalization parameters through float4 / float2, so
  // the natural 2- and 4-byte alignment of the element type is not enough.
  __shared__ __align__(16) __nv_bfloat16 s_pair_proj_w[kHeads * kPairChans];
  __shared__ __align__(16) float s_pair_ln_w[kPairChans];
  __shared__ __align__(16) float s_pair_ln_b[kPairChans];
  __shared__ __align__(16) float s_msa_ln_w[kMsaChans];
  __shared__ __align__(16) float s_msa_ln_b[kMsaChans];
  // Holds the pair logits first, then the row softmax weights in place.
  __shared__ float s_row_weights[kHeads * kResidues];
  __shared__ __align__(16) __nv_bfloat16 s_msa_norm[kResidues * kMsaChans];
  __shared__ __align__(16) __nv_bfloat16 s_gated[kMsaChans];
  __shared__ __align__(16) __nv_bfloat16 s_out[kMsaChans];

  const int seq = blockIdx.x;      // sequence row this CTA writes
  const int query = blockIdx.y;    // residue this CTA writes
  const int batch = blockIdx.z;
  const int n_seq = gridDim.x;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const __nv_bfloat16* pair_block =
      pair + (static_cast<long>(batch) * kResidues + query) * kResidues * kPairChans;
  const __nv_bfloat16* msa_block =
      msa + (static_cast<long>(batch) * n_seq + seq) * kResidues * kMsaChans;
  const __nv_bfloat16* mask_row =
      mask == nullptr
          ? nullptr
          : mask + (static_cast<long>(batch) * kResidues + query) * kResidues;

  for (int i = threadIdx.x; i < kHeads * kPairChans; i += kThreads) {
    s_pair_proj_w[i] = pair_proj_w[i];
  }
  for (int i = threadIdx.x; i < kPairChans; i += kThreads) {
    s_pair_ln_w[i] = __bfloat162float(pair_ln_w[i]);
    s_pair_ln_b[i] = __bfloat162float(pair_ln_b[i]);
  }
  for (int i = threadIdx.x; i < kMsaChans; i += kThreads) {
    s_msa_ln_w[i] = __bfloat162float(msa_ln_w[i]);
    s_msa_ln_b[i] = __bfloat162float(msa_ln_b[i]);
  }
  __syncthreads();

  // ---- layer_norm_pair + pair_logits -------------------------------------
  // Warp `warp` owns residues `warp` and `warp + kWarps`; lane `lane` owns the
  // four channels starting at 4*lane. The normalized row never leaves registers.
  //
  // Each lane's affine slice is four consecutive floats, so read it as one
  // float4. Reading the four scalars separately would have every 8th lane land
  // in the same bank at a different address -- a four-way conflict per access.
  const float4 pair_ln_scale =
      *reinterpret_cast<const float4*>(&s_pair_ln_w[kPairPerLane * lane]);
  const float4 pair_ln_shift =
      *reinterpret_cast<const float4*>(&s_pair_ln_b[kPairPerLane * lane]);
  const float pair_scale[kPairPerLane] = {pair_ln_scale.x, pair_ln_scale.y,
                                          pair_ln_scale.z, pair_ln_scale.w};
  const float pair_shift[kPairPerLane] = {pair_ln_shift.x, pair_ln_shift.y,
                                          pair_ln_shift.z, pair_ln_shift.w};

  #pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    const int key = warp + rep * kWarps;
    const __nv_bfloat16* row = pair_block + key * kPairChans;

    float v[kPairPerLane];
    {
      const __nv_bfloat162 lo = load_bf162(row + kPairPerLane * lane);
      const __nv_bfloat162 hi = load_bf162(row + kPairPerLane * lane + 2);
      v[0] = __low2float(lo);
      v[1] = __high2float(lo);
      v[2] = __low2float(hi);
      v[3] = __high2float(hi);
    }

    // Two-pass mean/variance in fp32, biased variance, matching the reference
    // LayerNorm's fp32 reduction over the bfloat16 input.
    float mean = warp_sum(v[0] + v[1] + v[2] + v[3]) * (1.0f / kPairChans);
    float sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < kPairPerLane; ++i) {
      const float d = v[i] - mean;
      sq += d * d;
    }
    const float rstd = rsqrtf(warp_sum(sq) * (1.0f / kPairChans) + pair_eps);

    #pragma unroll
    for (int i = 0; i < kPairPerLane; ++i) {
      v[i] = to_bf16_and_back((v[i] - mean) * rstd * pair_scale[i] + pair_shift[i]);
    }

    // Project onto the per-head pair weight; fp32 accumulation, then rounded to
    // bfloat16 exactly where the reference's linear layer emits bfloat16.
    #pragma unroll
    for (int head = 0; head < kHeads; ++head) {
      const __nv_bfloat16* w = s_pair_proj_w + head * kPairChans + kPairPerLane * lane;
      const __nv_bfloat162 wlo = load_bf162(w);
      const __nv_bfloat162 whi = load_bf162(w + 2);
      float acc = v[0] * __low2float(wlo) + v[1] * __high2float(wlo) +
                  v[2] * __low2float(whi) + v[3] * __high2float(whi);
      acc = warp_sum(acc);
      if (lane == 0) {
        s_row_weights[head * kResidues + key] = to_bf16_and_back(acc);
      }
    }

    // The mask enters as the reference's finite inf*(mask-1) in bfloat16, never
    // as a true -inf: a fully masked row must reduce to uniform weights rather
    // than NaN.
    if (lane == 0 && mask_row != nullptr) {
      const float diff = to_bf16_and_back(__bfloat162float(mask_row[key]) - 1.0f);
      const float bias = to_bf16_and_back(mask_inf * diff);
      #pragma unroll
      for (int head = 0; head < kHeads; ++head) {
        float* slot = &s_row_weights[head * kResidues + key];
        *slot = to_bf16_and_back(*slot + bias);
      }
    }
  }
  __syncthreads();

  // ---- row_softmax over residues, one head per warp ----------------------
  {
    const float x = lane < kResidues ? s_row_weights[warp * kResidues + lane]
                                     : -INFINITY;
    const float peak = warp_max(x);
    const float e = lane < kResidues ? __expf(x - peak) : 0.0f;
    const float total = warp_sum(e);
    if (lane < kResidues) {
      s_row_weights[warp * kResidues + lane] = to_bf16_and_back(e / total);
    }
  }

  // ---- layer_norm_msa ----------------------------------------------------
  // Same ownership as the pair side: two residues per warp, two channels per
  // lane, written out as bfloat16 pairs so consecutive lanes cover all banks.
  // The affine slice is two consecutive floats, read as one float2 for the same
  // reason the pair side reads a float4.
  const int msa_chan = kMsaPerLane * lane;
  const float2 msa_ln_scale = *reinterpret_cast<const float2*>(&s_msa_ln_w[msa_chan]);
  const float2 msa_ln_shift = *reinterpret_cast<const float2*>(&s_msa_ln_b[msa_chan]);

  #pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    const int key = warp + rep * kWarps;
    const __nv_bfloat162 raw = load_bf162(msa_block + key * kMsaChans + msa_chan);
    const float v0 = __low2float(raw);
    const float v1 = __high2float(raw);

    const float mean = warp_sum(v0 + v1) * (1.0f / kMsaChans);
    const float d0 = v0 - mean;
    const float d1 = v1 - mean;
    const float rstd = rsqrtf(warp_sum(d0 * d0 + d1 * d1) * (1.0f / kMsaChans) + msa_eps);

    const __nv_bfloat162 norm = __floats2bfloat162_rn(
        d0 * rstd * msa_ln_scale.x + msa_ln_shift.x,
        d1 * rstd * msa_ln_scale.y + msa_ln_shift.y);
    *reinterpret_cast<__nv_bfloat162*>(&s_msa_norm[key * kMsaChans + msa_chan]) = norm;
  }
  __syncthreads();

  // ---- weighted_mean, then value_project / gate_project ------------------
  // Warp `warp` is head `warp`. It first averages the normalized MSA rows under
  // that head's softmax weights, which is the reassociation that removes the
  // per-residue value projection, then reduces the averaged vector against the
  // head's eight value rows and the same eight gate rows.
  {
    const int head = warp;
    const int chan = kMsaPerLane * lane;
    float u0 = 0.0f;
    float u1 = 0.0f;
    #pragma unroll
    for (int key = 0; key < kResidues; ++key) {
      const float w = s_row_weights[head * kResidues + key];
      const __nv_bfloat162 mv = load_bf162(&s_msa_norm[key * kMsaChans + chan]);
      u0 += w * __low2float(mv);
      u1 += w * __high2float(mv);
    }

    const __nv_bfloat162 query_norm = load_bf162(&s_msa_norm[query * kMsaChans + chan]);
    const float q0 = __low2float(query_norm);
    const float q1 = __high2float(query_norm);

    #pragma unroll
    for (int c = 0; c < kHeadChans; ++c) {
      const int channel = head * kHeadChans + c;
      const __nv_bfloat162 vw = load_bf162(value_w + channel * kMsaChans + chan);
      const __nv_bfloat162 gw = load_bf162(gate_w + channel * kMsaChans + chan);
      float value = u0 * __low2float(vw) + u1 * __high2float(vw);
      float gate = q0 * __low2float(gw) + q1 * __high2float(gw);
      value = warp_sum(value);
      gate = warp_sum(gate);
      if (lane == 0) {
        const float averaged = to_bf16_and_back(value);
        const float logit = to_bf16_and_back(gate);
        const float gated = to_bf16_and_back(1.0f / (1.0f + __expf(-logit)));
        s_gated[channel] = __float2bfloat16(averaged * gated);
      }
    }
  }
  __syncthreads();

  // ---- output_project ---------------------------------------------------
  {
    const __nv_bfloat162 y = load_bf162(&s_gated[kMsaPerLane * lane]);
    const float y0 = __low2float(y);
    const float y1 = __high2float(y);
    #pragma unroll
    for (int t = 0; t < kMsaChans / kWarps; ++t) {
      const int j = warp * (kMsaChans / kWarps) + t;
      const __nv_bfloat162 ow = load_bf162(out_w + j * kMsaChans + kMsaPerLane * lane);
      float acc = y0 * __low2float(ow) + y1 * __high2float(ow);
      acc = warp_sum(acc);
      if (lane == 0) {
        s_out[j] = __float2bfloat16(acc);
      }
    }
  }
  __syncthreads();

  if (threadIdx.x < kMsaChans / kMsaPerLane) {
    __nv_bfloat16* dst =
        out + ((static_cast<long>(batch) * n_seq + seq) * kResidues + query) * kMsaChans +
        kMsaPerLane * threadIdx.x;
    *reinterpret_cast<__nv_bfloat162*>(dst) =
        load_bf162(&s_out[kMsaPerLane * threadIdx.x]);
  }
}

void check_weight(const torch::Tensor& t, const char* name, int64_t rows, int64_t cols,
                  const torch::Device& device) {
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
  TORCH_CHECK(t.device() == device, name, " must live on ", device);
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  if (cols < 0) {
    TORCH_CHECK(t.dim() == 1 && t.size(0) == rows, name, " must be [", rows, "]");
  } else {
    TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols, name,
                " must be [", rows, ", ", cols, "]");
  }
}

const __nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
  return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr<at::BFloat16>());
}

}  // namespace

torch::Tensor msa_pair_weighted_avg(
    torch::Tensor msa,
    torch::Tensor pair,
    std::optional<torch::Tensor> mask,
    torch::Tensor msa_ln_weight,
    torch::Tensor msa_ln_bias,
    torch::Tensor pair_ln_weight,
    torch::Tensor pair_ln_bias,
    torch::Tensor pair_proj_weight,
    torch::Tensor value_weight,
    torch::Tensor gate_weight,
    torch::Tensor out_weight,
    double mask_inf,
    double msa_eps,
    double pair_eps) {
  TORCH_CHECK(msa.is_cuda(), "msa must be a CUDA tensor");
  TORCH_CHECK(pair.is_cuda(), "pair must be a CUDA tensor");
  TORCH_CHECK(msa.scalar_type() == torch::kBFloat16, "msa must be bfloat16");
  TORCH_CHECK(pair.scalar_type() == torch::kBFloat16, "pair must be bfloat16");
  TORCH_CHECK(msa.is_contiguous(), "msa must be contiguous");
  TORCH_CHECK(pair.is_contiguous(), "pair must be contiguous");
  TORCH_CHECK(msa.dim() == 4, "msa must be [B, N_seq, N_res, C_m]");
  TORCH_CHECK(pair.dim() == 4, "pair must be [B, N_res, N_res, C_z]");

  const int64_t batch = pair.size(0);
  const int64_t n_seq = msa.size(1);
  TORCH_CHECK(msa.size(0) == batch, "msa and pair must agree on the batch dim");
  TORCH_CHECK(pair.size(1) == kResidues && pair.size(2) == kResidues,
              "fused kernel is specialized for N_res == ", kResidues);
  TORCH_CHECK(pair.size(3) == kPairChans,
              "fused kernel is specialized for C_z == ", kPairChans);
  TORCH_CHECK(msa.size(2) == kResidues, "msa N_res must be ", kResidues);
  TORCH_CHECK(msa.size(3) == kMsaChans,
              "fused kernel is specialized for C_m == ", kMsaChans);
  TORCH_CHECK(batch > 0 && n_seq > 0, "empty batch or sequence dim");
  // The batch indexes grid.z and the sequence grid.x, so both inherit the
  // driver's per-dimension limits.
  TORCH_CHECK(batch <= 65535, "batch ", batch, " exceeds the grid.z limit of 65535");
  TORCH_CHECK(n_seq <= 2147483647, "N_seq ", n_seq, " exceeds the grid.x limit");

  const torch::Device device = msa.device();
  TORCH_CHECK(pair.device() == device, "msa and pair must share a device");
  check_weight(msa_ln_weight, "layer_norm_m.weight", kMsaChans, -1, device);
  check_weight(msa_ln_bias, "layer_norm_m.bias", kMsaChans, -1, device);
  check_weight(pair_ln_weight, "layer_norm_z.weight", kPairChans, -1, device);
  check_weight(pair_ln_bias, "layer_norm_z.bias", kPairChans, -1, device);
  check_weight(pair_proj_weight, "linear_z.weight", kHeads, kPairChans, device);
  check_weight(value_weight, "linear_v.weight", kMsaChans, kMsaChans, device);
  check_weight(gate_weight, "linear_g.weight", kMsaChans, kMsaChans, device);
  check_weight(out_weight, "linear_o.weight", kMsaChans, kMsaChans, device);

  const __nv_bfloat16* mask_ptr = nullptr;
  if (mask.has_value()) {
    const torch::Tensor& mk = mask.value();
    TORCH_CHECK(mk.scalar_type() == torch::kBFloat16, "mask must be bfloat16");
    TORCH_CHECK(mk.device() == device, "mask must share the input device");
    TORCH_CHECK(mk.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(mk.dim() == 3 && mk.size(0) == batch && mk.size(1) == kResidues &&
                    mk.size(2) == kResidues,
                "mask must be [B, N_res, N_res]");
    mask_ptr = bf16_ptr(mk);
  }

  // Bind to the inputs' device before allocating or launching: without this the
  // launch would go to whatever device happens to be current, which is the wrong
  // one whenever the caller passes tensors from a non-current device.
  const at::cuda::CUDAGuard device_guard(msa.device());

  torch::Tensor out = torch::empty_like(msa);

  const dim3 grid(static_cast<unsigned>(n_seq), static_cast<unsigned>(kResidues),
                  static_cast<unsigned>(batch));
  auto stream = at::cuda::getCurrentCUDAStream();
  msa_pair_weighted_avg_kernel<<<grid, kThreads, 0, stream>>>(
      bf16_ptr(msa), bf16_ptr(pair), mask_ptr, bf16_ptr(msa_ln_weight),
      bf16_ptr(msa_ln_bias), bf16_ptr(pair_ln_weight), bf16_ptr(pair_ln_bias),
      bf16_ptr(pair_proj_weight), bf16_ptr(value_weight), bf16_ptr(gate_weight),
      bf16_ptr(out_weight), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<float>(mask_inf), static_cast<float>(msa_eps),
      static_cast<float>(pair_eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


def _extension_build_directory() -> str:
    """Workspace-local, git-ignored build tree.

    Keeps build products out of the shared ``~/.cache/torch_extensions`` tree so a
    build here neither reads nor is confused by anything another run left behind.

    A failure to create it deliberately propagates. Passing ``None`` to
    ``load_inline`` would silently redirect the build into that shared cache, which
    is the one place a stale build of this very operator is likely to already
    exist; losing the fused path is preferable to compiling somewhere unhygienic,
    and the guarded build turns the error into a reference-path fallback anyway.
    """
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(workspace, ".torch_extensions", _FUSED_EXT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _build_extension():
    """Compile the fused kernel.

    Two environment hazards are handled here. ``TORCH_CUDA_ARCH_LIST`` is exported
    ambiently as a six-architecture list, which would compile six times over and
    emit ``sm_100`` rather than ``sm_100a``; it is pinned for the duration of the
    build and restored afterwards. And the build must announce itself and stream
    ninja's progress, because a silent cold compile looks like a stalled process
    to a watchdog that reads log mtime.
    """
    from torch.utils.cpp_extension import load_inline

    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    print(
        f"[{_FUSED_EXT_NAME}] building fused MSA pair-weighted-averaging kernel "
        f"for arch 10.0a (one-time JIT compile; streaming ninja progress) ...",
        file=sys.stderr,
        flush=True,
    )
    try:
        return load_inline(
            name=_FUSED_EXT_NAME,
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["msa_pair_weighted_avg"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "--expt-relaxed-constexpr",
            ],
            build_directory=_extension_build_directory(),
            verbose=True,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# Built once, at import. Compilation needs no GPU, and deferring it to the first
# call would put a multi-second build (and ninja's worker threads) inside the
# timed region. Any failure leaves the handle unavailable and the reference path
# active rather than breaking import.
_EXT = None
_EXT_ERROR: str | None = None
if os.environ.get("FK_AF3_MSA_DISABLE_FUSED", "") not in ("", "0"):
    _EXT_ERROR = "disabled via FK_AF3_MSA_DISABLE_FUSED"
else:
    try:
        _EXT = _build_extension()
    except Exception as exc:  # noqa: BLE001 - a build failure must not break import
        _EXT_ERROR = f"{type(exc).__name__}: {exc}"
        print(
            f"[{_FUSED_EXT_NAME}] build failed, using the reference path: {_EXT_ERROR}",
            file=sys.stderr,
            flush=True,
        )


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

    The captured configuration is served by a single fused CUDA kernel. Every
    other input -- a different channel count or residue count, a different dtype
    or device, a non-contiguous input, grad enabled, or a missing extension --
    takes the reference path, so the module stays a drop-in replacement.

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

        # Which path served each call, so a run can be audited rather than
        # trusted: a correctness pass obtained through the reference path is not
        # evidence that the fused kernel works. Every return from forward sets
        # last_path, including the early return, so the indicator always describes
        # the call that just happened rather than an earlier one.
        self.fused_calls = 0
        self.reference_calls = 0
        self.early_return_calls = 0
        self.last_path = "none"

        # True only when this instance's shape constants match the fused kernel's
        # compile-time specialization; checked once here instead of per call.
        self._shape_is_fusable = (
            c_m == _FUSED_MSA_CHANS
            and c_z == _FUSED_PAIR_CHANS
            and no_heads == _FUSED_HEADS
            and c_hidden == _FUSED_HEAD_CHANS
        )

    @property
    def fused_available(self) -> bool:
        """Whether the fused extension compiled and loaded."""
        return _EXT is not None

    @property
    def fused_unavailable_reason(self) -> str | None:
        return _EXT_ERROR

    def _fused_applies(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> bool:
        if _EXT is None or not self._shape_is_fusable or torch.is_grad_enabled():
            return False
        n_res = _FUSED_RESIDUES
        if m.dim() != 4 or z.dim() != 4:
            return False
        if m.dtype is not torch.bfloat16 or z.dtype is not torch.bfloat16:
            return False
        if not m.is_cuda or not z.is_cuda or m.device != z.device:
            return False
        if not m.is_contiguous() or not z.is_contiguous():
            return False
        if z.shape[1] != n_res or z.shape[2] != n_res or z.shape[3] != self.c_z:
            return False
        if m.shape[0] != z.shape[0] or m.shape[2] != n_res or m.shape[3] != self.c_m:
            return False
        # An empty batch or sequence has nothing to launch, and a batch beyond the
        # grid.z limit cannot be launched at all; both belong on the reference path
        # rather than raising out of the kernel.
        if not 0 < m.shape[0] <= _MAX_FUSED_BATCH or m.shape[1] <= 0:
            return False
        if not _arch_supported(m.device):
            return False
        if mask is not None and (
            mask.dtype is not torch.bfloat16
            or not mask.is_cuda
            or mask.device != z.device
            or not mask.is_contiguous()
            or mask.dim() != 3
            or mask.shape[0] != z.shape[0]
            or mask.shape[1] != n_res
            or mask.shape[2] != n_res
        ):
            return False
        # The harness shares weights with strict=False, so a parameter that was
        # renamed or reshaped upstream would silently keep its own random values.
        # Validating the tensors the kernel is about to read means a mismatch
        # falls back to the reference path instead of reading a wrong layout.
        device = m.device
        for param, shape in (
            (self.layer_norm_m.weight, (self.c_m,)),
            (self.layer_norm_m.bias, (self.c_m,)),
            (self.layer_norm_z.weight, (self.c_z,)),
            (self.layer_norm_z.bias, (self.c_z,)),
            (self.linear_z.weight, (self.no_heads, self.c_z)),
            (self.linear_v.weight, (self.c_m, self.c_m)),
            (self.linear_g.weight, (self.c_m, self.c_m)),
            (self.linear_o.weight, (self.c_m, self.c_m)),
        ):
            if (
                param is None
                or param.dtype is not torch.bfloat16
                or param.device != device
                or tuple(param.shape) != shape
                or not param.is_contiguous()
            ):
                return False
        if self.layer_norm_m.normalized_shape != (self.c_m,):
            return False
        if self.layer_norm_z.normalized_shape != (self.c_z,):
            return False
        return True

    def _reference_forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # Pair bias: [*, 1, no_heads, N_res, N_res]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        # Value projection
        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)  # [*, N_seq, H, N_res, C_hidden]

        # Weighted average: [*, N_seq, H, N_res, C_hidden]
        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        # Gating
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g

        # Flatten heads and project
        o = o.reshape(o.shape[:-2] + (-1,))
        o = self.linear_o(o)

        return o

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res, N_res] pair mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if z is None:
            self.early_return_calls += 1
            self.last_path = "early-return"
            return m

        if self._fused_applies(m, z, mask):
            self.fused_calls += 1
            self.last_path = "fused"
            return _EXT.msa_pair_weighted_avg(
                m,
                z,
                mask,
                self.layer_norm_m.weight,
                self.layer_norm_m.bias,
                self.layer_norm_z.weight,
                self.layer_norm_z.bias,
                self.linear_z.weight,
                self.linear_v.weight,
                self.linear_g.weight,
                self.linear_o.weight,
                self.inf,
                self.layer_norm_m.eps,
                self.layer_norm_z.eps,
            )

        self.reference_calls += 1
        self.last_path = "reference"
        return self._reference_forward(m, z, mask)
