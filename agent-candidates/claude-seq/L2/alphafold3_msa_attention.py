"""MSA pair-weighted averaging for AlphaFold3 (Algorithm 10).

Weighted averaging over the MSA representation using pair activations,
NOT key-query self-attention.

Reference: openfold3/core/model/layers/msa.py MSAPairWeightedAveraging

Why one kernel
--------------
The captured case is tiny: ``m`` is [1, 8, 16, 64] and ``z`` is [1, 16, 16, 128],
about 2M multiply-accumulates, which is well under a microsecond of arithmetic on
this GPU.  The baseline costs 230-310 us because it issues ~30 eager ops at ~7 us
of dispatch each.  Nothing about the math is worth tuning; the op count is.  So
the whole algorithm -- two LayerNorms, four projections, the masked softmax, the
weighted average, the sigmoid gate and the output projection -- runs as a single
launch behind a single C++ call, and ``forward`` does no tensor work in Python.

Three things make the fusion fit in one block per (sequence, query) pair, with no
inter-block communication and no grid sync:

* **The LayerNorm affines are folded into the weights they feed.**  For the pair
  path, ``zn[k,e] = (z[k,e] - mu_k) * rstd_k * lnw[e] + lnb[e]``, so with
  ``A[e,h] = lnw[e] * Wz[h,e]`` and ``cz[h] = sum_e lnb[e] * Wz[h,e]``,
  ``z_proj[k,h] = rstd_k * (sum_e z[k,e] * A[e,h] - mu_k * sum_e A[e,h]) + cz[h]``.
  A normalized tensor is never materialized -- a raw row plus its two moments is
  enough -- which is what lets the pair row stay in registers.  The MSA path
  folds the same way into ``linear_v`` and ``linear_g``.
* **The weighted average is reassociated through ``linear_v``.**  The baseline
  forms ``v = linear_v(m)`` for every residue and then contracts it with the pair
  weights, which per block is ``N*D*C_m`` = 65536 MACs.  Contracting the other way
  first -- ``u[h,d] = sum_k w[h,k] * mn[k,d]``, then
  ``o[h,c] = sum_d u[h,d] * Wv[h*C_h+c, d]`` -- is ``H*C_m*N + D*C_m`` = 12288, a
  5x cut, because ``C_h`` (8) is much smaller than ``C_m`` (64).  Globally, with
  ``v`` shared across all queries, the baseline order is the cheaper one; per
  block, where ``v`` would be recomputed for every query, it is not.
* **Only the gate needs a single MSA row.**  ``g`` is indexed by the query
  residue, so a block owning one query reads one row of ``mn`` for it, while the
  weighted average needs all of them.

The folded weights are packed once, on the first call, into one fp32 buffer (the
pair projection, which every key reads, plus the four bias vectors) and one bf16
buffer (``linear_v`` / ``linear_g`` / ``linear_o``, each read exactly once per
block).  The pack is rebuilt if the parameters are reloaded or the module is
moved.  Anything the kernel does not cover -- another channel configuration,
N > 32, a non-contiguous input, grad enabled, no GPU, no nvcc -- falls through to
the reference path below, which is the baseline implementation unchanged.

See the CUDA source's own header for why the kernel is shaped the way it is; the
short version is that at this size it is bound by exposed global latency and by
the number of barrier-separated stages, not by arithmetic.
"""

from __future__ import annotations

import os

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


_CUDA_SRC = r"""
// One launch for AF3 MSA pair-weighted averaging (Algorithm 10).
//
// Block (b, s, q) owns one output row out[b, s, q, :].  It needs the pair tile
// z[b, q, :, :] (which gives this query's N softmax weights) and the MSA tile
// m[b, s, :, :] (the values being averaged), and nothing from any other block --
// no grid sync, no scratch buffer, no second kernel.
//
// At this size (~2M multiply-accumulates in total, ~90 per thread) the kernel is
// entirely latency-bound, so its shape is set by two budgets ncu measured, not
// by arithmetic:
//
//   * **Exposed global latency.**  Every block reads all of Wv / Wg / Wo, each
//     element exactly once, and with one block per SM there are only a handful
//     of warps per scheduler to cover the misses -- so what costs time is the
//     number of dependent round trips, not bytes.  Hence: the weights are bf16
//     (half the cache lines); each matrix is *pre-permuted on the host* into the
//     order its consuming thread reads it, so a thread's whole slice is one
//     16-byte load instead of eight strided ones; and every global read --
//     weights, MSA row, pair row, mask -- is issued at the top of the kernel and
//     consumed several stages later, so the latency hides behind real work.
//     (Copying the weights into shared memory instead measured ~60% slower: it
//     replaces that free overlap with a barrier on the whole transfer.)
//   * **Stages.**  Each barrier-separated stage exposes its own latency with
//     nothing to overlap it, so removing a stage beats removing instructions.
//     The pair-row LayerNorm is fused into the projection that consumes it: one
//     warp owns one key row, normalizes it in registers and contracts it against
//     all H heads there, so the row never round-trips through shared memory.
//     What is left is five barriers.
//
// LayerNorm affines are folded into the weights they feed (see the Python
// docstring), so the kernel reads raw rows plus their two moments and never
// materializes a normalized tensor.  Everything accumulates in fp32.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>

namespace {

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void from_f(float &d, float s) { d = s; }
__device__ __forceinline__ void from_f(__half &d, float s) { d = __float2half_rn(s); }
__device__ __forceinline__ void from_f(__nv_bfloat16 &d, float s) { d = __float2bfloat16_rn(s); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// K consecutive bf16 weights -> fp32 registers, as 16-byte loads when the host
// layout allows it (it does for every instantiated configuration).
template <int K>
__device__ __forceinline__ void load_w(float (&dst)[K], const __nv_bfloat16 *src) {
  if constexpr (K % 8 == 0) {
#pragma unroll
    for (int v = 0; v < K / 8; ++v) {
      const uint4 raw = *reinterpret_cast<const uint4 *>(src + v * 8);
      const __nv_bfloat16 *h = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[v * 8 + j] = to_f(h[j]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < K; ++j) dst[j] = to_f(src[j]);
  }
}

// Largest power of two <= n (>= 1).  A stage's partial sums are reduced with a
// lane butterfly, so its width has to divide 32.
constexpr int pow2_le(int n) { return n < 2 ? 1 : 2 * pow2_le(n / 2); }
constexpr int ilog2(int n) { return n < 2 ? 0 : 1 + ilog2(n / 2); }

// How many ways a stage splits its contraction across threads.  Compile-time:
// depends only on the channel dims and the block width, never on N.  A split of
// 1 means one thread per output and no reduction at all.
template <int NT, int W> constexpr int split() {
  return NT / W > 32 ? 32 : (NT / W > 0 ? pow2_le(NT / W) : 1);
}

// fp32 weights: At[H][CZ] (the folded pair projection, transposed so a lane reads
// it as float4), then sumA[H], cz[H], cv[D], cg[D].
template <int CZ, int CM, int H, int CH> constexpr int w32_floats() {
  return CZ * H + 2 * H + 2 * (H * CH);
}
// bf16 weights: WvA[D][CM], WgA[D][CM] and Wo[CM][D] -- each in its natural
// [out][in] order, which is exactly the order the split slots read it.
template <int CZ, int CM, int H, int CH> constexpr int w16_elems() {
  return 3 * CM * (H * CH);
}

// Shared memory a block needs for N residues: the staged pair projection, the
// normalized MSA tile, and the three small per-stage handoffs.  Everything else
// -- the pair row, the value/gate/output weights -- stays in registers.
template <int CZ, int CM, int H, int CH, int NT> constexpr int smem_floats(int N) {
  return (CZ * H + 2 * H) + N * CM + 2 * N * H + H * CM + (H * CH);
}

// ---------------------------------------------------------------------------
// CZ = c_z, CM = c_m, H = no_heads, CH = c_hidden, NT = threads/block.  N
// (residues) and the batch/sequence extents are runtime.  One block per
// (batch, sequence, query).
// ---------------------------------------------------------------------------
template <typename T, int CZ, int CM, int H, int CH, int NT>
__global__ __launch_bounds__(NT) void msa_pwa_kernel(
    const T *__restrict__ mp, const T *__restrict__ zp, const T *__restrict__ maskp,
    const float *__restrict__ w32, const __nv_bfloat16 *__restrict__ w16,
    T *__restrict__ outp, int S, int N, int nqlog, float inf, float eps_m,
    float eps_z) {
  constexpr int D = H * CH;
  constexpr int NW = NT / 32;
  constexpr int MV = CM / 32;               // m elements per lane per row
  constexpr int ZV = CZ / 32;               // z elements per lane per row
  constexpr int PU = split<NT, H * CM>();   // average: ways the N sum splits
  constexpr int PO = split<NT, D>();        // gate/value: ways the C_m sum splits
  constexpr int PE = split<NT, CM>();       // output: ways the D sum splits
  constexpr int SU = NT / PU, SO = NT / PO, SE = NT / PE;
  constexpr int LPO = ilog2(PO), LPE = ilog2(PE);
  constexpr int KV = CM / PO;  // WvA / WgA elements per thread
  constexpr int KO = D / PE;   // WoT elements per thread
  static_assert(CM % 32 == 0 && CZ % 128 == 0, "rows are lane-split, 4 per lane");
  static_assert(NT % 32 == 0 && NT == PO * SO && NT == PE * SE, "whole warps, exact slots");
  static_assert(D <= SO && CM <= SE, "one output per thread in the last two stages");
  static_assert(CM % PO == 0 && D % PE == 0, "even splits");
  static_assert(2 * KV + KO <= 32, "the weight slices have to stay in registers");

  constexpr int offAt = 0;              // in w32: At[H][CZ]
  constexpr int offSumA = offAt + CZ * H;
  constexpr int offCz = offSumA + H;
  constexpr int offCv = offCz + H;
  constexpr int offCg = offCv + D;
  constexpr int offWvA = 0;              // in w16
  constexpr int offWgA = offWvA + CM * D;
  constexpr int offWoT = offWgA + CM * D;
  constexpr int ASH = CZ * H + 2 * H;    // At, sumA and cz: the shared prefix

  extern __shared__ float smem[];
  float *At = smem;                  // [H][CZ] + sumA[H] + cz[H]
  float *ms = At + ASH;              // [N][CM]  (m - mu) * rstd
  float *zpj = ms + N * CM;          // [N][H]   z_proj
  float *zw = zpj + N * H;           // [H][N]   softmax weights
  float *us = zw + H * N;            // [H][CM]  reassociated average
  float *ogs = us + H * CM;          // [D]      gated average, flattened heads

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int bs = blockIdx.x / N;
  const int q = blockIdx.x - bs * N;
  const int b = bs / S;

  // =========================================================================
  // Issue every global read first and consume none of it yet: the whole kernel
  // pays one exposed latency instead of one per stage.
  // =========================================================================
  // The threads sharing an output channel are *adjacent lanes* (part is the low
  // bits of tid), which buys two things: their partial sums reduce with a lane
  // butterfly instead of a trip through shared memory and a barrier, and the
  // slice each one contracts over is contiguous in the weight's natural
  // [out][in] order -- so the load is a single 16-byte load and a warp's loads
  // cover one contiguous 512-byte run.
  const int partO = tid & (PO - 1), hc0 = tid >> LPO;
  const int partE = tid & (PE - 1), e0 = tid >> LPE;
  float wv[KV] = {}, wg[KV] = {}, wo[KO] = {}, cvb = 0.f, cgb = 0.f;
  if (hc0 < D) {
    load_w(wv, w16 + offWvA + hc0 * CM + partO * KV);
    load_w(wg, w16 + offWgA + hc0 * CM + partO * KV);
    if (partO == 0) {  // the bias joins the lane that writes the result
      cvb = w32[offCv + hc0];
      cgb = w32[offCg + hc0];
    }
  }
  if (e0 < CM) load_w(wo, w16 + offWoT + e0 * D + partE * KO);

  // MSA and pair rows, with four (resp. two) contiguous elements per lane so
  // each is a single wide load.  Warp `w` owns rows w, w + NW, ...; with the
  // captured N = 16 and 16 warps that is exactly one row each.
  const T *msrc = mp + ((long)bs * N) * CM + MV * lane;
  const T *zsrc = zp + (((long)b * N + q) * N) * CZ + 4 * lane;
  float mv[MV], zv[ZV];
#pragma unroll
  for (int j = 0; j < MV; ++j) mv[j] = warp < N ? to_f(msrc[warp * CM + j]) : 0.f;
#pragma unroll
  for (int j = 0; j < ZV; ++j)
    zv[j] = warp < N ? to_f(zsrc[warp * CZ + (j / 4) * 128 + (j & 3)]) : 0.f;
  float mbias = 0.f;
  if (maskp != nullptr && lane < N)
    mbias = inf * (to_f(maskp[((long)b * N + q) * N + lane]) - 1.f);

  {
    const float4 *s4 = reinterpret_cast<const float4 *>(w32);
    float4 *d4 = reinterpret_cast<float4 *>(At);
#pragma unroll
    for (int j = 0; j < (ASH / 4 + NT - 1) / NT; ++j) {
      const int i = tid + j * NT;
      if (i < ASH / 4) d4[i] = s4[i];
    }
    if constexpr (ASH % 4)
      for (int i = (ASH & ~3) + tid; i < ASH; i += NT) At[i] = w32[i];
  }

  // =========================================================================
  // MSA tile LayerNorm.  The affine is folded into Wv / Wg, so only the
  // centered, scaled row is published.  This runs before the barrier that
  // publishes At, so the pair projection's weights arrive for free.
  // =========================================================================
  for (int r = warp; r < N; r += NW) {
    if (r != warp) {
#pragma unroll
      for (int j = 0; j < MV; ++j) mv[j] = to_f(msrc[r * CM + j]);
    }
    float sum = 0.f, sq = 0.f;
#pragma unroll
    for (int j = 0; j < MV; ++j) {
      sum += mv[j];
      sq += mv[j] * mv[j];
    }
    sum = warp_sum(sum);
    sq = warp_sum(sq);
    const float mu = sum / CM;
    const float rs = rsqrtf(fmaxf(sq / CM - mu * mu, 0.f) + eps_m);
#pragma unroll
    for (int j = 0; j < MV; ++j) ms[r * CM + MV * lane + j] = (mv[j] - mu) * rs;
  }
  __syncthreads();  // publishes At (and ms)

  // =========================================================================
  // Pair-row LayerNorm fused with the projection: warp `r` normalizes key row r
  // in registers and contracts it against all H heads there, so the row never
  // reaches shared memory and the two stages cost one barrier instead of two.
  //   z_proj[r][h] = rstd_r * (sum_e z[r][e] * A[e][h] - mu_r * sumA[h]) + cz[h]
  // =========================================================================
  for (int r = warp; r < N; r += NW) {
    if (r != warp) {
#pragma unroll
      for (int j = 0; j < ZV; ++j)
        zv[j] = to_f(zsrc[r * CZ + (j / 4) * 128 + (j & 3)]);
    }
    float sum = 0.f, sq = 0.f;
#pragma unroll
    for (int j = 0; j < ZV; ++j) {
      sum += zv[j];
      sq += zv[j] * zv[j];
    }
    sum = warp_sum(sum);
    sq = warp_sum(sq);
    const float mu = sum / CZ;
    const float rs = rsqrtf(fmaxf(sq / CZ - mu * mu, 0.f) + eps_z);

    float acc[H] = {};
#pragma unroll
    for (int g = 0; g < ZV / 4; ++g) {
      const int ev = g * 128 + 4 * lane;
#pragma unroll
      for (int h = 0; h < H; ++h) {
        // At is [H][CZ], so these four weights are one 16-byte shared load and
        // the warp's lanes cover 32 distinct banks.
        const float4 aw = *reinterpret_cast<const float4 *>(At + offAt + h * CZ + ev);
        acc[h] += zv[4 * g] * aw.x + zv[4 * g + 1] * aw.y + zv[4 * g + 2] * aw.z +
                  zv[4 * g + 3] * aw.w;
      }
    }
#pragma unroll
    for (int h = 0; h < H; ++h) acc[h] = warp_sum(acc[h]);
    if (lane == 0) {
#pragma unroll
      for (int h = 0; h < H; ++h)
        zpj[r * H + h] = rs * (acc[h] - mu * At[offSumA + h]) + At[offCz + h];
    }
  }
  __syncthreads();

  // ---- masked softmax over k, one nq-lane group per head --------------------
  {
    const int nq = 1 << nqlog;
    const int h = tid >> nqlog, k = tid & (nq - 1);
    const bool live = (h < H) && (k < N);
    float val = live ? zpj[k * H + h] : -FLT_MAX;
    // mbias was loaded by lane k of every warp, so shuffle it into the lane that
    // needs it rather than re-reading the mask here.
    if (live && maskp != nullptr) val += __shfl_sync(0xffffffffu, mbias, k);
    float mx = val;
    for (int o = nq >> 1; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float ev = live ? __expf(val - mx) : 0.f;
    float sm = ev;
    for (int o = nq >> 1; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
    if (live) zw[h * N + k] = ev / sm;
  }
  __syncthreads();

  // ---- u[h][d] = sum_k zw[h][k] * ms[k][d] (the reassociated average) -------
  {
    const int part = tid & (PU - 1), slot = tid / PU;
    for (int base = 0; base < H * CM; base += SU) {
      const int i = base + slot;
      float a0 = 0.f, a1 = 0.f;
      if (i < H * CM) {
        const int h = i / CM, d = i - h * CM;
        const float *zh = zw + h * N;
        const float *md = ms + d;
        int k = part;
        for (; k + 3 * PU < N; k += 4 * PU) {  // N is runtime, so unrolling by
          a0 += zh[k] * md[k * CM];            // hand is the only ILP available
          a1 += zh[k + PU] * md[(k + PU) * CM];
          a0 += zh[k + 2 * PU] * md[(k + 2 * PU) * CM];
          a1 += zh[k + 3 * PU] * md[(k + 3 * PU) * CM];
        }
        for (; k < N; k += PU) a0 += zh[k] * md[k * CM];
      }
      float acc = a0 + a1;
#pragma unroll
      for (int o = PU >> 1; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
      if (part == 0 && i < H * CM) us[i] = acc;
    }
  }
  __syncthreads();

  // ---- o = u @ Wv, gate = sigmoid(mn[q] @ Wg), og = o * gate ----------------
  {
    float o = cvb, g = cgb;
    if (hc0 < D) {
      const float *uh = us + (hc0 / CH) * CM;
      const float *mq = ms + q * CM;
      const int d0 = partO * KV;
#pragma unroll
      for (int j = 0; j < KV; ++j) {
        o += uh[d0 + j] * wv[j];
        g += mq[d0 + j] * wg[j];
      }
    }
#pragma unroll
    for (int t = PO >> 1; t; t >>= 1) {  // adjacent lanes, so no barrier
      o += __shfl_xor_sync(0xffffffffu, o, t);
      g += __shfl_xor_sync(0xffffffffu, g, t);
    }
    if (partO == 0 && hc0 < D) ogs[hc0] = o * (1.f / (1.f + __expf(-g)));
  }
  __syncthreads();

  // ---- out[b, s, q, :] = og @ Wo^T -----------------------------------------
  {
    float a = 0.f;
    if (e0 < CM) {
      const int h0 = partE * KO;
#pragma unroll
      for (int j = 0; j < KO; ++j) a += ogs[h0 + j] * wo[j];
    }
#pragma unroll
    for (int t = PE >> 1; t; t >>= 1) a += __shfl_xor_sync(0xffffffffu, a, t);
    if (partE == 0 && e0 < CM) from_f(outp[((long)bs * N + q) * CM + e0], a);
  }
}

constexpr int kThreads = FK_MSA_NT;
// The softmax reduces inside one lane group, so a query's N keys must fit in a
// warp; the mask bias is likewise carried one key per lane.
constexpr int kNMax = 32;

}  // namespace

// ---------------------------------------------------------------------------
// A prepared call.  Everything constant -- the two folded weight buffers, their
// device pointers, the head geometry, inf and both epsilons -- is validated and
// bound once, so a call converts three Python arguments instead of ten.
// ---------------------------------------------------------------------------
class MsaPlan {
 public:
  MsaPlan(at::Tensor w32, at::Tensor w16, int64_t no_heads, int64_t c_hidden, double inf,
          double eps_m, double eps_z)
      : w32_(std::move(w32)), w16_(std::move(w16)), heads_((int)no_heads),
        hidden_((int)c_hidden), inf_((float)inf), eps_m_((float)eps_m),
        eps_z_((float)eps_z) {
    TORCH_CHECK(w32_.is_cuda() && w32_.is_contiguous() && w32_.scalar_type() == at::kFloat,
                "fp32 weight buffer must be contiguous CUDA float");
    TORCH_CHECK(w16_.is_cuda() && w16_.is_contiguous() &&
                    w16_.scalar_type() == at::kBFloat16,
                "bf16 weight buffer must be contiguous CUDA bfloat16");
    p32_ = w32_.const_data_ptr<float>();
    p16_ = (const __nv_bfloat16 *)w16_.const_data_ptr<at::BFloat16>();
  }

  // Returns an undefined tensor (None in Python) when the case is not covered,
  // so the caller can fall back without paying for a shape check in Python.
  at::Tensor run(const at::Tensor &m, const at::Tensor &z,
                 const c10::optional<at::Tensor> &mask) const {
    if (m.dim() != 4 || z.dim() != 4 || !m.is_cuda() || !z.is_cuda() ||
        !m.is_contiguous() || !z.is_contiguous())
      return at::Tensor();

    const int B = (int)m.size(0), S = (int)m.size(1), N = (int)m.size(2);
    const int CM = (int)m.size(3), CZ = (int)z.size(3);
    if (z.size(0) != B || z.size(1) != N || z.size(2) != N || N > kNMax || N < 1)
      return at::Tensor();

    const at::Tensor *maskp = nullptr;
    if (mask.has_value() && mask->defined()) {
      const at::Tensor &mk = *mask;
      if (mk.dim() != 3 || mk.size(0) != B || mk.size(1) != N || mk.size(2) != N ||
          !mk.is_contiguous() || mk.scalar_type() != m.scalar_type())
        return at::Tensor();
      maskp = &mk;
    }

    // Softmax group width: a power of two >= N.  Every head needs its own group,
    // and a group may not straddle a warp.
    int nqlog = 0;
    while ((1 << nqlog) < N) ++nqlog;
    if ((heads_ << nqlog) > kThreads) return at::Tensor();

    at::cuda::OptionalCUDAGuard guard(device_of(m));
    // empty_cuda goes straight to the caching allocator; at::empty would walk the
    // dispatcher, which is ~1 us of the few this call is allowed to cost.
    at::Tensor out = at::detail::empty_cuda(m.sizes(), m.scalar_type(), m.device(),
                                            c10::MemoryFormat::Contiguous);

#define FK_CASE(CZ_, CM_, H_, CH_)                                                     \
  if (CZ == CZ_ && CM == CM_ && heads_ == H_ && hidden_ == CH_) {                        \
    bool ok;                                                                            \
    if (m.scalar_type() == at::kBFloat16)                                               \
      ok = launch<__nv_bfloat16, CZ_, CM_, H_, CH_>(m, z, maskp, out, B, S, N, nqlog);   \
    else if (m.scalar_type() == at::kHalf)                                              \
      ok = launch<__half, CZ_, CM_, H_, CH_>(m, z, maskp, out, B, S, N, nqlog);          \
    else if (m.scalar_type() == at::kFloat)                                             \
      ok = launch<float, CZ_, CM_, H_, CH_>(m, z, maskp, out, B, S, N, nqlog);           \
    else                                                                                \
      ok = false;                                                                       \
    return ok ? out : at::Tensor();                                                     \
  }

    FK_CASE(128, 64, 8, 8)  // OpenFold3 MSA stack (the captured configuration)
#undef FK_CASE
    return at::Tensor();
  }

 private:
  template <typename T, int CZ, int CM, int H, int CH>
  bool launch(const at::Tensor &m, const at::Tensor &z, const at::Tensor *mask,
              at::Tensor &out, int B, int S, int N, int nqlog) const {
    if (w32_.numel() != w32_floats<CZ, CM, H, CH>() ||
        w16_.numel() != w16_elems<CZ, CM, H, CH>())
      return false;
    const int shmem = smem_floats<CZ, CM, H, CH, kThreads>(N) * (int)sizeof(float);
    if (shmem > 48 * 1024) return false;  // stays inside the default limit
    msa_pwa_kernel<T, CZ, CM, H, CH, kThreads>
        <<<B * S * N, kThreads, shmem, at::cuda::getCurrentCUDAStream()>>>(
            (const T *)m.const_data_ptr(), (const T *)z.const_data_ptr(),
            mask ? (const T *)mask->const_data_ptr() : nullptr, p32_, p16_,
            (T *)out.mutable_data_ptr(), S, N, nqlog, inf_, eps_m_, eps_z_);
    return true;
  }

  at::Tensor w32_, w16_;  // kept alive for p32_ / p16_
  const float *p32_;
  const __nv_bfloat16 *p16_;
  int heads_, hidden_;
  float inf_, eps_m_, eps_z_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.attr("threads") = pybind11::int_(kThreads);
  pybind11::class_<MsaPlan>(mod, "MsaPlan")
      .def(pybind11::init<at::Tensor, at::Tensor, int64_t, int64_t, double, double,
                          double>())
      .def("__call__", &MsaPlan::run);
}
"""

# Threads per block.  Measured (ncu, clocks locked) on the captured shape: 512
# is 11 us, 1024 is 14 us -- the wider block halves the work per thread but pays
# far more at the five barriers -- and 256 cannot hold the weight slices in
# registers.  With 128 blocks on 148 SMs there is one block per SM either way, so
# this is also what sets warps per scheduler.
_NT = 512


def _no_call(m, z, mask):
    """Stands in for the bound kernel when this module cannot use it, so
    ``forward`` never retries the (failed) setup."""
    return None


_FWD = None
_LOADED = False


def _pin_arch() -> None:
    override = os.environ.get("FK_TORCH_CUDA_ARCH_LIST", "")
    if override.strip():
        os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - no device: leave the ambient list alone
        return
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{'a' if major in (9, 10, 12) else ''}"


def _load() -> None:
    global _FWD, _LOADED
    _LOADED = True
    try:
        from torch.utils.cpp_extension import load_inline

        _pin_arch()
        ext = load_inline(
            name=f"fk_l2_af3_msa_pwa_{_NT}",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                f"-DFK_MSA_NT={_NT}",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=False,
        )
        _FWD = ext.MsaPlan
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the reference path
        _FWD = None


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

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

        if not _LOADED:
            _load()
        # The kernel entry point with the packed weights and every scalar
        # already bound; rebuilt whenever the parameters change (see
        # _pack_invalidate).  A plain instance attribute, so reading it in
        # forward is a dict hit rather than nn.Module.__getattr__, and so it
        # stays out of state_dict.
        self._call = None
        self._register_load_state_dict_pre_hook(self._pack_invalidate)

    # -- packed weights ----------------------------------------------------
    def _pack_invalidate(self, *args, **kwargs) -> None:
        self._call = None

    def _apply(self, *args, **kwargs):  # device / dtype moves
        self._call = None
        return super()._apply(*args, **kwargs)

    def __getstate__(self):
        # The prepared call holds a pybind object and device buffers; drop it so
        # the module stays picklable / deep-copyable.  It is rebuilt on demand.
        state = self.__dict__.copy()
        state["_call"] = None
        return state

    def _build_call(self):
        """Fold both LayerNorm affines into the projections they feed and pack
        everything into one fp32 buffer, output index innermost.

        z_proj[k,h] = rstd_k * (sum_e z[k,e]*A[e,h] - mu_k*sumA[h]) + cz[h]
        with A[e,h] = lnw_z[e]*Wz[h,e], and the analogous fold of lnw_m/lnb_m
        into linear_v / linear_g, so the kernel never materializes a normalized
        tensor.  Wo is transposed so a warp's reads are contiguous.
        """
        wz, wv, wg, wo = (self.linear_z.weight, self.linear_v.weight,
                          self.linear_g.weight, self.linear_o.weight)
        if _FWD is None or any(w is None for w in (wz, wv, wg, wo)) or not wz.is_cuda:
            self._call = _no_call
            return _no_call
        dev = wz.device

        def affine(ln, n):
            w = ln.weight
            b = ln.bias
            w = torch.ones(n, device=dev) if w is None else w.float()
            b = torch.zeros(n, device=dev) if b is None else b.float()
            return w, b

        lnzw, lnzb = affine(self.layer_norm_z, self.c_z)
        lnmw, lnmb = affine(self.layer_norm_m, self.c_m)
        wzf, wvf, wgf = wz.float(), wv.float(), wg.float()
        a = wzf * lnzw                          # [H, C_z], already At's layout
        w32 = torch.cat([
            a.reshape(-1), a.sum(1), (wzf * lnzb).sum(1),
            (wvf * lnmb).sum(1), (wgf * lnmb).sum(1),
        ])
        # Each of these is read once per block and never reused, so the cost is
        # cache lines in flight, not precision -- bf16 halves the lines.  Their
        # natural [out_channel, in_channel] order is already the order the
        # kernel's threads read them in, one 16-byte load each.
        w16 = torch.cat([
            (wvf * lnmw).reshape(-1),
            (wgf * lnmw).reshape(-1),
            wo.float().reshape(-1),
        ]).to(torch.bfloat16)
        self._call = _FWD(w32, w16, self.no_heads, self.c_hidden, self.inf,
                          self.layer_norm_m.eps, self.layer_norm_z.eps)
        return self._call

    # -- reference path ----------------------------------------------------
    def _reference(self, m, z, mask):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)

        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g
        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

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
            return m
        call = self._call
        if call is None:
            call = self._build_call()
        # The kernel produces a leaf tensor, so a training-mode caller has to go
        # down the autograd-capable reference path.
        if not torch.is_grad_enabled():
            out = call(m, z, mask)
            if out is not None:
                return out
        return self._reference(m, z, mask)
