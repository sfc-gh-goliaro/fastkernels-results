"""Sequence-local atom attention for AlphaFold3 -- fused CUDA implementation.

AtomAttentionEncoder (Algorithm 5) and AtomAttentionDecoder (Algorithm 6).

Reference: openfold3/core/model/layers/sequence_local_atom_attention.py

Why this is a launch-count problem
----------------------------------
At the captured shapes (368 atoms, 16 tokens, c_atom=128, a 3-block atom
transformer over 12 windows of 32x128) the whole operator is under a GFLOP, but
the reference expresses it as 866 (decoder) / 1297 (encoder) separate CUDA
kernels.  Measured on this GPU that is 3.5-4.8 ms of GPU time against ~25 us of
actual arithmetic: the operator is pure launch overhead, and an empty kernel
launch is ~3 us of it.  So the only lever that matters is *how many kernels*.

What the fused path does
------------------------
The captured shapes are driven by one C++ entry point per module, which launches
20 (decoder) / 22-24 (encoder) kernels -- every one of them a fused chain of what
were dozens of eager ops.  The structural wins, beyond plain elementwise fusion:

* **All AdaLN conditioning is one GEMM.**  Every AdaLN gate/shift and every
  AdaLN-Zero output gate is a function of the conditioning tensor ``s`` alone, so
  none of it depends on the residual stream.  The 21 eager [368,128]x[128,128]
  projections spread across the 3 blocks collapse into a single
  [368,128]x[128,3072], computed once before the stack runs.
* **The window gather moves behind the projections.**  The reference
  materializes a [12,128,128] key copy of the [368,128] residual stream -- a 4x
  duplication -- and then projects it.  Because LayerNorm, AdaLN and the K/V
  projections are all row-local, the fused path projects each *atom* once and
  gathers the projected K/V inside the attention kernel: a quarter of the work.
* **The pair bias is computed once for the stack.**  ``layer_norm_z(plm)`` plus
  all three blocks' ``linear_z`` is a single kernel over the pair tensor, so the
  1.5 MB [12,32,128,16] tensor is read once instead of once per block.
* **Attention stays in shared memory.**  32 queries x 128 keys x 32 channels per
  (window, head) fits, so the [12,4,32,128] score tensor is never written -- the
  reference writes and re-reads it three times per block (bmm, +bias, softmax).
* **The encoder's pair stack is one kernel.**  Offsets, inverse square
  distances, the valid-space-uid mask, the trunk pair projection, the
  single-rep outer sum and the whole 3-layer pair MLP are fused into one pass
  over the 786 k pair elements, which also emits the attention bias.

Numerics
--------
The reference materializes every intermediate as bfloat16, so the fused kernels
round to bf16 at exactly those boundaries (``rb`` in the CUDA source) rather than
carrying fp32 through.  LayerNorm reduces in fp32 (the ``promote_fp32`` path),
GEMMs accumulate in fp32 (what cuBLAS does), and softmax / silu / sigmoid use an
fp32 opmath with a bf16 result (torch's ``opmath_t`` convention).  Even the index
arithmetic is replayed bit-exactly: ``_get_block_key_indices`` promotes its int32
window indices to bfloat16 via ``n_real``, which rounds every index above 256 to
an even value and lets ``index == n_real`` appear (then flagged invalid).  The
mathematically intended indices would gather different atoms.

Because the problem cannot fill a B200 -- the whole atom transformer is ~0.25
GMAC -- every kernel here is latency-bound rather than throughput-bound: its cost
is (instructions per warp) x (stall cycles) / (warps in flight), and ncu confirms
~15 stall cycles per issued instruction at 2 warps per scheduler.  That drives
three choices that would otherwise look odd: all memory access is vectorized to
16 bytes per thread, grids are sized to put exactly one wmma output tile in every
warp (splitting K where the output is too narrow to supply enough tiles), and the
AdaLN conditioning and layer_norm_s are precomputed rather than recomputed inside
each of the kernels that consume them.

Scope
-----
The fused path claims only the configuration the captures use -- bf16 on CUDA,
c_atom=128, c_atom_pair=16, 4 heads of 32, n_query=32, n_key=128,
n_transition=2, AdaLN on, unit batch, and a standard ``DiffusionTransformer``
stack whose nine ``layer_norm_s`` weights agree (which is what lets the AdaLN
GEMM share one normalized operand).  Anything else -- other dims, fp32, a
custom ``transformer_cls``, no CUDA, autograd enabled -- falls through to the
reference implementation below, which is kept intact.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from ..L1.relu import ReLU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import Pad


# ---------------------------------------------------------------------------
# The fused CUDA extension.  Source is inlined so this file is the whole
# deliverable; it is JIT-compiled once per machine and cached by
# torch.utils.cpp_extension.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
// Fused sequence-local atom attention (AlphaFold3 Algorithms 5 & 6).
//
// The baseline runs 866 (decoder) / 1297 (encoder) separate CUDA kernels per
// forward for a problem whose whole arithmetic budget is ~0.25 GMAC: 368 atoms,
// 128 channels, a 3-block transformer over 12 windows of 32x128.  Measured on
// this GPU that is 3.5-4.8 ms of GPU time, i.e. the operator is pure overhead.
// Everything below optimizes for launch count first (the whole operator is one
// C++ call, ~20 kernels) and then for the only thing that matters once the
// kernels are fused: this problem cannot fill a B200, so each kernel is
// latency-bound, and its cost is (instructions per warp) x (stall cycles) /
// (warps per scheduler).  Hence the vectorized memory access everywhere (one
// 16-byte transaction per thread rather than eight 2-byte ones) and grids sized
// to put one wmma output tile in every warp.
//
// Numerics follow the reference at every tensor boundary: it materializes each
// intermediate as bfloat16, so each fused kernel rounds to bf16 exactly where an
// eager tensor would have been written (helper `rb`).  LayerNorm reduces in fp32
// (what promote_fp32 does), GEMMs accumulate in fp32 (what cuBLAS does), and
// softmax / silu / sigmoid use an fp32 opmath with a bf16 result (torch's
// opmath_t convention).  Even the index arithmetic is replayed bit-exactly --
// `_get_block_key_indices` promotes its int32 window indices to bfloat16 via
// `n_real`, which rounds every index above 256 to an even value; the
// mathematically intended indices would gather different atoms.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>

namespace {

namespace wm = nvcuda::wmma;
using bf = __nv_bfloat16;

constexpr int CDIM = 128;   // c_atom
constexpr int CZ = 16;      // c_atom_pair
constexpr int NH = 4;       // no_heads
constexpr int HD = 32;      // c_hidden
constexpr int NQ = 32;      // n_query
constexpr int NKY = 128;    // n_key
constexpr int CH = 256;     // n_transition * c_atom
constexpr int NGRP_N = 6;   // AdaLN groups fed from layer_norm_s(s)
constexpr int NGRP_R = 2;   // AdaLN-Zero output gates fed from raw s
constexpr float LN_EPS = 1e-5f;
constexpr float QSCALE = 0.176776695296637f;  // 1/sqrt(c_hidden)

__device__ __forceinline__ float g2f(bf v) { return __bfloat162float(v); }
__device__ __forceinline__ bf f2b(float v) { return __float2bfloat16(v); }
// Round an fp32 value through bfloat16, i.e. materialize it the way the eager
// reference would have.  Every fused step ends on one of these.
__device__ __forceinline__ float rb(float v) {
  return __bfloat162float(__float2bfloat16(v));
}
// Fast-path transcendentals.  The reference computes silu / sigmoid / softmax in
// an fp32 opmath and rounds the result to bf16, so only the first 8 mantissa bits
// survive; __expf's ~2 ulp is 7 orders of magnitude below that, while `expf`'s
// range reduction costs ~12 instructions against 2.  On kernels that evaluate 16
// sigmoids per thread this was a fifth of the instruction stream.
__device__ __forceinline__ float fexp(float x) { return __expf(x); }
__device__ __forceinline__ float sigf(float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

// 4- and 8-wide bf16 vectors: one 8- or 16-byte transaction instead of 4 or 8
// scalar ones.  Every buffer here has a leading dimension that is a multiple of
// 8 elements, so these are always aligned.
union U4 {
  uint2 u;
  bf h[4];
};
union U8 {
  uint4 u;
  bf h[8];
};
__device__ __forceinline__ U4 ld4(const bf *p) {
  U4 v;
  v.u = *reinterpret_cast<const uint2 *>(p);
  return v;
}
__device__ __forceinline__ void st4(bf *p, const U4 &v) {
  *reinterpret_cast<uint2 *>(p) = v.u;
}
__device__ __forceinline__ U8 ld8(const bf *p) {
  U8 v;
  v.u = *reinterpret_cast<const uint4 *>(p);
  return v;
}
__device__ __forceinline__ void st8(bf *p, const U8 &v) {
  *reinterpret_cast<uint4 *>(p) = v.u;
}

__device__ __forceinline__ void warp_sum(float &a) {
#pragma unroll
  for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
}
__device__ __forceinline__ void warp_max(float &a) {
#pragma unroll
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
}

// ---------------------------------------------------------------------------
// C[BM][BN] = A[BM][KK] * W[n0 .. n0+BN][KK]^T   (W row-major: a weight matrix)
//
// A and the fp32 accumulator live in shared memory so each kernel can run an
// arbitrary fused epilogue -- and, in k_oproj, feed a result straight back in as
// the next GEMM's A instead of round-tripping it to HBM.  wmma's col_major
// matrix_b view of a row-major [N][K] weight is exactly W^T, no transpose pass.
//
// Two details, each worth several x on these shapes (both measured with ncu):
//
// * **The weight tile is staged in shared memory.**  Loading matrix_b fragments
//   straight from global looks natural and is a trap: the accumulator chain
//   serializes the fragment loads, so the kernel pays ~700 cycles of latency per
//   k-step (78 k cycles elapsed against 5.8 k of SM-active work, 68% of all
//   stalls on L1TEX scoreboard).  A cooperative 16-byte-per-thread staging pass
//   has all its loads in flight at once, and the fragment loads that follow hit
//   shared memory at ~30 cycles.
// * **Padded shared leading dimensions.**  Unpadded, an ldmatrix's 8 lanes per
//   phase all land on the same banks (row stride a multiple of 128 B) -- an
//   8-way conflict on every fragment load, which is most of the instruction
//   stream.  The pads are chosen so consecutive lanes step exactly 4 banks: +8
//   elements for row-major A, +16 for col_major B (2 lanes per column, so the
//   row step must be 8 banks), +8 floats for the accumulator store.
// ---------------------------------------------------------------------------
constexpr int PADA = 8;   // bf16 row-major A
constexpr int PADB = 16;  // bf16 col_major B (staged weight tile)
constexpr int PADC = 8;   // fp32 accumulator

template <int BM, int BN, int KK, int KC, int NW, int KSPLIT = 1>
__device__ __forceinline__ void gemm_bt(const bf *__restrict__ As,
                                        const bf *__restrict__ W, int n0,
                                        bf *__restrict__ Wsh, float *__restrict__ Cs,
                                        int tid, int warp) {
  constexpr int LDA = KK + PADA, LDB = KC + PADB, LDC = BN + PADC;
  constexpr int MT = BM / 16, NT = BN / 16, TOT = MT * NT;
  constexpr int KSL = KK / KSPLIT;
  static_assert(NW == TOT * KSPLIT, "one output tile per warp per K slice");
  static_assert(KK % KSPLIT == 0 && KSL % KC == 0 && KC % 16 == 0, "K tiling");
  constexpr int NVEC = (BN * KC) / 8;
  constexpr int ROWV = KC / 8;
  constexpr int SNT = TOT * 32;  // threads cooperating on one slice's staging

  const int t = warp % TOT, sl = warp / TOT;
  const int mt = t / NT, nt = t - mt * NT;
  const int stid = tid - sl * SNT;
  bf *Ws = Wsh + (size_t)sl * BN * LDB;
  float *Cd = Cs + (size_t)sl * BM * LDC;

  wm::fragment<wm::accumulator, 16, 16, 16, float> c;
  wm::fill_fragment(c, 0.0f);
  for (int kc = sl * KSL; kc < (sl + 1) * KSL; kc += KC) {
    __syncthreads();
    for (int i = stid; i < NVEC; i += SNT) {
      const int row = i / ROWV, col = (i - row * ROWV) * 8;
      *reinterpret_cast<uint4 *>(Ws + row * LDB + col) =
          *reinterpret_cast<const uint4 *>(W + (size_t)(n0 + row) * KK + kc + col);
    }
    __syncthreads();
#pragma unroll
    for (int k0 = 0; k0 < KC; k0 += 16) {
      wm::fragment<wm::matrix_a, 16, 16, 16, bf, wm::row_major> af;
      wm::fragment<wm::matrix_b, 16, 16, 16, bf, wm::col_major> bfr;
      wm::load_matrix_sync(af, As + mt * 16 * LDA + kc + k0, LDA);
      wm::load_matrix_sync(bfr, Ws + (size_t)nt * 16 * LDB + k0, LDB);
      wm::mma_sync(c, af, bfr, c);
    }
  }
  wm::store_matrix_sync(Cd + mt * 16 * LDC + nt * 16, c, LDC, wm::mem_row_major);
}

// Sum the K-slice partials.  Splitting K is what keeps the narrow kernels (whose
// output is only [368,128] = 184 wmma tiles) from running out of warps: this
// problem is small enough that a kernel's cost is total-warp-instructions over
// warps-in-flight, so 4 slices of K is a straight 4x.
template <int KSPLIT, int BM, int LDC>
__device__ __forceinline__ float csum(const float *Cs, int r, int c) {
  float v = Cs[r * LDC + c];
#pragma unroll
  for (int s = 1; s < KSPLIT; s++) v += Cs[(size_t)s * BM * LDC + r * LDC + c];
  return v;
}

// Shared-arena sizes for one gemm_bt instantiation.
template <int BM, int BN, int KK, int KC, int KSPLIT = 1>
struct GemmSh {
  static constexpr int A = BM * (KK + PADA) * (int)sizeof(bf);
  static constexpr int C = KSPLIT * BM * (BN + PADC) * (int)sizeof(float);
  static constexpr int B = KSPLIT * BN * (KC + PADB) * (int)sizeof(bf);
  static constexpr int TOTAL = A + C + B;
};

// One warp LayerNorms one row of KK channels (KK a multiple of 128), fp32
// reduction, optional affine, bf16 result -- the promote_fp32 path.  4 channels
// per lane per chunk, so the row costs KK/128 vector loads instead of KK/32
// scalar ones.
template <int KK>
__device__ __forceinline__ void warp_ln(const bf *__restrict__ x,
                                        const bf *__restrict__ w, bf *out,
                                        int lane) {
  constexpr int NC = KK / 128;
  U4 v[NC];
  float sum = 0.f;
#pragma unroll
  for (int t = 0; t < NC; t++) {
    v[t] = ld4(x + t * 128 + lane * 4);
#pragma unroll
    for (int e = 0; e < 4; e++) sum += g2f(v[t].h[e]);
  }
  warp_sum(sum);
  const float m = sum * (1.0f / KK);
  float sq = 0.f;
#pragma unroll
  for (int t = 0; t < NC; t++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float d = g2f(v[t].h[e]) - m;
      sq += d * d;
    }
  warp_sum(sq);
  const float r = rsqrtf(sq * (1.0f / KK) + LN_EPS);
#pragma unroll
  for (int t = 0; t < NC; t++) {
    U4 o;
    const U4 wv = w ? ld4(w + t * 128 + lane * 4) : U4{};
#pragma unroll
    for (int e = 0; e < 4; e++) {
      float y = (g2f(v[t].h[e]) - m) * r;
      if (w) y *= g2f(wv.h[e]);
      o.h[e] = f2b(y);
    }
    st4(out + t * 128 + lane * 4, o);
  }
}

// LayerNorm + AdaLN of one 128-channel row: g * (layer_norm_a(a) + linear_s(s)),
// with the gate and shift already projected into `sp`.  bf16 rounding at both
// the norm output and the product, as the reference has them.
__device__ __forceinline__ void warp_adaln(const bf *__restrict__ ar,
                                           const bf *__restrict__ gate,
                                           const bf *__restrict__ shift,
                                           bf *out, int lane) {
  const U4 v = ld4(ar + lane * 4);
  float sum = 0.f;
#pragma unroll
  for (int e = 0; e < 4; e++) sum += g2f(v.h[e]);
  warp_sum(sum);
  const float m = sum * (1.0f / CDIM);
  float sq = 0.f;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const float d = g2f(v.h[e]) - m;
    sq += d * d;
  }
  warp_sum(sq);
  const float rs = rsqrtf(sq * (1.0f / CDIM) + LN_EPS);
  const U4 gv = ld4(gate + lane * 4), sv = ld4(shift + lane * 4);
  U4 o;
#pragma unroll
  for (int e = 0; e < 4; e++)
    o.h[e] = f2b(rb(g2f(gv.h[e]) * rb(rb((g2f(v.h[e]) - m) * rs) + g2f(sv.h[e]))));
  st4(out + lane * 4, o);
}

// ---------------------------------------------------------------------------
// Window-key index table.
//
// Replays `_get_block_key_indices` exactly, including its accidental bfloat16
// promotion: `total_shift` comes out of a bf16 `n_real`, so `initial +
// total_shift` is evaluated in bf16 and every index above 256 snaps to an even
// value (and index == n_real appears, which the reference then marks invalid).
//
// Three outputs, all tiny: `kidx` clamped gather index, `kgeo` 1 where the
// window slot is a real slot (the reference's ~invalid_mask, which is what
// zeroes the gathered activations), `kvalid` = kgeo * atom_mask[kidx] (which is
// what enters the attention bias).  `qvalid` is the query-side mask, padded.
// ---------------------------------------------------------------------------
__global__ void k_index(const bf *__restrict__ mask, int N, int NB,
                        int *__restrict__ kidx, float *__restrict__ kgeo,
                        float *__restrict__ kvalid, float *__restrict__ qvalid) {
  __shared__ float red[32];
  __shared__ float s_nr, s_nm1, s_hi;
  const int tid = threadIdx.x;
  float acc = 0.f;
  for (int i = tid; i < N; i += blockDim.x) acc += g2f(mask[i]);
  const int lane = tid & 31, w = tid >> 5;
  warp_sum(acc);
  if (lane == 0) red[w] = acc;
  __syncthreads();
  if (tid == 0) {
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
    const float nr = rb(t);  // atom_mask.sum(-1), a bf16 tensor
    s_nr = nr;
    s_nm1 = rb(nr - 1.0f);   // (n_real - 1), still bf16
    s_hi = fmaxf(s_nm1, 0.f);
  }
  __syncthreads();

  const int Np = NB * NQ;
  for (int i = tid; i < Np; i += blockDim.x)
    qvalid[i] = (i < N) ? g2f(mask[i]) : 0.f;

  for (int t = tid; t < NB * NKY; t += blockDim.x) {
    const int b = t / NKY, j = t - b * NKY;
    const int center = NQ / 2 + b * NQ;
    const int init0 = center - NKY / 2;
    const int initL = center + NKY / 2 - 1;
    const int under = init0 < 0 ? -init0 : 0;
    float ovf = rb(rb((float)initL) - s_nm1);
    ovf = fmaxf(ovf, 0.f);
    const float shift = (under > 0) ? rb((float)under) : rb(-ovf);
    const float fin = rb(rb((float)(init0 + j)) + shift);
    const bool inv = (fin < 0.f) || (fin >= s_nr);
    const int idx = (int)fminf(fmaxf(fin, 0.f), s_hi);
    kidx[t] = idx;
    kgeo[t] = inv ? 0.f : 1.f;
    kvalid[t] = inv ? 0.f : g2f(mask[idx]);
  }
}

// ---------------------------------------------------------------------------
// layer_norm_z(plm) folded with all three blocks' linear_z: writes the attention
// bias directly, so the 1.5 MB [12,32,128,16] pair tensor is read once for the
// whole stack instead of once per transformer block.
// ---------------------------------------------------------------------------
__global__ void k_zbias(const bf *__restrict__ plm, const bf *__restrict__ wlnz,
                        const bf *__restrict__ Wz, int nblk, int NB,
                        bf *__restrict__ zbias) {
  extern __shared__ float smz[];
  float *swz = smz;                   // [nblk*NH*CZ]
  float *swl = smz + nblk * NH * CZ;  // [CZ]
  for (int i = threadIdx.x; i < nblk * NH * CZ; i += blockDim.x) swz[i] = g2f(Wz[i]);
  for (int i = threadIdx.x; i < CZ; i += blockDim.x) swl[i] = g2f(wlnz[i]);
  __syncthreads();

  const int bq = blockIdx.x;
  const int b = bq / NQ, q = bq - b * NQ;
  const int k = threadIdx.x;
  const bf *p = plm + ((size_t)bq * NKY + k) * CZ;
  U8 lo = ld8(p), hi = ld8(p + 8);
  float x[CZ];
  float m = 0.f;
#pragma unroll
  for (int c = 0; c < 8; c++) {
    x[c] = g2f(lo.h[c]);
    x[c + 8] = g2f(hi.h[c]);
    m += x[c] + x[c + 8];
  }
  m *= 1.0f / CZ;
  float v = 0.f;
#pragma unroll
  for (int c = 0; c < CZ; c++) {
    const float d = x[c] - m;
    v += d * d;
  }
  const float r = rsqrtf(v * (1.0f / CZ) + LN_EPS);
#pragma unroll
  for (int c = 0; c < CZ; c++) x[c] = rb((x[c] - m) * r * swl[c]);
  for (int i = 0; i < nblk; i++)
#pragma unroll
    for (int h = 0; h < NH; h++) {
      float a = 0.f;
#pragma unroll
      for (int c = 0; c < CZ; c++) a += x[c] * swz[(i * NH + h) * CZ + c];
      zbias[((((size_t)i * NB + b) * NH + h) * NQ + q) * NKY + k] = f2b(a);
    }
}

// ---------------------------------------------------------------------------
// All AdaLN conditioning for the whole stack, in one GEMM.
//
// Every AdaLN gate/shift and every AdaLN-Zero output gate is a function of `s`
// alone, so none of it depends on the residual stream: the 21 eager
// [368,128]x[128,128] projections spread across the 3 transformer blocks
// collapse into a single [368,128]x[128,3072].  The six per-block groups that
// consume layer_norm_s(s) share one normalized operand (the reference's nine
// layer_norm_s weights are checked identical on the host, which is what makes
// that legal); the two that consume raw `s` take the unnormalized one.
// `sig`/`bias` carry the per-column epilogue so the packing stays declarative.
//
// layer_norm_s(s) arrives precomputed (from whichever kernel produced `s`):
// recomputing it here would run it once per column tile, i.e. 48 times over, and
// that redundancy was two thirds of this kernel.
// ---------------------------------------------------------------------------
template <int BM, int BN, int NW>
__global__ void k_sproj(const bf *__restrict__ s, const bf *__restrict__ snorm,
                        int N, const bf *__restrict__ W,
                        const float *__restrict__ bias,
                        const unsigned char *__restrict__ sig, int NCOL, int SPLIT,
                        bf *__restrict__ out) {
  using SH = GemmSh<BM, BN, CDIM, CDIM>;
  constexpr int LDA = CDIM + PADA, LDC = BN + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;
  const bool raw = (n0 >= SPLIT);

  const bf *src = raw ? s : snorm;
  for (int r = warp; r < BM; r += NW) {
    const int j = m0 + r;
    st4(As + r * LDA + lane * 4,
        (j < N) ? ld4(src + (size_t)j * CDIM + lane * 4) : U4{});
  }
  gemm_bt<BM, BN, CDIM, CDIM, NW>(As, W, n0, Wsh, Cs, tid, warp);
  __syncthreads();
  // 8 output columns per thread: one 16-byte store per thread.
  for (int i = tid * 8; i < BM * BN; i += NW * 32 * 8) {
    const int r = i / BN, c = i - r * BN;
    const int j = m0 + r;
    if (j >= N) continue;
    const int col = n0 + c;
    U8 o;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float v = rb(Cs[r * LDC + c + e] + bias[col + e]);
      if (sig[col + e]) v = rb(sigf(v));
      o.h[e] = f2b(v);
    }
    st8(out + (size_t)j * NCOL + col, o);
  }
}

// ---------------------------------------------------------------------------
// Per-transformer-block stage 1: layer_norm_a -> the two AdaLNs -> q/gate/k/v.
//
// The reference materializes blocked copies of the residual stream ([12,32,128]
// queries and [12,128,128] keys, the latter a 4x duplication of the same 368
// rows) and projects both.  Every step here is row-local, so the window gather
// is deferred to the attention kernel and all four projections run once per
// *atom* against a single concatenated [512,128] weight -- a quarter of the
// reference's K/V work.
// ---------------------------------------------------------------------------
template <int BM, int BN, int NW>
__global__ void k_qkvg(const bf *__restrict__ a, int N, const bf *__restrict__ sp,
                       int NCOL, int off_qg, int off_qs, int off_kg, int off_ks,
                       const bf *__restrict__ W, const float *__restrict__ bq,
                       bf *__restrict__ qout, bf *__restrict__ gout,
                       bf *__restrict__ kout, bf *__restrict__ vout) {
  using SH = GemmSh<BM, BN, CDIM, CDIM>;
  constexpr int LDA = CDIM + PADA, LDC = BN + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;
  const bool qside = (n0 < 2 * CDIM);
  const int off_g = qside ? off_qg : off_kg;
  const int off_s = qside ? off_qs : off_ks;

  for (int r = warp; r < BM; r += NW) {
    const int j = m0 + r;
    if (j >= N) {
      st4(As + r * LDA + lane * 4, U4{});
      continue;
    }
    const bf *spr = sp + (size_t)j * NCOL;
    warp_adaln(a + (size_t)j * CDIM, spr + off_g, spr + off_s, As + r * LDA, lane);
  }
  gemm_bt<BM, BN, CDIM, CDIM, NW>(As, W, n0, Wsh, Cs, tid, warp);
  __syncthreads();
  for (int i = tid * 8; i < BM * BN; i += NW * 32 * 8) {
    const int r = i / BN, c = i - r * BN;
    const int j = m0 + r;
    if (j >= N) continue;
    const int col = n0 + c;
    U8 o;
    if (col < CDIM) {
      // linear_q (with bias), then the 1/sqrt(c_hidden) scale -- two bf16 steps
#pragma unroll
      for (int e = 0; e < 8; e++)
        o.h[e] = f2b(rb(Cs[r * LDC + c + e] + bq[col + e]) * QSCALE);
      st8(qout + (size_t)j * CDIM + col, o);
    } else if (col < 2 * CDIM) {
#pragma unroll
      for (int e = 0; e < 8; e++) o.h[e] = f2b(sigf(rb(Cs[r * LDC + c + e])));
      st8(gout + (size_t)j * CDIM + col - CDIM, o);
    } else {
#pragma unroll
      for (int e = 0; e < 8; e++) o.h[e] = f2b(Cs[r * LDC + c + e]);
      st8((col < 3 * CDIM ? kout : vout) + (size_t)j * CDIM +
              (col < 3 * CDIM ? col - 2 * CDIM : col - 3 * CDIM),
          o);
    }
  }
}

// ---------------------------------------------------------------------------
// Per-transformer-block stage 2: sequence-local attention, one block per
// (window, head).
//
// 32 queries x 128 keys x 32 channels fits in shared memory, so the score matrix
// never reaches HBM -- the reference writes and re-reads a [1,12,4,32,128] score
// tensor three times per block (bmm, +bias, softmax).  Both matmuls run on
// tensor cores against shared operands: the gathered keys are staged as a
// col_major B tile directly (no transpose pass), and V is staged transposed so
// the context matmul is the same shape of problem.  The key/value gather happens
// here, on the already-projected K/V, so the 4x window overlap costs loads
// rather than FLOPs.
// ---------------------------------------------------------------------------
template <int NW>
__global__ void k_attn(const bf *__restrict__ qin, const bf *__restrict__ kin,
                       const bf *__restrict__ vin, const bf *__restrict__ gin,
                       const bf *__restrict__ zbias, const int *__restrict__ kidx,
                       const float *__restrict__ kgeo,
                       const float *__restrict__ kvalid,
                       const float *__restrict__ qvalid, int N,
                       bf *__restrict__ opre) {
  constexpr int LDQ = HD + PADA;    // [32][40]  bf16, matrix_a of the score GEMM
  constexpr int LDK = HD + PADB;    // [128][48] bf16, matrix_b (gathered keys)
  constexpr int LDV = NKY + PADB;   // [32][144] bf16, matrix_b (V, transposed)
  constexpr int LDS = NKY + PADC;   // [32][136] fp32, scores
  constexpr int LDP = NKY + PADA;   // [32][136] bf16, matrix_a of the context GEMM
  constexpr int LDO = HD + PADC;    // [32][40]  fp32, context
  extern __shared__ __align__(16) char smem[];
  bf *sAq = reinterpret_cast<bf *>(smem);
  bf *sBk = sAq + NQ * LDQ;
  bf *sBv = sBk + NKY * LDK;
  bf *sAp = sBv + HD * LDV;
  float *sC = reinterpret_cast<float *>(sAp + NQ * LDP);
  float *sO = sC + NQ * LDS;
  float *skv = sO + 4 * NQ * LDO;
  float *sqv = skv + NKY;
  int *sidx = reinterpret_cast<int *>(sqv + NQ);

  const int b = blockIdx.x, h = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  for (int i = tid; i < NKY; i += NW * 32) {
    sidx[i] = kidx[b * NKY + i];
    skv[i] = kvalid[b * NKY + i];
  }
  for (int i = tid; i < NQ; i += NW * 32) sqv[i] = qvalid[b * NQ + i];
  __syncthreads();
  for (int i = tid * 4; i < NQ * HD; i += NW * 32 * 4) {
    const int q = i / HD, c = i - q * HD;
    const int j = b * NQ + q;
    st4(sAq + q * LDQ + c, (j < N) ? ld4(qin + (size_t)j * CDIM + h * HD + c) : U4{});
  }
  for (int i = tid * 4; i < NKY * HD; i += NW * 32 * 4) {
    const int kk = i / HD, c = i - kk * HD;
    const bool ok = kgeo[b * NKY + kk] > 0.f;
    const int m = sidx[kk];
    const U4 kv = ok ? ld4(kin + (size_t)m * CDIM + h * HD + c) : U4{};
    const U4 vv = ok ? ld4(vin + (size_t)m * CDIM + h * HD + c) : U4{};
    st4(sBk + kk * LDK + c, kv);
#pragma unroll
    for (int e = 0; e < 4; e++) sBv[(c + e) * LDV + kk] = vv.h[e];
  }
  __syncthreads();

  // scores = Q K^T : [32,32] x [128,32]^T, one 16x16 tile per warp
  if (warp < (NQ / 16) * (NKY / 16)) {
    const int mt = warp / (NKY / 16), nt = warp - mt * (NKY / 16);
    wm::fragment<wm::accumulator, 16, 16, 16, float> c;
    wm::fill_fragment(c, 0.0f);
#pragma unroll
    for (int k0 = 0; k0 < HD; k0 += 16) {
      wm::fragment<wm::matrix_a, 16, 16, 16, bf, wm::row_major> af;
      wm::fragment<wm::matrix_b, 16, 16, 16, bf, wm::col_major> bfr;
      wm::load_matrix_sync(af, sAq + mt * 16 * LDQ + k0, LDQ);
      wm::load_matrix_sync(bfr, sBk + nt * 16 * LDK + k0, LDK);
      wm::mma_sync(c, af, bfr, c);
    }
    wm::store_matrix_sync(sC + mt * 16 * LDS + nt * 16, c, LDS, wm::mem_row_major);
  }
  __syncthreads();

  // + mask bias + pair bias, then softmax: one warp per query row, 4 keys/lane
  const bf *zb = zbias + ((size_t)(b * NH + h) * NQ) * NKY;
  constexpr int QPER = NQ / NW > 0 ? NQ / NW : 1;
  for (int qi = 0; qi < QPER; qi++) {
    const int q = warp * QPER + qi;
    if (q >= NQ) break;
    const float qv = sqv[q];
    const float4 cv = *reinterpret_cast<const float4 *>(sC + q * LDS + lane * 4);
    const float4 mv = *reinterpret_cast<const float4 *>(skv + lane * 4);
    const U4 zv = ld4(zb + q * NKY + lane * 4);
    float sc[4];
    const float cvv[4] = {cv.x, cv.y, cv.z, cv.w};
    const float mvv[4] = {mv.x, mv.y, mv.z, mv.w};
    float mx = -INFINITY;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      float v = rb(rb(cvv[e]) + rb(1.0e9f * rb(rb(qv * mvv[e]) - 1.0f)));
      v = rb(v + g2f(zv.h[e]));
      sc[e] = v;
      mx = fmaxf(mx, v);
    }
    warp_max(mx);
    float ssum = 0.f;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      sc[e] = fexp(sc[e] - mx);
      ssum += sc[e];
    }
    warp_sum(ssum);
    const float inv = __frcp_rn(ssum);
    U4 pv;
#pragma unroll
    for (int e = 0; e < 4; e++) pv.h[e] = f2b(sc[e] * inv);
    st4(sAp + q * LDP + lane * 4, pv);
  }
  __syncthreads();

  // context = P V : [32,128] x [32,128]^T.  Only 4 output tiles, so the 128 keys
  // are split 4 ways and the partials summed in the epilogue -- otherwise 12 of
  // the 16 warps would idle through this GEMM.
  {
    constexpr int CT_ = (NQ / 16) * (HD / 16), KSL = NKY / 4;
    const int t = warp % CT_, sl = warp / CT_;
    const int mt = t / (HD / 16), nt = t - mt * (HD / 16);
    wm::fragment<wm::accumulator, 16, 16, 16, float> c;
    wm::fill_fragment(c, 0.0f);
#pragma unroll
    for (int k0 = 0; k0 < KSL; k0 += 16) {
      wm::fragment<wm::matrix_a, 16, 16, 16, bf, wm::row_major> af;
      wm::fragment<wm::matrix_b, 16, 16, 16, bf, wm::col_major> bfr;
      wm::load_matrix_sync(af, sAp + mt * 16 * LDP + sl * KSL + k0, LDP);
      wm::load_matrix_sync(bfr, sBv + nt * 16 * LDV + sl * KSL + k0, LDV);
      wm::mma_sync(c, af, bfr, c);
    }
    wm::store_matrix_sync(sO + sl * NQ * LDO + mt * 16 * LDO + nt * 16, c, LDO,
                          wm::mem_row_major);
  }
  __syncthreads();
  for (int i = tid * 4; i < NQ * HD; i += NW * 32 * 4) {
    const int q = i / HD, c = i - q * HD;
    const int j = b * NQ + q;
    if (j >= N) continue;
    const U4 gv = ld4(gin + (size_t)j * CDIM + h * HD + c);
    U4 o;
#pragma unroll
    for (int e = 0; e < 4; e++)
      o.h[e] = f2b(rb(sO[q * LDO + c + e] + sO[NQ * LDO + q * LDO + c + e] +
                      sO[2 * NQ * LDO + q * LDO + c + e] +
                      sO[3 * NQ * LDO + q * LDO + c + e]) *
                   g2f(gv.h[e]));
    st4(opre + (size_t)j * CDIM + h * HD + c, o);
  }
}

struct AttnSh {
  static constexpr int TOTAL =
      NQ * (HD + PADA) * 2 + NKY * (HD + PADB) * 2 + HD * (NKY + PADB) * 2 +
      NQ * (NKY + PADA) * 2 + NQ * (NKY + PADC) * 4 + 4 * NQ * (HD + PADC) * 4 +
      NKY * 4 + NQ * 4 + NKY * 4 + 64;
};

// ---------------------------------------------------------------------------
// Per-transformer-block stage 3: linear_o -> AdaLN-Zero output gate -> attention
// residual -> the transition block's AdaLN.
//
// A row-local chain, so one block owns 32 atoms end to end and the
// post-attention residual is fed straight back in from shared memory as the
// next stage's operand.  The grid can only be tiled over rows here (the o-proj's
// 128 output channels all feed the row's LayerNorm), which is why the wide
// SwiGLU projection that follows is a separate, 4x wider kernel.
// ---------------------------------------------------------------------------
template <int BM, int NW, int KSPL>
__global__ void k_oproj(const bf *__restrict__ opre, int N,
                        const bf *__restrict__ sp, int NCOL, int off_ada,
                        int off_tg, int off_ts, const bf *__restrict__ Wo,
                        bf *__restrict__ a, bf *__restrict__ tmid) {
  using SH = GemmSh<BM, CDIM, CDIM, CDIM / KSPL, KSPL>;
  constexpr int LDA = CDIM + PADA, LDC = CDIM + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = blockIdx.x * BM;

  for (int i = tid * 8; i < BM * CDIM; i += NW * 32 * 8) {
    const int r = i / CDIM, c = i - r * CDIM;
    const int j = m0 + r;
    st8(As + r * LDA + c, (j < N) ? ld8(opre + (size_t)j * CDIM + c) : U8{});
  }
  gemm_bt<BM, CDIM, CDIM, CDIM / KSPL, NW, KSPL>(As, Wo, 0, Wsh, Cs, tid, warp);
  __syncthreads();
  for (int i = tid * 8; i < BM * CDIM; i += NW * 32 * 8) {
    const int r = i / CDIM, c = i - r * CDIM;
    const int j = m0 + r;
    if (j >= N) continue;
    const U8 gv = ld8(sp + (size_t)j * NCOL + off_ada + c);
    const U8 av = ld8(a + (size_t)j * CDIM + c);
    U8 o;
#pragma unroll
    for (int e = 0; e < 8; e++)
      o.h[e] = f2b(g2f(av.h[e]) +
                      rb(g2f(gv.h[e]) * rb(csum<KSPL, BM, LDC>(Cs, r, c + e))));
    st8(a + (size_t)j * CDIM + c, o);
    st8(As + r * LDA + c, o);  // a_new is exactly a bf16 value; stage for the AdaLN
  }
  __syncthreads();
  for (int r = warp; r < BM; r += NW) {
    const int j = m0 + r;
    if (j >= N) continue;
    const bf *spr = sp + (size_t)j * NCOL;
    warp_adaln(As + r * LDA, spr + off_tg, spr + off_ts, tmid + (size_t)j * CDIM,
               lane);
  }
}

// SwiGLU's two input projections (c_atom -> 2 * n_transition * c_atom).  Split
// out of k_oproj so the grid can also tile the 512 output columns: 4x the warps
// of a rows-only grid, which on a problem this small is 4x the throughput.
template <int BM, int BN, int NW>
__global__ void k_swiglu_in(const bf *__restrict__ tmid, int N,
                            const bf *__restrict__ Wab, bf *__restrict__ hmid) {
  using SH = GemmSh<BM, BN, CDIM, CDIM>;
  constexpr int LDA = CDIM + PADA, LDC = BN + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;

  for (int i = tid * 8; i < BM * CDIM; i += NW * 32 * 8) {
    const int r = i / CDIM, c = i - r * CDIM;
    const int j = m0 + r;
    st8(As + r * LDA + c, (j < N) ? ld8(tmid + (size_t)j * CDIM + c) : U8{});
  }
  gemm_bt<BM, BN, CDIM, CDIM, NW>(As, Wab, n0, Wsh, Cs, tid, warp);
  __syncthreads();
  for (int i = tid * 8; i < BM * BN; i += NW * 32 * 8) {
    const int r = i / BN, c = i - r * BN;
    const int j = m0 + r;
    if (j >= N) continue;
    U8 o;
#pragma unroll
    for (int e = 0; e < 8; e++) o.h[e] = f2b(Cs[r * LDC + c + e]);
    st8(hmid + (size_t)j * (2 * CH) + n0 + c, o);
  }
}

// ---------------------------------------------------------------------------
// Per-transformer-block stage 4: SwiGLU -> linear_out -> output gate -> mask ->
// transition residual.  One block owns 32 atoms; the 256-wide activation never
// leaves shared memory.
// ---------------------------------------------------------------------------
template <int BM, int NW, int KSPL>
__global__ void k_trans_out(const bf *__restrict__ hmid, int N,
                            const bf *__restrict__ sp, int NCOL, int off_og,
                            const bf *__restrict__ Wout,
                            const bf *__restrict__ mask, bf *__restrict__ a) {
  using SH = GemmSh<BM, CDIM, CH, CH / KSPL, KSPL>;
  constexpr int LDA = CH + PADA, LDC = CDIM + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int m0 = blockIdx.x * BM;

  for (int i = tid * 8; i < BM * CH; i += NW * 32 * 8) {
    const int r = i / CH, c = i - r * CH;
    const int j = m0 + r;
    U8 o{};
    if (j < N) {
      const bf *hr = hmid + (size_t)j * (2 * CH);
      const U8 ha = ld8(hr + c), hbv = ld8(hr + CH + c);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float x = g2f(ha.h[e]);
        o.h[e] = f2b(rb(x * sigf(x)) * g2f(hbv.h[e]));
      }
    }
    st8(As + r * LDA + c, o);
  }
  gemm_bt<BM, CDIM, CH, CH / KSPL, NW, KSPL>(As, Wout, 0, Wsh, Cs, tid, warp);
  __syncthreads();
  for (int i = tid * 8; i < BM * CDIM; i += NW * 32 * 8) {
    const int r = i / CDIM, c = i - r * CDIM;
    const int j = m0 + r;
    if (j >= N) continue;
    const float mk = g2f(mask[j]);
    const U8 gv = ld8(sp + (size_t)j * NCOL + off_og + c);
    const U8 av = ld8(a + (size_t)j * CDIM + c);
    U8 o;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const float upd =
          rb(rb(g2f(gv.h[e]) * rb(csum<KSPL, BM, LDC>(Cs, r, c + e))) * mk);
      o.h[e] = f2b(g2f(av.h[e]) + upd);
    }
    st8(a + (size_t)j * CDIM + c, o);
  }
}

// ---------------------------------------------------------------------------
// Token-level projection (optionally preceded by a LayerNorm): linear_q_in(ai)
// in the decoder, layer_norm_s -> linear_s(si_trunk) in the encoder.
// ---------------------------------------------------------------------------
template <int KT, int NW, int KSPL>
__global__ void k_tokproj(const bf *__restrict__ tok, int T,
                          const bf *__restrict__ wln, const bf *__restrict__ W,
                          bf *__restrict__ out) {
  using SH = GemmSh<16, CDIM, KT, 32, KSPL>;
  constexpr int LDA = KT + PADA, LDC = CDIM + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = blockIdx.x * 16;
  for (int r = warp; r < 16; r += NW) {
    const int j = m0 + r;
    if (j >= T) {
#pragma unroll
      for (int t = 0; t < KT / 128; t++) st4(As + r * LDA + t * 128 + lane * 4, U4{});
    } else if (wln) {
      warp_ln<KT>(tok + (size_t)j * KT, wln, As + r * LDA, lane);
    } else {
#pragma unroll
      for (int t = 0; t < KT / 128; t++)
        st4(As + r * LDA + t * 128 + lane * 4,
            ld4(tok + (size_t)j * KT + t * 128 + lane * 4));
    }
  }
  gemm_bt<16, CDIM, KT, 32, NW, KSPL>(As, W, 0, Wsh, Cs, tid, warp);
  __syncthreads();
  for (int i = tid * 8; i < 16 * CDIM; i += NW * 32 * 8) {
    const int r = i / CDIM, c = i - r * CDIM;
    const int j = m0 + r;
    if (j >= T) continue;
    U8 o;
#pragma unroll
    for (int e = 0; e < 8; e++)
      o.h[e] = f2b(csum<KSPL, 16, LDC>(Cs, r, c + e));
    st8(out + (size_t)j * CDIM + c, o);
  }
}

// base + broadcast-of-a-token-feature, with the query padding rows zeroed so the
// downstream window reshape needs no separate pad kernel.  Also emits
// layer_norm_s(s) for the AdaLN GEMM, which would otherwise be recomputed once
// per column tile there.
template <int NW>
__global__ void k_gather_add(const bf *__restrict__ base,
                             const bf *__restrict__ proj,
                             const bf *__restrict__ s, const bf *__restrict__ wlns,
                             const long *__restrict__ a2t, int N, int Np,
                             bf *__restrict__ out, bf *__restrict__ snorm) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int j = blockIdx.x * NW + warp;
  if (j >= Np) return;
  if (j >= N) {
    st4(out + (size_t)j * CDIM + lane * 4, U4{});
    return;
  }
  const U4 bv = ld4(base + (size_t)j * CDIM + lane * 4);
  const U4 pv = ld4(proj + (size_t)a2t[j] * CDIM + lane * 4);
  U4 o;
#pragma unroll
  for (int e = 0; e < 4; e++) o.h[e] = f2b(g2f(bv.h[e]) + g2f(pv.h[e]));
  st4(out + (size_t)j * CDIM + lane * 4, o);
  warp_ln<CDIM>(s + (size_t)j * CDIM, wlns, snorm + (size_t)j * CDIM, lane);
}

// Final LayerNorm + narrow projection (c_atom -> 3) of the decoder.
template <int NW>
__global__ void k_final(const bf *__restrict__ a, int N, const bf *__restrict__ wln,
                        const bf *__restrict__ W, int OUTD, bf *__restrict__ out) {
  __shared__ __align__(16) bf srow[NW * CDIM];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int j = blockIdx.x * NW + warp;
  if (j >= N) return;
  warp_ln<CDIM>(a + (size_t)j * CDIM, wln, srow + warp * CDIM, lane);
  __syncwarp();
  const U4 tv = ld4(srow + warp * CDIM + lane * 4);
  for (int d = 0; d < OUTD; d++) {
    const U4 wv = ld4(W + (size_t)d * CDIM + lane * 4);
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < 4; e++) acc += g2f(tv.h[e]) * g2f(wv.h[e]);
    warp_sum(acc);
    if (lane == 0) out[(size_t)j * OUTD + d] = f2b(acc);
  }
}

// ===========================================================================
// Encoder-only stages (AF3 Algorithm 5, lines 1-12 plus the pair stack).
// ===========================================================================

// The five reference-feature linears share one input row, so they are one GEMM
// against a concatenated [c_atom, 3+1+1+c_elem+c_chars] weight.  This builds that
// row: positions, arcsinh(charge), the reference mask, the element one-hot and
// the flattened atom-name characters, zero-padded to a multiple of 128.
__global__ void k_refx(const bf *__restrict__ pos, const bf *__restrict__ chg,
                       const bf *__restrict__ rmask, const bf *__restrict__ elem,
                       const bf *__restrict__ chars, int N, int CE, int CN,
                       int KCL, bf *__restrict__ X) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * KCL) return;
  const int j = i / KCL, c = i - j * KCL;
  float v = 0.f;
  if (c < 3) v = g2f(pos[(size_t)j * 3 + c]);
  else if (c == 3) v = rb(asinhf(g2f(chg[j])));
  else if (c == 4) v = g2f(rmask[j]);
  else if (c < 5 + CE) v = g2f(elem[(size_t)j * CE + (c - 5)]);
  else if (c < 5 + CE + CN) v = g2f(chars[(size_t)j * CN + (c - 5 - CE)]);
  X[i] = f2b(v);
}

// layer_norm_z -> linear_z on the trunk pair representation: [T*T, c_z] -> [T*T, CZ].
__global__ void k_zproj(const bf *__restrict__ zij, int ROWS,
                        const bf *__restrict__ wln, const bf *__restrict__ W,
                        bf *__restrict__ out) {
  __shared__ __align__(16) bf srow[8 * CDIM];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nw = blockDim.x >> 5;
  const int r = blockIdx.x * nw + warp;
  if (r >= ROWS) return;
  warp_ln<CDIM>(zij + (size_t)r * CDIM, wln, srow + warp * CDIM, lane);
  __syncwarp();
  const U4 xv = ld4(srow + warp * CDIM + lane * 4);
  for (int ch = 0; ch < CZ; ch++) {
    const U4 wv = ld4(W + (size_t)ch * CDIM + lane * 4);
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < 4; e++) acc += g2f(xv.h[e]) * g2f(wv.h[e]);
    warp_sum(acc);
    if (lane == 0) out[(size_t)r * CZ + ch] = f2b(acc);
  }
}

// Atom conditioning finalized: cl += broadcast(linear_s(layer_norm_s(si_trunk))),
// ql = cl + linear_r(rl), and the two single-rep pair projections
// linear_l/linear_m(relu(cl)) -- the latter computed once per atom instead of
// once per window slot (the reference builds a 4x-duplicated key copy first).
template <int BM, int NW, int KSPL, bool NOISY>
__global__ void k_clfinal(const bf *__restrict__ clbase,
                          const bf *__restrict__ siproj,
                          const long *__restrict__ a2t, const bf *__restrict__ rl,
                          const bf *__restrict__ Wr, const bf *__restrict__ WLM,
                          const bf *__restrict__ wlns, int N, int Np,
                          bf *__restrict__ cl, bf *__restrict__ a,
                          bf *__restrict__ LM, bf *__restrict__ snorm) {
  using SH = GemmSh<BM, 2 * CZ, CDIM, CDIM / KSPL, KSPL>;
  constexpr int LDA = CDIM + PADA, LDC = 2 * CZ + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m0 = blockIdx.x * BM;

  for (int i = tid * 8; i < BM * CDIM; i += NW * 32 * 8) {
    const int r = i / CDIM, c = i - r * CDIM;
    const int j = m0 + r;
    if (j >= N) {
      st8(As + r * LDA + c, U8{});
      continue;
    }
    const U8 bv = ld8(clbase + (size_t)j * CDIM + c);
    U8 cv, av;
    if (NOISY) {
      const U8 sv = ld8(siproj + (size_t)a2t[j] * CDIM + c);
      const float r0 = g2f(rl[(size_t)j * 3 + 0]), r1 = g2f(rl[(size_t)j * 3 + 1]),
                  r2 = g2f(rl[(size_t)j * 3 + 2]);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float cval = rb(g2f(bv.h[e]) + g2f(sv.h[e]));
        const bf *wr = Wr + (size_t)(c + e) * 3;
        const float rv = rb(g2f(wr[0]) * r0 + g2f(wr[1]) * r1 + g2f(wr[2]) * r2);
        cv.h[e] = f2b(cval);
        av.h[e] = f2b(cval + rv);
      }
    } else {
      cv = bv;
      av = bv;
    }
    st8(cl + (size_t)j * CDIM + c, cv);
    st8(a + (size_t)j * CDIM + c, av);
    U8 rv;
#pragma unroll
    for (int e = 0; e < 8; e++) rv.h[e] = f2b(fmaxf(g2f(cv.h[e]), 0.f));
    st8(As + r * LDA + c, rv);
  }
  __syncthreads();
  for (int r = warp; r < BM; r += NW) {
    const int j = m0 + r;
    if (j < N)
      warp_ln<CDIM>(cl + (size_t)j * CDIM, wlns, snorm + (size_t)j * CDIM, lane);
  }
  // pad the query rows the window reshape will read
  for (int i = N + tid; i < Np; i += NW * 32)
#pragma unroll
    for (int c = 0; c < CDIM; c += 8) st8(a + (size_t)i * CDIM + c, U8{});
  gemm_bt<BM, 2 * CZ, CDIM, CDIM / KSPL, NW, KSPL>(As, WLM, 0, Wsh, Cs, tid, warp);
  __syncthreads();
  for (int i = tid * 4; i < BM * 2 * CZ; i += NW * 32 * 4) {
    const int r = i / (2 * CZ), c = i - r * (2 * CZ);
    const int j = m0 + r;
    if (j >= N) continue;
    U4 o;
#pragma unroll
    for (int e = 0; e < 4; e++)
      o.h[e] = f2b(csum<KSPL, BM, LDC>(Cs, r, c + e));
    st4(LM + (size_t)j * (2 * CZ) + c, o);
  }
}

// ---------------------------------------------------------------------------
// The whole atom pair representation, in one kernel.
//
// Fuses what the reference spends ~150 eager kernels on over a 786 k-element
// [12,32,128,16] tensor: the windowed offset vectors and their inverse square
// distances, the reference-space-uid validity mask, all three
// RefAtomFeatureEmbedder pair linears, the trunk pair projection gathered into
// window form, the linear_l/linear_m outer sum, the 3-layer pair MLP with its
// residual, the block mask, and finally layer_norm_z + all three transformer
// blocks' linear_z -- so the pair tensor is written once and read once, and the
// attention bias comes out of the same pass.
//
// One block owns one (window, query) row of 128 keys; one warp owns 16 of those
// keys, which is exactly one wmma tile, so the three 16x16 MLP layers are three
// mma instructions with a shared-memory round trip between them for the ReLU.
// Two lanes share an element (8 of its 16 channels each), so the per-element
// reductions -- the distance, the LayerNorm, the linear_z dots -- finish with a
// single width-2 shuffle.
// ---------------------------------------------------------------------------
template <int NW, bool NOISY>
__global__ void k_plm(const bf *__restrict__ pos, const bf *__restrict__ uid,
                      const bf *__restrict__ LM, const bf *__restrict__ zt,
                      const long *__restrict__ a2t, const int *__restrict__ kidx,
                      const float *__restrict__ kgeo,
                      const float *__restrict__ kvalid,
                      const float *__restrict__ qvalid,
                      const float *__restrict__ Wpsc, const bf *__restrict__ Wmlp,
                      const bf *__restrict__ wlnz, const bf *__restrict__ Wz,
                      int N, int T, int NB, int nblk, bf *__restrict__ plm,
                      bf *__restrict__ zbias) {
  constexpr int LDP = CZ + PADA;    // bf16 matrix_a tile: [16][24]
  constexpr int LDQ = CZ + PADC;    // fp32 accumulator tile: [16][24]
  __shared__ __align__(16) bf Ash[NW * 16 * LDP];
  __shared__ __align__(16) float Qsh[NW * 16 * LDQ];
  __shared__ float sW[CZ * 5];
  __shared__ float sWz[16 * NH * CZ];
  __shared__ float slnz[CZ];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int i = tid; i < CZ * 5; i += NW * 32) sW[i] = Wpsc[i];
  for (int i = tid; i < nblk * NH * CZ; i += NW * 32) sWz[i] = g2f(Wz[i]);
  for (int i = tid; i < CZ; i += NW * 32) slnz[i] = g2f(wlnz[i]);
  __syncthreads();

  const int bq = blockIdx.x;
  const int b = bq / NQ, q = bq - b * NQ;
  const int jq = b * NQ + q;                 // padded query atom
  const int el = lane >> 1;                  // key within this warp's tile
  const int c0 = (lane & 1) * 8;             // channel half
  const int kk = warp * 16 + el;             // key within the window
  bf *At = Ash + warp * 16 * LDP;
  float *Qt = Qsh + warp * 16 * LDQ;

  const int m = kidx[b * NKY + kk];
  const float geo = kgeo[b * NKY + kk];
  const float bm = rb(qvalid[jq] * kvalid[b * NKY + kk]);
  const bool qin = jq < N;

  // offsets, |offset|^-2 and the same-space mask, all in the reference's order
  float dlm[3];
#pragma unroll
  for (int c = 0; c < 3; c++) {
    const float dl = qin ? g2f(pos[(size_t)jq * 3 + c]) : 0.f;
    const float dm = geo > 0.f ? g2f(pos[(size_t)m * 3 + c]) : 0.f;
    dlm[c] = rb(rb(dl - dm) * bm);
  }
  const float vl = qin ? g2f(uid[jq]) : 0.f;
  const float vm = geo > 0.f ? g2f(uid[m]) : 0.f;
  const float vlm = rb((vl == vm ? 1.0f : 0.0f) * bm);
  const float sqsum = rb(rb(dlm[0] * dlm[0]) + rb(dlm[1] * dlm[1]) + rb(dlm[2] * dlm[2]));
  const float invsq = rb(1.0f / rb(1.0f + sqsum));

  const U8 lv = qin ? ld8(LM + (size_t)jq * (2 * CZ) + c0) : U8{};
  const U8 mv = geo > 0.f ? ld8(LM + (size_t)m * (2 * CZ) + CZ + c0) : U8{};
  U8 ztv{};
  if (NOISY) {
    const int qtok = qin ? (int)a2t[jq] : 0;
    const int ktok = (int)a2t[m];
    if (geo > 0.f) ztv = ld8(zt + ((size_t)qtok * T + ktok) * CZ + c0);
  }

  float res[8];
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const int ch = c0 + e;
    const float *wo = sW + ch * 5;
    float p = rb(rb(wo[0] * dlm[0] + wo[1] * dlm[1] + wo[2] * dlm[2]) * vlm);
    p = rb(p + rb(rb(wo[3] * invsq) * vlm));
    p = rb(p + rb(rb(wo[4] * vlm) * vlm));
    if (NOISY) p = rb(p + rb(g2f(ztv.h[e]) * bm));
    p = rb(p + rb(rb(g2f(lv.h[e]) + g2f(mv.h[e])) * bm));
    res[e] = p;
    At[el * LDP + ch] = f2b(fmaxf(p, 0.f));  // pair_mlp starts with a ReLU
  }
  __syncwarp();

  // three 16x16 layers; ReLU between them, through the shared tile
  wm::fragment<wm::matrix_a, 16, 16, 16, bf, wm::row_major> af;
  wm::fragment<wm::matrix_b, 16, 16, 16, bf, wm::col_major> bfr;
  wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
  for (int layer = 0; layer < 3; layer++) {
    wm::load_matrix_sync(af, At, LDP);
    wm::load_matrix_sync(bfr, Wmlp + (size_t)layer * CZ * CZ, CZ);
    wm::fill_fragment(acc, 0.0f);
    wm::mma_sync(acc, af, bfr, acc);
    wm::store_matrix_sync(Qt, acc, LDQ, wm::mem_row_major);
    __syncwarp();
    if (layer < 2) {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int ch = c0 + e;
        At[el * LDP + ch] = f2b(fmaxf(rb(Qt[el * LDQ + ch]), 0.f));
      }
      __syncwarp();
    }
  }

  // residual + block mask; this is the returned pair representation
  U8 xv;
  float x[8];
  float sum = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    x[e] = rb(rb(res[e] + rb(Qt[el * LDQ + c0 + e])) * bm);
    xv.h[e] = f2b(x[e]);
    sum += x[e];
  }
  st8(plm + ((size_t)bq * NKY + kk) * CZ + c0, xv);

  // layer_norm_z + every transformer block's linear_z -> attention bias
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);
  const float mu = sum * (1.0f / CZ);
  float var = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const float d = x[e] - mu;
    var += d * d;
  }
  var += __shfl_xor_sync(0xffffffffu, var, 1);
  const float rs = rsqrtf(var * (1.0f / CZ) + LN_EPS);
  float zn[8];
#pragma unroll
  for (int e = 0; e < 8; e++) zn[e] = rb((x[e] - mu) * rs * slnz[c0 + e]);
  for (int i = 0; i < nblk; i++)
#pragma unroll
    for (int h = 0; h < NH; h++) {
      const float *wz = sWz + (i * NH + h) * CZ + c0;
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) s += zn[e] * wz[e];
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      if (c0 == 0)
        zbias[((((size_t)i * NB + b) * NH + h) * NQ + q) * NKY + kk] = f2b(s);
    }
}

// ---------------------------------------------------------------------------
// Encoder output: mask the atom representation (this is the returned ql), project
// it to the token width, ReLU.
// ---------------------------------------------------------------------------
template <int BM, int BN, int NW>
__global__ void k_atomproj(const bf *__restrict__ a, const bf *__restrict__ mask,
                           int N, const bf *__restrict__ Wq, int CT,
                           bf *__restrict__ qlout, bf *__restrict__ proj) {
  using SH = GemmSh<BM, BN, CDIM, CDIM>;
  constexpr int LDA = CDIM + PADA, LDC = BN + PADC;
  extern __shared__ __align__(16) char smem[];
  bf *As = reinterpret_cast<bf *>(smem);
  float *Cs = reinterpret_cast<float *>(smem + SH::A);
  bf *Wsh = reinterpret_cast<bf *>(smem + SH::A + SH::C);
  const int tid = threadIdx.x, warp = tid >> 5;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;

  for (int i = tid * 8; i < BM * CDIM; i += NW * 32 * 8) {
    const int r = i / CDIM, c = i - r * CDIM;
    const int j = m0 + r;
    U8 o{};
    if (j < N) {
      const float mk = g2f(mask[j]);
      const U8 av = ld8(a + (size_t)j * CDIM + c);
#pragma unroll
      for (int e = 0; e < 8; e++) o.h[e] = f2b(g2f(av.h[e]) * mk);
      if (n0 == 0) st8(qlout + (size_t)j * CDIM + c, o);
    }
    st8(As + r * LDA + c, o);
  }
  gemm_bt<BM, BN, CDIM, CDIM, NW>(As, Wq, n0, Wsh, Cs, tid, warp);
  __syncthreads();
  for (int i = tid * 8; i < BM * BN; i += NW * 32 * 8) {
    const int r = i / BN, c = i - r * BN;
    const int j = m0 + r;
    if (j >= N) continue;
    U8 o;
#pragma unroll
    for (int e = 0; e < 8; e++) o.h[e] = f2b(fmaxf(rb(Cs[r * LDC + c + e]), 0.f));
    st8(proj + (size_t)j * CT + n0 + c, o);
  }
}

// Mean-pool the atom projection into tokens.  One block per (token, channel
// tile): it first compacts the atoms belonging to its token (a 3-iteration scan
// of the atom->token map), then sums only those rows -- so the reference's bf16
// scatter_add becomes a deterministic fp32 accumulation with no atomics, and the
// inner loop is ~N/T coalesced reads instead of N dependent ones.
__global__ void k_agg(const bf *__restrict__ proj, const bf *__restrict__ mask,
                      const long *__restrict__ a2t, int N, int CT,
                      bf *__restrict__ ai) {
  extern __shared__ __align__(16) char smem[];
  int *slist = reinterpret_cast<int *>(smem);
  float *smk = reinterpret_cast<float *>(slist + N);
  __shared__ int scount;
  const int t = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) scount = 0;
  __syncthreads();
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    if ((int)a2t[j] == t) {
      const int p = atomicAdd(&scount, 1);
      slist[p] = j;
      smk[p] = g2f(mask[j]);
    }
  __syncthreads();
  const int n = scount;
  if (d >= CT) return;
  float acc = 0.f, cnt = 0.f;
  for (int i = 0; i < n; i++) {
    acc += rb(g2f(proj[(size_t)slist[i] * CT + d]) * smk[i]);
    cnt += smk[i];
  }
  ai[(size_t)t * CT + d] = f2b(acc / fmaxf(cnt, 1.0f));
}

// ---------------------------------------------------------------------------
// Host side: workspace layout + the entry points.  Driving the whole operator
// from C++ matters on its own: a launch costs ~3 us of CPU time here against
// ~9 us from Python, and there are only ~20 of them left.
// ---------------------------------------------------------------------------
static inline size_t algn(size_t x) { return (x + 255u) & ~(size_t)255u; }

struct Layout {
  size_t a, qh, gh, kf, vf, opre, tmid, hmid, sproj, zbias, proj, kidx, kgeo,
      kvalid, qvalid, snorm, refx, clbase, siproj, zt, lm, tokproj, total;
};

static Layout make_layout(int N, int NB, int nblk) {
  const int Np = NB * NQ;
  Layout L;
  size_t o = 0;
  auto take = [&](size_t bytes) {
    const size_t r = o;
    o += algn(bytes);
    return r;
  };
  L.a = take(sizeof(bf) * (size_t)Np * CDIM);
  L.qh = take(sizeof(bf) * (size_t)Np * CDIM);
  L.gh = take(sizeof(bf) * (size_t)Np * CDIM);
  L.kf = take(sizeof(bf) * (size_t)N * CDIM);
  L.vf = take(sizeof(bf) * (size_t)N * CDIM);
  L.opre = take(sizeof(bf) * (size_t)Np * CDIM);
  L.tmid = take(sizeof(bf) * (size_t)N * CDIM);
  L.hmid = take(sizeof(bf) * (size_t)N * 2 * CH);
  L.sproj = take(sizeof(bf) * (size_t)N * nblk * (NGRP_N + NGRP_R) * CDIM);
  L.zbias = take(sizeof(bf) * (size_t)nblk * NB * NH * NQ * NKY);
  L.proj = take(sizeof(bf) * 512u * CDIM);
  L.kidx = take(sizeof(int) * (size_t)NB * NKY);
  L.kgeo = take(sizeof(float) * (size_t)NB * NKY);
  L.kvalid = take(sizeof(float) * (size_t)NB * NKY);
  L.qvalid = take(sizeof(float) * (size_t)Np);
  L.snorm = take(sizeof(bf) * (size_t)N * CDIM);
  // encoder-only scratch
  L.refx = take(sizeof(bf) * (size_t)N * 512);
  L.clbase = take(sizeof(bf) * (size_t)N * CDIM);
  L.siproj = take(sizeof(bf) * 512u * CDIM);
  L.zt = take(sizeof(bf) * 512u * 512u * CZ / 4);
  L.lm = take(sizeof(bf) * (size_t)N * 2 * CZ);
  L.tokproj = take(sizeof(bf) * (size_t)N * 1024);
  L.total = o;
  return L;
}

#define PTR(T, off) reinterpret_cast<T *>(w + L.off)
#define CPTR(T, off) reinterpret_cast<const T *>(w + L.off)

// Launch geometry: one wmma output tile per warp (TPW == 1 wherever the tile
// count allows it), because warps-in-flight is what bounds a latency-bound
// kernel.  Kernels whose shared arena exceeds the 48 KB static limit need the
// opt-in once per process; the bool keeps it off the per-call path.
using SprojSh = GemmSh<64, 64, CDIM, CDIM>;
using QkvgSh = GemmSh<32, CDIM, CDIM, CDIM>;
using OprojSh = GemmSh<16, CDIM, CDIM, CDIM / 2, 2>;
using SwgSh = GemmSh<32, CDIM, CDIM, CDIM>;
using TransSh = GemmSh<16, CDIM, CH, CH / 2, 2>;
using ClfSh = GemmSh<32, 2 * CZ, CDIM, CDIM / 4, 4>;
using AtomSh = GemmSh<32, CDIM, CDIM, CDIM>;
template <int KT>
using TokSh = GemmSh<16, CDIM, KT, 32, 4>;

static bool g_smem_ready = false;

static void init_smem() {
  if (g_smem_ready) return;
  g_smem_ready = true;
  auto set = [](const void *f, int b) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
  };
  set((const void *)k_sproj<64, 64, 16>, SprojSh::TOTAL);
  set((const void *)k_attn<16>, AttnSh::TOTAL);
  set((const void *)k_qkvg<32, CDIM, 16>, QkvgSh::TOTAL);
  set((const void *)k_oproj<16, 16, 2>, OprojSh::TOTAL);
  set((const void *)k_swiglu_in<32, CDIM, 16>, SwgSh::TOTAL);
  set((const void *)k_trans_out<16, 16, 2>, TransSh::TOTAL);
  set((const void *)k_clfinal<32, 16, 4, true>, ClfSh::TOTAL);
  set((const void *)k_clfinal<32, 16, 4, false>, ClfSh::TOTAL);
  set((const void *)k_atomproj<32, CDIM, 16>, AtomSh::TOTAL);
  set((const void *)k_tokproj<128, 32, 4>, TokSh<128>::TOTAL);
  set((const void *)k_tokproj<256, 32, 4>, TokSh<256>::TOTAL);
  set((const void *)k_tokproj<384, 32, 4>, TokSh<384>::TOTAL);
  set((const void *)k_tokproj<768, 32, 4>, TokSh<768>::TOTAL);
  set((const void *)k_tokproj<512, 32, 4>, TokSh<512>::TOTAL);
}

static void launch_tokproj(int KT, const bf *tok, int T, const bf *wln,
                           const bf *W, bf *out, cudaStream_t st) {
  const int g = (T + 15) / 16;
#define TOKCASE(K)                                                            \
  case K:                                                                     \
    k_tokproj<K, 32, 4><<<g, 1024, TokSh<K>::TOTAL, st>>>(tok, T, wln, W, out);     \
    break;
  switch (KT) {
    TOKCASE(128)
    TOKCASE(256)
    TOKCASE(384)
    TOKCASE(512)
    TOKCASE(768)
    default: TORCH_CHECK(false, "unsupported token width ", KT);
  }
#undef TOKCASE
}

// The 3-block DiffusionTransformer stack: 5 kernels per block plus two shared
// preambles (the pair bias and every AdaLN conditioning term), against the
// reference's ~250 eager ops per block.
static void run_stack_nozb(const Layout &L, char *w, int N, int NB, int nblk,
                           const bf *s, const bf *mask, const bf *Wsp,
                           const float *Bsp, const unsigned char *Sig,
                           const bf *Wqkvg, const float *Bq, const bf *Wo,
                           const bf *Wab, const bf *Wout, cudaStream_t st) {
  const int NCOL = nblk * (NGRP_N + NGRP_R) * CDIM;
  const int RAWB = nblk * NGRP_N * CDIM;
  const int rows32 = (N + 31) / 32;
  const int rows16 = (N + 15) / 16;
  bf *a = PTR(bf, a);

  k_sproj<64, 64, 16><<<dim3((N + 63) / 64, NCOL / 64), 512, SprojSh::TOTAL, st>>>(
      s, CPTR(bf, snorm), N, Wsp, Bsp, Sig, NCOL, RAWB, PTR(bf, sproj));

  for (int i = 0; i < nblk; i++) {
    const int oqg = (i * NGRP_N + 0) * CDIM, oqs = (i * NGRP_N + 1) * CDIM;
    const int okg = (i * NGRP_N + 2) * CDIM, oks = (i * NGRP_N + 3) * CDIM;
    const int otg = (i * NGRP_N + 4) * CDIM, ots = (i * NGRP_N + 5) * CDIM;
    const int oad = RAWB + (i * NGRP_R + 0) * CDIM;
    const int oog = RAWB + (i * NGRP_R + 1) * CDIM;
    k_qkvg<32, CDIM, 16><<<dim3(rows32, 4), 512, QkvgSh::TOTAL, st>>>(
        a, N, CPTR(bf, sproj), NCOL, oqg, oqs, okg, oks,
        Wqkvg + (size_t)i * 4 * CDIM * CDIM, Bq + (size_t)i * CDIM, PTR(bf, qh),
        PTR(bf, gh), PTR(bf, kf), PTR(bf, vf));
    k_attn<16><<<dim3(NB, NH), 512, AttnSh::TOTAL, st>>>(
        CPTR(bf, qh), CPTR(bf, kf), CPTR(bf, vf), CPTR(bf, gh),
        CPTR(bf, zbias) + (size_t)i * NB * NH * NQ * NKY, CPTR(int, kidx),
        CPTR(float, kgeo), CPTR(float, kvalid), CPTR(float, qvalid), N,
        PTR(bf, opre));
    k_oproj<16, 16, 2><<<rows16, 512, OprojSh::TOTAL, st>>>(
        CPTR(bf, opre), N, CPTR(bf, sproj), NCOL, oad, otg, ots,
        Wo + (size_t)i * CDIM * CDIM, a, PTR(bf, tmid));
    k_swiglu_in<32, CDIM, 16><<<dim3(rows32, 2 * CH / CDIM), 512, SwgSh::TOTAL, st>>>(
        CPTR(bf, tmid), N, Wab + (size_t)i * 2 * CH * CDIM, PTR(bf, hmid));
    k_trans_out<16, 16, 2><<<rows16, 512, TransSh::TOTAL, st>>>(
        CPTR(bf, hmid), N, CPTR(bf, sproj), NCOL, oog,
        Wout + (size_t)i * CDIM * CH, mask, a);
  }
}

// Surface launch failures (bad launch config, shared-memory request) here rather
// than letting a silently-skipped kernel return uninitialized output.
static void check_launch(const char *what) {
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "fk_af3 ", what, ": ", cudaGetErrorString(e));
}

static const bf *bfp(const at::Tensor &t) {
  return reinterpret_cast<const bf *>(t.data_ptr());
}
static bf *bfpm(at::Tensor &t) { return reinterpret_cast<bf *>(t.data_ptr()); }

void decoder_forward(at::Tensor out, at::Tensor ws, at::Tensor ai, at::Tensor ql,
                     at::Tensor cl, at::Tensor plm, at::Tensor mask,
                     at::Tensor a2t, at::Tensor Wqin, at::Tensor wlnz,
                     at::Tensor Wz, at::Tensor Wsp, at::Tensor Bsp,
                     at::Tensor Sig, at::Tensor wlns, at::Tensor Wqkvg,
                     at::Tensor Bq, at::Tensor Wo, at::Tensor Wab,
                     at::Tensor Wout, at::Tensor wlnfin, at::Tensor Wfin,
                     int64_t nblk_) {
  const at::cuda::CUDAGuard guard(ql.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  const int N = (int)mask.numel();
  const int T = (int)ai.size(0);
  const int KT = (int)ai.size(1);
  const int NB = (int)plm.size(0);
  const int nblk = (int)nblk_;
  const int Np = NB * NQ;
  const Layout L = make_layout(N, NB, nblk);
  init_smem();
  TORCH_CHECK(ws.numel() >= (int64_t)L.total, "workspace too small");
  char *w = reinterpret_cast<char *>(ws.data_ptr());

  k_index<<<1, 256, 0, st>>>(bfp(mask), N, NB, PTR(int, kidx), PTR(float, kgeo),
                             PTR(float, kvalid), PTR(float, qvalid));
  launch_tokproj(KT, bfp(ai), T, nullptr, bfp(Wqin), PTR(bf, proj), st);
  k_gather_add<8><<<(Np + 7) / 8, 256, 0, st>>>(
      bfp(ql), CPTR(bf, proj), bfp(cl), bfp(wlns), a2t.data_ptr<long>(), N, Np,
      PTR(bf, a), PTR(bf, snorm));
  k_zbias<<<NB * NQ, NKY, (nblk * NH * CZ + CZ) * sizeof(float), st>>>(
      bfp(plm), bfp(wlnz), bfp(Wz), nblk, NB, PTR(bf, zbias));
  run_stack_nozb(L, w, N, NB, nblk, bfp(cl), bfp(mask), bfp(Wsp),
                 Bsp.data_ptr<float>(),
                 reinterpret_cast<const unsigned char *>(Sig.data_ptr()),
                 bfp(Wqkvg), Bq.data_ptr<float>(), bfp(Wo), bfp(Wab), bfp(Wout),
                 st);
  k_final<8><<<(N + 7) / 8, 256, 0, st>>>(CPTR(bf, a), N, bfp(wlnfin), bfp(Wfin),
                                          (int)Wfin.size(0), bfpm(out));
  check_launch("decoder_forward");
}

void encoder_forward(at::Tensor ai_out, at::Tensor ql_out, at::Tensor cl_out,
                     at::Tensor plm_out, at::Tensor ws, at::Tensor ref_pos,
                     at::Tensor ref_chg, at::Tensor ref_mask, at::Tensor ref_elem,
                     at::Tensor ref_chars, at::Tensor ref_uid, at::Tensor mask,
                     at::Tensor a2t, at::Tensor rl, at::Tensor si, at::Tensor zij,
                     at::Tensor Wcl, at::Tensor Wsi, at::Tensor wln_si,
                     at::Tensor wln_z, at::Tensor Wzt, at::Tensor Wr,
                     at::Tensor WLM, at::Tensor Wpsc, at::Tensor Wmlp,
                     at::Tensor wlnz, at::Tensor Wz, at::Tensor Wsp,
                     at::Tensor Bsp, at::Tensor Sig, at::Tensor wlns,
                     at::Tensor Wqkvg, at::Tensor Bq, at::Tensor Wo,
                     at::Tensor Wab, at::Tensor Wout, at::Tensor Wq,
                     int64_t nblk_, int64_t ntok_, bool noisy) {
  const at::cuda::CUDAGuard guard(mask.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  const int N = (int)mask.numel();
  const int ntok = (int)ntok_;
  const int CT = (int)Wq.size(0);
  const int NB = (int)plm_out.size(0);
  const int nblk = (int)nblk_;
  const int Np = NB * NQ;
  const int CE = (int)ref_elem.size(1), CN = (int)ref_chars.size(1);
  const int KCL = (int)Wcl.size(1);
  const Layout L = make_layout(N, NB, nblk);
  init_smem();
  TORCH_CHECK(ws.numel() >= (int64_t)L.total, "workspace too small");
  char *w = reinterpret_cast<char *>(ws.data_ptr());
  const long *a2tp = a2t.data_ptr<long>();

  k_index<<<1, 256, 0, st>>>(bfp(mask), N, NB, PTR(int, kidx), PTR(float, kgeo),
                             PTR(float, kvalid), PTR(float, qvalid));
  k_refx<<<(N * KCL + 255) / 256, 256, 0, st>>>(
      bfp(ref_pos), bfp(ref_chg), bfp(ref_mask), bfp(ref_elem), bfp(ref_chars), N,
      CE, CN, KCL, PTR(bf, refx));
  launch_tokproj(KCL, CPTR(bf, refx), N, nullptr, bfp(Wcl), PTR(bf, clbase), st);
  if (noisy) {
    launch_tokproj((int)Wsi.size(1), bfp(si), ntok, bfp(wln_si), bfp(Wsi),
                   PTR(bf, siproj), st);
    const int zrows = (int)(zij.numel() / CDIM);
    k_zproj<<<(zrows + 7) / 8, 256, 0, st>>>(bfp(zij), zrows, bfp(wln_z),
                                             bfp(Wzt), PTR(bf, zt));
    k_clfinal<32, 16, 4, true><<<(N + 31) / 32, 512, ClfSh::TOTAL, st>>>(
        CPTR(bf, clbase), CPTR(bf, siproj), a2tp, bfp(rl), bfp(Wr), bfp(WLM),
        bfp(wlns), N, Np, bfpm(cl_out), PTR(bf, a), PTR(bf, lm), PTR(bf, snorm));
    k_plm<8, true><<<NB * NQ, 256, 0, st>>>(
        bfp(ref_pos), bfp(ref_uid), CPTR(bf, lm), CPTR(bf, zt), a2tp,
        CPTR(int, kidx), CPTR(float, kgeo), CPTR(float, kvalid),
        CPTR(float, qvalid), Wpsc.data_ptr<float>(), bfp(Wmlp), bfp(wlnz),
        bfp(Wz), N, ntok, NB, nblk, bfpm(plm_out), PTR(bf, zbias));
  } else {
    k_clfinal<32, 16, 4, false><<<(N + 31) / 32, 512, ClfSh::TOTAL, st>>>(
        CPTR(bf, clbase), nullptr, a2tp, nullptr, nullptr, bfp(WLM), bfp(wlns), N,
        Np, bfpm(cl_out), PTR(bf, a), PTR(bf, lm), PTR(bf, snorm));
    k_plm<8, false><<<NB * NQ, 256, 0, st>>>(
        bfp(ref_pos), bfp(ref_uid), CPTR(bf, lm), nullptr, a2tp, CPTR(int, kidx),
        CPTR(float, kgeo), CPTR(float, kvalid), CPTR(float, qvalid),
        Wpsc.data_ptr<float>(), bfp(Wmlp), bfp(wlnz), bfp(Wz), N, ntok, NB, nblk,
        bfpm(plm_out), PTR(bf, zbias));
  }
  run_stack_nozb(L, w, N, NB, nblk, bfp(cl_out), bfp(mask), bfp(Wsp),
                 Bsp.data_ptr<float>(),
                 reinterpret_cast<const unsigned char *>(Sig.data_ptr()),
                 bfp(Wqkvg), Bq.data_ptr<float>(), bfp(Wo), bfp(Wab), bfp(Wout),
                 st);
  bf *projbuf = PTR(bf, tokproj);
  k_atomproj<32, CDIM, 16><<<dim3((N + 31) / 32, CT / CDIM), 512, AtomSh::TOTAL, st>>>(
      CPTR(bf, a), bfp(mask), N, bfp(Wq), CT, bfpm(ql_out), projbuf);
  k_agg<<<dim3(ntok, (CT + 127) / 128), 128,
          N * (sizeof(int) + sizeof(float)), st>>>(projbuf, bfp(mask), a2tp, N, CT,
                                                  bfpm(ai_out));
  check_launch("encoder_forward");
}

int64_t ws_bytes(int64_t N, int64_t NB, int64_t nblk) {
  return (int64_t)make_layout((int)N, (int)NB, (int)nblk).total;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decoder_forward", &decoder_forward, "fused AtomAttentionDecoder");
  m.def("encoder_forward", &encoder_forward, "fused AtomAttentionEncoder");
  m.def("ws_bytes", &ws_bytes, "workspace size in bytes");
}
"""

_EXT = None
_LOADED = False


def _pin_arch() -> None:
    """Build for the local arch only.

    The ambient ``TORCH_CUDA_ARCH_LIST`` here lists six architectures, which
    turns a ~1 min build into a ~6 min one.  Mirrors ``infra/cuda_ext.py``,
    including its ``FASTKERNELS_CUDA_ARCH_LIST`` escape hatch.
    """
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - no device: leave the ambient list alone
        return
    suffix = "a" if major in (9, 10, 12) else ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _load() -> None:
    """JIT-compile the fused kernels; leave ``_EXT`` None if that is impossible."""
    global _EXT, _LOADED
    _LOADED = True
    try:
        from torch.utils.cpp_extension import load_inline

        _pin_arch()
        _EXT = load_inline(
            name="fk_l2_af3_atom_attention",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the reference path
        _EXT = None


# ---------------------------------------------------------------------------
# Weight packing for the atom transformer stack.
#
# Built once (the parameters are loaded before the first forward) and keyed on a
# sentinel parameter's storage + version, so an in-place ``load_state_dict`` or a
# ``.to(dtype)`` that replaces the storage rebuilds it.  Two identity compares
# per call against the ~50 us of work they guard.
# ---------------------------------------------------------------------------
_NGRP_N = 6   # AdaLN groups fed from layer_norm_s(s)
_NGRP_R = 2   # AdaLN-Zero output gates fed from raw s


class _StackPack:
    """Concatenated, contiguous bf16 weights for one DiffusionTransformer stack."""

    __slots__ = ("nblk", "wlnz", "Wz", "Wsp", "Bsp", "Sig", "wlns", "Wqkvg",
                 "Bq", "Wo", "Wab", "Wout")


def _is_plain_ln(ln, n, want_weight=True, want_bias=False):
    return (isinstance(getattr(ln, "normalized_shape", None), tuple)
            and ln.normalized_shape == (n,)
            and float(ln.eps) == 1e-5
            and (ln.weight is not None) == want_weight
            and (ln.bias is not None) == want_bias)


# The one configuration the fused kernels are specialized for (what the
# captures use).  Anything else falls through to the reference path.
_C_ATOM, _C_Z, _C_HID, _N_HEAD, _N_TRANS = 128, 16, 32, 4, 2
# Reduction widths the fused token-projection kernel is instantiated for, and the
# bounds the fixed-size workspace regions impose.
_TOK_K = (128, 256, 384, 512, 768)
_MAX_ATOM, _MAX_TOKEN = 4096, 256


def _pack_stack(tf, c_atom):
    """Pack a ``DiffusionTransformer`` for the fused path, or return None."""
    c_z, c_hidden, no_heads, n_transition = _C_Z, _C_HID, _N_HEAD, _N_TRANS
    blocks = getattr(tf, "blocks", None)
    if blocks is None or len(blocks) == 0 or not getattr(tf, "use_cross_attention", False):
        return None
    if not _is_plain_ln(getattr(tf, "layer_norm_z", None), c_z):
        return None
    ln_s_ref = None
    rows_n, bias_n, sig_n, rows_r, bias_r = [], [], [], [], []
    Wz, Wqkvg, Bq, Wo, Wab, Wout = [], [], [], [], [], []
    for blk in blocks:
        apb = getattr(blk, "attention_pair_bias", None)
        ct = getattr(blk, "conditioned_transition", None)
        if apb is None or ct is None:
            return None
        if not (getattr(apb, "use_ada_layer_norm", False)
                and getattr(apb, "n_query", None) == 32
                and getattr(apb, "n_key", None) == 128
                and float(getattr(apb, "inf", 0.0)) == 1e9):
            return None
        mha = getattr(apb, "mha", None)
        if mha is None or mha.linear_g is None or mha.linear_q.bias is None:
            return None
        if (mha.c_hidden != c_hidden or mha.no_heads != no_heads
                or mha.linear_o.weight.shape != (c_atom, c_atom)
                or apb.c_z != c_z):
            return None
        adalns = (apb.layer_norm_a_q, apb.layer_norm_a_k, ct.layer_norm)
        for ada in adalns:
            # layer_norm_a must be the bare normalization (create_scale=False)
            if not _is_plain_ln(ada.layer_norm_a, c_atom, want_weight=False):
                return None
            if not _is_plain_ln(ada.layer_norm_s, c_atom):
                return None
            if ln_s_ref is None:
                ln_s_ref = ada.layer_norm_s.weight
            elif not torch.equal(ada.layer_norm_s.weight, ln_s_ref):
                # One shared normalized operand for the fused AdaLN GEMM is only
                # valid when every layer_norm_s agrees.
                return None
        for ada in adalns:
            rows_n += [ada.linear_g.weight, ada.linear_s.weight]
            bias_n += [ada.linear_g.bias,
                       torch.zeros_like(ada.linear_s.weight[:, 0])]
            sig_n += [1, 0]
        rows_r += [apb.linear_ada_out.weight, ct.linear_g.weight]
        bias_r += [apb.linear_ada_out.bias, ct.linear_g.bias]
        Wz.append(apb.linear_z.weight)
        Wqkvg.append(torch.cat([mha.linear_q.weight, mha.linear_g.weight,
                                mha.linear_k.weight, mha.linear_v.weight], 0))
        Bq.append(mha.linear_q.bias)
        Wo.append(mha.linear_o.weight)
        Wab.append(torch.cat([ct.swiglu.linear_a.weight,
                              ct.swiglu.linear_b.weight], 0))
        Wout.append(ct.linear_out.weight)
    ch = n_transition * c_atom
    if Wab[0].shape != (2 * ch, c_atom) or Wout[0].shape != (c_atom, ch):
        return None
    if Wz[0].shape != (no_heads, c_z):
        return None

    # The AdaLN GEMM's column order is [per block: g_q, s_q, g_k, s_k, g_t, s_t]
    # from layer_norm_s(s), then [per block: ada_out, transition gate] from raw s.
    order_n, order_b, order_s = [], [], []
    for i in range(len(blocks)):
        for g in range(_NGRP_N):
            order_n.append(rows_n[i * _NGRP_N + g])
            order_b.append(bias_n[i * _NGRP_N + g])
            order_s.append(sig_n[i * _NGRP_N + g])
    for i in range(len(blocks)):
        for g in range(_NGRP_R):
            order_n.append(rows_r[i * _NGRP_R + g])
            order_b.append(bias_r[i * _NGRP_R + g])
            order_s.append(1)

    p = _StackPack()
    p.nblk = len(blocks)
    dev = ln_s_ref.device
    p.wlnz = tf.layer_norm_z.weight.detach().contiguous()
    p.wlns = ln_s_ref.detach().contiguous()
    p.Wz = torch.stack([w.detach() for w in Wz]).contiguous()
    p.Wsp = torch.cat([w.detach() for w in order_n], 0).contiguous()
    p.Bsp = torch.cat([b.detach().float() for b in order_b], 0).contiguous()
    p.Sig = torch.tensor(
        [s for s in order_s for _ in range(c_atom)], dtype=torch.uint8, device=dev)
    p.Wqkvg = torch.stack([w.detach() for w in Wqkvg]).contiguous()
    p.Bq = torch.stack([b.detach().float() for b in Bq]).contiguous()
    p.Wo = torch.stack([w.detach() for w in Wo]).contiguous()
    p.Wab = torch.stack([w.detach() for w in Wab]).contiguous()
    p.Wout = torch.stack([w.detach() for w in Wout]).contiguous()
    return p


_WS_CACHE: dict[tuple, torch.Tensor] = {}


def _workspace(n_atom, n_blocks, nblk, device):
    """Scratch buffer, cached per (shape, device).  Pure scratch: every kernel
    writes its region before reading it, so reuse across calls is safe and saves
    an allocation on a path where an allocation is a measurable fraction."""
    key = (n_atom, n_blocks, nblk, device)
    ws = _WS_CACHE.get(key)
    if ws is None:
        nbytes = _EXT.ws_bytes(n_atom, n_blocks, nblk)
        ws = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _WS_CACHE[key] = ws
    return ws


def _unit_batch(t, ndim_feat):
    """True if every leading (batch) dim of *t* is 1."""
    return all(s == 1 for s in t.shape[:-ndim_feat])


def _fk_cached(mod, sentinel, builder):
    """Cache *builder*'s result on *mod*, invalidated when *sentinel*'s storage or
    version changes (an in-place ``load_state_dict``, a ``.to(dtype)``)."""
    key = (sentinel.data_ptr(), sentinel._version)
    if getattr(mod, "_fk_key", None) == key:
        return mod._fk_val
    val = builder()
    mod._fk_key = key
    mod._fk_val = val
    return val


def _pack_decoder(mod):
    """Packed weights for AtomAttentionDecoder's fused path, or None."""
    try:
        w = mod.linear_q_in.weight
        if w.shape[0] != _C_ATOM or w.shape[1] not in _TOK_K:
            return None
        sp = _pack_stack(mod.atom_transformer, _C_ATOM)
        if sp is None or not _is_plain_ln(mod.layer_norm, _C_ATOM):
            return None
        return (sp, w.detach().contiguous(),
                mod.layer_norm.weight.detach().contiguous(),
                mod.linear_q_out.weight.detach().contiguous())
    except Exception:  # noqa: BLE001 - an unexpected module tree just falls back
        return None


def _fk_common_ok(ql, cl, plm, am):
    """Shape / dtype / layout gate for AtomAttentionDecoder's fused path."""
    if _EXT is None or torch.is_grad_enabled():
        return None
    if not (ql.is_cuda and ql.dtype is torch.bfloat16 and ql.is_contiguous()):
        return None
    for t in (cl, plm, am):
        if (t.dtype is not torch.bfloat16 or not t.is_cuda
                or not t.is_contiguous()):
            return None
    if ql.shape[-1] != _C_ATOM or cl.shape[-1] != _C_ATOM or plm.dim() < 5:
        return None
    nb, nq, nk, cz = plm.shape[-4:]
    if (nq, nk, cz) != (32, 128, _C_Z):
        return None
    n_atom = am.shape[-1]
    if (am.numel() != n_atom or n_atom > _MAX_ATOM or nb != -(-n_atom // nq)
            or ql.shape[-2] != n_atom or cl.shape[-2] != n_atom):
        return None
    if not (_unit_batch(ql, 2) and _unit_batch(cl, 2) and _unit_batch(plm, 4)):
        return None
    return n_atom, nb


def _pack_encoder(mod):
    """Packed weights for AtomAttentionEncoder's fused path, or None."""
    try:
        sp = _pack_stack(mod.atom_transformer, _C_ATOM)
        if sp is None:
            return None
        e = mod.ref_atom_feature_embedder
        c_elem = e.linear_ref_element.weight.shape[1]
        c_chars = e.linear_ref_atom_chars.weight.shape[1]
        kcl = 5 + c_elem + c_chars
        kcl_pad = (kcl + 127) // 128 * 128
        if kcl_pad not in (128, 256, 384, 512, 768):
            return None
        rows = [e.linear_ref_pos.weight, e.linear_ref_charge.weight,
                e.linear_ref_mask.weight, e.linear_ref_element.weight,
                e.linear_ref_atom_chars.weight]
        if any(r.shape[0] != _C_ATOM for r in rows):
            return None
        Wcl = torch.zeros(_C_ATOM, kcl_pad, dtype=rows[0].dtype,
                          device=rows[0].device)
        col = 0
        for r in rows:
            Wcl[:, col:col + r.shape[1]] = r.detach()
            col += r.shape[1]
        # linear_ref_offset | linear_inv_sq_dists | linear_valid_mask, as fp32
        if (e.linear_ref_offset.weight.shape != (_C_Z, 3)
                or e.linear_inv_sq_dists.weight.shape != (_C_Z, 1)
                or e.linear_valid_mask.weight.shape != (_C_Z, 1)):
            return None
        Wpsc = torch.cat([e.linear_ref_offset.weight,
                          e.linear_inv_sq_dists.weight,
                          e.linear_valid_mask.weight], 1).detach().float().contiguous()
        mlp = [m for m in mod.pair_mlp if isinstance(m, Linear)]
        if len(mlp) != 3 or any(m.weight.shape != (_C_Z, _C_Z) for m in mlp):
            return None
        Wmlp = torch.stack([m.weight.detach() for m in mlp]).contiguous()
        if (mod.linear_l.weight.shape != (_C_Z, _C_ATOM)
                or mod.linear_m.weight.shape != (_C_Z, _C_ATOM)):
            return None
        WLM = torch.cat([mod.linear_l.weight, mod.linear_m.weight],
                        0).detach().contiguous()
        lq = mod.linear_q[0]
        if (lq.bias is not None or lq.weight.shape[1] != _C_ATOM
                or lq.weight.shape[0] % _C_ATOM or lq.weight.shape[0] > 1024):
            return None
        Wq = lq.weight.detach().contiguous()

        npe = mod.noisy_position_embedder
        empty = Wq.new_empty(0)
        noisy = npe is not None
        Wsi = wln_si = wln_z = Wzt = Wr = empty
        if noisy:
            c_s = npe.linear_s.weight.shape[1]
            if (c_s not in _TOK_K or npe.layer_norm_z.weight is None
                    or npe.linear_z.weight.shape != (_C_Z, _C_ATOM)
                    or npe.linear_r.weight.shape != (_C_ATOM, 3)
                    or not _is_plain_ln(npe.layer_norm_s, c_s)
                    or not _is_plain_ln(npe.layer_norm_z, _C_ATOM)):
                return None
            Wsi = npe.linear_s.weight.detach().contiguous()
            wln_si = npe.layer_norm_s.weight.detach().contiguous()
            wln_z = npe.layer_norm_z.weight.detach().contiguous()
            Wzt = npe.linear_z.weight.detach().contiguous()
            Wr = npe.linear_r.weight.detach().contiguous()
        return (sp, Wcl, Wpsc, Wmlp, WLM, Wq, Wsi, wln_si, wln_z, Wzt, Wr,
                noisy, c_elem, c_chars, empty)
    except Exception:  # noqa: BLE001 - an unexpected module tree just falls back
        return None


def _get_block_key_indices(
    atom_mask: torch.Tensor, n_query: int, n_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized computation of key-block gather indices.

    Returns:
        safe_indices: [*, N_blocks, n_key] clamped indices
        invalid_mask: [*, N_blocks, n_key] True where index is out of range
    """
    batch_dims = atom_mask.shape[:-1]
    n_atom = atom_mask.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    device = atom_mask.device
    offset = n_query // 2

    subset_centers = offset + torch.arange(num_blocks, device=device) * n_query
    subset_centers = subset_centers.reshape(*(1,) * len(batch_dims), num_blocks)
    subset_centers = subset_centers.expand(*batch_dims, num_blocks)

    n_real = atom_mask.sum(dim=-1, keepdim=True).expand(*batch_dims, num_blocks)

    initial = (
        subset_centers.unsqueeze(-1)
        + torch.arange(-n_key // 2, n_key // 2, device=device)
    ).int()

    underflow = torch.relu(-initial[..., 0])
    overflow = torch.relu(initial[..., -1] - (n_real - 1))
    total_shift = torch.where(underflow > 0, underflow, -overflow)
    final = initial + total_shift.unsqueeze(-1)

    n_real_exp = n_real.unsqueeze(-1)
    invalid = (final < 0) | (final >= n_real_exp)
    safe = torch.clamp(final, torch.zeros_like(n_real_exp), (n_real_exp - 1).clamp(min=0))

    return safe.long(), invalid


def _convert_single_rep_to_blocks(
    ql: torch.Tensor,
    n_query: int,
    n_key: int,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert flat atom representation to windowed block format (vectorized).

    Args:
        ql: [*, N_atom, C] atom features
        n_query: block height
        n_key: block width
        atom_mask: [*, N_atom] mask

    Returns:
        ql_query: [*, N_blocks, n_query, C]
        ql_key:   [*, N_blocks, n_key, C]
        mask_blocks: [*, N_blocks, n_query, n_key] or None
    """
    batch_dims = ql.shape[:-2]
    n_atom, c = ql.shape[-2], ql.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    pad_q = (-n_atom) % n_query

    if pad_q > 0:
        ql = Pad()(ql, (0, 0, 0, pad_q))
        if atom_mask is not None:
            atom_mask = Pad()(atom_mask, (0, pad_q))

    ql_query = ql.reshape(*batch_dims, num_blocks, n_query, c)

    if atom_mask is None:
        atom_mask = ql.new_ones(*batch_dims, n_atom + pad_q)

    atom_mask = atom_mask.expand(*batch_dims, -1)
    key_indices, invalid_mask = _get_block_key_indices(atom_mask, n_query, n_key)

    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
    ql_flat = ql.reshape(flat_batch, n_atom + pad_q, c)
    idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, c)

    ql_key_flat = torch.gather(ql_flat, 1, idx_expanded)
    mask_flat = invalid_mask.reshape(flat_batch, num_blocks * n_key).unsqueeze(-1).expand(-1, -1, c)
    ql_key_flat.masked_fill_(mask_flat, 0.0)
    ql_key = ql_key_flat.reshape(*batch_dims, num_blocks, n_key, c)

    mask_q = atom_mask.reshape(*batch_dims, num_blocks, n_query)
    mask_k_valid = (~invalid_mask).to(atom_mask.dtype)
    atom_mask_at_keys = torch.gather(
        atom_mask.reshape(flat_batch, -1), 1,
        idx_flat,
    ).reshape(*batch_dims, num_blocks, n_key)
    mask_k_valid = mask_k_valid * atom_mask_at_keys
    mask_blocks = mask_q.unsqueeze(-1) * mask_k_valid.unsqueeze(-2)

    return ql_query, ql_key, mask_blocks


_apply_block_indices = _convert_single_rep_to_blocks


def _convert_pair_rep_to_blocks(
    batch: dict,
    zij_trunk: torch.Tensor,
    n_query: int,
    n_key: int,
) -> torch.Tensor:
    """Convert pair representation to block format for atom attention (vectorized).

    Args:
        batch: needs atom_mask, atom_to_token_index
        zij_trunk: [*, N_token, N_token, C_z]
        n_query: block height
        n_key: block width

    Returns:
        [*, N_blocks, n_query, n_key, C_z]
    """
    atom_mask = batch["atom_mask"]
    n_atoms = atom_mask.shape[-1]
    batch_dims = zij_trunk.shape[:-3]
    c_z = zij_trunk.shape[-1]

    if "atom_to_token_index" in batch:
        atom_to_token = batch["atom_to_token_index"]
        if atom_to_token.dim() > 1:
            atom_to_token = atom_to_token[0]
    else:
        n_token = zij_trunk.shape[-2]
        atom_to_token = torch.arange(n_token, device=zij_trunk.device)
        if n_atoms > n_token:
            atom_to_token = atom_to_token.repeat_interleave(
                (n_atoms + n_token - 1) // n_token
            )[:n_atoms]

    num_blocks = math.ceil(n_atoms / n_query)
    pad_q = (-n_atoms) % n_query

    atk_padded = Pad()(atom_to_token, (0, pad_q))
    q_indices = atk_padded.reshape(num_blocks, n_query)

    atom_mask_exp = atom_mask.expand(*batch_dims, -1)
    key_indices, invalid_mask = _get_block_key_indices(atom_mask_exp, n_query, n_key)

    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1

    atk_flat = atom_to_token.expand(flat_batch, -1)
    key_idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    k_token_flat = torch.gather(atk_flat, 1, key_idx_flat.clamp(min=0, max=n_atoms - 1))
    k_indices = k_token_flat.reshape(flat_batch, num_blocks, n_key)

    zij_flat = zij_trunk.reshape(flat_batch, *zij_trunk.shape[-3:])
    batch_idx = torch.arange(flat_batch, device=zij_trunk.device).view(-1, 1, 1, 1)
    q_idx = q_indices.long().unsqueeze(0).expand(flat_batch, -1, -1)

    plm = zij_flat[batch_idx, q_idx.unsqueeze(-1), k_indices.unsqueeze(-2)]

    inv_expanded = invalid_mask.reshape(flat_batch, num_blocks, n_key)
    plm.masked_fill_(inv_expanded[:, :, None, :, None].expand_as(plm), 0.0)

    pair_mask = _get_pair_atom_block_mask(
        atom_mask=atom_mask_exp, num_blocks=num_blocks,
        n_query=n_query, n_key=n_key, pad_q=pad_q,
        key_indices=key_indices, invalid_mask=invalid_mask,
    )
    plm = plm * pair_mask.reshape(flat_batch, num_blocks, n_query, n_key, 1)
    plm = plm.reshape(*batch_dims, num_blocks, n_query, n_key, c_z)

    return plm


def _get_pair_atom_block_mask(
    atom_mask: torch.Tensor,
    num_blocks: int,
    n_query: int,
    n_key: int,
    pad_q: int,
    key_indices: torch.Tensor,
    invalid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute pair atom block mask."""
    batch_dims = atom_mask.shape[:-1]
    flat_batch = int(math.prod(batch_dims)) if batch_dims else 1
    mask_flat = atom_mask.reshape(flat_batch, -1)

    mask_padded = Pad()(mask_flat, (0, pad_q))
    mask_q = mask_padded.reshape(flat_batch, num_blocks, n_query)

    idx_flat = key_indices.reshape(flat_batch, num_blocks * n_key)
    mask_k_vals = torch.gather(mask_flat, 1, idx_flat.clamp(min=0, max=mask_flat.shape[-1] - 1))
    mask_k = mask_k_vals.reshape(flat_batch, num_blocks, n_key)
    inv_flat = invalid_mask.reshape(flat_batch, num_blocks, n_key)
    mask_k = mask_k * (~inv_flat).to(mask_k.dtype)

    pair_mask = mask_q.unsqueeze(-1) * mask_k.unsqueeze(-2)
    return pair_mask.reshape(*batch_dims, num_blocks, n_query, n_key)


def _broadcast_token_feat_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor | None,
    token_feat: torch.Tensor,
    atom_to_token_index: torch.Tensor | None = None,
    n_atoms: int | None = None,
) -> torch.Tensor:
    """Broadcast token-level features to atom-level.

    Args:
        token_mask: [*, N_token]
        num_atoms_per_token: [*, N_token] or None
        token_feat: [*, N_token, C]
        atom_to_token_index: [*, N_atom] optional direct mapping
        n_atoms: total number of atoms if atom_to_token_index not provided

    Returns:
        [*, N_atom, C]
    """
    if atom_to_token_index is not None:
        idx = atom_to_token_index.long()
        while idx.dim() < token_feat.dim() - 1:
            idx = idx.unsqueeze(1)
        idx = idx.expand(*token_feat.shape[:-2], idx.shape[-1])
        return torch.gather(
            token_feat, -2,
            idx.unsqueeze(-1).expand(*idx.shape, token_feat.shape[-1]),
        )

    if num_atoms_per_token is not None:
        return torch.repeat_interleave(
            token_feat, num_atoms_per_token.long(), dim=-2,
        )

    return token_feat


def _aggregate_atom_feat_to_tokens(
    token_mask: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_feat: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Aggregate atom-level features to token-level.

    Args:
        token_mask: [*, N_token]
        atom_to_token_index: [N_atom]
        atom_mask: [*, N_atom]
        atom_feat: [*, N_atom, C]
        mode: "mean" or "sum"

    Returns:
        [*, N_token, C]
    """
    n_token = token_mask.shape[-1]
    c = atom_feat.shape[-1]
    batch_shape = atom_feat.shape[:-2]

    atom_mask_expanded = atom_mask.expand(*batch_shape, -1)

    result = atom_feat.new_zeros(*batch_shape, n_token, c)
    masked_feat = atom_feat * atom_mask_expanded[..., None]

    idx = atom_to_token_index.long().expand(*batch_shape, -1)
    result.scatter_add_(-2, idx.unsqueeze(-1).expand_as(masked_feat), masked_feat)

    if mode == "mean":
        counts = torch.zeros(*batch_shape, n_token, dtype=result.dtype, device=result.device)
        counts.scatter_add_(-1, idx, atom_mask_expanded.to(dtype=result.dtype))
        counts = counts.clamp(min=1.0)
        result = result / counts.unsqueeze(-1)

    return result


__targets__ = ["AtomAttentionEncoder", "AtomAttentionDecoder"]


class RefAtomFeatureEmbedder(nn.Module):
    """Embeds reference atom features (Algorithm 5, lines 1-6).

    Args:
        c_atom_ref_element: Reference element one-hot dim (119)
        c_atom_ref_name_chars: Reference atom name chars dim (256 = 4*64)
        c_atom: Atom single conditioning dim
        c_atom_pair: Atom pair conditioning dim
    """

    def __init__(
        self,
        c_atom_ref_element: int = 119,
        c_atom_ref_name_chars: int = 256,
        c_atom: int = 128,
        c_atom_pair: int = 16,
    ):
        super().__init__()
        self.linear_ref_pos = Linear(3, c_atom, bias=False)
        self.linear_ref_charge = Linear(1, c_atom, bias=False)
        self.linear_ref_mask = Linear(1, c_atom, bias=False)
        self.linear_ref_element = Linear(c_atom_ref_element, c_atom, bias=False)
        self.linear_ref_atom_chars = Linear(c_atom_ref_name_chars, c_atom, bias=False)
        self.linear_ref_offset = Linear(3, c_atom_pair, bias=False)
        self.linear_inv_sq_dists = Linear(1, c_atom_pair, bias=False)
        self.linear_valid_mask = Linear(1, c_atom_pair, bias=False)

    def forward(
        self,
        batch: dict,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = batch["ref_pos"].dtype

        cl = self.linear_ref_pos(batch["ref_pos"])
        cl = cl + self.linear_ref_charge(
            torch.arcsinh(batch["ref_charge"].unsqueeze(-1))
        )
        cl = cl + self.linear_ref_mask(batch["ref_mask"].unsqueeze(-1).to(dtype=dtype))
        cl = cl + self.linear_ref_element(batch["ref_element"].to(dtype=dtype))
        cl = cl + self.linear_ref_atom_chars(
            batch["ref_atom_name_chars"].flatten(start_dim=-2).to(dtype=dtype)
        )

        d_l, d_m, atom_mask = _convert_single_rep_to_blocks(
            ql=batch["ref_pos"],
            n_query=n_query, n_key=n_key,
            atom_mask=batch["atom_mask"],
        )
        v_l, v_m, _ = _convert_single_rep_to_blocks(
            ql=batch["ref_space_uid"].unsqueeze(-1),
            n_query=n_query, n_key=n_key,
            atom_mask=batch["atom_mask"],
        )

        dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * atom_mask.unsqueeze(-1)
        vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).to(
            dtype=dlm.dtype
        ) * atom_mask.unsqueeze(-1)

        plm = self.linear_ref_offset(dlm) * vlm

        inv_sq_dists = 1.0 / (1 + torch.sum(dlm ** 2, dim=-1, keepdim=True))
        plm = plm + self.linear_inv_sq_dists(inv_sq_dists) * vlm
        plm = plm + self.linear_valid_mask(vlm) * vlm

        return cl, plm


class NoisyPositionEmbedder(nn.Module):
    """Embeds noisy positions and trunk embeddings (Algorithm 5, lines 8-12).

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_atom: Atom single conditioning channel dimension
        c_atom_pair: Atom pair conditioning channel dimension
    """

    def __init__(self, c_s: int, c_z: int, c_atom: int, c_atom_pair: int):
        super().__init__()
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(c_s, c_atom, bias=False)
        self.layer_norm_z = LayerNorm(c_z, create_offset=False)
        self.linear_z = Linear(c_z, c_atom_pair, bias=False)
        self.linear_r = Linear(3, c_atom, bias=False)

    def forward(
        self,
        batch: dict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        rl: torch.Tensor,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        si_trunk_proj = self.linear_s(self.layer_norm_s(si_trunk))
        si_trunk_proj = _broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch.get("num_atoms_per_token"),
            token_feat=si_trunk_proj,
            atom_to_token_index=batch.get("atom_to_token_index"),
        )
        cl = cl + si_trunk_proj

        zij_trunk_proj = self.linear_z(self.layer_norm_z(zij_trunk))
        zij_trunk_block = _convert_pair_rep_to_blocks(
            batch=batch, zij_trunk=zij_trunk_proj,
            n_query=n_query, n_key=n_key,
        )
        plm = plm + zij_trunk_block

        ql = cl + self.linear_r(rl)

        return cl, plm, ql


class AtomAttentionEncoder(nn.Module):
    """AF3 Algorithm 5: Atom attention encoder.

    Args:
        c_atom: Atom single representation channel dimension
        c_atom_pair: Atom pair representation channel dimension
        c_token: Token single representation output channel dimension
        c_atom_ref_element: Reference element one-hot dim
        c_atom_ref_name_chars: Reference atom name chars dim
        add_noisy_pos: Whether to embed noisy positions and trunk reps
        c_s: Single representation dim (optional, needed if add_noisy_pos)
        c_z: Pair representation dim (optional, needed if add_noisy_pos)
        c_hidden: Per-head hidden dim for atom transformer
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition blocks per transformer block
        n_query: Block height for sequence-local attention
        n_key: Block width for sequence-local attention
        use_ada_layer_norm: Whether to use AdaLN
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int = 384,
        c_atom_ref_element: int = 119,
        c_atom_ref_name_chars: int = 256,
        add_noisy_pos: bool = False,
        c_s: int | None = None,
        c_z: int | None = None,
        c_hidden: int = 32,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        n_query: int = 32,
        n_key: int = 128,
        use_ada_layer_norm: bool = True,
        transformer_cls=None,
    ):
        super().__init__()
        if not _LOADED:
            _load()
        if transformer_cls is None:
            from ..L3.alphafold3_diffusion_transformer import DiffusionTransformer
            transformer_cls = DiffusionTransformer

        self.n_query = n_query
        self.n_key = n_key

        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            c_atom_ref_element=c_atom_ref_element,
            c_atom_ref_name_chars=c_atom_ref_name_chars,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
        )

        self.noisy_position_embedder: NoisyPositionEmbedder | None = None
        if add_noisy_pos:
            assert c_s is not None and c_z is not None
            self.noisy_position_embedder = NoisyPositionEmbedder(
                c_s=c_s, c_z=c_z, c_atom=c_atom, c_atom_pair=c_atom_pair,
            )

        self.relu = ReLU()
        self.linear_l = Linear(c_atom, c_atom_pair, bias=False)
        self.linear_m = Linear(c_atom, c_atom_pair, bias=False)

        self.pair_mlp = nn.Sequential(
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
            ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False),
        )

        self.atom_transformer = transformer_cls(
            c_a=c_atom, c_s=c_atom, c_z=c_atom_pair,
            c_hidden=c_hidden, no_heads=no_heads,
            no_blocks=no_blocks, n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query, n_key=n_key,
        )

        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, bias=False),
            ReLU(),
        )

    def _fused_forward(self, batch, rl, si_trunk, zij_trunk):
        """One C++ call for the whole encoder, or None if this config is not claimed."""
        if _EXT is None or torch.is_grad_enabled():
            return None
        need = ("atom_mask", "atom_to_token_index", "token_mask", "ref_pos",
                "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
                "ref_space_uid")
        if any(k not in batch for k in need):
            return None
        am, a2t = batch["atom_mask"], batch["atom_to_token_index"]
        pos, uid = batch["ref_pos"], batch["ref_space_uid"]
        chars = batch["ref_atom_name_chars"]
        feats = (pos, batch["ref_charge"], batch["ref_mask"],
                 batch["ref_element"], chars, uid, am)
        if any(t.dtype is not torch.bfloat16 or not t.is_cuda or not t.is_contiguous()
               for t in feats):
            return None
        n_atom = am.shape[-1]
        nb = -(-n_atom // self.n_query)
        if (self.n_query, self.n_key) != (32, 128) or n_atom > _MAX_ATOM:
            return None
        if not (am.numel() == n_atom and pos.shape[-2:] == (n_atom, 3)
                and uid.numel() == n_atom and _unit_batch(pos, 2)):
            return None
        idx = a2t.reshape(-1)
        if idx.numel() != n_atom or idx.dtype is not torch.int64:
            return None
        packed = _fk_cached(self, self.linear_l.weight,
                            lambda: _pack_encoder(self))
        if packed is None:
            return None
        (sp, Wcl, Wpsc, Wmlp, WLM, Wq, Wsi, wln_si, wln_z, Wzt, Wr, noisy,
         c_elem, c_chars, empty) = packed
        if noisy != (rl is not None and self.noisy_position_embedder is not None):
            return None
        n_tok = batch["token_mask"].shape[-1]
        if n_tok > _MAX_TOKEN:
            return None
        lead_cl = pos.shape[:-2]
        lead_q = lead_cl
        if noisy:
            if not all(t is not None and t.dtype is torch.bfloat16 and t.is_cuda
                       and t.is_contiguous() for t in (rl, si_trunk, zij_trunk)):
                return None
            if not (rl.shape[-2:] == (n_atom, 3) and _unit_batch(rl, 2)
                    and si_trunk.shape[-2:] == (n_tok, wln_si.shape[0])
                    and _unit_batch(si_trunk, 2)
                    and zij_trunk.shape[-3:] == (n_tok, n_tok, _C_ATOM)
                    and _unit_batch(zij_trunk, 3)):
                return None
            lead_q = rl.shape[:-2]
        dev, c_tok = am.device, Wq.shape[0]
        cl_out = torch.empty(lead_cl + (n_atom, _C_ATOM), dtype=torch.bfloat16,
                             device=dev)
        ql_out = torch.empty(lead_q + (n_atom, _C_ATOM), dtype=torch.bfloat16,
                             device=dev)
        plm_out = torch.empty(lead_cl + (nb, 32, 128, _C_Z),
                              dtype=torch.bfloat16, device=dev)
        ai_out = torch.empty(lead_q + (n_tok, c_tok), dtype=torch.bfloat16,
                             device=dev)
        _EXT.encoder_forward(
            ai_out.view(n_tok, c_tok), ql_out.view(n_atom, _C_ATOM),
            cl_out.view(n_atom, _C_ATOM), plm_out.view(nb, 32, 128, _C_Z),
            _workspace(n_atom, nb, sp.nblk, dev), pos.view(n_atom, 3),
            batch["ref_charge"].view(-1), batch["ref_mask"].view(-1),
            batch["ref_element"].view(n_atom, c_elem),
            chars.view(n_atom, c_chars), uid.view(-1), am.view(-1), idx,
            rl.view(n_atom, 3) if noisy else empty,
            si_trunk.view(n_tok, -1) if noisy else empty,
            zij_trunk.view(-1, _C_ATOM) if noisy else empty,
            Wcl, Wsi, wln_si, wln_z, Wzt, Wr, WLM, Wpsc, Wmlp, sp.wlnz, sp.Wz,
            sp.Wsp, sp.Bsp, sp.Sig, sp.wlns, sp.Wqkvg, sp.Bq, sp.Wo, sp.Wab,
            sp.Wout, Wq, sp.nblk, n_tok, noisy)
        return ai_out, ql_out, cl_out, plm_out

    def forward(
        self,
        batch: dict,
        rl: torch.Tensor | None = None,
        si_trunk: torch.Tensor | None = None,
        zij_trunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ai: [*, N_token, c_token] token representation
            ql: [*, N_atom, c_atom] atom single representation
            cl: [*, N_atom, c_atom] atom single conditioning
            plm: [*, N_blocks, n_query, n_key, c_atom_pair] atom pair rep
        """
        fused = self._fused_forward(batch, rl, si_trunk, zij_trunk)
        if fused is not None:
            return fused

        atom_mask = batch["atom_mask"]

        cl, plm = self.ref_atom_feature_embedder(
            batch=batch, n_query=self.n_query, n_key=self.n_key,
        )

        if rl is not None and self.noisy_position_embedder is not None:
            cl, plm, ql = self.noisy_position_embedder(
                batch=batch, cl=cl, plm=plm,
                si_trunk=si_trunk, zij_trunk=zij_trunk, rl=rl,
                n_query=self.n_query, n_key=self.n_key,
            )
        else:
            ql = cl.clone()

        cl_l, cl_m, block_mask = _convert_single_rep_to_blocks(
            ql=cl, n_query=self.n_query, n_key=self.n_key, atom_mask=atom_mask,
        )

        cl_lm = (
            self.linear_l(self.relu(cl_l.unsqueeze(-2)))
            + self.linear_m(self.relu(cl_m.unsqueeze(-3)))
        )
        if block_mask is not None:
            cl_lm = cl_lm * block_mask.unsqueeze(-1)

        plm = plm + cl_lm
        plm = plm + self.pair_mlp(plm)
        if block_mask is not None:
            plm = plm * block_mask.unsqueeze(-1)

        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=atom_mask,
        )

        ql = ql * atom_mask.unsqueeze(-1)

        atom_proj = self.linear_q(ql)

        if "atom_to_token_index" in batch:
            ai = _aggregate_atom_feat_to_tokens(
                token_mask=batch["token_mask"],
                atom_to_token_index=batch["atom_to_token_index"],
                atom_mask=atom_mask,
                atom_feat=atom_proj,
                mode="mean",
            )
        else:
            ai = atom_proj

        return ai, ql, cl, plm


class AtomAttentionDecoder(nn.Module):
    """AF3 Algorithm 6: Atom attention decoder.

    Args:
        c_atom: Atom single representation channel dimension
        c_atom_pair: Atom pair representation channel dimension
        c_token: Token diffusion channel dimension
        c_hidden: Per-head hidden dim
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition blocks per transformer block
        n_query: Block height
        n_key: Block width
        use_ada_layer_norm: Whether to use AdaLN
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int = 768,
        c_hidden: int = 32,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        n_query: int = 32,
        n_key: int = 128,
        use_ada_layer_norm: bool = True,
        transformer_cls=None,
    ):
        super().__init__()
        if not _LOADED:
            _load()
        if transformer_cls is None:
            from ..L3.alphafold3_diffusion_transformer import DiffusionTransformer
            transformer_cls = DiffusionTransformer

        self.linear_q_in = Linear(c_token, c_atom, bias=False)

        self.atom_transformer = transformer_cls(
            c_a=c_atom, c_s=c_atom, c_z=c_atom_pair,
            c_hidden=c_hidden, no_heads=no_heads,
            no_blocks=no_blocks, n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query, n_key=n_key,
        )

        self.layer_norm = LayerNorm(c_atom, create_offset=False)
        self.linear_q_out = Linear(c_atom, 3, bias=False)

    def _fused_forward(self, batch, ai, ql, cl, plm):
        """One C++ call for the whole decoder, or None if this config is not claimed."""
        am = batch.get("atom_mask")
        a2t = batch.get("atom_to_token_index")
        if am is None or a2t is None:
            return None
        common = _fk_common_ok(ql, cl, plm, am)
        if common is None:
            return None
        n_atom, nb = common
        if not (ai.dtype is torch.bfloat16 and ai.is_cuda and ai.is_contiguous()
                and _unit_batch(ai, 2) and ai.shape[-2] <= 512):
            return None
        packed = _fk_cached(self, self.linear_q_in.weight,
                            lambda: _pack_decoder(self))
        if packed is None or ai.shape[-1] != self.linear_q_in.weight.shape[1]:
            return None
        sp, Wqin, wlnfin, Wfin = packed
        idx = a2t.reshape(-1)
        if idx.numel() != n_atom or idx.dtype is not torch.int64:
            return None
        out = torch.empty(ql.shape[:-1] + (Wfin.shape[0],),
                          dtype=torch.bfloat16, device=ql.device)
        _EXT.decoder_forward(
            out.view(n_atom, -1), _workspace(n_atom, nb, sp.nblk, ql.device),
            ai.view(-1, ai.shape[-1]), ql.view(n_atom, _C_ATOM),
            cl.view(n_atom, _C_ATOM), plm.view(nb, 32, 128, _C_Z),
            am.view(-1), idx, Wqin, sp.wlnz, sp.Wz, sp.Wsp, sp.Bsp, sp.Sig,
            sp.wlns, sp.Wqkvg, sp.Bq, sp.Wo, sp.Wab, sp.Wout, wlnfin, Wfin,
            sp.nblk)
        return out

    def forward(
        self,
        batch: dict,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            rl_update: [*, N_atom, 3] atom position updates
        """
        fused = self._fused_forward(batch, ai, ql, cl, plm)
        if fused is not None:
            return fused

        ai_broadcast = _broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch.get("num_atoms_per_token"),
            token_feat=self.linear_q_in(ai),
            atom_to_token_index=batch.get("atom_to_token_index"),
        )
        ql = ql + ai_broadcast

        ql = self.atom_transformer(
            a=ql, s=cl, z=plm, mask=batch["atom_mask"],
        )

        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update
