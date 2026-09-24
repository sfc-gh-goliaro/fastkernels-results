"""SwiGLU activation and AdaLN for AlphaFold3 (L2 composites).

SwiGLU: SiLU(linear_a(x)) * linear_b(x)
AdaLN: Adaptive Layer Normalization

Reference: openfold3/core/model/primitives/activations.py SwiGLU
           openfold3/core/model/primitives/normalization.py AdaLN

Both composites are *launch bound* at the captured sizes.  The largest case here
is a pair of 16x768 @ 768x1536 matmuls -- 75 MFLOP, tens of nanoseconds of B200
tensor-core time -- while the harness measures several microseconds for every
dispatched torch op, in steps of about one kernel slot.  The baseline AdaLN
issues thirteen of them (each ``LayerNorm`` promotes to fp32, normalizes and
casts back; then two GEMMs, a sigmoid, an add and a multiply) and measures
~70 us; SwiGLU issues four and measures ~25 us.  So the lever that matters is
collapsing each composite into *one* kernel.

Two fused CUDA kernels do that, one per class.  Both compute the same shape --
``epilogue(X @ Wa^T, X @ Wb^T)`` with X either the input (SwiGLU) or
``LayerNorm(s)`` (AdaLN) -- and share one inner loop, ``accum``; the design notes
for the mapping, the access width and the tile choice are in the CUDA source,
where the code they explain lives.  The short version: k is split across the warp
so the row-major weight reads stay contiguous, the activation tile is staged in
shared memory as fp32 at an access width that is bank-conflict free, and the tile
shape is picked per call to land near 20 warps/SM (these grids are small enough
that occupancy, not traffic, binds).

Rounding is matched to the reference op by op -- at every point where torch would
have materialized a bf16 tensor (each LayerNorm output, each GEMM result, the
SiLU/sigmoid result, the inner add) the kernel rounds through bf16 too -- so the
only numerical difference left is fp32 accumulation order inside the dot
products.

The weights are read in place; nothing is concatenated or re-laid-out, so
``state_dict`` keys and shapes are exactly the baseline's (the harness shares
weights baseline -> candidate with ``load_state_dict``).  Anything the kernels do
not claim -- a dtype other than bf16, a non-contiguous input, a channel count
that is not a multiple of 8, a shape whose broadcast is not the ``a`` shape, or a
grad-enabled caller that needs a differentiable graph -- falls through to the
baseline chain.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

_CUDA_SRC = r"""// Fused AlphaFold3 SwiGLU / AdaLN: one launch each.
//
// Both composites are  O = epilogue(X @ Wa^T, X @ Wb^T)  with X either the input
// (SwiGLU) or LayerNorm(s) (AdaLN), at sizes where the whole thing is launch
// bound -- the baseline chains 4 and 13 torch ops respectively, and a dispatched
// op costs more than the arithmetic.  So both kernels have the same shape, and
// the only thing that matters about the inner loop is that it not waste memory
// pipeline slots.
//
// Mapping: k split across the warp
// --------------------------------
// The obvious mapping -- one thread per output, walking its own weight row --
// makes a warp read 32 rows of a row-major W[n][k] at one k, so every load
// scatters over 32 cache lines.  Measured: l1tex at 90% of peak with DRAM at
// 2%, and 6-10x slower than the torch chain.
//
// Here the *k* axis is split across the warp (``KS`` lanes over k, the remaining
// ``NS = 32 / KS`` over n), so the KS lanes of a group read KS consecutive
// 8-byte chunks of one weight row -- one contiguous burst.  Each lane keeps a
// partial dot product for every row of the tile; the partials fold with a
// segmented ``__shfl_xor`` at the end.  KS is chosen per call -- see the tile
// selection block below, which is where its trade-off against the reduction cost
// and against BM is written down.
//
// Granularity: 4 halves (8 bytes) per lane per step
// -------------------------------------------------
// The activation tile is staged in shared memory as fp32 -- it is re-read BM
// times per loaded weight chunk, so the bf16 -> fp32 conversion is paid once at
// stage time rather than in the inner loop.  Its *access width* then sets the
// bank behaviour: a lane owning W consecutive k values reads the staged row at a
// stride of W floats, so bank = (lane * W) % 32 and only 32 / W banks are ever
// touched.  W = 8 (a 16-byte weight load) therefore costs an 8-way conflict, and
// measured at that width the conflict *is* the kernel: 31.7k of 31.7k SM-active
// cycles.  W = 4 makes each lane's read one naturally aligned
// ``float4``, i.e. one distinct 4-bank group per lane, conflict free, and still
// leaves the weight load a contiguous 8 bytes per lane.
//
// Results go back through a small shared tile so the epilogue reads and writes
// (the second pass over ``a``, the gate bias, the store) are contiguous along n.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

using bf16 = __nv_bfloat16;

struct alignas(8) V4 {   // one weight chunk: 4 halves
  bf16 d[4];
};
struct alignas(16) V8 {  // widest load, used for the LayerNorm passes
  bf16 d[8];
};

// Round an fp32 value through bf16: reproduce a torch op that would have
// materialized a bf16 tensor at this point in the reference chain.
__device__ __forceinline__ float bf16r(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float wsum32(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

constexpr int NWARP = 8;   // warps per block

// acc[r] += x[r, c*4 .. c*4+4] . w  for every row of the tile.
template <int KS, int BM>
__device__ __forceinline__ void accum(const float *xs, int ldx, int kq, int ksub,
                                      const V4 *pa, const V4 *pb, float *aa,
                                      float *ab) {
  for (int c = ksub; c < kq; c += KS) {
    const V4 av = pa[c], bv = pb[c];
    float af[4], bv4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      af[j] = __bfloat162float(av.d[j]);
      bv4[j] = __bfloat162float(bv.d[j]);
    }
#pragma unroll
    for (int r = 0; r < BM; ++r) {
      // One aligned float4 per lane -> one bank group per lane, no conflict.
      const float4 x = *reinterpret_cast<const float4 *>(xs + (size_t)r * ldx + c * 4);
      aa[r] = fmaf(x.x, af[0], aa[r]);
      ab[r] = fmaf(x.x, bv4[0], ab[r]);
      aa[r] = fmaf(x.y, af[1], aa[r]);
      ab[r] = fmaf(x.y, bv4[1], ab[r]);
      aa[r] = fmaf(x.z, af[2], aa[r]);
      ab[r] = fmaf(x.z, bv4[2], ab[r]);
      aa[r] = fmaf(x.w, af[3], aa[r]);
      ab[r] = fmaf(x.w, bv4[3], ab[r]);
    }
  }
}

template <int KS, int BM>
__device__ __forceinline__ void reduce_pairs(float *aa, float *ab) {
  if (KS > 1) {
#pragma unroll
    for (int r = 0; r < BM; ++r) {
#pragma unroll
      for (int off = 1; off < KS; off <<= 1) {
        aa[r] += __shfl_xor_sync(0xffffffffu, aa[r], off);
        ab[r] += __shfl_xor_sync(0xffffffffu, ab[r], off);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SwiGLU:  O[m, n] = silu(X[m, :] . WA[n, :]) * (X[m, :] . WB[n, :])
// ---------------------------------------------------------------------------
template <int KS, int BM>
__global__ __launch_bounds__(NWARP * 32) void swiglu_k(
    const bf16 *__restrict__ X, const bf16 *__restrict__ WA,
    const bf16 *__restrict__ WB, bf16 *__restrict__ O, int M, int N, int K) {
  constexpr int NS = 32 / KS;
  constexpr int BN = NWARP * NS;
  constexpr int NTHR = NWARP * 32;

  extern __shared__ float sh[];
  float *xs = sh;                   // [BM][K]
  float *ob = sh + (size_t)BM * K;  // [2][BM][BN]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kq = K >> 2;

  // --- stage the activation tile as fp32; consecutive tids take consecutive
  //     chunks of one row, so the global read is contiguous
  for (int i = tid; i < BM * kq; i += NTHR) {
    const int r = i / kq, c = i - r * kq;
    const int mm = m0 + r;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (mm < M) {
      const V4 s = reinterpret_cast<const V4 *>(X + (size_t)mm * K)[c];
      v = make_float4(__bfloat162float(s.d[0]), __bfloat162float(s.d[1]),
                      __bfloat162float(s.d[2]), __bfloat162float(s.d[3]));
    }
    *reinterpret_cast<float4 *>(xs + (size_t)r * K + c * 4) = v;
  }
  __syncthreads();

  const int ksub = lane & (KS - 1), nsub = lane / KS;
  const int n = n0 + warp * NS + nsub;
  float aa[BM], ab[BM];
#pragma unroll
  for (int r = 0; r < BM; ++r) { aa[r] = 0.f; ab[r] = 0.f; }
  if (n < N) {
    accum<KS, BM>(xs, K, kq, ksub, reinterpret_cast<const V4 *>(WA + (size_t)n * K),
              reinterpret_cast<const V4 *>(WB + (size_t)n * K), aa, ab);
  }
  reduce_pairs<KS, BM>(aa, ab);
  if (ksub == 0) {
    const int col = warp * NS + nsub;
#pragma unroll
    for (int r = 0; r < BM; ++r) {
      ob[(size_t)r * BN + col] = aa[r];
      ob[(size_t)BM * BN + r * BN + col] = ab[r];
    }
  }
  __syncthreads();

  for (int i = tid; i < BM * BN; i += NTHR) {
    const int r = i / BN, c = i - r * BN;
    const int mm = m0 + r, nn = n0 + c;
    if (mm < M && nn < N) {
      // linear_a / linear_b each land in bf16; F.silu rounds again.
      const float ga = bf16r(ob[(size_t)r * BN + c]);
      const float gb = bf16r(ob[(size_t)BM * BN + r * BN + c]);
      const float sl = bf16r(ga / (1.f + __expf(-ga)));
      O[(size_t)mm * N + nn] = __float2bfloat16(sl * gb);
    }
  }
}

// ---------------------------------------------------------------------------
// AdaLN:  sn = LN(s) * lnw;  g = sigmoid(sn @ WG^T + BG);
//         O  = g * (LN(a) + sn @ WS^T)
// ---------------------------------------------------------------------------
template <int KS, int BM>
__global__ __launch_bounds__(NWARP * 32) void adaln_k(
    const bf16 *__restrict__ A, const bf16 *__restrict__ S,
    const bf16 *__restrict__ LNW, const bf16 *__restrict__ WG,
    const bf16 *__restrict__ BG, const bf16 *__restrict__ WS,
    bf16 *__restrict__ O, int M, int CA, int CS, float eps_s, float eps_a) {
  constexpr int NS = 32 / KS;
  constexpr int BN = NWARP * NS;
  constexpr int NTHR = NWARP * 32;

  extern __shared__ float sh[];
  float *xs = sh;                             // [BM][CS]   s_norm, fp32
  float *ob = sh + (size_t)BM * CS;            // [2][BM][BN]
  float *st = ob + (size_t)2 * BM * BN;        // [2][BM]    mean_a, rstd_a

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kvs = CS >> 3, kva = CA >> 3, kqs = CS >> 2;

  // --- phase A: warp `warp` owns row `warp` of the tile.  The rows are read
  //     straight from global each pass (coalesced, and L1-resident after the
  //     first), which keeps shared memory write-only here.
  for (int row = warp; row < BM; row += NWARP) {
    const int mm = m0 + row;
    float *xr = xs + (size_t)row * CS;
    if (mm >= M) {
      for (int c = lane; c < kqs; c += 32)
        *reinterpret_cast<float4 *>(xr + c * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (lane == 0) { st[row] = 0.f; st[BM + row] = 1.f; }
    } else {
      const V8 *sp = reinterpret_cast<const V8 *>(S + (size_t)mm * CS);
      float sum = 0.f;
      for (int c = lane; c < kvs; c += 32) {
        const V8 v = sp[c];
#pragma unroll
        for (int j = 0; j < 8; ++j) sum += __bfloat162float(v.d[j]);
      }
      const float mean = wsum32(sum) / (float)CS;
      float sq = 0.f;
      for (int c = lane; c < kvs; c += 32) {
        const V8 v = sp[c];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float t = __bfloat162float(v.d[j]) - mean;
          sq = fmaf(t, t, sq);
        }
      }
      const float rstd = rsqrtf(wsum32(sq) / (float)CS + eps_s);
      for (int c = lane; c < kqs; c += 32) {
        const V4 v = reinterpret_cast<const V4 *>(sp)[c];
        const V4 w = reinterpret_cast<const V4 *>(LNW)[c];
        float o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          o[j] = bf16r((__bfloat162float(v.d[j]) - mean) * rstd *
                       __bfloat162float(w.d[j]));
        *reinterpret_cast<float4 *>(xr + c * 4) = make_float4(o[0], o[1], o[2], o[3]);
      }
      // layer_norm_a statistics (scale- and offset-free)
      const V8 *ap = reinterpret_cast<const V8 *>(A + (size_t)mm * CA);
      float sa = 0.f;
      for (int c = lane; c < kva; c += 32) {
        const V8 v = ap[c];
#pragma unroll
        for (int j = 0; j < 8; ++j) sa += __bfloat162float(v.d[j]);
      }
      const float mean_a = wsum32(sa) / (float)CA;
      float sqa = 0.f;
      for (int c = lane; c < kva; c += 32) {
        const V8 v = ap[c];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float t = __bfloat162float(v.d[j]) - mean_a;
          sqa = fmaf(t, t, sqa);
        }
      }
      const float rsa = rsqrtf(wsum32(sqa) / (float)CA + eps_a);
      if (lane == 0) { st[row] = mean_a; st[BM + row] = rsa; }
    }
  }
  __syncthreads();

  const int ksub = lane & (KS - 1), nsub = lane / KS;
  const int n = n0 + warp * NS + nsub;
  float aa[BM], ab[BM];
#pragma unroll
  for (int r = 0; r < BM; ++r) { aa[r] = 0.f; ab[r] = 0.f; }
  if (n < CA) {
    accum<KS, BM>(xs, CS, kqs, ksub, reinterpret_cast<const V4 *>(WG + (size_t)n * CS),
              reinterpret_cast<const V4 *>(WS + (size_t)n * CS), aa, ab);
  }
  reduce_pairs<KS, BM>(aa, ab);
  if (ksub == 0) {
    const int col = warp * NS + nsub;
#pragma unroll
    for (int r = 0; r < BM; ++r) {
      ob[(size_t)r * BN + col] = aa[r];
      ob[(size_t)BM * BN + r * BN + col] = ab[r];
    }
  }
  __syncthreads();

  for (int i = tid; i < BM * BN; i += NTHR) {
    const int r = i / BN, c = i - r * BN;
    const int mm = m0 + r, nn = n0 + c;
    if (mm < M && nn < CA) {
      const float gp = bf16r(ob[(size_t)r * BN + c] + __bfloat162float(BG[nn]));
      const float g = bf16r(1.f / (1.f + __expf(-gp)));
      const float ls = bf16r(ob[(size_t)BM * BN + r * BN + c]);
      const float an =
          bf16r((__bfloat162float(A[(size_t)mm * CA + nn]) - st[r]) * st[BM + r]);
      O[(size_t)mm * CA + nn] = __float2bfloat16(g * bf16r(an + ls));
    }
  }
}

constexpr int64_t MAX_SHMEM = 48 * 1024;

// ---------------------------------------------------------------------------
// Tile selection
//
// KS (lanes cooperating on k) and BM (rows per tile) trade off against each
// other, and both were swept against every captured shape:
//
//   * KS wide  -> the weight burst is wide (32 lanes x 8 B contiguous) and few
//     column tiles, but the tail reduction costs BM*2*log2(KS) shuffles, and
//     SHFL retires at a quarter of the FFMA rate.  KS narrow -> the burst
//     degrades to 16-byte fragments at 50% sector use.  KS in [8, 16] is the
//     usable window; within it, leave each lane >= 4 chunks so the reduction
//     stays amortized (>= 2 when K is too small for that).
//   * BM large -> each loaded weight chunk feeds more rows (less weight
//     traffic), but a lane then carries 2*BM accumulators, which caps occupancy
//     by registers, and there are fewer blocks.  These grids are small enough
//     that occupancy, not traffic, is the binding constraint: profiling the
//     768x1536 SwiGLU at BM=8 showed 0.6 waves and 32% achieved occupancy,
//     stalled on L1TEX latency.  So BM is picked to land near 20 warps/SM.
// ---------------------------------------------------------------------------
inline int pick_ks(int64_t kq) {
  // Widest split <= 16 that leaves >= 4 chunks per lane; below 8 that is a bad
  // trade against sector efficiency, so relax the chunk count instead.
  for (int ks = 16; ks >= 2; ks >>= 1)
    if (kq % ks == 0 && kq / ks >= 4) {
      if (ks >= 8) return ks;
      break;
    }
  for (int ks = 8; ks >= 2; ks >>= 1)
    if (kq % ks == 0 && kq / ks >= 2) return ks;
  return 1;
}

// Rows per tile that put ~20 warps on each of the 148 SMs, rounded to a power of
// two.  The crossover is below the geometric midpoint: the cost curve is flat
// above the target and steep below it (too few rows per lane means the weights
// are re-read once more per row tile), so rounding up is the safer error.
inline int pick_bm(int64_t outputs, int ks) {
  const double want = (double)outputs * ks / (32.0 * 20.0 * 148.0);
  int bm = 2;
  while (bm < 16 && want > bm * 1.2) bm <<= 1;
  return bm;
}

// Joint choice.  A BM over 8 means the tile is starved of blocks rather than of
// work, so halve KS first (which widens the column tile and drops the shuffle
// count) and re-derive; only then clamp.
inline void pick_tile(int64_t outputs, int64_t kq, int64_t rowbytes, int *ks_out,
                      int *bm_out) {
  int ks = pick_ks(kq);
  int bm = pick_bm(outputs, ks);
  while (bm > 8 && ks >= 4 && kq % (ks / 2) == 0 && kq / (ks / 2) >= 2) {
    ks /= 2;
    bm = pick_bm(outputs, ks);
  }
  if (bm > 8) bm = 8;
  // The staged activation tile has to fit in shared memory.
  while (bm > 2 && (int64_t)bm * rowbytes > 40 * 1024) bm >>= 1;
  *ks_out = ks;
  *bm_out = bm;
}

inline void check_bf16(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(),
              name, ": need a contiguous cuda bf16 tensor");
}

}  // namespace

torch::Tensor fk_swiglu(const torch::Tensor &x, const torch::Tensor &wa,
                        const torch::Tensor &wb) {
  check_bf16(x, "x");
  check_bf16(wa, "linear_a.weight");
  check_bf16(wb, "linear_b.weight");
  const int64_t K = x.size(-1);
  TORCH_CHECK(wa.dim() == 2 && wa.size(1) == K && wb.sizes() == wa.sizes(),
              "swiglu: weight shape mismatch");
  const int64_t N = wa.size(0), M = x.numel() / K;
  TORCH_CHECK(K >= 4 && K % 4 == 0 && M > 0 && N > 0, "swiglu: unsupported K/M/N");
  int ks, bm;
  pick_tile(M * N, K >> 2, K * 4, &ks, &bm);
  const int bn = NWARP * (32 / ks);
  const int64_t shmem = ((int64_t)bm * K + 2 * bm * bn) * 4;
  TORCH_CHECK(shmem <= MAX_SHMEM, "swiglu: tile does not fit in shared memory");

  auto sizes = x.sizes().vec();
  sizes.back() = N;
  auto out = torch::empty(sizes, x.options());

  const dim3 grd((unsigned)((N + bn - 1) / bn), (unsigned)((M + bm - 1) / bm));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16 *xp = (const bf16 *)x.const_data_ptr();
  const bf16 *ap = (const bf16 *)wa.const_data_ptr();
  const bf16 *bp = (const bf16 *)wb.const_data_ptr();
  bf16 *op = (bf16 *)out.data_ptr();
#define LAUNCH(KSV, BMV)                                                            \
  swiglu_k<KSV, BMV><<<grd, NWARP * 32, (size_t)shmem, stream>>>(                   \
      xp, ap, bp, op, (int)M, (int)N, (int)K)
#define LAUNCH_BM(BMV)                                                              \
  switch (ks) {                                                                     \
    case 32: LAUNCH(32, BMV); break;                                                \
    case 16: LAUNCH(16, BMV); break;                                                \
    case 8:  LAUNCH(8, BMV);  break;                                                \
    case 4:  LAUNCH(4, BMV);  break;                                                \
    case 2:  LAUNCH(2, BMV);  break;                                                \
    default: LAUNCH(1, BMV);  break;                                                \
  }
  switch (bm) {
    case 16: LAUNCH_BM(16); break;
    case 8:  LAUNCH_BM(8);  break;
    case 2:  LAUNCH_BM(2);  break;
    default: LAUNCH_BM(4);  break;
  }
#undef LAUNCH_BM
#undef LAUNCH
  return out;
}

torch::Tensor fk_adaln(const torch::Tensor &a, const torch::Tensor &s,
                       const torch::Tensor &lnw, const torch::Tensor &wg,
                       const torch::Tensor &bg, const torch::Tensor &ws,
                       double eps_s, double eps_a) {
  check_bf16(a, "a");
  check_bf16(s, "s");
  check_bf16(lnw, "layer_norm_s.weight");
  check_bf16(wg, "linear_g.weight");
  check_bf16(bg, "linear_g.bias");
  check_bf16(ws, "linear_s.weight");
  const int64_t CA = a.size(-1), CS = s.size(-1);
  TORCH_CHECK(wg.dim() == 2 && wg.size(0) == CA && wg.size(1) == CS &&
                  ws.sizes() == wg.sizes() && lnw.numel() == CS && bg.numel() == CA,
              "adaln: parameter shape mismatch");
  TORCH_CHECK(CA % 8 == 0 && CS % 8 == 0, "adaln: channels must be a multiple of 8");
  // The reference broadcasts a_norm against linear_s(s_norm), whose shape is
  // s.shape[:-1] + (c_a,).  This kernel writes an a-shaped output, so require
  // the layouts under which that *is* the broadcast: a's trailing dims match
  // s's (last one excepted) and any extra leading dims of a are singleton.
  TORCH_CHECK(a.dim() >= s.dim(), "adaln: a has fewer dims than s");
  const int64_t off = a.dim() - s.dim();
  for (int64_t i = 0; i < off; ++i)
    TORCH_CHECK(a.size(i) == 1, "adaln: non-singleton leading dim in a");
  for (int64_t i = 0; i + 1 < s.dim(); ++i)
    TORCH_CHECK(a.size(off + i) == s.size(i), "adaln: a/s shape mismatch");
  const int64_t M = a.numel() / CA;
  TORCH_CHECK(M > 0 && s.numel() / CS == M, "adaln: row count mismatch");
  int ks, bm;
  pick_tile(M * CA, CS >> 2, CS * 4, &ks, &bm);
  const int bn = NWARP * (32 / ks);
  const int64_t shmem = ((int64_t)bm * CS + 2 * bm * bn + 2 * bm) * 4;
  TORCH_CHECK(shmem <= MAX_SHMEM, "adaln: tile does not fit in shared memory");

  auto out = torch::empty(a.sizes(), a.options());

  const dim3 grd((unsigned)((CA + bn - 1) / bn), (unsigned)((M + bm - 1) / bm));
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16 *ap = (const bf16 *)a.const_data_ptr();
  const bf16 *sp = (const bf16 *)s.const_data_ptr();
  const bf16 *lp = (const bf16 *)lnw.const_data_ptr();
  const bf16 *gp = (const bf16 *)wg.const_data_ptr();
  const bf16 *bp = (const bf16 *)bg.const_data_ptr();
  const bf16 *wp = (const bf16 *)ws.const_data_ptr();
  bf16 *op = (bf16 *)out.data_ptr();
#define LAUNCH(KSV, BMV)                                                            \
  adaln_k<KSV, BMV><<<grd, NWARP * 32, (size_t)shmem, stream>>>(                    \
      ap, sp, lp, gp, bp, wp, op, (int)M, (int)CA, (int)CS, (float)eps_s,           \
      (float)eps_a)
#define LAUNCH_BM(BMV)                                                              \
  switch (ks) {                                                                     \
    case 32: LAUNCH(32, BMV); break;                                                \
    case 16: LAUNCH(16, BMV); break;                                                \
    case 8:  LAUNCH(8, BMV);  break;                                                \
    case 4:  LAUNCH(4, BMV);  break;                                                \
    case 2:  LAUNCH(2, BMV);  break;                                                \
    default: LAUNCH(1, BMV);  break;                                                \
  }
  switch (bm) {
    case 16: LAUNCH_BM(16); break;
    case 8:  LAUNCH_BM(8);  break;
    case 2:  LAUNCH_BM(2);  break;
    default: LAUNCH_BM(4);  break;
  }
#undef LAUNCH_BM
#undef LAUNCH
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swiglu", &fk_swiglu, "fused SwiGLU");
  m.def("adaln", &fk_adaln, "fused AdaLN");
}
"""

_EXT = None
_SWIGLU = None
_ADALN = None
_LOADED = False


def _pin_arch() -> None:
    """Build for the local arch only; the ambient list has six of them."""
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
    global _EXT, _SWIGLU, _ADALN, _LOADED
    _LOADED = True
    try:
        from torch.utils.cpp_extension import load_inline

        _pin_arch()
        _EXT = load_inline(
            name="fk_l2_alphafold3_swiglu_fused",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=False,
        )
        _SWIGLU = _EXT.swiglu
        _ADALN = _EXT.adaln
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the torch path
        _EXT = None
        _SWIGLU = None
        _ADALN = None


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)
        if not _LOADED:
            _load()
        # Bound on the first forward, not here: the harness rewrites parameter
        # storage (dtype cast, then load_state_dict) between construction and
        # the first call.  A tuple, so nn.Module.__setattr__ keeps it out of
        # ``_parameters`` -- registering the weights twice would add phantom
        # state_dict keys.
        self._wt = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wt = self._wt
        if wt is None:
            wt = self._bind()
        if wt is not False and not torch.is_grad_enabled():
            try:
                return _SWIGLU(x, wt[0], wt[1])
            except Exception:  # noqa: BLE001 - input the kernel does not claim
                pass
        return self.silu(self.linear_a(x)) * self.linear_b(x)

    def _bind(self):
        wt = False if _SWIGLU is None else (self.linear_a.weight, self.linear_b.weight)
        self._wt = wt
        return wt


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)
        if not _LOADED:
            _load()
        self._wt = None

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        wt = self._wt
        if wt is None:
            wt = self._bind()
        if wt is not False and not torch.is_grad_enabled():
            try:
                return _ADALN(a, s, wt[0], wt[1], wt[2], wt[3], wt[4], wt[5])
            except Exception:  # noqa: BLE001 - input the kernel does not claim
                pass
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))

    def _bind(self):
        lnw, bg = self.layer_norm_s.weight, self.linear_g.bias
        if (_ADALN is None or lnw is None or bg is None
                or self.layer_norm_s.bias is not None
                or self.layer_norm_a.weight is not None
                or self.layer_norm_a.bias is not None):
            wt = False
        else:
            wt = (lnw, self.linear_g.weight, bg, self.linear_s.weight,
                  float(self.layer_norm_s.eps), float(self.layer_norm_a.eps))
        self._wt = wt
        return wt
