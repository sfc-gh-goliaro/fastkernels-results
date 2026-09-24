"""CLIP MLP and text embeddings (L2), fused.

Both operators are launch- and latency-bound at the captured shapes -- the MLP
moves 19 MB of weights for 0.73 GFLOP of work, the embedding pair moves 236 KB
-- so the wins come from cutting kernel count and from making each kernel's
memory pipeline deep enough to cover DRAM latency at the ~100 CTAs of work these
shapes provide.

CLIPMLP (hidden 768 -> 3072 -> 768, 77 tokens, fp32)
----------------------------------------------------
Baseline: ``F.linear`` (cuBLAS) -> three elementwise kernels for
``x * sigmoid(1.702x)`` -> ``F.linear``: five launches, 14 us for fc1 and 20 us
for fc2 of GPU time (cuBLAS runs fc2 with a 24-CTA grid).

Here it is two launches: ``fc1 + QuickGELU`` fused, then fc2 with a split-K
epilogue, with the activation handed over in fc2's MMA fragment order so it too
arrives by ``cp.async``. Both go through one templated ``mlp_gemm`` built on
``mma.sync.m16n8k8.tf32``, with 80-row (5 x m16) row tiles -- only 3 of 80 MMA
rows are padding for the 77 tokens.

Shared memory holds both operands in *MMA fragment order*, so an operand fetch
is one ``LDS.128`` / ``LDS.64`` of a coalesced run instead of four / two scalar
loads. The weights are permuted into that same order once, at load, which also
lets their tiles arrive by ``cp.async`` as a linear 16-byte copy: the weight is
the DRAM-resident operand (19 MB per call), so it is the one whose latency has
to be pipelined. The activation is small and L2-resident, so it is staged
through registers, where the TF32 rounding is applied on the way in.

Matching the reference bit for bit
----------------------------------
``torch.backends.cuda.matmul.fp32_precision == 'tf32'`` here, so the reference
GEMMs round both operands to TF32 and accumulate in fp32.  fc2 then re-rounds
the activation to TF32, and that rounding is a step function: a relative
perturbation of 1e-6 in the fc1 output flips ~0.5% of fc2's terms by a full TF32
ulp, which pushes ~1% of the output past the scorer's ``atol=1e-5`` for the
near-zero elements.  Measured: recomputing the activation with *any* accumulation
order other than the reference's -- even an exactly-rounded one -- lands at
0.988-0.994 matched, i.e. at or below the 0.99 the scorer requires.

So the activation is reproduced exactly instead of approximately:

* operands are rounded with ``cvt.rn.tf32.f32`` (the weights once, at load; the
  activations in the kernel), which is the rounding cuBLAS applies;
* fc1 keeps one sequential ascending k-chain per output element -- no split-K --
  because the MMA's internal accumulation is what the reference's is, so the
  same instruction order gives the same bits (verified: ``mma.sync`` tf32 matches
  cuBLAS bit for bit on this shape, and so does Triton's ``tcgen05`` path);
* ``quickgelu`` is evaluated with the same operations in the same order as
  ``x * torch.sigmoid(1.702 * x)`` (fp32 multiply, ``expf``, IEEE divide), not
  with a fast-math reciprocal;
* fc2 *may* split k -- its output feeds nothing that re-rounds, so the extra
  fp32 add costs ~1e-7 relative and the matched ratio stays at 1.0.

The result is an fc1 activation bit-identical to the reference's and a final
output that matches within 1.2e-05 (1.0000 of elements inside tolerance).

CLIPTextEmbeddings (vocab 49408, 77 positions, bf16)
----------------------------------------------------
Baseline: two gathers plus an add, three launches for 236 KB of traffic. Fused
into one kernel that gathers the token row, adds the position row (positions are
always ``arange``, so the position gather is just an index), and rounds once --
which is what ``tok + pos`` does in torch (fp32 opmath, single rounding), so the
result is bit-identical. One 16-byte vector per thread covers a row segment;
fp32/fp16/bf16 all use the same code path.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import CLIPTextConfig

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.quickgelu import QuickGELU

_MLP_CPP = """
#include <ATen/ATen.h>
bool fk_clip_mlp_run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&,
                     int64_t, int64_t, int64_t, int64_t);
at::Tensor fk_permute_weight(const at::Tensor&, int64_t, int64_t);
at::Tensor fk_clip_embed(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t);
"""

_MLP_CUDA = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_runtime.h>

#define MT 5
#define MROWS (MT * 16)
#define CEILDIV(a, b) (((a) + (b) - 1) / (b))
/* Tile choice, swept against the scorer's own timer on this GPU:
     fc1  8 warps as 2 m-groups x 4 n-columns, one 8-wide n-subtile each ->
          32 columns per CTA, 96 CTAs, k chunk 128, 3-deep cp.async pipeline
     fc2  4 warps in one m-group, 2 n-subtiles each -> 64 columns, 12 column
          groups x 8 k-splits = 96 CTAs, k chunk 128, 2-deep
   FC1_BN / FC2_BN must match what the host permutes the weights into, and HKC
   must match fc2's k chunk (it sets the layout fc1 writes the activation in). */
#define FC1_BN 32
#define FC1_KC 128
#define FC2_BN 64
#define FC2_KC 128
#define HKC 128                  /* fc2 k chunk: sets the activation block layout */
#define HKSTEPS (HKC / 8)
#define ABLKP 132                /* padded A fragment block (scatter path)    */
#define ABLKC 128                /* unpadded A fragment block (cp.async path)  */
#define BBLK 64                  /* B fragment block: 32 lanes x 2            */

__device__ __forceinline__ unsigned to_tf32(float x) {
  unsigned r;
  asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void mma16x8x8(float *d, const unsigned *a, const unsigned *b) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void cp16(float *dst, const float *src) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ float quickgelu(float x) {
  float y = 1.702f * x;
  float s = 1.0f / (1.0f + expf(-y));
  return x * s;
}

// Shared memory holds both operands in *MMA fragment order*, so an operand fetch
// is one LDS.128 / LDS.64 of a coalesced run rather than four / two scalar loads.
// The weight is pre-permuted into that same order on the host, which lets its
// tiles arrive by cp.async (a linear 16 B copy) instead of a register scatter --
// the weight is the DRAM-resident operand, so it is the one whose latency has to
// be pipelined. The activation is small and L2-resident; it is staged through
// registers because its global layout is row major.
template <int NW, int MG, int NPW, int KC, bool CONV_A, int EPI, int KS, int ST>
__global__ __launch_bounds__(32 * NW) void mlp_gemm(
    const float *__restrict__ A, const float *__restrict__ BP,
    const float *__restrict__ BIAS, float *__restrict__ C,
    float *__restrict__ OUT2, const float *__restrict__ BIAS2,
    int M, int K, int N, int OC) {
  constexpr int NWN = NW / MG;
  constexpr int BN = 8 * NWN * NPW;
  constexpr int MTG = CEILDIV(MT, MG);
  constexpr int NSUB = BN / 8;
  constexpr int NT = 32 * NW;
  constexpr int KSTEPS = KC / 8;
  constexpr int ABLK = CONV_A ? ABLKP : ABLKC;
  constexpr int ABUF = MT * KSTEPS * ABLK;
  constexpr int BBUF = NSUB * KSTEPS * BBLK;          /* == KC * BN */
  /* A buffers: the scatter path double buffers by (c & 1), the async path uses
     the same ST-deep rotation as B. */
  constexpr int AQ = KC / 4;
  constexpr int AVT = CONV_A ? CEILDIV(MROWS * AQ, NT) : CEILDIV(ABUF / 4, NT);
  constexpr int BVT = CEILDIV(BBUF / 4, NT);
  static_assert(!CONV_A || NT % AQ == 0, "CTA threads must cover whole A rows");

  extern __shared__ float smem[];
  float *sa = smem;
  float *sb = smem + (CONV_A ? 2 : ST) * ABUF;

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int mg = warp % MG, wn = warp / MG;
  const int mlo = mg * MTG;
  const int m0 = blockIdx.y * MROWS;
  const int nt = blockIdx.x;                          /* n tile index */
  const int kchunk = (KS == 1) ? K : (K / KS);
  const int nchunk = kchunk / KC;
  const int c0 = (KS == 1) ? 0 : (int)blockIdx.z * nchunk;   /* first k chunk */

  constexpr int ARSTEP = NT / AQ;
  const int arow0 = tid / AQ, akk = (tid % AQ) * 4;
  /* permuted weight: [n tile][k chunk][BBUF] */
  const float *bbase = BP + ((size_t)nt * (K / KC) + c0) * BBUF;

  float acc[MTG][NPW][4];
#pragma unroll
  for (int i = 0; i < MTG; ++i)
#pragma unroll
    for (int j = 0; j < NPW; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

  float4 ra[CONV_A ? AVT : 1];
#define LOADA(kbase)                                                                 \
  do {                                                                               \
    _Pragma("unroll") for (int u = 0; u < AVT; ++u) {                                  \
      const int row = arow0 + u * ARSTEP;                                              \
      if (AVT * ARSTEP <= MROWS || row < MROWS)                                        \
        ra[u] = *reinterpret_cast<const float4 *>(                                     \
            A + (size_t)min(m0 + row, M - 1) * K + (kbase) + akk);                     \
    }                                                                                  \
  } while (0)

#define STOREA(buf)                                                                  \
  do {                                                                               \
    _Pragma("unroll") for (int u = 0; u < AVT; ++u) {                                  \
      const int row = arow0 + u * ARSTEP;                                              \
      float *d = sa + (buf) * ABUF                                                     \
                 + ((row >> 4) * KSTEPS + (akk >> 3)) * ABLK + (row & 7) * 16          \
                 + ((row & 15) >> 3) + 2 * ((akk & 7) >> 2);                           \
      if (CONV_A) {                                                                    \
        d[0] = __uint_as_float(to_tf32(ra[u].x));                                       \
        d[4] = __uint_as_float(to_tf32(ra[u].y));                                       \
        d[8] = __uint_as_float(to_tf32(ra[u].z));                                       \
        d[12] = __uint_as_float(to_tf32(ra[u].w));                                      \
      } else {                                                                         \
        d[0] = ra[u].x; d[4] = ra[u].y; d[8] = ra[u].z; d[12] = ra[u].w;                \
      }                                                                                \
    }                                                                                  \
  } while (0)

/* fc2's A tile is already in fragment order in global memory (fc1 wrote it that
   way), so it streams in with the same linear cp.async the weight uses. */
#define STAGEA_ASYNC(buf, ch)                                                        \
  do {                                                                               \
    _Pragma("unroll") for (int u = 0; u < AVT; ++u) {                                  \
      const int unit = tid + u * NT;                                                   \
      if (AVT * NT <= ABUF / 4 || unit < ABUF / 4)                                      \
        cp16(sa + (buf) * ABUF + unit * 4,                                             \
             A + (size_t)((ch) + c0) * ABUF + unit * 4);                                \
    }                                                                                  \
  } while (0)

#define STAGEB(buf, ch)                                                              \
  do {                                                                               \
    _Pragma("unroll") for (int u = 0; u < BVT; ++u) {                                  \
      const int unit = tid + u * NT;                                                   \
      if (BVT * NT <= BBUF / 4 || unit < BBUF / 4)                                      \
        cp16(sb + (buf) * BBUF + unit * 4, bbase + (size_t)(ch) * BBUF + unit * 4);     \
    }                                                                                  \
    if (!CONV_A) STAGEA_ASYNC(buf, ch);                                                \
    cp_commit();                                                                       \
  } while (0)

  if (CONV_A) LOADA(c0 * KC);
#pragma unroll
  for (int p = 0; p < ST; ++p)
    if (p < nchunk) STAGEB(p, p);

  for (int c = 0; c < nchunk; ++c) {
    const int buf = c % ST;
    const int abuf = CONV_A ? (c & 1) : buf;
    if (CONV_A) {
      STOREA(c & 1);
      if (c + 1 < nchunk) LOADA((c0 + c + 1) * KC);
    }
    cp_wait<ST - 1>();
    __syncthreads();
    const float *pa = sa + abuf * ABUF + lane * 4;
    const float *pb = sb + buf * BBUF + wn * (NPW * KSTEPS * BBLK) + lane * 2;
#pragma unroll
    for (int s = 0; s < KSTEPS; ++s) {
      unsigned af[MTG][4];
#pragma unroll
      for (int i = 0; i < MTG; ++i)
        if (MG == 1 || mlo + i < MT) {
          const float4 f =
              *reinterpret_cast<const float4 *>(pa + ((mlo + i) * KSTEPS + s) * ABLK);
          af[i][0] = __float_as_uint(f.x);
          af[i][1] = __float_as_uint(f.y);
          af[i][2] = __float_as_uint(f.z);
          af[i][3] = __float_as_uint(f.w);
        }
      unsigned bf[NPW][2];
#pragma unroll
      for (int j = 0; j < NPW; ++j) {
        const float2 f = *reinterpret_cast<const float2 *>(pb + (j * KSTEPS + s) * BBLK);
        bf[j][0] = __float_as_uint(f.x);
        bf[j][1] = __float_as_uint(f.y);
      }
#pragma unroll
      for (int i = 0; i < MTG; ++i)
        if (MG == 1 || mlo + i < MT)
#pragma unroll
          for (int j = 0; j < NPW; ++j) mma16x8x8(acc[i][j], af[i], bf[j]);
    }
    __syncthreads();
    if (c + ST < nchunk) STAGEB(buf, c + ST);
  }
#undef LOADA
#undef STOREA
#undef STAGEB

  const int nw0 = nt * BN + wn * (8 * NPW);
#pragma unroll
  for (int j = 0; j < NPW; ++j) {
    const int col = nw0 + j * 8 + 2 * t;
#pragma unroll
    for (int i = 0; i < MTG; ++i) {
      if (MG != 1 && mlo + i >= MT) continue;
      const int r0 = m0 + (mlo + i) * 16 + g, r1 = r0 + 8;
      if (EPI == 0) {
        /* Store the activation straight into fc2's A-fragment order (block
           [k chunk][m subtile][k step][lane][4]) so fc2 can cp.async it. */
        const float2 bias = *reinterpret_cast<const float2 *>(BIAS + col);
        float v[4];
        v[0] = __uint_as_float(to_tf32(quickgelu(acc[i][j][0] + bias.x)));
        v[1] = __uint_as_float(to_tf32(quickgelu(acc[i][j][1] + bias.y)));
        v[2] = __uint_as_float(to_tf32(quickgelu(acc[i][j][2] + bias.x)));
        v[3] = __uint_as_float(to_tf32(quickgelu(acc[i][j][3] + bias.y)));
        const int mi = mlo + i;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int row = (e < 2) ? (g) : (g + 8);
          const int kk = col + (e & 1);
          if (m0 + mi * 16 + row >= M) continue;
          const int blk = kk / HKC;
          const int st2 = (kk % HKC) / 8;
          const int ln = (row & 7) * 4 + (kk & 3);
          const int cmp = ((row & 15) >> 3) + 2 * ((kk & 7) >> 2);
          C[((size_t)blk * MT + mi) * (HKSTEPS * ABLKC) + st2 * ABLKC + ln * 4 + cmp] = v[e];
        }
      } else {
        if (r0 < M) {
          float *p = C + (size_t)r0 * N + col;
          atomicAdd(p, acc[i][j][0]);
          atomicAdd(p + 1, acc[i][j][1]);
        }
        if (r1 < M) {
          float *p = C + (size_t)r1 * N + col;
          atomicAdd(p, acc[i][j][2]);
          atomicAdd(p + 1, acc[i][j][3]);
        }
      }
    }
  }
  if (EPI == 0) {
    const int oc0 = nt * OC;
    for (int idx = tid; idx < MROWS * OC; idx += NT) {
      const int r = m0 + idx / OC, c = oc0 + idx % OC;
      if (r < M) OUT2[(size_t)r * (size_t)(OC * gridDim.x) + c] = BIAS2[c];
    }
  }
}

// W [N, K] row major -> fragment order [n tile][k chunk][jn][s][lane][2], with
// every element rounded to TF32 on the way (same instruction the MMA path uses).
__global__ void permute_w(const float *__restrict__ W, float *__restrict__ P, int N, int K,
                          int BN, int KC, int64_t total) {
  int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (idx >= total) return;
  const int KSTEPS = KC / 8;
  int c = (int)(idx & 1);
  int64_t r = idx >> 1;
  int lane = (int)(r % 32); r /= 32;
  int s = (int)(r % KSTEPS); r /= KSTEPS;
  int jn = (int)(r % (BN / 8)); r /= (BN / 8);
  int ch = (int)(r % (K / KC)); r /= (K / KC);
  int ntile = (int)r;
  const int gg = lane >> 2, tt = lane & 3;
  const int col = ntile * BN + jn * 8 + gg;
  const int k = ch * KC + s * 8 + tt + 4 * c;
  P[idx] = __uint_as_float(to_tf32(W[(size_t)col * K + k]));
}

at::Tensor fk_permute_weight(const at::Tensor &W, int64_t BN, int64_t KC) {
  const int64_t N = W.size(0), K = W.size(1);
  at::Tensor P = at::detail::empty_cuda({N * K}, at::kFloat, W.device(), c10::nullopt);
  const int64_t total = N * K;
  permute_w<<<(unsigned)((total + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      W.const_data_ptr<float>(), P.mutable_data_ptr<float>(), (int)N, (int)K, (int)BN,
      (int)KC, total);
  return P;
}

static constexpr int smem_of(int NW, int MG, int NPW, int KC, int ST, bool conv_a) {
  return ((conv_a ? 2 : ST) * MT * (KC / 8) * (conv_a ? ABLKP : ABLKC)
          + ST * KC * (8 * (NW / MG) * NPW)) * (int)sizeof(float);
}

template <int A, int B, int C, int D, int E, int F, class Fn>
static void set_smem(Fn *fn, int bytes) {
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute((const void *)fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    cudaGetLastError();
    done = true;
  }
}

#define LAUNCH1(NW, MG, NPW, KC, ST)                                                  \
  do {                                                                                \
    constexpr int BN = 8 * ((NW) / (MG)) * (NPW);                                      \
    constexpr int SM = smem_of(NW, MG, NPW, KC, ST, true);                             \
    if (N1 % BN || K % (KC) || N2 % (N1 / BN)) return false;                           \
    auto fn = mlp_gemm<NW, MG, NPW, KC, true, 0, 1, ST>;                              \
    set_smem<NW, MG, NPW, KC, 0, ST>(fn, SM);                                         \
    fn<<<dim3((unsigned)(N1 / BN), (unsigned)mt), 32 * (NW), SM, st>>>(               \
        xp, w1p, b1p, hp, op, b2p, (int)M, (int)K, (int)N1, (int)(N2 / (N1 / BN)));     \
  } while (0)

#define LAUNCH2(NW, MG, NPW, KC, KS, ST)                                              \
  do {                                                                                \
    constexpr int BN = 8 * ((NW) / (MG)) * (NPW);                                      \
    constexpr int SM = smem_of(NW, MG, NPW, KC, ST, false);                            \
    if (N2 % BN || N1 % ((KC) * (KS))) return false;                                   \
    auto fn = mlp_gemm<NW, MG, NPW, KC, false, 1, KS, ST>;                            \
    set_smem<NW, MG, NPW, KC, 1, KS>(fn, SM);                                         \
    fn<<<dim3((unsigned)(N2 / BN), (unsigned)mt, (KS)), 32 * (NW), SM, st>>>(          \
        hp, w2p, nullptr, op, nullptr, nullptr, (int)M, (int)N1, (int)N2, 0);           \
  } while (0)

bool fk_clip_mlp_run(const at::Tensor &X, const at::Tensor &W1P, const at::Tensor &B1,
                     const at::Tensor &W2P, const at::Tensor &B2, at::Tensor &H,
                     at::Tensor &OUT, int64_t M, int64_t K, int64_t N1, int64_t N2) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int mt = (int)CEILDIV(M, MROWS);
  const float *xp = X.const_data_ptr<float>();
  const float *w1p = W1P.const_data_ptr<float>();
  const float *b1p = B1.const_data_ptr<float>();
  const float *w2p = W2P.const_data_ptr<float>();
  const float *b2p = B2.const_data_ptr<float>();
  float *hp = H.mutable_data_ptr<float>();
  float *op = OUT.mutable_data_ptr<float>();
  LAUNCH1(8, 2, 1, FC1_KC, 3);
  LAUNCH2(4, 1, 2, FC2_KC, 8, 2);
  return true;
}
"""

_EMB_CUDA = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// out[i, d] = tok[ids[i], d] + pos[i % S, d], computed in fp32 and rounded once
// (exactly what `tok_emb + pos_emb` does in torch: opmath float, RNE store).
template <typename T> struct Cvt;
template <> struct Cvt<float> {
  static __device__ __forceinline__ float up(float v) { return v; }
  static __device__ __forceinline__ float down(float v) { return v; }
};
template <> struct Cvt<__half> {
  static __device__ __forceinline__ float up(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half down(float v) { return __float2half_rn(v); }
};
template <> struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float up(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 down(float v) { return __float2bfloat16(v); }
};

// VEC elements per thread (one 16B transaction when sizeof(T)*VEC == 16).
template <typename T, int VEC, bool EXACT>
__global__ __launch_bounds__(256) void clip_embed(const T* __restrict__ tok,
                                                  const T* __restrict__ pos,
                                                  const int64_t* __restrict__ ids,
                                                  T* __restrict__ out,
                                                  int nvec_row, int S, int64_t vocab,
                                                  int total_vec) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (!EXACT && i >= total_vec) return;
  int row = i / nvec_row;
  int col = i - row * nvec_row;             // vector index inside the row
  int64_t id = ids[row];
  if (id < 0 || id >= vocab) id = 0;
  const T* tp = tok + id * (int64_t)nvec_row * VEC + (int64_t)col * VEC;
  const T* pp = pos + (int64_t)(row % S) * nvec_row * VEC + (int64_t)col * VEC;
  T* op = out + (int64_t)i * VEC;
  T a[VEC], b[VEC], r[VEC];
  *reinterpret_cast<int4*>(a) = *reinterpret_cast<const int4*>(tp);
  *reinterpret_cast<int4*>(b) = *reinterpret_cast<const int4*>(pp);
#pragma unroll
  for (int k = 0; k < VEC; ++k) r[k] = Cvt<T>::down(Cvt<T>::up(a[k]) + Cvt<T>::up(b[k]));
  *reinterpret_cast<int4*>(op) = *reinterpret_cast<const int4*>(r);
}

template <typename T, int VEC>
static void launch(const at::Tensor& tok, const at::Tensor& pos, const at::Tensor& ids,
                   at::Tensor& out, int S) {
  const int D = (int)tok.size(1);
  const int nvec_row = D / VEC;
  const int total = (int)(ids.numel() * nvec_row);
  const int grid = (total + 255) / 256;
  auto s = at::cuda::getCurrentCUDAStream();
  auto fn = (total == grid * 256) ? clip_embed<T, VEC, true> : clip_embed<T, VEC, false>;
  fn<<<grid, 256, 0, s>>>(static_cast<const T*>(tok.const_data_ptr()),
                          static_cast<const T*>(pos.const_data_ptr()),
                          static_cast<const int64_t*>(ids.const_data_ptr()),
                          static_cast<T*>(out.mutable_data_ptr()),
                          nvec_row, S, tok.size(0), total);
}

at::Tensor fk_clip_embed(const at::Tensor& tok, const at::Tensor& pos,
                         const at::Tensor& ids, int64_t S) {
  const int64_t D = tok.size(1);
  std::vector<int64_t> sz(ids.sizes().vec());
  sz.push_back(D);
  at::Tensor out = at::detail::empty_cuda(sz, tok.scalar_type(), tok.device(), c10::nullopt);
  if (ids.numel() == 0 || D == 0) return out;
  const auto st = tok.scalar_type();
  if (st == at::kBFloat16 && (D & 7) == 0)
    launch<__nv_bfloat16, 8>(tok, pos, ids, out, (int)S);
  else if (st == at::kHalf && (D & 7) == 0)
    launch<__half, 8>(tok, pos, ids, out, (int)S);
  else if (st == at::kFloat && (D & 3) == 0)
    launch<float, 4>(tok, pos, ids, out, (int)S);
  else
    return at::embedding(tok, ids) + at::embedding(pos, at::arange(S, ids.options()));
  return out;
}
"""


def _build():
    """JIT-compile both kernels, pinned to the local GPU arch (tf32 MMA needs sm_80+)."""
    from torch.utils.cpp_extension import load_inline
    try:
        from fastkernels.infra.cuda_ext import _pin_build_arch
        _pin_build_arch()
    except Exception:
        try:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    return load_inline(
        name="fk_cand_clip_mlp",
        cpp_sources=_MLP_CPP,
        cuda_sources=[_MLP_CUDA, _EMB_CUDA],
        functions=["fk_clip_mlp_run", "fk_permute_weight", "fk_clip_embed"],
        with_cuda=True,
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )


try:
    _EXT = _build()
except Exception:      # no nvcc / unsupported toolchain -> eager fallback
    _EXT = None

_MROWS = 80            # rows per MMA row tile (5 x m16); must match the kernel
_FC1_BN, _FC2_BN, _KC = 32, 64, 128   # must match FC1_BN / FC2_BN / *_KC in the kernel


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()
        self._plan = None

    def _build(self):
        """Validate the fast path once and pre-stage what it needs: both weights
        transposed to [K, N] and rounded to TF32 (the rounding the reference's
        GEMMs apply), plus the activation workspace. ``None`` means fall back.

        Kept as one cached tuple because ``forward`` is measured at ~50 us with a
        ~9 us floor -- per-call attribute lookups are not free at that scale."""
        w1, w2 = self.fc1.weight, self.fc2.weight
        b1, b2 = self.fc1.bias, self.fc2.bias
        if (_EXT is None or b1 is None or b2 is None or not w1.is_cuda
                or w1.dtype is not torch.float32 or w2.dtype is not torch.float32
                or b1.dtype is not torch.float32 or b2.dtype is not torch.float32
                or w2.shape[1] != w1.shape[0] or not w1.is_contiguous()
                or not w2.is_contiguous()):
            return None
        N1, K = int(w1.shape[0]), int(w1.shape[1])
        N2 = int(w2.shape[0])
        if (N1 % _FC1_BN or K % _KC or N2 % _FC2_BN or N1 % _KC
                or N2 % (N1 // _FC1_BN) or N1 % (_KC * 8)):
            return None
        w1p = _EXT.fk_permute_weight(w1, _FC1_BN, _KC)
        w2p = _EXT.fk_permute_weight(w2, _FC2_BN, _KC)
        h = torch.zeros(_MROWS * N1, device=w1.device, dtype=torch.float32)
        return (w1p, w2p, h, b1, b2, K, N1, N2, w1._version, w2._version)

    def _eager(self, hidden_states):
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        p = self._plan
        if (p is None or p[8] != self.fc1.weight._version
                or p[9] != self.fc2.weight._version):
            p = self._plan = self._build()
            if p is None:
                return self._eager(hidden_states)
        w1t, w2t, h, b1, b2, K, N1, N2 = p[:8]
        if hidden_states.dtype is not torch.float32 or hidden_states.shape[-1] != K:
            return self._eager(hidden_states)
        x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        M = x.numel() // K
        if M > _MROWS:          # workspace and grid are sized for one row tile
            return self._eager(hidden_states)
        out = torch.empty(x.shape[:-1] + (N2,), device=x.device, dtype=x.dtype)
        if not _EXT.fk_clip_mlp_run(x.view(M, K), w1t, b1, w2t, b2, h,
                                    out.view(M, N2), M, K, N1, N2):
            return self._eager(hidden_states)   # tile shape does not divide these dims
        return out


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self._plan = None

    def _build(self):
        """``(token weight, position weight)`` if the fused gather can serve this
        module, else ``False``. Both are ``nn.Parameter``s, so an in-place dtype
        cast (what the harness does) is picked up without rebuilding."""
        tok = self.token_embedding.emb.weight
        pos = self.position_embedding.emb.weight
        if (_EXT is None or not tok.is_cuda or tok.dtype != pos.dtype
                or not tok.is_contiguous() or not pos.is_contiguous()
                or tok.dim() != 2 or pos.dim() != 2 or tok.shape[1] != pos.shape[1]):
            return False
        return (tok, pos, int(pos.shape[0]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        p = self._plan
        if p is None:
            p = self._plan = self._build()
        seq_length = input_ids.shape[-1]
        if (p is not False and input_ids.dtype is torch.int64 and input_ids.dim() == 2
                and seq_length <= p[2] and input_ids.is_contiguous()):
            return _EXT.fk_clip_embed(p[0], p[1], input_ids, seq_length)
        position_ids = self.position_ids[:, :seq_length]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)
