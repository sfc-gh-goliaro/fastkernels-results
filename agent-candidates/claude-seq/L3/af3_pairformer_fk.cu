// Fused AlphaFold3 PairFormer stack -- 14 kernels per block, bit-exact.
//
// Why bit-exact and not merely "close": over 48 blocks this stack amplifies a
// single bf16 ULP perturbation of the input until ~58% of the output elements
// fall outside the scorer's tolerance (measured: perturbing one element of z by
// one ULP in the *reference* diverges exactly as much as a fused-but-inexact
// kernel).  So the only correct fast implementation is one that reproduces the
// reference's arithmetic exactly.  Everything here does:
//
//   * every matmul uses mma.sync.m16n8k16 with fp32 accumulation walked
//     sequentially in k -- byte-identical to cuBLAS on every shape in this
//     stack (verified elementwise over millions of outputs, including the
//     k=24 attention contraction which is zero-padded to 32)
//   * LayerNorm replicates ATen's vectorized_layer_norm_kernel exactly: the
//     same Welford recurrence, the same four-elements-per-lane assignment, the
//     same intra-warp shuffle-down tree and the same 4-way inter-warp tree
//   * softmax replicates softmax_warp_forward for dim=16 (WARP_SIZE=16, an
//     XOR butterfly for max then for sum)
//   * sigmoid/silu/mul/add/div use the same fp32 expressions ATen's
//     TensorIterator uses and round to bf16 at exactly the same points, so this
//     file must NOT be compiled with --use_fast_math
//
// A kernel boundary is placed only where a reduction crosses CTA boundaries;
// everything else is fused, taking the reference's ~143 kernels per block to 14.
//
// One reference quirk is load-bearing: TriangleAttention's triangle bias is
// ``_permute_final_dims(linear_z(x), (2,0,1)).unsqueeze(-4)``, which inserts the
// new axis at position 1 -- so the bias is indexed [h][q][k] over the *whole*
// pair matrix and broadcast across the attention's leading residue axis, rather
// than being per-row.  Hence the separate ``k_ta_pre`` pass.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <vector>

using bf16 = __nv_bfloat16;

// ---------------- captured AF3 PairFormer configuration ----------------
#define NRES 16
#define NTOK 256          // NRES*NRES
#define CZ   128          // c_z
#define TAH  4            // no_heads_pair
#define TAD  32           // c_hidden_pair_att
#define TAHD 128          // TAH*TAD
#define PTH  512          // transition_n * c_z
#define CS   384          // c_s
#define APH  16           // no_heads_pair_bias
#define APD  24           // c_hidden_pair_bias
#define APDP 32           // per-head dim padded to the mma k step
#define APHD 384          // APH*APD
#define APQ  (APH * APDP) // padded q/k row stride
#define STH  1536         // transition_n * c_s

#define SZ   (CZ + 8)     // shared row stride for 128-wide rows
#define SS   (CS + 8)     // shared row stride for 384-wide rows
#define S16  20           // padded stride for 16-wide shared rows: a stride of
                          // 16 bf16 puts all eight mma operand rows in the same
                          // bank group (8-way conflict); 20 spreads them
#define NW   8            // warps per CTA
#define NTHR (NW * 32)
#define EPSL 1e-5f
#define RS32 5.656854249492380195f   // sqrt(32)
#define RS24 4.898979485566356f      // sqrt(24)

// ---------------- scalar primitives (ATen-identical) ----------------
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16  f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ bf16 sigb(bf16 x) { return f2b(1.f / (1.f + expf(-b2f(x)))); }
__device__ __forceinline__ bf16 silub(bf16 x) { float a = b2f(x); return f2b(a / (1.f + expf(-a))); }
__device__ __forceinline__ bf16 mulb(bf16 a, bf16 b) { return f2b(b2f(a) * b2f(b)); }
__device__ __forceinline__ bf16 addb(bf16 a, bf16 b) { return f2b(b2f(a) + b2f(b)); }
__device__ __forceinline__ bf16 maskbias(bf16 m) {
  return f2b(1e9f * b2f(f2b(b2f(m) - 1.f)));
}

// ---------------- mma m16n8k16, sequential k ----------------
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One 16x8 output tile from a *strided* pair of operands: A is the 16-row
// operand (row stride lda), B the 8-row operand (row stride ldb).  Used where B
// is produced inside the kernel (attention k/v).
template <int K>
__device__ __forceinline__ void mma_tile(float* d, const bf16* A, int lda,
                                         const bf16* B, int ldb) {
  const int t = threadIdx.x & 31, g = t >> 2, t4 = (t & 3) * 2;
  const bf16* a0 = A + (size_t)g * lda + t4;
  const bf16* a1 = A + (size_t)(g + 8) * lda + t4;
  const bf16* b0 = B + (size_t)g * ldb + t4;
#pragma unroll
  for (int k = 0; k < K; k += 16) {
    uint32_t a[4];
    a[0] = *(const uint32_t*)(a0 + k);
    a[1] = *(const uint32_t*)(a1 + k);
    a[2] = *(const uint32_t*)(a0 + k + 8);
    a[3] = *(const uint32_t*)(a1 + k + 8);
    mma16816(d, a, *(const uint32_t*)(b0 + k), *(const uint32_t*)(b0 + k + 8));
  }
}

// Same, but B is a weight tile pre-swizzled on the host into mma B-fragment
// order: element ((k_step * 32 + lane) * 4 + e).  One 8-byte load per lane per
// k-step covers the whole fragment, and the warp's 32 loads are one contiguous
// 256-byte transaction instead of eight scattered sectors.  With only ~8 warps
// resident per SM there is nothing else to hide memory latency with, so the
// k-steps are issued in batches of UN: UN loads in flight, then UN mmas.
#define WTILE(K) ((size_t)((K) / 16) * 128)   // swizzled elements per n-tile

// ncu on this stack: every kernel sits at ~2% of peak on every unit with
// ~15% issue activity and 13-30 cycles of warp latency per instruction issued.
// The problem is too small to fill the machine (16-128 CTAs of 8 warps), so the
// only lever is instruction-level parallelism: the weight loads have to be
// issued in deep batches, otherwise each mma waits out a full memory round trip.
// ``UN`` k-steps of the B operand are loaded before any mma runs; the A operand
// comes from shared memory and is cheap enough to fetch per mma.
template <int K>
__device__ __forceinline__ void mma_sw(float* d, const bf16* A, int lda,
                                       const bf16* Bt) {
  constexpr int NKS = K / 16;
  constexpr int UN = NKS <= 32 ? NKS : 32;
  const int t = threadIdx.x & 31, g = t >> 2, t4 = (t & 3) * 2;
  const bf16* a0 = A + (size_t)g * lda + t4;
  const bf16* a1 = A + (size_t)(g + 8) * lda + t4;
  const bf16* b = Bt + t * 4;
#pragma unroll
  for (int ks0 = 0; ks0 < NKS; ks0 += UN) {
    uint2 bb[UN];
#pragma unroll
    for (int u = 0; u < UN; ++u) bb[u] = *(const uint2*)(b + (size_t)(ks0 + u) * 128);
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      const int k = (ks0 + u) * 16;
      uint32_t aa[4];
      aa[0] = *(const uint32_t*)(a0 + k);
      aa[1] = *(const uint32_t*)(a1 + k);
      aa[2] = *(const uint32_t*)(a0 + k + 8);
      aa[3] = *(const uint32_t*)(a1 + k + 8);
      mma16816(d, aa, bb[u].x, bb[u].y);
    }
  }
}

// NP independent weight planes sharing one A operand (the 5 triangle-multiply
// projections, the two SwiGLU projections).  Batching the planes together as
// well multiplies the loads in flight by NP, turning NP sequential memory round
// trips into one.
template <int K, int NP>
__device__ __forceinline__ void mma_sw_np(float d[NP][4], const bf16* A, int lda,
                                          const bf16* Bt, size_t pstride) {
  constexpr int NKS = K / 16;
  constexpr int UN = (NKS * NP <= 40) ? NKS
                   : ((NKS % 8 == 0) ? 8 : ((NKS % 4 == 0) ? 4 : 1));
  const int t = threadIdx.x & 31, g = t >> 2, t4 = (t & 3) * 2;
  const bf16* a0 = A + (size_t)g * lda + t4;
  const bf16* a1 = A + (size_t)(g + 8) * lda + t4;
  const bf16* b = Bt + t * 4;
#pragma unroll
  for (int ks0 = 0; ks0 < NKS; ks0 += UN) {
    uint2 bb[UN][NP];
#pragma unroll
    for (int u = 0; u < UN; ++u)
#pragma unroll
      for (int pp = 0; pp < NP; ++pp)
        bb[u][pp] = *(const uint2*)(b + pp * pstride + (size_t)(ks0 + u) * 128);
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      const int k = (ks0 + u) * 16;
      uint32_t aa[4];
      aa[0] = *(const uint32_t*)(a0 + k);
      aa[1] = *(const uint32_t*)(a1 + k);
      aa[2] = *(const uint32_t*)(a0 + k + 8);
      aa[3] = *(const uint32_t*)(a1 + k + 8);
#pragma unroll
      for (int pp = 0; pp < NP; ++pp) mma16816(d[pp], aa, bb[u][pp].x, bb[u][pp].y);
    }
  }
}

#define MMA_ROW0 ((threadIdx.x & 31) >> 2)
#define MMA_COL0 (((threadIdx.x & 31) & 3) * 2)
#define ZERO4(d) do { d[0] = d[1] = d[2] = d[3] = 0.f; } while (0)

// ---------------- LayerNorm: ATen vectorized_layer_norm_kernel ----------------
struct WD { float mean, sigma2, count; };

// Welford, but with every element count pinned as a template argument.  The
// reference divides -- ``1.f/new_count`` in the online update and ``1.f/count``
// in the merge -- twelve times per row, and a full-precision fp32 divide is a
// ~20 instruction sequence.  With ~8 warps per SM this stack is issue bound
// (ncu: 15% issue activity, 13 cycles of warp latency per instruction), so those
// divides were the single biggest instruction term.  Every count here is known
// at compile time -- the online update runs over exactly four elements, and the
// reduction tree doubles a fixed count at each level -- and constant-folding
// ``1.f/CNT`` is correctly rounded, hence bit-identical to the divide.
template <int CNT>
__device__ __forceinline__ WD w_online(float val, WD c) {
  constexpr float rcp = 1.0f / (float)CNT;
  float delta = val - c.mean;
  float nm = c.mean + delta * rcp;
  return {nm, c.sigma2 + delta * (val - nm), (float)CNT};
}

// Mirrors cuWelfordCombine(dataB, dataA); CA/CB are dataA.count / dataB.count.
// The fmaf forms are not cosmetic: the reference's ``nA*A.mean + nB*B.mean``
// compiles to fma(nA, A.mean, nB*B.mean), and writing it as a plain expression
// lets nvcc pick the other association, which disagrees on ~3e-6 of elements --
// enough to fail once amplified across 48 blocks.  (Only reachable with a
// 384-wide norm, where the inter-warp merges have non-zero counts on both sides.)
template <int CA, int CB>
__device__ __forceinline__ WD w_comb(WD B, WD A) {
  constexpr int CT = CA + CB;
  if (CT == 0) return {0.f, 0.f, 0.f};
  constexpr float coef = 1.0f / (float)(CT > 0 ? CT : 1);
  constexpr float nA = (float)CA * coef, nB = (float)CB * coef;
  float delta = B.mean - A.mean;
  return {__fmaf_rn(nA, A.mean, nB * B.mean),
          __fmaf_rn(delta * delta * (float)CA, nB, A.sigma2 + B.sigma2),
          (float)CT};
}
__device__ __forceinline__ WD w_down(WD w, int off) {
  return {__shfl_down_sync(0xffffffffu, w.mean, off),
          __shfl_down_sync(0xffffffffu, w.sigma2, off),
          __shfl_down_sync(0xffffffffu, w.count, off)};
}

// One warp, one row of N bf16 promoted to fp32 (exactly what ``x.float()`` does).
// N is a multiple of 128; ATen's launch is (32, 4) threads, so a row is split
// into N/128 groups of 32 float4 and merged with its 4-way inter-warp tree.
// One warp, one row of N bf16 promoted to fp32 (exactly what ``x.float()``
// does).  N is a multiple of 128; ATen launches (32, 4) threads, so a row is
// split into N/128 groups of 32 float4 and merged with its 4-way inter-warp
// tree.  The row is loaded once (8 bytes per lane per group) and kept in
// registers for the affine pass -- at this occupancy a second pass over global
// memory costs more than the whole reduction.
template <int N>
__device__ __forceinline__ void ln_stage(const bf16* row, int lane, float* v) {
  constexpr int NV = N / 128;
#pragma unroll
  for (int g = 0; g < NV; ++g) {
    uint2 raw = *(const uint2*)(row + (g * 32 + lane) * 4);
    const bf16* p = (const bf16*)&raw;
#pragma unroll
    for (int e = 0; e < 4; ++e) v[g * 4 + e] = b2f(p[e]);
  }
}

template <int N>
__device__ __forceinline__ void ln_stats(const float* v, float& mean, float& rstd) {
  constexpr int NV = N / 128;
  WD wd[NV];
#pragma unroll
  for (int g = 0; g < NV; ++g) {
    WD s{0.f, 0.f, 0.f};
    s = w_online<1>(v[g * 4 + 0], s);
    s = w_online<2>(v[g * 4 + 1], s);
    s = w_online<3>(v[g * 4 + 2], s);
    s = w_online<4>(v[g * 4 + 3], s);
    wd[g] = s;
  }
  // intra-warp shuffle-down tree: the count doubles at every level
#pragma unroll
  for (int g = 0; g < NV; ++g) {
    wd[g] = w_comb<4, 4>(wd[g], w_down(wd[g], 16));
    wd[g] = w_comb<8, 8>(wd[g], w_down(wd[g], 8));
    wd[g] = w_comb<16, 16>(wd[g], w_down(wd[g], 4));
    wd[g] = w_comb<32, 32>(wd[g], w_down(wd[g], 2));
    wd[g] = w_comb<64, 64>(wd[g], w_down(wd[g], 1));
  }
  // ATen's 4-way inter-warp tree; groups past NV are the empty warps
  constexpr int C0 = 128, C1 = (NV > 1) ? 128 : 0;
  constexpr int C2 = (NV > 2) ? 128 : 0, C3 = (NV > 3) ? 128 : 0;
  const WD z{0.f, 0.f, 0.f};
  WD a0 = wd[0];
  WD a1 = (NV > 1) ? wd[NV > 1 ? 1 : 0] : z;
  WD a2 = (NV > 2) ? wd[NV > 2 ? 2 : 0] : z;
  WD a3 = (NV > 3) ? wd[NV > 3 ? 3 : 0] : z;
  WD A0 = w_comb<C2, C0>(a0, a2);
  WD A1 = w_comb<C3, C1>(a1, a3);
  WD F = w_comb<C1 + C3, C0 + C2>(A0, A1);
  mean = __shfl_sync(0xffffffffu, F.mean, 0);
  rstd = rsqrtf(__shfl_sync(0xffffffffu, F.sigma2, 0) / float(N) + EPSL);
}

// LayerNorm one row into ``out`` (and optionally a second copy into ``out2``).
template <int N>
__device__ __forceinline__ void ln_apply(const bf16* row, bf16* out, int lane,
                                         const float* w, const float* b,
                                         bf16* out2 = nullptr) {
  constexpr int NV = N / 128;
  float v[NV * 4];
  ln_stage<N>(row, lane, v);
  float mean, rstd;
  ln_stats<N>(v, mean, rstd);
#pragma unroll
  for (int g = 0; g < NV; ++g) {
    const int c0 = (g * 32 + lane) * 4;
    const float4 gw = *(const float4*)(w + c0);
    const float4 bb = *(const float4*)(b + c0);
    bf16 r[4];
    r[0] = f2b(gw.x * (rstd * (v[g * 4 + 0] - mean)) + bb.x);
    r[1] = f2b(gw.y * (rstd * (v[g * 4 + 1] - mean)) + bb.y);
    r[2] = f2b(gw.z * (rstd * (v[g * 4 + 2] - mean)) + bb.z);
    r[3] = f2b(gw.w * (rstd * (v[g * 4 + 3] - mean)) + bb.w);
    *(uint2*)(out + c0) = *(const uint2*)r;
    if (out2 != nullptr) *(uint2*)(out2 + c0) = *(const uint2*)r;
  }
}

// Normalise ROWS rows at once.  One warp has to walk the rows one at a time
// (the Welford tree is a warp-wide reduction over a single row), but at ~8 warps
// per SM there is nothing to hide the row's load latency with, so every row's
// load -- and the affine parameters, which are the same for all of them -- is
// issued up front and the reductions run out of registers afterwards.  Doing
// this one row at a time costs one full memory round trip per row and was the
// single largest term in the profile.
template <int N, int ROWS>
__device__ __forceinline__ void ln_rows(const bf16* in, int in_stride,
                                        bf16* out, int out_stride, int lane,
                                        const float* w, const float* b,
                                        bf16* out2 = nullptr, int out2_stride = 0) {
  constexpr int NV = N / 128;
  float v[ROWS][NV * 4];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) ln_stage<N>(in + (size_t)r * in_stride, lane, v[r]);
  float gw[NV * 4], bb[NV * 4];
#pragma unroll
  for (int g = 0; g < NV; ++g) {
    const int c0 = (g * 32 + lane) * 4;
    *(float4*)(gw + g * 4) = *(const float4*)(w + c0);
    *(float4*)(bb + g * 4) = *(const float4*)(b + c0);
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float mean, rstd;
    ln_stats<N>(v[r], mean, rstd);
#pragma unroll
    for (int g = 0; g < NV; ++g) {
      const int c0 = (g * 32 + lane) * 4;
      bf16 o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[e] = f2b(gw[g * 4 + e] * (rstd * (v[r][g * 4 + e] - mean)) + bb[g * 4 + e]);
      *(uint2*)(out + (size_t)r * out_stride + c0) = *(const uint2*)o;
      if (out2 != nullptr)
        *(uint2*)(out2 + (size_t)r * out2_stride + c0) = *(const uint2*)o;
    }
  }
}

// ---------------- softmax over 16 (ATen softmax_warp_forward) ----------------
__device__ __forceinline__ bf16 sm16(bf16 x) {
  float e = b2f(x), m = e;
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) {
    float o = __shfl_xor_sync(0xffffffffu, m, off, 16);
    m = m > o ? m : o;
  }
  e = expf(e - m);
  float s = e;
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off, 16);
  return f2b(e / s);
}

// ============================================================ kernels

// ---- triangle multiplication head ----------------------------------------
// LayerNorm(z) -> [a_p|a_g|b_p|b_g|g] -> a = mask*sig(a_g)*a_p, b likewise,
// gsig = sig(g).  Rows are independent: CTA = one 16-row tile x half the
// channel groups.
template <int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_tm_head(
    const bf16* __restrict__ z, const bf16* __restrict__ mask,
    const float* __restrict__ lnp, const bf16* __restrict__ W5,
    bf16* __restrict__ A, bf16* __restrict__ B, bf16* __restrict__ G) {
  __shared__ __align__(16) bf16 zln[NRES * SZ];
  const int mt = blockIdx.y, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bf16* zb = z + (size_t)mt * NRES * CZ;
  ln_rows<CZ, NRES / NWARP>(zb + (size_t)wid * CZ, NWARP * CZ,
                            zln + (size_t)wid * SZ, NWARP * SZ, lane, lnp, lnp + CZ);
  __syncthreads();

  const int nt = blockIdx.x * NWARP + wid;         // 0..15
  float d[5][4];
#pragma unroll
  for (int p = 0; p < 5; ++p) ZERO4(d[p]);
  mma_sw_np<CZ, 5>(d, zln, SZ, W5 + (size_t)nt * WTILE(CZ),
                   (size_t)(CZ / 8) * WTILE(CZ));
  const int r0 = MMA_ROW0, c0 = nt * 8 + MMA_COL0;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const size_t row = (size_t)(mt * NRES + r0 + h * 8);
    const bf16 mk = mask[row];
    bf16 av[2], bv[2], gv[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int j = h * 2 + u;
      av[u] = mulb(mulb(mk, sigb(f2b(d[1][j]))), f2b(d[0][j]));
      bv[u] = mulb(mulb(mk, sigb(f2b(d[3][j]))), f2b(d[2][j]));
      gv[u] = sigb(f2b(d[4][j]));
    }
    const size_t o = row * CZ + c0;
    *(uint32_t*)(A + o) = *(const uint32_t*)av;
    *(uint32_t*)(B + o) = *(const uint32_t*)bv;
    *(uint32_t*)(G + o) = *(const uint32_t*)gv;
  }
}

// ---- triangle multiplication product ------------------------------------
// x[m][n][c] = sum_t a[.][c] * b[.][c].  The reference runs this as a batched
// 16x16x16 bf16 gemm on tensor cores; a scalar FFMA reduction over the 16 terms
// is *not* bit-identical (86 disagreements in 9.4M outputs measured over the
// whole stack, and fp64-exact accumulation gives 17), so it has to go through
// mma as well -- which means batching over channels, hence its own kernel.
#define PNC 4                       // channels per CTA (4 bf16 = one 8B store)
template <bool OUTGOING, int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_tm_prod(
    const bf16* __restrict__ A, const bf16* __restrict__ B,
    bf16* __restrict__ X) {
  __shared__ __align__(16) bf16 as[PNC * NRES * S16];
  __shared__ __align__(16) bf16 bs[PNC * NRES * S16];
  const int c0 = blockIdx.x * PNC, wid = threadIdx.x >> 5;
  for (int task = threadIdx.x; task < 2 * NRES * NRES; task += NWARP * 32) {
    const int op = task >> 8, mt = task & 255, m = mt >> 4, t = mt & 15;
    const int row = OUTGOING ? (m * NRES + t) : (t * NRES + m);
    const bf16* p = (op ? B : A) + (size_t)row * CZ + c0;
    bf16* dst = (op ? bs : as) + (size_t)m * S16 + t;
#pragma unroll
    for (int c = 0; c < PNC; ++c) dst[c * NRES * S16] = p[c];
  }
  __syncthreads();
  const int r0 = MMA_ROW0, cc = MMA_COL0;
  // One warp owns all PNC channels of one n-tile, so the PNC results for a given
  // (m, n) are adjacent in X and go out as a single 16-byte store; the other way
  // round (one channel per warp) makes every store a lone 2-byte scatter.
  for (int nt = wid; nt < NRES / 8; nt += NWARP) {
    float d[PNC][4];
#pragma unroll
    for (int c = 0; c < PNC; ++c) {
      ZERO4(d[c]);
      mma_tile<NRES>(d[c], as + (size_t)c * NRES * S16, S16,
                     bs + (size_t)c * NRES * S16 + (size_t)(nt * 8) * S16, S16);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int m = r0 + h * 8, n = nt * 8 + cc + u;
        bf16 v[PNC];
#pragma unroll
        for (int c = 0; c < PNC; ++c) v[c] = f2b(d[c][h * 2 + u]);
        *(uint2*)(X + (size_t)(m * NRES + n) * CZ + c0) = *(const uint2*)v;
      }
  }
}

// ---- triangle multiplication tail ----------------------------------------
// LayerNorm(x) -> linear_z -> * gsig -> z += .  Rows are independent.
template <int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_tm_fin(
    const bf16* __restrict__ X, const bf16* __restrict__ G,
    const float* __restrict__ lnp, const bf16* __restrict__ Wz,
    bf16* __restrict__ z) {
  __shared__ __align__(16) bf16 xl[NRES * SZ];
  const int mt = blockIdx.x, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  ln_rows<CZ, NRES / NWARP>(X + (size_t)(mt * NRES + wid) * CZ, NWARP * CZ,
                            xl + (size_t)wid * SZ, NWARP * SZ, lane, lnp, lnp + CZ);
  __syncthreads();
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
  {
    const int nt = blockIdx.y * NWARP + wid;
    float d[4]; ZERO4(d);
    mma_sw<CZ>(d, xl, SZ, Wz + (size_t)nt * WTILE(CZ));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const size_t row = (size_t)(mt * NRES + r0 + h * 8);
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const size_t o = row * CZ + nt * 8 + c0 + u;
        z[o] = addb(z[o], mulb(f2b(d[h * 2 + u]), G[o]));
      }
    }
  }
}

// ---- triangle attention, pass 1: LayerNorm + q|k|v|g + triangle bias -------
// Rows are independent here, so the CTA owns one 16-row slice of the (optionally
// transposed) pair view and one eighth of the 64 projection tiles -- 128 CTAs,
// each touching ~20 KB.  Folding this into the attention kernel instead (16 CTAs,
// each reading the whole 128 KB projection) measured 3x slower: with ~8 warps per
// SM the achievable per-SM read bandwidth is what binds, not total traffic.
//
// TB[h][a][b] = linear_z(LN(x))[a][b][h]: the reference's triangle bias is
// ``permute(linear_z(x),(2,0,1)).unsqueeze(-4)``, and unsqueeze(-4) on a 4-d
// tensor inserts at position 1 -- so the bias is indexed by (query, key) over the
// whole pair matrix and broadcast across the leading residue axis, not sliced per
// row.  ``x`` row (a, b) is z row a*16+b for the starting node and b*16+a for the
// ending node, and either way this CTA's rows are x rows (mt, r), so the bias
// index is mt*16+r in both cases.
template <bool START, int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_ta_qkvg(
    const bf16* __restrict__ z, const float* __restrict__ lnp,
    const bf16* __restrict__ Wqkvg, const bf16* __restrict__ Wzt,
    bf16* __restrict__ Q, bf16* __restrict__ K, bf16* __restrict__ VT,
    bf16* __restrict__ G, bf16* __restrict__ TB) {
  __shared__ __align__(16) bf16 xl[NRES * SZ];
  const int mt = blockIdx.y, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int RSTRIDE = START ? CZ : NRES * CZ;   // step between this CTA's rows
  const size_t r0off = START ? (size_t)mt * NRES * CZ : (size_t)mt * CZ;
  ln_rows<CZ, NRES / NWARP>(z + r0off + (size_t)wid * RSTRIDE, NWARP * RSTRIDE,
                            xl + (size_t)wid * SZ, NWARP * SZ, lane, lnp, lnp + CZ);
  __syncthreads();
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
  const int nt = blockIdx.x * NWARP + wid;          // 0..63
  {
    float d[4]; ZERO4(d);
    mma_sw<CZ>(d, xl, SZ, Wqkvg + (size_t)nt * WTILE(CZ));
    const int plane = nt / (TAHD / 8), col = (nt % (TAHD / 8)) * 8 + c0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = r0 + h * 8;
      const size_t o = (size_t)(mt * NRES + r) * CZ + col;
      bf16 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const bf16 x = f2b(d[h * 2 + u]);
        if (plane == 0)      v[u] = f2b(b2f(x) / RS32);
        else if (plane == 3) v[u] = sigb(x);
        else                 v[u] = x;
      }
      if (plane == 2) {
#pragma unroll
        for (int u = 0; u < 2; ++u)
          VT[((size_t)mt * CZ + col + u) * NRES + r] = v[u];
      } else {
        bf16* dst = (plane == 0) ? Q : ((plane == 1) ? K : G);
        *(uint32_t*)(dst + o) = *(const uint32_t*)v;
      }
    }
  }
  if (blockIdx.x == 0 && wid == 0) {
    float d[4]; ZERO4(d);
    mma_sw<CZ>(d, xl, SZ, Wzt);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int r = r0 + h * 8, hh = c0 + u;
        if (hh < TAH) TB[(size_t)hh * NTOK + mt * NRES + r] = f2b(d[h * 2 + u]);
      }
  }
}

// ---- triangle attention, pass 2: masked softmax attention + gate -----------
template <bool START, int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_ta_attn(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ VT, const bf16* __restrict__ G,
    const bf16* __restrict__ TB, const bf16* __restrict__ mask,
    bf16* __restrict__ OS) {
  __shared__ __align__(16) bf16 qs[NRES * SZ];
  __shared__ __align__(16) bf16 ks[NRES * SZ];
  __shared__ __align__(16) bf16 vt[TAHD * S16];
  __shared__ __align__(16) bf16 sc[TAH * NRES * S16];
  __shared__ bf16 mb[NRES];
  const int i = blockIdx.x, wid = threadIdx.x >> 5, nthr = NWARP * 32;
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
  for (int idx = threadIdx.x * 4; idx < NRES * CZ; idx += nthr * 4) {
    const int r = idx / CZ, c = idx - r * CZ;
    *(uint2*)(qs + r * SZ + c) = *(const uint2*)(Q + (size_t)(i * NRES + r) * CZ + c);
    *(uint2*)(ks + r * SZ + c) = *(const uint2*)(K + (size_t)(i * NRES + r) * CZ + c);
  }
  for (int idx = threadIdx.x * 4; idx < TAHD * NRES; idx += nthr * 4) {
    const int c = idx / NRES, r = idx - c * NRES;
    *(uint2*)(vt + c * S16 + r) = *(const uint2*)(VT + (size_t)(i * CZ + c) * NRES + r);
  }
  if (threadIdx.x < NRES)
    mb[threadIdx.x] = maskbias(mask[START ? (i * NRES + threadIdx.x)
                                          : (threadIdx.x * NRES + i)]);
  __syncthreads();
  for (int w = wid; w < TAH * (NRES / 8); w += NWARP) {
    const int hd = w / (NRES / 8), nt = w % (NRES / 8);
    float d[4]; ZERO4(d);
    mma_tile<TAD>(d, qs + hd * TAD, SZ, ks + (size_t)(nt * 8) * SZ + hd * TAD, SZ);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int q = r0 + h * 8, key = nt * 8 + c0 + u;
        sc[(hd * NRES + q) * S16 + key] =
            addb(addb(f2b(d[h * 2 + u]), mb[key]),
                 TB[((size_t)hd * NRES + q) * NRES + key]);
      }
  }
  __syncthreads();
  {
    const int l16 = threadIdx.x & 15, sub = (threadIdx.x >> 4) & 1;
    for (int row = wid * 2 + sub; row < TAH * NRES; row += NWARP * 2)
      sc[row * S16 + l16] = sm16(sc[row * S16 + l16]);
  }
  __syncthreads();
  for (int w = wid; w < TAH * (TAD / 8); w += NWARP) {
    const int hd = w / (TAD / 8), nt = w % (TAD / 8);
    float d[4]; ZERO4(d);
    mma_tile<NRES>(d, sc + (size_t)hd * NRES * S16, S16,
                   vt + (size_t)(hd * TAD + nt * 8) * S16, S16);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int q = r0 + h * 8;
      const int c = hd * TAD + nt * 8 + c0;
      bf16 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u)
        v[u] = mulb(f2b(d[h * 2 + u]), G[(size_t)(i * NRES + q) * CZ + c + u]);
      *(uint32_t*)(OS + (size_t)(i * NRES + q) * CZ + c) = *(const uint32_t*)v;
    }
  }
}

// ---- triangle attention, pass 3: linear_o + residual ----------------------
template <bool START, int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_ta_out(
    const bf16* __restrict__ OS, const bf16* __restrict__ Wo,
    bf16* __restrict__ z) {
  __shared__ __align__(16) bf16 os[NRES * SZ];
  const int mt = blockIdx.y, wid = threadIdx.x >> 5, nthr = NWARP * 32;
  for (int idx = threadIdx.x * 4; idx < NRES * CZ; idx += nthr * 4) {
    const int r = idx / CZ, c = idx - r * CZ;
    *(uint2*)(os + r * SZ + c) = *(const uint2*)(OS + (size_t)(mt * NRES + r) * CZ + c);
  }
  __syncthreads();
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
  const int nt = blockIdx.x * NWARP + wid;
  float d[4]; ZERO4(d);
  mma_sw<TAHD>(d, os, SZ, Wo + (size_t)nt * WTILE(TAHD));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = r0 + h * 8;
    const size_t row = START ? (size_t)(mt * NRES + r) : (size_t)(r * NRES + mt);
    bf16 v[2];
#pragma unroll
    for (int u = 0; u < 2; ++u)
      v[u] = addb(z[row * CZ + nt * 8 + c0 + u], f2b(d[h * 2 + u]));
    *(uint32_t*)(z + row * CZ + nt * 8 + c0) = *(const uint32_t*)v;
  }
}

// ---- transition head (LayerNorm + SwiGLU hidden) --------------------------
template <int CIN, int HID, int MT, int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_mlp_head(
    const bf16* __restrict__ X, const float* __restrict__ lnp,
    const bf16* __restrict__ Wab, bf16* __restrict__ H) {
  constexpr int LD = CIN + 8;
  __shared__ __align__(16) bf16 xl[MT * NRES * LD];
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * MT * NRES;
  ln_rows<CIN, MT * NRES / NWARP>(X + (size_t)(row0 + wid) * CIN, NWARP * CIN,
                                  xl + (size_t)wid * LD, NWARP * LD, lane,
                                  lnp, lnp + CIN);
  __syncthreads();
  const int nt = blockIdx.y * NWARP + wid;
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    float dd[2][4]; ZERO4(dd[0]); ZERO4(dd[1]);
    const bf16* a = xl + (size_t)mt * NRES * LD;
    mma_sw_np<CIN, 2>(dd, a, LD, Wab + (size_t)nt * WTILE(CIN),
                      (size_t)(HID / 8) * WTILE(CIN));
    const float* da = dd[0]; const float* db = dd[1];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const size_t row = (size_t)(row0 + mt * NRES + r0 + h * 8);
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int j = h * 2 + u;
        H[row * HID + nt * 8 + c0 + u] = mulb(silub(f2b(da[j])), f2b(db[j]));
      }
    }
  }
}

// ---- transition tail (linear_out, mask, residual) -------------------------
template <int HID, int COUT, int MT, int NWARP, bool STAGE>
__global__ __launch_bounds__(NWARP * 32) void k_mlp_tail(
    const bf16* __restrict__ H, const bf16* __restrict__ Wout,
    const bf16* __restrict__ mask, bf16* __restrict__ X) {
  // Without staging every warp re-reads the whole hidden block from global (the
  // mma A operand is the same for all n-tiles), which is NWARP times the traffic.
  constexpr int LDH = HID + 8;
  __shared__ __align__(16) bf16 hs[STAGE ? MT * NRES * LDH : 1];
  const int wid = threadIdx.x >> 5;
  const int row0 = blockIdx.x * MT * NRES;
  if (STAGE) {
    for (int i = threadIdx.x * 8; i < MT * NRES * HID; i += NWARP * 32 * 8) {
      const int r = i / HID, c = i - r * HID;
      *(uint4*)(hs + r * LDH + c) = *(const uint4*)(H + (size_t)(row0 + r) * HID + c);
    }
    __syncthreads();
  }
  const int nt = blockIdx.y * NWARP + wid;
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    float d[4]; ZERO4(d);
    mma_sw<HID>(d, STAGE ? (hs + (size_t)mt * NRES * LDH)
                         : (H + (size_t)(row0 + mt * NRES) * HID),
                STAGE ? LDH : HID, Wout + (size_t)nt * WTILE(HID));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const size_t row = (size_t)(row0 + mt * NRES + r0 + h * 8);
      const bf16 mk = mask[row];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const size_t o = row * COUT + nt * 8 + c0 + u;
        X[o] = addb(X[o], mulb(f2b(d[h * 2 + u]), mk));
      }
    }
  }
}

// ---- attention pair bias, pass 1 ----------------------------------------
// blockIdx.x < APB_NBIAS : LayerNorm(z) -> linear_z -> pair bias [h][i][j]
// else                   : LayerNorm(s) -> q|k|v|g (q: +bias, /sqrt(d))
#define APB_NBIAS NRES
template <int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_apb_pre(
    const bf16* __restrict__ z, const bf16* __restrict__ s,
    const float* __restrict__ lnz, const bf16* __restrict__ Wzb,
    const float* __restrict__ lna, const bf16* __restrict__ Wqkvg,
    const bf16* __restrict__ qbias, bf16* __restrict__ BIAS,
    bf16* __restrict__ QS, bf16* __restrict__ KS, bf16* __restrict__ VT,
    bf16* __restrict__ GS) {
  __shared__ __align__(16) bf16 zl[NRES * SS];
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
  if (blockIdx.x < APB_NBIAS) {
    const int row0 = blockIdx.x * NRES;
    ln_rows<CZ, NRES / NWARP>(z + (size_t)(row0 + wid) * CZ, NWARP * CZ,
                              zl + (size_t)wid * SZ, NWARP * SZ, lane, lnz, lnz + CZ);
    __syncthreads();
    for (int nt = wid; nt < APH / 8; nt += NWARP) {
      float d[4]; ZERO4(d);
      mma_sw<CZ>(d, zl, SZ, Wzb + (size_t)nt * WTILE(CZ));
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int zrow = row0 + r0 + h * 8;
          BIAS[(size_t)(nt * 8 + c0 + u) * NTOK + zrow] = f2b(d[h * 2 + u]);
        }
    }
  } else {
    bf16* sl = zl;
    ln_rows<CS, NRES / NWARP>(s + (size_t)wid * CS, NWARP * CS,
                              sl + (size_t)wid * SS, NWARP * SS, lane, lna, lna + CS);
    __syncthreads();
    const int nt = (blockIdx.x - APB_NBIAS) * NWARP + wid;   // 0..191
    float d[4]; ZERO4(d);
    mma_sw<CS>(d, sl, SS, Wqkvg + (size_t)nt * WTILE(CS));
    const int plane = nt / (APHD / 8), col = (nt % (APHD / 8)) * 8 + c0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int q = r0 + h * 8;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = col + u;                 // head*APD + d
        const int hh = c / APD, dd = c - hh * APD;
        const float acc = d[h * 2 + u];
        if (plane == 0)
          QS[(size_t)q * APQ + hh * APDP + dd] =
              f2b(b2f(f2b(acc + b2f(qbias[c]))) / RS24);
        else if (plane == 1) KS[(size_t)q * APQ + hh * APDP + dd] = f2b(acc);
        else if (plane == 2) VT[(size_t)c * NRES + q] = f2b(acc);
        else                 GS[(size_t)q * APHD + c] = sigb(f2b(acc));
      }
    }
  }
}

// ---- attention pair bias, pass 2 ----------------------------------------
template <int NWARP>
__global__ __launch_bounds__(NWARP * 32) void k_apb_post(
    const bf16* __restrict__ mask, const bf16* __restrict__ Wo,
    const bf16* __restrict__ QS, const bf16* __restrict__ KS,
    const bf16* __restrict__ VT, const bf16* __restrict__ GS,
    const bf16* __restrict__ BIAS, bf16* __restrict__ s) {
  __shared__ __align__(16) bf16 sc[APH * NRES * S16];
  __shared__ __align__(16) bf16 os[NRES * SS];
  __shared__ bf16 mb[NRES];
  const int wid = threadIdx.x >> 5;
  const int r0 = MMA_ROW0, c0 = MMA_COL0;
  if (threadIdx.x < NRES) mb[threadIdx.x] = maskbias(mask[threadIdx.x]);
  __syncthreads();
  for (int w = wid; w < APH * (NRES / 8); w += NWARP) {
    const int hd = w / (NRES / 8), nt = w % (NRES / 8);
    float d[4]; ZERO4(d);
    mma_tile<APDP>(d, QS + hd * APDP, APQ, KS + (size_t)(nt * 8) * APQ + hd * APDP, APQ);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int q = r0 + h * 8, key = nt * 8 + c0 + u;
        sc[(hd * NRES + q) * S16 + key] =
            addb(addb(f2b(d[h * 2 + u]), mb[key]),
                 BIAS[((size_t)hd * NRES + q) * NRES + key]);
      }
  }
  __syncthreads();
  {
    const int l16 = threadIdx.x & 15, sub = (threadIdx.x >> 4) & 1;
    for (int row = wid * 2 + sub; row < APH * NRES; row += NWARP * 2)
      sc[row * S16 + l16] = sm16(sc[row * S16 + l16]);
  }
  __syncthreads();
  for (int w = wid; w < APH * (APD / 8); w += NWARP) {
    const int hd = w / (APD / 8), nt = w % (APD / 8);
    float d[4]; ZERO4(d);
    mma_tile<NRES>(d, sc + (size_t)hd * NRES * S16, S16,
                   VT + (size_t)(hd * APD + nt * 8) * NRES, NRES);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int q = r0 + h * 8, c = hd * APD + nt * 8 + c0 + u;
        os[q * SS + c] = mulb(f2b(d[h * 2 + u]), GS[(size_t)q * APHD + c]);
      }
  }
  __syncthreads();
  const int nt = blockIdx.x * NWARP + wid;
  float d[4]; ZERO4(d);
  mma_sw<APHD>(d, os, SS, Wo + (size_t)nt * WTILE(APHD));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int q = r0 + h * 8;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const size_t o = (size_t)q * CS + nt * 8 + c0 + u;
      s[o] = addb(s[o], f2b(d[h * 2 + u]));
    }
  }
}

// ============================================================ driver
#define WB_TM   (6 * CZ * CZ)                            // W5 (5) + Wz
#define WB_TA   (4 * TAHD * CZ + 8 * CZ + CZ * TAHD)     // qkvg + Wz(pad 8) + Wo
#define WB_PT   (2 * PTH * CZ + CZ * PTH)
#define WB_APB  (APH * CZ + 4 * APHD * CS + 512 + CS * CS)
#define WB_ST   (2 * STH * CS + CS * STH)
#define WB_BLK  (2 * WB_TM + 2 * WB_TA + WB_PT + WB_APB + WB_ST)
#define WF_BLK  (8 * CZ + 4 * CZ + 2 * CZ + 2 * CZ + 2 * CS + 2 * CS)

struct Scr { bf16 *A, *B, *G, *X, *HZ, *XL, *TB, *QS, *KS, *VT, *GS, *BI, *HS; };

// ``budget`` (-1 = unlimited) caps how many kernels of the block run; used only
// by the stage-by-stage bit-exactness test.
#define LAUNCH(k) do { if (*budget == 0) return; --(*budget); k; } while (0)

// The pair stream and the single stream are independent: z_{i+1} depends only on
// z_i, and s_{i+1} on (s_i, z_i).  So once block i's pair update is finished the
// single update for block i can run *alongside* block i+1's pair update.  These
// kernels use ~2% of the machine each (the problem is 16 tokens wide), so the
// overlap is nearly free.  ``z`` itself is the only shared input: the single side
// reads it in ``k_apb_pre``, so the pair side waits for that before the next
// ``k_tm_fin`` -- the first kernel of the following block that writes z.
static void run_pair(const bf16*& w, const float*& f, bf16* z,
                     const bf16* pmask, const Scr& sc, cudaStream_t st,
                     cudaEvent_t wait_before_write, int* budget) {
  // --- triangle multiplication: outgoing then incoming
  for (int pass = 0; pass < 2; ++pass) {
    const float* ln_in = f; const float* ln_out = f + 2 * CZ; f += 4 * CZ;
    const bf16* W5 = w; const bf16* Wz = w + 5 * CZ * CZ; w += WB_TM;
    LAUNCH((k_tm_head<8><<<dim3(CZ / 8 / 8, NRES), 8 * 32, 0, st>>>(
        z, pmask, ln_in, W5, sc.A, sc.B, sc.G)));
    if (pass == 0) LAUNCH((k_tm_prod<true, 2><<<CZ / PNC, 2 * 32, 0, st>>>(sc.A, sc.B, sc.X)));
    else           LAUNCH((k_tm_prod<false, 2><<<CZ / PNC, 2 * 32, 0, st>>>(sc.A, sc.B, sc.X)));
    if (pass == 0 && wait_before_write) cudaStreamWaitEvent(st, wait_before_write, 0);
    LAUNCH((k_tm_fin<8><<<dim3(NRES, CZ / 8 / 8), 8 * 32, 0, st>>>(
        sc.X, sc.G, ln_out, Wz, z)));
  }
  // --- triangle attention: starting then ending node
  for (int pass = 0; pass < 2; ++pass) {
    const float* ln = f; f += 2 * CZ;
    const bf16* Wq = w; const bf16* Wzt = w + 4 * TAHD * CZ;
    const bf16* Wo = Wzt + 8 * CZ; w += WB_TA;
    if (pass == 0) {
      LAUNCH((k_ta_qkvg<true, 8><<<dim3(4 * TAHD / 8 / 8, NRES), 8 * 32, 0, st>>>(
          z, ln, Wq, Wzt, sc.A, sc.B, sc.X, sc.G, sc.TB)));
      LAUNCH((k_ta_attn<true, 8><<<NRES, 8 * 32, 0, st>>>(
          sc.A, sc.B, sc.X, sc.G, sc.TB, pmask, sc.XL)));
      LAUNCH((k_ta_out<true, 8><<<dim3(CZ / 8 / 8, NRES), 8 * 32, 0, st>>>(sc.XL, Wo, z)));
    } else {
      LAUNCH((k_ta_qkvg<false, 8><<<dim3(4 * TAHD / 8 / 8, NRES), 8 * 32, 0, st>>>(
          z, ln, Wq, Wzt, sc.A, sc.B, sc.X, sc.G, sc.TB)));
      LAUNCH((k_ta_attn<false, 8><<<NRES, 8 * 32, 0, st>>>(
          sc.A, sc.B, sc.X, sc.G, sc.TB, pmask, sc.XL)));
      LAUNCH((k_ta_out<false, 8><<<dim3(CZ / 8 / 8, NRES), 8 * 32, 0, st>>>(sc.XL, Wo, z)));
    }
  }
  // --- pair transition
  {
    const float* ln = f; f += 2 * CZ;
    const bf16* Wab = w; const bf16* Wout = w + 2 * PTH * CZ; w += WB_PT;
    LAUNCH((k_mlp_head<CZ, PTH, 4, 8><<<dim3(4, PTH / 8 / 8), 8 * 32, 0, st>>>(
        z, ln, Wab, sc.HZ)));
    LAUNCH((k_mlp_tail<PTH, CZ, 1, 8, true><<<dim3(NRES, CZ / 8 / 8), 8 * 32, 0, st>>>(
        sc.HZ, Wout, pmask, z)));
  }
}

static void run_single(const bf16*& w, const float*& f, const bf16* z, bf16* s,
                       const bf16* smask, const Scr& sc, cudaStream_t st,
                       cudaEvent_t after_z_read, int* budget) {
  {
    const float* lnz = f; const float* lna = f + 2 * CZ; f += 2 * CZ + 2 * CS;
    const bf16* Wzb = w; const bf16* Wqkvg = w + APH * CZ;
    const bf16* qb = Wqkvg + 4 * APHD * CS; const bf16* Wo = qb + 512;
    w += WB_APB;
    LAUNCH((k_apb_pre<8><<<APB_NBIAS + 4 * APHD / 8 / 8, 8 * 32, 0, st>>>(
        z, s, lnz, Wzb, lna, Wqkvg, qb, sc.BI, sc.QS, sc.KS, sc.VT, sc.GS)));
    if (after_z_read) cudaEventRecord(after_z_read, st);
    LAUNCH((k_apb_post<8><<<CS / 8 / 8, 8 * 32, 0, st>>>(
        smask, Wo, sc.QS, sc.KS, sc.VT, sc.GS, sc.BI, s)));
  }
  {
    const float* ln = f;
    const bf16* Wab = w; const bf16* Wout = w + 2 * STH * CS;
    LAUNCH((k_mlp_head<CS, STH, 1, 8><<<dim3(1, STH / 8 / 8), 8 * 32, 0, st>>>(
        s, ln, Wab, sc.HS)));
    LAUNCH((k_mlp_tail<STH, CS, 1, 1, false><<<dim3(1, CS / 8 / 1), 1 * 32, 0, st>>>(
        sc.HS, Wout, smask, s)));
  }
}

// Persistent side stream + per-block events; both have to exist before capture.
struct Pipe {
  cudaStream_t side = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
  std::vector<cudaEvent_t> zdone, sread;
  void ensure(int nb) {
    if (!side) {
      cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking);
      cudaEventCreateWithFlags(&fork, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
    }
    while ((int)zdone.size() < nb) {
      cudaEvent_t a, b;
      cudaEventCreateWithFlags(&a, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&b, cudaEventDisableTiming);
      zdone.push_back(a); sread.push_back(b);
    }
  }
};
static Pipe g_pipe;

void pairformer(torch::Tensor s, torch::Tensor z, torch::Tensor smask,
                torch::Tensor pmask, torch::Tensor wbf, torch::Tensor wf32,
                torch::Tensor scratch, int64_t nblocks, int64_t nstages) {
  const at::cuda::OptionalCUDAGuard guard(device_of(z));
  auto main = at::cuda::getCurrentCUDAStream().stream();
  bf16* sp = (bf16*)scratch.data_ptr();
  Scr sc;
  size_t o = 0;
#define TAKE(fld, n) do { sc.fld = sp + o; o += (n); } while (0)
  TAKE(A, NTOK * CZ); TAKE(B, NTOK * CZ); TAKE(G, NTOK * CZ); TAKE(X, NTOK * CZ);
  TAKE(HZ, NTOK * PTH); TAKE(XL, NTOK * CZ); TAKE(TB, TAH * NTOK);
  TAKE(QS, NRES * APQ); TAKE(KS, NRES * APQ); TAKE(VT, APHD * NRES);
  TAKE(GS, NRES * APHD); TAKE(BI, APH * NTOK); TAKE(HS, NRES * STH);
#undef TAKE
  TORCH_CHECK(scratch.numel() >= (int64_t)o, "scratch too small");
  const bf16* wb = (const bf16*)wbf.data_ptr();
  const float* wf = wf32.data_ptr<float>();
  bf16* zp = (bf16*)z.data_ptr();
  bf16* spp = (bf16*)s.data_ptr();
  const bf16* sm = (const bf16*)smask.data_ptr();
  const bf16* pm = (const bf16*)pmask.data_ptr();
  int budget = nstages > 0 ? (int)nstages : -1;

  if (nstages > 0) {                     // debug path: strict program order
    for (int64_t b = 0; b < nblocks; ++b) {
      const bf16* w = wb + b * WB_BLK; const float* f = wf + b * WF_BLK;
      run_pair(w, f, zp, pm, sc, main, nullptr, &budget);
      run_single(w, f, zp, spp, sm, sc, main, nullptr, &budget);
    }
    return;
  }

  g_pipe.ensure((int)nblocks);
  cudaEventRecord(g_pipe.fork, main);
  cudaStreamWaitEvent(g_pipe.side, g_pipe.fork, 0);
  for (int64_t b = 0; b < nblocks; ++b) {
    const bf16* w = wb + b * WB_BLK; const float* f = wf + b * WF_BLK;
    run_pair(w, f, zp, pm, sc, main,
             b > 0 ? g_pipe.sread[b - 1] : nullptr, &budget);
    cudaEventRecord(g_pipe.zdone[b], main);
    cudaStreamWaitEvent(g_pipe.side, g_pipe.zdone[b], 0);
    run_single(w, f, zp, spp, sm, sc, g_pipe.side, g_pipe.sread[b], &budget);
  }
  cudaEventRecord(g_pipe.join, g_pipe.side);
  cudaStreamWaitEvent(main, g_pipe.join, 0);
}

int64_t wb_block() { return WB_BLK; }
int64_t wf_block() { return WF_BLK; }
int64_t scratch_size() {
  return 5 * NTOK * CZ + NTOK * PTH + TAH * NTOK + 2 * NRES * APQ
       + APHD * NRES + NRES * APHD + APH * NTOK + NRES * STH;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pairformer", &pairformer);
  m.def("wb_block", &wb_block);
  m.def("wf_block", &wf_block);
  m.def("scratch_size", &scratch_size);
}
