// Fused AlphaFold3 atom-level (cross-attention) diffusion-transformer stack.
//
// The captured geometry is 368 atoms x 128 channels, 3 blocks, sequence-local
// attention with n_query=32 / n_key=128, and a 1.5 MB blocked pair tensor.  The
// arithmetic is ~0.7 GFLOP -- a few microseconds of a B200 -- so, exactly as in
// the token-level stack, the cost is launch count and synchronization, not work.
// Composed out of the frozen L2 winners it is 16 launches and 184 us of device
// time; this is one launch.
//
// The shape of the solution follows from two measurements on this machine:
//
//   * one SM sustains only ~45-60 GB/s of streaming global reads, so a design
//     where every CTA reads all 1.8 MB of weights costs ~35 us on the weight
//     stream alone.  The weights therefore have to be split by output column.
//   * a device-wide barrier costs ~1.0 us but a *cluster* barrier costs 0.25 us,
//     and distributed shared memory makes the exchange that follows it free.
//
// So: 96 CTAs in 24 clusters of 4.  Cluster `mt` owns atom rows [16mt, 16mt+16)
// -- always inside one key block, since n_query = 32 -- and rank `cs` of a
// cluster owns channels [32cs, 32cs+32) of every 128-wide tensor, which for
// c_hidden=32 is exactly attention head `cs`.  Every GEMM is split by output
// column across the four ranks, so each CTA reads a quarter of the weights;
// wherever the next GEMM needs the full reduction dimension, the four slices are
// pulled back together through distributed shared memory behind a cluster
// barrier.  Only the k/v projections need to cross cluster boundaries (a key
// block gathers atoms from anywhere), so there is exactly one device-wide
// barrier per transformer block.
//
// The gather indices are reproduced bit-exactly, including the reference's
// accidental bf16 index arithmetic: `_get_block_key_indices` promotes int32
// indices to bf16 when it subtracts `n_real - 1`, and bf16 cannot represent odd
// integers above 256, so for the later key blocks the index list genuinely
// contains duplicates (128 slots -> 65 distinct atoms) and one out-of-range slot
// that the invalid mask then zeroes.  See `blk_indices`.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

#define NWARP    16
#define THREADS  (NWARP * 32)
#define PAD      8
#define CLS      4          /* cluster size == no_heads == 128 / 32 channels */

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rb(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ uint32_t lds32(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint64_t ldg64(const void* p) {
  return *reinterpret_cast<const uint64_t*>(p);
}
struct __align__(8) H2 { __nv_bfloat162 h[2]; };
struct V4 { bf16 x[4]; };

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A-fragment of tile (., kt) of a row-major [16][ld] shared tile.
__device__ __forceinline__ void ld_afrag(uint32_t (&a)[4], const bf16* A, int ld,
                                         int kt) {
  const int lane = threadIdx.x & 31;
  const bf16* p = A + (lane >> 2) * ld + kt * 16 + ((lane & 3) << 1);
  a[0] = lds32(p);
  a[1] = lds32(p + 8 * ld);
  a[2] = lds32(p + 8);
  a[3] = lds32(p + 8 * ld + 8);
}
// B-fragment of tile (nt, kt) of a row-major [N][ld] shared tile.
__device__ __forceinline__ void ld_bfrag_s(uint32_t& b0, uint32_t& b1,
                                           const bf16* B, int ld, int nt, int kt) {
  const int lane = threadIdx.x & 31;
  const bf16* p = B + (nt * 8 + (lane >> 2)) * ld + kt * 16 + ((lane & 3) << 1);
  b0 = lds32(p);
  b1 = lds32(p + 8);
}
// B-fragment of tile (nt, kt) when the tile is stored *transposed*, [K][ld] with
// the reduction index as the row: what the P*V product needs out of v[slot][ch].
__device__ __forceinline__ void ld_bfrag_t(uint32_t& b0, uint32_t& b1,
                                           const bf16* T, int ld, int nt, int kt) {
  const int lane = threadIdx.x & 31;
  const int n = nt * 8 + (lane >> 2);
  const int k = kt * 16 + ((lane & 3) << 1);
  const bf16* p = T + (size_t)k * ld + n;
  uint32_t l0 = (uint32_t)__bfloat16_as_ushort(p[0]);
  uint32_t h0 = (uint32_t)__bfloat16_as_ushort(p[ld]);
  uint32_t l1 = (uint32_t)__bfloat16_as_ushort(p[8 * ld]);
  uint32_t h1 = (uint32_t)__bfloat16_as_ushort(p[9 * ld]);
  b0 = l0 | (h0 << 16);
  b1 = l1 | (h1 << 16);
}
// Packed-B operand address.  Weights arrive in mma B-fragment order with the
// fragments of two adjacent k-tiles adjacent inside each lane's slot, so one
// 16-byte load feeds two mmas: on this machine a per-lane 8-byte stream tops out
// near 22 GB/s per SM against ~45 GB/s for 16-byte, and it halves the
// instruction count as well.
template <int KT2>
__device__ __forceinline__ const bf16* bfrag2(const bf16* W, int nt, int kt2) {
  return W + ((size_t)nt * KT2 + kt2) * 256 + ((threadIdx.x & 31) << 3);
}

// NT output n-tiles x NKT k-tiles against a row-major [16][lda] A tile, with
// every B-fragment load issued before the first mma.  These GEMMs are tiny --
// one k-sweep is 8-16 eight-byte loads per lane -- so with a load inside the
// k-loop the warp simply waits out L2 latency NKT times in a row; issuing them
// all up front is worth ~5x on the conditioner stage.
template <int NT, int NKT>
__device__ __forceinline__ void gemm_nt(float (&acc)[NT][4], const bf16* A, int lda,
                                        const bf16* W, int nt0) {
  constexpr int KT2 = NKT / 2;
  static_assert(NKT % 2 == 0, "two k-tiles per packed load");
  uint4 bv[KT2][NT];
#pragma unroll
  for (int k2 = 0; k2 < KT2; k2++)
#pragma unroll
    for (int n = 0; n < NT; n++)
      bv[k2][n] = *(const uint4*)bfrag2<KT2>(W, nt0 + n, k2);
  // Left to itself ptxas sinks every one of these back next to the mma that
  // consumes it, which leaves one load in flight per warp and turns the stage
  // into a chain of L2 round trips -- 3x on the conditioners, measured.
  asm volatile("" ::: "memory");
#pragma unroll
  for (int n = 0; n < NT; n++) {
    acc[n][0] = 0.f; acc[n][1] = 0.f; acc[n][2] = 0.f; acc[n][3] = 0.f;
  }
#pragma unroll
  for (int k2 = 0; k2 < KT2; k2++) {
    uint32_t a0[4], a1[4];
    ld_afrag(a0, A, lda, 2 * k2);
    ld_afrag(a1, A, lda, 2 * k2 + 1);
#pragma unroll
    for (int n = 0; n < NT; n++) {
      mma16816(acc[n], a0, bv[k2][n].x, bv[k2][n].y);
      mma16816(acc[n], a1, bv[k2][n].z, bv[k2][n].w);
    }
  }
}


// Pull the cluster's four column slices together into one full tile.  Written as
// "read every rank, then store" on purpose: with the store inside the per-rank
// loop each rank costs its own distributed-shared round trip, and there are four
// of these exchanges per transformer block.
template <int ROWS, int CSLW, int SLD, int DLD>
__device__ __forceinline__ void cl_copy(bf16* dst, bf16* src, cg::cluster_group& cl) {
  constexpr int WU = CSLW / 2, NU = ROWS * WU;
  constexpr int REP = (NU + THREADS - 1) / THREADS;
  const int tid = threadIdx.x;
  const uint32_t* rp[CLS];
  uint32_t v[CLS][REP];
#pragma unroll
  for (int q = 0; q < CLS; q++) rp[q] = (const uint32_t*)cl.map_shared_rank(src, q);
#pragma unroll
  for (int q = 0; q < CLS; q++)
#pragma unroll
    for (int r = 0; r < REP; r++) {
      const int u = tid + r * THREADS;
      if (u < NU) v[q][r] = rp[q][(u / WU) * (SLD / 2) + (u % WU)];
    }
  asm volatile("" ::: "memory");
#pragma unroll
  for (int q = 0; q < CLS; q++)
#pragma unroll
    for (int r = 0; r < REP; r++) {
      const int u = tid + r * THREADS;
      if (u < NU)
        ((uint32_t*)(dst + (size_t)(u / WU) * DLD + q * CSLW))[u % WU] = v[q][r];
    }
}

// Same, but the four slices are added into the (replicated) residual stream.
template <int ROWS, int CSLW, int SLD, int DLD>
__device__ __forceinline__ void cl_add(bf16* dst, bf16* src, cg::cluster_group& cl,
                                       int r0, int na) {
  constexpr int WU = CSLW / 4, NU = ROWS * WU;
  constexpr int REP = (NU + THREADS - 1) / THREADS;
  const int tid = threadIdx.x;
  const H2* rp[CLS];
  H2 v[CLS][REP];
#pragma unroll
  for (int q = 0; q < CLS; q++) rp[q] = (const H2*)cl.map_shared_rank(src, q);
#pragma unroll
  for (int q = 0; q < CLS; q++)
#pragma unroll
    for (int r = 0; r < REP; r++) {
      const int u = tid + r * THREADS;
      if (u < NU) v[q][r] = rp[q][(u / WU) * (SLD / 4) + (u % WU)];
    }
  asm volatile("" ::: "memory");
#pragma unroll
  for (int q = 0; q < CLS; q++)
#pragma unroll
    for (int r = 0; r < REP; r++) {
      const int u = tid + r * THREADS;
      const int row = u / WU;
      if (u < NU && r0 + row < na) {
        H2* d = (H2*)(dst + (size_t)row * DLD + q * CSLW);
        H2 x = d[u % WU];
        x.h[0] = __hadd2(x.h[0], v[q][r].h[0]);
        x.h[1] = __hadd2(x.h[1], v[q][r].h[1]);
        d[u % WU] = x;
      }
    }
}

// ------------------------------------------------------------- device barrier
struct Sync { unsigned* cnt; unsigned* gen; };
__device__ __forceinline__ void stv(unsigned* p, unsigned v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void gbar(Sync s, int ncta, unsigned& gen) {
  __syncthreads();
  if (threadIdx.x < 32) {
    if (threadIdx.x == 0) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      unsigned t = atomicAdd(s.cnt, 1u);
      if (t == (unsigned)(ncta - 1)) {
        stv(s.cnt, 0u);
        asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(s.gen), "r"(gen + 1)
                     : "memory");
      }
    }
    unsigned v;
    do {
      asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(s.gen) : "memory");
    } while (v == gen);
  }
  __syncthreads();
  gen++;
}

// Weight sections; the host fills Params::off with per-block element offsets.
enum {
  W_LNQ = 0, W_GSQ, W_BGQ, W_LNK, W_GSK, W_BGK, W_LN2, W_GS2, W_BG2,
  W_AO, W_BAO, W_GC, W_BGC, W_Z, W_QKVG, W_BQ, W_O, W_SG, W_OUT, W_NSECT
};

struct Params {
  const bf16* a; const bf16* s; const bf16* z; const bf16* mask;
  const bf16* lnz;            // module-level layer_norm_z weight
  bf16* out;
  const bf16* W; long wstride;
  long off[W_NSECT];
  bf16* kv;                   // [2][2][NP][C]  double-buffered k / v
  unsigned* cnt; unsigned* gen;
  int NB, dbg;
  float eps, qdiv, inf;
  int has_mask;
};

// g * (bf16(layer_norm(x)) + b) for one four-element group, two bf16x2 ops.
__device__ __forceinline__ H2 adaln4(const float (&v)[4], float mu, float rs,
                                     H2 gv, H2 bv) {
  H2 o;
  o.h[0] = __hmul2(gv.h[0], __hadd2(__floats2bfloat162_rn((v[0] - mu) * rs,
                                                          (v[1] - mu) * rs),
                                    bv.h[0]));
  o.h[1] = __hmul2(gv.h[1], __hadd2(__floats2bfloat162_rn((v[2] - mu) * rs,
                                                          (v[3] - mu) * rs),
                                    bv.h[1]));
  return o;
}

// ---------------------------------------------------------------------------
// The reference's key-block gather, reproduced operation for operation.
//
// `_get_block_key_indices` builds int32 indices, then subtracts `n_real - 1`
// where n_real is a bf16 sum of the atom mask.  Type promotion makes the whole
// tail bf16, so both the index and the shift round to bf16 before they are
// added -- and bf16 has a spacing of 2 above 256.  Doing this in fp32 changes
// the gather for the last six key blocks.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bf(float x) { return rb(x); }

struct BlkIdx { int idx; float valid; };

template <int NQ, int NK>
__device__ BlkIdx blk_index(int qb, int j, float nreal) {
  const int sc = NQ / 2 + qb * NQ;
  const int i0 = sc - NK / 2, iL = sc + NK / 2 - 1;
  const float nm1 = bf(nreal - 1.f);              // (n_real - 1) in bf16
  const int under = i0 < 0 ? -i0 : 0;             // relu(-initial[..,0]), int32
  float ov = bf(bf((float)iL) - nm1);             // int32 promoted to bf16 first
  if (ov < 0.f) ov = 0.f;
  const float shift = under > 0 ? bf((float)under) : -ov;
  const float fin = bf(bf((float)(sc + j - NK / 2)) + shift);
  const float lim = nm1 > 0.f ? nm1 : 0.f;
  float safe = fin < 0.f ? 0.f : (fin > lim ? lim : fin);
  BlkIdx r;
  r.idx = (int)safe;
  r.valid = (fin < 0.f || fin >= nreal) ? 0.f : 1.f;
  return r;
}

// ###########################################################################
template <int NA, int NP, int C, int S, int H, int D, int FH, int CZ,
          int NQ, int NK, int NBK, int G>
__global__ __launch_bounds__(THREADS) __cluster_dims__(CLS, 1, 1)
void dit_cross(Params p) {
  constexpr int R = NP / (G / CLS);        // atom rows per cluster (16)
  constexpr int CSL = C / CLS;             // channel slice per rank (32)
  constexpr int HSL = FH / CLS;            // hidden slice per rank (64)
  constexpr int LDA = C + PAD, LDS = CSL + PAD, LDK = NK + PAD,
                LDH = FH + PAD, LDHS = HSL + PAD;
  constexpr int NKS = S / 16, NKC = C / 16, NKF = FH / 16, NKD = D / 16,
                NKK = NK / 16;
  constexpr int NTC = CSL / 8;             // n-tiles per rank of a C-wide output
  static_assert(R == 16 && CSL == 32 && H == CLS && D == CSL, "mapping");
  static_assert(NQ % R == 0, "a row tile must stay inside one key block");

  extern __shared__ char smem[];
  // Full (replicated) tiles first, then the per-rank slices that the cluster
  // exchanges.  Slice buffers must sit at the same offset in every rank.
  bf16* sa   = (bf16*)smem;                      // [R][LDA]  residual stream
  bf16* saq  = sa   + (size_t)R * LDA;           // [R][LDA]  a_q  / a2
  bf16* sak  = saq  + (size_t)R * LDA;           // [R][LDA]  a_k
  bf16* ssn  = sak  + (size_t)R * LDA;           // [4][R][LDA] s_nq,s_nk,s_n2,s
  bf16* scF  = ssn  + (size_t)4 * R * LDA;       // [8][R][LDA] conditioners
  bf16* soF  = scF  + (size_t)8 * R * LDA;       // [R][LDA]  gated attn output
  bf16* shF  = soF  + (size_t)R * LDA;           // [R][LDH]  swiglu hidden
  bf16* sK   = shF  + (size_t)R * LDH;           // [NK][LDS] gathered k slice
  bf16* sV   = sK   + (size_t)NK * LDS;          // [NK][LDS] gathered v slice
  bf16* sSC  = sV   + (size_t)NK * LDS;          // [R][LDK]  scores / probs
  bf16* sZB  = sSC  + (size_t)R * LDK;           // [3][R][LDK] pair bias
  bf16* sQ   = sZB  + (size_t)3 * R * LDK;       // [R][LDS]  q slice (head cs)
  bf16* sGT  = sQ   + (size_t)R * LDS;           // [R][LDS]  gate slice
  bf16* sO   = sGT  + (size_t)R * LDS;           // [R][LDS]  o slice
  bf16* scs  = sO   + (size_t)R * LDS;           // [8][R][LDS] conditioner slice
  bf16* shs  = scs  + (size_t)8 * R * LDS;       // [R][LDHS] hidden slice
  bf16* sup  = shs  + (size_t)R * LDHS;          // [R][LDS]  residual update
  bf16* szs  = sup  + (size_t)R * LDS;           // [3][H][R][CSL] zbias slice
  float* smu = (float*)(szs + (size_t)3 * H * R * CSL);
  float* srs = smu + R;
  float* smk = srs + R;                          // [R] atom mask
  int*   sid = (int*)(smk + R);                  // [NK] gather index
  float* svd = (float*)(sid + NK);               // [NK] slot validity

  cg::cluster_group cl = cg::this_cluster();
  const int cs = cl.block_rank(), mt = blockIdx.x / CLS;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int r0 = mt * R, qb = r0 / NQ, qi0 = r0 - qb * NQ;
  const int ch0 = cs * CSL;
  Sync sy{p.cnt, p.gen};
  // Every thread reads the generation counter itself: no CTA can advance it
  // before all of them have arrived at the first barrier.
  unsigned gen;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(gen) : "l"(p.gen) : "memory");

  // ------------------------------------------------- mask, gather, residual
  {
    float ms = 0.f;
    for (int i = tid; i < NA; i += THREADS) ms += p.has_mask ? b2f(p.mask[i]) : 1.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) ms += __shfl_xor_sync(0xffffffffu, ms, o);
    if (lane == 0) smu[w % R] = ms;          // per-warp partial in smu (R >= NWARP)
  }
  __syncthreads();
  float nreal;
  {
    float t = 0.f;
    for (int i = 0; i < NWARP; i++) t += smu[i];
    nreal = rb(t);                            // bf16 sum, as torch does it
  }
  for (int i = tid; i < NK; i += THREADS) {
    BlkIdx b = blk_index<NQ, NK>(qb, i, nreal);
    // `masked_fill_` zeroes the gathered row on *invalid* alone; the atom mask at
    // the key only enters the bias, so the two flags are kept apart.
    sid[i] = b.valid != 0.f ? b.idx : -1;
    float am = (b.idx < NA) ? (p.has_mask ? b2f(p.mask[b.idx]) : 1.f) : 0.f;
    svd[i] = rb(b.valid * am);                // (~invalid) * atom_mask_at_keys
  }
  if (tid < R) {
    const int r = r0 + tid;
    smk[tid] = (r < NA) ? (p.has_mask ? b2f(p.mask[r]) : 1.f) : 0.f;
  }
  // a (padded with zero rows) and s into shared; s is reused every block.
  for (int i = tid; i < R * (C / 4); i += THREADS) {
    const int r = i / (C / 4), c4 = i - r * (C / 4);
    V4 va{}, vs{};
    if (r0 + r < NA) {
      va = ((const V4*)(p.a + (size_t)(r0 + r) * C))[c4];
      vs = ((const V4*)(p.s + (size_t)(r0 + r) * S))[c4];
    }
    ((V4*)(sa + (size_t)r * LDA))[c4] = va;
    ((V4*)(ssn + (size_t)3 * R * LDA + (size_t)r * LDA))[c4] = vs;
  }
  __syncthreads();

  // --------------------------------------------- prologue: pair bias from z
  // Each rank reads the j-slice [CSL*cs, +CSL) of its own rows of z, normalizes
  // over c_z and applies all three blocks' linear_z; the cluster then swaps so
  // every rank ends up with its own head for all NK key slots.
  if (!(p.dbg & 1)) {
    const bf16* Wz0 = p.W + p.off[W_Z];
    // z is [NBK][NQ][NK][CZ]; this cluster's rows are qi0 + [0, R).
    for (int u = tid; u < R * CSL; u += THREADS) {
      const int i = u / CSL, jj = u - i * CSL, j = cs * CSL + jj;
      const bf16* zp = p.z + ((size_t)(qb * NQ + qi0 + i) * NK + j) * CZ;
      float v[CZ];
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < CZ; c++) { v[c] = b2f(zp[c]); s1 += v[c]; s2 += v[c] * v[c]; }
      const float mu = s1 / CZ;
      const float rs = rsqrtf(fmaxf(s2 / CZ - mu * mu, 0.f) + p.eps);
#pragma unroll
      for (int c = 0; c < CZ; c++) v[c] = rb((v[c] - mu) * rs * b2f(p.lnz[c]));
      for (int b = 0; b < p.NB; b++) {
        const bf16* Wz = Wz0 + (size_t)b * p.wstride;
#pragma unroll
        for (int h = 0; h < H; h++) {
          float acc = 0.f;
#pragma unroll
          for (int c = 0; c < CZ; c++) acc += v[c] * b2f(Wz[h * CZ + c]);
          szs[(((size_t)b * H + h) * R + i) * CSL + jj] = f2b(acc);
        }
      }
    }
  }
  cl.sync();
  for (int q = 0; q < CLS; q++) {
    const bf16* rp = (const bf16*)cl.map_shared_rank(szs, q);
    for (int u = tid; u < p.NB * R * (CSL / 2); u += THREADS) {
      const int t = u / (CSL / 2), j2 = u - t * (CSL / 2);
      const int b = t / R, i = t - b * R;
      ((uint32_t*)(sZB + ((size_t)b * R + i) * LDK + q * CSL))[j2] =
          ((const uint32_t*)(rp + (((size_t)b * H + cs) * R + i) * CSL))[j2];
    }
  }
  __syncthreads();

  // =========================================================== block loop
  for (int b = 0; b < ((p.dbg & 2) ? 0 : p.NB); b++) {
    const bf16* Wb = p.W + (size_t)b * p.wstride;
    bf16* gk = p.kv + (size_t)(b & 1) * 2 * NP * C;
    bf16* gv = gk + (size_t)NP * C;

    // ------------------------------------------- conditioners (column slice)
    // layer_norm_s of s (three different weights, one set of statistics), then
    // the eight [C][S] conditioner matrices restricted to this rank's columns.
    if (!(p.dbg & 4)) {
      const bf16* sraw = ssn + (size_t)3 * R * LDA;
      if (w < R) {
        const H2* xp = (const H2*)(sraw + (size_t)w * LDA);
        float s1 = 0.f, s2 = 0.f;
        float v[S / 4 / 32][4];
#pragma unroll
        for (int i = lane, u = 0; i < S / 4; i += 32, u++) {
          H2 t = xp[i];
          float2 f0 = __bfloat1622float2(t.h[0]), f1 = __bfloat1622float2(t.h[1]);
          v[u][0] = f0.x; v[u][1] = f0.y; v[u][2] = f1.x; v[u][3] = f1.y;
          s1 += (f0.x + f0.y) + (f1.x + f1.y);
          s2 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y,
               fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, s2))));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
          s1 += __shfl_xor_sync(0xffffffffu, s1, o);
          s2 += __shfl_xor_sync(0xffffffffu, s2, o);
        }
        const float mu = s1 / S;
        const float rs = rsqrtf(fmaxf(s2 / S - mu * mu, 0.f) + p.eps);
#pragma unroll
        for (int k = 0; k < 3; k++) {
          const bf16* wl = Wb + p.off[k == 0 ? W_LNQ : k == 1 ? W_LNK : W_LN2];
          H2* yp = (H2*)(ssn + ((size_t)k * R + w) * LDA);
#pragma unroll
          for (int i = lane, u = 0; i < S / 4; i += 32, u++) {
            H2 wv = ((const H2*)wl)[i], o;
            float2 w0 = __bfloat1622float2(wv.h[0]), w1 = __bfloat1622float2(wv.h[1]);
            o.h[0] = __floats2bfloat162_rn((v[u][0] - mu) * rs * w0.x,
                                           (v[u][1] - mu) * rs * w0.y);
            o.h[1] = __floats2bfloat162_rn((v[u][2] - mu) * rs * w1.x,
                                           (v[u][3] - mu) * rs * w1.y);
            yp[i] = o;
          }
        }
      }
      __syncthreads();
      // 8 kinds x NTC n-tiles = 32 tiles; warp w takes two adjacent n-tiles of
      // one kind, so both share the A operand and one hoisted load batch.
      constexpr int NTP = 8 * NTC / NWARP;        // n-tiles per warp (2)
      {
        const int kind = w / (NTC / NTP), nsub = (w % (NTC / NTP)) * NTP;
        int sec, nt, bsec, src;
        switch (kind) {
          case 0: sec = W_GSQ; nt = 0;      bsec = W_BGQ; src = 0; break;
          case 1: sec = W_GSQ; nt = C / 8;  bsec = -1;    src = 0; break;
          case 2: sec = W_GSK; nt = 0;      bsec = W_BGK; src = 1; break;
          case 3: sec = W_GSK; nt = C / 8;  bsec = -1;    src = 1; break;
          case 4: sec = W_GS2; nt = 0;      bsec = W_BG2; src = 2; break;
          case 5: sec = W_GS2; nt = C / 8;  bsec = -1;    src = 2; break;
          case 6: sec = W_AO;  nt = 0;      bsec = W_BAO; src = 3; break;
          default: sec = W_GC; nt = 0;      bsec = W_BGC; src = 3; break;
        }
        const bf16* Wm = Wb + p.off[sec];
        const bf16* A = ssn + (size_t)src * R * LDA;
        float acc[NTP][4];
        gemm_nt<NTP, NKS>(acc, A, LDA, Wm, nt + cs * NTC + nsub);
        const bf16* bi = bsec >= 0 ? Wb + p.off[bsec] : nullptr;
        bf16* dst = scs + ((size_t)kind * R + (lane >> 2)) * LDS;
#pragma unroll
        for (int n = 0; n < NTP; n++) {
          const int lc = (nsub + n) * 8 + ((lane & 3) << 1);   // slice-local
#pragma unroll
          for (int q = 0; q < 2; q++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
              float x = acc[n][q * 2 + e];
              if (bi) x = sigm(rb(x + b2f(bi[ch0 + lc + e])));
              dst[(size_t)q * 8 * LDS + lc + e] = f2b(x);
            }
          }
        }
      }
      cl.sync();
      cl_copy<8 * R, CSL, LDS, LDA>(scF, scs, cl);
      __syncthreads();
    }

    // ------------------------------------------- a_q / a_k (full, replicated)
    if (w < R && !(p.dbg & 8)) {
      const H2* xp = (const H2*)(sa + (size_t)w * LDA);
      float s1 = 0.f, s2 = 0.f;
      float v[C / 4 / 32][4];
#pragma unroll
      for (int i = lane, u = 0; i < C / 4; i += 32, u++) {
        H2 t = xp[i];
        float2 f0 = __bfloat1622float2(t.h[0]), f1 = __bfloat1622float2(t.h[1]);
        v[u][0] = f0.x; v[u][1] = f0.y; v[u][2] = f1.x; v[u][3] = f1.y;
        s1 += (f0.x + f0.y) + (f1.x + f1.y);
        s2 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y,
             fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, s2))));
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      }
      const float mu = s1 / C, rs = rsqrtf(fmaxf(s2 / C - mu * mu, 0.f) + p.eps);
      const H2* g0 = (const H2*)(scF + (size_t)(0 * R + w) * LDA);
      const H2* b0 = (const H2*)(scF + (size_t)(1 * R + w) * LDA);
      const H2* g1 = (const H2*)(scF + (size_t)(2 * R + w) * LDA);
      const H2* b1 = (const H2*)(scF + (size_t)(3 * R + w) * LDA);
      H2* yq = (H2*)(saq + (size_t)w * LDA);
      H2* yk = (H2*)(sak + (size_t)w * LDA);
#pragma unroll
      for (int i = lane, u = 0; i < C / 4; i += 32, u++) {
        yq[i] = adaln4(v[u], mu, rs, g0[i], b0[i]);
        yk[i] = adaln4(v[u], mu, rs, g1[i], b1[i]);
      }
    }
    __syncthreads();

    // --------------------------------- q / k / v / gate (this rank's columns)
    if (!(p.dbg & 16)) {
      const bf16* Wq = Wb + p.off[W_QKVG];
      const bf16* bq = Wb + p.off[W_BQ];
      for (int u = w; u < 4 * NTC; u += NWARP) {
        const int m = u / NTC, nsub = u - m * NTC;
        const bf16* A = (m == 0 || m == 3) ? saq : sak;
        float acc1[1][4];
        gemm_nt<1, NKC>(acc1, A, LDA, Wq, m * (C / 8) + cs * NTC + nsub);
        float* acc = acc1[0];
        const int col = (cs * NTC + nsub) * 8 + ((lane & 3) << 1);
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int r = (lane >> 2) + q * 8;
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const float x = acc[q * 2 + e];
            const int cc = col + e, lc = cc - ch0;
            if (m == 0) sQ[(size_t)r * LDS + lc] = f2b(rb(rb(x + b2f(bq[cc])) / p.qdiv));
            else if (m == 1) gk[(size_t)(r0 + r) * C + cc] = f2b(x);
            else if (m == 2) gv[(size_t)(r0 + r) * C + cc] = f2b(x);
            else sGT[(size_t)r * LDS + lc] = f2b(sigm(rb(x)));
          }
        }
      }
    }
    if (!(p.dbg & 8192)) gbar(sy, G, gen);

    // -------------------------------------------------- attention (head cs)
    if (!(p.dbg & 32)) {
      constexpr int NU = NK * (CSL / 4), REP = NU / THREADS;
      static_assert(NU % THREADS == 0, "gather split");
      V4 kk[REP], vv[REP];
      int jj[REP], cc[REP];
#pragma unroll
      for (int r = 0; r < REP; r++) {
        const int u = tid + r * THREADS;
        jj[r] = u / (CSL / 4); cc[r] = u - jj[r] * (CSL / 4);
        const int src = sid[jj[r]];
        kk[r] = V4{}; vv[r] = V4{};
        if (src >= 0) {
          kk[r] = ((const V4*)(gk + (size_t)src * C + ch0))[cc[r]];
          vv[r] = ((const V4*)(gv + (size_t)src * C + ch0))[cc[r]];
        }
      }
#pragma unroll
      for (int r = 0; r < REP; r++) {
        ((V4*)(sK + (size_t)jj[r] * LDS))[cc[r]] = kk[r];
        ((V4*)(sV + (size_t)jj[r] * LDS))[cc[r]] = vv[r];
      }
    }
    __syncthreads();
    // scores: NK/8 n-tiles over 16 warps, NKD k-tiles each.
    for (int nt = (p.dbg & 64) ? NK / 8 : w; nt < NK / 8; nt += NWARP) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kt = 0; kt < NKD; kt++) {
        uint32_t af[4], b0, b1;
        ld_afrag(af, sQ, LDS, kt);
        ld_bfrag_s(b0, b1, sK, LDS, nt, kt);
        mma16816(acc, af, b0, b1);
      }
      const bf16* zb = sZB + (size_t)b * R * LDK;
      const int col = nt * 8 + ((lane & 3) << 1);
#pragma unroll
      for (int q = 0; q < 2; q++) {
        const int r = (lane >> 2) + q * 8;
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int j = col + e;
          const float bm = rb(rb(rb(smk[r] * svd[j]) - 1.f) * p.inf);
          float x = rb(rb(rb(acc[q * 2 + e]) + bm) + b2f(zb[(size_t)r * LDK + j]));
          sSC[(size_t)r * LDK + j] = f2b(x);
        }
      }
    }
    __syncthreads();
    if (w < R && !(p.dbg & 128)) {                  // softmax, one row per warp
      bf16* row = sSC + (size_t)w * LDK;
      float v[NK / 32];
      float mx = -1e30f;
#pragma unroll
      for (int i = lane, u = 0; i < NK; i += 32, u++) {
        v[u] = b2f(row[i]);
        mx = fmaxf(mx, v[u]);
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      float sm = 0.f;
#pragma unroll
      for (int u = 0; u < NK / 32; u++) { v[u] = expf(v[u] - mx); sm += v[u]; }
#pragma unroll
      for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
#pragma unroll
      for (int i = lane, u = 0; i < NK; i += 32, u++) row[i] = f2b(v[u] / sm);
    }
    __syncthreads();
    // o = P V, gated: D/8 n-tiles over the first D/8 warps.
    if (w < D / 8 && !(p.dbg & 256)) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kt = 0; kt < NKK; kt++) {
        uint32_t af[4], b0, b1;
        ld_afrag(af, sSC, LDK, kt);
        ld_bfrag_t(b0, b1, sV, LDS, w, kt);
        mma16816(acc, af, b0, b1);
      }
      const int col = w * 8 + ((lane & 3) << 1);
#pragma unroll
      for (int q = 0; q < 2; q++) {
        const int r = (lane >> 2) + q * 8;
#pragma unroll
        for (int e = 0; e < 2; e++)
          sO[(size_t)r * LDS + col + e] =
              f2b(rb(rb(acc[q * 2 + e]) * b2f(sGT[(size_t)r * LDS + col + e])));
      }
    }
    cl.sync();
    cl_copy<R, CSL, LDS, LDA>(soF, sO, cl);
    __syncthreads();
    // linear_o -> residual, this rank's columns
    {
      const bf16* Wo = Wb + p.off[W_O];
      if (w < NTC && !(p.dbg & 512)) {
        float acc1[1][4];
        gemm_nt<1, NKC>(acc1, soF, LDA, Wo, cs * NTC + w);
        float* acc = acc1[0];
        const int col = w * 8 + ((lane & 3) << 1);
        const bf16* go = scF + (size_t)6 * R * LDA;
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int r = (lane >> 2) + q * 8;
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const int lc = col + e;
            sup[(size_t)r * LDS + lc] =
                f2b(rb(rb(acc[q * 2 + e]) * b2f(go[(size_t)r * LDA + ch0 + lc])));
          }
        }
      }
    }
    cl.sync();
    cl_add<R, CSL, LDS, LDA>(sa, sup, cl, r0, NA);
    __syncthreads();

    // ------------------------------------------- conditioned transition block
    if (w < R && !(p.dbg & 1024)) {
      const H2* xp = (const H2*)(sa + (size_t)w * LDA);
      float s1 = 0.f, s2 = 0.f;
      float v[C / 4 / 32][4];
#pragma unroll
      for (int i = lane, u = 0; i < C / 4; i += 32, u++) {
        H2 t = xp[i];
        float2 f0 = __bfloat1622float2(t.h[0]), f1 = __bfloat1622float2(t.h[1]);
        v[u][0] = f0.x; v[u][1] = f0.y; v[u][2] = f1.x; v[u][3] = f1.y;
        s1 += (f0.x + f0.y) + (f1.x + f1.y);
        s2 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y,
             fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, s2))));
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
        s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      }
      const float mu = s1 / C, rs = rsqrtf(fmaxf(s2 / C - mu * mu, 0.f) + p.eps);
      const H2* g2 = (const H2*)(scF + (size_t)(4 * R + w) * LDA);
      const H2* b2 = (const H2*)(scF + (size_t)(5 * R + w) * LDA);
      H2* yp = (H2*)(saq + (size_t)w * LDA);
#pragma unroll
      for (int i = lane, u = 0; i < C / 4; i += 32, u++)
        yp[i] = adaln4(v[u], mu, rs, g2[i], b2[i]);
    }
    __syncthreads();
    {   // swiglu, this rank's HSL hidden channels (2 HSL packed rows)
      const bf16* Wsg = Wb + p.off[W_SG];
      constexpr int NTH = HSL / 4;              // n-tiles of the interleaved pack
      for (int u = (p.dbg & 2048) ? NTH : w; u < NTH; u += NWARP) {
        float acc1[1][4];
        gemm_nt<1, NKC>(acc1, saq, LDA, Wsg, cs * NTH + u);
        float* acc = acc1[0];
        const int t = u * 4 + ((lane & 3) >> 0);   // (a,b) pair index in slice
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int r = (lane >> 2) + q * 8;
          const float ha = rb(acc[q * 2]), hb = rb(acc[q * 2 + 1]);
          shs[(size_t)r * LDHS + t] = f2b(rb(rb(ha * sigm(ha)) * hb));
        }
      }
    }
    cl.sync();
    cl_copy<R, HSL, LDHS, LDH>(shF, shs, cl);
    __syncthreads();
    {   // linear_out -> gated, masked residual
      const bf16* Wo = Wb + p.off[W_OUT];
      if (w < NTC && !(p.dbg & 4096)) {
        float acc1[1][4];
        gemm_nt<1, NKF>(acc1, shF, LDH, Wo, cs * NTC + w);
        float* acc = acc1[0];
        const int col = w * 8 + ((lane & 3) << 1);
        const bf16* gc = scF + (size_t)7 * R * LDA;
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int r = (lane >> 2) + q * 8;
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const int lc = col + e;
            float x = rb(rb(acc[q * 2 + e]) * b2f(gc[(size_t)r * LDA + ch0 + lc]));
            sup[(size_t)r * LDS + lc] = f2b(rb(x * smk[r]));
          }
        }
      }
    }
    cl.sync();
    cl_add<R, CSL, LDS, LDA>(sa, sup, cl, r0, NA);
    __syncthreads();
  }

  // The last residual exchange reads the cluster peers' shared memory, and
  // nothing downstream synchronizes the cluster again -- without this barrier a
  // rank that finishes its output store can exit while a slower peer is still
  // reading from it, which compute-sanitizer reports as an invalid shared read
  // and the driver eventually turns into a launch failure.
  cl.sync();

  // ------------------------------------------------------------------ output
  for (int u = tid; u < R * (CSL / 4); u += THREADS) {
    const int r = u / (CSL / 4), c4 = u - r * (CSL / 4);
    if (r0 + r >= NA) continue;
    ((V4*)(p.out + (size_t)(r0 + r) * C + ch0))[c4] =
        ((const V4*)(sa + (size_t)r * LDA + ch0))[c4];
  }
}

// ###########################################################################
#define CFG_X 368, 384, 128, 128, 4, 32, 256, 16, 32, 128, 12, 96
using PX = int;

template <int NA, int NP, int C, int S, int H, int D, int FH, int CZ,
          int NQ, int NK, int NBK, int G>
constexpr size_t smem_bytes() {
  constexpr int R = NP / (G / CLS), CSL = C / CLS, HSL = FH / CLS;
  constexpr int LDA = C + PAD, LDS = CSL + PAD, LDK = NK + PAD,
                LDH = FH + PAD, LDHS = HSL + PAD;
  size_t n = 0;
  n += (size_t)R * LDA;            // sa
  n += (size_t)R * LDA;            // saq
  n += (size_t)R * LDA;            // sak
  n += (size_t)4 * R * LDA;        // ssn
  n += (size_t)8 * R * LDA;        // scF
  n += (size_t)R * LDA;            // soF
  n += (size_t)R * LDH;            // shF
  n += (size_t)2 * NK * LDS;       // sK, sV
  n += (size_t)R * LDK;            // sSC
  n += (size_t)3 * R * LDK;        // sZB
  n += (size_t)3 * R * LDS;        // sQ, sGT, sO
  n += (size_t)8 * R * LDS;        // scs
  n += (size_t)R * LDHS;           // shs
  n += (size_t)R * LDS;            // sup
  n += (size_t)3 * H * R * CSL;    // szs
  return n * 2 + (size_t)(3 * R) * 4 + (size_t)NK * 8 + 64;
}

at::Tensor dit_cross_fwd(const at::Tensor& a, const at::Tensor& s, const at::Tensor& z,
                         const at::Tensor& mask, const at::Tensor& lnz,
                         const at::Tensor& W, const at::Tensor& offs,
                         const at::Tensor& kv, const at::Tensor& sync,
                         int64_t NB, int64_t wstride, double eps, double qdiv,
                         double inf, int64_t has_mask, int64_t dbg) {
  at::Tensor out = at::empty({368, 128}, a.options());
  Params p{};
  p.a = (const bf16*)a.data_ptr();
  p.s = (const bf16*)s.data_ptr();
  p.z = (const bf16*)z.data_ptr();
  p.mask = has_mask ? (const bf16*)mask.data_ptr() : nullptr;
  p.lnz = (const bf16*)lnz.data_ptr();
  p.out = (bf16*)out.data_ptr();
  p.W = (const bf16*)W.data_ptr();
  p.wstride = wstride;
  const int64_t* o = offs.data_ptr<int64_t>();
  for (int i = 0; i < W_NSECT; i++) p.off[i] = o[i];
  p.kv = (bf16*)kv.data_ptr();
  p.cnt = (unsigned*)sync.data_ptr();
  p.gen = p.cnt + 1;
  p.NB = (int)NB; p.dbg = (int)dbg;
  p.eps = (float)eps; p.qdiv = (float)qdiv; p.inf = (float)inf;
  p.has_mask = (int)has_mask;
  TORCH_CHECK(NB == 3, "dit_cross: the pair-bias buffer is sized for 3 blocks");

  constexpr int smb = (int)smem_bytes<CFG_X>();
  static int done = 0;
  if (!done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)dit_cross<CFG_X>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smb));
    done = 1;
  }
  dit_cross<CFG_X><<<96, THREADS, smb, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t dit_cross_smem() { return (int64_t)smem_bytes<CFG_X>(); }

// How many clusters the device can hold at once: the device-wide barrier needs
// every one of them resident, and cluster placement is coarser than CTA
// placement, so this is checked once before the fast path is enabled.
int64_t dit_cross_clusters() {
  constexpr int smb = (int)smem_bytes<CFG_X>();
  C10_CUDA_CHECK(cudaFuncSetAttribute((void*)dit_cross<CFG_X>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smb));
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  cfg.gridDim = dim3(96, 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = smb;
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CLS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  cudaOccupancyMaxActiveClusters(&n, (void*)dit_cross<CFG_X>, &cfg);
  return (int64_t)n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dit_cross_fwd", &dit_cross_fwd);
  m.def("dit_cross_smem", &dit_cross_smem);
  m.def("dit_cross_clusters", &dit_cross_clusters);
}
