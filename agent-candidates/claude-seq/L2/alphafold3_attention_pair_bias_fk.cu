// Fused AttentionPairBias / CrossAttentionPairBias (AlphaFold3, Algorithm 24).
//
// At the captured shapes both operators are pure launch overhead.  The whole
// AttentionPairBias forward is ~125 MFLOP over 16 tokens -- about 1.5 us of
// arithmetic on a B200 -- and the reference composition needs ~200 us for it,
// spread over 32 kernels.  CrossAttentionPairBias needs 117 kernels for ~645 us,
// most of them the elementwise steps of ``_get_block_key_indices`` plus the
// blocked gather.  Measured on the bench's own timing loop a launch costs a flat
// ~4.1 us whatever it does, so the figure of merit is the launch count:
//
//   AttentionPairBias       3 launches (2 without AdaLN)
//   CrossAttentionPairBias  2 launches
//
// Everything not separated by a true data dependency shares a launch, including
// work that is cheaper to recompute per block than to synchronize on (the pair
// bias, the attention itself, every LayerNorm statistic).
//
// Two layout choices carry most of the performance:
//
// * Every matrix consumed as a wmma B operand is packed in 16x16 tile-major
//   order (``_tile16`` on the Python side).  Loading a B fragment out of a plain
//   row-major [n][k] matrix touches 16 rows that are ``ldb`` apart, so one
//   fragment costs 16 scattered 32-byte transactions; tile-major makes it one
//   contiguous 512-byte run.  That alone was a ~10x difference on these shapes.
// * The per-head dim is padded to a multiple of 16 (c_hidden is 24 for one
//   captured variant) so no fragment ever straddles a head, and q/k/v are
//   written tile-major too, which serves the score GEMM (q as A, k as B
//   col-major) and the value GEMM (v as B row-major) from the same bytes.
//
// Fragment loads are also prefetched UNROLL-deep: with only a few warps resident
// there is nothing else to hide load latency behind, and one-load-per-mma left
// the first version of these kernels stalled ~50x longer than it computed.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace {

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

constexpr int WARP = 32;
constexpr int TILE = 16;        // wmma M/N/K
constexpr int TSZ = TILE * TILE;  // elements in one packed tile
constexpr int MAXCZ = 32;       // c_z bound for the register-staged bias row
constexpr int UNR = 8;          // fragment prefetch depth          // fragment prefetch depth          // fragment prefetch depth          // fragment prefetch depth

// Packed-buffer sections are padded to a multiple of TSZ so that every tile base
// inside one is 512-byte aligned.
__host__ __device__ __forceinline__ int64_t apad(int64_t n) {
  return (n + TSZ - 1) / TSZ * TSZ;
}

__device__ __forceinline__ float bf(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ float sig(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = WARP / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = WARP / 2; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Bulk copies.  Element-at-a-time copies of these staging buffers were costing
// more than the GEMMs they feed: 12288 two-byte loads with a dynamic trip count
// gets unrolled only ~4 deep, so the copy ran at a few GB/s per SM.  16-byte
// accesses give 8x fewer iterations and enough in flight to saturate.
__device__ __forceinline__ void vcopy(bf16* dst, const bf16* src, size_t n, int tid,
                                      int nthr) {
  for (size_t e = (size_t)tid * 8; e < n; e += (size_t)nthr * 8)
    *(float4*)(dst + e) = *(const float4*)(src + e);
}

__device__ __forceinline__ void vcopyf(float* dst, const float* src, size_t n, int tid,
                                       int nthr) {
  for (size_t e = (size_t)tid * 4; e < n; e += (size_t)nthr * 4)
    *(float4*)(dst + e) = *(const float4*)(src + e);
}

// bf16 dot product with 16-byte loads.  The score and pair-bias reductions run
// over 16-32 elements; at two bytes per load they were issue-bound on LDS.U16.
__device__ __forceinline__ float dot_bf(const bf16* x, const bf16* y, int n) {
  float acc = 0.f;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    float4 xv = *(const float4*)(x + i), yv = *(const float4*)(y + i);
    const bf16* xb = (const bf16*)&xv;
    const bf16* yb = (const bf16*)&yv;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += __bfloat162float(xb[j]) * __bfloat162float(yb[j]);
  }
  for (; i < n; ++i) acc += __bfloat162float(x[i]) * __bfloat162float(y[i]);
  return acc;
}

// LayerNorm statistics of one row, one warp, two passes (mean, then the sum of
// squared deviations) to match torch's Welford reduction rather than the less
// stable E[x^2]-E[x]^2.  A null row means an all-zero (padded) row.
__device__ __forceinline__ void row_stats(const bf16* row, int n, int lane,
                                          float eps, float& mean, float& rstd) {
  if (row == nullptr) {
    mean = 0.f;
    rstd = rsqrtf(eps);
    return;
  }
  float s = 0.f;
  for (int i = lane; i < n; i += WARP) s += __bfloat162float(row[i]);
  mean = warp_sum(s) / n;
  float q = 0.f;
  for (int i = lane; i < n; i += WARP) {
    float d = __bfloat162float(row[i]) - mean;
    q += d * d;
  }
  rstd = rsqrtf(warp_sum(q) / n + eps);
}

// acc += A[16 rows, :] * B[16 cols, :]^T over ktiles 16-wide steps.  Both operands
// are 16x16 tile-major: consecutive k-tiles are TSZ elements apart, and one tile
// holds [row][col] so the same bytes serve A (row-major) and B (col-major).
//
// The k-loop cascades 8 -> 4 -> 2 -> 1 tiles per chunk, issuing every B fragment
// load of a chunk before any mma.  Without that, each mma waits on the load right
// before it: wmma lowers a fragment load to four generic 32-bit loads and only a
// handful stay in flight, so a one-load-per-mma loop runs at one memory latency
// per k-step.  The cascade matters because K is split across warps, which leaves
// chunks that are not multiples of 8.
template <int U, typename FragAcc>
__device__ __forceinline__ void mm_chunk(FragAcc& acc, const bf16* A, const bf16* B) {
  typedef wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, bf16, wmma::row_major> FA;
  typedef wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, bf16, wmma::col_major> FB;
  FB bfr[U];
#pragma unroll
  for (int u = 0; u < U; ++u) wmma::load_matrix_sync(bfr[u], B + (size_t)u * TSZ, TILE);
#pragma unroll
  for (int u = 0; u < U; ++u) {
    FA af;
    wmma::load_matrix_sync(af, A + (size_t)u * TSZ, TILE);
    wmma::mma_sync(acc, af, bfr[u], acc);
  }
}

template <int UMAX, typename FragAcc>
__device__ __forceinline__ void mm_tiled(FragAcc& acc, const bf16* A, const bf16* B,
                                         int ktiles) {
  int t = 0;
  if (UMAX >= 8)
    for (; t + 8 <= ktiles; t += 8)
      mm_chunk<8>(acc, A + (size_t)t * TSZ, B + (size_t)t * TSZ);
  if (t + 4 <= ktiles) {
    mm_chunk<4>(acc, A + (size_t)t * TSZ, B + (size_t)t * TSZ);
    t += 4;
  }
  if (t + 2 <= ktiles) {
    mm_chunk<2>(acc, A + (size_t)t * TSZ, B + (size_t)t * TSZ);
    t += 2;
  }
  if (t < ktiles) mm_chunk<1>(acc, A + (size_t)t * TSZ, B + (size_t)t * TSZ);
}

// One output tile per block: every warp takes a slice of K and the partials are
// summed through shared memory.  With a single warp per tile the k-loop is one
// long chain of dependent fragment loads -- wmma compiles a fragment load to
// four generic 32-bit loads and only ~11 stay in flight -- and that chain, not
// bandwidth, set the runtime of the first version of these kernels.  Splitting K
// across 8-16 warps shortens each chain proportionally.
__device__ __forceinline__ void ks_gemm(float* sPart, const bf16* A, const bf16* B,
                                        int ktiles, int warps, int warp) {
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> f;
  wmma::fill_fragment(f, 0.f);
  const int per = (ktiles + warps - 1) / warps;
  const int k0 = warp * per, k1 = min(ktiles, k0 + per);
  if (k0 < k1) mm_tiled<UNR>(f, A + (size_t)k0 * TSZ, B + (size_t)k0 * TSZ, k1 - k0);
  wmma::store_matrix_sync(sPart + (size_t)warp * TSZ, f, TILE, wmma::mem_row_major);
}

// Sum of one tile's per-warp partials for element e (call after __syncthreads()).
__device__ __forceinline__ float ks_sum(const float* sPart, int warps, int e) {
  float v = 0.f;
  for (int w = 0; w < warps; ++w) v += sPart[(size_t)w * TSZ + e];
  return v;
}

// Tile-major element address helpers.  ``TILED(base, mt, kt, nk)`` is the tile at
// (m-tile, k-tile) of a matrix with nk k-tiles; +(i*TILE+j) selects the element.
#define TILED(base, mt, kt, nk) ((base) + ((size_t)(mt) * (nk) + (kt)) * TSZ)

// ---------------------------------------------------------------------------
// Flags shared with the Python packer.
// ---------------------------------------------------------------------------
constexpr int F_ADA = 1 << 0;
constexpr int F_GATING = 1 << 1;
constexpr int F_LNZ_B = 1 << 2;   // layer_norm_z has an offset
constexpr int F_LNS_B = 1 << 3;   // AdaLN.layer_norm_s has an offset
constexpr int F_LNA_W = 1 << 4;   // non-ada layer_norm_a has a scale
constexpr int F_LNA_B = 1 << 5;   // non-ada layer_norm_a has an offset
constexpr int F_LNSK_B = 1 << 6;  // CrossAttention: k-side layer_norm_s offset
constexpr int F_LNK_W = 1 << 7;   // CrossAttention: non-ada k-side scale
constexpr int F_LNK_B = 1 << 8;   // CrossAttention: non-ada k-side offset

struct Cfg {
  int N;     // tokens (AttentionPairBias) / padded atom count (CrossAttention)
  int Nt;    // N rounded up to a multiple of TILE
  int C, Cs, Cz, H, D, Dp, HDp;
  int nC, nCs, nHD;  // tile counts: C/16, Cs/16, HDp/16
  float eps, inf, qscale;
  int flags;
  int nAtom, Q, K, NB, QS;  // CrossAttentionPairBias only
};

struct APBW {
  const bf16 *wqkvg, *bq, *wo, *wz, *lnzw, *lnzb;
  const bf16 *lnsw, *lnsb, *adaw, *adagb, *adaow, *adaob;
  const bf16 *lnaw, *lnab;
};

struct XW {
  const bf16 *wqg, *bq, *wkv, *woT, *wz;
  const bf16 *lnsqw, *lnsqb, *adaqw, *adaqgb;
  const bf16 *lnskw, *lnskb, *adakw, *adakgb;
  const bf16 *adaow, *adaob;
  const bf16 *lnqw, *lnqb, *lnkw, *lnkb;
};

// ###########################################################################
// AttentionPairBias
// ###########################################################################

// Stage 1 (AdaLN only): x = sigmoid(Wg s_norm + bg) * (LN(a) + Ws s_norm).
// Each warp owns a 16-column tile of x and recomputes s_norm and a's LayerNorm
// statistics itself; both are a few KB, so that is cheaper than a launch.
__global__ void apb_prep(Cfg cfg, APBW w, const bf16* __restrict__ a,
                         const bf16* __restrict__ s, bf16* __restrict__ x) {
  extern __shared__ char smem[];
  const int warps = blockDim.x / WARP;
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;

  bf16* sS = (bf16*)smem;                       // s_norm, tile-major [Nt/16][nCs]
  float* sMean = (float*)(sS + (size_t)cfg.Nt * cfg.Cs);
  float* sRstd = sMean + cfg.Nt;
  float* sPg = sRstd + cfg.Nt;                  // [warps][256]
  float* sPs = sPg + (size_t)warps * TSZ;

  for (int r = warp; r < cfg.Nt; r += warps) {
    const bf16* src = (r < cfg.N) ? s + (size_t)r * cfg.Cs : nullptr;
    float mean, rstd;
    row_stats(src, cfg.Cs, lane, cfg.eps, mean, rstd);
    for (int i = lane; i < cfg.Cs; i += WARP) {
      float v = src ? __bfloat162float(src[i]) : 0.f;
      float y = (v - mean) * rstd * __bfloat162float(w.lnsw[i]);
      if (w.lnsb) y += __bfloat162float(w.lnsb[i]);
      TILED(sS, r / TILE, i / TILE, cfg.nCs)[(r % TILE) * TILE + i % TILE] =
          __float2bfloat16(y);
    }
    const bf16* ar = (r < cfg.N) ? a + (size_t)r * cfg.C : nullptr;
    row_stats(ar, cfg.C, lane, cfg.eps, mean, rstd);
    if (lane == 0) { sMean[r] = mean; sRstd[r] = rstd; }
  }
  __syncthreads();

  const int tile = blockIdx.x;
  const int c0 = tile * TILE;
  for (int mt = 0; mt < cfg.Nt / TILE; ++mt) {
    const bf16* A = TILED(sS, mt, 0, cfg.nCs);
    ks_gemm(sPg, A, TILED(w.adaw, tile, 0, cfg.nCs), cfg.nCs, warps, warp);
    ks_gemm(sPs, A, TILED(w.adaw, cfg.nC + tile, 0, cfg.nCs), cfg.nCs, warps, warp);
    __syncthreads();
    for (int e = threadIdx.x; e < TSZ; e += blockDim.x) {
      int r = mt * TILE + e / TILE, c = c0 + e % TILE;
      if (r >= cfg.N) continue;
      float g = sig(ks_sum(sPg, warps, e) + __bfloat162float(w.adagb[c]));
      float an = (__bfloat162float(a[(size_t)r * cfg.C + c]) - sMean[r]) * sRstd[r];
      TILED(x, mt, tile, cfg.nC)[e] = __float2bfloat16(g * (an + ks_sum(sPs, warps, e)));
    }
    __syncthreads();
  }
}

// Stage 2: q/k/v/gate projections plus, on the trailing blocks, the pair bias
// zb[h][q][k] = <LN(z[q][k]), wz[h]>.  The two are independent, so one launch.
__global__ void apb_qkvg(Cfg cfg, APBW w, const bf16* __restrict__ a,
                         const bf16* __restrict__ xin, const bf16* __restrict__ z,
                         bf16* __restrict__ qkvg, float* __restrict__ zb,
                         int mainBlocks, int pairsPerWarp) {
  extern __shared__ char smem[];
  const int warps = blockDim.x / WARP;
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  const bool ada = cfg.flags & F_ADA;

  if (blockIdx.x >= mainBlocks) {  // ---- pair-bias task ----
    // linear_z is [H][Cz] and every lane owns one head, so reading it in the
    // reduction loop had all 32 lanes hitting addresses Cz apart -- one
    // transaction per lane per element.  Staged transposed, lanes read
    // consecutive halves instead, which is both coalesced and bank-conflict free.
    bf16* sWzT = (bf16*)smem;                      // [Cz][H]
    float* sZn = (float*)(sWzT + (size_t)cfg.H * cfg.Cz) + (size_t)warp * cfg.Cz;
    for (int e = threadIdx.x; e < cfg.H * cfg.Cz; e += blockDim.x)
      sWzT[(e % cfg.Cz) * cfg.H + e / cfg.Cz] = w.wz[e];
    __syncthreads();
    const int pairsPerBlock = warps * pairsPerWarp;
    int p0 = (blockIdx.x - mainBlocks) * pairsPerBlock + warp * pairsPerWarp;
    for (int p = p0; p < p0 + pairsPerWarp; ++p) {
      if (p >= cfg.N * cfg.N) return;
      const bf16* zr = z + (size_t)p * cfg.Cz;
      float mean, rstd;
      row_stats(zr, cfg.Cz, lane, cfg.eps, mean, rstd);
      for (int i = lane; i < cfg.Cz; i += WARP) {
        float y = (__bfloat162float(zr[i]) - mean) * rstd * __bfloat162float(w.lnzw[i]);
        if (w.lnzb) y += __bfloat162float(w.lnzb[i]);
        sZn[i] = bf(y);
      }
      __syncwarp();
      int q = p / cfg.N, k = p % cfg.N;
      if (lane < cfg.H) {
        float acc = 0.f;
        for (int i = 0; i < cfg.Cz; ++i)
          acc += sZn[i] * __bfloat162float(sWzT[(size_t)i * cfg.H + lane]);
        zb[((size_t)lane * cfg.Nt + q) * cfg.Nt + k] = acc;
      }
      __syncwarp();
    }
    return;
  }

  // ---- projection task ----
  bf16* sX = (bf16*)smem;  // tile-major [Nt/16][nC]
  float* sAcc = (float*)(sX + (size_t)cfg.Nt * cfg.C);
  if (!ada) {
    for (int r = warp; r < cfg.Nt; r += warps) {
      const bf16* ar = (r < cfg.N) ? a + (size_t)r * cfg.C : nullptr;
      float mean, rstd;
      row_stats(ar, cfg.C, lane, cfg.eps, mean, rstd);
      for (int i = lane; i < cfg.C; i += WARP) {
        float v = ar ? __bfloat162float(ar[i]) : 0.f;
        float y = (v - mean) * rstd;
        if (w.lnaw) y *= __bfloat162float(w.lnaw[i]);
        if (w.lnab) y += __bfloat162float(w.lnab[i]);
        TILED(sX, r / TILE, i / TILE, cfg.nC)[(r % TILE) * TILE + i % TILE] =
            __float2bfloat16(y);
      }
    }
    __syncthreads();
  }

  const int tile = blockIdx.x;
  const bf16* Amat = ada ? xin : sX;   // tile-major either way
  const int nplane = (cfg.flags & F_GATING) ? 4 : 3;
  const int plane = tile / cfg.nHD;
  const int ht = tile - plane * cfg.nHD;   // tile index within [HDp]
  const int col0 = ht * TILE;
  float* sPart = (float*)(sX + (size_t)cfg.Nt * cfg.C);
  // q/k/v/gate are written tile-major as [plane][h][mtile][dtile][16][16]; the
  // score and value GEMMs read those tiles directly.
  const int h = col0 / cfg.Dp, dt = (col0 - h * cfg.Dp) / TILE;
  const int ndt = cfg.Dp / TILE, nmt = cfg.Nt / TILE;

  for (int mt = 0; mt < nmt; ++mt) {
    ks_gemm(sPart, TILED(Amat, mt, 0, cfg.nC), TILED(w.wqkvg, tile, 0, cfg.nC), cfg.nC,
            warps, warp);
    __syncthreads();
    bf16* dst = qkvg + ((((size_t)plane * cfg.H + h) * nmt + mt) * ndt + dt) * TSZ;
    for (int e = threadIdx.x; e < TSZ; e += blockDim.x) {
      float v = ks_sum(sPart, warps, e);
      if (plane == 0) v = bf(v + __bfloat162float(w.bq[col0 + e % TILE])) * cfg.qscale;
      else if (plane == 3) v = sig(v);
      dst[e] = __float2bfloat16(v);
    }
    __syncthreads();
  }
}

// Stage 3: attention for every head (recomputed per block -- ~100 wmma ops,
// against 4.1 us for a launch), gating, the output projection and the AdaLN-Zero
// output gate.  Each warp owns a 16-column tile of the output.
__global__ void apb_attn_out(Cfg cfg, APBW w, const bf16* __restrict__ s,
                             const bf16* __restrict__ mask,
                             const bf16* __restrict__ qkvg,
                             const float* __restrict__ zb, bf16* __restrict__ out) {
  extern __shared__ char smem[];
  const int warps = blockDim.x / WARP;
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  const bool ada = cfg.flags & F_ADA;
  const bool gating = cfg.flags & F_GATING;
  const int nmt = cfg.Nt / TILE, ndt = cfg.Dp / TILE, nkt = cfg.Nt / TILE;

  const int scLd = cfg.Nt + 1;  // pad the score row stride: with one lane per
                                // query row, stride Nt would put every lane in
                                // the same pair of shared-memory banks.
  const size_t plane = (size_t)cfg.H * nmt * ndt * TSZ;

  bf16* sO = (bf16*)smem;                                   // tile-major [nmt][nHD]
  bf16* sS = sO + (size_t)cfg.Nt * cfg.HDp;                 // tile-major [nmt][nCs]
  bf16* sP = sS + (size_t)(ada ? cfg.Nt * cfg.Cs : 0);      // [warps][nkt][256]
  float* sSc = (float*)(sP + (size_t)warps * TILE * cfg.Nt);  // [warps][16][scLd]
  float* sPo = sSc + (size_t)warps * TILE * scLd;           // [warps][256]
  float* sPg = sPo + (size_t)warps * TSZ;                   // [warps][256]
  bf16* pp = sP + (size_t)warp * TILE * cfg.Nt;
  float* sc = sSc + (size_t)warp * TILE * scLd;
  float* acc = sPo + (size_t)warp * TSZ;   // reused as wmma staging pre-reduction

  const bf16* qh0 = qkvg;
  const bf16* kh0 = qkvg + plane;
  const bf16* vh0 = qkvg + 2 * plane;
  const bf16* gb = qkvg + 3 * plane;
  if (ada)
    for (size_t e = (size_t)threadIdx.x * 8; e < (size_t)cfg.Nt * cfg.Cs;
         e += (size_t)blockDim.x * 8) {
      int r = e / cfg.Cs, i = e % cfg.Cs;
      bf16* d = TILED(sS, r / TILE, i / TILE, cfg.nCs) + (r % TILE) * TILE + i % TILE;
      if (r < cfg.N) *(float4*)d = *(const float4*)(s + e);
      else *(float4*)d = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  __syncthreads();

  // ---- attention, all heads ----
  for (int h = warp; h < cfg.H; h += warps) {
    const size_t hoff = (size_t)h * nmt * ndt * TSZ;
    const bf16* qh = qh0 + hoff;
    const bf16* kh = kh0 + hoff;
    const bf16* vh = vh0 + hoff;
    const bf16* gh = gb + hoff;
    for (int mt = 0; mt < nmt; ++mt) {
      for (int kt = 0; kt < nkt; ++kt) {
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> f;
        wmma::fill_fragment(f, 0.f);
        mm_tiled<UNR>(f, TILED(qh, mt, 0, ndt), TILED(kh, kt, 0, ndt), ndt);
        wmma::store_matrix_sync(acc, f, TILE, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < TSZ; e += WARP)
          sc[(e / TILE) * scLd + kt * TILE + e % TILE] = acc[e];
        __syncwarp();
      }
      // Softmax over keys, one lane per query row -- no cross-lane reductions.
      // The mask bias is folded in at bf16 precision, so an all-masked row
      // collapses to uniform exactly as the baseline's bf16 (score + -1e9) does.
      if (lane < TILE) {
        const int qrow = mt * TILE + lane;
        float* row = sc + (size_t)lane * scLd;
        bf16* prow = pp + (size_t)0;
        if (qrow >= cfg.N) {
          for (int k = 0; k < cfg.Nt; ++k)
            TILED(prow, 0, k / TILE, nkt)[lane * TILE + k % TILE] = __float2bfloat16(0.f);
        } else {
          const float* zbr = zb + ((size_t)h * cfg.Nt + qrow) * cfg.Nt;
          float mx = -INFINITY;
          for (int k = 0; k < cfg.Nt; ++k) {
            float v = -INFINITY;
            if (k < cfg.N) {
              float mk = mask ? __bfloat162float(mask[k]) : 1.f;
              v = bf(bf(row[k]) + bf(cfg.inf * (mk - 1.f)));
              v = bf(v + zbr[k]);
            }
            row[k] = v;
            mx = fmaxf(mx, v);
          }
          float sum = 0.f;
          for (int k = 0; k < cfg.Nt; ++k) {
            float e = (row[k] == -INFINITY) ? 0.f : __expf(row[k] - mx);
            row[k] = e;
            sum += e;
          }
          float inv = 1.f / sum;
          for (int k = 0; k < cfg.Nt; ++k)
            TILED(prow, 0, k / TILE, nkt)[lane * TILE + k % TILE] =
                __float2bfloat16(row[k] * inv);
        }
      }
      __syncwarp();
      for (int dt = 0; dt < ndt; ++dt) {
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> f;
        wmma::fill_fragment(f, 0.f);
        for (int kt = 0; kt < nkt; ++kt) {
          wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, bf16, wmma::row_major> af;
          wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, bf16, wmma::row_major> bfr;
          wmma::load_matrix_sync(af, TILED(pp, 0, kt, nkt), TILE);
          wmma::load_matrix_sync(bfr, TILED(vh, kt, dt, ndt), TILE);
          wmma::mma_sync(f, af, bfr, f);
        }
        wmma::store_matrix_sync(acc, f, TILE, wmma::mem_row_major);
        __syncwarp();
        const bf16* gt = TILED(gh, mt, dt, ndt);
        bf16* ot = TILED(sO, mt, h * ndt + dt, cfg.nHD);
        for (int e = lane; e < TSZ; e += WARP) {
          float v = bf(acc[e]);
          if (gating) v *= __bfloat162float(gt[e]);
          ot[e] = __float2bfloat16(v);
        }
        __syncwarp();
      }
    }
  }
  __syncthreads();

  // ---- output projection, gated by sigmoid(linear_ada_out(s)) ----
  const int tile = blockIdx.x;
  const int c0 = tile * TILE;
  for (int mt = 0; mt < nmt; ++mt) {
    ks_gemm(sPo, TILED(sO, mt, 0, cfg.nHD), TILED(w.wo, tile, 0, cfg.nHD), cfg.nHD,
            warps, warp);
    if (ada)
      ks_gemm(sPg, TILED(sS, mt, 0, cfg.nCs), TILED(w.adaow, tile, 0, cfg.nCs), cfg.nCs,
              warps, warp);
    __syncthreads();
    for (int e = threadIdx.x; e < TSZ; e += blockDim.x) {
      int r = mt * TILE + e / TILE, c = c0 + e % TILE;
      if (r >= cfg.N) continue;
      float v = ks_sum(sPo, warps, e);
      if (ada) v *= sig(ks_sum(sPg, warps, e) + __bfloat162float(w.adaob[c]));
      out[(size_t)r * cfg.C + c] = __float2bfloat16(v);
    }
    __syncthreads();
  }
}


// ###########################################################################
// CrossAttentionPairBias (sequence-local atom attention)
// ###########################################################################
//
// The baseline materializes the blocked key view with a gather
// (``_convert_single_rep_to_blocks``), which duplicates every atom into
// ~n_key/n_query blocks -- 1536 key rows for 368 atoms -- and then LayerNorms and
// projects all of them.  A blocked key row is a pure function of the atom it
// gathers from, so here the projections run once per atom (384 rows) and the
// block view is applied later, where the attention reads k/v.
//
// The gather indices are reproduced exactly, including the baseline's bf16
// arithmetic: ``final = initial.to(bf16) + shift`` rounds indices >= 257 to even
// values and pushes the last key of the trailing blocks up to n_real, where the
// baseline then marks it invalid.  Recomputing that in fp32 would gather
// different atoms, so bf() is applied at the same points.

struct KeyGeom {
  float nReal;  // bf16(sum(mask)), as the baseline's mask.sum(-1) produces
  float nRm1;   // bf16(nReal - 1), the clamp bound
  float first;  // first unshifted key index of this block
  float shift;
};

__device__ __forceinline__ KeyGeom key_geom(const Cfg& cfg, const bf16* mask, int b,
                                            float* wred) {
  const int nw = blockDim.x / WARP;
  float sum = 0.f;
  if (mask)
    for (int i = threadIdx.x; i < cfg.nAtom; i += blockDim.x)
      sum += __bfloat162float(mask[i]);
  else if (threadIdx.x == 0)
    sum = (float)cfg.nAtom;
  sum = warp_sum(sum);
  if (threadIdx.x % WARP == 0) wred[threadIdx.x / WARP] = sum;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < nw; ++i) tot += wred[i];
  KeyGeom g;
  g.nReal = bf(tot);
  g.nRm1 = bf(g.nReal - 1.f);
  int center = cfg.Q / 2 + b * cfg.Q;
  g.first = (float)(center - cfg.K / 2);
  int last = center + cfg.K / 2 - 1;
  float under = fmaxf(0.f, -g.first);
  float over = fmaxf(0.f, bf(bf((float)last) - g.nRm1));
  g.shift = (under > 0.f) ? under : -over;
  return g;
}

// ``invalid`` means the shifted index fell outside the real atoms, which the
// baseline handles by zeroing the gathered row (``masked_fill_`` in
// ``_convert_single_rep_to_blocks``) as well as masking the score.  The zeroing is
// only observable for a query row whose every key is masked -- there the softmax
// is uniform rather than zero, so those rows average the *zeroed* values -- but
// that is exactly what happens for padded atoms under a partial mask.
__device__ __forceinline__ int key_index(const KeyGeom& g, int j, float& valid,
                                         bool& invalid, const bf16* mask, int nAtom) {
  float fin = bf(bf(g.first + (float)j) + g.shift);
  invalid = (fin < 0.f) || (fin >= g.nReal);
  int idx = (int)fminf(fmaxf(fin, 0.f), g.nRm1);
  float mk = (mask == nullptr) ? 1.f : (idx < nAtom ? __bfloat162float(mask[idx]) : 0.f);
  valid = invalid ? 0.f : mk;
  return idx;
}

// Stage 1: per-atom AdaLN (query and key variants) and then q / gate / k / v plus
// the AdaLN-Zero output gate.  One block owns 16 atom rows; the AdaLN stage and
// the projections that consume it are separated by __syncthreads, not a launch.
__global__ void xapb_prep(Cfg cfg, XW w, const bf16* __restrict__ a,
                          const bf16* __restrict__ s, bf16* __restrict__ qgkv,
                          bf16* __restrict__ osc) {
  extern __shared__ char smem[];
  const int warps = blockDim.x / WARP;
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  const bool ada = cfg.flags & F_ADA;
  const bool gating = cfg.flags & F_GATING;
  const int r0 = blockIdx.x * TILE;
  const int nplane = gating ? 4 : 3;

  bf16* sAn = (bf16*)smem;                              // [16][C] row-major, LN(a)
  bf16* sSraw = sAn + (size_t)TILE * cfg.C;             // tile-major, raw s
  bf16* sNq = sSraw + (size_t)TILE * cfg.Cs;            // tile-major, layer_norm_s (q)
  bf16* sNk = sNq + (size_t)TILE * cfg.Cs;              // tile-major, layer_norm_s (k)
  bf16* sXq = sNk + (size_t)TILE * cfg.Cs;              // tile-major
  bf16* sXk = sXq + (size_t)TILE * cfg.C;
  float* sA = (float*)(sXk + (size_t)TILE * cfg.C);     // [warps][256] accumulators

  // ---- LayerNorms.  One warp per atom row; the AdaLN s-statistics are shared
  // between the query and key sides (only the scale/offset differ), so the row is
  // reduced once rather than twice.
  for (int r = warp; r < TILE; r += warps) {
    int t = r0 + r;
    const bf16* ar = (t < cfg.nAtom) ? a + (size_t)t * cfg.C : nullptr;
    float mean, rstd;
    row_stats(ar, cfg.C, lane, cfg.eps, mean, rstd);
    for (int i = lane; i < cfg.C; i += WARP) {
      float v = ar ? __bfloat162float(ar[i]) : 0.f;
      sAn[(size_t)r * cfg.C + i] = __float2bfloat16((v - mean) * rstd);
    }
    const bf16* sr = (t < cfg.nAtom && s) ? s + (size_t)t * cfg.Cs : nullptr;
    float sm, sr2;
    if (ada) row_stats(sr, cfg.Cs, lane, cfg.eps, sm, sr2);
    for (int i = lane; i < cfg.Cs; i += WARP) {
      float v = sr ? __bfloat162float(sr[i]) : 0.f;
      int off = (r % TILE) * TILE + i % TILE;
      TILED(sSraw, 0, i / TILE, cfg.nCs)[off] = __float2bfloat16(v);
      if (ada) {
        float y = (v - sm) * sr2;
        float yq = y * __bfloat162float(w.lnsqw[i]);
        float yk = y * __bfloat162float(w.lnskw[i]);
        if (w.lnsqb) yq += __bfloat162float(w.lnsqb[i]);
        if (w.lnskb) yk += __bfloat162float(w.lnskb[i]);
        TILED(sNq, 0, i / TILE, cfg.nCs)[off] = __float2bfloat16(yq);
        TILED(sNk, 0, i / TILE, cfg.nCs)[off] = __float2bfloat16(yk);
      }
    }
  }
  __syncthreads();

  // ---- AdaLN: x_q and x_k.  One warp per (column tile, side, gate/shift) so all
  // warps work; the four products are combined after a single barrier.
  if (ada) {
    const int ntask = cfg.nC * 4;
    for (int t = warp; t < ntask; t += warps) {
      const int tile = t >> 2, side = (t >> 1) & 1, which = t & 1;
      const bf16* A = side ? sNk : sNq;
      const bf16* WA = side ? w.adakw : w.adaqw;
      const int row = which ? cfg.nC + tile : tile;  // linear_g rows, then linear_s
      wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> f;
      wmma::fill_fragment(f, 0.f);
      mm_tiled<UNR>(f, A, TILED(WA, row, 0, cfg.nCs), cfg.nCs);
      wmma::store_matrix_sync(sA + (size_t)t * TSZ, f, TILE, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 2 * cfg.nC * TSZ; e += blockDim.x) {
      const int side = e / (cfg.nC * TSZ), rest = e - side * cfg.nC * TSZ;
      const int tile = rest / TSZ, el = rest % TSZ;
      const int r = el / TILE, c = tile * TILE + el % TILE;
      const float* ag = sA + ((size_t)tile * 4 + side * 2) * TSZ + el;
      const float* as = ag + TSZ;
      const bf16* GB = side ? w.adakgb : w.adaqgb;
      float g = sig(*ag + __bfloat162float(GB[c]));
      float an = __bfloat162float(sAn[(size_t)r * cfg.C + c]);
      TILED(side ? sXk : sXq, 0, tile, cfg.nC)[el] = __float2bfloat16(g * (an + *as));
    }
  } else {
    for (int e = threadIdx.x; e < TILE * cfg.C; e += blockDim.x) {
      const int tile = (e % cfg.C) / TILE;
      const int r = e / cfg.C, c = e % cfg.C;
      const int el = (r % TILE) * TILE + c % TILE;
      float an = __bfloat162float(sAn[(size_t)r * cfg.C + c]);
      float vq = an, vk = an;
      if (w.lnqw) vq *= __bfloat162float(w.lnqw[c]);
      if (w.lnqb) vq += __bfloat162float(w.lnqb[c]);
      if (w.lnkw) vk *= __bfloat162float(w.lnkw[c]);
      if (w.lnkb) vk += __bfloat162float(w.lnkb[c]);
      TILED(sXq, 0, tile, cfg.nC)[el] = __float2bfloat16(vq);
      TILED(sXk, 0, tile, cfg.nC)[el] = __float2bfloat16(vk);
    }
  }
  __syncthreads();

  // ---- q / k / v / gate, and the AdaLN-Zero output gate (independent of x).
  const int nqkvg = nplane * cfg.nHD;
  const int ntask = nqkvg + (ada ? cfg.nC : 0);
  for (int t = warp; t < ntask; t += warps) {
    float* acc = sA + (size_t)warp * TSZ;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> f;
    wmma::fill_fragment(f, 0.f);
    if (t < nqkvg) {
      const int plane = t / cfg.nHD, ht = t - plane * cfg.nHD, col0 = ht * TILE;
      const bf16* A;
      const bf16* B;
      switch (plane) {
        case 0: A = sXq; B = TILED(w.wqg, ht, 0, cfg.nC); break;
        case 1: A = sXk; B = TILED(w.wkv, ht, 0, cfg.nC); break;
        case 2: A = sXk; B = TILED(w.wkv, cfg.nHD + ht, 0, cfg.nC); break;
        default: A = sXq; B = TILED(w.wqg, cfg.nHD + ht, 0, cfg.nC); break;
      }
      mm_tiled<UNR>(f, A, B, cfg.nC);
      wmma::store_matrix_sync(acc, f, TILE, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < TSZ; e += WARP) {
        int r = r0 + e / TILE, c = col0 + e % TILE;
        float v = acc[e];
        if (plane == 0) v = bf(v + __bfloat162float(w.bq[c])) * cfg.qscale;
        else if (plane == 3) v = sig(v);
        qgkv[((size_t)plane * cfg.N + r) * cfg.HDp + c] = __float2bfloat16(v);
      }
    } else {
      const int tile = t - nqkvg, c0 = tile * TILE;
      mm_tiled<UNR>(f, sSraw, TILED(w.adaow, tile, 0, cfg.nCs), cfg.nCs);
      wmma::store_matrix_sync(acc, f, TILE, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < TSZ; e += WARP) {
        int r = r0 + e / TILE, c = c0 + e % TILE;
        if (r >= cfg.nAtom) continue;
        osc[(size_t)r * cfg.C + c] =
            __float2bfloat16(sig(acc[e] + __bfloat162float(w.adaob[c])));
      }
    }
    __syncwarp();
  }
}

// Stage 2: pair bias, sequence-local attention, gating, output projection and the
// output gate -- one block per (key block, query sub-tile).  z is the only large
// tensor (1.5 MB at the captured shape) and is read exactly once; the query
// sub-tiling exists to keep enough blocks resident to stream it at full
// bandwidth.  k/v are gathered into shared memory once per block.
__global__ void xapb_attn_out(Cfg cfg, XW w, const bf16* __restrict__ z,
                              const bf16* __restrict__ mask,
                              const bf16* __restrict__ qgkv,
                              const bf16* __restrict__ osc, bf16* __restrict__ out) {
  extern __shared__ char smem[];
  const int nsub = cfg.Q / cfg.QS;
  const int b = blockIdx.x / nsub, sub = blockIdx.x % nsub;
  const int q0 = sub * cfg.QS, t0 = b * cfg.Q + q0;
  const bool gating = cfg.flags & F_GATING;
  const bool ada = cfg.flags & F_ADA;
  const int nw = blockDim.x / WARP;
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;

  bf16* sK = (bf16*)smem;                                   // [K][HDp]
  bf16* sV = sK + (size_t)cfg.K * cfg.HDp;
  bf16* sQ = sV + (size_t)cfg.K * cfg.HDp;                  // [QS][HDp]
  bf16* sG = sQ + (size_t)cfg.QS * cfg.HDp;
  bf16* sOf = sG + (size_t)cfg.QS * cfg.HDp;
  bf16* sZ = sOf + (size_t)cfg.QS * cfg.HDp;                // [QS][K][Cz]
  bf16* sWo = sZ + (size_t)cfg.QS * cfg.K * cfg.Cz;         // [HDp][C] linear_o
  float* sSc = (float*)(sWo + (size_t)cfg.C * cfg.HDp);     // [H][QS][K]
  float* sKv = sSc + (size_t)cfg.H * cfg.QS * cfg.K;        // [K]
  float* sMq = sKv + cfg.K;                                 // [QS]
  float* sRed = sMq + cfg.QS;                               // [WARP]

  const bf16* qb = qgkv;
  const bf16* kb = qgkv + (size_t)cfg.N * cfg.HDp;
  const bf16* vb = qgkv + (size_t)2 * cfg.N * cfg.HDp;
  const bf16* gbuf = qgkv + (size_t)3 * cfg.N * cfg.HDp;

  KeyGeom g = key_geom(cfg, mask, b, sRed);

  // Gather this block's keys.  Vector copies: HDp is a multiple of 8 so one warp
  // moves a whole row per instruction.
  const int vpr = cfg.HDp / 8;
  const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int e = threadIdx.x; e < cfg.K * vpr; e += blockDim.x) {
    int j = e / vpr, u = (e % vpr) * 8;
    float valid;
    bool invalid;
    int idx = key_index(g, j, valid, invalid, mask, cfg.nAtom);
    if (u == 0) sKv[j] = valid;
    *(float4*)(sK + (size_t)j * cfg.HDp + u) =
        invalid ? zero4 : *(const float4*)(kb + (size_t)idx * cfg.HDp + u);
    *(float4*)(sV + (size_t)j * cfg.HDp + u) =
        invalid ? zero4 : *(const float4*)(vb + (size_t)idx * cfg.HDp + u);
  }
  for (int e = threadIdx.x; e < cfg.QS * vpr; e += blockDim.x) {
    int i = e / vpr, u = (e % vpr) * 8;
    *(float4*)(sQ + (size_t)i * cfg.HDp + u) =
        *(const float4*)(qb + (size_t)(t0 + i) * cfg.HDp + u);
    if (gating)
      *(float4*)(sG + (size_t)i * cfg.HDp + u) =
          *(const float4*)(gbuf + (size_t)(t0 + i) * cfg.HDp + u);
  }
  // z[b][q0..q0+QS][all keys][all channels] is one contiguous run.
  const size_t zn = (size_t)cfg.QS * cfg.K * cfg.Cz;
  const bf16* zsrc = z + (size_t)t0 * cfg.K * cfg.Cz;
  for (size_t e = threadIdx.x * 8; e < zn; e += (size_t)blockDim.x * 8)
    *(float4*)(sZ + e) = *(const float4*)(zsrc + e);
  vcopy(sWo, w.woT, (size_t)cfg.C * cfg.HDp, threadIdx.x, blockDim.x);
  for (int i = threadIdx.x; i < cfg.QS; i += blockDim.x) {
    int t = t0 + i;
    sMq[i] = (t >= cfg.nAtom) ? 0.f : (mask ? __bfloat162float(mask[t]) : 1.f);
  }
  __syncthreads();

  // Scores: q.k, then the mask bias, then the pair bias -- the baseline's order,
  // each step rounded to bf16 so a fully masked row collapses the same way.
  const int nsc = cfg.H * cfg.QS * cfg.K;
  for (int e = threadIdx.x; e < nsc; e += blockDim.x) {
    int j = e % cfg.K, i = (e / cfg.K) % cfg.QS, h = e / (cfg.K * cfg.QS);
    float dot = dot_bf(sQ + (size_t)i * cfg.HDp + h * cfg.Dp,
                       sK + (size_t)j * cfg.HDp + h * cfg.Dp, cfg.D);
    float zbv = dot_bf(sZ + ((size_t)i * cfg.K + j) * cfg.Cz,
                       w.wz + (size_t)h * cfg.Cz, cfg.Cz);
    float bm = bf(sMq[i] * sKv[j]);
    sSc[e] = bf(bf(bf(dot) + bf(cfg.inf * (bm - 1.f))) + bf(zbv));
  }
  __syncthreads();

  for (int r = warp; r < cfg.H * cfg.QS; r += nw) {
    float* row = sSc + (size_t)r * cfg.K;
    float mx = -INFINITY;
    for (int j = lane; j < cfg.K; j += WARP) mx = fmaxf(mx, row[j]);
    mx = warp_max(mx);
    float sum = 0.f;
    for (int j = lane; j < cfg.K; j += WARP) {
      float e = __expf(row[j] - mx);
      row[j] = e;
      sum += e;
    }
    float inv = 1.f / warp_sum(sum);
    for (int j = lane; j < cfg.K; j += WARP) row[j] = bf(row[j] * inv);
  }
  __syncthreads();

  for (int e = threadIdx.x; e < cfg.QS * cfg.HDp; e += blockDim.x) {
    int i = e / cfg.HDp, c = e % cfg.HDp, h = c / cfg.Dp;
    const float* row = sSc + (size_t)(h * cfg.QS + i) * cfg.K;
    float av = 0.f;
    for (int j = 0; j < cfg.K; ++j)
      av += row[j] * __bfloat162float(sV[(size_t)j * cfg.HDp + c]);
    av = bf(av);
    if (gating) av *= __bfloat162float(sG[e]);
    sOf[e] = __float2bfloat16(av);
  }
  __syncthreads();

  // Output projection.  linear_o is packed transposed ([HDp][C]) so that lanes
  // walking c read contiguous weights.
  for (int e = threadIdx.x; e < cfg.QS * cfg.C; e += blockDim.x) {
    int i = e / cfg.C, c = e % cfg.C, t = t0 + i;
    if (t >= cfg.nAtom) continue;
    float av = 0.f;
    for (int j = 0; j < cfg.HDp; ++j)
      av += __bfloat162float(sOf[(size_t)i * cfg.HDp + j]) *
            __bfloat162float(sWo[(size_t)j * cfg.C + c]);
    if (ada) av *= __bfloat162float(osc[(size_t)t * cfg.C + c]);
    out[(size_t)t * cfg.C + c] = __float2bfloat16(av);
  }
}

// ###########################################################################
// Host side
// ###########################################################################

// Walks the packed buffer in the order ``_build_pack`` writes it.
struct Cursor {
  const bf16* p;
  int64_t left;
  const bf16* take(int64_t n) {
    if (n == 0) return nullptr;
    TORCH_CHECK(left >= n, "packed weight buffer too small");
    const bf16* r = p;
    int64_t step = apad(n);
    p += step;
    left -= step;
    return r;
  }
};

APBW apb_weights(const at::Tensor& pack, const Cfg& c) {
  Cursor cur{(const bf16*)pack.data_ptr(), pack.numel()};
  APBW w{};
  const int nplane = (c.flags & F_GATING) ? 4 : 3;
  w.wqkvg = cur.take((int64_t)nplane * c.HDp * c.C);
  w.bq = cur.take(c.HDp);
  w.wo = cur.take((int64_t)c.C * c.HDp);
  w.wz = cur.take((int64_t)c.H * c.Cz);
  w.lnzw = cur.take(c.Cz);
  w.lnzb = cur.take((c.flags & F_LNZ_B) ? c.Cz : 0);
  if (c.flags & F_ADA) {
    w.lnsw = cur.take(c.Cs);
    w.lnsb = cur.take((c.flags & F_LNS_B) ? c.Cs : 0);
    w.adaw = cur.take((int64_t)2 * c.C * c.Cs);
    w.adagb = cur.take(c.C);
    w.adaow = cur.take((int64_t)c.C * c.Cs);
    w.adaob = cur.take(c.C);
  } else {
    w.lnaw = cur.take((c.flags & F_LNA_W) ? c.C : 0);
    w.lnab = cur.take((c.flags & F_LNA_B) ? c.C : 0);
  }
  return w;
}

XW xapb_weights(const at::Tensor& pack, const Cfg& c) {
  Cursor cur{(const bf16*)pack.data_ptr(), pack.numel()};
  XW w{};
  const int nqg = (c.flags & F_GATING) ? 2 : 1;
  w.wqg = cur.take((int64_t)nqg * c.HDp * c.C);
  w.bq = cur.take(c.HDp);
  w.wkv = cur.take((int64_t)2 * c.HDp * c.C);
  w.woT = cur.take((int64_t)c.C * c.HDp);
  w.wz = cur.take((int64_t)c.H * c.Cz);
  if (c.flags & F_ADA) {
    w.lnsqw = cur.take(c.Cs);
    w.lnsqb = cur.take((c.flags & F_LNS_B) ? c.Cs : 0);
    w.adaqw = cur.take((int64_t)2 * c.C * c.Cs);
    w.adaqgb = cur.take(c.C);
    w.lnskw = cur.take(c.Cs);
    w.lnskb = cur.take((c.flags & F_LNSK_B) ? c.Cs : 0);
    w.adakw = cur.take((int64_t)2 * c.C * c.Cs);
    w.adakgb = cur.take(c.C);
    w.adaow = cur.take((int64_t)c.C * c.Cs);
    w.adaob = cur.take(c.C);
  } else {
    w.lnqw = cur.take((c.flags & F_LNA_W) ? c.C : 0);
    w.lnqb = cur.take((c.flags & F_LNA_B) ? c.C : 0);
    w.lnkw = cur.take((c.flags & F_LNK_W) ? c.C : 0);
    w.lnkb = cur.take((c.flags & F_LNK_B) ? c.C : 0);
  }
  return w;
}

// Raising the dynamic shared-memory cap is a driver call, so it is done once per
// kernel rather than once per forward (it was costing more than the kernels).
template <typename K>
void set_smem(K kernel, size_t bytes) {
  static size_t granted = 0;
  if (bytes > 48u * 1024u && bytes > granted) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    granted = bytes;
  }
}

const bf16* opt_ptr(const at::Tensor& t) {
  return t.defined() && t.numel() ? (const bf16*)t.data_ptr() : nullptr;
}

// Leading dims must all be 1: the captured workload is always batch 1, and a
// batched fast path would never be exercised.  Anything the fast path does not
// cover returns an undefined tensor, and the module falls back to the reference.
bool batch_is_one(const at::Tensor& t, int nontrivial) {
  for (int64_t i = 0; i + nontrivial < t.dim(); ++i)
    if (t.size(i) != 1) return false;
  return true;
}

bool ok_input(const at::Tensor& t, int nontrivial) {
  return t.defined() && t.is_cuda() && t.is_contiguous() &&
         t.scalar_type() == at::kBFloat16 && t.dim() >= nontrivial &&
         batch_is_one(t, nontrivial);
}

at::Tensor apb_forward(const at::Tensor& pack, const at::Tensor& a, const at::Tensor& z,
                       const at::Tensor& s, const at::Tensor& mask, int64_t C,
                       int64_t Cs, int64_t Cz, int64_t H, int64_t D, double eps,
                       double inf, int64_t flags) {
  if (!ok_input(a, 2) || !ok_input(z, 3)) return at::Tensor();
  const bool ada = flags & F_ADA;
  if (ada && (!ok_input(s, 2) || s.dim() > a.dim())) return at::Tensor();
  if (mask.defined() && mask.numel() && !ok_input(mask, 1)) return at::Tensor();

  Cfg c{};
  c.N = (int)a.size(-2);
  c.Nt = (int)((c.N + TILE - 1) / TILE * TILE);
  c.C = (int)C;
  c.Cs = (int)Cs;
  c.Cz = (int)Cz;
  c.H = (int)H;
  c.D = (int)D;
  c.Dp = (int)((D + TILE - 1) / TILE * TILE);
  c.HDp = c.H * c.Dp;
  c.nC = c.C / TILE;
  c.nCs = c.Cs / TILE;
  c.nHD = c.HDp / TILE;
  c.eps = (float)eps;
  c.inf = (float)inf;
  c.qscale = 1.f / std::sqrt((float)D);
  c.flags = (int)flags;
  if (c.N > 32 || a.size(-1) != C || z.size(-1) != Cz || z.size(-2) != c.N ||
      z.size(-3) != c.N || (ada && (s.size(-1) != Cs || s.size(-2) != c.N)) ||
      (mask.defined() && mask.numel() && mask.size(-1) != c.N))
    return at::Tensor();

  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto st = at::cuda::getCurrentCUDAStream();
  APBW w = apb_weights(pack, c);
  auto opts = a.options();
  auto out = at::empty(a.sizes(), opts);
  const bf16* ap = (const bf16*)a.data_ptr();
  const bf16* sp = opt_ptr(s);
  const bf16* mp = opt_ptr(mask);
  const int nplane = (flags & F_GATING) ? 4 : 3;
  at::Tensor x;
  if (ada) {
    x = at::empty({c.Nt, c.C}, opts);
    // One column tile per block, 16 warps splitting K.
    const int warps = 16, thr = warps * WARP;
    size_t sm = sizeof(bf16) * c.Nt * c.Cs +
                sizeof(float) * (2 * c.Nt + (size_t)warps * 2 * TSZ);
    set_smem(apb_prep, sm);
    apb_prep<<<c.nC, thr, sm, st>>>(c, w, ap, sp, (bf16*)x.data_ptr());
  }

  auto qkvg = at::empty({4, c.Nt, c.HDp}, opts);
  auto zb = at::empty({c.H, c.Nt, c.Nt}, opts.dtype(at::kFloat));
  {
    const int warps = 16, thr = warps * WARP;
    int mainBlocks = nplane * c.nHD;
    int ppw = 2;
    int zbBlocks = (c.N * c.N + warps * ppw - 1) / (warps * ppw);
    size_t sm = std::max(sizeof(bf16) * c.Nt * c.C + sizeof(float) * warps * TSZ,
                         sizeof(bf16) * c.H * c.Cz + sizeof(float) * warps * c.Cz);
    set_smem(apb_qkvg, sm);
    apb_qkvg<<<mainBlocks + zbBlocks, thr, sm, st>>>(
        c, w, ap, ada ? (const bf16*)x.data_ptr() : nullptr, (const bf16*)z.data_ptr(),
        (bf16*)qkvg.data_ptr(), zb.data_ptr<float>(), mainBlocks, ppw);
  }
  {
    const int warps = 16, thr = warps * WARP;
    size_t sm = sizeof(bf16) * (c.Nt * c.HDp + (ada ? c.Nt * c.Cs : 0) +
                                (size_t)warps * TILE * c.Nt) +
                sizeof(float) * (size_t)warps * (TILE * (c.Nt + 1) + 2 * TSZ);
    set_smem(apb_attn_out, sm);
    apb_attn_out<<<c.nC, thr, sm, st>>>(c, w, sp, mp, (const bf16*)qkvg.data_ptr(),
                                        zb.data_ptr<float>(), (bf16*)out.data_ptr());
  }
  return out;
}

at::Tensor xapb_forward(const at::Tensor& pack, const at::Tensor& a, const at::Tensor& z,
                        const at::Tensor& s, const at::Tensor& mask, int64_t C,
                        int64_t Cs, int64_t Cz, int64_t H, int64_t D, int64_t nQuery,
                        int64_t nKey, double eps, double inf, int64_t flags) {
  if (!ok_input(a, 2) || !ok_input(z, 4)) return at::Tensor();
  const bool ada = flags & F_ADA;
  if (ada && (!ok_input(s, 2) || s.dim() > a.dim())) return at::Tensor();
  if (mask.defined() && mask.numel() && !ok_input(mask, 1)) return at::Tensor();

  Cfg c{};
  c.nAtom = (int)a.size(-2);
  c.Q = (int)nQuery;
  c.K = (int)nKey;
  c.NB = (int)((c.nAtom + c.Q - 1) / c.Q);
  c.N = c.NB * c.Q;
  c.Nt = c.N;
  c.C = (int)C;
  c.Cs = (int)Cs;
  c.Cz = (int)Cz;
  c.H = (int)H;
  c.D = (int)D;
  c.Dp = (int)((D + TILE - 1) / TILE * TILE);
  c.HDp = c.H * c.Dp;
  c.nC = c.C / TILE;
  c.nCs = c.Cs / TILE;
  c.nHD = c.HDp / TILE;
  c.eps = (float)eps;
  c.inf = (float)inf;
  c.qscale = 1.f / std::sqrt((float)D);
  c.flags = (int)flags;
  if (c.N % TILE || c.Cz > MAXCZ || c.HDp % 8 || a.size(-1) != C ||
      (ada && (s.size(-1) != Cs || s.size(-2) != c.nAtom)) || z.size(-1) != Cz ||
      z.size(-2) != c.K || z.size(-3) != c.Q || z.size(-4) != c.NB ||
      (mask.defined() && mask.numel() && mask.size(-1) != c.nAtom))
    return at::Tensor();

  // Query sub-tiling: keep enough blocks resident to stream z at full bandwidth.
  int qs = c.Q;
  while (qs > 1 && c.NB * (c.Q / qs) < 96) qs >>= 1;
  c.QS = qs;

  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto st = at::cuda::getCurrentCUDAStream();
  XW w = xapb_weights(pack, c);
  auto opts = a.options();
  auto out = at::empty(a.sizes(), opts);
  auto qgkv = at::empty({4, c.N, c.HDp}, opts);
  auto osc = ada ? at::empty({c.N, c.C}, opts) : at::empty({0}, opts);
  const bf16* sp = opt_ptr(s);
  const bf16* mp = opt_ptr(mask);
  {
    // One warp per (column tile, matrix) task: 32 warps means each warp issues
    // ~8x fewer dependent fragment loads than 8 warps did.
    const int warps = 32, thr = warps * WARP;
    size_t sm = sizeof(bf16) * (size_t)TILE * (3 * c.C + 3 * c.Cs) +
                sizeof(float) * std::max(warps, 4 * c.nC) * TSZ;
    set_smem(xapb_prep, sm);
    xapb_prep<<<c.N / TILE, thr, sm, st>>>(c, w, (const bf16*)a.data_ptr(), sp,
                                           (bf16*)qgkv.data_ptr(),
                                           ada ? (bf16*)osc.data_ptr() : nullptr);
  }
  {
    const int thr = 1024;
    size_t sm = sizeof(bf16) * (2 * (size_t)c.K * c.HDp + 3 * (size_t)c.QS * c.HDp +
                                (size_t)c.QS * c.K * c.Cz + (size_t)c.C * c.HDp) +
                sizeof(float) * ((size_t)c.H * c.QS * c.K + c.K + c.QS + WARP);
    set_smem(xapb_attn_out, sm);
    xapb_attn_out<<<c.NB * (c.Q / c.QS), thr, sm, st>>>(
        c, w, (const bf16*)z.data_ptr(), mp, (const bf16*)qgkv.data_ptr(),
        ada ? (const bf16*)osc.data_ptr() : nullptr, (bf16*)out.data_ptr());
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apb_forward", &apb_forward);
  m.def("xapb_forward", &xapb_forward);
}
