// Fused Kimi MoE for the decode / small-token regime.
//
// The baseline hands the whole routed MoE to flashinfer's trtllm-gen kernel.
// That kernel's *GPU* time is close to the HBM roofline, but its host-side
// entry costs ~615us per call in this build (measured: the native TVM-FFI
// call alone, tactic selection included), and the benchmark times wall-clock
// latency per call.  At M<=443 the whole forward only needs 60-620us of GPU
// work, so the module is host-bound by an order of magnitude at M=1.
//
// This file replaces the entire small-M path with one host call, and there are
// two decompositions behind it.
//
// M > 16 -- expert-major, wmma (five kernels, one launch each):
//
//   gate    : the router projection, in the L1 kernel's exact reduction order.
//   route   : sigmoid+bias top-8, renormalised weights, per-expert token
//             lists, block table (prefix sum in the last block), zeroes the
//             fp32 output accumulator.
//   gemm1   : per (expert, 16-token tile, N-tile): x @ w13^T -> SwiGLU -> hbuf
//   gemm2   : per (expert, 16-token tile, H-tile): hbuf @ w2^T, scaled by the
//             routing weight and atomically accumulated into fp32.
//   finalize: fp32 -> bf16, and re-zero the counters for the next call.
//
// M <= 16 -- token-major, SIMT (three kernels):
//
//   gate+route : projection and top-8 in one kernel, joined by a per-token-block
//                arrival counter; emits the (expert, weight) pair list.
//   gemm1_tok  : per (token, expert slot, row group), a warp per expert slot.
//   gemm2_tok  : same, and the 9 partials of an output element land in 9 warps
//                of one CTA, so the cross-expert reduction is one __syncthreads
//                and the result is written bf16 exactly once.
//
// The token-major path needs no block table, no per-expert counts, no fp32
// accumulator, no atomics and no finalize pass: at tiny M each activated expert
// holds one token, so the work list is exactly TOPK+1 = 9 pairs per token, a
// static count the host can size the grid on. Above M=16 the duplicate-expert
// weight re-reads that costs (0% at M=1, ~13% at M=8, ~46% at M=26) outweigh it
// and the 16-token wmma tile wins.
//
// The shared expert is expert index E: its gate_up / down weights have exactly
// the same per-expert layout as w13[e] / w2[e], its token list is the identity
// and its routing weight is 1, so it rides the same two GEMM kernels instead of
// three separate PyTorch ops plus a final add.
//
// Why m16n16k16 tensor cores above M=16 and not SIMT FMA: the token dimension is
// tiny, so
// the problem is HBM-bound (3.6GB of expert weights once all 256 experts are
// touched, ~450us at peak).  But FP32 FMA peak is only ~57 TFLOP/s at the
// benchmark's locked 1500MHz clock, and at M=443 the routed GEMMs need 50
// GFLOP -- 880us, i.e. FMA would become the bottleneck by 2x.  Padding the
// token dimension to 16 and using bf16 MMA makes compute free at every M in
// range (58 GFLOP padded, <130us even at a fifth of tensor-core peak) and
// leaves the kernel purely bandwidth-bound, which is the roofline we want.
//
// Numerics: the routing replicates trtllm's own formulation exactly --
// sigmoid_accurate(x) = 0.5*tanh(0.5x)+0.5 in fp32 on the fp32 router logits,
// plus the bf16 bias widened to fp32, top-8 on the *biased* score, weights
// taken from the *unbiased* sigmoid and renormalised before the
// routed_scaling_factor.  Identical inputs therefore give bitwise-identical
// expert selection; only the final weight keeps fp32 where trtllm rounds to
// bf16.  Both GEMMs accumulate in fp32 and the cross-expert reduction is fp32.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

using bf16 = __nv_bfloat16;
using namespace nvcuda;

constexpr int WM = 16;  // MMA tile / token-tile height

// Programmatic dependent launch. All four kernels chain strictly, and the L1
// router GEMM ahead of them already emits its own trigger, so every launch in
// the decode path can overlap its grid setup with its producer's tail. That is
// worth ~2us per launch here (the stream advances in whole ~2.048us quanta on
// B200), i.e. most of a 45us M=1 budget spread over five launches.
//
// Correctness: `pdl_wait` sits before the first load of producer-written data,
// and `pdl_trigger` at the very end of every block -- including the one block
// of route_kernel that builds the block table after all the others have
// finished, since a dependent grid is released only once *every* producer block
// has triggered or completed.
__device__ __forceinline__ void pdl_wait() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// trtllm's routing sigmoid (RoutingKernel.cuh: sigmoid_accurate).
__device__ __forceinline__ float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// Total order on floats as unsigned ints, so (score, index) fits one u64 key
// and a plain max-reduce breaks ties toward the lower expert index.
__device__ __forceinline__ unsigned int ford(float f) {
  unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------------------
// route: one block per token.
//
// Emits, for every expert e that token t selected, an entry (t, weight) in
// e's list.  The last block to arrive turns the per-expert counts into the
// block table consumed by both GEMM kernels, so the whole routing stage is a
// single launch with no host round-trip for the work count.
// ---------------------------------------------------------------------------
template <int E, int TOPK>
__global__ __launch_bounds__(256) void route_kernel(
    const float* __restrict__ logits, const bf16* __restrict__ bias, int M, int H,
    float scale, int bm, int* __restrict__ cnt, int* __restrict__ tlist,
    float* __restrict__ twt, int* __restrict__ meta, int* __restrict__ blkE,
    int* __restrict__ blkT, float* __restrict__ accf) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  pdl_wait();

  __shared__ int   sel[TOPK];
  __shared__ float ssig[TOPK];
  __shared__ int   is_last;

  // Zero this token's slice of the fp32 accumulator (H is a multiple of 4).
  {
    float4* dst = reinterpret_cast<float4*>(accf + (size_t)t * H);
    const int n4 = H >> 2;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = tid; i < n4; i += 256) dst[i] = z;
  }
  // The shared expert takes every token, in order, with weight 1.
  if (tid == 0) {
    tlist[(size_t)E * M + t] = t;
    twt[(size_t)E * M + t] = 1.0f;
  }

  // Top-8 in a single warp: lane l owns experts l, l+32, ..., l+224.
  if (tid < 32) {
    constexpr int PL = E / 32;  // candidates per lane
    const int lane = tid;
    float sb[PL], sg[PL];
    int   id[PL];
#pragma unroll
    for (int j = 0; j < PL; ++j) {
      const int e = lane + 32 * j;
      const float s = sigmoid_accurate(logits[(size_t)t * E + e]);
      sg[j] = s;
      sb[j] = s + __bfloat162float(bias[e]);
      id[j] = e;
    }
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      unsigned long long best = 0ull;
#pragma unroll
      for (int j = 0; j < PL; ++j) {
        const unsigned long long p =
            ((unsigned long long)ford(sb[j]) << 32) | (unsigned int)(E - 1 - id[j]);
        best = p > best ? p : best;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
      { const unsigned long long o = __shfl_xor_sync(0xffffffffu, best, off);
        best = o > best ? o : best; }
      const int wid = E - 1 - (int)(unsigned int)(best & 0xffffffffu);
      if (lane == 0) sel[k] = wid;
#pragma unroll
      for (int j = 0; j < PL; ++j)
        if (id[j] == wid) {
          ssig[k] = sg[j];
          sb[j] = -INFINITY;
        }
    }
  }
  __syncthreads();

  if (tid < TOPK) {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) sum += ssig[k];
    const int e = sel[tid];
    const int pos = atomicAdd(&cnt[e], 1);
    tlist[(size_t)e * M + pos] = t;
    twt[(size_t)e * M + pos] = ssig[tid] * scale / sum;
  }

  // Last block builds the block table.  At M=1 this block is trivially it, and
  // the release fence plus the arrival counter are pure latency on the shortest
  // path there is.
  if (M > 1) {
    __threadfence();
    if (tid == 0) is_last = (atomicAdd(&meta[0], 1) == M - 1);
  } else if (tid == 0) {
    is_last = 1;
  }
  __syncthreads();
  if (!is_last) {
    pdl_trigger();
    return;
  }

  __shared__ int nb[E + 1];
  __shared__ int off[E + 1];
  nb[tid] = (cnt[tid] + bm - 1) / bm;
  if (tid == 0) {
    cnt[E] = M;
    nb[E] = (M + bm - 1) / bm;
  }
  __syncthreads();
  // Hillis-Steele inclusive scan over the first E entries.
  off[tid] = nb[tid];
  __syncthreads();
  for (int d = 1; d < E; d <<= 1) {
    const int v = (tid >= d) ? off[tid - d] : 0;
    __syncthreads();
    off[tid] += v;
    __syncthreads();
  }
  const int start = off[tid] - nb[tid];  // exclusive
  if (tid == 0) meta[1] = off[E - 1] + nb[E];
  for (int j = 0; j < nb[tid]; ++j) {
    blkE[start + j] = tid;
    blkT[start + j] = j * bm;
  }
  if (tid == 0) {
    const int s = off[E - 1];
    for (int j = 0; j < nb[E]; ++j) {
      blkE[s + j] = E;
      blkT[s + j] = j * bm;
    }
  }
  pdl_trigger();
}

// ---------------------------------------------------------------------------
// Streaming GEMM core, shared by both expert GEMMs.
//
// Both are the same shape family: a [<=BM, K] activation tile against a [N, K]
// weight tile with K contiguous, N-major, accumulating in fp32.  The whole
// small-M regime is HBM-bound, so the only thing that matters is keeping
// enough weight bytes in flight per SM.  A first version had wmma read the B
// fragments straight from global (legal, and the 32B segments waste no DRAM
// sectors) and reached only 0.84 TB/s of 8: with 4 warps and no pipelining an
// SM had ~1KB outstanding where saturating B200 needs ~40KB.
//
// Hence the cp.async pipeline below: STAGES-1 chunks of both operands are in
// flight at once, which decouples issue from consumption without needing more
// warps or more CTAs -- important because at M=1 only 8 experts are activated,
// so the grid barely covers the SMs and per-CTA parallelism is all there is.
//
// TM is the token-tile height in units of the 16-row MMA tile.  It has to be at
// least the largest per-expert token count, because a second tile re-reads that
// expert's entire weight: at M=443 the counts are Binomial(3544, 1/256), mean
// 13.8 with sd 3.7, so ~28% of experts exceed 16 and TM=1 there cost ~28% extra
// HBM traffic.  Padding the token dimension instead is free -- MMA throughput is
// two orders of magnitude off the critical path here.
//
// Shared tiles are padded to BK+8 so the column-major B fragment reads stay off
// a single bank; 8 also keeps ldm a multiple of 8 as wmma requires for bf16.
// ---------------------------------------------------------------------------

// gemm1: h = silu(x @ w1^T) * (x @ w3^T) for one (expert, token tile, N tile).
// w13[e] is [2I, H] row-major: the gate rows are the first BN of the staged
// tile and the up rows the second BN, so one tile feeds both halves of SwiGLU
// and the intermediate never reaches global memory in fp32.
// GU=2 gives each warp a single 16-wide output tile of *one* SwiGLU half
// instead of both, doubling the warps per CTA at identical shared-memory
// footprint. That matters because the warp count here is otherwise pinned by
// the problem: with GU=1 the total is 9*I/16 warps at M=1 no matter how BN and
// NW trade off, i.e. ~4 warps/SM, too few to hide the shared-memory fragment
// loads and MMAs behind the streaming loads.
template <int TM, int BN, int NW, int BK, int STAGES, int GU>
__global__ __launch_bounds__(32 * NW) void gemm1_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ w13,
    const bf16* __restrict__ sgu, const int* __restrict__ blkE,
    const int* __restrict__ blkT, const int* __restrict__ cnt,
    const int* __restrict__ tlist, const int* __restrict__ meta,
    bf16* __restrict__ hbuf, int M, int H, int I, int E) {
  constexpr int BM = 16 * TM;
  constexpr int NWG = NW / GU;          // warps per SwiGLU half
  constexpr int SUB = BN / (16 * NWG);  // 16-wide tiles per warp
  constexpr int NACC = 2 / GU;          // accumulator sets per warp
  constexpr int THREADS = 32 * NW;
  constexpr int LDW = BK + 8;
  constexpr int WROWS = 2 * BN;
  constexpr int WSTAGE = WROWS * LDW;
  constexpr int XSTAGE = BM * LDW;
  constexpr int VPT = 8;         // bf16 per 16B cp.async
  constexpr int TPR = BK / VPT;  // threads covering one row
  constexpr int RSTEP = THREADS / TPR;
  constexpr int WITER = WROWS / RSTEP;
  static_assert(BN == 16 * NWG * SUB && (GU == 1 || GU == 2), "BN must tile 16*NWG");
  static_assert(WROWS % RSTEP == 0 && THREADS % TPR == 0, "row tiling");

  const int b = blockIdx.x;
  pdl_wait();
  if (b >= meta[1]) {
    pdl_trigger();
    return;
  }
  const int e = blkE[b];
  const int t0 = blkT[b];
  const int ntok = min(BM, cnt[e] - t0);
  if (ntok <= 0) {
    pdl_trigger();
    return;
  }

  const bf16* W = (e == E) ? sgu : (w13 + (size_t)e * 2 * I * H);
  const int n0 = blockIdx.y * BN;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wh = (GU == 2) ? (warp / NWG) : 0;  // 0 = gate half, 1 = up half
  const int ws = (GU == 2) ? (warp % NWG) : warp;

  extern __shared__ bf16 sm1[];
  bf16* sW = sm1;
  bf16* sX = sm1 + STAGES * WSTAGE;

  const int* tl = tlist + (size_t)e * M + t0;
  const int lr = threadIdx.x / TPR;
  const int lc = (threadIdx.x % TPR) * VPT;

  // Rows past ntok stay zero in every stage, so the padded MMA rows contribute
  // zeros instead of garbage and the loader needs no per-stage branch.
  for (int r = lr; r < BM; r += RSTEP)
    if (r >= ntok)
      for (int st = 0; st < STAGES; ++st)
        *reinterpret_cast<uint4*>(sX + st * XSTAGE + r * LDW + lc) =
            make_uint4(0u, 0u, 0u, 0u);

  const int nchunk = H / BK;
  auto stage_in = [&](int st, int c) {
    const int kc = c * BK;
    bf16* dW = sW + st * WSTAGE;
#pragma unroll
    for (int it = 0; it < WITER; ++it) {
      const int r = lr + it * RSTEP;
      const int row = (r < BN) ? (n0 + r) : (I + n0 + r - BN);
      __pipeline_memcpy_async(dW + r * LDW + lc, W + (size_t)row * H + kc + lc, 16);
    }
    bf16* dX = sX + st * XSTAGE;
    for (int r = lr; r < ntok; r += RSTEP)
      __pipeline_memcpy_async(dX + r * LDW + lc,
                              x + (size_t)tl[r] * H + kc + lc, 16);
    __pipeline_commit();
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NACC][TM][SUB];
#pragma unroll
  for (int h = 0; h < NACC; ++h)
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
      for (int s = 0; s < SUB; ++s) wmma::fill_fragment(acc[h][m][s], 0.f);

#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) stage_in(st, st);

  for (int c = 0; c < nchunk; ++c) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    const bf16* pW = sW + (c % STAGES) * WSTAGE;
    const bf16* pX = sX + (c % STAGES) * XSTAGE;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa[TM];
#pragma unroll
      for (int m = 0; m < TM; ++m)
        wmma::load_matrix_sync(fa[m], pX + m * 16 * LDW + kk * 16, LDW);
#pragma unroll
      for (int s = 0; s < SUB; ++s) {
        const int nl = (ws * SUB + s) * 16;
#pragma unroll
        for (int h = 0; h < NACC; ++h) {
          const int row = (GU == 2) ? (wh * BN + nl) : (h * BN + nl);
          wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> fb;
          wmma::load_matrix_sync(fb, pW + row * LDW + kk * 16, LDW);
#pragma unroll
          for (int m = 0; m < TM; ++m)
            wmma::mma_sync(acc[h][m][s], fa[m], fb, acc[h][m][s]);
        }
      }
    }
    const int nx = c + STAGES - 1;
    if (nx < nchunk)
      stage_in(nx % STAGES, nx);
    else
      __pipeline_commit();  // empty batch: keeps wait_prior's depth exact
  }

  // SwiGLU epilogue through the (now dead) staging buffer.  With GU=2 the two
  // halves of a column live in different warps, so the combine crosses a block
  // barrier; with GU=1 each warp owns its own pair and __syncwarp suffices.
  __syncthreads();
  constexpr int TILE = SUB * 256;
  float* base = reinterpret_cast<float*>(sm1);
  bf16* hb = hbuf + (size_t)b * BM * I;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    float* mine = base + ((GU == 2) ? (wh * NWG + ws) : (2 * warp)) * TILE;
#pragma unroll
    for (int s = 0; s < SUB; ++s) {
      wmma::store_matrix_sync(mine + s * 256, acc[0][m][s], 16, wmma::mem_row_major);
      if (GU == 1)
        wmma::store_matrix_sync(mine + TILE + s * 256, acc[NACC - 1][m][s], 16,
                                wmma::mem_row_major);
    }
    if (GU == 2) __syncthreads(); else __syncwarp();
    // GU=2: gate tiles occupy the first NWG slots and up tiles the next NWG, so
    // one strided pass over NWG*TILE pairs them up.
    const int span = (GU == 2) ? (NWG * TILE) : TILE;
    const int step = (GU == 2) ? THREADS : 32;
    for (int idx = ((GU == 2) ? (int)threadIdx.x : lane); idx < span; idx += step) {
      const int sl = (GU == 2) ? (idx / TILE) : ws;
      const int in = (GU == 2) ? (idx % TILE) : idx;
      const float* gp = (GU == 2) ? (base + (idx / TILE) * TILE) : mine;
      const float* up = (GU == 2) ? (base + (NWG + idx / TILE) * TILE) : (mine + TILE);
      const int r = m * 16 + ((in >> 4) & 15);
      const float g = gp[in];
      const float h = (g / (1.f + __expf(-g))) * up[in];
      hb[(size_t)r * I + n0 + (sl * SUB + (in >> 8)) * 16 + (in & 15)] =
          __float2bfloat16(r < ntok ? h : 0.f);
    }
    if (GU == 2) __syncthreads(); else __syncwarp();
  }
  pdl_trigger();
}

// gemm2: out[t] += weight * (h @ w2^T).  w2[e] is [H, I] row-major, so the same
// column-major B trick applies with ldm=I, and the routing weight folds into
// the fp32 atomic, so neither the scale nor the shared-expert add needs a pass.
template <int TM, int BH, int NW, int BK, int STAGES>
__global__ __launch_bounds__(32 * NW) void gemm2_kernel(
    const bf16* __restrict__ hbuf, const bf16* __restrict__ w2,
    const bf16* __restrict__ sdn, const int* __restrict__ blkE,
    const int* __restrict__ blkT, const int* __restrict__ cnt,
    const int* __restrict__ tlist, const float* __restrict__ twt,
    const int* __restrict__ meta, float* __restrict__ accf, int M, int H, int I,
    int E) {
  constexpr int BM = 16 * TM;
  constexpr int SUB = BH / (16 * NW);
  constexpr int THREADS = 32 * NW;
  constexpr int LDW = BK + 8;
  constexpr int WSTAGE = BH * LDW;
  constexpr int ASTAGE = BM * LDW;
  constexpr int VPT = 8;
  constexpr int TPR = BK / VPT;
  constexpr int RSTEP = THREADS / TPR;
  constexpr int WITER = BH / RSTEP;
  static_assert(BH == 16 * NW * SUB, "BH must tile 16*NW");
  static_assert(BH % RSTEP == 0 && THREADS % TPR == 0, "row tiling");

  const int b = blockIdx.x;
  pdl_wait();
  if (b >= meta[1]) {
    pdl_trigger();
    return;
  }
  const int e = blkE[b];
  const int t0 = blkT[b];
  const int ntok = min(BM, cnt[e] - t0);
  if (ntok <= 0) {
    pdl_trigger();
    return;
  }

  const bf16* W = (e == E) ? sdn : (w2 + (size_t)e * H * I);
  const int h0 = blockIdx.y * BH;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bf16* A = hbuf + (size_t)b * BM * I;

  extern __shared__ bf16 sm2[];
  bf16* sW = sm2;
  bf16* sA = sm2 + STAGES * WSTAGE;

  const int lr = threadIdx.x / TPR;
  const int lc = (threadIdx.x % TPR) * VPT;
  const int nchunk = I / BK;
  const int arows = min(BM, ((ntok + 15) / 16) * 16);  // whole MMA tiles only

  auto stage_in = [&](int st, int c) {
    const int kc = c * BK;
    bf16* dW = sW + st * WSTAGE;
#pragma unroll
    for (int it = 0; it < WITER; ++it) {
      const int r = lr + it * RSTEP;
      __pipeline_memcpy_async(dW + r * LDW + lc,
                              W + (size_t)(h0 + r) * I + kc + lc, 16);
    }
    bf16* dA = sA + st * ASTAGE;
    for (int r = lr; r < arows; r += RSTEP)
      __pipeline_memcpy_async(dA + r * LDW + lc, A + (size_t)r * I + kc + lc, 16);
    __pipeline_commit();
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TM][SUB];
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int s = 0; s < SUB; ++s) wmma::fill_fragment(acc[m][s], 0.f);

#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) stage_in(st, st);

  const int mtiles = arows / 16;
  for (int c = 0; c < nchunk; ++c) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    const bf16* pW = sW + (c % STAGES) * WSTAGE;
    const bf16* pA = sA + (c % STAGES) * ASTAGE;
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> fa[TM];
#pragma unroll
      for (int m = 0; m < TM; ++m)
        if (m < mtiles)
          wmma::load_matrix_sync(fa[m], pA + m * 16 * LDW + kk * 16, LDW);
#pragma unroll
      for (int s = 0; s < SUB; ++s) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> fb;
        wmma::load_matrix_sync(fb, pW + (warp * SUB + s) * 16 * LDW + kk * 16, LDW);
#pragma unroll
        for (int m = 0; m < TM; ++m)
          if (m < mtiles) wmma::mma_sync(acc[m][s], fa[m], fb, acc[m][s]);
      }
    }
    const int nx = c + STAGES - 1;
    if (nx < nchunk)
      stage_in(nx % STAGES, nx);
    else
      __pipeline_commit();
  }

  __syncthreads();
  float* sc = reinterpret_cast<float*>(sm2) + warp * (SUB * 256);
  const int* tl = tlist + (size_t)e * M + t0;
  const float* tw = twt + (size_t)e * M + t0;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    if (m >= mtiles) break;
#pragma unroll
    for (int s = 0; s < SUB; ++s)
      wmma::store_matrix_sync(sc + s * 256, acc[m][s], 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < SUB * 256; idx += 32) {
      const int r = m * 16 + ((idx >> 4) & 15);
      if (r >= ntok) continue;
      const int h = h0 + (warp * SUB + (idx >> 8)) * 16 + (idx & 15);
      atomicAdd(&accf[(size_t)tl[r] * H + h], tw[r] * sc[idx]);
    }
    __syncwarp();
  }
  pdl_trigger();
}

// Widen one bf16 lane of a 16B vector load to fp32.
__device__ __forceinline__ float bf2f(const uint4& v, int k) {
  return __bfloat162float(reinterpret_cast<const bf16*>(&v)[k]);
}

// ---------------------------------------------------------------------------
// Token-major tiny-M path: three kernels, no block table, no fp32 accumulator,
// no atomics, no finalize pass.
//
// At tiny M every activated expert holds essentially one token, so the natural
// work item is the (token, expert slot) pair -- and there are exactly
// TOPK+1 = 9 of them per token (8 routed plus the shared expert). That count is
// *static*, so the work list needs no prefix sum, no per-expert counts and no
// device round-trip: pair j = t*(TOPK+1) + w, and the grid is sized on the host.
//
// The decomposition assigns pair j to warp w of the CTA at (t, row group), so:
//   * gemm1 stages the token's x row in shared once and all 9 warps read it;
//   * gemm2's 9 partials for one output element live in 9 warps of the *same*
//     CTA, so the cross-expert reduction is one __syncthreads and the routing
//     weight folds into it -- the output is written bf16 exactly once.
//
// The price is that a duplicate expert (two tokens picking the same one) reads
// its weights twice: 0% at M=1, ~13% extra at M=8, ~46% at M=26. So this path
// wins at the bottom of the range and the expert-major tiles take over above it.
// ---------------------------------------------------------------------------

// route: one warp per token. Writes the (expert, weight) pair list directly.
template <int E, int TOPK>
__global__ __launch_bounds__(32) void route_tok_kernel(
    const float* __restrict__ logits, const bf16* __restrict__ bias,
    float scale, int* __restrict__ eid, float* __restrict__ wgt) {
  const int t = blockIdx.x;
  const int lane = threadIdx.x;
  pdl_wait();

  __shared__ int   sel[TOPK];
  __shared__ float ssig[TOPK];

  constexpr int PL = E / 32;
  float sb[PL], sg[PL];
  int   id[PL];
#pragma unroll
  for (int j = 0; j < PL; ++j) {
    const int e = lane + 32 * j;
    const float s = sigmoid_accurate(logits[(size_t)t * E + e]);
    sg[j] = s;
    sb[j] = s + __bfloat162float(bias[e]);
    id[j] = e;
  }
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    unsigned long long best = 0ull;
#pragma unroll
    for (int j = 0; j < PL; ++j) {
      const unsigned long long p =
          ((unsigned long long)ford(sb[j]) << 32) | (unsigned int)(E - 1 - id[j]);
      best = p > best ? p : best;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    { const unsigned long long o = __shfl_xor_sync(0xffffffffu, best, off);
      best = o > best ? o : best; }
    const int wid = E - 1 - (int)(unsigned int)(best & 0xffffffffu);
    if (lane == 0) sel[k] = wid;
#pragma unroll
    for (int j = 0; j < PL; ++j)
      if (id[j] == wid) {
        ssig[k] = sg[j];
        sb[j] = -INFINITY;
      }
  }
  __syncwarp();

  if (lane <= TOPK) {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) sum += ssig[k];
    const int slot = t * (TOPK + 1) + lane;
    if (lane < TOPK) {
      eid[slot] = sel[lane];
      wgt[slot] = ssig[lane] * scale / sum;
    } else {
      eid[slot] = E;   // shared expert
      wgt[slot] = 1.0f;
    }
  }
  pdl_trigger();
}

// gemm1: grid (M, I/R), NWE = TOPK+1 warps. Warp w owns expert slot w and R
// (gate, up) row pairs, so SwiGLU stays inside the warp's registers.
template <int R, int NWE, int UNR>
__global__ __launch_bounds__(32 * NWE) void gemm1_tok_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ w13,
    const bf16* __restrict__ sgu, const int* __restrict__ eid,
    bf16* __restrict__ hbuf, int H, int I, int E) {
  constexpr int THREADS = 32 * NWE;
  static_assert(R <= 32, "one lane per output value in the epilogue");

  const int t = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  pdl_wait();

  extern __shared__ bf16 sX1t[];
  const int h8 = H >> 3;
  for (int c = threadIdx.x; c < h8; c += THREADS)
    *reinterpret_cast<uint4*>(sX1t + c * 8) =
        *reinterpret_cast<const uint4*>(x + (size_t)t * H + c * 8);

  const int j = t * NWE + warp;
  const int e = eid[j];
  const bf16* W = (e == E) ? sgu : (w13 + (size_t)e * 2 * I * H);
  __syncthreads();
  const bf16* px = sX1t + lane * 8;
  bf16* hbj = hbuf + (size_t)j * I;
  const int nstep = H >> 8;

  // Grid-stride over row groups. A one-CTA-per-row-group grid is
  // wave-quantised: 512 CTAs of 288 threads at 62 registers fit 3 per SM, i.e.
  // 444 slots, so the last 68 CTAs cost a whole second wave for 15% more work
  // and the kernel runs at 1/1.7 of its bandwidth. Sizing the grid to the
  // machine instead and looping makes every CTA's work equal.
  for (int ry = blockIdx.y; ry < (I / R); ry += gridDim.y) {
  const int n0 = ry * R;
  const bf16* pg = W + (size_t)n0 * H + lane * 8;
  const bf16* pu = W + (size_t)(I + n0) * H + lane * 8;

  float ag[R], au[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { ag[r] = 0.f; au[r] = 0.f; }

  for (int s0 = 0; s0 < nstep; s0 += UNR) {
    uint4 vg[UNR][R], vu[UNR][R], vx[UNR];
#pragma unroll
    for (int q = 0; q < UNR; ++q) {
      const int kc = (s0 + q) << 8;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        vg[q][r] = *reinterpret_cast<const uint4*>(pg + (size_t)r * H + kc);
        vu[q][r] = *reinterpret_cast<const uint4*>(pu + (size_t)r * H + kc);
      }
      vx[q] = *reinterpret_cast<const uint4*>(px + kc);
    }
#pragma unroll
    for (int q = 0; q < UNR; ++q)
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float a = bf2f(vx[q], k);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          ag[r] = fmaf(a, bf2f(vg[q][r], k), ag[r]);
          au[r] = fmaf(a, bf2f(vu[q][r], k), au[r]);
        }
      }
  }

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      ag[r] += __shfl_xor_sync(0xffffffffu, ag[r], off);
      au[r] += __shfl_xor_sync(0xffffffffu, au[r], off);
    }

  bf16* hb = hbj + n0;
#pragma unroll
  for (int r = 0; r < R; ++r)
    if (lane == r) {
      const float g = ag[r];
      hb[r] = __float2bfloat16((g / (1.f + __expf(-g))) * au[r]);
    }
  }
  pdl_trigger();
}

// gemm2: grid (M, H/R), NWE warps. Warp w computes its expert's partial for R
// output rows; the CTA then reduces the 9 weighted partials and writes bf16.
template <int R, int NWE, int UNR>
__global__ __launch_bounds__(32 * NWE) void gemm2_tok_kernel(
    const bf16* __restrict__ hbuf, const bf16* __restrict__ w2,
    const bf16* __restrict__ sdn, const int* __restrict__ eid,
    const float* __restrict__ wgt, bf16* __restrict__ out, int H, int I, int E) {
  const int t = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  pdl_wait();

  const int j = t * NWE + warp;
  const int e = eid[j];
  const bf16* W = (e == E) ? sdn : (w2 + (size_t)e * H * I);
  const bf16* pa = hbuf + (size_t)j * I + lane * 8;
  const float wj = wgt[j];
  const int nstep = I >> 8;
  __shared__ float red[NWE][R];

  // Grid-stride, same reason as gemm1_tok.
  for (int ry = blockIdx.y; ry < (H / R); ry += gridDim.y) {
  const int h0 = ry * R;
  const bf16* pw = W + (size_t)h0 * I + lane * 8;

  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;

  for (int s0 = 0; s0 < nstep; s0 += UNR) {
    uint4 vw[UNR][R], va[UNR];
#pragma unroll
    for (int q = 0; q < UNR; ++q) {
      const int kc = (s0 + q) << 8;
#pragma unroll
      for (int r = 0; r < R; ++r)
        vw[q][r] = *reinterpret_cast<const uint4*>(pw + (size_t)r * I + kc);
      va[q] = *reinterpret_cast<const uint4*>(pa + kc);
    }
#pragma unroll
    for (int q = 0; q < UNR; ++q)
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float a = bf2f(va[q], k);
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = fmaf(a, bf2f(vw[q][r], k), acc[r]);
      }
  }

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) red[warp][r] = acc[r] * wj;
  }
  __syncthreads();
  if (threadIdx.x < R) {
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < NWE; ++u) s += red[u][threadIdx.x];
    out[(size_t)t * H + h0 + threadIdx.x] = __float2bfloat16(s);
  }
  __syncthreads();  // `red` is reused by the next row group
  }
  pdl_trigger();
}

// ---------------------------------------------------------------------------
// Fused router projection + routing, for the token-major tiny path.
//
// The decode chain otherwise starts with a separate Python call into the L1
// `gate_linear` extension (~6us of host time and one ~2us stream quantum for a
// GEMV that moves 1.18MB).  Folding it in makes the whole fast path a single
// extension call.
//
// HARD constraint: expert *selection* must not move.  The L1 kernel's tiling is
// part of its result -- it is a K-split SIMT GEMM whose fp32 sum is taken in a
// specific order (each lane runs an fma chain over its own KPT 256-element
// chunks, then the 32 lane partials are added in lane order, then the WK warp
// partials in warp order).  A different order would perturb near-tie top-8
// decisions, and one flipped token at M=26 is 3.8% of the output elements --
// enough to fail the 0.99 threshold on its own.  So phase 1 below reproduces
// `rg_simt<BM_G, BNR, WK, KPT>` exactly, instantiated with the same
// (BM_G, BNR, WK, KPT) the L1 dispatcher would pick for this M, and
// `kimi_moe_gate_logits` exists so the host can assert bitwise equality before
// enabling the fused path at all.
//
// Phase 2 is the same routing as route_tok_kernel.  The two phases are joined by
// a per-token-block arrival counter: the last CTA of a column does the routing
// for that column's tokens, so no second launch and no host round trip.
// ---------------------------------------------------------------------------
template <int E, int TOPK, int BM_G, int BNR, int WK, int KPT>
__global__ __launch_bounds__(32 * WK) void gate_route_tok_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ gw,
    const bf16* __restrict__ bias, int M, float scale,
    float* __restrict__ lg, int* __restrict__ ctr, int* __restrict__ eid,
    float* __restrict__ wgt) {
  constexpr int K = WK * KPT * 256;  // = H, checked on the host

  const int n0 = blockIdx.x * BNR;
  const int tb = blockIdx.y;
  const int m0 = tb * BM_G;
  const int mrows = min(BM_G, M - m0);
  const int lane = threadIdx.x & 31;
  const int wk = threadIdx.x >> 5;
  const int kb = wk * (KPT * 256) + lane * 8;

  pdl_wait();

  const bf16* bp = gw + (size_t)n0 * K + kb;
  const bf16* ap = x + (size_t)m0 * K + kb;
  // Rows past M read a clamped (valid) row and are not stored, exactly as the
  // L1 kernel does, so the inner loop stays branch-free.
  int aoff[BM_G];
#pragma unroll
  for (int m = 0; m < BM_G; ++m) aoff[m] = (m < mrows ? m : mrows - 1) * K;

  float acc[BM_G][BNR];
#pragma unroll
  for (int m = 0; m < BM_G; ++m)
#pragma unroll
    for (int j = 0; j < BNR; ++j) acc[m][j] = 0.f;

#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    uint4 bv[BNR], av[BM_G];
#pragma unroll
    for (int j = 0; j < BNR; ++j)
      bv[j] = *reinterpret_cast<const uint4*>(bp + (size_t)j * K + i * 256);
#pragma unroll
    for (int m = 0; m < BM_G; ++m)
      av[m] = *reinterpret_cast<const uint4*>(ap + aoff[m] + i * 256);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bf[BNR], af[BM_G];
#pragma unroll
      for (int j = 0; j < BNR; ++j) bf[j] = bf2f(bv[j], k);
#pragma unroll
      for (int m = 0; m < BM_G; ++m) af[m] = bf2f(av[m], k);
#pragma unroll
      for (int m = 0; m < BM_G; ++m)
#pragma unroll
        for (int j = 0; j < BNR; ++j) acc[m][j] = fmaf(af[m], bf[j], acc[m][j]);
    }
  }

  __shared__ float sh[WK][32][BM_G * BNR + 1];
#pragma unroll
  for (int m = 0; m < BM_G; ++m)
#pragma unroll
    for (int j = 0; j < BNR; ++j) sh[wk][lane][m * BNR + j] = acc[m][j];
  __syncwarp();
  for (int t = lane; t < BM_G * BNR; t += 32) {
    float s = 0.f;
#pragma unroll
    for (int l = 0; l < 32; ++l) s += sh[wk][l][t];
    sh[wk][1][t] = s;  // row 1 is free: it held lane 1's partials, now consumed
  }
  __syncthreads();
  for (int t = threadIdx.x; t < BM_G * BNR; t += 32 * WK) {
    const int m = t / BNR, j = t % BNR;
    if (m < mrows) {
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < WK; ++q) s += sh[q][1][t];
      lg[(size_t)(m0 + m) * E + n0 + j] = s;
    }
  }

  // The last CTA of this token-block column routes the column's tokens.
  __shared__ int is_last;
  __threadfence();
  if (threadIdx.x == 0) is_last = (atomicAdd(&ctr[tb], 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!is_last) {
    pdl_trigger();
    return;
  }
  if (threadIdx.x == 0) ctr[tb] = 0;  // leave the counter clean for the next call
  __threadfence();

  __shared__ int   sel[WK][TOPK];
  __shared__ float ssig[WK][TOPK];
  constexpr int PL = E / 32;
  for (int mm = wk; mm < mrows; mm += WK) {
    const int t = m0 + mm;
    const volatile float* row = lg + (size_t)t * E;
    float sb[PL], sg[PL];
    int   id[PL];
#pragma unroll
    for (int j = 0; j < PL; ++j) {
      const int e = lane + 32 * j;
      const float s = sigmoid_accurate(row[e]);
      sg[j] = s;
      sb[j] = s + __bfloat162float(bias[e]);
      id[j] = e;
    }
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      unsigned long long best = 0ull;
#pragma unroll
      for (int j = 0; j < PL; ++j) {
        const unsigned long long p =
            ((unsigned long long)ford(sb[j]) << 32) | (unsigned int)(E - 1 - id[j]);
        best = p > best ? p : best;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
      { const unsigned long long o = __shfl_xor_sync(0xffffffffu, best, off);
        best = o > best ? o : best; }
      const int wid = E - 1 - (int)(unsigned int)(best & 0xffffffffu);
      if (lane == 0) sel[wk][k] = wid;
#pragma unroll
      for (int j = 0; j < PL; ++j)
        if (id[j] == wid) {
          ssig[wk][k] = sg[j];
          sb[j] = -INFINITY;
        }
    }
    __syncwarp();
    if (lane <= TOPK) {
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) sum += ssig[wk][k];
      const int slot = t * (TOPK + 1) + lane;
      if (lane < TOPK) {
        eid[slot] = sel[wk][lane];
        wgt[slot] = ssig[wk][lane] * scale / sum;
      } else {
        eid[slot] = E;
        wgt[slot] = 1.0f;
      }
    }
    __syncwarp();
  }
  pdl_trigger();
}

// Phase 1 only, exposed so the host can prove bitwise equality against the L1
// `gate_linear` output before enabling the fused path.
template <int BM_G, int BNR, int WK, int KPT>
__global__ __launch_bounds__(32 * WK) void gate_only_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ gw, int M, int N,
    float* __restrict__ lg) {
  constexpr int K = WK * KPT * 256;
  const int n0 = blockIdx.x * BNR;
  const int m0 = blockIdx.y * BM_G;
  const int mrows = min(BM_G, M - m0);
  const int lane = threadIdx.x & 31;
  const int wk = threadIdx.x >> 5;
  const int kb = wk * (KPT * 256) + lane * 8;
  pdl_wait();
  const bf16* bp = gw + (size_t)n0 * K + kb;
  const bf16* ap = x + (size_t)m0 * K + kb;
  int aoff[BM_G];
#pragma unroll
  for (int m = 0; m < BM_G; ++m) aoff[m] = (m < mrows ? m : mrows - 1) * K;
  float acc[BM_G][BNR];
#pragma unroll
  for (int m = 0; m < BM_G; ++m)
#pragma unroll
    for (int j = 0; j < BNR; ++j) acc[m][j] = 0.f;
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    uint4 bv[BNR], av[BM_G];
#pragma unroll
    for (int j = 0; j < BNR; ++j)
      bv[j] = *reinterpret_cast<const uint4*>(bp + (size_t)j * K + i * 256);
#pragma unroll
    for (int m = 0; m < BM_G; ++m)
      av[m] = *reinterpret_cast<const uint4*>(ap + aoff[m] + i * 256);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bf[BNR], af[BM_G];
#pragma unroll
      for (int j = 0; j < BNR; ++j) bf[j] = bf2f(bv[j], k);
#pragma unroll
      for (int m = 0; m < BM_G; ++m) af[m] = bf2f(av[m], k);
#pragma unroll
      for (int m = 0; m < BM_G; ++m)
#pragma unroll
        for (int j = 0; j < BNR; ++j) acc[m][j] = fmaf(af[m], bf[j], acc[m][j]);
    }
  }
  __shared__ float sh[WK][32][BM_G * BNR + 1];
#pragma unroll
  for (int m = 0; m < BM_G; ++m)
#pragma unroll
    for (int j = 0; j < BNR; ++j) sh[wk][lane][m * BNR + j] = acc[m][j];
  __syncwarp();
  for (int t = lane; t < BM_G * BNR; t += 32) {
    float s = 0.f;
#pragma unroll
    for (int l = 0; l < 32; ++l) s += sh[wk][l][t];
    sh[wk][1][t] = s;
  }
  __syncthreads();
  for (int t = threadIdx.x; t < BM_G * BNR; t += 32 * WK) {
    const int m = t / BNR, j = t % BNR;
    if (m < mrows) {
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < WK; ++q) s += sh[q][1][t];
      lg[(size_t)(m0 + m) * N + n0 + j] = s;
    }
  }
  pdl_trigger();
}

// ---------------------------------------------------------------------------
// finalize: fp32 accumulator -> bf16 output, and reset the counters so the
// next call needs no separate memset launch.
// ---------------------------------------------------------------------------
template <int E>
__global__ __launch_bounds__(256) void finalize_kernel(
    const float* __restrict__ accf, bf16* __restrict__ out, long n,
    int* __restrict__ cnt, int* __restrict__ meta) {
  pdl_wait();
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(accf[i]);
  if (blockIdx.x == 0) {
    if (threadIdx.x < E) cnt[threadIdx.x] = 0;
    if (threadIdx.x == 0) cnt[E] = 0;
    if (threadIdx.x < 2) meta[threadIdx.x] = 0;
  }
}

// Opt in to the >48KB dynamic shared-memory carveout the pipelines need.
}  // namespace

// One tuned (token-tile, tile shape, pipeline depth) triple per token bucket.
// The shapes are fixed (H=2304, I=1024, 256 experts, top-8), so the only free
// variable is M, and each bucket trades three things: enough CTAs to cover 148
// SMs (which at M=1 is the binding constraint -- only 8 experts are touched),
// enough bytes in flight per SM, and few enough N/H tiles per block that
// re-reading the activation and the intermediate stays a small tax.
// Launch with programmatic stream serialization so the kernel's grid setup
// overlaps its producer's tail (see pdl_wait / pdl_trigger).
template <typename F, typename... Args>
inline void launch_pdl(F fn, dim3 grid, dim3 block, size_t smem, cudaStream_t st,
                       Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.numAttrs = 1;
  cfg.attrs = attrs;
  cudaLaunchKernelEx(&cfg, fn, args...);
}

struct Cfg {
  int bm;      // tokens per block (16*TM)
  int g1_bn, g1_nw, g1_ctas;
  int g2_bh, g2_nw, g2_ctas;
  size_t g1_smem, g2_smem;
  void (*g1)(const bf16*, const bf16*, const bf16*, const int*, const int*,
             const int*, const int*, const int*, bf16*, int, int, int, int);
  void (*g2)(const bf16*, const bf16*, const bf16*, const int*, const int*,
             const int*, const int*, const float*, const int*, float*, int, int,
             int, int);
};

template <int TM, int BN, int NW1, int BK1, int ST1, int BH, int NW2, int BK2,
          int ST2, int GU = 1>
Cfg make_cfg(int I, int H) {
  Cfg c{};
  c.bm = 16 * TM;
  c.g1_bn = BN;
  c.g1_nw = NW1;
  c.g1_ctas = I / BN;
  c.g2_bh = BH;
  c.g2_nw = NW2;
  c.g2_ctas = H / BH;
  c.g1_smem = (size_t)ST1 * (2 * BN + 16 * TM) * (BK1 + 8) * sizeof(bf16);
  // The epilogue reuses the staging buffer; with GU=2 it needs 2*NWG*SUB tiles.
  const size_t epi1 = (size_t)2 * (BN / 16) * 256 * sizeof(float);
  if (epi1 > c.g1_smem) c.g1_smem = epi1;
  c.g2_smem = (size_t)ST2 * (BH + 16 * TM) * (BK2 + 8) * sizeof(bf16);
  c.g1 = gemm1_kernel<TM, BN, NW1, BK1, ST1, GU>;
  c.g2 = gemm2_kernel<TM, BH, NW2, BK2, ST2>;
  cudaFuncSetAttribute((void*)c.g1, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)c.g1_smem);
  cudaFuncSetAttribute((void*)c.g2, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)c.g2_smem);
  return c;
}

constexpr int kNumCfg = 4;

// Token-major tiny-M path (route_tok / gemm1_tok / gemm2_tok). Separate table
// because the kernels take a different argument list: there is no block table,
// no per-expert count and no fp32 accumulator to thread through.
struct TokCfg {
  int g1_r, g1_nwe, g1_ctas;
  int g2_r, g2_nwe, g2_ctas;
  size_t g1_smem;
  void (*g1)(const bf16*, const bf16*, const bf16*, const int*, bf16*, int, int,
             int);
  void (*g2)(const bf16*, const bf16*, const bf16*, const int*, const float*,
             bf16*, int, int, int);
};

// GY caps the grid's row-group dimension: each CTA then strides over
// ceil(rows/GY) groups.  0 means one CTA per row group (no striding).
template <int R1, int UNR1, int R2, int UNR2, int NWE, int GY = 0>
TokCfg make_tok(int I, int H) {
  TokCfg c{};
  c.g1_r = R1;
  c.g1_nwe = NWE;
  c.g1_ctas = (GY && GY < I / R1) ? GY : I / R1;
  c.g2_r = R2;
  c.g2_nwe = NWE;
  c.g2_ctas = (GY && GY < H / R2) ? GY : H / R2;
  c.g1_smem = (size_t)H * sizeof(bf16);
  c.g1 = gemm1_tok_kernel<R1, NWE, UNR1>;
  c.g2 = gemm2_tok_kernel<R2, NWE, UNR2>;
  cudaFuncSetAttribute((void*)c.g1, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)c.g1_smem);
  return c;
}

constexpr int kNumTok = 4;

// Above this the duplicate-expert weight re-read of the token-major path
// (0% at M=1, ~13% at M=8, ~46% at M=26) outweighs its advantages and the
// 16-token wmma tile wins. Measured; see ITERATIONS.md.
constexpr int kTokMaxM = 16;

// Index of the shipped token-major tile in tok_table.
constexpr int kTokBest = 0;

const TokCfg* tok_table(int I, int H) {
  static const TokCfg t[kNumTok] = {
      //        R1 UN1  R2 UN2  NWE
      // 0: the shipped point. R1=1 gives gemm1 twice the CTAs (a warp already
      // has two independent streams from its gate/up row pair), and R2=8 makes
      // a gemm2 warp's weight run 8 contiguous w2 rows = 16KB.
      make_tok<  1,  3,  8,  4,  9>(I, H),
      // 1-3: the nearest measured alternatives, kept for cheap re-probing on a
      // future arch. The full 23-entry sweep is in ITERATIONS.md.
      make_tok<  1,  3,  2,  4,  9>(I, H),
      make_tok<  2,  3,  2,  4,  9>(I, H),
      make_tok<  1,  9,  8,  4,  9>(I, H),
  };
  return t;
}

// Pruned to the measured winners; the wider search (32 tile/pipeline/warp-split
// combinations) is recorded in ITERATIONS.md so it need not be recompiled here.
const Cfg* cfg_table(int I, int H) {
  static const Cfg t[kNumCfg] = {
      //       TM  BN NW1 BK1 ST1   BH NW2 BK2 ST2  GU
      // 0: M=1. Only 8 experts are activated, so gemm1 has 9 blocks and the
      // grid barely covers the SMs; there one fat CTA per SM with a 256B run
      // per weight row beats more, thinner CTAs (55us vs 68us end to end).
      make_cfg<1, 64, 4, 128, 3, 64, 4, 128, 3>(I, H),
      // 1: the general small-M tile. Best from M=2 to M=320.
      make_cfg<1, 64, 4, 64, 3, 64, 4, 64, 3>(I, H),
      // 2: 32-token tiles above M=320, where per-expert counts (mean 13.8, sd
      // 3.7 at M=443) start to overflow a 16-token tile and re-read weights.
      make_cfg<2, 64, 4, 64, 3, 128, 4, 64, 3>(I, H),
      // 3: gate/up split across warp halves (8 warps/CTA at the same shared
      // memory). Kept because it is the one variant that is a genuine tie with
      // config 1 rather than a loss, and it is the cheapest place to re-probe
      // whether a future arch is warp- rather than bandwidth-limited.
      make_cfg<1, 64, 8, 64, 3, 64, 4, 64, 3, 2>(I, H),
  };
  return t;
}

// Bucket -> config, measured (see ITERATIONS.md). Ids >= kNumCfg select the
// token-major tiny path.
int pick_cfg(int M) {
  if (M <= kTokMaxM) return kNumCfg + kTokBest;  // token-major tiny path
  if (M <= 320) return 1;
  return 2;
}

// Upper bound on the block-table length for a given token-tile height, from M
// alone (no device round-trip): at most min(E, TOPK*M) experts are active, each
// needs one tile plus one per extra `bm` tokens, and the shared expert needs
// ceil(M/bm).
int nb_bound(int M, int bm) {
  const int pairs = 8 * M;
  const int act = pairs < 256 ? pairs : 256;
  return act + pairs / bm + (M + bm - 1) / bm + 1;
}

// Block-table capacity: the smallest `bm` in the wmma table gives the most
// blocks. The token-major path needs no block table at all.
int kimi_moe_num_blocks(int M) { return nb_bound(M, 16); }

// Intermediate-buffer capacity in rows of I.  Each config needs
// nb_bound(M, bm) * bm rows; take the max over the table's token-tile heights.
int kimi_moe_hbuf_rows(int M) {
  int best = 9 * M;  // token-major path: (TOPK+1) rows per token
  const int bms[] = {16, 32};
  for (int bm : bms) {
    const int r = nb_bound(M, bm) * bm;
    if (r > best) best = r;
  }
  return best;
}

// `logits` may be an empty tensor, in which case `gate_w` is used to compute the
// router projection here instead -- one fewer Python call on the wmma path, in
// the tiling `kimi_moe_gate_logits` proves bitwise-equal to the L1 kernel's.
void kimi_moe_fused(const at::Tensor& x, const at::Tensor& logits,
                    const at::Tensor& bias, const at::Tensor& w13,
                    const at::Tensor& w2, const at::Tensor& sgu,
                    const at::Tensor& sdn, at::Tensor& out, at::Tensor& cnt,
                    at::Tensor& tlist, at::Tensor& twt, at::Tensor& meta,
                    at::Tensor& blkE, at::Tensor& blkT, at::Tensor& hbuf,
                    at::Tensor& accf, const at::Tensor& gate_w, at::Tensor& lgbuf,
                    double scale, int64_t cfg_id) {
  const int M = (int)x.size(0);
  const int H = (int)x.size(1);
  const int E = (int)bias.size(0);
  const int I = (int)w2.size(2);
  TORCH_CHECK(E == 256, "kimi_moe_fused: expects 256 experts");
  TORCH_CHECK(H % 64 == 0 && I % 256 == 0, "kimi_moe_fused: bad H/I");

  cudaStream_t s = at::cuda::getCurrentCUDAStream();

  const bool fuse_gate = logits.numel() == 0;
  const float* lg = fuse_gate ? lgbuf.data_ptr<float>() : logits.data_ptr<float>();
  const bf16* bp = reinterpret_cast<const bf16*>(bias.data_ptr());
  const bf16* xp = reinterpret_cast<const bf16*>(x.data_ptr());
  const bf16* w13p = reinterpret_cast<const bf16*>(w13.data_ptr());
  const bf16* w2p = reinterpret_cast<const bf16*>(w2.data_ptr());
  const bf16* sgup = reinterpret_cast<const bf16*>(sgu.data_ptr());
  const bf16* sdnp = reinterpret_cast<const bf16*>(sdn.data_ptr());
  bf16* outp = reinterpret_cast<bf16*>(out.data_ptr());
  bf16* hbp = reinterpret_cast<bf16*>(hbuf.data_ptr());
  float* af = accf.data_ptr<float>();
  int* cp = cnt.data_ptr<int>();
  int* tp = tlist.data_ptr<int>();
  float* wp = twt.data_ptr<float>();
  int* mp = meta.data_ptr<int>();
  int* bE = blkE.data_ptr<int>();
  int* bT = blkT.data_ptr<int>();

  const int sel = (cfg_id >= 0 && cfg_id < kNumCfg + kNumTok) ? (int)cfg_id
                                                               : pick_cfg(M);
  if (sel >= kNumCfg) {
    // Token-major path: `tlist` / `twt` hold the (expert, weight) pair list
    // (9 entries per token, statically sized), so no other scratch is touched
    // and neither `cnt` / `meta` nor `accf` participate.
    const TokCfg& tc = tok_table(I, H)[sel - kNumCfg];
    launch_pdl(route_tok_kernel<256, 8>, dim3(M), dim3(32), 0, s, lg, bp,
               (float)scale, tp, wp);
    launch_pdl(tc.g1, dim3(M, tc.g1_ctas), dim3(32 * tc.g1_nwe), tc.g1_smem, s,
               xp, w13p, sgup, tp, hbp, H, I, E);
    launch_pdl(tc.g2, dim3(M, tc.g2_ctas), dim3(32 * tc.g2_nwe), 0, s, hbp, w2p,
               sdnp, tp, wp, outp, H, I, E);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kimi_moe_fused: launch failed");
    return;
  }

  const Cfg* tab = cfg_table(I, H);
  const Cfg& cf = tab[sel];
  const int NB = nb_bound(M, cf.bm);

  if (fuse_gate) {
    TORCH_CHECK(H == 2304 && E == 256, "kimi_moe_fused: gate fusion needs H=2304");
    float* lgp = lgbuf.data_ptr<float>();
    // One expert per warp (256 CTAs): 3.1us at M=1 against 3.8 at two per warp,
    // and the expert tile is free to differ from the L1 kernel's since only
    // (WK, KPT) fix the reduction order.
    if (M <= 4)
      launch_pdl(gate_only_kernel<1, 1, 1, 9>, dim3(256, M), dim3(32), 0, s, xp,
                 reinterpret_cast<const bf16*>(gate_w.data_ptr()), M, E, lgp);
    else
      launch_pdl(gate_only_kernel<4, 1, 3, 3>, dim3(256, (M + 3) / 4), dim3(96), 0,
                 s, xp, reinterpret_cast<const bf16*>(gate_w.data_ptr()), M, E, lgp);
  }
  launch_pdl(route_kernel<256, 8>, dim3(M), dim3(256), 0, s, lg, bp, M, H,
             (float)scale, cf.bm, cp, tp, wp, mp, bE, bT, af);
  launch_pdl(cf.g1, dim3(NB, cf.g1_ctas), dim3(32 * cf.g1_nw), cf.g1_smem, s, xp,
             w13p, sgup, bE, bT, cp, tp, mp, hbp, M, H, I, E);
  launch_pdl(cf.g2, dim3(NB, cf.g2_ctas), dim3(32 * cf.g2_nw), cf.g2_smem, s, hbp,
             w2p, sdnp, bE, bT, cp, tp, wp, mp, af, M, H, I, E);

  const long n = (long)M * H;
  launch_pdl(finalize_kernel<256>, dim3((int)((n + 255) / 256)), dim3(256), 0, s,
             af, outp, n, cp, mp);
  // Cheap (no sync) and catches a bad launch config immediately rather than
  // as silent zeros in the output.
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kimi_moe_fused: launch failed");
}


// Fully fused token-major entry: router projection + routing + both expert
// GEMMs + SwiGLU + the weighted reduction, in three launches and with no
// Python-side kernel call at all.  `eid` / `wgt` hold the (expert, weight) pair
// list, `lg` the router logits and `ctr` the per-token-block arrival counters
// (all reused decode scratch; the wmma path's `cnt` / `meta` / `accf` are
// untouched, and every counter is left at zero for the next call).
void kimi_moe_fused_tok(const at::Tensor& x, const at::Tensor& gate_w,
                        const at::Tensor& bias, const at::Tensor& w13,
                        const at::Tensor& w2, const at::Tensor& sgu,
                        const at::Tensor& sdn, at::Tensor& out, at::Tensor& eid,
                        at::Tensor& wgt, at::Tensor& lg, at::Tensor& ctr,
                        at::Tensor& hbuf, double scale, int64_t cfg_id,
                        int64_t gate_mode) {
  const int M = (int)x.size(0);
  const int H = (int)x.size(1);
  const int E = (int)bias.size(0);
  const int I = (int)w2.size(2);
  TORCH_CHECK(E == 256 && H == 2304, "kimi_moe_fused_tok: expects E=256, H=2304");

  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const bf16* xp = reinterpret_cast<const bf16*>(x.data_ptr());
  const bf16* gwp = reinterpret_cast<const bf16*>(gate_w.data_ptr());
  const bf16* bp = reinterpret_cast<const bf16*>(bias.data_ptr());
  const bf16* w13p = reinterpret_cast<const bf16*>(w13.data_ptr());
  const bf16* w2p = reinterpret_cast<const bf16*>(w2.data_ptr());
  const bf16* sgup = reinterpret_cast<const bf16*>(sgu.data_ptr());
  const bf16* sdnp = reinterpret_cast<const bf16*>(sdn.data_ptr());
  bf16* outp = reinterpret_cast<bf16*>(out.data_ptr());
  int* ep = eid.data_ptr<int>();
  float* wp = wgt.data_ptr<float>();
  float* lp = lg.data_ptr<float>();
  int* cp = ctr.data_ptr<int>();
  bf16* hbp = reinterpret_cast<bf16*>(hbuf.data_ptr());

  int sel = (cfg_id >= kNumCfg && cfg_id < kNumCfg + kNumTok)
                ? (int)cfg_id - kNumCfg
                : pick_cfg(M) - kNumCfg;
  if (sel < 0) sel = kTokBest;  // M past the bucket: use the tuned tile
  const TokCfg& tc = tok_table(I, H)[sel];

  // (WK, KPT) must match what the L1 gate_linear dispatcher picks for this M --
  // that pair *is* the reduction order.  BM_G and BNR only choose which
  // accumulators a CTA owns, so they are free to be retuned here.
  //
  const int ntb = (M <= 4) ? M : (M + 3) / 4;
  if (gate_mode == 0) {
    // Shipped: projection and routing in one kernel, one expert per warp (256
    // CTAs), joined by a per-token-block arrival counter. Measured 6.94us at
    // M=1 against 7.56 at two experts per warp, 8.78 at four, and 7.42 for the
    // two-kernel split -- and it is one launch rather than two.
    if (M <= 4)
      launch_pdl(gate_route_tok_kernel<256, 8, 1, 1, 1, 9>, dim3(256, ntb),
                 dim3(32), 0, s, xp, gwp, bp, M, (float)scale, lp, cp, ep, wp);
    else
      launch_pdl(gate_route_tok_kernel<256, 8, 4, 1, 3, 3>, dim3(256, ntb),
                 dim3(96), 0, s, xp, gwp, bp, M, (float)scale, lp, cp, ep, wp);
  } else {
    // Split again, for re-probing: the fused kernel pays a fence in every CTA
    // and a serial tail in the last one, which on another arch may cost more
    // than a second PDL launch.
    if (M <= 4)
      launch_pdl(gate_only_kernel<1, 1, 1, 9>, dim3(256, ntb), dim3(32), 0, s, xp,
                 gwp, M, E, lp);
    else
      launch_pdl(gate_only_kernel<4, 1, 3, 3>, dim3(256, ntb), dim3(96), 0, s, xp,
                 gwp, M, E, lp);
    launch_pdl(route_tok_kernel<256, 8>, dim3(M), dim3(32), 0, s, lp, bp,
               (float)scale, ep, wp);
  }
  launch_pdl(tc.g1, dim3(M, tc.g1_ctas), dim3(32 * tc.g1_nwe), tc.g1_smem, s, xp,
             w13p, sgup, ep, hbp, H, I, E);
  launch_pdl(tc.g2, dim3(M, tc.g2_ctas), dim3(32 * tc.g2_nwe), 0, s, hbp, w2p,
             sdnp, ep, wp, outp, H, I, E);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kimi_moe_fused_tok: launch failed");
}

// Router projection alone, in the exact tiling the fused path uses.  The host
// gate compares this bitwise against the L1 gate_linear output before turning
// the fused path on, so a build where gate_linear falls back to cuBLAS (or to
// F.linear) can never silently change expert selection.
at::Tensor kimi_moe_gate_logits(const at::Tensor& x, const at::Tensor& gate_w) {
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)gate_w.size(0);
  TORCH_CHECK(K == 2304 && N == 256, "kimi_moe_gate_logits: expects K=2304, N=256");
  auto out = at::empty({x.size(0), gate_w.size(0)}, x.options().dtype(at::kFloat));
  const bf16* xp = reinterpret_cast<const bf16*>(x.data_ptr());
  const bf16* gwp = reinterpret_cast<const bf16*>(gate_w.data_ptr());
  float* lp = out.data_ptr<float>();
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  if (M <= 4)
    gate_only_kernel<1, 2, 1, 9><<<dim3(128, M), dim3(32), 0, s>>>(xp, gwp, M, N, lp);
  else
    gate_only_kernel<4, 8, 3, 3><<<dim3(32, (M + 3) / 4), dim3(96), 0, s>>>(
        xp, gwp, M, N, lp);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kimi_moe_gate_logits: launch failed");
  return out;
}


#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kimi_moe_fused", &kimi_moe_fused, "Fused small-M Kimi MoE");
  m.def("kimi_moe_num_blocks", &kimi_moe_num_blocks, "Block-table upper bound");
  m.def("kimi_moe_hbuf_rows", &kimi_moe_hbuf_rows, "Intermediate-buffer rows");
  m.def("kimi_moe_fused_tok", &kimi_moe_fused_tok, "Fused small-M Kimi MoE, gate included");
  m.def("kimi_moe_gate_logits", &kimi_moe_gate_logits, "Router GEMV in the fused path's tiling");
  m.def("kimi_moe_num_cfg", []() { return (int64_t)kNumCfg; }, "First token-major config id");
  m.def("kimi_moe_tok_max_m", []() { return (int64_t)kTokMaxM; }, "Largest M on the token-major path");
}
