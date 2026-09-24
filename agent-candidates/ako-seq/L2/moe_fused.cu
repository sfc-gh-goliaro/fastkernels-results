// Fused small-token shared-expert MoE for L2/shared_expert_moe.
//
// The benched configuration is hidden=2048, num_experts=512, top_k=10,
// intermediate_per_tp=512 (shared expert too), softmax routing with
// renormalization and a sigmoid-gated shared expert. In that regime every
// active expert sees only ~1-2 tokens, so the whole layer is weight-streaming,
// not GEMM: per expert w13 is 4.19 MB and w2 2.10 MB, and the arithmetic
// intensity is ~1 token per weight element. The reference path spends nearly
// all of its wall clock in host-side dispatch (flashinfer's trtllm wrapper is
// ~740 us of Python per call at M=1 against 42 us of GPU work), so this file
// collapses the entire layer into four kernels behind one pybind entry point:
//
//   1. router_kernel  -- x @ gate_w^T -> bf16 logits (bit-compatible rounding)
//   2. route_kernel   -- per-token top-k + renormalize, per-expert work lists,
//                        the shared expert's sigmoid gate, and zeroing the
//                        fp32 accumulator
//   3. expert_kernel  -- one CTA per (active expert x intermediate chunk),
//                        streaming w13 then w2 exactly once and keeping the
//                        SwiGLU intermediate on chip; the shared expert rides
//                        the same code path as extra grid slots
//   4. cast_kernel    -- fp32 accumulator -> bf16 output
//
// Top-k ties matter: with 512 bf16 logits the rank-10 and rank-11 logits are
// exactly equal for ~10% of tokens, and the reference resolves those toward the
// LOWER expert index. ``topk_key()`` below encodes exactly that.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }

// bf16 -> fp32 straight off a packed bf16x2 register. A bf16 *is* the top half
// of the fp32 with the same value, so each of these is one integer instruction;
// ``__bfloat162float`` applied to an array element instead costs a half-word
// extract plus the shift. In the expert inner loop that is measurably the
// difference between 13.9 and 16.9 TMAC/s (dev/probe_alu.py).
__device__ __forceinline__ float lo2f(unsigned u) { return __int_as_float(u << 16); }
__device__ __forceinline__ float hi2f(unsigned u) {
  return __int_as_float(u & 0xFFFF0000u);
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x * sigmoidf_(x); }

// Top-k selection key.
//
// The reference resolves an exact tie between two router logits toward the
// LOWER expert index, and with 512 bf16 logits ~10% of tokens have such a tie at
// the top-k boundary, so this rule is load-bearing rather than cosmetic. Packing
// the order into one uint32 makes it exact and lets each selection round reduce
// with a single ``__reduce_max_sync`` instead of a five-step shuffle chain: the
// logits are bf16, so 16 bits carry the whole value and the low half is free.
//
// High 16 bits: the bf16 pattern mapped to a monotone unsigned order.
// Low 16 bits: 0xFFFF - index, so a larger index sorts lower and an exact tie
// resolves to the smaller index.
__device__ __forceinline__ unsigned topk_key(bf16 v, int i) {
  const unsigned short b = __bfloat16_as_ushort(v);
  const unsigned short u = (b & 0x8000u) ? (unsigned short)(~(unsigned)b)
                                         : (unsigned short)(b | 0x8000u);
  return ((unsigned)u << 16) | (0xFFFFu - (unsigned)i);
}

__device__ __forceinline__ int key_index(unsigned k) {
  return (int)(0xFFFFu - (k & 0xFFFFu));
}

__device__ __forceinline__ float key_value(unsigned k) {
  const unsigned short u = (unsigned short)(k >> 16);
  const unsigned short b = (u & 0x8000u) ? (unsigned short)(u & 0x7FFFu)
                                         : (unsigned short)(~(unsigned)u);
  return __bfloat162float(__ushort_as_bfloat16(b));
}

// ---------------------------------------------------------------------------
// 1. Router projection: logits[m, e] = sum_c x[m, c] * gate_w[e, c].
//
// Same shape as expert phase 1 -- a reduction over the contiguous dimension of
// both operands -- so it uses the same warp-per-weight-row mapping: each warp
// owns EPW expert rows, holds one row at a time in registers, and sweeps the
// BM staged token rows out of shared memory. The token tile is staged once, not
// per reduction chunk, so there is no repeated block barrier and the row loads
// issue as one burst of eight 16-byte loads per lane.
//
// Rounding matches the reference exactly: an fp32 accumulation rounded once to
// bf16, which is what a cuBLAS bf16 GEMM with bf16 output produces. That
// matters because the routing top-k is decided on these values.
//
// Block (0,0) also clears the routing counters that kernel 2 accumulates into.
// ---------------------------------------------------------------------------
template <int H, int BM, int EPW, int NT>
__global__ __launch_bounds__(NT) void router_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ gw,
    bf16* __restrict__ logits, int M, int E,
    int* __restrict__ cnt, int* __restrict__ n_active) {
  constexpr int NW = NT / 32;
  constexpr int BE = NW * EPW;
  constexpr int CIT = H / 256;
  extern __shared__ char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);        // [BM][H]

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int m0 = blockIdx.y * BM;
  const int e0 = blockIdx.x * BE + warp * EPW;

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    for (int i = tid; i < E; i += NT) cnt[i] = 0;
    if (tid == 0) *n_active = 0;
  }

  // Rows past M clamp to the last token so the tile always holds finite data;
  // their logits are simply not stored.
  for (int i = tid; i < BM * (H / 8); i += NT) {
    const int r = i / (H / 8), c = (i % (H / 8)) * 8;
    const int m = min(m0 + r, M - 1);
    *reinterpret_cast<uint4*>(xs + r * H + c) =
        *reinterpret_cast<const uint4*>(x + (size_t)m * H + c);
  }
  __syncthreads();

  float acc[EPW][BM];
#pragma unroll
  for (int p = 0; p < EPW; ++p)
#pragma unroll
    for (int t = 0; t < BM; ++t) acc[p][t] = 0.f;

#pragma unroll
  for (int p = 0; p < EPW; ++p) {
    const int e = e0 + p;
    if (e >= E) break;
    const bf16* wr = gw + (size_t)e * H + lane * 8;
#pragma unroll 4
    for (int it = 0; it < CIT; ++it) {
      const uint4 w4 = *reinterpret_cast<const uint4*>(wr + it * 256);
      const bf16* wv = reinterpret_cast<const bf16*>(&w4);
#pragma unroll
      for (int t = 0; t < BM; ++t) {
        const uint4 x4 =
            *reinterpret_cast<const uint4*>(xs + t * H + lane * 8 + it * 256);
        const bf16* xv = reinterpret_cast<const bf16*>(&x4);
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[p][t] = fmaf(b2f(wv[q]), b2f(xv[q]), acc[p][t]);
      }
    }
  }

#pragma unroll
  for (int p = 0; p < EPW; ++p) {
    const int e = e0 + p;
    if (e >= E) break;
#pragma unroll
    for (int t = 0; t < BM; ++t) {
      float v = acc[p][t];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      if (lane == 0 && m0 + t < M)
        logits[(size_t)(m0 + t) * E + e] = __float2bfloat16(v);
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Routing + shared-expert gate + accumulator init. One CTA per token.
//
// Reads the token's E bf16 logits, takes the softmax in fp32, selects the top-k
// with the lower-index tie-break, renormalizes, and pushes (token, weight) onto
// each selected expert's list. The first push for an expert also appends it to
// a compacted active-expert list, which bounds kernel 3's grid without a host
// sync. Also folds in the shared expert's sigmoid gate projection and clears
// this token's row of the fp32 accumulator.
// ---------------------------------------------------------------------------
template <int NT, int KMAX>
__global__ __launch_bounds__(NT) void route_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ logits,
    const bf16* __restrict__ sh_gate_w, int M, int E, int H, int K, int Tmax,
    int* __restrict__ cnt, int* __restrict__ tok, float* __restrict__ twt,
    int* __restrict__ n_active, int* __restrict__ active,
    float* __restrict__ gate_scale, float* __restrict__ accum) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  extern __shared__ char smem[];
  unsigned* key = reinterpret_cast<unsigned*>(smem);   // [E] packed order keys
  __shared__ float rv[NT / 32];
  __shared__ unsigned sel[KMAX];
  __shared__ float w_s[KMAX];

  // --- everything independent of the selection, in parallel -----------------
  // The expert kernel accumulates into ``accum`` with fp32 atomics, so this is
  // where it gets cleared.
  {
    float4* a4 = reinterpret_cast<float4*>(accum + (size_t)t * H);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = tid; i < H / 4; i += NT) a4[i] = z;
  }
  const bf16* lg = logits + (size_t)t * E;
  for (int i = tid; i < E; i += NT) key[i] = topk_key(lg[i], i);
  // Shared-expert gate projection, accumulated in fp32.
  float g = 0.f;
  for (int c = tid * 8; c < H; c += NT * 8) {
    const uint4 xv = *reinterpret_cast<const uint4*>(x + (size_t)t * H + c);
    const uint4 wv = *reinterpret_cast<const uint4*>(sh_gate_w + c);
    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
    const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
    for (int q = 0; q < 8; ++q) g = fmaf(b2f(xp[q]), b2f(wp[q]), g);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) g += __shfl_down_sync(0xffffffffu, g, o);
  if (lane == 0) rv[warp] = g;
  __syncthreads();

  // --- top-k, entirely inside warp 0 ---------------------------------------
  // K rounds of a block-wide argmax would be K pairs of block barriers, which
  // at M=1 (one CTA, nothing to overlap with) is most of this kernel's cost.
  // One warp needs no barriers, and with the packed key each round is a strided
  // scan plus one warp-reduce instruction.
  if (warp == 0) {
    for (int k = 0; k < K; ++k) {
      unsigned best = 0;
      for (int i = lane; i < E; i += 32) best = max(best, key[i]);
      best = __reduce_max_sync(0xffffffffu, best);
      if (lane == 0) {
        sel[k] = best;
        key[key_index(best)] = 0;
      }
      __syncwarp();
    }
    // sel[0] is the max, so exp(l - lmax) <= 1. Dividing by the sum over the
    // selected set is exactly the reference's softmax-then-renormalize: the
    // global softmax denominator cancels against the renormalization.
    if (lane == 0) {
      const float lmax = key_value(sel[0]);
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        w_s[k] = __expf(key_value(sel[k]) - lmax);
        sum += w_s[k];
      }
      const float inv = 1.f / (sum + 1e-20f);
      for (int k = 0; k < K; ++k) w_s[k] *= inv;
    }
    __syncwarp();
    // Publish the work items. The first push for an expert also appends it to
    // the compacted active list, which bounds the expert kernel's grid with no
    // host sync.
    if (lane < K) {
      const int e = key_index(sel[lane]);
      const int slot = atomicAdd(&cnt[e], 1);
      if (slot < Tmax) {
        tok[(size_t)e * Tmax + slot] = t;
        twt[(size_t)e * Tmax + slot] = w_s[lane];
      }
      if (slot == 0) active[atomicAdd(n_active, 1)] = e;
    }
  } else if (warp == 1 && lane == 0) {
    float sum = 0.f;
    for (int w = 0; w < NT / 32; ++w) sum += rv[w];
    gate_scale[t] = sigmoidf_(sum);
  }
}

// ---------------------------------------------------------------------------
// 3. Expert compute. Grid is (slot, chunk).
//
// Slots [0, nxr) are routed experts, indexed through the compacted active list;
// slots [nxr, nxr + nsh) are the shared expert, one per group of TG tokens.
// Splitting the shared expert over token groups matters: with one slot it
// becomes a single CTA streaming its 6.3 MB once per token group, which at
// M=445 is 176 MB of serial traffic in one CTA and dominates everything else.
// Its re-reads across groups are L2 hits (the whole shared expert is 6.3 MB),
// so the extra traffic is a few percent, not a few hundred percent.
//
// The chunk dimension splits the intermediate dim, which is what supplies
// parallelism when only a handful of experts are active: at M=1 ten experts
// would otherwise occupy ten SMs out of 148.
//
// Phase 1 streams the w13 gate/up row pair for each intermediate column and
// reduces against the staged token rows, leaving the SwiGLU result in shared
// memory. Phase 2 streams w2^T (stored [N, H], so an intermediate-column chunk
// is a contiguous slab) and accumulates into the fp32 output with atomics.
// Neither expert activation ever reaches HBM.
// ---------------------------------------------------------------------------

// One token group, with the group size TN fixed at compile time.
//
// TN being compile-time is worth 1.66x, not a rounding error. With a runtime
// token count the inner loop needs a data-dependent `break`, and ptxas will not
// hoist the next iteration's weight loads across it -- measured 28.6 GB/s per SM
// versus 47.5 GB/s (88% of HBM peak) once the trip count is constant. Groups
// are therefore padded up to a power of two, the padding aliasing token 0 with
// weight 0 so it stages real (finite) data and contributes nothing.
template <int H, int NN, int TN, int TG, int NT, int JU>
__device__ __forceinline__ void expert_group(
    const bf16* __restrict__ WA, const bf16* __restrict__ WB, const bf16* xs,
    float* av, const int* tid_s, const float* tsc_s, float* __restrict__ accum,
    int Nc, int j0) {
  constexpr int VEC = H / NT;      // output columns per thread in phase 2
  constexpr int NW = NT / 32;
  constexpr int CIT = H / 256;     // phase-1 c steps (32 lanes x 8 bf16)
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;

  // ---- phase 1: g = w1[j].x, u = w3[j].x, a[j] = silu(g) * u ---------------
  for (int j = j0 + warp; j < j0 + Nc; j += NW) {
    const bf16* rg = WA + (size_t)j * H + lane * 8;
    const bf16* ru = WA + (size_t)(NN + j) * H + lane * 8;
    float ag[TN], au[TN];
#pragma unroll
    for (int t = 0; t < TN; ++t) { ag[t] = 0.f; au[t] = 0.f; }
    // Unrolled by 4, not fully: holding all CIT load pairs live costs 64
    // registers and drops the kernel to one CTA per SM, which measured slower
    // than the extra loads in flight are worth.
#pragma unroll 4
    for (int it = 0; it < CIT; ++it) {
      const uint4 g4 = *reinterpret_cast<const uint4*>(rg + it * 256);
      const uint4 u4 = *reinterpret_cast<const uint4*>(ru + it * 256);
      const unsigned* gu = reinterpret_cast<const unsigned*>(&g4);
      const unsigned* uu = reinterpret_cast<const unsigned*>(&u4);
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        const uint4 x4 =
            *reinterpret_cast<const uint4*>(xs + t * H + lane * 8 + it * 256);
        const unsigned* xu = reinterpret_cast<const unsigned*>(&x4);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float xl = lo2f(xu[q]), xh = hi2f(xu[q]);
          ag[t] = fmaf(lo2f(gu[q]), xl, ag[t]);
          ag[t] = fmaf(hi2f(gu[q]), xh, ag[t]);
          au[t] = fmaf(lo2f(uu[q]), xl, au[t]);
          au[t] = fmaf(hi2f(uu[q]), xh, au[t]);
        }
      }
    }
#pragma unroll
    for (int t = 0; t < TN; ++t) {
      float g = ag[t], u = au[t];
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, o);
        u += __shfl_down_sync(0xffffffffu, u, o);
      }
      if (lane == 0) av[(size_t)(j - j0) * TG + t] = siluf_(g) * u;
    }
  }
  __syncthreads();

  // ---- phase 2: out += scale * w2[:, chunk] . a[chunk] ---------------------
  float oacc[VEC][TN];
#pragma unroll
  for (int q = 0; q < VEC; ++q)
#pragma unroll
    for (int t = 0; t < TN; ++t) oacc[q][t] = 0.f;
  const int hb = tid * VEC;
  const bf16* wbase = WB + (size_t)j0 * H + hb;
  for (int j = 0; j < Nc; j += JU) {
    uint4 wraw[JU];
#pragma unroll
    for (int u = 0; u < JU; ++u) {
      if (VEC == 8) {
        wraw[u] = *reinterpret_cast<const uint4*>(wbase + (size_t)(j + u) * H);
      } else {
        const uint2 v = *reinterpret_cast<const uint2*>(wbase + (size_t)(j + u) * H);
        wraw[u].x = v.x;
        wraw[u].y = v.y;
      }
    }
#pragma unroll
    for (int u = 0; u < JU; ++u) {
      const unsigned* wp = reinterpret_cast<const unsigned*>(&wraw[u]);
      float wv[VEC];
#pragma unroll
      for (int q = 0; q < VEC; q += 2) {
        wv[q] = lo2f(wp[q / 2]);
        wv[q + 1] = hi2f(wp[q / 2]);
      }
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        const float a = av[(size_t)(j + u) * TG + t];
#pragma unroll
        for (int q = 0; q < VEC; ++q) oacc[q][t] = fmaf(wv[q], a, oacc[q][t]);
      }
    }
  }
  // Accumulate in fp32, not into the bf16 output directly.
  //
  // Packed bf16x2 atomics would save a launch and half the atomic bytes, and on
  // average they look fine -- but an element receives (top_k + 1) * chunks
  // contributions, which is 176 at M=1, and rounding a running sum to bf16 that
  // many times put 1.2% of a single token's elements outside tolerance on a
  // tested seed (matched 0.98779 against a 0.99 bar). At M=1 one token is the
  // entire budget, so the extra launch is the cheaper side of the trade.
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const float sc = tsc_s[t];
    if (sc == 0.f) continue;      // padding slot, or a genuinely zero weight
    float* dst = accum + (size_t)tid_s[t] * H + hb;
#pragma unroll
    for (int q = 0; q < VEC; ++q) atomicAdd(dst + q, sc * oacc[q][t]);
  }
}

template <int H, int NN, int TG, int NT, int MINB, int JU>
__global__ __launch_bounds__(NT, MINB) void expert_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ w13,
    const bf16* __restrict__ w2t, const bf16* __restrict__ sh_gu,
    const bf16* __restrict__ sh_dnt, const int* __restrict__ cnt,
    const int* __restrict__ tok, const float* __restrict__ twt,
    const int* __restrict__ n_active, const int* __restrict__ active,
    const float* __restrict__ gate_scale, float* __restrict__ accum,
    int M, int Tmax, int Nc, int nxr) {
  constexpr int VEC = H / NT;
  static_assert(H % NT == 0 && (VEC == 4 || VEC == 8), "bad VEC");

  extern __shared__ char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);              // [TG][H]
  float* av = reinterpret_cast<float*>(xs + TG * H);     // [NN][TG]
  __shared__ int tid_s[TG];
  __shared__ float tsc_s[TG];

  static_assert(TG <= 16, "GROUP dispatch covers group widths up to 16");
  const int xi = blockIdx.x, ci = blockIdx.y;
  const bool is_shared = (xi >= nxr);
  const bf16* WA;
  const bf16* WB;
  int t_begin, t_end;
  const int* tlist = nullptr;
  const float* wlist = nullptr;
  if (is_shared) {
    WA = sh_gu;
    WB = sh_dnt;
    t_begin = (xi - nxr) * TG;
    t_end = min(M, t_begin + TG);
  } else {
    if (xi >= *n_active) return;
    const int e = active[xi];
    int tcnt = cnt[e];
    if (tcnt > Tmax) tcnt = Tmax;
    WA = w13 + (size_t)e * 2 * NN * H;
    WB = w2t + (size_t)e * NN * H;
    tlist = tok + (size_t)e * Tmax;
    wlist = twt + (size_t)e * Tmax;
    t_begin = 0;
    t_end = tcnt;
  }
  // Chunk width rather than chunk count: the count is chosen for load balance
  // and need not divide NN, so the last chunk can be short. NN and Nc are both
  // multiples of 8, so phase 2's unroll-by-8 stays exact.
  const int j0 = ci * Nc;
  if (j0 >= NN) return;
  const int ncur = min(Nc, NN - j0);
  const int tid = threadIdx.x;

  for (int tb = t_begin; tb < t_end; tb += TG) {
    const int tn = min(TG, t_end - tb);
    if (tid < TG) {
      int t = 0;
      float sc = 0.f;
      if (tid < tn) {
        t = is_shared ? (tb + tid) : tlist[tb + tid];
        sc = is_shared ? gate_scale[t] : wlist[tb + tid];
      }
      tid_s[tid] = t;
      tsc_s[tid] = sc;
    }
    __syncthreads();
    for (int i = tid; i < tn * (H / 8); i += NT) {
      const int r = i / (H / 8), c = (i % (H / 8)) * 8;
      *reinterpret_cast<uint4*>(xs + r * H + c) =
          *reinterpret_cast<const uint4*>(x + (size_t)tid_s[r] * H + c);
    }
    __syncthreads();

    // Dispatch on the *exact* group size. Rounding up to a power of two would
    // pad every expert's last group, and at M=445 (mean 8.7 tokens per expert)
    // padding is the dominant cost: a group width of 16 with rounding spends
    // 1.8x the arithmetic it needs, while a width of 4 re-reads each expert's
    // 6.3 MB nearly three times. Exact counts pay neither.
#define GROUP(TN)                                                              \
  expert_group<H, NN, ((TN) <= TG ? (TN) : TG), TG, NT, JU>(                   \
      WA, WB, xs, av, tid_s, tsc_s, accum, ncur, j0)
    switch (tn) {
      case 1: GROUP(1); break;
      case 2: GROUP(2); break;
      case 3: GROUP(3); break;
      case 4: GROUP(4); break;
      case 5: GROUP(5); break;
      case 6: GROUP(6); break;
      case 7: GROUP(7); break;
      case 8: GROUP(8); break;
      case 9: case 10: GROUP(10); break;
      case 11: case 12: GROUP(12); break;
      case 13: case 14: GROUP(14); break;
      default: GROUP(TG); break;
    }
#undef GROUP
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// 4. fp32 accumulator -> bf16 output.
// ---------------------------------------------------------------------------
__global__ void cast_kernel(const float* __restrict__ src, bf16* __restrict__ dst,
                            long n) {
  const long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) {
    const float4 v = *reinterpret_cast<const float4*>(src + i);
    const bf16 o[4] = {__float2bfloat16(v.x), __float2bfloat16(v.y),
                       __float2bfloat16(v.z), __float2bfloat16(v.w)};
    *reinterpret_cast<uint2*>(dst + i) = *reinterpret_cast<const uint2*>(o);
  } else {
    for (long k = i; k < n; ++k) dst[k] = __float2bfloat16(src[k]);
  }
}

// ---------------------------------------------------------------------------
// 5. Fused MoE epilogue for the large-token path.
//
// trtllm-gen can hand back its *unfinalized* GEMM-2 output -- one row per
// (token, selected expert) in permuted order -- plus the renormalized expert
// weights and the expanded-index -> permuted-row map. Finalizing it is a
// weighted sum of top_k rows per token, and the layer then needs two more passes
// over the same [M, H] rows: the shared expert's gate projection, and
// ``routed + shared * sigmoid(gate)``.
//
// Run as the reference does that is three kernels and 3.6 passes over the
// activation. At M=16384 they measure 0.167 + 0.019 + 0.031 = 0.217 ms, and
// trtllm's own finalize moves 738 MB at 4.4 TB/s where this machine reaches
// ~6.9 TB/s. One kernel does all of it in a single pass: 872 MB, ~0.126 ms.
//
// One CTA per token, NT threads x VEC columns covering H exactly, so the gate
// dot is computed once per token rather than once per output tile.
// ---------------------------------------------------------------------------
template <int H, int NT, int KT>
__global__ __launch_bounds__(NT) void finalize_gated_kernel(
    const bf16* __restrict__ g2,      // [P, H]   unfinalized GEMM-2 output
    const int* __restrict__ idx,      // [M * K]  expanded idx -> permuted row
    const bf16* __restrict__ ew,      // [M, K]   renormalized expert weights
    const bf16* __restrict__ shared,  // [M, H]   shared expert, un-gated
    const bf16* __restrict__ xin,     // [M, H]   for the gate projection
    const bf16* __restrict__ gw,      // [H]      shared-expert gate weight
    bf16* __restrict__ out,           // [M, H]
    int K, int P) {
  constexpr int VEC = H / NT;
  constexpr int NW = NT / 32;
  static_assert(VEC == 8, "finalize expects H == NT * 8");
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;

  __shared__ float red[NW];
  __shared__ int is_[32];
  __shared__ float ws_[32];
  if (tid < K) {
    is_[tid] = idx[(size_t)t * K + tid];
    ws_[tid] = b2f(ew[(size_t)t * K + tid]);
  }

  // Gate projection, fp32 accumulation -- same as the reference epilogue, which
  // also keeps this dot in fp32 rather than rounding it through bf16.
  const int c = tid * VEC;
  const uint4 xv = *reinterpret_cast<const uint4*>(xin + (size_t)t * H + c);
  const uint4 wv = *reinterpret_cast<const uint4*>(gw + c);
  const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
  const unsigned* wu = reinterpret_cast<const unsigned*>(&wv);
  float g = 0.f;
#pragma unroll
  for (int q = 0; q < VEC / 2; ++q) {
    g = fmaf(lo2f(xu[q]), lo2f(wu[q]), g);
    g = fmaf(hi2f(xu[q]), hi2f(wu[q]), g);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) g += __shfl_down_sync(0xffffffffu, g, o);
  if (lane == 0) red[warp] = g;
  __syncthreads();
  float gsum = 0.f;
#pragma unroll
  for (int w = 0; w < NW; ++w) gsum += red[w];
  const float scale = sigmoidf_(gsum);

  float acc[VEC];
#pragma unroll
  for (int q = 0; q < VEC; ++q) acc[q] = 0.f;
  // All K row addresses are independent, so a compile-time trip count lets
  // ptxas keep every gather in flight at once; KT == 0 is the generic path.
  const int kn = (KT > 0) ? KT : K;
#pragma unroll
  for (int k = 0; k < kn; ++k) {
    if (KT == 0 && k >= K) break;
    const int r = is_[k];
    if (r < 0 || r >= P) continue;
    const float w = ws_[k];
    const uint4 v = *reinterpret_cast<const uint4*>(g2 + (size_t)r * H + c);
    const unsigned* vu = reinterpret_cast<const unsigned*>(&v);
#pragma unroll
    for (int q = 0; q < VEC / 2; ++q) {
      acc[2 * q] = fmaf(w, lo2f(vu[q]), acc[2 * q]);
      acc[2 * q + 1] = fmaf(w, hi2f(vu[q]), acc[2 * q + 1]);
    }
  }

  const uint4 sv = *reinterpret_cast<const uint4*>(shared + (size_t)t * H + c);
  const unsigned* su = reinterpret_cast<const unsigned*>(&sv);
  bf16 o[VEC];
#pragma unroll
  for (int q = 0; q < VEC / 2; ++q) {
    o[2 * q] = __float2bfloat16(fmaf(scale, lo2f(su[q]), acc[2 * q]));
    o[2 * q + 1] = __float2bfloat16(fmaf(scale, hi2f(su[q]), acc[2 * q + 1]));
  }
  *reinterpret_cast<uint4*>(out + (size_t)t * H + c) =
      *reinterpret_cast<const uint4*>(o);
}

struct ExpertArgs {
  const bf16* x;
  const bf16* w13;
  const bf16* w2t;
  const bf16* sh_gu;
  const bf16* sh_dnt;
  const int* cnt;
  const int* tok;
  const float* twt;
  const int* n_active;
  const int* active;
  const float* gate_scale;
  float* accum;
  int M;
  int Tmax;
  int K;
  int E;
  int N;
  int s_ovr;
  cudaStream_t st;
};

// Which (TG, NT, MINB, JU) shape to run. Tuned; ``ovr`` >= 0 forces one.
int cfg_of(int M, long ovr) {
  if (ovr >= 0) return (int)ovr;
  (void)M;
  // A narrow group (4) wins at every measured token count. A wide one amortizes
  // the weight stream over more tokens, but 16 tokens of staged activations plus
  // their SwiGLU intermediate is 96 KB of shared memory, which halves occupancy
  // to one CTA per SM -- and the bandwidth that costs outweighs the re-reads it
  // saves, even at M=445 where each expert holds ~9 tokens.
  return 1;
}

template <int TG, int NT, int MINB, int JU>
void launch_experts(const ExpertArgs& a) {
  const int nxr = (int)std::min<long>(a.E, (long)a.M * a.K);
  const int nsh = (a.M + TG - 1) / TG;
  const int nx = nxr + nsh;
  // Chunk width over the intermediate dim. Two effects pull in opposite
  // directions and both were measured (dev/sweep.py):
  //
  //  * Wide chunks (few, fat blocks) lose to load imbalance. Blocks cost the
  //    same but experts hold different token counts and over-provisioned slots
  //    exit immediately, so a grid close to the machine size leaves a long tail:
  //    at M=26 one chunk of 512 costs 422 us against 251 us at 48.
  //  * Narrow chunks pay a per-chunk atomic pass and token-tile staging, and at
  //    M=1 a width of 8 costs 43 us against 35 us at 32-48.
  //
  // Widths are multiples of 8 so phase 2's unroll-by-8 stays exact; the width
  // need not divide N, so the last chunk may be short.
  int Nc;
  if (a.s_ovr > 0) {
    Nc = (int)a.s_ovr;
  } else {
    constexpr int NSM = 148;   // SMs on a B200
    Nc = a.M <= 160 ? 48 : 96;
    // With only a handful of active experts even that is too coarse to fill the
    // machine, so narrow it until the grid covers every SM.
    while (Nc > 16 && (long)nx * ((a.N + Nc - 1) / Nc) < NSM) Nc -= 8;
  }
  const int S = (a.N + Nc - 1) / Nc;
  const size_t sm = (size_t)TG * 2048 * sizeof(bf16) + (size_t)512 * TG * 4;
  auto fn = expert_kernel<2048, 512, TG, NT, MINB, JU>;
  static bool set_once = false;
  if (!set_once) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) ==
                    cudaSuccess,
                "cannot opt in to ", sm, " bytes of dynamic shared memory");
    set_once = true;
  }
  fn<<<dim3(nx, S), NT, sm, a.st>>>(a.x, a.w13, a.w2t, a.sh_gu, a.sh_dnt, a.cnt,
                                    a.tok, a.twt, a.n_active, a.active,
                                    a.gate_scale, a.accum, a.M, a.Tmax, Nc, nxr);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "expert launch failed: ", cudaGetErrorString(e),
              " TG=", TG, " NT=", NT, " Nc=", Nc, " nx=", nx, " smem=", sm);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry point. One pybind call, four launches, no host sync.
// ---------------------------------------------------------------------------
void shared_expert_moe_fused(
    at::Tensor x,          // [M, H]      bf16
    at::Tensor gate_w,     // [E, H]      bf16
    at::Tensor w13,        // [E, 2N, H]  bf16
    at::Tensor w2t,        // [E, N, H]   bf16
    at::Tensor sh_gu,      // [2N, H]     bf16
    at::Tensor sh_dnt,     // [N, H]      bf16
    at::Tensor sh_gate_w,  // [H]         bf16
    at::Tensor out,        // [M, H]      bf16
    at::Tensor ws,         // workspace, uint8
    int64_t top_k,
    int64_t s_ovr,         // 0 = auto: intermediate-chunk width
    int64_t cfg_ovr) {     // <0 = auto: expert-kernel (TG, NT, MINB, JU) shape
  const int M = (int)x.size(0);
  const int H = (int)x.size(1);
  const int E = (int)gate_w.size(0);
  const int N = (int)w2t.size(1);
  const int K = (int)top_k;
  TORCH_CHECK(H == 2048 && N == 512, "fused path expects H=2048, N=512");
  TORCH_CHECK(K <= 32, "fused path expects top_k <= 32");

  cudaStream_t st = at::cuda::getCurrentCUDAStream();

  const bf16* xp = (const bf16*)x.data_ptr();
  const int Tmax = M;

  // Carve the workspace.
  char* wsp = (char*)ws.data_ptr();
  size_t off = 0;
  auto take = [&](size_t bytes) {
    char* p = wsp + off;
    off += (bytes + 255) & ~(size_t)255;
    return p;
  };
  bf16* logits = (bf16*)take((size_t)M * E * 2);
  int* cnt = (int*)take((size_t)E * 4);
  int* n_active = (int*)take(4);
  int* active = (int*)take((size_t)E * 4);
  int* tok = (int*)take((size_t)E * Tmax * 4);
  float* twt = (float*)take((size_t)E * Tmax * 4);
  float* gate_scale = (float*)take((size_t)M * 4);
  float* accum = (float*)take((size_t)M * H * 4);
  TORCH_CHECK(off <= (size_t)ws.numel(), "workspace too small");

  // --- 1. router ---
  {
    constexpr int NT = 256;
#define LAUNCH_ROUTER(BM, EPW)                                                    \
  do {                                                                           \
    constexpr int BE = (NT / 32) * (EPW);                                         \
    const size_t sm = (size_t)(BM) * 2048 * sizeof(bf16);                         \
    auto fn = router_kernel<2048, BM, EPW, NT>;                                   \
    static bool set_once = false;                                                 \
    if (!set_once) {                                                              \
      cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                           (int)sm);                                             \
      set_once = true;                                                            \
    }                                                                             \
    fn<<<dim3((E + BE - 1) / BE, (M + (BM) - 1) / (BM)), NT, sm, st>>>(           \
        xp, (const bf16*)gate_w.data_ptr(), logits, M, E, cnt, n_active);         \
  } while (0)
    // The token tile is padded to BM, so an oversized BM both stages and
    // multiplies duplicate rows: at M=1, BM=16 is 16x the arithmetic and 16x
    // the staging of BM=1. One expert row per warp keeps enough CTAs busy when
    // there is only one token block; two amortizes the tile's shared reads.
    if (M <= 2) {
      LAUNCH_ROUTER(1, 1);
    } else if (M <= 8) {
      LAUNCH_ROUTER(4, 1);
    } else if (M <= 32) {
      LAUNCH_ROUTER(16, 1);
    } else {
      LAUNCH_ROUTER(16, 2);
    }
#undef LAUNCH_ROUTER
  }

  // --- 2. routing + gate + accumulator init ---
  {
    constexpr int NT = 256;
    const size_t sm = (size_t)E * sizeof(float);
    route_kernel<NT, 32><<<M, NT, sm, st>>>(
        xp, logits, (const bf16*)sh_gate_w.data_ptr(), M, E, H, K, Tmax,
        cnt, tok, twt, n_active, active, gate_scale, accum);
  }

  // --- 3. experts (routed + shared) ---
  {
    const ExpertArgs ea{xp,
                        (const bf16*)w13.data_ptr(),
                        (const bf16*)w2t.data_ptr(),
                        (const bf16*)sh_gu.data_ptr(),
                        (const bf16*)sh_dnt.data_ptr(),
                        cnt,
                        tok,
                        twt,
                        n_active,
                        active,
                        gate_scale,
                        accum,
                        M,
                        Tmax,
                        K,
                        E,
                        N,
                        (int)s_ovr,
                        st};
    switch (cfg_of(M, cfg_ovr)) {
      case 0: launch_experts<4, 512, 1, 8>(ea); break;
      case 1: launch_experts<4, 512, 2, 8>(ea); break;
      case 2: launch_experts<8, 512, 1, 8>(ea); break;
      case 3: launch_experts<16, 512, 1, 4>(ea); break;
      // NT=256 halves the thread count, which makes VEC 8 instead of 4 -- so
      // phase 2 loads w2 with 16-byte `uint4`s per lane rather than 8-byte
      // `uint2`s, and more CTAs fit per SM. Round 1 swept (group width x chunk
      // width) but held NT at 512 throughout, so this axis is new.
      case 4: launch_experts<4, 256, 2, 8>(ea); break;
      case 5: launch_experts<4, 256, 4, 8>(ea); break;
      case 6: launch_experts<8, 256, 2, 8>(ea); break;
      default: launch_experts<4, 256, 1, 8>(ea); break;
    }
  }

  // --- 4. cast ---
  {
    const long n = (long)M * H;
    constexpr int NT = 256;
    const long blocks = (n / 4 + NT - 1) / NT;
    cast_kernel<<<(int)blocks, NT, 0, st>>>(accum, (bf16*)out.data_ptr(), n);
  }

  // A launch that fails configuration validation is silent and looks exactly
  // like a very fast kernel, so check before returning.
  {
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused MoE launch failed: ",
                cudaGetErrorString(err), " (M=", M, " K=", K, " S_ovr=", s_ovr,
                " cfg=", cfg_ovr, ")");
  }
}

// ---------------------------------------------------------------------------
// Host entry point for the fused epilogue. One launch replaces trtllm-gen's
// finalize, the shared-expert gate gemv and the gated add.
// ---------------------------------------------------------------------------
void moe_finalize_gated(
    at::Tensor g2,        // [P, H]  bf16, trtllm-gen unfinalized GEMM-2 output
    at::Tensor idx,       // [M*K]   int32
    at::Tensor ew,        // [M, K]  bf16
    at::Tensor shared,    // [M, H]  bf16
    at::Tensor x,         // [M, H]  bf16
    at::Tensor gate_w,    // [H]     bf16
    at::Tensor out) {     // [M, H]  bf16
  const int M = (int)out.size(0);
  const int H = (int)out.size(1);
  const int P = (int)g2.size(0);
  const int K = (int)(idx.numel() / std::max(M, 1));
  TORCH_CHECK(H == 2048, "fused epilogue expects H=2048, got ", H);
  TORCH_CHECK(K >= 1 && K <= 32, "fused epilogue expects 1 <= top_k <= 32");
  TORCH_CHECK(g2.size(1) == H && shared.size(0) == M && shared.size(1) == H,
              "fused epilogue shape mismatch");
  if (M == 0) return;
  constexpr int NT = 256;
  auto st = at::cuda::getCurrentCUDAStream();
  const bf16* g2p = (const bf16*)g2.data_ptr();
  const int* ip = (const int*)idx.data_ptr();
  const bf16* ewp = (const bf16*)ew.data_ptr();
  const bf16* shp = (const bf16*)shared.data_ptr();
  const bf16* xp = (const bf16*)x.data_ptr();
  const bf16* gwp = (const bf16*)gate_w.data_ptr();
  bf16* op = (bf16*)out.data_ptr();
#define FIN(KT)                                                                \
  finalize_gated_kernel<2048, NT, KT>                                          \
      <<<M, NT, 0, st>>>(g2p, ip, ewp, shp, xp, gwp, op, K, P)
  switch (K) {
    case 8: FIN(8); break;
    case 10: FIN(10); break;
    case 12: FIN(12); break;
    case 16: FIN(16); break;
    default: FIN(0); break;
  }
#undef FIN
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fused epilogue launch failed: ",
              cudaGetErrorString(err), " (M=", M, " K=", K, ")");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_expert_moe_fused", &shared_expert_moe_fused,
        "Fused shared-expert MoE for small token counts");
  m.def("moe_finalize_gated", &moe_finalize_gated,
        "Weighted top-k row reduction + shared-expert sigmoid gate + add");
}
