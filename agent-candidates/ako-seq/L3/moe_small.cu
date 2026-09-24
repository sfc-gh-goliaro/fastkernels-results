// L3-owned routed-expert MoE for the small-token regime.
//
// The benched configuration is hidden=2048, num_experts=512, top_k=10,
// intermediate=512 per expert (and for the shared expert), softmax routing with
// renormalization, sigmoid-gated shared expert. At M=1 the layer selects 10 of
// 512 experts, so it reads 11 x 6.29 MB = 69.2 MB of weights for 34.6 M MACs --
// pure weight streaming, arithmetic intensity ~0.5 MAC per byte.
//
// WHAT THE ROOFLINE ACTUALLY IS HERE. Not the ~10 us a 6.9 TB/s figure implies. On
// this B200 achievable read bandwidth depends strongly on transfer size, and 69 MB
// is small: a read-only kernel gets 7.06 TB/s on 4 GB, 6.22 on 1 GB, 4.08 on
// 138 MB and 3.55 on 69 MB (tools/bwprobe2.py, with the harness's per-iteration
// write-flush). Over the real 11-slab footprint the read-only floor is 20.5 us
// with ordinary loads and 18.4 us with an evict-first policy, and it does *not*
// move with the access pattern: warp-per-row, flat contiguous slabs, an even
// device-side split over 148 CTAs and 8-stage TMA (``cp.async.bulk``) all land
// within 10% of each other, for grids from 88 to 592 CTAs (tools/bwprobe3.py).
// Multi-stage prefetch has nothing to buy -- 176 CTAs x 4 KB per warp is already
// ~19 MB in flight against the ~4 MB Little's law asks for.
//
// So the L2 child's expert kernel at 23.9 us was within 15% of its floor, and the
// 43 us it took at M=1 was mostly elsewhere:
//
//   L2 child, M=1: expert 23.9 / route 6.3 / router 5.0 / cast 1.7 us of GPU
//                  time, 43.0 us of wall clock (~1.2 us of graph dispatch gap
//                  per launch)
//   this file:     expert 17.1 / route 3.3 / router 4.1 / cast 1.4, 30.7 us wall
//
// Four changes, in decreasing order of what they were worth:
//
//  1. ``red.global.add.v4.f32`` for the fp32 accumulator pass. It was 176 CTAs x
//     2048 *scalar* reductions, which priced at 4.1 us of the 36.9 (turning them
//     into plain stores, cfg 6, gives 32.8) -- and in vector form it is faster
//     than plain stores.
//  2. ``.L2::evict_first`` on the weight stream. Every byte is read once and the
//     working set is 3.2 GB against a 132 MB L2, so the lines have no reuse value;
//     without the hint each one evicts a dirty line and drags a writeback with it.
//  3. A register-resident top-k. E / 32 = 16 packed keys per lane, one warp per
//     token, so a selection round is a register tree-max plus one ``redux.sync``
//     rather than a strided shared-memory scan with block barriers in a single
//     CTA. The shared-expert gate projection and the accumulator clear moved into
//     the router, where they hide behind a memory-bound CTA.
//  4. A router grid that covers the machine (128 CTAs for a 2 MB read, not 64),
//     with the expert-row loads issued *before* the token-staging barrier so the
//     two cold round trips overlap.
//
// Everything the router decides is bit-identical to the reference by construction
// (see ``router_kernel`` and ``topk_key``); everything downstream of it only has
// to land inside the harness's bf16 tolerance, which is what buys the freedom to
// reassociate the expert dot products and the shared-expert gate.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

// Column slices the shared-expert gate projection is split into inside the
// router (one warp's worth of columns each, for H = 2048).
constexpr int GP = 8;

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float b2f(bf16 v) { return __bfloat162float(v); }
// bf16 -> fp32 straight off a packed bf16x2 register: a bf16 is the top half of
// the fp32 with the same value, so each of these is one integer instruction.
__device__ __forceinline__ float lo2f(unsigned u) { return __int_as_float(u << 16); }
__device__ __forceinline__ float hi2f(unsigned u) {
  return __int_as_float(u & 0xFFFF0000u);
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

// L2 eviction policy for the weight stream.
//
// Every weight byte is read exactly once, so the lines have no reuse value -- and
// the harness's timing loop *writes* a 2x-L2 buffer before every iteration, so
// each line the stream allocates evicts a dirty one and drags a writeback with
// it. Marking the stream evict-first lets it recycle its own lines instead:
// measured on the read-only probe over the real 11-slab footprint, 20.5 us ->
// 18.4 us (tools/bwprobe3.py).
//
// The eviction-priority *modifiers* (``ld.global.L2::evict_first``) need a 32-byte
// vector load on sm_100, which the phase mappings here do not use; the
// ``createpolicy`` + ``.L2::cache_hint`` form takes any width, and since the
// policy is a register operand the mode can be chosen at run time without
// duplicating a single load site.
__device__ __forceinline__ unsigned long long l2_policy(int mode) {
  unsigned long long p;
  if (mode == 1) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  } else if (mode == 2) {
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  }
  return p;
}
__device__ __forceinline__ uint4 ld16(const void* p, unsigned long long pol) {
  uint4 v;
  asm volatile("ld.global.nc.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
// Vector fp32 reductions. The accumulator pass is 176 CTAs x VEC floats per
// thread at M=1, and priced on its own it costs 4.1 us of a 36.9 us MoE (cfg 6
// below turns it into a plain store and lands at 32.8 us). ``red.global.add`` on
// sm_100 takes a .v2/.v4 vector, so the same bytes go in a quarter of the
// operations; the destination is ``accum + token * 2048 + tid * VEC``, which is
// 16-byte aligned for both VEC values.
__device__ __forceinline__ void red4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(a),
               "f"(b), "f"(c), "f"(d)
               : "memory");
}

__device__ __forceinline__ uint2 ld8(const void* p, unsigned long long pol) {
  uint2 v;
  asm volatile("ld.global.nc.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
               : "=r"(v.x), "=r"(v.y) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ float siluf_(float x) { return x * sigmoidf_(x); }

// Top-k selection key (identical rule to the L2 child's, which is what the
// reference does).
//
// The reference resolves an exact tie between two router logits toward the LOWER
// expert index, and with 512 bf16 logits ~10% of tokens have such a tie at the
// top-k boundary, so the rule is load-bearing. Packing value and index into one
// uint32 makes the comparison exact and lets a selection round reduce with a
// single ``__reduce_max_sync``.
//
// High 16 bits: the bf16 pattern mapped to a monotone unsigned order.
// Low 16 bits: 0xFFFF - index, so a larger index sorts lower and a tie resolves
// to the smaller index.
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
// 1. Routing: top-k + renormalize + per-expert work lists.
// ---------------------------------------------------------------------------
// One token's top-k, renormalize and work-list publication, by one warp.
//
// The L2 child spent 6.1 us of a single CTA here: K rounds of a *shared-memory*
// strided argmax, plus the shared expert's gate gemv, plus zeroing the fp32
// accumulator. E / 32 = 16 packed keys fit in registers, so each round becomes a
// register tree-max plus one ``redux.sync`` -- no shared memory, no barrier, and
// one warp per token instead of one CTA per token.
//
// The selected set, its order and the renormalized weights are identical to the
// L2 child's, and so to the reference: ``max`` over the packed keys is
// associative, and exp(l - lmax) over the selected set divided by its own sum is
// exactly softmax-then-renormalize, the global denominator cancelling.
//
// ``CG`` bypasses L1 on the logits read, which is required when this runs as the
// router's own epilogue: those lines were written by *other* CTAs.
template <int EPL, int CG>
__device__ __forceinline__ void route_one(
    const bf16* __restrict__ logits, int t, int M, int E, int K, int Tmax,
    int* __restrict__ cnt, int* __restrict__ tok, float* __restrict__ twt,
    int* __restrict__ n_active, int* __restrict__ active,
    const float* __restrict__ gate_part, float* __restrict__ gate_scale) {
  const int lane = threadIdx.x % 32;
  // Finish the shared expert's gate: the router computed GP column partials, on
  // CTAs that were waiting on memory anyway.
  if (lane == 0) {
    float g = 0.f;
#pragma unroll
    for (int i = 0; i < GP; ++i) {
      const float* q = gate_part + (size_t)i * M + t;
      g += CG ? __ldcg(q) : *q;
    }
    gate_scale[t] = sigmoidf_(g);
  }
  unsigned key[EPL];
  const bf16* lg = logits + (size_t)t * E;
#pragma unroll
  for (int i = 0; i < EPL; ++i) {
    const int e = i * 32 + lane;
    bf16 v;
    if (e < E) {
      if (CG) {
        unsigned short raw = __ldcg(reinterpret_cast<const unsigned short*>(lg + e));
        v = __ushort_as_bfloat16(raw);
      } else {
        v = lg[e];
      }
    }
    key[i] = (e < E) ? topk_key(v, e) : 0u;
  }

  // Every lane runs the same K rounds, so ``sum`` is warp-uniform and summed in
  // the same k order the reference renormalizes in; only lane k keeps round k's
  // winner, which is what the publication step below needs.
  float mywt = 0.f;
  int myexp = 0;
  float lmax = 0.f, sum = 0.f;
  for (int k = 0; k < K; ++k) {
    unsigned best = 0;
#pragma unroll
    for (int i = 0; i < EPL; ++i) best = max(best, key[i]);
    best = __reduce_max_sync(0xffffffffu, best);
    // Retire the winner: exactly one (lane, slot) holds it, keys being unique.
#pragma unroll
    for (int i = 0; i < EPL; ++i)
      if (key[i] == best) key[i] = 0u;
    const float v = key_value(best);
    if (k == 0) lmax = v;
    const float w = __expf(v - lmax);
    sum += w;
    if (lane == k) {
      mywt = w;
      myexp = key_index(best);
    }
  }
  // Publish the work items, one lane per selected expert. The first push for an
  // expert also appends it to the compacted active list, which bounds the expert
  // kernel's grid with no host sync.
  if (lane < K) {
    const int slot = atomicAdd(&cnt[myexp], 1);
    if (slot < Tmax) {
      tok[(size_t)myexp * Tmax + slot] = t;
      twt[(size_t)myexp * Tmax + slot] = mywt / (sum + 1e-20f);
    }
    if (slot == 0) active[atomicAdd(n_active, 1)] = myexp;
  }
}

template <int NT, int EPL>
__global__ __launch_bounds__(NT) void route_kernel(
    const bf16* __restrict__ logits, int M, int E, int K, int Tmax,
    int* __restrict__ cnt, int* __restrict__ tok, float* __restrict__ twt,
    int* __restrict__ n_active, int* __restrict__ active,
    const float* __restrict__ gate_part, float* __restrict__ gate_scale) {
  constexpr int NW = NT / 32;
  const int t = blockIdx.x * NW + threadIdx.x / 32;
  if (t >= M) return;
  route_one<EPL, 0>(logits, t, M, E, K, Tmax, cnt, tok, twt, n_active, active,
                    gate_part, gate_scale);
}

// ---------------------------------------------------------------------------
// 2. Router: logits[m, e] = sum_c x[m, c] * gate_w[e, c], plus the routing
//    prologue (counters, fp32 accumulator) and optionally the routing itself.
//
// The reduction mapping is *not* free to change. These logits decide a top-10 of
// 512 whose 10th/11th gap is ~0.006, so one bf16 ulp here flips a token's expert
// set and a flipped token is a whole wrong output row -- at M=1 that is 100% of
// the elements against a 99% bar. What makes this bit-compatible with the
// reference's cuBLAS bf16 GEMM is one fp32 accumulation of the whole row rounded
// once to bf16, and that is preserved exactly: one warp owns one expert row, its
// 32 lanes each take a fixed 8-element column slice, and the cross-lane sum is
// the same 5-step ``__shfl_down_sync`` tree. Only the *block* shape changed
// (NT=128, so 128 CTAs cover the machine for what was a 64-CTA 2 MB read).
//
// The extra work folded in here is all independent of routing and lands on CTAs
// that are waiting on memory anyway:
//   * blockIdx.x == 0 clears the per-expert counters and the active-list length,
//   * the low CTAs clear the fp32 output accumulator,
//   * the first GP CTAs each take one warp's worth of columns of the shared
//     expert's gate projection, which the L2 child did as a whole 2048-wide dot in
//     the single-CTA routing kernel.
// ---------------------------------------------------------------------------
template <int H, int BM, int NT, int RU, int MERGE, int EPL>
__global__ __launch_bounds__(NT) void router_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ gw,
    bf16* __restrict__ logits, int M, int E, int K, int Tmax,
    int* __restrict__ cnt, int* __restrict__ tok, float* __restrict__ twt,
    int* __restrict__ n_active, int* __restrict__ active,
    int* __restrict__ done, int* __restrict__ rdone,
    const bf16* __restrict__ sh_gate_w, float* __restrict__ gate_part,
    float* __restrict__ gate_scale, float* __restrict__ accum, int acc_blocks,
    int pol_mode) {
  constexpr int NW = NT / 32;
  constexpr int CIT = H / 256;
  extern __shared__ char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);        // [BM][H]

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int m0 = blockIdx.y * BM;
  const int e0 = blockIdx.x * NW + warp;

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    for (int i = tid; i < E; i += NT) cnt[i] = 0;
    if (tid == 0) {
      *n_active = 0;
      *done = 0;      // the expert kernel's arrival counter
    }
  }
  // Clear the fp32 accumulator the expert kernel atomically adds into, spread
  // over the first ``acc_blocks`` CTAs of the first token block.
  if (blockIdx.y == 0 && blockIdx.x < acc_blocks) {
    const long n4 = (long)M * (H / 4);
    float4* a4 = reinterpret_cast<float4*>(accum);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (long i = (long)blockIdx.x * NT + tid; i < n4; i += (long)acc_blocks * NT)
      a4[i] = z;
  }

  // Issue this warp's expert-row loads *before* staging the token rows, so the
  // two cold HBM round trips overlap instead of being separated by the staging
  // barrier. That is worth ~1 us of a 5 us kernel: the read itself is only 2 MB.
  uint4 w4[CIT];
  if (e0 < E) {
    const unsigned long long pol = l2_policy(pol_mode);
    const bf16* wr = gw + (size_t)e0 * H + lane * 8;
#pragma unroll RU
    for (int it = 0; it < CIT; ++it) w4[it] = ld16(wr + it * 256, pol);
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

  // Expert row dot products. The mapping is *not* free to change: these logits
  // decide a top-10 of 512 whose 10th/11th gap is ~0.006, so one bf16 ulp here
  // flips a token's expert set, and a flipped token is a whole wrong output row
  // (100% of the elements at M=1, against a 99% bar). What makes it
  // bit-compatible with the reference's cuBLAS bf16 GEMM is one fp32
  // accumulation of the whole row rounded once to bf16, and that is preserved
  // exactly: one warp owns one expert row, its 32 lanes take fixed 8-element
  // column slices, and the cross-lane sum is the same 5-step shuffle tree. Only
  // the block shape and the load scheduling changed.
  if (e0 < E) {
    float acc[BM];
#pragma unroll
    for (int t = 0; t < BM; ++t) acc[t] = 0.f;
#pragma unroll
    for (int it = 0; it < CIT; ++it) {
      const bf16* wv = reinterpret_cast<const bf16*>(&w4[it]);
#pragma unroll
      for (int t = 0; t < BM; ++t) {
        const uint4 x4 =
            *reinterpret_cast<const uint4*>(xs + t * H + lane * 8 + it * 256);
        const bf16* xv = reinterpret_cast<const bf16*>(&x4);
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[t] = fmaf(b2f(wv[q]), b2f(xv[q]), acc[t]);
      }
    }
#pragma unroll
    for (int t = 0; t < BM; ++t) {
      float v = acc[t];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
      if (lane == 0 && m0 + t < M) logits[(size_t)(m0 + t) * E + e0] = __float2bfloat16(v);
    }
  }

  // A GP-way split of the shared expert's gate projection, over the token rows
  // this CTA has already staged. Doing the whole 2048-wide dot in one CTA (which
  // is what the L2 child's routing kernel did) puts a cold 4 KB weight load on
  // somebody's critical path; with H / GP == 32 lanes x 8 elements one warp
  // covers a slice exactly, so there is no cross-warp reduction and nothing to
  // zero -- each CTA *stores* its own partial and the routing step sums GP of
  // them. Reassociating that sum is fine: it feeds sigmoid() and the shared
  // expert's scale, both downstream of the router.
  static_assert(H / GP == 256, "GP must give one warp's worth of columns");
  if (blockIdx.x < GP && warp == 0) {
    const int c0 = blockIdx.x * (H / GP) + lane * 8;
    const uint4 wv = *reinterpret_cast<const uint4*>(sh_gate_w + c0);
    const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
    for (int t = 0; t < BM; ++t) {
      if (m0 + t >= M) break;
      const uint4 xv = *reinterpret_cast<const uint4*>(xs + t * H + c0);
      const bf16* xp = reinterpret_cast<const bf16*>(&xv);
      float g = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) g = fmaf(b2f(xp[q]), b2f(wp[q]), g);
#pragma unroll
      for (int o = 16; o; o >>= 1) g += __shfl_down_sync(0xffffffffu, g, o);
      if (lane == 0) gate_part[(size_t)blockIdx.x * M + m0 + t] = g;
    }
  }

  // Optionally the routing epilogue too, in whichever CTA arrives last (MERGE).
  // The top-k needs every logit, so it cannot start until the whole grid is done
  // -- but it does not need a *launch* for that. An arrival counter plus a
  // release fence costs each CTA one atomic over the 16 bf16 it wrote, against
  // the ~1.5 us of graph dispatch gap a second kernel costs. (Contrast the same
  // trick applied to the output cast, where the fence has to wait for 2048 fp32
  // atomics per CTA and loses -- see ``expert_kernel``.)
  //
  // ``rdone`` must be zero on entry: the workspace is allocated zeroed and the
  // last CTA puts it back, so a completed launch always leaves it clean.
  if (MERGE) {
    __threadfence();
    __shared__ bool last;
    if (tid == 0)
      last = (atomicAdd(rdone, 1) == (int)(gridDim.x * gridDim.y) - 1);
    __syncthreads();
    if (!last) return;
    if (tid == 0) *rdone = 0;
    for (int t = warp; t < M; t += NW)
      route_one<EPL, 1>(logits, t, M, E, K, Tmax, cnt, tok, twt, n_active,
                        active, gate_part, gate_scale);
  }
}

// ---------------------------------------------------------------------------
// 3. Expert compute + cast.
//
// Slots index the compacted active-expert list first and then the shared expert,
// one slot per group of TG tokens; each slot's 512 intermediate columns are cut
// into chunks. MODE 1 (shipped) makes that a static 2-D grid, MODE 0 an even
// device-side split over a grid the caller picks -- see ``expert_kernel``.
//
// Phase 1 streams the w13 gate/up row pair for each intermediate column and
// reduces against the staged token rows, leaving SwiGLU in shared memory. Phase
// 2 streams w2t (stored [N, H], so a column chunk is a contiguous slab) and
// accumulates into the fp32 output with atomics. Neither expert activation
// reaches HBM.
//
// fp32 accumulation, not packed bf16x2 atomics: an element receives (top_k + 1) x
// (chunks per expert) contributions, and rounding a running sum to bf16 that many
// times put 1.2% of a single token's elements outside tolerance on a tested seed.
//
// The cast back to bf16 stays its own launch. Folding it into whichever CTA
// arrives last removes a launch and its ~1.2 us of graph dispatch gap, and was a
// 1.5 us *loss* while the reductions were scalar (the ``__threadfence()`` that
// publishes the accumulator has to wait for all of that CTA's reductions to
// land); with the vector form it is a wash, 30.72 against 30.69 us. Kept behind
// ``flags`` bit 0 either way.
// ---------------------------------------------------------------------------
template <int H, int NN, int TN, int TG, int NT, int JU, int UN, int NOATOM>
__device__ __forceinline__ void expert_tile(
    const bf16* __restrict__ WA, const bf16* __restrict__ WB, const bf16* xs,
    float* av, const int* tid_s, const float* tsc_s, float* __restrict__ accum,
    int j0, int jn, unsigned long long pol) {
  constexpr int VEC = H / NT;
  constexpr int NW = NT / 32;
  constexpr int CIT = H / 256;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;

  // ---- phase 1: a[j] = silu(w1[j].x) * (w3[j].x) --------------------------
  for (int jj = warp; jj < jn; jj += NW) {
    const int j = j0 + jj;
    const bf16* rg = WA + (size_t)j * H + lane * 8;
    const bf16* ru = WA + (size_t)(NN + j) * H + lane * 8;
    float ag[TN], au[TN];
#pragma unroll
    for (int t = 0; t < TN; ++t) { ag[t] = 0.f; au[t] = 0.f; }
#pragma unroll UN
    for (int it = 0; it < CIT; ++it) {
      const uint4 g4 = ld16(rg + it * 256, pol);
      const uint4 u4 = ld16(ru + it * 256, pol);
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
      if (lane == 0) av[(size_t)jj * TG + t] = siluf_(g) * u;
    }
  }
  __syncthreads();

  // ---- phase 2: out += scale * w2[:, chunk] . a[chunk] -------------------
  float oacc[VEC][TN];
#pragma unroll
  for (int q = 0; q < VEC; ++q)
#pragma unroll
    for (int t = 0; t < TN; ++t) oacc[q][t] = 0.f;
  const int hb = tid * VEC;
  const bf16* wbase = WB + (size_t)j0 * H + hb;
  int jj = 0;
  for (; jj + JU <= jn; jj += JU) {
    uint4 wraw[JU];
#pragma unroll
    for (int u = 0; u < JU; ++u) {
      if (VEC == 8) {
        wraw[u] = ld16(wbase + (size_t)(jj + u) * H, pol);
      } else {
        const uint2 v = ld8(wbase + (size_t)(jj + u) * H, pol);
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
        const float a = av[(size_t)(jj + u) * TG + t];
#pragma unroll
        for (int q = 0; q < VEC; ++q) oacc[q][t] = fmaf(wv[q], a, oacc[q][t]);
      }
    }
  }
  for (; jj < jn; ++jj) {
    uint4 wraw;
    if (VEC == 8) {
      wraw = ld16(wbase + (size_t)jj * H, pol);
    } else {
      const uint2 v = ld8(wbase + (size_t)jj * H, pol);
      wraw.x = v.x;
      wraw.y = v.y;
    }
    const unsigned* wp = reinterpret_cast<const unsigned*>(&wraw);
    float wv[VEC];
#pragma unroll
    for (int q = 0; q < VEC; q += 2) {
      wv[q] = lo2f(wp[q / 2]);
      wv[q + 1] = hi2f(wp[q / 2]);
    }
#pragma unroll
    for (int t = 0; t < TN; ++t) {
      const float a = av[(size_t)jj * TG + t];
#pragma unroll
      for (int q = 0; q < VEC; ++q) oacc[q][t] = fmaf(wv[q], a, oacc[q][t]);
    }
  }
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const float sc = tsc_s[t];
    if (sc == 0.f) continue;      // padding slot, or a genuinely zero weight
    float* dst = accum + (size_t)tid_s[t] * H + hb;
    if (NOATOM) {
      // A *diagnostic*: this produces a wrong answer (every chunk overwrites the
      // others) and exists only to price the reduction pass.
#pragma unroll
      for (int q = 0; q < VEC; ++q) dst[q] = sc * oacc[q][t];
    } else {
#pragma unroll
      for (int q = 0; q < VEC; q += 4)
        red4(dst + q, sc * oacc[q][t], sc * oacc[q + 1][t], sc * oacc[q + 2][t],
             sc * oacc[q + 3][t]);
    }
  }
}

// One (slot, [j0, j0+jn)) piece of work: stage the slot's tokens, then run the
// two phases over TJ-row tiles.
template <int H, int NN, int TG, int NT, int JU, int UN, int TJ, int NOATOM>
__device__ __forceinline__ void run_piece(
    const bf16* __restrict__ x, const bf16* __restrict__ w13,
    const bf16* __restrict__ w2t, const bf16* __restrict__ sh_gu,
    const bf16* __restrict__ sh_dnt, const int* __restrict__ cnt,
    const int* __restrict__ tok, const float* __restrict__ twt,
    const int* __restrict__ active, const float* __restrict__ gate_scale,
    float* __restrict__ accum, bf16* xs, float* av, int* tid_s, float* tsc_s,
    int M, int Tmax, int nsh_base, int slot, int j0, int jn,
    unsigned long long pol) {
  const int tid = threadIdx.x;
  const bool is_shared = (slot >= nsh_base);
  const bf16* WA;
  const bf16* WB;
  int t_begin, t_end;
  const int* tlist = nullptr;
  const float* wlist = nullptr;
  if (is_shared) {
    WA = sh_gu;
    WB = sh_dnt;
    t_begin = (slot - nsh_base) * TG;
    t_end = min(M, t_begin + TG);
  } else {
    const int e = active[slot];
    int tcnt = cnt[e];
    if (tcnt > Tmax) tcnt = Tmax;
    WA = w13 + (size_t)e * 2 * NN * H;
    WB = w2t + (size_t)e * NN * H;
    tlist = tok + (size_t)e * Tmax;
    wlist = twt + (size_t)e * Tmax;
    t_begin = 0;
    t_end = tcnt;
  }
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
    // Dispatch on the *exact* group size where we can: rounding up to a power of
    // two pads every expert's last group, and padding is the dominant cost once
    // an expert holds only a token or two.
    for (int jt = 0; jt < jn; jt += TJ) {
      const int jc = min(TJ, jn - jt);
#define GROUP(N)                                                               \
  expert_tile<H, NN, ((N) <= TG ? (N) : TG), TG, NT, JU, UN, NOATOM>(                  \
      WA, WB, xs, av, tid_s, tsc_s, accum, j0 + jt, jc, pol)
      switch (tn) {
        case 1: GROUP(1); break;
        case 2: GROUP(2); break;
        case 3: GROUP(3); break;
        case 4: GROUP(4); break;
        case 5: case 6: GROUP(6); break;
        case 7: case 8: GROUP(8); break;
        default: GROUP(TG); break;
      }
#undef GROUP
      __syncthreads();
    }
  }
}

// MODE 0: a 1-D grid whose CTAs each take an equal slice of the (slot x
//         intermediate-column) space, computed on the device where the active
//         expert count is known. Balanced for any count, but the slice is only a
//         multiple of the warp count when the grid divides it.
// MODE 1: the L2 child's static grid -- blockIdx.x is the slot, blockIdx.y a
//         fixed-width column chunk. Wastes CTAs when fewer experts are active
//         than top_k * M, but a chunk width that is a multiple of the warp count
//         keeps phase 1 perfectly balanced inside the CTA.
template <int H, int NN, int TG, int NT, int MINB, int JU, int UN, int TJ,
          int MODE, int NOATOM>
__global__ __launch_bounds__(NT, MINB) void expert_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ w13,
    const bf16* __restrict__ w2t, const bf16* __restrict__ sh_gu,
    const bf16* __restrict__ sh_dnt, const int* __restrict__ cnt,
    const int* __restrict__ tok, const float* __restrict__ twt,
    const int* __restrict__ n_active, const int* __restrict__ active,
    const float* __restrict__ gate_scale, float* __restrict__ accum,
    bf16* __restrict__ out, int* __restrict__ done, int M, int Tmax, int Nc,
    int nxr, int pol_mode) {
  constexpr int VEC = H / NT;
  static_assert(H % NT == 0 && (VEC == 4 || VEC == 8), "bad VEC");

  extern __shared__ char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);              // [TG][H]
  float* av = reinterpret_cast<float*>(xs + TG * H);     // [TJ][TG]
  __shared__ int tid_s[TG];
  __shared__ float tsc_s[TG];

  const int nact = *n_active;
  const int nsh = (M + TG - 1) / TG;
  const int tid = threadIdx.x;
  const unsigned long long pol = l2_policy(pol_mode);

  if (MODE == 0) {
    const long units = (long)(nact + nsh) * NN;
    const long lo = (long)blockIdx.x * units / gridDim.x;
    const long hi = (long)(blockIdx.x + 1) * units / gridDim.x;
    for (long u = lo; u < hi;) {
      const int slot = (int)(u / NN);
      const int j0 = (int)(u % NN);
      const int jn = (int)min((long)(NN - j0), hi - u);
      u += jn;
      run_piece<H, NN, TG, NT, JU, UN, TJ, NOATOM>(
          x, w13, w2t, sh_gu, sh_dnt, cnt, tok, twt, active, gate_scale, accum,
          xs, av, tid_s, tsc_s, M, Tmax, nact, slot, j0, jn, pol);
    }
  } else {
    const int slot = blockIdx.x;
    const int j0 = blockIdx.y * Nc;
    const bool live = (slot < nxr ? slot < nact : slot < nxr + nsh) && j0 < NN;
    if (live) {
      run_piece<H, NN, TG, NT, JU, UN, TJ, NOATOM>(
          x, w13, w2t, sh_gu, sh_dnt, cnt, tok, twt, active, gate_scale, accum,
          xs, av, tid_s, tsc_s, M, Tmax, nxr, slot, j0, min(Nc, NN - j0), pol);
    }
  }

  // ---- fp32 accumulator -> bf16 output, in the last CTA to arrive ---------
  // The router zeroed ``done`` earlier in the same call, so this costs one atomic
  // per CTA and saves a launch plus its ~1.3 us of graph dispatch gap. Every CTA
  // reaches here, including ones with no work, or the count would never
  // complete.
  if (done == nullptr) return;      // the caller kept the cast as its own launch
  __threadfence();
  __shared__ bool last;
  if (tid == 0) {
    const int nb = (int)(gridDim.x * gridDim.y);
    last = (atomicAdd(done, 1) == nb - 1);
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  const long n = (long)M * H;
  for (long i = (long)tid * 4; i + 3 < n; i += (long)NT * 4) {
    // ``__ldcg``, not a plain load: these lines were written by *other* CTAs'
    // atomics, which land in L2 and leave L1 alone.
    const float4 v = __ldcg(reinterpret_cast<const float4*>(accum + i));
    const bf16 o[4] = {__float2bfloat16(v.x), __float2bfloat16(v.y),
                       __float2bfloat16(v.z), __float2bfloat16(v.w)};
    *reinterpret_cast<uint2*>(out + i) = *reinterpret_cast<const uint2*>(o);
  }
}

// Separate cast, for the A/B against folding it into the expert kernel.
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

struct Args {
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
  bf16* out;
  int* done;
  int M;
  int K;
  int E;
  int Tmax;
  int nctas;
  int nchunk;     // MODE 1: column chunk width (0 = auto)
  int pol_mode;   // L2 eviction policy for the weight stream
  bool fold_cast;
  cudaStream_t st;
};

template <int TG, int NT, int MINB, int JU, int UN, int TJ, int MODE, int NOATOM>
void launch_experts(const Args& a) {
  constexpr int H = 2048, NN = 512;
  const size_t sm = (size_t)TG * H * sizeof(bf16) + (size_t)TJ * TG * 4;
  auto fn = expert_kernel<H, NN, TG, NT, MINB, JU, UN, TJ, MODE, NOATOM>;
  static bool set_once = false;
  if (!set_once) {
    TORCH_CHECK(cudaFuncSetAttribute(
                    fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) ==
                    cudaSuccess,
                "cannot opt in to ", sm, " bytes of dynamic shared memory");
    set_once = true;
  }
  bf16* out = a.fold_cast ? a.out : nullptr;
  int* done = a.fold_cast ? a.done : nullptr;
  if (MODE == 0) {
    fn<<<a.nctas, NT, sm, a.st>>>(a.x, a.w13, a.w2t, a.sh_gu, a.sh_dnt, a.cnt,
                                  a.tok, a.twt, a.n_active, a.active,
                                  a.gate_scale, a.accum, out, done, a.M, a.Tmax,
                                  0, 0, a.pol_mode);
  } else {
    const int nxr = (int)std::min<long>((long)a.E, (long)a.M * a.K);
    const int nsh = (a.M + TG - 1) / TG;
    // Chunk width over the intermediate dim, as a multiple of the warp count so
    // phase 1 divides evenly across the CTA's warps, narrowed until the grid
    // covers the machine.
    constexpr int NSM = 148;
    constexpr int NW = NT / 32;
    // Two effects pull in opposite directions, and both were measured. Wide
    // chunks (few, fat CTAs) lose memory parallelism: at M=8 a width of 256
    // (164 CTAs) costs 158 us against 111 us at 48 (902 CTAs). Narrow chunks pay
    // a per-chunk atomic pass and token-tile staging, and at M=1 a width of 8
    // costs 43 us against 35 us at 32-48. Widths are multiples of the CTA's warp
    // count so phase 1 divides evenly across warps; the width need not divide NN,
    // so the last chunk may be short.
    int Nc = a.nchunk > 0 ? a.nchunk : (a.M <= 160 ? 3 * NW : 6 * NW);
    while (Nc > NW && (long)(nxr + nsh) * ((NN + Nc - 1) / Nc) < NSM) Nc -= NW;
    const int S = (NN + Nc - 1) / Nc;
    fn<<<dim3(nxr + nsh, S), NT, sm, a.st>>>(
        a.x, a.w13, a.w2t, a.sh_gu, a.sh_dnt, a.cnt, a.tok, a.twt, a.n_active,
        a.active, a.gate_scale, a.accum, out, done, a.M, a.Tmax, Nc, nxr,
        a.pol_mode);
  }
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "expert launch failed: ", cudaGetErrorString(e),
              " TG=", TG, " NT=", NT, " mode=", MODE, " smem=", sm);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry point. One pybind call, three launches (two with the cast folded),
// no host sync, nothing allocated.
// ---------------------------------------------------------------------------
void moe_small(
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
    int64_t grid_ovr,      // 0 = auto; >0 = MODE 0 CTA count; <0 = MODE 1 chunk
                           //   width (columns per CTA), negated
    int64_t cfg_ovr,       // <0 = auto expert-kernel shape
    int64_t flags) {       // 0: fold cast; 1: merge routing; 4-6: router
                       // shape; 8-9: L2 eviction policy
  const int M = (int)x.size(0);
  const int H = (int)x.size(1);
  const int E = (int)gate_w.size(0);
  const int N = (int)w2t.size(1);
  const int K = (int)top_k;
  TORCH_CHECK(H == 2048 && N == 512, "moe_small expects H=2048, N=512");
  TORCH_CHECK(K <= 32, "moe_small expects top_k <= 32");
  TORCH_CHECK(E == 512, "moe_small expects E=512");

  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  // L2 eviction policy for the weight stream. evict-first is the default because
  // the stream has no reuse and the harness's write-flush leaves L2 full of dirty
  // lines: 36.9 us against 41.0 us for the whole MoE at M=1.
  //   flags bits 8-9: 0 = evict_first (default), 1 = evict_normal, 2 = evict_last
  const int polsel = (int)((flags >> 8) & 3);
  const int pol_mode = (polsel == 0) ? 1 : (polsel == 1 ? 0 : 2);
  const bf16* xp = (const bf16*)x.data_ptr();
  const int Tmax = M;

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
  float* gate_part = (float*)take((size_t)GP * M * 4);
  int* done = (int*)take(4);
  int* rdone = (int*)take(4);
  float* accum = (float*)take((size_t)M * H * 4);
  TORCH_CHECK(off <= (size_t)ws.numel(), "workspace too small");

  // --- 1. router (+ counters, accumulator clear, shared-expert gate partials,
  //        and the routing itself when merged) ---
  const bool merge = (flags & 2) != 0;
  {
#define LAUNCH_ROUTER(BM, NT, RU, MG)                                            \
  do {                                                                          \
    constexpr int NW = (NT) / 32;                                               \
    const int acc_blocks =                                                      \
        min((E + NW - 1) / NW,                                                  \
            (int)(((long)M * (H / 4) + (NT) - 1) / (NT)));                      \
    const size_t sm = (size_t)(BM) * 2048 * sizeof(bf16);                       \
    auto fn = router_kernel<2048, BM, NT, RU, MG, 16>;                          \
    static bool set_once = false;                                               \
    if (!set_once) {                                                            \
      cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                           (int)sm);                                            \
      set_once = true;                                                          \
    }                                                                           \
    fn<<<dim3((E + NW - 1) / NW, (M + (BM) - 1) / (BM)), NT, sm, st>>>(         \
        xp, (const bf16*)gate_w.data_ptr(), logits, M, E, K, Tmax, cnt, tok,     \
        twt, n_active, active, done, rdone,                                     \
        (const bf16*)sh_gate_w.data_ptr(), gate_part, gate_scale, accum,        \
        acc_blocks, pol_mode);                                                  \
  } while (0)
    // The token tile is padded to BM, so an oversized BM both stages and
    // multiplies duplicate rows: at M=1, BM=16 is 16x the arithmetic and 16x the
    // staging of BM=1.
#define LAUNCH_ROUTER_BM(NT, RU, MG)                                             \
  do {                                                                          \
    if (M <= 2) {                                                               \
      LAUNCH_ROUTER(1, NT, RU, MG);                                             \
    } else if (M <= 8) {                                                        \
      LAUNCH_ROUTER(4, NT, RU, MG);                                             \
    } else {                                                                    \
      LAUNCH_ROUTER(16, NT, RU, MG);                                            \
    }                                                                           \
  } while (0)
#define LAUNCH_ROUTER_MG(NT, RU)                                                 \
  do {                                                                          \
    if (merge) {                                                                \
      LAUNCH_ROUTER_BM(NT, RU, 1);                                              \
    } else {                                                                    \
      LAUNCH_ROUTER_BM(NT, RU, 0);                                              \
    }                                                                           \
  } while (0)
    switch ((int)((flags >> 4) & 7)) {
      case 1: LAUNCH_ROUTER_MG(256, 8); break;
      case 2: LAUNCH_ROUTER_MG(128, 4); break;
      default: LAUNCH_ROUTER_MG(128, 8); break;
    }
#undef LAUNCH_ROUTER_MG
#undef LAUNCH_ROUTER_BM
#undef LAUNCH_ROUTER
  }

  // --- 2. routing, unless the router did it in its last CTA ---
  if (!merge) {
    constexpr int NT = 256;
    constexpr int NW = NT / 32;
    route_kernel<NT, 16><<<(M + NW - 1) / NW, NT, 0, st>>>(
        logits, M, E, K, Tmax, cnt, tok, twt, n_active, active, gate_part,
        gate_scale);
  }

  // --- 3. experts (routed + shared) + cast ---
  {
    constexpr int NSM = 148;
    // One wave of CTAs is enough: the whole kernel is one pass over the weights,
    // and an even split means every CTA carries the same number of intermediate
    // columns whatever the active-expert count turns out to be.
    int nctas = (int)(grid_ovr > 0 ? grid_ovr : 0);
    const long nslot_max = std::min<long>((long)E, (long)M * K) + (M + 3) / 4;
    const long max_units = nslot_max * N;
    if (nctas <= 0) nctas = NSM;
    if (nctas > max_units) nctas = (int)max_units;
    const int nchunk = (int)(grid_ovr < 0 ? -grid_ovr : 0);
    Args a{xp,
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
           (bf16*)out.data_ptr(),
           done,
           M,
           K,
           E,
           Tmax,
           nctas,
           nchunk,
           pol_mode,
           (flags & 1) != 0,
           st};
    const int cfg = (cfg_ovr >= 0) ? (int)cfg_ovr : 0;
    switch (cfg) {
      //              TG  NT  MINB JU  UN  TJ MODE NOATOM
      case 0: launch_experts<4, 512, 2, 8, 4, 64, 1, 0>(a); break;
      case 1: launch_experts<4, 512, 2, 8, 8, 64, 1, 0>(a); break;
      case 2: launch_experts<4, 512, 2, 8, 4, 64, 0, 0>(a); break;
      case 3: launch_experts<4, 256, 4, 8, 4, 64, 1, 0>(a); break;
      case 4: launch_experts<8, 512, 2, 8, 4, 64, 1, 0>(a); break;
      case 5: launch_experts<4, 512, 2, 4, 4, 64, 1, 0>(a); break;
      // Diagnostic only -- wrong output, prices the fp32 atomic pass.
      case 6: launch_experts<4, 512, 2, 8, 4, 64, 1, 1>(a); break;
      default: launch_experts<4, 512, 2, 8, 4, 64, 1, 0>(a); break;
    }
    if ((flags & 1) == 0) {
      const long n = (long)M * H;
      constexpr int CNT = 256;
      const long blocks = (n / 4 + CNT - 1) / CNT;
      cast_kernel<<<(int)blocks, CNT, 0, st>>>(accum, (bf16*)out.data_ptr(), n);
    }
  }

  {
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_small launch failed: ",
                cudaGetErrorString(err), " (M=", M, " K=", K, ")");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_small", &moe_small,
        "Fused routed-expert MoE for small token counts");
}
