// MoE block alignment: histogram -> padded per-expert prefix -> scatter.
//
// The stock SGLang kernel runs moe_align_block_size_kernel<<<2, 1024>>> (block 0
// histograms + scans all numel serially, block 1 alone sentinel-fills the whole
// padded buffer) and then a second launch to scatter with one global atomicAdd
// per token: two launches, 146 of 148 B200 SMs idle.
//
// Measured cost model for this op under the harness (cold L2, CUDA events around
// an eager call):
//
//     measured_us ~= 7.15 (event + input-copy floor) + kernel_device_us
//     each launch costs a further 2.06 us
//     an empty kernel's own cold-L2 device time is 1.46 us
//
// so the measurement is quantised in 2.05 us steps, launch count is the coarsest
// lever, and bandwidth is irrelevant (the largest benched case moves < 3 MB).
// Everything below is about launch count, dependent round trips and instruction
// count.  See ITERATIONS.md for the calibration and the measured dead ends.
//
// Two kernels, one launch either way:
//
//   redundant  numel <= red_max.  Every block histograms the *whole* input, once
//              for the part before its own slice and once for the rest, so it
//              can derive the global padded prefix and its own per-expert base
//              with zero inter-block communication.  grid*numel shared atomics
//              is far cheaper than the ~4.5 us a grid barrier costs at these
//              sizes, and unlike a single-block kernel it uses every SM.
//   fused      larger numel, where redundant histogramming would dominate.  Many
//              blocks plus a software grid barrier: phase A histograms a slice
//              and publishes it with one atomicAdd per (block, expert) whose
//              return value is that block's arrival-order base inside the
//              expert; the last block to arrive scans; phase B turns bases into
//              absolute slots and scatters.  The grid is capped at the
//              occupancy-derived resident block count so it cannot deadlock.
//
// Both write the sentinel only into the per-expert padding gaps (plus the unused
// tail past the padded total).  Those slots are disjoint from every slot the
// scatter writes, so no block ever has to be ordered against another.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define FK_WARP 32

namespace {

__device__ __forceinline__ int warp_incl_scan(int x) {
  const int lane = threadIdx.x & (FK_WARP - 1);
#pragma unroll
  for (int o = 1; o < FK_WARP; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  return x;
}

// Exclusive scan of ceil(count/block_size)*block_size over `num_experts`
// entries, by one block.  `load(e)` supplies the count, `emit(e, excl, cnt)`
// consumes the result; returns the padded total (uniform across threads).
template <int THREADS, typename Load, typename Emit>
__device__ __forceinline__ int padded_prefix_scan(
    int num_experts, int block_size, int32_t* s_warp, Load load, Emit emit) {
  constexpr int NWARP = THREADS / FK_WARP;
  const int tid = threadIdx.x;
  const int lane = tid & (FK_WARP - 1);
  const int warp = tid / FK_WARP;
  int running = 0;
  for (int base = 0; base < num_experts; base += THREADS) {
    const int e = base + tid;
    const int cnt = (e < num_experts) ? load(e) : 0;
    const int pad = (cnt + block_size - 1) / block_size * block_size;
    const int incl = warp_incl_scan(pad);
    if (lane == FK_WARP - 1) s_warp[warp] = incl;
    __syncthreads();
    if (tid < FK_WARP) {
      int v = (tid < NWARP) ? s_warp[tid] : 0;
      v = warp_incl_scan(v);
      if (tid < NWARP) s_warp[tid] = v;
    }
    __syncthreads();
    if (e < num_experts)
      emit(e, running + incl - pad + (warp ? s_warp[warp - 1] : 0), cnt);
    running += s_warp[NWARP - 1];
    __syncthreads();
  }
  return running;
}

// block -> owning expert, for every padded block.  The prefix is non-decreasing,
// so the owner is the last expert whose padded start is <= the block's first slot
// (experts with count 0 own no block and the <= skips past them).
template <int THREADS>
__device__ __forceinline__ void write_expert_ids(
    const int32_t* __restrict__ s_pref,
    int32_t* __restrict__ expert_ids,
    int num_experts,
    int block_size,
    int num_blocks,
    int first,
    int step) {
  for (int b = first; b < num_blocks; b += step) {
    const int start = b * block_size;
    int lo = 0, hi = num_experts;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (s_pref[mid] <= start)
        lo = mid + 1;
      else
        hi = mid;
    }
    expert_ids[b] = lo - 1;
  }
}

__device__ __forceinline__ void bump4(int32_t* s_dst, const int4 x) {
  atomicAdd(&s_dst[x.x], 1);
  atomicAdd(&s_dst[x.y], 1);
  atomicAdd(&s_dst[x.z], 1);
  atomicAdd(&s_dst[x.w], 1);
}

// Rank one 4-id vector inside each id's expert with a shared atomic and drop the
// token index at the resulting absolute slot.
__device__ __forceinline__ void scatter4(
    int32_t* out, int32_t* s_cur, const int4 x, int i) {
  out[atomicAdd(&s_cur[x.x], 1)] = i;
  out[atomicAdd(&s_cur[x.y], 1)] = i + 1;
  out[atomicAdd(&s_cur[x.z], 1)] = i + 2;
  out[atomicAdd(&s_cur[x.w], 1)] = i + 3;
}

// Histogram topk_ids[lo, hi) into a shared array.  With VEC4, lo is a multiple of
// 4 and the pointer is 16 B aligned, so the vector body is always aligned.
// UNROLL vector loads are issued back-to-back before the dependent atomics: a
// plain load-then-atomic loop stalls once per iteration on a cold miss.
template <int THREADS, int UNROLL, bool VEC4, int REPL = 1>
__device__ __forceinline__ void hist_range(
    const int32_t* __restrict__ topk_ids,
    int lo,
    int hi,
    int32_t* s_base,
    int stride_r = 0) {
  const int tid = threadIdx.x;
  // Each lane owns one replica; with a stride of num_experts+1, two lanes in
  // different replicas hitting the same expert also hit different banks, which is
  // what makes replication worth its extra reduction pass.
  int32_t* const s_dst = (REPL > 1) ? s_base + (tid % REPL) * stride_r : s_base;
  if (VEC4) {
    const int4* const in4 = reinterpret_cast<const int4*>(topk_ids + lo);
    const int nv = (hi - lo) >> 2;
    int v = tid;
    for (; v + (UNROLL - 1) * THREADS < nv; v += UNROLL * THREADS) {
      int4 x[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) x[u] = in4[v + u * THREADS];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) bump4(s_dst, x[u]);
    }
    for (; v < nv; v += THREADS) bump4(s_dst, in4[v]);
    for (int i = lo + (nv << 2) + tid; i < hi; i += THREADS)
      atomicAdd(&s_dst[topk_ids[i]], 1);
  } else {
    for (int i = lo + tid; i < hi; i += THREADS)
      atomicAdd(&s_dst[topk_ids[i]], 1);
  }
}

// Scatter topk_ids[lo, hi); same unrolling as hist_range.
template <int THREADS, int UNROLL, bool VEC4>
__device__ __forceinline__ void scatter_range(
    const int32_t* __restrict__ topk_ids,
    int32_t* out,
    int32_t* s_cur,
    int lo,
    int hi) {
  const int tid = threadIdx.x;
  if (VEC4) {
    const int4* const in4 = reinterpret_cast<const int4*>(topk_ids + lo);
    const int nv = (hi - lo) >> 2;
    int v = tid;
    for (; v + (UNROLL - 1) * THREADS < nv; v += UNROLL * THREADS) {
      int4 x[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) x[u] = in4[v + u * THREADS];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u)
        scatter4(out, s_cur, x[u], lo + ((v + u * THREADS) << 2));
    }
    for (; v < nv; v += THREADS) scatter4(out, s_cur, in4[v], lo + (v << 2));
    for (int i = lo + (nv << 2) + tid; i < hi; i += THREADS)
      out[atomicAdd(&s_cur[topk_ids[i]], 1)] = i;
  } else {
    for (int i = lo + tid; i < hi; i += THREADS)
      out[atomicAdd(&s_cur[topk_ids[i]], 1)] = i;
  }
}

// Sentinel into this block's share of the padding: the gap after each of its
// assigned experts, plus the unused tail past the padded total.  Disjoint from
// every slot the scatter writes, so no block needs ordering against another.
//
// Experts are handed out per *warp*, not per block: a gap is shorter than
// block_size, so a block-wide loop leaves all but the first block_size lanes idle
// and serialises num_experts/gridDim iterations.  Per-warp there are
// gridDim*NWARP claimants, which for a small grid is the difference between ~7
// dependent iterations and ~1 (measured ~1 us on [314,8]).
template <int THREADS, bool WARP_PER_EXPERT>
__device__ __forceinline__ void fill_padding(
    int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ s_pref,
    const int32_t* __restrict__ s_gap,
    int num_experts,
    int numel,
    int total,
    int max_padded,
    int gtid,
    int stride) {
  constexpr int NWARP = THREADS / FK_WARP;
  const int tid = threadIdx.x;
  const int warp = tid / FK_WARP;
  const int lane = tid & (FK_WARP - 1);
  const int first = WARP_PER_EXPERT ? blockIdx.x * NWARP + warp : blockIdx.x;
  const int step = WARP_PER_EXPERT ? gridDim.x * NWARP : gridDim.x;
  const int off = WARP_PER_EXPERT ? lane : tid;
  const int inc = WARP_PER_EXPERT ? FK_WARP : THREADS;
  for (int e = first; e < num_experts; e += step) {
    const int gap_end = s_pref[e + 1];
    for (int i = s_gap[e] + off; i < gap_end; i += inc)
      sorted_token_ids[i] = numel;
  }
  for (int i = total + gtid; i < max_padded; i += stride)
    sorted_token_ids[i] = numel;
}

// ---------------------------------------------------------------------------
// redundant: many blocks, no barrier, no global communication
// ---------------------------------------------------------------------------
template <int THREADS, int UNROLL, bool VEC4, int REPL>
__global__ void __launch_bounds__(THREADS) moe_align_redundant_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_post_pad,
    const int num_experts,
    const int block_size,
    const int numel,
    const int max_padded) {
  constexpr int NWARP = THREADS / FK_WARP;
  extern __shared__ int32_t smem[];
  int32_t* const s_before = smem;                  // hist[0, start) -> cursor
  int32_t* const s_rest = smem + num_experts;      // hist[start, numel) -> gap
  int32_t* const s_pref = smem + 2 * num_experts;  // [E+1]
  int32_t* const s_repl = smem + 3 * num_experts + 1;  // [2][REPL][E+1]
  __shared__ int32_t s_warp[NWARP];

  const int tid = threadIdx.x;
  const int stride = gridDim.x * THREADS;
  const int gtid = blockIdx.x * THREADS + tid;

  // Slice boundaries are multiples of 4 so the vector loads stay aligned.
  const int per = ((numel + gridDim.x - 1) / gridDim.x + 3) & ~3;
  int start = blockIdx.x * per;
  if (start > numel) start = numel;
  int end = start + per;
  if (end > numel) end = numel;

  // Two replicated histograms, strided by num_experts+1 so lanes landing in
  // different replicas also land in different banks.
  const int stride_r = num_experts + 1;
  for (int i = tid; i < 2 * REPL * stride_r; i += THREADS) s_repl[i] = 0;
  __syncthreads();

  hist_range<THREADS, UNROLL, VEC4, REPL>(topk_ids, 0, start, s_repl, stride_r);
  hist_range<THREADS, UNROLL, VEC4, REPL>(topk_ids, start, numel,
                                          s_repl + REPL * stride_r, stride_r);
  __syncthreads();
  for (int e = tid; e < num_experts; e += THREADS) {
    int b = 0, r = 0;
#pragma unroll
    for (int k = 0; k < REPL; ++k) {
      b += s_repl[k * stride_r + e];
      r += s_repl[(REPL + k) * stride_r + e];
    }
    s_before[e] = b;
    s_rest[e] = r;
  }
  __syncthreads();

  const int total = padded_prefix_scan<THREADS>(
      num_experts, block_size, s_warp,
      [&](int e) { return s_before[e] + s_rest[e]; },
      [&](int e, int excl, int) { s_pref[e] = excl; });
  if (gtid == 0) *num_tokens_post_pad = total;
  // s_before -> this block's write cursor, s_rest -> start of the padding gap.
  for (int e = tid; e < num_experts; e += THREADS) {
    const int base = s_pref[e] + s_before[e];
    s_rest[e] = base + s_rest[e];
    s_before[e] = base;
  }
  if (tid == 0) s_pref[num_experts] = total;
  __syncthreads();

  write_expert_ids<THREADS>(s_pref, expert_ids, num_experts, block_size,
                            total / block_size, gtid, stride);
  fill_padding<THREADS, true>(sorted_token_ids, s_pref, s_rest, num_experts,
                              numel, total, max_padded, gtid, stride);
  scatter_range<THREADS, UNROLL, VEC4>(topk_ids, sorted_token_ids, s_before,
                                       start, end);
}

// ---------------------------------------------------------------------------
// fused: many blocks + software grid barrier
// ---------------------------------------------------------------------------
// Workspace (int32, zero-initialised once by the caller):
//   [0]              arrival counter (reset by the scan block every launch)
//   [1]              epoch           (monotonic, never reset)
//   [2 .. 2+E)       per-expert counts, re-zeroed by the scan block's atomicExch
//   [2+E .. 3+2E)    padded exclusive prefix, total at index E
//   [3+2E .. 3+3E)   gap start per expert (prefix[e] + count[e])
template <int THREADS, int UNROLL, bool VEC4>
__global__ void __launch_bounds__(THREADS) moe_align_fused_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tokens_post_pad,
    int32_t* ws,
    const int num_experts,
    const int block_size,
    const int numel,
    const int max_padded,
    const int spin_sleep_ns) {
  constexpr int NWARP = THREADS / FK_WARP;
  extern __shared__ int32_t smem[];
  int32_t* const s_hist = smem;                     // count -> base -> cursor
  int32_t* const s_pref = smem + num_experts;       // [E+1]
  int32_t* const s_gap = s_pref + num_experts + 1;  // [E]
  __shared__ int32_t s_warp[NWARP];
  __shared__ int32_t s_is_last;

  int32_t* const w_arrive = ws;
  int32_t* const w_epoch = ws + 1;
  int32_t* const w_counts = ws + 2;
  int32_t* const w_prefix = ws + 2 + num_experts;
  int32_t* const w_gap = w_prefix + num_experts + 1;

  const int tid = threadIdx.x;
  const int stride = gridDim.x * THREADS;
  const int gtid = blockIdx.x * THREADS + tid;
  // Contiguous per-block slice; phases A and B must agree on it.  Multiples of 4
  // so the vector loads stay aligned.
  const int per = ((numel + gridDim.x - 1) / gridDim.x + 3) & ~3;
  int start = blockIdx.x * per;
  if (start > numel) start = numel;
  int end = start + per;
  if (end > numel) end = numel;

  // The harness flushes L2 before every call, so the first touch of the tiny
  // workspace would be a full DRAM round trip sitting on the barrier's critical
  // path.  Pull it into L2 now, behind the input load, instead.
  if (tid == 0)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(w_counts) : "memory");
  if (tid == FK_WARP)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(w_prefix) : "memory");

  // Snapshot the epoch in a register, not shared memory: parking it in shared
  // would put this cold load on the critical path of the __syncthreads() below
  // even though only the spin needs it.
  const int my_epoch =
      (tid == 0) ? *reinterpret_cast<volatile int32_t*>(w_epoch) : 0;
  for (int e = tid; e < num_experts; e += THREADS) s_hist[e] = 0;
  __syncthreads();

  // ---- phase A ----------------------------------------------------------
  hist_range<THREADS, UNROLL, VEC4>(topk_ids, start, end, s_hist);
  __syncthreads();

  // Publish the slice histogram; atomicAdd hands back this block's
  // arrival-order exclusive prefix inside each expert.
  for (int e = tid; e < num_experts; e += THREADS)
    s_hist[e] = atomicAdd(&w_counts[e], s_hist[e]);

  // ---- grid barrier -----------------------------------------------------
  // Phase A only wrote through atomics, which are coherent at L2, so the release
  // needs no __threadfence() (which measured ~1 us here).
  __syncthreads();
  if (tid == 0)
    s_is_last = (atomicAdd(w_arrive, 1) == static_cast<int>(gridDim.x) - 1);
  __syncthreads();

  if (s_is_last) {
    // atomicExch to read: device scope, so no stale L1 line, and it leaves the
    // counter zeroed for the next launch in one instruction.  Publishing with
    // atomicExch likewise avoids a release fence.
    const int total = padded_prefix_scan<THREADS>(
        num_experts, block_size, s_warp,
        [&](int e) { return atomicExch(&w_counts[e], 0); },
        [&](int e, int excl, int cnt) {
          atomicExch(&w_prefix[e], excl);
          atomicExch(&w_gap[e], excl + cnt);
        });
    if (tid == 0) {
      atomicExch(&w_prefix[num_experts], total);
      *num_tokens_post_pad = total;
      atomicExch(w_arrive, 0);
    }
    __syncthreads();
    if (tid == 0) atomicAdd(w_epoch, 1);
  } else if (tid == 0) {
    if (spin_sleep_ns > 0) {
      while (*reinterpret_cast<volatile int32_t*>(w_epoch) == my_epoch)
        __nanosleep(spin_sleep_ns);
    } else {
      while (*reinterpret_cast<volatile int32_t*>(w_epoch) == my_epoch)
        ;
    }
  }
  __syncthreads();

  // ---- phase B ----------------------------------------------------------
  // prefix[E+1] and gap[E] are adjacent in the workspace, so pull them in as one
  // contiguous coalesced run.  __ldcg bypasses L1, so a stale line cannot shadow
  // the scan block's device-scope writes.
  for (int i = tid; i < 2 * num_experts + 1; i += THREADS)
    s_pref[i] = __ldcg(w_prefix + i);
  __syncthreads();
  for (int e = tid; e < num_experts; e += THREADS) s_hist[e] += s_pref[e];
  const int total = s_pref[num_experts];
  __syncthreads();

  write_expert_ids<THREADS>(s_pref, expert_ids, num_experts, block_size,
                            total / block_size, gtid, stride);
  // The fused path always runs with grid >= num_experts, so one expert per block
  // with the whole block on it beats splitting it per warp.
  fill_padding<THREADS, false>(sorted_token_ids, s_pref, s_gap, num_experts,
                               numel, total, max_padded, gtid, stride);
  scatter_range<THREADS, UNROLL, VEC4>(topk_ids, sorted_token_ids, s_hist,
                                       start, end);
}

__global__ void moe_align_nop_kernel(int32_t* out) {
  if (threadIdx.x == 1024) out[0] = 1;
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
int env_int(const char* name, int fallback) {
  const char* v = getenv(name);
  return (v && *v) ? atoi(v) : fallback;
}

// Swept values are in ITERATIONS.md; these are the winners.  Block sizes and the
// unroll depth are compile-time (they pick the instantiation), the rest are
// read once from the environment so a sweep needs no rebuild.
constexpr int kRedThreads = 128;   // 128 beat 256 and 512 consistently
constexpr int kFusedThreads = 256;
constexpr int kUnroll = 4;         // 8 was within noise, 2 was worse

struct Tuning {
  int red_max;     // barrier-free redundant kernel while numel <= this
  int red_grid;    // more than ~64 is flat: it does not shrink the per-block work
  int repl;        // histogram replicas in the redundant kernel
  int fused_grid;  // 0 = derive from the work; the SM count measured best
  int count_per_thread;
  int fill_per_thread;
  int spin_sleep_ns;
  Tuning()
      : red_max(env_int("FK_MOE_RED_MAX", 16384)),
        repl(env_int("FK_MOE_REPL", 2)),
        red_grid(env_int("FK_MOE_RED_GRID", 64)),
        fused_grid(env_int("FK_MOE_GRID", 148)),
        count_per_thread(env_int("FK_MOE_CPT", 4)),
        fill_per_thread(env_int("FK_MOE_FPT", 4)),
        spin_sleep_ns(env_int("FK_MOE_SLEEP", 32)) {}
};

const Tuning& tuning() {
  static Tuning t;
  return t;
}

int resident_blocks(const void* fn, int threads, int smem) {
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, threads, smem);
  if (per_sm < 1) per_sm = 1;
  return at::cuda::getCurrentDeviceProperties()->multiProcessorCount * per_sm;
}

// vec4 is the only runtime-selected template argument.
#define FK_BY_VEC(CALL) \
  do {                  \
    if (vec4)           \
      CALL(true);       \
    else                \
      CALL(false);      \
  } while (0)

#define FK_LAUNCH_RED_1(V, R)                                             \
  moe_align_redundant_kernel<kRedThreads, kUnroll, V, R>                   \
      <<<grid, kRedThreads, red_smem, stream>>>(                           \
          topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad,     \
          num_experts, block_size, numel, max_padded)

#define FK_LAUNCH_RED(V)     \
  do {                       \
    if (repl >= 4)           \
      FK_LAUNCH_RED_1(V, 4); \
    else if (repl >= 2)      \
      FK_LAUNCH_RED_1(V, 2); \
    else                     \
      FK_LAUNCH_RED_1(V, 1); \
  } while (0)

#define FK_LAUNCH_FUSED(V)                                                 \
  do {                                                                     \
    const auto fn = moe_align_fused_kernel<kFusedThreads, kUnroll, V>;      \
    static int cached = -1;                                                 \
    static int cap = 1;                                                     \
    if (smem != cached) {                                                   \
      cap = resident_blocks(reinterpret_cast<const void*>(fn),              \
                            kFusedThreads, smem);                           \
      cached = smem;                                                        \
    }                                                                       \
    const int g = grid > cap ? cap : grid;                                  \
    fn<<<g, kFusedThreads, smem, stream>>>(                                 \
        topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, ws,     \
        num_experts, block_size, numel, max_padded, t.spin_sleep_ns);        \
  } while (0)

void launch_align(
    const int32_t* topk_ids,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* num_tokens_post_pad,
    int32_t* ws,
    int num_experts,
    int block_size,
    int numel,
    int max_padded,
    cudaStream_t stream) {
  const Tuning& t = tuning();
  const int smem = (3 * num_experts + 1) * static_cast<int>(sizeof(int32_t));
  const bool vec4 =
      (numel % 4 == 0) && (reinterpret_cast<uintptr_t>(topk_ids) % 16 == 0);

  if (numel <= t.red_max) {
    const int repl = t.repl;
    const int red_smem =
        (3 * num_experts + 1 + 2 * repl * (num_experts + 1)) *
        static_cast<int>(sizeof(int32_t));
    int grid = t.red_grid;
    const int useful = (numel + kRedThreads - 1) / kRedThreads;
    if (grid > useful) grid = useful;
    if (grid < 1) grid = 1;
    FK_BY_VEC(FK_LAUNCH_RED);
    return;
  }

  int grid = t.fused_grid;
  if (grid <= 0) {
    const int per_block_count = kFusedThreads * t.count_per_thread;
    const int per_block_fill = kFusedThreads * t.fill_per_thread;
    grid = (numel + per_block_count - 1) / per_block_count;
    const int fg = (max_padded - numel + per_block_fill - 1) / per_block_fill;
    if (fg > grid) grid = fg;
    if (grid < 1) grid = 1;
  }
  FK_BY_VEC(FK_LAUNCH_FUSED);
}

}  // namespace

void moe_align_fast(
    torch::Tensor topk_ids,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor ws,
    int64_t num_experts,
    int64_t block_size) {
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
              "topk_ids must be int32");
  // Both kernels hold 3*E+1 ints of shared state (plus the redundant kernel's
  // replicas); past that a launch would fail without the opt-in attribute, so
  // fail loudly here instead.
  TORCH_CHECK((3 * num_experts + 1 + 4 * (num_experts + 1)) * 4 <= 48 * 1024,
              "num_experts too large for this kernel's shared memory: ",
              num_experts);
  launch_align(topk_ids.data_ptr<int32_t>(),
               sorted_token_ids.data_ptr<int32_t>(),
               expert_ids.data_ptr<int32_t>(),
               num_tokens_post_pad.data_ptr<int32_t>(), ws.data_ptr<int32_t>(),
               static_cast<int>(num_experts), static_cast<int>(block_size),
               static_cast<int>(topk_ids.numel()),
               static_cast<int>(sorted_token_ids.numel()),
               at::cuda::getCurrentCUDAStream());
}

// Calibration only: cost of getting an empty kernel onto the GPU under the same
// cold-L2 conditions the harness measures in.
void moe_align_nop(torch::Tensor out, int64_t threads) {
  moe_align_nop_kernel<<<1, static_cast<int>(threads), 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<int32_t>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_fast", &moe_align_fast, "MoE align, one launch (CUDA)");
  m.def("moe_align_nop", &moe_align_nop, "empty kernel (calibration)");
}
