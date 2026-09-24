// Fused MSA feature embedding: one kernel for AF3 Algorithm 8, lines 1-4.
//
// The module this replaces is five host operations -- a `torch.cat` of three
// tensors, two bias-free `F.linear`s, an `unsqueeze` and a broadcast add -- which
// issue four kernel launches and two intermediates. None of that is arithmetic:
// the captured shape is 740 K MACs over ~100 KB, which a B200 does in noise. What
// it costs is host issue time.
//
// Measured inside a faithful reproduction of the harness's own timed window
// (profile/measure_costs.py, which drives bench.py's `_collect_cases` /
// `_make_call` / `_time_module`): a bare aten launch adds ~2-4 us of window and
// one `F.linear` adds ~14 us, the extra ~10 us being cuBLAS's host-side heuristic
// selection -- which two GEMMs this small should never see. The baseline module
// adds ~40 us on top of a ~34 us floor that is the harness re-copying the
// forward's nine tensor leaves, a floor both arms pay. Collapsing all five
// operations into one kernel behind one pybind crossing is the whole
// optimization.
//
// Restructured, with C = msa.size(-1), M = c_m, Ks = c_s_input:
//
//   m[b,s,t,n] = sum_c msa[b,s,t,c] * Wm[n,c]                     (feature proj)
//              + hd[b,s,t] * Wm[n,C] + dv[b,s,t] * Wm[n,C+1]
//              + sum_j s_input[b,t,j] * Ws[n,j]                   (token proj)
//
// Three things fall out, and they remove work rather than fusing it. The
// concatenated feature vector is never materialized: `has_deletion` and
// `deletion_value` are columns C and C+1, so they enter as two extra
// scalar-times-weight-row terms. The token projection carries no `s` index, so it
// is computed once per token and reused across all S rows -- the Ks-long
// reduction runs over B*T tokens, not B*S*T rows. And the broadcast add is the
// epilogue: the feature projection never reaches memory, the store is the sum.
//
// Layouts this file does not claim are reported in-band by `plan()` below --
// `msa_module_embed` returns an undefined `at::Tensor`, which reaches Python as
// `None`, and the caller routes to the reference forward. There is deliberately
// no exception caught around a launch: a launch that starts has to be a launch
// that is correct.

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kWarp = 32;
constexpr int kMaxBlock = 1024;

// Launch geometry. A block owns one (b, t) token and a contiguous group of
// `n_group` output channels, so the grid is (B*T, ceil(M / n_group)) and one warp
// drives one channel's Ks-long reduction.
//
// Splitting the channel axis across blocks rather than giving one block the whole
// M is what makes this kernel fill the device, and it costs nothing in duplicated
// arithmetic: p[n] is needed only by the block that owns n, so no reduction is
// recomputed. Only the two staging reads (the token vector, and the feature rows)
// are repeated per channel group.
//
// One block per (b, t) was the obvious shape and it was the slow one: on the
// captured case that is 16 blocks of 256 threads, i.e. ~1 warp per SM across 148
// SMs, and each warp then had to cover 8 channels x 15 strided loads of `Ws` with
// nothing to hide the latency behind. Measured 12.7 us of device time for a
// kernel whose arithmetic is ~40 MACs per available lane. Splitting the channels
// raised the captured case to 128 blocks and 5.06 us. Both constants come from
// sweeping `msa_module_embed_tuned` -- see profile/tune_geometry.py.
constexpr int kDefaultBlock = 256;
constexpr int kDefaultNGroup = 8;

// Rows of the S x M output slab staged at once. Bounded, and that is the point:
// staging the whole slab is what would make the shared-memory requirement scale
// with MSA depth, and the real model path reaches S = 512, which at C + 2 = 34
// would ask for ~69 KB. Tiling removes the S term from the byte formula entirely,
// so no bound on S is needed and the kernel stays usable at full depth.
constexpr int kTileRows = 16;

// CUDA's maximum extent of the second grid dimension. The channel groups ride on
// grid.y, so a channel count above 65535 * n_group is outside the claimed domain
// and declines rather than building an invalid launch -- the launch check would
// otherwise raise out of the entry point, which is exactly what the in-band
// decline exists to avoid. At the default group width that bound is M > 524280,
// which no reachable configuration approaches (the real path runs M = 64).
constexpr int64_t kMaxGridY = 65535;

// The default per-block dynamic shared-memory limit, carried as a float count
// rather than a byte count so admission can subtract terms from a budget instead
// of forming a product and comparing it afterwards. Asserted rather than opted
// past: nothing in candidate/L1/ calls cudaFuncSetAttribute, and a configuration
// that does not fit is a configuration to decline, not to widen the limit for.
constexpr int64_t kMaxSmemBytes = 48 * 1024;
constexpr int64_t kMaxSmemFloats = kMaxSmemBytes / static_cast<int64_t>(sizeof(float));

// Row stride of the staged transposed weight: the smallest *odd* value that is at
// least the channel-group width. The staging loop walks consecutive c for one n,
// so consecutive lanes write `stride` floats apart; a stride sharing a factor with
// the 32-bank count serializes them, and a width that is itself a multiple of 32
// puts all 32 lanes in one bank. Only an odd stride is coprime with 32 for every
// width -- `width + 1` is not, since it is even whenever the width is odd. `| 1`
// leaves an odd width alone (already coprime, and already wide enough not to
// overlap) and rounds an even one up by one. The unit-stride read in the
// projection loop is unaffected either way; ncu measures that read at 0 bank
// conflicts and 1.000 wavefronts per instruction.
__host__ __device__ inline int staged_wm_stride(int n_group) { return n_group | 1; }

// Independent `Ws` loads issued per warp iteration on the generic path, where the
// reduction length is a runtime value and the trip count cannot be unrolled. The
// four offsets a warp interleaves are independent addresses, so this is four loads
// in flight instead of one, and it splits the FMA chain into four.
constexpr int kNUnroll = 4;

// The exact shape the compile-time specialization is built for -- the captured
// variant, and nothing else. It is deliberately the whole tuple rather than the
// family `(C + 2 == 34, Ks == 449, S <= tile)`: only this tuple is scored, only it
// is validated by `validate.py`, and admitting neighbours to a separately compiled
// arm means shipping an instantiation that nothing measures. Everything else takes
// the generic arm, which is written for arbitrary extents. See the comment on
// `msa_embed_kernel` for what the specialization buys.
constexpr int kSpecB = 1;
constexpr int kSpecS = 8;
constexpr int kSpecT = 16;
constexpr int kSpecC = 32;
constexpr int kSpecM = 64;
constexpr int kSpecKs = 449;                       // c_s_input
constexpr int kSpecFeatCols = kSpecC + 2;          // 32 msa channels + 2 deletion columns
static_assert(kSpecS <= kTileRows,
              "the specialization assumes the whole MSA depth fits one staging tile");

// Element traits. bf16 only, and the omission is the allow-list: fp32 cannot be
// admitted because PyTorch's fp32 GEMM reference routes to a TF32 kernel whose
// own deviation (~1.3e-2) already exceeds the fp32 tolerance band, so an exact
// fp32 kernel cannot match the thing it is compared against (measured in
// candidate/L1/linear.py). fp16 shares bf16's tolerance band but has no probe of
// its own, and an unmeasured claim is not a claim. Adding either means adding a
// specialization here, a dispatch arm below, and the measurement first.
template <typename T>
struct ElemTraits;

template <>
struct ElemTraits<__nv_bfloat16> {
  __device__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16_rn(x); }
};

// ---------------------------------------------------------------------------
// The kernel. blockIdx.x is the (b, t) token; blockIdx.y is the channel group.
// The block produces m[b, :, t, n0 : n0 + ng].
//
// `FEAT` and `KS` are the feature width and reduction length when known at compile
// time, 0 when they are runtime values; `SINGLE_TILE` says the whole MSA depth
// fits one staging tile. The scored shape is instantiated with all three fixed,
// which buys three things the generic instantiation cannot have:
//
//   * The Ks-long reduction splits into a fixed count of rounds where every lane
//     is valid plus one predicated tail, so it unrolls completely and issues ~14
//     independent loads per lane instead of the generic path's 4.
//     `long_scoreboard` -- waiting on global loads -- is 43.5% of this kernel's
//     stall cycles, so memory-level parallelism is the dominant lever.
//   * The 34-column feature loop unrolls completely.
//   * Two of the four barriers disappear. With one tile the feature rows can be
//     staged alongside `s_input` and `Wm` *before* the token projection instead of
//     after it, so their load latency overlaps the reduction, and the end-of-tile
//     barrier has nothing left to protect.
//
// Everything else -- the indexing, the staging layout, the accumulation order --
// is shared. The accumulator grouping is deliberately identical in both arms
// (`lane + 32k` accumulates into `a[k % 4]`), so the two instantiations sum in the
// same order and agree bit for bit.
// ---------------------------------------------------------------------------
template <typename scalar_t, int FEAT, int KS, bool SINGLE_TILE>
__global__ void msa_embed_kernel(scalar_t* __restrict__ out,
                                 const scalar_t* __restrict__ msa,
                                 const scalar_t* __restrict__ has_deletion,
                                 const scalar_t* __restrict__ deletion_value,
                                 const scalar_t* __restrict__ s_input,
                                 const scalar_t* __restrict__ wm,
                                 const scalar_t* __restrict__ ws,
                                 int S, int T, int C, int M, int Ks,
                                 int n_group, int tile_rows) {
  using Traits = ElemTraits<scalar_t>;

  const int feat_cols = FEAT ? FEAT : (C + 2);
  const int ks = KS ? KS : Ks;
  const int n0 = blockIdx.y * n_group;
  const int ng = min(n_group, M - n0);
  const int wm_stride = staged_wm_stride(n_group);

  // One dynamic allocation, partitioned. Everything staged is staged as fp32
  // because every value is read many times and the accumulation is fp32 anyway,
  // so the upcast is paid once per value instead of once per use.
  //
  // The token vector is staged only on the generic path. Staging it is what forces
  // the reduction to wait on a barrier -- it cannot start until every thread's
  // share of `s_input` has landed in shared memory -- and that barrier is what
  // splits the kernel's global reads into two *dependent* round trips. Reading
  // `s_input` straight from global instead costs a warp-contiguous, L1-resident
  // reload per channel and buys a single round trip and a single barrier. Only the
  // specialization takes that trade, because it is the arm whose reduction length
  // is fixed and therefore fully unrolled; see the barrier structure below.
  const int sin_floats = KS ? 0 : ks;
  extern __shared__ float smem[];
  float* sin_sh = smem;                                  // [sin_floats]
  float* p_sh = sin_sh + sin_floats;                     // [n_group]
  float* wmT_sh = p_sh + n_group;                        // [feat_cols][wm_stride]
  float* feat_sh = wmT_sh + feat_cols * wm_stride;       // [tile_rows][feat_cols]

  const int b = blockIdx.x / T;
  const int t = blockIdx.x - b * T;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Stage one tile of feature rows: msa channels, then the two deletion columns.
  auto stage_features = [&](int s0, int rows) {
    for (int i = tid; i < rows * feat_cols; i += nthreads) {
      const int r = i / feat_cols;
      const int c = i - r * feat_cols;
      const int64_t row = (static_cast<int64_t>(b) * S + (s0 + r)) * T + t;
      float v;
      if (c < C) {
        v = Traits::to_f(msa[row * C + c]);
      } else if (c == C) {
        v = Traits::to_f(has_deletion[row]);
      } else {
        v = Traits::to_f(deletion_value[row]);
      }
      feat_sh[r * feat_cols + c] = v;
    }
  };

  // Project one tile of feature rows and add the token projection as the epilogue.
  // Flat over the rows x ng slab, so every (r, n) is assigned exactly once for any
  // ng. Deriving n as tid % ng and striding by blockDim / ng instead is only
  // correct when ng divides blockDim: at blockDim = 256 and ng = 96 the stride
  // collapses to 2 while threads 0 and 96 both target n = 0 of different rows --
  // duplicated, unsynchronized stores.
  auto project_and_store = [&](int s0, int rows) {
    for (int idx = tid; idx < rows * ng; idx += nthreads) {
      const int r = idx / ng;
      const int n = idx - r * ng;
      const float* f = feat_sh + r * feat_cols;
      const float* w = wmT_sh + n;
      float acc = p_sh[n];
      if constexpr (FEAT) {
#pragma unroll
        for (int c = 0; c < FEAT; ++c) acc = fmaf(f[c], w[c * wm_stride], acc);
      } else {
        for (int c = 0; c < feat_cols; ++c) acc = fmaf(f[c], w[c * wm_stride], acc);
      }
      // The only rounding in the whole computation. The baseline rounds the two
      // projections to bf16 separately and rounds their sum again; accumulating
      // both in fp32 and rounding once is closer to exact -- measured against an
      // fp64 reference at half the baseline's maximum error -- and was checked
      // against the harness's own comparator at matched = 1.0, max_abs = 7.8e-3
      // against atol = 1e-2 (profile/numerics_probe.py).
      out[(static_cast<int64_t>(b) * S + (s0 + r)) * T * M +
          static_cast<int64_t>(t) * M + (n0 + n)] = Traits::from_f(acc);
    }
  };

  // -- staging -------------------------------------------------------------
  // The token vector, upcast once and then read ng times out of shared memory.
  // Generic path only -- the specialization reads it from global in the reduction.
  if constexpr (!KS) {
    const scalar_t* src = s_input + (static_cast<int64_t>(b) * T + t) * ks;
    for (int j = tid; j < ks; j += nthreads) sin_sh[j] = Traits::to_f(src[j]);
  }

  // This group's columns of Wm, transposed on the way in. Transposed because in
  // the projection loop adjacent threads hold adjacent n, so wmT_sh[c][n] is then
  // read with unit stride; untransposed it would be a stride-(C+2) gather.
  // Walking the flat source index keeps the global read contiguous, and the padded
  // destination stride keeps the write off a single bank.
  for (int i = tid; i < feat_cols * ng; i += nthreads) {
    const int n = i / feat_cols;
    const int c = i - n * feat_cols;
    wmT_sh[c * wm_stride + n] =
        Traits::to_f(wm[static_cast<int64_t>(n0 + n) * feat_cols + c]);
  }
  // With one tile the feature rows are staged here rather than after the token
  // projection, so their global-load latency joins the same round trip as
  // everything else. `feat_sh` is disjoint from everything the reduction touches.
  if constexpr (SINGLE_TILE) stage_features(0, S);
  // The generic path has to publish `sin_sh` before its reduction can read it.
  // The specialization does not, so its one barrier moves *after* the reduction
  // and every global read in the kernel -- Wm, the feature rows, `s_input` and
  // `Ws` -- issues before it, in one round trip instead of two.
  if constexpr (!KS) __syncthreads();

  // -- token projection: p[n] = sum_j s_input[b,t,j] * Ws[n,j] --------------
  // One warp per output channel, reducing over j *within* the warp. That
  // orientation is what makes Ws readable as it lies: the 32 lanes of an iteration
  // read 32 consecutive j of one row, so no staging and no transpose of Ws is
  // needed. The reads are contiguous in the reduction index but not sector-aligned
  // at row boundaries -- at Ks = 449 successive rows start 898 bytes apart and
  // 898 mod 32 = 2, so 1 row in 16 starts on a sector boundary and the rest
  // straddle (ncu: 2.57 sectors per request against a 2.0 ideal for bf16).
  {
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int nwarps = nthreads / kWarp;
    const scalar_t* srow = s_input + (static_cast<int64_t>(b) * T + t) * ks;
    for (int n = warp; n < ng; n += nwarps) {
      const scalar_t* wsn = ws + static_cast<int64_t>(n0 + n) * ks;
      // `lane + 32k` accumulates into `a[k % 4]` in both arms, so the two
      // instantiations sum in the same order and agree bit for bit. Under the full
      // unroll below the index folds to a constant and the array stays in
      // registers.
      float a[kNUnroll] = {0.0f, 0.0f, 0.0f, 0.0f};
      if constexpr (KS) {
        // A fixed reduction length splits into rounds where every lane is valid
        // and one predicated tail, so the body carries no per-load predicate and
        // the compiler can issue every load before consuming any of them.
        constexpr int kFull = KS / kWarp;
#pragma unroll
        for (int k = 0; k < kFull; ++k) {
          const int j = lane + k * kWarp;
          // Both operands come from global here. The 32 lanes of one k read 32
          // consecutive elements of each row, so both reads are warp-contiguous,
          // and `srow` is shared by every channel in the block so it is L1-resident
          // after the first warp touches it.
          a[k % kNUnroll] =
              fmaf(Traits::to_f(srow[j]), Traits::to_f(wsn[j]), a[k % kNUnroll]);
        }
        // Whatever is left is shorter than a warp: Ks = 449 leaves exactly one
        // lane with a final element to add.
        const int jt = lane + kFull * kWarp;
        if (jt < KS) {
          a[kFull % kNUnroll] = fmaf(Traits::to_f(srow[jt]), Traits::to_f(wsn[jt]),
                                     a[kFull % kNUnroll]);
        }
      } else {
        // `k` is carried explicitly so the tail keeps accumulating into the same
        // `a[k % 4]` the unrolled body would have used. The main loop advances k by
        // kNUnroll, so `(k + u) % kNUnroll == u` inside it -- the four loads stay
        // unconditional and in flight, and the grouping still matches the arm
        // above element for element.
        int j = lane, k = 0;
        for (; j + (kNUnroll - 1) * kWarp < ks; j += kNUnroll * kWarp, k += kNUnroll) {
#pragma unroll
          for (int u = 0; u < kNUnroll; ++u) {
            const int ju = j + u * kWarp;
            a[u] = fmaf(sin_sh[ju], Traits::to_f(wsn[ju]), a[u]);
          }
        }
        // A tail shorter than the unrolled step is predicated by the loop bound
        // itself; Ks = 449 leaves the final round with one active lane.
        for (; j < ks; j += kWarp, ++k) {
          a[k % kNUnroll] = fmaf(sin_sh[j], Traits::to_f(wsn[j]), a[k % kNUnroll]);
        }
      }
      float acc = (a[0] + a[1]) + (a[2] + a[3]);
#pragma unroll
      for (int shift = kWarp / 2; shift > 0; shift >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, shift);
      }
      if (lane == 0) p_sh[n] = acc;
    }
  }
  __syncthreads();

  // -- feature projection, with the broadcast add as the epilogue -----------
  if constexpr (SINGLE_TILE) {
    // Features are already staged and p_sh is published, so this is the last
    // phase: nothing follows that the tile buffer would need protecting from.
    project_and_store(0, S);
  } else {
    for (int s0 = 0; s0 < S; s0 += tile_rows) {
      const int rows = min(tile_rows, S - s0);
      stage_features(s0, rows);
      __syncthreads();
      project_and_store(s0, rows);
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Zero-barrier variant for the scored tuple, kept for measurement.
//
// Every extent is a compile-time constant here, and one warp owns one output channel
// end to end: it reduces the token projection itself, keeps the result in a register
// broadcast across its lanes, and writes its own channel's rows. Nothing is shared
// between warps, so there is no shared memory and no `__syncthreads()` at all.
//
// The accumulation order is deliberately identical to the staged kernel's -- the same
// `a[k % 4]` grouping, the same `__shfl_down` tree, then a broadcast of lane 0's value
// rather than a store-and-reload -- so it is bit-identical to it.
//
// This is not the shipped path. It is slower, and it is retained because the reason
// is worth keeping: it trades one barrier for reading `Wm` and the feature rows
// redundantly per warp instead of once per block. The same trade measured in the
// opposite direction in `profile/step_budget.py`, where a no-shared-memory kernel
// that merely touches the operator's bytes costs *more* than the staged kernel that
// projects them. Shared-memory staging is buying more than the barrier costs.
template <typename scalar_t>
__global__ void msa_embed_kernel_warp_private(scalar_t* __restrict__ out,
                                              const scalar_t* __restrict__ msa,
                                              const scalar_t* __restrict__ has_deletion,
                                              const scalar_t* __restrict__ deletion_value,
                                              const scalar_t* __restrict__ s_input,
                                              const scalar_t* __restrict__ wm,
                                              const scalar_t* __restrict__ ws) {
  using Traits = ElemTraits<scalar_t>;
  constexpr int C = kSpecC, FEAT = kSpecFeatCols, KS = kSpecKs;
  constexpr int S = kSpecS, T = kSpecT, M = kSpecM;
  constexpr int kFull = KS / kWarp;

  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  const int n = blockIdx.y * (blockDim.x / kWarp) + warp;
  if (n >= M) return;
  const int b = blockIdx.x / T;
  const int t = blockIdx.x - b * T;

  const scalar_t* srow = s_input + (static_cast<int64_t>(b) * T + t) * KS;
  const scalar_t* wsn = ws + static_cast<int64_t>(n) * KS;
  float a[kNUnroll] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int k = 0; k < kFull; ++k) {
    const int j = lane + k * kWarp;
    a[k % kNUnroll] = fmaf(Traits::to_f(srow[j]), Traits::to_f(wsn[j]), a[k % kNUnroll]);
  }
  const int jt = lane + kFull * kWarp;
  if (jt < KS) {
    a[kFull % kNUnroll] =
        fmaf(Traits::to_f(srow[jt]), Traits::to_f(wsn[jt]), a[kFull % kNUnroll]);
  }
  float p = (a[0] + a[1]) + (a[2] + a[3]);
#pragma unroll
  for (int shift = kWarp / 2; shift > 0; shift >>= 1) {
    p += __shfl_down_sync(0xffffffffu, p, shift);
  }
  // Broadcast rather than publish: the reduced value never leaves the warp.
  p = __shfl_sync(0xffffffffu, p, 0);

  // Lane l owns MSA row l. All lanes read the same `Wm` element each step, which the
  // L1 broadcasts; the feature reads are per-lane and strided.
  const scalar_t* wmn = wm + static_cast<int64_t>(n) * FEAT;
  for (int s = lane; s < S; s += kWarp) {
    const int64_t row = (static_cast<int64_t>(b) * S + s) * T + t;
    float acc = p;
#pragma unroll
    for (int c = 0; c < FEAT; ++c) {
      const float f = (c < C) ? Traits::to_f(msa[row * C + c])
                              : (c == C ? Traits::to_f(has_deletion[row])
                                        : Traits::to_f(deletion_value[row]));
      acc = fmaf(f, Traits::to_f(wmn[c]), acc);
    }
    out[row * M + n] = Traits::from_f(acc);
  }
}

// ---------------------------------------------------------------------------
// Host-side admission. Every predicate guards something the kernel relies on,
// and all of them run before anything is allocated or launched.
//
// The ordering is load-bearing, not stylistic. Every derived quantity -- `C + 2`,
// the channel-tile count, the grid product, the staging budget -- is formed only
// after the extents it is built from have been bounded, and each is written in a
// form that cannot overflow rather than one that is checked after the fact.
// Signed overflow is undefined behaviour, so a check that runs on an
// already-overflowed value is not a check. This is concrete, not theoretical: a
// CUDA bf16 tensor of shape [0, 0, 0, INT64_MAX] is constructible because it holds
// no elements, and it reaches the `wm.size(1) != C + 2` relation.
// ---------------------------------------------------------------------------

struct Plan {
  bool ok = false;
  bool specialized = false;
  int64_t B = 0, S = 0, T = 0, C = 0, M = 0, Ks = 0;
  int block = 0;
  int n_group = 0;
  int64_t n_tiles = 0;
  int tile_rows = 0;
  int64_t smem_bytes = 0;
};

bool dtype_supported(at::ScalarType t) { return t == at::kBFloat16; }

// A raw kernel writing into a fresh tensor builds no graph, so a grad-requiring
// call has to take the differentiable ATen path. The two autograd predicates are
// separate concerns: requires_grad covers reverse mode, and a forward-mode dual
// carries a tangent while reporting requires_grad() == false, so it needs its own
// check. A lazily-applied negative bit leaves un-negated values in storage, which
// a raw pointer read would consume with the wrong signs.
bool operand_ok(const at::Tensor& x, const at::Tensor& ref) {
  if (!x.defined() || !x.is_cuda() || x.layout() != at::kStrided) return false;
  if (x.scalar_type() != ref.scalar_type() || x.device() != ref.device()) return false;
  if (!x.is_contiguous() || x.is_neg()) return false;
  if (x.requires_grad() && c10::GradMode::is_enabled()) return false;
  if (x._fw_grad(/*level=*/0).defined()) return false;
  return true;
}

// True when `a * b` is representable in int64_t, for non-negative a and b. Compares
// by division so the product is never formed unless it is known to fit: signed
// overflow is undefined behaviour, so forming it and checking afterwards is not a
// check at all.
constexpr bool mul_fits(int64_t a, int64_t b) {
  return a == 0 || b == 0 || a <= std::numeric_limits<int64_t>::max() / b;
}

// Pinned at compile time, including one tuple that does *not* fit -- the
// materialized six-tensor operand set at that bound is far larger than any device's
// memory, so the helper is what gets tested, not an allocation.
static_assert(mul_fits(0, std::numeric_limits<int64_t>::max()), "zero always fits");
static_assert(mul_fits(std::numeric_limits<int64_t>::max(), 1), "identity fits");
static_assert(mul_fits(1LL << 31, 1LL << 31), "2^62 fits");
static_assert(!mul_fits(1LL << 32, 1LL << 31), "2^63 does not fit");
static_assert(!mul_fits(std::numeric_limits<int64_t>::max(), 2), "2x the max cannot fit");
// The scored output, and an output count that cannot be represented.
static_assert(mul_fits(static_cast<int64_t>(kSpecB) * kSpecT * kSpecS, kSpecM),
              "the scored output element count must be representable");
static_assert(!mul_fits((1LL << 40) * (1LL << 20), 1LL << 10),
              "a 2^70 output element count must be rejected");

// Subtract one staging term from the remaining float budget, declining if it does
// not fit.
bool take_budget(int64_t& budget, int64_t count) {
  if (count < 0 || count > budget) return false;
  budget -= count;
  return true;
}

// The same for a two-dimensional term, comparing by division before multiplying so
// the product is never formed unless it is known to fit.
bool take_budget_product(int64_t& budget, int64_t rows, int64_t cols) {
  if (rows < 0 || cols < 0) return false;
  if (rows == 0 || cols == 0) return true;
  if (rows > budget / cols) return false;   // cols > 0 here, so this is safe
  return take_budget(budget, rows * cols);
}

Plan plan(const at::Tensor& msa, const at::Tensor& has_deletion,
          const at::Tensor& deletion_value, const at::Tensor& s_input,
          const at::Tensor& wm, const at::Tensor& ws, int64_t block,
          int64_t n_group, int64_t variant) {
  Plan p;
  const int64_t kIntMax = std::numeric_limits<int>::max();

  // Geometry first: it is caller-supplied on the tuning entry point, and the
  // derived quantities below divide by and multiply with it.
  if (block <= 0 || block > kMaxBlock || block % kWarp != 0) return p;
  if (n_group <= 0 || n_group > kIntMax) return p;

  if (!msa.defined() || !msa.is_cuda() || !dtype_supported(msa.scalar_type())) return p;
  // Autocast rewrites the reference path's `F.linear` to the autocast dtype, so
  // under fp16 autocast the baseline returns fp16 while this kernel would return
  // bf16 -- a dtype and a precision mismatch, measured. The cast belongs to the
  // ATen ops that implement it, so an autocast-active call is the reference path's
  // to serve. This is ambient dispatcher state rather than a property of any
  // operand, so it cannot live in the per-operand loop below.
  if (at::autocast::is_autocast_enabled(msa.device().type())) return p;
  for (const at::Tensor* x : {&msa, &has_deletion, &deletion_value, &s_input, &wm, &ws}) {
    if (!operand_ok(*x, msa)) return p;
  }
  if (msa.dim() != 4 || has_deletion.dim() != 3 || deletion_value.dim() != 3 ||
      s_input.dim() != 3 || wm.dim() != 2 || ws.dim() != 2) {
    return p;
  }

  // Base extents, straight off the tensors. Sizes are non-negative for any valid
  // tensor; the guard states the assumption the arithmetic below rests on.
  const int64_t B = msa.size(0), S = msa.size(1), T = msa.size(2), C = msa.size(3);
  const int64_t M = wm.size(0), Ks = ws.size(1);
  if (B < 0 || S < 0 || T < 0 || C < 0 || M < 0 || Ks < 0) return p;

  // Bound every extent *before* forming anything from it.
  //   - S carries staging-tile headroom: the row loop advances by the tile height
  //     in int arithmetic, so at S == INT_MAX the final increment would overflow
  //     and the loop would not terminate. C == 0 makes such a shape cheap to build.
  //   - C carries headroom for the two deletion columns, so `C + 2` is then safe.
  if (S > kIntMax - kTileRows) return p;
  if (C > kIntMax - 2) return p;
  if (T > kIntMax || M > kIntMax || Ks > kIntMax) return p;

  // Only now is the feature width a safe expression to form.
  const int64_t feat_cols = C + 2;
  if (wm.size(1) != feat_cols || ws.size(0) != M) return p;
  if (has_deletion.size(0) != B || has_deletion.size(1) != S ||
      has_deletion.size(2) != T) {
    return p;
  }
  if (deletion_value.size(0) != B || deletion_value.size(1) != S ||
      deletion_value.size(2) != T) {
    return p;
  }
  if (s_input.size(0) != B || s_input.size(1) != T || s_input.size(2) != Ks) return p;

  // Grid extents. grid.x is the token count and grid.y the channel-tile count;
  // the product is bounded by division rather than formed and then checked, and
  // the ceiling is taken with a remainder rather than by adding a divisor.
  if (T != 0 && B > kIntMax / T) return p;
  const int64_t tokens = B * T;
  const int64_t n_tiles = M / n_group + (M % n_group != 0 ? 1 : 0);
  if (n_tiles > kMaxGridY) return p;

  // The output holds `tokens * S * M` elements. That count is the numel handed to
  // the allocator below, and it is also the range of the kernel's int64 output
  // index. Neither is bounded by any input tensor -- an input `numel` is
  // representable by construction, but the *output's* is a product this function
  // invents -- so it has to be proven here, before anything is allocated. An
  // overflowing tuple is otherwise inside the numeric domain this function claims,
  // and the entry point would raise out of the allocator rather than decline
  // in-band.
  if (!mul_fits(tokens, S)) return p;
  const int64_t out_rows = tokens * S;
  if (!mul_fits(out_rows, M)) return p;

  // Staging budget, subtracted term by term in floats. `ng_alloc` is what the
  // kernel actually indexes with -- the host passes it as `n_group` -- so the
  // device partitioning and this budget agree by construction rather than by
  // coincidence.
  const int64_t ng_alloc = std::min<int64_t>(n_group, std::max<int64_t>(M, 1));
  const int64_t tile_rows = std::min<int64_t>(S, kTileRows);
  // The specialization is another instantiation of this same kernel, eligible only
  // for the exact captured tuple, and only once every predicate above has passed.
  // It is decided here rather than at the end because it changes the staging
  // layout: it reads the token vector from global instead of staging it, so it asks
  // for `Ks` fewer floats. The budget below and the device-side partition are
  // parameterized by the same flag, so they cannot drift apart.
  const bool specialized = variant != 0 && B == kSpecB && S == kSpecS &&
                           T == kSpecT && C == kSpecC && M == kSpecM &&
                           Ks == kSpecKs;

  int64_t budget = kMaxSmemFloats;
  if (!take_budget(budget, specialized ? 0 : Ks)) return p;                // sin_sh
  if (!take_budget(budget, ng_alloc)) return p;                            // p_sh
  if (!take_budget_product(budget, feat_cols, ng_alloc | 1)) return p;     // wmT_sh
  if (!take_budget_product(budget, tile_rows, feat_cols)) return p;        // feat_sh

  p.ok = true;
  p.specialized = specialized;
  p.B = B; p.S = S; p.T = T; p.C = C; p.M = M; p.Ks = Ks;
  p.block = static_cast<int>(block);
  p.n_group = static_cast<int>(ng_alloc);
  p.n_tiles = n_tiles;
  p.tile_rows = static_cast<int>(tile_rows);
  p.smem_bytes = (kMaxSmemFloats - budget) * static_cast<int64_t>(sizeof(float));
  return p;
}

at::Tensor run(const at::Tensor& msa, const at::Tensor& has_deletion,
               const at::Tensor& deletion_value, const at::Tensor& s_input,
               const at::Tensor& wm, const at::Tensor& ws, int64_t block,
               int64_t n_group, int64_t variant) {
  const Plan p = plan(msa, has_deletion, deletion_value, s_input, wm, ws, block,
                      n_group, variant);
  if (!p.ok) return at::Tensor();

  const c10::cuda::CUDAGuard guard(msa.device());
  // Straight from the CUDA allocator rather than through the `at::empty`
  // dispatcher hop, which measured 0.26 us cheaper per call in
  // candidate/L1/rms_norm_kernels.cu. Safe only because this path is never reached
  // under tracing: the caller routes to pure PyTorch whenever the compiler is
  // active, so nothing here is ever traced or graph-captured.
  at::Tensor out = at::detail::empty_cuda({p.B, p.S, p.T, p.M}, msa.scalar_type(),
                                          msa.device(), c10::MemoryFormat::Contiguous);
  // A degenerate extent leaves nothing to compute, and B * T or n_tiles can
  // themselves be zero here -- an empty grid is illegal, so the shape is the whole
  // answer.
  if (out.numel() == 0) return out;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(p.B * p.T), static_cast<unsigned>(p.n_tiles));
  auto* out_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const auto* msa_p = reinterpret_cast<const __nv_bfloat16*>(msa.const_data_ptr());
  const auto* hd_p = reinterpret_cast<const __nv_bfloat16*>(has_deletion.const_data_ptr());
  const auto* dv_p =
      reinterpret_cast<const __nv_bfloat16*>(deletion_value.const_data_ptr());
  const auto* si_p = reinterpret_cast<const __nv_bfloat16*>(s_input.const_data_ptr());
  const auto* wm_p = reinterpret_cast<const __nv_bfloat16*>(wm.const_data_ptr());
  const auto* ws_p = reinterpret_cast<const __nv_bfloat16*>(ws.const_data_ptr());
  const int S = static_cast<int>(p.S), T = static_cast<int>(p.T);
  const int C = static_cast<int>(p.C), M = static_cast<int>(p.M);
  const int Ks = static_cast<int>(p.Ks);

  if (p.specialized && variant == 2) {
    // Measurement-only arm: one warp per channel, so the channel group has to be
    // exactly the warp count. Anything else is declined rather than mis-launched.
    if (p.n_group != p.block / kWarp) return at::Tensor();
    msa_embed_kernel_warp_private<__nv_bfloat16>
        <<<grid, p.block, 0, stream>>>(out_p, msa_p, hd_p, dv_p, si_p, wm_p, ws_p);
  } else if (p.specialized) {
    msa_embed_kernel<__nv_bfloat16, kSpecFeatCols, kSpecKs, true>
        <<<grid, p.block, p.smem_bytes, stream>>>(out_p, msa_p, hd_p, dv_p, si_p, wm_p,
                                                  ws_p, S, T, C, M, Ks, p.n_group,
                                                  p.tile_rows);
  } else {
    msa_embed_kernel<__nv_bfloat16, 0, 0, false>
        <<<grid, p.block, p.smem_bytes, stream>>>(out_p, msa_p, hd_p, dv_p, si_p, wm_p,
                                                  ws_p, S, T, C, M, Ks, p.n_group,
                                                  p.tile_rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

// Returns the embedded output, or an undefined tensor -- which reaches Python as
// None -- when the layout is not claimed. The caller uses that answer to route to
// the reference forward. One crossing does both the allocation and the launch: at
// several microseconds per host operation in the scored window, a separate
// `torch.empty` on the Python side would cost a measurable fraction of the whole
// addressable budget.
at::Tensor msa_module_embed(const at::Tensor& msa, const at::Tensor& has_deletion,
                            const at::Tensor& deletion_value, const at::Tensor& s_input,
                            const at::Tensor& wm, const at::Tensor& ws) {
  return run(msa, has_deletion, deletion_value, s_input, wm, ws, kDefaultBlock,
             kDefaultNGroup, /*variant=*/1);
}

// Launch-geometry and instantiation override, for the local configuration sweep
// only. The scored path goes through `msa_module_embed` and never pays for the extra
// arguments. `variant` selects the arm: 0 forces the generic instantiation onto a
// shape the specialization would otherwise claim (which is how the generic arm stays
// measured on the shape that matters), 1 is the shipped staged specialization, and 2
// is the zero-barrier warp-private variant kept for comparison.
at::Tensor msa_module_embed_tuned(const at::Tensor& msa, const at::Tensor& has_deletion,
                                  const at::Tensor& deletion_value,
                                  const at::Tensor& s_input, const at::Tensor& wm,
                                  const at::Tensor& ws, int64_t block, int64_t n_group,
                                  int64_t variant) {
  return run(msa, has_deletion, deletion_value, s_input, wm, ws, block, n_group, variant);
}

// The admission decision without the launch, for tests: whether the layout is
// claimed, the dimensions read off it, the geometry chosen, the staging budget it
// would ask for, and which instantiation it would run.
std::vector<int64_t> describe_claim(const at::Tensor& msa, const at::Tensor& has_deletion,
                                    const at::Tensor& deletion_value,
                                    const at::Tensor& s_input, const at::Tensor& wm,
                                    const at::Tensor& ws) {
  const Plan p = plan(msa, has_deletion, deletion_value, s_input, wm, ws, kDefaultBlock,
                      kDefaultNGroup, /*variant=*/1);
  return {p.ok ? 1 : 0, p.B,     p.S,          p.T,         p.C,
          p.M,          p.Ks,    p.block,      p.n_group,   p.n_tiles,
          p.tile_rows,  p.smem_bytes,          p.specialized ? 1 : 0};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("msa_module_embed", &msa_module_embed,
        "Fused MSA feature embedding; None when the layout is not claimed");
  m.def("msa_module_embed_tuned", &msa_module_embed_tuned,
        "msa_module_embed with a launch-geometry and instantiation override",
        pybind11::arg("msa"), pybind11::arg("has_deletion"),
        pybind11::arg("deletion_value"), pybind11::arg("s_input"), pybind11::arg("wm"),
        pybind11::arg("ws"), pybind11::arg("block"), pybind11::arg("n_group"),
        pybind11::arg("variant") = 1);
  m.def("describe_claim", &describe_claim,
        "Admission decision, dimensions, geometry, staging bytes and instantiation");
}
