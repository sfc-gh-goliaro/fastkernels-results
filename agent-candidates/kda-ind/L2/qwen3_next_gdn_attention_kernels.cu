// Fused post-recurrence stage for the Qwen3-Next GDN block.
//
// Replaces three launches -- the gated RMS norm, the fp32->bf16 narrowing of the
// recurrent state, and its scatter into the cache -- with one. The two pieces of
// work are independent, so a single grid carries both and branches on the block
// index, the same way the surrounding Triton kernels pack unrelated head ranges
// into one launch.
//
// The gate arrives as a strided view of the input-projection output rather than a
// compact tensor, so its row stride is an explicit argument. That is the whole
// reason this stage cannot simply call the gated-norm wrapper: flattening token
// and head on a strided gate is not expressible as a view.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>
#include <utility>

namespace {

constexpr int kHeadDim = 128;
constexpr int kWarp = 32;
// Eight bf16 per lane is a 16-byte load, so one head row is sixteen lanes and its
// reduction is half a warp -- no shared memory and no block barrier. Two bytes per
// lane instead left the kernel latency-bound at a third of achievable bandwidth.
constexpr int kVec = 8;
constexpr int kLanesPerRow = kHeadDim / kVec;  // 16
constexpr int kRowsPerBlock = 8;               // (token, head) rows per block
constexpr int kNormThreads = kLanesPerRow * kRowsPerBlock;  // 128
constexpr int kScatterThreads = kNormThreads;

static_assert(kHeadDim == kLanesPerRow * kVec, "lanes must tile one head row");
static_assert(kLanesPerRow <= kWarp, "a row's reduction must stay inside a warp");

struct alignas(16) BF16x8 {
  __nv_bfloat16 v[kVec];
};

struct alignas(16) F32x4 {
  float v[4];
};

struct alignas(8) BF16x4 {
  __nv_bfloat16 v[4];
};



// Sum across `Width` neighbouring lanes; every participating lane holds the result.
template <int Width>
__device__ __forceinline__ float lane_reduce_add(float v) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

// RMSNorm(o) * silu(z), sixteen lanes per (token, head) row, so one warp covers two.
//
// The association matters: the reference forms the gate first and then multiplies,
// ``y *= z * sigmoid(z)``, so a left-associated ``y * z * sigmoid(z)`` would not
// round identically. The reduction is in fp32 across the row's sixteen lanes and the
// result is rounded to bf16 exactly once, at the store.
__device__ void gated_norm_block(const __nv_bfloat16* __restrict__ o,
                                 const __nv_bfloat16* __restrict__ z,
                                 const float* __restrict__ weight,
                                 __nv_bfloat16* __restrict__ y, long z_row,
                                 long y_row, long tokens, int heads, float eps,
                                 long block_first_row) {
  const int row_in_block = threadIdx.x / kLanesPerRow;
  const int lane = threadIdx.x % kLanesPerRow;
  const long row = block_first_row + row_in_block;  // flat (token, head) index
  const long total_rows = tokens * static_cast<long>(heads);
  if (row >= total_rows) return;

  const long token = row / heads;
  const int head = static_cast<int>(row - token * heads);
  const int col = lane * kVec;

  const BF16x8 ov = *reinterpret_cast<const BF16x8*>(o + row * kHeadDim + col);
  const BF16x8 zv = *reinterpret_cast<const BF16x8*>(
      z + token * z_row + static_cast<long>(head) * kHeadDim + col);
  float wv[kVec];
  const float* wp = weight + static_cast<long>(head) * kHeadDim + col;
  *reinterpret_cast<F32x4*>(wv) = *reinterpret_cast<const F32x4*>(wp);
  *reinterpret_cast<F32x4*>(wv + 4) = *reinterpret_cast<const F32x4*>(wp + 4);

  float x[kVec];
  float partial = 0.0f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    x[i] = __bfloat162float(ov.v[i]);
    partial += x[i] * x[i];
  }
  const float mean_square =
      lane_reduce_add<kLanesPerRow>(partial) / static_cast<float>(kHeadDim);
  const float rstd = rsqrtf(mean_square + eps);

  BF16x8 out;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float zf = __bfloat162float(zv.v[i]);
    const float gate = zf / (1.0f + __expf(-zf));  // z * sigmoid(z)
    out.v[i] = __float2bfloat16(x[i] * rstd * wv[i] * gate);
  }
  *reinterpret_cast<BF16x8*>(y + token * y_row +
                             static_cast<long>(head) * kHeadDim + col) = out;
}

// Narrow the fp32 final state to the cache dtype and scatter it, reading each
// destination row index off the device so no host synchronization is introduced.
__device__ void state_scatter_block(const float* __restrict__ final_state,
                                    __nv_bfloat16* __restrict__ cache,
                                    const long* __restrict__ state_index,
                                    long per_sequence, long batch,
                                    long scatter_blocks, long scatter_block) {
  // per_sequence is a multiple of the head dim, so it is a multiple of four and
  // the vectorized walk never straddles a sequence boundary.
  const long vec_per_seq = per_sequence / 4;
  const long total = batch * vec_per_seq;
  const long stride = scatter_blocks * kScatterThreads;
  for (long i = scatter_block * kScatterThreads + threadIdx.x; i < total;
       i += stride) {
    const long seq = i / vec_per_seq;
    const long offset = (i - seq * vec_per_seq) * 4;
    const F32x4 src =
        *reinterpret_cast<const F32x4*>(final_state + seq * per_sequence + offset);
    BF16x4 out;
#pragma unroll
    for (int j = 0; j < 4; ++j) out.v[j] = __float2bfloat16(src.v[j]);
    *reinterpret_cast<BF16x4*>(
        cache + state_index[seq] * per_sequence + offset) = out;
  }
}

__global__ void gdn_post_recurrence_kernel(
    const __nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ z,
    const float* __restrict__ weight, __nv_bfloat16* __restrict__ y,
    const float* __restrict__ final_state, __nv_bfloat16* __restrict__ cache,
    const long* __restrict__ state_index, long z_row, long y_row, long tokens,
    int heads, float eps, long norm_blocks, long per_sequence, long batch,
    long scatter_blocks) {
  const long block = blockIdx.x;
  if (block < norm_blocks) {
    gated_norm_block(o, z, weight, y, z_row, y_row, tokens, heads, eps,
                     block * kRowsPerBlock);
  } else if (cache != nullptr) {
    state_scatter_block(final_state, cache, state_index, per_sequence, batch,
                        scatter_blocks, block - norm_blocks);
  }
}

}  // namespace

// ``o``/``y`` are compact ``[tokens, heads, 128]``; ``z`` carries an explicit row
// stride because it is a column slice of the projection output. Passing
// ``final_state``/``cache``/``state_index`` as undefined tensors runs the norm
// alone, which is what the decode path wants -- it updates the recurrent state
// in place inside its own kernel.
void gdn_post_recurrence(const at::Tensor& o, const at::Tensor& z,
                         const at::Tensor& weight, at::Tensor& y,
                         const std::optional<at::Tensor>& final_state,
                         const std::optional<at::Tensor>& cache,
                         const std::optional<at::Tensor>& state_index,
                         double eps) {
  TORCH_CHECK(o.is_cuda() && z.is_cuda() && y.is_cuda() && weight.is_cuda(),
              "expected CUDA tensors");
  TORCH_CHECK(z.device() == o.device() && y.device() == o.device() &&
                  weight.device() == o.device(),
              "every argument must share one device");
  TORCH_CHECK(o.scalar_type() == at::kBFloat16, "o must be bfloat16");
  TORCH_CHECK(z.scalar_type() == at::kBFloat16, "z must be bfloat16");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16, "y must be bfloat16");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "norm weight must be float32");
  TORCH_CHECK(o.dim() == 3 && o.size(2) == kHeadDim, "o must be [T, H, 128]");
  TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
  TORCH_CHECK(z.stride(1) == 1 && y.stride(1) == 1, "z and y need unit column stride");

  const long tokens = o.size(0);
  const int heads = static_cast<int>(o.size(1));
  TORCH_CHECK(weight.numel() == static_cast<long>(heads) * kHeadDim,
              "norm weight must be replicated once per head");
  TORCH_CHECK(z.size(0) == tokens && z.size(1) == static_cast<long>(heads) * kHeadDim,
              "z must be [T, H*128]");

  const long norm_rows = tokens * static_cast<long>(heads);
  const long norm_blocks = (norm_rows + kRowsPerBlock - 1) / kRowsPerBlock;

  TORCH_CHECK(y.dim() == 2 && y.size(0) == tokens &&
                  y.size(1) == static_cast<long>(heads) * kHeadDim,
              "y must be [tokens, heads*128], got ", y.sizes());

  long per_sequence = 0, batch = 0, scatter_blocks = 0;
  const float* state_ptr = nullptr;
  __nv_bfloat16* cache_ptr = nullptr;
  const long* index_ptr = nullptr;
  // All three scatter arguments or none: two of the three would silently run the
  // norm alone, which for a caller that meant to scatter is a lost state write.
  const int scatter_args = static_cast<int>(final_state.has_value()) +
                           static_cast<int>(cache.has_value()) +
                           static_cast<int>(state_index.has_value());
  TORCH_CHECK(scatter_args == 0 || scatter_args == 3,
              "final_state, cache and state_index must be given together or not "
              "at all; got ", scatter_args, " of 3");
  if (scatter_args == 3) {
    const at::Tensor& fs = *final_state;
    const at::Tensor& cc = *cache;
    const at::Tensor& si = *state_index;
    TORCH_CHECK(fs.scalar_type() == at::kFloat, "final state must be float32");
    TORCH_CHECK(cc.scalar_type() == at::kBFloat16, "cache must be bfloat16");
    TORCH_CHECK(si.scalar_type() == at::kLong, "state index must be int64");
    TORCH_CHECK(fs.is_contiguous() && cc.is_contiguous() && si.is_contiguous(),
                "state tensors must be contiguous");
    TORCH_CHECK(fs.device() == o.device() && cc.device() == o.device() &&
                    si.device() == o.device(),
                "state tensors must share the output's device");
    TORCH_CHECK(fs.dim() == 4 && cc.dim() == 4, "state tensors must be 4-D");
    TORCH_CHECK(fs.size(1) == static_cast<long>(heads) &&
                    cc.size(1) == static_cast<long>(heads),
                "state head count must match the output's");
    TORCH_CHECK(fs.size(2) == kHeadDim && fs.size(3) == kHeadDim,
                "state must be [B, H, 128, 128]");
    batch = fs.size(0);
    TORCH_CHECK(si.numel() == batch, "one state index per sequence");
    // Every dimension below is already checked, so this is exact and needs no
    // divide-by-zero guard.
    per_sequence = static_cast<long>(heads) * kHeadDim * kHeadDim;
    TORCH_CHECK(cc.size(2) == kHeadDim && cc.size(3) == kHeadDim,
                "cache must be [slots, heads, 128, 128], got ", cc.sizes());
    TORCH_CHECK(cc.numel() >= per_sequence,
                "cache must hold at least one state row");
    static_assert(kHeadDim % 4 == 0,
                  "the vectorized scatter walks four floats at a time");
    const long items = batch * per_sequence / 4;
    scatter_blocks = std::min<long>(
        1024, (items + kScatterThreads - 1) / kScatterThreads);
    state_ptr = fs.data_ptr<float>();
    cache_ptr = reinterpret_cast<__nv_bfloat16*>(cc.data_ptr());
    index_ptr = si.data_ptr<long>();
  }

  // One block size for both halves -- kScatterThreads is defined as kNormThreads --
  // and the scatter is a grid-stride loop, so it does not care how many it gets.
  const long blocks = norm_blocks + scatter_blocks;
  if (blocks == 0) return;
  const c10::cuda::CUDAGuard guard(o.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn_post_recurrence_kernel<<<blocks, kNormThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(o.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(z.const_data_ptr()),
      weight.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), state_ptr, cache_ptr,
      index_ptr, z.stride(0), y.stride(0), tokens, heads,
      static_cast<float>(eps), norm_blocks, per_sequence, batch, scatter_blocks);
}


// ---------------------------------------------------------------------------
// Fused pre-recurrence stage
// ---------------------------------------------------------------------------
//
// Replaces two launches -- the varlen causal conv with its SiLU and state
// write-back, and the post-conv split with the L2 norm and gate formation -- with
// one, so the 8192-wide conv result never round-trips to HBM.
//
// The arithmetic has to match the two kernels it replaces term for term, and
// same-named operations are not the same function. The correspondence used below
// was established by comparing bit patterns over a wide fp32 range
// (profile/p1_math_probe/), not assumed:
//
//     tl.exp(x)              -> __expf(x)
//     tl.log(x)              -> logf(x)                       (also torch.log)
//     tl.sqrt(x)             -> sqrt.approx.f32               (inline PTX)
//     1.0 / tl.sqrt(x)       -> __fdividef(1.0f, sqrt.approx.f32(x))
//     tl.sigmoid(x)          -> __fdividef(1.0f, 1.0f + __expf(-x))
//     x / (1 + tl.exp(-x))   -> __fdividef(x, 1.0f + __expf(-x))
//     fp32_acc += bf16*bf16  -> __bfloat162float(__hmul(a, b))
//     torch.exp(x)           -> expf(x)
//
// The last two matter most. Widening the conv taps before multiplying, instead of
// multiplying in bf16 and widening the product, is wrong by a full bf16 ulp
// (3.9e-3 relative). And the forget gate's outer exponential is the *precise* one,
// because the reference applies torch.exp to it after the Triton kernel has run,
// while every exponential inside the gate is Triton's fast one.

namespace {

constexpr int kConvWidth = 4;
constexpr int kStateLen = kConvWidth - 1;
// Four channels per lane is an 8-byte load, which makes one 128-wide head exactly
// one warp: the L2 reduction becomes a warp shuffle with no shared memory and no
// block barrier, and each block covers four head groups instead of one. One
// channel per lane left this kernel issuing 2-byte loads at ~1.5 TB/s.
constexpr int kPreVec = 4;
constexpr int kPreLanesPerGroup = kHeadDim / kPreVec;   // 32 == one warp
constexpr int kPreGroupsPerBlock = 4;
constexpr int kPreThreads = kPreLanesPerGroup * kPreGroupsPerBlock;  // 128
constexpr float kL2NormEps = 1e-6f;

static_assert(kPreLanesPerGroup == kWarp,
              "one head group must map to exactly one warp");
constexpr float kSoftplusThreshold = 20.0f;
constexpr int kNullBlockId = 0;

__device__ __forceinline__ float sqrt_approx(float v) {
  float r;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

// fp32_acc += bf16_a * bf16_b, with the product rounded to bf16 first.
//
// GDN_MUTATE_* are deliberate defects, compiled only by the mutation harness in
// profile/p1_parity/harness/. They exist so each parity check can be shown to fail
// against a wrong implementation before it is trusted on the right one; none is
// ever defined in a production build.
__device__ __forceinline__ float tap(const __nv_bfloat16 x,
                                     const __nv_bfloat16 w) {
#ifdef GDN_MUTATE_WIDEN_TAPS
  // Widening before the multiply is the natural way to write this and is wrong by
  // a full bf16 ulp.
  return __bfloat162float(x) * __bfloat162float(w);
#else
  return __bfloat162float(__hmul(x, w));
#endif
}

__device__ __forceinline__ float silu(float v) {
  return __fdividef(v, 1.0f + __expf(-v));
}

// How the projection is read in the conv loop. GDN_PRE_LOAD_POLICY is set only by the
// tuning harness; the default is a plain load, which lets the compiler choose.
//   1 = __ldg      (non-coherent read-only path)
//   2 = ld.global.nc.L1::evict_last  (keep: the next tile re-reads these taps)
//   3 = ld.global.nc.L1::evict_first (stream: do not retain)
__device__ __forceinline__ BF16x4 load_taps(const __nv_bfloat16* p) {
#if defined(GDN_PRE_LOAD_POLICY) && GDN_PRE_LOAD_POLICY == 1
  BF16x4 out;
  const int2 raw = __ldg(reinterpret_cast<const int2*>(p));
  *reinterpret_cast<int2*>(out.v) = raw;
  return out;
#elif defined(GDN_PRE_LOAD_POLICY) && GDN_PRE_LOAD_POLICY == 2
  BF16x4 out;
  int2 raw;
  asm("ld.global.nc.L1::evict_last.v2.s32 {%0, %1}, [%2];"
      : "=r"(raw.x), "=r"(raw.y) : "l"(p));
  *reinterpret_cast<int2*>(out.v) = raw;
  return out;
#elif defined(GDN_PRE_LOAD_POLICY) && GDN_PRE_LOAD_POLICY == 3
  BF16x4 out;
  int2 raw;
  asm("ld.global.nc.L1::evict_first.v2.s32 {%0, %1}, [%2];"
      : "=r"(raw.x), "=r"(raw.y) : "l"(p));
  *reinterpret_cast<int2*>(out.v) = raw;
  return out;
#else
  return *reinterpret_cast<const BF16x4*>(p);
#endif
}

#ifdef GDN_PRE_MIN_BLOCKS_PER_SM
__global__ __launch_bounds__(kPreThreads, GDN_PRE_MIN_BLOCKS_PER_SM)
#else
__global__
#endif
void gdn_pre_recurrence_kernel(
    const __nv_bfloat16* __restrict__ mixed,   // [tokens, conv_dim], row stride
    const __nv_bfloat16* __restrict__ conv_w,  // [conv_dim, 4]
    __nv_bfloat16* __restrict__ conv_state,    // [slots, conv_dim, 3]
    const int* __restrict__ cu_seqlens,        // [B + 1]
    const bool* __restrict__ has_initial_state,
    const int* __restrict__ cache_index,       // [B]
    const __nv_bfloat16* __restrict__ a_in,    // [tokens, HV], row stride
    const __nv_bfloat16* __restrict__ b_in,
    const __nv_bfloat16* __restrict__ A_log,   // [HV]
    const __nv_bfloat16* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ q_out,         // [tokens, H, 128]
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,         // [tokens, HV, 128]
    float* __restrict__ gate_out,              // [tokens, HV]
    float* __restrict__ beta_out,
    __nv_bfloat16* __restrict__ conv_out,      // optional parity tap
    long mixed_row, long ab_row, long state_slot_stride, long state_tok_stride,
    int heads_k, int heads_v, int tile_tokens, long conv_dim) {
  const int seq = blockIdx.z;
  const int lane = threadIdx.x % kPreLanesPerGroup;
  const int group = blockIdx.y * kPreGroupsPerBlock +
                    static_cast<int>(threadIdx.x) / kPreLanesPerGroup;

  const int seq_start = cu_seqlens[seq];
  const int seqlen = cu_seqlens[seq + 1] - seq_start;
  const long tile_first = static_cast<long>(blockIdx.x) * tile_tokens;
  if (tile_first >= seqlen) return;

  const int slot = cache_index[seq];
  if (slot == kNullBlockId) return;            // padded cache line: skip

  const int q_groups = heads_k;                // 128 channels per K head
  const int channel0 = group * kHeadDim + lane * kPreVec;

  // The conv weight is [conv_dim, 4], so one lane's four channels are four
  // consecutive rows of it and one 8-byte load covers one channel's four taps.
  __nv_bfloat16 w[kPreVec][kConvWidth];
#pragma unroll
  for (int c = 0; c < kPreVec; ++c) {
    const BF16x4 row = *reinterpret_cast<const BF16x4*>(
        conv_w + static_cast<long>(channel0 + c) * kConvWidth);
#pragma unroll
    for (int j = 0; j < kConvWidth; ++j) w[c][j] = row.v[j];
  }

  __nv_bfloat16* state = conv_state + static_cast<long>(slot) * state_slot_stride +
                         static_cast<long>(channel0);
  const __nv_bfloat16* x_base =
      mixed + static_cast<long>(seq_start) * mixed_row + channel0;

  // --- three history taps -------------------------------------------------
  __nv_bfloat16 col0[kPreVec], col1[kPreVec], col2[kPreVec];
  const bool has_init = has_initial_state != nullptr && has_initial_state[seq];
  if (tile_first == 0) {
    if (has_init) {
#pragma unroll
      for (int c = 0; c < kPreVec; ++c) {
        col0[c] = state[c + 0 * state_tok_stride];
        col1[c] = state[c + 1 * state_tok_stride];
        col2[c] = state[c + 2 * state_tok_stride];
      }
#ifdef GDN_MUTATE_CROSS_SEQUENCE_HISTORY
    } else if (seq_start >= kStateLen) {
      // Taking a sequence's history from the tokens preceding it in the flat token
      // axis, which belong to the previous sequence, instead of from the conv
      // state. Only tile zero sits at a sequence start, so this is the only place
      // a boundary can be crossed.
      const __nv_bfloat16* flat = mixed + channel0;
      const long t0 = seq_start;
      *reinterpret_cast<BF16x4*>(col0) =
          *reinterpret_cast<const BF16x4*>(flat + (t0 - 3) * mixed_row);
      *reinterpret_cast<BF16x4*>(col1) =
          *reinterpret_cast<const BF16x4*>(flat + (t0 - 2) * mixed_row);
      *reinterpret_cast<BF16x4*>(col2) =
          *reinterpret_cast<const BF16x4*>(flat + (t0 - 1) * mixed_row);
#endif
    } else {
#pragma unroll
      for (int c = 0; c < kPreVec; ++c) {
        col0[c] = col1[c] = col2[c] = __float2bfloat16(0.0f);
      }
    }
    // The first tile of a sequence is also the one block that writes this
    // sequence's new conv state, taken from the last three *pre-activation*
    // inputs. It reads them straight out of the projection, which no block
    // writes, so this needs no ordering against the conv itself. Every old value
    // it needs is read before anything is written.
    if (kStateLen <= seqlen) {
      BF16x4 s0 = *reinterpret_cast<const BF16x4*>(
          x_base + static_cast<long>(seqlen - 3) * mixed_row);
      BF16x4 s1 = *reinterpret_cast<const BF16x4*>(
          x_base + static_cast<long>(seqlen - 2) * mixed_row);
      BF16x4 s2 = *reinterpret_cast<const BF16x4*>(
          x_base + static_cast<long>(seqlen - 1) * mixed_row);
#ifdef GDN_MUTATE_POST_ACT_STATE
      // Storing the SiLU'd value instead of the pre-activation input. The conv
      // state must carry the raw inputs, because the next step convolves them.
#pragma unroll
      for (int c = 0; c < kPreVec; ++c) {
        s0.v[c] = __float2bfloat16(silu(__bfloat162float(s0.v[c])));
        s1.v[c] = __float2bfloat16(silu(__bfloat162float(s1.v[c])));
        s2.v[c] = __float2bfloat16(silu(__bfloat162float(s2.v[c])));
      }
#endif
      *reinterpret_cast<BF16x4*>(state + 0 * state_tok_stride) = s0;
      *reinterpret_cast<BF16x4*>(state + 1 * state_tok_stride) = s1;
      *reinterpret_cast<BF16x4*>(state + 2 * state_tok_stride) = s2;
    } else {
      // Shorter than the state: shift the old state left and append what exists.
      const int shift = kStateLen - seqlen;
      __nv_bfloat16 next[kStateLen][kPreVec];
#pragma unroll
      for (int j = 0; j < kStateLen; ++j) {
        const int from_old = j + seqlen;
        const int t = j - shift;
        for (int c = 0; c < kPreVec; ++c) {
          if (from_old < kStateLen) {
            next[j][c] = has_init ? state[c + from_old * state_tok_stride]
                                  : __float2bfloat16(0.0f);
          } else {
            next[j][c] = (t >= 0 && t < seqlen)
                             ? x_base[static_cast<long>(t) * mixed_row + c]
                             : __float2bfloat16(0.0f);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kStateLen; ++j) {
        for (int c = 0; c < kPreVec; ++c) {
          state[c + j * state_tok_stride] = next[j][c];
        }
      }
    }
  } else {
#ifdef GDN_MUTATE_ZERO_HISTORY
    // Zeroing the history at every tile boundary instead of carrying the three
    // preceding inputs.
#pragma unroll
    for (int c = 0; c < kPreVec; ++c) {
      col0[c] = col1[c] = col2[c] = __float2bfloat16(0.0f);
    }
#else
    // A later tile's history is always inside its own sequence, because only tile
    // zero sits at a sequence start and that case is handled above. A mutant that
    // indexed the flat token axis here would be indistinguishable from this code,
    // which is why the cross-sequence defect is injected at tile zero instead.
    // Later tiles read their history from the projection at the three token
    // positions preceding the tile. Tiles are at least kStateLen wide, so those
    // positions are always inside the sequence.
    *reinterpret_cast<BF16x4*>(col0) = load_taps(x_base + (tile_first - 3) * mixed_row);
    *reinterpret_cast<BF16x4*>(col1) = load_taps(x_base + (tile_first - 2) * mixed_row);
    *reinterpret_cast<BF16x4*>(col2) = load_taps(x_base + (tile_first - 1) * mixed_row);
#endif
  }

  // --- gate parameters, for the value groups only -------------------------
  const bool is_v = group >= 2 * q_groups;
  const int head = is_v ? group - 2 * q_groups
                        : (group < q_groups ? group : group - q_groups);
  float a_log_v = 0.0f, dt_v = 0.0f;
  if (is_v && lane == 0) {
    a_log_v = __bfloat162float(A_log[head]);
    dt_v = __bfloat162float(dt_bias[head]);
  }

  const int seg = static_cast<int>(min(static_cast<long>(tile_tokens),
                                       seqlen - tile_first));
  for (int it = 0; it < seg; ++it) {
    const long token = tile_first + it;
    __nv_bfloat16 xcur[kPreVec];
    *reinterpret_cast<BF16x4*>(xcur) = load_taps(x_base + token * mixed_row);

    float act_f[kPreVec];
    __nv_bfloat16 act[kPreVec];
#pragma unroll
    for (int c = 0; c < kPreVec; ++c) {
      // Term order matches the reference's static_range: fp32 addition is not
      // associative, so the taps are accumulated oldest first.
      float acc = 0.0f;
      acc += tap(col0[c], w[c][0]);
      acc += tap(col1[c], w[c][1]);
      acc += tap(col2[c], w[c][2]);
      acc += tap(xcur[c], w[c][3]);
      col0[c] = col1[c];
      col1[c] = col2[c];
      col2[c] = xcur[c];
      // The reference stores the activated conv result to a bf16 buffer and the
      // next stage reads it back, so the rounding to bf16 happens here whether or
      // not the value ever reaches memory.
      act[c] = __float2bfloat16(silu(acc));
#ifdef GDN_MUTATE_NO_ROUND
      // Keeping the activation in fp32 for the L2 norm skips the rounding the
      // reference's intermediate bf16 buffer imposes.
      act_f[c] = silu(acc);
#else
      act_f[c] = __bfloat162float(act[c]);
#endif
    }

    const long out_token = static_cast<long>(seq_start) + token;
    if (conv_out != nullptr) {
      *reinterpret_cast<BF16x4*>(conv_out + out_token * conv_dim + channel0) =
          *reinterpret_cast<const BF16x4*>(act);
    }

    if (is_v) {
      *reinterpret_cast<BF16x4*>(
          v_out + (out_token * heads_v + head) * kHeadDim + lane * kPreVec) =
          *reinterpret_cast<const BF16x4*>(act);
      if (lane == 0) {
        const float x = __bfloat162float(a_in[out_token * ab_row + head]) + dt_v;
        const float sp_raw = (x > 0.0f) ? x + logf(1.0f + __expf(-x))
                                        : logf(1.0f + __expf(x));
        const float sp = (x <= kSoftplusThreshold) ? sp_raw : x;
        const float g = -__expf(a_log_v) * sp;
        // Precise exponential: the reference applies torch.exp to g after the
        // Triton kernel, so folding it in here has to use the precise one.
        gate_out[out_token * heads_v + head] = expf(g);
        const float bv = __bfloat162float(b_in[out_token * ab_row + head]);
        beta_out[out_token * heads_v + head] =
            __fdividef(1.0f, 1.0f + __expf(-bv));
      }
    } else {
      float sq = 0.0f;
#pragma unroll
      for (int c = 0; c < kPreVec; ++c) sq += act_f[c] * act_f[c];
      sq = lane_reduce_add<kPreLanesPerGroup>(sq);
      const float inv = __fdividef(1.0f, sqrt_approx(sq + kL2NormEps));
      __nv_bfloat16* dst = (group < q_groups) ? q_out : k_out;
      __nv_bfloat16 norm[kPreVec];
#pragma unroll
      for (int c = 0; c < kPreVec; ++c) {
        norm[c] = __float2bfloat16(act_f[c] * inv);
      }
      *reinterpret_cast<BF16x4*>(
          dst + (out_token * heads_k + head) * kHeadDim + lane * kPreVec) =
          *reinterpret_cast<const BF16x4*>(norm);
    }
  }
}

}  // namespace

void gdn_pre_recurrence(const at::Tensor& mixed, const at::Tensor& conv_weight,
                        at::Tensor& conv_state, const at::Tensor& cu_seqlens,
                        const std::optional<at::Tensor>& has_initial_state,
                        const at::Tensor& cache_index, const at::Tensor& a_in,
                        const at::Tensor& b_in, const at::Tensor& A_log,
                        const at::Tensor& dt_bias, at::Tensor& q_out,
                        at::Tensor& k_out, at::Tensor& v_out,
                        at::Tensor& gate_out, at::Tensor& beta_out,
                        const std::optional<at::Tensor>& conv_out,
                        int64_t max_seqlen, int64_t tile_tokens) {
  TORCH_CHECK(mixed.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(mixed.scalar_type() == at::kBFloat16, "mixed must be bfloat16");
  {
    const at::Device dev = mixed.device();
    const std::pair<const char*, const at::Tensor*> args[] = {
        {"conv_weight", &conv_weight}, {"conv_state", &conv_state},
        {"cu_seqlens", &cu_seqlens},   {"cache_index", &cache_index},
        {"a_in", &a_in},               {"b_in", &b_in},
        {"A_log", &A_log},             {"dt_bias", &dt_bias},
        {"q_out", &q_out},             {"k_out", &k_out},
        {"v_out", &v_out},             {"gate_out", &gate_out},
        {"beta_out", &beta_out},
    };
    for (const auto& [name, t] : args) {
      TORCH_CHECK(t->is_cuda() && t->device() == dev, name,
                  " must be CUDA and share the projection's device");
    }
    // These six are reinterpreted as bf16 by the kernel, so their scalar type is
    // not a formality: a fp32 tensor here would be read as twice as many bf16
    // values with no complaint from anything.
    const std::pair<const char*, const at::Tensor*> bf16_args[] = {
        {"conv_weight", &conv_weight}, {"conv_state", &conv_state},
        {"a_in", &a_in},               {"b_in", &b_in},
        {"A_log", &A_log},             {"dt_bias", &dt_bias},
    };
    for (const auto& [name, t] : bf16_args) {
      TORCH_CHECK(t->scalar_type() == at::kBFloat16, name, " must be bfloat16");
    }
    if (has_initial_state.has_value()) {
      TORCH_CHECK(has_initial_state->device() == dev &&
                      has_initial_state->scalar_type() == at::kBool,
                  "has_initial_state must be a bool tensor on the same device");
      TORCH_CHECK(has_initial_state->numel() == cu_seqlens.numel() - 1,
                  "one has_initial_state flag per sequence");
    }
    if (conv_out.has_value()) {
      TORCH_CHECK(conv_out->device() == dev &&
                      conv_out->scalar_type() == at::kBFloat16,
                  "conv_out tap must be bf16 on the same device");
      TORCH_CHECK(conv_out->size(0) == mixed.size(0) &&
                      conv_out->size(1) == mixed.size(1),
                  "conv_out tap must match the projection's shape");
    }
  }
  {
    const std::pair<const char*, const at::Tensor*> outs[] = {
        {"q_out", &q_out}, {"k_out", &k_out}, {"v_out", &v_out}};
    for (const auto& [name, t] : outs) {
      TORCH_CHECK(t->scalar_type() == at::kBFloat16, name, " must be bfloat16");
      TORCH_CHECK(t->dim() == 3 && t->size(0) == mixed.size(0) &&
                      t->size(2) == kHeadDim,
                  name, " must be [tokens, heads, 128], got ", t->sizes());
    }
  }
  TORCH_CHECK(mixed.dim() == 2 && a_in.dim() == 2 && b_in.dim() == 2,
              "the projection and its gate slices must be 2-D");
  TORCH_CHECK(a_in.size(0) == mixed.size(0) && b_in.size(0) == mixed.size(0) &&
                  a_in.size(1) == v_out.size(1) &&
                  b_in.size(1) == v_out.size(1),
              "a and b must be [tokens, value heads]");
  // The conv state is indexed as [slot, channel, tap] with the channel innermost;
  // an unexpected shape or tap stride would silently address the wrong history.
  TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == mixed.size(1) &&
                  conv_state.size(2) >= kStateLen,
              "conv_state must be [slots, conv_dim, >=3], got ",
              conv_state.sizes());
  TORCH_CHECK(conv_weight.size(0) == mixed.size(1),
              "one conv weight row per channel");
  TORCH_CHECK(gate_out.scalar_type() == at::kFloat &&
                  beta_out.scalar_type() == at::kFloat,
              "gate and beta must be float32");
  TORCH_CHECK(gate_out.dim() == 2 && gate_out.size(0) == mixed.size(0) &&
                  gate_out.size(1) == v_out.size(1) &&
                  beta_out.sizes() == gate_out.sizes(),
              "gate and beta must be [tokens, value heads]");
  TORCH_CHECK(A_log.numel() == v_out.size(1) &&
                  dt_bias.numel() == v_out.size(1),
              "A_log and dt_bias must carry one entry per value head");
  TORCH_CHECK(q_out.size(1) == k_out.size(1),
              "q and k must carry the same head count");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2 &&
                  cu_seqlens.is_contiguous(),
              "cu_seqlens must be a contiguous 1-D [B+1]");
  TORCH_CHECK(cache_index.dim() == 1 && cache_index.is_contiguous() &&
                  cache_index.numel() == cu_seqlens.numel() - 1,
              "one contiguous cache index per sequence");
  TORCH_CHECK(conv_weight.size(1) == kConvWidth, "conv width must be 4");
  TORCH_CHECK(conv_weight.stride(1) == 1 && conv_weight.stride(0) == kConvWidth,
              "conv weight must be compact [conv_dim, 4]");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
  TORCH_CHECK(cache_index.scalar_type() == at::kInt, "cache index must be int32");
  TORCH_CHECK(q_out.size(2) == kHeadDim && v_out.size(2) == kHeadDim,
              "this kernel is specialized to 128-wide heads");
  TORCH_CHECK(tile_tokens >= kStateLen,
              "a tile must be at least as wide as the conv history");
  TORCH_CHECK(q_out.is_contiguous() && k_out.is_contiguous() &&
                  v_out.is_contiguous() && gate_out.is_contiguous() &&
                  beta_out.is_contiguous(),
              "outputs must be contiguous");
  TORCH_CHECK(mixed.stride(1) == 1 && a_in.stride(1) == 1 && b_in.stride(1) == 1,
              "projection views need unit column stride");
  TORCH_CHECK(conv_state.stride(1) == 1, "conv state must be channel-innermost");

  const int heads_k = static_cast<int>(q_out.size(1));
  const int heads_v = static_cast<int>(v_out.size(1));
  const long batch = cu_seqlens.numel() - 1;
  const int groups = 2 * heads_k + heads_v;
  TORCH_CHECK(static_cast<long>(groups) * kHeadDim == mixed.size(1),
              "channel groups must tile the conv dimension exactly");
  if (max_seqlen == 0) return;   // batch >= 1 already, from numel() >= 2

  TORCH_CHECK(groups % kPreGroupsPerBlock == 0,
              "head groups must tile the block's group count");
  const long tiles = (max_seqlen + tile_tokens - 1) / tile_tokens;
  const c10::cuda::CUDAGuard guard(mixed.device());
  const dim3 grid(static_cast<unsigned>(tiles),
                  static_cast<unsigned>(groups / kPreGroupsPerBlock),
                  static_cast<unsigned>(batch));
  gdn_pre_recurrence_kernel<<<grid, kPreThreads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(mixed.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(conv_weight.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(conv_state.data_ptr()),
      cu_seqlens.const_data_ptr<int>(),
      has_initial_state.has_value()
          ? has_initial_state->const_data_ptr<bool>()
          : nullptr,
      cache_index.const_data_ptr<int>(),
      reinterpret_cast<const __nv_bfloat16*>(a_in.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(b_in.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(A_log.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(dt_bias.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr()),
      gate_out.data_ptr<float>(), beta_out.data_ptr<float>(),
      conv_out.has_value()
          ? reinterpret_cast<__nv_bfloat16*>(conv_out->data_ptr())
          : nullptr,
      mixed.stride(0), a_in.stride(0), conv_state.stride(0),
      conv_state.stride(2), heads_k, heads_v,
      static_cast<int>(tile_tokens), mixed.size(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_post_recurrence", &gdn_post_recurrence,
        "Gated RMS norm over a strided gate, fused with the recurrent-state "
        "scatter");
  m.def("gdn_pre_recurrence", &gdn_pre_recurrence,
        "Causal conv, SiLU, split, L2 norm and gate formation in one pass over "
        "the strided projection");
}
