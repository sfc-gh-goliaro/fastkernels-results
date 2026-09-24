// Fused temporal axial attention: rotary rotation, causal attention over the frame
// axis, and both layout shuffles, in one warp-per-(batch, spatial, head) kernel.
//
// The operator this serves projects a (bsz, T, H, W, dim) activation to a fused qkv
// buffer, then attends over the *temporal* axis independently for every spatial
// position and head. T is 2..6 in practice while bsz*H*W*heads is in the thousands,
// so the attention is a few thousand copies of a tiny problem and the surrounding
// permutes are pure layout churn. Everything between the two projections therefore
// collapses into this one launch: the three permute-copies, both rotary chains, the
// attention, and both output relayouts.
//
// Decomposition: head_dim 64 in a 2-byte dtype is exactly 32 lanes x 2 elements, so
// one warp reading one (t, head) row issues a single fully-utilised 128-byte burst,
// and lane l's two elements are exactly rotary pair f = l -- the rotation is then a
// swap, two packed multiplies and one packed add on registers the lane already
// holds, with no shuffle and no shared memory. Scores are warp-reduced to a
// lane-uniform value, so the <=8-wide softmax is recomputed redundantly in every
// lane and needs no further communication. There is no cross-warp traffic and
// therefore no __syncthreads anywhere. This is the geometry the frozen
// candidate/L1/_tiny_seq_attn.cuh measured as best for the identical attention
// problem, and the attention half below is a port of it.
//
// Every stride is a runtime argument; the caller checks the layout rather than the
// kernel assuming it.
//
// This header is deliberately free of any framework dependency so the profiling
// harness and the PyTorch binding compile the *same* kernel source.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace temporal_axial {

// The only head extent the packed-pair decomposition covers: 32 lanes x 2 elements.
constexpr int kHeadDim = 64;
// Pairs per row, i.e. rotary pairs, i.e. lanes.
constexpr int kPairsPerRow = kHeadDim / 2;
// Widest temporal extent with a compiled specialisation.
constexpr int kMaxSeq = 8;

// Packed-pair traits.
//
// Two distinct precisions are in play and the split is deliberate. The rotation
// runs in *packed input precision* because that is what makes it bit-exact against
// the reference, which evaluates it on tensors of the activation dtype: each of the
// two products and their sum rounds to that dtype. Everything downstream of the
// rotation -- the score dot products, the softmax and the p.v accumulation -- widens
// to fp32 through float2, where the reference's own attention kernel accumulates;
// packed arithmetic there would silently accumulate at input precision.
//
// `rotate` is written in PTX rather than as `__hadd2(__hmul2(...), __hmul2(...))`
// because nvcc contracts that expression into a single unrounded HFMA2, which drops
// one of the three roundings. Measured on the fp16 captured window, the contracted
// form differed from the reference by one ulp on about 12% of elements -- always in
// the direction the extra precision predicts, which is exactly the kind of "more
// accurate, therefore not equal" divergence this operator has to avoid. The three
// `.rn` instructions below are the contract, stated where it cannot be optimised
// away. A blanket -fmad=false would also work, but it would change the score dot
// product in the attention half, which is a faithful port of a kernel that was
// compiled with contraction on.
template <typename T>
struct Packed;

template <>
struct Packed<__half> {
  using vec = __half2;
  static __device__ __forceinline__ float2 to_float2(vec x) { return __half22float2(x); }
  static __device__ __forceinline__ vec from_float2(float2 x) { return __float22half2_rn(x); }
  static __device__ __forceinline__ vec swap(vec x) { return __lowhigh2highlow(x); }
  // (lo, lo) -- the cosine, broadcast to both halves of the pair.
  static __device__ __forceinline__ vec splat_low(vec x) {
    return __half2half2(__low2half(x));
  }
  // (-hi, hi) -- the sine with the rotation's sign already folded into the low
  // half. Negating the multiplier rather than the product is exact and free: half
  // multiplication is sign-symmetric, so the magnitude and the rounding are
  // unchanged.
  static __device__ __forceinline__ vec splat_high_signed(vec x) {
    const __half hi = __high2half(x);
    return __halves2half2(__hneg(hi), hi);
  }
  static __device__ __forceinline__ vec rotate(vec x, vec swapped, vec cos2, vec nsin2) {
    union { vec packed; unsigned int raw; } out, a, b, c, d;
    a.packed = x; b.packed = swapped; c.packed = cos2; d.packed = nsin2;
    asm("{\n\t"
        ".reg .b32 p, q;\n\t"
        "mul.rn.f16x2 p, %1, %3;\n\t"
        "mul.rn.f16x2 q, %2, %4;\n\t"
        "add.rn.f16x2 %0, p, q;\n\t"
        "}"
        : "=r"(out.raw)
        : "r"(a.raw), "r"(b.raw), "r"(c.raw), "r"(d.raw));
    return out.packed;
  }
};

template <>
struct Packed<__nv_bfloat16> {
  using vec = __nv_bfloat162;
  static __device__ __forceinline__ float2 to_float2(vec x) { return __bfloat1622float2(x); }
  static __device__ __forceinline__ vec from_float2(float2 x) { return __float22bfloat162_rn(x); }
  static __device__ __forceinline__ vec swap(vec x) { return __lowhigh2highlow(x); }
  static __device__ __forceinline__ vec splat_low(vec x) {
    return __bfloat162bfloat162(__low2bfloat16(x));
  }
  static __device__ __forceinline__ vec splat_high_signed(vec x) {
    const __nv_bfloat16 hi = __high2bfloat16(x);
    return __halves2bfloat162(__hneg(hi), hi);
  }
  static __device__ __forceinline__ vec rotate(vec x, vec swapped, vec cos2, vec nsin2) {
    union { vec packed; unsigned int raw; } out, a, b, c, d;
    a.packed = x; b.packed = swapped; c.packed = cos2; d.packed = nsin2;
    asm("{\n\t"
        ".reg .b32 p, q;\n\t"
        "mul.rn.bf16x2 p, %1, %3;\n\t"
        "mul.rn.bf16x2 q, %2, %4;\n\t"
        "add.rn.bf16x2 %0, p, q;\n\t"
        "}"
        : "=r"(out.raw)
        : "r"(a.raw), "r"(b.raw), "r"(c.raw), "r"(d.raw));
    return out.packed;
  }
};

// One rotary pair, rotated at input precision.
//
// The reference is ``(x * table.cos()) + (rotate_half(x) * table.sin())`` evaluated
// on tensors of the activation dtype, so each of the two products and their sum
// rounds to that dtype -- three roundings, not one. Accumulating the products in
// fp32 and rounding once would be *more* accurate and therefore not equal. Factored
// out so a test can compile this exact code against the reference (tests/
// _rotation_probe.cu) rather than inferring the rotation from an attention output.
template <typename T>
__device__ __forceinline__ typename Packed<T>::vec
rotate_pair(typename Packed<T>::vec x, typename Packed<T>::vec cos_sin) {
  using Traits = Packed<T>;
  return Traits::rotate(x, Traits::swap(x), Traits::splat_low(cos_sin),
                        Traits::splat_high_signed(cos_sin));
}

// Element strides of the (batch, time, spatial) axes of one activation tensor.
// head_dim is the unit-stride axis and the head axis is a fixed kHeadDim step
// inside a row, so neither needs an entry.
struct Layout {
  long long batch;
  long long time;
  long long spatial;
};

// Column offsets of the q, k and v blocks inside a fused row, in elements.
struct QkvOffsets {
  long long q;
  long long k;
  long long v;
};

// SEQ is the exact temporal extent, so every loop unrolls completely and no row is
// predicated off. CAUSAL selects j <= i; the diagonal is always included, so no row is
// fully masked and, for finite inputs, the denominator is at least exp(0) = 1 and
// cannot be zero. Non-finite inputs still propagate: a NaN or an infinite score makes
// the max-subtract produce NaN, exactly as the reference attention does for the same
// input, so this is equivalence rather than robustness.
template <typename T, int SEQ, bool CAUSAL>
__global__ void kernel(const T* __restrict__ qkv,
                       const T* __restrict__ cos_sin,
                       T* __restrict__ out,
                       int pairs,
                       int heads,
                       int spatial,
                       Layout in_layout,
                       Layout out_layout,
                       QkvOffsets off,
                       float scale) {
  using Traits = Packed<T>;
  using vec = typename Traits::vec;

  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int pair = blockIdx.x * static_cast<int>(blockDim.x >> 5) + warp;
  // Warp-uniform: every lane of a warp exits together, which is what makes the
  // full-mask shuffles below well-defined.
  if (pair >= pairs) return;

  // Heads innermost, so consecutive warps of a block walk consecutive heads of one
  // spatial position -- their row loads are adjacent 128-byte bursts inside the
  // same fused row.
  const int head = pair % heads;
  const int flat = pair / heads;
  const int b = flat / spatial;
  const int s = flat - b * spatial;

  const long long slot = static_cast<long long>(b) * in_layout.batch
                       + static_cast<long long>(s) * in_layout.spatial
                       + static_cast<long long>(head) * kHeadDim
                       + static_cast<long long>(lane) * 2;
  const T* qp = qkv + slot + off.q;
  const T* kp = qkv + slot + off.k;
  const T* vp = qkv + slot + off.v;
  T* op = out + static_cast<long long>(b) * out_layout.batch
             + static_cast<long long>(s) * out_layout.spatial
             + static_cast<long long>(head) * kHeadDim
             + static_cast<long long>(lane) * 2;

  // One packed (cos, sin) per (t, lane): lane l's rotary pair is f = l, so the
  // whole table a warp needs for one t is 128 contiguous bytes.
  const vec* table = reinterpret_cast<const vec*>(cos_sin) + lane;

  vec qr[SEQ], kr[SEQ], vr[SEQ];
#pragma unroll
  for (int t = 0; t < SEQ; ++t) {
    const long long step = static_cast<long long>(t) * in_layout.time;
    vec q = *reinterpret_cast<const vec*>(qp + step);
    vec k = *reinterpret_cast<const vec*>(kp + step);
    vr[t] = *reinterpret_cast<const vec*>(vp + step);

    const vec cs = table[t * kPairsPerRow];
    qr[t] = rotate_pair<T>(q, cs);
    kr[t] = rotate_pair<T>(k, cs);
  }

  // One query row at a time: holding all SEQ^2 scores at once buys nothing here and
  // costs registers.
#pragma unroll
  for (int i = 0; i < SEQ; ++i) {
    const int jmax = CAUSAL ? i : (SEQ - 1);
    // Scale the query row once, before the dot product, rather than scaling each
    // score after the reduction. Two multiplies instead of up to SEQ, and it keeps
    // the reduction operating on the same magnitudes the reference kernel uses --
    // scaling afterwards lets the unscaled sum overflow fp32 for inputs whose
    // exponent range allows it (reachable in bf16), which would turn the subsequent
    // max-subtract into inf - inf.
    float2 qi = Traits::to_float2(qr[i]);
    qi.x *= scale;
    qi.y *= scale;

    // Per-lane partial dot products for the whole row first, then one butterfly
    // over the array. The reduction is a five-step dependent chain either way, but
    // batching it this way gives those five steps up to SEQ independent shuffles to
    // overlap instead of serialising SEQ separate chains.
    float score[SEQ];
#pragma unroll
    for (int j = 0; j < SEQ; ++j) {
      if (j <= jmax) {
        const float2 kj = Traits::to_float2(kr[j]);
        score[j] = qi.x * kj.x + qi.y * kj.y;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int j = 0; j < SEQ; ++j) {
        if (j <= jmax) {
          score[j] += __shfl_xor_sync(0xffffffffu, score[j], offset);
        }
      }
    }

    float row_max = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < SEQ; ++j) {
      if (j <= jmax) {
        row_max = fmaxf(row_max, score[j]);
      }
    }

    float denom = 0.0f;
    float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int j = 0; j < SEQ; ++j) {
      if (j <= jmax) {
        const float p = expf(score[j] - row_max);
        denom += p;
        const float2 vj = Traits::to_float2(vr[j]);
        acc.x = fmaf(p, vj.x, acc.x);
        acc.y = fmaf(p, vj.y, acc.y);
      }
    }

    const float inv = 1.0f / denom;
    acc.x *= inv;
    acc.y *= inv;
    *reinterpret_cast<vec*>(op + static_cast<long long>(i) * out_layout.time) =
        Traits::from_float2(acc);
  }
}

// Launch geometry, fixed. At the captured bsz*H*W*heads = 2304 pairs, 8 warps per
// block is 288 CTAs against 148 SMs -- 1.95x the SM count, which is the margin a
// non-persistent kernel needs, since a grid at or below the SM count leaves the
// machine unable to overlap anything.
constexpr int kWarpsPerBlock = 8;

// Host-side dispatch over the compile-time temporal extent and mask mode. Returns
// false if the extent is outside the compiled window, so callers can route the
// input elsewhere instead of launching nothing.
template <typename T>
bool launch(const T* qkv, const T* cos_sin, T* out,
            int batch, int seq, int spatial, int heads,
            Layout in_layout, Layout out_layout, QkvOffsets off,
            float scale, bool causal, int warps_per_block, cudaStream_t stream) {
  if (batch <= 0 || spatial <= 0 || heads <= 0 || warps_per_block <= 0) return false;
  const int pairs = batch * spatial * heads;
  if (pairs <= 0) return false;
  const int threads = warps_per_block * 32;
  const int blocks = (pairs + warps_per_block - 1) / warps_per_block;

#define TEMPORAL_AXIAL_DISPATCH(N)                                               \
  case N:                                                                        \
    if (causal) {                                                                \
      kernel<T, N, true><<<blocks, threads, 0, stream>>>(                        \
          qkv, cos_sin, out, pairs, heads, spatial, in_layout, out_layout, off,  \
          scale);                                                                \
    } else {                                                                     \
      kernel<T, N, false><<<blocks, threads, 0, stream>>>(                       \
          qkv, cos_sin, out, pairs, heads, spatial, in_layout, out_layout, off,  \
          scale);                                                                \
    }                                                                            \
    return true;

  switch (seq) {
    TEMPORAL_AXIAL_DISPATCH(1)
    TEMPORAL_AXIAL_DISPATCH(2)
    TEMPORAL_AXIAL_DISPATCH(3)
    TEMPORAL_AXIAL_DISPATCH(4)
    TEMPORAL_AXIAL_DISPATCH(5)
    TEMPORAL_AXIAL_DISPATCH(6)
    TEMPORAL_AXIAL_DISPATCH(7)
    TEMPORAL_AXIAL_DISPATCH(8)
    default:
      return false;
  }
#undef TEMPORAL_AXIAL_DISPATCH
}

}  // namespace temporal_axial
