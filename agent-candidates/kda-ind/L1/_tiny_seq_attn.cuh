// Tiny-sequence dense attention: one warp per (batch, head) pair.
//
// Serves the temporal-attention regime where the sequence extent is a handful of
// tokens but the (batch x head) product is in the thousands -- e.g. 2304 pairs of
// a 5x5 problem at head_dim 64. Stock flash kernels tile 128x64 there, so >96% of
// every tile is masked padding and latency becomes independent of the real work.
//
// Decomposition: head_dim 64 with fp16/bf16 means 32 lanes x 2 elements exactly
// covers one row, so a warp reading a row issues a single fully-utilised 128-byte
// burst. Each lane keeps its own 2-element slice of every q/k/v row in registers;
// scores are warp-reduced to a lane-uniform value, so the <=8-wide softmax is
// recomputed redundantly in every lane and needs no further communication. There
// is no cross-warp traffic and therefore no __syncthreads anywhere.
//
// Every stride is a runtime argument: the value tensor in this regime is commonly
// a transpose(0,1) view of a fused qkv slice, and materialising it contiguously
// would cost more than the entire target latency.
//
// This header is deliberately free of any framework dependency so the profiling
// harness and the PyTorch binding compile the *same* kernel source.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace tinyseq {

// Packed-pair traits. Conversions go through float2 explicitly so the score and
// epilogue accumulations really happen in fp32 -- packed half arithmetic would
// silently accumulate at input precision.
template <typename T>
struct Packed;

template <>
struct Packed<__half> {
  using vec = __half2;
  static __device__ __forceinline__ float2 to_float2(vec x) { return __half22float2(x); }
  static __device__ __forceinline__ vec from_float2(float2 x) { return __float22half2_rn(x); }
};

template <>
struct Packed<__nv_bfloat16> {
  using vec = __nv_bfloat162;
  static __device__ __forceinline__ float2 to_float2(vec x) { return __bfloat1622float2(x); }
  static __device__ __forceinline__ vec from_float2(float2 x) { return __float22bfloat162_rn(x); }
};

// Strides of one 4-D (batch, seq, head, head_dim) tensor, in elements. head_dim is
// always the unit-stride axis, so it needs no entry.
struct Strides {
  long long batch;
  long long seq;
  long long head;
};

// SEQ is the exact sequence extent, so every loop below unrolls completely and no
// row is ever predicated off. CAUSAL selects j <= i; the diagonal is always
// included, so no row is fully masked and -inf / NaN cannot arise.
template <typename T, int SEQ, bool CAUSAL>
__global__ void kernel(const T* __restrict__ q_ptr,
                       const T* __restrict__ k_ptr,
                       const T* __restrict__ v_ptr,
                       T* __restrict__ o_ptr,
                       int pairs,
                       int batch,
                       int heads,
                       int head_major,
                       float scale,
                       Strides qs, Strides ks, Strides vs, Strides os) {
  using vec = typename Packed<T>::vec;

  const int lane = threadIdx.x & 31;
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int pair = blockIdx.x * static_cast<int>(blockDim.x >> 5) + warp;
  // Warp-uniform: every lane of a warp exits together, which is what makes the
  // full-mask shuffles below well-defined.
  if (pair >= pairs) return;

  int b, h;
  if (head_major) {
    h = pair / batch;
    b = pair - h * batch;
  } else {
    b = pair / heads;
    h = pair - b * heads;
  }

  // head_dim is the unit-stride axis in every layout we accept, so a lane's two
  // elements are adjacent and the pair load is a single 4-byte access.
  const long long d = static_cast<long long>(lane) * 2;
  const T* qb = q_ptr + b * qs.batch + h * qs.head + d;
  const T* kb = k_ptr + b * ks.batch + h * ks.head + d;
  const T* vb = v_ptr + b * vs.batch + h * vs.head + d;
  T* ob = o_ptr + b * os.batch + h * os.head + d;

  vec qr[SEQ], kr[SEQ], vr[SEQ];
#pragma unroll
  for (int s = 0; s < SEQ; ++s) {
    qr[s] = *reinterpret_cast<const vec*>(qb + s * qs.seq);
    kr[s] = *reinterpret_cast<const vec*>(kb + s * ks.seq);
    vr[s] = *reinterpret_cast<const vec*>(vb + s * vs.seq);
  }

  // One query row at a time: holding all SEQ^2 scores at once buys nothing here
  // and costs registers.
#pragma unroll
  for (int i = 0; i < SEQ; ++i) {
    const int jmax = CAUSAL ? i : (SEQ - 1);
    // Scale the query row once, before the dot product, rather than scaling each
    // score after the reduction. Two multiplies instead of up to SEQ, and it keeps
    // the reduction operating on the same magnitudes the reference kernels use --
    // scaling afterwards lets the unscaled sum overflow fp32 for inputs whose
    // exponent range allows it (reachable in bf16), which would turn the
    // subsequent max-subtract into inf - inf.
    float2 qi = Packed<T>::to_float2(qr[i]);
    qi.x *= scale;
    qi.y *= scale;

    // Per-lane partial dot products for the whole row first, then one butterfly
    // over the array. The reduction is a five-step dependent chain either way, but
    // batching it this way gives those five steps up to SEQ independent shuffles
    // to overlap instead of serialising SEQ separate chains.
    float score[SEQ];
#pragma unroll
    for (int j = 0; j < SEQ; ++j) {
      if (j <= jmax) {
        const float2 kj = Packed<T>::to_float2(kr[j]);
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
        const float2 vj = Packed<T>::to_float2(vr[j]);
        acc.x = fmaf(p, vj.x, acc.x);
        acc.y = fmaf(p, vj.y, acc.y);
      }
    }

    const float inv = 1.0f / denom;
    acc.x *= inv;
    acc.y *= inv;
    *reinterpret_cast<vec*>(ob + i * os.seq) = Packed<T>::from_float2(acc);
  }
}

// Host-side dispatch over the compile-time sequence extent and mask mode.
// Returns false if seq is outside the supported window, so callers can route the
// input elsewhere instead of launching nothing.
template <typename T>
bool launch(const T* q, const T* k, const T* v, T* o,
            int batch, int seq, int heads,
            Strides qs, Strides ks, Strides vs, Strides os,
            float scale, bool causal, int warps_per_block, int head_major,
            cudaStream_t stream) {
  const int pairs = batch * heads;
  if (pairs <= 0 || warps_per_block <= 0) return false;
  const int threads = warps_per_block * 32;
  const int blocks = (pairs + warps_per_block - 1) / warps_per_block;

#define TINY_SEQ_DISPATCH(N)                                                     \
  case N:                                                                        \
    if (causal) {                                                                \
      kernel<T, N, true><<<blocks, threads, 0, stream>>>(                         \
          q, k, v, o, pairs, batch, heads, head_major, scale, qs, ks, vs, os);    \
    } else {                                                                     \
      kernel<T, N, false><<<blocks, threads, 0, stream>>>(                        \
          q, k, v, o, pairs, batch, heads, head_major, scale, qs, ks, vs, os);    \
    }                                                                            \
    return true;

  switch (seq) {
    TINY_SEQ_DISPATCH(1)
    TINY_SEQ_DISPATCH(2)
    TINY_SEQ_DISPATCH(3)
    TINY_SEQ_DISPATCH(4)
    TINY_SEQ_DISPATCH(5)
    TINY_SEQ_DISPATCH(6)
    TINY_SEQ_DISPATCH(7)
    TINY_SEQ_DISPATCH(8)
    default:
      return false;
  }
#undef TINY_SEQ_DISPATCH
}

}  // namespace tinyseq
