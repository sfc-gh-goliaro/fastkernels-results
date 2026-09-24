// Fused per-head QK-norm + RoPE, applied in place to the packed QKV buffer.
//
// The baseline runs this as three or more passes: RMSNorm reads the Q slice of
// the QKV buffer and writes a fresh contiguous Q, RoPE reads and rewrites it,
// and (for M-RoPE) the cos/sin rows are first gathered into two temporaries.
// One pass suffices: a warp owns one head, keeps the row in registers between
// the sum-of-squares reduction and the rotation, and writes back into the QKV
// buffer, so Q and K stay strided views of it and the attention kernel reads
// them there.
//
// Numerics follow the baseline exactly on the norm (accumulate x*x in fp32,
// rsqrt(sum/D + eps), one rounding of x*s*w) and on 1-D RoPE (cos/sin read from
// the activation-dtype cache, rotation in fp32, one rounding).  The M-RoPE
// path rotates in fp32 where the baseline's Triton kernel rotates in bf16,
// which is strictly closer to the exact result.
//
// Layout: one thread block per (token, head chunk).  A head of D = 32*E
// elements is held by one warp, E bf16 per lane, as a single 2E-byte vector
// load; the RoPE partner element sits exactly 16 lanes away, so the pairing is
// a __shfl_xor by 16 rather than a second memory pass.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <type_traits>
#include <cuda_runtime.h>

namespace {

using bf16 = __nv_bfloat16;

template <int E>
struct alignas(E * 2) BVec {
  bf16 x[E];
};

constexpr unsigned kFull = 0xffffffffu;

// ROPE: 0 = none, 1 = 1-D (shared position for all sections),
//       2 = M-RoPE with contiguous T/H/W sections,
//       3 = M-RoPE with the sections interleaved every third index (Qwen3-VL).



// ---------------------------------------------------------------------------
// Block-scaled FP8 GEMV: out[m, n] = sum_k a[m, k] * w[n, k], M <= 2.
//
// At one or two tokens the projection is pure weight bandwidth -- 37 MB of FP8
// weight for the QKV projection, 33 MB for the output projection -- and a
// tensor-core GEMM throws away 128x of its shape.  One warp per output row,
// with the activation quantized into shared memory by the block itself so the
// whole linear is a single launch and no intermediate tensors are allocated.
//
// Lane ``l`` owns 16 contiguous FP8 weight bytes, so a warp reads 512 bytes of
// the weight row per step, i.e. four whole 128-element K-blocks.  Each lane
// therefore sits inside exactly one K-block and can fold that block's scale
// product into its own accumulator, which leaves a single warp reduction for
// the entire row instead of one per block.
//
// ``weight_scale`` is DeepGEMM's packed UE8M0 layout: int32[N, K/512] with
// stride (1, N), byte ``b`` of word (n, j) holding the biased exponent of
// K-block 4j+b.  All 32 lanes of a step read the same word, so the decode is a
// broadcast load plus a shift.
// ---------------------------------------------------------------------------
template <int MT>
__global__ void fp8_gemv_kernel(
    const bf16* __restrict__ a, const __nv_fp8_storage_t* __restrict__ w,
    const int32_t* __restrict__ wscale, const bf16* __restrict__ bias,
    bf16* __restrict__ out, const int M, const int K, const int N,
    const int groups) {
  extern __shared__ __align__(16) char smem_raw[];
  __nv_fp8_storage_t* qa = reinterpret_cast<__nv_fp8_storage_t*>(smem_raw);
  float* sa = reinterpret_cast<float*>(qa + MT * K);

  // --- quantize the activation into shared memory (reference semantics) -----
  const int total_groups = M * groups;
  for (int g = threadIdx.x >> 3; g < total_groups; g += (blockDim.x >> 3)) {
    const int lane8 = threadIdx.x & 7;
    const int64_t base = static_cast<int64_t>(g) * 128 + lane8 * 16;
    alignas(16) bf16 regs[16];
    {
      uint4* dst = reinterpret_cast<uint4*>(&regs[0]);
      const uint4* src = reinterpret_cast<const uint4*>(a + base);
      dst[0] = src[0];
      dst[1] = src[1];
    }
    float absmax = 1e-10f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
      absmax = fmaxf(absmax, fabsf(__bfloat162float(regs[i])));
    const unsigned mask = 0xffu << (threadIdx.x & 24u);
    absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, 4));
    absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, 2));
    absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, 1));
    float y_s;
    if (lane8 == 0) {
      y_s = absmax / 448.0f;
      y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
      sa[g] = y_s;
    }
    y_s = __shfl_sync(mask, y_s, 0, 8);
    __nv_fp8x2_storage_t qv[8];
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      float2 q;
      q.x = fminf(fmaxf(__bfloat162float(regs[i]) / y_s, -448.0f), 448.0f);
      q.y = fminf(fmaxf(__bfloat162float(regs[i + 1]) / y_s, -448.0f), 448.0f);
      qv[i / 2] = __nv_cvt_float2_to_fp8x2(q, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint4*>(qa + base) =
        *reinterpret_cast<const uint4*>(&qv[0]);
  }
  __syncthreads();

  // --- one warp per output row --------------------------------------------
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= N) return;

  const int sub = lane & 7;        // 16-byte slot inside the K-block
  const int blk = lane >> 3;       // which of the four K-blocks of this step
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;

  const __nv_fp8_storage_t* wrow = w + static_cast<int64_t>(row) * K;
  for (int g0 = 0; g0 < groups; g0 += 4) {
    const int g = g0 + blk;
    const int64_t off = static_cast<int64_t>(g) * 128 + sub * 16;
    const uint4 wv = *reinterpret_cast<const uint4*>(wrow + off);
    // Packed UE8M0: one broadcast word per step, one byte per K-block.
    const int32_t word = wscale[static_cast<int64_t>(g0 >> 2) * N + row];
    const int e = (word >> (8 * blk)) & 0xFF;
    const float ws = __int_as_float(e << 23);
    const __nv_fp8x2_storage_t* wp =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(&wv);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < M) {
        const uint4 av = *reinterpret_cast<const uint4*>(
            qa + static_cast<int64_t>(m) * K + off);
        const __nv_fp8x2_storage_t* ap =
            reinterpret_cast<const __nv_fp8x2_storage_t*>(&av);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float2 af =
              __half22float2(__nv_cvt_fp8x2_to_halfraw2(ap[i], __NV_E4M3));
          const float2 wf =
              __half22float2(__nv_cvt_fp8x2_to_halfraw2(wp[i], __NV_E4M3));
          part += af.x * wf.x + af.y * wf.y;
        }
        acc[m] += part * sa[m * groups + g] * ws;
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m < M) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        acc[m] += __shfl_xor_sync(kFull, acc[m], off);
    }
  }
  if (lane == 0) {
    const float b = bias ? __bfloat162float(bias[row]) : 0.f;
#pragma unroll
    for (int m = 0; m < MT; ++m)
      if (m < M)
        out[static_cast<int64_t>(m) * N + row] = __float2bfloat16(acc[m] + b);
  }
}

// ---------------------------------------------------------------------------
// Per-token-group FP8 activation quantization (group = 128, UE8M0 scales,
// column-major scale layout).
//
// This is the pass in front of the block-scaled FP8 GEMM.  The reference kernel
// stages every group through shared memory behind a __syncthreads and runs at
// ~0.94 TB/s, which at 16384 tokens costs 215 us for the QKV projection's
// activation and 422 us for the output projection's -- more than either GEMM,
// which DeepGEMM finishes at ~3.5 PFLOP/s.  Here a group is 8 lanes x 16 bf16,
// held in registers across the absmax reduce, the scale, and the quantize, so
// the group is read once (two 16-byte loads) and written once (one 16-byte
// store) with no shared memory and no block barrier.
//
// Numerics are the reference's, operation for operation -- absmax seeded with
// eps, ``local_absmax / 448``, then ``exp2f(ceilf(log2f(fmaxf(fabsf(y_s),
// 1e-10f))))`` for the UE8M0 rounding, then ``fminf(fmaxf(x / y_s, -448),
// 448)`` into e4m3 -- so the quantized activation and its scales come out
// bit-identical and the GEMM result is unchanged.
// ---------------------------------------------------------------------------
template <int kThreadsPerGroup>
__global__ void quant_fp8_group128_kernel(
    const bf16* __restrict__ input, __nv_fp8_storage_t* __restrict__ out_q,
    float* __restrict__ out_s, const int groups_per_row, const int m_rows,
    const int64_t num_groups, const float eps, const float fp8_min,
    const float fp8_max) {
  constexpr int kVec = 128 / kThreadsPerGroup;
  using LoadV = typename std::conditional<kVec == 16, uint4, uint2>::type;
  using StoreV = typename std::conditional<kVec == 16, uint4, uint2>::type;

  const int64_t gid =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x)
      / kThreadsPerGroup;
  if (gid >= num_groups) return;
  const int lane = threadIdx.x % kThreadsPerGroup;

  const int64_t base = gid * 128 + lane * kVec;
  alignas(16) bf16 regs[kVec];
  {
    // kVec = 16 -> two 16-byte loads; kVec = 8 -> one 16-byte load.
    uint4* dst = reinterpret_cast<uint4*>(&regs[0]);
    const uint4* src = reinterpret_cast<const uint4*>(input + base);
#pragma unroll
    for (int i = 0; i < kVec * 2 / 16; ++i) dst[i] = src[i];
  }

  float absmax = eps;
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    absmax = fmaxf(absmax, fabsf(__bfloat162float(regs[i])));

  // Reduce across the lanes of this group (one octet / half of the warp).
  const unsigned mask = (kThreadsPerGroup == 8)
                            ? (0xffu << (threadIdx.x & 24u))
                            : ((threadIdx.x & 16u) ? 0xffff0000u : 0x0000ffffu);
  if constexpr (kThreadsPerGroup == 16)
    absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, 8));
  absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, 4));
  absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, 2));
  absmax = fmaxf(absmax, __shfl_xor_sync(mask, absmax, 1));

  // Only lane 0 evaluates the (multi-instruction) log2/ceil/exp2 chain; the
  // rest of the group reads it back off the shuffle.
  float y_s;
  if (lane == 0) {
    y_s = absmax / fp8_max;
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }
  y_s = __shfl_sync(mask, y_s, 0, kThreadsPerGroup);

  if (lane == 0) {
    // Column-major scales: element (row, group) lives at group * m_rows + row.
    const int row = static_cast<int>(gid / groups_per_row);
    const int grp = static_cast<int>(gid % groups_per_row);
    out_s[static_cast<int64_t>(grp) * m_rows + row] = y_s;
  }

  __nv_fp8x2_storage_t qv[kVec / 2];
#pragma unroll
  for (int i = 0; i < kVec; i += 2) {
    float2 q;
    q.x = fminf(fmaxf(__bfloat162float(regs[i]) / y_s, fp8_min), fp8_max);
    q.y = fminf(fmaxf(__bfloat162float(regs[i + 1]) / y_s, fp8_min), fp8_max);
    qv[i / 2] = __nv_cvt_float2_to_fp8x2(q, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<StoreV*>(out_q + base) =
      *reinterpret_cast<const StoreV*>(&qv[0]);
}

template <int E>
__global__ void attn_small_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, bf16* __restrict__ out,
    const int64_t qs0, const int64_t qs1, const int64_t ks0, const int64_t ks1,
    const int64_t vs0, const int64_t vs1, const int64_t os0, const int64_t os1,
    const int32_t* __restrict__ cu, const float* __restrict__ sinks,
    const int window_left, const float scale, const int group) {
  using V = BVec<E>;
  const int lane = threadIdx.x & 31;
  const int seq = blockIdx.z;
  const int h = blockIdx.y;
  const int begin = cu[seq];
  const int len = cu[seq + 1] - begin;
  const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (i >= len) return;
  const int kvh = h / group;

  const V qv = *reinterpret_cast<const V*>(
      q + static_cast<int64_t>(begin + i) * qs0 + h * qs1 + lane * E);
  float qf[E];
#pragma unroll
  for (int e = 0; e < E; ++e) qf[e] = __bfloat162float(qv.x[e]);

  float m, lsum, acc[E];
  if (sinks != nullptr) {
    m = sinks[h];
    lsum = 1.0f;
  } else {
    m = -INFINITY;
    lsum = 0.0f;
  }
#pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.0f;

  const int lo = (window_left >= 0) ? max(0, i - window_left) : 0;
  const bf16* kp = k + static_cast<int64_t>(begin + lo) * ks0 + kvh * ks1
                   + lane * E;
  const bf16* vp = v + static_cast<int64_t>(begin + lo) * vs0 + kvh * vs1
                   + lane * E;
  for (int j = lo; j <= i; ++j, kp += ks0, vp += vs0) {
    const V kv = *reinterpret_cast<const V*>(kp);
    float dot = 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) dot += qf[e] * __bfloat162float(kv.x[e]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(kFull, dot, off);

    const float sc = dot * scale;
    const float mn = fmaxf(m, sc);
    const float corr = __expf(m - mn);
    const float pe = __expf(sc - mn);
    lsum = lsum * corr + pe;
    const V vv = *reinterpret_cast<const V*>(vp);
#pragma unroll
    for (int e = 0; e < E; ++e)
      acc[e] = acc[e] * corr + pe * __bfloat162float(vv.x[e]);
    m = mn;
  }

  const float inv = 1.0f / lsum;
  V ov;
#pragma unroll
  for (int e = 0; e < E; ++e) ov.x[e] = __float2bfloat16(acc[e] * inv);
  *reinterpret_cast<V*>(out + static_cast<int64_t>(begin + i) * os0 + h * os1
                        + lane * E) = ov;
}

template <int E, bool HAS_NORM, int ROPE>
__global__ void qk_norm_rope_kernel(
    bf16* __restrict__ qkv, const int64_t row_stride,
    const bf16* __restrict__ qw, const bf16* __restrict__ kw, const float eps,
    const bf16* __restrict__ cache,
    const int64_t* __restrict__ positions, const int64_t pos_row_stride,
    const int64_t pos_tok_stride, const int p0, const int p1,
    const int n_q_heads, const int total_heads, const int heads_per_block) {
  constexpr int D = 32 * E;
  constexpr int HALF = D / 2;
  using V = BVec<E>;

  const int tok = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n_warps = blockDim.x >> 5;

  const int h_begin = blockIdx.y * heads_per_block;
  int h_end = h_begin + heads_per_block;
  if (h_end > total_heads) h_end = total_heads;

  float cosv[E], sinv[E];
  if constexpr (ROPE != 0) {
    const int base = (lane & 15) * E;
    if constexpr (ROPE == 1) {
      const bf16* c = cache + positions[tok * pos_tok_stride] * D;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        cosv[e] = __bfloat162float(c[base + e]);
        sinv[e] = __bfloat162float(c[HALF + base + e]);
      }
    } else {
#pragma unroll
      for (int e = 0; e < E; ++e) {
        const int jj = base + e;
        int sec;
        if constexpr (ROPE == 2) {
          sec = jj < p0 ? 0 : (jj < p1 ? 1 : 2);
        } else {
          // Mirrors the reference masks: index jj belongs to H when
          // jj % 3 == 1 && jj <= 3 * section_h, to W when jj % 3 == 2 &&
          // jj <= 3 * section_w, and to T otherwise.
          const int r = jj % 3;
          sec = (r == 1 && jj <= p0) ? 1 : ((r == 2 && jj <= p1) ? 2 : 0);
        }
        const bf16* c =
            cache + positions[sec * pos_row_stride + tok * pos_tok_stride] * D;
        cosv[e] = __bfloat162float(c[jj]);
        sinv[e] = __bfloat162float(c[HALF + jj]);
      }
    }
  }

  bf16* row = qkv + static_cast<int64_t>(tok) * row_stride;

  for (int h = h_begin + warp; h < h_end; h += n_warps) {
    bf16* hp = row + h * D;
    V vv = *reinterpret_cast<const V*>(hp + lane * E);

    float x[E];
#pragma unroll
    for (int e = 0; e < E; ++e) x[e] = __bfloat162float(vv.x[e]);

    if constexpr (HAS_NORM) {
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < E; ++e) ss += x[e] * x[e];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(kFull, ss, off);
      const float s = rsqrtf(ss / D + eps);
      const bf16* w = (h < n_q_heads) ? qw : kw;
      const V wv = *reinterpret_cast<const V*>(w + lane * E);
#pragma unroll
      for (int e = 0; e < E; ++e) {
        x[e] = __bfloat162float(
            __float2bfloat16(x[e] * s * __bfloat162float(wv.x[e])));
      }
    }

    if constexpr (ROPE == 1) {
      // Reference: the CUDA rotary kernel promotes to fp32, rotates, and
      // rounds once -- bit-identical here.
      const bool first = lane < 16;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        const float y = __shfl_xor_sync(kFull, x[e], 16);
        x[e] = first ? (x[e] * cosv[e] - y * sinv[e])
                     : (x[e] * cosv[e] + y * sinv[e]);
      }
    } else if constexpr (ROPE >= 2) {
      // Reference: the M-RoPE Triton kernel's operands are bf16, and the
      // generated code rounds the product that involves the *second* half of
      // the pair while contracting the other into the add:
      //     out_lo = bf16(x_lo * cos - bf16(x_hi * sin))
      //     out_hi = bf16(x_lo * sin + bf16(x_hi * cos))
      // Reproducing that placement of the roundings matters: rotating entirely
      // in fp32 is more accurate but drifts far enough through attention to
      // push ~0.7% of output elements outside the scorer's per-element band,
      // while rounding every product rounds one time too many.  This form is
      // bit-identical to the reference on the captured shapes.
      const bool first = lane < 16;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        const float partner = __shfl_xor_sync(kFull, x[e], 16);
        const float lo = first ? x[e] : partner;
        const float hi = first ? partner : x[e];
        x[e] = first
            ? lo * cosv[e]
                  - __bfloat162float(__float2bfloat16(hi * sinv[e]))
            : lo * sinv[e]
                  + __bfloat162float(__float2bfloat16(hi * cosv[e]));
      }
    }

#pragma unroll
    for (int e = 0; e < E; ++e) vv.x[e] = __float2bfloat16(x[e]);
    *reinterpret_cast<V*>(hp + lane * E) = vv;
  }
}

template <int E, bool HAS_NORM, int ROPE>
void launch(torch::Tensor& qkv, const bf16* qw, const bf16* kw, float eps,
            const bf16* cache, const int64_t* pos, int64_t pos_row_stride,
            int64_t pos_tok_stride, int p0, int p1, int n_q_heads,
            int total_heads, int n_tokens) {
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  // One block per token (warps loop over heads, reusing the cos/sin they
  // already hold) once there are enough tokens to fill the device; below that,
  // split the heads across blocks instead so the launch is not one-SM wide.
  const int heads_per_block =
      (n_tokens >= 512) ? total_heads : kWarps;
  const int hchunks = (total_heads + heads_per_block - 1) / heads_per_block;
  dim3 grid(n_tokens, hchunks);
  qk_norm_rope_kernel<E, HAS_NORM, ROPE>
      <<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<bf16*>(qkv.data_ptr()), qkv.stride(0), qw, kw, eps,
          cache, pos, pos_row_stride, pos_tok_stride, p0, p1, n_q_heads,
          total_heads, heads_per_block);
}

template <int E>
void dispatch_flags(torch::Tensor& qkv, const bf16* qw, const bf16* kw,
                    float eps, const bf16* cache, const int64_t* pos,
                    int64_t pos_row_stride, int64_t pos_tok_stride, int p0,
                    int p1, int rope, int n_q_heads, int total_heads,
                    int n_tokens) {
  const bool hn = (qw != nullptr);
#define CASE(HN, R)                                                          \
  launch<E, HN, R>(qkv, qw, kw, eps, cache, pos, pos_row_stride,             \
                   pos_tok_stride, p0, p1, n_q_heads, total_heads, n_tokens)
  if (hn) {
    if (rope == 3) CASE(true, 3);
    else if (rope == 2) CASE(true, 2);
    else if (rope == 1) CASE(true, 1);
    else CASE(true, 0);
  } else {
    if (rope == 3) CASE(false, 3);
    else if (rope == 2) CASE(false, 2);
    else if (rope == 1) CASE(false, 1);
    else CASE(false, 0);
  }
#undef CASE
}

}  // namespace

// qkv is modified in place.  ``qw``/``kw`` empty -> no norm; ``rope`` 0..3.
// ``p0``/``p1`` are the section bounds: (s_t, s_t + s_h) for rope == 2 and
// (3 * s_h, 3 * s_w) for the interleaved rope == 3.
void qk_norm_rope(torch::Tensor qkv, torch::Tensor qw, torch::Tensor kw,
                  double eps, torch::Tensor cos_sin, torch::Tensor positions,
                  int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim,
                  int64_t p0, int64_t p1, int64_t rope) {
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16, "qkv must be bfloat16");
  TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
  const int n_tokens = static_cast<int>(qkv.size(0));
  if (n_tokens == 0) return;

  const bf16* qw_p =
      qw.numel() ? reinterpret_cast<const bf16*>(qw.data_ptr()) : nullptr;
  const bf16* kw_p =
      kw.numel() ? reinterpret_cast<const bf16*>(kw.data_ptr()) : nullptr;
  const bf16* cache_p =
      cos_sin.numel() ? reinterpret_cast<const bf16*>(cos_sin.data_ptr())
                      : nullptr;
  const int64_t* pos_p =
      positions.numel() ? positions.data_ptr<int64_t>() : nullptr;
  const int64_t pos_row_stride = (rope >= 2) ? positions.stride(0) : 0;
  const int64_t pos_tok_stride =
      positions.numel() ? positions.stride(-1) : 0;
  const int total_heads = static_cast<int>(n_q_heads + n_kv_heads);

  if (head_dim == 128) {
    dispatch_flags<4>(qkv, qw_p, kw_p, static_cast<float>(eps), cache_p, pos_p,
                      pos_row_stride, pos_tok_stride, static_cast<int>(p0),
                      static_cast<int>(p1), static_cast<int>(rope),
                      static_cast<int>(n_q_heads), total_heads, n_tokens);
  } else {
    dispatch_flags<2>(qkv, qw_p, kw_p, static_cast<float>(eps), cache_p, pos_p,
                      pos_row_stride, pos_tok_stride, static_cast<int>(p0),
                      static_cast<int>(p1), static_cast<int>(rope),
                      static_cast<int>(n_q_heads), total_heads, n_tokens);
  }
}




// Block-scaled FP8 GEMV for M <= 2: quantizes the activation itself, so this is
// the whole linear in one launch.
torch::Tensor fp8_gemv(torch::Tensor a, torch::Tensor w, torch::Tensor wscale,
                       c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.is_contiguous());
  TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.is_contiguous());
  TORCH_CHECK(wscale.scalar_type() == at::kInt && wscale.stride(0) == 1);
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(w.size(1) == K && K % 512 == 0 && M <= 2);
  const int groups = K / 128;

  auto out = torch::empty({a.size(0), w.size(0)}, a.options());
  const bf16* bias_p =
      (bias.has_value() && bias->numel())
          ? reinterpret_cast<const bf16*>(bias->data_ptr()) : nullptr;

  constexpr int kThreads = 256;
  const int rows_per_block = kThreads / 32;
  const dim3 grid((N + rows_per_block - 1) / rows_per_block);
  auto stream = at::cuda::getCurrentCUDAStream();
#define GLAUNCH(MT)                                                           \
  do {                                                                        \
    const size_t smem = static_cast<size_t>(MT) * K                           \
                        + static_cast<size_t>(MT) * groups * sizeof(float);    \
    fp8_gemv_kernel<MT><<<grid, kThreads, smem, stream>>>(                     \
        reinterpret_cast<const bf16*>(a.data_ptr()),                           \
        reinterpret_cast<const __nv_fp8_storage_t*>(w.data_ptr()),             \
        wscale.data_ptr<int32_t>(), bias_p,                                    \
        reinterpret_cast<bf16*>(out.data_ptr()), M, K, N, groups);             \
  } while (0)
  if (M == 1) GLAUNCH(1);
  else GLAUNCH(2);
#undef GLAUNCH
  return out;
}

// Short-sequence varlen causal attention.  q: [T, Hq, D], k/v: [T, Hkv, D]
// (arbitrary token/head strides), out: [T, Hq, D] contiguous.
torch::Tensor attn_small(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                        torch::Tensor cu_seqlens, torch::Tensor sinks,
                        int64_t window_left, double scale,
                        int64_t max_seqlen) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16);
  const int64_t D = q.size(2);
  const int64_t Hq = q.size(1);
  TORCH_CHECK(D == 64 || D == 128);
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1);
  const int num_seqs = static_cast<int>(cu_seqlens.size(0)) - 1;
  const int group = static_cast<int>(Hq / k.size(1));

  auto out = torch::empty({q.size(0), Hq, D}, q.options());
  const float* sinks_p =
      sinks.numel() ? sinks.data_ptr<float>() : nullptr;

  constexpr int kThreads = 128;
  const int rows = kThreads / 32;
  dim3 grid((static_cast<int>(max_seqlen) + rows - 1) / rows,
            static_cast<int>(Hq), num_seqs);
  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(EE)                                                            \
  attn_small_kernel<EE><<<grid, kThreads, 0, stream>>>(                       \
      reinterpret_cast<const bf16*>(q.data_ptr()),                            \
      reinterpret_cast<const bf16*>(k.data_ptr()),                            \
      reinterpret_cast<const bf16*>(v.data_ptr()),                            \
      reinterpret_cast<bf16*>(out.data_ptr()), q.stride(0), q.stride(1),       \
      k.stride(0), k.stride(1), v.stride(0), v.stride(1), out.stride(0),       \
      out.stride(1), cu_seqlens.data_ptr<int32_t>(), sinks_p,                  \
      static_cast<int>(window_left), static_cast<float>(scale), group)
  if (D == 128) LAUNCH(4);
  else LAUNCH(2);
#undef LAUNCH
  return out;
}


// In-place per-token-group FP8 quantization of a contiguous [M, K] activation.
// ``out_s`` must be the column-major (stride (1, M)) scale view the block-scaled
// GEMM expects.
void quant_fp8_group128(torch::Tensor input, torch::Tensor out_q,
                        torch::Tensor out_s) {
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.is_contiguous());
  TORCH_CHECK(input.dim() == 2 && out_q.is_contiguous());
  TORCH_CHECK(out_q.scalar_type() == at::kFloat8_e4m3fn);
  const int m_rows = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  TORCH_CHECK(K % 128 == 0);
  TORCH_CHECK(out_s.scalar_type() == at::kFloat && out_s.stride(0) == 1
              && out_s.stride(1) == m_rows);
  const int groups_per_row = K / 128;
  const int64_t num_groups = static_cast<int64_t>(m_rows) * groups_per_row;

  // 8 lanes x 16 bf16 per group measured fastest on this device: two 16-byte
  // loads and one 16-byte store per lane beat 16 lanes x 8 (which halves the
  // store width) by ~1.3x on the large activations.
  constexpr int kThreads = 256;
  constexpr int lanes_per_group = 8;
  const int64_t threads_needed = num_groups * lanes_per_group;
  const int64_t blocks = (threads_needed + kThreads - 1) / kThreads;
  auto stream = at::cuda::getCurrentCUDAStream();
#define QLAUNCH(LPG)                                                          \
  quant_fp8_group128_kernel<LPG><<<static_cast<int>(blocks), kThreads, 0,     \
                                   stream>>>(                                 \
      reinterpret_cast<const bf16*>(input.data_ptr()),                        \
      reinterpret_cast<__nv_fp8_storage_t*>(out_q.data_ptr()),                \
      out_s.data_ptr<float>(), groups_per_row, m_rows, num_groups, 1e-10f,    \
      -448.0f, 448.0f)
  QLAUNCH(lanes_per_group);
#undef QLAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_norm_rope", &qk_norm_rope, "fused per-head QK RMSNorm + RoPE");
  m.def("fp8_gemv", &fp8_gemv, "block-scaled FP8 GEMV (M <= 2)");
  m.def("attn_small", &attn_small, "short-sequence varlen causal attention");
  m.def("quant_fp8_group128", &quant_fp8_group128,
        "per-token-group FP8 activation quantization (group 128, UE8M0)");
}
