// Custom kernels for the LlamaAttention candidate.  Three of them:
//
//   qkv_post          fused per-head RMSNorm + RoPE, in place on the packed
//                     QKV projection output
//   quant_fp8_groups  per-token-group FP8 activation quantization for the
//                     block-scaled projections
//   attn_short        varlen causal attention for short key spans (decode)
//
// Each replaces a chain of eager ops (or a library kernel) that measurement
// showed to be the bottleneck for this layer; see the comment above each.
//
// ---------------------------------------------------------------------------
// qkv_post
//
// One pass over the packed [N, (nh + 2*nkv)*hd] projection output applies, in
// place and per head:
//     optional per-head RMSNorm (Qwen3 q_norm / k_norm)  ->  optional RoPE
// so the norm + rope + gather + contiguous-copy chain the eager path walks
// (five passes over q, plus a bf16 cast of the whole cos/sin cache) collapses
// into a single read-modify-write.  V is untouched: the attention kernel reads
// it straight out of the packed buffer as a strided view, exactly as the
// baseline does for the configs without QK-norm.
//
// One warp per (token, head): lane L owns the CH = hd/32 contiguous elements at
// [L*CH, L*CH+CH).  RoPE pairs element i with i + hd/2, which by construction
// lives in lane L^16, so the rotation needs a single warp shuffle and no shared
// memory.
// ---------------------------------------------------------------------------

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

// Adjacent heads processed by one warp.  The kernel is bandwidth bound and one
// head per warp leaves a single request per lane in flight, so batching heads
// lifts throughput a lot: measured 0.91 -> 1.08 -> 1.23 TB/s at 1 / 2 / 4 heads
// for a 1000-token prefill (1.14 -> 1.56 -> 1.90 at 16384).  Four is fastest for
// prefill but adds a launch's worth of blocks at N=1, where this layer is host
// bound, so two is the balanced choice.
constexpr int kHeadsPerWarp = 2;

constexpr int kRopeNone = 0;   // no rotary
constexpr int kRope1D = 1;     // positions [N]         (neox, full head)
constexpr int kRopeMInter = 2; // positions [3, N]      (mrope, interleaved)
constexpr int kRopeMSect = 3;  // positions [3, N]      (mrope, sectioned)

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// mrope section -> which of the three position rows feeds angle j.
template <int ROPE>
__device__ __forceinline__ int mrope_dim(int j, int sec_t, int sec_h,
                                         int sec_w) {
  if (ROPE == kRopeMInter) {
    // Mirrors the reference Triton masks, including their ``<=`` bound.
    const int m = j % 3;
    if (m == 1 && j <= 3 * sec_h) return 1;
    if (m == 2 && j <= 3 * sec_w) return 2;
    return 0;
  }
  if (j < sec_t) return 0;
  if (j < sec_t + sec_h) return 1;
  return 2;
}

template <int HD, bool HAS_NORM, int ROPE, int HW>
__global__ void qkv_post_kernel(__nv_bfloat16 *__restrict__ qkv,
                                const int64_t *__restrict__ positions,
                                const __nv_bfloat16 *__restrict__ q_weight,
                                const __nv_bfloat16 *__restrict__ k_weight,
                                const __nv_bfloat16 *__restrict__ cos_sin,
                                const float eps, const int num_tokens,
                                const int row_stride, const int nh,
                                const int nkv, const int sec_t, const int sec_h,
                                const int sec_w) {
  constexpr int CH = HD / 32;   // elements per lane
  constexpr int CH2 = CH / 2;   // bf16x2 packets per lane
  constexpr int HALF = HD / 2;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int head0 = (blockIdx.x * (blockDim.x >> 5) + warp) * HW;
  const int nheads = nh + nkv;
  if (head0 >= nheads) return;
  const int token = blockIdx.y;

  __nv_bfloat16 *row = qkv + (size_t)token * (size_t)row_stride;
  // Heads are laid out [q heads][k heads][v heads]; we only touch q and k.
  // A warp owns HW *adjacent* heads and issues all of their loads before
  // touching any of them, so the (bandwidth-bound) kernel keeps several
  // requests per lane in flight instead of one.
  __nv_bfloat16 *ptr[HW];
  __nv_bfloat162 v[HW][CH2];
#pragma unroll
  for (int h = 0; h < HW; ++h) {
    ptr[h] = row + (size_t)(head0 + h) * HD + (size_t)lane * CH;
    if (head0 + h < nheads) {
      const __nv_bfloat162 *src =
          reinterpret_cast<const __nv_bfloat162 *>(ptr[h]);
#pragma unroll
      for (int i = 0; i < CH2; ++i) v[h][i] = src[i];
    }
  }

#pragma unroll
  for (int h = 0; h < HW; ++h) {
    if (head0 + h >= nheads) break;
    const int head = head0 + h;
    float x[CH];
#pragma unroll
    for (int i = 0; i < CH2; ++i) {
      x[2 * i] = __bfloat162float(v[h][i].x);
      x[2 * i + 1] = __bfloat162float(v[h][i].y);
    }

    if (HAS_NORM) {
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < CH; ++i) ss += x[i] * x[i];
      ss = warp_reduce_sum(ss);
      const float s = rsqrtf(ss / (float)HD + eps);
      const __nv_bfloat16 *w = (head < nh) ? q_weight : k_weight;
      const __nv_bfloat162 *w2 =
          reinterpret_cast<const __nv_bfloat162 *>(w + lane * CH);
#pragma unroll
      for (int i = 0; i < CH2; ++i) {
        const __nv_bfloat162 wv = w2[i];
        // vLLM's rms_norm_kernel: bf16(x * rsqrt(mean + eps) * weight).
        x[2 * i] = __bfloat162float(
            __float2bfloat16(x[2 * i] * s * __bfloat162float(wv.x)));
        x[2 * i + 1] = __bfloat162float(
            __float2bfloat16(x[2 * i + 1] * s * __bfloat162float(wv.y)));
      }
    }

    if (ROPE != kRopeNone) {
      const int jbase = (lane & 15) * CH;
      const int64_t *pos_row = positions + token;
      int64_t pos0 = 0;
      if (ROPE == kRope1D) pos0 = pos_row[0];

      float xr[CH];
#pragma unroll
      for (int i = 0; i < CH; ++i) {
        const int j = jbase + i;
        int64_t pos = pos0;
        if (ROPE != kRope1D) {
          const int d = mrope_dim<ROPE>(j, sec_t, sec_h, sec_w);
          pos = pos_row[(size_t)d * num_tokens];
        }
        const __nv_bfloat16 *crow = cos_sin + pos * HD;
        const float partner = __shfl_xor_sync(0xffffffffu, x[i], 16);
        if (ROPE == kRope1D) {
          // The reference CUDA kernel rotates in fp32 and rounds once, on store.
          const float c = __bfloat162float(crow[j]);
          const float s = __bfloat162float(crow[HALF + j]);
          xr[i] = (lane < 16) ? (x[i] * c - partner * s) : (x[i] * c + partner * s);
        } else {
          // The reference m-rope kernel is Triton over bf16 tensors, so the
          // rotation runs in native bf16 -- and the NVPTX backend contracts one
          // product of each pair into the add.  Which one differs per half, as its
          // PTX shows: the first half is a subtract, whose *left* product is
          // fused,
          //     mul.bf16x2 t, sin, x2 ; fma.rn.bf16x2 out1, cos, x1, -t
          // and the second an add, whose *right* product is fused,
          //     mul.bf16x2 t, cos, x2 ; fma.rn.bf16x2 out2, sin, x1,  t
          // so one product rounds to bf16 and the other does not, mirrored
          // between the halves.  Emitting the same bf16 instructions makes this
          // bit-identical to the reference; an fp32 rewrite differs by a ulp on
          // ~19% of elements, and through the softmax that pushes ~0.7% of the
          // layer's outputs outside the comparison tolerance.
          const __nv_bfloat16 c = crow[j];
          const __nv_bfloat16 s = crow[HALF + j];
          const __nv_bfloat16 self = __float2bfloat16(x[i]);
          const __nv_bfloat16 other = __float2bfloat16(partner);
          xr[i] = __bfloat162float(
              (lane < 16) ? __hfma(c, self, __hneg(__hmul(s, other)))
                          : __hfma(s, other, __hmul(c, self)));
        }
      }
#pragma unroll
      for (int i = 0; i < CH; ++i) x[i] = xr[i];
    }

    __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(ptr[h]);
#pragma unroll
    for (int i = 0; i < CH2; ++i) {
      dst[i] = __nv_bfloat162(__float2bfloat16(x[2 * i]),
                              __float2bfloat16(x[2 * i + 1]));
    }
  }
}

#define DISPATCH_QKV_POST(HD, HAS_NORM, ROPE)                                  \
  qkv_post_kernel<HD, HAS_NORM, ROPE, kHeadsPerWarp><<<grid, block, 0, stream>>>(\
      qkv_ptr, pos_ptr, qw_ptr, kw_ptr, cs_ptr, (float)eps, (int)num_tokens,   \
      (int)row_stride, (int)nh, (int)nkv, (int)sec_t, (int)sec_h, (int)sec_w)

#define DISPATCH_ROPE(HD, HAS_NORM)                                            \
  switch (rope_mode) {                                                         \
    case kRopeNone: DISPATCH_QKV_POST(HD, HAS_NORM, kRopeNone); break;         \
    case kRope1D: DISPATCH_QKV_POST(HD, HAS_NORM, kRope1D); break;             \
    case kRopeMInter: DISPATCH_QKV_POST(HD, HAS_NORM, kRopeMInter); break;     \
    default: DISPATCH_QKV_POST(HD, HAS_NORM, kRopeMSect); break;               \
  }

#define DISPATCH_NORM(HD)                                                      \
  if (has_norm) {                                                              \
    DISPATCH_ROPE(HD, true);                                                   \
  } else {                                                                     \
    DISPATCH_ROPE(HD, false);                                                  \
  }

} // namespace

void qkv_post(at::Tensor qkv, at::Tensor positions,
              std::optional<at::Tensor> q_weight,
              std::optional<at::Tensor> k_weight,
              std::optional<at::Tensor> cos_sin, double eps, int64_t nh,
              int64_t nkv, int64_t head_dim, int64_t rope_mode, int64_t sec_t,
              int64_t sec_h, int64_t sec_w) {
  TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1);
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16);
  const int64_t num_tokens = qkv.size(0);
  const int64_t row_stride = qkv.stride(0);
  const bool has_norm = q_weight.has_value();
  if (num_tokens == 0) return;

  auto *qkv_ptr = reinterpret_cast<__nv_bfloat16 *>(qkv.data_ptr());
  const int64_t *pos_ptr =
      rope_mode == kRopeNone ? nullptr
                             : positions.data_ptr<int64_t>();
  const __nv_bfloat16 *qw_ptr =
      has_norm ? reinterpret_cast<const __nv_bfloat16 *>(q_weight->data_ptr())
               : nullptr;
  const __nv_bfloat16 *kw_ptr =
      has_norm ? reinterpret_cast<const __nv_bfloat16 *>(k_weight->data_ptr())
               : nullptr;
  const __nv_bfloat16 *cs_ptr =
      cos_sin.has_value()
          ? reinterpret_cast<const __nv_bfloat16 *>(cos_sin->data_ptr())
          : nullptr;

  const int warps_per_block = 8;
  const int64_t per_block = warps_per_block * kHeadsPerWarp;
  const int64_t heads = nh + nkv;
  dim3 grid((heads + per_block - 1) / per_block, num_tokens);
  dim3 block(32 * warps_per_block);
  const c10::cuda::CUDAGuard guard(qkv.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (head_dim == 128) {
    DISPATCH_NORM(128);
  } else if (head_dim == 64) {
    DISPATCH_NORM(64);
  } else if (head_dim == 256) {
    DISPATCH_NORM(256);
  } else {
    TORCH_CHECK(false, "qkv_post: unsupported head_dim ", head_dim);
  }
}


// ---------------------------------------------------------------------------
// Per-token-group FP8 quantization (activation side of the block-scaled GEMM).
//
// Numerically identical to the reference ``per_token_group_quant_8bit_kernel``
// (same eps, same absmax -> scale -> UE8M0 rounding, same divide-then-clamp)
// but one warp owns one 128-wide group and streams it straight through
// registers: no shared-memory staging, no 16-thread groups, one 8-byte load
// and one 4-byte store per lane.
// ---------------------------------------------------------------------------

namespace {

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

template <bool UE8M0>
__global__ void quant_fp8_groups_kernel(const __nv_bfloat16 *__restrict__ x,
                                        __nv_fp8_e4m3 *__restrict__ out,
                                        float *__restrict__ scales,
                                        const int groups_per_row,
                                        const int num_rows,
                                        const int scale_row_stride,
                                        const int scale_col_stride) {
  // One warp per 128-wide group.  Warps in a block take *consecutive rows of
  // the same group column* so their scale stores (the layout DeepGEMM wants is
  // column-major) land in one contiguous 32-byte line instead of 8 lines a
  // whole row-stride apart.
  const int col = blockIdx.x;
  const int row = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= num_rows) return;
  const int lane = threadIdx.x & 31;

  const int64_t base =
      ((int64_t)row * groups_per_row + col) * 128 + lane * 4;
  const __nv_bfloat162 *src = reinterpret_cast<const __nv_bfloat162 *>(x + base);
  const __nv_bfloat162 a = src[0];
  const __nv_bfloat162 b = src[1];
  float f[4] = {__bfloat162float(a.x), __bfloat162float(a.y),
                __bfloat162float(b.x), __bfloat162float(b.y)};

  float amax = 1e-10f;  // reference seeds the reduction with eps
#pragma unroll
  for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(f[i]));
  amax = warp_reduce_max(amax);

  float y_s = amax / 448.0f;
  if (UE8M0) y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));

  if (lane == 0) {
    scales[(int64_t)row * scale_row_stride + (int64_t)col * scale_col_stride] =
        y_s;
  }

  __nv_fp8x4_e4m3 packed;
  float4 qf;
  qf.x = fminf(fmaxf(f[0] / y_s, -448.0f), 448.0f);
  qf.y = fminf(fmaxf(f[1] / y_s, -448.0f), 448.0f);
  qf.z = fminf(fmaxf(f[2] / y_s, -448.0f), 448.0f);
  qf.w = fminf(fmaxf(f[3] / y_s, -448.0f), 448.0f);
  packed = __nv_fp8x4_e4m3(qf);
  *reinterpret_cast<__nv_fp8x4_e4m3 *>(out + base) = packed;
}

} // namespace

void quant_fp8_groups(at::Tensor x, at::Tensor out, at::Tensor scales,
                      bool ue8m0) {
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.numel() % 128 == 0);
  const int groups_per_row = (int)(x.size(x.dim() - 1) / 128);
  const int num_rows = (int)(x.numel() / 128 / groups_per_row);
  const int threads = 256;
  const int warps = threads / 32;
  dim3 grid(groups_per_row, (num_rows + warps - 1) / warps);
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto *xp = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
  auto *op = reinterpret_cast<__nv_fp8_e4m3 *>(out.data_ptr());
  auto *sp = scales.data_ptr<float>();
  const int srs = (int)scales.stride(0);
  const int scs = (int)scales.stride(1);
  if (ue8m0) {
    quant_fp8_groups_kernel<true><<<grid, threads, 0, stream>>>(
        xp, op, sp, groups_per_row, num_rows, srs, scs);
  } else {
    quant_fp8_groups_kernel<false><<<grid, threads, 0, stream>>>(
        xp, op, sp, groups_per_row, num_rows, srs, scs);
  }
}

// ---------------------------------------------------------------------------
// Varlen causal attention for very short key spans (the decode shapes).
//
// For a batch whose whole job is one query row against a handful of keys,
// FlashAttention's Blackwell kernel costs ~10 us on the device and ~35 us of
// *host* time in its CuTeDSL launcher -- and at these sizes the layer is host
// bound, so that launch is the dominant term.  This kernel does the same math
// -- GQA, causal mask, left sliding window, attention sinks, fp32 accumulation,
// online softmax -- with one warp per (token, head) and no shared memory, for
// ~5 us of host time and ~3 us on the device.  It is dispatched only for small
// ``max_seqlen_k``; past roughly 32 keys per query the one-key-per-iteration
// reduction loses to FlashAttention's tensor cores, so longer spans keep the
// FlashAttention path.
// ---------------------------------------------------------------------------

namespace {

template <int HD>
__global__ void attn_short_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const __nv_bfloat16 *__restrict__ sinks,
    __nv_bfloat16 *__restrict__ out, const int32_t *__restrict__ cu_q,
    const int32_t *__restrict__ cu_k, const int num_seqs, const int nh,
    const int gqa, const float scale, const int window_left,
    const int q_row_stride, const int k_row_stride, const int v_row_stride,
    const int q_head_stride, const int k_head_stride, const int v_head_stride,
    const int out_row_stride) {
  constexpr int CH = HD / 32;
  const int token = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (head >= nh) return;

  // Locate the sequence this token belongs to (num_seqs is small).
  int s = 0;
  while (s + 1 < num_seqs && cu_q[s + 1] <= token) ++s;
  const int q_beg = cu_q[s];
  const int q_len = cu_q[s + 1] - q_beg;
  const int k_beg = cu_k[s];
  const int k_len = cu_k[s + 1] - k_beg;
  const int q_abs = (token - q_beg) + (k_len - q_len);

  int lo = 0;
  if (window_left >= 0) lo = max(0, q_abs - window_left);
  const int hi = min(q_abs, k_len - 1);

  const int kvh = head / gqa;
  const __nv_bfloat16 *qp = q + (size_t)token * q_row_stride +
                            (size_t)head * q_head_stride + lane * CH;
  float qv[CH];
  {
    const __nv_bfloat162 *src = reinterpret_cast<const __nv_bfloat162 *>(qp);
#pragma unroll
    for (int i = 0; i < CH / 2; ++i) {
      const __nv_bfloat162 t = src[i];
      qv[2 * i] = __bfloat162float(t.x);
      qv[2 * i + 1] = __bfloat162float(t.y);
    }
  }

  float m = -INFINITY, l = 0.f;
  if (sinks != nullptr) {
    m = __bfloat162float(sinks[head]);
    l = 1.f;
  }
  float acc[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) acc[i] = 0.f;

  for (int j = lo; j <= hi; ++j) {
    const __nv_bfloat162 *kp = reinterpret_cast<const __nv_bfloat162 *>(
        k + (size_t)(k_beg + j) * k_row_stride + (size_t)kvh * k_head_stride +
        lane * CH);
    const __nv_bfloat162 *vp = reinterpret_cast<const __nv_bfloat162 *>(
        v + (size_t)(k_beg + j) * v_row_stride + (size_t)kvh * v_head_stride +
        lane * CH);
    float dot = 0.f;
    float vv[CH];
#pragma unroll
    for (int i = 0; i < CH / 2; ++i) {
      const __nv_bfloat162 kt = kp[i];
      const __nv_bfloat162 vt = vp[i];
      dot = fmaf(qv[2 * i], __bfloat162float(kt.x), dot);
      dot = fmaf(qv[2 * i + 1], __bfloat162float(kt.y), dot);
      vv[2 * i] = __bfloat162float(vt.x);
      vv[2 * i + 1] = __bfloat162float(vt.y);
    }
    dot = warp_reduce_sum(dot) * scale;
    const float m_new = fmaxf(m, dot);
    const float corr = __expf(m - m_new);
    const float p = __expf(dot - m_new);
    l = l * corr + p;
    m = m_new;
#pragma unroll
    for (int i = 0; i < CH; ++i) acc[i] = fmaf(acc[i], corr, p * vv[i]);
  }

  const float inv = (l > 0.f) ? (1.f / l) : 0.f;
  __nv_bfloat162 *dst = reinterpret_cast<__nv_bfloat162 *>(
      out + (size_t)token * out_row_stride + (size_t)head * HD + lane * CH);
#pragma unroll
  for (int i = 0; i < CH / 2; ++i) {
    dst[i] = __nv_bfloat162(__float2bfloat16(acc[2 * i] * inv),
                            __float2bfloat16(acc[2 * i + 1] * inv));
  }
}

} // namespace

at::Tensor attn_short(at::Tensor q, at::Tensor k, at::Tensor v,
                      at::Tensor cu_q, at::Tensor cu_k,
                      std::optional<at::Tensor> sinks, double scale,
                      int64_t window_left) {
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3);
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1);
  TORCH_CHECK(q.scalar_type() == at::kBFloat16);
  const int N = (int)q.size(0), nh = (int)q.size(1), hd = (int)q.size(2);
  const int nkv = (int)k.size(1);
  TORCH_CHECK(nh % nkv == 0);
  auto out = at::empty({N, nh * hd}, q.options());
  if (N == 0) return out;
  const int threads = 128;
  const int warps = threads / 32;
  dim3 grid(N, (nh + warps - 1) / warps);
  const c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16 *sp =
      sinks.has_value()
          ? reinterpret_cast<const __nv_bfloat16 *>(sinks->data_ptr())
          : nullptr;
#define LAUNCH_ATTN_SHORT(HD)                                                  \
  attn_short_kernel<HD><<<grid, threads, 0, stream>>>(                         \
      reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr()),                   \
      reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr()),                   \
      reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()), sp,               \
      reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()),                       \
      cu_q.data_ptr<int32_t>(), cu_k.data_ptr<int32_t>(),                      \
      (int)cu_q.size(0) - 1, nh, nh / nkv, (float)scale, (int)window_left,     \
      (int)q.stride(0), (int)k.stride(0), (int)v.stride(0), (int)q.stride(1),  \
      (int)k.stride(1), (int)v.stride(1), nh *hd)
  if (hd == 128) {
    LAUNCH_ATTN_SHORT(128);
  } else if (hd == 64) {
    LAUNCH_ATTN_SHORT(64);
  } else if (hd == 256) {
    LAUNCH_ATTN_SHORT(256);
  } else {
    TORCH_CHECK(false, "attn_short: unsupported head_dim ", hd);
  }
#undef LAUNCH_ATTN_SHORT
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkv_post", &qkv_post, "fused qkv split + qk-norm + rope (in place)");
  m.def("quant_fp8_groups", &quant_fp8_groups, "per-token-group fp8 quant");
  m.def("attn_short", &attn_short, "varlen causal attention, short key spans");
}
