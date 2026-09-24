// Fast per-token-group (group=128) FP8 E4M3 quantization for Blackwell.
//
// One 128-element group is handled by 16 lanes, 8 bf16 each: a single 16 B
// vector load in, a single 8 B vector store out, the group absmax reduced with
// four ``__shfl_xor_sync`` steps.  No shared memory and no ``__syncthreads``
// (the baseline stages the group through smem and pays a block barrier), so the
// kernel runs at copy bandwidth.  ILP groups per lane are loaded before any is
// reduced, which keeps enough loads in flight to saturate HBM.
//
// Two scale-output modes share the kernel:
//   kSmodeF32    -- float32 power-of-two scale per group, flat in group index
//                   (the vLLM layout ``PerTokenGroupQuantFp8`` must produce).
//   kSmodeE8m0T  -- the UE8M0 exponent byte replicated 4x -- the MXFP8
//                   scale-factor layout with 32-element granularity that
//                   ``tl.dot_scaled`` / ``tcgen05.mma.block_scale`` consumes --
//                   written K-major as int32[K/128, M].  That scatters the
//                   scale store (4 B into a 32 B sector, ~8% of the kernel's
//                   traffic) but makes the GEMM's per-(K-block, M-tile) scale
//                   load one contiguous vector, which is the better trade.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {

constexpr int kSmodeF32 = 0;
constexpr int kSmodeE8m0T = 1;

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short h) {
  return __uint_as_float(static_cast<unsigned int>(h) << 16);
}

__device__ __forceinline__ float half_bits_to_f32(unsigned short h) {
  return __half2float(__ushort_as_half(h));
}

template <bool IS_BF16>
__device__ __forceinline__ float elem_to_f32(unsigned short h) {
  return IS_BF16 ? bf16_bits_to_f32(h) : half_bits_to_f32(h);
}

// absmax -> UE8M0 exponent field (bias 127), matching deep_gemm's
// ``ceil_to_ue8m0``: round the fp32 scale up to the next power of two.
__device__ __forceinline__ int ue8m0_exp(float absmax) {
  float s = fmaxf(absmax, 1e-10f) * (1.0f / 448.0f);
  unsigned int b = __float_as_uint(s);
  int e = static_cast<int>((b >> 23) & 0xFFu) + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
  return e < 1 ? 1 : (e > 254 ? 254 : e);
}

template <int ILP, int SMODE, bool IS_BF16>
__global__ __launch_bounds__(256) void quant_group128_kernel(
    const uint4* __restrict__ xv, uint2* __restrict__ qv,
    void* __restrict__ sbuf, long long ngroups, int groups_per_row,
    int num_rows) {
  const int lane = threadIdx.x & 15;
  const int sg = threadIdx.x >> 4;
  const long long base = static_cast<long long>(blockIdx.x) * (16 * ILP) + sg;

  uint4 v[ILP];
  long long g[ILP];
  bool ok[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) {
    g[i] = base + static_cast<long long>(i) * 16;
    ok[i] = g[i] < ngroups;
    if (ok[i]) v[i] = xv[g[i] * 16 + lane];
  }

#pragma unroll
  for (int i = 0; i < ILP; ++i) {
    if (!ok[i]) continue;
    const unsigned short* hp = reinterpret_cast<const unsigned short*>(&v[i]);
    float f[8];
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      f[j] = elem_to_f32<IS_BF16>(hp[j]);
      m = fmaxf(m, fabsf(f[j]));
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));

    const int e = ue8m0_exp(m);
    const float sc = __uint_as_float(static_cast<unsigned int>(e) << 23);
    const float inv = 1.0f / sc;

    uint2 out;
    unsigned short* op = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 t = make_float2(f[2 * j] * inv, f[2 * j + 1] * inv);
      op[j] = __nv_cvt_float2_to_fp8x2(t, __NV_SATFINITE, __NV_E4M3);
    }
    qv[g[i] * 16 + lane] = out;

    if (lane == 0) {
      if (SMODE == kSmodeF32) {
        static_cast<float*>(sbuf)[g[i]] = sc;
      } else {
        // K-major scale: sbuf[kb * num_rows + row], so the GEMM reads one
        // contiguous uint32 vector per (k-block, M-tile).
        const long long row = g[i] / groups_per_row;
        const long long kb = g[i] - row * groups_per_row;
        static_cast<unsigned int*>(sbuf)[kb * num_rows + row] =
            static_cast<unsigned int>(e) * 0x01010101u;
      }
    }
  }
}

template <int SMODE>
void launch_quant(const torch::Tensor& x, torch::Tensor& q, void* sbuf,
                  int groups_per_row = 0, int num_rows = 0) {
  TORCH_CHECK(x.is_contiguous() && q.is_contiguous());
  TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn, "output must be e4m3");
  TORCH_CHECK(x.numel() % 128 == 0, "numel must be a multiple of 128");
  const long long ngroups = x.numel() / 128;
  const bool is_bf16 = x.scalar_type() == at::kBFloat16;
  TORCH_CHECK(is_bf16 || x.scalar_type() == at::kHalf,
              "only bfloat16/float16 input is supported");

  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* xv = static_cast<const uint4*>(x.const_data_ptr());
  auto* qv = static_cast<uint2*>(q.data_ptr());

  // Pick ILP so the grid still fills the device; more groups per lane keeps
  // more loads in flight (the kernel is pure HBM traffic).
  auto blocks = [&](int ilp) { return (ngroups + 16LL * ilp - 1) / (16LL * ilp); };
  int ilp = 8;
  if (blocks(8) < 1024) ilp = 4;
  if (blocks(4) < 1024) ilp = 2;
  if (blocks(2) < 512) ilp = 1;

#define FK_LAUNCH(ILP_V)                                                     \
  do {                                                                       \
    if (is_bf16)                                                             \
      quant_group128_kernel<ILP_V, SMODE, true>                              \
          <<<blocks(ILP_V), 256, 0, stream>>>(xv, qv, sbuf, ngroups,         \
                                             groups_per_row, num_rows);      \
    else                                                                     \
      quant_group128_kernel<ILP_V, SMODE, false>                             \
          <<<blocks(ILP_V), 256, 0, stream>>>(xv, qv, sbuf, ngroups,         \
                                             groups_per_row, num_rows);      \
  } while (0)

  switch (ilp) {
    case 1: FK_LAUNCH(1); break;
    case 2: FK_LAUNCH(2); break;
    case 4: FK_LAUNCH(4); break;
    default: FK_LAUNCH(8); break;
  }
#undef FK_LAUNCH
}

}  // namespace

// out_scale: float32, flat group-major (row-major [M, K/128]).
void per_token_group_quant_fp8_f32(const torch::Tensor& x, torch::Tensor& q,
                                   torch::Tensor& s) {
  TORCH_CHECK(s.is_contiguous() && s.scalar_type() == at::kFloat);
  TORCH_CHECK(s.numel() * 128 == x.numel());
  launch_quant<kSmodeF32>(x, q, s.data_ptr());
}

// out_scale: int32 [K/128, M], the UE8M0 exponent byte replicated 4x (K-major).
void per_token_group_quant_fp8_e8m0_t(const torch::Tensor& x, torch::Tensor& q,
                                      torch::Tensor& s) {
  TORCH_CHECK(s.is_contiguous() && s.element_size() == 4);
  TORCH_CHECK(x.dim() == 2 && s.dim() == 2);
  const int M = static_cast<int>(x.size(0));
  const int gpr = static_cast<int>(x.size(1)) / 128;
  TORCH_CHECK(s.size(0) == gpr && s.size(1) == M);
  launch_quant<kSmodeE8m0T>(x, q, s.data_ptr(), gpr, M);
}


// ---------------------------------------------------------------------------
// Fused small-M block-scaled FP8 linear (decode / M <= 16).
//
// At M=1 the MMA path is pure waste: the problem is weight-bandwidth bound
// (N*K FP8 bytes) and a 128-row tile throws away 128x of the tensor core.  This
// kernel instead quantizes the whole activation into shared memory once per
// block and then has one warp per weight row stream that row through FMAs, so
// the whole linear is a single launch at copy bandwidth.
//
//   out[m, n] = sum_kb sa[m, kb] * sb[n, kb] * sum_{k in kb} a[m, k] * b[n, k]
//
// ``wsi`` is the captured DeepGEMM packed UE8M0 layout, int32[N, K/512] with
// stride (1, N): one uniform (broadcast) load per warp covers the 4 K-blocks of
// a 512-byte step, and lane>>3 selects the byte.
// ---------------------------------------------------------------------------

namespace {

__device__ __forceinline__ void fp8x2_to_float2(unsigned short packed,
                                                float& lo, float& hi) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
  float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
  lo = f.x;
  hi = f.y;
}

template <int MMAX, int WARPS, bool IS_BF16>
__global__ __launch_bounds__(WARPS * 32) void small_m_linear_kernel(
    const void* __restrict__ x, const unsigned char* __restrict__ wq,
    const int* __restrict__ wsi, long s_wn, long s_wk,
    __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  extern __shared__ __align__(16) char smem[];
  unsigned char* aq = reinterpret_cast<unsigned char*>(smem);   // [M, K] fp8
  const int gpr = K >> 7;
  float* sa = reinterpret_cast<float*>(smem + (size_t)M * K);   // [M, gpr]

  // A warp walks its weight row 512 B at a time; issuing UNROLL of those loads
  // before consuming any is what keeps enough bytes in flight to saturate HBM.
  // They are issued *before* the activation quantization below because they do
  // not depend on it -- that hides the quantization phase (and its barrier)
  // behind the weight fetch instead of serializing after it.
  constexpr int UNROLL = 4;
  const int warp = threadIdx.x >> 5;
  const int lane32 = threadIdx.x & 31;
  const int row = blockIdx.x * WARPS + warp;
  const bool row_ok = row < N;
  const unsigned char* brow = wq + (size_t)row * K;
  uint4 bv[UNROLL];
  int word[UNROLL];
  bool act[UNROLL];
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    const int k0 = u * 512;
    act[u] = row_ok && k0 < K;
    if (act[u]) {
      bv[u] = *reinterpret_cast<const uint4*>(brow + k0 + lane32 * 16);
      word[u] = wsi[row * s_wn + (k0 >> 9) * s_wk];
    }
  }

  // ---- phase 1: quantize the activation into shared memory -----------------
  {
    const int lane = threadIdx.x & 15;
    const int sg = threadIdx.x >> 4;
    const int ngroups = M * gpr;
    const uint4* xv = static_cast<const uint4*>(x);
    for (int g = sg; g < ngroups; g += (WARPS * 32) / 16) {
      uint4 v = xv[(size_t)g * 16 + lane];
      const unsigned short* hp = reinterpret_cast<const unsigned short*>(&v);
      float f[8];
      float m = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        f[j] = elem_to_f32<IS_BF16>(hp[j]);
        m = fmaxf(m, fabsf(f[j]));
      }
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
      const int e = ue8m0_exp(m);
      const float sc = __uint_as_float(static_cast<unsigned int>(e) << 23);
      const float inv = 1.0f / sc;
      uint2 o;
      unsigned short* op = reinterpret_cast<unsigned short*>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 t = make_float2(f[2 * j] * inv, f[2 * j + 1] * inv);
        op[j] = __nv_cvt_float2_to_fp8x2(t, __NV_SATFINITE, __NV_E4M3);
      }
      *reinterpret_cast<uint2*>(aq + (size_t)g * 128 + lane * 8) = o;
      if (lane == 0) sa[g] = sc;
    }
  }
  __syncthreads();

  // ---- phase 2: one warp per weight row ------------------------------------
  const int lane = lane32;
  float acc[MMAX];
#pragma unroll
  for (int m = 0; m < MMAX; ++m) acc[m] = 0.0f;

  const int kb_lane = lane >> 3;          // which of the 4 K-blocks in a step
  const int byte_sel = (lane >> 3) * 8;   // UE8M0 byte within the packed word
  for (int kk = 0; kk < K; kk += 512 * UNROLL) {
    if (kk != 0) {
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int k0 = kk + u * 512;
        act[u] = row_ok && k0 < K;
        if (act[u]) {
          bv[u] = *reinterpret_cast<const uint4*>(brow + k0 + lane * 16);
          word[u] = wsi[row * s_wn + (k0 >> 9) * s_wk];
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      if (!act[u]) continue;
      const int k0 = kk + u * 512;
      const unsigned short* bp = reinterpret_cast<const unsigned short*>(&bv[u]);
      float bf[16];
#pragma unroll
      for (int j = 0; j < 8; ++j) fp8x2_to_float2(bp[j], bf[2 * j], bf[2 * j + 1]);
      const float sb = __uint_as_float(
          static_cast<unsigned int>((word[u] >> byte_sel) & 0xFF) << 23);
      const int kb = (k0 >> 7) + kb_lane;
#pragma unroll
      for (int m = 0; m < MMAX; ++m) {
        if (m >= M) break;
        const uint4 av =
            *reinterpret_cast<const uint4*>(aq + (size_t)m * K + k0 + lane * 16);
        const unsigned short* ap = reinterpret_cast<const unsigned short*>(&av);
        float p = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float a0, a1;
          fp8x2_to_float2(ap[j], a0, a1);
          p = fmaf(a0, bf[2 * j], p);
          p = fmaf(a1, bf[2 * j + 1], p);
        }
        acc[m] = fmaf(p, sb * sa[m * gpr + kb], acc[m]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MMAX; ++m) {
    if (m >= M) break;
    float v = acc[m];
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (lane == 0 && row_ok) out[(size_t)m * N + row] = __float2bfloat16(v);
  }
}

}  // namespace

// Fused small-M path: BF16 activation in, BF16 out, no intermediate buffers.
torch::Tensor small_m_linear(const torch::Tensor& x, const torch::Tensor& wq,
                             const torch::Tensor& wsi) {
  TORCH_CHECK(x.is_contiguous() && wq.is_contiguous());
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(wq.size(0));
  TORCH_CHECK(wq.size(1) == K && K % 512 == 0);
  TORCH_CHECK(M <= 4);
  const bool is_bf16 = x.scalar_type() == at::kBFloat16;

  auto out = torch::empty({M, N}, x.options().dtype(at::kBFloat16));
  const size_t smem = (size_t)M * K + (size_t)M * (K / 128) * sizeof(float);
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kWarps = 8;
  const int blocks = (N + kWarps - 1) / kWarps;

#define FK_SMALL_M(MMAX_V)                                                   \
  do {                                                                       \
    if (is_bf16) {                                                           \
      auto k = small_m_linear_kernel<MMAX_V, kWarps, true>;                  \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                           (int)smem);                                       \
      k<<<blocks, kWarps * 32, smem, stream>>>(                              \
          x.const_data_ptr(), static_cast<const unsigned char*>(             \
              wq.const_data_ptr()),                                          \
          wsi.const_data_ptr<int>(), wsi.stride(0), wsi.stride(1),         \
          static_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K);             \
    } else {                                                                 \
      auto k = small_m_linear_kernel<MMAX_V, kWarps, false>;                 \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                           (int)smem);                                       \
      k<<<blocks, kWarps * 32, smem, stream>>>(                              \
          x.const_data_ptr(), static_cast<const unsigned char*>(             \
              wq.const_data_ptr()),                                          \
          wsi.const_data_ptr<int>(), wsi.stride(0), wsi.stride(1),         \
          static_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K);             \
    }                                                                        \
  } while (0)

  if (M == 1) {
    FK_SMALL_M(1);
  } else if (M <= 2) {
    FK_SMALL_M(2);
  } else {
    FK_SMALL_M(4);
  }
#undef FK_SMALL_M
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_e8m0_t", &per_token_group_quant_fp8_e8m0_t,
        "per-token-group FP8 quant, UE8M0 scale bytes, K-major");
  m.def("small_m_linear", &small_m_linear,
        "fused small-M block-scaled FP8 linear");
  m.def("quant_f32", &per_token_group_quant_fp8_f32,
        "per-token-group FP8 quant, float32 power-of-two scales");
}
