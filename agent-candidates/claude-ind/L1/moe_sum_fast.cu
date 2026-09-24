// Vectorized MoE top-k reduction (bf16/fp16/fp32) with fp32 accumulation.
//
// The op is purely memory bound: read topk*M*D elements, write M*D. The
// baseline moves 2 bytes per thread per load; here every thread moves 16-byte
// vectors (uint4), so a warp issues one 512-byte request per expert and each
// thread keeps TOPK*VPT loads in flight. One block covers a contiguous slice of
// a [topk, D] input slab (D=4096, topk=8 -> 64 KB) and consecutive blocks walk
// consecutive slabs, which keeps the DRAM stream sequential.
//
// Measured on a B200 (topk=8, D=4096, bf16): 172 us for the 16384-row case
// (1.21 GB moved, ~7.0 TB/s) against 252 us for the baseline's scalar loads.
// Launch shapes come from tune/sweep2.py; see pick_cfg.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <type_traits>

namespace {

constexpr int kMaxGridY = 65535;

// ---------------------------------------------------------------------------
// 16-byte vector traits: scalars per vector, fold-into-fp32, pack-back.
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<__nv_bfloat16> {
  static constexpr int N = 8;
  __device__ __forceinline__ static void accum(const uint4& v, float* a) {
    const unsigned int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // bf16 -> fp32 is just a 16-bit shift of the bit pattern.
      a[2 * i + 0] += __uint_as_float(w[i] << 16);
      a[2 * i + 1] += __uint_as_float(w[i] & 0xffff0000u);
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* a) {
    uint4 r;
    unsigned int* o = reinterpret_cast<unsigned int*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_bfloat162 p = __floats2bfloat162_rn(a[2 * i + 0], a[2 * i + 1]);
      o[i] = *reinterpret_cast<unsigned int*>(&p);
    }
    return r;
  }
};

template <>
struct VecTraits<__half> {
  static constexpr int N = 8;
  __device__ __forceinline__ static void accum(const uint4& v, float* a) {
    const unsigned int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
      a[2 * i + 0] += f.x;
      a[2 * i + 1] += f.y;
    }
  }
  __device__ __forceinline__ static uint4 pack(const float* a) {
    uint4 r;
    unsigned int* o = reinterpret_cast<unsigned int*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __half2 p = __floats2half2_rn(a[2 * i + 0], a[2 * i + 1]);
      o[i] = *reinterpret_cast<unsigned int*>(&p);
    }
    return r;
  }
};

template <>
struct VecTraits<float> {
  static constexpr int N = 4;
  __device__ __forceinline__ static void accum(const uint4& v, float* a) {
    const unsigned int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] += __uint_as_float(w[i]);
  }
  __device__ __forceinline__ static uint4 pack(const float* a) {
    uint4 r;
    float* o = reinterpret_cast<float*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = a[i];
    return r;
  }
};

// HINT 0: normal loads/stores. HINT 2: last-use ("streaming") hints on both,
// which is a measurable win once the working set no longer fits in L2.
__device__ __forceinline__ uint4 load_vec(const uint4* p, int hint) {
  return hint ? __ldcs(p) : __ldg(p);
}
__device__ __forceinline__ void store_vec(uint4* p, const uint4& v, int hint) {
  if (hint)
    __stcs(p, v);
  else
    *p = v;
}

// ---------------------------------------------------------------------------
// grid = (column chunks, rows in this launch); one output row per blockIdx.y.
//   vpr = D / VecTraits<T>::N   (16B vectors per row)
//   VPT = vectors per thread, strided by BLOCK so every load stays coalesced
// TOPK == 0 selects the runtime topk.
// ---------------------------------------------------------------------------
template <typename T, int TOPK, int VPT, int BLOCK, bool EXACT, int HINT>
__global__ __launch_bounds__(BLOCK) void moe_sum_vec_kernel(
    uint4* __restrict__ out, const uint4* __restrict__ in, int vpr, int topk_rt) {
  constexpr int NS = VecTraits<T>::N;
  const int topk = (TOPK > 0) ? TOPK : topk_rt;
  const long long row = blockIdx.y;
  const uint4* __restrict__ ip = in + row * (long long)topk * vpr;
  uint4* __restrict__ op = out + row * (long long)vpr;
  const int c0 = blockIdx.x * (BLOCK * VPT) + threadIdx.x;

  float acc[VPT][NS];
#pragma unroll
  for (int j = 0; j < VPT; ++j)
#pragma unroll
    for (int i = 0; i < NS; ++i) acc[j][i] = 0.f;

  if (EXACT) {
    for (int k = 0; k < topk; ++k) {
      const uint4* base = ip + (long long)k * vpr + c0;
      uint4 v[VPT];
#pragma unroll
      for (int j = 0; j < VPT; ++j) v[j] = load_vec(base + j * BLOCK, HINT);
#pragma unroll
      for (int j = 0; j < VPT; ++j) VecTraits<T>::accum(v[j], acc[j]);
    }
#pragma unroll
    for (int j = 0; j < VPT; ++j)
      store_vec(op + c0 + j * BLOCK, VecTraits<T>::pack(acc[j]), HINT);
  } else {
    bool ok[VPT];
#pragma unroll
    for (int j = 0; j < VPT; ++j) ok[j] = (c0 + j * BLOCK) < vpr;
    for (int k = 0; k < topk; ++k) {
      const uint4* base = ip + (long long)k * vpr + c0;
      uint4 v[VPT];
#pragma unroll
      for (int j = 0; j < VPT; ++j)
        if (ok[j]) v[j] = load_vec(base + j * BLOCK, HINT);
#pragma unroll
      for (int j = 0; j < VPT; ++j)
        if (ok[j]) VecTraits<T>::accum(v[j], acc[j]);
    }
#pragma unroll
    for (int j = 0; j < VPT; ++j)
      if (ok[j]) store_vec(op + c0 + j * BLOCK, VecTraits<T>::pack(acc[j]), HINT);
  }
}

// Scalar fallback: D not a multiple of the vector width, or a misaligned base.
template <typename T, int TOPK>
__global__ void moe_sum_scalar_kernel(T* __restrict__ out, const T* __restrict__ in,
                                      int d, int topk_rt) {
  const int topk = (TOPK > 0) ? TOPK : topk_rt;
  const long long row = blockIdx.y;
  const long long ib = row * (long long)topk * d;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < d;
       idx += blockDim.x * gridDim.x) {
    float x = 0.f;
    for (int k = 0; k < topk; ++k)
      x += static_cast<float>(in[ib + (long long)k * d + idx]);
    out[row * (long long)d + idx] = static_cast<T>(x);
  }
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------
struct Cfg {
  int block;
  int vpt;
  int hint;
};

// Picked from the B200 sweep (tune/sweep2.py), keyed on the number of 16-byte
// output vectors: tiny problems want one small block per row so the work
// spreads over SMs, mid-size wants half-row blocks, and above L2 the streaming
// cache hints pay off (515 us vs 538 us on the 16384-row case).
inline Cfg pick_cfg(long long total_vec) {
  if (total_vec <= 1024) return {64, 1, 0};
  if (total_vec <= 2 * 1024 * 1024) return {128, 2, 0};
  return {256, 2, 2};
}

template <typename T, int TOPK, int VPT, int BLOCK, int HINT>
void launch_one(void* out, const void* in, long long M, int vpr, int topk,
                int chunks, bool exact, cudaStream_t stream) {
  uint4* o = reinterpret_cast<uint4*>(out);
  const uint4* i = reinterpret_cast<const uint4*>(in);
  // gridDim.y is capped at 65535; split the row range if needed.
  for (long long off = 0; off < M; off += kMaxGridY) {
    const int rows = (int)std::min<long long>(kMaxGridY, M - off);
    dim3 grid((unsigned)chunks, (unsigned)rows);
    const uint4* ip = i + off * (long long)topk * vpr;
    uint4* op = o + off * (long long)vpr;
    if (exact)
      moe_sum_vec_kernel<T, TOPK, VPT, BLOCK, true, HINT>
          <<<grid, BLOCK, 0, stream>>>(op, ip, vpr, topk);
    else
      moe_sum_vec_kernel<T, TOPK, VPT, BLOCK, false, HINT>
          <<<grid, BLOCK, 0, stream>>>(op, ip, vpr, topk);
  }
}

// The captured workload is bf16 with topk=8; that pair gets the tuned grid of
// launch shapes. Everything else runs one fixed shape (256 threads, one vector
// per thread) -- same kernel, no extra instantiations.
#define FK_HOT_HINT(VPT, BLOCK)                                                          \
  (cfg.hint ? launch_one<T, 8, VPT, BLOCK, 2>(out, in, M, vpr, 8, chunks, exact, stream) \
            : launch_one<T, 8, VPT, BLOCK, 0>(out, in, M, vpr, 8, chunks, exact, stream))

#define FK_HOT_VPT(BLOCK)              \
  if (cfg.vpt >= 2) {                  \
    FK_HOT_HINT(2, BLOCK);             \
  } else {                             \
    FK_HOT_HINT(1, BLOCK);             \
  }

template <typename T>
void launch_typed(void* out, const void* in, long long M, int d, int topk, Cfg cfg,
                  cudaStream_t stream) {
  constexpr int NS = VecTraits<T>::N;
  const bool aligned =
      ((reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(in)) & 15u) == 0;
  if (d % NS != 0 || !aligned) {
    for (long long off = 0; off < M; off += kMaxGridY) {
      const int rows = (int)std::min<long long>(kMaxGridY, M - off);
      dim3 grid((unsigned)((d + 255) / 256), (unsigned)rows);
      const T* ip = reinterpret_cast<const T*>(in) + off * (long long)topk * d;
      T* op = reinterpret_cast<T*>(out) + off * (long long)d;
      if (topk == 8)
        moe_sum_scalar_kernel<T, 8><<<grid, 256, 0, stream>>>(op, ip, d, topk);
      else
        moe_sum_scalar_kernel<T, 0><<<grid, 256, 0, stream>>>(op, ip, d, topk);
    }
    return;
  }

  const int vpr = d / NS;
  if (cfg.block <= 0) cfg = pick_cfg(M * (long long)vpr);
  // Never give a block more columns than a row has.
  while (cfg.block * cfg.vpt > vpr && cfg.vpt > 1) cfg.vpt /= 2;
  while (cfg.block * cfg.vpt > vpr && cfg.block > 32) cfg.block /= 2;
  const int per_block = cfg.block * cfg.vpt;
  const int chunks = (vpr + per_block - 1) / per_block;
  const bool exact = (chunks * per_block == vpr);

  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (topk == 8) {
      switch (cfg.block) {
        case 32: FK_HOT_VPT(32) break;
        case 64: FK_HOT_VPT(64) break;
        case 128: FK_HOT_VPT(128) break;
        case 512: FK_HOT_VPT(512) break;
        default: FK_HOT_VPT(256) break;
      }
      return;
    }
  }

  const int gchunks = (vpr + 255) / 256;
  const bool gexact = (gchunks * 256 == vpr);
  switch (topk) {
    case 2:
      launch_one<T, 2, 1, 256, 0>(out, in, M, vpr, topk, gchunks, gexact, stream);
      break;
    case 3:
      launch_one<T, 3, 1, 256, 0>(out, in, M, vpr, topk, gchunks, gexact, stream);
      break;
    case 4:
      launch_one<T, 4, 1, 256, 0>(out, in, M, vpr, topk, gchunks, gexact, stream);
      break;
    case 8:
      launch_one<T, 8, 1, 256, 0>(out, in, M, vpr, topk, gchunks, gexact, stream);
      break;
    default:
      launch_one<T, 0, 1, 256, 0>(out, in, M, vpr, topk, gchunks, gexact, stream);
      break;
  }
}

inline void launch_any(void* out, const void* in, at::ScalarType dt, long long M, int d,
                       int topk, Cfg cfg, cudaStream_t stream) {
  switch (dt) {
    case at::ScalarType::BFloat16:
      launch_typed<__nv_bfloat16>(out, in, M, d, topk, cfg, stream);
      break;
    case at::ScalarType::Half:
      launch_typed<__half>(out, in, M, d, topk, cfg, stream);
      break;
    case at::ScalarType::Float:
      launch_typed<float>(out, in, M, d, topk, cfg, stream);
      break;
    default:
      TORCH_CHECK(false, "moe_sum_fast: unsupported dtype ", dt);
  }
}

}  // namespace

// input: [M*topk, D] contiguous; output: [M, D] contiguous.
void moe_sum(torch::Tensor input, torch::Tensor output, int64_t topk) {
  const int d = (int)output.size(-1);
  const long long M = output.numel() / d;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  launch_any(output.data_ptr(), input.data_ptr(), input.scalar_type(), M, d, (int)topk,
             Cfg{-1, -1, 0}, at::cuda::getCurrentCUDAStream());
}

// Same op with the launch shape forced, for tune/*.py.
void moe_sum_tuned(torch::Tensor input, torch::Tensor output, int64_t topk, int64_t block,
                   int64_t vpt, int64_t hint) {
  const int d = (int)output.size(-1);
  const long long M = output.numel() / d;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  launch_any(output.data_ptr(), input.data_ptr(), input.scalar_type(), M, d, (int)topk,
             Cfg{(int)block, (int)vpt, (int)hint}, at::cuda::getCurrentCUDAStream());
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sum", &moe_sum, "MoE sum reduction (vectorized CUDA)");
  m.def("moe_sum_tuned", &moe_sum_tuned, "MoE sum reduction with a forced launch shape");
}
