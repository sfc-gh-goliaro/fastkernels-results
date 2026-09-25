#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

struct alignas(32) Bf16x16 {
  uint32_t u[8];
};

struct alignas(16) Bf16x8 {
  uint32_t u[4];
};

__device__ __forceinline__ Bf16x16 load256(const Bf16x16* ptr) {
  Bf16x16 value;
  asm volatile(
      "ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(value.u[0]), "=r"(value.u[1]), "=r"(value.u[2]),
        "=r"(value.u[3]), "=r"(value.u[4]), "=r"(value.u[5]),
        "=r"(value.u[6]), "=r"(value.u[7])
      : "l"(ptr));
  return value;
}

__device__ __forceinline__ void store256(Bf16x16* ptr, const Bf16x16& value) {
  asm volatile(
      "st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
      :
      : "l"(ptr), "r"(value.u[0]), "r"(value.u[1]), "r"(value.u[2]),
        "r"(value.u[3]), "r"(value.u[4]), "r"(value.u[5]),
        "r"(value.u[6]), "r"(value.u[7])
      : "memory");
}

__device__ __forceinline__ Bf16x8 load128(const Bf16x8* ptr) {
  Bf16x8 value;
  *reinterpret_cast<uint4*>(value.u) =
      __ldg(reinterpret_cast<const uint4*>(ptr));
  return value;
}

__device__ __forceinline__ void store128(Bf16x8* ptr,
                                         const Bf16x8& value) {
  *reinterpret_cast<uint4*>(ptr) =
      *reinterpret_cast<const uint4*>(value.u);
}

__device__ __forceinline__ Bf16x16 activate_and_multiply(
    Bf16x16 gate, const Bf16x16& up) {
  auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(gate.u);
  const auto* up_pairs = reinterpret_cast<const __nv_bfloat162*>(up.u);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float2 g = __bfloat1622float2(gate_pairs[i]);
    const float2 u = __bfloat1622float2(up_pairs[i]);
    g.x *= __fdividef(1.0f, 1.0f + __expf(-g.x));
    g.y *= __fdividef(1.0f, 1.0f + __expf(-g.y));
    g.x *= u.x;
    g.y *= u.y;
    gate_pairs[i] = __floats2bfloat162_rn(g.x, g.y);
  }
  return gate;
}

__device__ __forceinline__ Bf16x8 activate_and_multiply128(
    Bf16x8 gate, const Bf16x8& up) {
  auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(gate.u);
  const auto* up_pairs = reinterpret_cast<const __nv_bfloat162*>(up.u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 g = __bfloat1622float2(gate_pairs[i]);
    const float2 u = __bfloat1622float2(up_pairs[i]);
    g.x *= __fdividef(1.0f, 1.0f + __expf(-g.x));
    g.y *= __fdividef(1.0f, 1.0f + __expf(-g.y));
    g.x *= u.x;
    g.y *= u.y;
    gate_pairs[i] = __floats2bfloat162_rn(g.x, g.y);
  }
  return gate;
}

__global__ void narrow_row_kernel(__nv_bfloat16* out,
                                  const __nv_bfloat16* input, int rows,
                                  int d) {
  const int row = blockIdx.x;
  const int i = threadIdx.x;
  const int vectors = d / 8;
  if (row >= rows || i >= vectors) {
    return;
  }
  const auto* gate = reinterpret_cast<const Bf16x8*>(
      input + static_cast<int64_t>(row) * 2 * d);
  const auto* up = reinterpret_cast<const Bf16x8*>(
      input + static_cast<int64_t>(row) * 2 * d + d);
  auto* dst =
      reinterpret_cast<Bf16x8*>(out + static_cast<int64_t>(row) * d);
  Bf16x8 g = load128(gate + i);
  const Bf16x8 u = load128(up + i);
  store128(dst + i, activate_and_multiply128(g, u));
}

template <int BlockSize>
__global__ void row_kernel(__nv_bfloat16* out, const __nv_bfloat16* input,
                           int rows, int d) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const auto* gate = reinterpret_cast<const Bf16x16*>(
      input + static_cast<int64_t>(row) * 2 * d);
  const auto* up = reinterpret_cast<const Bf16x16*>(
      input + static_cast<int64_t>(row) * 2 * d + d);
  auto* dst =
      reinterpret_cast<Bf16x16*>(out + static_cast<int64_t>(row) * d);
  const int vectors = d / 16;
  for (int i = threadIdx.x; i < vectors; i += BlockSize) {
    Bf16x16 g = load256(gate + i);
    const Bf16x16 u = load256(up + i);
    store256(dst + i, activate_and_multiply(g, u));
  }
}

template <int RowsPerBlock>
__global__ void grouped_rows_kernel(__nv_bfloat16* out,
                                    const __nv_bfloat16* input, int rows,
                                    int d) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * RowsPerBlock + warp;
  if (row >= rows) {
    return;
  }
  const auto* gate = reinterpret_cast<const Bf16x16*>(
      input + static_cast<int64_t>(row) * 2 * d);
  const auto* up = reinterpret_cast<const Bf16x16*>(
      input + static_cast<int64_t>(row) * 2 * d + d);
  auto* dst =
      reinterpret_cast<Bf16x16*>(out + static_cast<int64_t>(row) * d);
  const int vectors = d / 16;
  for (int i = lane; i < vectors; i += 32) {
    Bf16x16 g = load256(gate + i);
    const Bf16x16 u = load256(up + i);
    store256(dst + i, activate_and_multiply(g, u));
  }
}

template <int BlockSize>
__global__ void tiled_rows_kernel(__nv_bfloat16* out,
                                  const __nv_bfloat16* input, int rows,
                                  int d) {
  const int row = blockIdx.y;
  const int i = blockIdx.x * BlockSize + threadIdx.x;
  const int vectors = d / 16;
  if (row >= rows || i >= vectors) {
    return;
  }
  const auto* gate = reinterpret_cast<const Bf16x16*>(
      input + static_cast<int64_t>(row) * 2 * d);
  const auto* up = reinterpret_cast<const Bf16x16*>(
      input + static_cast<int64_t>(row) * 2 * d + d);
  auto* dst =
      reinterpret_cast<Bf16x16*>(out + static_cast<int64_t>(row) * d);
  Bf16x16 g = load256(gate + i);
  const Bf16x16 u = load256(up + i);
  store256(dst + i, activate_and_multiply(g, u));
}

__global__ void scalar_kernel(__nv_bfloat16* out,
                              const __nv_bfloat16* input, int64_t count,
                              int d) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < count;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = index / d;
    const int col = index - row * d;
    const float gate =
        __bfloat162float(input[row * static_cast<int64_t>(2 * d) + col]);
    const float up =
        __bfloat162float(input[row * static_cast<int64_t>(2 * d) + d + col]);
    const float silu = gate / (1.0f + __expf(-gate));
    out[index] = __float2bfloat16_rn(silu * up);
  }
}

void launch(__nv_bfloat16* out, const __nv_bfloat16* input, int rows, int d,
            cudaStream_t stream) {
  if (d % 16 != 0) {
    const int64_t count = static_cast<int64_t>(rows) * d;
    const int blocks =
        static_cast<int>(std::min<int64_t>((count + 255) / 256, 4096));
    scalar_kernel<<<blocks, 256, 0, stream>>>(out, input, count, d);
    return;
  }

  if (d <= 512 && rows <= 8) {
    narrow_row_kernel<<<rows, std::min(d / 8, 1024), 0, stream>>>(
        out, input, rows, d);
  } else if (d <= 512) {
    grouped_rows_kernel<4><<<(rows + 3) / 4, 128, 0, stream>>>(
        out, input, rows, d);
  } else if (d >= 8192 && rows > 128) {
    tiled_rows_kernel<448>
        <<<dim3((d / 16 + 447) / 448, rows), 448, 0, stream>>>(
            out, input, rows, d);
  } else if (d >= 8192) {
    tiled_rows_kernel<256>
        <<<dim3((d / 16 + 255) / 256, rows), 256, 0, stream>>>(
            out, input, rows, d);
  } else if (d >= 2048) {
    row_kernel<512><<<rows, 512, 0, stream>>>(out, input, rows, d);
  } else {
    row_kernel<256><<<rows, 256, 0, stream>>>(out, input, rows, d);
  }
}

torch::Tensor silu_and_mul(torch::Tensor input) {
  const int d = static_cast<int>(input.size(-1) / 2);
  const int rows = static_cast<int>(input.numel() / input.size(-1));
  std::vector<int64_t> sizes(input.sizes().begin(), input.sizes().end());
  sizes.back() = d;
  auto out = torch::empty(sizes, input.options());
  if (rows == 0 || d == 0) {
    return out;
  }

  const c10::cuda::CUDAGuard guard(input.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  launch(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
         reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),
         rows, d, stream);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("silu_and_mul", &silu_and_mul);
}
