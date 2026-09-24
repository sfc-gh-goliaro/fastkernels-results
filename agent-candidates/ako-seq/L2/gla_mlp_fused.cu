// Weight-streaming SwiGLU MLP for the tiny-M decode path.
//
//   h[m, j] = silu(x[m, :] . Wg[j, :]) * (x[m, :] . Wu[j, :])      j < I
//   y[m, n] = h[m, :] . Wd[n, :]                                   n < H
//
// Why this exists at all -- cuBLAS is *not* slow here in FLOP terms, it is slow
// in blocks.  With H=2560, I=6912, bf16 and M <= 4 rows the whole op is three
// GEMVs over 106 MB of weights, and the only thing that matters is how close to
// the streaming ceiling those 106 MB can be read.  Measured on a B200 inside the
// benchmark's own timing loop (which leaves L2 full of dirty lines from its
// 253 MiB `l2.zero_()`, so ~3.9 TB/s is the ceiling, not the 5.4 TB/s this part
// reaches on a 2 GB working set):
//
//   * a kernel that only *reads* 106 MB: 2.54 TB/s at 148 blocks, 3.25 at 296,
//     3.50 at 592+.  Block count, not bandwidth, is the knob.
//   * cuBLAS gate+up (N=13824) reaches 3.16 TB/s -- it is fine.
//   * cuBLAS down (N=2560, K=6912) reaches only 2.5 TB/s at M=1: the output is
//     M x 2560, so a 128x128 tiling yields ~20 blocks on 148 SMs.
//
// So both kernels below are shaped to put ~600-900 blocks on the machine while
// reading every weight byte exactly once, which no tiling of these GEMMs can do:
//
//   `gu_sam`  partitions the *output* dim I.  Block b owns BI values of j and
//             reads Wg[j0:j0+BI, :] and Wu[j0:j0+BI, :] -- contiguous rows, read
//             once, I/BI blocks (864 at BI=8).  It also fuses the SiLU-and-mul,
//             so h is produced directly and the gate/up intermediate never
//             reaches memory at all.
//   `down_gv` partitions the *output* dim H.  Block b owns BN values of n and
//             reads Wd[n0:n0+BN, :] -- again contiguous rows, read once, H/BN
//             blocks (320-640).  No split-K, so no atomics and no fp32 scratch.
//
// The vector operand (x for `gu_sam`, h for `down_gv`) is staged in shared
// memory and re-read once per weight row.  That is exactly one shared-memory
// byte per weight byte per row of M, i.e. 70.8 MB and 35.4 MB of shared traffic
// at M=1 -- 2.5 us at the 28 TB/s aggregate, against 20 us of HBM time.  Holding
// it in registers instead would need 108 registers per lane for h and buys
// nothing.
//
// Numerics reproduce the reference chain rather than improving on it: the
// reference's gate/up come out of a bf16 cuBLAS GEMM, so the fp32 accumulators
// are rounded to bf16 *before* the activation, SiLU is evaluated in fp32 via the
// same one-MUFU `tanh.approx.f32` identity the L1 `silu`/`silu_and_mul` winners
// use, and the gate*up product is a bf16 multiply.  Down accumulates in fp32,
// like cuBLAS.
//
// The two launches are chained with programmatic dependent launch, matching the
// L1 winners: `down_gv` sets up its grid and computes its Wd addresses while
// `gu_sam`'s tail drains.  `gu_sam` runs 864 blocks on 148 SMs (~3 waves), well
// above the >1.5-waves threshold where triggering completion is safe.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <set>

namespace fk_gla {

#define DEVINL __device__ __forceinline__

// 16-byte payload = 8 bf16.
struct alignas(16) V16 { uint32_t d[4]; };

DEVINL float silu_approx(float v) {
  float h = 0.5f * v, t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return h * (1.0f + t);
}

// acc += dot(w, xv) over the 8 bf16 lanes of one payload.
DEVINL void fma8(float& acc, const V16& w, const V16& xv) {
  const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w);
  const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 wf = __bfloat1622float2(wb[i]);
    const float2 xf = __bfloat1622float2(xb[i]);
    acc = fmaf(wf.x, xf.x, acc);
    acc = fmaf(wf.y, xf.y, acc);
  }
}

DEVINL float warp_sum(float v) {
#pragma unroll
  for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
  return v;
}

// ---------------------------------------------------------------------------
// gu_sam: h[MC, I] = silu(x @ Wg^T) * (x @ Wu^T), partitioned over I.
//   grid = I / BI, block = WARPS * 32.  NVL = H / 256 payload-columns per lane.
// ---------------------------------------------------------------------------
template <int MC, int BI, int WARPS, int UF>
__global__ __launch_bounds__(WARPS * 32) void gu_sam(
    const V16* __restrict__ x,        // [MC, H/8]
    const V16* __restrict__ wgu,      // [2I, H/8]
    __nv_bfloat16* __restrict__ hout, // [MC, I]
    int I, int nvrow) {               // nvrow = H/8
  // Each warp owns ROWS of the 2*BI weight rows this block covers, and walks
  // them together: ROWS*UF payload loads are in flight at once, and the x
  // payload for a given column is converted once for all ROWS.
  constexpr int ROWS = (2 * BI) / WARPS;
  static_assert(ROWS >= 1 && ROWS * WARPS == 2 * BI, "2*BI must be a multiple of WARPS");
  extern __shared__ char smem[];
  V16* xs = reinterpret_cast<V16*>(smem);                    // [MC][nvrow]
  float* sred = reinterpret_cast<float*>(xs + MC * nvrow);   // [2*BI][MC]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int j0 = blockIdx.x * BI;

  for (int i = tid; i < MC * nvrow; i += WARPS * 32) xs[i] = x[i];
  __syncthreads();

  // Warp w owns rows w*ROWS .. w*ROWS+ROWS-1 of the 2*BI-row window; the first
  // BI are gate rows, the rest up rows, so a row maps to either half of wgu.
  const V16* wp[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int rr = warp * ROWS + r;
    const int grow = (rr < BI) ? (j0 + rr) : (I + j0 + rr - BI);
    wp[r] = wgu + (size_t)grow * nvrow + lane;
  }
  float acc[ROWS][MC];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < MC; ++m) acc[r][m] = 0.f;

  // Lane `lane` walks payload columns lane, lane+32, ... so the warp's 32 lanes
  // cover 512 contiguous bytes per step: every step is one full burst.
  for (int t = 0; t < nvrow; t += 32 * UF) {
    V16 w[ROWS][UF];
#pragma unroll
    for (int u = 0; u < UF; ++u)
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int tt = t + 32 * u;
        w[r][u] = (tt < nvrow) ? wp[r][tt] : V16{{0u, 0u, 0u, 0u}};
      }
#pragma unroll
    for (int u = 0; u < UF; ++u) {
      const int tt = t + 32 * u;
      if (tt < nvrow) {
#pragma unroll
        for (int m = 0; m < MC; ++m) {
          const V16 xv = xs[m * nvrow + tt + lane];
#pragma unroll
          for (int r = 0; r < ROWS; ++r) fma8(acc[r][m], w[r][u], xv);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float s = warp_sum(acc[r][m]);
      if (lane == 0) sred[(warp * ROWS + r) * MC + m] = s;
    }
  __syncthreads();

  for (int i = tid; i < BI * MC; i += WARPS * 32) {
    const int jj = i / MC, m = i - jj * MC;
    // Reference order: bf16 GEMM output -> fp32 SiLU -> bf16 -> bf16 multiply.
    const float g = __bfloat162float(__float2bfloat16(sred[jj * MC + m]));
    const __nv_bfloat16 u = __float2bfloat16(sred[(BI + jj) * MC + m]);
    hout[(size_t)m * I + j0 + jj] = __hmul(__float2bfloat16(silu_approx(g)), u);
  }
  // Signal the dependent grid only once *every* thread of this block is done
  // writing h and those writes are device-visible: __syncthreads() alone orders
  // them within the block, and the trigger is a per-block event, so without the
  // fence a thread could signal while another's store is still in flight.
  __syncthreads();
  __threadfence();
  cudaTriggerProgrammaticLaunchCompletion();
}

// ---------------------------------------------------------------------------
// down_gv: y[MC, H] = h @ Wd^T, partitioned over H.  No split-K.
//   grid = H / BN, block = WARPS * 32.
//   SPLIT warps cooperate per output row when BN < WARPS.
// ---------------------------------------------------------------------------
template <int MC, int BN, int WARPS, int UF>
__global__ __launch_bounds__(WARPS * 32) void down_gv(
    const __nv_bfloat16* __restrict__ h, // [MC, I]
    const V16* __restrict__ wd,          // [H, I/8]
    __nv_bfloat16* __restrict__ y,       // [MC, H]
    int H, int nvrow) {                  // nvrow = I/8
  // SPLIT warps share an output row when there are more warps than rows, each
  // taking a strided slice of the reduction; otherwise a warp owns ROWS rows and
  // walks them together for ROWS*UF loads in flight.
  constexpr int SPLIT = (BN >= WARPS) ? 1 : (WARPS / BN);
  constexpr int ROWS = (BN >= WARPS) ? (BN / WARPS) : 1;
  static_assert(ROWS * WARPS == BN || SPLIT * BN == WARPS, "BN/WARPS mismatch");
  extern __shared__ char smem[];
  V16* hs = reinterpret_cast<V16*>(smem);                          // [MC][nvrow]
  float* sred = reinterpret_cast<float*>(hs + MC * nvrow);         // [BN][SPLIT][MC]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int n0 = blockIdx.x * BN;
  const int part = (SPLIT == 1) ? 0 : (warp % SPLIT);
  const int rbase = (SPLIT == 1) ? (warp * ROWS) : (warp / SPLIT);

  const V16* wp[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
    wp[r] = wd + (size_t)(n0 + rbase + r) * nvrow + lane;
  float acc[ROWS][MC];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < MC; ++m) acc[r][m] = 0.f;

  const V16* hv = reinterpret_cast<const V16*>(h);
  // Everything above touches only our own constants and Wd addresses; wait for
  // the producer only here, immediately before the first load of the h it wrote.
  cudaGridDependencySynchronize();
  for (int i = tid; i < MC * nvrow; i += WARPS * 32) hs[i] = hv[i];
  __syncthreads();

  for (int t = part * 32; t < nvrow; t += 32 * UF * SPLIT) {
    V16 w[ROWS][UF];
#pragma unroll
    for (int u = 0; u < UF; ++u)
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int tt = t + 32 * u * SPLIT;
        w[r][u] = (tt < nvrow) ? wp[r][tt] : V16{{0u, 0u, 0u, 0u}};
      }
#pragma unroll
    for (int u = 0; u < UF; ++u) {
      const int tt = t + 32 * u * SPLIT;
      if (tt < nvrow) {
#pragma unroll
        for (int m = 0; m < MC; ++m) {
          const V16 xv = hs[m * nvrow + tt + lane];
#pragma unroll
          for (int r = 0; r < ROWS; ++r) fma8(acc[r][m], w[r][u], xv);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < MC; ++m) {
      const float s = warp_sum(acc[r][m]);
      if (lane == 0) sred[((rbase + r) * SPLIT + part) * MC + m] = s;
    }
  __syncthreads();
  for (int i = tid; i < BN * MC; i += WARPS * 32) {
    const int r = i / MC, m = i - r * MC;
    float s = sred[(r * SPLIT) * MC + m];
#pragma unroll
    for (int p = 1; p < SPLIT; ++p) s += sred[(r * SPLIT + p) * MC + m];
    y[(size_t)m * H + n0 + r] = __float2bfloat16(s);
  }
}

// ---------------------------------------------------------------------------
// Launch helpers.  Programmatic stream serialization must be set on the launch
// or the device-side griddepcontrol intrinsics are no-ops.
// ---------------------------------------------------------------------------
inline bool& pdl_enabled() { static bool on = true; return on; }

// Dynamic shared memory above 48 KB (MC=4 stages 55 KB of h) needs an explicit
// opt-in per kernel; without it the launch fails with cudaErrorInvalidValue.
inline void opt_in_smem(const void* k, size_t bytes) {
  if (bytes <= 48u * 1024u) return;
  static std::set<const void*> done;
  if (done.insert(k).second)
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
}

template <typename F, typename... A>
inline void launch_pdl(F kern, int grid, int block, size_t shmem,
                       cudaStream_t stream, bool pdl, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = shmem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

// cfg selects the (BI/BN, WARPS) pair; kept switchable so the geometry can be
// swept with the benchmark's own timing loop rather than guessed.
#define GU_CASE(ID, BI, WARPS, UF)                                             \
  case ID: {                                                                   \
    const size_t sh = sizeof(V16) * MC * nvx + sizeof(float) * 2 * (BI) * MC;   \
    auto k = gu_sam<MC, BI, WARPS, UF>;                                        \
    opt_in_smem((const void*)k, sh);                                           \
    launch_pdl(k, I / (BI), (WARPS) * 32, sh, stream, pdl_enabled(),            \
               xp, wp, hp, I, nvx);                                            \
    break;                                                                     \
  }

#define DN_CASE(ID, BN, WARPS, UF)                                                    \
  case ID: {                                                                          \
    constexpr int SP = ((BN) >= (WARPS)) ? 1 : ((WARPS) / (BN));                       \
    const size_t sh = sizeof(V16) * MC * nvh + sizeof(float) * (BN) * SP * MC;          \
    auto k = down_gv<MC, BN, WARPS, UF>;                                              \
    opt_in_smem((const void*)k, sh);                                                  \
    launch_pdl(k, H / (BN), (WARPS) * 32, sh, stream, pdl_enabled(),                   \
               hp2, wdp, yp, H, nvh);                                                  \
    break;                                                                            \
  }

template <int MC>
static void run_mc(const at::Tensor& x, const at::Tensor& wgu,
                   const at::Tensor& wd, at::Tensor& h, at::Tensor& y,
                   int gu_cfg, int dn_cfg) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int H = (int)x.size(1);
  const int I = (int)wd.size(1);
  const int nvx = H / 8;
  const int nvh = I / 8;
  const V16* xp = reinterpret_cast<const V16*>(x.const_data_ptr());
  const V16* wp = reinterpret_cast<const V16*>(wgu.const_data_ptr());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr());
  switch (gu_cfg) {
    GU_CASE(0, 8, 8, 2)   // 864 blocks, ROWS=2  -- iter-2 winner shape
    GU_CASE(1, 8, 8, 5)
    GU_CASE(2, 8, 8, 10)
    GU_CASE(3, 4, 4, 2)   // 1728 blocks, ROWS=2
    GU_CASE(4, 4, 4, 5)
    GU_CASE(5, 16, 8, 2)  // 432 blocks, ROWS=4
    GU_CASE(6, 16, 8, 5)
    GU_CASE(7, 4, 8, 2)   // 1728 blocks, ROWS=1
    GU_CASE(8, 4, 8, 5)
    GU_CASE(9, 4, 8, 10)
    GU_CASE(10, 8, 16, 2) // 864 blocks, ROWS=1, 512 thr
    GU_CASE(11, 8, 16, 5)
    GU_CASE(12, 2, 4, 5)  // 3456 blocks, ROWS=1
    GU_CASE(13, 12, 8, 2) // 576 blocks, ROWS=3
    GU_CASE(14, 12, 8, 5)
    GU_CASE(15, 8, 4, 5)  // 864 blocks, ROWS=4
    default: TORCH_CHECK(false, "bad gu_cfg");
  }
  const __nv_bfloat16* hp2 = reinterpret_cast<const __nv_bfloat16*>(h.const_data_ptr());
  const V16* wdp = reinterpret_cast<const V16*>(wd.const_data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  switch (dn_cfg) {
    DN_CASE(0, 4, 8, 3)   // 640 blocks, SPLIT=2  -- iter-2 winner shape
    DN_CASE(1, 4, 8, 9)
    DN_CASE(2, 2, 8, 3)   // 1280 blocks, SPLIT=4
    DN_CASE(3, 2, 8, 9)
    DN_CASE(4, 8, 8, 3)   // 320 blocks, ROWS=1
    DN_CASE(5, 8, 8, 9)
    DN_CASE(6, 8, 8, 27)
    DN_CASE(7, 16, 8, 3)  // 160 blocks, ROWS=2
    DN_CASE(8, 16, 8, 9)
    DN_CASE(9, 4, 4, 9)   // 640 blocks, ROWS=1
    DN_CASE(10, 2, 4, 9)  // 1280 blocks, SPLIT=2
    DN_CASE(11, 1, 8, 3)  // 2560 blocks, SPLIT=8
    DN_CASE(12, 1, 8, 9)
    DN_CASE(13, 4, 16, 3) // 640 blocks, SPLIT=4
    DN_CASE(14, 8, 16, 3) // 320 blocks, SPLIT=2
    DN_CASE(15, 2, 2, 9)  // 1280 blocks, ROWS=1, 64 thr
    default: TORCH_CHECK(false, "bad dn_cfg");
  }
}

}  // namespace fk_gla

// x: [M, H] contiguous bf16, wgu: [2I, H] contiguous bf16, wd: [H, I] contiguous bf16
// returns y: [M, H] bf16.  Two launches, chained with PDL.
at::Tensor gla_mlp_stream(const at::Tensor& x, const at::Tensor& wgu,
                          const at::Tensor& wd, int64_t gu_cfg, int64_t dn_cfg) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && wgu.scalar_type() == at::kBFloat16
              && wd.scalar_type() == at::kBFloat16);
  const int M = (int)x.size(0);
  const int H = (int)x.size(1);
  const int I = (int)wd.size(1);
  TORCH_CHECK(wgu.size(0) == 2 * (int64_t)I && wgu.size(1) == H && wd.size(0) == H);
  TORCH_CHECK(H % 256 == 0 && I % 256 == 0, "shape not supported by the stream path");
  // 16-byte payload loads: every operand base must be 16B aligned.  The
  // benchmark's shifting pool steps by 256 bytes so this always holds, but a
  // caller with an odd view would otherwise fault.
  TORCH_CHECK(((reinterpret_cast<uintptr_t>(x.const_data_ptr()) |
                reinterpret_cast<uintptr_t>(wgu.const_data_ptr()) |
                reinterpret_cast<uintptr_t>(wd.const_data_ptr())) & 15u) == 0,
              "stream path needs 16B-aligned operands");
  const at::cuda::CUDAGuard guard(x.device());
  at::Tensor h(at::detail::empty_cuda({M, I}, at::kBFloat16, x.device(), std::nullopt));
  at::Tensor y(at::detail::empty_cuda({M, H}, at::kBFloat16, x.device(), std::nullopt));
  switch (M) {
    case 1: fk_gla::run_mc<1>(x, wgu, wd, h, y, (int)gu_cfg, (int)dn_cfg); break;
    case 2: fk_gla::run_mc<2>(x, wgu, wd, h, y, (int)gu_cfg, (int)dn_cfg); break;
    case 3: fk_gla::run_mc<3>(x, wgu, wd, h, y, (int)gu_cfg, (int)dn_cfg); break;
    case 4: fk_gla::run_mc<4>(x, wgu, wd, h, y, (int)gu_cfg, (int)dn_cfg); break;
    default: TORCH_CHECK(false, "stream path supports M <= 4, got ", M);
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gla_mlp_stream", &gla_mlp_stream,
        "Weight-streaming SwiGLU MLP for tiny M (two PDL-chained launches)");
  m.def("set_pdl", [](bool on) { fk_gla::pdl_enabled() = on; },
        "Toggle programmatic dependent launch (measurement only)");
}
