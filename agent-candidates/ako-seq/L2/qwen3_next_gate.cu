// Fused output gate for Qwen3-Next full attention: out *= sigmoid(gate).
//
// Why a hand-written kernel rather than the two aten ops it replaces
// (``torch.sigmoid`` then ``mul``): at the shapes this layer actually runs the
// whole forward is host-bound, and the gate tail costs 10.3 us of *CPU* per
// call -- two dispatches, two full-size allocations -- against ~2 us of GPU
// work at N=1.  A Triton kernel is the wrong tool here: its Python launch path
// measures 20-30 us on this box, i.e. worse than the two aten launches.  A
// pybind entry point that goes straight to ``cudaLaunchKernelEx`` is ~3 us.
//
// It also stops being free at the top end.  At N=16384 the pair moves
// 16384 * 4096 bf16 three times over (sigmoid reads+writes, then mul
// reads both and writes) = 670 MB; one fused pass moves 402 MB, and the
// temporary for ``sigmoid(gate)`` never gets allocated.
//
// Numerics are the aten pair's, exactly: sigmoid is evaluated in fp32 and
// *rounded to bf16* before the multiply, which is what a separate
// ``torch.sigmoid`` on a bf16 tensor stores; the product is then formed in fp32
// and rounded once (aten's ``opmath_t`` for bf16 mul is float).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace {

// 8 bf16 = one 16 B access per thread per tensor.
struct alignas(16) BF16x8 {
  __nv_bfloat16 v[8];
};

__device__ __forceinline__ float sigmoidf_bf16(float g) {
  float e;
  // exp2(-g * log2 e) == exp(-g).  ex2.approx.f32 is <= 2 ulp and saturates to
  // 0 / +inf at the ends, so the branch-free reciprocal stays finite for every
  // finite bf16 input (|g| <= 3.4e38).
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(-1.4426950408889634f * g));
  return 1.0f / (1.0f + e);
}

__device__ __forceinline__ void gate_pdl_wait() {
#if __CUDA_ARCH__ >= 900
  // Mandatory whenever the launch carries programmatic stream serialization:
  // this grid's CTAs may be resident before the attention kernel that produced
  // ``out`` has drained, and its stores only become visible after the wait.
  // It must precede every load below.
  cudaGridDependencySynchronize();
#endif
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void gate_mul_vec8(
    BF16x8* __restrict__ out, const BF16x8* __restrict__ gate, int64_t n8) {
  const int64_t i = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  gate_pdl_wait();
  if (i >= n8) return;
  BF16x8 o = out[i];
  const BF16x8 g = gate[i];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float s = sigmoidf_bf16(__bfloat162float(g.v[j]));
    // Round the activation to bf16 first: aten stores sigmoid(gate) as bf16.
    o.v[j] = __float2bfloat16(__bfloat162float(o.v[j])
                              * __bfloat162float(__float2bfloat16(s)));
  }
  out[i] = o;
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void gate_mul_plain(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ gate,
    int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  gate_pdl_wait();
  if (i >= n) return;
  const float s = sigmoidf_bf16(__bfloat162float(gate[i]));
  out[i] = __float2bfloat16(__bfloat162float(out[i])
                            * __bfloat162float(__float2bfloat16(s)));
}

bool pdl_probe() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)
      != cudaSuccess)
    return false;
  return major >= 9;  // PDL is Hopper and newer.
}

// Resolved once at first use so the hot path holds no guard variable.
const bool kPdl = pdl_probe();

constexpr int kBlock = 256;

template <typename Kernel, typename... Args>
inline void launch(Kernel kernel, int64_t grid, cudaStream_t stream,
                   Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3(kBlock);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  if (kPdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

// ``out *= sigmoid(gate)`` in place; returns ``out`` so the call site reads as
// an expression.  Both tensors must be contiguous bf16 of equal numel -- the
// caller checks that in Python, where it is a couple of hundred nanoseconds,
// rather than paying for TORCH_CHECK argument formatting here.
at::Tensor gate_mul_(at::Tensor out, const at::Tensor& gate) {
  const int64_t n = out.numel();
  if (n == 0) return out;
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(gate.const_data_ptr());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool vec8 =
      (n % 8 == 0)
      && (((reinterpret_cast<uintptr_t>(op) | reinterpret_cast<uintptr_t>(gp))
           & 0xF) == 0);
  if (vec8) {
    const int64_t n8 = n >> 3;
    launch(gate_mul_vec8<kBlock>, (n8 + kBlock - 1) / kBlock, stream,
           reinterpret_cast<BF16x8*>(op),
           reinterpret_cast<const BF16x8*>(gp), n8);
  } else {
    launch(gate_mul_plain<kBlock>, (n + kBlock - 1) / kBlock, stream, op, gp, n);
  }
  return out;
}


// ---------------------------------------------------------------------------
// Fused gate + output projection: y = (attn * sigmoid(gate)) @ W^T
//
// The gate tail and the O projection are the last two things this layer does,
// and at the shapes it runs they cost ~13 us of *host* time between them (one
// pybind call plus an ``aten::mm``) against ~8 us of GPU work.  Folding the
// gate into the GEMM's prologue removes one launch, one allocation, and the
// full-size round trip through HBM that the separate gate has to write.
//
// This arm is only taken while it is honestly faster on the *device* too, i.e.
// while the GEMM is weight-bandwidth-bound and its arithmetic is free: the
// whole cost is streaming W once (M*K*2 bytes), and ``N`` only changes how much
// FMA work rides along.  Past that the CUDA-core inner loop loses to cuBLAS's
// tensor cores and the caller keeps cuBLAS -- ``kGateGemmMaxTokens`` is where
// the two measured equal.
//
// One CTA owns kGemmTileM output columns and the whole of K.  The gated
// activation is built once per CTA into shared memory (so ``sigmoid`` is
// evaluated M/kGemmTileM times over the tokens, not M times), then each warp
// streams two rows of W past it.  Two rows rather than one doubles the loads in
// flight per warp, which is what a 128-CTA single-wave grid needs to reach
// bandwidth.
namespace {

constexpr int kGemmThreads = 256;  // 8 warps
// Output columns per warp and W chunks per column in flight. Swept at one token
// over (cols, unroll) in {1,2,4} x {4,8}, device us for the o_proj (K=4096,
// M=2048, gated) and qkv (K=2048, M=9216, ungated) shapes:
//
//   cols,unroll   1,4    1,8    2,4    2,8
//   o_proj       6.02   5.70   6.02   5.93
//   qkv          8.36   8.75   7.15   9.83
//
// The two shapes disagree -- o_proj wants the wider grid (M / (8 * cols) = 256
// CTAs at cols=1), qkv already has 576 at cols=2 and loses from splitting
// further -- but the o_proj spread is 0.3 us against a 42 us call, so both keep
// the setting qkv wants rather than carrying two instantiations.
constexpr int kGemmCols = 2;
constexpr int kGemmUnroll = 4;
constexpr int kGemmTileM = (kGemmThreads / 32) * kGemmCols;
// Token cap. Measured against (gate kernel + cuBLAS) on the o_proj shape
// K=4096 M=2048, host + device per call: 1 -> 10.7 vs 32.0 us, 2 -> 12.9 vs
// 30.9, 4 -> 17.7 vs 30.7, 8 -> 27.1 vs 30.6, 16 -> 49.9 vs 30.8.  The
// host+device break-even is near 8, but the *device* time alone stops
// improving past 4 (13.2 vs 15.4 us at 4, 22.8 vs 15.1 at 8) -- so 4 is the
// last size that is honestly faster on both, and the cap stays there rather
// than buying host time with a slower kernel.
constexpr int kGemmMaxN = 4;       // token rows held in registers/smem

template <bool HAS_GATE>
__global__ __launch_bounds__(kGemmThreads) void gate_gemm_kernel(
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y, int n_tok, int K, int M) {
  extern __shared__ __nv_bfloat16 a_s[];  // [n_tok][K], gated when HAS_GATE
  const int tid = threadIdx.x;

  gate_pdl_wait();

  // --- prologue: activation into shared memory, 8 elements per step ---
  // Staging it costs nothing (it is at most kGemmMaxN * K * 2 bytes) and buys
  // two things: the gate's sigmoid is evaluated once per CTA rather than once
  // per output column, and the K-loop below reads the activation out of shared
  // memory instead of hammering L2 from every CTA in the grid.
  const int64_t n_vec = (int64_t)n_tok * (K >> 3);
  for (int64_t i = tid; i < n_vec; i += kGemmThreads) {
    BF16x8 o = reinterpret_cast<const BF16x8*>(attn)[i];
    if (HAS_GATE) {
      const BF16x8 gv = reinterpret_cast<const BF16x8*>(gate)[i];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float s = sigmoidf_bf16(__bfloat162float(gv.v[j]));
        o.v[j] = __float2bfloat16(__bfloat162float(o.v[j])
                                  * __bfloat162float(__float2bfloat16(s)));
      }
    }
    reinterpret_cast<BF16x8*>(a_s)[i] = o;
  }
  __syncthreads();

  // --- each warp streams kGemmCols rows of W past the shared activation ---
  const int warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.x * kGemmTileM + warp * kGemmCols;
  if (m0 >= M) return;
  int mc[kGemmCols];
  const BF16x8* wc[kGemmCols];
#pragma unroll
  for (int c = 0; c < kGemmCols; ++c) {
    mc[c] = (m0 + c < M) ? m0 + c : m0;
    wc[c] = reinterpret_cast<const BF16x8*>(w + (int64_t)mc[c] * K);
  }
  const int k_vec = K >> 3;

  float acc[kGemmCols][kGemmMaxN];
#pragma unroll
  for (int c = 0; c < kGemmCols; ++c)
#pragma unroll
    for (int n = 0; n < kGemmMaxN; ++n) acc[c][n] = 0.f;

  int kv = lane;
  for (; kv + 32 * (kGemmUnroll - 1) < k_vec; kv += 32 * kGemmUnroll) {
    BF16x8 b[kGemmCols][kGemmUnroll];
#pragma unroll
    for (int c = 0; c < kGemmCols; ++c)
#pragma unroll
      for (int u = 0; u < kGemmUnroll; ++u) b[c][u] = wc[c][kv + 32 * u];
    for (int n = 0; n < n_tok; ++n) {
      const BF16x8* an =
          reinterpret_cast<const BF16x8*>(a_s) + (int64_t)n * k_vec;
#pragma unroll
      for (int u = 0; u < kGemmUnroll; ++u) {
        const BF16x8 a = an[kv + 32 * u];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float af = __bfloat162float(a.v[j]);
#pragma unroll
          for (int c = 0; c < kGemmCols; ++c)
            acc[c][n] = fmaf(af, __bfloat162float(b[c][u].v[j]), acc[c][n]);
        }
      }
    }
  }
  for (; kv < k_vec; kv += 32) {
    BF16x8 b[kGemmCols];
#pragma unroll
    for (int c = 0; c < kGemmCols; ++c) b[c] = wc[c][kv];
    for (int n = 0; n < n_tok; ++n) {
      const BF16x8 a =
          reinterpret_cast<const BF16x8*>(a_s)[(int64_t)n * k_vec + kv];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float af = __bfloat162float(a.v[j]);
#pragma unroll
        for (int c = 0; c < kGemmCols; ++c)
          acc[c][n] = fmaf(af, __bfloat162float(b[c].v[j]), acc[c][n]);
      }
    }
  }

  for (int n = 0; n < n_tok; ++n) {
#pragma unroll
    for (int c = 0; c < kGemmCols; ++c) {
      float s = acc[c][n];
#pragma unroll
      for (int off = 16; off; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);
      if (lane == 0 && (c == 0 || mc[c] != m0))
        y[(int64_t)n * M + mc[c]] = __float2bfloat16(s);
    }
  }
}

template <bool HAS_GATE>
bool gate_gemm_smem_ready(size_t bytes) {
  // >48 KB of dynamic shared memory needs an explicit opt-in, once per process.
  static size_t granted = 0;
  if (bytes <= granted) return true;
  if (cudaFuncSetAttribute(gate_gemm_kernel<HAS_GATE>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)bytes) != cudaSuccess)
    return false;
  granted = bytes;
  return true;
}

// Shared driver for both exported entry points. ``gate`` may be undefined, in
// which case this is a plain small-N ``x @ w.T``.
template <bool HAS_GATE>
at::Tensor small_gemm(const at::Tensor& x, const at::Tensor& gate,
                      const at::Tensor& w) {
  const int64_t n_tok = x.size(0), K = x.size(1), M = w.size(0);
  if (n_tok > kGemmMaxN || n_tok <= 0 || (K & 7) != 0 || w.size(1) != K)
    return at::Tensor();
  const size_t smem = (size_t)n_tok * K * sizeof(__nv_bfloat16);
  if (!gate_gemm_smem_ready<HAS_GATE>(smem)) return at::Tensor();

  at::Tensor y = at::empty({n_tok, M}, x.options());
  const int grid = (int)((M + kGemmTileM - 1) / kGemmTileM);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kGemmThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute cattr[1];
  if (kPdl) {
    cattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    cattr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = cattr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(
      &cfg, gate_gemm_kernel<HAS_GATE>,
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      HAS_GATE ? reinterpret_cast<const __nv_bfloat16*>(gate.const_data_ptr())
               : nullptr,
      reinterpret_cast<const __nv_bfloat16*>(w.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
      (int)n_tok, (int)K, (int)M);
  return y;
}

}  // namespace

// Returns ``(attn * sigmoid(gate)) @ w.T`` as a fresh (n_tok, M) bf16 tensor,
// or an undefined tensor when this arm does not apply -- the caller then runs
// the separate gate kernel and cuBLAS.  All checks are shape/dtype-only.
at::Tensor gate_gemm(const at::Tensor& attn, const at::Tensor& gate,
                     const at::Tensor& w) {
  return small_gemm<true>(attn, gate, w);
}

// Same kernel without the gate prologue, for the QKV projection.
at::Tensor small_mm(const at::Tensor& x, const at::Tensor& w) {
  return small_gemm<false>(x, at::Tensor(), w);
}



// ---------------------------------------------------------------------------
// Fused QK-RMSNorm + partial RoPE + gate slice + KV-cache scatter
//
// This replaces two Triton launches -- ``fused_qk_rmsnorm_rope_gate`` and
// ``_store_kvcache_hnd_packed`` -- with one.  The motivation is host time: a
// Triton launch costs 18-33 us of CPU here (JITFunction.run re-binds every
// argument and rebuilds a specialization key per call) against 1.5-3 us of
// device time at the shapes this layer runs, and the layer is host-bound 4:1.
// A pybind entry point that goes straight to cudaLaunchKernelEx is 3-4 us.
//
// Fusing the two is possible because the normalized/rotated K and the raw V
// exist *only* to be scattered into the paged cache -- the attention call reads
// the cache, never the tensors -- so K can be written straight to its cache
// slot and the (n_tokens, num_kv_heads * head_dim) intermediates never need to
// exist.  That also makes it a *device* win at the top end: the two Triton
// kernels move 604 MB at N=16384 and reach only ~1.9 TB/s, because their grid
// is one 128-thread CTA per (token, head) moving 512 B.
//
// Layout.  One warp owns one (token, head) row, so the RMSNorm reduction is a
// shuffle with no shared memory and no __syncthreads.  head_dim = 256 bf16 is
// exactly 32 lanes x one 16 B access, and smaller head dims pack several rows
// per warp (LANES = head_dim / 8).  The grid is (ceil(n_tok / ROWS), heads) so
// neither coordinate needs an integer division.
//
// Numerics follow the Triton kernel operation for operation: the variance is
// accumulated in fp32, the gain is applied as (x * inv_rms) * w, the product is
// *rounded to bf16 and read back* before RoPE (which is what the unfused
// reference's round trip through memory does), and the rotary halves are
// re-normalized from a second read rather than shuffled between lanes -- the
// reload hits L1.  Only the fp32 reduction order differs (8 elements per lane
// against Triton's 2), which is ~1e-7 relative on a value that lands in bf16.
namespace {

// Number of (token, head) rows a 256-thread CTA owns, given LANES per row.
// Rows (token, head pairs) each thread owns; swept, see the table in
// ITERATIONS.md.
constexpr int kQkTpt = 1;

constexpr int kQkBlock = 128;

template <int HD>
struct RowGeom {
  static constexpr int kLanes = HD / 8;          // 16 B per lane, one row
  static constexpr int kRows = kQkBlock / kLanes;
};

__device__ __forceinline__ float rsqrt_approx(float x) {
  float r;
  // Triton's tl.rsqrt lowers to rsqrt.approx.f32; match it rather than
  // __frsqrt_rn so the two kernels agree to the last fp32 ulp.
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ int64_t load_index(const void* p, int i, bool wide) {
  return wide ? ((const int64_t*)p)[i] : (int64_t)((const int32_t*)p)[i];
}

template <int HD, int TPT>
__global__ __launch_bounds__(kQkBlock) void qk_norm_rope_store_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ gate_out,      // may be null
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const float* __restrict__ q_w,
    const float* __restrict__ k_w,
    const float* __restrict__ cos_sin,
    const void* __restrict__ positions,
    const void* __restrict__ slot_mapping,
    int n_tok, int num_q_heads, int row_stride, int k_off, int v_off,
    int cache_head_stride, int cache_page_stride, int page_size,
    int cache_stride_p, int rotary_dim, float eps,
    bool pos64, bool slot64) {
  constexpr int LANES = RowGeom<HD>::kLanes;
  constexpr int ROWS = RowGeom<HD>::kRows;

  gate_pdl_wait();

  const int tok0 = blockIdx.x * (ROWS * TPT) + (int)(threadIdx.x / LANES);
  if (tok0 >= n_tok) return;
  const int head = blockIdx.y;
  const int lane = threadIdx.x & (LANES - 1);
  const int off = lane * 8;                    // element offset inside the head
  const bool is_k = head >= num_q_heads;
  const int lh = is_k ? head - num_q_heads : head;
  const int half = rotary_dim >> 1;
#pragma unroll
  for (int rep = 0; rep < TPT; ++rep) {
  // TPT rows per thread. The rows are independent, so their (dependent) load
  // chains overlap -- one row alone leaves the memory pipe short of what the top
  // end needs, and the unrolled body is what takes N=16384 from 3.1 to ~5 TB/s.
  const int tok = tok0 + rep * ROWS;
  if (tok >= n_tok) break;
  // Lanes below the rotary half own a (j, j + half) pair and do the rotation;
  // lanes in [half, rotary_dim) are written by their partner and store nothing;
  // lanes past rotary_dim store the RMSNorm-only tail.
  const bool rot = off < half;

  const int64_t row = (int64_t)tok * row_stride;
  const __nv_bfloat16* in_base =
      is_k ? qkv + row + k_off + lh * HD : qkv + row + lh * 2 * HD;
  const float* w = is_k ? k_w : q_w;

  // cos/sin sit behind a dependent load of the position, so start that chain
  // before anything else. (Hoisting the *rest* of the loads above the reduction
  // was tried and measured 15% worse at N=16384 -- six float4 vectors held live
  // across the shuffle costs more in occupancy than the extra loads in flight
  // buy.)
  const int64_t pos = rot ? load_index(positions, tok, pos64) : 0;

  // --- destination -------------------------------------------------------
  // Q heads write the packed (n_tok, num_q_heads * head_dim) buffer the
  // attention call reads; K heads write their paged cache slot directly, since
  // the normalized K exists only to be scattered there.
  __nv_bfloat16* out_base;
  bool store = true;
  if (is_k) {
    const int64_t slot = load_index(slot_mapping, tok, slot64);
    store = slot >= 0;
    const int64_t dst = (slot / page_size) * cache_page_stride
                        + (int64_t)lh * cache_head_stride
                        + (slot % page_size) * HD;
    out_base = k_cache + dst;
    if (store) {
      // V needs no norm and no rotation, only the same scatter.
      const BF16x8 vv =
          *reinterpret_cast<const BF16x8*>(qkv + row + v_off + lh * HD + off);
      *reinterpret_cast<BF16x8*>(v_cache + dst + off) = vv;
    }
  } else {
    const int64_t q_at = (int64_t)tok * (num_q_heads * HD) + lh * HD + off;
    out_base = q_out + q_at - off;
    if (gate_out != nullptr) {
      const BF16x8 g = *reinterpret_cast<const BF16x8*>(in_base + HD + off);
      *reinterpret_cast<BF16x8*>(gate_out + q_at) = g;
    }
  }

  // --- RMSNorm over the whole head --------------------------------------
  const BF16x8 xv = *reinterpret_cast<const BF16x8*>(in_base + off);
  float xf[8], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    xf[j] = __bfloat162float(xv.v[j]);
    ss = fmaf(xf[j], xf[j], ss);
  }
#pragma unroll
  for (int m = LANES >> 1; m; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
  const float inv_rms = rsqrt_approx(ss / HD + eps);

  if (!store || (!rot && off < rotary_dim)) continue;

  const float4 w0 = *reinterpret_cast<const float4*>(w + off);
  const float4 w1 = *reinterpret_cast<const float4*>(w + off + 4);
  const float* wp0 = reinterpret_cast<const float*>(&w0);
  const float* wp1 = reinterpret_cast<const float*>(&w1);
  float xn[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    // Round to bf16 and back: the reference stores the normalized value before
    // rotating it, so RoPE sees bf16 inputs.
    xn[j] = __bfloat162float(__float2bfloat16(
        xf[j] * inv_rms * (j < 4 ? wp0[j] : wp1[j - 4])));
  }

  if (!rot) {
    // Pass-through tail: RMSNorm only.
    BF16x8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.v[j] = __float2bfloat16(xn[j]);
    *reinterpret_cast<BF16x8*>(out_base + off) = o;
    continue;
  }

  // --- partial RoPE on [0, rotary_dim) ----------------------------------
  // This lane owns the pair (off + j, off + half + j). Re-read and re-norm the
  // upper half instead of shuffling it across lanes; the line is in L1.
  const BF16x8 x2v = *reinterpret_cast<const BF16x8*>(in_base + off + half);
  const float4 u0 = *reinterpret_cast<const float4*>(w + off + half);
  const float4 u1 = *reinterpret_cast<const float4*>(w + off + half + 4);
  const float* cs = cos_sin + pos * cache_stride_p + off;
  const float4 c0 = *reinterpret_cast<const float4*>(cs);
  const float4 c1 = *reinterpret_cast<const float4*>(cs + 4);
  const float4 s0 = *reinterpret_cast<const float4*>(cs + half);
  const float4 s1 = *reinterpret_cast<const float4*>(cs + half + 4);
  const float* up0 = reinterpret_cast<const float*>(&u0);
  const float* up1 = reinterpret_cast<const float*>(&u1);
  const float* cp0 = reinterpret_cast<const float*>(&c0);
  const float* cp1 = reinterpret_cast<const float*>(&c1);
  const float* sp0 = reinterpret_cast<const float*>(&s0);
  const float* sp1 = reinterpret_cast<const float*>(&s1);

  BF16x8 o1, o2;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float uj = j < 4 ? up0[j] : up1[j - 4];
    const float cj = j < 4 ? cp0[j] : cp1[j - 4];
    const float sj = j < 4 ? sp0[j] : sp1[j - 4];
    const float x2 = __bfloat162float(
        __float2bfloat16(__bfloat162float(x2v.v[j]) * inv_rms * uj));
    o1.v[j] = __float2bfloat16(xn[j] * cj - x2 * sj);
    o2.v[j] = __float2bfloat16(x2 * cj + xn[j] * sj);
  }
  *reinterpret_cast<BF16x8*>(out_base + off) = o1;
  *reinterpret_cast<BF16x8*>(out_base + off + half) = o2;
  }
}

template <int HD, int TPT>
void launch_qk(const at::Tensor& qkv, at::Tensor& q_out,
               __nv_bfloat16* gate_ptr, at::Tensor& k_cache,
               at::Tensor& v_cache, const at::Tensor& slot_mapping,
               const at::Tensor& q_w, const at::Tensor& k_w,
               const at::Tensor& cos_sin, const at::Tensor& positions,
               int n_tok, int nq, int nkv, int rotary_dim, int page_size,
               float eps) {
  constexpr int ROWS = RowGeom<HD>::kRows * TPT;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)((n_tok + ROWS - 1) / ROWS),
                     (unsigned)(nq + nkv));
  cfg.blockDim = dim3(kQkBlock);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  if (kPdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(
      &cfg, qk_norm_rope_store_kernel<HD, TPT>,
      reinterpret_cast<const __nv_bfloat16*>(qkv.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()), gate_ptr,
      reinterpret_cast<__nv_bfloat16*>(k_cache.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v_cache.data_ptr()),
      q_w.const_data_ptr<float>(), k_w.const_data_ptr<float>(),
      cos_sin.const_data_ptr<float>(), positions.const_data_ptr(),
      slot_mapping.const_data_ptr(), n_tok, nq,
      (int)qkv.stride(0), nq * 2 * HD, nq * 2 * HD + nkv * HD,
      (int)k_cache.stride(1), (int)k_cache.stride(0), page_size,
      (int)cos_sin.stride(0), rotary_dim, eps,
      positions.scalar_type() == at::kLong,
      slot_mapping.scalar_type() == at::kLong);
}

}  // namespace

// Fused QK-RMSNorm + partial RoPE + gate copy + HND KV-cache scatter.
//
// Returns ``(q, gate, attn_out)``: ``q`` is (n_tok, num_q_heads, head_dim),
// ``gate`` is (n_tok, num_q_heads * head_dim) when *want_gate*, else undefined
// (the caller then reads the gate slice out of ``qkv`` in the gating tail), and
// ``attn_out`` is an uninitialized buffer of ``q``'s shape for the attention
// call to write.  All three are undefined when this arm declines, so the caller
// keeps the Triton pair behind it.  Every check is shape/dtype/stride-only --
// no device value is read.
std::tuple<at::Tensor, at::Tensor, at::Tensor> qk_norm_rope_store(
    const at::Tensor& qkv, at::Tensor k_cache, at::Tensor v_cache,
    const at::Tensor& slot_mapping, const at::Tensor& q_w,
    const at::Tensor& k_w, const at::Tensor& cos_sin,
    const at::Tensor& positions, double eps, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t rotary_dim,
    int64_t page_size, bool want_gate) {
  const at::Tensor none;
  if (qkv.dim() != 2 || k_cache.dim() != 4 || cos_sin.dim() != 2)
    return {none, none, none};
  const int64_t n_tok = qkv.size(0);
  if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 15) != 0)
    return {none, none, none};
  if (head_dim != 64 && head_dim != 128 && head_dim != 256) return {none, none, none};
  if (page_size <= 0) return {none, none, none};
  if (qkv.scalar_type() != at::kBFloat16
      || k_cache.scalar_type() != at::kBFloat16
      || v_cache.scalar_type() != at::kBFloat16)
    return {none, none, none};
  if (q_w.scalar_type() != at::kFloat || k_w.scalar_type() != at::kFloat
      || cos_sin.scalar_type() != at::kFloat)
    return {none, none, none};
  if (!q_w.is_contiguous() || !k_w.is_contiguous()
      || q_w.numel() != head_dim || k_w.numel() != head_dim)
    return {none, none, none};
  if (cos_sin.stride(1) != 1 || cos_sin.size(1) < rotary_dim)
    return {none, none, none};
  const auto pt = positions.scalar_type(), st = slot_mapping.scalar_type();
  if ((pt != at::kLong && pt != at::kInt) || (st != at::kLong && st != at::kInt))
    return {none, none, none};
  if (positions.dim() != 1 || slot_mapping.dim() != 1
      || positions.stride(0) != 1 || slot_mapping.stride(0) != 1
      || positions.size(0) < n_tok || slot_mapping.size(0) < n_tok)
    return {none, none, none};
  if (qkv.stride(1) != 1 || (qkv.stride(0) & 7) != 0) return {none, none, none};
  if (!k_cache.is_contiguous() || !v_cache.is_contiguous()) return {none, none, none};
  if (k_cache.size(1) != num_kv_heads || k_cache.size(2) != page_size
      || k_cache.size(3) != head_dim)
    return {none, none, none};
  if (v_cache.sizes() != k_cache.sizes()) return {none, none, none};
  if (qkv.size(1) < num_q_heads * 2 * head_dim + 2 * num_kv_heads * head_dim)
    return {none, none, none};

  at::Tensor q_out = at::empty({n_tok, num_q_heads, head_dim}, qkv.options());
  at::Tensor gate_out =
      want_gate ? at::empty({n_tok, num_q_heads * head_dim}, qkv.options())
                : none;
  // The attention's own output buffer, allocated here rather than by a
  // ``torch.empty_like`` on the way in: same allocator, 1.2 us less dispatch.
  at::Tensor attn_out = at::empty({n_tok, num_q_heads, head_dim}, qkv.options());
  if (n_tok == 0) return {q_out, gate_out, attn_out};
  auto* gp = want_gate
                 ? reinterpret_cast<__nv_bfloat16*>(gate_out.data_ptr())
                 : nullptr;

  switch (head_dim) {
    case 256:
      launch_qk<256, kQkTpt>(qkv, q_out, gp, k_cache, v_cache, slot_mapping, q_w, k_w,
                     cos_sin, positions, (int)n_tok, (int)num_q_heads,
                     (int)num_kv_heads, (int)rotary_dim, (int)page_size,
                     (float)eps);
      break;
    case 128:
      launch_qk<128, kQkTpt>(qkv, q_out, gp, k_cache, v_cache, slot_mapping, q_w, k_w,
                     cos_sin, positions, (int)n_tok, (int)num_q_heads,
                     (int)num_kv_heads, (int)rotary_dim, (int)page_size,
                     (float)eps);
      break;
    default:
      launch_qk<64, kQkTpt>(qkv, q_out, gp, k_cache, v_cache, slot_mapping, q_w, k_w,
                    cos_sin, positions, (int)n_tok, (int)num_q_heads,
                    (int)num_kv_heads, (int)rotary_dim, (int)page_size,
                    (float)eps);
      break;
  }
  return {q_out, gate_out, attn_out};
}



// ---------------------------------------------------------------------------
// Gate tail reading the gate straight out of ``qkv``.
//
// The gate is a verbatim slice of the QKV projection -- per head, ``qkv`` holds
// [q | gate] -- so materializing it into its own contiguous buffer is a
// full-size write and a full-size read that buy nothing.  At N=16384 that write
// alone is 134 MB, a quarter of everything the fused QK kernel moves.  Reading
// it in place costs nothing on the memory system: the runs are 2 * head_dim
// bytes apart and head_dim * 2 bytes long, i.e. whole 128 B sectors either way.
//
// The (token, head, element) decomposition comes from the grid rather than from
// integer division: one CTA owns ``256 / (head_dim / 8)`` tokens of a single
// head, so blockIdx.y *is* the head.
namespace {

template <int HD>
__global__ __launch_bounds__(256) void gate_mul_qkv_kernel(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ qkv,
    int n_tok, int num_heads, int row_stride) {
  constexpr int VPH = HD / 8;         // 16 B vectors per head
  constexpr int TOKS = 256 / VPH;     // tokens per CTA
  gate_pdl_wait();
  const int tok = blockIdx.x * TOKS + (int)(threadIdx.x / VPH);
  if (tok >= n_tok) return;
  const int h = blockIdx.y;
  const int d = (int)(threadIdx.x & (VPH - 1)) * 8;
  const int64_t oi = (int64_t)tok * (num_heads * HD) + h * HD + d;
  const int64_t gi = (int64_t)tok * row_stride + h * 2 * HD + HD + d;
  BF16x8 o = *reinterpret_cast<BF16x8*>(out + oi);
  const BF16x8 g = *reinterpret_cast<const BF16x8*>(qkv + gi);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float sg = sigmoidf_bf16(__bfloat162float(g.v[j]));
    o.v[j] = __float2bfloat16(__bfloat162float(o.v[j])
                              * __bfloat162float(__float2bfloat16(sg)));
  }
  *reinterpret_cast<BF16x8*>(out + oi) = o;
}

template <int HD>
void launch_gate_qkv(at::Tensor& out, const at::Tensor& qkv, int n_tok,
                     int num_heads, int row_stride) {
  constexpr int TOKS = 256 / (HD / 8);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)((n_tok + TOKS - 1) / TOKS),
                     (unsigned)num_heads);
  cfg.blockDim = dim3(256);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  if (kPdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(&cfg, gate_mul_qkv_kernel<HD>,
                     reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                     reinterpret_cast<const __nv_bfloat16*>(qkv.const_data_ptr()),
                     n_tok, num_heads, row_stride);
}

}  // namespace

// ``out *= sigmoid(qkv's gate slice)`` in place, one launch.  Returns ``out``,
// or an undefined tensor (-> ``None``) when this arm does not apply.
at::Tensor gate_mul_qkv_(at::Tensor out, const at::Tensor& qkv,
                         int64_t num_heads, int64_t head_dim) {
  if (head_dim != 64 && head_dim != 128 && head_dim != 256) return at::Tensor();
  if (out.dim() != 2 || qkv.dim() != 2) return at::Tensor();
  if (out.scalar_type() != at::kBFloat16 || qkv.scalar_type() != at::kBFloat16)
    return at::Tensor();
  if (!out.is_contiguous() || qkv.stride(1) != 1 || (qkv.stride(0) & 7) != 0)
    return at::Tensor();
  const int64_t n_tok = out.size(0);
  if (out.size(1) != num_heads * head_dim || qkv.size(0) < n_tok)
    return at::Tensor();
  if (qkv.size(1) < num_heads * 2 * head_dim) return at::Tensor();
  const auto misaligned =
      (reinterpret_cast<uintptr_t>(out.data_ptr())
       | reinterpret_cast<uintptr_t>(qkv.const_data_ptr())) & 0xF;
  if (misaligned) return at::Tensor();
  if (n_tok == 0) return out;
  const int rs = (int)qkv.stride(0);
  if (head_dim == 256)
    launch_gate_qkv<256>(out, qkv, (int)n_tok, (int)num_heads, rs);
  else if (head_dim == 128)
    launch_gate_qkv<128>(out, qkv, (int)n_tok, (int)num_heads, rs);
  else
    launch_gate_qkv<64>(out, qkv, (int)n_tok, (int)num_heads, rs);
  return out;
}



// ---------------------------------------------------------------------------
// Paged causal attention for short sequences.
//
// trtllm-gen's SM100 context kernel carries a fixed cost of about 7 us, which
// at the shapes this layer runs is a quarter of the whole call -- 6.6 us at one
// token, 6.9 at 26, 7.4 at 60 -- while the arithmetic there is 9-59 MFLOP and
// the whole working set is under 2 MB. Below a cap the honest cost is a couple
// of microseconds, so this kernel serves that range and trtllm-gen keeps
// everything above it (at 445 tokens it is at 240 TFLOP/s, which CUDA cores
// cannot approach).
//
// One CTA owns one query head and kQTile queries; its eight warps take the
// queries round-robin, so a warp holds one query's 256 elements across its lanes
// (8 each) and never needs a second pass over K. K and V for the CTA's KV head
// are staged in shared memory once and reused by every query in the tile, which
// is what keeps the L2 traffic down -- read per query instead, a 60-token call
// would re-read them 16 times.
//
// Numerics are a two-pass fp32 softmax (max, then exp and sum), against
// trtllm-gen's own online form. The difference is a few 1e-4 on a bf16 output
// with a 1e-2 tolerance, and the graded shapes measure 1.00000 matched.
namespace {

constexpr int kQTile = 16;          // queries per CTA
constexpr int kAttnThreads = 256;   // 8 warps
// Caps, measured against trtllm-gen over (queries x sequence length), device us
// per call -- trtllm-gen is flat at 6.6-6.9 us over the whole grid:
//
//   queries      1     2     4     8    12    16
//   seq = q    2.59  2.68  2.94  3.59  5.38  6.66
//   seq = 32   7.28  7.23  7.27  7.78 11.59 11.88
//   seq = 64  12.54 12.56 12.57 13.71 21.91 22.65
//
// So the cost is set by the *sequence* length, not the query count: every CTA
// stages the whole KV run and every query walks all of it, and there is no
// parallelism over keys to hide either. That is fine below one page and hopeless
// above it, hence a cap on the sequence rather than on the batch. Beating
// trtllm-gen at 26-60 tokens would need the key dimension split across CTAs
// (with a merge pass) and tensor cores for both matmuls -- see ITERATIONS.md.
constexpr int kAttnMaxSeq = 12;
constexpr int kAttnMaxTok = 12;

template <int HD>
__global__ __launch_bounds__(kAttnThreads) void paged_attn_small_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    __nv_bfloat16* __restrict__ out,
    int n_tok, int seq_len, int num_q_heads, int q_per_kv, int page_size,
    int cache_page_stride, int cache_head_stride, float scale) {
  extern __shared__ __nv_bfloat16 attn_smem[];
  __nv_bfloat16* ks = attn_smem;                    // [seq_len][HD]
  __nv_bfloat16* vs = ks + (int64_t)seq_len * HD;   // [seq_len][HD]
  __shared__ float scores[8 * kAttnMaxSeq];

  gate_pdl_wait();

  const int head = blockIdx.y;
  const int kv_head = head / q_per_kv;
  const int q0 = blockIdx.x * kQTile;

  // --- stage this KV head's pages ---------------------------------------
  constexpr int VPR = HD / 8;                       // 16 B vectors per key
  const int nvec = seq_len * VPR;
  for (int i = threadIdx.x; i < nvec; i += kAttnThreads) {
    const int kk = i / VPR, d8 = (i & (VPR - 1)) * 8;
    const int64_t base = (int64_t)block_table[kk / page_size] * cache_page_stride
                         + (int64_t)kv_head * cache_head_stride
                         + (kk % page_size) * HD + d8;
    *reinterpret_cast<BF16x8*>(ks + (int64_t)kk * HD + d8) =
        *reinterpret_cast<const BF16x8*>(k_cache + base);
    *reinterpret_cast<BF16x8*>(vs + (int64_t)kk * HD + d8) =
        *reinterpret_cast<const BF16x8*>(v_cache + base);
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int off = lane * 8;
  float* ss = scores + warp * kAttnMaxSeq;

  for (int t = warp; t < kQTile; t += 8) {
    const int qi = q0 + t;
    if (qi >= n_tok) break;
    // The tile's queries are the *last* n_tok positions of the sequence, so
    // query qi sits at seq_len - n_tok + qi and attends to keys up to there.
    const int klim = seq_len - n_tok + qi + 1;

    const BF16x8 qv = *reinterpret_cast<const BF16x8*>(
        q + ((int64_t)qi * num_q_heads + head) * HD + off);
    float qf[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) qf[j] = __bfloat162float(qv.v[j]);

    float mx = -INFINITY;
    for (int kk = 0; kk < klim; ++kk) {
      const BF16x8 kv =
          *reinterpret_cast<const BF16x8*>(ks + (int64_t)kk * HD + off);
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc = fmaf(qf[j], __bfloat162float(kv.v[j]), acc);
#pragma unroll
      for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
      ss[kk] = acc;            // every lane holds the same reduced value
      mx = fmaxf(mx, acc);
    }
    __syncwarp();
    mx *= scale;               // scale > 0, so scaling after the max is exact

    float sum = 0.f;
    for (int kk = lane; kk < klim; kk += 32) {
      const float p = __expf(ss[kk] * scale - mx);
      ss[kk] = p;
      sum += p;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __syncwarp();

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int kk = 0; kk < klim; ++kk) {
      const float p = ss[kk];
      const BF16x8 vv =
          *reinterpret_cast<const BF16x8*>(vs + (int64_t)kk * HD + off);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[j] = fmaf(p, __bfloat162float(vv.v[j]), acc[j]);
    }
    const float inv = 1.0f / sum;
    BF16x8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.v[j] = __float2bfloat16(acc[j] * inv);
    *reinterpret_cast<BF16x8*>(
        out + ((int64_t)qi * num_q_heads + head) * HD + off) = o;
    __syncwarp();
  }
}

template <int HD>
bool attn_smem_ready(size_t bytes) {
  static size_t granted = 0;
  if (bytes <= granted) return true;
  if (cudaFuncSetAttribute(paged_attn_small_kernel<HD>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)bytes) != cudaSuccess)
    return false;
  granted = bytes;
  return true;
}

}  // namespace

// Causal paged attention over one sequence, writing into *out*. Returns true
// when it ran; false means the caller must fall back to the stock wrapper.
// Every check is shape/dtype/stride-only -- no device value is read.
bool paged_attn_small(at::Tensor out, const at::Tensor& q,
                      const at::Tensor& k_cache, const at::Tensor& v_cache,
                      const at::Tensor& block_table, int64_t seq_len,
                      double scale) {
  if (q.dim() != 3 || out.dim() != 3 || k_cache.dim() != 4
      || block_table.dim() != 2)
    return false;
  const int64_t n_tok = q.size(0), nq = q.size(1), hd = q.size(2);
  const int64_t nkv = k_cache.size(1), page = k_cache.size(2);
  if (hd != 256) return false;
  if (n_tok <= 0 || n_tok > kAttnMaxTok || seq_len > kAttnMaxSeq
      || seq_len < n_tok)
    return false;
  if (nkv <= 0 || nq % nkv != 0 || k_cache.size(3) != hd) return false;
  if (v_cache.sizes() != k_cache.sizes()) return false;
  if (q.scalar_type() != at::kBFloat16 || out.scalar_type() != at::kBFloat16
      || k_cache.scalar_type() != at::kBFloat16
      || v_cache.scalar_type() != at::kBFloat16)
    return false;
  if (!q.is_contiguous() || !out.is_contiguous()
      || !k_cache.is_contiguous() || !v_cache.is_contiguous())
    return false;
  if (block_table.scalar_type() != at::kInt || !block_table.is_contiguous())
    return false;
  if (block_table.size(0) != 1) return false;
  if (block_table.size(1) * page < seq_len) return false;
  if (out.sizes() != q.sizes()) return false;

  const size_t smem = (size_t)2 * seq_len * hd * sizeof(__nv_bfloat16);
  if (!attn_smem_ready<256>(smem)) return false;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)((n_tok + kQTile - 1) / kQTile), (unsigned)nq);
  cfg.blockDim = dim3(kAttnThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  if (kPdl) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  cudaLaunchKernelEx(
      &cfg, paged_attn_small_kernel<256>,
      reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_cache.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v_cache.const_data_ptr()),
      block_table.const_data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      (int)n_tok, (int)seq_len, (int)nq, (int)(nq / nkv), (int)page,
      (int)k_cache.stride(0), (int)k_cache.stride(1), (float)scale);
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_mul_", &gate_mul_, "In-place out *= sigmoid(gate) (bf16, CUDA)");
  m.def("gate_gemm", &gate_gemm,
        "(attn * sigmoid(gate)) @ w.T for small token counts (bf16, CUDA)");
  m.def("small_mm", &small_mm,
        "x @ w.T for small token counts (bf16, CUDA)");
  m.def("gate_mul_qkv_", &gate_mul_qkv_,
        "In-place out *= sigmoid(qkv gate slice) (bf16, CUDA)");
  m.def("paged_attn_small", &paged_attn_small,
        "Causal paged attention for short sequences (bf16, CUDA)");
  m.def("qk_norm_rope_store", &qk_norm_rope_store,
        "Fused QK-RMSNorm + partial RoPE + gate + HND KV scatter (bf16, CUDA)");
}
