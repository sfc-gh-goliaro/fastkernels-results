// Decode-sized shared-expert MoE: the whole layer from one host call.
//
// At one to a few tokens this layer is not a GEMM problem, it is a launch-cost
// problem. Its GPU work is a couple of dozen microseconds of weight streaming,
// but issuing it as six Triton kernels costs ~15 us of host time *each* -- the
// timing events measure the gaps, not the math. So the whole layer sits behind a
// single pybind entry point that launches five plain CUDA kernels: router
// projection, top-k, gate/up gemv with SwiGLU, down gemv, top-k reduction.
//
// Everything here assumes the small-token regime and exploits it:
//   * No expert grouping. With `numel` pairs and E >> numel almost every pair
//     has its own expert, so each (token, slot) pair is handled independently
//     and no alignment/histogram pass is needed.
//   * No tensor cores. Every output element is one dot product over H (or I),
//     so these are bandwidth kernels, not MMA kernels.
//   * The shared expert is expert `E` with routing weight `sigmoid(gate)`, and
//     its gate is row `E` of the router weight, so it needs no separate GEMMs,
//     activation, gemv or epilogue add.
//
// The shape that matters for all three dot-product kernels is memory-level
// parallelism: at one token there are only a few hundred CTAs, so each warp
// walks ROWS output rows at once against a shared input tile. That gives ROWS
// independent 16 B loads in flight per lane and reuses each input element ROWS
// times, instead of one dependent load per FMA group.
//
// Top-k must reproduce the reference's selection, not merely a valid one: the
// reference routes on the bf16 output of a bf16 `F.linear`, where near-ties in
// the top-k tail are common, and trtllm-gen breaks them towards the smaller
// expert index. Both are matched here (`round_bf16`, `better`).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kWarp = 32;
constexpr int kVec = 8;  // bf16/fp16 elements per 16 B load

// The reference's router projection is an `F.linear` in the weight dtype, so its
// logits reach the routing rounded to that dtype. Both supported dtypes are
// 16-bit sign-magnitude, which is what lets the top-k key packing below work.
template <typename T>
struct Bits16;
template <>
struct Bits16<__nv_bfloat16> {
  static __device__ __forceinline__ unsigned int of(float v) {
    return __bfloat16_as_ushort(__float2bfloat16(v));
  }
  static __device__ __forceinline__ float val(unsigned int b) {
    return __bfloat162float(__ushort_as_bfloat16((unsigned short)b));
  }
};
template <>
struct Bits16<__half> {
  static __device__ __forceinline__ unsigned int of(float v) {
    return __half_as_ushort(__float2half_rn(v));
  }
  static __device__ __forceinline__ float val(unsigned int b) {
    return __half2float(__ushort_as_half((unsigned short)b));
  }
};

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_f<__half>(__half v) {
  return __half2float(v);
}

// acc[r] = dot(w[r][:], sx[:]) for ROWS consecutive weight rows of length K, by
// one warp, with 16 B vector loads and all ROWS loads issued before any is
// consumed. The input tile is read from shared memory as 16 B per lane too
// (stored in T, not widened to float): at 8 floats per lane the strided fp32
// reads would be an 8-way bank conflict, which at these ROWS counts costs as
// much as the global traffic it feeds. Butterfly-reduced, so every lane ends up
// with every acc[r] and the ROWS stores can go out from ROWS lanes.
template <typename T, int ROWS>
__device__ __forceinline__ void warp_rows_dot(const T* __restrict__ w, long ldw,
                                              const T* __restrict__ sx, int K, int lane,
                                              float (&acc)[ROWS]) {
  float a[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) a[r] = 0.f;
  for (int k = lane * kVec; k < K; k += kWarp * kVec) {
    float4 q[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) q[r] = *reinterpret_cast<const float4*>(w + r * ldw + k);
    const float4 xq = *reinterpret_cast<const float4*>(sx + k);
    const T* xe = reinterpret_cast<const T*>(&xq);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float xv = to_f<T>(xe[j]);
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
        a[r] = fmaf(to_f<T>(reinterpret_cast<const T*>(&q[r])[j]), xv, a[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float t = a[r];
#pragma unroll
    for (int off = kWarp / 2; off; off >>= 1) t += __shfl_xor_sync(0xffffffff, t, off);
    acc[r] = t;
  }
}

// --------------------------------------------------------------------------
// Router projection: part[sk][m][n] = sum_{k in chunk sk} x[m][k] * rw[n][k].
// --------------------------------------------------------------------------
template <typename T, int WARPS, int ROWS>
__global__ void router_kernel(const T* __restrict__ x, const T* __restrict__ rw,
                              float* __restrict__ part, int M, int N, int H, int sxm,
                              int split) {
  constexpr int PER_BLOCK = WARPS * ROWS;
  const int m = blockIdx.y;
  const int sk = blockIdx.z;
  const int kc = H / split;
  const int k0 = sk * kc;
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;

  extern __shared__ char raw[];
  T* sx = reinterpret_cast<T*>(raw);  // [kc]
  for (int k = threadIdx.x; k < kc; k += WARPS * kWarp) sx[k] = x[(long)m * sxm + k0 + k];
  __syncthreads();

  const int n0 = blockIdx.x * PER_BLOCK + warp * ROWS;
  float* dst = part + ((long)sk * M + m) * N;
  if (n0 + ROWS <= N) {
    float acc[ROWS];
    warp_rows_dot<T, ROWS>(rw + (long)n0 * H + k0, H, sx, kc, lane, acc);
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
      if (lane == r) dst[n0 + r] = acc[r];
  } else {
    // Ragged tail (N is E+1): one row at a time.
    for (int n = n0; n < min(n0 + ROWS, N); ++n) {
      float one[1];
      warp_rows_dot<T, 1>(rw + (long)n * H + k0, H, sx, kc, lane, one);
      if (lane == 0) dst[n] = one[0];
    }
  }
}

// --------------------------------------------------------------------------
// Top-k + renormalize, one warp per token.
//
// TK rounds of a warp-wide argmax give exactly the reference's selection:
// masking the winner by *index* stops duplicate scores from knocking out two
// experts at once, and ties inside a round go to the smaller index. softmax over
// the selected logits equals softmax-then-topk-then-renormalize, so no
// full-vector sum is needed. `SPLIT` is a template parameter so the partial-sum
// loads are independent instead of a dependent chain -- with one resident warp
// there is nothing to hide their latency behind.
// --------------------------------------------------------------------------
// One round of top-k is a warp-wide max over (score, index) pairs. Packing the
// pair into a single u32 -- bf16 score in the high half ordered as an unsigned
// int, `65535 - index` in the low half -- turns that into one
// `redux.sync.max.u32`, and makes ties resolve towards the smaller expert index
// exactly as trtllm-gen's `TopKRedType` does. The scores are already rounded to
// bf16 (that is what the reference routes on), so nothing is lost by comparing
// in bf16. This matters: at one token there is a single resident warp, so a
// five-step shuffle tree per round costs pure dependent latency with nothing to
// hide it behind, and top-k was 3x the two gemvs combined before.
template <typename T>
__device__ __forceinline__ unsigned int pack_key(float v, int idx) {
  unsigned int b = Bits16<T>::of(v);
  b = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
  return (b << 16) | (unsigned int)(65535 - idx);
}

template <typename T>
__device__ __forceinline__ float unpack_val(unsigned int key) {
  unsigned int b = key >> 16;
  return Bits16<T>::val((b & 0x8000u) ? (b & 0x7fffu) : (~b & 0xffffu));
}

__device__ __forceinline__ int unpack_idx(unsigned int key) {
  return 65535 - (int)(key & 0xffffu);
}

__device__ __forceinline__ unsigned int warp_max_u32(unsigned int v) {
  unsigned int r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}

// --------------------------------------------------------------------------
// Top-k + renormalize, one warp per token.
//
// `softmax(l)_e / sum_{e in topk} softmax(l)_e` is a softmax over the selected
// logits, so the renormalized weights need no full-vector sum. `SPLIT` is a
// template parameter so the router partials are summed with independent loads
// rather than a dependent chain.
// --------------------------------------------------------------------------
template <typename T, int WARPS, int VMAX, int SPLIT>
__global__ void topk_kernel(const float* __restrict__ part, float* __restrict__ tw,
                            int* __restrict__ te, int M, int N, int E, int TK, int S,
                            int has_shared) {
  const int warp = threadIdx.x / kWarp;
  const int m = blockIdx.x * WARPS + warp;
  const int lane = threadIdx.x % kWarp;
  if (m >= M) return;
  const float* base = part + (long)m * N;
  const long skip = (long)M * N;

  unsigned int key[VMAX];
#pragma unroll
  for (int j = 0; j < VMAX; ++j) {
    const int e = lane + j * kWarp;
    float s = 0.f;
#pragma unroll
    for (int sk = 0; sk < SPLIT; ++sk) s += (e < E) ? base[sk * skip + e] : 0.f;
    key[j] = (e < E) ? pack_key<T>(s, e) : 0u;
  }

  // Round r's winner is parked in lane r, so the selected set never leaves
  // registers and renormalizing is a single warp reduction.
  float myv = 0.f;
  int myi = 0;
  float top = 0.f;
  for (int r = 0; r < TK; ++r) {
    unsigned int lmax = 0u;
#pragma unroll
    for (int j = 0; j < VMAX; ++j) lmax = max(lmax, key[j]);
    const unsigned int g = warp_max_u32(lmax);
    const int gi = unpack_idx(g);
    const int gj = gi / kWarp;
#pragma unroll
    for (int j = 0; j < VMAX; ++j)
      if (j == gj && gi % kWarp == lane) key[j] = 0u;
    const float gv = unpack_val<T>(g);
    if (r == 0) top = gv;
    if (lane == r) {
      myv = __expf(gv - top);
      myi = gi;
    }
  }
  float sum = myv;
#pragma unroll
  for (int off = kWarp / 2; off; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
  if (lane < TK) {
    tw[m * S + lane] = myv / sum;
    te[m * S + lane] = myi;
  }
  if (has_shared && lane == TK) {
    float g = 0.f;
#pragma unroll
    for (int sk = 0; sk < SPLIT; ++sk) g += base[sk * skip + E];
    tw[m * S + TK] = 1.f / (1.f + __expf(-g));
    te[m * S + TK] = E;
  }
}

// --------------------------------------------------------------------------
// GEMM1 as a gemv per pair: h[pair][i] = silu(x . w_gate[i]) * (x . w_up[i]).
// `w13` stores the gate/up rows interleaved (2i gate, 2i+1 up), so the two rows
// one activation needs are adjacent and a warp's ROWS-row group covers whole
// activations.
// --------------------------------------------------------------------------
template <typename T, int WARPS, int ROWS>
__global__ void gemv1_kernel(const T* __restrict__ x, const int* __restrict__ te,
                             const T* __restrict__ w13, T* __restrict__ h, int H, int I, int S,
                             int sxm) {
  constexpr int PER_BLOCK = WARPS * ROWS;  // weight rows, i.e. PER_BLOCK/2 activations
  const int pair = blockIdx.x;
  const int r0 = blockIdx.y * PER_BLOCK;
  const int e = te[pair];
  const int m = pair / S;
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;

  extern __shared__ char raw[];
  T* sx = reinterpret_cast<T*>(raw);  // [H]
  for (int k = threadIdx.x; k < H; k += WARPS * kWarp) sx[k] = x[(long)m * sxm + k];
  __syncthreads();

  const int rb = r0 + warp * ROWS;
  float acc[ROWS];
  warp_rows_dot<T, ROWS>(w13 + ((long)e * 2 * I + rb) * H, H, sx, H, lane, acc);
#pragma unroll
  for (int j = 0; j < ROWS / 2; ++j) {
    if (lane == j) {
      const float g = acc[2 * j];
      h[(long)pair * I + rb / 2 + j] = (T)(g / (1.f + __expf(-g)) * acc[2 * j + 1]);
    }
  }
}

// --------------------------------------------------------------------------
// GEMM2 as a gemv per pair, routing weight applied on the way out. Written to
// `out[pair]` so the reduction reads a token's S slots contiguously.
// --------------------------------------------------------------------------
template <typename T, int WARPS, int ROWS>
__global__ void gemv2_kernel(const T* __restrict__ h, const int* __restrict__ te,
                             const float* __restrict__ tw, const T* __restrict__ w2,
                             T* __restrict__ out, int H, int I) {
  constexpr int PER_BLOCK = WARPS * ROWS;
  const int pair = blockIdx.x;
  const int n0 = blockIdx.y * PER_BLOCK;
  const int e = te[pair];
  const float w = tw[pair];
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;

  extern __shared__ char raw[];
  T* sh = reinterpret_cast<T*>(raw);  // [I]
  for (int i = threadIdx.x; i < I; i += WARPS * kWarp) sh[i] = h[(long)pair * I + i];
  __syncthreads();

  const int nb = n0 + warp * ROWS;
  float acc[ROWS];
  warp_rows_dot<T, ROWS>(w2 + ((long)e * H + nb) * I, I, sh, I, lane, acc);
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
    if (lane == r) out[(long)pair * H + nb + r] = (T)(acc[r] * w);
}

// --------------------------------------------------------------------------
// Sum a token's S slot outputs (routing weights already applied). 16 B per
// thread: one bf16 per thread would make each warp's read of a slot row 64 B,
// half a cache line, and this kernel moves 800 MiB at prefill sizes.
// --------------------------------------------------------------------------
template <typename T, int BN>
__global__ void reduce_kernel(const T* __restrict__ slots, T* __restrict__ out, int H, int S) {
  const int m = blockIdx.x;
  const int n = (blockIdx.y * BN + threadIdx.x) * kVec;
  if (n >= H) return;
  const T* p = slots + (long)m * S * H + n;
  float acc[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j) acc[j] = 0.f;
  for (int s = 0; s < S; ++s) {
    const float4 q = *reinterpret_cast<const float4*>(p + (long)s * H);
    const T* e = reinterpret_cast<const T*>(&q);
#pragma unroll
    for (int j = 0; j < kVec; ++j) acc[j] += to_f<T>(e[j]);
  }
  T r[kVec];
#pragma unroll
  for (int j = 0; j < kVec; ++j) r[j] = (T)acc[j];
  *reinterpret_cast<float4*>(out + (long)m * H + n) = *reinterpret_cast<const float4*>(r);
}

// Scratch and derived sizes, cached across calls. The small path runs one layer
// at a time and the shape repeats every step, so the host side of a forward is a
// single comparison plus the launches themselves -- which is the whole point of
// this path: at one token it is the host, not the GPU, that sets the latency.
struct SmallPlan {
  int M = -1, H = 0, I = 0, E = 0, TK = 0, S = 0, N = 0, numel = 0, split = 1, rblocks = 0;
  int sxm = 0;
  at::ScalarType st = at::kBFloat16;
  int8_t dev = -1;
  torch::Tensor part, tw, te, h, slots;
};
SmallPlan g_plan;

template <int RW, int RROWS>
void build_plan(SmallPlan& p, const torch::Tensor& x, int E, int TK, bool has_shared) {
  p.M = x.size(0);
  p.H = x.size(1);
  p.sxm = x.stride(0);
  p.E = E;
  p.TK = TK;
  p.st = x.scalar_type();
  p.dev = x.device().index();
  p.S = TK + (has_shared ? 1 : 0);
  p.N = E + (has_shared ? 1 : 0);
  p.numel = p.M * p.S;
  p.rblocks = (p.N + RW * RROWS - 1) / (RW * RROWS);
  // Split the router's K until it covers a couple of waves; the top-k kernel
  // folds the partials back together.
  p.split = 1;
  while (p.split < 8 && (long)p.rblocks * p.M * p.split * 2 < 296 &&
         p.H % (2 * p.split * kWarp * kVec) == 0)
    p.split *= 2;
  auto fo = x.options().dtype(torch::kFloat32);
  p.part = torch::empty({(long)p.split * p.M * p.N}, fo);
  p.tw = torch::empty({p.numel}, fo);
  p.te = torch::empty({p.numel}, x.options().dtype(torch::kInt32));
  p.slots = torch::empty({(long)p.numel * p.H}, x.options());
}

template <typename T>
torch::Tensor run(const torch::Tensor& x, const torch::Tensor& rw, const torch::Tensor& w13,
                  const torch::Tensor& w2, int64_t top_k, int64_t num_experts, bool has_shared) {
  constexpr int RW = 8, RROWS = 4;    // router: 32 expert rows per CTA
  constexpr int GW = 8, G1ROWS = 4;   // gemv1: 32 weight rows = 16 activations
  constexpr int G2W = 8, G2ROWS = 4;  // gemv2: 32 output rows
  constexpr int VMAX = 16;            // top-k: E <= 512
  constexpr int RBN = 128;

  SmallPlan& p = g_plan;
  const int I = w2.size(2);
  if (p.M != (int)x.size(0) || p.H != (int)x.size(1) || p.I != I || p.E != (int)num_experts ||
      p.TK != (int)top_k || p.st != x.scalar_type() || p.dev != x.device().index() ||
      p.sxm != (int)x.stride(0)) {
    build_plan<RW, RROWS>(p, x, num_experts, top_k, has_shared);
    p.I = I;
    p.h = torch::empty({(long)p.numel * I}, x.options());
  }
  const int M = p.M, H = p.H, S = p.S, N = p.N, numel = p.numel, split = p.split;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::empty({M, H}, x.options());

  const T* xp = reinterpret_cast<const T*>(x.const_data_ptr());
  const T* rwp = reinterpret_cast<const T*>(rw.const_data_ptr());
  const T* w13p = reinterpret_cast<const T*>(w13.const_data_ptr());
  const T* w2p = reinterpret_cast<const T*>(w2.const_data_ptr());
  T* hp = reinterpret_cast<T*>(p.h.data_ptr());
  T* sp = reinterpret_cast<T*>(p.slots.data_ptr());
  T* op = reinterpret_cast<T*>(out.data_ptr());
  float* partp = p.part.data_ptr<float>();
  float* twp = p.tw.data_ptr<float>();
  int* tep = p.te.data_ptr<int>();

  router_kernel<T, RW, RROWS><<<dim3(p.rblocks, M, split), RW * kWarp,
                                (H / split) * sizeof(T), stream>>>(
      xp, rwp, partp, M, N, H, p.sxm, split);
#define TOPK(SP)                                                  \
  topk_kernel<T, 1, VMAX, SP><<<M, kWarp, 0, stream>>>(            \
      partp, twp, tep, M, N, p.E, p.TK, S, has_shared ? 1 : 0)
  switch (split) {
    case 1: TOPK(1); break;
    case 2: TOPK(2); break;
    case 4: TOPK(4); break;
    default: TOPK(8); break;
  }
#undef TOPK
  gemv1_kernel<T, GW, G1ROWS><<<dim3(numel, 2 * I / (GW * G1ROWS)), GW * kWarp,
                                H * sizeof(T), stream>>>(xp, tep, w13p, hp, H, I, S, p.sxm);
  gemv2_kernel<T, G2W, G2ROWS><<<dim3(numel, H / (G2W * G2ROWS)), G2W * kWarp,
                                 I * sizeof(T), stream>>>(hp, tep, twp, w2p, sp, H, I);
  reduce_kernel<T, RBN><<<dim3(M, (H / kVec + RBN - 1) / RBN), RBN, 0, stream>>>(sp, op, H, S);
  return out;
}

}  // namespace

// Standalone top-k for the grouped-GEMM path, which computes its router logits
// with a Triton MMA kernel but wants this reduction: a Triton top-k over 512
// experts costs 20 block-wide reductions per token (75 us at 16k tokens) where
// the packed-key form costs ten `redux` instructions (~20 us).
void moe_topk(torch::Tensor part, torch::Tensor tw, torch::Tensor te,
              torch::Tensor logit_dtype, int64_t num_experts, int64_t top_k, bool has_shared) {
  TORCH_CHECK(part.dim() == 3 && part.is_contiguous(), "part must be [split, M, N]");
  const int split = part.size(0);
  const int M = part.size(1);
  const int N = part.size(2);
  const int E = num_experts;
  const int TK = top_k;
  const int S = tw.size(1);
  TORCH_CHECK(E <= 512 && TK <= 32, "E <= 512 and top_k <= 32");
  auto stream = at::cuda::getCurrentCUDAStream();
  // One warp per CTA: the kernel is a handful of instructions per token, so
  // spreading tokens over CTAs (and therefore SMs) beats packing them.
  constexpr int TW_ = 1, VMAX = 16;
  const int grid = M;
  const float* pp = part.const_data_ptr<float>();
  float* twp = tw.data_ptr<float>();
  int* tep = te.data_ptr<int>();
#define TOPK_ONLY(T, SP)                                                     \
  topk_kernel<T, TW_, VMAX, SP><<<grid, TW_ * kWarp, 0, stream>>>(            \
      pp, twp, tep, M, N, E, TK, S, has_shared ? 1 : 0)
#define TOPK_SPLIT(T)                                                        \
  switch (split) {                                                            \
    case 1: TOPK_ONLY(T, 1); break;                                           \
    case 2: TOPK_ONLY(T, 2); break;                                           \
    case 4: TOPK_ONLY(T, 4); break;                                           \
    case 8: TOPK_ONLY(T, 8); break;                                           \
    default: TORCH_CHECK(false, "split must be 1, 2, 4 or 8");                 \
  }
  if (logit_dtype.scalar_type() == at::kBFloat16) {
    TOPK_SPLIT(__nv_bfloat16);
  } else {
    TORCH_CHECK(logit_dtype.scalar_type() == at::kHalf, "bf16 or fp16 only");
    TOPK_SPLIT(__half);
  }
#undef TOPK_SPLIT
#undef TOPK_ONLY
}

// Standalone top-k reduction for the grouped-GEMM path: allocating the output
// and launching a Triton kernel are two python-level calls, and on this host a
// python-level call is 15-25 us -- the same order as the whole reduction.
torch::Tensor moe_reduce(torch::Tensor slots, int64_t M, int64_t S) {
  TORCH_CHECK(slots.dim() == 2 && slots.is_contiguous(), "slots must be [M*S, H]");
  const int H = slots.size(1);
  auto out = torch::empty({M, H}, slots.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int RBN = 128;
  TORCH_CHECK(H % kVec == 0, "H must be a multiple of 8");
  const dim3 grid(M, (H / kVec + RBN - 1) / RBN);
  if (slots.scalar_type() == at::kBFloat16) {
    reduce_kernel<__nv_bfloat16, RBN><<<grid, RBN, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(slots.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, S);
  } else {
    TORCH_CHECK(slots.scalar_type() == at::kHalf, "bf16 or fp16 only");
    reduce_kernel<__half, RBN><<<grid, RBN, 0, stream>>>(
        reinterpret_cast<const __half*>(slots.const_data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()), H, S);
  }
  return out;
}

torch::Tensor shared_expert_moe_small(torch::Tensor x, torch::Tensor rw, torch::Tensor w13,
                                      torch::Tensor w2, int64_t top_k, int64_t num_experts,
                                      bool has_shared) {
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "x must be row-major 2-D");
  if (x.scalar_type() == at::kBFloat16)
    return run<__nv_bfloat16>(x, rw, w13, w2, top_k, num_experts, has_shared);
  TORCH_CHECK(x.scalar_type() == at::kHalf, "bf16 or fp16 only");
  return run<__half>(x, rw, w13, w2, top_k, num_experts, has_shared);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_expert_moe_small", &shared_expert_moe_small,
        "fused shared-expert MoE for small token counts");
  m.def("moe_reduce", &moe_reduce, "sum a token's top-k slot outputs");
  m.def("moe_topk", &moe_topk, "softmax top-k + renormalize from split-K router partials");
}
