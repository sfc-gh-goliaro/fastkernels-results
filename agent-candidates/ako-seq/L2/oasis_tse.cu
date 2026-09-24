// Oasis timestep embedder: the whole module in two PDL-chained kernels.
//
//   out[B,N2] = silu(emb(t) @ W1^T + b1) @ W2^T + b2
//   emb(t)[b,k] = cos(t[b]*f[k]) for k < 128,  sin(t[b]*f[k-128]) for k >= 128
//
// Everything below is driven by three measurements on B200 under the
// fastkernels bench timing loop (full numbers in ITERATIONS.md):
//
//  1. The window is pure device time with the host far ahead (the harness
//     enqueues a 253MiB l2.zero_() before every start event) and it is quantised
//     to ~2.048us. The harness' own input copy alone measures ~7us; one added
//     tiny op 11.1us, two 13.5us, ten 42.1us. An F.linear reading 0.5MB and one
//     reading 4MB both land on the same 15.3us step, so the 5MB of weights this
//     module needs is nearly free and the *only* thing that matters is how many
//     kernels run. The reference is ten ops; this is two.
//  2. A kernel costs two quanta: one for the launch pipeline (an empty kernel
//     measures +1 quantum) and one for having any duration at all. A
//     `grid.sync()` costs one. That predicts a single cooperative kernel with a
//     grid barrier would be cheaper -- it is not, it measures a quantum *worse*
//     (15.4us vs 13.3us), because a cooperative launch does not benefit from
//     PDL. Two PDL kernels is the measured optimum.
//  3. Programmatic dependent launch is worth two quanta (13.3us with, 17.4us
//     without). It lets a kernel run its grid setup and every load that does not
//     depend on the producer *while the previous op on the stream is still
//     executing*, so both kernels issue their whole weight slice before
//     `griddepcontrol.wait`. The `"memory"` clobber on that barrier is
//     load-bearing: without it nvcc sinks those loads past it, since their first
//     use is after it.
//
// Shape choices: a warp owns one output column and reduces over K inside itself,
// because row-major W[n,k] is already the layout a lane-strided float4 walk over
// k wants -- so there is no pre-transposed weight copy, which matters because the
// harness overwrites the weights with `load_state_dict` after construction.
// Kernel 1 runs 8 warps per block (128 blocks, ~1 per SM). Kernel 2 runs 16
// warps per block with RS=2 warps sharing each column and splitting the B rows
// between them: its cost otherwise steps up a whole quantum as soon as B >= 3
// (per-thread work scales with B), and splitting rows is the one axis that fixes
// that without giving up block count -- 9.2/9.2/9.3/11.2/11.2us across B=2..6
// against 9.1/11.1/11.2/11.2/11.2 with one warp per column. Everything else
// swept (columns per warp, split-K over the reduction, block widths 2..32) was
// neutral or worse; see ITERATIONS.md.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define PDL_TRIGGER() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#else
#define PDL_WAIT()
#define PDL_TRIGGER()
#endif

// cuBLAS' fp32 GEMM path on this GPU is TF32 (torch reports allow_tf32=True /
// float32_matmul_precision "high") and it converts both operands fp32 -> tf32
// with round-to-nearest-even before the MMA. Reproducing that conversion is what
// makes this kernel agree with the reference to ~1e-5 instead of ~1e-4: exact
// fp32 math is *more* accurate than the reference and lands outside the
// harness' atol 1e-5 / rtol 1e-3 window on ~15% of elements (matched_ratio
// 0.85). Only cuBLAS' M == 1 GEMV stays exact fp32, so B == 1 takes the
// reference path in kernel.py.
__device__ __forceinline__ float tf32(float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  unsigned r;
  asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
  return __uint_as_float(r);
#else
  // Same round-to-nearest-even on the low 13 mantissa bits, without the sm_90
  // conversion instruction.
  const unsigned i = __float_as_uint(v);
  return __uint_as_float((i + 0x0fffu + ((i >> 13) & 1u)) & 0xffffe000u);
#endif
}

__device__ __forceinline__ float4 tf32x4(float4 v) {
  return make_float4(tf32(v.x), tf32(v.y), tf32(v.z), tf32(v.w));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float dot4(const float4& a, const float4& b) {
  float s = a.x * b.x;
  s = fmaf(a.y, b.y, s);
  s = fmaf(a.z, b.z, s);
  s = fmaf(a.w, b.w, s);
  return s;
}

// ---------------------------------------------------------------------------
// Kernel 1: sinusoidal embedding + first GEMM + bias + SiLU.
//   h[b,n] = tf32(silu(b1[n] + sum_{k<256} emb[b,k] * W1[n,k]))
// K is fixed at 256 = 32 lanes x 8 floats, so a lane's slice of a weight row is
// exactly two float4. h is stored already rounded to tf32 -- bit-identical to
// letting kernel 2 round it on read, but off kernel 2's critical path.
// ---------------------------------------------------------------------------
template <int B, int WPB, int RS>
__global__ __launch_bounds__(32 * WPB) void tse_k1(
    float* __restrict__ h,             // [B, N]
    const long long* __restrict__ tp,  // [B]
    const float* __restrict__ freqs,   // [128] cached, never rebuilt per call
    const float* __restrict__ w1,      // [N, 256]
    const float* __restrict__ bias1,   // [N]
    int N) {
  constexpr int K = 256, HALF = K / 2, TPB = 32 * WPB;
  // RS warps share an output column and split the B rows between them
  // (strided, so the split stays balanced for odd B), which stops a thread's
  // work from scaling with B.
  constexpr int RPW = (B + RS - 1) / RS;  // rows per warp
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rg = warp % RS;
  const int n = blockIdx.x * (WPB / RS) + warp / RS;
  const bool live = n < N;

  // Producer-independent prologue: this warp's 8 weights, issued before the
  // barrier so the fetch overlaps the previous op on the stream.
  float4 wa = make_float4(0.f, 0.f, 0.f, 0.f), wb = wa;
  if (live) {
    const float4* wp =
        reinterpret_cast<const float4*>(w1 + (size_t)n * K + lane * 8);
    wa = tf32x4(wp[0]);
    wb = tf32x4(wp[1]);
  }
  __shared__ __align__(16) float emb[B][K];

  PDL_WAIT();  // t[] is written by the preceding op on the stream

  // The [B,256] embedding lives here and nowhere else -- no arange, no exp, no
  // int64->fp32 cast, no outer product, no cos/sin pair of kernels, no cat, and
  // no global intermediate. HALF*B items spread over TPB threads rather than
  // HALF items x B rows each, so no thread idles when TPB > HALF.
#pragma unroll 1
  for (int i = threadIdx.x; i < HALF * B; i += TPB) {
    const int b = i / HALF, k = i & (HALF - 1);
    float sv, cv;
    // Accurate libdevice sincos. The MUFU pair (__sincosf) is ~4x cheaper and
    // was tried: its argument-reduction error is too large even against the
    // tf32 quantum this gets rounded to, giving matched_ratio 0.981-0.989 at
    // t ~ 1e3 -- below the harness' 0.99 bar. Do not re-try it.
    sincosf(static_cast<float>(tp[b]) * freqs[k], &sv, &cv);
    emb[b][k] = tf32(cv);
    emb[b][k + HALF] = tf32(sv);
  }
  __syncthreads();

  float acc[RPW];
#pragma unroll
  for (int j = 0; j < RPW; ++j) {
    const int b = rg + j * RS;
    if (RS == 1 || b < B) {
      const float4 ea = *reinterpret_cast<const float4*>(&emb[b][lane * 8]);
      const float4 eb = *reinterpret_cast<const float4*>(&emb[b][lane * 8 + 4]);
      acc[j] = warp_sum(dot4(wa, ea) + dot4(wb, eb));
    }
  }
  // Direct 4B stores from lane 0, deliberately *not* staged through shared
  // memory into wide coalesced stores: measured, the extra __syncthreads costs
  // more than the scattered transactions save (this kernel alone stays at 9.2us
  // through B=4 with these and slips to 11.2us with the staged version).
  if (live && lane == 0) {
    const float bv = bias1[n];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
      const int b = rg + j * RS;
      if (RS == 1 || b < B) {
        const float z = acc[j] + bv;
        h[(size_t)b * N + n] = tf32(z / (1.0f + expf(-z)));
      }
    }
  }
  PDL_TRIGGER();
}

// ---------------------------------------------------------------------------
// Kernel 2: second GEMM + bias.  out[b,n] = b2[n] + sum_{k<K} h[b,k]*W2[n,k]
// KIT = K/128, i.e. the float4 per lane of this warp's weight row.
// ---------------------------------------------------------------------------
template <int B, int WPB, int KIT, int RS>
__global__ __launch_bounds__(32 * WPB) void tse_k2(
    float* __restrict__ out,          // [B, N]
    const float* __restrict__ h,      // [B, K], already tf32-rounded
    const float* __restrict__ w2,     // [N, K]
    const float* __restrict__ bias2,  // [N]
    int N) {
  constexpr int K = KIT * 128, TPB = 32 * WPB;
  constexpr int RPW = (B + RS - 1) / RS;  // rows per warp (see kernel 1)
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rg = warp % RS;
  const int n = blockIdx.x * (WPB / RS) + warp / RS;
  const bool live = n < N;

  // Producer-independent prologue: this warp's entire weight row. Across the
  // grid that is the whole 4MB of W2, fetched before the barrier.
  float4 wv[KIT];
  {
    const float4* wp =
        reinterpret_cast<const float4*>(w2 + (size_t)(live ? n : 0) * K);
#pragma unroll
    for (int i = 0; i < KIT; ++i)
      wv[i] = live ? tf32x4(wp[i * 32 + lane]) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  __shared__ __align__(16) float sh[B][K];

  PDL_WAIT();  // h[] is written by kernel 1

  // Every warp in the block needs all of h, so stage it once in shared memory.
  // Reading it straight from global instead (L1-served, WPB-fold redundant, no
  // barrier) was measured a full quantum slower.
#pragma unroll 1
  for (int i = threadIdx.x * 4; i < B * K; i += TPB * 4)
    *reinterpret_cast<float4*>(&sh[0][i]) =
        *reinterpret_cast<const float4*>(h + i);
  __syncthreads();

  float acc[RPW];
#pragma unroll
  for (int j = 0; j < RPW; ++j) acc[j] = 0.0f;
#pragma unroll
  for (int i = 0; i < KIT; ++i) {
    const int k0 = (i * 32 + lane) * 4;
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
      const int b = rg + j * RS;
      if (RS == 1 || b < B)
        acc[j] += dot4(wv[i], *reinterpret_cast<const float4*>(&sh[b][k0]));
    }
  }
#pragma unroll
  for (int j = 0; j < RPW; ++j) {
    const int b = rg + j * RS;
    const float v = warp_sum(acc[j]);
    if (lane == 0 && live && (RS == 1 || b < B))
      out[(size_t)b * N + n] = v + bias2[n];
  }
  PDL_TRIGGER();
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
template <typename F, typename... Args>
inline void launch_pdl(F kernel, int blocks, int threads, cudaStream_t stream,
                       bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.numAttrs = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cudaLaunchKernelEx(&cfg, kernel, args...);
}

int env_int(const char* name, int fallback) {
  const char* v = getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  const int x = atoi(v);
  return x > 0 ? x : fallback;
}

struct Cfg {
  int wpb1, wpb2;  // warps per block, per kernel
  bool pdl;
  int mode;  // 0 = both kernels; 1 = kernel 1 only, 2 = kernel 2 only (probes)
};

// Defaults are the measured winners; the env knobs exist for the sweep harness.
Cfg g_cfg = {env_int("FK_OASIS_WPB1", 8), env_int("FK_OASIS_WPB2", 162),
             env_int("FK_OASIS_PDL", 1) != 0, env_int("FK_OASIS_MODE", 1) - 1};

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// (warps per block, warps per column). Row splitting only pays if it keeps the
// block count up, so it comes with a wider block.
#define WPB_SWITCH(LAUNCH, BB, W) \
  switch (W) {                    \
    case 2:                       \
      LAUNCH(BB, 2, 1);           \
      break;                      \
    case 8:                       \
      LAUNCH(BB, 8, 1);           \
      break;                      \
    case 16:                      \
      LAUNCH(BB, 16, 1);          \
      break;                      \
    case 82:                      \
      LAUNCH(BB, 8, 2);           \
      break;                      \
    case 162:                     \
      LAUNCH(BB, 16, 2);          \
      break;                      \
    case 164:                     \
      LAUNCH(BB, 16, 4);          \
      break;                      \
    case 322:                     \
      LAUNCH(BB, 32, 2);          \
      break;                      \
    case 324:                     \
      LAUNCH(BB, 32, 4);          \
      break;                      \
    default:                      \
      LAUNCH(BB, 4, 1);           \
      break;                      \
  }

#define LAUNCH_K1(BB, WW, RR)                                                 \
  launch_pdl(tse_k1<BB, WW, RR>, CEIL_DIV(N1, (WW) / (RR)), 32 * (WW), stream, \
             c.pdl, hp, tp, fp, w1p, b1p, N1)

#define LAUNCH_K2(BB, WW, RR)                                              \
  launch_pdl(tse_k2<BB, WW, 8, RR>, CEIL_DIV(N2, (WW) / (RR)), 32 * (WW),  \
             stream, c.pdl, op, hp, w2p, b2p, N2)

}  // namespace

// Grid-shape override for the sweep harness in ITERATIONS.md.
void fk_oasis_set_cfg(int64_t wpb1, int64_t wpb2, int64_t pdl, int64_t mode) {
  g_cfg.wpb1 = static_cast<int>(wpb1);
  g_cfg.wpb2 = static_cast<int>(wpb2);
  g_cfg.pdl = pdl != 0;
  g_cfg.mode = static_cast<int>(mode);
}

// t: int64 [B] (cuda, contiguous, 2 <= B <= 8), freqs: fp32 [128],
// w1: [N1,256], b1: [N1], w2: [N2,N1], b2: [N2].  Returns fp32 [B, N2].
// N1 must be 1024 (kernel 2's K and its shared-memory staging buffer are
// compile-time constants); kernel.py gates on that.
at::Tensor fk_oasis_tse(const at::Tensor& t, const at::Tensor& freqs,
                        const at::Tensor& w1, const at::Tensor& b1,
                        const at::Tensor& w2, const at::Tensor& b2) {
  const int B = static_cast<int>(t.size(0));
  const int N1 = static_cast<int>(w1.size(0));
  const int N2 = static_cast<int>(w2.size(0));
  at::Tensor h = at::empty({B, N1}, w1.options());
  at::Tensor out = at::empty({B, N2}, w1.options());

  float* hp = h.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const long long* tp = reinterpret_cast<const long long*>(t.data_ptr());
  const float* fp = freqs.data_ptr<float>();
  const float* w1p = w1.data_ptr<float>();
  const float* b1p = b1.data_ptr<float>();
  const float* w2p = w2.data_ptr<float>();
  const float* b2p = b2.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Cfg& c = g_cfg;

#define DISPATCH_B(BB)                                 \
  case BB:                                             \
    if (c.mode != 2) WPB_SWITCH(LAUNCH_K1, BB, c.wpb1); \
    if (c.mode != 1) WPB_SWITCH(LAUNCH_K2, BB, c.wpb2); \
    break;

  switch (B) {
    DISPATCH_B(2)
    DISPATCH_B(3)
    DISPATCH_B(4)
    DISPATCH_B(5)
    DISPATCH_B(6)
    DISPATCH_B(7)
    DISPATCH_B(8)
    default:
      TORCH_CHECK(false, "fk_oasis_tse: unsupported batch ", B);
  }
#undef DISPATCH_B
  return out;
}
