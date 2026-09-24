// SiLU-and-Mul: flat vector-space, occupancy-aware launch geometry.
//
//   out[t, j] = silu(in[t, j]) * in[t, d + j]      d = in.size(-1)/2
//
// Design (vs. the vLLM one-block-per-token reference, grid=num_tokens /
// block=min(d/vec,1024)):
//  * Work is indexed in a flat vector space (row recovered by a 64-bit magic
//    multiply), so block width is independent of `d`. The reference gives
//    d=384 a 24-thread block and d=14336 only num_tokens blocks.
//  * Grid sized from need vs. SMs x resident-blocks x waves, with a
//    grid-stride loop over BLOCK*ILP-vector tiles.
//  * Vector width and ILP are template parameters, not a num_tokens>128 gate.
//    Swept: 16B vectors and ILP=1 win on every captured shape, so that is
//    what dispatch_auto ships; the wider paths stay for the probe.
//  * silu(x) = 0.5*x*(1 + tanh(0.5*x)) via tanh.approx.f32 -- one MUFU per
//    element instead of the two (ex2 + rcp) an expf sigmoid needs. The
//    gate multiply is packed (__hmul2), which is also what the eager
//    reference does (silu rounds to bf16 before the multiply).
//  * cudaLaunchKernelEx + programmatic stream serialization, with
//    cudaGridDependencySynchronize() before the first load: lets the block
//    scheduler start us while the producer kernel's tail drains.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <algorithm>

namespace fk {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  #define FK_HAS_256B 1
#else
  #define FK_HAS_256B 0
#endif

// ---------------------------------------------------------------- vector I/O
template <int VB>
struct Vec;

template <>
struct alignas(32) Vec<32> {
  uint32_t d[8];
};

template <>
struct alignas(16) Vec<16> {
  uint32_t d[4];
};

template <int VB>
__device__ __forceinline__ void ldv(Vec<VB>& v, const void* p);

template <>
__device__ __forceinline__ void ldv<32>(Vec<32>& v, const void* p) {
#if FK_HAS_256B
  asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v.d[0]), "=r"(v.d[1]), "=r"(v.d[2]), "=r"(v.d[3]),
                 "=r"(v.d[4]), "=r"(v.d[5]), "=r"(v.d[6]), "=r"(v.d[7])
               : "l"(p));
#else
  const int4* q = reinterpret_cast<const int4*>(p);
  *reinterpret_cast<int4*>(&v.d[0]) = __ldg(q);
  *reinterpret_cast<int4*>(&v.d[4]) = __ldg(q + 1);
#endif
}

template <>
__device__ __forceinline__ void ldv<16>(Vec<16>& v, const void* p) {
  *reinterpret_cast<int4*>(&v.d[0]) = __ldg(reinterpret_cast<const int4*>(p));
}

template <int VB>
__device__ __forceinline__ void stv(void* p, const Vec<VB>& v);

template <>
__device__ __forceinline__ void stv<32>(void* p, const Vec<32>& v) {
#if FK_HAS_256B
  asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :
               : "l"(p), "r"(v.d[0]), "r"(v.d[1]), "r"(v.d[2]), "r"(v.d[3]),
                 "r"(v.d[4]), "r"(v.d[5]), "r"(v.d[6]), "r"(v.d[7])
               : "memory");
#else
  int4* q = reinterpret_cast<int4*>(p);
  q[0] = *reinterpret_cast<const int4*>(&v.d[0]);
  q[1] = *reinterpret_cast<const int4*>(&v.d[4]);
#endif
}

template <>
__device__ __forceinline__ void stv<16>(void* p, const Vec<16>& v) {
  *reinterpret_cast<int4*>(p) = *reinterpret_cast<const int4*>(&v.d[0]);
}

// ----------------------------------------------------------------- math core
// silu(x) = x * sigmoid(x) = 0.5*x*(1 + tanh(0.5*x)).
// tanh.approx.f32 is ~2^-22 relative, so the (1 + tanh) cancellation for
// x << 0 stays far inside the bf16 output's own rounding.
__device__ __forceinline__ float silu_f(float x) {
  float h = 0.5f * x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return fmaf(h, t, h);
#else
  return x / (1.0f + __expf(-x));
#endif
}

// Ops<T, PMUL>::apply handles one 32-bit lane of a vector.
//   PMUL=true  : round silu to the storage type, then multiply packed.
//   PMUL=false : keep the gate multiply in fp32 (one fewer rounding).
template <typename T, bool PMUL>
struct Ops;

template <bool PMUL>
struct Ops<__nv_bfloat16, PMUL> {
  static __device__ __forceinline__ uint32_t apply(uint32_t xr, uint32_t yr) {
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(&xr);
    const __nv_bfloat162 y = *reinterpret_cast<const __nv_bfloat162*>(&yr);
    const float2 fx = __bfloat1622float2(x);
    __nv_bfloat162 o;
    if constexpr (PMUL) {
      o = __hmul2(__float22bfloat162_rn(
                      make_float2(silu_f(fx.x), silu_f(fx.y))),
                  y);
    } else {
      const float2 fy = __bfloat1622float2(y);
      o = __float22bfloat162_rn(
          make_float2(silu_f(fx.x) * fy.x, silu_f(fx.y) * fy.y));
    }
    return *reinterpret_cast<const uint32_t*>(&o);
  }
};

template <bool PMUL>
struct Ops<__half, PMUL> {
  static __device__ __forceinline__ uint32_t apply(uint32_t xr, uint32_t yr) {
    const __half2 x = *reinterpret_cast<const __half2*>(&xr);
    const __half2 y = *reinterpret_cast<const __half2*>(&yr);
    const float2 fx = __half22float2(x);
    __half2 o;
    if constexpr (PMUL) {
      o = __hmul2(__float22half2_rn(make_float2(silu_f(fx.x), silu_f(fx.y))),
                  y);
    } else {
      const float2 fy = __half22float2(y);
      o = __float22half2_rn(
          make_float2(silu_f(fx.x) * fy.x, silu_f(fx.y) * fy.y));
    }
    return *reinterpret_cast<const uint32_t*>(&o);
  }
};

template <bool PMUL>
struct Ops<float, PMUL> {
  static __device__ __forceinline__ uint32_t apply(uint32_t xr, uint32_t yr) {
    const float r = silu_f(*reinterpret_cast<const float*>(&xr)) *
                    *reinterpret_cast<const float*>(&yr);
    return *reinterpret_cast<const uint32_t*>(&r);
  }
};

// ------------------------------------------------------------- fast PDL path
// nv    : vectors per output row (= d / (VB/sizeof(T)))
// magic : floor(2^64/nv)+1 -- hi64(k*magic) == k/nv whenever k*nv < 2^64
template <typename T, int VB, int BLOCK, int ILP, bool PMUL, uint32_t NVC = 0>
__global__ __launch_bounds__(BLOCK) void silu_mul_flat(
    T* __restrict__ out, const T* __restrict__ in, unsigned long long magic,
    uint32_t d, uint32_t total_vecs, uint32_t full_tiles, bool rev) {
  constexpr uint32_t VE = VB / sizeof(T);
  constexpr int LANES = VB / 4;
  constexpr uint32_t TV = static_cast<uint32_t>(BLOCK) * ILP;
  using V = Vec<VB>;
  const uint32_t tid = threadIdx.x;
  const uint32_t nblk = gridDim.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif

  for (uint32_t i = blockIdx.x; i < full_tiles; i += nblk) {
    const uint32_t tile = rev ? (full_tiles - 1u - i) : i;
    const uint32_t base = tile * TV + tid;
    V xv[ILP], yv[ILP];
    uint64_t xo[ILP];
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      const uint32_t k = base + static_cast<uint32_t>(u) * BLOCK;
      const uint32_t t = NVC ? (k / NVC)
                             : static_cast<uint32_t>(__umul64hi(k, magic));
      xo[u] = static_cast<uint64_t>(k) * VE + static_cast<uint64_t>(t) * d;
    }
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      ldv<VB>(xv[u], in + xo[u]);
      ldv<VB>(yv[u], in + xo[u] + d);
    }
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      V o;
#pragma unroll
      for (int j = 0; j < LANES; ++j)
        o.d[j] = Ops<T, PMUL>::apply(xv[u].d[j], yv[u].d[j]);
      const uint32_t k = base + static_cast<uint32_t>(u) * BLOCK;
      stv<VB>(out + static_cast<uint64_t>(k) * VE, o);
    }
  }

  // Tail: the < TV vectors that do not form a whole tile.
  for (uint32_t k = full_tiles * TV + blockIdx.x * BLOCK + tid; k < total_vecs;
       k += nblk * BLOCK) {
    const uint32_t t = NVC ? (k / NVC)
                           : static_cast<uint32_t>(__umul64hi(k, magic));
    const uint64_t o = static_cast<uint64_t>(k) * VE +
                       static_cast<uint64_t>(t) * d;
    V xv, yv, ov;
    ldv<VB>(xv, in + o);
    ldv<VB>(yv, in + o + d);
#pragma unroll
    for (int j = 0; j < LANES; ++j)
      ov.d[j] = Ops<T, PMUL>::apply(xv.d[j], yv.d[j]);
    stv<VB>(out + static_cast<uint64_t>(k) * VE, ov);
  }
}

// --------------------------------------------------------- decomposition probe
// Dev-only (silu_and_mul_probe). The shipped geometry with parts of the body
// removed, so the harness's timed window can be split into launch / load /
// store instead of guessed at:
//   0  nothing (one block, grid-dependency sync only)
//   1  shipped grid, sync only, no memory traffic
//   2  loads only -- both halves read, discarded through a never-taken store
//   3  stores only -- output written, input never read
//   4  full body (identical to the shipped kernel)
//   5  loads only, but from `alt` -- a cold buffer the producer copy never
//      wrote, so this is the same traffic with no L2 residency to inherit
//   6  full body reading `alt` instead of the copy's output
//   7  full body with the grid-dependency sync moved *after* address
//      formation, so the pre-sync preamble is as long as possible
//   8  full body + cudaTriggerProgrammaticLaunchCompletion() after the
//      last store issue
//   9  both 7 and 8
template <typename T, int VB, int BLOCK, int KIND>
__global__ __launch_bounds__(BLOCK) void silu_mul_probe(
    T* __restrict__ out, const T* __restrict__ in, const T* __restrict__ alt,
    unsigned long long magic, uint32_t d, uint32_t total_vecs) {
  constexpr uint32_t VE = VB / sizeof(T);
  constexpr int LANES = VB / 4;
  using V = Vec<VB>;
  const uint32_t stride = gridDim.x * BLOCK;
  constexpr bool LATE_SYNC = (KIND == 7 || KIND == 9);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (LATE_SYNC) {
    // Everything the kernel can do without touching the producer's output:
    // index math, magic divide, address formation. Then sync.
    const uint32_t k0 = blockIdx.x * BLOCK + threadIdx.x;
    const uint32_t t0 = static_cast<uint32_t>(__umul64hi(k0, magic));
    const uint64_t o0 =
        static_cast<uint64_t>(k0) * VE + static_cast<uint64_t>(t0) * d;
    __threadfence_block();  // keep the address math above the sync
    cudaGridDependencySynchronize();
    if (o0 == ~0ull) reinterpret_cast<uint32_t*>(out)[0] = 1u;  // never taken
  } else {
    cudaGridDependencySynchronize();
  }
#endif
  if constexpr (KIND == 0) return;
  uint32_t acc = 0;
  for (uint32_t k = blockIdx.x * BLOCK + threadIdx.x; k < total_vecs;
       k += stride) {
    if constexpr (KIND == 1) {
      acc += k;  // index math only
      continue;
    }
    const uint32_t t = static_cast<uint32_t>(__umul64hi(k, magic));
    const uint64_t o =
        static_cast<uint64_t>(k) * VE + static_cast<uint64_t>(t) * d;
    V xv, yv, ov;
    const T* src = (KIND == 5 || KIND == 6) ? alt : in;
    static_assert(KIND != 3 || true, "");
    if constexpr (KIND != 3) {
      ldv<VB>(xv, src + o);
      ldv<VB>(yv, src + o + d);
    }
    if constexpr (KIND == 2 || KIND == 5) {
#pragma unroll
      for (int j = 0; j < LANES; ++j) acc ^= xv.d[j] ^ yv.d[j];
      continue;
    }
#pragma unroll
    for (int j = 0; j < LANES; ++j)
      ov.d[j] = (KIND == 3) ? 0x3f803f80u : Ops<T, true>::apply(xv.d[j], yv.d[j]);
    stv<VB>(out + static_cast<uint64_t>(k) * VE, ov);
  }
  // Sink that keeps KIND 1/2 alive without ever executing: `acc` is data- or
  // index-dependent, so the compiler cannot fold the comparison away.
  if constexpr (KIND == 1 || KIND == 2 || KIND == 5)
    if (acc == 0xDEADBEEFu) reinterpret_cast<uint32_t*>(out)[0] = acc;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (KIND == 8 || KIND == 9) cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// --------------------------------------------------------- generic fallback
// Scalar, 64-bit indexed; used when d is not a multiple of the vector width,
// the tensors are not vector-aligned, or the row is a single vector.
template <typename T>
__global__ void silu_mul_scalar(T* __restrict__ out, const T* __restrict__ in,
                                uint32_t d, unsigned long long magic, int shift,
                                int64_t total) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < total; i += stride) {
    const uint32_t t = (shift >= 0)
                           ? static_cast<uint32_t>(i >> shift)
                           : static_cast<uint32_t>(__umul64hi(
                                 static_cast<unsigned long long>(i), magic));
    const int64_t o = i + static_cast<int64_t>(t) * d;
    out[i] = static_cast<T>(silu_f(static_cast<float>(in[o])) *
                            static_cast<float>(in[o + d]));
  }
}

// ------------------------------------------------------------------- launch
// hi64(k * magic_for(nv)) == k / nv for every k with k*nv < 2^64.
// Requires nv >= 2: nv == 1 needs the constant 2^64, which wraps to 0.
inline unsigned long long magic_for(uint32_t nv) {
  return (~0ull / nv) + 1ull;
}

// log2(n) if n is a power of two, else -1.
inline int pow2_shift(uint32_t n) {
  if (n == 0 || (n & (n - 1)) != 0) return -1;
  int s = 0;
  while ((1u << s) != n) ++s;
  return s;
}

inline int sm_count() {
  static const int n = [] {
    int dev = 0, v = 148;
    if (cudaGetDevice(&dev) == cudaSuccess)
      cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
    return v;
  }();
  return n;
}

// Set by the tuned probe only, to measure what PDL is worth per shape.
// Without the attribute the device-side cudaGridDependencySynchronize()
// degrades to a no-op, so the kernel stays correct either way.
inline bool& pdl_enabled() {
  static bool on = true;
  return on;
}

template <typename F, typename... A>
inline void launch_pdl(F kern, dim3 grid, dim3 block, cudaStream_t stream,
                       A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl_enabled() ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

// Probe-only launcher: the launch-attribute variants measured in round 2.
//   0 stream serialization (what ships)      1 + programmatic event
//   2 + programmatic event, triggerAtBlockStart
//   3 programmatic event only (no serialization)   4 no attributes at all
// All four measured neutral-or-worse; see ITERATIONS.md.
template <typename F, typename... A>
inline void launch_probe(F kern, dim3 grid, dim3 block, cudaStream_t stream,
                         int lmode, A... args) {
  static cudaEvent_t ev = [] {
    cudaEvent_t e = nullptr;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    return e;
  }();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (lmode != 3 && lmode != 4) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (lmode == 1 || lmode == 2 || lmode == 3) {
    attr[n].id = cudaLaunchAttributeProgrammaticEvent;
    attr[n].val.programmaticEvent.event = ev;
    attr[n].val.programmaticEvent.flags = 0;
    attr[n].val.programmaticEvent.triggerAtBlockStart = (lmode == 2) ? 1 : 0;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

template <typename T, int VB, int BLOCK, int ILP, bool PMUL, uint32_t NVC = 0>
inline void run_flat(T* out, const T* in, uint32_t nv, uint32_t d,
                     uint32_t total_vecs, int cap_waves, cudaStream_t stream,
                     bool rev = false) {
  constexpr uint32_t TV = static_cast<uint32_t>(BLOCK) * ILP;
  const uint32_t full_tiles = total_vecs / TV;
  uint32_t need = (total_vecs + TV - 1) / TV;
  if (need == 0) need = 1;
  const uint32_t cap = static_cast<uint32_t>(sm_count()) *
                       static_cast<uint32_t>(cap_waves) *
                       static_cast<uint32_t>(std::max(1, 2048 / BLOCK));
  const uint32_t grid = std::min(need, std::max(1u, cap));
  launch_pdl(silu_mul_flat<T, VB, BLOCK, ILP, PMUL, NVC>, dim3(grid), dim3(BLOCK),
             stream, out, in, magic_for(nv), d, total_vecs, full_tiles, rev);
}

}  // namespace fk

// ===========================================================================
//                             host dispatch
// ===========================================================================
namespace fk {

// X(id, VB, BLOCK, ILP) -- shared by the tuned probe and the auto path.
#define FK_CONFIGS(X)   \
  X(0, 32, 32, 1)       \
  X(1, 32, 64, 1)       \
  X(2, 32, 128, 1)      \
  X(3, 32, 256, 1)      \
  X(4, 32, 512, 1)      \
  X(5, 32, 1024, 1)     \
  X(6, 32, 64, 2)       \
  X(7, 32, 128, 2)      \
  X(8, 32, 256, 2)      \
  X(9, 32, 256, 4)      \
  X(10, 32, 512, 4)     \
  X(11, 32, 1024, 2)    \
  X(12, 16, 32, 1)      \
  X(13, 16, 64, 1)      \
  X(14, 16, 128, 1)     \
  X(15, 16, 256, 1)     \
  X(16, 16, 512, 1)     \
  X(17, 16, 1024, 1)    \
  X(18, 16, 64, 2)      \
  X(19, 16, 256, 2)     \
  X(20, 16, 256, 4)     \
  X(21, 16, 1024, 2)

struct Plan {
  int64_t num_tokens;
  uint32_t d;          // elements per half row
  uint32_t nv32, nv16; // vectors per output row at 32B / 16B
  uint32_t tv32, tv16; // total vectors at 32B / 16B
  bool ok32, ok16;
};

template <typename T>
inline Plan make_plan(const torch::Tensor& out, const torch::Tensor& in) {
  Plan p{};
  p.d = static_cast<uint32_t>(in.size(-1) / 2);
  p.num_tokens = in.numel() / in.size(-1);
  const uintptr_t pi = reinterpret_cast<uintptr_t>(in.const_data_ptr());
  const uintptr_t po = reinterpret_cast<uintptr_t>(out.data_ptr());
  auto fill = [&](uint32_t VB, uint32_t& nv, uint32_t& tv, bool& ok) {
    const uint32_t VE = VB / sizeof(T);
    nv = (p.d % VE == 0) ? p.d / VE : 0;
    const int64_t t64 = static_cast<int64_t>(nv) * p.num_tokens;
    // nv == 1 would need a 2^64 magic constant; punt to the scalar path.
    ok = nv > 1 && t64 < (int64_t)0xFFFFFFFFll &&
         (pi & (VB - 1)) == 0 && (po & (VB - 1)) == 0;
    tv = ok ? static_cast<uint32_t>(t64) : 0;
  };
  fill(32, p.nv32, p.tv32, p.ok32);
  fill(16, p.nv16, p.tv16, p.ok16);
  return p;
}

template <typename T>
inline void run_scalar(T* out, const T* in, uint32_t d, int64_t total,
                       cudaStream_t stream) {
  const int block = 256;
  const int64_t need = std::max<int64_t>(1, (total + block - 1) / block);
  const uint32_t grid = static_cast<uint32_t>(
      std::min<int64_t>(need, (int64_t)sm_count() * 32));
  const int shift = pow2_shift(d);
  launch_pdl(silu_mul_scalar<T>, dim3(grid), dim3(block), stream, out, in, d,
             shift >= 0 ? 0ull : magic_for(d), shift, total);
}

// ------------------------------------------------------------- tuned probe
// Dev-only entry point behind silu_and_mul_tuned(); nothing in the shipped
// path reaches it. `mode` bits: 1 = packed gate multiply, 2 = reverse
// traversal, 4 = compile-time nv, 8 = drop the PDL launch attribute. Driven
// by tools/micro.py --mode sweep|table.
template <int VB, int BLOCK, int ILP>
inline void probe_one(const Plan& p, torch::Tensor& out, torch::Tensor& in,
                      int cap_waves, int mode, cudaStream_t stream) {
  using T = __nv_bfloat16;
  T* o = reinterpret_cast<T*>(out.data_ptr());
  const T* i = reinterpret_cast<const T*>(in.const_data_ptr());
  const uint32_t nv = (VB == 32) ? p.nv32 : p.nv16;
  const uint32_t tv = (VB == 32) ? p.tv32 : p.tv16;
  TORCH_CHECK((VB == 32) ? p.ok32 : p.ok16, "config not applicable to shape");
  const bool rev = (mode & 2) != 0;
  if ((mode & 4) && VB == 16 && ILP == 1) {
    switch (nv) {
#define FK_NVC(N)                                                        \
  case N:                                                                \
    run_flat<T, 16, BLOCK, 1, true, N>(o, i, nv, p.d, tv, cap_waves,     \
                                       stream, rev);                     \
    return;
      FK_NVC(32) FK_NVC(48) FK_NVC(576) FK_NVC(1792)
#undef FK_NVC
      default:
        TORCH_CHECK(false, "no compile-time nv instantiation for nv=", nv);
    }
  }
  if (mode & 1)
    run_flat<T, VB, BLOCK, ILP, true>(o, i, nv, p.d, tv, cap_waves, stream, rev);
  else
    run_flat<T, VB, BLOCK, ILP, false>(o, i, nv, p.d, tv, cap_waves, stream, rev);
}

void silu_and_mul_tuned(torch::Tensor& out, torch::Tensor& in, int64_t cfg,
                        int64_t cap_waves, int64_t mode) {
  TORCH_CHECK(in.scalar_type() == at::kBFloat16, "tuned probe is bf16 only");
  const c10::cuda::CUDAGuard guard(in.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Plan p = make_plan<__nv_bfloat16>(out, in);
  if (p.num_tokens == 0) return;
  const int m = static_cast<int>(mode);
  const int w = static_cast<int>(cap_waves);
  pdl_enabled() = (m & 8) == 0;
  switch (cfg) {
#define FK_CASE(ID, VB, B, I)                       \
  case ID:                                          \
    probe_one<VB, B, I>(p, out, in, w, m, stream);   \
    break;
    FK_CONFIGS(FK_CASE)
#undef FK_CASE
    default:
      TORCH_CHECK(false, "bad cfg id");
  }
}

// ------------------------------------------------------ decomposition probe
// Dev-only entry point behind silu_and_mul_probe(); nothing in the shipped
// path reaches it. Driven by tools/micro.py --mods probe0..probe4.
void silu_and_mul_probe(torch::Tensor& out, torch::Tensor& in,
                        torch::Tensor& alt, int64_t kind, int64_t pdl,
                        int64_t frac, int64_t grid_override, int64_t lmode) {
  TORCH_CHECK(in.scalar_type() == at::kBFloat16, "probe is bf16 only");
  using T = __nv_bfloat16;
  const c10::cuda::CUDAGuard guard(in.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Plan p = make_plan<T>(out, in);
  if (p.num_tokens == 0) return;
  TORCH_CHECK(p.ok16, "probe needs the 16B path");
  T* o = reinterpret_cast<T*>(out.data_ptr());
  const T* i = reinterpret_cast<const T*>(in.const_data_ptr());
  const T* alt_p = reinterpret_cast<const T*>(alt.const_data_ptr());
  const int lm = (pdl != 0) ? static_cast<int>(lmode) : 4;
  constexpr int BLOCK = 256;
  // frac > 1 processes only the first 1/frac of the vectors: a continuous knob
  // on how much work sits in the timed window, to test whether the window is
  // quantized or bandwidth-limited.
  const uint32_t tv = std::max(1u, static_cast<uint32_t>(
      p.tv16 / static_cast<uint32_t>(std::max<int64_t>(1, frac))));
  const uint32_t need = (tv + BLOCK - 1) / BLOCK;
  uint32_t grid = (kind == 0) ? 1u : std::max(1u, need);
  if (grid_override > 0)
    grid = std::min(grid, static_cast<uint32_t>(grid_override));
  const unsigned long long magic = magic_for(p.nv16);
#define FK_PROBE(K)                                                        \
  case K:                                                                  \
    launch_probe(silu_mul_probe<T, 16, BLOCK, K>, dim3(grid), dim3(BLOCK),  \
                 stream, lm, o, i, alt_p, magic, p.d, tv);                  \
    break;
  switch (kind) {
    FK_PROBE(0) FK_PROBE(1) FK_PROBE(2) FK_PROBE(3) FK_PROBE(4)
    FK_PROBE(5) FK_PROBE(6) FK_PROBE(7) FK_PROBE(8) FK_PROBE(9)
    default:
      TORCH_CHECK(false, "bad probe kind");
  }
#undef FK_PROBE
}

// ---------------------------------------------------------------- auto path
// One geometry covers every shape: 16B vectors, one vector per thread, packed
// gate multiply. Only the block width adapts, and only for a reason that is
// not shape-fitted -- measured on B200, a launch whose grid is under ~8 blocks
// pays a ~2us penalty (it cannot spread over the SMs while the producer
// kernel's tail drains), so narrow the block until we have at least 8 of them.
// Sweep evidence: at [60,512] (1920 vectors) BLOCK 32/64/128/256 all land on
// the 7.1us harness floor while 512 (4 blocks) and 1024 (2 blocks) cost 9.1us;
// at [16384,28672] BLOCK 256 is the fastest of 32..1024 (measured on the real
// harness: 0.4884 ms at 256 vs 0.4904 at 128 and 0.4925 at 512).
//
// FK_AUTO_WIDE is the one knob worth re-checking on new hardware -- rebuild
// with -DFK_AUTO_WIDE=128 or 512 and diff CAND ms.
#ifndef FK_AUTO_WIDE
  #define FK_AUTO_WIDE 256
#endif

inline int auto_block(uint32_t total_vecs) {
  uint32_t want = total_vecs / 8u;
  if (want >= 256u) return FK_AUTO_WIDE;
  if (want >= 128u) return 128;
  if (want >= 64u) return 64;
  return 32;
}

inline size_t l2_bytes() {
  static const size_t n = [] {
    int dev = 0, v = 126 * 1024 * 1024;
    if (cudaGetDevice(&dev) == cudaSuccess)
      cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, dev);
    return static_cast<size_t>(v);
  }();
  return n;
}

template <typename T>
inline void dispatch_auto(torch::Tensor& out, torch::Tensor& in,
                          cudaStream_t stream) {
  const Plan p = make_plan<T>(out, in);
  if (p.num_tokens == 0) return;
  T* o = reinterpret_cast<T*>(out.data_ptr());
  const T* i = reinterpret_cast<const T*>(in.const_data_ptr());
  if (!p.ok16) {
    run_scalar<T>(o, i, p.d, p.num_tokens * (int64_t)p.d, stream);
    return;
  }
  // Walking the flat index backwards pays off only when the producer's output
  // is larger than L2: then its most recently written rows -- the high
  // addresses -- are still resident when we start. Below that the whole input
  // is cached either way and the order is irrelevant.
  const bool rev =
      static_cast<size_t>(in.numel()) * sizeof(T) > l2_bytes();
  const uint32_t tv = p.tv16;
  switch (auto_block(tv)) {
    case 512:
      run_flat<T, 16, 512, 1, true>(o, i, p.nv16, p.d, tv, 1 << 20, stream, rev);
      break;
    case 256:
      run_flat<T, 16, 256, 1, true>(o, i, p.nv16, p.d, tv, 1 << 20, stream, rev);
      break;
    case 128:
      run_flat<T, 16, 128, 1, true>(o, i, p.nv16, p.d, tv, 1 << 20, stream, rev);
      break;
    case 64:
      run_flat<T, 16, 64, 1, true>(o, i, p.nv16, p.d, tv, 1 << 20, stream, rev);
      break;
    default:
      run_flat<T, 16, 32, 1, true>(o, i, p.nv16, p.d, tv, 1 << 20, stream, rev);
      break;
  }
}

}  // namespace fk

void silu_and_mul(torch::Tensor& out, torch::Tensor& in) {
  // A zero-width last dim would divide by zero when deriving num_tokens.
  if (in.numel() == 0 || in.size(-1) == 0) return;
  fk::pdl_enabled() = true;
  const c10::cuda::CUDAGuard guard(in.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (in.scalar_type()) {
    case at::kBFloat16:
      fk::dispatch_auto<__nv_bfloat16>(out, in, stream);
      break;
    case at::kHalf:
      fk::dispatch_auto<__half>(out, in, stream);
      break;
    case at::kFloat:
      fk::dispatch_auto<float>(out, in, stream);
      break;
    default:
      TORCH_CHECK(false, "silu_and_mul: unsupported dtype ", in.scalar_type());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &silu_and_mul, "SiLU-and-Mul (CUDA)");
  m.def("silu_and_mul_tuned", &fk::silu_and_mul_tuned,
        "SiLU-and-Mul with explicit launch config (bf16 probe)");
  m.def("silu_and_mul_probe", &fk::silu_and_mul_probe,
        "launch/load/store decomposition probe (dev only)");
}
