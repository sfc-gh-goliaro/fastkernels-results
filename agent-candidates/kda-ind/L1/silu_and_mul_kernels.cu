// SiLU-and-Mul over a contiguous [..., 2d] tensor, for Blackwell (sm_100a).
//
//   d          = input.size(-1) / 2
//   out[r, c]  = silu(input[r, c]) * input[r, c + d]        c in [0, d)
//
// Rows are strided by 2*d elements, not by input.size(-1). That is the same
// truncating arithmetic the vLLM reference kernel uses, so an odd trailing
// dimension mis-strides here exactly as it does there and the two agree
// element for element instead of diverging on an input nobody captures.
//
// Two launch geometries live here and produce identical results:
//
//   flat     one flat iteration space over output vectors, grid-stride. Block
//            and grid size are then free of the problem shape, which is what
//            lets a 256-thread block cover a 1-row problem and a 16384-row one.
//            Recovering (row, col) from the flat index needs one division by
//            dvec, done with a host-computed magic multiply so no divide
//            instruction is issued.
//
//   rowwise  one block per row with an intra-block column loop. Needs no
//            division at all, at the cost of tying the grid to the row count.
//            Kept as a measured control for the flat geometry rather than as
//            dead code -- the flat kernel only ships because it wins.
//
// The activation is selectable so that the accuracy and the cost of the cheap
// form can both be measured against the accurate form on the real GPU instead
// of argued from an error bound on paper. See ActKind below.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// 256-bit (v8.u32) PTX load/store needs an SM100+ target and a CUDA 12.9+
// toolkit. Where either is missing the 32-byte access is still available -- it
// just decomposes into two 128-bit accesses -- so the width ladder stays a pure
// alignment decision and can never select an instruction the binary lacks.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDART_VERSION) && CUDART_VERSION >= 12090
  #define FK_HAS_256B_PTX 1
#else
  #define FK_HAS_256B_PTX 0
#endif

namespace fk_silu_and_mul {

// ---------------------------------------------------------------------------
// Element traits: everything is computed in fp32 through the natural packed
// pair of each dtype, so one conversion instruction covers two elements.
// ---------------------------------------------------------------------------
template <typename T>
struct PairTraits;

template <>
struct PairTraits<__nv_bfloat16> {
  using pair_t = __nv_bfloat162;
  static __device__ __forceinline__ float2 to_f2(pair_t v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ pair_t from_f2(float2 v) { return __floats2bfloat162_rn(v.x, v.y); }
  static __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16_rn(v); }
};

template <>
struct PairTraits<__half> {
  using pair_t = __half2;
  static __device__ __forceinline__ float2 to_f2(pair_t v) { return __half22float2(v); }
  static __device__ __forceinline__ pair_t from_f2(float2 v) { return __floats2half2_rn(v.x, v.y); }
  static __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from_f(float v) { return __float2half_rn(v); }
};

template <>
struct PairTraits<float> {
  using pair_t = float2;
  static __device__ __forceinline__ float2 to_f2(pair_t v) { return v; }
  static __device__ __forceinline__ pair_t from_f2(float2 v) { return v; }
  static __device__ __forceinline__ float to_f(float v) { return v; }
  static __device__ __forceinline__ float from_f(float v) { return v; }
};

// A BYTES-wide vector holding whole packed pairs. The wide load and store move
// it as raw 32-bit words; the arithmetic sees the pairs.
template <typename T, int BYTES>
struct alignas(BYTES) PackedVec {
  using pair_t = typename PairTraits<T>::pair_t;
  static constexpr int kWords = BYTES / 4;
  static constexpr int kPairs = BYTES / static_cast<int>(sizeof(pair_t));
  static_assert(kPairs >= 1, "vector must hold at least one packed pair");
  static_assert(BYTES % static_cast<int>(sizeof(pair_t)) == 0, "pair must tile the vector");
  pair_t p[kPairs];

  __device__ __forceinline__ uint32_t* words() {
    return reinterpret_cast<uint32_t*>(p);
  }
  __device__ __forceinline__ const uint32_t* words() const {
    return reinterpret_cast<const uint32_t*>(p);
  }
};

// ---------------------------------------------------------------------------
// Width-specific global access. Read-only loads take the non-coherent path.
// ---------------------------------------------------------------------------
template <int BYTES>
struct WideIO;

template <>
struct WideIO<32> {
  static __device__ __forceinline__ void load(uint32_t* w, const void* p) {
#if FK_HAS_256B_PTX
    asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]),
                   "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
                 : "l"(p));
#else
    const uint4* q = reinterpret_cast<const uint4*>(p);
    const uint4 lo = __ldg(q);
    const uint4 hi = __ldg(q + 1);
    w[0] = lo.x; w[1] = lo.y; w[2] = lo.z; w[3] = lo.w;
    w[4] = hi.x; w[5] = hi.y; w[6] = hi.z; w[7] = hi.w;
#endif
  }

  static __device__ __forceinline__ void store(void* p, const uint32_t* w) {
#if FK_HAS_256B_PTX
    asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
                 :
                 : "l"(p), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]),
                   "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7])
                 : "memory");
#else
    uint4* q = reinterpret_cast<uint4*>(p);
    q[0] = make_uint4(w[0], w[1], w[2], w[3]);
    q[1] = make_uint4(w[4], w[5], w[6], w[7]);
#endif
  }
};

template <>
struct WideIO<16> {
  static __device__ __forceinline__ void load(uint32_t* w, const void* p) {
    const uint4 r = __ldg(reinterpret_cast<const uint4*>(p));
    w[0] = r.x; w[1] = r.y; w[2] = r.z; w[3] = r.w;
  }
  static __device__ __forceinline__ void store(void* p, const uint32_t* w) {
    *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
  }
};

template <>
struct WideIO<8> {
  static __device__ __forceinline__ void load(uint32_t* w, const void* p) {
    const uint2 r = __ldg(reinterpret_cast<const uint2*>(p));
    w[0] = r.x; w[1] = r.y;
  }
  static __device__ __forceinline__ void store(void* p, const uint32_t* w) {
    *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
  }
};

template <>
struct WideIO<4> {
  static __device__ __forceinline__ void load(uint32_t* w, const void* p) {
    w[0] = __ldg(reinterpret_cast<const uint32_t*>(p));
  }
  static __device__ __forceinline__ void store(void* p, const uint32_t* w) {
    *reinterpret_cast<uint32_t*>(p) = w[0];
  }
};

// ---------------------------------------------------------------------------
// Activation.
//
// The reference kernel evaluates x / (1 + expf(-x)): a full-precision expf plus
// an IEEE div.rn.f32, which the profile shows costs about four
// transcendental-pipe slots per output element and makes the largest shape
// compute-bound rather than bandwidth-bound.
//
// sigmoid(x) = 0.5 * (1 + tanh(x/2)) turns that into a single tanh.approx.f32
// (one MUFU) plus two FMAs. The PTX ISA specifies a maximum *relative* error of
// 2^-11 for tanh.approx.f32; since |tanh| <= 1 that is also an absolute bound,
// conservatively. Measured on this device over every operand the bf16 path can
// generate -- all 65280 finite bf16 gate patterns, whose halves are exactly
// representable in fp32 -- the implied error is 7.05e-6 absolute (2^-17.1) and
// 1.48e-5 relative (2^-16.0), roughly 33x inside the specified bound. That is a
// measurement of this implementation, not an architectural guarantee, which is
// why every variant below stays reachable and is compared on the GPU against the
// reference kernel's own output rather than argued from the bound.
//
// The approximation's *relative* error on silu is not small everywhere. For
// x <~ -16 the instruction saturates to -1, 1 + tanh(x/2) underflows to zero and
// approximate silu becomes exactly 0 against a true value of ~1e-7, so relative
// error reaches 100%. Absolute error stays under 9.71e-6 across the whole gate
// range, and the benchmark's rule is atol + rtol*|ref| with atol = 1e-2 for
// bf16, so atol carries the bound in exactly the region where the relative error
// is worst. It stops carrying it once |up| is large enough for rtol*|ref| to
// dominate; the measured crossover is between |up| = 2^12 and 2^13 for both bf16
// and fp16, against |up| <~ 6 for the benchmark's standard-normal inputs.
//
// kExpRound reproduces the reference arithmetic exactly, including its rounding
// of the activated half to the storage dtype before the multiply, and is the
// bit-exact anchor the other variants are judged against.
// ---------------------------------------------------------------------------
enum ActKind : int {
  kExpRound = 0,      // expf + IEEE divide, activated half rounded (reference-exact)
  kTanhRound = 1,     // tanh.approx.f32, activated half rounded
  kTanhFused = 2,     // tanh.approx.f32, whole expression kept in fp32
  kFastExpRound = 3,  // __expf + __fdividef, activated half rounded
  // tanh.approx per lane, but redone with the accurate form on any lane whose
  // product is large enough that a single output ulp would exceed a third of the
  // benchmark's tolerance bound. Exists to measure whether the sweep criterion is
  // reachable at all; the guard magnitude is a runtime parameter.
  kTanhGuarded = 4,
  kActKindCount = 5,
};

__device__ __forceinline__ float tanh_approx(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
#else
  return tanhf(x);
#endif
}

template <int KIND>
__device__ __forceinline__ float silu(float x) {
  if constexpr (KIND == kExpRound) {
    return x / (1.0f + expf(-x));
  } else if constexpr (KIND == kFastExpRound) {
    return __fdividef(x, 1.0f + __expf(-x));
  } else {
    return 0.5f * x * (1.0f + tanh_approx(0.5f * x));
  }
}

__host__ __device__ __forceinline__ constexpr bool rounds_activated_half(int kind) {
  return kind != kTanhFused;
}

// Default guard for kTanhGuarded when the caller does not pick one. Any value at or
// below 0.744 makes every single-ulp disagreement fall inside a third of the bound;
// see tests/check_correctness.py for the derivation and the measured frontier.
constexpr float kDefaultUpGuard = 0.744f;

// bf16 and fp16 both take the cheap form. Measured over every finite gate
// pattern with |up| <= 40: worst margin 0.8439 of the bound for bf16 (a 2-ulp
// output disagreement at gate 5.9375, up 40) and 0.1399 for fp16, every element
// inside the bound, and NaN/Inf classification identical to the reference.
template <typename T>
struct DefaultActKind {
  static constexpr int value = kTanhRound;
};
template <>
struct DefaultActKind<float> {
  // fp32 is scored at atol = 1e-5, rtol = 1e-3, a thousand times tighter than
  // bf16, and the approximation measures 3.77x outside that bound on a dense
  // sweep. fp32 gets the accurate form, which is bit-exact against the reference.
  static constexpr int value = kExpRound;
};

// One lane of the guarded form. The rounding of the activated half to the storage
// dtype happens on both branches, because the reference does it and the branch must
// not change which arithmetic is being compared -- only how it is computed.
template <typename T>
__device__ __forceinline__ float guarded_silu(float gate_f, float up_f, float guard) {
  using PT = PairTraits<T>;
  float a = PT::to_f(PT::from_f(silu<kTanhRound>(gate_f)));
  if (fabsf(a * up_f) > guard) {
    a = PT::to_f(PT::from_f(silu<kExpRound>(gate_f)));
  }
  return a;
}

template <typename T, int KIND>
__device__ __forceinline__ typename PairTraits<T>::pair_t gate_mul(
    typename PairTraits<T>::pair_t gate, typename PairTraits<T>::pair_t up,
    float guard) {
  using PT = PairTraits<T>;
  float2 g = PT::to_f2(gate);
  const float2 u = PT::to_f2(up);
  if constexpr (KIND == kTanhGuarded) {
    g.x = guarded_silu<T>(g.x, u.x, guard);
    g.y = guarded_silu<T>(g.y, u.y, guard);
  } else {
    g.x = silu<KIND>(g.x);
    g.y = silu<KIND>(g.y);
    if constexpr (rounds_activated_half(KIND)) {
      g = PT::to_f2(PT::from_f2(g));
    }
  }
  g.x *= u.x;
  g.y *= u.y;
  return PT::from_f2(g);
}

template <typename T, int KIND>
__device__ __forceinline__ T gate_mul_scalar(T gate, T up, float guard) {
  using PT = PairTraits<T>;
  const float u = PT::to_f(up);
  float g;
  if constexpr (KIND == kTanhGuarded) {
    g = guarded_silu<T>(PT::to_f(gate), u, guard);
  } else {
    g = silu<KIND>(PT::to_f(gate));
    if constexpr (rounds_activated_half(KIND)) {
      g = PT::to_f(PT::from_f(g));
    }
  }
  return PT::from_f(g * u);
}

// ---------------------------------------------------------------------------
// Division by the per-row vector count without a divide instruction.
//
// (__umulhi(n, magic) + n) >> shift == n / divisor, the add-then-shift
// formulation used by ATen/cuda/detail/IntegerDivider.cuh. Valid only for
// dividends up to INT32_MAX, which is why the host refuses the 32-bit kernels
// once the largest vector offset it would form crosses that bound.
// ---------------------------------------------------------------------------
struct FastDivider {
  uint32_t magic;
  uint32_t shift;
};

__device__ __forceinline__ uint32_t fast_div(uint32_t n, FastDivider fd) {
  return (__umulhi(n, fd.magic) + n) >> fd.shift;
}

static FastDivider make_fast_divider(uint32_t divisor) {
  TORCH_CHECK(divisor >= 1, "fast divider needs a positive divisor");
  TORCH_CHECK(divisor <= 0x80000000u, "fast divider divisor exceeds 2^31");
  uint32_t shift = 0;
  while ((1ull << shift) < static_cast<uint64_t>(divisor)) ++shift;
  const uint64_t one = 1;
  const uint64_t magic =
      ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
  TORCH_CHECK(magic > 0 && magic <= 0xffffffffull, "fast divider magic overflowed");
  return FastDivider{static_cast<uint32_t>(magic), shift};
}

// ---------------------------------------------------------------------------
// Kernels.
//
// A 256-thread block at 32 or fewer registers per thread is 8 resident blocks
// and every warp slot filled, against the reference kernel's 43.75% ceiling on
// the largest shape. __launch_bounds__ states the intent so the register
// allocator is held to it rather than trusted to land there.
// ---------------------------------------------------------------------------
constexpr int kMaxBlock = 256;

// <= 0 means "one block per kMaxBlock output vectors, however many that is".
// See the launch policy below for why the measured default is uncapped.
constexpr int64_t kDefaultBlocksPerSM = -1;

template <typename T, int BYTES, int KIND>
__global__ __launch_bounds__(kMaxBlock) void flat_kernel(
    T* __restrict__ out, const T* __restrict__ in,
    uint32_t nvec, uint32_t dvec, FastDivider dvec_div, float guard) {
  using Vec = PackedVec<T, BYTES>;
  const Vec* in_v = reinterpret_cast<const Vec*>(in);
  Vec* out_v = reinterpret_cast<Vec*>(out);
  const uint32_t row_stride = dvec * 2u;
  const uint32_t stride = gridDim.x * blockDim.x;

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    const uint32_t row = fast_div(i, dvec_div);
    const uint32_t col = i - row * dvec;
    const uint32_t gate_i = row * row_stride + col;

    Vec gate, up;
    WideIO<BYTES>::load(gate.words(), in_v + gate_i);
    WideIO<BYTES>::load(up.words(), in_v + gate_i + dvec);
#pragma unroll
    for (int j = 0; j < Vec::kPairs; ++j) {
      gate.p[j] = gate_mul<T, KIND>(gate.p[j], up.p[j], guard);
    }
    WideIO<BYTES>::store(out_v + i, gate.words());
  }
}

// Same mapping in 64-bit index space, for an output whose vector offsets no
// longer fit the 32-bit magic-multiply bound. Correctness path: it pays a real
// 64-bit division per element rather than pretending the magic constant still
// applies.
template <typename T, int BYTES, int KIND>
__global__ __launch_bounds__(kMaxBlock) void flat_kernel_wide_index(
    T* __restrict__ out, const T* __restrict__ in, int64_t nvec, int64_t dvec,
    float guard) {
  using Vec = PackedVec<T, BYTES>;
  const Vec* in_v = reinterpret_cast<const Vec*>(in);
  Vec* out_v = reinterpret_cast<Vec*>(out);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < nvec; i += stride) {
    const int64_t row = i / dvec;
    const int64_t col = i - row * dvec;
    const int64_t gate_i = row * dvec * 2 + col;

    Vec gate, up;
    WideIO<BYTES>::load(gate.words(), in_v + gate_i);
    WideIO<BYTES>::load(up.words(), in_v + gate_i + dvec);
#pragma unroll
    for (int j = 0; j < Vec::kPairs; ++j) {
      gate.p[j] = gate_mul<T, KIND>(gate.p[j], up.p[j], guard);
    }
    WideIO<BYTES>::store(out_v + i, gate.words());
  }
}

// Division-free control geometry: one block per row, columns walked inside the
// block. Measured against flat_kernel on every benchmarked shape.
template <typename T, int BYTES, int KIND>
__global__ __launch_bounds__(kMaxBlock) void rowwise_kernel(
    T* __restrict__ out, const T* __restrict__ in, uint32_t dvec, float guard) {
  using Vec = PackedVec<T, BYTES>;
  const int64_t row = blockIdx.x;
  const Vec* gate_row = reinterpret_cast<const Vec*>(in) + row * static_cast<int64_t>(dvec) * 2;
  const Vec* up_row = gate_row + dvec;
  Vec* out_row = reinterpret_cast<Vec*>(out) + row * static_cast<int64_t>(dvec);

  for (uint32_t c = threadIdx.x; c < dvec; c += blockDim.x) {
    Vec gate, up;
    WideIO<BYTES>::load(gate.words(), gate_row + c);
    WideIO<BYTES>::load(up.words(), up_row + c);
#pragma unroll
    for (int j = 0; j < Vec::kPairs; ++j) {
      gate.p[j] = gate_mul<T, KIND>(gate.p[j], up.p[j], guard);
    }
    WideIO<BYTES>::store(out_row + c, gate.words());
  }
}

// Element-at-a-time fallback for trailing widths or runtime alignments no
// vector width can serve (an odd d, a deliberately misaligned view). Correct
// rather than fast, and the only path that can express an odd row stride.
template <typename T, int KIND, typename index_t>
__global__ __launch_bounds__(kMaxBlock) void scalar_kernel(
    T* __restrict__ out, const T* __restrict__ in, index_t total, index_t d,
    float guard) {
  const index_t stride = static_cast<index_t>(gridDim.x) * blockDim.x;
  for (index_t i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const index_t row = i / d;
    const index_t col = i - row * d;
    const index_t gate_i = row * d * 2 + col;
    out[i] = gate_mul_scalar<T, KIND>(in[gate_i], in[gate_i + d], guard);
  }
}

// Measurement helpers, not part of the operator.
//
// copy_kernel is the cheapest kernel that still moves the output's worth of
// bytes at each shape, so a shape's speedup can be compared against something
// reachable instead of against the harness's no-kernel floor.
template <typename T, int BYTES>
__global__ __launch_bounds__(kMaxBlock) void copy_kernel(
    T* __restrict__ out, const T* __restrict__ in, uint32_t nvec) {
  using Vec = PackedVec<T, BYTES>;
  const Vec* in_v = reinterpret_cast<const Vec*>(in);
  Vec* out_v = reinterpret_cast<Vec*>(out);
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    Vec v;
    WideIO<BYTES>::load(v.words(), in_v + i);
    WideIO<BYTES>::store(out_v + i, v.words());
  }
}

__global__ void empty_kernel() {}

// ---------------------------------------------------------------------------
// Host-side launch policy.
// ---------------------------------------------------------------------------
enum Geometry : int {
  kGeometryAuto = 0,
  kGeometryFlat = 1,
  kGeometryRowwise = 2,
  kGeometryScalar = 3,
};

// What the dispatcher decided, so a test can assert the wide path is the branch
// actually taken instead of assuming it.
struct Plan {
  int64_t n = 0;
  int64_t d = 0;
  int64_t vec_bytes = 0;   // 0 -> scalar path
  int64_t dvec = 0;
  int64_t nvec = 0;
  int64_t block = 0;
  int64_t grid = 0;
  int64_t geometry = 0;
  int64_t act_kind = 0;
  double up_guard = kDefaultUpGuard;
  bool wide_index = false;
  bool gate_aligned = false;
  bool up_aligned = false;
  bool out_aligned = false;
  int64_t magic = 0;
  int64_t shift = 0;
};

static int64_t max_vector_bytes(const torch::Tensor& input, const torch::Tensor& out,
                                int64_t d, int64_t elem_size, int64_t pair_bytes) {
  const auto in_addr = reinterpret_cast<uintptr_t>(input.const_data_ptr());
  const auto out_addr = reinterpret_cast<uintptr_t>(out.const_data_ptr());
  for (int64_t bytes : {32, 16, 8, 4}) {
    if (bytes < pair_bytes || bytes % pair_bytes != 0) continue;
    const int64_t v_elems = bytes / elem_size;
    // d divisible by the vector width covers three things at once: the gate
    // half, the +d offset to the up half, and the 2d row stride are all whole
    // numbers of vectors.
    if (d % v_elems != 0) continue;
    if (in_addr % static_cast<uintptr_t>(bytes) != 0) continue;
    if (out_addr % static_cast<uintptr_t>(bytes) != 0) continue;
    return bytes;
  }
  return 0;
}

static int64_t pick_block(int64_t work_items) {
  if (work_items >= kMaxBlock) return kMaxBlock;
  // Round up to a whole warp so no launch carries a partial warp, and never go
  // below one warp.
  const int64_t warps = (work_items + 31) / 32;
  return std::max<int64_t>(32, std::min<int64_t>(kMaxBlock, warps * 32));
}

static int64_t pick_grid(int64_t work_items, int64_t block, int sm_count,
                         int64_t blocks_per_sm) {
  const int64_t waves = (work_items + block - 1) / block;
  if (blocks_per_sm <= 0) return std::max<int64_t>(1, waves);
  const int64_t cap = static_cast<int64_t>(sm_count) * blocks_per_sm;
  return std::max<int64_t>(1, std::min<int64_t>(waves, cap));
}

#define FK_DISPATCH_DTYPE(SCALAR_TYPE, NAME, ...)                            \
  [&] {                                                                      \
    switch (SCALAR_TYPE) {                                                   \
      case at::ScalarType::BFloat16: {                                       \
        using scalar_t = __nv_bfloat16;                                      \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case at::ScalarType::Half: {                                           \
        using scalar_t = __half;                                             \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case at::ScalarType::Float: {                                          \
        using scalar_t = float;                                              \
        return __VA_ARGS__();                                                \
      }                                                                      \
      default:                                                               \
        TORCH_CHECK(false, NAME ": unsupported dtype ", SCALAR_TYPE);         \
    }                                                                        \
  }()


// The non-default activations exist to be measured, and every benchmarked shape
// runs the 32-byte path, so they are only instantiated there. Narrower widths
// carry each dtype's shipping activation only, which keeps the instantiation
// count (and the build) bounded.
#define FK_DISPATCH_ACT(BYTES_CONST, ACT_KIND, LAUNCH)                        \
  do {                                                                        \
    constexpr int kDefaultKind = DefaultActKind<scalar_t>::value;             \
    const int kind_ = (ACT_KIND) < 0 ? kDefaultKind : (ACT_KIND);            \
    if constexpr ((BYTES_CONST) == 32) {                                      \
      switch (kind_) {                                                        \
        case kExpRound: LAUNCH(kExpRound); break;                             \
        case kTanhRound: LAUNCH(kTanhRound); break;                           \
        case kTanhFused: LAUNCH(kTanhFused); break;                           \
        case kFastExpRound: LAUNCH(kFastExpRound); break;                     \
        case kTanhGuarded: LAUNCH(kTanhGuarded); break;                       \
        default: TORCH_CHECK(false, "unknown activation variant ", kind_);     \
      }                                                                       \
    } else {                                                                  \
      TORCH_CHECK(kind_ == kDefaultKind, "activation variant ", kind_,        \
                  " is only built for the 32-byte vector path");              \
      LAUNCH(kDefaultKind);                                                   \
    }                                                                         \
  } while (0)

template <typename scalar_t, int BYTES>
static void launch_vector(const Plan& p, scalar_t* out_p, const scalar_t* in_p,
                          int act_kind, cudaStream_t stream) {
  if constexpr (BYTES >= static_cast<int>(sizeof(typename PairTraits<scalar_t>::pair_t))) {
    const float guard = static_cast<float>(p.up_guard);
    const dim3 grid(static_cast<unsigned>(p.grid));
    const dim3 block(static_cast<unsigned>(p.block));
    if (p.geometry == kGeometryRowwise) {
      const auto dvec = static_cast<uint32_t>(p.dvec);
#define FK_LAUNCH(KIND)                                                       \
  rowwise_kernel<scalar_t, BYTES, KIND><<<grid, block, 0, stream>>>(out_p, in_p, dvec, guard)
      FK_DISPATCH_ACT(BYTES, act_kind, FK_LAUNCH);
#undef FK_LAUNCH
    } else if (p.wide_index) {
#define FK_LAUNCH(KIND)                                                       \
  flat_kernel_wide_index<scalar_t, BYTES, KIND>                               \
      <<<grid, block, 0, stream>>>(out_p, in_p, p.nvec, p.dvec, guard)
      FK_DISPATCH_ACT(BYTES, act_kind, FK_LAUNCH);
#undef FK_LAUNCH
    } else {
      const auto nvec = static_cast<uint32_t>(p.nvec);
      const auto dvec = static_cast<uint32_t>(p.dvec);
      const FastDivider div = FastDivider{static_cast<uint32_t>(p.magic),
                                          static_cast<uint32_t>(p.shift)};
#define FK_LAUNCH(KIND)                                                       \
  flat_kernel<scalar_t, BYTES, KIND>                                          \
      <<<grid, block, 0, stream>>>(out_p, in_p, nvec, dvec, div, guard)
      FK_DISPATCH_ACT(BYTES, act_kind, FK_LAUNCH);
#undef FK_LAUNCH
    }
  } else {
    TORCH_CHECK(false, "a ", BYTES, "-byte vector cannot hold one packed pair of this dtype");
  }
}

template <typename scalar_t>
static void launch_scalar(const Plan& p, scalar_t* out_p, const scalar_t* in_p,
                          int act_kind, cudaStream_t stream) {
  constexpr int kDefaultKind = DefaultActKind<scalar_t>::value;
  const int kind = act_kind < 0 ? kDefaultKind : act_kind;
  TORCH_CHECK(kind == kDefaultKind,
              "activation variant ", kind, " is not built for the scalar path");
  const dim3 grid(static_cast<unsigned>(p.grid));
  const dim3 block(static_cast<unsigned>(p.block));
  const int64_t total = p.n * p.d;
  const float guard = static_cast<float>(p.up_guard);
  if (p.wide_index) {
    scalar_kernel<scalar_t, kDefaultKind, int64_t>
        <<<grid, block, 0, stream>>>(out_p, in_p, total, p.d, guard);
  } else {
    scalar_kernel<scalar_t, kDefaultKind, uint32_t>
        <<<grid, block, 0, stream>>>(out_p, in_p, static_cast<uint32_t>(total),
                                     static_cast<uint32_t>(p.d), guard);
  }
}

// ---------------------------------------------------------------------------
// The operator.
// ---------------------------------------------------------------------------
struct Prepared {
  torch::Tensor input;
  torch::Tensor out;
  Plan plan;
  bool empty = false;
};

// force_wide_index: -1 selects the index width from the largest offset the
// launch would form, 0 pins the 32-bit kernels and 1 pins the 64-bit ones. The
// overrides exist so both paths can be exercised on a tensor small enough to
// allocate, and so the 32-bit bound can be shown to be a real bound rather than
// a conservative guess.
static Prepared prepare(const torch::Tensor& input_arg, int64_t geometry,
                        int64_t block_override, int64_t blocks_per_sm,
                        int64_t act_kind, int64_t force_wide_index, double up_guard,
                        int64_t force_vec_bytes) {
  TORCH_CHECK(input_arg.is_cuda(), "silu_and_mul: input must be a CUDA tensor");
  TORCH_CHECK(input_arg.dim() >= 1, "silu_and_mul: input must have at least one dimension");
  TORCH_CHECK(input_arg.scalar_type() == at::ScalarType::BFloat16 ||
                  input_arg.scalar_type() == at::ScalarType::Half ||
                  input_arg.scalar_type() == at::ScalarType::Float,
              "silu_and_mul: unsupported dtype ", input_arg.scalar_type());

  Prepared r;
  // The reference kernel reads its input linearly from storage and ignores
  // strides, which only agrees with the mathematical definition when the input
  // is contiguous. Materializing a contiguous copy keeps this operator correct
  // on inputs the reference would read wrongly; the benchmark only ever hands
  // over contiguous views, so this costs nothing there.
  r.input = input_arg.is_contiguous() ? input_arg : input_arg.contiguous();

  const int64_t last = r.input.size(-1);
  const int64_t d = last / 2;
  auto sizes = r.input.sizes().vec();
  sizes.back() = d;
  r.out = torch::empty(sizes, r.input.options());

  const int64_t n = last == 0 ? 0 : r.input.numel() / last;
  r.plan.n = n;
  r.plan.d = d;
  r.plan.act_kind = act_kind < 0 ? -1 : act_kind;
  r.plan.up_guard = up_guard > 0.0 ? up_guard : kDefaultUpGuard;
  if (n == 0 || d == 0 || r.out.numel() == 0) {
    r.empty = true;
    return r;
  }

  const int64_t elem = r.input.element_size();
  const int64_t pair_bytes = 2 * elem;
  int64_t vec_bytes = max_vector_bytes(r.input, r.out, d, elem, pair_bytes);
  // force_vec_bytes overrides the ladder, including past what the runtime alignment
  // supports. It exists only so the misaligned-access negative in
  // tests/check_negatives.py can execute the fault the width check prevents; nothing in
  // the operator's own path ever sets it.
  if (force_vec_bytes > 0) {
    TORCH_CHECK(force_vec_bytes == 32 || force_vec_bytes == 16 || force_vec_bytes == 8 ||
                    force_vec_bytes == 4,
                "force_vec_bytes must be 4, 8, 16 or 32");
    vec_bytes = force_vec_bytes;
  }
  if (geometry == kGeometryScalar) vec_bytes = 0;
  TORCH_CHECK(geometry != kGeometryRowwise || vec_bytes != 0,
              "the row-per-block geometry is only built for the vector path");
  r.plan.vec_bytes = vec_bytes;
  r.plan.geometry = geometry == kGeometryAuto
                        ? (vec_bytes == 0 ? kGeometryScalar : kGeometryFlat)
                        : geometry;

  const auto in_addr = reinterpret_cast<uintptr_t>(r.input.const_data_ptr());
  const auto out_addr = reinterpret_cast<uintptr_t>(r.out.const_data_ptr());
  const uintptr_t up_addr = in_addr + static_cast<uintptr_t>(d * elem);
  r.plan.gate_aligned = vec_bytes != 0 && in_addr % static_cast<uintptr_t>(vec_bytes) == 0;
  r.plan.up_aligned = vec_bytes != 0 && up_addr % static_cast<uintptr_t>(vec_bytes) == 0;
  r.plan.out_aligned = vec_bytes != 0 && out_addr % static_cast<uintptr_t>(vec_bytes) == 0;

  int64_t work_items;
  if (vec_bytes == 0) {
    r.plan.dvec = d;
    r.plan.nvec = n * d;
    work_items = r.plan.nvec;
    // The scalar path forms offsets up to 2*n*d elements.
    r.plan.wide_index = 2 * r.plan.nvec > INT32_MAX;
  } else {
    const int64_t v_elems = vec_bytes / elem;
    r.plan.dvec = d / v_elems;
    r.plan.nvec = n * r.plan.dvec;
    work_items = r.plan.nvec;
    // The largest vector offset the kernel forms is the last row's up half,
    // (n-1)*2*dvec + dvec + (dvec-1) = 2*nvec - 1. Bound the index width on
    // that, not on nvec.
    r.plan.wide_index = 2 * r.plan.nvec > INT32_MAX;
  }
  if (force_wide_index >= 0) r.plan.wide_index = force_wide_index != 0;
  if (vec_bytes != 0 && !r.plan.wide_index) {
    const FastDivider div = make_fast_divider(static_cast<uint32_t>(r.plan.dvec));
    r.plan.magic = div.magic;
    r.plan.shift = div.shift;
  }

  const int sm_count =
      at::cuda::getDeviceProperties(r.input.device().index())->multiProcessorCount;

  if (r.plan.geometry == kGeometryRowwise) {
    // One block per row; the block only needs to cover the columns.
    r.plan.block = block_override > 0 ? block_override : pick_block(r.plan.dvec);
    r.plan.grid = n;
  } else {
    r.plan.block = block_override > 0 ? block_override : pick_block(work_items);
    // One block per 256 output vectors, uncapped. Capping the grid at the
    // resident block capacity was the expected win -- fewer, longer-lived blocks
    // instead of tens of thousands of one-shot ones -- and measurement says the
    // opposite. On [16384,28672] a cap of 8 blocks/SM costs 4.7% (521.1 us against
    // 496.7), and the cost falls off monotonically as the cap is raised: 1.232x at
    // 8 blocks/SM, 1.266x at 32, 1.279x at 64, 1.293x uncapped.
    //
    // What that establishes is that fewer grid-stride iterations per thread is
    // better here, not why. Note a cap of 8 blocks/SM still reaches full
    // theoretical occupancy, so queued blocks cannot be adding resident warps;
    // loop and address overhead, block turnover, memory-partition behaviour and
    // tail effects are all live candidates alongside the per-thread dependency
    // chain between successive iterations. Separating them would need a capped vs
    // uncapped profile pair, which is not worth it while the shipped default is
    // the fastest measured option.
    //
    // The cap stays available through the tuned entry point, and the SM count is
    // queried rather than hard-coded, but it is off by default.
    r.plan.grid = pick_grid(work_items, r.plan.block, sm_count,
                            blocks_per_sm != 0 ? blocks_per_sm : kDefaultBlocksPerSM);
  }
  TORCH_CHECK(r.plan.block >= 1 && r.plan.block <= kMaxBlock,
              "silu_and_mul: block size ", r.plan.block, " outside [1, ", kMaxBlock, "]");
  TORCH_CHECK(r.plan.grid >= 1 && r.plan.grid <= 2147483647L,
              "silu_and_mul: grid size ", r.plan.grid, " outside [1, 2^31)");
  return r;
}

static torch::Tensor run(const torch::Tensor& input_arg, int64_t act_kind,
                         int64_t geometry, int64_t block_override,
                         int64_t blocks_per_sm, int64_t force_wide_index,
                         double up_guard, int64_t force_vec_bytes) {
  const c10::cuda::CUDAGuard device_guard(input_arg.device());
  Prepared r = prepare(input_arg, geometry, block_override, blocks_per_sm, act_kind,
                       force_wide_index, up_guard, force_vec_bytes);
  if (r.empty) return r.out;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int kind = static_cast<int>(r.plan.act_kind);

  FK_DISPATCH_DTYPE(r.input.scalar_type(), "silu_and_mul", [&] {
    auto* out_p = reinterpret_cast<scalar_t*>(r.out.mutable_data_ptr());
    const auto* in_p = reinterpret_cast<const scalar_t*>(r.input.const_data_ptr());
    switch (r.plan.vec_bytes) {
      case 32: launch_vector<scalar_t, 32>(r.plan, out_p, in_p, kind, stream); break;
      case 16: launch_vector<scalar_t, 16>(r.plan, out_p, in_p, kind, stream); break;
      case 8: launch_vector<scalar_t, 8>(r.plan, out_p, in_p, kind, stream); break;
      case 4: launch_vector<scalar_t, 4>(r.plan, out_p, in_p, kind, stream); break;
      default: launch_scalar<scalar_t>(r.plan, out_p, in_p, kind, stream); break;
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return r.out;
}

torch::Tensor silu_and_mul(const torch::Tensor& input) {
  return run(input, -1, kGeometryAuto, 0, 0, -1, kDefaultUpGuard, 0);
}

torch::Tensor silu_and_mul_tuned(const torch::Tensor& input, int64_t act_kind,
                                 int64_t geometry, int64_t block, int64_t blocks_per_sm,
                                 int64_t force_wide_index, double up_guard,
                                 int64_t force_vec_bytes) {
  return run(input, act_kind, geometry, block, blocks_per_sm, force_wide_index, up_guard,
             force_vec_bytes);
}

// ---------------------------------------------------------------------------
// Introspection and measurement entry points.
// ---------------------------------------------------------------------------
py::dict plan_for(const torch::Tensor& input, int64_t act_kind, int64_t geometry,
                  int64_t block, int64_t blocks_per_sm, int64_t force_wide_index,
                  double up_guard, int64_t force_vec_bytes) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  Prepared r = prepare(input, geometry, block, blocks_per_sm, act_kind, force_wide_index,
                       up_guard, force_vec_bytes);
  const int kind = r.plan.act_kind < 0
                       ? FK_DISPATCH_DTYPE(r.input.scalar_type(), "plan_for",
                                           [&] { return DefaultActKind<scalar_t>::value; })
                       : static_cast<int>(r.plan.act_kind);
  py::dict out;
  out["n"] = r.plan.n;
  out["d"] = r.plan.d;
  out["vec_bytes"] = r.plan.vec_bytes;
  out["dvec"] = r.plan.dvec;
  out["nvec"] = r.plan.nvec;
  out["block"] = r.plan.block;
  out["grid"] = r.plan.grid;
  out["geometry"] = r.plan.geometry;
  out["act_kind"] = static_cast<int64_t>(kind);
  out["wide_index"] = r.plan.wide_index;
  out["gate_aligned"] = r.plan.gate_aligned;
  out["up_aligned"] = r.plan.up_aligned;
  out["out_aligned"] = r.plan.out_aligned;
  out["up_guard"] = r.plan.up_guard;
  out["magic"] = r.plan.magic;
  out["shift"] = r.plan.shift;
  out["empty"] = r.empty;
  out["has_256b_ptx_host_view"] = (CUDART_VERSION >= 12090);
  return out;
}

// (magic, shift) for a divisor, so the host-side check that
// (umulhi(i, magic) + i) >> shift == i / divisor can be run in Python over the
// exact index range each shape uses.
std::vector<int64_t> fast_divider_for(int64_t divisor) {
  const FastDivider fd = make_fast_divider(static_cast<uint32_t>(divisor));
  return {static_cast<int64_t>(fd.magic), static_cast<int64_t>(fd.shift)};
}

// Registers per thread and spill bytes for the shipped kernels, so the
// occupancy argument rests on the compiler's own numbers.
py::dict kernel_attributes() {
  py::dict out;
  auto add = [&](const char* name, const void* fn) {
    cudaFuncAttributes attr{};
    const cudaError_t err = cudaFuncGetAttributes(&attr, fn);
    TORCH_CHECK(err == cudaSuccess, "cudaFuncGetAttributes(", name, "): ",
                cudaGetErrorString(err));
    py::dict entry;
    entry["num_regs"] = attr.numRegs;
    entry["local_size_bytes"] = static_cast<int64_t>(attr.localSizeBytes);
    entry["max_threads_per_block"] = attr.maxThreadsPerBlock;
    entry["shared_size_bytes"] = static_cast<int64_t>(attr.sharedSizeBytes);
    out[name] = entry;
  };
  add("flat_bf16_32B_tanh_round",
      reinterpret_cast<const void*>(&flat_kernel<__nv_bfloat16, 32, kTanhRound>));
  add("flat_bf16_32B_exp_round",
      reinterpret_cast<const void*>(&flat_kernel<__nv_bfloat16, 32, kExpRound>));
  add("flat_bf16_32B_tanh_fused",
      reinterpret_cast<const void*>(&flat_kernel<__nv_bfloat16, 32, kTanhFused>));
  add("rowwise_bf16_32B_tanh_round",
      reinterpret_cast<const void*>(&rowwise_kernel<__nv_bfloat16, 32, kTanhRound>));
  return out;
}

// Floor references. copy_out moves exactly the output's bytes with no
// arithmetic; launch_only allocates the output and launches an empty kernel.
torch::Tensor copy_out(const torch::Tensor& input) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  Prepared r = prepare(input, kGeometryFlat, 0, 0, -1, -1, kDefaultUpGuard, 0);
  if (r.empty) return r.out;
  TORCH_CHECK(r.plan.vec_bytes == 32 && !r.plan.wide_index,
              "copy_out is a measurement floor for the 32-byte 32-bit-index path only");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(r.plan.grid));
  const dim3 block(static_cast<unsigned>(r.plan.block));
  const auto nvec = static_cast<uint32_t>(r.plan.nvec);
  FK_DISPATCH_DTYPE(r.input.scalar_type(), "copy_out", [&] {
    copy_kernel<scalar_t, 32><<<grid, block, 0, stream>>>(
        reinterpret_cast<scalar_t*>(r.out.mutable_data_ptr()),
        reinterpret_cast<const scalar_t*>(r.input.const_data_ptr()), nvec);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return r.out;
}

torch::Tensor launch_only(const torch::Tensor& input) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const int64_t last = input.size(-1);
  auto sizes = input.sizes().vec();
  sizes.back() = last / 2;
  torch::Tensor out = torch::empty(sizes, input.options());
  empty_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fk_silu_and_mul

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &fk_silu_and_mul::silu_and_mul,
        "SiLU-and-Mul, flat vectorized grid-stride (CUDA)");
  m.def("silu_and_mul_tuned", &fk_silu_and_mul::silu_and_mul_tuned,
        "SiLU-and-Mul with an explicit activation variant and launch geometry",
        py::arg("input"), py::arg("act_kind") = -1, py::arg("geometry") = 0,
        py::arg("block") = 0, py::arg("blocks_per_sm") = 0,
        py::arg("force_wide_index") = -1, py::arg("up_guard") = fk_silu_and_mul::kDefaultUpGuard,
        py::arg("force_vec_bytes") = 0);
  m.def("plan_for", &fk_silu_and_mul::plan_for,
        "The launch decision the dispatcher would make for this input",
        py::arg("input"), py::arg("act_kind") = -1, py::arg("geometry") = 0,
        py::arg("block") = 0, py::arg("blocks_per_sm") = 0,
        py::arg("force_wide_index") = -1, py::arg("up_guard") = fk_silu_and_mul::kDefaultUpGuard,
        py::arg("force_vec_bytes") = 0);
  m.def("fast_divider_for", &fk_silu_and_mul::fast_divider_for,
        "(magic, shift) for the host-computed magic-multiply divider");
  m.def("kernel_attributes", &fk_silu_and_mul::kernel_attributes,
        "Registers per thread and spill bytes for the shipped kernels");
  m.def("copy_out", &fk_silu_and_mul::copy_out,
        "Measurement floor: move the output's bytes with no arithmetic");
  m.def("launch_only", &fk_silu_and_mul::launch_only,
        "Measurement floor: allocate the output and launch an empty kernel");
  m.attr("ACT_EXP_ROUND") = static_cast<int>(fk_silu_and_mul::kExpRound);
  m.attr("ACT_TANH_ROUND") = static_cast<int>(fk_silu_and_mul::kTanhRound);
  m.attr("ACT_TANH_FUSED") = static_cast<int>(fk_silu_and_mul::kTanhFused);
  m.attr("ACT_FASTEXP_ROUND") = static_cast<int>(fk_silu_and_mul::kFastExpRound);
  m.attr("ACT_TANH_GUARDED") = static_cast<int>(fk_silu_and_mul::kTanhGuarded);
  m.attr("DEFAULT_UP_GUARD") = fk_silu_and_mul::kDefaultUpGuard;
  m.attr("GEOMETRY_AUTO") = static_cast<int>(fk_silu_and_mul::kGeometryAuto);
  m.attr("GEOMETRY_FLAT") = static_cast<int>(fk_silu_and_mul::kGeometryFlat);
  m.attr("GEOMETRY_ROWWISE") = static_cast<int>(fk_silu_and_mul::kGeometryRowwise);
  m.attr("GEOMETRY_SCALAR") = static_cast<int>(fk_silu_and_mul::kGeometryScalar);
}
