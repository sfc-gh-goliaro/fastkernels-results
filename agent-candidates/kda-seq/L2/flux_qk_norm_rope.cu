// Fused qk-norm + RoPE, rewriting the q and k regions of a packed QKV buffer
// in place.
//
// The baseline reaches the same values in five launches -- RMSNorm on q,
// RMSNorm on k, an fp64->activation-dtype cast of cos and sin, rotary on q,
// rotary on k -- reading and writing the q|k half of every row about five times
// over. This kernel touches it once: 113 MB of read+write at N=4608 against
// roughly 450 MB today.
//
// The numerics are transcribed from the two frozen L1 kernels this replaces
// (`rms_norm_kernels.cu`'s `rms_norm_warp_rows` and `diffusion_rope.py`'s
// `rotary_token`) rather than merely being equivalent to them. Same pairwise
// sum-of-squares, same `__shfl_xor_sync` butterfly order, same
// `(x * scale) * w` association with one rounding, same intermediate rounding of
// the normalized value back to the activation dtype before the rotation, same
// fp32 rotation packed with a single round-to-nearest-even. Being *more*
// accurate than the reference would be a divergence like any other: the fast and
// fallback paths have to agree.
//
// Three structural properties make the fusion work:
//
//   * head_dim 128 in a 2-byte dtype is 16 lanes of one 128-bit access, so a
//     row is a sub-warp segment and its reduction is a shuffle butterfly with no
//     shared memory and no barrier. This is the geometry the frozen norm's
//     launcher independently arrives at, which is what makes bit-exactness
//     reachable rather than merely hoped for.
//   * The interleaved layout puts both partners of a rotary pair adjacent in
//     memory, so one 128-bit access is four whole pairs and every partner a lane
//     needs is already in its own registers. That is what lets the rewrite be in
//     place without a barrier between the read and the write.
//   * A CTA is pinned to one token by the launch geometry (blockIdx.y is the
//     token), so the token's 1 KiB of fp64 coefficients is loaded and rounded
//     once per CTA into shared memory and reused by all of the CTA's rows.
//     Without that staging each lane would reload and re-round its own four cos
//     and four sin per row: 32 bytes of coefficient per 16 bytes of activation,
//     and at N=4608 some 28.3 million fp64->bf16 conversions per launch
//     (4608 tokens x 48 rows x 16 lanes x 8) against 1.77 million staged
//     (3 CTAs per token x 4608 x 128) -- a 16x reduction, since 48 rows at 16
//     rows per block is three CTAs per token rather than one. Cache hints cannot
//     remove a conversion; staging can.
//
// Layouts this file does not claim are reported in band -- `false` from the entry
// points, before any launch -- so the caller routes them to its reference path
// with nothing written. `qk_norm_rope_claims` exists so a caller needing two
// launches (the joint variant, whose two streams use different norm weights and
// different rotary offsets) can validate both *before* issuing either, because a
// second launch declined after the first has run would leave a half-rewritten
// buffer. There is deliberately no exception caught around a launch: a launch
// that starts has to be a launch that is correct.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace flux_qk {

constexpr int kWarp = 32;
// One 128-bit access carries 8 contiguous 16-bit elements, which in the
// interleaved layout is 4 whole rotary pairs.
constexpr int kVec = 8;
constexpr int kPairsPerVec = kVec / 2;
constexpr int kMaxLanes = 32;
constexpr int kBlock = 256;
constexpr int64_t kMaxGridY = 65535;

// ---------------------------------------------------------------------------
// Element traits. Only the two 16-bit floating types get a path: they are the
// only activation dtypes this operator's traffic uses, and the ones where the
// packed conversions below exist.
// ---------------------------------------------------------------------------

template <typename T>
struct ElemTraits;

template <>
struct ElemTraits<__nv_bfloat16> {
  using pack_t = __nv_bfloat162;
  __device__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static float2 to_f2(pack_t x) { return __bfloat1622float2(x); }
  __device__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16(x); }
  __device__ static pack_t pack_f2(float2 x) { return __float22bfloat162_rn(x); }
};

template <>
struct ElemTraits<__half> {
  using pack_t = __half2;
  __device__ static float to_f(__half x) { return __half2float(x); }
  __device__ static float2 to_f2(pack_t x) { return __half22float2(x); }
  __device__ static __half from_f(float x) { return __float2half_rn(x); }
  __device__ static pack_t pack_f2(float2 x) { return __float22half2_rn(x); }
};

// A kVec-wide bundle, aligned so the compiler emits one 128-bit access for it.
template <typename T>
struct alignas(sizeof(T) * kVec) Bundle {
  T e[kVec];
};

// Sum of squares in fp32, accumulated pairwise. The pair grouping and the
// `z.x * z.x + z.y * z.y` shape are the frozen `sum_squares`, not an equivalent
// rewriting of it: a flat elementwise loop sums in a different order and lands on
// a different last bit.
template <typename T>
__device__ __forceinline__ float sum_squares(const Bundle<T>& b) {
  using Tr = ElemTraits<T>;
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < kVec; i += 2) {
    float2 z = Tr::to_f2(typename Tr::pack_t{b.e[i], b.e[i + 1]});
    acc += z.x * z.x + z.y * z.y;
  }
  return acc;
}

// ---------------------------------------------------------------------------
// Coefficient rounding
//
// The reference is `cos.to(query.dtype)` -- a torch cast -- followed by the
// frozen rotary reading the result. So the value the rotation sees is the fp64
// coefficient rounded *through* the activation dtype, and reading the wider
// input straight into fp32 would make this kernel more accurate than the thing
// it is compared against.
//
// fp64 goes through fp32 on the way, because that is what torch does:
// `static_cast<BFloat16>(double)` converts to float first and then rounds to
// bf16, so the double rounding is part of the reference, not an artifact here.
// ---------------------------------------------------------------------------

template <typename T, typename C>
struct CoefCast;

template <typename T>
struct CoefCast<T, double> {
  __device__ static T to_elem(double x) {
    return ElemTraits<T>::from_f(__double2float_rn(x));
  }
};

template <typename T>
struct CoefCast<T, float> {
  __device__ static T to_elem(float x) { return ElemTraits<T>::from_f(x); }
};

template <typename T>
struct CoefCast<T, __nv_bfloat16> {
  __device__ static T to_elem(__nv_bfloat16 x) {
    return ElemTraits<T>::from_f(__bfloat162float(x));
  }
};

template <typename T>
struct CoefCast<T, __half> {
  __device__ static T to_elem(__half x) {
    return ElemTraits<T>::from_f(__half2float(x));
  }
};

// ---------------------------------------------------------------------------
// The kernel
//
// Grid is (row-chunks within a token, tokens). Pinning a CTA to one token is
// what makes the staged coefficients correct and is also why `is_q`, the weight
// bundle and the lane index are all loop-invariant: a thread keeps the same
// (head, lane) for every token it visits, so the weight is loaded once for the
// lifetime of the thread rather than once per row.
//
// LANES and WALK_TOKENS are template parameters rather than arguments because the
// first profile of this kernel showed both costing real time as runtime values.
// With a runtime `lanes`, `threadIdx.x / lanes` is an integer division (it showed
// up as math_pipe_throttle on that line) and the reduction butterfly is a loop
// nvcc cannot unroll (branch_resolving, 258 pcsamp samples). WALK_TOKENS exists
// because `grid_y = min(num_tokens, 65535)`, so for every shape this operator
// sees the token loop runs exactly once -- and the trailing barrier that protects
// the next iteration was the second-largest stall in the kernel (543 samples)
// while protecting nothing.
// ---------------------------------------------------------------------------

// One token's coefficients, rounded to the activation dtype once for the whole
// CTA. This is the difference between 590 thousand conversions per launch and
// ~28 million: without it every lane re-rounds its own four cos and four sin for
// every row it handles, and no cache hint can remove a conversion.
template <typename T, typename C>
__device__ __forceinline__ void stage_coefficients(
    T* __restrict__ s_cos, T* __restrict__ s_sin, const C* __restrict__ cos,
    const C* __restrict__ sin, int64_t coef_row, int half_dim) {
  for (int i = static_cast<int>(threadIdx.x); i < half_dim;
       i += static_cast<int>(blockDim.x)) {
    s_cos[i] = CoefCast<T, C>::to_elem(cos[coef_row + i]);
    s_sin[i] = CoefCast<T, C>::to_elem(sin[coef_row + i]);
  }
}

// Normalize and rotate one (token, head) row in place, from the staged
// coefficients. Every expression here is transcribed from the frozen kernels;
// see the file header.
template <typename T, int LANES>
__device__ __forceinline__ void rewrite_row(
    T* __restrict__ row, const Bundle<T>& w, const T* __restrict__ s_cos,
    const T* __restrict__ s_sin, int lane, unsigned mask, int head_dim,
    float eps) {
  using B = Bundle<T>;
  using Tr = ElemTraits<T>;

  const B x = reinterpret_cast<const B*>(row)[lane];

  // -- RMSNorm, transcribed from the frozen warp-row kernel --
  float acc = sum_squares(x);
#pragma unroll
  for (int off = LANES >> 1; off; off >>= 1) {
    acc += __shfl_xor_sync(mask, acc, off);
  }
  const float scale = rsqrtf(acc / head_dim + eps);

  B y;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    // (x * scale) * w, in that association, rounded exactly once.
    y.e[i] = Tr::from_f(Tr::to_f(x.e[i]) * scale * Tr::to_f(w.e[i]));
  }

  // -- Interleaved rotation, transcribed from the frozen rotary --
  //
  // `y` is read back out of the activation dtype even though the value is
  // already in a register: the reference rounds here, and skipping it would
  // perturb q and k by ~0.2% relative in the *accurate* direction, which is
  // still a divergence from the thing being compared against.
  const int pair_base = lane * kPairsPerVec;
  B out;
#pragma unroll
  for (int j = 0; j < kPairsPerVec; ++j) {
    const float2 e = Tr::to_f2(typename Tr::pack_t{y.e[2 * j], y.e[2 * j + 1]});
    const float cf = Tr::to_f(s_cos[pair_base + j]);
    const float sf = Tr::to_f(s_sin[pair_base + j]);
    float2 r;
    // fp32 throughout: the coefficients arrive as unbounded values, not as true
    // cosines and sines, so the subtraction genuinely cancels.
    r.x = e.x * cf - e.y * sf;
    r.y = e.y * cf + e.x * sf;
    const typename Tr::pack_t p = Tr::pack_f2(r);
    out.e[2 * j] = p.x;
    out.e[2 * j + 1] = p.y;
  }
  reinterpret_cast<B*>(row)[lane] = out;
}

template <typename T, typename C, int LANES, bool WALK_TOKENS>
__global__ __launch_bounds__(kBlock) void qk_norm_rope_kernel(
    T* __restrict__ qkv,
    const T* __restrict__ w_q,
    const T* __restrict__ w_k,
    const C* __restrict__ cos,
    const C* __restrict__ sin,
    const int64_t row_stride,
    const int64_t num_tokens,
    const int64_t pos_offset,
    const int num_heads,
    const int rows_per_token,   // num_heads + num_kv_heads
    const int head_dim,
    const int half_dim,         // head_dim / 2
    const float eps) {
  using B = Bundle<T>;

  // Staged cos then sin, already rounded to the activation dtype: head_dim
  // elements, 256 B at head_dim=128.
  extern __shared__ __align__(16) char smem_raw[];
  T* s_cos = reinterpret_cast<T*>(smem_raw);
  T* s_sin = s_cos + half_dim;

  constexpr int kRowsPerBlock = kBlock / LANES;
  const int lane = threadIdx.x & (LANES - 1);
  const int row_in_token =
      blockIdx.x * kRowsPerBlock + static_cast<int>(threadIdx.x) / LANES;
  // Uniform across a row's lane segment, so the masked shuffles below stay
  // converged even when a sibling segment in the same warp is idle.
  const bool live = row_in_token < rows_per_token;

  const unsigned mask =
      (LANES == kWarp)
          ? 0xffffffffu
          : (((1u << LANES) - 1u)
             << ((threadIdx.x & (kWarp - 1)) & ~(LANES - 1)));

  // Which region this thread's row lives in, and therefore which norm weight.
  // Loop-invariant: the row's head never changes, only its token.
  const bool is_q = row_in_token < num_heads;
  const int head_in_region = is_q ? row_in_token : row_in_token - num_heads;
  const int64_t col_base =
      (is_q ? 0 : static_cast<int64_t>(num_heads) * head_dim) +
      static_cast<int64_t>(head_in_region) * head_dim;

  B w;
  if (live) {
    w = reinterpret_cast<const B*>(is_q ? w_q : w_k)[lane];
  }

  if (WALK_TOKENS) {
#pragma unroll 1
    for (int64_t token = blockIdx.y; token < num_tokens; token += gridDim.y) {
      stage_coefficients<T, C>(s_cos, s_sin, cos, sin,
                               (pos_offset + token) * half_dim, half_dim);
      __syncthreads();
      if (live) {
        rewrite_row<T, LANES>(qkv + token * row_stride + col_base, w, s_cos,
                              s_sin, lane, mask, head_dim, eps);
      }
      // Keeps the next iteration's staging from overwriting coefficients a
      // lagging warp is still reading.
      __syncthreads();
    }
  } else {
    // The grid covers every token, so the launch index *is* the token and there
    // is no next iteration to protect: one barrier, not two.
    const int64_t token = blockIdx.y;
    stage_coefficients<T, C>(s_cos, s_sin, cos, sin,
                             (pos_offset + token) * half_dim, half_dim);
    __syncthreads();
    if (live) {
      rewrite_row<T, LANES>(qkv + token * row_stride + col_base, w, s_cos,
                            s_sin, lane, mask, head_dim, eps);
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side predicates
//
// Every one of these runs before any launch and reports in band. A predicate
// evaluated after a launch, or a decline signalled by an exception, would leave
// the caller unable to tell "declined" from "partially rewritten".
// ---------------------------------------------------------------------------

bool is_pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

bool aligned_to(const void* p, std::uintptr_t bytes) {
  return (reinterpret_cast<std::uintptr_t>(p) % bytes) == 0;
}

// Byte extent a strided 2-D tensor spans, from its base to its last element.
int64_t byte_extent(const at::Tensor& t, int64_t row_stride, int64_t width) {
  const int64_t elems = (t.size(0) - 1) * row_stride + width;
  return elems * t.element_size();
}

// Whether two tensors can touch the same byte. The kernel takes every pointer as
// `__restrict__` and rewrites `qkv` while reading the others, so an alias is
// undefined behaviour and a race at once -- and a view contrived to alias can
// satisfy every shape, dtype, contiguity and alignment test above.
bool overlaps(const void* a_base, int64_t a_bytes, const at::Tensor& b) {
  const char* pa = static_cast<const char*>(a_base);
  const char* pb = static_cast<const char*>(b.const_data_ptr());
  return pa < pb + b.numel() * b.element_size() && pb < pa + a_bytes;
}

bool elem_dtype_ok(const at::Tensor& t) {
  return t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf;
}

bool coef_dtype_ok(const at::Tensor& t) {
  return t.scalar_type() == at::kDouble || t.scalar_type() == at::kFloat ||
         t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf;
}

// A weight indexable as a plain [head_dim] bundle array on the device. Same
// conditions the frozen norm's `weight_ok` imposes, for the same reason.
bool weight_ok(const at::Tensor& w, const at::Tensor& qkv, int64_t head_dim) {
  return w.defined() && w.is_cuda() && w.is_contiguous() && w.dim() == 1 &&
         w.numel() == head_dim && w.scalar_type() == qkv.scalar_type() &&
         w.device() == qkv.device() && aligned_to(w.const_data_ptr(), 16);
}

// Everything the launch needs, once the predicates have accepted it.
struct Shape {
  int64_t num_tokens;
  int64_t row_stride;
  int head_dim;
  int lanes;
  int half_dim;
  int rows_per_token;
};

bool check(const at::Tensor& qkv, const at::Tensor& w_q, const at::Tensor& w_k,
           const at::Tensor& cos, const at::Tensor& sin, int64_t num_heads,
           int64_t num_kv_heads, int64_t pos_offset, Shape* out) {
  if (!qkv.is_cuda() || !elem_dtype_ok(qkv)) return false;
  // 2-D only: the caller hands over a flat [tokens, packed_width] view, and
  // collapsing leading dimensions here would mean reproducing the frozen norm's
  // stride analysis for no shape this operator actually sees.
  if (qkv.dim() != 2) return false;
  if (qkv.stride(-1) != 1) return false;

  const int64_t num_tokens = qkv.size(0);
  const int64_t width = qkv.size(1);
  if (num_tokens <= 0 || width <= 0) return false;  // an empty grid is illegal

  if (num_heads <= 0 || num_kv_heads <= 0) return false;
  if (num_heads > std::numeric_limits<int>::max() ||
      num_kv_heads > std::numeric_limits<int>::max()) {
    return false;
  }
  if (num_heads + 2 * num_kv_heads <= 0) return false;
  // `rows_per_token` is narrowed to int for the kernel, so the *sum* needs the
  // bound, not just each term.
  if (num_heads + num_kv_heads > std::numeric_limits<int>::max()) return false;
  if (width % (num_heads + 2 * num_kv_heads) != 0) return false;
  const int64_t head_dim = width / (num_heads + 2 * num_kv_heads);
  // The q, k and v regions must tile the row exactly: a width that merely
  // divides evenly would still admit a head_dim the caller did not mean.
  if (num_heads * head_dim + 2 * num_kv_heads * head_dim != width) return false;
  if (head_dim <= 0 || head_dim > std::numeric_limits<int>::max()) return false;
  if (head_dim % kVec != 0) return false;

  const int64_t lanes = head_dim / kVec;
  // A power-of-two lane count no wider than a warp is what makes the row a
  // sub-warp segment with a shuffle-only reduction and a mask rather than a
  // modulo.
  if (!is_pow2(lanes) || lanes > kMaxLanes) return false;
  if (head_dim % 2 != 0) return false;

  const int64_t row_stride = qkv.stride(0);
  // Every row must start on a bundle boundary, not merely the first one.
  if (row_stride % kVec != 0) return false;
  // Rows must not overlap. `stride(0) == 0` (an expanded view) or any stride
  // below the row width passes every other test here and would have distinct
  // token CTAs rewriting the same storage concurrently.
  if (row_stride < width) return false;
  if (!aligned_to(qkv.const_data_ptr(), 16)) return false;

  if (!weight_ok(w_q, qkv, head_dim)) return false;
  if (!weight_ok(w_k, qkv, head_dim)) return false;

  if (!cos.defined() || !sin.defined()) return false;
  if (!cos.is_cuda() || !sin.is_cuda()) return false;
  if (cos.device() != qkv.device() || sin.device() != qkv.device()) return false;
  if (!coef_dtype_ok(cos) || cos.scalar_type() != sin.scalar_type()) return false;
  if (!cos.is_contiguous() || !sin.is_contiguous()) return false;
  if (cos.dim() != 2 || sin.dim() != 2) return false;
  if (cos.sizes() != sin.sizes()) return false;
  // Full rotary only: the vectorized path rotates the whole head, so there is no
  // untouched tail to copy through. Compared by division rather than by doubling
  // cos.size(1), which is caller-supplied and would overflow the signed multiply.
  if (cos.size(1) != head_dim / 2) return false;
  if (!aligned_to(cos.const_data_ptr(), 8) ||
      !aligned_to(sin.const_data_ptr(), 8)) {
    return false;
  }

  // Coefficients are indexed by absolute position, so the window this call reads
  // has to lie inside them.
  if (pos_offset < 0) return false;
  if (pos_offset > cos.size(0) - num_tokens) return false;

  // No input may alias the buffer being rewritten.
  const void* qkv_base = qkv.const_data_ptr();
  const int64_t qkv_bytes = byte_extent(qkv, row_stride, width);
  if (overlaps(qkv_base, qkv_bytes, w_q) || overlaps(qkv_base, qkv_bytes, w_k) ||
      overlaps(qkv_base, qkv_bytes, cos) || overlaps(qkv_base, qkv_bytes, sin)) {
    return false;
  }

  if (out != nullptr) {
    out->num_tokens = num_tokens;
    out->row_stride = row_stride;
    out->head_dim = static_cast<int>(head_dim);
    out->lanes = static_cast<int>(lanes);
    out->half_dim = static_cast<int>(head_dim / 2);
    out->rows_per_token = static_cast<int>(num_heads + num_kv_heads);
  }
  return true;
}

template <typename T, typename C, int LANES, bool WALK_TOKENS>
void launch_tiled(const at::Tensor& qkv, const at::Tensor& w_q,
                  const at::Tensor& w_k, const at::Tensor& cos,
                  const at::Tensor& sin, const Shape& s, int64_t num_heads,
                  int64_t pos_offset, double eps, dim3 grid, size_t smem,
                  cudaStream_t stream) {
  qk_norm_rope_kernel<T, C, LANES, WALK_TOKENS><<<grid, kBlock, smem, stream>>>(
      reinterpret_cast<T*>(qkv.data_ptr()),
      reinterpret_cast<const T*>(w_q.const_data_ptr()),
      reinterpret_cast<const T*>(w_k.const_data_ptr()),
      reinterpret_cast<const C*>(cos.const_data_ptr()),
      reinterpret_cast<const C*>(sin.const_data_ptr()),
      s.row_stride, s.num_tokens, pos_offset, static_cast<int>(num_heads),
      s.rows_per_token, s.head_dim, s.half_dim, static_cast<float>(eps));
}

// `lanes` is a power of two <= 32 by predicate, so the instantiation set is
// closed at six widths x two token-walk flavors.
#define DISPATCH_LANES(L, ...)          \
  switch (L) {                           \
    case 32: {                           \
      constexpr int LANES = 32;           \
      __VA_ARGS__;                        \
      break;                              \
    }                                     \
    case 16: {                            \
      constexpr int LANES = 16;           \
      __VA_ARGS__;                        \
      break;                              \
    }                                     \
    case 8: {                             \
      constexpr int LANES = 8;            \
      __VA_ARGS__;                        \
      break;                              \
    }                                     \
    case 4: {                             \
      constexpr int LANES = 4;            \
      __VA_ARGS__;                        \
      break;                              \
    }                                     \
    case 2: {                             \
      constexpr int LANES = 2;            \
      __VA_ARGS__;                        \
      break;                              \
    }                                     \
    default: {                            \
      constexpr int LANES = 1;            \
      __VA_ARGS__;                        \
      break;                              \
    }                                     \
  }

template <typename T, typename C>
void launch(const at::Tensor& qkv, const at::Tensor& w_q, const at::Tensor& w_k,
            const at::Tensor& cos, const at::Tensor& sin, const Shape& s,
            int64_t num_heads, int64_t pos_offset, double eps,
            cudaStream_t stream) {
  const int rows_per_block = kBlock / s.lanes;
  const int64_t grid_x = (s.rows_per_token + rows_per_block - 1) / rows_per_block;
  const int64_t grid_y = std::min<int64_t>(s.num_tokens, kMaxGridY);
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
  const size_t smem = static_cast<size_t>(s.head_dim) * sizeof(T);
  // Only a token count past the launch-dimension limit needs the walking
  // flavor, and no shape this operator sees comes close.
  const bool walk = grid_y < s.num_tokens;

  if (walk) {
    DISPATCH_LANES(s.lanes, {
      launch_tiled<T, C, LANES, true>(qkv, w_q, w_k, cos, sin, s, num_heads,
                                      pos_offset, eps, grid, smem, stream);
    });
  } else {
    DISPATCH_LANES(s.lanes, {
      launch_tiled<T, C, LANES, false>(qkv, w_q, w_k, cos, sin, s, num_heads,
                                       pos_offset, eps, grid, smem, stream);
    });
  }
}

// Two-level dispatch over (activation dtype, coefficient dtype). The matrix is
// closed and small: 2 x 4 = 8 instantiations in one translation unit.
template <typename T>
void dispatch_coef(const at::Tensor& qkv, const at::Tensor& w_q,
                   const at::Tensor& w_k, const at::Tensor& cos,
                   const at::Tensor& sin, const Shape& s, int64_t num_heads,
                   int64_t pos_offset, double eps, cudaStream_t stream) {
  switch (cos.scalar_type()) {
    case at::kDouble:
      launch<T, double>(qkv, w_q, w_k, cos, sin, s, num_heads, pos_offset, eps,
                        stream);
      break;
    case at::kFloat:
      launch<T, float>(qkv, w_q, w_k, cos, sin, s, num_heads, pos_offset, eps,
                       stream);
      break;
    case at::kBFloat16:
      launch<T, __nv_bfloat16>(qkv, w_q, w_k, cos, sin, s, num_heads,
                               pos_offset, eps, stream);
      break;
    default:
      launch<T, __half>(qkv, w_q, w_k, cos, sin, s, num_heads, pos_offset, eps,
                        stream);
      break;
  }
}

}  // namespace flux_qk

// Predicates only: no launch, nothing written. A caller that needs two launches
// over one buffer validates both through this first, so a decline on the second
// cannot leave the first launch's rewrite behind.
bool qk_norm_rope_claims(const at::Tensor& qkv, const at::Tensor& w_q,
                         const at::Tensor& w_k, const at::Tensor& cos,
                         const at::Tensor& sin, int64_t num_heads,
                         int64_t num_kv_heads, int64_t pos_offset) {
  return flux_qk::check(qkv, w_q, w_k, cos, sin, num_heads, num_kv_heads,
                        pos_offset, nullptr);
}

// Rewrites the q and k regions of `qkv` in place. Returns false -- having
// written nothing -- for any layout the kernel does not claim.
bool qk_norm_rope(const at::Tensor& qkv, const at::Tensor& w_q,
                  const at::Tensor& w_k, const at::Tensor& cos,
                  const at::Tensor& sin, int64_t num_heads,
                  int64_t num_kv_heads, int64_t pos_offset, double eps) {
  flux_qk::Shape s;
  if (!flux_qk::check(qkv, w_q, w_k, cos, sin, num_heads, num_kv_heads,
                      pos_offset, &s)) {
    return false;
  }

  const c10::cuda::CUDAGuard device_guard(qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (qkv.scalar_type() == at::kBFloat16) {
    flux_qk::dispatch_coef<__nv_bfloat16>(qkv, w_q, w_k, cos, sin, s, num_heads,
                                          pos_offset, eps, stream);
  } else {
    flux_qk::dispatch_coef<__half>(qkv, w_q, w_k, cos, sin, s, num_heads,
                                   pos_offset, eps, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_norm_rope", &qk_norm_rope,
        "Fused RMSNorm + interleaved RoPE rewriting the q|k regions of a packed "
        "QKV buffer in place; returns false, having written nothing, for any "
        "layout the kernel does not claim.",
        py::arg("qkv"), py::arg("w_q"), py::arg("w_k"), py::arg("cos"),
        py::arg("sin"), py::arg("num_heads"), py::arg("num_kv_heads"),
        py::arg("pos_offset"), py::arg("eps"));
  m.def("qk_norm_rope_claims", &qk_norm_rope_claims,
        "Whether qk_norm_rope would claim this layout. Evaluates the same "
        "predicates without launching, so a caller needing several launches over "
        "one buffer can validate all of them before issuing any.",
        py::arg("qkv"), py::arg("w_q"), py::arg("w_k"), py::arg("cos"),
        py::arg("sin"), py::arg("num_heads"), py::arg("num_kv_heads"),
        py::arg("pos_offset"));
}
