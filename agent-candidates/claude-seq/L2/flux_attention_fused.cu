// Fused q/k RMSNorm + interleaved (GPT-J) RoPE, in place on a packed QKV buffer.
//
// The FLUX attention prologue is, per call, a fused-QKV GEMM followed by four
// memory-bound passes over the q and k halves of its output (q-norm, k-norm,
// q-rope, k-rope) -- plus, in the dual-stream blocks, three `cat`s that splice
// the text stream in front of the image stream.  Every one of those passes reads
// and writes the same ~57 MB of q/k, so the prologue costs 4-8x the HBM traffic
// the work actually needs.
//
// This kernel does all of it in one pass, in place:
//
//   * The caller hands us the *joint* QKV buffer with both streams already in
//     their final row order (the two GEMMs write disjoint row ranges of one
//     allocation via `addmm(out=)`), so the concatenation disappears entirely.
//   * q and k occupy `[0, (nq+nk)*head_dim)` of every row -- one contiguous run
//     of `nq+nk` head-sized segments -- so the whole prologue is a single
//     grid-over-rows kernel.  v is never touched.
//   * One block owns one row.  A 16-lane group owns one head segment
//     (head_dim=128 -> 8 bf16 per lane, one 16 B access); a 128-thread block
//     therefore retires 8 head segments per pass and keeps `PASSES` of them in
//     flight at once, so all of a thread's loads are issued before the first
//     reduction consumes one.
//   * cos/sin for the row are read once per lane (4 rotary pairs, 32 B) and
//     reused across every head of the row -- instead of once per (head, pass)
//     as a separate rope kernel would, and without the separate fp64->bf16 cast
//     pass the reference needs (`image_rotary_emb` is captured as fp64).
//   * The two streams' q/k norms use different weights (`norm_q`/`norm_added_q`);
//     which pair applies is a function of the row alone, so the choice is
//     uniform per block and costs one predicated pointer select at entry.
//
// Numerics follow the reference op-for-op: the sum of squares accumulates in
// fp32, the norm result is rounded to bf16 exactly where the reference's
// `rms_norm` kernel stores it, cos/sin are rounded to the activation dtype
// exactly where the reference's `cos.to(query.dtype)` does, and the rotation
// itself runs in fp32.  Only the order of the variance reduction differs.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kHeadDim = 128;   // bf16 elements per head segment
constexpr int kLanes = 16;      // lanes per head segment (8 elements each)
constexpr int kGroups = 8;      // head segments retired per pass
constexpr int kBlock = kLanes * kGroups;

// 16-byte chunk of 8 bf16 lanes.  Held as uint32_t pairs: a struct of
// __nv_bfloat162 members is copied member-wise and ptxas then emits four 32-bit
// accesses instead of one 128-bit one.
struct alignas(16) Pack {
  uint32_t w[4];
};

// bf16 <-> fp32 without the conversion pipe.
//
// This kernel's arithmetic is trivial next to the number of format conversions
// it needs (a sum of squares, a scale, a rotation -- but four conversions per
// element), and `cvt` is a quarter-rate instruction: a first version that used
// `__bfloat1622float2` / `__float2bfloat16` everywhere ran 2.5x slower than a
// bare copy of the same bytes with the same access pattern, entirely on
// conversion throughput.  bf16 shares fp32's exponent field, so widening is an
// exact 16-bit shift (denormals, inf and NaN included) and costs a full-rate
// integer op instead.  Narrowing still needs a rounding instruction, but
// `cvt.rn.bf16x2.f32` does a pair at a time.
template <typename T>
struct Cvt;

template <>
struct Cvt<__nv_bfloat16> {
  // The two lanes of a packed pair, widened to fp32.
  static __device__ __forceinline__ float lo(uint32_t v) {
    return __uint_as_float(v << 16);
  }
  static __device__ __forceinline__ float hi(uint32_t v) {
    return __uint_as_float(v & 0xffff0000u);
  }
  // Round a pair to bf16 (one instruction) and keep it packed.
  static __device__ __forceinline__ uint32_t round2(float a, float b) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
    uint32_t v;
    __builtin_memcpy(&v, &p, sizeof(v));
    return v;
  }
  static __device__ __forceinline__ float round1(float v) {
    return lo(round2(v, v));
  }
};

template <>
struct Cvt<__half> {
  static __device__ __forceinline__ float lo(uint32_t v) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(v)));
  }
  static __device__ __forceinline__ float hi(uint32_t v) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(v >> 16)));
  }
  static __device__ __forceinline__ uint32_t round2(float a, float b) {
    const __half2 p = __floats2half2_rn(a, b);
    uint32_t v;
    __builtin_memcpy(&v, &p, sizeof(v));
    return v;
  }
  static __device__ __forceinline__ float round1(float v) {
    return __half2float(__float2half(v));
  }
};

// cos/sin arrive in whatever dtype the model computed them in (fp64 for the
// captured FLUX shapes).  The reference casts them to the activation dtype
// before the rope kernel reads them back as fp32, so round through T here too --
// otherwise the rotation would run at higher precision than the reference's.
template <typename T, typename C>
__device__ __forceinline__ float rot_coeff(C v) {
  return Cvt<T>::round1(static_cast<float>(v));
}

// One block per row of the QKV buffer; `PASSES * kGroups` head segments per row.
template <typename T, typename C, int PASSES, bool TWO_STREAM>
__global__ __launch_bounds__(kBlock) void qk_norm_rope_kernel(
    T* __restrict__ qkv,              // [rows, row_stride], q|k in [0, n_items*128)
    const C* __restrict__ cosp,       // [>= rows, 64]
    const C* __restrict__ sinp,
    const T* __restrict__ wq,         // [128] image-stream q scale
    const T* __restrict__ wk,
    const T* __restrict__ wq2,        // [128] text-stream q scale (TWO_STREAM)
    const T* __restrict__ wk2,
    int n_enc, int nq_items, int64_t row_stride, float eps, float inv_h) {
  const int lane = threadIdx.x & (kLanes - 1);
  const int group = threadIdx.x >> 4;
  const int row = blockIdx.x;

  // Rotary pairs [4*lane, 4*lane+4) -- the pairs covered by this lane's 8
  // elements, identical for every head segment it touches.
  float cf[4], sf[4];
  {
    const int64_t off = (int64_t)row * (kHeadDim / 2) + lane * 4;
    const C* __restrict__ cp = cosp + off;
    const C* __restrict__ sp = sinp + off;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      cf[j] = rot_coeff<T, C>(cp[j]);
      sf[j] = rot_coeff<T, C>(sp[j]);
    }
  }

  // Per-stream norm scales: uniform across the block.
  const T* wq_sel = wq;
  const T* wk_sel = wk;
  if constexpr (TWO_STREAM) {
    if (row < n_enc) {
      wq_sel = wq2;
      wk_sel = wk2;
    }
  }
  // Widened once per thread and reused by every head segment it owns.
  float wqf[8], wkf[8];
  {
    const Pack wqv = reinterpret_cast<const Pack*>(wq_sel)[lane];
    const Pack wkv = reinterpret_cast<const Pack*>(wk_sel)[lane];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wqf[2 * j] = Cvt<T>::lo(wqv.w[j]);
      wqf[2 * j + 1] = Cvt<T>::hi(wqv.w[j]);
      wkf[2 * j] = Cvt<T>::lo(wkv.w[j]);
      wkf[2 * j + 1] = Cvt<T>::hi(wkv.w[j]);
    }
  }

  T* __restrict__ base = qkv + (int64_t)row * row_stride + lane * 8;
  Pack* ptr[PASSES];
  Pack x[PASSES];
#pragma unroll
  for (int u = 0; u < PASSES; ++u)
    ptr[u] = reinterpret_cast<Pack*>(base + (u * kGroups + group) * kHeadDim);
  // All PASSES loads are issued before the first is consumed.
#pragma unroll
  for (int u = 0; u < PASSES; ++u) x[u] = *ptr[u];

  float scale[PASSES];
#pragma unroll
  for (int u = 0; u < PASSES; ++u) {
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float f0 = Cvt<T>::lo(x[u].w[j]);
      const float f1 = Cvt<T>::hi(x[u].w[j]);
      a0 = fmaf(f0, f0, a0);
      a1 = fmaf(f1, f1, a1);
    }
    float acc = a0 + a1;
#pragma unroll
    for (int off = kLanes / 2; off > 0; off >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, off, kLanes);
    scale[u] = rsqrtf(acc * inv_h + eps);
  }

#pragma unroll
  for (int u = 0; u < PASSES; ++u) {
    const bool is_k = (u * kGroups + group) >= nq_items;
    const float s = scale[u];
    Pack o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float w0 = is_k ? wkf[2 * j] : wqf[2 * j];
      const float w1 = is_k ? wkf[2 * j + 1] : wqf[2 * j + 1];
      // Rounded to the storage dtype here, exactly where the reference's
      // rms_norm kernel stores it -- then widened back for the rotation.
      const uint32_t n = Cvt<T>::round2(Cvt<T>::lo(x[u].w[j]) * s * w0,
                                        Cvt<T>::hi(x[u].w[j]) * s * w1);
      const float n0 = Cvt<T>::lo(n);
      const float n1 = Cvt<T>::hi(n);
      o.w[j] = Cvt<T>::round2(n0 * cf[j] - n1 * sf[j], n1 * cf[j] + n0 * sf[j]);
    }
    *ptr[u] = o;
  }
}

template <typename T, typename C, bool TWO_STREAM>
void launch(const torch::Tensor& qkv, const torch::Tensor& cos, const torch::Tensor& sin,
            const torch::Tensor& wq, const torch::Tensor& wk,
            const T* wq2, const T* wk2, int rows, int n_items, int n_enc, int nq_items,
            int64_t row_stride, double eps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const float inv_h = 1.0f / static_cast<float>(kHeadDim);
  auto* qp = reinterpret_cast<T*>(qkv.data_ptr());
  const auto* cp = reinterpret_cast<const C*>(cos.data_ptr());
  const auto* sp = reinterpret_cast<const C*>(sin.data_ptr());
  const auto* wqp = reinterpret_cast<const T*>(wq.data_ptr());
  const auto* wkp = reinterpret_cast<const T*>(wk.data_ptr());

#define LAUNCH_PASSES(P)                                                          \
  qk_norm_rope_kernel<T, C, P, TWO_STREAM><<<rows, kBlock, 0, stream>>>(          \
      qp, cp, sp, wqp, wkp, wq2, wk2, n_enc, nq_items, row_stride,                \
      static_cast<float>(eps), inv_h);                                            \
  break
  switch (n_items / kGroups) {
    case 1: LAUNCH_PASSES(1);
    case 2: LAUNCH_PASSES(2);
    case 3: LAUNCH_PASSES(3);
    case 4: LAUNCH_PASSES(4);
    case 5: LAUNCH_PASSES(5);
    case 6: LAUNCH_PASSES(6);
    case 8: LAUNCH_PASSES(8);
    case 10: LAUNCH_PASSES(10);
    case 12: LAUNCH_PASSES(12);
    case 16: LAUNCH_PASSES(16);
    default:
      TORCH_CHECK(false, "qk_norm_rope: unsupported head count ", n_items);
  }
#undef LAUNCH_PASSES
}

template <typename T>
void dispatch_cos(const torch::Tensor& qkv, const torch::Tensor& cos,
                  const torch::Tensor& sin, const torch::Tensor& wq,
                  const torch::Tensor& wk, const c10::optional<torch::Tensor>& wq2,
                  const c10::optional<torch::Tensor>& wk2, int rows, int n_items,
                  int n_enc, int nq_items, int64_t row_stride, double eps) {
  const bool two = wq2.has_value() && wk2.has_value() && n_enc > 0;
  const T* p2q = two ? reinterpret_cast<const T*>(wq2->data_ptr()) : nullptr;
  const T* p2k = two ? reinterpret_cast<const T*>(wk2->data_ptr()) : nullptr;

#define DISPATCH_TWO(C)                                                           \
  if (two)                                                                        \
    launch<T, C, true>(qkv, cos, sin, wq, wk, p2q, p2k, rows, n_items, n_enc,      \
                       nq_items, row_stride, eps);                                 \
  else                                                                            \
    launch<T, C, false>(qkv, cos, sin, wq, wk, p2q, p2k, rows, n_items, n_enc,     \
                        nq_items, row_stride, eps)

  switch (cos.scalar_type()) {
    case at::kDouble:
      DISPATCH_TWO(double);
      break;
    case at::kFloat:
      DISPATCH_TWO(float);
      break;
    case at::kBFloat16:
      DISPATCH_TWO(__nv_bfloat16);
      break;
    case at::kHalf:
      DISPATCH_TWO(__half);
      break;
    default:
      TORCH_CHECK(false, "qk_norm_rope: unsupported cos dtype ", cos.scalar_type());
  }
#undef DISPATCH_TWO
}

}  // namespace

// In-place q/k RMSNorm + interleaved RoPE over a packed [rows, row_stride] QKV
// buffer.  Raises (-> Python RuntimeError) for anything outside the fast path so
// the caller can fall back to the unfused reference chain.
void qk_norm_rope_(torch::Tensor qkv, torch::Tensor cos, torch::Tensor sin,
                   torch::Tensor wq, torch::Tensor wk,
                   c10::optional<torch::Tensor> wq2, c10::optional<torch::Tensor> wk2,
                   int64_t n_enc, int64_t nq_heads, int64_t nk_heads, double eps) {
  TORCH_CHECK(qkv.is_cuda() && cos.is_cuda() && sin.is_cuda(), "qk_norm_rope: cuda only");
  TORCH_CHECK(qkv.dim() == 2, "qk_norm_rope: need 2d qkv");
  TORCH_CHECK(qkv.stride(1) == 1, "qk_norm_rope: qkv rows must be contiguous");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "qk_norm_rope: need 2d cos/sin");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "qk_norm_rope: cos/sin strided");
  TORCH_CHECK(cos.size(1) == kHeadDim / 2 && sin.size(1) == kHeadDim / 2,
              "qk_norm_rope: partial rotation unsupported");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(), "qk_norm_rope: cos/sin dtype");
  TORCH_CHECK(wq.scalar_type() == qkv.scalar_type() && wk.scalar_type() == qkv.scalar_type(),
              "qk_norm_rope: norm weight dtype");
  TORCH_CHECK(wq.is_contiguous() && wk.is_contiguous() && wq.numel() == kHeadDim &&
                  wk.numel() == kHeadDim,
              "qk_norm_rope: norm weight shape");

  const int rows = static_cast<int>(qkv.size(0));
  const int64_t row_stride = qkv.stride(0);
  const int n_items = static_cast<int>(nq_heads + nk_heads);
  TORCH_CHECK(n_items % kGroups == 0, "qk_norm_rope: (nq+nk) % 8");
  TORCH_CHECK((int64_t)n_items * kHeadDim <= qkv.size(1),
              "qk_norm_rope: qkv row too narrow");
  TORCH_CHECK(cos.size(0) >= rows && sin.size(0) >= rows, "qk_norm_rope: seqlen_ro < rows");
  TORCH_CHECK(row_stride % 8 == 0, "qk_norm_rope: row stride not 16B-aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0,
              "qk_norm_rope: qkv base not 16B-aligned");
  if (wq2.has_value()) {
    TORCH_CHECK(wq2->numel() == kHeadDim && wk2->numel() == kHeadDim &&
                    wq2->is_contiguous() && wk2->is_contiguous() &&
                    wq2->scalar_type() == qkv.scalar_type() &&
                    wk2->scalar_type() == qkv.scalar_type(),
                "qk_norm_rope: added norm weight");
  }
  if (rows == 0) return;

  const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
  switch (qkv.scalar_type()) {
    case at::kBFloat16:
      dispatch_cos<__nv_bfloat16>(qkv, cos, sin, wq, wk, wq2, wk2, rows, n_items,
                                  static_cast<int>(n_enc),
                                  static_cast<int>(nq_heads), row_stride, eps);
      break;
    case at::kHalf:
      dispatch_cos<__half>(qkv, cos, sin, wq, wk, wq2, wk2, rows, n_items,
                           static_cast<int>(n_enc), static_cast<int>(nq_heads),
                           row_stride, eps);
      break;
    default:
      TORCH_CHECK(false, "qk_norm_rope: unsupported dtype ", qkv.scalar_type());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_norm_rope_", &qk_norm_rope_,
        "fused in-place q/k RMSNorm + interleaved RoPE on a packed QKV buffer");
}
