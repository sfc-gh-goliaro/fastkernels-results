// Fused per-head RMSNorm + interleaved (GPT-J) RoPE over a packed
// [B, S, 3, H, D] q/k/v buffer, applied in place.
//
// The FLUX attention pre-attention path is pure streaming glue: the QKV
// projection writes q|k|v, then q and k are each read and rewritten by a
// per-head RMSNorm and again by the rotary embedding, and in the dual-stream
// case the text and image halves are additionally concatenated (a third
// read+write of all three tensors).  Every one of those passes touches the same
// 28 MB tensors, so the whole path is bandwidth-bound with nothing but launch
// count and traffic to win.
//
// This kernel collapses it: the QKV GEMMs write straight into row ranges of one
// packed buffer (so the concatenation never happens), and a single launch reads
// each q/k head row once, reduces it in fp32, scales by the norm weight,
// rotates in registers and stores once.  The value rows are never touched.
//
// Numerics are the composition of the two ops being replaced, rounding included:
//   * the fp32 sum of squares is reduced by shuffle over the LPR lanes that own
//     the row, and the scale is rsqrtf(sumsq / D + eps) -- as in rmsnorm_ako.cu;
//   * the normalized value is rounded to the storage dtype *before* the
//     rotation, because the ops it replaces store it and re-read it there;
//   * cos/sin are rounded through the storage dtype too, standing in for the
//     `cos.to(query.dtype)` cast the caller no longer does -- so an fp64
//     (cos, sin) pair is consumed directly, with no separate cast pass.
//
// Lane layout: a row of D elements is 16-byte-vectorized into D/8 vectors, and a
// group of LPR = D/(8*VPR) lanes owns one row, VPR vectors per lane (VPR == 2 at
// head_dim 128).  Two vectors per lane rather than one halves the number of
// shuffle levels the sum-of-squares reduction needs (3 instead of 4 at head_dim
// 128) and doubles the loads in flight per thread; the fp32 widening of the
// inputs is then kept in registers so the rotation pass does not re-convert
// every element.  That combination is 18% faster than one-vector-per-lane and
// bit-identical to it -- see the reduction-order note below.
//
// Reduction order is preserved exactly: `pv[k2]` holds the partial sum of the
// vector at `lane + k2*LPR`, and folding pv widest-stride-first reproduces the
// leading levels of the shuffle tree a one-vector-per-lane kernel would run over
// D/8 lanes, so the fp32 sum is associated identically and the bf16 result is
// bit-for-bit the same.
//
// Row indexing: within a token's 3*H*D block, q occupies [0, H*D) and k occupies
// [H*D, 2*H*D), so a flat row index r in [0, 2H) addresses both -- r < H is a q
// head, r >= H a k head, and the byte offset is r*D either way.  The text and
// image streams use different norm weights (norm_added_q/k vs norm_q/k) but are
// adjacent rows of one buffer, so the weight vector is selected per block from
// the token index against `s_split` rather than by launching twice.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

// 24 lane groups per block: measured faster than 128, 256 or 384 threads at both
// captured sequence lengths.
constexpr int THREADS_PER_BLOCK = 192;

template <class T>
struct Tr;
template <>
struct Tr<__nv_bfloat16> {
  static __device__ __forceinline__ float to_f(__nv_bfloat16 a) { return __bfloat162float(a); }
  static __device__ __forceinline__ __nv_bfloat16 from_f(float a) { return __float2bfloat16_rn(a); }
};
template <>
struct Tr<__half> {
  static __device__ __forceinline__ float to_f(__half a) { return __half2float(a); }
  static __device__ __forceinline__ __half from_f(float a) { return __float2half_rn(a); }
};

// 128-bit vector of eight 16-bit elements: one per lane per row.
template <class T>
struct alignas(16) V8 {
  T d[8];
};

// Widening reads spelled out per type rather than left to implicit conversion,
// which the half/bf16 headers gate behind __CUDA_NO_*_CONVERSIONS__.
__device__ __forceinline__ float wide(double v) { return (float)v; }
__device__ __forceinline__ float wide(float v) { return v; }
__device__ __forceinline__ float wide(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float wide(__half v) { return __half2float(v); }

// One (cos, sin) value, widened, rounded to the storage dtype and widened again:
// that stands in for the caller's `cos.to(query.dtype)`, which is why an fp64
// (cos, sin) pair can be consumed here directly instead of by a cast pass.
template <class T, class C>
__device__ __forceinline__ float cs_cvt(C v) {
  return Tr<T>::to_f(Tr<T>::from_f(wide(v)));
}

// A group of LPR lanes owns one row of D elements (VPR 16-byte vectors each);
// GROUPS such groups per block; each group walks NLOAD rows strided by GROUPS,
// which keeps every iteration's block-wide access one contiguous burst.
template <class T, class C, int D, int VPR, int NLOAD, bool ROPE>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void qk_norm_rope_kernel(
    T* __restrict__ p, const C* __restrict__ cosp, const C* __restrict__ sinp,
    const T* __restrict__ w_q, const T* __restrict__ w_k, const T* __restrict__ wt_q,
    const T* __restrict__ wt_k, const int nheads, const int s_split, const float eps,
    const int64_t tok_vecs) {
  using V = V8<T>;
  constexpr int RVEC = D / 8;         // 16-byte vectors per row
  constexpr int LPR = RVEC / VPR;     // lanes per row
  constexpr int GROUPS = THREADS_PER_BLOCK / LPR;
  constexpr int HALF = D / 2;         // (cos, sin) pairs per row
  // The fp32 widening is kept in registers only while it fits comfortably; a
  // block walking many rows would spill instead, which costs more than the
  // re-conversion it saves.
  constexpr bool F32 = (NLOAD * VPR <= 4);
  const int R = 2 * nheads;

  const int tid = threadIdx.x;
  const int lane = tid & (LPR - 1);
  const int g = tid / LPR;
  constexpr unsigned LMASK = (LPR >= 32) ? 0xffffffffu : ((1u << LPR) - 1u);
  const unsigned mask = LMASK << ((tid & 31) - lane);

  // Token index within its sequence == the block index, because the grid covers
  // exactly one sequence and the caller launches once per batch element with `p`
  // already advanced.  That indirection earns its keep: deriving the index as
  // `blockIdx.x % seq` here instead measured 11-14% slower, and not because of the
  // division's own cost -- the SASS branches around it entirely when seq covers the
  // grid.  It is the division *sequence* (I2F/MUFU.RCP/IMAD chain) holding
  // temporaries live across the branch, which takes the kernel from 64 to 70
  // registers and so from 5 resident blocks per SM to 4.  At 64 registers x 192
  // threads this kernel sits exactly on the 5-block boundary, so anything that
  // needs a seventh register costs a fifth of the occupancy.
  const int s = blockIdx.x;

  // Which of the two streams' norm weights this token uses.  The vectors are read
  // where they are used rather than hoisted into registers here: they are D bytes
  // shared by every block, so they hit L1, and against the register budget above
  // hoisting them is worth about 2%.
  const bool txt = (s < s_split);

  // The token's cos/sin row is staged through shared memory: every group needs
  // the same D/2 pairs, so reading them once per block instead of once per group
  // removes what was the kernel's densest stream of load instructions.
  __shared__ float sh[ROPE ? 2 * HALF : 1];
  float cf[VPR * 4], sf[VPR * 4];
  if (ROPE) {
    const int64_t base = (int64_t)s * HALF;
    for (int i = tid; i < HALF; i += THREADS_PER_BLOCK) {
      sh[i] = cs_cvt<T, C>(cosp[base + i]);
      sh[HALF + i] = cs_cvt<T, C>(sinp[base + i]);
    }
    __syncthreads();
    // A lane owns 8 elements == 4 interleaved pairs per vector, whatever D is.
#pragma unroll
    for (int k = 0; k < VPR; ++k)
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int pi = (lane + k * LPR) * 4 + m;
        cf[k * 4 + m] = sh[pi];
        sf[k * 4 + m] = sh[HALF + pi];
      }
  }

  V* __restrict__ vp = reinterpret_cast<V*>(p) + (int64_t)blockIdx.x * tok_vecs;

  int rows[NLOAD];
#pragma unroll
  for (int i = 0; i < NLOAD; ++i) rows[i] = g + i * GROUPS;

  // All rows are loaded up front so the reductions and the stores never wait on
  // a load issued after them.
  V buf[NLOAD][VPR];
#pragma unroll
  for (int i = 0; i < NLOAD; ++i)
    if (rows[i] < R)
#pragma unroll
      for (int k = 0; k < VPR; ++k) buf[i][k] = vp[(int64_t)rows[i] * RVEC + lane + k * LPR];

  float xf[F32 ? NLOAD : 1][F32 ? VPR * 8 : 1];
  if (F32) {
#pragma unroll
    for (int i = 0; i < NLOAD; ++i)
#pragma unroll
      for (int k = 0; k < VPR; ++k)
#pragma unroll
        for (int j = 0; j < 8; ++j) xf[i][k * 8 + j] = Tr<T>::to_f(buf[i][k].d[j]);
  }

  float sc[NLOAD];
#pragma unroll
  for (int i = 0; i < NLOAD; ++i) {
    float pv[VPR];
#pragma unroll
    for (int k = 0; k < VPR; ++k) {
      float acc = 0.f;
      if (rows[i] < R) {
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const float a = F32 ? xf[i][k * 8 + j] : Tr<T>::to_f(buf[i][k].d[j]);
          const float b = F32 ? xf[i][k * 8 + j + 1] : Tr<T>::to_f(buf[i][k].d[j + 1]);
          acc += a * a + b * b;
        }
      }
      pv[k] = acc;
    }
    // Fold the per-vector sums widest-stride-first: this is exactly the leading
    // levels of the shuffle tree a one-vector-per-lane kernel runs, so the fp32
    // association -- and therefore the bf16 result -- is unchanged.
#pragma unroll
    for (int w = VPR / 2; w > 0; w >>= 1)
#pragma unroll
      for (int t = 0; t < w; ++t) pv[t] += pv[t + w];
    float acc = pv[0];
    // Every lane of a group shares rows[i], so the group is uniformly active
    // and all LPR lanes reach the shuffle.
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1) acc += __shfl_xor_sync(mask, acc, off);
    sc[i] = rsqrtf(acc / (float)D + eps);
  }

#pragma unroll
  for (int i = 0; i < NLOAD; ++i) {
    if (rows[i] >= R) continue;
    const V* __restrict__ wp = reinterpret_cast<const V*>(
        (rows[i] < nheads) ? (txt ? wt_q : w_q) : (txt ? wt_k : w_k));
    const float s_i = sc[i];
#pragma unroll
    for (int k = 0; k < VPR; ++k) {
      const V w = wp[lane + k * LPR];
      V o;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float x0 = F32 ? xf[i][k * 8 + 2 * m] : Tr<T>::to_f(buf[i][k].d[2 * m]);
        const float x1 = F32 ? xf[i][k * 8 + 2 * m + 1] : Tr<T>::to_f(buf[i][k].d[2 * m + 1]);
        const float xa = x0 * s_i * Tr<T>::to_f(w.d[2 * m]);
        const float xb = x1 * s_i * Tr<T>::to_f(w.d[2 * m + 1]);
        if (ROPE) {
          // The norm result is rounded here because the two ops this replaces
          // store it and read it back before rotating.
          const float na = Tr<T>::to_f(Tr<T>::from_f(xa));
          const float nb = Tr<T>::to_f(Tr<T>::from_f(xb));
          const float c = cf[k * 4 + m], sn = sf[k * 4 + m];
          o.d[2 * m] = Tr<T>::from_f(na * c - nb * sn);
          o.d[2 * m + 1] = Tr<T>::from_f(na * sn + nb * c);
        } else {
          o.d[2 * m] = Tr<T>::from_f(xa);
          o.d[2 * m + 1] = Tr<T>::from_f(xb);
        }
      }
      vp[(int64_t)rows[i] * RVEC + lane + k * LPR] = o;
    }
  }
}

struct Plan {
  int nheads, seq, s_split, D, batch;
  int64_t tok_vecs;
  float eps;
};

// Two vectors per lane at head_dim 128, one at 64: either way a group is 8 lanes
// and a block is THREADS_PER_BLOCK/8 groups.
template <int D>
struct Vpr {
  static constexpr int value = (D >= 128) ? 2 : 1;
};

template <class T, class C, int D, bool ROPE>
bool launch_nload(int nload, const Plan& pl, cudaStream_t st, void* p, const void* c,
                  const void* s, const void* wq, const void* wk, const void* wtq,
                  const void* wtk) {
  constexpr int VPR = Vpr<D>::value;
  T* pp = reinterpret_cast<T*>(p);
  const C* cp = reinterpret_cast<const C*>(c);
  const C* sp = reinterpret_cast<const C*>(s);
  const T* a = reinterpret_cast<const T*>(wq);
  const T* b = reinterpret_cast<const T*>(wk);
  const T* x = reinterpret_cast<const T*>(wtq);
  const T* y = reinterpret_cast<const T*>(wtk);
  // One launch per batch element, each over exactly one sequence: see the comment
  // on `s` in the kernel.  Every captured call has batch 1, so this is one launch.
  const int64_t bstride = (int64_t)pl.seq * pl.tok_vecs * 8;
#define FK_NL(N)                                                                        \
  case N:                                                                               \
    for (int64_t bi = 0; bi < pl.batch; ++bi)                                           \
      qk_norm_rope_kernel<T, C, D, VPR, N, ROPE>                                        \
          <<<(unsigned)pl.seq, THREADS_PER_BLOCK, 0, st>>>(pp + bi * bstride, cp, sp, a, \
                                                          b, x, y, pl.nheads,           \
                                                          pl.s_split, pl.eps,           \
                                                          pl.tok_vecs);                 \
    return true;
  switch (nload) {
    FK_NL(1)
    FK_NL(2)
    FK_NL(3)
    FK_NL(4)
    FK_NL(6)
    FK_NL(8)
    default:
      return false;
  }
#undef FK_NL
}

template <class T, int D, bool ROPE>
bool launch_cdtype(at::ScalarType cdt, int nload, const Plan& pl, cudaStream_t st, void* p,
                   const void* c, const void* s, const void* wq, const void* wk,
                   const void* wtq, const void* wtk) {
  switch (cdt) {
    case at::kDouble:
      return launch_nload<T, double, D, ROPE>(nload, pl, st, p, c, s, wq, wk, wtq, wtk);
    case at::kFloat:
      return launch_nload<T, float, D, ROPE>(nload, pl, st, p, c, s, wq, wk, wtq, wtk);
    case at::kBFloat16:
      return launch_nload<T, __nv_bfloat16, D, ROPE>(nload, pl, st, p, c, s, wq, wk, wtq,
                                                     wtk);
    case at::kHalf:
      return launch_nload<T, __half, D, ROPE>(nload, pl, st, p, c, s, wq, wk, wtq, wtk);
    default:
      return false;
  }
}

template <class T>
bool launch_D(int D, bool rope, at::ScalarType cdt, int nload, const Plan& pl,
              cudaStream_t st, void* p, const void* c, const void* s, const void* wq,
              const void* wk, const void* wtq, const void* wtk) {
  if (D == 128) {
    return rope ? launch_cdtype<T, 128, true>(cdt, nload, pl, st, p, c, s, wq, wk, wtq, wtk)
                : launch_cdtype<T, 128, false>(cdt, nload, pl, st, p, c, s, wq, wk, wtq,
                                               wtk);
  }
  if (D == 64) {
    return rope ? launch_cdtype<T, 64, true>(cdt, nload, pl, st, p, c, s, wq, wk, wtq, wtk)
                : launch_cdtype<T, 64, false>(cdt, nload, pl, st, p, c, s, wq, wk, wtq,
                                              wtk);
  }
  return false;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

// packed: [B, S, 3, H, D], contiguous, bf16/fp16, modified in place.
// cos/sin: [>=S, D/2] contiguous (any float dtype), or absent for norm-only.
// w_q/w_k: norm weights for tokens at or after `s_split`; wt_q/wt_k for the
// tokens before it (the text stream).  Returns false, having done nothing, when
// the shape/layout is outside what this kernel covers.
bool flux_qk_norm_rope(at::Tensor packed, std::optional<at::Tensor> cos,
                       std::optional<at::Tensor> sin, at::Tensor w_q, at::Tensor w_k,
                       std::optional<at::Tensor> wt_q, std::optional<at::Tensor> wt_k,
                       int64_t s_split, double eps) {
  if (!packed.is_cuda() || packed.dim() != 5 || !packed.is_contiguous()) return false;
  const auto dt = packed.scalar_type();
  if (dt != at::kBFloat16 && dt != at::kHalf) return false;
  if (packed.size(2) != 3) return false;

  const int64_t B = packed.size(0), S = packed.size(1), Hh = packed.size(3),
                D = packed.size(4);
  if (D != 64 && D != 128) return false;
  if (B < 1 || S < 1 || Hh < 1) return false;
  // S is the grid's x dimension and B its y dimension, whose CUDA limit is 65535.
  if (S > 2147483647LL || B > 65535LL) return false;

  const bool rope = cos.has_value() && sin.has_value();
  at::ScalarType cdt = at::kFloat;
  const void *cp = nullptr, *sp = nullptr;
  if (rope) {
    at::Tensor c = *cos, s = *sin;
    if (c.dim() == 3) {  // (1, seqlen_ro, D/2)
      if (c.size(0) < 1 || s.dim() != 3) return false;
      c = c.select(0, 0);
      s = s.select(0, 0);
    }
    if (c.dim() != 2 || s.dim() != 2) return false;
    if (!c.is_contiguous() || !s.is_contiguous()) return false;
    if (c.sizes() != s.sizes() || c.scalar_type() != s.scalar_type()) return false;
    if (c.size(1) * 2 != D || c.size(0) < S) return false;
    if (c.device() != packed.device() || s.device() != packed.device()) return false;
    cdt = c.scalar_type();
    cp = c.data_ptr();
    sp = s.data_ptr();
  }

  // The text weights are only read when some token actually falls before the
  // split; otherwise the image weights stand in so the pointers are never null.
  at::Tensor tq = (s_split > 0 && wt_q.has_value()) ? *wt_q : w_q;
  at::Tensor tk = (s_split > 0 && wt_k.has_value()) ? *wt_k : w_k;
  if (s_split < 0 || s_split > S) return false;
  for (const at::Tensor& w : {w_q, w_k, tq, tk}) {
    if (!w.is_cuda() || w.scalar_type() != dt || !w.is_contiguous()) return false;
    if (w.numel() != D || !aligned16(w.data_ptr())) return false;
  }
  if (!aligned16(packed.data_ptr())) return false;

  // A group is 8 lanes at both supported head_dims (see Vpr).
  const int groups = THREADS_PER_BLOCK / 8;
  // Rows past 2*heads are predicated off inside the kernel, so the row count
  // need not divide the group count: round the per-thread row budget up to an
  // instantiated value rather than declining the call.
  const int R = 2 * (int)Hh;
  int nload = 0;
  for (const int cand : {1, 2, 3, 4, 6, 8}) {
    if (cand * groups >= R) {
      nload = cand;
      break;
    }
  }
  if (nload == 0) return false;  // more heads than one block can walk

  Plan pl;
  pl.nheads = (int)Hh;
  pl.seq = (int)S;
  pl.s_split = (int)s_split;
  pl.D = (int)D;
  pl.batch = (int)B;
  pl.tok_vecs = 3 * (int64_t)Hh * D / 8;
  pl.eps = (float)eps;

  const c10::cuda::CUDAGuard guard(packed.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (dt == at::kBFloat16)
    return launch_D<__nv_bfloat16>((int)D, rope, cdt, nload, pl, st, packed.data_ptr(), cp,
                                   sp, w_q.data_ptr(), w_k.data_ptr(), tq.data_ptr(),
                                   tk.data_ptr());
  return launch_D<__half>((int)D, rope, cdt, nload, pl, st, packed.data_ptr(), cp, sp,
                          w_q.data_ptr(), w_k.data_ptr(), tq.data_ptr(), tk.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flux_qk_norm_rope", &flux_qk_norm_rope,
        "Fused per-head RMSNorm + interleaved RoPE over a packed [B,S,3,H,D] buffer");
}
