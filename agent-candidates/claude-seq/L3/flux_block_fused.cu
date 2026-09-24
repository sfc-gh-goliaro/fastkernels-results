// Fused elementwise/LayerNorm epilogues for the FLUX L3 transformer blocks.
//
// The two blocks spend ~20-25% of their GPU time in *broadcast* elementwise
// kernels that torch's eager path runs at ~1.5-1.8 TB/s on sm_100 (a B200 does
// ~5.7 TB/s on a streaming pass over the same buffers): every
// `x * (1 + scale[:, None]) + shift[:, None]` and every `gate.unsqueeze(1) * y`
// reads and writes the whole [S, 3072] activation through an unvectorized,
// index-computing kernel.  The dual-stream block issues twelve of those per
// call plus four LayerNorms and four residual adds -- sixteen full passes over
// 28 MB.
//
// Everything here collapses a chain of those into one pass:
//
//   ln_mod        y = LN(x) * (1 + scale) + shift
//   add_ln_mod    x' = x + gate * a ;  y = LN(x') * (1 + scale) + shift
//   gated_add     out = res + gate * y
//
// each over up to two independent row segments (the image and text streams, or
// the text/image halves of the single-stream block's concatenated sequence) so
// one launch serves both -- `grid.y` picks the segment, and the 512-row text
// stream costs no extra launch latency.
//
// Three things decide the speed of these kernels, in the order they were found:
//
// * **The modulation vectors live in shared memory, not registers.**  They are
//   the same 3072 values for every row of a segment, so hoisting them per
//   thread is tempting -- but it costs 48 registers, which caps the grid at
//   ~4 blocks/SM and leaves the kernel short of loads in flight.
// * **The next row is loaded before the current row's reduction**, so a block's
//   memory pipeline is not idle across the barrier.
// * **The arithmetic is done in packed `bf16x2` / `half2`, not in fp32.**  The
//   scalar form needs ~9 fp32<->bf16 conversions per element and
//   `cvt.rn.bf16.f32` issues at a quarter of the fp32 rate.
//   `mul.rn.bf16x2` / `add.rn.bf16x2` (sm_90+) compute the exact product/sum of
//   two bf16 values and round once -- *bit-identical* to the reference's
//   "promote to fp32, operate, round back" -- at one instruction per two
//   elements, cutting that to ~1.5.  Only the LayerNorm reduction and its
//   `(x - mean) * rstd` stay in fp32.
//
// What is left on the table, so the next person does not re-run it: the
// LayerNorm kernels move their bytes at ~1.7-2.2 TB/s against ~3.2 TB/s for a
// plain `copy_` of the same buffers, and none of the usual suspects accounts for
// it.  Register pressure (58/78 registers, 6-8 blocks/SM), the block-wide
// barrier, bytes in flight, and the conversion count were each addressed and
// each moved the number by less than measurement noise; a warp-per-row form
// with no barrier at all and 4x the data in flight measured 12% *slower* at
// S = 4608, and a sweep of 32..384 threads x 1..16 blocks/SM is flat to within
// 5%.  `gated_add`, the same memory pattern minus the reduction, reaches 2.5
// TB/s -- so the reduction's serial `sum`/`sq` dependency chain is the leading
// remaining suspect, and breaking it needs multiple accumulators, which changes
// the summation order (see the note on that below).
//
// Semantics follow the reference exactly where it is observable: every
// intermediate the reference materializes as a tensor is rounded to the
// activation dtype here too.  That matters -- folding the chain into one fp32
// expression is *more* accurate, and therefore disagrees with the reference by
// up to two ulp wherever `shift` cancels the scaled value, which is exactly
// where its relative tolerance is tightest.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

// --- packed-pair traits: bf16x2 / half2 --------------------------------------
template <typename T>
struct Pack;

// `mul` and `add` are spelled as inline PTX rather than as `__hmul2` / `__hadd2`
// because nvcc contracts the intrinsic pair into a single-rounding
// `fma.rn.bf16x2`.  The reference rounds the product to bf16 before adding (it
// materializes it as a tensor), and dropping that rounding moves ~27% of the
// elements by one ulp -- which this block amplifies into a failed comparison.
template <>
struct Pack<__nv_bfloat16> {
  using P = __nv_bfloat162;
  union U {
    unsigned u;
    P p;
  };
  static __device__ __forceinline__ float2 unpack(P v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ P pack(float a, float b) {
    return __floats2bfloat162_rn(a, b);
  }
  static __device__ __forceinline__ P mul(P a, P b) {
    U x, y, r;
    x.p = a;
    y.p = b;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r.u) : "r"(x.u), "r"(y.u));
    return r.p;
  }
  static __device__ __forceinline__ P add(P a, P b) {
    U x, y, r;
    x.p = a;
    y.p = b;
    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r.u) : "r"(x.u), "r"(y.u));
    return r.p;
  }
};

template <>
struct Pack<__half> {
  using P = __half2;
  union U {
    unsigned u;
    P p;
  };
  static __device__ __forceinline__ float2 unpack(P v) { return __half22float2(v); }
  static __device__ __forceinline__ P pack(float a, float b) {
    return __floats2half2_rn(a, b);
  }
  static __device__ __forceinline__ P mul(P a, P b) {
    U x, y, r;
    x.p = a;
    y.p = b;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r.u) : "r"(x.u), "r"(y.u));
    return r.p;
  }
  static __device__ __forceinline__ P add(P a, P b) {
    U x, y, r;
    x.p = a;
    y.p = b;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(r.u) : "r"(x.u), "r"(y.u));
    return r.p;
  }
};

// The widest load/store a thread can issue, as packed pairs.
template <typename T>
struct alignas(16) Vec {
  static constexpr int NP = 16 / (2 * sizeof(T));
  typename Pack<T>::P p[NP];
};

// Sum (a, b) across the block, leaving the total in every thread.  One barrier:
// the partials are summed redundantly by all threads rather than by one warp
// plus a broadcast.  Consecutive rows use different `sm` halves, so no
// anti-dependency barrier is needed.
__device__ __forceinline__ void warp_red2(float &a, float &b) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, off);
    b += __shfl_xor_sync(0xffffffffu, b, off);
  }
}

__device__ __forceinline__ void block_red2(float &a, float &b, float *sm) {
  const int nw = blockDim.x >> 5;
  warp_red2(a, b);
  if (nw == 1) return;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) {
    sm[wid] = a;
    sm[32 + wid] = b;
  }
  __syncthreads();
  float sa = 0.f, sb = 0.f;
  for (int i = 0; i < nw; ++i) {
    sa += sm[i];
    sb += sm[32 + i];
  }
  a = sa;
  b = sb;
}

// ---------------------------------------------------------------------------
// LayerNorm + modulation, optionally preceded by a gated residual add.
// ---------------------------------------------------------------------------
template <typename T>
struct SegLN {
  const T *x;      // input rows
  const T *a;      // gated addend (HAS_A only)
  T *xo;           // x + gate * a  (HAS_A only)
  T *y;            // LN(.) * (1 + scale) + shift
  const T *shift;  // [n]
  const T *scale;  // [n]
  const T *gate;   // [n]  (HAS_A only)
  int rows;
};

template <typename T, int VPT, bool HAS_A>
__global__ __launch_bounds__(384) void ln_mod_k(SegLN<T> s0, SegLN<T> s1, int n,
                                               int nvec, float eps) {
  using V = Vec<T>;
  constexpr int NP = V::NP;
  using PK = Pack<T>;
  const SegLN<T> s = (blockIdx.y == 0) ? s0 : s1;
  if (blockIdx.x >= (unsigned)s.rows) return;

  extern __shared__ char smem[];
  V *sh_shift = reinterpret_cast<V *>(smem);
  V *sh_scale = sh_shift + nvec;
  V *sh_gate = sh_scale + nvec;
  __shared__ float sm[128];

  const int tid = threadIdx.x, nt = blockDim.x;
  {
    const V *SH = reinterpret_cast<const V *>(s.shift);
    const V *SC = reinterpret_cast<const V *>(s.scale);
    const V *GA = reinterpret_cast<const V *>(s.gate);
    for (int j = tid; j < nvec; j += nt) {
      sh_shift[j] = SH[j];
      V b = SC[j], o;
#pragma unroll
      for (int q = 0; q < NP; ++q) {
        const float2 f = PK::unpack(b.p[q]);
        o.p[q] = PK::pack(1.0f + f.x, 1.0f + f.y);
      }
      sh_scale[j] = o;
      if (HAS_A) sh_gate[j] = GA[j];
    }
  }
  __syncthreads();

  const float inv_n = 1.0f / (float)n;
  int idx[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int j = tid + i * nt;
    idx[i] = j < nvec ? j : -1;
  }

  const long step = (long)gridDim.x;
  V cx[VPT], ca[VPT], nx[VPT], na[VPT];
  long row = (long)blockIdx.x;
  {
    const V *xv = reinterpret_cast<const V *>(s.x + row * (long)n);
    const V *av = reinterpret_cast<const V *>(s.a + row * (long)n);
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (idx[i] >= 0) {
        cx[i] = xv[idx[i]];
        if (HAS_A) ca[i] = av[idx[i]];
      }
  }
  int parity = 0;
  for (; row < (long)s.rows; row += step) {
    const long nrow = row + step;
    if (nrow < (long)s.rows) {
      const V *xn = reinterpret_cast<const V *>(s.x + nrow * (long)n);
      const V *an = reinterpret_cast<const V *>(s.a + nrow * (long)n);
#pragma unroll
      for (int i = 0; i < VPT; ++i)
        if (idx[i] >= 0) {
          nx[i] = xn[idx[i]];
          if (HAS_A) na[i] = an[idx[i]];
        }
    }
    float sum = 0.f, sq = 0.f;
    if (HAS_A) {
      V *xov = reinterpret_cast<V *>(s.xo + row * (long)n);
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        const int j = idx[i];
        if (j >= 0) {
          const V g = sh_gate[j];
          V o;
#pragma unroll
          for (int q = 0; q < NP; ++q)
            o.p[q] = PK::add(cx[i].p[q], PK::mul(g.p[q], ca[i].p[q]));
          xov[j] = o;
          // The LayerNorm reads the rounded residual, as the reference does.
          cx[i] = o;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (idx[i] >= 0) {
#pragma unroll
        for (int q = 0; q < NP; ++q) {
          // Element order, one accumulate at a time: pairing the two halves of
          // a packed value first perturbs `sum` in the last fp32 bits, and the
          // block is sensitive enough to that (via the attention it feeds) to
          // move percents of the output outside the reference's tolerance.
          const float2 f = PK::unpack(cx[i].p[q]);
          sum += f.x;
          sq = fmaf(f.x, f.x, sq);
          sum += f.y;
          sq = fmaf(f.y, f.y, sq);
        }
      }
    block_red2(sum, sq, sm + (parity ? 64 : 0));
    parity ^= 1;
    const float mean = sum * inv_n;
    const float rstd = rsqrtf(fmaxf(sq * inv_n - mean * mean, 0.f) + eps);
    V *yv = reinterpret_cast<V *>(s.y + row * (long)n);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int j = idx[i];
      if (j >= 0) {
        const V sc = sh_scale[j], sft = sh_shift[j];
        V o;
#pragma unroll
        for (int q = 0; q < NP; ++q) {
          const float2 f = PK::unpack(cx[i].p[q]);
          const typename PK::P nrm =
              PK::pack((f.x - mean) * rstd, (f.y - mean) * rstd);
          o.p[q] = PK::add(PK::mul(nrm, sc.p[q]), sft.p[q]);
        }
        yv[j] = o;
      }
    }
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      cx[i] = nx[i];
      if (HAS_A) ca[i] = na[i];
    }
  }
}


// ---------------------------------------------------------------------------
// out = res + gate * y
// ---------------------------------------------------------------------------
template <typename T>
struct SegGA {
  const T *res;
  const T *y;
  T *out;
  const T *gate;
  int rows;
};

template <typename T, int VPT>
__global__ __launch_bounds__(384) void gated_add_k(SegGA<T> s0, SegGA<T> s1, int n,
                                                   int nvec) {
  using V = Vec<T>;
  using PK = Pack<T>;
  constexpr int NP = V::NP;
  const SegGA<T> s = (blockIdx.y == 0) ? s0 : s1;
  if (blockIdx.x >= (unsigned)s.rows) return;

  extern __shared__ char smem[];
  V *sh_gate = reinterpret_cast<V *>(smem);
  const int tid = threadIdx.x, nt = blockDim.x;
  {
    const V *GA = reinterpret_cast<const V *>(s.gate);
    for (int j = tid; j < nvec; j += nt) sh_gate[j] = GA[j];
  }
  __syncthreads();

  int idx[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int j = tid + i * nt;
    idx[i] = j < nvec ? j : -1;
  }
  const long step = (long)gridDim.x;
  for (long row = (long)blockIdx.x; row < (long)s.rows; row += step) {
    const V *rv = reinterpret_cast<const V *>(s.res + row * (long)n);
    const V *yv = reinterpret_cast<const V *>(s.y + row * (long)n);
    V *ov = reinterpret_cast<V *>(s.out + row * (long)n);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int j = idx[i];
      if (j >= 0) {
        const V rr = rv[j], yy = yv[j], g = sh_gate[j];
        V o;
#pragma unroll
        for (int q = 0; q < NP; ++q)
          o.p[q] = PK::add(rr.p[q], PK::mul(g.p[q], yy.p[q]));
        ov[j] = o;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
// Block width to aim for, and the resident-block budget for the persistent
// grid.  Swept over 32..384 threads x 1..16 blocks/SM at both captured
// sequence lengths; everything from 96x4 to 192x8 lands within measurement
// noise of each other, and this is the middle of that plateau.
constexpr int kTargetThreads = 128;
constexpr int kBlocksPerSM = 8;

int sm_count() {
  static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

inline int round32(int v) { return ((v + 31) / 32) * 32; }

// Smallest vectors-per-thread whose block still covers a row within the target
// width; vpt = 0 when no supported VPT does.
inline void pick(int nvec, int &vpt, int &threads) {
  static const int kV[] = {1, 2, 3, 4, 6, 8, 12, 16};
  for (int i = 0; i < 8; ++i) {
    const int t = round32((nvec + kV[i] - 1) / kV[i]);
    if (t <= kTargetThreads) {
      vpt = kV[i];
      threads = t < 32 ? 32 : t;
      return;
    }
  }
  vpt = 0;
  threads = 0;
}

inline void check_rows(const at::Tensor &t, int64_t rows, int64_t n, const char *what) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), what, ": must be contiguous CUDA");
  TORCH_CHECK(t.numel() == rows * n, what, ": wrong size");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.const_data_ptr()) & 15) == 0, what,
              ": needs 16B alignment");
}

// A [n]-sized slice of a flat modulation tensor.
template <typename T>
inline const T *slice(const at::Tensor &mod, int64_t off, int64_t n) {
  TORCH_CHECK(off + n <= mod.numel(), "modulation slice out of range");
  return reinterpret_cast<const T *>(mod.const_data_ptr()) + off;
}

template <typename T>
void run_ln_mod(SegLN<T> s0, SegLN<T> s1, int n, float eps, bool has_a) {
  int vpt = 0, threads = 0;
  const int nvec = n / (16 / (int)sizeof(T));
  pick(nvec, vpt, threads);
  TORCH_CHECK(vpt > 0, "ln_mod: row too wide");
  const int rows = s0.rows > s1.rows ? s0.rows : s1.rows;
  const int cap = kBlocksPerSM * sm_count();
  const dim3 grid((unsigned)(rows < cap ? rows : cap), s1.rows > 0 ? 2u : 1u);
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shmem = (size_t)(has_a ? 3 : 2) * (size_t)n * sizeof(T);
#define LN_LAUNCH(V, A)                                                      \
  ln_mod_k<T, V, A><<<grid, threads, shmem, stream>>>(s0, s1, n, nvec, eps)
#define LN_VPT(A)                                                            \
  switch (vpt) {                                                             \
    case 1: LN_LAUNCH(1, A); break;                                          \
    case 2: LN_LAUNCH(2, A); break;                                          \
    case 3: LN_LAUNCH(3, A); break;                                          \
    case 4: LN_LAUNCH(4, A); break;                                          \
    case 6: LN_LAUNCH(6, A); break;                                          \
    case 8: LN_LAUNCH(8, A); break;                                          \
    case 12: LN_LAUNCH(12, A); break;                                        \
    default: LN_LAUNCH(16, A); break;                                        \
  }
  if (has_a) {
    LN_VPT(true)
  } else {
    LN_VPT(false)
  }
#undef LN_VPT
#undef LN_LAUNCH
}

template <typename T>
void run_gated_add(SegGA<T> s0, SegGA<T> s1, int n) {
  int vpt = 0, threads = 0;
  const int nvec = n / (16 / (int)sizeof(T));
  pick(nvec, vpt, threads);
  TORCH_CHECK(vpt > 0, "gated_add: row too wide");
  const int rows = s0.rows > s1.rows ? s0.rows : s1.rows;
  const int cap = kBlocksPerSM * sm_count();
  const dim3 grid((unsigned)(rows < cap ? rows : cap), s1.rows > 0 ? 2u : 1u);
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t shmem = (size_t)n * sizeof(T);
#define GA_LAUNCH(V)                                                         \
  gated_add_k<T, V><<<grid, threads, shmem, stream>>>(s0, s1, n, nvec)
  switch (vpt) {
    case 1: GA_LAUNCH(1); break;
    case 2: GA_LAUNCH(2); break;
    case 3: GA_LAUNCH(3); break;
    case 4: GA_LAUNCH(4); break;
    case 6: GA_LAUNCH(6); break;
    case 8: GA_LAUNCH(8); break;
    case 12: GA_LAUNCH(12); break;
    default: GA_LAUNCH(16); break;
  }
#undef GA_LAUNCH
}

inline void check_dtype(at::ScalarType dt) {
  TORCH_CHECK(dt == at::kBFloat16 || dt == at::kHalf,
              "flux_block_fused: only bf16 / fp16 activations");
}

}  // namespace

// x0/x1: [rows_i, n] inputs.  mod_i: flat conditioning vector; shift is at
// `shift_off`, scale at `shift_off + n`.  Writes y0/y1.
void ln_mod(const at::Tensor &x0, const at::Tensor &y0, const at::Tensor &mod0,
            int64_t shift_off0, const c10::optional<at::Tensor> &x1,
            const c10::optional<at::Tensor> &y1,
            const c10::optional<at::Tensor> &mod1, int64_t shift_off1,
            double eps) {
  const auto dt = x0.scalar_type();
  check_dtype(dt);
  const int64_t n = x0.size(-1);
  const int64_t r0 = x0.numel() / n;
  check_rows(x0, r0, n, "x0");
  check_rows(y0, r0, n, "y0");
  int64_t r1 = 0;
  if (x1.has_value()) {
    r1 = x1->numel() / n;
    TORCH_CHECK(x1->size(-1) == n && x1->scalar_type() == dt, "x1: mismatch");
    check_rows(*x1, r1, n, "x1");
    check_rows(*y1, r1, n, "y1");
  }
  const c10::cuda::CUDAGuard guard(x0.device());
#define LN_BODY(T)                                                             \
  {                                                                            \
    SegLN<T> s0{}, s1{};                                                       \
    s0.x = reinterpret_cast<const T *>(x0.const_data_ptr());                   \
    s0.y = reinterpret_cast<T *>(y0.data_ptr());                               \
    s0.shift = slice<T>(mod0, shift_off0, n);                                  \
    s0.scale = s0.shift + n;                                                   \
    s0.rows = (int)r0;                                                         \
    if (r1 > 0) {                                                              \
      s1.x = reinterpret_cast<const T *>(x1->const_data_ptr());                \
      s1.y = reinterpret_cast<T *>(y1->data_ptr());                            \
      s1.shift = slice<T>(*mod1, shift_off1, n);                               \
      s1.scale = s1.shift + n;                                                 \
      s1.rows = (int)r1;                                                       \
    }                                                                          \
    run_ln_mod<T>(s0, s1, (int)n, (float)eps, false);                          \
  }
  if (dt == at::kBFloat16) LN_BODY(__nv_bfloat16) else LN_BODY(__half)
#undef LN_BODY
}

// xo_i = x_i + gate_i * a_i ;  y_i = LN(xo_i) * (1 + scale_i) + shift_i
// gate at `gate_off`, shift at `shift_off`, scale at `shift_off + n`.
void add_ln_mod(const at::Tensor &x0, const at::Tensor &a0, const at::Tensor &xo0,
                const at::Tensor &y0, const at::Tensor &mod0, int64_t gate_off0,
                int64_t shift_off0, const c10::optional<at::Tensor> &x1,
                const c10::optional<at::Tensor> &a1,
                const c10::optional<at::Tensor> &xo1,
                const c10::optional<at::Tensor> &y1,
                const c10::optional<at::Tensor> &mod1, int64_t gate_off1,
                int64_t shift_off1, double eps) {
  const auto dt = x0.scalar_type();
  check_dtype(dt);
  const int64_t n = x0.size(-1);
  const int64_t r0 = x0.numel() / n;
  check_rows(x0, r0, n, "x0");
  check_rows(a0, r0, n, "a0");
  check_rows(xo0, r0, n, "xo0");
  check_rows(y0, r0, n, "y0");
  int64_t r1 = 0;
  if (x1.has_value()) {
    r1 = x1->numel() / n;
    TORCH_CHECK(x1->size(-1) == n && x1->scalar_type() == dt, "x1: mismatch");
    check_rows(*x1, r1, n, "x1");
    check_rows(*a1, r1, n, "a1");
    check_rows(*xo1, r1, n, "xo1");
    check_rows(*y1, r1, n, "y1");
  }
  const c10::cuda::CUDAGuard guard(x0.device());
#define ALN_BODY(T)                                                            \
  {                                                                            \
    SegLN<T> s0{}, s1{};                                                       \
    s0.x = reinterpret_cast<const T *>(x0.const_data_ptr());                   \
    s0.a = reinterpret_cast<const T *>(a0.const_data_ptr());                   \
    s0.xo = reinterpret_cast<T *>(xo0.data_ptr());                             \
    s0.y = reinterpret_cast<T *>(y0.data_ptr());                               \
    s0.gate = slice<T>(mod0, gate_off0, n);                                    \
    s0.shift = slice<T>(mod0, shift_off0, n);                                  \
    s0.scale = s0.shift + n;                                                   \
    s0.rows = (int)r0;                                                         \
    if (r1 > 0) {                                                              \
      s1.x = reinterpret_cast<const T *>(x1->const_data_ptr());                \
      s1.a = reinterpret_cast<const T *>(a1->const_data_ptr());                \
      s1.xo = reinterpret_cast<T *>(xo1->data_ptr());                          \
      s1.y = reinterpret_cast<T *>(y1->data_ptr());                            \
      s1.gate = slice<T>(*mod1, gate_off1, n);                                 \
      s1.shift = slice<T>(*mod1, shift_off1, n);                               \
      s1.scale = s1.shift + n;                                                 \
      s1.rows = (int)r1;                                                       \
    }                                                                          \
    run_ln_mod<T>(s0, s1, (int)n, (float)eps, true);                           \
  }
  if (dt == at::kBFloat16) ALN_BODY(__nv_bfloat16) else ALN_BODY(__half)
#undef ALN_BODY
}

// out_i = res_i + gate_i * y_i
void gated_add(const at::Tensor &res0, const at::Tensor &y0, const at::Tensor &out0,
               const at::Tensor &mod0, int64_t gate_off0,
               const c10::optional<at::Tensor> &res1,
               const c10::optional<at::Tensor> &y1,
               const c10::optional<at::Tensor> &out1,
               const c10::optional<at::Tensor> &mod1, int64_t gate_off1) {
  const auto dt = res0.scalar_type();
  check_dtype(dt);
  const int64_t n = res0.size(-1);
  const int64_t r0 = res0.numel() / n;
  check_rows(res0, r0, n, "res0");
  check_rows(y0, r0, n, "y0");
  check_rows(out0, r0, n, "out0");
  int64_t r1 = 0;
  if (res1.has_value()) {
    r1 = res1->numel() / n;
    TORCH_CHECK(res1->size(-1) == n && res1->scalar_type() == dt, "res1: mismatch");
    check_rows(*res1, r1, n, "res1");
    check_rows(*y1, r1, n, "y1");
    check_rows(*out1, r1, n, "out1");
  }
  const c10::cuda::CUDAGuard guard(res0.device());
#define GA_BODY(T)                                                             \
  {                                                                            \
    SegGA<T> s0{}, s1{};                                                       \
    s0.res = reinterpret_cast<const T *>(res0.const_data_ptr());               \
    s0.y = reinterpret_cast<const T *>(y0.const_data_ptr());                   \
    s0.out = reinterpret_cast<T *>(out0.data_ptr());                           \
    s0.gate = slice<T>(mod0, gate_off0, n);                                    \
    s0.rows = (int)r0;                                                         \
    if (r1 > 0) {                                                              \
      s1.res = reinterpret_cast<const T *>(res1->const_data_ptr());            \
      s1.y = reinterpret_cast<const T *>(y1->const_data_ptr());                \
      s1.out = reinterpret_cast<T *>(out1->data_ptr());                        \
      s1.gate = slice<T>(*mod1, gate_off1, n);                                 \
      s1.rows = (int)r1;                                                       \
    }                                                                          \
    run_gated_add<T>(s0, s1, (int)n);                                          \
  }
  if (dt == at::kBFloat16) GA_BODY(__nv_bfloat16) else GA_BODY(__half)
#undef GA_BODY
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_mod", &ln_mod, "LayerNorm + adaLN modulation (1-2 segments)");
  m.def("add_ln_mod", &add_ln_mod,
        "gated residual add, then LayerNorm + adaLN modulation (1-2 segments)");
  m.def("gated_add", &gated_add, "res + gate * y (1-2 segments)");
}
