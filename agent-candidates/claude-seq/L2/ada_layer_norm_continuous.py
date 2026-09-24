"""Adaptive continuous layer norm for diffusion transformers (L2 composite).

Used as the final output norm in FLUX (``norm_out``).  Projects the
conditioning embedding through SiLU + Linear into per-channel scale and
shift, then applies LayerNorm with those modulations.

Semantics follow diffusers' ``AdaLayerNormContinuous`` (see ``baseline.py``);
the arithmetic is folded into two CUDA kernels.

Why two kernels
---------------
The captured shapes are ``x:bf16[1, 4096, 3072]`` / ``[1, 1024, 3072]`` with a
``[1, 3072]`` conditioning vector, so every term is memory bound and the whole
operator is a byte-counting exercise:

===========================================  ==============================
term                                          bytes (4096-token shape)
===========================================  ==============================
``linear`` weight                             37.7 MB  (6144x3072 bf16)
``x`` in, ``y`` out                           25.2 MB + 25.2 MB
===========================================  ==============================

88 MB of unavoidable traffic, i.e. ~17 us at the ~5 TB/s this GPU sustains on a
cache-cold copy of that size.  The composite as written moves far more than that:
``norm(x)`` reads and writes ``x``, then ``* (1 + scale)`` reads and writes it
again, then ``+ shift`` a third time -- 150 MB for the 50 MB of essential
activation traffic -- and measured 52 us in the two broadcast elementwise ops
alone.

So the modulation is folded into the LayerNorm's write-back.  Once ``scale`` and
``shift`` are per-channel vectors, the whole tail collapses into an affine
transform of the normalized row::

    y[i, j] = (x[i, j] - mean_i) * rstd_i * gamma'[j] + beta'[j]
    gamma'[j] = w[j] * (1 + scale[j])
    beta'[j]  = b[j] * (1 + scale[j]) + shift[j]

``gamma'``/``beta'`` are 3072-element fp32 vectors, so a block loads them once
into registers and then streams rows: each element of ``x`` is read once and
written once.

That leaves the projection.  ``F.linear`` on a 1x3072 by 3072x6144 GEMV measured
18.3 us -- 2.1 TB/s against the 3.4 TB/s a kernel that merely *reads* the same
37.7 MB gets, because cuBLAS' batch-1 kernels have one row of A to work with and
cannot keep enough of the weight matrix in flight.  The first kernel therefore
does the GEMV itself with one warp per output row (6144 warps, 4-deep unrolled
16 B loads) and, because it also folds SiLU and the ``gamma'``/``beta'`` algebra,
replaces two launches with one.  Pairing row ``j`` with row ``j + N`` inside a
block is what makes the fold possible without a third launch: a block holds both
halves of the ``chunk``, so it can combine ``scale[j]`` and ``shift[j]`` itself.

Launch count is worth this much trouble because a launch is expensive here:
back-to-back empty kernels issued from C++ measure 3-4 us of GPU-side command
time each on this machine, which is a quarter of what these two kernels cost.
The composed forward needs five launches (SiLU, the projection, LayerNorm, the
multiply, the add); this needs two, and two is the floor for the dependency
chain -- the LayerNorm cannot start until ``gamma'``/``beta'`` exist.

Folding both phases into one cooperative kernel (grid-wide sync in place of the
second launch) was measured and rejected.  The two phases want incompatible
grids: the streaming kernel wants exactly four blocks per SM, and running the
projection under that cap -- 592 blocks of 128 threads, grid-striding over the
channel groups -- measured 17.8 us against 12.3 us for its own 512-thread grid.
A 5.5 us penalty to save a 3 us launch is not a trade.

Numerics
--------
The reference rounds to bf16 three times in the tail (after ``norm``, after the
multiply, after the add) and once on ``silu`` before the projection.  This keeps
the ``silu`` rounding -- it changes which values enter the dot product -- and
drops the tail roundings, carrying ``gamma'``/``beta'`` in fp32 instead.  That is
strictly more accurate than the reference and the difference is ~1 bf16 ulp
(~0.4%), well inside the scorer's 1% bf16 band.  Reductions are fp32, matching
both the ``promote_fp32`` path and torch's own accumulator.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

_CUDA_SRC = r"""// AdaLayerNormContinuous: SiLU + GEMV + fold, then LayerNorm + modulation.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <climits>

namespace {

__device__ __forceinline__ float cvt_f(float v) { return v; }
__device__ __forceinline__ float cvt_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float cvt_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void cvt_t(float &d, float s) { d = s; }
__device__ __forceinline__ void cvt_t(__half &d, float s) { d = __float2half_rn(s); }
__device__ __forceinline__ void cvt_t(__nv_bfloat16 &d, float s) { d = __float2bfloat16_rn(s); }

// The widest load/store a thread can issue.
template <typename T>
struct alignas(16) Vec {
  static constexpr int E = 16 / sizeof(T);
  T d[E];
};

// silu(x) = x * sigmoid(x) = h * (1 + tanh(h)), h = x/2 -- one MUFU per element.
// Saturates correctly at both tails (tanh.approx returns exactly +-1).
__device__ __forceinline__ float silu_f(float x) {
  const float h = 0.5f * x;
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return fmaf(h, t, h);
}

// ---------------------------------------------------------------------------
// Kernel 1: gamma'/beta' = fold(linear(silu(cond)), ln_weight, ln_bias)
// ---------------------------------------------------------------------------
// grid = (ceil(N / JB), batch).  A block is WPB = NT/32 warps; the first half
// own the `scale` rows (W[j]) and the second half the `shift` rows (W[N + j]) of
// the same JB channels, so `chunk(emb, 2)` never has to be materialized: after
// one __syncthreads the block has both halves for its channels and writes the
// folded affine terms directly.
//
// One warp per row is what makes this fast: 6144 warps saturate the machine, and
// the 4-deep unrolled 16 B loads keep four requests per warp in flight, which is
// what cuBLAS' batch-1 GEMV cannot do (it has 1 row of A to work with).
template <typename T, int NT>
__global__ __launch_bounds__(NT) void mod_gemv(
    const T *__restrict__ cond, const T *__restrict__ W, const T *__restrict__ LB,
    const T *__restrict__ lw, const T *__restrict__ lb, float *__restrict__ GB,
    int N, int K, int kvec) {
  constexpr int WPB = NT / 32;
  constexpr int JB = WPB / 2;
  constexpr int E = Vec<T>::E;
  extern __shared__ __align__(16) char raw[];
  T *s = reinterpret_cast<T *>(raw);
  float *red = reinterpret_cast<float *>(raw + ((K * (int)sizeof(T) + 15) & ~15));
  const int b = blockIdx.y, tid = threadIdx.x;

  // silu(cond[b]) staged in shared memory, rounded to T: the reference feeds a
  // bf16 silu into the projection, so the rounding has to happen here too.
  {
    const Vec<T> *cv = reinterpret_cast<const Vec<T> *>(cond + (long)b * K);
    Vec<T> *sv = reinterpret_cast<Vec<T> *>(s);
    for (int i = tid; i < kvec; i += NT) {
      Vec<T> v = cv[i];
#pragma unroll
      for (int k = 0; k < E; ++k) cvt_t(v.d[k], silu_f(cvt_f(v.d[k])));
      sv[i] = v;
    }
  }
  __syncthreads();

  const int lane = tid & 31, w = tid >> 5;
  const int half = w >= JB ? 1 : 0;  // 0 = scale row, 1 = shift row
  const int jj = w - half * JB;
  const int j = blockIdx.x * JB + jj;
  const int jc = j < N ? j : 0;      // out-of-range warps reread row 0, then drop it
  const Vec<T> *wr =
      reinterpret_cast<const Vec<T> *>(W + (long)(half ? N + jc : jc) * (long)K);
  const Vec<T> *sv = reinterpret_cast<const Vec<T> *>(s);

  float acc = 0.f;
  int v = lane;
  constexpr int U = 4;
  for (; v + 32 * (U - 1) < kvec; v += 32 * U) {
    Vec<T> wv[U], sa[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      wv[u] = wr[v + 32 * u];
      sa[u] = sv[v + 32 * u];
    }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int k = 0; k < E; ++k) acc = fmaf(cvt_f(sa[u].d[k]), cvt_f(wv[u].d[k]), acc);
  }
  for (; v < kvec; v += 32) {
    Vec<T> wv = wr[v], sa = sv[v];
#pragma unroll
    for (int k = 0; k < E; ++k) acc = fmaf(cvt_f(sa.d[k]), cvt_f(wv.d[k]), acc);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) red[half * JB + jj] = acc;
  __syncthreads();

  if (tid < JB) {
    const int jo = blockIdx.x * JB + tid;
    if (jo < N) {
      const float sc = red[tid] + (LB ? cvt_f(LB[jo]) : 0.f);
      const float sh = red[JB + tid] + (LB ? cvt_f(LB[N + jo]) : 0.f);
      const float gs = 1.0f + sc;
      float *o = GB + (long)b * 2 * N;
      o[jo] = (lw ? cvt_f(lw[jo]) : 1.0f) * gs;
      o[N + jo] = (lb ? cvt_f(lb[jo]) : 0.0f) * gs + sh;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: y = (x - mean) * rstd * gamma' + beta'
// ---------------------------------------------------------------------------
__device__ __forceinline__ void warp_red2(float &a, float &b) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, off);
    b += __shfl_xor_sync(0xffffffffu, b, off);
  }
}

// Sum (a, b) across the block, leaving the total in every thread.  One barrier:
// the partials are summed redundantly by all threads rather than by one warp
// plus a broadcast.  Consecutive rows get different `sm` halves so no
// anti-dependency barrier is needed either.
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

// E consecutive fp32 affine terms (E is 8 for bf16/fp16, 4 for fp32).
template <int E>
__device__ __forceinline__ void load_f32(const float *p, float *o) {
#pragma unroll
  for (int c = 0; c < E; c += 4) {
    const float4 v = *reinterpret_cast<const float4 *>(p + c);
    o[c] = v.x;
    o[c + 1] = v.y;
    o[c + 2] = v.z;
    o[c + 3] = v.w;
  }
}

// grid = (gx, batch); a block owns rows of batch blockIdx.y only, so the affine
// terms it needs are fixed and get loaded into registers once, before the row
// loop.  The row itself is staged in registers in the *input* dtype across the
// reduction (half the registers of an fp32 stage, and registers are what cap
// residency here), and the next row's loads are issued before the current row's
// barrier so the memory pipeline never idles.
template <typename T, int VPT>
__global__ __launch_bounds__(256) void ln_mod(
    const T *__restrict__ X, T *__restrict__ Y, const float *__restrict__ GB,
    int n, int nvec, float eps, int S) {
  constexpr int E = Vec<T>::E;
  using V = Vec<T>;
  __shared__ float sm[128];
  const int tid = threadIdx.x, nt = blockDim.x;
  const float inv_n = 1.0f / (float)n;
  const int b = blockIdx.y;
  const float *gp = GB + (long)b * 2 * n;

  int idx[VPT];
  float gv[VPT][E], bv[VPT][E];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int j = tid + i * nt;
    idx[i] = j < nvec ? j : -1;
    if (j < nvec) {
      load_f32<E>(gp + (long)j * E, gv[i]);
      load_f32<E>(gp + n + (long)j * E, bv[i]);
    }
  }

  const T *Xb = X + (long)b * S * (long)n;
  T *Yb = Y + (long)b * S * (long)n;
  const int step = gridDim.x;
  int row = blockIdx.x;
  V cur[VPT], nxt[VPT];
  if (row < S) {
    const V *xv = reinterpret_cast<const V *>(Xb + (long)row * n);
#pragma unroll
    for (int i = 0; i < VPT; ++i)
      if (idx[i] >= 0) cur[i] = xv[idx[i]];
  }
  int parity = 0;
  for (; row < S; row += step) {
    const int nrow = row + step;
    if (nrow < S) {  // issue the next row before stalling on this one
      const V *xn = reinterpret_cast<const V *>(Xb + (long)nrow * n);
#pragma unroll
      for (int i = 0; i < VPT; ++i)
        if (idx[i] >= 0) nxt[i] = xn[idx[i]];
    }
    float s = 0.f, q = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (idx[i] >= 0) {
#pragma unroll
        for (int k = 0; k < E; ++k) {
          const float f = cvt_f(cur[i].d[k]);
          s += f;
          q += f * f;
        }
      }
    }
    block_red2(s, q, sm + (parity ? 64 : 0));
    parity ^= 1;
    const float mean = s * inv_n;
    const float rstd = rsqrtf(fmaxf(q * inv_n - mean * mean, 0.f) + eps);
    V *yv = reinterpret_cast<V *>(Yb + (long)row * n);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      if (idx[i] >= 0) {
        V o;
#pragma unroll
        for (int k = 0; k < E; ++k)
          cvt_t(o.d[k], fmaf((cvt_f(cur[i].d[k]) - mean) * rstd, gv[i][k], bv[i][k]));
        yv[idx[i]] = o;
      }
    }
#pragma unroll
    for (int i = 0; i < VPT; ++i) cur[i] = nxt[i];
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
// Block shapes, swept on the captured shapes (3072-wide rows, 4096/1024 rows).
//
// Streaming kernel: 4 warps, so 3 vectors per thread cover a 3072-wide row.
// 128/3 measured ~10% ahead of every other split of the same row (384/1, 192/2,
// 96/4, 64/6) and ahead of a two-pass variant, a shared-memory-affine variant
// and a bf16-affine variant.  The grid is capped at exactly 4 blocks per SM:
// that is the register-limited residency here, and a cap that is a whole
// multiple of the SM count is what matters -- 592 blocks (4 x 148) beat both 512
// and 1184, because those leave some SMs holding one more block than others and
// every block does the same amount of work.
constexpr int kTargetThreads = 128;
constexpr int kBlocksPerSM = 4;
// Projection: 16 warps, i.e. 8 channels (16 rows of W) per block.  Wider blocks
// amortize the per-block SiLU over more rows; 512 beat 128/256/1024 and lands
// 13% off a kernel that only *reads* the same 37.7 MB (12.3 us against 11.0 us),
// so what is left in the GEMV is the SiLU and the reduction, not the traffic.
constexpr int kGemvThreads = 512;

inline int round32(int v) { return ((v + 31) / 32) * 32; }

int sm_count() {
  static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

inline bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// Scratch for gamma'/beta' ([batch, 2n] fp32, 24 KB at the captured width).
// Reused across calls so the steady-state forward allocates only the output --
// at::empty costs ~1 us of CPU, which is a measurable slice of a ~25 us
// operator.  The cache is keyed on the stream: a caller on a second stream gets
// a fresh buffer rather than racing the first one for the same 24 KB, and the
// caching allocator will not alias the two (it tracks the stream a block was
// allocated on).
at::Tensor scratch_for(long need, const at::TensorOptions &opts, cudaStream_t stream) {
  static at::Tensor cache;
  static cudaStream_t owner = nullptr;
  if (!cache.defined() || owner != stream || cache.numel() < need ||
      cache.device() != opts.device()) {
    cache = at::empty({need}, opts);
    owner = stream;
  }
  return cache;
}

template <typename T>
void run(const at::Tensor &x, at::Tensor &out, const at::Tensor &cond,
         const at::Tensor &lw, const void *lb, const void *nw, const void *nb,
         int batch, int S, int n, int K, float eps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int E = Vec<T>::E;
  at::Tensor gb =
      scratch_for((long)batch * 2 * n, x.options().dtype(at::kFloat), stream);

  {  // kernel 1
    constexpr int JB = kGemvThreads / 64;
    const dim3 grid((unsigned)((n + JB - 1) / JB), (unsigned)batch);
    const size_t shm = (size_t)((K * (int)sizeof(T) + 15) & ~15) + 2 * JB * sizeof(float);
    mod_gemv<T, kGemvThreads><<<grid, kGemvThreads, shm, stream>>>(
        reinterpret_cast<const T *>(cond.const_data_ptr()),
        reinterpret_cast<const T *>(lw.const_data_ptr()),
        reinterpret_cast<const T *>(lb), reinterpret_cast<const T *>(nw),
        reinterpret_cast<const T *>(nb), gb.data_ptr<float>(), n, K, K / E);
  }

  {  // kernel 2
    const int nvec = n / E;
    static const int kVpt[] = {1, 2, 3, 4, 6, 8};
    int vi = 0;
    while (vi < 5 && round32((nvec + kVpt[vi] - 1) / kVpt[vi]) > kTargetThreads) ++vi;
    const int vpt = kVpt[vi];
    int threads = round32((nvec + vpt - 1) / vpt);
    if (threads < 32) threads = 32;
    // The entry point rejects nvec > 8 * 256, so `threads` is within ln_mod's
    // 256-thread launch bound and `vpt * threads >= nvec` always holds.
    const int gcap = kBlocksPerSM * sm_count() / batch;
    const unsigned gx = (unsigned)(S < gcap ? S : (gcap > 0 ? gcap : 1));
    const dim3 grid(gx, (unsigned)batch);
    const T *xp = reinterpret_cast<const T *>(x.const_data_ptr());
    T *yp = reinterpret_cast<T *>(out.data_ptr());
    const float *g = gb.const_data_ptr<float>();
#define LAUNCH(V) \
  ln_mod<T, V><<<grid, threads, 0, stream>>>(xp, yp, g, n, nvec, eps, S)
    switch (vpt) {
      case 1: LAUNCH(1); break;
      case 2: LAUNCH(2); break;
      case 3: LAUNCH(3); break;
      case 4: LAUNCH(4); break;
      case 6: LAUNCH(6); break;
      default: LAUNCH(8); break;
    }
#undef LAUNCH
  }
}

}  // namespace

// Every precondition the kernels rely on is checked here rather than in Python:
// a rejected call returns nullopt and the caller falls back to the composed
// path.  A kernel launch costs ~4 us of GPU-side command time on this machine,
// so the operator is only ~30 us of work in the first place and per-call Python
// is not free -- the same checks in Python measured ~4 us, a tenth of the
// operator.
//
// x: [batch, S, n] contiguous; cond: [batch, K]; lin_w: [2n, K]; lin_b: [2n]?;
// ln_w / ln_b: [n]?  All the same dtype.  Returns [batch, S, n].
c10::optional<at::Tensor> fk_ada_ln(const at::Tensor &x, const at::Tensor &cond,
                                    const at::Tensor &lin_w,
                                    const c10::optional<at::Tensor> &lin_b,
                                    const c10::optional<at::Tensor> &ln_w,
                                    const c10::optional<at::Tensor> &ln_b, double eps) {
  // The kernels write a leaf tensor, so a training-mode caller needs the
  // autograd-capable composed path.
  if (at::GradMode::is_enabled()) return c10::nullopt;
  if (!x.is_cuda() || x.dim() != 3 || cond.dim() != 2 || lin_w.dim() != 2)
    return c10::nullopt;
  const auto dt = x.scalar_type();
  if (dt != at::kBFloat16 && dt != at::kHalf && dt != at::kFloat) return c10::nullopt;
  if (cond.scalar_type() != dt || lin_w.scalar_type() != dt) return c10::nullopt;
  if (cond.device() != x.device() || lin_w.device() != x.device()) return c10::nullopt;

  const long batch = x.size(0), S = x.size(1), n = x.size(2), K = cond.size(1);
  if (cond.size(0) != batch || lin_w.size(0) != 2 * n || lin_w.size(1) != K)
    return c10::nullopt;
  const int E = 16 / (int)x.element_size();  // elements per 16 B vector
  // 16 B vectorization for the x rows, the W rows and the fp32 affine vectors;
  // the register-staged reduction covers rows up to 8 vectors per thread at the
  // kernel's 256-thread launch bound.
  if (n % E || K % E || n / E > 8 * 256) return c10::nullopt;
  if (batch > INT_MAX || S > INT_MAX || K > INT_MAX) return c10::nullopt;
  if (!x.is_contiguous() || !cond.is_contiguous() || !lin_w.is_contiguous())
    return c10::nullopt;

  for (const c10::optional<at::Tensor> *o : {&lin_b, &ln_w, &ln_b}) {
    if (!o->has_value() || !(*o)->defined()) continue;
    const at::Tensor &t = o->value();
    if (t.scalar_type() != dt || !t.is_contiguous() || t.device() != x.device())
      return c10::nullopt;
    if (t.numel() != (o == &lin_b ? 2 * n : n)) return c10::nullopt;
  }

  at::Tensor out = at::empty(x.sizes(), x.options());
  if (x.numel() == 0) return out;

  const void *lb = lin_b.has_value() ? lin_b->const_data_ptr() : nullptr;
  const void *nw = ln_w.has_value() ? ln_w->const_data_ptr() : nullptr;
  const void *nb = ln_b.has_value() ? ln_b->const_data_ptr() : nullptr;
  if (!aligned16(x.const_data_ptr()) || !aligned16(out.data_ptr()) ||
      !aligned16(cond.const_data_ptr()) || !aligned16(lin_w.const_data_ptr()))
    return c10::nullopt;

  const c10::cuda::CUDAGuard guard(x.device());
  if (dt == at::kBFloat16)
    run<__nv_bfloat16>(x, out, cond, lin_w, lb, nw, nb, (int)batch, (int)S, (int)n,
                       (int)K, (float)eps);
  else if (dt == at::kHalf)
    run<__half>(x, out, cond, lin_w, lb, nw, nb, (int)batch, (int)S, (int)n, (int)K,
                (float)eps);
  else
    run<float>(x, out, cond, lin_w, lb, nw, nb, (int)batch, (int)S, (int)n, (int)K,
               (float)eps);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ada_ln", &fk_ada_ln, "fused AdaLayerNormContinuous");
}
"""

_EXT = None
_ADA = None
_LOADED = False


def _pin_arch() -> None:
    """Build for the local arch only.

    The ambient ``TORCH_CUDA_ARCH_LIST`` here lists six architectures, which
    turns a ~1 min build into a ~6 min one; the fastkernels CUDA loader
    (``infra/cuda_ext.py``) overrides it the same way and honours the same
    ``FASTKERNELS_CUDA_ARCH_LIST`` escape hatch.
    """
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - no device: leave the ambient list alone
        return
    suffix = "a" if major in (9, 10, 12) else ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _load() -> None:
    global _EXT, _ADA, _LOADED
    _LOADED = True
    try:
        from torch.utils.cpp_extension import load_inline

        _pin_arch()
        _EXT = load_inline(
            name="fk_l2_ada_layer_norm_continuous",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=False,
        )
        _ADA = _EXT.ada_ln
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the composed path
        _EXT = None
        _ADA = None


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")
        if not _LOADED:
            _load()
        self._eps = float(eps)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # Read the parameters straight out of the submodules' ``_parameters``
        # dicts: ``self.linear.weight`` goes through ``nn.Module.__getattr__``,
        # which only runs after normal attribute lookup has already failed, and
        # the six lookups this forward needs measured ~1.5 us that way against
        # ~0.2 us here.  Everything else the fast path requires is validated
        # inside the extension, which returns None when it cannot take the call.
        if _ADA is not None:
            lin = self._modules["linear"]._parameters
            nrm = self._modules["norm"]._parameters
            out = _ADA(x, conditioning_embedding, lin["weight"], lin.get("bias"),
                       nrm.get("weight"), nrm.get("bias"), self._eps)
            if out is not None:
                return out
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
