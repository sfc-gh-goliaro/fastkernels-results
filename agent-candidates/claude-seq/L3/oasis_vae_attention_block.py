"""Oasis VAE attention block -- the residual/LayerNorm plumbing fused away.

The block is ``x + attn(norm1(x))`` then ``h + mlp(norm2(h))``.  The three GEMMs
inside ``attn``/``mlp`` and the fused rotary+flash kernel are already the frozen
L1/L2 winners, and neither has slack left: a Triton tile loses to cuBLAS on all
four GEMM shapes here (0.63-0.87x), and sweeping the attention kernel's
tile/warp/stage space reproduces the frozen choice as the best of 108
configurations -- deleting its in-loop rotary entirely would only buy 21%, and
hoisting the rotary into its own pass costs more than that.  So what is left at
this level is the *glue*: two LayerNorms and two residual adds.  On the captured
shapes the glue is not a rounding error -- it was 30.8 us of the 224 us at
``[6, 576, 1024]`` and 39.1 us of the 131 us at ``[1, 576, 1024]``.

All of it is launch count and traffic, not arithmetic.  These tensors are 1.2 -
7 MB, small enough that a kernel's fixed cost is comparable to its payload: at
``[1, 576, 1024]`` a plain copy and a full LayerNorm measure the same, and the
block's wall time exceeds the sum of its kernels' by ~17 us either way.  Three
things follow, and this file does all three.

**1. Fold each residual add into the norm that consumes it.**  ``x + attn(...)``
was a standalone elementwise kernel that wrote ``h`` to DRAM for ``norm2`` to
read straight back.  :func:`res_norm_row` stages the row in registers once and
emits both ``h`` and ``norm(h) * w + b`` from that single load: 35 MB and two
launches become 28 MB and one.

**2. Transpose and normalize the channel-major capture in one launch.**  The
``[1, 576, 1024]`` capture has stride ``(589824, 1, 576)``, so every op that
touches ``x`` falls off torch's vectorized path onto the scalar
``elementwise_kernel<128, 4>``, and the frozen ``LayerNorm`` additionally calls
``.contiguous()`` on it -- the baseline composition pays for *two* permuted
copies of ``x`` plus two scalar-strided adds, 30.7 us of the 39.1 us, at
0.3 TB/s.  :func:`tr_ln` does the transpose, the reduction and the normalize
together; the comment on the kernel covers the two things that had to be right
before one launch beat two (CTA count when the reduction axis is the strided
one, and loads in flight).

**3. Let the fc2 GEMM close the second residual.**  ``out = h + fc2(g)`` is the
same contraction either way; handing cuBLAS ``h`` as the GEMM's ``C`` with
``beta = 1`` (an in-place ``addmm_``) retires the last elementwise kernel
instead of adding 21 MB of round trip after it.  The fc2 *bias* has to be inside
``h`` for that to be exact, so the norm kernel adds it to the ``h`` it writes --
after taking the statistics, which are ``norm2``'s and must not see it.  On the
captured shapes cuBLAS also picks a better kernel for the ``beta = 1`` form than
for the plain one, so the four GEMMs together get *cheaper*: 99.7 us against
103.3 us at ``[6, 576, 1024]``.

Per-kernel GPU time (CUPTI) and candidate latency (``python validate.py``):

    [6, 576, 1024]   10 kernels -> 8   glue 30.8 us -> 20.5 us   0.224 -> 0.217 ms
    [1, 576, 1024]   12 kernels -> 8   glue 39.1 us -> 16.9 us   0.131 -> 0.105 ms

The GPUs here are shared and their clocks unlocked, so absolute latency drifts
run to run -- the candidate by a few percent, the *baseline* by up to 40%
(0.33-0.59 ms on the same shape), which is why the numbers above are candidate
side and per kernel.  Over the runs measured the reported speedup went from
2.45x / 2.98x to 2.6-2.7x / 3.5-4.7x.

Numerics follow the baseline op-for-op: the residual add rounds to fp16 first
(torch's ``add`` computes in fp32 and rounds once) and the norm's statistics are
taken from that rounded value, in fp32, with fp32 affine -- which is what
``LayerNorm(promote_fp32=True)`` does.  The two deviations are the fc2 bias
(rounded into ``h`` rather than into the GEMM epilogue, ~1 fp16 ulp of ``h``)
and the fp32 reduction order.  Measured max |err| against the baseline is
3.9e-03 to 5.9e-03 across bench runs -- one fp16 ulp at the output magnitude --
against a bound of ``1e-2 + 1e-2*|y|`` on 99% of elements, which every element
clears.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_vae_attention import OasisVAEAttention

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------
_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<at::Tensor> fk_res_norm(const at::Tensor& x, const c10::optional<at::Tensor>& r,
                                    const c10::optional<at::Tensor>& w,
                                    const c10::optional<at::Tensor>& b, double eps,
                                    const c10::optional<at::Tensor>& hbias);
"""

_CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <vector>

namespace {

__device__ __forceinline__ float cvt_f(float v) { return v; }
__device__ __forceinline__ float cvt_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float cvt_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void cvt_t(float &d, float s) { d = s; }
__device__ __forceinline__ void cvt_t(__half &d, float s) { d = __float2half_rn(s); }
__device__ __forceinline__ void cvt_t(__nv_bfloat16 &d, float s) { d = __float2bfloat16_rn(s); }

// Widest load/store a thread can issue.
template <typename T>
struct alignas(16) Vec {
  static constexpr int E = 16 / sizeof(T);
  T d[E];
};

// ---------------------------------------------------------------------------
// h = x (+ r) (+ hbias);  n = ((h - mean) * rstd) * w + b   -- one warp per row
//
// ``hbias`` lands in the stored ``h`` only, *after* the statistics are taken, so
// the norm sees the true residual while the tensor left in memory is already
// the ``C`` operand the next GEMM wants.
//
// The row is staged in registers *in the input dtype* between the reduction and
// the write-back, so it is loaded once; fp16 staging (rather than fp32) halves
// the registers per element, and registers are what cap residency here.  The
// reduction is warp-local, so there is no shared memory and no __syncthreads
// anywhere in the kernel -- which is the point of one warp per row rather than
// one block per row: at n = 1024 a block-wide reduction's barrier costs more
// than the 32 lanes' worth of loads it would add.
// ---------------------------------------------------------------------------
template <typename T, int VPT, bool HAS_R, bool HAS_HB>
__global__ __launch_bounds__(128) void res_norm_row(
    const T *__restrict__ X, const T *__restrict__ R, const T *__restrict__ HB,
    T *__restrict__ H, T *__restrict__ N, const T *__restrict__ W,
    const T *__restrict__ B, int n, int nvec, float eps, long rows) {
  using V = Vec<T>;
  constexpr int E = V::E;
  const int lane = threadIdx.x & 31;
  const int wpb = blockDim.x >> 5;
  const long gstride = (long)gridDim.x * wpb;
  const float inv_n = 1.f / (float)n;
  const V *WV = reinterpret_cast<const V *>(W);
  const V *BV = reinterpret_cast<const V *>(B);
  const V *HBV = reinterpret_cast<const V *>(HB);

  for (long row = (long)blockIdx.x * wpb + (threadIdx.x >> 5); row < rows;
       row += gstride) {
    const V *xv = reinterpret_cast<const V *>(X + row * (long)n);
    V h[VPT];
    float s = 0.f, q = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int j = lane + i * 32;
      if (j < nvec) {
        h[i] = xv[j];
        if (HAS_R) {
          const V rv = reinterpret_cast<const V *>(R + row * (long)n)[j];
#pragma unroll
          for (int k = 0; k < E; ++k)
            cvt_t(h[i].d[k], cvt_f(h[i].d[k]) + cvt_f(rv.d[k]));
        }
#pragma unroll
        for (int k = 0; k < E; ++k) {
          const float f = cvt_f(h[i].d[k]);
          s += f;
          q += f * f;
        }
      }
    }
    if (H) {
      V *hv = reinterpret_cast<V *>(H + row * (long)n);
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        const int j = lane + i * 32;
        if (j < nvec) {
          if (HAS_HB) {
            V o = HBV[j];
#pragma unroll
            for (int k = 0; k < E; ++k)
              cvt_t(o.d[k], cvt_f(h[i].d[k]) + cvt_f(o.d[k]));
            hv[j] = o;
          } else {
            hv[j] = h[i];
          }
        }
      }
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      s += __shfl_xor_sync(0xffffffffu, s, off);
      q += __shfl_xor_sync(0xffffffffu, q, off);
    }
    const float mean = s * inv_n;
    const float rstd = rsqrtf(fmaxf(q * inv_n - mean * mean, 0.f) + eps);
    V *nv = reinterpret_cast<V *>(N + row * (long)n);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const int j = lane + i * 32;
      if (j < nvec) {
        V wv, bv, o;
        if (W) wv = WV[j];
        if (B) bv = BV[j];
#pragma unroll
        for (int k = 0; k < E; ++k) {
          float f = (cvt_f(h[i].d[k]) - mean) * rstd;
          if (W) f *= cvt_f(wv.d[k]);
          if (B) f += cvt_f(bv.d[k]);
          cvt_t(o.d[k], f);
        }
        nv[j] = o;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Channel-major x -> contiguous h + its LayerNorm, in one launch.
//
// x's (s, c) plane is stored as [C][S]: s is the contiguous axis, and the norm
// reduces along c.  Two things fall out of that, and both had to be handled
// before one launch beat two.
//
// *Parallelism.*  There are only S/BS = 18 row tiles, and splitting c across
// CTAs would need a second launch to join the partial sums -- which is exactly
// the launch this kernel exists to remove.  So every CTA reduces the *whole* c
// range for its BS rows and writes only the BC-wide slice it owns: C/BC times
// the reads, but 144 CTAs instead of 18, and the redundant reads are L2 hits
// (9 MB of L2 against 1.2 MB of DRAM).
//
// *Loads in flight.*  The obvious mapping -- one thread per (s, c) element,
// stepping along c -- issues 1024 dependent 2-byte loads per thread and measured
// 17.6 us, i.e. ~230 cycles per iteration with nothing overlapping: at 144 CTAs
// there is one CTA per SM, so eight warps have to cover all the latency there
// is.  Instead each thread takes E consecutive *s* values as one 16 B load and
// carries E running sums, which cuts the loop to C/(NT/SG) = 16 iterations of
// 16 B and lets the pipeline fill.  The per-(s, c) sums then live one per
// register rather than one per thread, so the cross-thread reduction is a
// shuffle over the lanes that share an s group.
//
// The store phase re-reads x instead of staging the slice in shared memory: it
// needs the transpose (8 consecutive c for one s), and the shared-memory layout
// that makes those reads conflict-free is exactly the one that makes the
// staging writes conflict-heavy.  Re-reading costs ~9 MB of L2 and no barrier.
// ---------------------------------------------------------------------------
template <typename T, int BS, int BC, int NT>
__global__ __launch_bounds__(NT) void tr_ln(
    const T *__restrict__ X, const T *__restrict__ HB, T *__restrict__ H,
    T *__restrict__ N, const T *__restrict__ W, const T *__restrict__ B, int S,
    int C, long sbx, float eps) {
  using V = Vec<T>;
  constexpr int E = V::E;
  constexpr int SG = BS / E;   // s groups per row tile
  constexpr int CG = NT / SG;  // c rows visited per iteration
  constexpr int NW = NT / 32;
  constexpr int TPR = BC / E;  // store threads per row
  __shared__ float red[2][NW][BS];
  __shared__ float ms[2][BS];

  const int tid = threadIdx.x;
  const int sg = tid % SG, cg = tid / SG;
  const int s0 = blockIdx.y * BS, c0 = blockIdx.x * BC;
  const T *xb = X + (long)blockIdx.z * sbx;

  float sum[E], sq[E];
#pragma unroll
  for (int k = 0; k < E; ++k) {
    sum[k] = 0.f;
    sq[k] = 0.f;
  }
  const T *xs = xb + s0 + sg * E;
  for (int c = cg; c < C; c += CG) {
    const V v = *reinterpret_cast<const V *>(xs + (long)c * S);
#pragma unroll
    for (int k = 0; k < E; ++k) {
      const float f = cvt_f(v.d[k]);
      sum[k] += f;
      sq[k] += f * f;
    }
  }
  // Lanes sharing an s group differ only in the lane bits above SG.
#pragma unroll
  for (int k = 0; k < E; ++k) {
#pragma unroll
    for (int off = SG; off < 32; off <<= 1) {
      sum[k] += __shfl_xor_sync(0xffffffffu, sum[k], off);
      sq[k] += __shfl_xor_sync(0xffffffffu, sq[k], off);
    }
  }
  if ((tid & 31) < SG) {
    const int wid = tid >> 5;
#pragma unroll
    for (int k = 0; k < E; ++k) {
      red[0][wid][sg * E + k] = sum[k];
      red[1][wid][sg * E + k] = sq[k];
    }
  }
  __syncthreads();
  if (tid < BS) {
    float a = 0.f, b2 = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) {
      a += red[0][i][tid];
      b2 += red[1][i][tid];
    }
    const float inv_n = 1.f / (float)C;
    const float mean = a * inv_n;
    ms[0][tid] = mean;
    ms[1][tid] = rsqrtf(fmaxf(b2 * inv_n - mean * mean, 0.f) + eps);
  }
  __syncthreads();

  const V *WV = reinterpret_cast<const V *>(W);
  const V *BV = reinterpret_cast<const V *>(B);
  const V *HBV = reinterpret_cast<const V *>(HB);
  const long hbase = (long)blockIdx.z * S * C;
  for (int j = tid; j < BS * TPR; j += NT) {
    const int sw = j / TPR, cw = (j % TPR) * E;
    const int jv = (c0 + cw) / E;
    const float mean = ms[0][sw], rstd = ms[1][sw];
    V ho, no, wv, bv, hbv;
    if (W) wv = WV[jv];
    if (B) bv = BV[jv];
    if (HB) hbv = HBV[jv];
    const T *xr = xb + (long)(c0 + cw) * S + s0 + sw;
#pragma unroll
    for (int k = 0; k < E; ++k) {
      const float f = cvt_f(xr[(long)k * S]);
      float g = f;
      if (HB) g += cvt_f(hbv.d[k]);
      cvt_t(ho.d[k], g);
      float u = (f - mean) * rstd;
      if (W) u *= cvt_f(wv.d[k]);
      if (B) u += cvt_f(bv.d[k]);
      cvt_t(no.d[k], u);
    }
    const long off = hbase + (long)(s0 + sw) * C + c0 + cw;
    *reinterpret_cast<V *>(H + off) = ho;
    *reinterpret_cast<V *>(N + off) = no;
  }
}

// ---------------------------------------------------------------------------
// h[b][s][c] = x[b][s][c] (+ r) (+ hbias) for a channel-major x.  Used only for
// the residual-on-a-channel-major-x case, which the captures do not hit; the
// classic 32x32 shared tile keeps both the global read and the global write
// contiguous in their own fast axis.
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(256) void tr_add(const T *__restrict__ X,
                                              const T *__restrict__ R,
                                              const T *__restrict__ HB,
                                              T *__restrict__ H, int S, int C,
                                              long sbx) {
  __shared__ T tile[32][33];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int c0 = blockIdx.x * 32, s0 = blockIdx.y * 32;
  const T *xb = X + (long)blockIdx.z * sbx;
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    const int c = c0 + ty + k, s = s0 + tx;
    if (c < C && s < S) tile[tx][ty + k] = xb[(long)c * S + s];
  }
  __syncthreads();
  const long hb = (long)blockIdx.z * S * C;
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    const int s = s0 + ty + k, c = c0 + tx;
    if (s < S && c < C) {
      const long o = hb + (long)s * C + c;
      float v = cvt_f(tile[ty + k][tx]);
      if (R) v += cvt_f(R[o]);
      if (HB) v += cvt_f(HB[c]);
      cvt_t(H[o], v);
    }
  }
}

// ---------------------------------------------------------------------------
// tail: any n, any alignment.  One block per row, two passes over the row.
// ---------------------------------------------------------------------------
template <typename T, bool HAS_R, bool HAS_HB>
__global__ __launch_bounds__(256) void res_norm_gen(
    const T *__restrict__ X, const T *__restrict__ R, const T *__restrict__ HB,
    T *__restrict__ H, T *__restrict__ N, const T *__restrict__ W,
    const T *__restrict__ B, int n, float eps, long rows) {
  __shared__ float sm[2][8];
  const int tid = threadIdx.x, nt = blockDim.x, nw = nt >> 5;
  const float inv_n = 1.f / (float)n;
  for (long row = (long)blockIdx.x; row < rows; row += (long)gridDim.x) {
    const T *xr = X + row * (long)n;
    const T *rr = HAS_R ? R + row * (long)n : nullptr;
    float s = 0.f, q = 0.f;
    for (int i = tid; i < n; i += nt) {
      T t = xr[i];
      if (HAS_R) cvt_t(t, cvt_f(t) + cvt_f(rr[i]));
      const float f = cvt_f(t);
      s += f;
      q += f * f;
      if (H) {
        float g = f;
        if (HAS_HB) g += cvt_f(HB[i]);
        cvt_t(H[row * (long)n + i], g);
      }
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      s += __shfl_xor_sync(0xffffffffu, s, off);
      q += __shfl_xor_sync(0xffffffffu, q, off);
    }
    if (nw > 1) {
      if ((tid & 31) == 0) {
        sm[0][tid >> 5] = s;
        sm[1][tid >> 5] = q;
      }
      __syncthreads();
      s = 0.f;
      q = 0.f;
      for (int i = 0; i < nw; ++i) {
        s += sm[0][i];
        q += sm[1][i];
      }
    }
    const float mean = s * inv_n;
    const float rstd = rsqrtf(fmaxf(q * inv_n - mean * mean, 0.f) + eps);
    T *nr = N + row * (long)n;
    for (int i = tid; i < n; i += nt) {
      float f = cvt_f(xr[i]);
      if (HAS_R) {
        T t;
        cvt_t(t, f + cvt_f(rr[i]));
        f = cvt_f(t);
      }
      f = (f - mean) * rstd;
      if (W) f *= cvt_f(W[i]);
      if (B) f += cvt_f(B[i]);
      cvt_t(nr[i], f);
    }
    if (nw > 1) __syncthreads();
  }
}

int sm_count() {
  static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

inline bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// One warp per row, four warps per block: the grid is then wide enough to fill
// the machine at 576 rows and still only ~6 blocks/SM at 3456.
constexpr int kWarpsPerBlock = 4;

template <typename T, bool HAS_R, bool HAS_HB>
void launch_row(const T *x, const T *r, const T *hb, T *h, T *nn, const T *w,
                const T *b, int n, long rows, float eps, bool vec_ok,
                cudaStream_t stream) {
  const int nvec = n / Vec<T>::E;
  if (vec_ok && nvec <= 32 * 8) {
    const unsigned grid = (unsigned)((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
#define LAUNCH(VP)                                                             \
  res_norm_row<T, VP, HAS_R, HAS_HB><<<grid, 32 * kWarpsPerBlock, 0, stream>>>( \
      x, r, hb, h, nn, w, b, n, nvec, eps, rows)
    if (nvec <= 32) {
      LAUNCH(1);
    } else if (nvec <= 64) {
      LAUNCH(2);
    } else if (nvec <= 96) {
      LAUNCH(3);
    } else if (nvec <= 128) {
      LAUNCH(4);
    } else if (nvec <= 192) {
      LAUNCH(6);
    } else {
      LAUNCH(8);
    }
#undef LAUNCH
    return;
  }
  const long gcap = 8L * sm_count();
  const unsigned grid = (unsigned)(rows < gcap ? rows : gcap);
  int threads = n < 256 ? ((n + 31) / 32) * 32 : 256;
  if (threads < 32) threads = 32;
  res_norm_gen<T, HAS_R, HAS_HB>
      <<<grid, threads, 0, stream>>>(x, r, hb, h, nn, w, b, n, eps, rows);
}

template <typename T>
void launch_row_d(const T *x, const T *r, const T *hb, T *h, T *nn, const T *w,
                  const T *b, int n, long rows, float eps, bool vec_ok,
                  cudaStream_t stream) {
  if (r) {
    if (hb)
      launch_row<T, true, true>(x, r, hb, h, nn, w, b, n, rows, eps, vec_ok, stream);
    else
      launch_row<T, true, false>(x, r, hb, h, nn, w, b, n, rows, eps, vec_ok, stream);
  } else if (hb) {
    launch_row<T, false, true>(x, r, hb, h, nn, w, b, n, rows, eps, vec_ok, stream);
  } else {
    launch_row<T, false, false>(x, r, hb, h, nn, w, b, n, rows, eps, vec_ok, stream);
  }
}

// 32 rows x 128 channels per CTA: 144 CTAs on the captured [1, 576, 1024].
constexpr int kTrBS = 32, kTrBC = 128, kTrNT = 256;

template <typename T>
void launch_tr_ln(const T *x, const T *hb, T *h, T *nn, const T *w, const T *b,
                  int S, int C, long sbx, int batch, float eps,
                  cudaStream_t stream) {
  dim3 grid(C / kTrBC, (S + kTrBS - 1) / kTrBS, (unsigned)batch);
  tr_ln<T, kTrBS, kTrBC, kTrNT>
      <<<grid, kTrNT, 0, stream>>>(x, hb, h, nn, w, b, S, C, sbx, eps);
}

const at::Tensor *opt(const c10::optional<at::Tensor> &o) {
  if (o.has_value() && o->defined()) return &o.value();
  return nullptr;
}

// x is [B, S, C] with the (S, C) plane laid out as [C][S].
bool channel_major(const at::Tensor &x) {
  if (x.dim() != 3) return false;
  const long S = x.size(1), C = x.size(2);
  return S > 0 && C > 0 && x.stride(1) == 1 && x.stride(2) == S &&
         x.stride(0) == S * C;
}

}  // namespace

// Returns {h, n} with h = x (+ r) (+ hbias) contiguous and n = LN(x + r).
// h aliases x when there is nothing to add to it and x is already contiguous.
std::vector<at::Tensor> fk_res_norm(const at::Tensor &x_in,
                                    const c10::optional<at::Tensor> &r_opt,
                                    const c10::optional<at::Tensor> &w_opt,
                                    const c10::optional<at::Tensor> &b_opt,
                                    double eps,
                                    const c10::optional<at::Tensor> &hb_opt) {
  TORCH_CHECK(x_in.is_cuda(), "res_norm: cuda only");
  const auto dt = x_in.scalar_type();
  TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16 || dt == at::kFloat,
              "res_norm: unsupported dtype");
  const at::Tensor *r = opt(r_opt);
  const at::Tensor *w = opt(w_opt);
  const at::Tensor *b = opt(b_opt);
  const at::Tensor *hbt = opt(hb_opt);
  const int n = (int)x_in.size(-1);
  const long rows = n ? x_in.numel() / n : 0;
  const c10::cuda::CUDAGuard guard(x_in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  at::Tensor wc, bc, hbc;
  const void *wp = nullptr, *bp = nullptr, *hbp = nullptr;
  if (w) {
    TORCH_CHECK(w->numel() == n && w->scalar_type() == dt, "res_norm: bad weight");
    wc = w->is_contiguous() ? *w : w->contiguous();
    wp = wc.const_data_ptr();
  }
  if (b) {
    TORCH_CHECK(b->numel() == n && b->scalar_type() == dt, "res_norm: bad bias");
    bc = b->is_contiguous() ? *b : b->contiguous();
    bp = bc.const_data_ptr();
  }
  if (hbt) {
    TORCH_CHECK(hbt->numel() == n && hbt->scalar_type() == dt, "res_norm: bad hbias");
    hbc = hbt->is_contiguous() ? *hbt : hbt->contiguous();
    hbp = hbc.const_data_ptr();
  }
  if (r) {
    TORCH_CHECK(r->is_contiguous() && r->scalar_type() == dt &&
                    r->numel() == x_in.numel(),
                "res_norm: residual must be contiguous and match x");
  }

  at::Tensor nrm = at::empty(x_in.sizes(), x_in.options());
  const int E = dt == at::kFloat ? 4 : 8;

  // -- channel-major x, nothing to add: transpose + norm in one launch -------
  if (!x_in.is_contiguous() && !r && channel_major(x_in) && dt != at::kFloat &&
      (int)x_in.size(2) % kTrBC == 0 && (int)x_in.size(1) % kTrBS == 0 &&
      n % E == 0 &&
      aligned16(x_in.const_data_ptr()) && (!wp || aligned16(wp)) &&
      (!bp || aligned16(bp)) && (!hbp || aligned16(hbp))) {
    at::Tensor h = at::empty(x_in.sizes(), x_in.options());
    const int S = (int)x_in.size(1), C = (int)x_in.size(2);
    if (rows) {
#define DISP_TR(T)                                                             \
  launch_tr_ln<T>((const T *)x_in.const_data_ptr(), (const T *)hbp,            \
                  (T *)h.data_ptr(), (T *)nrm.data_ptr(), (const T *)wp,       \
                  (const T *)bp, S, C, x_in.stride(0), (int)x_in.size(0),      \
                  (float)eps, stream)
      if (dt == at::kHalf) {
        DISP_TR(__half);
      } else {
        DISP_TR(__nv_bfloat16);
      }
#undef DISP_TR
    }
    return {h, nrm};
  }

  at::Tensor x = x_in;
  at::Tensor h;
  bool folded = false;  // the add already happened during the transpose
  if (!x_in.is_contiguous()) {
    if (channel_major(x_in) && dt != at::kFloat) {
      h = at::empty(x_in.sizes(), x_in.options());
      const int S = (int)x_in.size(1), C = (int)x_in.size(2);
      dim3 grid((C + 31) / 32, (S + 31) / 32, (unsigned)x_in.size(0));
      dim3 blk(32, 8);
#define DISP_TA(T)                                                             \
  tr_add<T><<<grid, blk, 0, stream>>>(                                         \
      (const T *)x_in.const_data_ptr(),                                        \
      r ? (const T *)r->const_data_ptr() : nullptr, (const T *)hbp,            \
      (T *)h.data_ptr(), S, C, x_in.stride(0))
      if (dt == at::kHalf) {
        DISP_TA(__half);
      } else {
        DISP_TA(__nv_bfloat16);
      }
#undef DISP_TA
      x = h;
      folded = true;
    } else {
      x = x_in.contiguous();
    }
  }
  const bool has_r = r && !folded;
  const bool has_hb = hbp && !folded;
  if (!folded) h = (has_r || has_hb) ? at::empty(x_in.sizes(), x_in.options()) : x;
  if (!rows || !n) return {h, nrm};

  const void *rp = has_r ? r->const_data_ptr() : nullptr;
  const bool vec_ok = (n % E == 0) && aligned16(x.const_data_ptr()) &&
                      aligned16(nrm.data_ptr()) && (!wp || aligned16(wp)) &&
                      (!bp || aligned16(bp)) && (!rp || aligned16(rp)) &&
                      (!has_hb || aligned16(hbp)) && aligned16(h.data_ptr());
  // h is only written when something has to be added to it; otherwise it is x.
  const bool write_h = has_r || has_hb;

#define DISPATCH(T)                                                            \
  launch_row_d<T>((const T *)x.const_data_ptr(), (const T *)rp,                \
                  has_hb ? (const T *)hbp : nullptr,                           \
                  write_h ? (T *)h.data_ptr() : nullptr, (T *)nrm.data_ptr(),  \
                  (const T *)wp, (const T *)bp, n, rows, (float)eps, vec_ok,   \
                  stream)
  if (dt == at::kHalf) {
    DISPATCH(__half);
  } else if (dt == at::kBFloat16) {
    DISPATCH(__nv_bfloat16);
  } else {
    DISPATCH(float);
  }
#undef DISPATCH
  return {h, nrm};
}
"""


def _load_ext():
    from torch.utils.cpp_extension import load_inline

    if not torch.cuda.is_available():
        return None
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is None:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + (
            "a" if major in (9, 10, 12) else ""
        )
    elif override.strip():
        os.environ["TORCH_CUDA_ARCH_LIST"] = override
    try:
        return load_inline(
            name="fk_oasis_vae_block_ext",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["fk_res_norm"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _EXT = _load_ext()
except Exception:  # pragma: no cover - fall back to the composed path
    _EXT = None

_DTYPES = (torch.float16, torch.bfloat16)


class OasisVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)

    def _norm_ok(self, norm, x) -> bool:
        return (
            norm.weight is not None
            and norm.bias is not None
            and norm.weight.dtype == x.dtype
            and norm.normalized_shape == (x.shape[-1],)
        )

    def _fast_ok(self, x: torch.Tensor) -> bool:
        return (
            _EXT is not None
            and x.is_cuda
            and x.dtype in _DTYPES
            and x.dim() == 3
            and x.shape[-1] % 8 == 0
            and self._norm_ok(self.norm1, x)
            and self._norm_ok(self.norm2, x)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fast_ok(x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        mlp, fc2 = self.mlp, self.mlp.fc2
        # Close the second residual in fc2's GEMM epilogue (beta = 1), which
        # needs fc2's bias to already be inside h.
        w2 = fc2.weight
        fold = (
            fc2.bias is not None
            and w2.dim() == 2
            and w2.stride(1) == 1
            and w2.shape[0] == x.shape[-1]
            and fc2.bias.dtype == x.dtype
            and w2.dtype == x.dtype
        )
        h1, n1 = _EXT.fk_res_norm(
            x, None, self.norm1.weight, self.norm1.bias, self.norm1.eps, None
        )
        h2, n2 = _EXT.fk_res_norm(
            h1,
            self.attn(n1),
            self.norm2.weight,
            self.norm2.bias,
            self.norm2.eps,
            fc2.bias if fold else None,
        )
        g = mlp.act(mlp.fc1(n2))
        if not fold:
            return h2 + fc2(g)
        h2.view(-1, h2.shape[-1]).addmm_(g.reshape(-1, g.shape[-1]), w2.t())
        return h2
