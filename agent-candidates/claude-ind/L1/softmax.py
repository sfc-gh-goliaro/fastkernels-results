"""Softmax / LogSoftmax activations.

``Softmax`` is a hand-written CUDA implementation.  The captured workload is a
mix of

* contiguous last-dim reductions (row lengths 77 / 400 / 512), and
* reductions along a *strided* axis of a permuted view -- e.g. YOLO's DFL
  ``[4, 16, 4, 8400]`` softmax over dim 1, or a permuted ``[1, 1, 8, 16, 16]``,

where ``F.softmax`` has to materialise a contiguous copy first (two kernels).
Everything here runs in a single launch that reads the strided input directly
and writes a contiguous output, so per-call latency is ~one kernel launch plus
one output allocation.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r'''
// Softmax forward: contiguous-row warp kernel + strided register kernel.
#include <ATen/ATen.h>
#include <ATen/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cfloat>

#define MAXND 6
#define MAXRANK 8

namespace {

constexpr int kSMTarget = 148;

template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Vec {
  T d[VEC];
};

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---------------------------------------------------------------------------
// Contiguous rows, reduce over the contiguous last dim: one warp per row.
// Needs ncols <= 32*VEC*ITER and ncols % VEC == 0.
// ---------------------------------------------------------------------------
template <typename T, int VEC, int ITER>
__global__ __launch_bounds__(256) void softmax_row(const T* __restrict__ in, T* __restrict__ out,
                                                   int ncols, long nrows) {
  using V = Vec<T, VEC>;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= nrows) return;
  const T* rp = in + row * (long)ncols;
  T* op = out + row * (long)ncols;

  float v[ITER][VEC];
  float m = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < ITER; ++i) {
    const int col = (i * 32 + lane) * VEC;
    if (col < ncols) {
      V t = *reinterpret_cast<const V*>(rp + col);
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        v[i][j] = static_cast<float>(t.d[j]);
        m = fmaxf(m, v[i][j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < VEC; ++j) v[i][j] = -FLT_MAX;
    }
  }
  m = warp_max(m);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < ITER; ++i)
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      v[i][j] = __expf(v[i][j] - m);
      s += v[i][j];
    }
  s = warp_sum(s);
  const float inv = 1.f / s;
#pragma unroll
  for (int i = 0; i < ITER; ++i) {
    const int col = (i * 32 + lane) * VEC;
    if (col < ncols) {
      V t;
#pragma unroll
      for (int j = 0; j < VEC; ++j) t.d[j] = static_cast<T>(v[i][j] * inv);
      *reinterpret_cast<V*>(op + col) = t;
    }
  }
}

// Long contiguous rows: one block per row, two passes over memory.
template <typename T, int NT>
__global__ __launch_bounds__(NT) void softmax_row_long(const T* __restrict__ in,
                                                       T* __restrict__ out, int ncols,
                                                       long nrows) {
  __shared__ float red[NT / 32];
  const T* rp = in + (long)blockIdx.x * ncols;
  T* op = out + (long)blockIdx.x * ncols;
  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < ncols; c += NT) m = fmaxf(m, static_cast<float>(rp[c]));
  m = warp_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int k = 1; k < NT / 32; ++k) m = fmaxf(m, red[k]);
  float s = 0.f;
  for (int c = threadIdx.x; c < ncols; c += NT) s += __expf(static_cast<float>(rp[c]) - m);
  s = warp_sum(s);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  s = red[0];
#pragma unroll
  for (int k = 1; k < NT / 32; ++k) s += red[k];
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < ncols; c += NT)
    op[c] = static_cast<T>(__expf(static_cast<float>(rp[c]) - m) * inv);
}

// ---------------------------------------------------------------------------
// Strided reduce: one thread per row, reduce dim held in registers.  The
// non-reduced dims are collapsed into NG groups; group 0 (smallest input
// stride) maps to blockIdx.x/threadIdx.x and groups 1..2 to blockIdx.y/z, so
// no integer division is needed.  32-bit offsets.
// ---------------------------------------------------------------------------
template <typename T, int R, int NG>
__global__ void softmax_strided(const T* __restrict__ in, T* __restrict__ out, int ri, int ro,
                                int n0, int i0, int o0, int i1, int o1, int i2, int o2) {
  const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c0 >= n0) return;
  int io = c0 * i0, oo = c0 * o0;
  if (NG > 1) {
    io += (int)blockIdx.y * i1;
    oo += (int)blockIdx.y * o1;
  }
  if (NG > 2) {
    io += (int)blockIdx.z * i2;
    oo += (int)blockIdx.z * o2;
  }
  float v[R];
  float m = -FLT_MAX;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    v[r] = static_cast<float>(in[io + r * ri]);
    m = fmaxf(m, v[r]);
  }
  float s = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    v[r] = __expf(v[r] - m);
    s += v[r];
  }
  const float inv = 1.f / s;
#pragma unroll
  for (int r = 0; r < R; ++r) out[oo + r * ro] = static_cast<T>(v[r] * inv);
}

// Same, runtime R (three passes over the strided row, no register array).
template <typename T, int NG>
__global__ void softmax_strided_dyn(const T* __restrict__ in, T* __restrict__ out, int R, int ri,
                                    int ro, int n0, int i0, int o0, int i1, int o1, int i2,
                                    int o2) {
  const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c0 >= n0) return;
  int io = c0 * i0, oo = c0 * o0;
  if (NG > 1) {
    io += (int)blockIdx.y * i1;
    oo += (int)blockIdx.y * o1;
  }
  if (NG > 2) {
    io += (int)blockIdx.z * i2;
    oo += (int)blockIdx.z * o2;
  }
  float m = -FLT_MAX;
  for (int r = 0; r < R; ++r) m = fmaxf(m, static_cast<float>(in[io + r * ri]));
  float s = 0.f;
  for (int r = 0; r < R; ++r) s += __expf(static_cast<float>(in[io + r * ri]) - m);
  const float inv = 1.f / s;
  for (int r = 0; r < R; ++r)
    out[oo + r * ro] = static_cast<T>(__expf(static_cast<float>(in[io + r * ri]) - m) * inv);
}

// Fully generic fallback: 64-bit offsets, runtime rank, one thread per row.
struct Dims {
  int size[MAXND];
  long istr[MAXND];
  long ostr[MAXND];
};

template <typename T>
__global__ void softmax_generic(const T* __restrict__ in, T* __restrict__ out, int R, long ri,
                                long ro, long nrows, int nd, Dims d) {
  long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) return;
  long io = 0, oo = 0;
  for (int k = 0; k < nd; ++k) {
    const long c = (k == nd - 1) ? row : row % d.size[k];
    if (k != nd - 1) row /= d.size[k];
    io += c * d.istr[k];
    oo += c * d.ostr[k];
  }
  float m = -FLT_MAX;
  for (int r = 0; r < R; ++r) m = fmaxf(m, static_cast<float>(in[io + r * ri]));
  float s = 0.f;
  for (int r = 0; r < R; ++r) s += __expf(static_cast<float>(in[io + r * ri]) - m);
  const float inv = 1.f / s;
  for (int r = 0; r < R; ++r)
    out[oo + r * ro] = static_cast<T>(__expf(static_cast<float>(in[io + r * ri]) - m) * inv);
}

// ---------------------------------------------------------------------------
// Launch plan.  Building it costs a few hundred ns of pointer chasing through
// the TensorImpl, which is a measurable slice of the ~4 us end-to-end cost of
// one call, so the last plan is memoized on the input's metadata.
// ---------------------------------------------------------------------------
enum Mode { MODE_ROW = 0, MODE_ROW_LONG = 1, MODE_STRIDED = 2, MODE_GENERIC = 3 };

struct Plan {
  int mode;
  int R;
  long nrows;
  // row
  int vec, iter, warps;
  // strided
  int ng, nt, n0, i0, o0, i1, o1, i2, o2, g1, g2, ri32, ro32;
  // generic
  long ri, ro;
  int nd;
  Dims d;
};

struct Key {
  int8_t valid, dtype, rank, dim;
  long sizes[MAXRANK];
  long strides[MAXRANK];
};

__attribute__((always_inline)) inline bool key_match(const Key& k, const at::Tensor& x, int dim,
                                                    int rank, int8_t dt) {
  if (!k.valid || k.rank != rank || k.dim != dim || k.dtype != dt) return false;
  const long* s = x.sizes().data();
  const long* t = x.strides().data();
  for (int i = 0; i < rank; ++i)
    if (k.sizes[i] != s[i] || k.strides[i] != t[i]) return false;
  return true;
}

// Pick (VEC, ITER) for a row of `ncols`: widest 16B-capable vector that divides
// ncols, then the smallest tile that covers the row.
inline bool pick_row(int ncols, int elem_size, int* vec, int* iter) {
  static const int kTiles[][2] = {{8, 1}, {8, 2}, {4, 1}, {4, 2}, {4, 3}, {8, 4}, {4, 4},
                                  {2, 1}, {2, 2}, {2, 4}, {4, 8}, {2, 8}, {1, 1}, {1, 2},
                                  {1, 3}, {1, 4}, {1, 8}};
  const int v16 = 16 / elem_size;
  for (auto& t : kTiles) {
    if (t[0] > v16 || ncols % t[0] || ncols > 32 * t[0] * t[1]) continue;
    *vec = t[0];
    *iter = t[1];
    return true;
  }
  return false;
}

bool build_plan(const at::Tensor& x, int dim, int rank, Plan& p) {
  p.R = static_cast<int>(x.size(dim));
  p.nrows = x.numel() / p.R;

  if (dim == rank - 1 && x.is_contiguous()) {
    if (pick_row(p.R, static_cast<int>(x.element_size()), &p.vec, &p.iter)) {
      p.mode = MODE_ROW;
      int w = 8;
      while (w > 1 && (p.nrows + w - 1) / w < kSMTarget) w >>= 1;
      p.warps = w;
    } else {
      p.mode = MODE_ROW_LONG;
    }
    return true;
  }

  // Collapse the non-reduced dims: fastest input stride first, carrying both
  // the input stride and the (contiguous) output stride.
  long sz[MAXRANK], is_[MAXRANK], os_[MAXRANK];
  int n = 0;
  {
    long ostr_all[MAXRANK];
    long acc = 1;
    for (int i = rank - 1; i >= 0; --i) {
      ostr_all[i] = acc;
      acc *= x.size(i);
    }
    for (int i = 0; i < rank; ++i) {
      if (i == dim || x.size(i) <= 1) continue;
      sz[n] = x.size(i);
      is_[n] = x.stride(i);
      os_[n] = ostr_all[i];
      ++n;
    }
  }
  for (int a = 1; a < n; ++a) {  // insertion sort by input stride, ascending
    const long s = sz[a], i1 = is_[a], o1 = os_[a];
    int b = a - 1;
    while (b >= 0 && is_[b] > i1) {
      sz[b + 1] = sz[b];
      is_[b + 1] = is_[b];
      os_[b + 1] = os_[b];
      --b;
    }
    sz[b + 1] = s;
    is_[b + 1] = i1;
    os_[b + 1] = o1;
  }
  int m = 0;
  for (int a = 0; a < n; ++a) {
    if (m > 0 && is_[a] == is_[m - 1] * sz[m - 1] && os_[a] == os_[m - 1] * sz[m - 1]) {
      sz[m - 1] *= sz[a];
    } else {
      sz[m] = sz[a];
      is_[m] = is_[a];
      os_[m] = os_[a];
      ++m;
    }
  }
  if (m == 0) {
    sz[0] = 1;
    is_[0] = 0;
    os_[0] = 0;
    m = 1;
  }
  p.ri = x.stride(dim);
  p.ro = 1;
  for (int i = rank - 1; i > dim; --i) p.ro *= x.size(i);

  long maxoff = (long)(p.R - 1) * (p.ri > p.ro ? p.ri : p.ro);
  for (int k = 0; k < m; ++k) {
    const long a = (sz[k] - 1) * is_[k], b = (sz[k] - 1) * os_[k];
    maxoff += a > b ? a : b;
  }
  const bool ok32 = maxoff < 0x7ffffff0L;
  const bool grid_ok = m <= 3 && (m < 2 || sz[1] <= 65535) && (m < 3 || sz[2] <= 65535);
  if (ok32 && grid_ok) {
    p.mode = MODE_STRIDED;
    p.ng = m;
    p.n0 = (int)sz[0];
    p.i0 = (int)is_[0];
    p.o0 = (int)os_[0];
    p.g1 = m > 1 ? (int)sz[1] : 1;
    p.i1 = m > 1 ? (int)is_[1] : 0;
    p.o1 = m > 1 ? (int)os_[1] : 0;
    p.g2 = m > 2 ? (int)sz[2] : 1;
    p.i2 = m > 2 ? (int)is_[2] : 0;
    p.o2 = m > 2 ? (int)os_[2] : 0;
    p.ri32 = (int)p.ri;
    p.ro32 = (int)p.ro;
    int nt = 256;
    while (nt > 32) {
      if ((long)((p.n0 + nt - 1) / nt) * p.g1 * p.g2 >= kSMTarget) break;
      nt >>= 1;
    }
    if (p.n0 < nt) nt = ((p.n0 + 31) / 32) * 32;
    p.nt = nt;
    return true;
  }
  if (m > MAXND) return false;
  p.mode = MODE_GENERIC;
  p.nd = m;
  for (int k = 0; k < m; ++k) {
    p.d.size[k] = (int)sz[k];
    p.d.istr[k] = is_[k];
    p.d.ostr[k] = os_[k];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
template <typename T>
void launch_row(const T* in, T* out, const Plan& p, cudaStream_t st) {
  const long blocks = (p.nrows + p.warps - 1) / p.warps;
  const int nt = 32 * p.warps;
#define ROW(V, I)                                                                \
  if (p.vec == V && p.iter == I) {                                               \
    softmax_row<T, V, I><<<blocks, nt, 0, st>>>(in, out, p.R, p.nrows);          \
    return;                                                                      \
  }
  if constexpr (sizeof(T) <= 2) {
    ROW(8, 1) ROW(8, 2) ROW(8, 4)
  }
  ROW(4, 1) ROW(4, 2) ROW(4, 3) ROW(4, 4) ROW(4, 8)
  ROW(2, 1) ROW(2, 2) ROW(2, 4) ROW(2, 8)
  ROW(1, 1) ROW(1, 2) ROW(1, 3) ROW(1, 4) ROW(1, 8)
#undef ROW
}

template <typename T, int NG>
void launch_strided_ng(const T* in, T* out, const Plan& p, cudaStream_t st) {
  const dim3 grid((p.n0 + p.nt - 1) / p.nt, NG > 1 ? p.g1 : 1, NG > 2 ? p.g2 : 1);
#define SR(n)                                                                                \
  case n:                                                                                    \
    softmax_strided<T, n, NG><<<grid, p.nt, 0, st>>>(in, out, p.ri32, p.ro32, p.n0, p.i0,     \
                                                     p.o0, p.i1, p.o1, p.i2, p.o2);          \
    return;
  switch (p.R) {
    SR(2) SR(3) SR(4) SR(5) SR(6) SR(7) SR(8) SR(16) SR(32)
    default:
      softmax_strided_dyn<T, NG><<<grid, p.nt, 0, st>>>(in, out, p.R, p.ri32, p.ro32, p.n0, p.i0,
                                                        p.o0, p.i1, p.o1, p.i2, p.o2);
      return;
  }
#undef SR
}

template <typename T>
void launch_plan(const T* in, T* out, const Plan& p, cudaStream_t st) {
  switch (p.mode) {
    case MODE_ROW:
      return launch_row<T>(in, out, p, st);
    case MODE_ROW_LONG:
      softmax_row_long<T, 512><<<p.nrows, 512, 0, st>>>(in, out, p.R, p.nrows);
      return;
    case MODE_STRIDED:
      if (p.ng == 1) return launch_strided_ng<T, 1>(in, out, p, st);
      if (p.ng == 2) return launch_strided_ng<T, 2>(in, out, p, st);
      return launch_strided_ng<T, 3>(in, out, p, st);
    default:
      softmax_generic<T><<<(p.nrows + 127) / 128, 128, 0, st>>>(in, out, p.R, p.ri, p.ro, p.nrows,
                                                                p.nd, p.d);
      return;
  }
}

}  // namespace

#define DTYPE_SWITCH(SCALAR_T, ...)                         \
  switch (SCALAR_T) {                                       \
    case at::kFloat: {                                      \
      using scalar_t = float;                               \
      __VA_ARGS__;                                          \
      break;                                                \
    }                                                       \
    case at::kHalf: {                                       \
      using scalar_t = at::Half;                            \
      __VA_ARGS__;                                          \
      break;                                                \
    }                                                       \
    case at::kBFloat16: {                                   \
      using scalar_t = at::BFloat16;                        \
      __VA_ARGS__;                                          \
      break;                                                \
    }                                                       \
    default:                                                \
      TORCH_CHECK(false, "softmax_fwd: unsupported dtype"); \
  }

at::Tensor softmax_fwd(const at::Tensor& x, int64_t dim_in) {
  static thread_local Key key{};
  static thread_local Plan plan{};

  const int rank = static_cast<int>(x.dim());
  int dim = static_cast<int>(dim_in < 0 ? dim_in + rank : dim_in);
  const auto sty = x.scalar_type();
  const bool supported = (sty == at::kFloat || sty == at::kHalf || sty == at::kBFloat16);
  // Anything outside the covered set (fp64, CPU, autograd, rank > 8) is bounced
  // back to the caller, which falls through to F.softmax.
  TORCH_CHECK(supported && x.is_cuda() && !x.requires_grad() && rank >= 1 &&
                  rank <= MAXRANK && dim >= 0 && dim < rank,
              "softmax_fwd: unsupported input");

  at::Tensor out = at::cuda::empty(x.sizes(), x.options());
  if (x.numel() == 0) return out;

  if (!key_match(key, x, dim, rank, (int8_t)sty)) {
    key.valid = 0;
    if (!build_plan(x, dim, rank, plan)) {  // rank > MAXND after collapsing
      at::Tensor xc = x.contiguous();
      at::Tensor o2 = softmax_fwd(xc, dim);
      out.copy_(o2);
      return out;
    }
    key.rank = (int8_t)rank;
    key.dim = (int8_t)dim;
    key.dtype = (int8_t)sty;
    for (int i = 0; i < rank; ++i) {
      key.sizes[i] = x.size(i);
      key.strides[i] = x.stride(i);
    }
    key.valid = 1;
  }

  auto st = c10::cuda::getCurrentCUDAStream();
  DTYPE_SWITCH(sty, launch_plan<scalar_t>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), plan,
                                          st));
  return out;
}
'''

_CPP_DECL = "at::Tensor softmax_fwd(const at::Tensor& x, int64_t dim);"


def _pin_arch() -> None:
    """Build only for the local arch; the default list compiles 7 of them."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    major, minor = torch.cuda.get_device_capability()
    # sm_90/100/120 need the 'a' (architecture-specific) variant.
    suffix = "a" if major in (9, 10, 12) else ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _build():
    from torch.utils.cpp_extension import load_inline

    _pin_arch()
    return load_inline(
        name="fk_l1_softmax_v1",
        cpp_sources=_CPP_DECL,
        cuda_sources=_CUDA_SRC,
        functions=["softmax_fwd"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


try:
    _softmax_fwd = _build().softmax_fwd
except Exception:  # no CUDA / no nvcc -> keep the reference path
    _softmax_fwd = None


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        self._fwd = _softmax_fwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self._fwd
        if fwd is not None:
            try:
                return fwd(x, self.dim)
            except Exception:
                pass  # dtype / layout the kernel does not cover
        return F.softmax(x, dim=self.dim)


class LogSoftmax(nn.Module):
    """Numerically-stable log-softmax. Used by the TTT-E2E inner-loop CE loss."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.dim)
