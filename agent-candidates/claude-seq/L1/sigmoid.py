"""Sigmoid activation: 1 / (1 + exp(-x)).

Hand-written CUDA elementwise kernel. The captured workload is a handful of tiny
tensors (8K-200K elements, contiguous) plus one 2.7M-element fp16 tensor that is
a *strided* view (a YOLO head slice, ``stride=[1209600, 8400, 1]``), so the op is
launch/latency bound at the small end and bandwidth bound at the large end.

Three things buy the speedup over ``torch.sigmoid``:

* 16-byte vectorized load -> ``ex2.approx`` sigmoid -> 16-byte store, single
  pass, none of the TensorIterator setup;
* a strided input is handled in place by iterating (outer dims) x (contiguous
  inner run) with the same 16-byte vectors, so no ``contiguous()`` copy and no
  per-element offset arithmetic;
* programmatic dependent launch (sm_90+): the grid is scheduled while the
  preceding kernel in the stream is still draining, and blocks call
  ``cudaGridDependencySynchronize()`` before touching the input, so launch
  latency hides behind the producer instead of adding to it.
"""

from __future__ import annotations

import os
import subprocess

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CPP = r"""
at::Tensor fk_sigmoid(const at::Tensor& x);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>

// ---------------------------------------------------------------------------
// sigmoid(x) = 1/(1+exp(-x)), evaluated with the hardware approximations:
// __expf is one ex2.approx (plus the log2(e) scale) and __fdividef is one
// rcp.approx. Relative error ~1e-6, far tighter than the 1e-2 fp16/bf16
// tolerance, and it saturates correctly at both ends (x -> -inf gives 0,
// x -> +inf gives 1).
// ---------------------------------------------------------------------------
template <typename T> struct FkAcc { using type = float; };
template <> struct FkAcc<double> { using type = double; };

template <typename T>
__device__ __forceinline__ T fk_sig(T v) {
  using C = typename FkAcc<T>::type;
  if constexpr (std::is_same<C, double>::value) return (T)(1.0 / (1.0 + exp(-(double)v)));
  else return (T)__fdividef(1.0f, 1.0f + __expf(-(float)v));
}

template <typename T>
__device__ __forceinline__ void fk_sig16(float4& v) {
  T* p = reinterpret_cast<T*>(&v);
#pragma unroll
  for (int j = 0; j < 16 / (int)sizeof(T); ++j) p[j] = fk_sig(p[j]);
}

// Up to 4 collapsed outer dims of a strided input; the inner run is contiguous.
struct FkOuter { long size[4]; long stride[4]; int ndim; };

// --------------------------- contiguous input ------------------------------
template <typename T, int NT, bool PDL>
__global__ __launch_bounds__(NT) void fk_flat_vec(const T* __restrict__ in,
                                                  T* __restrict__ out, long n) {
  constexpr int VEC = 16 / (int)sizeof(T);
  long i = ((long)blockIdx.x * NT + threadIdx.x) * VEC;
#if __CUDA_ARCH__ >= 900
  if (PDL) cudaGridDependencySynchronize();
#endif
  if (i + VEC <= n) {
    float4 v = *reinterpret_cast<const float4*>(in + i);
    fk_sig16<T>(v);
    *reinterpret_cast<float4*>(out + i) = v;
  } else if (i < n) {
#pragma unroll 1
    for (long j = i; j < n; ++j) out[j] = fk_sig(in[j]);
  }
}

template <typename T, int NT, bool PDL>
__global__ __launch_bounds__(NT) void fk_flat_scalar(const T* __restrict__ in,
                                                     T* __restrict__ out, long n) {
  long i = (long)blockIdx.x * NT + threadIdx.x;
  long step = (long)gridDim.x * NT;
#if __CUDA_ARCH__ >= 900
  if (PDL) cudaGridDependencySynchronize();
#endif
  for (; i < n; i += step) out[i] = fk_sig(in[i]);
}

// ---------------------------- strided input --------------------------------
// grid = (chunks along the contiguous inner run, collapsed outer index).
// The outer offset costs at most four 64-bit divmods per *block*.
template <typename T, int NT, bool VECTOR, bool PDL>
__global__ __launch_bounds__(NT) void fk_rows(const T* __restrict__ in,
                                              T* __restrict__ out, long inner,
                                              FkOuter o) {
  constexpr int VEC = VECTOR ? 16 / (int)sizeof(T) : 1;
  long r = blockIdx.y, ioff = 0;
#pragma unroll
  for (int d = 0; d < 4; ++d)
    if (d < o.ndim) { long q = r % o.size[d]; r /= o.size[d]; ioff += q * o.stride[d]; }
  long ooff = (long)blockIdx.y * inner;
  long c = ((long)blockIdx.x * NT + threadIdx.x) * VEC;
#if __CUDA_ARCH__ >= 900
  if (PDL) cudaGridDependencySynchronize();
#endif
  if (VECTOR && c + VEC <= inner) {
    float4 v = *reinterpret_cast<const float4*>(in + ioff + c);
    fk_sig16<T>(v);
    *reinterpret_cast<float4*>(out + ooff + c) = v;
  } else if (c < inner) {
#pragma unroll 1
    for (long j = c; j < inner && j < c + VEC; ++j) out[ooff + j] = fk_sig(in[ioff + j]);
  }
}

// ------------------------------- launching ---------------------------------
static bool fk_pdl() {
  static int cached = -1;
  if (cached < 0) {
    int dev = 0, major = 0;
    cached = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess)
      cached = (major >= 9) ? 1 : 0;
  }
  return cached == 1;
}

// Launch the PDL instantiation where the hardware supports it, else the plain
// one (identical signature, so both share one function-pointer type).
template <typename FN, typename... A>
static inline void fk_go(FN pdl_fn, FN plain_fn, dim3 grid, int nt, cudaStream_t st,
                         A... args) {
  if (fk_pdl()) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(nt, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, pdl_fn, args...);
  } else {
    plain_fn<<<grid, nt, 0, st>>>(args...);
  }
}

#define FK_NT 256
#define FK_MAX_ROWS 65535

// Collapse (sizes, strides) into a contiguous inner run plus <=4 outer dims.
// The same merge is valid for the contiguous output, so the output index is
// simply (outer index) * inner + (inner index).
static bool fk_collapse(const at::Tensor& x, long& inner, FkOuter& o) {
  const auto sz = x.sizes(), sv = x.strides();
  std::vector<long> s, t;
  for (int i = 0; i < (int)sz.size(); ++i)
    if (sz[i] != 1) { s.push_back(sz[i]); t.push_back(sv[i]); }
  if (s.empty()) { inner = 1; o.ndim = 0; return true; }
  if (t.back() != 1) return false;
  std::vector<long> ms{s.back()}, mt{1L};
  for (int i = (int)s.size() - 2; i >= 0; --i) {
    if (t[i] == ms.back() * mt.back()) ms.back() *= s[i];
    else { ms.push_back(s[i]); mt.push_back(t[i]); }
  }
  o.ndim = (int)ms.size() - 1;
  if (o.ndim > 4) return false;
  inner = ms[0];
  for (int d = 0; d < 4; ++d) {
    o.size[d] = d < o.ndim ? ms[d + 1] : 1;
    o.stride[d] = d < o.ndim ? mt[d + 1] : 0;
  }
  return true;
}

at::Tensor fk_sigmoid(const at::Tensor& x) {
  if (!x.is_cuda() || !x.is_floating_point() || x.numel() == 0) return at::sigmoid(x);
  const c10::cuda::CUDAGuard guard(x.device());

  const bool contig = x.is_contiguous();
  long inner = 0;
  FkOuter o{};
  bool rows_ok = false;
  long nrows = 1;
  if (!contig) {
    rows_ok = fk_collapse(x, inner, o);
    if (rows_ok) {
      long r = 1;
      for (int d = 0; d < o.ndim; ++d) r *= o.size[d];
      rows_ok = (r <= FK_MAX_ROWS);
      nrows = r;
    }
  }
  // Anything the strided kernel cannot map (inner run not innermost, >4 outer
  // dims, or too many rows) falls back to one gather into a contiguous buffer.
  const at::Tensor src = (contig || rows_ok) ? x : x.contiguous();
  const bool use_rows = !contig && rows_ok;

  at::Tensor out = at::detail::empty_cuda(x.sizes(), x.scalar_type(), x.device(),
                                          at::MemoryFormat::Contiguous);
  cudaStream_t st = at::cuda::getCurrentCUDAStream(x.device().index());
  const long n = src.numel();
  const int esz = (int)src.element_size();
  const int vec = 16 / esz;
  bool aligned =
      ((reinterpret_cast<uintptr_t>(src.const_data_ptr()) |
        reinterpret_cast<uintptr_t>(out.data_ptr())) & 15u) == 0;
  if (use_rows) {
    aligned = aligned && (inner % vec == 0);
    for (int d = 0; d < o.ndim; ++d) aligned = aligned && (o.stride[d] % vec == 0);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, src.scalar_type(),
                                  "fk_sigmoid", [&] {
    const scalar_t* ip = src.const_data_ptr<scalar_t>();
    scalar_t* op = out.mutable_data_ptr<scalar_t>();
    if (use_rows) {
      const int bx = (int)((inner + (long)FK_NT * (aligned ? vec : 1) - 1) /
                           ((long)FK_NT * (aligned ? vec : 1)));
      const dim3 grid(bx, (unsigned)nrows, 1);
      if (aligned)
        fk_go(fk_rows<scalar_t, FK_NT, true, true>, fk_rows<scalar_t, FK_NT, true, false>,
              grid, FK_NT, st, ip, op, inner, o);
      else
        fk_go(fk_rows<scalar_t, FK_NT, false, true>, fk_rows<scalar_t, FK_NT, false, false>,
              grid, FK_NT, st, ip, op, inner, o);
    } else if (aligned) {
      const dim3 grid((unsigned)((n + (long)FK_NT * vec - 1) / ((long)FK_NT * vec)), 1, 1);
      fk_go(fk_flat_vec<scalar_t, FK_NT, true>, fk_flat_vec<scalar_t, FK_NT, false>,
            grid, FK_NT, st, ip, op, n);
    } else {
      long b = (n + FK_NT - 1) / FK_NT;
      const dim3 grid((unsigned)(b > 65535 ? 65535 : b), 1, 1);
      fk_go(fk_flat_scalar<scalar_t, FK_NT, true>, fk_flat_scalar<scalar_t, FK_NT, false>,
            grid, FK_NT, st, ip, op, n);
    }
  });
  return out;
}
"""

_FLAGS = ["-O3", "--expt-relaxed-constexpr"]


def _local_arch() -> str | None:
    """Local compute capability, without creating a CUDA context (nvidia-smi)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    mapped = [f"{c}a" if c.split(".")[0] in ("9", "10", "12") else c for c in caps]
    return " ".join(mapped) or None


def _build():
    """JIT the extension, pinned to the local arch so the build stays quick."""
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    arch = _local_arch()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name="fk_l1_sigmoid_v2",
            cpp_sources=_CPP,
            cuda_sources=_CUDA,
            functions=["fk_sigmoid"],
            extra_cuda_cflags=_FLAGS,
            verbose=False,
        )
    finally:
        if arch:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev


_fk_sigmoid = _build().fk_sigmoid


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fk_sigmoid(x)
