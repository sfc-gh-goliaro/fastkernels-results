"""T5-style RMSNorm with fp32 variance computation -- single fused CUDA kernel.

Semantics are HuggingFace's ``T5LayerNorm``: the sum of squares and the
``rsqrt`` are computed in fp32, the normalized activation is rounded back to
the weight dtype, and only then multiplied by the weight.  The baseline spends
eight separate elementwise/reduction kernels (and two fp32 temporaries twice
the size of the input) on that; here one pass reads each row into registers,
reduces it, and writes the scaled row straight back -- so HBM sees exactly one
read and one write of the bf16 tensor.

For the captured shape ([1, 512, 4096] bf16) that is ~2 us of work on a B200,
i.e. within a couple of microseconds of an empty kernel launch.  Anything the
fast path cannot handle (non-bf16, ragged hidden size, unaligned or
non-contiguous input) falls through to the eager reference implementation.
"""

from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float warp_all_reduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ uint4 scale_vec(uint4 xv, uint4 wv, float inv) {
  const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
  const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
  uint4 o;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __bfloat1622float2(xp[k]);
    float2 g = __bfloat1622float2(wp[k]);
    // HF parity: (x * rsqrt) is rounded to bf16 before the weight multiply.
    float a = __bfloat162float(__float2bfloat16_rn(f.x * inv));
    float b = __bfloat162float(__float2bfloat16_rn(f.y * inv));
    op[k] = __floats2bfloat162_rn(a * g.x, b * g.y);
  }
  return o;
}

// One row of H = TPR * VPT * 8 bf16 values per TPR threads, RPB row-groups per
// block.  X is read once into registers, reduced across the row group, then
// scaled and stored -- a single pass in each direction.
template <int TPR, int RPB, int VPT>
__global__ __launch_bounds__(TPR* RPB) void t5_ln_fast(
    const uint4* __restrict__ X, const uint4* __restrict__ W,
    uint4* __restrict__ Y, int nrows, float inv_n, float eps) {
  constexpr int VPR = TPR * VPT;  // uint4 per row
  constexpr int WPR = TPR / 32;   // warps per row (>=1)
  const int lane = threadIdx.x % TPR;
  const int sub = threadIdx.x / TPR;
  const int row = blockIdx.x * RPB + sub;

  __shared__ float smem[RPB * WPR];

  uint4 v[VPT];
  float acc = 0.f;
  if (row < nrows) {
    const uint4* xr = X + (size_t)row * VPR;
#pragma unroll
    for (int i = 0; i < VPT; ++i) v[i] = xr[lane + i * TPR];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v[i]);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 f = __bfloat1622float2(p[k]);
        acc = fmaf(f.x, f.x, acc);
        acc = fmaf(f.y, f.y, acc);
      }
    }
  }
  acc = warp_all_reduce(acc);
  float total = acc;
  if (WPR > 1) {
    if ((threadIdx.x & 31) == 0) smem[sub * WPR + lane / 32] = acc;
    __syncthreads();
    total = 0.f;
#pragma unroll
    for (int i = 0; i < WPR; ++i) total += smem[sub * WPR + i];
  }
  if (row >= nrows) return;

  const float inv = rsqrtf(total * inv_n + eps);
  uint4* yr = Y + (size_t)row * VPR;
#pragma unroll
  for (int i = 0; i < VPT; ++i)
    yr[lane + i * TPR] = scale_vec(v[i], W[lane + i * TPR], inv);
}

// Any H that is a multiple of 8: two passes over the row, the second served
// from L2.  One block per row.
template <int NT>
__global__ __launch_bounds__(NT) void t5_ln_generic(
    const uint4* __restrict__ X, const uint4* __restrict__ W,
    uint4* __restrict__ Y, int vecs, float inv_n, float eps) {
  const uint4* xr = X + (size_t)blockIdx.x * vecs;
  uint4* yr = Y + (size_t)blockIdx.x * vecs;
  float acc = 0.f;
  for (int i = threadIdx.x; i < vecs; i += NT) {
    uint4 t = xr[i];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&t);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 f = __bfloat1622float2(p[k]);
      acc = fmaf(f.x, f.x, acc);
      acc = fmaf(f.y, f.y, acc);
    }
  }
  acc = warp_all_reduce(acc);
  __shared__ float smem[NT / 32];
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x / 32] = acc;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) total += smem[i];
  const float inv = rsqrtf(total * inv_n + eps);
  for (int i = threadIdx.x; i < vecs; i += NT)
    yr[i] = scale_vec(xr[i], W[i], inv);
}

}  // namespace

#define LAUNCH(TPR, RPB, VPT)                                              \
  t5_ln_fast<TPR, RPB, VPT><<<(nrows + (RPB)-1) / (RPB), (TPR) * (RPB), 0, \
                              stream>>>(xp, wp, yp, nrows, inv_n, epsf);   \
  return y;

torch::Tensor t5_rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto y = torch::empty_like(x);
  const int64_t H = x.size(-1);
  const int nrows = (int)(x.numel() / H);
  const int vecs = (int)(H / 8);
  const float inv_n = 1.0f / (float)H;
  const float epsf = (float)eps;
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint4* xp = reinterpret_cast<const uint4*>(x.data_ptr());
  const uint4* wp = reinterpret_cast<const uint4*>(w.data_ptr());
  uint4* yp = reinterpret_cast<uint4*>(y.data_ptr());

  // TPR * VPT must equal vecs; 256 threads per row is the fastest arrangement
  // measured on B200 for the 4096-wide captured case.
  switch (vecs) {
    case 512: LAUNCH(256, 1, 2)     // H = 4096
    case 256: LAUNCH(256, 1, 1)     // H = 2048
    case 1024: LAUNCH(256, 1, 4)    // H = 8192
    case 2048: LAUNCH(256, 1, 8)    // H = 16384
    case 128: LAUNCH(128, 2, 1)     // H = 1024
    case 64: LAUNCH(64, 4, 1)       // H = 512
    case 32: LAUNCH(32, 8, 1)       // H = 256
    default: break;
  }
  t5_ln_generic<256><<<nrows, 256, 0, stream>>>(xp, wp, yp, vecs, inv_n, epsf);
  return y;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor t5_rms_norm(torch::Tensor x, torch::Tensor w, double eps);
"""

_ext = None
_ext_failed = False
_lock = threading.Lock()


def _load_ext():
    """JIT-build (once) the fused kernel; ``None`` if that is not possible."""
    global _ext, _ext_failed
    if _ext is not None or _ext_failed:
        return _ext
    with _lock:
        if _ext is not None or _ext_failed:
            return _ext
        try:
            from torch.utils.cpp_extension import load_inline

            prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}.{minor}" + ("a" if major in (9, 10, 12) else "")
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            try:
                _ext = load_inline(
                    name="fk_t5_layer_norm_fused",
                    cpp_sources=_CPP_SRC,
                    cuda_sources=_CUDA_SRC,
                    functions=["t5_rms_norm"],
                    extra_cuda_cflags=["-O3", "--use_fast_math"],
                    verbose=False,
                )
            finally:
                if prev is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = prev
        except Exception:
            _ext_failed = True
            _ext = None
    return _ext


def _fast_ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and w.dtype is torch.bfloat16
        and x.is_contiguous()
        and w.is_contiguous()
        and x.dim() >= 1
        and x.size(-1) == w.numel()
        and x.size(-1) % 8 == 0
        and x.numel() > 0
        and (x.data_ptr() % 16) == 0
        and (w.data_ptr() % 16) == 0
    )


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if _fast_ok(hidden_states, weight):
            ext = _load_ext()
            if ext is not None:
                return ext.t5_rms_norm(
                    hidden_states, weight, self.variance_epsilon)

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon)

        if weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(weight.dtype)

        return weight * hidden_states
