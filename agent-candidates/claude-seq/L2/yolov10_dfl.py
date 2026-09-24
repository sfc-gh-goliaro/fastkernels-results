"""YOLOv10 Distribution Focal Loss layer — one fused CUDA kernel.

The reference layer is

    softmax over the 16 "bin" channels of ``x.view(b, 4, 16, a)``  ->
    1x1 conv with weight ``arange(16)``                            ->  view(b, 4, a)

i.e. per ``(b, j, i)`` the expected value of a 16-way distribution whose logits
sit at ``x[b, j*16 + k, i]``.  Composed out of the L1 kernels that is two
launches (strided softmax, then a 1x1 implicit-GEMM conv) plus two output
allocations, and it materialises the ``[b, 16, 4, a]`` probability tensor --
16x the size of the result -- so it moves ~3x the bytes this needs to.

Here one kernel does the whole thing.  A thread owns one anchor of one
``(b, j)`` pair and holds that anchor's ``C`` logits in registers, so the
reduction never leaves the register file: the input is read exactly once and the
only thing written is the ``[b, 4, a]`` result.  ``forward`` is one call into the
extension, which also matters: the whole layer is a few microseconds of GPU
work, so the two extra ``nn.Module.__call__``s of the composed path are not
free either.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.softmax import Softmax

_CUDA_SRC = r'''
// Fused DFL: softmax over C bins + dot with the bin weights, one launch.
#include <ATen/ATen.h>
#include <ATen/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

namespace {

constexpr int kSMTarget = 148;   // B200
constexpr int kMaxC = 32;
constexpr float kLog2e = 1.4426950408889634f;

// Single MUFU.EX2.  The softmax shift is folded into the argument, so a logit
// costs one FFMA plus this.
__device__ __forceinline__ float ex2(float a) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(float v) { return v; }

__device__ __forceinline__ void from_f(float v, __half* o) { *o = __float2half_rn(v); }
__device__ __forceinline__ void from_f(float v, __nv_bfloat16* o) { *o = __float2bfloat16(v); }
__device__ __forceinline__ void from_f(float v, float* o) { *o = v; }

// wdt: 0 = half, 1 = bfloat16, 2 = float.  Kept out of the unrolled bin loop:
// a dtype branch inside it costs several times the loads themselves.
__device__ __forceinline__ float load_w1(const void* w, int k, int wdt) {
  if (wdt == 0) return __half2float(static_cast<const __half*>(w)[k]);
  if (wdt == 1) return __bfloat162float(static_cast<const __nv_bfloat16*>(w)[k]);
  return static_cast<const float*>(w)[k];
}

// x: [B, 4*C, A] contiguous.  out: [B, 4, A] contiguous.
//
// One thread per (anchor, (b, j)) pair.  blockIdx.y indexes the pair -- input
// row base bj*C*A, output row base bj*A -- so there is no integer division, and
// threadIdx.x runs along the anchor axis, which is the contiguous one, so a
// warp's 16 bin loads are 16 coalesced 64 B accesses.  The anchor's C logits
// live in registers across both passes (nothing spills, nothing is re-read);
// note that the register array must only ever be indexed by the unrolled loop
// counter -- taking its address, e.g. to reinterpret_cast it to a vector type,
// pushes it to local memory and costs ~3x.
template <typename T, int C>
__global__ void dfl_kernel(const T* __restrict__ x, const void* __restrict__ w, int wdt,
                           T* __restrict__ out, int A) {
  __shared__ float wsh[C];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const long bj = blockIdx.y;

  // Get the C strided loads in flight before the (L1-resident) weight fetch.
  T v[C];
  if (i < A) {
    const T* p = x + bj * (long)C * (long)A + i;
#pragma unroll
    for (int k = 0; k < C; ++k) v[k] = p[(long)k * (long)A];
  }
  if ((int)threadIdx.x < C) wsh[threadIdx.x] = load_w1(w, threadIdx.x, wdt);
  __syncthreads();
  if (i >= A) return;

  float m = to_f(v[0]);
#pragma unroll
  for (int k = 1; k < C; ++k) m = fmaxf(m, to_f(v[k]));
  m *= -kLog2e;

  // Unnormalised softmax, accumulated straight into <1, p> and <weight, p>.
  float s = 0.f, ws = 0.f;
#pragma unroll
  for (int k = 0; k < C; ++k) {
    const float e = ex2(fmaf(to_f(v[k]), kLog2e, m));
    s += e;
    ws = fmaf(wsh[k], e, ws);
  }
  from_f(ws / s, out + bj * (long)A + i);
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
template <typename T, int C>
void launch_c(const T* x, const void* w, int wdt, T* out, int A, long BJ, cudaStream_t st) {
  int nt = 128;
  while (nt > 32 && (long)((A + nt - 1) / nt) * BJ < kSMTarget) nt >>= 1;
  if (A < nt) nt = ((A + 31) / 32) * 32;
  const dim3 grid((unsigned)((A + nt - 1) / nt), (unsigned)BJ);
  dfl_kernel<T, C><<<grid, nt, 0, st>>>(x, w, wdt, out, A);
}

template <typename T>
bool launch(const T* x, const void* w, int wdt, T* out, int C, int A, long BJ, cudaStream_t st) {
  switch (C) {
    case 4: launch_c<T, 4>(x, w, wdt, out, A, BJ, st); return true;
    case 8: launch_c<T, 8>(x, w, wdt, out, A, BJ, st); return true;
    case 16: launch_c<T, 16>(x, w, wdt, out, A, BJ, st); return true;
    case 32: launch_c<T, 32>(x, w, wdt, out, A, BJ, st); return true;
    default: return false;
  }
}

inline int wdt_of(at::ScalarType s) {
  if (s == at::kHalf) return 0;
  if (s == at::kBFloat16) return 1;
  if (s == at::kFloat) return 2;
  return -1;
}

}  // namespace

at::Tensor dfl_fwd(const at::Tensor& x, const at::Tensor& w, int64_t c1) {
  const auto sty = x.scalar_type();
  const int wdt = wdt_of(w.scalar_type());
  const int C = static_cast<int>(c1);
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.dim() == 3 && x.is_contiguous() &&
                  w.is_contiguous() && !x.requires_grad() && wdt >= 0 &&
                  (sty == at::kHalf || sty == at::kBFloat16 || sty == at::kFloat) &&
                  C > 0 && C <= kMaxC && x.size(1) == 4 * (long)C && w.numel() == C,
              "dfl_fwd: unsupported input");

  const long B = x.size(0);
  const long A = x.size(2);
  at::Tensor out = at::cuda::empty({B, 4, A}, x.options());
  if (out.numel() == 0) return out;
  // 32-bit offsets inside the kernel (bj * C * A fits, as does A itself).
  TORCH_CHECK(A * 4L * (long)C * B < 0x7ffffff0L, "dfl_fwd: too large");

  auto st = c10::cuda::getCurrentCUDAStream();
  const void* wp = w.const_data_ptr();
  bool ok;
  switch (sty) {
    case at::kHalf:
      ok = launch<__half>((const __half*)x.const_data_ptr(), wp, wdt, (__half*)out.data_ptr(), C,
                          (int)A, B * 4, st);
      break;
    case at::kBFloat16:
      ok = launch<__nv_bfloat16>((const __nv_bfloat16*)x.const_data_ptr(), wp, wdt,
                                 (__nv_bfloat16*)out.data_ptr(), C, (int)A, B * 4, st);
      break;
    default:
      ok = launch<float>((const float*)x.const_data_ptr(), wp, wdt, (float*)out.data_ptr(), C,
                         (int)A, B * 4, st);
      break;
  }
  TORCH_CHECK(ok, "dfl_fwd: unsupported c1");
  return out;
}
'''

_CPP_DECL = "at::Tensor dfl_fwd(const at::Tensor& x, const at::Tensor& w, int64_t c1);"


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
        name="fk_l2_yolov10_dfl_v7",
        cpp_sources=_CPP_DECL,
        cuda_sources=_CUDA_SRC,
        functions=["dfl_fwd"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


try:
    _dfl_fwd = _build().dfl_fwd
except Exception:  # no CUDA / no nvcc -> keep the reference path
    _dfl_fwd = None


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)
        # Flat view of the bin weights, cached in __dict__ rather than as a
        # module attribute so forward skips nn.Module.__getattr__.  Filled on the
        # first call, since .half()/.cuda() replace conv.weight before then.
        object.__setattr__(self, "_wflat", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._wflat
        if w is None:
            w = self.conv.weight.detach().reshape(-1)
            if _dfl_fwd is None or not w.is_cuda:
                return self._ref(x)
            object.__setattr__(self, "_wflat", w)
        try:
            return _dfl_fwd(x, w, self.c1)
        except Exception:  # dtype / layout / c1 the kernel does not cover
            return self._ref(x)

    def _ref(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(self._softmax(x.view(b, 4, self.c1, a).transpose(2, 1))).view(b, 4, a)
