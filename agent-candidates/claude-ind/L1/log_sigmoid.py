"""Log-sigmoid activation: log(1 / (1 + exp(-x))) = -softplus(-x).

Custom CUDA kernel. ``F.logsigmoid`` evaluates ``exp`` + ``log1p`` at full
libdevice precision, which makes the op compute bound (~1.8 TB/s on a B200)
even though it is a trivially memory-bound elementwise map. Here the identity

    logsigmoid(x) = min(x, 0) - log1p(exp(-|x|))

is evaluated with one MUFU.EX2 (``exp2``) plus a degree-3 polynomial for
``log1p(z) / z`` on ``z = exp(-|x|) in [0, 1]`` (max error 5e-4, far inside the
bf16 + 1e-2 bench tolerance). One transcendental per element is free at HBM
speed on this GPU; two are not. With 16B vector accesses and three items per
thread the kernel runs at pure-copy bandwidth (~6.0 TB/s).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#define LOG2E 1.4426950408889634f

// -log1p(z)/z on z in [0, 1], Chebyshev degree 3 (|err| <= 5.1e-4 in log1p).
#define A0  7.389876e-02f
#define A1 -2.518742e-01f
#define A2  4.846352e-01f
#define A3 -9.993012e-01f

__device__ __forceinline__ float lsig(float x) {
  float z = __builtin_exp2f(-fabsf(x) * LOG2E);   // exp(-|x|), one MUFU.EX2
  float p = fmaf(A0, z, A1);
  p = fmaf(p, z, A2);
  p = fmaf(p, z, A3);
  return fmaf(p, z, fminf(x, 0.f));               // min(x,0) - log1p(z)
}

typedef __nv_bfloat162 b2;
struct V8 { b2 a, b, c, d; };                      // 8 bf16 = 16 bytes

__device__ __forceinline__ V8 apply(V8 v) {
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    b2* q = reinterpret_cast<b2*>(&v) + k;
    float2 f = __bfloat1622float2(*q);
    f.x = lsig(f.x);
    f.y = lsig(f.y);
    *q = __float22bfloat162_rn(f);
  }
  return v;
}
__device__ __forceinline__ V8 ldx(const V8* p) {   // evict-first: streamed once
  int4 t = __ldcs(reinterpret_cast<const int4*>(p));
  return *reinterpret_cast<V8*>(&t);
}
__device__ __forceinline__ void stx(V8* p, V8 v) {
  __stcs(reinterpret_cast<int4*>(p), *reinterpret_cast<int4*>(&v));
}

constexpr int NT = 256;   // threads / block
constexpr int IT = 3;     // 16B items / thread (3 keeps enough loads in flight
                          // to stay at copy bandwidth; 2 and 4 both measure slower)

__global__ __launch_bounds__(NT) void lsig_vec(const V8* __restrict__ in,
                                               V8* __restrict__ out, int n8) {
  int base = blockIdx.x * (NT * IT) + threadIdx.x;
  if (base + (IT - 1) * NT < n8) {
    V8 v[IT];
    #pragma unroll
    for (int j = 0; j < IT; ++j) v[j] = ldx(in + base + j * NT);
    #pragma unroll
    for (int j = 0; j < IT; ++j) v[j] = apply(v[j]);
    #pragma unroll
    for (int j = 0; j < IT; ++j) stx(out + base + j * NT, v[j]);
  } else {
    #pragma unroll
    for (int j = 0; j < IT; ++j) {
      int i = base + j * NT;
      if (i < n8) stx(out + i, apply(ldx(in + i)));
    }
  }
}

// Fallback for a numel that is not a multiple of 8 / unaligned storage.
__global__ __launch_bounds__(256) void lsig_scalar(const __nv_bfloat16* __restrict__ in,
                                                   __nv_bfloat16* __restrict__ out,
                                                   long long n) {
  long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16_rn(lsig(__bfloat162float(in[i])));
}

at::Tensor logsigmoid(const at::Tensor& x) {
  at::Tensor xc = x.is_contiguous() ? x : x.contiguous();
  at::Tensor out = at::empty_like(xc);
  int64_t n = xc.numel();
  if (n == 0) return out;
  const c10::cuda::CUDAGuard guard(xc.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const void* ip = xc.const_data_ptr();
  void* op = out.data_ptr();
  bool aligned = ((reinterpret_cast<uintptr_t>(ip) | reinterpret_cast<uintptr_t>(op)) & 15) == 0;
  if ((n & 7) == 0 && aligned) {
    int64_t n8 = n >> 3;
    int64_t blocks = (n8 + NT * IT - 1) / (NT * IT);
    lsig_vec<<<blocks, NT, 0, stream>>>(static_cast<const V8*>(ip),
                                        static_cast<V8*>(op), (int)n8);
  } else {
    int64_t blocks = (n + 255) / 256;
    lsig_scalar<<<blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(ip),
                                            static_cast<__nv_bfloat16*>(op), (long long)n);
  }
  return out;
}
"""

_CPP_SRC = "at::Tensor logsigmoid(const at::Tensor& x);"


def _build():
    from torch.utils.cpp_extension import load_inline

    if torch.cuda.is_available():
        os.environ.setdefault(
            "TORCH_CUDA_ARCH_LIST", "%d.%d" % torch.cuda.get_device_capability()
        )
    return load_inline(
        name="fk_log_sigmoid_v1",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["logsigmoid"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


try:
    _EXT = _build()
except Exception:  # pragma: no cover -- keep the op usable if the JIT build fails
    _EXT = None

# ``n & 7 == 0`` is guaranteed by every captured shape (last dim 1280); the
# kernel still handles the ragged case itself.
_FAST = _EXT is not None


class LogSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _FAST and x.dtype == torch.bfloat16 and x.is_cuda:
            return _EXT.logsigmoid(x)
        return F.logsigmoid(x)
