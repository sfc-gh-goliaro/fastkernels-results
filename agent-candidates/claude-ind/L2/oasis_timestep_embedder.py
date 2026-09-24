"""Oasis timestep embedding, fused into a single CUDA kernel.

The baseline runs ~13 tiny eager kernels (arange/exp/mul/cos/sin/cat, addmm,
silu, addmm) for a problem whose whole working set is the 5 MB of MLP weights,
so latency is entirely launch-bound.  Everything is folded into one kernel:

  emb = [cos(t*f), sin(t*f)]   ->   h = silu(emb @ W1^T + b1)   ->   h @ W2^T + b2

A grid-wide spin barrier separates the two GEMVs; both weight matrices are
loaded into registers *before* the barrier so the 5 MB fetch overlaps the
embedding, the first GEMV and the barrier itself.

torch's fp32 matmul path uses TF32 on this GPU, so the kernel reproduces it:
matmul inputs are rounded to tf32 (round-to-nearest-even) as they are loaded and
accumulated in fp32, which matches cuBLAS TF32 to ~1e-6 (the baseline's own
distance from an exact fp32 result is ~1e-4).
"""

from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define FE 256
#define HALFE 128
#define HS 1024
#define NWARP 8
#define NTHR (NWARP * 32)
#define NBLK (HS / NWARP)
#define W1V (FE / 128)
#define W2V (HS / 128)
#define MAXB 8

__device__ __forceinline__ float tf32(float x) {
  unsigned i = __float_as_uint(x);
  i = (i + 0x0FFFu + ((i >> 13) & 1u)) & 0xFFFFE000u;
  return __uint_as_float(i);
}

__device__ __forceinline__ float4 tf32x4(const float4& v) {
  float4 r;
  r.x = tf32(v.x); r.y = tf32(v.y); r.z = tf32(v.z); r.w = tf32(v.w);
  return r;
}

__device__ __forceinline__ float dot4(const float4& a, const float4& b) {
  float r = a.x * b.x;
  r += a.y * b.y;
  r += a.z * b.z;
  r += a.w * b.w;
  return r;
}

template <int B>
__global__ __launch_bounds__(NTHR) void fused(
    const long* __restrict__ tp, const float* __restrict__ freqs,
    const float* __restrict__ w1, const float* __restrict__ b1,
    const float* __restrict__ w2, const float* __restrict__ b2,
    float* __restrict__ h, float* __restrict__ out,
    unsigned long long* __restrict__ bar, unsigned long long target) {
  __shared__ __align__(16) float sm[B * HS];
  const int tid = threadIdx.x;
  const int lane = tid & 31, w = tid >> 5;
  const int row = blockIdx.x * NWARP + w;

  // Issue every weight load up front so the whole 5 MB is in flight while the
  // embedding, the first GEMV and the barrier run.
  const float4* w1r = reinterpret_cast<const float4*>(w1 + (size_t)row * FE);
  float4 a[W1V];
#pragma unroll
  for (int m = 0; m < W1V; ++m) a[m] = tf32x4(w1r[m * 32 + lane]);
  const float4* w2r = reinterpret_cast<const float4*>(w2 + (size_t)row * HS);
  float4 c[W2V];
#pragma unroll
  for (int m = 0; m < W2V; ++m) c[m] = tf32x4(w2r[m * 32 + lane]);
  const float bias1 = b1[row], bias2 = b2[row];

  // One (batch, frequency) pair per thread, spread over the whole block so the
  // trig latency is not serialised across B.
#pragma unroll
  for (int j = tid; j < B * HALFE; j += NTHR) {
    const int b = j >> 7, i = j & (HALFE - 1);
    float s, co;
    sincosf((float)tp[b] * freqs[i], &s, &co);
    sm[b * FE + i] = tf32(co);
    sm[b * FE + HALFE + i] = tf32(s);
  }
  __syncthreads();

  float acc[B];
#pragma unroll
  for (int b = 0; b < B; ++b) acc[b] = 0.f;
#pragma unroll
  for (int m = 0; m < W1V; ++m) {
    const int i = (m * 32 + lane) * 4;
#pragma unroll
    for (int b = 0; b < B; ++b)
      acc[b] += dot4(a[m], *reinterpret_cast<const float4*>(&sm[b * FE + i]));
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) {
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
  }
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < B; ++b) {
      const float v = acc[b] + bias1;
      h[b * HS + row] = tf32(v / (1.f + expf(-v)));
    }
  }

  // Grid-wide barrier.  The fence orders this thread's h stores; __syncthreads
  // then guarantees every warp in the block has fenced before thread 0
  // announces the block's arrival.
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    atomicAdd(bar, 1ULL);
    volatile unsigned long long* flag = bar;
    while (*flag < target) {}
  }
  __syncthreads();

  {
    float4* sm4 = reinterpret_cast<float4*>(sm);
    const float4* h4 = reinterpret_cast<const float4*>(h);
#pragma unroll
    for (int b = 0; b < B; ++b)
      if (tid < HS / 4) sm4[b * (HS / 4) + tid] = __ldcg(&h4[b * (HS / 4) + tid]);
  }
  __syncthreads();

#pragma unroll
  for (int b = 0; b < B; ++b) acc[b] = 0.f;
#pragma unroll
  for (int m = 0; m < W2V; ++m) {
    const int kk = (m * 32 + lane) * 4;
#pragma unroll
    for (int b = 0; b < B; ++b)
      acc[b] += dot4(c[m], *reinterpret_cast<const float4*>(&sm[b * HS + kk]));
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) {
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
  }
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < B; ++b) out[b * HS + row] = acc[b] + bias2;
  }
}

struct Plan {
  const float *freqs, *w1, *b1, *w2, *b2;
  float* h;
  unsigned long long* bar;
  unsigned long long gen;
  at::Tensor scratch;
  at::TensorOptions opts;
};
static std::vector<Plan> g_plans;

// The grid barrier requires every block to be resident simultaneously.
static bool grid_fits() {
  int dev = 0, sms = 0, per_sm = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
    return false;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (void*)fused<MAXB>,
                                                    NTHR, 0) != cudaSuccess)
    return false;
  return per_sm > 0 && (long)per_sm * sms >= NBLK;
}

int64_t make_plan(at::Tensor freqs, at::Tensor w1, at::Tensor b1, at::Tensor w2,
                  at::Tensor b2) {
  if (!grid_fits()) return -1;
  Plan p;
  p.freqs = freqs.data_ptr<float>();
  p.w1 = w1.data_ptr<float>();
  p.b1 = b1.data_ptr<float>();
  p.w2 = w2.data_ptr<float>();
  p.b2 = b2.data_ptr<float>();
  p.opts = w1.options();
  p.scratch = at::zeros({MAXB * HS + 4}, p.opts);
  p.h = p.scratch.data_ptr<float>();
  p.bar = reinterpret_cast<unsigned long long*>(p.h + MAXB * HS);
  p.gen = 0;
  g_plans.push_back(p);
  return (int64_t)g_plans.size() - 1;
}

#define LAUNCH(B)                                                           \
  fused<B><<<NBLK, NTHR, 0, st>>>(tp, p.freqs, p.w1, p.b1, p.w2, p.b2, p.h, \
                                  op, p.bar, p.gen* NBLK);                  \
  break;

at::Tensor run(int64_t id, at::Tensor t) {
  Plan& p = g_plans[(size_t)id];
  const int B = (int)t.size(0);
  at::Tensor out = at::empty({B, HS}, p.opts);
  float* op = out.data_ptr<float>();
  const long* tp = (const long*)t.data_ptr();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  ++p.gen;
  switch (B) {
    case 1: LAUNCH(1)
    case 2: LAUNCH(2)
    case 3: LAUNCH(3)
    case 4: LAUNCH(4)
    case 5: LAUNCH(5)
    case 6: LAUNCH(6)
    case 7: LAUNCH(7)
    case 8: LAUNCH(8)
    default: TORCH_CHECK(false, "unsupported batch ", B);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_plan", &make_plan);
  m.def("run", &run);
}
"""

_MIN_B = 2
_MAX_B = 8
_ext = None


def _load_ext():
    """JIT-build the fused kernel (cached by source hash); None if unavailable."""
    global _ext
    if _ext is not None:
        return _ext if _ext is not False else None
    _ext = False
    try:
        from torch.utils.cpp_extension import _get_build_directory, load

        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = (
            arch + "a" if major in (9, 10, 12) else arch)
        name = "oasis_tse_" + hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
        src = os.path.join(_get_build_directory(name, verbose=False), "kernel.cu")
        if not os.path.exists(src) or open(src).read() != _CUDA_SRC:
            with open(src, "w") as fh:
                fh.write(_CUDA_SRC)
        _ext = load(name=name, sources=[src], extra_cuda_cflags=["-O3"],
                    verbose=False)
    except Exception:
        _ext = False
    return _ext if _ext is not False else None


class OasisTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Linear(frequency_embedding_size, hidden_size, bias=True),
                SiLU(),
                Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._fusable = (hidden_size == 1024 and frequency_embedding_size == 256)
        self._plan = None
        self._run = None

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def _build_plan(self):
        """Capture the device pointers the fused kernel reads (once)."""
        self._plan = False
        if not self._fusable:
            return
        w1, b1 = self.mlp[0].weight, self.mlp[0].bias
        w2, b2 = self.mlp[2].weight, self.mlp[2].bias
        if tuple(w1.shape) != (1024, 256) or tuple(w2.shape) != (1024, 1024):
            return
        if (b1 is None or b2 is None or not w1.is_cuda
                or w1.dtype is not torch.float32 or w2.dtype is not torch.float32
                or not (w1.is_contiguous() and w2.is_contiguous()
                        and b1.is_contiguous() and b2.is_contiguous())):
            return
        ext = _load_ext()
        if ext is None:
            return
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=w1.device)
            / half,
        )
        # The plan caches device pointers; keep references so they stay alive.
        # Weights are read live (and rounded to tf32 inside the kernel), so
        # in-place weight updates are picked up without rebuilding the plan.
        self._keep = (freqs, w1.detach(), b1.detach(), w2.detach(), b2.detach())
        plan = ext.make_plan(*self._keep)
        if plan < 0:          # grid cannot be co-resident: barrier unsafe
            return
        self._plan = plan
        self._run = ext.run

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self._plan is None:
            self._build_plan()
        # cuBLAS dispatches a *non*-TF32 gemv when the batch is 1, so the fused
        # kernel's tf32 emulation would not match the baseline there.
        if (self._plan is not False and t.is_cuda
                and _MIN_B <= t.shape[0] <= _MAX_B):
            return self._run(self._plan, t)
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x
