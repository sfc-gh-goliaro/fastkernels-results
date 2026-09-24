"""Log-sigmoid activation on a single hand-written CUDA kernel (B200 / sm_100).

Same contract as ``baseline.py``: ``LogSigmoid()`` takes no constructor arguments and
``forward(x)`` returns a tensor with the same shape and dtype as ``x``.

Math. ``logsigmoid(x) = -log(1 + exp(-x))``, evaluated in the stable form
``min(x, 0) - log1p(exp(-|x|))``. On device that is two special-function-unit
operations per element -- ``__expf`` (MUFU.EX2 + a multiply) and ``__logf``
(MUFU.LG2 + a multiply) -- against libdevice's multi-instruction ``expf`` /
``log1pf``, which is where the whole gain on large inputs comes from. Eager
PyTorch already issues 128-bit loads and stores for a unary bfloat16 op on this
toolchain (``thread_work_size() == 8`` for CUDA >= 12.8), so the 16-byte packets
below buy parity on memory access, not an advantage; the arithmetic is the lever.

A polynomial ``log1p`` would replace the second special-function operation with a
few FMAs and is cheap to drop in. It is not used here because the two-intrinsic
form is essentially exact -- max abs error ~3e-05 over the whole finite bfloat16
domain against a 1e-2 comparison bound, where a degree-3 polynomial measured
7.8e-03 and would spend nearly all of that margin.

Profiling settled which resource actually limits this kernel, and it is not
memory: at ``[181, 1081, 1280]`` the special-function pipe runs at 82 % of peak
while FMA sits at 27 %, and DRAM traffic is 0.893 GiB at 4.13 TB/s, only about
half of peak bandwidth. So the polynomial form is the best-motivated next step
rather than a pointless trade -- it is left out of this version on accuracy
margin, not because the arithmetic is free. See
``profile/log_sigmoid_v2_ldg128_stream_1gb/REPORT.md``.

A 64K-entry bfloat16 lookup table was also considered and rejected: random
16-bit indices average roughly three-way shared-memory bank conflicts, putting
effective throughput below the special-function rate it would replace.

The fast path is deliberately narrow -- contiguous, 16-byte-aligned, bfloat16,
CUDA, under 2**31 elements, not requiring grad -- and everything else is
delegated to ``at::log_sigmoid`` inside C++, so there is exactly one device code
path to reason about and no scalar fallback kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Distinct name so this extension never collides with another operator's build
# in a shared cache.
_EXT_NAME = "fk_log_sigmoid_bf16"

# Resolved from this file, never from the process working directory: the
# benchmark harness imports this module by path from wherever it happens to be
# running. ``load_inline`` does not create the directory itself.
_BUILD_DIR = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXT_NAME

# Used when the capability query is unavailable (no initialized CUDA device at
# import time). This workspace targets B200.
_DEFAULT_ARCH = "10.0"

# Test hook: appended to the nvcc flags so a build failure can be provoked
# without editing this file, which is how the "compiled kernel is what was
# measured" assertion gets a negative case.
_EXTRA_NVCC_ENV = "FK_LOG_SIGMOID_EXTRA_NVCC_FLAGS"

_CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <cstdint>

at::Tensor logsigmoid_forward(const at::Tensor& x);
int64_t fastpath_calls();
int64_t delegated_calls();
void reset_call_counters();
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/ops/log_sigmoid.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdint>
#include <optional>

namespace {

// How many calls each path served. Relaxed atomics, one increment per call, so
// a benchmark run can be shown to have gone through the kernel rather than
// through ATen -- the build-status flag alone only proves the extension loaded,
// not that any particular call used it.
std::atomic<uint64_t> g_fastpath_calls{0};
std::atomic<uint64_t> g_delegated_calls{0};

}  // namespace

int64_t fastpath_calls() {
  return static_cast<int64_t>(g_fastpath_calls.load(std::memory_order_relaxed));
}

int64_t delegated_calls() {
  return static_cast<int64_t>(g_delegated_calls.load(std::memory_order_relaxed));
}

void reset_call_counters() {
  g_fastpath_calls.store(0, std::memory_order_relaxed);
  g_delegated_calls.store(0, std::memory_order_relaxed);
}

namespace {

// Block size, and the ceiling on resident blocks (148 SMs x 8). Both are named
// so a later occupancy or grid sweep is a constant change rather than a
// restructuring; neither is an occupancy guarantee on its own.
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 148 * 8;

// 16-byte packet: eight bfloat16 per thread per load and per store.
constexpr int kPackElems = 8;

// At or above this element count the packet index would still fit in int32
// (it is 8x smaller), but the input is delegated to ATen rather than widening
// the address arithmetic in the inner loop for a case no capture reaches.
constexpr int64_t kMaxFastNumel = int64_t{1} << 31;

// Eight bfloat16 in one 16-byte access. The global load and store go through the
// uint4 member: declaring the packet as an array of __nv_bfloat162 and copying it
// whole compiles to four 32-bit LDG/STG rather than one LDG.E.128 / STG.E.128
// (confirmed in SASS), so the wide type is what crosses the memory boundary and
// the pair view is used only on registers.
union alignas(16) BF16x8 {
  uint4 raw;
  __nv_bfloat162 h[4];
};

__device__ __forceinline__ bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

__device__ __forceinline__ float logsigmoid_f32(float x) {
  const float t = __expf(-fabsf(x));   // MUFU.EX2 + 1 mul
  const float g = __logf(1.0f + t);    // MUFU.LG2 + 1 mul
  // fminf rather than the cheaper (x - fabsf(x)) * 0.5f, which yields NaN at
  // x = +inf. fminf(NaN, 0.0f) returns 0.0f, so NaN propagation rests on the
  // log term instead -- which does propagate it.
  return fminf(x, 0.0f) - g;
}

__device__ __forceinline__ __nv_bfloat162 logsigmoid_bf162(__nv_bfloat162 v) {
  const float2 f = __bfloat1622float2(v);
  return __floats2bfloat162_rn(logsigmoid_f32(f.x), logsigmoid_f32(f.y));
}

// Grid-stride over 16-byte packets. The leftover ``ntail`` elements (fewer than
// eight, and zero for every captured shape) are finished by the first ``ntail``
// threads of block 0 in this same launch; a second launch would cost more than
// the tail itself. The grid is floored at one block by the caller so this
// always runs.
__global__ __launch_bounds__(kThreads) void logsigmoid_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int nvec,
    int ntail) {
  const uint4* __restrict__ in_v = reinterpret_cast<const uint4*>(in);
  uint4* __restrict__ out_v = reinterpret_cast<uint4*>(out);

  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    BF16x8 p;
    p.raw = in_v[i];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      p.h[k] = logsigmoid_bf162(p.h[k]);
    }
    out_v[i] = p.raw;
  }

  const int t = static_cast<int>(threadIdx.x);
  if (blockIdx.x == 0 && t < ntail) {
    const int j = nvec * kPackElems + t;
    out[j] = __float2bfloat16(logsigmoid_f32(__bfloat162float(in[j])));
  }
}

}  // namespace

at::Tensor logsigmoid_forward(const at::Tensor& x) {
  // Everything the packet kernel requires, checked before anything is
  // allocated, so a delegated input never pays for an output it will not use.
  // The two autograd predicates are separate concerns: requires_grad covers
  // reverse mode, and a forward-mode dual carries a tangent while reporting
  // requires_grad() == false, so it needs its own check (this is the same
  // predicate ATen's own isFwGradDefined uses).
  if (!x.is_cuda() || x.scalar_type() != at::kBFloat16 || !x.is_contiguous() ||
      x.numel() >= kMaxFastNumel ||
      (x.requires_grad() && c10::GradMode::is_enabled()) ||
      x._fw_grad(/*level=*/0).defined()) {
    g_delegated_calls.fetch_add(1, std::memory_order_relaxed);
    return at::log_sigmoid(x);
  }

  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t n = x.numel();

  if (n == 0) {
    g_fastpath_calls.fetch_add(1, std::memory_order_relaxed);
    return at::detail::empty_cuda(x.sizes(), x.scalar_type(), x.device(),
                                  std::nullopt);
  }

  const auto* in = static_cast<const __nv_bfloat16*>(x.const_data_ptr());
  if ((reinterpret_cast<uintptr_t>(in) & 15) != 0) {
    g_delegated_calls.fetch_add(1, std::memory_order_relaxed);
    return at::log_sigmoid(x);
  }

  at::Tensor out = at::detail::empty_cuda(x.sizes(), x.scalar_type(),
                                          x.device(), std::nullopt);
  auto* dst = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  // The caching allocator hands back 512-byte-aligned blocks, so this never
  // fires in practice; it is here so the kernel's alignment precondition is
  // checked rather than assumed. This is the one path that allocates and then
  // delegates anyway -- unreachable in practice, and the alternative would be to
  // trust the allocator's alignment silently.
  if ((reinterpret_cast<uintptr_t>(dst) & 15) != 0) {
    g_delegated_calls.fetch_add(1, std::memory_order_relaxed);
    return at::log_sigmoid(x);
  }

  const int nvec = static_cast<int>(n / kPackElems);
  const int ntail = static_cast<int>(n % kPackElems);

  // Floored at one block: without the floor, every 0 < numel < 8 gives nvec == 0
  // and therefore a zero-block launch, which leaves the output unwritten and
  // never runs the tail.
  int blocks = (nvec + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  if (blocks < 1) {
    blocks = 1;
  }

  g_fastpath_calls.fetch_add(1, std::memory_order_relaxed);
  logsigmoid_bf16_kernel<<<blocks, kThreads, 0,
                           at::cuda::getCurrentCUDAStream(x.device().index())>>>(
      in, dst, nvec, ntail);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


def _gencode_flag() -> str:
    """``-gencode`` for the visible device's compute capability.

    Passed directly rather than through ``TORCH_CUDA_ARCH_LIST``:
    ``cpp_extension._get_cuda_arch_flags`` returns nothing once a caller-supplied
    flag mentions ``arch``, so this pins the target without touching the
    environment the rest of the process compiles under. The ambient list in this
    workspace names seven architectures -- roughly six times the compile time for
    six binaries that would never run, and a stall-watchdog risk on a cold build.
    """
    arch = _DEFAULT_ARCH
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}.{minor}"
    except Exception:
        pass
    tag = arch.replace(".", "")
    return f"-gencode=arch=compute_{tag},code=sm_{tag}"


def _build():
    """Compile and load the extension, or return ``None`` after reporting why not."""
    flags = ["-O3", "-lineinfo", _gencode_flag()]  # -lineinfo for ncu correlation
    extra = os.environ.get(_EXTRA_NVCC_ENV, "").split()
    if extra:
        flags += extra

    try:
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
        return load_inline(
            name=_EXT_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["logsigmoid_forward", "fastpath_calls",
                       "delegated_calls", "reset_call_counters"],
            extra_cuda_cflags=flags,
            build_directory=str(_BUILD_DIR),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, but say so
        print(
            f"[{_EXT_NAME}] CUDA extension build failed, falling back to "
            f"F.logsigmoid (unoptimized, 1.00x): {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


_EXT = _build()

#: ``True`` only when the CUDA extension compiled and loaded. A run measured
#: with this ``False`` is a measurement of ``F.logsigmoid`` and must not be
#: recorded as an optimized result.
FASTPATH_ACTIVE: bool = _EXT is not None

# One module-level global, resolved once at import: ``forward`` is a single call
# with no branch and no attribute chain, because four of the five benched shapes
# are CPU-enqueue-bound and the entire addressable budget there is ~0.6 us.
_logsigmoid = _EXT.logsigmoid_forward if FASTPATH_ACTIVE else F.logsigmoid


class LogSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _logsigmoid(x)
