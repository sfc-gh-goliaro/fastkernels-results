"""SiLU (Swish) for B200 / sm_100: one fp32-accumulating elementwise pass.

The activation is a single streaming pass over the input, so the only lever
against ATen's ``silu_kernel`` -- which already issues 16 bytes per thread per
access for 2-byte dtypes -- is the arithmetic cost per element. ATen evaluates
``x / (1 + exp(-x))`` with a precise ``expf`` and an IEEE divide; the kernel
built here keeps ATen's access width and substitutes a cheaper approximate
form, computed in fp32 and rounded once to the input dtype exactly as ATen's
``opmath_type<BFloat16|Half> = float`` does.

Two numeric forms are compiled, selected by ``_FORM``:

* ``FORM_EXP_DIV`` -- ``__fdividef(x, 1 + __expf(-x))``: ``ex2.approx.f32`` plus
  ``div.approx.f32``, dropping ``expf``'s range reduction and ``div.rn``'s
  Newton-Raphson fixup.
* ``FORM_TANH`` -- ``0.5*x*(1 + tanh.approx.f32(0.5*x))``: a single
  special-function instruction. Not shipped: it crosses a bfloat16 rounding tie
  near x = +5.94 and misses the accuracy gate.

Every input the vectorized path does not cover -- non-CUDA, non-contiguous,
dtypes other than bfloat16/float16, data pointers that are not 16-byte
aligned, or autograd -- is delegated to ``at::silu``, so this module can never
be more wrong than the baseline. ``EXTENSION_STATUS`` is the empty string
exactly when the compiled path is live, and carries the build error otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Numeric forms compiled into the extension; see the module docstring.
FORM_EXP_DIV = 0
FORM_TANH = 1

# Shipped configuration. The form is decided by an exhaustive sweep over all
# 65,536 bfloat16 and all 65,536 float16 encodings (tests/accuracy_sweep.py):
# FORM_TANH's approximation pushes the result across a bfloat16 rounding tie near
# x = +5.94 -- the exact value sits 5.6e-7 from the midpoint, inside the
# instruction's measured 7.9e-6 error -- which costs 45% of the comparison bound,
# while FORM_EXP_DIV is bitwise identical to ATen on every finite bfloat16
# encoding.
#
# The launch geometry comes from a measured sweep
# (profile/p1_candidate_v1/harness/tuning_sweep.py).
#
# ``_WAVES_PER_SM`` is a work-partitioning multiplier, not an occupancy figure. The
# sweep found an optimum in the middle of its range -- at 8 a 2.5 GiB input takes
# ~8% longer, and an uncapped grid ~42% longer because it launches 660k blocks --
# but the mechanism behind the small-cap loss is *not* established: at this block
# size 8 blocks of 192 threads exactly fill the register-limited residency, so
# there is no trailing wave to blame. See profile/p1_candidate_v2/REPORT.md 3c. At
# 128 the cap only binds above ~39M elements, so the smaller shapes keep the fully
# covering grid that measured best for them.
#
# ``_BLOCK`` is the winner of a randomized interleaved paired comparison of 128 /
# 192 / 256 threads over 11 rounds on one allocation: 192 beat 256 by 2.1 us
# (2.05 timer ticks, so beyond noise) and tied with 128 at 0.6 us.
#
# ``_WORDS_PER_THREAD`` stays at 1 because it makes no measurable difference on
# the only case with real streaming work: once ``_WAVES_PER_SM`` binds, it is the
# cap and not the requested work per thread that fixes the grid, and the whole
# 4x4 block/words grid at waves=128 spans just 855-869 us.
_FORM = FORM_EXP_DIV
_BLOCK = 192
_WORDS_PER_THREAD = 1
_WAVES_PER_SM = 128

_EXTENSION_NAME = "fk_silu_sm100_v1"
_CUDA_ARCH = "10.0"

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace {

// Matches the width ATen's vectorized elementwise kernel already uses for
// 2-byte dtypes on sm_100; there is nothing to win on access width here.
constexpr int kBytesPerAccess = 16;

constexpr int kFormExpDiv = 0;
constexpr int kFormTanh = 1;

// ex2.approx.f32 + div.approx.f32, without expf's range reduction or div.rn's
// reciprocal/Newton-Raphson/fixup chain.
__device__ __forceinline__ float silu_exp_div(const float x) {
  return __fdividef(x, 1.0f + __expf(-x));
}

// Inline PTX rather than __tanhf, so the single-instruction lowering is
// guaranteed rather than left to the compiler's discretion.
__device__ __forceinline__ float tanh_approx(const float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// silu(x) = x*sigmoid(x) = 0.5*x*(1 + tanh(0.5*x)).
__device__ __forceinline__ float silu_tanh(const float x) {
  const float half_x = 0.5f * x;
  return half_x * (1.0f + tanh_approx(half_x));
}

template <int kForm>
__device__ __forceinline__ float silu_f32(const float x) {
  if constexpr (kForm == kFormTanh) {
    return silu_tanh(x);
  } else {
    return silu_exp_div(x);
  }
}

// Explicit conversions only: load_inline compiles with
// -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__, and both
// intrinsics below round to nearest even, matching ATen's static_cast back from
// its fp32 accumulator.
template <typename T>
struct Convert;

template <>
struct Convert<__nv_bfloat16> {
  __device__ __forceinline__ static float to_f32(const __nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ __forceinline__ static __nv_bfloat16 from_f32(const float v) {
    return __float2bfloat16(v);
  }
};

template <>
struct Convert<__half> {
  __device__ __forceinline__ static float to_f32(const __half v) {
    return __half2float(v);
  }
  __device__ __forceinline__ static __half from_f32(const float v) {
    return __float2half(v);
  }
};

// uint4 carries the 16-byte alignment the vectorized access needs; the union
// makes the reinterpretation between the access word and its elements explicit.
template <typename T, int kElems>
union AccessWord {
  uint4 raw;
  T elem[kElems];
};

// One tail store, plus its accounting. Bundling them means the counter records
// stores rather than loop iterations, so a duplicated store cannot slip past it.
template <typename T, bool kCountWrites>
__device__ __forceinline__ void tail_store(T* __restrict__ out, int* __restrict__ counts,
                                           const int64_t i, const T value) {
  out[i] = value;
  if constexpr (kCountWrites) {
    atomicAdd(&counts[i], 1);
  }
}

// The trailing (elements % kElems) values. Owned by block 0 alone when kOwnTail:
// every block falls out of the vectorized grid-stride loop, so letting them all
// write this range would make each of those addresses a multi-writer store.
//
// kCountWrites is compiled out of the shipped kernel (counts is nullptr there). It
// exists so tests/kernel_probes.py can instantiate *this* body with an atomic
// counter and prove each tail address is written exactly once, instead of proving
// it about a separate copy of the loop.
template <typename T, int kForm, bool kOwnTail = true, bool kCountWrites = false>
__device__ __forceinline__ void silu_tail(const T* __restrict__ in, T* __restrict__ out,
                                          int* __restrict__ counts, const int64_t words,
                                          const int64_t elements) {
  constexpr int kElems = kBytesPerAccess / sizeof(T);
  if constexpr (kOwnTail) {
    if (blockIdx.x != 0) {
      return;
    }
  }
  for (int64_t i = words * kElems + threadIdx.x; i < elements; i += blockDim.x) {
    tail_store<T, kCountWrites>(
        out, counts, i, Convert<T>::from_f32(silu_f32<kForm>(Convert<T>::to_f32(in[i]))));
  }
}

template <typename T, int kForm>
__global__ void silu_vectorized_kernel(const T* __restrict__ in,
                                       T* __restrict__ out,
                                       const int64_t words,
                                       const int64_t elements) {
  constexpr int kElems = kBytesPerAccess / sizeof(T);

  const uint4* __restrict__ in_words = reinterpret_cast<const uint4*>(in);
  uint4* __restrict__ out_words = reinterpret_cast<uint4*>(out);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += stride) {
    AccessWord<T, kElems> word;
    word.raw = in_words[i];
#pragma unroll
    for (int k = 0; k < kElems; ++k) {
      word.elem[k] = Convert<T>::from_f32(silu_f32<kForm>(Convert<T>::to_f32(word.elem[k])));
    }
    out_words[i] = word.raw;
  }

  silu_tail<T, kForm>(in, out, nullptr, words, elements);
}

template <typename T, int kForm>
void launch(const at::Tensor& in, at::Tensor& out, const int block, const int words_per_thread,
            const int waves_per_sm) {
  constexpr int kElems = kBytesPerAccess / sizeof(T);
  const int64_t elements = in.numel();
  const int64_t words = elements / kElems;

  // ATen caches cudaDeviceProp per device, so this is a per-device lookup
  // rather than a process-wide static.
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // Two independent controls: words_per_thread sets how much work each thread is
  // asked for, which fixes the covering grid; waves_per_sm then caps that grid.
  const int64_t per_block = static_cast<int64_t>(block) * words_per_thread;
  const int64_t covering = (words + per_block - 1) / per_block;
  const int64_t capped = std::min<int64_t>(covering, static_cast<int64_t>(sm_count) * waves_per_sm);
  // Never zero: elements < kElems leaves words == 0 but still has a tail.
  const int64_t blocks = std::max<int64_t>(capped, 1);

  silu_vectorized_kernel<T, kForm><<<static_cast<unsigned int>(blocks),
                                     static_cast<unsigned int>(block), 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const T*>(in.const_data_ptr()), static_cast<T*>(out.data_ptr()), words,
      elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_launch(const at::Tensor& in, at::Tensor& out, const int64_t form, const int block,
                     const int words_per_thread, const int waves_per_sm) {
  const bool tanh_form = (form == kFormTanh);
  if (in.scalar_type() == at::kBFloat16) {
    if (tanh_form) {
      launch<__nv_bfloat16, kFormTanh>(in, out, block, words_per_thread, waves_per_sm);
    } else {
      launch<__nv_bfloat16, kFormExpDiv>(in, out, block, words_per_thread, waves_per_sm);
    }
  } else {
    if (tanh_form) {
      launch<__half, kFormTanh>(in, out, block, words_per_thread, waves_per_sm);
    } else {
      launch<__half, kFormExpDiv>(in, out, block, words_per_thread, waves_per_sm);
    }
  }
}

bool is_access_aligned(const void* p) {
  return reinterpret_cast<uintptr_t>(p) % kBytesPerAccess == 0;
}

}  // namespace

at::Tensor silu_forward(const at::Tensor& x, const int64_t form, const int64_t block,
                        const int64_t words_per_thread, const int64_t waves_per_sm) {
  if (!x.is_cuda() || !x.is_contiguous()) {
    return at::silu(x);
  }
  if (at::GradMode::is_enabled() && x.requires_grad()) {
    return at::silu(x);
  }
  const auto dtype = x.scalar_type();
  if (dtype != at::kBFloat16 && dtype != at::kHalf) {
    return at::silu(x);
  }

  TORCH_CHECK(block > 0 && block <= 1024 && block % 32 == 0,
              "block must be a positive multiple of 32 up to 1024, got ", block);
  TORCH_CHECK(words_per_thread > 0, "words_per_thread must be positive, got ", words_per_thread);
  TORCH_CHECK(waves_per_sm > 0, "waves_per_sm must be positive, got ", waves_per_sm);

  const at::cuda::CUDAGuard device_guard(x.device());
  auto out = at::empty_like(x);
  if (x.numel() == 0) {
    return out;  // a zero-block grid is an invalid launch configuration
  }
  // Any view whose storage offset breaks 16-byte alignment goes back to ATen.
  if (!is_access_aligned(x.const_data_ptr()) || !is_access_aligned(out.data_ptr())) {
    return at::silu(x);
  }

  dispatch_launch(x, out, form, static_cast<int>(block), static_cast<int>(words_per_thread),
                  static_cast<int>(waves_per_sm));
  return out;
}
"""

_CPP_SOURCE = r"""
at::Tensor silu_forward(const at::Tensor& x, int64_t form, int64_t block,
                        int64_t words_per_thread, int64_t waves_per_sm);
"""


def _build_directory() -> str | None:
    """Persistent per-workspace build cache, so a warm import never calls nvcc."""
    try:
        path = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXTENSION_NAME
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None  # fall back to cpp_extension's own default location
    return str(path)


def _load_extension(arch: str | None = _CUDA_ARCH):
    # Pinning the arch list does two things, and only the first is conditional on
    # the environment:
    #   * it keeps the build single-arch. This workspace's shell exports
    #     TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX", so without pinning,
    #     the same source compiles for six architectures instead of one.
    #   * it is outright required wherever the variable is unset or "native": that
    #     branch of _get_cuda_arch_flags iterates torch.cuda.device_count() and then
    #     indexes the resulting list, raising IndexError with no visible GPU. The
    #     failure only surfaces when nvcc is actually invoked, so a warm build cache
    #     hides it -- which is exactly why a fresh workspace needs the pin.
    # ``arch=None`` skips it, which is how that path is tested.
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is None:
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["silu_forward"],
            # -lineinfo lets ncu attribute SASS to source. --use_fast_math is
            # deliberately absent: every approximation here is an explicit,
            # auditable intrinsic.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=_build_directory(),
            verbose=False,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list


# Built at import time, never lazily inside forward: ninja spawns subprocesses,
# and the benchmark harness fails any candidate that grows the thread count
# during its timing window.
EXTENSION_STATUS = ""
try:
    _EXT = _load_extension()
except Exception as exc:  # noqa: BLE001 - a build failure must stay importable
    _EXT = None
    EXTENSION_STATUS = f"{type(exc).__name__}: {exc}"
    print(
        f"[silu] CUDA extension unavailable, falling back to F.silu: {EXTENSION_STATUS}",
        file=sys.stderr,
        flush=True,
    )


def _launch_with(x: torch.Tensor, *, form: int | None = None, block: int | None = None,
                 words_per_thread: int | None = None,
                 waves_per_sm: int | None = None) -> torch.Tensor:
    """The compiled path with individual launch parameters overridden.

    Only the tuning sweeps and the test suite use this; ``forward`` calls the
    extension directly so the scored path carries no extra indirection.
    """
    return _EXT.silu_forward(
        x,
        _FORM if form is None else form,
        _BLOCK if block is None else block,
        _WORDS_PER_THREAD if words_per_thread is None else words_per_thread,
        _WAVES_PER_SM if waves_per_sm is None else waves_per_sm,
    )


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _EXT is None:
            return F.silu(x)
        return _EXT.silu_forward(x, _FORM, _BLOCK, _WORDS_PER_THREAD, _WAVES_PER_SM)
