"""ReLU activation: max(0, x).

Hand-written elementwise CUDA kernel. The measured cost of a ReLU at these
captured sizes (49K-786K bf16 elements, 0.2-3.1 MB of read+write traffic) is
almost entirely per-launch serialization, not bandwidth, so the kernel is
built around three ideas:

* **Programmatic Dependent Launch (sm_90+).** The kernel is launched with
  ``cudaLaunchAttributeProgrammaticStreamSerialization`` and gates its first
  load on ``cudaGridDependencySynchronize()``. The grid becomes resident and
  does its address arithmetic *while the producer kernel ahead of it in the
  stream is still draining*, so the inter-kernel launch latency is overlapped
  instead of paid. Correctness is unaffected: the barrier still orders every
  load after all preceding writes on the stream.
* **1024 bytes per block.** Whether the launch latency is hidden turns out to
  be a per-call coin flip whose odds depend on how long the kernel runs, and
  moving exactly ``BS * WB == 1024`` bytes per block is the measured optimum:
  the three best configurations of a 12-point (access width x block size)
  sweep are precisely the three that move 1024 B/block, and everything at
  512 or >=1536 B/block is far worse. See ``ITERATIONS.md``.
* **4 bytes per thread.** Within that family -- 4B/256, 8B/128, 16B/64 all
  share the same grid -- the *narrowest* access wins, because it puts four
  times as many threads on the same bytes and the kernel finishes sooner.
  A 4-byte access also needs only 4-byte alignment, so the branch-free fast
  path covers strictly more inputs than a 16-byte one would.

ReLU itself is done as a sign-bit mask on packed 16-bit lanes
(``__vcmpgeu2`` / ``__vcmpgtu2``), never unpacking to fp32, and is bit-exact
with ``F.relu`` including -0.0 -> +0.0, -inf -> +0.0 and NaN payloads passed
through unchanged.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

if torch.cuda.is_available():  # pin arch: avoids compiling 7 unused gencodes
    _cap = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cap[0]}.{_cap[1]}"

from torch.utils.cpp_extension import load_inline  # noqa: E402

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdint>

// MODE 0: two bf16 lanes per word.  1: two fp16 lanes.  2: one fp32 lane.
//
// relu(v) == F.relu(v) bit-for-bit:
//   NaN   -> v unchanged (payload and sign preserved)
//   -0.0  -> +0.0        (sign bit set, not NaN -> masked to zero)
//   -inf  -> +0.0
//   +x    -> v unchanged
//
// A 16-bit lane widened to a word has a zero upper half, which is inert in
// both packed compares (+0.0 is neither NaN nor negative), so the same code
// serves the per-lane tail paths.
template <int MODE>
__device__ __forceinline__ unsigned int relu_word(unsigned int v) {
    if (MODE == 0) {
        unsigned int nan = __vcmpgtu2(v & 0x7fff7fffu, 0x7f807f80u);
        unsigned int neg = __vcmpgeu2(v, 0x80008000u);
        return v & (~neg | nan);
    } else if (MODE == 1) {
        unsigned int nan = __vcmpgtu2(v & 0x7fff7fffu, 0x7c007c00u);
        unsigned int neg = __vcmpgeu2(v, 0x80008000u);
        return v & (~neg | nan);
    } else {
        bool nan = (v & 0x7fffffffu) > 0x7f800000u;
        return (v >= 0x80000000u && !nan) ? 0u : v;
    }
}

#define BS 256                 // threads per block
#define WB 4                   // bytes per thread -> BS * WB == 1024 B/block

// Fast path: one 4B word per thread, grid covers the buffer exactly.
template <int MODE>
__global__ __launch_bounds__(BS) void relu_v(
        const unsigned int* __restrict__ in, unsigned int* __restrict__ out) {
    unsigned int i = blockIdx.x * BS + threadIdx.x;
    cudaGridDependencySynchronize();   // orders the load after the producer
    out[i] = relu_word<MODE>(in[i]);
}

// Generic: bounds-checked word body + a per-lane tail (nw words, then ntail
// lanes starting at in_t/out_t).  ntail is 0 or 1: only a 16-bit dtype with an
// odd element count leaves a partial word.
template <int MODE, typename LANE>
__global__ __launch_bounds__(BS) void relu_v_tail(
        const unsigned int* __restrict__ in, unsigned int* __restrict__ out,
        unsigned int nw, unsigned int ntail,
        const LANE* __restrict__ in_t, LANE* __restrict__ out_t) {
    unsigned int i = blockIdx.x * BS + threadIdx.x;
    cudaGridDependencySynchronize();
    if (i < nw) out[i] = relu_word<MODE>(in[i]);
    else if (i - nw < ntail) {
        unsigned int j = i - nw;
        out_t[j] = (LANE)relu_word<MODE>((unsigned int)in_t[j]);
    }
}

// Base pointer not 4B-aligned (a 16-bit view starting on an odd element):
// per-lane only.
template <int MODE, typename LANE>
__global__ __launch_bounds__(BS) void relu_lane(const LANE* __restrict__ in,
                                                LANE* __restrict__ out,
                                                unsigned int n) {
    unsigned int i = blockIdx.x * BS + threadIdx.x;
    cudaGridDependencySynchronize();
    if (i < n) out[i] = (LANE)relu_word<MODE>((unsigned int)in[i]);
}

// >= 8 GiB, where a 32-bit lane index would wrap: grid-stride with 64-bit
// indices.  Unreachable for the captured shapes; here so the contract holds.
template <int MODE, typename LANE>
__global__ __launch_bounds__(BS) void relu_huge(const LANE* __restrict__ in,
                                                LANE* __restrict__ out,
                                                size_t n) {
    size_t i = (size_t)blockIdx.x * BS + threadIdx.x;
    size_t stride = (size_t)gridDim.x * BS;
    cudaGridDependencySynchronize();
    for (; i < n; i += stride)
        out[i] = (LANE)relu_word<MODE>((unsigned int)in[i]);
}

// Every launch carries the PDL attribute. With no producer ahead of it in the
// stream the barrier is satisfied immediately, so this is never a pessimisation.
template <typename K, typename... A>
static inline void launch_pdl(K kernel, unsigned grid, cudaStream_t s, A... a) {
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(BS);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = s;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, a...);
}

static const int64_t HUGE_BYTES = (int64_t)1 << 33;

template <int MODE, typename LANE>
static void launch_typed(const void* in, void* out, int64_t nbytes,
                         cudaStream_t s) {
    const unsigned lanes_per_word = WB / (unsigned)sizeof(LANE);
    bool aligned = (((uintptr_t)in | (uintptr_t)out) & (WB - 1)) == 0;
    if (nbytes >= HUGE_BYTES) {
        size_t n = (size_t)nbytes / sizeof(LANE);
        size_t need = (n + BS - 1) / BS;
        unsigned grid = (unsigned)(need < (1u << 20) ? need : (1u << 20));
        launch_pdl(relu_huge<MODE, LANE>, grid, s,
                   (const LANE*)in, (LANE*)out, n);
    } else if (aligned && (nbytes % (BS * WB)) == 0) {
        launch_pdl(relu_v<MODE>, (unsigned)(nbytes / (BS * WB)), s,
                   (const unsigned int*)in, (unsigned int*)out);
    } else if (aligned) {
        unsigned nw = (unsigned)(nbytes / WB);
        unsigned ntail = (unsigned)((nbytes % WB) / sizeof(LANE));
        unsigned total = nw + ntail;
        launch_pdl(relu_v_tail<MODE, LANE>, (total + BS - 1) / BS, s,
                   (const unsigned int*)in, (unsigned int*)out, nw, ntail,
                   (const LANE*)in + (size_t)nw * lanes_per_word,
                   (LANE*)out + (size_t)nw * lanes_per_word);
    } else {
        unsigned n = (unsigned)(nbytes / sizeof(LANE));
        launch_pdl(relu_lane<MODE, LANE>, (n + BS - 1) / BS, s,
                   (const LANE*)in, (LANE*)out, n);
    }
}

// mode: 0 bf16, 1 fp16, 2 fp32
extern "C" void ako_relu_launch(const void* in, void* out, int64_t nbytes,
                                int mode, cudaStream_t s) {
    if (nbytes <= 0) return;
    if (mode == 0) launch_typed<0, unsigned short>(in, out, nbytes, s);
    else if (mode == 1) launch_typed<1, unsigned short>(in, out, nbytes, s);
    else launch_typed<2, unsigned int>(in, out, nbytes, s);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

extern "C" void ako_relu_launch(const void*, void*, int64_t, int, cudaStream_t);

static inline int mode_of(at::ScalarType t) {
    switch (t) {
        case at::kBFloat16: return 0;
        case at::kHalf:     return 1;
        case at::kFloat:    return 2;
        default:            return -1;
    }
}

static at::Tensor relu_core(const at::Tensor& x) {
    const int mode = mode_of(x.scalar_type());
    TORCH_CHECK(x.is_cuda() && mode >= 0,
                "relu kernel supports CUDA bfloat16/float16/float32, got ",
                x.toString(), " on ", x.device());
    const c10::cuda::CUDAGuard guard(x.device());
    const cudaStream_t s = at::cuda::getCurrentCUDAStream();

    // Dense (contiguous or e.g. channels-last) inputs whose empty_like shares
    // their layout map element-for-element in memory order, so a flat pass is
    // exactly right and keeps F.relu's output layout.
    if (x.is_non_overlapping_and_dense()) {
        at::Tensor out = at::empty_like(x);
        if (out.strides().equals(x.strides())) {
            ako_relu_launch(x.const_data_ptr(), out.data_ptr(),
                            x.numel() * x.element_size(), mode, s);
            return out;
        }
    }
    at::Tensor xc = x.contiguous();
    at::Tensor out = at::empty_like(xc);
    ako_relu_launch(xc.const_data_ptr(), out.data_ptr(),
                    xc.numel() * xc.element_size(), mode, s);
    return out;
}

// METH_O: no pybind argument parsing on the hot path.
static PyObject* py_relu(PyObject*, PyObject* arg) {
    HANDLE_TH_ERRORS
    if (!THPVariable_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "relu() expected a Tensor");
        return nullptr;
    }
    return THPVariable_Wrap(relu_core(THPVariable_Unpack(arg)));
    END_HANDLE_TH_ERRORS
}

static PyMethodDef k_methods[] = {
    {"relu", (PyCFunction)py_relu, METH_O, "elementwise relu"},
    {nullptr, nullptr, 0, nullptr},
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    PyModule_AddObject(m.ptr(), "relu", PyCFunction_New(&k_methods[0], nullptr));
}
"""

# load_inline caches the built .so by *name*, not by source, so the name
# carries a source hash -- editing the kernel can never serve a stale binary.
_NAME = "ako_relu_pdl_" + hashlib.sha1(
    (_CPP_SRC + _CUDA_SRC).encode()).hexdigest()[:12]

_C = load_inline(
    name=_NAME,
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=None,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

_relu = _C.relu

if torch.cuda.is_available():  # load the CUDA module now, not on the first call
    _relu(torch.zeros(512, dtype=torch.bfloat16, device="cuda"))


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _relu(x)
