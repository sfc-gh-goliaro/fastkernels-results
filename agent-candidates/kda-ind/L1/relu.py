"""ReLU on B200 via a 128-bit-wide CUDA elementwise kernel, with an F.relu fallback.

What this is
------------
A drop-in replacement for ``F.relu`` backed by one CUDA C++ translation unit
JIT-compiled at import through ``load_inline``. The kernel moves 16 bytes per
thread (``LDG.E.128`` / ``STG.E.128``, confirmed in SASS) and computes ReLU with
integer operations on the raw float bits. Any input the kernel does not claim --
CPU, unsupported dtype, non-contiguous, empty, gradient-tracking -- goes to
``F.relu``, and so does everything if the build fails.

Measured latency (B200, 148 SMs, 126.5 MiB L2, torch 2.11.0+cu130 / CUDA 13.0)
-----------------------------------------------------------------------------
The benchmark's timed window is ``pool.next() + module(x=...)``, so a full
device-to-device copy of the input is inside every measurement and is not under
this module's control. Median of 50 after 10 warmups, 253 MiB L2 flush before
each iteration, six to ten independent repeats:

    copy alone, no relu at all         7.168 us   <- the floor of the metric
    empty kernel, grid-matched        +2.048 us   (up to 768 CTAs)
    a *live* 128-bit load, no store   +4.06  us
    a 128-bit store, no load          +2.02 us (49 K) .. +3.98 us (786 K)
    F.relu                            11.264 - 11.280 us
    this kernel                       11.216 - 11.312 us

A live load alone already costs what the whole operator costs, so on the read
side there is nothing left to recover: any correct ReLU must read every input
element. Across ten independent repeats of all five captured shapes this kernel
and ``F.relu`` trade places at random inside a +/- 0.6 % band whose measured
run-to-run spread is 0.57 %.

The supported claim is: **no qualifying speedup and no material regression.** Four
official ``validate.py`` runs pass with every per-shape speedup in 0.9943-1.0071,
none of which reproduces in sign or magnitude, and the count-weighted mean lands at
1.0012, 1.0026, 1.0018 and 0.9968 -- on both sides of parity, and never near the
1.01x bar for recording a win.

Two auxiliary randomized interleaved paired sessions (A/B, plus A/A and C/C null
controls) were run and are archived under ``metrics/``. One did resolve a favourable
difference on the dominant shape, whose count-weighted aggregate was 1.0032x --
below the 1.01x bar -- and which the second session did not reproduce. The A/A null
control also produced an interval excluding zero, about 14x smaller in magnitude, so
an interval excluding zero at this pair count is not on its own evidence of a real
difference. The numbers live in ``docs/phase1-findings.md`` section 1.1 rather than
being restated here -- one place, so they cannot drift.

That +4.1 us marginal cost is flat in problem size, flat across 212 launch
configurations, and did not change when the input had just been read -- a second
ReLU over the same bytes costs the same +4.14 us as the first -- so it is not a
cold-start effect. Whether it is warp-level latency exposure at 0.32 waves per SM
or a dispatch-side granularity is not settled here; see
``profile/relu_b200_vec128_vs_aten/REPORT.md``.

That parity is a property of the captured shapes, not of the kernel. The same
kernel is genuinely faster once the problem leaves the latency plateau:

    numel      F.relu     this kernel
    786432    11.264 us    11.264 us    1.00x
      2 M     11.248 us    11.232 us    1.00x
      4 M     15.344 us    13.296 us    1.15x
     16 M     29.696 us    25.520 us    1.16x

The five captured shapes span 49152 - 786432 elements, all inside the plateau.

Chosen configuration, and why
-----------------------------
16 bytes per thread, one vector per thread, 256 threads per block, exact grid with
a bounds check. A sweep of access width (32/64/128 bit and scalar) x vectors per
thread (1/2/4/8) x block size (32/64/128/256/512) x exact-vs-grid-stride x
``__launch_bounds__`` present-or-absent -- 212 configurations on the dominant shape
and 211 on the smallest, each against an empty kernel launched at its *own* block
count and block size -- produced no configuration that beat any other outside the
noise floor. Two independent high-repeat runs ranked them in different orders. So
the choice is made on other grounds: 128-bit is the widest single access, one
vector per thread is the smallest register footprint, and 256 threads keeps the
grid at 384 CTAs on the largest captured shape, inside the observed CTA range where
the grid-matched empty-kernel control costs 2.048 us rather than the 4.096 us it
costs at 1536 CTAs and above.

Element operation, and why it is integer code
---------------------------------------------
``F.relu`` on CUDA preserves a NaN's sign and payload bit for bit, and maps -0.0
to +0.0. Four plausible formulations were compared bitwise against ``F.relu`` on
the same device over
``{+-0.0, +-1.0, +-2.5, +-min_norm, +-subnormal, +-Inf, qNaN+-, sNaN+-}``:

    __hmax2_nan(v, 0)          returns canonical 0x7fff for every NaN -- wrong
    __hmax2(v, 0)              NaN-quieting, NaN -> +0.0 -- wrong
    x < 0 ? 0 : x              returns -0.0 for -0.0 -- wrong
    x <= 0 ? 0 : x             correct for bfloat16/float16, but for float nvcc
                               contracts it into FMNMX, which canonicalizes NaN
    integer form below         bit-exact for all three dtypes

The integer form cannot be folded into a float min/max because it never enters the
float domain. The comparison must be against a CUDA reference: CPU ``F.relu``
returns -0.0 for -0.0 and would pass a wrong kernel.

Deliberate limitations
----------------------
* The kernel handles contiguous CUDA bfloat16 / float16 / float32 only. Everything
  else, including gradient-tracking inputs, delegates to ``F.relu`` -- there is no
  backward formula here, so falling back is what keeps this a real drop-in rather
  than something that silently detaches a graph.
* Non-contiguous inputs are not gathered; they go to ``F.relu``.
* The op is registered with ``torch.library`` so tracing and ``FakeTensor`` work,
  but registration failing is non-fatal: the extension is then called directly.
* The extension also exports ``relu_into(input, out) -> path_code``, which exists
  only so the self-check can supply a misaligned output and assert which of the
  wide / wide+tail / scalar paths ran. ``forward`` never calls it. It shares the
  same ``dispatch`` as ``relu_wide``, so what it exercises is the real path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VECTOR_BYTES = 16
_BLOCK = 256
# Mirrors the kPath* constants in the CUDA source; returned by the relu_into test hook.
PATH_WIDE, PATH_WIDE_TAIL, PATH_SCALAR = 0, 1, 2
PATH_NAMES = {PATH_WIDE: "wide", PATH_WIDE_TAIL: "wide_tail", PATH_SCALAR: "scalar"}
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_OP_NAMESPACE = "fastkernels_candidate_relu"

_CPP_SOURCE = """
#include <torch/extension.h>
at::Tensor relu_wide(const at::Tensor& x);
int64_t relu_into(const at::Tensor& x, at::Tensor out);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

// One traits struct per float format. Only the sign bit and the largest finite
// bit pattern are needed: everything above MAX_FINITE in magnitude is a NaN.
struct Bf16Format {
    using Bits = unsigned short;
    static constexpr Bits SIGN = 0x8000u;
    static constexpr Bits MAX_FINITE = 0x7F80u;
};
struct Fp16Format {
    using Bits = unsigned short;
    static constexpr Bits SIGN = 0x8000u;
    static constexpr Bits MAX_FINITE = 0x7C00u;
};
struct Fp32Format {
    using Bits = unsigned int;
    static constexpr Bits SIGN = 0x80000000u;
    static constexpr Bits MAX_FINITE = 0x7F800000u;
};

// ReLU as bit manipulation. A negative number -- which includes -0.0 -- becomes
// +0.0, matching CUDA F.relu. A NaN has magnitude above +Inf, so it falls through
// with its sign and payload intact, which CUDA F.relu also does. Staying in the
// integer domain is deliberate: the equivalent float expression
// `x <= 0 ? 0 : x` gets contracted into FMNMX, which canonicalizes NaN payloads.
template <typename Format>
__device__ __forceinline__ typename Format::Bits relu_bits(typename Format::Bits b) {
    using Bits = typename Format::Bits;
    const Bits magnitude = static_cast<Bits>(b & static_cast<Bits>(~Format::SIGN));
    const bool negative_number =
        (b & Format::SIGN) != 0 && magnitude <= Format::MAX_FINITE;
    return negative_number ? static_cast<Bits>(0) : b;
}

// 16 bytes per thread. `tail_lanes` elements are left over when the element count
// is not a whole number of vectors; one extra thread mops them up so that any
// aligned input still costs a single kernel launch.
template <typename Format>
__global__ void relu_vector_kernel(const uint4* __restrict__ in,
                                   uint4* __restrict__ out,
                                   long long vector_count,
                                   int tail_lanes) {
    using Bits = typename Format::Bits;
    constexpr int LANES = static_cast<int>(sizeof(uint4) / sizeof(Bits));
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < vector_count) {
        uint4 v = in[i];
        Bits* lane = reinterpret_cast<Bits*>(&v);
#pragma unroll
        for (int k = 0; k < LANES; ++k) lane[k] = relu_bits<Format>(lane[k]);
        out[i] = v;
    } else if (i == vector_count && tail_lanes > 0) {
        const Bits* src = reinterpret_cast<const Bits*>(in) + vector_count * LANES;
        Bits* dst = reinterpret_cast<Bits*>(out) + vector_count * LANES;
        for (int k = 0; k < tail_lanes; ++k) dst[k] = relu_bits<Format>(src[k]);
    }
}

// The path for pointers that are not 16-byte aligned. Never taken by the
// benchmark -- its pool shifts by 256 bytes per iteration -- but a storage-offset
// view such as `t[1:]` reaches it.
template <typename Format>
__global__ void relu_scalar_kernel(const typename Format::Bits* __restrict__ in,
                                   typename Format::Bits* __restrict__ out,
                                   long long n) {
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = relu_bits<Format>(in[i]);
}

namespace {
constexpr int kBlock = 256;
constexpr int kVectorBytes = 16;

// Which of the three code paths a launch took. Returned so a test can assert the
// path instead of inferring it from the result -- the wide, tail and scalar paths
// all produce identical values, so a correctness check alone cannot tell them
// apart, and an untested path would look tested.
constexpr int64_t kPathWide = 0;      // 16 B per thread, no leftover lanes
constexpr int64_t kPathWideTail = 1;  // 16 B per thread plus a ragged remainder
constexpr int64_t kPathScalar = 2;    // one lane per thread; a pointer is not 16 B aligned

inline bool is_wide_aligned(const void* p) {
    return reinterpret_cast<uintptr_t>(p) % kVectorBytes == 0;
}

template <typename Format>
int64_t launch(const void* ip, void* op, long long n, cudaStream_t stream) {
    using Bits = typename Format::Bits;
    constexpr int LANES = static_cast<int>(sizeof(uint4) / sizeof(Bits));
    if (is_wide_aligned(ip) && is_wide_aligned(op)) {
        const long long vector_count = n / LANES;
        const int tail_lanes = static_cast<int>(n % LANES);
        const long long threads_needed = vector_count + (tail_lanes > 0 ? 1 : 0);
        const long long blocks = (threads_needed + kBlock - 1) / kBlock;
        relu_vector_kernel<Format><<<blocks, kBlock, 0, stream>>>(
            reinterpret_cast<const uint4*>(ip), reinterpret_cast<uint4*>(op),
            vector_count, tail_lanes);
        return tail_lanes > 0 ? kPathWideTail : kPathWide;
    }
    const long long blocks = (n + kBlock - 1) / kBlock;
    relu_scalar_kernel<Format><<<blocks, kBlock, 0, stream>>>(
        reinterpret_cast<const Bits*>(ip), reinterpret_cast<Bits*>(op), n);
    return kPathScalar;
}

// The single dispatch point. Both entry points below go through here, so a test
// driving `relu_into` exercises exactly the code path `relu_wide` would take.
int64_t dispatch(const at::Tensor& x, at::Tensor& out, cudaStream_t stream) {
    const long long n = x.numel();
    const void* ip = x.const_data_ptr();
    void* op = out.data_ptr();
    int64_t path = kPathWide;
    switch (x.scalar_type()) {
        case at::kBFloat16: path = launch<Bf16Format>(ip, op, n, stream); break;
        case at::kHalf:     path = launch<Fp16Format>(ip, op, n, stream); break;
        case at::kFloat:    path = launch<Fp32Format>(ip, op, n, stream); break;
        default:
            TORCH_CHECK(false, "relu: unsupported dtype ", x.scalar_type());
    }
    // Fail here rather than in some later, unrelated synchronization.
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return path;
}
}  // namespace

at::Tensor relu_wide(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "relu_wide: expected a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "relu_wide: expected a contiguous tensor");
    // Guard before empty_like: without it the allocation and the launch would go
    // to the current device rather than to x's device.
    const at::cuda::CUDAGuard guard(x.device());
    at::Tensor out = at::empty_like(x);
    if (x.numel() == 0) return out;
    dispatch(x, out, at::cuda::getCurrentCUDAStream());
    return out;
}

// Test hook. Writes into a caller-provided output so a test can supply a
// deliberately misaligned destination -- which `empty_like` never produces, since
// the caching allocator hands back 512-byte-aligned blocks -- and returns the path
// code so the wide / wide+tail / scalar choice is observable. Not used by forward.
int64_t relu_into(const at::Tensor& x, at::Tensor out) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda(), "relu_into: expected CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(),
                "relu_into: expected contiguous tensors");
    TORCH_CHECK(x.scalar_type() == out.scalar_type(), "relu_into: dtype mismatch");
    TORCH_CHECK(x.sizes() == out.sizes(), "relu_into: shape mismatch");
    TORCH_CHECK(x.device() == out.device(), "relu_into: device mismatch");
    const at::cuda::CUDAGuard guard(x.device());
    if (x.numel() == 0) return kPathWide;
    return dispatch(x, out, at::cuda::getCurrentCUDAStream());
}
"""

# Set by _load_extension(). A build failure leaves the handle None and the reason
# in _BUILD_ERROR, which is the only record of it -- import must not raise, or the
# whole operator is reported as a runtime error instead of falling back.
_EXTENSION = None
_BUILD_ERROR: str | None = None


def _target_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _load_extension():
    """Compile at import time, into a pinned directory, without ever raising.

    Import time is the right moment: the benchmark snapshots its integrity before
    importing the candidate and counts threads only around its timing loop, so
    ninja's workers are long gone by then. A first-call build would put them
    inside the measured region.
    """
    global _EXTENSION, _BUILD_ERROR
    if not torch.cuda.is_available():
        _BUILD_ERROR = "no CUDA device available at import; using F.relu"
        return
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        from torch.utils.cpp_extension import load_inline

        arch = _target_arch()
        # The name carries a hash of the source so an edit can never silently
        # reuse a stale .so from the pinned directory.
        digest = hashlib.sha1(
            (_CUDA_SOURCE + _CPP_SOURCE).encode("utf-8")).hexdigest()[:10]
        name = f"relu_wide_sm{arch.replace('.', '')}_{digest}"
        build_dir = (Path(__file__).resolve().parents[2]
                     / ".torch_extensions" / name)
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        _EXTENSION = load_inline(
            name=name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["relu_wide", "relu_into"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - degrading to F.relu beats not importing
        _EXTENSION = None
        _BUILD_ERROR = f"{type(exc).__name__}: {exc}"
    finally:
        # Leaving this mutated would change how unrelated extensions compile.
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list


_load_extension()


def _register_op():
    """Expose the kernel as a real operator so tracing and FakeTensor work.

    Optional by design: if registration is unavailable or the name is already
    taken, the extension is called directly instead.
    """
    if _EXTENSION is None:
        return None
    try:
        existing = getattr(torch.ops, _OP_NAMESPACE, None)
        if existing is not None and hasattr(existing, "relu"):
            return existing.relu

        @torch.library.custom_op(f"{_OP_NAMESPACE}::relu", mutates_args=())
        def relu(x: torch.Tensor) -> torch.Tensor:
            return _EXTENSION.relu_wide(x)

        @relu.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        return relu
    except Exception:  # noqa: BLE001 - registration is a convenience, not a requirement
        return None


_RELU_OP = _register_op()


def _kernel_call(x: torch.Tensor) -> torch.Tensor:
    if _RELU_OP is not None:
        return _RELU_OP(x)
    return _EXTENSION.relu_wide(x)


def _can_use_kernel(x: torch.Tensor) -> bool:
    """Every condition the kernel relies on, all of them cheap host-side checks.

    Host cost is free here: the benchmark's 253 MiB L2 flush leaves roughly 45 us
    of host slack per iteration, so the CPU runs far ahead of the GPU and these
    checks never reach the measured window.
    """
    return (
        _EXTENSION is not None
        and x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and x.is_contiguous()
        and x.numel() > 0
        # No backward formula lives here, so a graph-tracking input must go to
        # F.relu rather than be silently detached.
        and not x.requires_grad
    )


def build_report() -> dict[str, object]:
    """Inspectable build state, so a silent fallback is still diagnosable."""
    return {
        "extension_loaded": _EXTENSION is not None,
        "registered_op": _RELU_OP is not None,
        "build_error": _BUILD_ERROR,
        "vector_bytes": _VECTOR_BYTES,
        "block": _BLOCK,
        "supported_dtypes": [str(d) for d in _SUPPORTED_DTYPES],
    }


class ReLU(nn.Module):
    """``F.relu`` with a 128-bit CUDA kernel underneath, and ``F.relu`` behind that."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _can_use_kernel(x):
            return _kernel_call(x)
        return F.relu(x)
