"""QuickGELU activation: x * sigmoid(1.702 * x), fused into a single CUDA kernel.

The baseline expression evaluates as three elementwise kernels (mul, sigmoid, mul).
At the captured shape -- float32[1, 77, 3072], 0.90 MiB -- the benchmark harness charges a
fixed cost per kernel launch rather than per byte moved: inside one event pair, k copies of a
torch elementwise op cost 2.9 + 4.1k us, and a 1-element kernel costs the same as a
236,544-element one. Collapsing three launches into one is therefore the entire available win;
0.90 MiB is far below this GPU's bandwidth-delay product, so there is no bandwidth to recover.
Evidence: profile/phase1_launch_model/analysis/cost-model.md.

That cost model also says what is *not* a win here. The 128-bit vectorized access, the grid-stride
loop and the launch geometry below are free at this shape -- any legal configuration measures the
same. They are written this way for robustness at other shapes, not for measured latency at this
one.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path


import torch
import torch.nn as nn

# Dtypes the fused kernel handles. Everything else (fp64, integers, a CPU tensor, ...) still works
# -- it routes to the torch expression at the bottom of this file.
#
# The half types reproduce torch's rounding, not just its arithmetic: torch evaluates
# `x * sigmoid(1.702 * x)` as a sequence of half-precision tensor ops, so each intermediate is
# rounded back to the storage dtype before the next one. The kernel does the same, which is why the
# half paths agree with torch exactly rather than merely within the harness's (1e-2, 1e-2) budget.
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_EXTENSION_NAME = "quickgelu_fused"

_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor quick_gelu(const at::Tensor& x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quick_gelu", &quick_gelu, "Fused QuickGELU: x * sigmoid(1.702 * x)");
}
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

// Cap on resident blocks, as a multiple of the SM count. The captured shape needs 231 blocks and
// so never reaches this cap; it exists only to bound the grid for arbitrarily large inputs. The
// exact multiplier is one of the choices the cost model shows to be free at this shape.
constexpr int kMaxBlocksPerSM = 32;

template <typename scalar_t>
__device__ __forceinline__ scalar_t quick_gelu_elem(scalar_t xv) {
    // Mirrors torch's own evaluation, including where it rounds.
    //
    // torch computes this as three tensor ops, so for a half input each intermediate is rounded
    // back to the storage dtype before the next op consumes it. Casting through scalar_t at the
    // same two points reproduces that exactly. For float, scalar_t is float and both casts are
    // identities, so this is the plain fp32 expression `xv * (1 / (1 + expf(-1.702f * xv)))`.
    //
    // Accurate expf rather than __expf: the fast intrinsic would sit near 1e-6 relative error,
    // comfortably inside the harness budget, but exact agreement is worth more than an
    // approximation that buys nothing measurable here.
    const float x = static_cast<float>(xv);
    const scalar_t t = static_cast<scalar_t>(1.702f * x);
    const scalar_t s = static_cast<scalar_t>(1.0f / (1.0f + expf(-static_cast<float>(t))));
    return static_cast<scalar_t>(x * static_cast<float>(s));
}

// A VEC-wide pack, aligned so that a load or store of one pack is a single 128-bit access when
// VEC covers 16 bytes.
template <typename scalar_t, int VEC>
struct alignas(sizeof(scalar_t) * VEC) Pack {
    scalar_t v[VEC];
};

template <typename scalar_t, int VEC>
__global__ __launch_bounds__(kBlockSize) void quick_gelu_kernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ out,
        int64_t n_packs,
        int64_t n) {
    using pack_t = Pack<scalar_t, VEC>;
    const auto* x_packs = reinterpret_cast<const pack_t*>(x);
    auto* out_packs = reinterpret_cast<pack_t*>(out);

    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    // Grid-stride so any n_packs works under a capped grid, including n_packs < blockDim.x.
    for (int64_t i = tid; i < n_packs; i += stride) {
        pack_t in = x_packs[i];
        pack_t result;
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            result.v[k] = quick_gelu_elem<scalar_t>(in.v[k]);
        }
        out_packs[i] = result;
    }

    // Ragged remainder. VEC is chosen from pointer alignment alone, never from divisibility, so
    // this loop is reachable for any n whose length is not a multiple of VEC -- there is one code
    // path, not a vectorized path plus an unreachable tail.
    for (int64_t i = n_packs * VEC + tid; i < n; i += stride) {
        out[i] = quick_gelu_elem<scalar_t>(x[i]);
    }
}

int64_t grid_blocks(int64_t n_packs) {
    // getCurrentDeviceProperties() hands back ATen's per-device cached cudaDeviceProp, so the SM
    // count costs no query per call and stays correct if the current device changes.
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t needed = (n_packs + kBlockSize - 1) / kBlockSize;
    const int64_t cap = static_cast<int64_t>(sm_count) * kMaxBlocksPerSM;
    return needed < cap ? needed : cap;
}

template <typename scalar_t, int VEC>
void launch_packed(const scalar_t* x, scalar_t* out, int64_t n, cudaStream_t stream) {
    const int64_t n_packs = n / VEC;
    // n_packs can be 0 for n < VEC; the bulk loop then does nothing and the remainder loop does
    // all the work, so the grid must still be at least one block.
    const int64_t blocks = std::max<int64_t>(grid_blocks(n_packs), 1);
    quick_gelu_kernel<scalar_t, VEC><<<blocks, kBlockSize, 0, stream>>>(x, out, n_packs, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_for_dtype(const at::Tensor& x, at::Tensor& out, int64_t n, cudaStream_t stream) {
    const auto* x_ptr = x.const_data_ptr<scalar_t>();
    auto* out_ptr = out.mutable_data_ptr<scalar_t>();

    // Alignment is read from the tensor data pointers, not from the storage pointers: the harness
    // hands us a view into a pool slot at a non-zero storage offset, so the storage base says
    // nothing about the address the kernel will actually read.
    const bool aligned_16b =
        (reinterpret_cast<uintptr_t>(x_ptr) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0);

    constexpr int kVecWidth = 16 / sizeof(scalar_t);  // 128-bit access
    if (aligned_16b) {
        launch_packed<scalar_t, kVecWidth>(x_ptr, out_ptr, n, stream);
    } else {
        launch_packed<scalar_t, 1>(x_ptr, out_ptr, n, stream);
    }
}

}  // namespace

at::Tensor quick_gelu(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "quick_gelu: expected a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "quick_gelu: expected a contiguous tensor");

    const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
    at::Tensor out = at::empty_like(x);

    const int64_t n = x.numel();
    // Ahead of any launch-config arithmetic, so a zero-element input can never produce a
    // zero-block grid. empty_like has already given the caller a correctly shaped result.
    if (n == 0) {
        return out;
    }

    // Launch on torch's current stream. A bare <<<grid, block>>> would go to the legacy default
    // stream, which could race the harness's pool copy that produces this input and could fall
    // outside the event pair that times us.
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Restricted to the three dtypes the Python guard admits; anything else never reaches here.
    AT_DISPATCH_SWITCH(
        x.scalar_type(), "quick_gelu",
        AT_DISPATCH_CASE(at::kFloat, [&] { launch_for_dtype<float>(x, out, n, stream); })
        AT_DISPATCH_CASE(at::kHalf, [&] { launch_for_dtype<at::Half>(x, out, n, stream); })
        AT_DISPATCH_CASE(at::kBFloat16, [&] { launch_for_dtype<at::BFloat16>(x, out, n, stream); }));
    return out;
}
"""


def _log_once(message: str) -> None:
    """Record a build problem on stderr, without ever becoming a failure itself."""
    try:
        print(f"[quickgelu] {message}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a closed stderr must not turn into an import error
        pass


def _probe_writable(directory: Path) -> bool:
    """Can we actually write here? Existence is not writability.

    The probe name is unique per process: a fixed name would let two concurrently probing
    processes unlink each other's file and mis-report the directory as unusable.
    """
    probe = directory / f".write-probe-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        probe.write_text("ok")
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
    return True


def _candidate_build_directories(subdir: str) -> list[Path]:
    """Build-directory choices, best first.

    An explicit directory is always supplied to load_inline. Passing None would hand control to
    torch's default directory, which would then get none of the locking below -- the very failure
    mode this path exists to prevent.
    """
    choices: list[Path] = []

    # Workspace-local, so the ninja cache survives across runs and stays with the operator.
    choices.append(Path(__file__).resolve().parents[2] / ".torch_extensions" / subdir)

    # Torch's own default root, resolved rather than delegated.
    try:
        from torch.utils.cpp_extension import _get_build_directory

        choices.append(Path(_get_build_directory(subdir, False)))
    except Exception:  # noqa: BLE001 - private helper; absence must not be fatal
        env_root = os.environ.get("TORCH_EXTENSIONS_DIR")
        if env_root:
            choices.append(Path(env_root) / subdir)

    # Last resort, always writable in practice.
    choices.append(Path(tempfile.gettempdir()) / f"quickgelu-ext-{os.getuid()}" / subdir)
    return choices


def _select_build_directory(subdir: str) -> Path | None:
    for directory in _candidate_build_directories(subdir):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if _probe_writable(directory):
            return directory
    return None


def _acquire_build_lock(build_dir: Path, timeout_s: float):
    """Serialize cooperating builders on an OS lock, or return None on timeout.

    This is the load-bearing piece of the whole build path. torch's FileBaton is a bare
    os.path.exists() check on a lock file with no owner and no timeout, and every kill path in this
    stack SIGKILLs the process group -- so a baton left behind by a killed builder would make every
    later import spin forever. Deleting it unconditionally is not a fix: an identical source hash
    proves two processes want the same build, not that the other one is dead.

    An OS lock does prove it. flock is released by the kernel when its holder dies, so a baton we
    find *while holding this lock* cannot belong to a living cooperating builder and is therefore
    genuinely stale. The lock file sits beside the build directory rather than inside it, so it also
    covers directory selection.
    """
    lock_path = build_dir.parent / f"{build_dir.name}.buildlock"
    deadline = time.monotonic() + timeout_s
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return None
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                os.close(fd)
                return None
        if time.monotonic() >= deadline:
            # Bounded wait, so a wedged peer degrades to the torch fallback instead of hanging the
            # run until the harness watchdog kills the process group.
            os.close(fd)
            return None
        time.sleep(0.1)


def _release_build_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _clear_stale_baton(build_dir: Path) -> None:
    """Remove a torch FileBaton left behind by a killed builder.

    Only ever called while holding the build lock, which is what makes "stale" a fact rather than a
    guess.
    """
    try:
        (build_dir / "lock").unlink()
    except OSError:
        pass


# How long to wait for another cooperating builder before giving up and using the torch fallback.
# A cold build of this kernel is ~20 s; this is generous but still far below the harness's 600 s
# stall watchdog. A plain literal: nothing at module scope may run code that can fail.
_DEFAULT_BUILD_LOCK_TIMEOUT_S = 300.0

# Test hook: name a stage to fail, to prove the fallback really covers that stage. Never set in
# normal operation. os.environ.get cannot raise, so this is safe at module scope.
_FAULT_STAGE = os.environ.get("QUICKGELU_FAULT_INJECT")


def _maybe_fail(stage: str) -> None:
    if _FAULT_STAGE == stage:
        raise RuntimeError(f"injected fault at stage {stage!r}")


def _build_lock_timeout() -> float:
    """Resolve the build-lock timeout, rejecting any value that would defeat the bounded wait.

    Called from inside the guarded path, never at module scope: `float()` raises on a malformed
    string, and an exception there would escape module import and take the whole bench run down.

    `float()` also accepts "nan" and "inf", which are the dangerous cases rather than the obvious
    ones. `_acquire_build_lock` polls until `time.monotonic() >= deadline`; against a NaN deadline
    that comparison is always false, and an infinite deadline never expires -- either would silently
    convert the bounded wait into exactly the unbounded hang the lock exists to prevent. So the value
    must be finite and strictly positive, and anything else is rejected loudly.
    """
    raw = os.environ.get("QUICKGELU_BUILD_LOCK_TIMEOUT")
    if raw is None:
        return _DEFAULT_BUILD_LOCK_TIMEOUT_S
    timeout = float(raw)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(
            f"QUICKGELU_BUILD_LOCK_TIMEOUT must be a finite positive number, got {raw!r}"
        )
    return timeout


def _load_extension():
    """Compile and return the fused extension, or None if it is unavailable for any reason.

    Building happens at module import, not on the first forward: the harness snapshots
    threading.active_count() immediately before timing and reports a reward hack if it grows, and a
    first-call compile would otherwise land inside a correctness round.

    Every statement that could fail sits inside the one try below, because an exception escaping
    module import would take the entire bench run down -- a build problem must cost us speed, never
    correctness. KeyboardInterrupt and SystemExit are deliberately *not* caught: swallowing an
    operator's Ctrl-C during a long nvcc run, or an interpreter shutdown, would be worse than
    falling back.
    """
    arch_list_was_set = False
    previous_arch_list = None
    lock_fd = None
    try:
        _maybe_fail("entry")
        if not torch.cuda.is_available():
            return None

        _maybe_fail("import")
        from torch.utils.cpp_extension import load_inline

        _maybe_fail("digest")
        digest = hashlib.sha256((_CPP_SOURCE + _CUDA_SOURCE).encode()).hexdigest()[:8]

        _maybe_fail("build_dir")
        build_dir = _select_build_directory(f"{_EXTENSION_NAME}_{digest}")
        if build_dir is None:
            raise RuntimeError("no writable build directory among the candidates")

        _maybe_fail("timeout")
        lock_timeout_s = _build_lock_timeout()

        _maybe_fail("lock")
        lock_fd = _acquire_build_lock(build_dir, lock_timeout_s)
        if lock_fd is None:
            raise RuntimeError(f"could not acquire the build lock within {lock_timeout_s:g}s")
        _clear_stale_baton(build_dir)

        _maybe_fail("capability")
        major, minor = torch.cuda.get_device_capability()

        _maybe_fail("environ")
        previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        # The inherited list spans six architectures; this kernel only ever runs on the local GPU,
        # and nvcc time is charged against the harness's stall watchdog.
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        arch_list_was_set = True

        _maybe_fail("build")
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=None,
            extra_cflags=["-O3"],
            # -O3 here is the host-side optimization level; device optimization is -Xptxas -O3
            # and is already nvcc's default. --generate-line-info costs nothing at runtime and
            # makes SASS map back to these lines when profiling.
            extra_cuda_cflags=["-O3", "--generate-line-info"],
            build_directory=str(build_dir),
            # Keeps ninja progress flowing to the worker log so a cold build does not look like a
            # stalled process to the harness watchdog.
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring: import must not raise
        _log_once(f"fused CUDA extension unavailable, using torch fallback: {exc!r}")
        return None
    finally:
        # Restore only if we got as far as changing it, so a failure before that point leaves the
        # environment exactly as we found it.
        if arch_list_was_set:
            if previous_arch_list is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
        if lock_fd is not None:
            _release_build_lock(lock_fd)


_EXT = _load_extension()


class QuickGELU(nn.Module):
    # No __init__: the module has no parameters, no buffers and no state. The harness feeds a
    # fresh data_ptr every iteration, so nothing about a call may be cached on self.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _EXT is not None
            and x.is_cuda
            and x.dtype in _SUPPORTED_DTYPES
            # Not x.contiguous(): materializing a contiguous copy would add a second kernel
            # inside the measured window, which costs more than the fusion saves.
            and x.is_contiguous()
            and x.numel() > 0
            # The raw kernel has no autograd formula, so anything that needs a backward pass
            # goes through the differentiable torch expression instead.
            and not (x.requires_grad and torch.is_grad_enabled())
        ):
            return _EXT.quick_gelu(x)
        return x * torch.sigmoid(1.702 * x)


# --------------------------------------------------------------------------------------------
# Optional fallback tracing.
#
# Answers "does the scored workload ever leave the fused path?" without costing anything when the
# answer is not being asked. Setting QUICKGELU_TRACE_FALLBACK swaps in an instrumented forward once,
# here at import; the method above is left untouched, so the normal hot path has no counter, no flag
# check and no extra branch.
# --------------------------------------------------------------------------------------------

_FALLBACK_REASON_ORDER = (
    "extension-unavailable",
    "not-cuda",
    "unsupported-dtype",
    "non-contiguous",
    "empty",
    "autograd",
)
_fallback_seen: dict[str, int] = {}


def _fallback_reason(x: torch.Tensor) -> str | None:
    """Why this tensor cannot take the fused path, or None if it can."""
    if _EXT is None:
        return "extension-unavailable"
    if not x.is_cuda:
        return "not-cuda"
    if x.dtype not in _SUPPORTED_DTYPES:
        return "unsupported-dtype"
    if not x.is_contiguous():
        return "non-contiguous"
    if x.numel() == 0:
        return "empty"
    if x.requires_grad and torch.is_grad_enabled():
        return "autograd"
    return None


def _forward_traced(self, x: torch.Tensor) -> torch.Tensor:
    reason = _fallback_reason(x)
    if reason is None:
        return _EXT.quick_gelu(x)
    count = _fallback_seen.get(reason, 0) + 1
    _fallback_seen[reason] = count
    if count == 1:
        # Once per reason, not once per call: the point is to notice a route being taken at all.
        _log_once(f"fallback taken: {reason}")
    return x * torch.sigmoid(1.702 * x)


def fallback_counts() -> dict[str, int]:
    """Fallback tally, empty when tracing is off or no fallback was taken."""
    return dict(_fallback_seen)


if os.environ.get("QUICKGELU_TRACE_FALLBACK"):
    QuickGELU.forward = _forward_traced
