"""Primitive tensor manipulation ops: fused trailing-zero-append ``Pad``.

The captured ``Pad`` workload is a *trailing append of zeros*: ``value`` is zero, a single
dimension is padded on its right only, and every dimension ahead of it has extent 1. In flat
memory the result is therefore a contiguous copy of ``x`` followed by a zero tail, which one
kernel can write in a single pass.

``F.pad`` dispatches to ``at::constant_pad_nd``, which spends two GPU commands on the same
thing -- an ``empty`` + ``fill_`` elementwise kernel plus a device-to-device ``copy_`` of the
head. Every captured output is at most 98 KB, so the operation is bound by per-command latency
rather than bandwidth; halving the command count is the whole optimization, and the fused
kernel also moves fewer bytes than the fill-plus-copy pair.

Anything the fused path declines is handed straight back to ``F.pad``, so behaviour outside the
captured regime is unchanged by construction rather than by re-implementation -- including the
exception ``F.pad`` raises for a malformed ``pad`` *value* (odd length, longer than ``2 * ndim``,
cropping past an extent), which reaches the predicate, is declined, and is then raised by ``F.pad``
itself. The one gap: an ill-*typed* ``pad`` (a tensor, a list of floats) is rejected by pybind's
argument conversion before the predicate runs, so the exception type still matches ``F.pad``'s
``TypeError`` but the message differs. That covers observability too: an
active ``TorchFunctionMode`` or ``TorchDispatchMode`` would see ``F.pad`` issue
``aten.constant_pad_nd`` and would see the fused path issue only ``aten.empty``, so a mode stack
of any kind sends the call down the generic path -- as does Dynamo tracing, which cannot handle
the entry point's optional return.

If the CUDA extension cannot be built the module degrades to ``F.pad``: still correct, but at
baseline speed, and it says so loudly on stderr so a silent regression to baseline is not
mistaken for a passing optimization.

Recovery note: the build directory is workspace-owned (``.torch_extensions/``). If a build is
killed mid-compile, ``torch.utils.file_baton.FileBaton.wait()`` spins forever on the leftover
``lock`` file; delete ``<build_dir>/lock`` to recover.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Architecture used when no GPU is visible at build time (B200 / sm_100 is the target device).
# Pinning to a single architecture keeps the build ~28 s cheaper than the ambient six-target
# list and lets the cache be pre-warmed without leasing a GPU.
_DEFAULT_CUDA_ARCH = "10.0"

_SOURCE_CPP = r"""
#include <torch/extension.h>

#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <vector>

std::optional<at::Tensor> pad_trailing_zero(const at::Tensor& x, std::vector<int64_t> pad,
                                            double value);
int64_t fused_calls();
int64_t declined_calls();
std::vector<int64_t> access_width_calls();
void reset_counters();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pad", &pad_trailing_zero,
        "Append a zero tail in one fused pass; None if the input needs generic padding (CUDA)",
        py::arg("x"), py::arg("pad"), py::arg("value"));
  m.def("fused_calls", &fused_calls,
        "Calls served by the fused kernel (CUDA)");
  m.def("declined_calls", &declined_calls,
        "Calls declined by the fused path and handed back to the caller (CUDA)");
  m.def("access_width_calls", &access_width_calls,
        "Fused calls per access width, ordered 16/8/4/2/1 bytes (CUDA)");
  m.def("reset_counters", &reset_counters,
        "Zero the dispatch counters (CUDA)");
}
"""

_SOURCE_CUDA = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace {

// Widths are recorded in this order so the host can assert which instantiation ran.
constexpr int kWidths[] = {16, 8, 4, 2, 1};
constexpr int kNumWidths = 5;

std::atomic<int64_t> g_fused_calls{0};
std::atomic<int64_t> g_declined_calls{0};
std::atomic<int64_t> g_width_calls[kNumWidths] = {};

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 65535;

template <int BYTES> struct BytesToType {};
template <> struct BytesToType<16> { using Type = uint4;    static_assert(sizeof(Type) == 16); };
template <> struct BytesToType<8>  { using Type = uint64_t; static_assert(sizeof(Type) == 8); };
template <> struct BytesToType<4>  { using Type = uint32_t; static_assert(sizeof(Type) == 4); };
template <> struct BytesToType<2>  { using Type = uint16_t; static_assert(sizeof(Type) == 2); };
template <> struct BytesToType<1>  { using Type = uint8_t;  static_assert(sizeof(Type) == 1); };

// One predicated load and one unconditional store, so tail lanes issue no load at all.
// The grid is clamped, so the loop is what keeps an arbitrarily large input correct; for
// every captured case it runs a single iteration.
template <typename V>
__global__ void pad_trailing_zero_kernel(const V* __restrict__ src, V* __restrict__ dst,
                                         int64_t n_head, int64_t n_total) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_total; i += stride) {
    V v{};                       // all-zero bytes read back as zero in every accepted dtype
    if (i < n_head) v = src[i];
    dst[i] = v;
  }
}

template <int BYTES>
void launch_width(const at::Tensor& x, at::Tensor& out, int64_t head_bytes,
                  int64_t total_bytes) {
  using V = typename BytesToType<BYTES>::Type;
  const int64_t n_head = head_bytes / BYTES;
  const int64_t n_total = total_bytes / BYTES;
  const int64_t blocks = std::min<int64_t>((n_total + kThreads - 1) / kThreads, kMaxBlocks);
  pad_trailing_zero_kernel<V>
      <<<static_cast<unsigned int>(blocks), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<const V*>(x.data_ptr()), static_cast<V*>(out.data_ptr()),
          n_head, n_total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Only dtypes whose zero really is all-zero bytes AND which the generic path accepts a 0.0
// fill value for. An allowlist rather than a denylist because a byte-level kernel is only
// safe for representations it has been checked against: float8_e8m0fnu, for instance, has no
// representation for zero at all, so the generic path *raises* where zeroed bytes would
// silently succeed. Unlisted dtypes (float8 variants, complex, sub-byte, bits*) fall back.
bool is_zero_fillable_dtype(at::ScalarType t) {
  switch (t) {
    case at::kBFloat16:
    case at::kHalf:
    case at::kFloat:
    case at::kDouble:
    case at::kLong:
    case at::kInt:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kBool:
      return true;
    default:
      return false;
  }
}

// True only when `pad` describes a flat trailing append of zeros onto `x`, i.e. when the
// output is bit-for-bit a contiguous copy of `x` followed by zero bytes. Everything else --
// including anything malformed -- is declined so the caller can use the generic path.
bool is_flat_trailing_zero_append(const at::Tensor& x, const std::vector<int64_t>& pad,
                                  double value, int64_t* pad_dim) {
  const int64_t ndim = x.dim();
  const int64_t np = static_cast<int64_t>(pad.size());

  // Checked first: without a storage, querying strides or the data pointer throws. This is
  // what a functorch batched tensor (inside vmap), a meta tensor and a fake tensor look like.
  if (!x.unsafeGetTensorImpl()->has_storage()) return false;

  // Layouts and wrapper tensors whose data_ptr is not a plain dense buffer.
  if (!x.is_cuda() || x.layout() != at::kStrided || x.is_quantized() || x.is_nested()) {
    return false;
  }
  if (!is_zero_fillable_dtype(x.scalar_type())) return false;
  // The generic path refuses named tensors outright, so accepting them would turn an error
  // into a silent success.
  if (x.has_names()) return false;
  // Conjugate/negative bits and zero-tensors defer materialization, so a raw byte copy
  // would silently drop the pending operation.
  if (x.is_conj() || x.is_neg() || x._is_zerotensor()) return false;
  if (x.numel() == 0) return false;

  // Zeroing bytes writes *positive* zero, while F.pad(value=-0.0) really does write the
  // negative-zero bit pattern -- and torch.equal(+0.0, -0.0) is True, so nothing downstream
  // would catch the difference. NaN also lands here, since NaN != 0.0.
  if (value != 0.0 || std::signbit(value)) return false;

  // Under enabled grad mode the generic path builds an autograd graph; the fused kernel
  // cannot, so it must not claim such an input.
  if (at::GradMode::is_enabled() && x.requires_grad()) return false;
  // Forward-mode AD carries a tangent alongside the primal that the generic path propagates
  // into its output. A raw byte copy would silently drop it, so dual tensors fall back.
  if (x._fw_grad(/*level=*/0).defined()) return false;

  // Exactly the standard contiguous strides, which is stricter than is_contiguous(): that
  // ignores size-1 dimensions, so a channels_last tensor whose unit extents make it
  // ambiguously contiguous would pass -- and the generic path would then propagate its
  // channels_last stride pattern to the output, which a flat append cannot reproduce.
  {
    int64_t expected = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
      if (x.stride(i) != expected) return false;
      expected *= x.size(i);
    }
  }

  // `pad` runs last-dimension-first in (left, right) pairs, so pad.back() is the right pad
  // of dimension ndim - np/2. Reject anything the generic path would reject as well.
  if (np % 2 != 0 || np < 2 || np > 2 * ndim) return false;
  if (pad.back() <= 0) return false;
  for (int64_t i = 0; i + 1 < np; ++i) {
    if (pad[i] != 0) return false;   // a left pad, or any interior dimension padded
  }

  const int64_t d = ndim - np / 2;
  // Checked add: the padded extent is computed as x.size(d) + pad.back(), and signed overflow
  // there is undefined behaviour that would wrap to a negative size. Decline before adding and
  // let the generic path produce its own diagnostic.
  if (pad.back() > std::numeric_limits<int64_t>::max() - x.size(d)) return false;
  // With unit extents ahead of the padded dimension the flat index of an element is
  // i_d * T + rest in both tensors, so the pad is an append. Any larger leading extent
  // makes it a strided insert instead, which this kernel does not implement.
  for (int64_t i = 0; i < d; ++i) {
    if (x.size(i) != 1) return false;
  }
  *pad_dim = d;
  return true;
}

int width_slot(int width) {
  for (int i = 0; i < kNumWidths; ++i) {
    if (kWidths[i] == width) return i;
  }
  return kNumWidths - 1;
}

}  // namespace

std::optional<at::Tensor> pad_trailing_zero(const at::Tensor& x, std::vector<int64_t> pad,
                                            double value) {
  int64_t pad_dim = 0;
  if (!is_flat_trailing_zero_append(x, pad, value, &pad_dim)) {
    g_declined_calls.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }

  const c10::cuda::CUDAGuard device_guard(x.device());

  auto sizes = x.sizes().vec();
  sizes[pad_dim] += pad.back();
  at::Tensor out = at::empty(sizes, x.options());

  const int64_t head_bytes = x.numel() * x.element_size();
  const int64_t total_bytes = out.numel() * out.element_size();

  // Widest access whose size divides both byte counts and both base addresses. Dividing the
  // head byte count is what keeps the head/tail boundary on an element boundary, so no single
  // access ever straddles it.
  uint64_t g = std::gcd(static_cast<uint64_t>(head_bytes), static_cast<uint64_t>(total_bytes));
  g = std::gcd(g, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x.data_ptr())));
  g = std::gcd(g, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(out.data_ptr())));

  int width = 1;
  for (int i = 0; i < kNumWidths; ++i) {
    if (g % static_cast<uint64_t>(kWidths[i]) == 0) { width = kWidths[i]; break; }
  }

  switch (width) {
    case 16: launch_width<16>(x, out, head_bytes, total_bytes); break;
    case 8:  launch_width<8>(x, out, head_bytes, total_bytes);  break;
    case 4:  launch_width<4>(x, out, head_bytes, total_bytes);  break;
    case 2:  launch_width<2>(x, out, head_bytes, total_bytes);  break;
    default: launch_width<1>(x, out, head_bytes, total_bytes);  break;
  }

  g_width_calls[width_slot(width)].fetch_add(1, std::memory_order_relaxed);
  g_fused_calls.fetch_add(1, std::memory_order_relaxed);
  return out;
}

int64_t fused_calls() { return g_fused_calls.load(std::memory_order_relaxed); }

int64_t declined_calls() { return g_declined_calls.load(std::memory_order_relaxed); }

std::vector<int64_t> access_width_calls() {
  std::vector<int64_t> counts(kNumWidths);
  for (int i = 0; i < kNumWidths; ++i) {
    counts[i] = g_width_calls[i].load(std::memory_order_relaxed);
  }
  return counts;
}

void reset_counters() {
  g_fused_calls.store(0, std::memory_order_relaxed);
  g_declined_calls.store(0, std::memory_order_relaxed);
  for (int i = 0; i < kNumWidths; ++i) {
    g_width_calls[i].store(0, std::memory_order_relaxed);
  }
}
"""

#: Access widths in the order ``_C.access_width_calls()`` reports them.
ACCESS_WIDTHS = (16, 8, 4, 2, 1)

# Probes for anything that observes, traces or rewrites the call. Two are private torch APIs, so
# all three are resolved once here rather than assumed: if a future torch drops them there is no
# way to tell whether the call is being interposed, and the only safe answer is to stop using the
# fused path entirely.
_IS_FUNCTION_MODE_ENABLED = getattr(torch._C, "_is_torch_function_mode_enabled", None)
_DISPATCH_STACK_LEN = getattr(torch._C, "_len_torch_dispatch_stack", None)
_IS_COMPILING = getattr(torch.compiler, "is_compiling", None)
_MODE_PROBES_AVAILABLE = (callable(_IS_FUNCTION_MODE_ENABLED)
                          and callable(_DISPATCH_STACK_LEN)
                          and callable(_IS_COMPILING))


def _mode_stack_active() -> bool:
    """True when the call is being observed, traced or rewritten -- or when we cannot tell.

    Three distinct reasons the extension must not be entered:

    * an active ``TorchFunctionMode`` -- ``F.pad`` invokes its hook, the fused path would not;
    * an active ``TorchDispatchMode`` -- it sees ``F.pad`` issue ``aten.constant_pad_nd`` but
      would see the fused path issue only ``aten.empty``, so a rewriting mode changes the answer;
    * Dynamo tracing -- the pybind entry point returns ``Optional[Tensor]``, which Dynamo
      rejects outright ("torch.* op returned non-Tensor"), so calling it would make the candidate
      fail to compile on a model where the baseline compiles cleanly. The tracing probe is
      checked first for the reason given inline below.

    All three probes are plain integer/boolean reads, ~130 ns together against an ~11 us call.
    """
    if not _MODE_PROBES_AVAILABLE:
        return True
    # `is_compiling()` MUST be tested first. Dynamo rejects
    # `torch._C._len_torch_dispatch_stack()` itself with "torch.* op returned non-Tensor", so
    # evaluating it during tracing would break compilation from inside this very guard.
    # Short-circuiting on `is_compiling()` keeps the traced graph clean.
    return (_IS_COMPILING()
            or _IS_FUNCTION_MODE_ENABLED()
            or _DISPATCH_STACK_LEN() != 0)


def _pinned_arch() -> str:
    """Single build architecture, mirroring ``infra.cuda_ext._pin_build_arch``'s override."""
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST", "").strip()
    if override:
        return override
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}.{minor}"
    except Exception:
        pass
    return _DEFAULT_CUDA_ARCH


def _extension_name(arch: str) -> str:
    """Content-addressed name: stable across processes, distinct after any source edit.

    ``sha256`` rather than the builtin ``hash()``, whose string hashing is salted per process
    and would give every run a different name (and so a rebuild plus a fresh cache entry).
    """
    digest = hashlib.sha256()
    for part in (_SOURCE_CPP, _SOURCE_CUDA, arch, torch.__version__, str(torch.version.cuda)):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"fk_pad_trailing_zero_{digest.hexdigest()[:16]}"


def _build_root() -> Path:
    """A build directory this workspace owns, so a stale lock file is visible and removable.

    Workspace-only by design: no `$HOME` or `/tmp` fallback, both because the workspace rules
    say to work only here and because a fallback would silently scatter build products where a
    stale `FileBaton` lock could not be found. If this directory cannot be created the caller's
    `try/except` turns it into the loud degradation to `F.pad`.
    """
    base = Path(__file__).resolve().parents[2] / ".torch_extensions"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _load_extension():
    arch = _pinned_arch()
    name = _extension_name(arch)
    build_dir = _build_root() / name
    build_dir.mkdir(parents=True, exist_ok=True)

    # A cold build streams ninja progress. The bench worker redirects fd 1 onto fd 2, so that
    # progress lands in the log whose mtime the stall watchdog reads, and a compile cannot be
    # mistaken for a hang. A warm import prints nothing and compiles nothing.
    cold = not (build_dir / f"{name}.so").exists()
    if cold:
        print(f"[tensor_ops.Pad] building {name} for sm {arch} in {build_dir}",
              file=sys.stderr, flush=True)

    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=name,
            cpp_sources=_SOURCE_CPP,
            cuda_sources=_SOURCE_CUDA,
            functions=None,          # the module block is hand-written in _SOURCE_CPP
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            verbose=cold,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


try:
    _C = _load_extension()
except Exception as exc:  # toolchain problems must not turn into a wrong answer
    _C = None
    print(f"[tensor_ops.Pad] CUDA extension unavailable ({type(exc).__name__}: {exc}); "
          f"falling back to F.pad -- correct but NOT optimized",
          file=sys.stderr, flush=True)

if _C is not None and not _MODE_PROBES_AVAILABLE:
    print("[tensor_ops.Pad] torch interposition probes unavailable, so an active mode stack or "
          "Dynamo tracing cannot be detected; falling back to F.pad on every call -- correct "
          "but NOT optimized", file=sys.stderr, flush=True)


class Pad(nn.Module):
    """Functional padding op; a flat trailing append of zeros runs as one fused kernel."""

    def forward(
        self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0,
    ) -> torch.Tensor:
        # Three conditions must hold before the extension may be called, because each one is a
        # way for an observer to tell the fused kernel apart from `F.pad`:
        #   * `type(x) is torch.Tensor` -- a subclass must keep its __torch_function__ and get a
        #     same-subclass result back, not be silently unwrapped to a plain tensor.
        #   * nothing is interposing -- see `_mode_stack_active`: an observing or rewriting mode
        #     stack, or Dynamo tracing, must all see exactly what the baseline does.
        # Cost is ~130 ns against an ~11 us call, and every probe is a plain integer read.
        if (_C is not None and type(x) is torch.Tensor
                and not _mode_stack_active()):
            out = _C.pad(x, pad, 0.0 if value is None else value)
            if out is not None:
                return out
        return F.pad(x, pad, value=value)
