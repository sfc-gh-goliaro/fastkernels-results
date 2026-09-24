"""MaxPool2d with a separable same-shape 5x5 CUDA kernel for the captured shape.

The captured workload is a *same-shape* pool: ``kernel_size=5, stride=1, padding=2``
over ``float16[N,128,20,20]``, so the output has the input's spatial extent. Three
properties of that configuration drive the kernel:

* Every output window contains its own centre pixel, so no window is entirely padding.
  Clamping a window's bounds to the input is therefore equivalent to the reference's
  behaviour of skipping padding cells, and no ``-inf`` sentinel is needed -- ``max`` is
  idempotent, so duplicating an edge row or column cannot change the result. All
  boundary branching disappears.
* A 5x5 max is separable into a 5-tap max along the row followed by a 5-tap max down
  the column: 8 comparisons per output instead of 24. The reference kernel is
  instruction-issue and latency bound (it also always materialises an int64 argmax
  tensor it is not asked for), so cutting instructions is the relevant lever.
* One plane is 20x20 = 400 halves. With ``W <= 32`` a plane's columns fit one per lane,
  so the row pass is a cross-lane shuffle reduction and the column pass lives entirely
  in per-lane registers -- no shared memory and no barrier.

At this size the operator is pure latency, not bandwidth: the reference kernel profiles
at 0.37% of peak DRAM throughput against 38.8% compute throughput. So the thing to
minimise is the length of each warp's dependency chain, not the number of bytes it
reads. Each warp therefore takes a band of ``kBandRows`` output rows plus the halo rows
those outputs need, and reads that whole band up front with compile-time indices: every
row load is independent and they all issue together, so a warp pays one memory latency
instead of one per row. That costs ``(kBandRows + 2 * radius) / kBandRows`` row reads per
output row -- 2x here -- and is worth it by a wide margin; see
``profile/candidate_v2_mapping_variants/REPORT.md`` for the measured comparison against
a streaming one-load-per-row formulation, a shared-memory-staged variant, and a
thread-per-output control.

Anything outside that configuration -- another kernel size, stride or padding,
``ceil_mode``, a non-fp16 or non-CUDA or non-contiguous or 3-D input, ``W > 32``, an
empty tensor, or an input that needs a gradient -- routes to ``F.max_pool2d``. Exact
*numerical* equality with the reference is the guarantee (``max`` over a fixed set of
float16 values is order-independent); bit-pattern identity is not, and is not claimed:
the reference keeps the first of tied signed zeros in raster order and preserves NaN
payloads, whereas ``__hmax_nan`` prefers ``+0.0`` order-independently and canonicalises
NaN. Both propagate NaN, which is why ``__hmax_nan`` and not ``__hmax`` is used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_EXTENSION_NAME = "fk_l1_max_pool2d_same5x5"

# Output rows per warp, and warps per block. Both chosen by measurement: 4 rows per warp
# was the fastest of {4, 5, 7, 10, 20} and 20 (a whole plane per warp) is also the only
# one that spills.
_BAND_ROWS = 4
_WARPS_PER_BLOCK = 4

_CPP_SOURCE = """
#include <torch/extension.h>

// The generated binding translation unit does not see the .cu sources, so the entry
// point has to be declared here for it to compile.
at::Tensor max_pool2d_same5x5(const at::Tensor& x);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <limits>

namespace {

constexpr int kLanes = 32;
constexpr int kKernelSize = 5;
constexpr int kBandRows = __BAND_ROWS__;
constexpr int kWarpsPerBlock = __WARPS_PER_BLOCK__;

// The block is kLanes * kWarpsPerBlock threads, so it is a multiple of the warp size by
// construction -- which is what makes the early exit below warp-uniform and the
// full-mask shuffles safe. These pin the surrounding assumptions at compile time.
static_assert(kLanes == 32, "the column-to-lane mapping assumes a 32-lane warp");
static_assert(kBandRows >= 1, "each warp must own at least one output row");
static_assert(kWarpsPerBlock >= 1 && kLanes * kWarpsPerBlock <= 1024,
              "block size must be a positive multiple of the warp size within limits");
static_assert(kKernelSize % 2 == 1, "an even kernel size has no centre pixel to clamp to");

// Horizontal (2*RADIUS+1)-tap max of one row, given the value this lane holds for it.
//
// Lane `l` owns output column `col = min(l, W - 1)`: lanes at or past W shadow the last
// column, which keeps them reading a valid address and, more importantly, keeps them
// participating in every shuffle so the full-warp mask stays honest. Taps falling
// outside [0, W) clamp to the edge column, which is a no-op for max. Because lane `j`
// holds column `j` for every j < W, the clamped column index doubles as the source lane
// index.
template <int RADIUS>
__device__ __forceinline__ __half row_max(__half v, int col, int W) {
  __half m = v;
#pragma unroll
  for (int d = 1; d <= RADIUS; ++d) {
    const int right = (col + d < W) ? (col + d) : (W - 1);
    const int left = (col - d > 0) ? (col - d) : 0;
    m = __hmax_nan(m, __shfl_sync(0xffffffffu, v, right));
    m = __hmax_nan(m, __shfl_sync(0xffffffffu, v, left));
  }
  return m;
}

// One warp per (plane, band of BAND output rows), one lane per column.
//
// The band's outputs span input rows `row0 - RADIUS .. row0 + BAND - 1 + RADIUS`, which
// is BAND + 2*RADIUS rows. All of them are read first, with loop indices the compiler
// resolves at compile time, so the loads carry no mutual dependency and a warp waits on
// memory once rather than once per row. The horizontal taps are then independent of each
// other too, and each output row is its own KSIZE-deep max tree over the horizontal
// maxima -- every array subscript here is a compile-time constant, which is what keeps
// the two register arrays in registers.
template <int KSIZE, int BAND, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(kLanes* WARPS_PER_BLOCK) void same_shape_max_pool(
    const __half* __restrict__ input, __half* __restrict__ output, int tasks, int bands,
    int H, int W) {
  constexpr int RADIUS = KSIZE / 2;
  constexpr int ROWS = BAND + 2 * RADIUS;

  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int task = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  // Warp-uniform: `warp` is derived from threadIdx.x / 32 and blockDim.x is a multiple
  // of the warp size, so an entire warp leaves together and every surviving warp has
  // all 32 lanes active for the shuffles below.
  if (task >= tasks) return;

  const int plane = task / bands;
  const int row0 = (task - plane * bands) * BAND;

  const long long base = static_cast<long long>(plane) * H * W;
  const __half* __restrict__ src = input + base;
  __half* __restrict__ dst = output + base;

  const int col = (lane < W) ? lane : (W - 1);

  // Halo rows above row 0 and below row H-1 clamp onto the edge row. Clamping rather
  // than substituting a sentinel is what makes this correct: every output window holds
  // its own centre pixel, so duplicating an edge row cannot change any maximum.
  __half v[ROWS];
#pragma unroll
  for (int t = 0; t < ROWS; ++t) {
    int r = row0 - RADIUS + t;
    r = (r < 0) ? 0 : ((r > H - 1) ? (H - 1) : r);
    v[t] = src[static_cast<long long>(r) * W + col];
  }

  __half rows[ROWS];
#pragma unroll
  for (int t = 0; t < ROWS; ++t) {
    rows[t] = row_max<RADIUS>(v[t], col, W);
  }

  // Every shuffle is done by this point, so the predicated stores below cannot leave a
  // shuffle with an incomplete warp.
#pragma unroll
  for (int b = 0; b < BAND; ++b) {
    const int i = row0 + b;
    if (i < H && lane < W) {
      __half m = rows[b];
#pragma unroll
      for (int d = 1; d < KSIZE; ++d) {
        m = __hmax_nan(m, rows[b + d]);
      }
      dst[static_cast<long long>(i) * W + col] = m;
    }
  }
}

at::Tensor reference_pool(const at::Tensor& x) {
  return at::max_pool2d(x, {kKernelSize, kKernelSize}, {1, 1},
                        {kKernelSize / 2, kKernelSize / 2});
}

}  // namespace

at::Tensor max_pool2d_same5x5(const at::Tensor& x) {
  // Cheap, tensor-dependent re-checks of everything the kernel assumes. The host-side
  // configuration predicate has already established kernel_size/stride/padding/
  // ceil_mode; these are the properties only the tensor can answer for.
  //
  // is_neg() and is_conj() matter because the kernel reads raw storage through a typed
  // pointer: a lazily negated view (torch._neg_view) is contiguous fp16 CUDA 4-D and
  // passes every other check, but its logical values are the negation of what is in
  // memory, so reading the pointer would silently pool the wrong numbers.
  //
  // The gradient guards are two separate mechanisms. Reverse mode is caught by
  // requires_grad under an enabled GradMode. Forward mode is not: a dual tensor from
  // torch.autograd.forward_ad.make_dual has requires_grad false, and reading the primal
  // through a pointer would drop its tangent without any error.
  const bool eligible =
      x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 4 &&
      x.is_contiguous() && x.numel() > 0 && x.size(3) <= kLanes && !x.is_neg() &&
      !x.is_conj() && !(at::GradMode::is_enabled() && x.requires_grad()) &&
      !x._fw_grad(/*level=*/0).defined();
  if (!eligible) {
    return reference_pool(x);
  }

  const long long H = x.size(2);
  const long long W = x.size(3);
  const long long planes = x.size(0) * x.size(1);
  // H is narrowed to int for the kernel, and the kernel forms row0 - RADIUS + t and
  // row0 + b from it, so it needs headroom above H for those to stay in range. A
  // [1,1,2^31,1] fp16 tensor is only 4 GB and therefore genuinely allocatable here, so
  // this bound is reachable rather than theoretical.
  if (H > std::numeric_limits<int>::max() - 2 * kKernelSize) {
    return reference_pool(x);
  }
  const long long bands = (H + kBandRows - 1) / kBandRows;
  const long long tasks = planes * bands;
  if (tasks > std::numeric_limits<int>::max()) {
    return reference_pool(x);
  }

  const c10::cuda::CUDAGuard guard(x.device());
  // Fresh output every call, with canonical contiguous strides. at::empty_like would
  // instead carry over the input's stride metadata, which for a contiguous-degenerate
  // input (a singleton dimension whose stride is arbitrary) differs from what the
  // reference operator returns.
  at::Tensor out = at::empty(x.sizes(), x.options());

  const int blocks = static_cast<int>((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
  same_shape_max_pool<kKernelSize, kBandRows, kWarpsPerBlock>
      <<<blocks, kLanes * kWarpsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>()),
          reinterpret_cast<__half*>(out.mutable_data_ptr<at::Half>()),
          static_cast<int>(tasks), static_cast<int>(bands), static_cast<int>(H),
          static_cast<int>(W));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


def _as_int(value) -> int:
    """Accept only what the reference operator itself accepts as an extent.

    Deliberately strict: ``int(5.0)`` would be 5, but ``F.max_pool2d(x, 5.0)`` raises, so
    coercing here would make the fast path succeed on an input the reference rejects.
    Anything unrecognised raises and disables the fast path, which routes the call to the
    reference and reproduces its error exactly.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value).__name__}")
    return value


def _as_pair(value) -> tuple[int, int]:
    """Normalise an int / 1- or 2-element sequence spec to an ``(h, w)`` pair."""
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            h = w = value[0]
        elif len(value) == 2:
            h, w = value
        else:
            raise ValueError(f"expected 1 or 2 values, got {len(value)}")
    else:
        h = w = value
    return _as_int(h), _as_int(w)


def _workspace_root() -> Path:
    """Directory to anchor build products in.

    The harness imports this file in place, so ``__file__`` is the real path inside the
    operator workspace. Keeping build products here rather than in the shared
    ``~/.cache/torch_extensions`` is what stops the concurrently running sibling
    operator workspaces from contending over one build tree.
    """
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "validate.py").is_file():
            return parent
    return here


def _target_arch() -> str:
    """The single architecture to compile for, so a cold build is not multiplied by the
    six-architecture ambient list."""
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}.{minor}"
    except Exception:
        pass
    return "10.0"


def _build_extension():
    from torch.utils.cpp_extension import load_inline

    build_dir = _workspace_root() / ".torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _target_arch()
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE.replace(
                "__BAND_ROWS__", str(_BAND_ROWS)
            ).replace("__WARPS_PER_BLOCK__", str(_WARPS_PER_BLOCK)),
            functions=["max_pool2d_same5x5"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


#: Set when the extension compiled and loaded; when False the module still works, it
#: just runs ``F.max_pool2d`` everywhere.
FAST_PATH_AVAILABLE = False
#: Populated with the build failure when ``FAST_PATH_AVAILABLE`` is False.
BUILD_ERROR: str | None = None

_extension = None
try:
    _extension = _build_extension()
    FAST_PATH_AVAILABLE = True
except Exception as exc:  # a build failure must not make this module unimportable
    BUILD_ERROR = f"{type(exc).__name__}: {exc}"
    print(
        f"[{_EXTENSION_NAME}] CUDA extension unavailable, falling back to "
        f"F.max_pool2d for every input: {BUILD_ERROR}",
        file=sys.stderr,
        flush=True,
    )


#: Attributes the fast-path predicate depends on. Assigning any of them re-evaluates it.
_CONFIG_ATTRS = frozenset({"kernel_size", "stride", "padding", "ceil_mode"})


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self._use_fast_path = self._eligible_configuration()
        self._configured = True

    def _eligible_configuration(self) -> bool:
        """Whether this configuration is exactly the one the kernel implements.

        Narrow on purpose. Widening it to "any odd kernel with padding = kernel // 2"
        would send kernel_size=1 -- whose stride defaults to 1 and whose padding
        1 // 2 is 0 -- to the 5x5 kernel. ``self.stride`` is read rather than the
        constructor argument, so ``stride=None`` has already resolved to
        ``kernel_size`` the way the reference resolves it. ``ceil_mode`` must be
        literally False, not merely falsy: the reference rejects ``ceil_mode=0`` with a
        TypeError, so treating 0 as False would make this succeed where it raises. No
        tensor is touched and no CUDA call is made.
        """
        try:
            return (
                FAST_PATH_AVAILABLE
                and self.ceil_mode is False
                and _as_pair(self.kernel_size) == (5, 5)
                and _as_pair(self.stride) == (1, 1)
                and _as_pair(self.padding) == (2, 2)
            )
        except (TypeError, ValueError):
            return False

    def __setattr__(self, name: str, value) -> None:
        # The reference reads its configuration attributes on every call, so reassigning
        # one changes its behaviour. Re-evaluating here keeps a cached predicate honest
        # without putting the check on the per-call path.
        super().__setattr__(name, value)
        if name in _CONFIG_ATTRS and getattr(self, "_configured", False):
            super().__setattr__("_use_fast_path", self._eligible_configuration())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fast_path:
            # The extension re-checks the tensor-dependent conditions and falls back to
            # the reference operator itself if any of them does not hold.
            return _extension.max_pool2d_same5x5(x)
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
        )
