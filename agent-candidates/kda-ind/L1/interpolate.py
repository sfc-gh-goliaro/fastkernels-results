"""Interpolate wrapping F.interpolate, with a custom nearest-2x upsample kernel.

`F.interpolate(x, scale_factor=2, mode="nearest")` on a contiguous NCHW half-precision
tensor is a pure bit-level copy: every output element is a copy of some input element,
with no arithmetic at all.  ATen serves it with one thread per *output* element, each
moving 2 bytes and recomputing the nearest-source index for both spatial axes.  This
module keeps that expression as the fallback and routes exactly that one argument
combination to a kernel that moves the same bytes with vectorized accesses instead.

Index algebra.  Write the contiguous input as ``in[rows][width]`` with
``rows = N * C * H``, and the output as ``out[2 * rows][2 * width]``.  Output row
``p * (2H) + 2h`` equals ``2 * (p * H + h) = 2r``, so the plane index cancels: in flat
row order the output is every input row with each element duplicated, emitted twice.
Only ``rows`` and ``width`` reach the kernel — no division by ``H``.

Every argument combination the kernel is not proven to reproduce exactly — other modes,
other scale factors, the ``size=`` spelling, non-``None`` ``align_corners``, other
dtypes, non-contiguous or non-4-D or CPU inputs, empty tensors, and row widths or
pointer alignments the kernel cannot serve — reaches the unmodified baseline
expression, so this module is a strict behavioral superset of the baseline: bit-identical
results, or the same exception with the same message.

Some of those conditions are properties of the *execution context* rather than of the
arguments, because an extension call is opaque to machinery that needs to look inside it:

* while either tracer is running the baseline expression is used instead, so
  ``torch.compile(..., fullgraph=True)`` captures one graph rather than failing on a call
  it cannot trace, and ``torch.jit.trace`` records the real operator rather than a
  constant. Dynamo is detected in Python because it never reaches the extension; the
  TorchScript tracer is detected inside the extension, which refuses while a trace is in
  progress;
* a tensor carrying a forward-mode tangent is refused by the extension itself, which asks
  the tensor rather than asking whether a dual level is open, so it holds however the
  level was entered;
* a tensor the dispatcher needs to see through -- batched under ``vmap``, functionalized,
  fake or meta, sparse, or any tensor observed while a ``TorchDispatchMode`` is active --
  is refused before anything is allocated or any pointer is taken. Such a tensor is still
  a plain ``torch.Tensor`` in Python, so only the dispatcher can answer this;
* reverse-mode autograd follows the same principle: an input that would produce a
  differentiable result falls back rather than losing its history.

One residual difference, which is not a divergence in behavior but in wording:
``torch.jit.script`` rejects this module and the baseline alike -- neither is scriptable --
with different error messages.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Only these two of the 2-byte dtypes are accepted by F.interpolate itself; int16,
# uint16 and the 8-bit float types raise NotImplementedError there.  The gate is an
# explicit allowlist rather than an element-size test so that the fast path can never
# succeed where the baseline expression would raise.
_FAST_DTYPES = (torch.float16, torch.bfloat16)

# Consulted on every call, so they are resolved once here rather than walked per call:
# the scored measurement on the smaller shapes is bound by per-call host cost.
_TENSOR = torch.Tensor
_is_grad_enabled = torch.is_grad_enabled
_is_compiling = torch.compiler.is_compiling

# Fallback call count, for tests that need to prove which route an argument set took.
# Only the fallback is counted: the scored measurement on the smaller shapes is bound by
# per-call host cost, and a counter increment there is the same order of magnitude as the
# difference between the candidate and the baseline. An eager call that leaves this
# unchanged took the fast path. Calls made under Dynamo leave it unchanged (they return
# before reaching it); calls made under the TorchScript tracer do increment it, since the
# extension is the side that refuses them.
_FALLBACK_CALLS = 0

_CPP_DECLARATIONS = r"""
#include <torch/extension.h>

// Returns a freshly allocated NCHW-contiguous 2x nearest upsample of `x`, or an
// undefined tensor (None in Python) when the row width or pointer alignment is
// outside what the kernel can serve, in which case the caller must fall back.
at::Tensor nearest2x(const at::Tensor& x);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace {

// A block covers one input row with blockDim.x groups of VEC elements, so the row must
// fit within the hardware limit on threads per block; the widest vector serves the
// widest row.
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kWidestVector = 4;

// Per-width access types.  A thread loads `VEC` elements (2 * VEC bytes) and stores
// their duplication (4 * VEC bytes) twice, once into each sibling output row.
template <int VEC> struct AccessTypes;
template <> struct AccessTypes<1> { using Load = uint16_t; using Store = uint32_t; };
template <> struct AccessTypes<2> { using Load = uint32_t; using Store = uint2; };
template <> struct AccessTypes<4> { using Load = uint2;    using Store = uint4; };

// Duplicate every 16-bit element of the loaded vector in place: a b c d -> a a b b c c d d.
// __byte_perm builds each result word from two copies of one 16-bit half of the source,
// one PRMT per output word.  Selector nibble i (counted from the least significant)
// picks the source byte for result byte i, so for a source holding little-endian halves
// [b0 b1][b2 b3], 0x1010 yields [b0 b1 b0 b1] and 0x3232 yields [b2 b3 b2 b3].  This is
// a pure bit permutation: exact for every payload, including NaN, signed zero and
// denormals, in both float16 and bfloat16.
template <int VEC>
__device__ __forceinline__ typename AccessTypes<VEC>::Store
duplicate(typename AccessTypes<VEC>::Load v);

template <>
__device__ __forceinline__ uint32_t duplicate<1>(uint16_t v) {
  const uint32_t h = v;
  return h | (h << 16);
}

template <>
__device__ __forceinline__ uint2 duplicate<2>(uint32_t v) {
  uint2 d;
  d.x = __byte_perm(v, 0, 0x1010);
  d.y = __byte_perm(v, 0, 0x3232);
  return d;
}

template <>
__device__ __forceinline__ uint4 duplicate<4>(uint2 v) {
  uint4 d;
  d.x = __byte_perm(v.x, 0, 0x1010);
  d.y = __byte_perm(v.x, 0, 0x3232);
  d.z = __byte_perm(v.y, 0, 0x1010);
  d.w = __byte_perm(v.y, 0, 0x3232);
  return d;
}

// in : [rows][width], out : [2 * rows][2 * width], elements are 2 bytes wide.
//
// blockDim.x is exactly width / VEC, so threadIdx.x is the group of `VEC` elements
// within an input row and threadIdx.y selects the row: no integer division, no masked
// lanes.  The input offset reduces to VEC * (block_base + threadIdx.x + threadIdx.y *
// blockDim.x), so consecutive linear thread ids read strictly consecutive groups and the
// load side is gap-free.  Offsets are 64-bit throughout: a large input has an output
// element count past the signed 32-bit range.
template <int VEC>
__global__ void nearest2x_kernel(const uint16_t* __restrict__ in,
                                 uint16_t* __restrict__ out,
                                 int64_t rows, int width) {
  using Load = typename AccessTypes<VEC>::Load;
  using Store = typename AccessTypes<VEC>::Store;

  const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (r >= rows) {
    return;
  }
  const int64_t group = threadIdx.x;

  const Load v = *reinterpret_cast<const Load*>(in + r * width + VEC * group);
  const Store d = duplicate<VEC>(v);

  // Output rows 2r and 2r + 1 start at flat elements 4 * r * width and
  // 4 * r * width + 2 * width.
  uint16_t* o = out + 4 * r * width + 2 * VEC * group;
  *reinterpret_cast<Store*>(o) = d;
  *reinterpret_cast<Store*>(o + 2 * width) = d;
}

// Widest vector width whose load, store and sibling-store offsets are all naturally
// aligned for these two base pointers and this row width.  Returns 0 when even the
// scalar width cannot be served.
int select_vector_width(uintptr_t in_ptr, uintptr_t out_ptr, int width) {
  if (width % 4 == 0 && in_ptr % 8 == 0 && out_ptr % 16 == 0) {
    return 4;
  }
  if (width % 2 == 0 && in_ptr % 4 == 0 && out_ptr % 8 == 0) {
    return 2;
  }
  if (out_ptr % 4 == 0) {
    return 1;
  }
  return 0;
}

// Rows per block for a block of `groups` threads in x.  Prefers a whole number of warps
// and a block near 256 threads, and never exceeds the 1024-thread limit — the clamp
// matters for wide rows, where `groups` alone can already be most of a block.
int rows_per_block(int groups) {
  int rows_y = 32 / std::gcd(groups, 32);
  if (groups * rows_y > kMaxThreadsPerBlock) {
    rows_y = std::max(1, kMaxThreadsPerBlock / groups);
  }
  while (groups * rows_y * 2 <= 256) {
    rows_y *= 2;
  }
  return rows_y;
}

}  // namespace

at::Tensor nearest2x(const at::Tensor& x) {
  // First question, before anything is allocated or any pointer is taken: is this a tensor
  // the dispatcher needs to see through?  A batched tensor from vmap, a functional or
  // gradient wrapper, a fake or meta tensor, a sparse layout, or any tensor observed while
  // a TorchDispatchMode is active is still a plain `torch.Tensor` in Python, but its data
  // may not exist at all -- and taking a raw pointer to a wrapped allocation launches a
  // kernel against memory that is not there.  `isTensorSubclassLike` is the predicate ATen
  // itself uses for this, and it reports true whenever a dispatch mode is enabled.
  if (at::isTensorSubclassLike(x) || !x.has_storage()) {
    return at::Tensor();
  }
  // Every layout this kernel cannot serve is answered with an undefined tensor (None in
  // Python) so the caller can fall back; these are capability answers, not errors.  The
  // element width is the one genuine invariant: the caller's dtype allowlist guarantees
  // it, and a violation would mean the two sides disagree about the contract.
  TORCH_CHECK(x.element_size() == 2, "nearest2x expects 2-byte elements, got ",
              x.element_size());
  // The TorchScript tracer records dispatched operators, not this call, so a traced graph
  // that reached here would hold a constant and ignore its input.  Refusing sends the
  // caller to the baseline expression, which the tracer records faithfully.  Answered
  // before anything is allocated, so the graph stays clean.
  if (torch::jit::tracer::isTracing()) {
    return at::Tensor();
  }
  if (!x.is_cuda() || x.dim() != 4 || !x.is_contiguous() || x.numel() == 0) {
    return at::Tensor();
  }
  // A tensor carrying a forward-mode tangent is refused here rather than in Python: this
  // asks the tensor itself instead of asking whether a dual level is open, so it holds
  // however the level was entered, and it costs a pointer test.  Copying bits would
  // return a correct primal with the tangent silently dropped.
  if (x._fw_grad(/*level=*/0).defined()) {
    return at::Tensor();
  }

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(x));

  const int64_t n = x.size(0);
  const int64_t c = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);
  const int64_t rows = n * c * h;
  if (w > static_cast<int64_t>(kMaxThreadsPerBlock) * kWidestVector) {
    return at::Tensor();  // -> None: a row this wide needs more threads than a block has
  }
  const int width = static_cast<int>(w);

  at::Tensor out = at::empty({n, c, 2 * h, 2 * w}, x.options());

  const auto* in_ptr = static_cast<const uint16_t*>(x.const_data_ptr());
  auto* out_ptr = static_cast<uint16_t*>(out.mutable_data_ptr());
  const int vec = select_vector_width(reinterpret_cast<uintptr_t>(in_ptr),
                                      reinterpret_cast<uintptr_t>(out_ptr), width);
  if (vec == 0) {
    return at::Tensor();  // -> None: caller falls back
  }

  const int groups = width / vec;
  const int rows_y = rows_per_block(groups);
  const int64_t blocks = (rows + rows_y - 1) / rows_y;
  const int64_t max_grid_x = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  if (groups > kMaxThreadsPerBlock || groups * rows_y > kMaxThreadsPerBlock
      || blocks < 1 || blocks > max_grid_x) {
    return at::Tensor();  // -> None: no valid launch for this row width
  }

  const dim3 block(groups, rows_y);
  const dim3 grid(static_cast<unsigned int>(blocks));
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (vec) {
    case 4:
      nearest2x_kernel<4><<<grid, block, 0, stream>>>(in_ptr, out_ptr, rows, width);
      break;
    case 2:
      nearest2x_kernel<2><<<grid, block, 0, stream>>>(in_ptr, out_ptr, rows, width);
      break;
    default:
      nearest2x_kernel<1><<<grid, block, 0, stream>>>(in_ptr, out_ptr, rows, width);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


def _build_extension():
    """Compile the upsample extension, or return None if anything goes wrong.

    Called once at import.  A build failure must not raise: the harness imports this
    module once per worker process, and an exception here would lose every case instead
    of degrading to the baseline expression.
    """
    if not torch.cuda.is_available():
        # Nothing the extension serves can be reached without a device, and building for
        # every architecture in the ambient list would cost minutes for nothing.
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        overrides = {
            # Workspace-local build cache, so the on-disk cache cannot race sibling
            # operator workspaces that share a home directory.  `setdefault` semantics:
            # an ambient value wins.
            "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR")
            or str(Path(__file__).resolve().parent / ".torch_extensions"),
            # Build for the device this process will actually run on.  The ambient value
            # names several architectures, and compiling all of them multiplies the cold
            # build time for no benefit here.
            "TORCH_CUDA_ARCH_LIST": "%d.%d" % torch.cuda.get_device_capability(),
        }

        saved = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            return load_inline(
                name="fk_interpolate_nearest2x_v1",
                cpp_sources=_CPP_DECLARATIONS,
                cuda_sources=_CUDA_SOURCE,
                functions=["nearest2x"],
                extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False,
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    except Exception:
        return None


_ext = _build_extension()


def _is_scale_two(scale_factor) -> bool:
    """True only for a scale factor that denotes exactly 2 on both spatial axes.

    The comparison is exact on purpose: F.interpolate computes the output size as
    floor(input * scale), so a scale of 1.9999 is a different operator (a size-3 axis
    becomes 5, not 6) and must reach the baseline expression.
    """
    if isinstance(scale_factor, (int, float)):
        return scale_factor == 2
    if isinstance(scale_factor, (tuple, list)):
        return len(scale_factor) == 2 and all(
            isinstance(s, (int, float)) and s == 2 for s in scale_factor
        )
    return False


class Interpolate(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        if _is_compiling():
            # Dynamo refuses to trace the extension call, so tracing gets the baseline
            # expression verbatim, returned before anything else -- in particular before
            # the fallback counter, since mutating a module global is itself enough to stop
            # a full graph being captured.  The TorchScript tracer is handled on the other
            # side, by the extension refusing while a trace is in progress.
            return F.interpolate(
                x,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
            )

        # This gate judges *semantics* only, and each condition is one the extension
        # cannot see: `align_corners is None` is required rather than merely falsy
        # because F.interpolate raises ValueError for any non-None value in nearest mode;
        # the dtype allowlist is explicit because F.interpolate itself rejects the other
        # 2-byte dtypes; the exact tensor type test keeps tensor subclasses on the
        # baseline path where their dispatch is honored; an input that would produce a
        # differentiable result falls back, since the kernel's output carries no autograd
        # history.  Everything about *layout* — device,
        # rank, contiguity, empty tensors, row width, pointer alignment — is the
        # extension's own capability question, and it answers None for anything it cannot
        # serve.  Ordering matters: the scored metric on the smaller shapes is bound by
        # per-call host cost, so the cheapest and most selective tests come first, and the
        # scale test is inlined for the spelling the captured workload uses.
        if (
            _ext is not None
            and mode == "nearest"
            and size is None
            and align_corners is None
            and (scale_factor == 2.0 if type(scale_factor) is float
                 else _is_scale_two(scale_factor))
            and type(x) is _TENSOR
            and x.dtype in _FAST_DTYPES
            and not (_is_grad_enabled() and x.requires_grad)
        ):
            out = _ext.nearest2x(x)
            if out is not None:  # None: a layout the kernel does not serve
                return out

        global _FALLBACK_CALLS
        _FALLBACK_CALLS += 1
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
