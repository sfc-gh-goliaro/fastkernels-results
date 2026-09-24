"""YOLOv10 tensor concatenation op, with a CUDA kernel for the channel-concat case.

Concatenating contiguous NCHW tensors along the channel axis is a pure copy: output
element ``(n, c, h, w)`` comes from the first input while ``c < C0`` and from the second
input after that. Flattened per batch row it is a two-source contiguous copy, which one
16-byte-per-lane kernel does in a single launch.

At these sizes the copy is not the expensive part. Most shapes a YOLOv10 concat sees do not
fill the GPU once, so the kernel's cost in a stream is dominated by getting the grid onto the
machine rather than by moving the bytes. The kernel is therefore launched with programmatic
stream serialization so its grid can be staged while whatever produced its inputs is still
draining, and it waits on that dependency before its first load.

The kernel only covers the two-input, ``dim=1``, contiguous, 16-byte-alignable CUDA case.
``forward`` decides eligibility in Python *before* launching anything and hands every other
input to ``torch.cat``, so the module stays equivalent to the reference for inputs the
kernel was never written for (other ranks, other dims, more or fewer inputs, CPU tensors,
non-contiguous views, mismatched shapes or dtypes, empty tensors, autograd).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SOURCE = r'''
#include <ATen/ATen.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

// One 16-byte vector per thread. Wider tiles per thread were measured worse: they shrink the
// grid, and the small sub-one-wave shapes need enough blocks to reach every SM. 128 threads
// is the smallest block that still clears one block per SM on every shape this operator sees.
constexpr int kThreads = 128;

// Copy `vec_a` vectors from `src_a` followed by `vec_b` vectors from `src_b` into each batch
// row of `dst`. `Batched` drops the row offset arithmetic entirely when there is only one row.
// The grid-stride loop keeps the kernel correct for any grid the host picks.
template <int NT, bool Batched>
__global__ __launch_bounds__(NT) void concat_rows_kernel(
    const uint4* __restrict__ src_a, const uint4* __restrict__ src_b,
    uint4* __restrict__ dst, int vec_a, int vec_b) {
  // When this grid was launched with programmatic stream serialization it may already be
  // resident while the kernel that produces our inputs is still draining, so wait for that
  // dependency before the first global load. Everything above this point reads only the
  // launch configuration and kernel parameters. Launched without that attribute the wait is
  // a no-op and ordinary stream ordering already guarantees the inputs are complete.
  cudaGridDependencySynchronize();
  const int vec_row = vec_a + vec_b;
  const long long row_dst = Batched ? (long long)blockIdx.y * vec_row : 0;
  const long long row_a = Batched ? (long long)blockIdx.y * vec_a : 0;
  const long long row_b = Batched ? (long long)blockIdx.y * vec_b : 0;
  const int stride = NT * gridDim.x;

  for (int i = blockIdx.x * NT + threadIdx.x; i < vec_row; i += stride) {
    dst[row_dst + i] = (i < vec_a) ? src_a[row_a + i] : src_b[row_b + (i - vec_a)];
  }
}

// Programmatic dependent launch needs compute capability 9.0 or newer. Probed once; on
// anything older the plain launch path is used and the in-kernel wait is inert.
bool dependent_launch_supported() {
  static const bool supported = [] {
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)
        != cudaSuccess) {
      return false;
    }
    return major >= 9;
  }();
  return supported;
}

}  // namespace

// Preconditions (rank, dtype, device, contiguity, shape agreement, alignment, size, deferred
// negation/conjugation, dimension names) are established by the caller before this is reached.
// The three checks below are the invariants whose violation would corrupt memory or fail with
// an opaque launch error rather than a readable one, so they stay as a safety net.
at::Tensor concat_channels(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "concat_channels expects 4-D inputs");
  const int64_t vec = 16 / a.element_size();
  const int64_t rows = a.size(0), plane = a.size(2) * a.size(3);
  const int64_t elems_a = a.size(1) * plane, elems_b = b.size(1) * plane;
  TORCH_CHECK(elems_a % vec == 0 && elems_b % vec == 0,
              "concat_channels expects both channel segments to be 16-byte multiples");
  // A batch beyond the grid's y limit otherwise fails as a bare "invalid argument" and leaves
  // a sticky error on the CUDA context, poisoning later unrelated calls.
  TORCH_CHECK(rows <= 65535, "concat_channels expects at most 65535 batch rows, got ", rows);

  const c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor out = at::Tensor(at::detail::empty_cuda(
      {rows, a.size(1) + b.size(1), a.size(2), a.size(3)},
      a.scalar_type(), a.device(), c10::nullopt));

  const int vec_a = static_cast<int>(elems_a / vec);
  const int vec_b = static_cast<int>(elems_b / vec);
  const int grid_x = (vec_a + vec_b + kThreads - 1) / kThreads;

  const auto* pa = static_cast<const uint4*>(a.const_data_ptr());
  const auto* pb = static_cast<const uint4*>(b.const_data_ptr());
  auto* pout = static_cast<uint4*>(out.data_ptr());
  const dim3 grid = (rows == 1) ? dim3(grid_x)
                                : dim3(grid_x, static_cast<unsigned>(rows));

  if (dependent_launch_supported()) {
    // The concat is one small kernel serialized behind whichever kernel produced its
    // inputs, so most of its cost in a stream is grid setup rather than the copy itself.
    // Programmatic stream serialization lets the grid be staged during the producer's tail;
    // the in-kernel wait above is what keeps that safe.
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = 0;
    config.stream = c10::cuda::getCurrentCUDAStream();
    config.attrs = attrs;
    config.numAttrs = 1;
    const cudaError_t err = (rows == 1)
        ? cudaLaunchKernelEx(&config,
              concat_rows_kernel<kThreads, false>, pa, pb, pout, vec_a, vec_b)
        : cudaLaunchKernelEx(&config,
              concat_rows_kernel<kThreads, true>, pa, pb, pout, vec_a, vec_b);
    TORCH_CHECK(err == cudaSuccess, "concat_channels launch failed: ",
                cudaGetErrorString(err));
    return out;
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  if (rows == 1) {
    concat_rows_kernel<kThreads, false><<<grid, kThreads, 0, stream>>>(
        pa, pb, pout, vec_a, vec_b);
  } else {
    concat_rows_kernel<kThreads, true><<<grid, kThreads, 0, stream>>>(
        pa, pb, pout, vec_a, vec_b);
  }
  return out;
}
'''

_CPP_SOURCE = r'''
#include <ATen/ATen.h>
at::Tensor concat_channels(const at::Tensor&, const at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("concat_channels", &concat_channels);
}
'''


def _load_extension():
    """Build the channel-concat kernel, reporting rather than hiding a failure.

    The surrounding environment presets a multi-architecture ``TORCH_CUDA_ARCH_LIST``, which
    multiplies compile time several-fold, so it is overwritten (not ``setdefault``-ed, which
    would be a no-op here) for the duration of the build and restored afterwards.
    """
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    try:
        return load_inline(
            name="fk_yolov10_concat_cuda",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=None,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        ), None
    except Exception as exc:  # noqa: BLE001 - any build failure must degrade, not raise
        return None, repr(exc)
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


_EXTENSION, BUILD_ERROR = _load_extension()
KERNEL_AVAILABLE = _EXTENSION is not None

if not KERNEL_AVAILABLE:
    # Without this line a broken build is indistinguishable from a genuine parity result,
    # since every call would silently return the reference path's output.
    print(f"[yolov10_concat] CUDA kernel unavailable, using torch.cat: {BUILD_ERROR}",
          file=sys.stderr, flush=True)

_VECTOR_BYTES = 16
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Bounds the per-row vector count so the kernel's grid-stride increment (threads per block
# times grid width) cannot overflow a 32-bit int. 2^30 vectors is a 16 GiB batch row.
_MAX_VECTORS = 1 << 30
# Batch rows map to blockIdx.y, whose CUDA grid limit is 65535.
_MAX_GRID_Y = 65535


def _value_is_its_storage(t: torch.Tensor) -> bool:
    """Whether copying ``t``'s storage bytes reproduces ``t``'s logical value.

    The kernel is a byte copy, which is only equivalent to the reference when the tensor's value
    really is what its pointer addresses. Several kinds of plain, contiguous CUDA tensor break
    that: a pending negation or conjugation is recorded in metadata rather than applied to the
    bytes, and a zero-storage or functional tensor has no readable bytes at all -- its
    ``data_ptr()`` is null, which would pass an alignment test and then fault as a load from
    address zero.
    """
    if t.is_neg() or t.is_conj():
        return False
    return t.data_ptr() != 0 or t.numel() == 0


def _kernel_applies(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the channel-concat kernel can produce exactly ``torch.cat([a, b], 1)``.

    Checked before any launch so an unsupported input takes the reference path instead of
    turning a real failure (an allocation error, a bad launch) into a plausible-looking
    result.
    """
    if type(a) is not torch.Tensor or type(b) is not torch.Tensor:
        return False  # a subclass may redefine what cat means
    if not _value_is_its_storage(a) or not _value_is_its_storage(b):
        return False
    if a.has_names() or b.has_names():
        return False  # torch.cat propagates dimension names; a raw output has none
    if a.dtype is not b.dtype or a.dtype not in _SUPPORTED_DTYPES:
        return False
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        return False
    if a.dim() != 4 or b.dim() != 4:
        return False
    if not a.is_contiguous() or not b.is_contiguous():
        return False
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2] or a.shape[3] != b.shape[3]:
        return False
    if a.shape[0] > _MAX_GRID_Y:
        return False  # more batch rows than the grid's y dimension can address
    if a.requires_grad or b.requires_grad:
        return False  # the kernel is not differentiable
    if a.numel() == 0 or b.numel() == 0:
        return False
    vector = _VECTOR_BYTES // a.element_size()
    plane = a.shape[2] * a.shape[3]
    if (a.shape[1] * plane) % vector or (b.shape[1] * plane) % vector:
        return False
    if a.data_ptr() % _VECTOR_BYTES or b.data_ptr() % _VECTOR_BYTES:
        return False
    return (a.numel() + b.numel()) // vector <= _MAX_VECTORS


class YOLOConcat(nn.Module):
    kernel_available = KERNEL_AVAILABLE
    kernel_build_error = BUILD_ERROR

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        # ``self.d`` is public and mutable, so the axis is resolved per call. Only a builtin int
        # is interpreted here: ``bool`` is a subclass of int and ``1.0 == 1``, so comparing
        # loosely would route YOLOConcat(True) and YOLOConcat(1.0) into the kernel even though
        # torch.cat rejects both, and comparing ``None`` or a string would raise from here rather
        # than from torch.cat. Anything that is not exactly an int is handed over untouched, so
        # PyTorch decides what it means and which error it raises.
        dim = self.d
        if (_EXTENSION is not None and type(dim) is int
                and isinstance(xs, (list, tuple)) and len(xs) == 2):
            a, b = xs
            # Rank is 4 whenever the kernel applies, so a negative axis normalizes against 4.
            if _kernel_applies(a, b) and (dim + 4 if dim < 0 else dim) == 1:
                return _EXTENSION.concat_channels(a, b)
        return torch.cat(xs, dim)
