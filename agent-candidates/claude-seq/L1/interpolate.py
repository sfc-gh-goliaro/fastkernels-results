"""Interpolate: custom CUDA kernel for the hot 2x nearest-neighbour upsample.

The captured workload is exclusively ``mode="nearest"`` with ``scale_factor=2.0``
on contiguous fp16 NCHW tensors.  Nearest upsampling there is not arithmetic at
all -- every input element is replicated into a 2x2 output block -- so the op is
pure store bandwidth plus call overhead.  Two things make the ATen path slow:

* ``upsample_nearest2d`` runs one scalar element per thread and recomputes the
  source index (integer divides) for every *output* element.  On B200 it needs
  12.3 us for a 3.3 MB output where a plain copy of the same bytes takes 0.41 us
  -- ~30x off the roofline.
* at these sizes the op is dispatch-bound rather than bandwidth-bound: in the
  bench's timing window one kernel costs ~3.1 us regardless of its grid, so the
  only things that matter are the *number* of kernels (one is the floor) and any
  data movement beyond that floor.  Host-side cost does not register at all --
  a deliberate 10 us busy-wait inside ``forward`` changed the measured latency
  by 0.0 us -- so the win has to come from the kernel, not the Python prologue.

The kernel has each thread read 4 halves (one aligned 8-byte ``uint2``), widen
them to 8 halves with ``__byte_perm`` (h0 h0 h1 h1 h2 h2 h3 h3) and issue two
fully coalesced 16-byte stores -- one per output row the input row feeds.  That
is 1 load + 2 vector stores per 4 inputs, no per-element index math, and it is
bit-exact (values are copied, never converted), so it works for any 2-byte
float dtype.  ``W/4`` is a template parameter for the captured widths so the row
divide folds into a compile-time multiply-shift.  It sustains 5.8-6.3 TB/s on
large inputs, matching a pure ``copy_`` of the same bytes (ATen manages 1.1
TB/s), which puts the captured shapes ~1 us above the one-kernel floor.

The host path is kept lean too -- a ``METH_FASTCALL`` C entry point, argument
screening in C, ``at::detail::empty_cuda`` instead of the ``at::empty``
dispatcher hop -- which costs nothing to do but, per the above, buys nothing
either; a single kernel launch is the whole game.

Anything the fast path does not recognise (other modes, scales, ranks, dtypes,
non-contiguous or misaligned inputs, CPU tensors) falls through to
``F.interpolate`` unchanged.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/EmptyTensor.h>

// One thread per group of 4 input halves: 8 B in, 2 x 16 B out.
//   G   = W / 4 = groups per input row = uint4s per *output* row (2W/8)
//   in  : input row r starts at uint2 index r*G
//   out : input row r feeds output rows 2r and 2r+1, i.e. uint4 indices
//         (2r)*G + g and (2r+1)*G + g, so with i = r*G + g the two stores are
//         at j = i + r*G and j + G.
template <int G_STATIC>
__global__ __launch_bounds__(256) void up2x_nearest_kernel(
    const uint2 *__restrict__ in, uint4 *__restrict__ out,
    unsigned total, unsigned g_dyn) {
  const unsigned G = (G_STATIC > 0) ? (unsigned)G_STATIC : g_dyn;
  const unsigned i = blockIdx.x * 256u + threadIdx.x;
  if (i >= total) return;
  const unsigned r = i / G;
  const uint2 v = in[i];
  uint4 o;
  o.x = __byte_perm(v.x, 0u, 0x1010);  // h0 h0
  o.y = __byte_perm(v.x, 0u, 0x3232);  // h1 h1
  o.z = __byte_perm(v.y, 0u, 0x1010);  // h2 h2
  o.w = __byte_perm(v.y, 0u, 0x3232);  // h3 h3
  const unsigned j = i + r * G;
  out[j] = o;
  out[j + G] = o;
}

// Returns an undefined tensor when the fast path does not apply.
at::Tensor up2x_nearest(const at::Tensor &x) {
  const int64_t ndim = x.dim();
  // fp16 / bf16 only: ATen has no upsample_nearest2d for the other 2-byte
  // dtypes, so widening the accepted set would change observable behaviour.
  if (ndim != 4 || !x.is_cuda() || x.element_size() != 2 ||
      !x.is_floating_point() || !x.is_contiguous()) {
    return at::Tensor();
  }
  const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  const int64_t numel = N * C * H * W;
  if ((W & 3) != 0 || numel <= 0 || numel > (int64_t)1 << 31 ||
      (reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 7u) != 0) {
    return at::Tensor();
  }

  at::Tensor out = at::detail::empty_cuda({N, C, H * 2, W * 2}, x.scalar_type(),
                                          x.device(), c10::nullopt);
  const unsigned total = (unsigned)(numel >> 2);  // groups of 4 halves
  const unsigned G = (unsigned)(W >> 2);
  const unsigned blocks = (total + 255u) / 256u;
  const uint2 *ip = reinterpret_cast<const uint2 *>(x.const_data_ptr());
  uint4 *op = reinterpret_cast<uint4 *>(out.data_ptr());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (G) {
    case 5:   // W = 20
      up2x_nearest_kernel<5><<<blocks, 256, 0, stream>>>(ip, op, total, G);
      break;
    case 10:  // W = 40
      up2x_nearest_kernel<10><<<blocks, 256, 0, stream>>>(ip, op, total, G);
      break;
    case 20:  // W = 80
      up2x_nearest_kernel<20><<<blocks, 256, 0, stream>>>(ip, op, total, G);
      break;
    default:
      up2x_nearest_kernel<0><<<blocks, 256, 0, stream>>>(ip, op, total, G);
      break;
  }
  return out;
}
"""

# METH_FASTCALL entry point: screens (size, scale_factor, mode) and dispatches
# without building an argument tuple or going through pybind's casters.
_CPP_SRC = r"""
#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>

at::Tensor up2x_nearest(const at::Tensor &x);

static PyObject *g_nearest = nullptr;  // interned "nearest"

static inline bool is_two(PyObject *o) {
  if (PyFloat_CheckExact(o)) return PyFloat_AS_DOUBLE(o) == 2.0;
  if (PyLong_CheckExact(o)) return PyLong_AsLong(o) == 2;
  return false;
}

// up2x(x, size, scale_factor, mode) -> Tensor | None
static PyObject *up2x_fast(PyObject *, PyObject *const *args, Py_ssize_t nargs) {
  if (nargs != 4 || args[1] != Py_None || !THPVariable_Check(args[0])) {
    Py_RETURN_NONE;
  }
  PyObject *mode = args[3];
  if (mode != g_nearest &&
      !(PyUnicode_CheckExact(mode) &&
        PyUnicode_CompareWithASCIIString(mode, "nearest") == 0)) {
    Py_RETURN_NONE;
  }
  PyObject *sf = args[2];
  if (!is_two(sf)) {
    // also accept a per-dim (2, 2) / [2.0, 2.0]
    if (!(PyTuple_CheckExact(sf) && PyTuple_GET_SIZE(sf) == 2 &&
          is_two(PyTuple_GET_ITEM(sf, 0)) && is_two(PyTuple_GET_ITEM(sf, 1)))) {
      Py_RETURN_NONE;
    }
  }
  at::Tensor out;
  try {
    out = up2x_nearest(THPVariable_Unpack(args[0]));
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
  if (!out.defined()) Py_RETURN_NONE;
  return THPVariable_Wrap(std::move(out));
}

static PyMethodDef fk_methods[] = {
    {"up2x", (PyCFunction)(void *)up2x_fast, METH_FASTCALL, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  g_nearest = PyUnicode_InternFromString("nearest");
  PyModule_AddFunctions(m.ptr(), fk_methods);
}
"""


_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".build_interpolate")


def _build():
    from torch.utils.cpp_extension import load_inline

    os.makedirs(_BUILD_DIR, exist_ok=True)
    # An explicit -gencode keeps cpp_extension from expanding the ambient
    # TORCH_CUDA_ARCH_LIST into a seven-architecture fatbin.
    arch = "-gencode=arch=compute_100,code=sm_100"
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return load_inline(
        name="fk_interpolate_up2x",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3", arch],
        extra_cflags=["-O3"],
        build_directory=_BUILD_DIR,
        verbose=False,
    )


def _load():
    """Build the extension, retrying once from scratch so that a stale build
    cache (different torch/python/arch) cannot silently demote us to ATen."""
    try:
        return _build().up2x
    except Exception:
        import shutil

        shutil.rmtree(_BUILD_DIR, ignore_errors=True)
        return _build().up2x


try:
    _up2x = _load()
except Exception:  # pragma: no cover - no nvcc / no CUDA: stay on ATen
    _up2x = None


def _fallback(x, size, scale_factor, mode, align_corners):
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode,
                         align_corners=align_corners)


class Interpolate(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        if _up2x is not None:
            out = _up2x(x, size, scale_factor, mode)
            if out is not None:
                return out
        return _fallback(x, size, scale_factor, mode, align_corners)
