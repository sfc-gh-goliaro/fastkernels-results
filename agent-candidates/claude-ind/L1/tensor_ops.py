"""Primitive tensor manipulation ops.

L1 ops wrapping standard tensor utilities for use by L2+ composites.

``Pad`` is the hot op here: the captured calls are all tiny constant pads
(``[1, 368]`` .. ``[1, 1, 368, 128]``, one padded dim, zero fill), so the cost
is entirely per-launch overhead, not bandwidth. ``F.pad`` -> ``constant_pad_nd``
materializes the output with ``empty`` + ``fill_`` + a narrowed ``copy_``, i.e.
two GPU kernels plus two TensorIterator setups. The CUDA kernel below writes
every output element exactly once -- copied or filled -- in a single launch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fused constant-pad kernel.
#
# Every captured pad touches exactly one dimension, so the output decomposes
# into ``outer`` identical rows of ``row_out`` elements, each holding the
# corresponding input row at offset ``off`` and the fill value elsewhere. Rows
# ride on ``blockIdx.y`` so the kernel needs no integer division, and the
# copy/fill runs in the widest power-of-two unit (up to 16B) that divides every
# row offset and both base pointers. Anything outside that shape family
# (negative pads, >1 padded dim, non-contiguous input, exotic dtype) falls back
# to ``at::constant_pad_nd``, so semantics match ``F.pad`` exactly.
# ---------------------------------------------------------------------------
_PAD_CPP = r"""
#include <torch/extension.h>
at::Tensor fk_pad(const at::Tensor& x, std::vector<int64_t> pad,
                  std::optional<double> value);
"""

_PAD_CUDA = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#define FK_MAXDIM 8
#define FK_NT 256

// out[r * row_out + j] = (j - off) in [0, n_in) ? in[r * n_in + j - off] : fill
template <typename T>
__global__ void __launch_bounds__(FK_NT) fk_pad_rows(
    T* __restrict__ out, const T* __restrict__ in,
    long row_out, long n_in, long off, T fill) {
  const long j = (long)blockIdx.x * FK_NT + threadIdx.x;
  if (j >= row_out) return;
  const long k = j - off;
  out[(long)blockIdx.y * row_out + j] =
      (k >= 0 && k < n_in) ? in[(long)blockIdx.y * n_in + k] : fill;
}

// Hot path: one row, no left pad -> 32-bit indexing, no y dimension.
template <typename T>
__global__ void __launch_bounds__(FK_NT) fk_pad_tail(
    T* __restrict__ out, const T* __restrict__ in, int row_out, int n_in, T fill) {
  const int i = blockIdx.x * FK_NT + threadIdx.x;
  if (i < row_out) out[i] = (i < n_in) ? in[i] : fill;
}

// One element of *value* as raw bytes; false for dtypes we don't handle.
static inline bool fk_pattern(unsigned char* buf, at::ScalarType st, double v) {
  switch (st) {
    case at::kBFloat16: { c10::BFloat16 t = (float)v; std::memcpy(buf, &t, 2); return true; }
    case at::kHalf:     { c10::Half t = (float)v;     std::memcpy(buf, &t, 2); return true; }
    case at::kFloat:    { float t = (float)v;         std::memcpy(buf, &t, 4); return true; }
    case at::kDouble:   {                             std::memcpy(buf, &v, 8); return true; }
    case at::kLong:     { long long t = (long long)v; std::memcpy(buf, &t, 8); return true; }
    case at::kInt:      { int t = (int)v;             std::memcpy(buf, &t, 4); return true; }
    case at::kShort:    { short t = (short)v;         std::memcpy(buf, &t, 2); return true; }
    case at::kChar:     { signed char t = (signed char)v;     std::memcpy(buf, &t, 1); return true; }
    case at::kByte:     { unsigned char t = (unsigned char)v; std::memcpy(buf, &t, 1); return true; }
    case at::kBool:     { bool t = (v != 0.0);        std::memcpy(buf, &t, 1); return true; }
    default: return false;
  }
}

at::Tensor fk_pad(const at::Tensor& x, std::vector<int64_t> pad,
                  std::optional<double> value_opt) {
  const double value = value_opt.value_or(0.0);
  const at::IntArrayRef padref(pad.data(), pad.size());
  const int64_t np = (int64_t)pad.size();
  const int64_t nd = x.dim();
  // Anything outside the dense/CUDA/inference fast path defers to ATen. The
  // requires_grad bail-out matters because the kernel writes a raw buffer and
  // would silently drop the backward graph F.pad would have built.
  if ((np & 1) || np > 2 * nd || nd == 0 || nd > FK_MAXDIM || !x.is_cuda()
      || x.layout() != c10::kStrided || x.requires_grad() || !x.is_contiguous())
    return at::constant_pad_nd(x, padref, value);

  const int64_t* isz = x.sizes().data();
  int64_t osz[FK_MAXDIM];
  for (int64_t d = 0; d < nd; ++d) osz[d] = isz[d];

  // Locate the single padded dimension (pad[2i], pad[2i+1] apply to dim nd-1-i).
  int64_t d0 = -1, lpad = 0;
  for (int64_t i = 0; i < np / 2; ++i) {
    const int64_t l = pad[2 * i], r = pad[2 * i + 1];
    if (l < 0 || r < 0 || (d0 >= 0 && (l | r)))
      return at::constant_pad_nd(x, padref, value);  // crop, or >1 padded dim
    if (l | r) { d0 = nd - 1 - i; lpad = l; osz[d0] += l + r; }
  }
  if (d0 < 0) d0 = nd - 1;  // no-op pad: a plain copy

  int64_t inner = 1;
  for (int64_t d = d0 + 1; d < nd; ++d) inner *= isz[d];
  int64_t outer = 1;
  for (int64_t d = 0; d < d0; ++d) outer *= isz[d];

  const int esz = (int)x.element_size();
  unsigned char pat[16];
  if (outer > 65535 || !fk_pattern(pat, x.scalar_type(), value))
    return at::constant_pad_nd(x, padref, value);

  const c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor out = at::detail::empty_cuda(at::IntArrayRef(osz, nd),
                                          x.scalar_type(), x.device(), std::nullopt);
  const int64_t n_in = isz[d0] * inner;
  const int64_t row_out = osz[d0] * inner;
  const int64_t off = lpad * inner;
  if (row_out == 0 || outer == 0) return out;

  const char* ip = (const char*)x.data_ptr();
  char* op = (char*)out.data_ptr();
  // Widest unit that divides every byte offset and both base pointers. OR-ing
  // the offsets is sound: they are all tested against a power of two.
  int unit = esz;
  for (int u = 16; u > esz; u >>= 1) {
    if ((((row_out | n_in | off) * esz) % u) == 0 &&
        ((((uintptr_t)ip) | ((uintptr_t)op)) % u) == 0) { unit = u; break; }
  }
  for (int b = esz; b < unit; b <<= 1) std::memcpy(pat + b, pat, b);

  const int64_t ru = row_out * esz / unit, nu = n_in * esz / unit,
                ou = off * esz / unit;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned)((ru + FK_NT - 1) / FK_NT), (unsigned)outer, 1);

#define FK_LAUNCH(T)                                                           \
  do {                                                                         \
    T f;                                                                        \
    std::memcpy(&f, pat, sizeof(T));                                            \
    if (outer == 1 && ou == 0 && ru <= 0x7fffffff)                              \
      fk_pad_tail<T><<<grid, FK_NT, 0, stream>>>((T*)op, (const T*)ip,          \
                                                 (int)ru, (int)nu, f);          \
    else                                                                        \
      fk_pad_rows<T><<<grid, FK_NT, 0, stream>>>((T*)op, (const T*)ip, ru, nu,  \
                                                 ou, f);                        \
  } while (0)

  switch (unit) {
    case 16: FK_LAUNCH(uint4); break;
    case 8:  FK_LAUNCH(unsigned long long); break;
    case 4:  FK_LAUNCH(unsigned int); break;
    case 2:  FK_LAUNCH(unsigned short); break;
    default: FK_LAUNCH(unsigned char); break;
  }
#undef FK_LAUNCH
  return out;
}
"""


def _build_pad():
    """JIT-build the fused pad kernel; ``None`` if it can't be compiled."""
    try:
        # Importing cuda_ext pins TORCH_CUDA_ARCH_LIST to the local GPU, so the
        # build targets one arch instead of every gencode in the default list.
        from fastkernels.infra import cuda_ext  # noqa: F401
    except Exception:  # noqa: BLE001 -- arch pinning is an optimization only
        pass
    try:
        from torch.utils.cpp_extension import load_inline
        return load_inline(
            name="fk_tensor_ops_pad",
            cpp_sources=_PAD_CPP,
            cuda_sources=_PAD_CUDA,
            functions=["fk_pad"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        ).fk_pad
    except Exception:  # noqa: BLE001 -- fall back to F.pad
        return None


_fk_pad = _build_pad()

if _fk_pad is None:  # no nvcc / unsupported toolchain -- keep the baseline path
    def _fk_pad(x, pad, value):
        return F.pad(x, pad, value=value)


class Pad(nn.Module):
    """Functional padding op."""

    def forward(
        self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0,
    ) -> torch.Tensor:
        return _fk_pad(x, pad, value)


class OneHot(nn.Module):
    """Functional one-hot encoding op."""

    def forward(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(x, num_classes)


class Cat(nn.Module):
    """Tensor concatenation op."""

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(tensors, dim=self.dim)


class Exp(nn.Module):
    """Elementwise exponential op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)
