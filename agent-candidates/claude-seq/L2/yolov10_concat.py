"""YOLOv10 tensor concatenation op.

Every captured call is ``torch.cat([a, b], 1)`` on two contiguous NCHW fp16
feature maps (192..384 output channels, 20x20..80x80, batch 1/4), so the op is a
pure memory copy: read both inputs once, write ``numel(a) + numel(b)`` elements
once. Nothing here is compute; the only questions are how many bytes move and
how much of the measurement is launch latency.

For contiguous inputs a concat along ``dim`` is a byte-level interleave: the
output is ``outer = prod(sizes[:dim])`` rows, each holding ``n0`` bytes of ``a``
followed by ``n1`` bytes of ``b``. The kernel below copies those rows in 16B
units with the row index on ``blockIdx.y``, so a thread needs no division, no
strides descriptor and no per-tensor metadata load -- just one predicate
(``j < v0``) to pick its source, then one coalesced 128-bit load and store.
ATen's ``CatArrayBatchedCopy_vectorized`` is also 16B-wide but spreads the work
over ~4x more blocks (one block range per input, 128 threads each, one unit per
thread), which costs it real time once the copy is big: for the 9.4 MiB case it
measures 9.4us against 8.2us here (ncu, isolated), and 19.4us against 17.4us
under the bench's L2-flushed timing.

Below ~1 MiB of output there is nothing to win -- a kernel that returns
immediately measures the same as ATen's full copy, i.e. the case is entirely
launch latency -- so those sizes, and anything outside the shape family
(non-contiguous, mixed dtype, more than two inputs, ``outer > 65535``), are
handed back to ``at::cat``. Semantics therefore match the baseline exactly,
including dtype promotion and the autograd path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_CAT_CPP = r"""
#include <torch/extension.h>
at::Tensor fk_cat2(const at::Tensor& a, const at::Tensor& b, int64_t dim);
"""

_CAT_CUDA = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

#define FK_MIN_BYTES (1 << 20)

// One output row per blockIdx.y: [v0 units of *a*][v1 units of *b*]. Threads
// walk the row in NT-wide strides so both halves stay fully coalesced; the
// single `j < v0` predicate splits the two sources without any division.
template <typename T, int NT, int U>
__global__ void __launch_bounds__(NT) fk_cat2_rows(
    T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
    int v0, int v1) {
  const int vt = v0 + v1;
  const long r = (long)blockIdx.y;
  T* o = out + r * vt;
  const T* pa = a + r * v0;
  const T* pb = b + r * v1;
  const int base = blockIdx.x * (NT * U) + threadIdx.x;
  T t[U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int j = base + u * NT;
    if (j < vt) t[u] = (j < v0) ? pa[j] : pb[j - v0];
  }
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int j = base + u * NT;
    if (j < vt) o[j] = t[u];
  }
}

at::Tensor fk_cat2(const at::Tensor& a, const at::Tensor& b, int64_t dim_in) {
  const int64_t nd = a.dim();
  // Dense, CUDA, inference-only, same dtype/rank: everything else is ATen's.
  if (nd == 0 || nd != b.dim() || !a.is_cuda() || !b.is_cuda()
      || a.device() != b.device() || a.scalar_type() != b.scalar_type()
      || a.layout() != c10::kStrided || b.layout() != c10::kStrided
      || a.requires_grad() || b.requires_grad()
      || !a.is_contiguous() || !b.is_contiguous())
    return at::cat({a, b}, dim_in);

  const int64_t dim = at::maybe_wrap_dim(dim_in, nd);
  int64_t osz[8];
  if (nd > 8) return at::cat({a, b}, dim_in);
  for (int64_t d = 0; d < nd; ++d) {
    if (d != dim && a.size(d) != b.size(d)) return at::cat({a, b}, dim_in);
    osz[d] = (d == dim) ? a.size(d) + b.size(d) : a.size(d);
  }

  int64_t outer = 1;
  for (int64_t d = 0; d < dim; ++d) outer *= osz[d];
  int64_t inner = 1;
  for (int64_t d = dim + 1; d < nd; ++d) inner *= osz[d];
  const int64_t esz = a.element_size();
  const int64_t n0 = a.size(dim) * inner * esz;  // bytes of *a* per output row
  const int64_t n1 = b.size(dim) * inner * esz;
  // Under ~1 MiB of output the copy is pure launch latency (a do-nothing
  // kernel measures the same as ATen's cat at these sizes), so there is nothing
  // for a custom kernel to win -- hand those back to ATen.
  if (outer > 65535 || (n0 + n1) > (int64_t)0x7fffffff * 16
      || (n0 + n1) * outer < FK_MIN_BYTES)
    return at::cat({a, b}, dim_in);

  const c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor out = at::detail::empty_cuda(at::IntArrayRef(osz, nd),
                                          a.scalar_type(), a.device(), std::nullopt);

  const char* ap = (const char*)a.data_ptr();
  const char* bp = (const char*)b.data_ptr();
  char* op = (char*)out.data_ptr();
  // Widest power-of-two unit that divides both row spans and every base
  // pointer. OR-ing is sound: each term is tested against the same power of two.
  int unit = 1;
  for (int u = 16; u > 1; u >>= 1) {
    if ((((n0 | n1) % u) == 0)
        && ((((uintptr_t)ap | (uintptr_t)bp | (uintptr_t)op) % u) == 0)) { unit = u; break; }
  }

  const int v0 = (int)(n0 / unit), v1 = (int)(n1 / unit);
  const int64_t total = (int64_t)(v0 + v1) * outer;
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Two launch shapes, picked by total copy size. Both keep the whole GPU busy;
  // the wide one (128 threads x 4 units = 512 units per block) amortizes block
  // scheduling once the copy is long enough to be bandwidth-bound, while the
  // narrow one (256 threads x 2 units) keeps more blocks resident for copies
  // short enough to still be latency-bound.
#define FK_LAUNCH_T(T, NT, U)                                                 \
  do {                                                                        \
    dim3 g((unsigned)((v0 + v1 + (NT) * (U)-1) / ((NT) * (U))), (unsigned)outer); \
    fk_cat2_rows<T, NT, U><<<g, NT, 0, stream>>>(                              \
        (T*)op, (const T*)ap, (const T*)bp, v0, v1);                          \
  } while (0)
#define FK_LAUNCH(T)                                                          \
  do {                                                                        \
    if (total >= 400000) FK_LAUNCH_T(T, 128, 4);                               \
    else                 FK_LAUNCH_T(T, 256, 2);                               \
  } while (0)

  switch (unit) {
    case 16: FK_LAUNCH(uint4); break;
    case 8:  FK_LAUNCH(unsigned long long); break;
    case 4:  FK_LAUNCH(unsigned int); break;
    case 2:  FK_LAUNCH(unsigned short); break;
    default: FK_LAUNCH(unsigned char); break;
  }
#undef FK_LAUNCH
#undef FK_LAUNCH_T
  return out;
}
"""


def _build_cat() -> object | None:
    """JIT-build the two-input cat kernel; ``None`` if it can't be compiled."""
    try:
        # Importing cuda_ext pins TORCH_CUDA_ARCH_LIST to the local GPU, so the
        # build targets one arch instead of every gencode in the default list.
        from fastkernels.infra import cuda_ext  # noqa: F401
    except Exception:  # noqa: BLE001 -- arch pinning is an optimization only
        pass
    try:
        from torch.utils.cpp_extension import load_inline
        return load_inline(
            name="fk_yolov10_cat2",
            cpp_sources=_CAT_CPP,
            cuda_sources=_CAT_CUDA,
            functions=["fk_cat2"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        ).fk_cat2
    except Exception:  # noqa: BLE001 -- fall back to torch.cat
        return None


_fk_cat2 = _build_cat()


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if _fk_cat2 is not None and len(xs) == 2:
            return _fk_cat2(xs[0], xs[1], self.d)
        return torch.cat(xs, self.d)
