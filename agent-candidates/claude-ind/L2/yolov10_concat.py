"""YOLOv10 tensor concatenation op.

``torch.cat`` along the channel axis of a batch of contiguous NCHW activations is
a pure byte shuffle: for every image ``n`` the ``K`` input slabs are laid down
back to back in the output.  ATen routes this through its generic
``CatArrayBatchedCopy``, which drives the copy from per-input metadata and
re-derives an N-d index as it goes.  Here the layout is fixed, so one kernel can
stream the whole thing as 16-byte vectors with a single compare per vector; on
the captured shapes that lands within a few percent of a plain device-to-device
memcpy of the same volume.  The grid is also kept short and evenly divided,
because at these sizes launch cost is a large share of the total.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// Threads per block; 256 measured fastest across the captured shapes.
#define CAT_THREADS 256
// Cap on grid.x * grid.y.  Launch latency on Blackwell starts growing with the
// block count, and these copies are small enough that extra blocks buy nothing
// -- the grid-stride loop soaks up the remainder instead.
#define CAT_MAX_BLOCKS 2048

template <int K>
struct CatPlan {
  const uint4* __restrict__ p[K];  // input base pointers (16B units)
  int slab[K];                     // vectors per outer slice, per input
  int off[K + 1];                  // output prefix offsets, off[K] == out slab
};

// grid.y indexes the outer (batch) slice, grid.x tiles one output slab.
template <int K>
__global__ __launch_bounds__(CAT_THREADS) void cat_dim1_kernel(
    CatPlan<K> plan, uint4* __restrict__ out) {
  const int total = plan.off[K];
  const long b = blockIdx.y;
  uint4* __restrict__ o = out + b * (long)total;
  const int stride = gridDim.x * CAT_THREADS;
  for (int i = blockIdx.x * CAT_THREADS + threadIdx.x; i < total; i += stride) {
    // Pick the owning segment with compile-time struct indices so everything
    // stays in registers (an array lookup by a runtime index would spill).
    const uint4* __restrict__ src = plan.p[0] + b * (long)plan.slab[0];
    int base = 0;
#pragma unroll
    for (int k = 1; k < K; ++k) {
      const bool take = (i >= plan.off[k]);
      src = take ? (plan.p[k] + b * (long)plan.slab[k]) : src;
      base = take ? plan.off[k] : base;
    }
    o[i] = src[i - base];
  }
}

template <int K>
static void launch(const std::vector<at::Tensor>& xs, at::Tensor& out, long outer) {
  CatPlan<K> plan;
  int off = 0;
  for (int k = 0; k < K; ++k) {
    plan.p[k] = reinterpret_cast<const uint4*>(xs[k].data_ptr());
    plan.slab[k] = (int)((xs[k].numel() / outer) * xs[k].element_size() / 16);
    plan.off[k] = off;
    off += plan.slab[k];
  }
  plan.off[K] = off;
  if (off == 0) return;
  // Round the grid down to a whole number of grid-stride passes so every thread
  // copies the same number of vectors -- a ragged last pass leaves most of the
  // machine idle while a few blocks finish.
  const long need = (off + CAT_THREADS - 1) / CAT_THREADS;
  long capx = CAT_MAX_BLOCKS / outer;
  if (capx < 1) capx = 1;
  const long passes = (need + capx - 1) / capx;
  const int gx = (int)((need + passes - 1) / passes);
  dim3 grid((unsigned)gx, (unsigned)outer);
  cat_dim1_kernel<K><<<grid, CAT_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      plan, reinterpret_cast<uint4*>(out.data_ptr()));
}

// Same-shape-except-dim-1, contiguous, 16B-aligned fast path; anything else
// falls through to ATen.
static bool usable(const std::vector<at::Tensor>& xs) {
  const at::Tensor& a = xs[0];
  if (!a.is_cuda() || a.dim() < 2) return false;
  const long outer = a.size(0);
  if (outer < 1 || outer > 65535) return false;
  long total_vec = 0;
  for (const at::Tensor& t : xs) {
    if (t.scalar_type() != a.scalar_type() || t.device() != a.device()) return false;
    if (!t.is_contiguous() || t.dim() != a.dim() || t.size(0) != outer) return false;
    for (int64_t d = 2; d < a.dim(); ++d)
      if (t.size(d) != a.size(d)) return false;
    const long slab_bytes = (t.numel() / outer) * t.element_size();
    if (slab_bytes % 16 != 0) return false;
    if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 != 0) return false;
    total_vec += slab_bytes / 16;
  }
  return total_vec < (1L << 30);
}

at::Tensor cat_dim1(std::vector<at::Tensor> xs) {
  const int k = (int)xs.size();
  if (k < 2 || k > 4 || !usable(xs)) return at::cat(xs, 1);

  auto sizes = xs[0].sizes().vec();
  int64_t c = 0;
  for (const at::Tensor& t : xs) c += t.size(1);
  sizes[1] = c;
  at::Tensor out = at::empty(sizes, xs[0].options());
  const long outer = sizes[0];
  switch (k) {
    case 2: launch<2>(xs, out, outer); break;
    case 3: launch<3>(xs, out, outer); break;
    default: launch<4>(xs, out, outer); break;
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("cat_dim1", &cat_dim1); }
"""

_ext = None
_ext_failed = False


def _load_ext():
    """JIT-build the concat kernel once, pinned to the local arch."""
    global _ext, _ext_failed
    if _ext is not None or _ext_failed:
        return _ext
    try:
        import os

        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}"
        if major in (9, 10, 12):
            arch += "a"
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        _ext = load_inline(
            name="yolov10_concat_cat_dim1",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:  # pragma: no cover - fall back to ATen if nvcc is unhappy
        _ext_failed = True
        _ext = None
    return _ext


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if self.d == 1 and len(xs) > 1 and xs[0].is_cuda:
            ext = _load_ext()
            if ext is not None:
                return ext.cat_dim1(xs)
        return torch.cat(xs, self.d)
