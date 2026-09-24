"""Embedding lookup kernel: a hand-written CUDA row-gather.

``nn.Embedding`` forwards to ``aten::index_select``, which is fine for wide rows
but collapses for short ones -- on a 64-wide table it runs at ~0.15 TB/s, ~15x
off what the output write alone should cost. This replaces it with a single
flat-tiled gather kernel:

* the row is copied as raw bytes, so one kernel serves every dtype -- the
  vector width comes from the row's byte length / pointer alignment, not from
  ``weight.dtype``;
* work is tiled over the *flat* output (``n_ids * vectors_per_row``) rather than
  per row, so a 1-token lookup and a 262144-token lookup both get a balanced,
  fully coalesced grid;
* ``(row, column)`` comes back from the flat index via a 32-bit magic-number
  division, and each thread handles ``CPT`` independent vectors so the
  ``ids -> weight`` dependent-load chains overlap.

``self.emb`` is kept as the parameter holder so this module's ``state_dict``
keys still match the baseline's, and anything the kernel cannot serve (CPU
tensors, non-int64 ids, a non-contiguous weight) falls back to ``aten``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// 32-bit magic division: q = (__umulhi(mul, n) >> shr) == n / d for n < 2^31.
// ---------------------------------------------------------------------------
struct FastDiv { unsigned mul, shr; };

static inline FastDiv make_fastdiv(unsigned d) {  // d >= 2
  FastDiv f;
  unsigned e = 32u - (unsigned)__builtin_clz(d - 1u);   // ceil(log2(d))
  unsigned p = 31u + e;
  f.mul = (unsigned)(((1ull << p) + (unsigned long long)d - 1ull) / d);
  f.shr = e - 1u;
  return f;
}

// ---------------------------------------------------------------------------
// Flat-tiled gather. Block b owns output vectors [b*BX*CPT, (b+1)*BX*CPT);
// thread t inside it takes the CPT vectors t, t+BX, ... so every store is a
// contiguous BX*sizeof(V) run per warp-step.
// ---------------------------------------------------------------------------
template <typename V, int BX, int CPT>
__global__ __launch_bounds__(BX) void gather_flat(
    const V* __restrict__ w, const int64_t* __restrict__ ids, V* __restrict__ out,
    unsigned vpr, unsigned mul, unsigned shr, unsigned wrows, unsigned total) {
  const unsigned base = blockIdx.x * (unsigned)(BX * CPT) + threadIdx.x;
  if (base + (unsigned)((CPT - 1) * BX) < total) {
    unsigned row[CPT];
    V v[CPT];
#pragma unroll
    for (int k = 0; k < CPT; ++k)
      row[k] = __umulhi(mul, base + (unsigned)(k * BX)) >> shr;
    // ids[] first, so the CPT dependent weight loads can all be in flight.
    int64_t id[CPT];
#pragma unroll
    for (int k = 0; k < CPT; ++k) id[k] = ids[row[k]];
#pragma unroll
    for (int k = 0; k < CPT; ++k) {
      unsigned i = base + (unsigned)(k * BX);
      // Clamp: an out-of-range id is a caller bug (nn.Embedding raises on the
      // device) but must not turn into an illegal access here.
      int64_t r = (id[k] < 0 || id[k] >= (int64_t)wrows) ? 0 : id[k];
      v[k] = w[r * (int64_t)vpr + (i - row[k] * vpr)];
    }
#pragma unroll
    for (int k = 0; k < CPT; ++k) out[base + (unsigned)(k * BX)] = v[k];
  } else {
#pragma unroll
    for (int k = 0; k < CPT; ++k) {
      unsigned i = base + (unsigned)(k * BX);
      if (i < total) {
        unsigned r = __umulhi(mul, i) >> shr;
        int64_t id = ids[r];
        if (id < 0 || id >= (int64_t)wrows) id = 0;
        out[i] = w[id * (int64_t)vpr + (i - r * vpr)];
      }
    }
  }
}

// Grid-stride fallback: rows of one vector each, or > 2^31 output vectors.
template <typename V>
__global__ void gather_stride(const V* __restrict__ w, const int64_t* __restrict__ ids,
                              V* __restrict__ out, int64_t vpr, int64_t wrows,
                              int64_t total) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x) {
    int64_t r = i / vpr;
    int64_t id = ids[r];
    if (id < 0 || id >= wrows) id = 0;
    out[i] = w[id * vpr + (i - r * vpr)];
  }
}

template <typename V, int BX>
static void launch_cpt(const V* w, const int64_t* ids, V* out, unsigned vpr,
                       unsigned wrows, unsigned total, cudaStream_t s) {
  const FastDiv f = make_fastdiv(vpr);
  // ~1024 blocks is enough to fill the device; past that give each thread more
  // vectors instead of adding blocks.
  const unsigned units = (total + (unsigned)BX - 1u) / (unsigned)BX;
  const int cpt = units > 4096u ? 8 : (units > 2048u ? 4 : (units > 1024u ? 2 : 1));
  const unsigned tile = (unsigned)BX * (unsigned)cpt;
  const unsigned grid = (total + tile - 1u) / tile;
#define FK_LAUNCH(C) \
  gather_flat<V, BX, C><<<grid, BX, 0, s>>>(w, ids, out, vpr, f.mul, f.shr, wrows, total)
  switch (cpt) {
    case 8: FK_LAUNCH(8); break;
    case 4: FK_LAUNCH(4); break;
    case 2: FK_LAUNCH(2); break;
    default: FK_LAUNCH(1); break;
  }
#undef FK_LAUNCH
}

template <typename V>
static void launch(const void* wv, const int64_t* ids, void* ov, int64_t vpr,
                   int64_t nrows, int64_t wrows, cudaStream_t s) {
  const V* w = (const V*)wv;
  V* out = (V*)ov;
  const int64_t total = nrows * vpr;
  if (vpr == 1 || total >= (int64_t)1 << 31) {
    const int blocks = (int)std::min<int64_t>((total + 255) / 256, 16384);
    gather_stride<V><<<blocks, 256, 0, s>>>(w, ids, out, vpr, wrows, total);
  } else if (total <= 2048) {
    launch_cpt<V, 64>(w, ids, out, (unsigned)vpr, (unsigned)wrows, (unsigned)total, s);
  } else {
    launch_cpt<V, 256>(w, ids, out, (unsigned)vpr, (unsigned)wrows, (unsigned)total, s);
  }
}

torch::Tensor embed(const torch::Tensor& w, const torch::Tensor& ids) {
  const int64_t D = w.size(1);
  const int64_t row_bytes = D * w.element_size();

  std::vector<int64_t> osz;
  osz.reserve(ids.dim() + 1);
  for (int64_t i = 0; i < ids.dim(); ++i) osz.push_back(ids.size(i));
  osz.push_back(D);
  torch::Tensor out = at::empty(osz, w.options());

  const int64_t nrows = ids.numel();
  if (nrows == 0 || D == 0) return out;

  const torch::Tensor idx = ids.is_contiguous() ? ids : ids.contiguous();
  const void* wp = w.const_data_ptr();
  void* op = out.data_ptr();
  const int64_t* ip = idx.const_data_ptr<int64_t>();
  const int64_t wrows = w.size(0);
  cudaStream_t s = at::cuda::getCurrentCUDAStream();

  // Widest vector both pointers and the row pitch can carry.
  const uintptr_t a = (uintptr_t)wp | (uintptr_t)op | (uintptr_t)row_bytes;
  if ((a & 15) == 0)     launch<int4> (wp, ip, op, row_bytes / 16, nrows, wrows, s);
  else if ((a & 7) == 0) launch<int2> (wp, ip, op, row_bytes / 8,  nrows, wrows, s);
  else if ((a & 3) == 0) launch<int>  (wp, ip, op, row_bytes / 4,  nrows, wrows, s);
  else if ((a & 1) == 0) launch<short>(wp, ip, op, row_bytes / 2,  nrows, wrows, s);
  else                   launch<char> (wp, ip, op, row_bytes,      nrows, wrows, s);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("embed", &embed, "embedding row gather");
}
'''


def _build():
    """JIT-compile the gather kernel, pinned to the local GPU arch."""
    from torch.utils.cpp_extension import load_inline
    try:
        from fastkernels.infra.cuda_ext import _pin_build_arch
        _pin_build_arch()
    except Exception:
        try:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    return load_inline(
        name="fk_cand_embedding_gather",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


try:
    _EMBED = _build().embed
except Exception:  # no nvcc / unsupported toolchain -> plain aten path
    _EMBED = None


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)

    def forward(self, input_ids):
        w = self.emb.weight
        if (_EMBED is not None and w.is_cuda and w.dim() == 2 and w.is_contiguous()
                and input_ids.dtype == torch.int64):
            return _EMBED(w, input_ids)
        # padding_idx only affects the backward pass, so plain aten is exact here.
        return torch.embedding(w, input_ids)
