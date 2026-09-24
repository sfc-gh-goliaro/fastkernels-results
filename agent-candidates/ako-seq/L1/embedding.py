"""Embedding lookup: hand-written vectorized row-gather CUDA kernel.

``nn.Embedding`` forwards to ``aten::index_select``, whose CUDA kernel copies
**one element per thread**.  For a bf16 table that is a 2-byte store per thread
(a 64-byte transaction per warp), which measures ~0.24 TB/s on B200 for the
``[512,512] x dim=64`` capture.  The kernel below copies whole rows with the
widest vector access the row pitch allows (16 B / ``uint4`` for every captured
``embedding_dim``), so a warp issues full-cache-line stores instead.

Layout of one launch:

* ``TPR`` (threads-per-row) threads cooperate on one row, ``RPB = 256 / TPR``
  rows per block, so each group of ``TPR`` consecutive lanes writes ``TPR * 16``
  contiguous output bytes -- one 512 B store per warp when ``TPR >= 32``, and
  32/``TPR`` whole-cache-line stores below that.  No division anywhere.
* ``TPR`` is the largest power of two that divides the per-row vector count
  (capped at 256), so the column loop has no ragged tail.
* ``gridDim.x`` walks rows (grid-stride, capped so a fat gather stays a few
  waves); ``gridDim.y`` splits columns, which is what fills the SMs when the
  row count alone is smaller than the machine.

``padding_idx`` needs no runtime branch: ``nn.Embedding`` only zeroes that row
at construction time and masks its *gradient*; the forward result is a plain
row lookup.
"""

import hashlib
import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/SmallVector.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>

namespace {

// TPR = 1 << LOG_TPR threads per row, RPB = 256 / TPR rows per block.
template <typename VT, typename IT, int LOG_TPR>
__global__ __launch_bounds__(256) void fk_gather(
    const VT* __restrict__ w, const IT* __restrict__ idx, VT* __restrict__ out,
    int nvec, long long nrows, long long wrowvec) {
  constexpr int TPR = 1 << LOG_TPR;
  constexpr int RPB = 256 / TPR;
  const int r_in = (int)(threadIdx.x >> LOG_TPR);
  const int c = (int)(threadIdx.x & (TPR - 1));
  const int j0 = c + (int)blockIdx.y * TPR;
  const int jstep = TPR * (int)gridDim.y;
  const long long rstep = (long long)RPB * gridDim.x;
  for (long long row = (long long)blockIdx.x * RPB + r_in; row < nrows;
       row += rstep) {
    const VT* __restrict__ src = w + (long long)idx[row] * wrowvec;
    VT* __restrict__ dst = out + row * (long long)nvec;
    for (int j = j0; j < nvec; j += jstep) dst[j] = src[j];
  }
}

#define FK_LAUNCH(VT, IT, LOG)                                            \
  fk_gather<VT, IT, LOG><<<grid, 256, 0, stream>>>(                       \
      (const VT*)wp, (const IT*)ip, (VT*)op, nvec, nrows, wrowvec)

// Full TPR ladder -- only the 16 B path needs it (every real embedding_dim
// has a 16 B-aligned row pitch).
template <typename VT, typename IT>
void run_all(int log_tpr, dim3 grid, cudaStream_t stream, const void* wp,
             const void* ip, void* op, int nvec, long long nrows,
             long long wrowvec) {
  switch (log_tpr) {
    case 0: FK_LAUNCH(VT, IT, 0); break;
    case 1: FK_LAUNCH(VT, IT, 1); break;
    case 2: FK_LAUNCH(VT, IT, 2); break;
    case 3: FK_LAUNCH(VT, IT, 3); break;
    case 4: FK_LAUNCH(VT, IT, 4); break;
    case 5: FK_LAUNCH(VT, IT, 5); break;
    case 6: FK_LAUNCH(VT, IT, 6); break;
    case 7: FK_LAUNCH(VT, IT, 7); break;
    default: FK_LAUNCH(VT, IT, 8); break;
  }
}

// Narrow-vector fallback for an exotic row pitch: TPR = 256 handles any nvec.
template <typename VT, typename IT>
void run_wide(dim3 grid, cudaStream_t stream, const void* wp, const void* ip,
              void* op, int nvec, long long nrows, long long wrowvec) {
  FK_LAUNCH(VT, IT, 8);
}

#undef FK_LAUNCH

// Block budget for the row grid-stride loop.  Swept on B200 (148 SMs) over
// {1,2,4,8,32} waves x {128,256,512,1024} threads: 8 waves at 256 threads wins
// (fewer waves starves the 33 MB gather, more waves adds CTA-launch cost, and
// 512+ threads costs a quantum on the [1000]x4096 capture).
constexpr int kWaves = 8;

int block_cap() {
  static int cap = [] {
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 148;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms * kWaves;
  }();
  return cap;
}

}  // namespace

at::Tensor fk_embedding(const at::Tensor& weight, const at::Tensor& index) {
  TORCH_CHECK(weight.dim() == 2, "embedding weight must be 2-D");
  TORCH_CHECK(weight.is_cuda() && index.is_cuda(), "cuda tensors required");
  const auto itype = index.scalar_type();
  TORCH_CHECK(itype == at::kLong || itype == at::kInt,
              "index must be int64 or int32");

  const c10::cuda::CUDAGuard guard(weight.device());
  const int64_t D = weight.size(1);

  c10::SmallVector<int64_t, 6> osize;
  for (auto s : index.sizes()) osize.push_back(s);
  osize.push_back(D);
  at::Tensor out = at::empty(osize, weight.options());

  const int64_t nrows = index.numel();
  if (nrows == 0 || D == 0) return out;

  const at::Tensor idx = index.is_contiguous() ? index : index.contiguous();
  const at::Tensor w =
      weight.stride(1) == 1 ? weight : weight.contiguous();

  const int64_t esz = w.element_size();
  const int64_t rowbytes = D * esz;
  const int64_t stridebytes = w.stride(0) * esz;
  const void* wp = w.const_data_ptr();
  const void* ip = idx.const_data_ptr();
  void* op = out.data_ptr();

  // Widest vector access that both row pitches and both base pointers allow.
  // (Caching-allocator blocks are 512 B-aligned, so this always lands on 16 B
  // for a real table -- the reduction is only a guard against a fed-in view.)
  const auto bases = reinterpret_cast<uintptr_t>(wp) |
                     reinterpret_cast<uintptr_t>(op);
  int vb = 16;
  while (vb > 1 &&
         (rowbytes % vb || stridebytes % vb || bases % (uintptr_t)vb))
    vb >>= 1;

  const int64_t nvec64 = rowbytes / vb;
  TORCH_CHECK(nvec64 <= (int64_t)INT_MAX, "row too wide");
  const int nvec = (int)nvec64;
  const long long wrowvec = (long long)(stridebytes / vb);

  // Largest power of two dividing nvec (capped at 256) -> no ragged tail.
  int tpr = (int)std::min<int64_t>(nvec64 & (-nvec64), 256);
  if (tpr < 8) {  // pathological pitch: accept a tail, keep lanes coalesced
    tpr = 1;
    while (tpr < 256 && tpr < nvec) tpr <<= 1;
  }
  int log_tpr = 0;
  while ((1 << log_tpr) < tpr) ++log_tpr;

  const int rpb = 256 >> log_tpr;
  const int cap = block_cap();
  const int64_t rowblocks = (nrows + rpb - 1) / rpb;
  const int gx = (int)std::min<int64_t>(rowblocks, (int64_t)cap);
  int gy = 1;
  if (gx < cap) {
    const int colblocks = (nvec + tpr - 1) / tpr;
    gy = std::max(1, std::min(colblocks, cap / gx));
  }
  const dim3 grid(gx, gy, 1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (vb == 16) {
    if (itype == at::kLong)
      run_all<uint4, int64_t>(log_tpr, grid, stream, wp, ip, op, nvec, nrows,
                              wrowvec);
    else
      run_all<uint4, int32_t>(log_tpr, grid, stream, wp, ip, op, nvec, nrows,
                              wrowvec);
    return out;
  }
  const dim3 g1(gx, 1, 1);
#define FK_NARROW(VT)                                                     \
  if (itype == at::kLong)                                                 \
    run_wide<VT, int64_t>(g1, stream, wp, ip, op, nvec, nrows, wrowvec);   \
  else                                                                    \
    run_wide<VT, int32_t>(g1, stream, wp, ip, op, nvec, nrows, wrowvec);
  if (vb == 8) {
    FK_NARROW(uint2)
  } else if (vb == 4) {
    FK_NARROW(unsigned int)
  } else if (vb == 2) {
    FK_NARROW(unsigned short)
  } else {
    FK_NARROW(unsigned char)
  }
#undef FK_NARROW
  return out;
}
"""

_DECL = "#include <torch/extension.h>\nat::Tensor fk_embedding(const at::Tensor&, const at::Tensor&);\n"

_NAME = "fk_emb_gather_" + hashlib.sha1(_CUDA.encode()).hexdigest()[:12]

# Only build for the GPU actually present: the default arch list is 7 targets
# (~90 s of nvcc) versus ~12 s for one, and this compiles inside the bench
# worker on a cold cache.
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    try:
        _cc = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"
    except Exception:  # noqa: BLE001 - fall back to torch's default list
        pass

_EXT = load_inline(
    name=_NAME,
    cpp_sources=_DECL,
    cuda_sources=_CUDA,
    functions=["fk_embedding"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],  # no float math in the kernel; it is a byte copy
    verbose=False,
)

_GATHER = _EXT.fk_embedding


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)
        # Cache the weight in __dict__ (bypassing nn.Module.__setattr__, so it
        # never becomes a second state_dict key) to keep forward at one plain
        # attribute lookup. ``.to()`` / ``.cuda()`` / ``.half()`` replace the
        # Parameter *object*, so the alias is refreshed from _apply and from
        # load_state_dict rather than trusted for the module's lifetime.
        self.__dict__["_w"] = self.emb.weight

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self.__dict__["_w"] = self.emb.weight
        return out

    def load_state_dict(self, *args, **kwargs):
        out = super().load_state_dict(*args, **kwargs)
        self.__dict__["_w"] = self.emb.weight
        return out

    def forward(self, input_ids):
        return _GATHER(self._w, input_ids)
