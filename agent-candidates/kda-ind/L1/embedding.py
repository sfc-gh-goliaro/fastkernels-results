"""Embedding lookup kernel: a custom CUDA row gather.

Forward is exactly ``out[i, :] = weight[input_ids[i], :]``. ``padding_idx`` changes only
the initial weight (that row is zeroed) and the backward pass, so it plays no part here.

The profile in ``profile/embedding_baseline_v0/`` shows where the time goes. ATen's
``vectorized_gather_kernel`` spends one 32-thread block per output row, which for a
128-byte row means 262144 single-warp blocks, a theoretical occupancy capped at 50 % by
the SM's block-slot limit, 10.3 % achieved occupancy and 0.13 eligible warps per cycle
per scheduler. It is latency-bound at low occupancy, not arithmetic-bound and not
bandwidth-bound: ALU sits at 8 %, stores are already perfectly coalesced at 4
sectors/request, and DRAM runs at 0.12 % of peak. The fix is therefore granularity --
pack several rows into one block so that the block count drops and the SM fills up.
For a wide row ATen already does the right thing, so there the kernel below aims at
parity rather than at a win.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <vector>

namespace {

// A row gather does not care what the elements mean, so the copy is expressed in bytes
// and one kernel serves bf16, fp16 and fp32 alike. That removes a dispatch over scalar
// types from the host path.
//
// Rows go on grid.x and the chunks within a row on grid.y, which is what makes the
// indexing division-free: neither coordinate has to be recovered from a flattened one.
// A warp covers 32/BLOCK_X consecutive output rows times BLOCK_X chunks, so its
// stores are contiguous.
template <typename VecT, int BLOCK_X, typename IdxT>
__global__ void gather_rows_kernel(const VecT* __restrict__ src,
                                   VecT* __restrict__ dst,
                                   const IdxT* __restrict__ idx,
                                   long long n_rows,
                                   int chunks_per_row,
                                   long long n_src_rows) {
  static_assert(BLOCK_X > 0 && (BLOCK_X & (BLOCK_X - 1)) == 0 && BLOCK_X <= 256,
                "BLOCK_X must be a power of two no larger than the 256-thread block");
  constexpr int BLOCK_Y = 256 / BLOCK_X;
  const long long row = static_cast<long long>(blockIdx.x) * BLOCK_Y + threadIdx.y;
  const int chunk = static_cast<int>(blockIdx.y) * BLOCK_X + threadIdx.x;

  // One global index load and one range-validation sequence per row per column tile, at every
  // BLOCK_X: lane 0 of a row reads its index, validates it, and publishes the verdict for the
  // rest of the row through shared memory. A negative published value is the "do not touch
  // memory" signal, so a single shared word carries both the index and its verdict and each
  // consumer does one shared read.
  //
  // An out-of-range index is *prevented* from reaching memory rather than reported after the
  // fact: only the valid branch below reaches the weight load, so no lane of a bad row ever
  // computes a weight address, and the leader raises the error once past the barrier. Checking
  // in each consumer instead -- as an earlier revision did -- leaves every non-leader lane of a
  // bad row holding the bad index on a path that reaches the load, so the illegal access could
  // issue and correctness would rest on the trap winning that race.
  //
  // The handoff costs 13.6 % on [512, 512] against checking per warp, which is paid knowingly;
  // profile/embedding_gather_v3/REPORT.md has the six-variant study and the counters showing
  // the cost is the whole leader/publish/control sequence rather than the shared read alone.
  __shared__ long long s_row[BLOCK_Y];
  if (threadIdx.x == 0) {
    long long r = -1;
    // The guard is only so the leader does not read idx out of bounds; its -1 is never
    // observed, because a consumer of this slot has the same row and has already returned.
    if (row < n_rows) {
      r = static_cast<long long>(idx[row]);
      if (r < 0 || r >= n_src_rows) {
        r = -1;
      }
    }
    s_row[threadIdx.y] = r;
  }
  // Reached by every thread before any boundary return, so no lane can arrive at a barrier
  // its peers have already left. A row's lanes cannot cross a warp while BLOCK_X divides 32,
  // so the cheaper warp barrier is correct there; of the instantiated widths only BLOCK_X=256
  // spreads a row across the block. The block barrier would also be correct for a width that
  // exceeded 32 without covering a whole block, which none of the three instantiations does.
  if constexpr (BLOCK_X <= 32) {
    __syncwarp();
  } else {
    __syncthreads();
  }

  if (row >= n_rows || chunk >= chunks_per_row) {
    return;
  }
  const long long r = s_row[threadIdx.y];
  // The name is the assert's message text, and tools/test_gather.py matches on it.
  const bool row_index_in_range = r >= 0;
  if (!row_index_in_range) {
    if (threadIdx.x == 0) {
      CUDA_KERNEL_ASSERT(row_index_in_range);
    }
    return;
  }
  dst[row * chunks_per_row + chunk] =
      __ldg(src + r * static_cast<long long>(chunks_per_row) + chunk);
}

constexpr int kMaxGridY = 65535;

bool aligned_to(const void* p, size_t width) {
  return (reinterpret_cast<uintptr_t>(p) % width) == 0;
}

template <typename VecT, int BLOCK_X, typename IdxT>
void launch(const at::Tensor& weight, const at::Tensor& idx, at::Tensor& out,
            long long n_rows, int chunks_per_row, dim3 grid) {
  constexpr int BLOCK_Y = 256 / BLOCK_X;
  gather_rows_kernel<VecT, BLOCK_X, IdxT>
      <<<grid, dim3(BLOCK_X, BLOCK_Y), 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const VecT*>(weight.const_data_ptr()),
          reinterpret_cast<VecT*>(out.data_ptr()),
          idx.const_data_ptr<IdxT>(),
          n_rows, chunks_per_row, weight.size(0));
}

// BLOCK_X is chosen so a warp's stores stay contiguous and few lanes idle: 8 chunks per
// row puts four rows in a warp, while a wide row wants the whole block on one row. Three
// instantiations cover every shape without a per-case table.
template <typename VecT, typename IdxT>
bool launch_by_width(const at::Tensor& weight, const at::Tensor& idx, at::Tensor& out,
                     long long n_rows, int chunks_per_row) {
  const int block_x = chunks_per_row <= 8 ? 8 : (chunks_per_row <= 32 ? 32 : 256);
  const int block_y = 256 / block_x;
  const long long grid_x = (n_rows + block_y - 1) / block_y;
  const long long grid_y = (chunks_per_row + block_x - 1) / block_x;
  if (grid_y > kMaxGridY || grid_x > 2147483647LL) {
    return false;
  }
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));
  switch (block_x) {
    case 8:
      launch<VecT, 8, IdxT>(weight, idx, out, n_rows, chunks_per_row, grid);
      break;
    case 32:
      launch<VecT, 32, IdxT>(weight, idx, out, n_rows, chunks_per_row, grid);
      break;
    default:
      launch<VecT, 256, IdxT>(weight, idx, out, n_rows, chunks_per_row, grid);
      break;
  }
  return true;
}

template <typename IdxT>
bool launch_by_index(const at::Tensor& weight, const at::Tensor& idx, at::Tensor& out,
                     long long n_rows, long long row_bytes) {
  const void* s = weight.const_data_ptr();
  const void* d = out.data_ptr();
  // Widest access the row size and both base pointers permit. Allocator pointers are
  // 512-byte aligned, so in practice only row_bytes decides -- but checking the pointers
  // costs nothing on the host and removes an assumption.
  if (row_bytes % 16 == 0 && aligned_to(s, 16) && aligned_to(d, 16)) {
    return launch_by_width<uint4, IdxT>(weight, idx, out, n_rows,
                                        static_cast<int>(row_bytes / 16));
  }
  if (row_bytes % 4 == 0 && aligned_to(s, 4) && aligned_to(d, 4)) {
    return launch_by_width<unsigned int, IdxT>(weight, idx, out, n_rows,
                                               static_cast<int>(row_bytes / 4));
  }
  if (row_bytes % 2 == 0 && aligned_to(s, 2) && aligned_to(d, 2)) {
    return launch_by_width<unsigned short, IdxT>(weight, idx, out, n_rows,
                                                 static_cast<int>(row_bytes / 2));
  }
  return launch_by_width<unsigned char, IdxT>(weight, idx, out, n_rows,
                                              static_cast<int>(row_bytes));
}

}  // namespace

at::Tensor emb_gather(const at::Tensor& weight, const at::Tensor& idx) {
  TORCH_CHECK(weight.dim() == 2, "weight must be 2-D");
  TORCH_CHECK(weight.is_cuda() && idx.is_cuda(), "weight and indices must be on CUDA");
  TORCH_CHECK(weight.device() == idx.device(),
              "weight and indices must be on the same device");
  const auto itype = idx.scalar_type();
  TORCH_CHECK(itype == at::kLong || itype == at::kInt,
              "index tensor must be int64 or int32");

  // The benchmark gives each worker one device, but the stream and the allocation below
  // must follow the weight's device rather than whatever happens to be current.
  const c10::cuda::OptionalCUDAGuard device_guard(at::device_of(weight));

  // A non-contiguous input would make the flat row index read the wrong elements, and a
  // non-contiguous weight would break the byte-copy formulation outright. Both are off
  // the fast path for every shape this operator sees, so they hand off to ATen rather
  // than growing a second code path here.
  if (!weight.is_contiguous() || !idx.is_contiguous()) {
    return at::embedding(weight, idx);
  }

  // Output is idx.sizes() followed by embedding_dim; a 0-dim index yields just
  // {embedding_dim}, which is what torch.embedding returns too.
  std::vector<int64_t> sizes(idx.sizes().begin(), idx.sizes().end());
  sizes.push_back(weight.size(1));
  at::Tensor out = at::empty(sizes, weight.options());

  const long long n_rows = idx.numel();
  if (n_rows == 0) {
    return out;  // correctly shaped and empty; launching a zero-sized grid is illegal
  }

  const long long row_bytes = weight.size(1) * weight.element_size();
  if (row_bytes == 0) {
    return out;
  }

  const bool launched = (itype == at::kLong)
      ? launch_by_index<int64_t>(weight, idx, out, n_rows, row_bytes)
      : launch_by_index<int32_t>(weight, idx, out, n_rows, row_bytes);
  if (!launched) {
    // A row wide enough to overflow the grid's y dimension. Documented fallback rather
    // than a silently truncated grid.
    return at::embedding(weight, idx);
  }
  return out;
}
"""

_CPP_SOURCE = r"""
#include <ATen/ATen.h>
at::Tensor emb_gather(const at::Tensor& weight, const at::Tensor& idx);
"""


def _build_gather():
    """Compile the extension once, at import, and return the callable to use.

    A build failure must not take the operator down with it: an exception escaping this
    module is a hard error for the whole run, while falling back to ``torch.embedding``
    is merely slower. The choice is made here, once, so ``forward`` carries neither a
    branch nor an exception handler.
    """
    try:
        from torch.utils.cpp_extension import load_inline

        # Build for the one architecture actually present, passed as a compile flag rather
        # than through the process-wide TORCH_CUDA_ARCH_LIST: torch skips that variable
        # entirely once the cuda flags carry an `arch`, so nothing global is disturbed and a
        # later extension build in the same process is unaffected. The ambient value here
        # names six architectures, which would multiply compile time for no benefit. Plain
        # sm_XX suffices; nothing in the kernel needs the 'a' variant.
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major, minor = 10, 0
        arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"

        # Build beside the workspace instead of in the shared ~/.cache/torch_extensions,
        # where a name collision with another agent's build causes rebuild thrashing. Passed
        # explicitly for the same reason as the arch: no environment mutation. The name is
        # stable because torch already versions the build directory when the sources change.
        name = "fk_l1_embedding_gather"
        build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / name
        build_dir.mkdir(parents=True, exist_ok=True)

        ext = load_inline(
            name=name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["emb_gather"],
            extra_cuda_cflags=["-O3", arch_flag],
            build_directory=str(build_dir),
            verbose=False,
        )
        return ext.emb_gather
    except Exception:
        return torch.embedding


_gather = _build_gather()


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        # The submodule name is load-bearing: the harness shares weights with
        # ``load_state_dict(..., strict=False)`` against a baseline whose key is
        # ``emb.weight``, and a mismatched key is ignored silently rather than raised.
        # Keeping a real nn.Embedding also matches its init distribution and its
        # padding_idx row-zeroing exactly.
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        return _gather(self.emb.weight, input_ids)
