// Graph I/O staging for the fused MSA module stack.
//
// A CUDA graph replays with fixed addresses, so the stack's four inputs have to
// be staged into static buffers and its two outputs copied back out.  Six
// ``copy_`` calls, or two ``_foreach_copy_`` calls, cost more on this workload
// than they move: at 16 tokens the whole payload is 80 KB, so every launch is
// pure overhead (``_foreach_copy_``'s multi-tensor kernel measured 4.6 us a
// side).  Both directions are one launch here instead, over a flat uint4 index
// space spanning all segments -- one 16-byte copy per thread, ~1.3 us.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

#define MAXSEG 4

struct Plan {
  const uint4* src[MAXSEG];
  uint4* dst[MAXSEG];
  int end[MAXSEG];        // exclusive prefix sum of per-segment uint4 counts
  int nseg;
  int total;
};

__global__ void multicopy_kernel(__grid_constant__ const Plan p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.total) return;
  // MAXSEG is 4, so a linear scan is cheaper than any lookup structure and the
  // whole warp takes the same branch except at one segment boundary.
  int s = 0, base = 0;
#pragma unroll
  for (int k = 0; k < MAXSEG - 1; ++k)
    if (k + 1 < p.nseg && i >= p.end[k]) { s = k + 1; base = p.end[k]; }
  p.dst[s][i - base] = p.src[s][i - base];
}

}  // namespace

// Each of ``srcs``/``dsts`` is a same-length list of matching 16-byte-aligned,
// contiguous tensors whose byte count is a multiple of 16.
void multicopy(std::vector<at::Tensor> srcs, std::vector<at::Tensor> dsts) {
  TORCH_CHECK(srcs.size() == dsts.size() && !srcs.empty()
              && srcs.size() <= MAXSEG, "multicopy: bad segment count");
  Plan p;
  p.nseg = (int)srcs.size();
  int acc = 0;
  for (int i = 0; i < p.nseg; ++i) {
    const int64_t bytes = srcs[i].numel() * srcs[i].element_size();
    TORCH_CHECK(bytes == dsts[i].numel() * dsts[i].element_size()
                && bytes % 16 == 0, "multicopy: size mismatch");
    p.src[i] = (const uint4*)srcs[i].data_ptr();
    p.dst[i] = (uint4*)dsts[i].data_ptr();
    acc += (int)(bytes / 16);
    p.end[i] = acc;
  }
  for (int i = p.nseg; i < MAXSEG; ++i) {
    p.src[i] = nullptr; p.dst[i] = nullptr; p.end[i] = acc;
  }
  p.total = acc;
  const int threads = 128;
  multicopy_kernel<<<(acc + threads - 1) / threads, threads, 0,
                     c10::cuda::getCurrentCUDAStream()>>>(p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multicopy", &multicopy);
}
