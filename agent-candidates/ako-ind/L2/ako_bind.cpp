// Python binding for the CUTLASS Sm100 FFN GEMMs.  Deliberately does NOT include
// any CUTLASS header, so this translation unit compiles in a couple of seconds.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "ako_knobs.h"

using RunFn = int (*)(const void*, const void*, const void*, void*, int, int, int,
                      void*, cudaStream_t, const ako::Knobs*);
using WsFn = size_t (*)(int, int, int, const ako::Knobs*);

#define AKO_DECL(NAME)                                                            \
  extern "C" int ako_run_##NAME(const void*, const void*, const void*, void*,      \
                               int, int, int, void*, cudaStream_t,                 \
                               const ako::Knobs*);                                 \
  extern "C" size_t ako_ws_##NAME(int, int, int, const ako::Knobs*);

AKO_DECL(g1a)
AKO_DECL(g1b)
AKO_DECL(g2a)
AKO_DECL(g2b)

static RunFn kRun[] = {ako_run_g1a, ako_run_g1b, ako_run_g2a, ako_run_g2b};
static WsFn kWs[] = {ako_ws_g1a, ako_ws_g1b, ako_ws_g2a, ako_ws_g2b};
static constexpr int kNumCfg = 4;

static ako::Knobs mk(int64_t swizzle, int64_t raster, int64_t splits, int64_t decomp) {
  ako::Knobs k;
  k.swizzle = (int)swizzle;
  k.raster = (int)raster;
  k.splits = (int)splits;
  k.decomp = (int)decomp;
  return k;
}

// A [M,K] bf16 row-major, B [N,K] bf16 row-major (a torch Linear weight),
// bias [N] bf16, D [M,N] bf16 row-major.  Caller guarantees contiguity/dtype.
static void gemm(int64_t cfg, at::Tensor A, at::Tensor B, at::Tensor bias, at::Tensor D,
                 at::Tensor ws, int64_t swizzle, int64_t raster, int64_t splits,
                 int64_t decomp) {
  TORCH_CHECK(cfg >= 0 && cfg < kNumCfg, "bad cfg");
  const int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(0);
  ako::Knobs k = mk(swizzle, raster, splits, decomp);
  int st = kRun[cfg](A.data_ptr(), B.data_ptr(), bias.data_ptr(), D.data_ptr(),
                     M, N, K, ws.numel() ? ws.data_ptr() : nullptr,
                     at::cuda::getCurrentCUDAStream(), &k);
  TORCH_CHECK(st == 0, "cutlass gemm failed, status ", st, " cfg ", cfg);
}

static int64_t ws_bytes(int64_t cfg, int64_t M, int64_t N, int64_t K, int64_t swizzle,
                        int64_t raster, int64_t splits, int64_t decomp) {
  TORCH_CHECK(cfg >= 0 && cfg < kNumCfg, "bad cfg");
  ako::Knobs k = mk(swizzle, raster, splits, decomp);
  return (int64_t)kWs[cfg]((int)M, (int)N, (int)K, &k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm, py::arg("cfg"), py::arg("A"), py::arg("B"), py::arg("bias"),
        py::arg("D"), py::arg("ws"), py::arg("swizzle") = 0, py::arg("raster") = 0,
        py::arg("splits") = 1, py::arg("decomp") = 0);
  m.def("ws_bytes", &ws_bytes, py::arg("cfg"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("swizzle") = 0, py::arg("raster") = 0, py::arg("splits") = 1,
        py::arg("decomp") = 0);
  m.def("num_cfg", []() { return kNumCfg; });
}
