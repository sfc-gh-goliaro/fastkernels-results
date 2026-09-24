// PyTorch binding for the fused temporal axial attention kernel.
//
// The kernel itself lives in `_temporal_axial_fused.cuh`, free of framework
// headers, so the profiling harness under `profile/` compiles the identical source.
// This file is argument validation, output allocation, the
// at::Half/at::BFloat16 -> __half/__nv_bfloat16 reinterpretation at the boundary,
// and the two orchestration entry points that were measured against each other.
//
// The checks here duplicate the module's own eligibility predicate on purpose. The
// Python gate exists so an unclaimed input is *routed away* cheaply; these checks
// exist so a claimed input that is nevertheless malformed fails loudly instead of
// reading out of bounds.

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "_temporal_axial_fused.cuh"

namespace {

// Element strides of the (batch, time, spatial) axes of a
// (bsz, T, height, width, channels) activation. Height and width fold into one
// spatial axis, which is exact for the contiguous layout this entry requires:
// h * width_stride + w * w_stride == (h * width + w) * w_stride.
temporal_axial::Layout layout_of(const at::Tensor& t) {
  return {t.stride(0), t.stride(1), t.stride(3)};
}

template <typename Torch, typename Cuda>
bool run(const at::Tensor& qkv, const at::Tensor& cos_sin, const at::Tensor& out,
         int64_t heads, double scale, bool causal, int64_t warps_per_block,
         cudaStream_t stream) {
  const int64_t dim = out.size(4);
  return temporal_axial::launch<Cuda>(
      reinterpret_cast<const Cuda*>(qkv.const_data_ptr<Torch>()),
      reinterpret_cast<const Cuda*>(cos_sin.const_data_ptr<Torch>()),
      reinterpret_cast<Cuda*>(out.mutable_data_ptr<Torch>()),
      static_cast<int>(qkv.size(0)), static_cast<int>(qkv.size(1)),
      static_cast<int>(qkv.size(2) * qkv.size(3)), static_cast<int>(heads),
      layout_of(qkv), layout_of(out),
      {0, dim, 2 * dim},
      static_cast<float>(scale), causal, static_cast<int>(warps_per_block), stream);
}

}  // namespace

// qkv: (bsz, T, height, width, 3 * dim) -- the fused projection output, q/k/v in
//      that column order, heads major inside each block.
// cos_sin: (T, head_dim / 2, 2) packed (cos theta_f, sin theta_f) per (t, pair).
// Returns (bsz, T, height, width, dim), contiguous.
at::Tensor temporal_axial_fused(const at::Tensor& qkv, const at::Tensor& cos_sin,
                                int64_t heads, bool causal, double scale) {
  TORCH_CHECK(qkv.dim() == 5, "temporal_axial_fused: qkv must be 5-D "
              "(bsz, time, height, width, 3*dim), got ", qkv.dim(), "-D");
  TORCH_CHECK(cos_sin.dim() == 3, "temporal_axial_fused: cos_sin must be 3-D "
              "(time, head_dim/2, 2), got ", cos_sin.dim(), "-D");
  TORCH_CHECK(qkv.is_cuda() && cos_sin.is_cuda(),
              "temporal_axial_fused: inputs must be CUDA tensors");
  TORCH_CHECK(cos_sin.device() == qkv.device(),
              "temporal_axial_fused: qkv and cos_sin must be on one device");
  TORCH_CHECK(qkv.scalar_type() == cos_sin.scalar_type(),
              "temporal_axial_fused: qkv and cos_sin dtypes must match, got ",
              qkv.scalar_type(), " and ", cos_sin.scalar_type());
  // The spatial fold and the row-stride arithmetic are only valid for the layout the
  // projection that feeds this produces, and they are checked rather than assumed.
  //
  // is_contiguous() alone is not the check. It ignores the stride of any size-1 axis,
  // so a 5-D tensor with width 1 and an arbitrary stride(3) reports contiguous while
  // the fold `h * stride(2) + w * stride(3) == (h * width + w) * stride(3)` does not
  // hold for it. The three stride identities below are what the fold actually needs.
  TORCH_CHECK(qkv.is_contiguous() && cos_sin.is_contiguous(),
              "temporal_axial_fused: inputs must be contiguous");
  TORCH_CHECK(qkv.stride(4) == 1,
              "temporal_axial_fused: the channel axis must have unit stride, got ",
              qkv.stride(4));
  TORCH_CHECK(qkv.stride(3) == qkv.size(4),
              "temporal_axial_fused: width rows must be packed, got stride ",
              qkv.stride(3), " for ", qkv.size(4), " channels");
  TORCH_CHECK(qkv.stride(2) == qkv.size(3) * qkv.stride(3),
              "temporal_axial_fused: height and width must fold into one axis, got "
              "stride ", qkv.stride(2), " against ", qkv.size(3), " x ", qkv.stride(3));

  const int64_t bsz = qkv.size(0);
  const int64_t seq = qkv.size(1);
  const int64_t height = qkv.size(2);
  const int64_t width = qkv.size(3);
  const int64_t qkv_dim = qkv.size(4);

  TORCH_CHECK(heads > 0, "temporal_axial_fused: heads must be positive, got ", heads);
  TORCH_CHECK(qkv_dim % 3 == 0,
              "temporal_axial_fused: last qkv axis must be 3*dim, got ", qkv_dim);
  const int64_t dim = qkv_dim / 3;
  TORCH_CHECK(dim == heads * temporal_axial::kHeadDim,
              "temporal_axial_fused: needs head_dim ", temporal_axial::kHeadDim,
              ", got dim ", dim, " over ", heads, " heads");
  TORCH_CHECK(cos_sin.size(0) == seq && cos_sin.size(1) == temporal_axial::kPairsPerRow
                  && cos_sin.size(2) == 2,
              "temporal_axial_fused: cos_sin must be (", seq, ", ",
              temporal_axial::kPairsPerRow, ", 2), got (", cos_sin.size(0), ", ",
              cos_sin.size(1), ", ", cos_sin.size(2), ")");
  TORCH_CHECK(bsz > 0 && seq > 0 && height > 0 && width > 0,
              "temporal_axial_fused: every extent must be positive, got (", bsz, ", ",
              seq, ", ", height, ", ", width, ")");
  TORCH_CHECK(seq <= temporal_axial::kMaxSeq,
              "temporal_axial_fused: temporal extent ", seq,
              " exceeds the compiled window of ", temporal_axial::kMaxSeq);
  // Packed-pair loads and stores need a 4-byte base. data_ptr folds in the storage
  // offset; a contiguous 5-D tensor with an even head_dim then has every row
  // aligned, but the base itself is a property of the allocation.
  TORCH_CHECK(qkv.data_ptr() != nullptr && cos_sin.data_ptr() != nullptr,
              "temporal_axial_fused: null input storage");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(qkv.data_ptr()) & 3u) == 0
                  && (reinterpret_cast<uintptr_t>(cos_sin.data_ptr()) & 3u) == 0,
              "temporal_axial_fused: inputs must be 4-byte aligned");
  // Index arithmetic inside the kernel is 64-bit, but the pair index and the grid
  // are int, so the problem count has to fit.
  // The kernel takes the problem count as an int and rounds it up to a block count,
  // so the bound leaves room for that rounding rather than stopping at INT32_MAX.
  const int64_t pairs = bsz * height * width * heads;
  TORCH_CHECK(pairs > 0 && pairs <= static_cast<int64_t>(INT32_MAX)
                               - temporal_axial::kWarpsPerBlock,
              "temporal_axial_fused: ", pairs, " (batch, spatial, head) problems "
              "exceeds the addressable grid");

  const at::cuda::CUDAGuard guard(qkv.device());
  at::Tensor out = at::empty({bsz, seq, height, width, dim}, qkv.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  bool launched = false;
  if (qkv.scalar_type() == at::kHalf) {
    launched = run<at::Half, __half>(qkv, cos_sin, out, heads, scale, causal,
                                     temporal_axial::kWarpsPerBlock, stream);
  } else if (qkv.scalar_type() == at::kBFloat16) {
    launched = run<at::BFloat16, __nv_bfloat16>(qkv, cos_sin, out, heads, scale,
                                                causal,
                                                temporal_axial::kWarpsPerBlock,
                                                stream);
  } else {
    TORCH_CHECK(false, "temporal_axial_fused: dtype must be float16 or bfloat16, got ",
                qkv.scalar_type());
  }
  TORCH_CHECK(launched, "temporal_axial_fused: unsupported problem geometry "
              "(seq=", seq, ", bsz=", bsz, ", spatial=", height * width,
              ", heads=", heads, ")");
  // Surface an invalid launch configuration here rather than letting it resurface
  // as an unrelated error at the next synchronisation point.
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

// Single host entry for the whole operator: projection, fused kernel, projection.
// Exists because the graded window is a CUDA-event pair around a host-side call, so
// host enqueue gaps are inside the measurement whenever the GPU starves. Which of
// this and the Python-side orchestration ships is decided by measurement, not by
// argument; see profile/04_orchestration_ab.py.
at::Tensor temporal_axial_forward(const at::Tensor& x,
                                  const at::Tensor& qkv_weight,
                                  const at::Tensor& out_weight,
                                  const c10::optional<at::Tensor>& out_bias,
                                  const at::Tensor& cos_sin,
                                  int64_t heads, bool causal, double scale) {
  TORCH_CHECK(x.dim() == 5, "temporal_axial_forward: x must be 5-D, got ",
              x.dim(), "-D");
  TORCH_CHECK(x.is_cuda(), "temporal_axial_forward: x must be a CUDA tensor");
  const at::cuda::CUDAGuard guard(x.device());
  at::Tensor qkv = at::linear(x, qkv_weight);
  at::Tensor o = temporal_axial_fused(qkv, cos_sin, heads, causal, scale);
  return at::linear(o, out_weight, out_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("temporal_axial_fused", &temporal_axial_fused,
        "Rotary rotation, causal attention over the temporal axis, and both layout "
        "shuffles, one warp per (batch, spatial, head) problem.",
        pybind11::arg("qkv"), pybind11::arg("cos_sin"), pybind11::arg("heads"),
        pybind11::arg("causal"), pybind11::arg("scale"));
  m.def("temporal_axial_forward", &temporal_axial_forward,
        "Whole operator from one host call: qkv projection, fused kernel, output "
        "projection.",
        pybind11::arg("x"), pybind11::arg("qkv_weight"), pybind11::arg("out_weight"),
        pybind11::arg("out_bias"), pybind11::arg("cos_sin"), pybind11::arg("heads"),
        pybind11::arg("causal"), pybind11::arg("scale"));
}
