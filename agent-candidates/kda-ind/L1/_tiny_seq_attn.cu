// PyTorch binding for the tiny-sequence attention kernel.
//
// The kernel itself lives in `_tiny_seq_attn.cuh`, free of framework headers, so
// the profiling harness under `profile/` compiles the identical source. This file
// is only argument validation, output allocation and the at::Half/at::BFloat16 ->
// __half/__nv_bfloat16 reinterpretation at the boundary.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "_tiny_seq_attn.cuh"

namespace {

tinyseq::Strides strides_of(const at::Tensor& t) {
  // (batch, seq, head) -- head_dim is the unit-stride axis and needs no entry.
  return {t.stride(0), t.stride(1), t.stride(2)};
}

template <typename Torch, typename Cuda>
bool run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
         const at::Tensor& o, double scale, bool causal, int64_t warps_per_block,
         int64_t head_major, cudaStream_t stream) {
  return tinyseq::launch<Cuda>(
      reinterpret_cast<const Cuda*>(q.const_data_ptr<Torch>()),
      reinterpret_cast<const Cuda*>(k.const_data_ptr<Torch>()),
      reinterpret_cast<const Cuda*>(v.const_data_ptr<Torch>()),
      reinterpret_cast<Cuda*>(o.mutable_data_ptr<Torch>()),
      static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
      static_cast<int>(q.size(2)),
      strides_of(q), strides_of(k), strides_of(v), strides_of(o),
      static_cast<float>(scale), causal, static_cast<int>(warps_per_block),
      static_cast<int>(head_major), stream);
}

}  // namespace

// Returns a (batch, seq, heads, head_dim) tensor backed by
// (batch, heads, seq, head_dim)-contiguous storage -- exactly the strides the
// baseline's own `permute(0, 2, 1, 3)` produces, which also makes a warp's S row
// stores contiguous. The permuted *view* is what reaches the kernel, so all four
// tensors are indexed by (batch, seq, head) in the same order and there is no
// axis-order asymmetry to get wrong.
at::Tensor tiny_seq_attn(const at::Tensor& query, const at::Tensor& key,
                         const at::Tensor& value, double scale, bool causal,
                         int64_t warps_per_block, int64_t head_major) {
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "tiny_seq_attn: expected 4-D (batch, seq, heads, head_dim) inputs");
  TORCH_CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes(),
              "tiny_seq_attn: q/k/v shapes must match");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type(),
              "tiny_seq_attn: q/k/v dtypes must match");
  TORCH_CHECK(query.size(3) == 64, "tiny_seq_attn: head_dim must be 64");
  TORCH_CHECK(query.stride(3) == 1 && key.stride(3) == 1 && value.stride(3) == 1,
              "tiny_seq_attn: head_dim must be the unit-stride axis");
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "tiny_seq_attn: inputs must be CUDA tensors");
  TORCH_CHECK(warps_per_block >= 1 && warps_per_block <= 32,
              "tiny_seq_attn: warps_per_block out of range");

  const at::cuda::CUDAGuard guard(query.device());
  auto out = at::empty({query.size(0), query.size(2), query.size(1), query.size(3)},
                       query.options())
                 .permute({0, 2, 1, 3});
  auto stream = at::cuda::getCurrentCUDAStream();

  bool launched = false;
  if (query.scalar_type() == at::kHalf) {
    launched = run<at::Half, __half>(query, key, value, out, scale, causal,
                                     warps_per_block, head_major, stream);
  } else if (query.scalar_type() == at::kBFloat16) {
    launched = run<at::BFloat16, __nv_bfloat16>(query, key, value, out, scale, causal,
                                                warps_per_block, head_major, stream);
  } else {
    TORCH_CHECK(false, "tiny_seq_attn: dtype must be float16 or bfloat16");
  }
  TORCH_CHECK(launched, "tiny_seq_attn: unsupported problem geometry (seq=",
              query.size(1), ", batch=", query.size(0), ", heads=", query.size(2), ")");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tiny_seq_attn", &tiny_seq_attn,
        "Tiny-sequence dense attention, one warp per (batch, head) pair. Returns "
        "(batch, seq, heads, head_dim) backed by (batch, heads, seq, head_dim) "
        "storage.",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"),
        pybind11::arg("scale"), pybind11::arg("causal"),
        pybind11::arg("warps_per_block"), pybind11::arg("head_major"));
}
