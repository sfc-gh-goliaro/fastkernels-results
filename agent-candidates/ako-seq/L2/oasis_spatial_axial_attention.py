"""Oasis spatial axial attention -- dispatch-collapsed.

The captured workload is tiny and entirely overhead: bsz=1, time=2..6, a 9x16=144
token grid and dim=1024 -> heads*dim_head = 16*64.  That is two small GEMMs and
2..6 tiny attention problems per call, but the reference module issues **229 ATen
ops and 31 launches** to get there, which costs 371-462 us of CPU per call on
B200 -- a CUDA launch is ~5 us of CPU here and an ATen dispatch ~0.6-1 us, so the
op is bound by dispatch, not by math.  Measured end state: **1 ATen op, 3
launches, 23-24 us of CPU, 20-26 us of device time**, i.e. 0.038-0.042 ms per
call against the reference's 0.39-0.43 ms.

Four things are collapsed, in the order they were worth doing.

1. The axial rotary table is static.  ``get_axial_freqs(9, 16)`` was rebuilt on
   every forward (4 linspaces, 2 einsums, 2 repeat_interleaves, a
   broadcast_tensors and a cat), and ``oasis_apply_rotary_emb`` then recomputed
   ``freqs.cos()``/``freqs.sin()`` twice -- once for q and once for k.  All of
   that is memoized as two ``(seq, dim_head)`` tables.  The builder is still the
   frozen L1 ``OasisRotaryEmbedding.get_axial_freqs`` + ``Tensor.cos/sin``, so
   the tables are bit-identical to the reference's; the cache key carries the
   ``freqs`` parameter's storage identity *and* version counter, so any weight
   reload or in-place edit rebuilds them.

2. Everything between the qkv projection and attention is one kernel.  The
   reference did ``chunk`` + 3 ``permute`` + 2 full rotary chains (reshape,
   unbind, neg, stack, flatten, 2 muls, add, to) + 3 ``reshape``/``transpose``
   pairs, and because the reshapes follow permutes they each materialize a
   contiguous copy -- 9 copy kernels alone.  Instead ``oasis_rope_qkv`` rotates
   the q and k halves of the ``[bsz*time*144, 3*1024]`` projection output
   *in place* and hands back q, k and v as strided views of that one buffer.
   v never moves at all: ``(bsz*time, 144, 16, 64)`` with strides
   ``(144*3072, 3072, 64, 1)`` is exactly the layout L1 ``DenseAttention``
   consumes, and its 16-byte-alignment predicate holds at all three
   1024-element offsets.  One launch, no allocations, no copies.

   Bit-exactness: the reference's rotary runs in the *promoted* dtype of
   ``t * freqs.cos()``.  The benchmark casts every high-precision parameter to
   the input dtype, so ``freqs`` is fp16 too and each of the two muls and the
   add round to fp16 separately.  The kernel reproduces that rounding when the
   promoted type is low precision and keeps a single fp32 round when it is not,
   so both the fp16-freqs and fp32-freqs cases match ATen exactly.

3. The epilogue.  ``DenseAttention`` returns a ``(bsz*time, heads, 144, 64)``
   buffer viewed as ``(bsz*time, 144, heads, 64)``, and the reference's
   ``reshape`` then forced yet another contiguous copy before ``to_out``.  That
   transpose is now ``oasis_attn_merge_heads``, one launch straight into the
   ``(bsz*time*144, 1024)`` matrix the GEMM wants.  The reference's
   ``out.to(q.dtype)`` and its second ``reshape`` were no-ops on the captured
   shapes and are gone.

   Both GEMMs still go through the frozen L1 ``Linear``, but with a 2-D input:
   ``F.linear`` on a 5-D operand walks ``matmul`` -> ``reshape`` -> ``mm`` ->
   ``_unsafe_view``, while a 2-D operand reaches ``mm``/``addmm`` directly.

4. Those five launches are then captured into one CUDA graph per shape.  Even at
   5 launches the op was still dispatch bound (~57 us of CPU against ~20 us of
   device time), and 24 us of that CPU was L1 ``DenseAttention``'s Triton launch,
   which nothing at this level can reach.  A replay costs ~2 us of CPU.

5. The two copies that bracket the replay -- feeding the static input, and
   handing the caller a tensor the next replay will not overwrite -- are nodes
   *inside* the graph, and the endpoint pointer of each is rewritten per call
   with ``cudaGraphExecKernelNodeSetParams``.  Measured on this box, that costs
   0.44-0.54 us against 3.66 us for the ``cudaMemcpyAsync`` it replaces (and
   ``cudaGraphLaunch`` is 4.02 us), so the call went from three driver launches
   to **one** and from 18-21 us of CPU to 9.5-11.6 us.

   The graph is captured through ``torch.cuda.CUDAGraph(keep_graph=True)`` -- so
   PyTorch still owns the private memory pool and the allocator handshake -- and
   the extension then works on its ``raw_cuda_graph()`` / ``raw_cuda_graph_exec()``
   handles.  Endpoint nodes are identified by comparing each kernel node's host
   function pointer against the two endpoint kernels' own addresses, which is
   exact rather than positional.  Everything about that is guarded: a
   zero-or-many node match, a driver that rejects the exec-level param update,
   or a torch without ``raw_cuda_graph`` all fall back to r1's
   copy-in / replay / copy-out (``OASIS_AXIAL_NO_RETARGET=1`` forces it), and a
   caller whose buffer the graph cannot be pointed at -- non-contiguous, or not
   16-byte aligned -- is staged through ATen first.

   The output is still a private allocation per call.  Returning the graph's
   static buffer instead would save a further ~11% (measured in r1) but would
   alias across calls, which is not what a ``forward`` is allowed to do; the
   point of the retarget is to get that win *without* the aliasing.

After all five the op is device bound: ~10 us of CPU against ~26 us of device
time, which is 7 graph nodes that are at or near their floor.  An otherwise-empty
kernel node costs ~1.2 us and a node transition 0.40 us; the nodes cost 1.5
(input copy), 4.95 (to_qkv GEMM, ~1.1 PFLOP/s), 2.90 (rope), 5.97 (L1
attention), 2.05 (merge), 3.07 (to_out GEMM, ~590 TFLOP/s) and ~1.4 us (output
copy).  Both GEMMs are past the reach of anything hand-written here: warp-level
``mma``/``wmma`` measures 545 TFLOP/s on this device, so cuBLAS is already on
Blackwell's 5th-generation ``tcgen05`` path.  See ITERATIONS.md for the two
alternatives that were built, measured and rejected on that basis.

Everything is guarded: with no extension, with grad enabled, under an in-flight
capture, or on a shape/dtype the fused kernels do not cover, the module runs the
reference chain (still with the memoized tables, which are always valid).
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

# ---------------------------------------------------------------------------
# Fused CUDA kernels (built once at import; the .so is cached by content hash).
# ---------------------------------------------------------------------------
_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> oasis_rope_qkv(const at::Tensor& qkv, const at::Tensor& cos,
                                       const at::Tensor& sin, int64_t heads,
                                       bool acc_low, int64_t max_blocks);
at::Tensor oasis_attn_merge_heads(const at::Tensor& src, int64_t max_blocks);
void oasis_copy_into(const at::Tensor& dst, const at::Tensor& src);
at::Tensor oasis_copy_of(const at::Tensor& src);
void oasis_ep_reset();
void oasis_ep_copy(const at::Tensor& dst, const at::Tensor& src, int64_t slot);
int64_t oasis_plan_build(int64_t graph_ptr, int64_t exec_ptr,
                         const at::Tensor& staging, const at::Tensor& out_proto);
at::Tensor oasis_plan_run(int64_t plan, const at::Tensor& x);
void oasis_plan_free(int64_t plan);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oasis_rope_qkv", &oasis_rope_qkv);
  m.def("oasis_attn_merge_heads", &oasis_attn_merge_heads);
  m.def("oasis_copy_into", &oasis_copy_into);
  m.def("oasis_copy_of", &oasis_copy_of);
  m.def("oasis_ep_reset", &oasis_ep_reset);
  m.def("oasis_ep_copy", &oasis_ep_copy);
  m.def("oasis_plan_build", &oasis_plan_build);
  m.def("oasis_plan_run", &oasis_plan_run);
  m.def("oasis_plan_free", &oasis_plan_free);
}
"""

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

__device__ __forceinline__ float to_f(float x)         { return x; }
__device__ __forceinline__ float to_f(__half x)        { return __half2float(x); }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ void from_f(float& d, float v)         { d = v; }
__device__ __forceinline__ void from_f(__half& d, float v)        { d = __float2half_rn(v); }
__device__ __forceinline__ void from_f(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

// Round a float through T and back -- reproduces one ATen TensorIterator store.
template <typename T>
__device__ __forceinline__ float round_via(float v) {
  T t;
  from_f(t, v);
  return to_f(t);
}

// N elements of T with their natural alignment declared, so ptxas emits one
// wide ld/st instead of N narrow ones.  nvcc cannot prove 16-byte alignment
// through `base + row * stride + col` arithmetic, so without this the rope and
// merge kernels issue 8 scalar u16 loads per thread; the host side only selects
// an N whose alignment it has actually verified.
template <typename T, int N>
struct alignas(sizeof(T) * N < 16 ? sizeof(T) * N : 16) VecT {
  T e[N];
};

// ---------------------------------------------------------------------------
// Rotary, applied in place to the q and k halves of the qkv projection output.
//
// qkv is [rows, 3*C] with row stride `rs` and C = heads * D.  Column c of the
// q (t=0) or k (t=1) half is head c/D, lane d = c%D; the reference pairs lanes
// (2i, 2i+1) and computes, elementwise in the promoted dtype,
//     y[2i]   = x[2i]*cos - x[2i+1]*sin
//     y[2i+1] = x[2i+1]*cos + x[2i]*sin
// with cos/sin read from a [S, D] table at row s = row % S.  ACC_LOW rounds the
// two products to T before the add, which is what ATen does when the promoted
// dtype is fp16/bf16; otherwise the sum stays fp32 and rounds once on store.
//
// VEC lanes are handled per thread out of one aligned load, so consecutive
// threads cover consecutive addresses.
// ---------------------------------------------------------------------------
template <typename T, typename TC, bool ACC_LOW, int VEC>
__global__ void oasis_rope_kernel(T* __restrict__ qkv,
                                  const TC* __restrict__ cs,
                                  const TC* __restrict__ sn,
                                  int total, int vpr, int S, int D, int C,
                                  long long rs) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
  int v = i % vpr;                 // which VEC-wide chunk of the C columns
  int rest = i / vpr;
  int t = rest & 1;                // 0 -> q, 1 -> k
  int row = rest >> 1;
  int s = row % S;
  int c0 = v * VEC;

  T* p = qkv + row * rs + (long long)t * C + c0;
  const int d0 = c0 % D;
  const TC* cp = cs + (long long)s * D + d0;
  const TC* sp = sn + (long long)s * D + d0;

  using QV = VecT<T, VEC>;
  using CV = VecT<TC, VEC>;
  const QV x = *reinterpret_cast<const QV*>(p);
  const CV cv = *reinterpret_cast<const CV*>(cp);
  const CV sv = *reinterpret_cast<const CV*>(sp);

  QV out;
  #pragma unroll
  for (int j = 0; j < VEC; j += 2) {
    float x0 = to_f(x.e[j]), x1 = to_f(x.e[j + 1]);
    float c_lo = to_f(cv.e[j]),     s_lo = to_f(sv.e[j]);
    float c_hi = to_f(cv.e[j + 1]), s_hi = to_f(sv.e[j + 1]);
    float a0 = x0 * c_lo, b0 = -(x1 * s_lo);
    float a1 = x1 * c_hi, b1 = x0 * s_hi;
    if (ACC_LOW) {
      a0 = round_via<T>(a0); b0 = round_via<T>(b0);
      a1 = round_via<T>(a1); b1 = round_via<T>(b1);
    }
    from_f(out.e[j], a0 + b0);
    from_f(out.e[j + 1], a1 + b1);
  }
  *reinterpret_cast<QV*>(p) = out;
  }
}

// ---------------------------------------------------------------------------
// Merge attention heads: src is (B, S, H, D) with arbitrary leading strides and
// a unit last stride; dst is (B*S, H*D) contiguous -- the matrix `to_out` wants.
// ---------------------------------------------------------------------------
template <typename T, int VEC>
__global__ void oasis_merge_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                   int total, int dvec, int H, int S,
                                   long long s0, long long s1, long long s2, int D) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
  int dv = i % dvec;
  int r1 = i / dvec;
  int h = r1 % H;
  int r2 = r1 / H;
  int s = r2 % S;
  int b = r2 / S;
  const T* sp = src + b * s0 + (long long)s * s1 + (long long)h * s2 + dv * VEC;
  T* dp = dst + ((long long)(b * S + s) * H + h) * D + dv * VEC;
  using V = VecT<T, VEC>;
  *reinterpret_cast<V*>(dp) = *reinterpret_cast<const V*>(sp);
  }
}

// These kernels are launch-ramp bound, not bandwidth bound: on this device an
// otherwise-empty kernel costs ~1.2 us at a few CTAs but ~5 us at ~900, because
// the grid's dispatch ramp is part of the kernel's measured duration.  So the
// grid is capped and each thread strides over several elements instead.
inline int nblocks(int total, int threads, int64_t max_blocks) {
  int b = (total + threads - 1) / threads;
  if (max_blocks > 0 && b > (int)max_blocks) b = (int)max_blocks;
  return b < 1 ? 1 : b;
}

template <typename T, typename TC, bool ACC_LOW>
inline void rope_launch(at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin,
                        int rows, int S, int D, int C, long long rs, int vec,
                        int64_t max_blocks, cudaStream_t stream) {
  T* q = static_cast<T*>(qkv.mutable_data_ptr());
  const TC* cp = static_cast<const TC*>(cos.const_data_ptr());
  const TC* sp = static_cast<const TC*>(sin.const_data_ptr());
  int vpr = C / vec;
  int total = rows * vpr * 2;
  int threads = total < 256 ? ((total + 31) / 32) * 32 : 256;
  int blocks = nblocks(total, threads, max_blocks);
  if (vec == 8) {
    oasis_rope_kernel<T, TC, ACC_LOW, 8><<<blocks, threads, 0, stream>>>(
        q, cp, sp, total, vpr, S, D, C, rs);
  } else if (vec == 4) {
    oasis_rope_kernel<T, TC, ACC_LOW, 4><<<blocks, threads, 0, stream>>>(
        q, cp, sp, total, vpr, S, D, C, rs);
  } else {
    oasis_rope_kernel<T, TC, ACC_LOW, 2><<<blocks, threads, 0, stream>>>(
        q, cp, sp, total, vpr, S, D, C, rs);
  }
}

template <typename T>
inline void rope_dispatch_cs(at::Tensor& qkv, const at::Tensor& cos, const at::Tensor& sin,
                             bool acc_low, int rows, int S, int D, int C,
                             long long rs, int vec, int64_t max_blocks,
                             cudaStream_t stream) {
  switch (cos.scalar_type()) {
    case at::kHalf:
      if (acc_low) rope_launch<T, __half, true>(qkv, cos, sin, rows, S, D, C, rs, vec, max_blocks, stream);
      else         rope_launch<T, __half, false>(qkv, cos, sin, rows, S, D, C, rs, vec, max_blocks, stream);
      break;
    case at::kBFloat16:
      if (acc_low) rope_launch<T, __nv_bfloat16, true>(qkv, cos, sin, rows, S, D, C, rs, vec, max_blocks, stream);
      else         rope_launch<T, __nv_bfloat16, false>(qkv, cos, sin, rows, S, D, C, rs, vec, max_blocks, stream);
      break;
    case at::kFloat:
      if (acc_low) rope_launch<T, float, true>(qkv, cos, sin, rows, S, D, C, rs, vec, max_blocks, stream);
      else         rope_launch<T, float, false>(qkv, cos, sin, rows, S, D, C, rs, vec, max_blocks, stream);
      break;
    default: TORCH_CHECK(false, "unsupported cos/sin dtype");
  }
}

// A view onto `base`'s storage built without touching the dispatcher: three
// as_strided calls would otherwise be ~1.5 us of the per-call budget.
inline at::Tensor strided_view(const at::Tensor& base, at::IntArrayRef sizes,
                               at::IntArrayRef strides, int64_t offset) {
  auto impl = c10::make_intrusive<at::TensorImpl>(
      c10::Storage(base.storage()), base.key_set(), base.dtype());
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(offset);
  return at::Tensor(std::move(impl));
}

}  // namespace

std::vector<at::Tensor> oasis_rope_qkv(const at::Tensor& qkv, const at::Tensor& cos,
                                       const at::Tensor& sin, int64_t heads,
                                       bool acc_low, int64_t max_blocks) {
  TORCH_CHECK(qkv.is_cuda() && cos.is_cuda() && sin.is_cuda(), "cuda tensors required");
  TORCH_CHECK(qkv.dim() == 2 && qkv.stride(1) == 1, "qkv must be a 2-D row-major matrix");
  TORCH_CHECK(cos.dim() == 2 && cos.is_contiguous() && sin.sizes() == cos.sizes()
              && sin.is_contiguous() && sin.scalar_type() == cos.scalar_type(),
              "cos/sin must be matching contiguous [S, D] tables");
  const int64_t rows = qkv.size(0);
  const int64_t C3 = qkv.size(1);
  const int64_t rs = qkv.stride(0);
  const int64_t S = cos.size(0);
  const int64_t D = cos.size(1);
  const int64_t C = heads * D;
  TORCH_CHECK(C3 == 3 * C, "qkv width must be 3 * heads * dim_head");
  TORCH_CHECK(D % 2 == 0, "dim_head must be even");
  TORCH_CHECK(rows % S == 0, "rows must be a multiple of the table length");

  // Widest vector width that stays inside one head and whose alignment holds
  // for every address the kernel forms: the qkv rows (base + k*rs + t*C + v*VEC)
  // and the cos/sin rows (base + s*D + d0).  VecT<> below assumes exactly this.
  int vec = 0;
  const int64_t esz = qkv.element_size();
  const int64_t csz = cos.element_size();
  const auto qbase = reinterpret_cast<uintptr_t>(qkv.const_data_ptr());
  const auto cbase = reinterpret_cast<uintptr_t>(cos.const_data_ptr());
  const auto sbase = reinterpret_cast<uintptr_t>(sin.const_data_ptr());
  for (int cand : {8, 4, 2}) {
    const uintptr_t qa = std::min<int64_t>(16, cand * esz);
    const uintptr_t ca = std::min<int64_t>(16, cand * csz);
    if (D % cand == 0 && C % cand == 0 && rs % cand == 0
        && qbase % qa == 0 && cbase % ca == 0 && sbase % ca == 0) {
      vec = cand;
      break;
    }
  }
  TORCH_CHECK(vec != 0, "qkv/cos/sin layout is not 2-element aligned");

  const c10::cuda::CUDAGuard guard(qkv.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor mut = qkv;
  if (rows > 0) {
    switch (qkv.scalar_type()) {
      case at::kHalf:
        rope_dispatch_cs<__half>(mut, cos, sin, acc_low, (int)rows, (int)S, (int)D,
                                 (int)C, rs, vec, max_blocks, stream);
        break;
      case at::kBFloat16:
        rope_dispatch_cs<__nv_bfloat16>(mut, cos, sin, acc_low, (int)rows, (int)S, (int)D,
                                        (int)C, rs, vec, max_blocks, stream);
        break;
      case at::kFloat:
        rope_dispatch_cs<float>(mut, cos, sin, acc_low, (int)rows, (int)S, (int)D,
                                (int)C, rs, vec, max_blocks, stream);
        break;
      default: TORCH_CHECK(false, "unsupported qkv dtype");
    }
  }

  // q, k, v as (bt, S, heads, D) views of the one projection buffer.
  const int64_t bt = rows / S;
  const int64_t sizes[4] = {bt, S, heads, D};
  const int64_t strides[4] = {S * rs, rs, D, 1};
  const int64_t off = qkv.storage_offset();
  std::vector<at::Tensor> out;
  out.reserve(3);
  for (int t = 0; t < 3; ++t) {
    out.push_back(strided_view(qkv, at::IntArrayRef(sizes, 4),
                               at::IntArrayRef(strides, 4), off + t * C));
  }
  return out;
}

// ---------------------------------------------------------------------------
// The two copies that bracket a graph replay: feeding the static input and
// handing the caller a tensor that the next replay will not overwrite.
// `Tensor::copy_` and `Tensor::clone` cost ~5.5 us and ~7 us of *CPU* here
// (dispatcher + TensorIterator + allocator), against ~1.5 us of device time --
// on a call whose whole budget is ~30 us. A D2D `cudaMemcpyAsync` on the
// current stream does the same work for one pybind11 call.
// ---------------------------------------------------------------------------
void oasis_copy_into(const at::Tensor& dst, const at::Tensor& src) {
  TORCH_CHECK(dst.is_cuda() && src.is_cuda(), "cuda tensors required");
  TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(), "contiguous required");
  TORCH_CHECK(dst.scalar_type() == src.scalar_type() && dst.numel() == src.numel(),
              "dst/src must match in dtype and element count");
  const size_t n = src.nbytes();
  if (n == 0) return;
  const c10::cuda::CUDAGuard guard(dst.device());
  AT_CUDA_CHECK(cudaMemcpyAsync(dst.mutable_data_ptr(), src.const_data_ptr(), n,
                                cudaMemcpyDeviceToDevice,
                                at::cuda::getCurrentCUDAStream()));
}

at::Tensor oasis_copy_of(const at::Tensor& src) {
  TORCH_CHECK(src.is_cuda() && src.is_contiguous(), "contiguous cuda tensor required");
  const c10::cuda::CUDAGuard guard(src.device());
  at::Tensor dst(at::detail::empty_cuda(src.sizes(), src.scalar_type(),
                                        src.device(), std::nullopt));
  const size_t n = src.nbytes();
  if (n == 0) return dst;
  AT_CUDA_CHECK(cudaMemcpyAsync(dst.mutable_data_ptr(), src.const_data_ptr(), n,
                                cudaMemcpyDeviceToDevice,
                                at::cuda::getCurrentCUDAStream()));
  return dst;
}

// ---------------------------------------------------------------------------
// Retargetable CUDA-graph endpoints.
//
// r1 bracketed the replay with two `cudaMemcpyAsync`s -- one feeding the static
// input, one handing the caller a tensor the next replay will not overwrite.
// Each costs ~3.7 us of *CPU* (measured here) for ~1.5 us of device work, on a
// call whose whole budget is ~35 us, and `cudaGraphLaunch` is another 4.0 us:
// three driver launches, ~11.4 us, over half the CPU side.
//
// Both copies are now nodes *inside* the graph, and the endpoint pointer of
// each is rewritten per call with `cudaGraphExecKernelNodeSetParams`, measured
// at 0.44-0.54 us.  One driver launch per call instead of three, and the
// returned tensor is still a private allocation -- nothing aliases.
//
// The mechanism is generic on purpose, because the output endpoint becomes a
// different kernel once the head-merge is fused into `to_out`: a launcher
// records (host function pointer, grid, block, shared bytes, the argument blob,
// and the byte offset of the pointer to retarget), and the plan builder finds
// the one kernel node in the captured graph whose function pointer matches.
// Identification never guesses -- the endpoint kernels are ours, so the host
// function pointer is an exact key -- and a zero-or-many match aborts the plan,
// leaving the caller on r1's copy-in/copy-out replay.  Note that
// `cudaGraphKernelNodeGetParams` *fails* on the cuBLAS and Triton nodes in this
// graph (they are launched through the driver API), so its error has to be
// swallowed *and* cleared, or the stale `cudaErrorInvalidDeviceFunction` in the
// runtime's last-error slot surfaces on some unrelated ATen call later.
// ---------------------------------------------------------------------------
constexpr size_t OASIS_EP_MAX_ARGS = 256;
constexpr int OASIS_EP_SLOTS = 2;   // 0 = graph input source, 1 = graph output dest

struct OasisEpCopyArgs {
  const uint4* src;
  uint4* dst;
  long long n;
};

// Two instantiations so the two endpoints have distinct host function pointers.
template <int SLOT>
__global__ void oasis_ep_copy_kernel(OasisEpCopyArgs a) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.n) a.dst[i] = a.src[i];
}

namespace {

struct OasisEp {
  cudaGraphNode_t node = nullptr;
  void* func = nullptr;
  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};
  unsigned shmem = 0;
  size_t args_size = 0;
  size_t ptr_offset = 0;
  alignas(16) unsigned char args[OASIS_EP_MAX_ARGS] = {};
};

// Endpoints recorded by the launchers while a capture is in flight.
OasisEp g_ep_rec[OASIS_EP_SLOTS];
bool g_ep_seen[OASIS_EP_SLOTS] = {false, false};

void oasis_ep_record(int slot, void* func, dim3 grid, dim3 block, unsigned shmem,
                     const void* args, size_t args_size, size_t ptr_offset) {
  TORCH_CHECK(slot >= 0 && slot < OASIS_EP_SLOTS, "bad endpoint slot");
  TORCH_CHECK(args_size <= OASIS_EP_MAX_ARGS, "endpoint argument blob too large");
  OasisEp& e = g_ep_rec[slot];
  e.node = nullptr;
  e.func = func;
  e.grid = grid;
  e.block = block;
  e.shmem = shmem;
  e.args_size = args_size;
  e.ptr_offset = ptr_offset;
  std::memcpy(e.args, args, args_size);
  g_ep_seen[slot] = true;
}

// Record only while capturing: the same launchers run during warmup, where the
// launch is a real one and there is no node to remember.
inline void oasis_ep_maybe_record(cudaStream_t stream, int slot, void* func, dim3 grid,
                                  dim3 block, unsigned shmem, const void* args,
                                  size_t args_size, size_t ptr_offset) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &st) != cudaSuccess) { cudaGetLastError(); return; }
  if (st != cudaStreamCaptureStatusActive) return;
  oasis_ep_record(slot, func, grid, block, shmem, args, args_size, ptr_offset);
}

struct OasisPlan {
  cudaGraphExec_t exec = nullptr;
  OasisEp ep[OASIS_EP_SLOTS];
  at::Tensor staging;     // ATen landing pad for a caller we cannot point at
  at::Tensor hold;        // keeps the capture-time destination alive
  std::vector<int64_t> out_sizes;
  at::ScalarType dtype = at::kHalf;
  c10::Device dev{c10::kCUDA, 0};
  int64_t in_numel = 0;
  const void* cur_src = nullptr;
  const void* cur_dst = nullptr;
};

void oasis_ep_set(OasisPlan* p, int slot, const void* ptr) {
  OasisEp& e = p->ep[slot];
  std::memcpy(e.args + e.ptr_offset, &ptr, sizeof(void*));
  void* kp[1] = {e.args};
  cudaKernelNodeParams np{};
  np.func = e.func;
  np.gridDim = e.grid;
  np.blockDim = e.block;
  np.sharedMemBytes = e.shmem;
  np.kernelParams = kp;
  np.extra = nullptr;
  AT_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(p->exec, e.node, &np));
}

// The one kernel node in `g` whose host function pointer is `func`.
cudaGraphNode_t oasis_find_node(const std::vector<cudaGraphNode_t>& nodes, void* func) {
  cudaGraphNode_t hit = nullptr;
  int count = 0;
  for (cudaGraphNode_t nd : nodes) {
    cudaGraphNodeType t;
    if (cudaGraphNodeGetType(nd, &t) != cudaSuccess) { cudaGetLastError(); continue; }
    if (t != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams kp{};
    // Fails for cuBLAS / Triton nodes; clear it or it resurfaces elsewhere.
    if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) { cudaGetLastError(); continue; }
    if (kp.func == func) { ++count; hit = nd; }
  }
  TORCH_CHECK(count == 1, "endpoint kernel matched ", count, " graph nodes");
  return hit;
}

}  // namespace

void oasis_ep_reset() {
  g_ep_seen[0] = g_ep_seen[1] = false;
}

void oasis_ep_copy(const at::Tensor& dst, const at::Tensor& src, int64_t slot) {
  TORCH_CHECK(dst.is_cuda() && src.is_cuda(), "cuda tensors required");
  TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(), "contiguous required");
  TORCH_CHECK(dst.nbytes() == src.nbytes(), "endpoint copy size mismatch");
  TORCH_CHECK(slot == 0 || slot == 1, "bad endpoint slot");
  const size_t nb = src.nbytes();
  TORCH_CHECK(nb > 0 && nb % 16 == 0, "endpoint copy needs a positive 16-byte multiple");
  const auto sp = reinterpret_cast<uintptr_t>(src.const_data_ptr());
  const auto dp = reinterpret_cast<uintptr_t>(dst.const_data_ptr());
  TORCH_CHECK(sp % 16 == 0 && dp % 16 == 0, "endpoint copy needs 16-byte-aligned buffers");

  OasisEpCopyArgs a;
  a.src = reinterpret_cast<const uint4*>(sp);
  a.dst = reinterpret_cast<uint4*>(dp);
  a.n = (long long)(nb / 16);
  const int threads = 256;
  const int blocks = (int)((a.n + threads - 1) / threads);

  const c10::cuda::CUDAGuard guard(dst.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void* func;
  if (slot == 0) {
    oasis_ep_copy_kernel<0><<<blocks, threads, 0, stream>>>(a);
    func = (void*)&oasis_ep_copy_kernel<0>;
  } else {
    oasis_ep_copy_kernel<1><<<blocks, threads, 0, stream>>>(a);
    func = (void*)&oasis_ep_copy_kernel<1>;
  }
  AT_CUDA_CHECK(cudaGetLastError());
  oasis_ep_maybe_record(stream, (int)slot, func, dim3(blocks), dim3(threads), 0, &a,
                        sizeof(a),
                        slot == 0 ? offsetof(OasisEpCopyArgs, src)
                                  : offsetof(OasisEpCopyArgs, dst));
}

int64_t oasis_plan_build(int64_t graph_ptr, int64_t exec_ptr, const at::Tensor& staging,
                         const at::Tensor& out_proto) {
  TORCH_CHECK(graph_ptr != 0 && exec_ptr != 0, "null graph handles");
  TORCH_CHECK(g_ep_seen[0] && g_ep_seen[1], "capture did not record both endpoints");
  cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph_ptr);
  size_t n = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
  TORCH_CHECK(n > 0, "captured graph has no nodes");
  std::vector<cudaGraphNode_t> nodes(n);
  AT_CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));

  auto* p = new OasisPlan();
  p->exec = reinterpret_cast<cudaGraphExec_t>(exec_ptr);
  p->staging = staging;
  p->hold = out_proto;
  p->out_sizes.assign(out_proto.sizes().begin(), out_proto.sizes().end());
  p->dtype = out_proto.scalar_type();
  p->dev = out_proto.device();
  p->in_numel = staging.numel();
  try {
    for (int slot = 0; slot < OASIS_EP_SLOTS; ++slot) {
      p->ep[slot] = g_ep_rec[slot];
      p->ep[slot].node = oasis_find_node(nodes, g_ep_rec[slot].func);
    }
    // Prove the driver accepts an exec-level param update before anyone relies
    // on it: re-set both endpoints to the pointers they already hold.
    const c10::cuda::CUDAGuard guard(p->dev);
    const void* s0;
    const void* s1;
    std::memcpy(&s0, p->ep[0].args + p->ep[0].ptr_offset, sizeof(void*));
    std::memcpy(&s1, p->ep[1].args + p->ep[1].ptr_offset, sizeof(void*));
    oasis_ep_set(p, 0, s0);
    oasis_ep_set(p, 1, s1);
  } catch (...) {
    delete p;
    oasis_ep_reset();
    throw;
  }
  oasis_ep_reset();
  return (int64_t)(uintptr_t)p;
}

// One pybind call per forward: allocate the output, patch whichever endpoint
// pointer moved, launch.  The harness hands a fresh input address every
// iteration (its shifting pool steps 256 bytes per call) but frees the previous
// output before the next call, so in the steady state slot 0 is patched and
// slot 1 is not.
at::Tensor oasis_plan_run(int64_t plan, const at::Tensor& x) {
  auto* p = reinterpret_cast<OasisPlan*>(plan);
  TORCH_CHECK(p != nullptr, "null plan");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == p->dtype && x.numel() == p->in_numel,
              "input does not match the plan");
  const c10::cuda::CUDAGuard guard(p->dev);
  at::Tensor out(at::detail::empty_cuda(p->out_sizes, p->dtype, p->dev, std::nullopt));

  const void* sp = x.const_data_ptr();
  if (!x.is_contiguous() || (reinterpret_cast<uintptr_t>(sp) % 16) != 0) {
    // Rare: a caller we cannot point the graph at.  Stage through ATen, which
    // handles any layout, and let the in-graph copy run as a no-op self-copy.
    p->staging.copy_(x);
    sp = p->staging.const_data_ptr();
  }
  if (sp != p->cur_src) {
    oasis_ep_set(p, 0, sp);
    p->cur_src = sp;
  }
  const void* dp = out.const_data_ptr();
  if (dp != p->cur_dst) {
    oasis_ep_set(p, 1, dp);
    p->cur_dst = dp;
  }
  AT_CUDA_CHECK(cudaGraphLaunch(p->exec, at::cuda::getCurrentCUDAStream()));
  return out;
}

void oasis_plan_free(int64_t plan) {
  delete reinterpret_cast<OasisPlan*>(plan);
}

at::Tensor oasis_attn_merge_heads(const at::Tensor& src, int64_t max_blocks) {
  TORCH_CHECK(src.is_cuda() && src.dim() == 4 && src.stride(3) == 1,
              "src must be a 4-D cuda tensor with unit last stride");
  const int64_t B = src.size(0), S = src.size(1), H = src.size(2), D = src.size(3);
  const int64_t s0 = src.stride(0), s1 = src.stride(1), s2 = src.stride(2);

  const c10::cuda::CUDAGuard guard(src.device());
  at::Tensor dst(at::detail::empty_cuda({B * S, H * D}, src.scalar_type(),
                                        src.device(), std::nullopt));
  const int64_t n = B * S * H * D;
  if (n == 0) return dst;

  int vec = 1;
  const int64_t esz = src.element_size();
  const auto sbase = reinterpret_cast<uintptr_t>(src.const_data_ptr());
  for (int cand : {8, 4, 2}) {
    const uintptr_t a = std::min<int64_t>(16, cand * esz);
    if (D % cand == 0 && s0 % cand == 0 && s1 % cand == 0 && s2 % cand == 0
        && sbase % a == 0) {
      vec = cand;
      break;
    }
  }
  const int dvec = (int)(D / vec);
  const int total = (int)(n / vec);
  const int threads = total < 256 ? ((total + 31) / 32) * 32 : 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define OASIS_MERGE(T, V)                                                        \
  oasis_merge_kernel<T, V><<<nblocks(total, threads, max_blocks), threads, 0, stream>>>( \
      static_cast<const T*>(src.const_data_ptr()),                               \
      static_cast<T*>(dst.mutable_data_ptr()), total, dvec, (int)H, (int)S,       \
      s0, s1, s2, (int)D)
#define OASIS_MERGE_V(T)                     \
  switch (vec) {                             \
    case 8: OASIS_MERGE(T, 8); break;        \
    case 4: OASIS_MERGE(T, 4); break;        \
    case 2: OASIS_MERGE(T, 2); break;        \
    default: OASIS_MERGE(T, 1); break;       \
  }

  switch (src.scalar_type()) {
    case at::kHalf:     OASIS_MERGE_V(__half); break;
    case at::kBFloat16: OASIS_MERGE_V(__nv_bfloat16); break;
    case at::kFloat:    OASIS_MERGE_V(float); break;
    default: TORCH_CHECK(false, "unsupported src dtype");
  }
#undef OASIS_MERGE_V
#undef OASIS_MERGE
  return dst;
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    # Pin the arch list to the device actually present.  torch's default list
    # here is {75, 80, 86, 90, 100, 120}, and compiling for six of them costs
    # minutes per build for no benefit; the tag carries the arch so a cached .so
    # is never reused across GPUs.
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}.{cap[1]}"
    tag = hashlib.md5((_CPP_SRC + _CUDA_SRC + arch).encode()).hexdigest()[:10]
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=f"oasis_axial_fused_{tag}",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


_ext = None
if torch.cuda.is_available() and not os.environ.get("OASIS_AXIAL_NO_EXT"):
    try:
        _ext = _build()
    except Exception:  # pragma: no cover - fall back to the reference chain
        _ext = None

_rope_qkv = _ext.oasis_rope_qkv if _ext is not None else None
_merge_heads = _ext.oasis_attn_merge_heads if _ext is not None else None
_copy_into = _ext.oasis_copy_into if _ext is not None else None
_copy_of = _ext.oasis_copy_of if _ext is not None else None
_ep_reset = _ext.oasis_ep_reset if _ext is not None else None
_ep_copy = _ext.oasis_ep_copy if _ext is not None else None
_plan_build = _ext.oasis_plan_build if _ext is not None else None
_plan_run = _ext.oasis_plan_run if _ext is not None else None
_plan_free = _ext.oasis_plan_free if _ext is not None else None

_MISSING = object()
_LOW_PREC = (torch.float16, torch.bfloat16)
# CTA cap for the two hand-written kernels; 0 means one thread per element.
# Swept over {1,2,3,4,6,8,12}x SM count and 0 won or tied everywhere, so the
# grid-stride loop below runs a single iteration on the captured shapes -- the
# knob is kept because it is the only lever on these two kernels' grids.
_GRID_CAP = int(os.environ.get("OASIS_AXIAL_GRID_CAP", "0"))
# Set OASIS_AXIAL_NO_GRAPH=1 to keep the eager 5-launch path (NCU attribution).
_USE_GRAPH = not os.environ.get("OASIS_AXIAL_NO_GRAPH")
# Set OASIS_AXIAL_NO_RETARGET=1 to fall back to r1's copy-in / replay / copy-out.
_USE_RETARGET = (_plan_build is not None
                 and not os.environ.get("OASIS_AXIAL_NO_RETARGET")
                 and hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph")
                 and hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph_exec"))


class OasisSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")
        self.dim_head = dim_head
        # (height, width, freqs identity/version) -> (freqs, cos, sin) tables.
        self._rope_cache: dict = {}
        # (shape, dtype, buffer identities) -> retarget plan or r1 replay entry.
        self._graph_cache: dict = {}

    # -- memoized rotary tables ---------------------------------------------
    def _rope_tables(self, height: int, width: int):
        """``(freqs, cos, sin)`` for a ``height x width`` grid.

        ``freqs`` is exactly what ``get_axial_freqs`` returns (the reference
        chain still needs its ``(h, w, rot_dim)`` shape); ``cos``/``sin`` are the
        same values flattened to a contiguous ``(h*w, rot_dim)`` pair for the
        fused kernel.  The key pins the ``freqs`` parameter's storage, version
        and dtype, so a reloaded or edited weight rebuilds the tables.
        """
        f = self.rotary_emb.freqs
        key = (height, width, f.data_ptr(), f._version, f.dtype,
               tuple(f.shape), self.rotary_emb.dummy.device)
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        freqs = self.rotary_emb.get_axial_freqs(height, width)
        flat = freqs.reshape(-1, freqs.shape[-1])
        entry = (freqs, flat.cos().contiguous(), flat.sin().contiguous())
        if len(self._rope_cache) >= 8:      # a stale weight pointer can recur
            self._rope_cache.clear()
        self._rope_cache[key] = entry
        return entry

    # -- reference chain (fallback) -----------------------------------------
    def _forward_reference(self, x, freqs):
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        q = oasis_apply_rotary_emb(freqs, q)
        k = oasis_apply_rotary_emb(freqs, k)
        q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        out = self.attn(q, k, v, causal=False)
        out = out.reshape(bsz, time, height, width, self.heads, -1).reshape(
            bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

    # -- fused path ---------------------------------------------------------
    def _fused_core(self, x, cos, sin, acc_low):
        """``x`` (bsz, time, h, w, dim) -> the 2-D ``(bsz*time*h*w, dim)`` result."""
        # One GEMM into a [bsz*time*seq, 3*heads*dim_head] matrix.  A 2-D input
        # reaches mm directly instead of walking matmul/reshape/_unsafe_view.
        qkv = self.to_qkv(x.view(-1, x.shape[-1]))
        # Rotary in place on the q and k halves; q/k/v come back as strided
        # views of that same buffer, already in DenseAttention's layout.
        q, k, v = _rope_qkv(qkv, cos, sin, self.heads, acc_low, _GRID_CAP)
        out = self.attn(q, k, v, causal=False)
        # (bt, seq, heads, dim_head) -> the [bt*seq, heads*dim_head] matrix
        # to_out wants, in one launch.
        out = _merge_heads(out, _GRID_CAP)
        return self.to_out(out)

    def _forward_fused(self, x, cos, sin, acc_low):
        bsz, time, height, width, _ = x.shape
        return self._fused_core(x, cos, sin, acc_low).view(bsz, time, height, width, -1)

    def _capture_body(self, staging, cos, sin, acc_low, out_proto):
        """``_fused_core`` bracketed by the two retargetable endpoint nodes.

        Slot 0's source and slot 1's destination are the pointers rewritten per
        call; at capture time both point at buffers we own, so the recorded
        launch is a self-copy for slot 0 and never observed for slot 1.
        """
        _ep_copy(staging, staging, 0)
        _ep_copy(out_proto, self._fused_core(staging, cos, sin, acc_low), 1)

    # -- one graph per (shape, buffer identity) -----------------------------
    def _warm(self, fn, *args):
        """Run ``fn`` three times on a side stream, then rejoin.

        The first call JITs the Triton attention kernel and grows cuBLAS'
        workspace, neither of which may happen inside a capture.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(*args)
        torch.cuda.current_stream().wait_stream(side)

    def _capture_retarget(self, x, cos, sin, acc_low):
        """Capture with both bookend copies *inside* the graph, endpoints live.

        The returned plan owns nothing but the node handles: the ``cudaGraph_t``
        and ``cudaGraphExec_t`` belong to ``graph``, which the cache entry keeps
        alive alongside it.
        """
        staging = torch.empty_like(x)
        staging.copy_(x)
        out_proto = torch.empty(x.shape[:-1] + (self.to_out.weight.shape[0],),
                                dtype=x.dtype, device=x.device)
        self._warm(self._capture_body, staging, cos, sin, acc_low, out_proto)
        _ep_reset()
        graph = torch.cuda.CUDAGraph(keep_graph=True)   # node handles must survive
        with torch.cuda.graph(graph):
            self._capture_body(staging, cos, sin, acc_low, out_proto)
        graph.instantiate()
        plan = _plan_build(graph.raw_cuda_graph(), graph.raw_cuda_graph_exec(),
                           staging, out_proto)
        return (0, plan, graph)

    def _capture_replay(self, x, cos, sin, acc_low):
        """r1's fallback: fixed static buffers, two driver-launched copies."""
        static_x = torch.empty_like(x)
        static_x.copy_(x)
        self._warm(self._forward_fused, static_x, cos, sin, acc_low)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self._forward_fused(static_x, cos, sin, acc_low)
        return (1, graph, static_x, static_out)

    def _drop_graph_cache(self):
        for entry in self._graph_cache.values():
            if entry is not None and entry[0] == 0:
                _plan_free(entry[1])
        self._graph_cache.clear()

    def _graph_entry(self, x, cos, sin, acc_low):
        """Capture this shape, or return ``None`` if capture is impossible.

        Weights are read *through* the captured pointers, so an in-place weight
        update (``load_state_dict``) is honoured automatically; the key therefore
        pins buffer identities, not values, and a reassigned parameter or a
        rebuilt rotary table forces a re-capture.
        """
        key = (tuple(x.shape), x.dtype, x.device,
               self.to_qkv.weight.data_ptr(), self.to_out.weight.data_ptr(),
               self.to_out.bias.data_ptr(), cos.data_ptr(), sin.data_ptr())
        hit = self._graph_cache.get(key, _MISSING)
        if hit is not _MISSING:
            return hit                                  # entry, or None if capture failed
        entry = None
        if _USE_RETARGET:
            try:
                entry = self._capture_retarget(x, cos, sin, acc_low)
            except Exception:
                # No node identification, no exec-param update, or no
                # `raw_cuda_graph` on this driver/torch: use r1's replay.
                entry = None
        if entry is None:
            try:
                entry = self._capture_replay(x, cos, sin, acc_low)
            except Exception:
                self._graph_cache[key] = None           # never retry this key
                return None
        if len(self._graph_cache) >= 16:
            self._drop_graph_cache()
        self._graph_cache[key] = entry
        return entry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, dim = x.shape
        freqs, cos, sin = self._rope_tables(height, width)

        fused_ok = (_rope_qkv is not None
                    and x.is_cuda
                    and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and x.is_contiguous()
                    and cos.shape[1] == self.dim_head   # rot_dim covers all of it
                    and self.dim_head % 2 == 0)
        if not fused_ok:
            return self._forward_reference(x, freqs)

        acc_low = torch.promote_types(x.dtype, cos.dtype) in _LOW_PREC

        # Graph replay is only valid for inference: autograd needs the real op
        # sequence, and capture cannot run while another capture is in flight.
        if (_USE_GRAPH and not torch.is_grad_enabled() and not x.requires_grad
                and not torch.cuda.is_current_stream_capturing()):
            entry = self._graph_entry(x, cos, sin, acc_low)
            if entry is not None:
                if entry[0] == 0:
                    # One driver launch: the endpoint pointers are patched on
                    # the exec, so the graph reads `x` and writes a private
                    # output tensor directly.  Nothing aliases across calls.
                    return _plan_run(entry[1], x)
                graph, static_x, static_out = entry[1], entry[2], entry[3]
                _copy_into(static_x, x)
                graph.replay()
                # The graph writes one fixed buffer, so the caller gets a copy;
                # handing back the buffer itself would alias the next call.
                return _copy_of(static_out)

        return self._forward_fused(x, cos, sin, acc_low)
