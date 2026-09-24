"""Oasis VAE self-attention with a fused rotary stage.

Same ``__init__``/``forward`` contract as the baseline, and the same submodules under
the same names: ``qkv``/``proj`` are ``..L1.linear.Linear``, attention is
``..L1.dense_attention.DenseAttention(backend="sdpa")``, and ``rotary`` is
``..L1.oasis_rotary.OasisRotaryEmbedding``. Keeping those names is not cosmetic --
the harness shares weights with ``load_state_dict(..., strict=False)`` inside a bare
``except``, so a renamed parameter is not reported, it just leaves this module holding
its own random weights.

The one structural change is the rotary stage. The baseline calls
``oasis_apply_rotary_emb`` twice; each call recomputes ``cos``/``sin`` over
``rotary_freqs`` -- a table fixed at construction -- and materialises a chain of fp32
intermediates twice the width of the fp16 data, before a trailing ``reshape`` produces
the contiguous ``(bsz, num_heads, seq_len, head_dim)`` tensors attention consumes.
Here the trig is evaluated once in ``__init__`` into two non-persistent fp32 buffers,
and a single kernel reads the ``qkv`` GEMM output and writes those two tensors
directly. The layout it writes is byte-for-byte the baseline's, so SDPA keeps
selecting the same cuDNN kernel, and ``v`` still reaches attention as a free strided
view of the ``qkv`` slice with no copy.

Both scored batch sizes need this for different reasons. At ``bsz=6`` the rotary stage
is bandwidth on top of real work; at ``bsz=1`` the forward is dominated by host enqueue
of its eager operations, and the harness times a single un-synchronised forward between
two CUDA events, so dispatch the device cannot hide lands inside the measured window.
Collapsing the operation count is itself the win there.

The regime the fused path claims, as sufficient conditions:

  * fp16 input on CUDA, ``qkv`` output contiguous, all operand bases 16-byte aligned;
  * ``2 * rot_dim == head_dim`` and ``head_dim % 16 == 0`` -- exactly what the
    branch-free thread mapping needs, no more;
  * the ``cos``/``sin`` tables present, fp32, and on the same device as ``qkv``;
  * grad mode off, since the entry point is raw pybind and records no autograd node.

Everything else -- bf16, a partially rotated head, a CPU module, grad-enabled callers,
``torch.compile`` and CUDA-graph capture -- takes the reference path, which reproduces
the baseline's body against the same already-computed ``qkv``. ``torch.compile`` and
graph capture are deliberately out of scope: the bench uses neither, and supporting
them would mean a registered custom op with a fake-tensor rule for no scored benefit.

The arithmetic is bit-exact against the reference rather than merely inside the
harness's 1e-2 gate. The reference evaluates, in fp32,
``t_middle * freqs.cos() + rotate_half(t_middle) * freqs.sin()`` and rounds once on the
trailing cast, which per element is

    out[d] = t[d]*cos[s][d] + (d even ? -t[d+1] : t[d-1]) * sin[s][d]

as two fp32 multiplies, one fp32 add, and one round-to-nearest-even to fp16. The kernel
spells those out with ``__fmul_rn``/``__fadd_rn``/``__float2half_rn`` so no compiler
setting can contract them into an FMA. The kernel is latency-bound on memory with the FMA
pipe at 14.8% and the ALU pipe at 27.9% while warps stall on memory 14 deep (measured on the
shipped kernel, ``profile/rope_v2_shipped/REPORT.md``), so the extra instruction is free and
the exactness removes a whole class of debugging. ``profile/rope_parity.py`` shows what it
buys: an FMA-contracted build of this same source differs from the reference in 45 of
1 179 648 ``q`` elements while still passing the harness gate, so contraction would have been
invisible to the score and visible to anyone debugging.

Elements at ``d >= rot_dim`` are widened to fp32 and narrowed back rather than copied
verbatim. That looks like a no-op and is one for every finite value, both infinities and
both signed zeros -- but the reference reaches those elements through
``cat(fp32_transformed, fp16_passthrough)``, whose promote-and-round-back canonicalises a
NaN payload to ``0x7fff`` on CUDA. Reproducing the round trip keeps the bit-exactness claim
unconditional instead of holding only for non-NaN input.

Launch geometry is a pair of named constants set from the sweep in
``profile/rope_sweep.py`` (all points, including the rejected ones, in
``profile/rope_sweep.csv``). They are arguments of the extension entry point so the
sweep can vary them, but the module itself never reads them from the environment.
``FK_OASIS_VAE_ATTN_FUSED=0`` forces the reference path, for A/B testing only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

# Unique to this workspace: the name keys both the ninja build lock and the resulting
# .so, so it must not collide with any other operator's extension.
_EXTENSION_NAME = "fk_cand_oasis_vae_attention_rope"

# Launch geometry, fixed. 256 threads per block puts bsz=6 at 864 blocks and bsz=1 at
# 144 -- the latter just under B200's 148 SMs, which is why 128 was swept against it.
#
# What the sweep can resolve depends on which measurement you look at, and the two
# disagree in an instructive way (profile/rope_sweep.csv accumulates every run).
#
# On the *scored forward* the sweep resolves almost nothing: the run-to-run spread of one
# geometry is 1.8-2.8 us, and nine or ten of the ten geometries fall inside it. That is the
# expected shape once the kernel is known to be latency-bound at roughly one wave per SM --
# block size mostly re-partitions the same threads and cannot change the wave count -- and it
# is why the sweep reports a band and checks membership rather than naming a winner.
#
# On the *kernel in isolation* it does resolve, and that is what picks the constant. Medians
# over two runs, bsz=6 / bsz=1 in microseconds: 64 -> 13.3/9.3, 128 -> 13.3/9.3,
# 256 -> 13.3/9.3, 512 -> 13.3/10.2, 1024 -> 16.4/11.3. So 1024 is about 23% worse and 512
# pays 10% at bsz=1, while 64, 128 and 256 are indistinguishable. 256 is taken from that tie
# because it also keeps each warp's four stores inside one 1 KiB window of the destination,
# which head-major ordering is what provides.
#
# Deliberately not environment-configurable: a runtime knob would let the shipped module
# run the geometry the sweep does reject, and rejected points belong in the record under
# profile/, not in the dispatch path.
_ROPE_BLOCK_SIZE = 256
_ROPE_HEAD_MAJOR = True

# Chunk width the thread mapping is built around: 8 fp16 elements is one 16-byte
# vector load, and the rotate partner of every element sits inside the same chunk
# because the pairs are (8c, 8c+1), (8c+2, 8c+3), ... so no thread needs a
# neighbour's data. head_dim % 16 == 0 is what makes both halves chunk evenly.
_ROPE_CHUNK_ELEMS = 8

# 128-bit loads and stores need this much alignment from every operand base.
_ROPE_ALIGN_BYTES = 16

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Half.h>

#include <cuda_fp16.h>

#include <climits>
#include <cstdint>
#include <tuple>
#include <vector>

namespace {

// One thread owns one 8-element chunk of the rotated half and the mirror chunk of the
// passthrough half, for q and k alike. Every thread in the warp therefore executes the
// same instructions -- a "one thread per 16 bytes of the whole row" mapping would put
// rotate lanes and copy lanes in the same warp instead.
constexpr int kChunkElems = 8;

// Cap on the x extent of the grid, so a pathologically large problem loops rather than
// asking for a grid the driver would reject. The scored shapes never reach it.
constexpr int64_t kMaxBlocksX = 8192;

struct alignas(16) Half8 {
  __half v[kChunkElems];
};

// 32 bytes at 16-byte alignment: nvcc emits two 128-bit loads, which is all the table
// addresses guarantee (a 16-byte-aligned base plus an offset that is a multiple of
// rot_dim, itself a multiple of kChunkElems floats).
struct alignas(16) Float8 {
  float v[kChunkElems];
};

// The passthrough half is not a raw copy. The reference reaches it through
// cat(fp32_transformed, fp16_passthrough), which promotes the fp16 half to fp32 and then
// rounds the concatenation back to fp16 -- an identity for every finite value, both
// infinities and both signed zeros, but *not* for a NaN payload, which CUDA canonicalises
// to 0x7fff on the way through. Widening and narrowing each element reproduces that
// exactly (measured over all 2046 fp16 NaN encodings plus the infinities and signed
// zeros: identical to the reference on all of them), and on a kernel that is latency-bound
// with the FMA pipe at 14.8% the two extra conversions are free -- measured on the shipped
// kernel in profile/rope_v2_shipped/REPORT.md, where adding them left the register count at 40
// with 0 spills and did not raise the duration.
__device__ __forceinline__ Half8 requantize_chunk(const Half8& in) {
  Half8 out;
#pragma unroll
  for (int i = 0; i < kChunkElems; ++i) {
    out.v[i] = __float2half_rn(__half2float(in.v[i]));
  }
  return out;
}

// out[2i]   = a*cos[2i]   + (-b)*sin[2i]
// out[2i+1] = b*cos[2i+1] +    a *sin[2i+1]
// Spelled with the round-to-nearest intrinsics so the two multiplies and the add stay
// three separately rounded fp32 operations, matching the reference's eager mul/mul/add,
// and with a single round-to-nearest-even on the way back to fp16.
__device__ __forceinline__ Half8 rotate_chunk(
    const Half8& in, const Float8& cos_v, const Float8& sin_v) {
  Half8 out;
#pragma unroll
  for (int i = 0; i < kChunkElems / 2; ++i) {
    const float a = __half2float(in.v[2 * i]);
    const float b = __half2float(in.v[2 * i + 1]);
    out.v[2 * i] = __float2half_rn(
        __fadd_rn(__fmul_rn(a, cos_v.v[2 * i]), __fmul_rn(-b, sin_v.v[2 * i])));
    out.v[2 * i + 1] = __float2half_rn(
        __fadd_rn(__fmul_rn(b, cos_v.v[2 * i + 1]), __fmul_rn(a, sin_v.v[2 * i + 1])));
  }
  return out;
}

// The batch axis is a grid dimension rather than part of the linear index, so the
// per-thread decomposition costs two 32-bit divisions instead of a chain of 64-bit
// ones. kHeadMajor picks which axis the linear index walks fastest: head-major keeps a
// warp's stores inside one destination row group, sequence-major makes its qkv reads
// one contiguous run. Read and write volumes are equal, so which wins is measured.
template <bool kHeadMajor>
__global__ void oasis_vae_rope_kernel(
    const __half* __restrict__ qkv,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    __half* __restrict__ out_q,
    __half* __restrict__ out_k,
    const int per_batch,
    const int heads,
    const int seq_len,
    const int head_dim,
    const int rot_dim,
    const int chunks) {
  const int qkv_row = 3 * heads * head_dim;
  const int kv_offset = heads * head_dim;
  const int64_t qkv_base = static_cast<int64_t>(blockIdx.y) * seq_len * qkv_row;
  const int64_t out_base =
      static_cast<int64_t>(blockIdx.y) * heads * seq_len * head_dim;
  const int grid_stride = blockDim.x * gridDim.x;

  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < per_batch;
       t += grid_stride) {
    const int row = t / chunks;
    const int c = t - row * chunks;
    int h, s;
    if (kHeadMajor) {
      h = row / seq_len;
      s = row - h * seq_len;
    } else {
      s = row / heads;
      h = row - s * heads;
    }

    const int d_rot = c * kChunkElems;
    const int d_pass = rot_dim + d_rot;
    const int64_t src = qkv_base + static_cast<int64_t>(s) * qkv_row
                      + static_cast<int64_t>(h) * head_dim;
    const int64_t dst = out_base
                      + (static_cast<int64_t>(h) * seq_len + s) * head_dim;
    const int64_t tab = static_cast<int64_t>(s) * rot_dim + d_rot;

    // Every load first, so the four 16-byte fetches and the two table fetches are all
    // in flight before any arithmetic waits on them.
    const Half8 q_rot = *reinterpret_cast<const Half8*>(qkv + src + d_rot);
    const Half8 q_pass = *reinterpret_cast<const Half8*>(qkv + src + d_pass);
    const Half8 k_rot = *reinterpret_cast<const Half8*>(qkv + src + kv_offset + d_rot);
    const Half8 k_pass =
        *reinterpret_cast<const Half8*>(qkv + src + kv_offset + d_pass);
    // Issued once and used for both q and k, which is the point of giving one thread
    // the same chunk of each.
    const Float8 cos_v = *reinterpret_cast<const Float8*>(cos_table + tab);
    const Float8 sin_v = *reinterpret_cast<const Float8*>(sin_table + tab);

    *reinterpret_cast<Half8*>(out_q + dst + d_rot) =
        rotate_chunk(q_rot, cos_v, sin_v);
    *reinterpret_cast<Half8*>(out_q + dst + d_pass) = requantize_chunk(q_pass);
    *reinterpret_cast<Half8*>(out_k + dst + d_rot) =
        rotate_chunk(k_rot, cos_v, sin_v);
    *reinterpret_cast<Half8*>(out_k + dst + d_pass) = requantize_chunk(k_pass);
  }
}

bool aligned16(const at::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

// Everything the launch needs, derived once from the inputs. Splitting validation from
// the launch is what lets a caller that supplies its own destinations -- the coverage
// test does, so it can fill them with a known pattern first -- go through exactly the
// checks and exactly the launch the allocating entry point uses, rather than a
// reimplementation of them that could drift.
struct RopePlan {
  int64_t bsz, seq_len, heads, head_dim, rot_dim, chunks, per_batch, blocks_x;
  int64_t block_size;
  bool empty;
};

// Guards use TORCH_CHECK_VALUE, which surfaces as a Python ValueError, so "this input
// is not for me" stays distinguishable from a real failure: the caller retries on the
// reference path for ValueError only, and an out-of-memory or a CUDA fault propagates.
// The Python dispatcher already tests all of this; these are the second line, and the
// only line for anyone calling the extension directly.
RopePlan plan_rope(
    const at::Tensor& qkv,
    const at::Tensor& cos_table,
    const at::Tensor& sin_table,
    int64_t num_heads,
    int64_t block_size) {
  TORCH_CHECK_VALUE(qkv.is_cuda() && cos_table.is_cuda() && sin_table.is_cuda(),
                    "oasis_vae_rope: fused path needs CUDA tensors");
  TORCH_CHECK_VALUE(cos_table.device() == qkv.device()
                        && sin_table.device() == qkv.device(),
                    "oasis_vae_rope: fused path needs every operand on one device");
  TORCH_CHECK_VALUE(qkv.scalar_type() == at::ScalarType::Half,
                    "oasis_vae_rope: fused path is float16 only, got ",
                    qkv.scalar_type());
  TORCH_CHECK_VALUE(cos_table.scalar_type() == at::ScalarType::Float
                        && sin_table.scalar_type() == at::ScalarType::Float,
                    "oasis_vae_rope: fused path needs float32 cos/sin tables");
  TORCH_CHECK_VALUE(qkv.dim() == 3 && qkv.is_contiguous(),
                    "oasis_vae_rope: fused path needs a contiguous 3-D qkv");
  TORCH_CHECK_VALUE(cos_table.dim() == 2 && cos_table.is_contiguous()
                        && sin_table.is_contiguous()
                        && sin_table.sizes() == cos_table.sizes(),
                    "oasis_vae_rope: fused path needs contiguous 2-D cos/sin tables "
                    "of one shape");
  // A raw pybind entry records no autograd node, so a differentiable call goes back to
  // the reference path instead of silently losing its gradient.
  TORCH_CHECK_VALUE(!at::GradMode::is_enabled(),
                    "oasis_vae_rope: fused path is inference-only");

  const int64_t bsz = qkv.size(0);
  const int64_t seq_len = qkv.size(1);
  const int64_t heads = num_heads;
  // Bounded by qkv.size(2)/3 before any multiplication by heads, so a caller passing an
  // enormous num_heads cannot overflow the guard expression meant to reject it.
  TORCH_CHECK_VALUE(heads > 0 && heads <= qkv.size(2) / 3,
                    "oasis_vae_rope: num_heads ", heads,
                    " is not positive and at most a third of the qkv width ",
                    qkv.size(2));
  TORCH_CHECK_VALUE(qkv.size(2) % (3 * heads) == 0,
                    "oasis_vae_rope: qkv width ", qkv.size(2),
                    " is not 3 * ", heads, " * head_dim");
  const int64_t head_dim = qkv.size(2) / (3 * heads);
  const int64_t rot_dim = cos_table.size(1);
  // The four-chunk mapping gives one thread the chunk at d and its mirror at
  // rot_dim + d, which only covers the row when the rotated and passthrough halves are
  // the same width. A partially rotated head is a different kernel, not a wider guard.
  TORCH_CHECK_VALUE(2 * rot_dim == head_dim,
                    "oasis_vae_rope: fused path needs 2 * rot_dim == head_dim, got ",
                    rot_dim, " and ", head_dim);
  TORCH_CHECK_VALUE(head_dim % (2 * kChunkElems) == 0,
                    "oasis_vae_rope: fused path needs head_dim a multiple of ",
                    2 * kChunkElems, ", got ", head_dim);
  TORCH_CHECK_VALUE(cos_table.size(0) == seq_len,
                    "oasis_vae_rope: cos/sin rows ", cos_table.size(0),
                    " do not match seq_len ", seq_len);
  TORCH_CHECK_VALUE(block_size >= 32 && block_size <= 1024 && block_size % 32 == 0,
                    "oasis_vae_rope: block_size must be a multiple of 32 in [32, 1024]"
                    ", got ", block_size);
  // 128-bit loads and stores need 16-byte-aligned bases. Fresh allocations satisfy
  // this, but that is a property of the caching allocator rather than a guarantee.
  TORCH_CHECK_VALUE(aligned16(qkv) && aligned16(cos_table) && aligned16(sin_table),
                    "oasis_vae_rope: fused path needs 16-byte-aligned operands");

  const int64_t chunks = rot_dim / kChunkElems;
  const int64_t per_batch = heads * seq_len * chunks;
  if (bsz == 0 || per_batch == 0) {
    return RopePlan{bsz, seq_len, heads, head_dim, rot_dim, chunks, per_batch, 0,
                    block_size, true};
  }
  TORCH_CHECK_VALUE(bsz <= 65535,
                    "oasis_vae_rope: batch ", bsz, " exceeds the grid y extent");
  // The kernel takes its extents as int and derives 3 * heads * head_dim and
  // heads * head_dim from them in int, so every extent it is handed has to fit -- and so
  // does the widest of those products, which is exactly the qkv width. Everything else
  // the kernel multiplies is widened to int64 first. head_dim and rot_dim follow from the
  // width, and seq_len is checked in its own right.
  TORCH_CHECK_VALUE(qkv.size(2) <= INT_MAX && seq_len <= INT_MAX,
                    "oasis_vae_rope: qkv extents (", seq_len, ", ", qkv.size(2),
                    ") exceed the 32-bit extents this kernel takes");

  const int64_t wanted = (per_batch + block_size - 1) / block_size;
  const int64_t blocks_x = wanted < kMaxBlocksX ? wanted : kMaxBlocksX;
  // The linear thread index is 32-bit on purpose: a 64-bit division per thread would cost
  // more than all the arithmetic this kernel does. The grid-stride loop's last increment
  // therefore has to stay in range too -- it runs from below per_batch to at most
  // per_batch + blocks_x * block_size, and a signed int overflow there could wrap
  // negative, pass the loop test again and index out of bounds.
  TORCH_CHECK_VALUE(per_batch + blocks_x * block_size <= INT_MAX,
                    "oasis_vae_rope: per-batch thread count ", per_batch,
                    " plus one grid stride exceeds the 32-bit index this kernel uses");

  return RopePlan{bsz, seq_len, heads, head_dim, rot_dim, chunks, per_batch, blocks_x,
                  block_size, false};
}

// Vets the destinations and launches. Every caller reaches the kernel through here, so a
// caller-supplied destination is held to the same shape, dtype, layout and alignment the
// allocating entry point produces by construction.
void launch_rope(
    const RopePlan& plan,
    const at::Tensor& qkv,
    const at::Tensor& cos_table,
    const at::Tensor& sin_table,
    at::Tensor& out_q,
    at::Tensor& out_k,
    bool head_major) {
  const std::vector<int64_t> want{plan.bsz, plan.heads, plan.seq_len, plan.head_dim};
  for (const at::Tensor* out : {&out_q, &out_k}) {
    TORCH_CHECK_VALUE(out->sizes() == at::IntArrayRef(want),
                      "oasis_vae_rope: destination shape ", out->sizes(),
                      " is not (bsz, heads, seq_len, head_dim) = ",
                      at::IntArrayRef(want));
    TORCH_CHECK_VALUE(out->scalar_type() == qkv.scalar_type(),
                      "oasis_vae_rope: destination dtype ", out->scalar_type(),
                      " does not match qkv ", qkv.scalar_type());
    TORCH_CHECK_VALUE(out->device() == qkv.device(),
                      "oasis_vae_rope: destination is not on the qkv device");
    TORCH_CHECK_VALUE(out->is_contiguous(),
                      "oasis_vae_rope: destination must be contiguous");
    TORCH_CHECK_VALUE(aligned16(*out),
                      "oasis_vae_rope: destination is not 16-byte aligned");
  }
  if (plan.empty) {
    return;
  }

  const dim3 grid(static_cast<unsigned>(plan.blocks_x),
                  static_cast<unsigned>(plan.bsz));
  const dim3 block(static_cast<unsigned>(plan.block_size));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto head_major_tag) {
    oasis_vae_rope_kernel<decltype(head_major_tag)::value>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(qkv.data_ptr<at::Half>()),
            cos_table.data_ptr<float>(),
            sin_table.data_ptr<float>(),
            reinterpret_cast<__half*>(out_q.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out_k.data_ptr<at::Half>()),
            static_cast<int>(plan.per_batch),
            static_cast<int>(plan.heads),
            static_cast<int>(plan.seq_len),
            static_cast<int>(plan.head_dim),
            static_cast<int>(plan.rot_dim),
            static_cast<int>(plan.chunks));
  };
  if (head_major) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> oasis_vae_fused_rope(
    const at::Tensor& qkv,
    const at::Tensor& cos_table,
    const at::Tensor& sin_table,
    int64_t num_heads,
    int64_t block_size,
    bool head_major) {
  const RopePlan plan = plan_rope(qkv, cos_table, sin_table, num_heads, block_size);
  // Taken before the allocation so the outputs land on the input's device whatever the
  // ambient current device is, and so the launch goes to the caller's stream on it.
  const c10::cuda::CUDAGuard guard(qkv.device());
  // Two separate allocations, not one packed (2, bsz, heads, seq_len, head_dim) tensor sliced
  // in half. The packed form is an allowed design option and was implemented and measured; it
  // is rejected on two independent grounds, both recorded in profile/packed_qk_rejected.md:
  // slice 1 necessarily carries a non-zero storage offset where the baseline's k is a fresh
  // allocation at offset 0, which breaks the field-by-field descriptor equality this module
  // exists to preserve; and it measured about 59% slower at bsz=6 (0.1187 vs 0.0747 ms) over
  // two runs, the opposite of the host-cost saving it was expected to buy.
  const std::vector<int64_t> shape{plan.bsz, plan.heads, plan.seq_len, plan.head_dim};
  at::Tensor out_q = at::empty(shape, qkv.options());
  at::Tensor out_k = at::empty(shape, qkv.options());
  launch_rope(plan, qkv, cos_table, sin_table, out_q, out_k, head_major);
  return {out_q, out_k};
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <tuple>
#include <vector>

std::tuple<at::Tensor, at::Tensor> oasis_vae_fused_rope(
    const at::Tensor& qkv,
    const at::Tensor& cos_table,
    const at::Tensor& sin_table,
    int64_t num_heads,
    int64_t block_size,
    bool head_major);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oasis_vae_fused_rope", &oasis_vae_fused_rope,
        "Fused rotary embedding over a qkv GEMM output: rotate the leading half of "
        "each q and k head against precomputed cos/sin tables, copy the trailing "
        "half, and write both as contiguous (bsz, heads, seq_len, head_dim)");
}
"""


def _local_arch_list() -> str | None:
    """Local compute capability, in the form nvcc wants for this build.

    Compute capabilities 9.0 and up need the architecture-specific ``a`` variant.
    Returning None leaves ``TORCH_CUDA_ARCH_LIST`` as it is, which is the right thing
    when the capability cannot be read.
    """
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"


def _build_fused_extension():
    """Compile the fused rotary extension into a workspace-local build directory."""
    from torch.utils.cpp_extension import load_inline

    build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    # The environment ships a multi-architecture list. Compiling all of it would cost
    # minutes of the 900 s wall budget the whole bench invocation shares, for a kernel
    # that only ever runs on this GPU.
    arch = _local_arch_list()
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "--expt-relaxed-constexpr",
            ],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if arch is not None:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# Built eagerly, so nothing compiles inside a timed region and the toolchain's helper
# processes stay clear of the harness's no-new-threads guard, and behind a try so a
# build failure costs speed rather than correctness.
_FUSED_ROPE = None
FUSED_ROPE_ERROR: str | None = None

if os.environ.get("FK_OASIS_VAE_ATTN_FUSED", "1") == "0":
    FUSED_ROPE_ERROR = "disabled by FK_OASIS_VAE_ATTN_FUSED=0"
elif not torch.cuda.is_available():
    FUSED_ROPE_ERROR = "no CUDA device available at import"
else:
    try:
        _FUSED_ROPE = _build_fused_extension().oasis_vae_fused_rope
    except Exception as exc:  # noqa: BLE001 - serve everything from the reference path
        FUSED_ROPE_ERROR = f"{type(exc).__name__}: {exc}"
        # One line, at import, on stderr. Degrading quietly would hide real breakage
        # behind a plausible-looking 1.00x.
        print(f"[candidate L2/oasis_vae_attention] fused rotary unavailable, using the "
              f"reference rotary path: {FUSED_ROPE_ERROR}", file=sys.stderr, flush=True)

#: Whether the fused kernel is live. A benchmark taken with this False measured the
#: reference rotary path, not the kernel.
FUSED_AVAILABLE = _FUSED_ROPE is not None

#: Fused launches per batch size. Plain host-side ints: no threads, no device sync,
#: nothing the harness's integrity guards watch. This is what distinguishes "the
#: extension built" from "dispatch actually reached the kernel".
FUSED_CALLS: dict[int, int] = {}


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        self.register_buffer(
            "rotary_freqs",
            self.rotary.get_axial_freqs(frame_height, frame_width),
            persistent=False,
        )
        self.attn = DenseAttention(backend="sdpa")

        seq_len = frame_height * frame_width
        head_dim = dim // num_heads
        freqs = self.rotary_freqs
        rot_dim = freqs.shape[-1]

        # rotary_freqs is derived state: built here from construction-time arguments,
        # never written again, and not in state_dict, so cos/sin of it are constants
        # too. Registered non-persistent so module movement carries them while
        # load_state_dict weight sharing stays untouched.
        #
        # The trig runs on a CUDA device when there is one, because that is where the
        # baseline evaluates it: fp32 cos/sin differ between CPU and CUDA by up to
        # 1 ulp (measured 5.96e-8, not bit-equal), and a CUDA-to-CUDA move preserves
        # bits while a CPU-to-CUDA one does not. Far below fp16 resolution either way,
        # but bit-equality is the cheaper thing to debug against. With no CUDA device
        # the tables are built on CPU and the fused path is off, so the difference is
        # unreachable.
        table = freqs
        if freqs.numel() == seq_len * rot_dim:
            table = freqs.reshape(seq_len, rot_dim)
        trig_device = table.device
        if torch.cuda.is_available():
            try:
                table = table.to("cuda")
                trig_device = table.device
            except Exception:  # noqa: BLE001 - a driver problem is not a correctness problem
                pass
        self.register_buffer("rotary_cos", torch.cos(table), persistent=False)
        self.register_buffer("rotary_sin", torch.sin(table), persistent=False)

        self._seq_len = seq_len
        self._head_dim = head_dim
        # Every structural precondition, resolved once. What is left per call is a flat
        # conjunction of attribute reads -- which matters on an operator whose baseline
        # spends more time enqueuing than computing.
        self._fused_ready = (
            FUSED_AVAILABLE
            and trig_device.type == "cuda"
            and tuple(freqs.shape) == (frame_height, frame_width, rot_dim)
            and freqs.dtype is torch.float32
            and 2 * rot_dim == head_dim
            and head_dim % (2 * _ROPE_CHUNK_ELEMS) == 0
            and num_heads > 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qkv = self.qkv(x)
        # Both tables bound once: the conjunction reads each of them twice, and the
        # extension needs them anyway.
        cos = self.rotary_cos
        sin = self.rotary_sin
        # A flat conjunction of attribute reads and comparisons, resolved before the
        # extension call rather than inside it. Every term guards something the kernel
        # relies on; the C++ layer checks the same things again as defence in depth, but
        # reaching it and being rejected would mean paying an allocation and an exception
        # to learn what a comparison already knew.
        if (self._fused_ready
                and x.is_cuda
                and x.dtype is torch.float16
                and qkv.dtype is torch.float16
                and qkv.is_contiguous()
                and qkv.data_ptr() % _ROPE_ALIGN_BYTES == 0
                and cos.dtype is torch.float32
                and sin.dtype is torch.float32
                and cos.device == qkv.device
                and sin.device == qkv.device
                and cos.is_contiguous()
                and sin.is_contiguous()
                and cos.data_ptr() % _ROPE_ALIGN_BYTES == 0
                and sin.data_ptr() % _ROPE_ALIGN_BYTES == 0
                and not torch.is_grad_enabled()):
            try:
                q, k = _FUSED_ROPE(qkv, cos, sin, self.num_heads,
                                   _ROPE_BLOCK_SIZE, _ROPE_HEAD_MAJOR)
            except (ValueError, TypeError):
                # The C++ guards rejected something the conjunction above thought it
                # had covered. Correct to fall back, but it means the fast path is not
                # running, which FUSED_CALLS is what makes visible.
                pass
            else:
                FUSED_CALLS[bsz] = FUSED_CALLS.get(bsz, 0) + 1
                out = self.attn(q.transpose(1, 2), k.transpose(1, 2),
                                self._value_view(qkv, bsz))
                return self.proj(out.reshape(bsz, self._seq_len, -1))
        return self._reference_forward(qkv, bsz)

    def _value_view(self, qkv: torch.Tensor, bsz: int) -> torch.Tensor:
        """``v`` as attention receives it, through the baseline's own chain of views.

        Every step is zero-copy, so a shorter route to the same bytes exists --
        ``qkv.view(bsz, seq_len, 3, heads, head_dim)[:, :, 2]`` reaches them in two
        operations instead of seven. It is not used, because at ``bsz=1`` it produces a
        different stride on the batch axis: a size-1 axis is never indexed past 0, so
        PyTorch has no canonical stride to produce there and the value depends on which
        chain of views produced the tensor. Reproducing the baseline's chain reproduces the
        baseline's descriptor by construction, at both batch sizes, without encoding an
        artifact of how PyTorch collapses a size-1 dimension.
        """
        # narrow rather than chunk: chunk(3, -1)[2] issues three narrows and discards two,
        # and the piece it keeps has exactly this descriptor. The remaining four steps are
        # the baseline's verbatim, which is what makes the result equal by construction
        # rather than by a stride formula that would have to special-case a size-1 batch.
        width = self.num_heads * self._head_dim
        v = qkv.narrow(-1, 2 * width, width)
        v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1)
        v = v.permute(0, 3, 1, 2, 4)
        return v.reshape(bsz, self.num_heads, self._seq_len, -1).transpose(1, 2)

    def _reference_forward(self, qkv: torch.Tensor, bsz: int) -> torch.Tensor:
        """The baseline's body, against a qkv that has already been computed.

        Branching after the GEMM rather than re-entering the baseline's forward is what
        keeps the unclaimed path from issuing the qkv projection a second time.
        """
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        q = oasis_apply_rotary_emb(self.rotary_freqs, q)
        k = oasis_apply_rotary_emb(self.rotary_freqs, k)

        seq_len = self.frame_height * self.frame_width
        q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        out = self.attn(q, k, v)
        out = out.reshape(bsz, seq_len, -1)
        return self.proj(out)
