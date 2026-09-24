"""Kimi MLA dense prefill in six dispatches.

The baseline's dense-prefill forward issues nine device kernels: two projection
GEMMs, a ``.contiguous()`` for the strided latent split, the vendored
``rms_norm`` kernel, the ``kv_b`` GEMM, two ``elementwise`` fills that build
``k`` from ``k_nope`` and a broadcast ``k_pe``, the attention kernel, and the
output GEMM. Four of the five shapes ``fastkernels bench`` times for this
operator (N = 1, 26, 64, 443 tokens) are CPU-launch-bound -- median latency is
flat regardless of the work done, while a graph replay of the same body costs a
fraction of it -- so what dominates them is dispatch count times per-dispatch
host cost, not kernel efficiency.

This module keeps the baseline's math and its parameters and collapses the
sequence to six dispatches:

    mm(hidden_states, Wq.t())      -> q       [N, 32, 192] row-contiguous
    mm(hidden_states, Wkva.t())    -> kv      [N, 576]
    fused_rms_norm_pack(kv)        -> latent  [N, 576]  normed 512 | k_pe 64
    mm(latent, packed_kv_b)        -> pack    [N, 32, 320]  k | v, no copy
    ragged_prefill(q, k, v)        -> o       [N, 32, 128]
    mm(o.view(N, 4096), Wo.t())    -> out     [N, 2304]

Three of the removed kernels are pure data movement worth 10.5% of GPU time at
N = 16384; at the launch-bound shapes they are four fewer enqueues each.

Everything else about the operator is unchanged: the class subclasses the
baseline, overrides ``forward`` only, and delegates to the inherited
implementation for every state the fast path has not been validated against
(decode, sparse, mixed batches, chunked context, a populated KV cache, fp8
linears, tensor parallelism, CUDA-graph capture, ``torch.compile``).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# The candidate meta-path finder only claims names under
# ``fastkernels.tasks.candidate.``, so this absolute import loads the genuine
# baseline module and does not recurse back into this file -- verified under
# ``--standalone``, which is what ``validate.py`` passes.
from fastkernels.tasks.baseline.L2.kimi_mla_attention import (
    KimiMLAAttention as _BaselineKimiMLAAttention,
)

from ....infra.context import CUDAGraphMode, get_context

# Relative L1 imports alias to the baseline module objects under
# ``--standalone``, so the 512 MiB ragged workspace this shares with the
# baseline's own flashinfer path stays a single module-level allocation.
from ..L1.flashinfer_mla_sparse import (
    TrtllmRaggedPrefill,
    flashinfer_mla_sparse_available,
)

_RAGGED_PREFILL = "trtllm_ragged"
_FLASH_VARLEN = "flash_varlen"

# Set once if the trtllm-gen cubin fails to load at runtime. flashinfer itself is
# imported eagerly along the baseline's own import chain, so a missing package is
# already a hard failure for the baseline; what can still fail here is the lazy
# JIT/cubin load on the first call. A broad ``except`` around a kernel launch is
# imperfect -- an asynchronous fault can surface at a later synchronization -- so
# this is a convenience for a whole-process capability, not a correctness
# mechanism. Correctness rests on the fast-path guard.
_ragged_prefill_broken = False

# ``flashinfer_mla_sparse_available`` probes the device capability, which is a
# host call on a path budgeted in tens of microseconds; the answer cannot change
# within a process.
_ragged_prefill_supported: bool | None = None


def _ragged_prefill_usable() -> bool:
    global _ragged_prefill_supported
    if _ragged_prefill_supported is None:
        _ragged_prefill_supported = flashinfer_mla_sparse_available()
    return _ragged_prefill_supported and not _ragged_prefill_broken


# ---------------------------------------------------------------------------
# Fused RMSNorm + rope passthrough
#
# One program per token: normalize the latent lane and copy the rope lane into a
# single contiguous ``[N, 576]`` buffer, which is exactly the operand the packed
# ``kv_b`` GEMM needs. Two kernels become one, and the ``.contiguous()`` the
# vendored norm forces on the strided latent split goes away with them.
#
# There are two implementations of the same arithmetic. The CUDA one is used
# whenever it builds and both lanes are 8-element aligned, because it enqueues in
# about half the host time of a Triton launch (measured 8.0 us against 16.2 us at
# N = 64) and that is the largest removable dispatch cost on a path where four of
# the five benchmarked shapes are launch-bound. The two were measured
# bit-identical to each other and to an fp32 PyTorch reference, for unit and
# non-unit norm weights alike; Triton stays as the fallback for an environment
# where the extension cannot be compiled, and for unaligned lane widths.
# ---------------------------------------------------------------------------

_RMS_NORM_PACK_CUDA_DECL = (
    "void rms_norm_pack(torch::Tensor out, torch::Tensor inp, "
    "torch::Tensor weight, int64_t latent, double eps);"
)

_RMS_NORM_PACK_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kThreads = 128;
constexpr int kVec = 8;   // 8 bf16 elements = one 16-byte load

// Reproduces the vendored vllm::rms_norm_kernel's fp32 order: accumulate x*x in
// fp32, take rsqrtf(variance / hidden + eps), then store (bf16)(x * inv_rms * w)
// with a single rounding at the end. Only the reduction tree differs from that
// kernel's cub BlockReduce, which is far inside the benchmark's tolerance and was
// measured to make no difference at all at these widths.
__global__ void rms_norm_pack_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ inp,
    const __nv_bfloat16* __restrict__ weight,
    const int64_t in_row_stride, const int64_t out_row_stride,
    const int latent, const int rope, const float eps) {
  const int row = blockIdx.x;
  const __nv_bfloat16* src = inp + (int64_t)row * in_row_stride;
  __nv_bfloat16* dst = out + (int64_t)row * out_row_stride;

  const int nvec = latent / kVec;
  const float4* src4 = reinterpret_cast<const float4*>(src);
  const float4* w4 = reinterpret_cast<const float4*>(weight);
  float4* dst4 = reinterpret_cast<float4*>(dst);

  float acc = 0.f;
  for (int i = threadIdx.x; i < nvec; i += kThreads) {
    float4 raw = src4[i];
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float x = __bfloat162float(e[j]);
      acc += x * x;
    }
  }

  __shared__ float smem[kThreads / 32];
  __shared__ float s_inv_rms;
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) v += smem[i];
    s_inv_rms = rsqrtf(v / (float)latent + eps);
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;

  for (int i = threadIdx.x; i < nvec; i += kThreads) {
    float4 raw = src4[i];
    float4 wraw = w4[i];
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
    const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wraw);
    float4 o;
    __nv_bfloat16* od = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float x = __bfloat162float(e[j]);
      od[j] = __float2bfloat16(x * inv_rms * __bfloat162float(w[j]));
    }
    dst4[i] = o;
  }

  // The rope lane passes through verbatim; this is where it had to go anyway.
  const int rvec = rope / kVec;
  const float4* rsrc4 = reinterpret_cast<const float4*>(src + latent);
  float4* rdst4 = reinterpret_cast<float4*>(dst + latent);
  for (int i = threadIdx.x; i < rvec; i += kThreads) rdst4[i] = rsrc4[i];
}

}  // namespace

void rms_norm_pack(torch::Tensor out, torch::Tensor inp, torch::Tensor weight,
                   int64_t latent, double eps) {
  TORCH_CHECK(inp.dim() == 2 && out.dim() == 2, "expected 2-D tensors");
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16
              && out.scalar_type() == at::kBFloat16
              && weight.scalar_type() == at::kBFloat16, "expected bfloat16");
  const int64_t rope = inp.size(1) - latent;
  TORCH_CHECK(latent > 0 && rope > 0, "bad lane widths");
  TORCH_CHECK(latent % kVec == 0 && rope % kVec == 0,
              "both lanes must be 8-element aligned");
  TORCH_CHECK(inp.stride(1) == 1 && out.stride(1) == 1
              && weight.stride(0) == 1, "expected unit last-dim strides");
  TORCH_CHECK(weight.numel() == latent, "weight must cover the latent lane");
  TORCH_CHECK(out.size(0) == inp.size(0) && out.size(1) == inp.size(1),
              "out and inp must have the same shape");
  const int64_t rows = inp.size(0);
  if (rows == 0) return;
  c10::cuda::CUDAGuard guard(inp.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  rms_norm_pack_bf16<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(inp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      inp.stride(0), out.stride(0), (int)latent, (int)rope, (float)eps);
}
'''

_VEC_ALIGN = 8
_cuda_ext = None
_cuda_ext_failed = False


def _rms_norm_pack_ext():
    """JIT-compile the fused-norm extension once, or ``None`` if it cannot build."""
    global _cuda_ext, _cuda_ext_failed
    if _cuda_ext is not None or _cuda_ext_failed:
        return _cuda_ext
    try:
        from torch.utils.cpp_extension import load_inline

        # Pin the build to the local architecture so the one-time compile does
        # not fan out over every arch in torch's default list. Mirrors
        # ``infra.cuda_ext``'s own pinning, and leaves an explicit setting alone.
        override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
        if override:
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        elif override is None and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = (
                f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"
            )
        _cuda_ext = load_inline(
            name="kimi_mla_fused_norm_pack",
            cpp_sources=_RMS_NORM_PACK_CUDA_DECL,
            cuda_sources=_RMS_NORM_PACK_CUDA_SRC,
            functions=["rms_norm_pack"],
            extra_cuda_cflags=["-O3", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
            verbose=False,
        )
    except Exception:
        _cuda_ext_failed = True
        _cuda_ext = None
    return _cuda_ext


@triton.jit
def _rms_norm_pack_kernel(
    kv_ptr,
    weight_ptr,
    out_ptr,
    kv_row_stride,
    out_row_stride,
    eps,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    LATENT_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
):
    """RMSNorm the latent lane and copy the rope lane, one program per token.

    Same fp32 order as the vendored ``vllm::rms_norm_kernel`` -- and as the CUDA
    kernel above, with which it was measured bit-identical.
    """
    row = tl.program_id(0)
    row_in = kv_ptr + row * kv_row_stride
    row_out = out_ptr + row * out_row_stride

    offs = tl.arange(0, LATENT_BLOCK)
    mask = offs < LATENT
    x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / LATENT + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(row_out + offs, (x * inv_rms * w).to(out_ptr.dtype.element_ty),
             mask=mask)

    rope_offs = tl.arange(0, ROPE_BLOCK)
    rope_mask = rope_offs < ROPE
    k_pe = tl.load(row_in + LATENT + rope_offs, mask=rope_mask, other=0.0)
    tl.store(row_out + LATENT + rope_offs, k_pe.to(out_ptr.dtype.element_ty),
             mask=rope_mask)


def _fused_rms_norm_pack(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    latent_dim: int,
    *,
    prefer_cuda: bool = True,
) -> torch.Tensor:
    """Normalize ``kv[:, :latent_dim]`` and pass ``kv[:, latent_dim:]`` through."""
    num_tokens, width = kv.shape
    rope_dim = width - latent_dim
    out = torch.empty_like(kv)

    if (prefer_cuda
            and kv.dtype is torch.bfloat16
            and latent_dim % _VEC_ALIGN == 0
            and rope_dim % _VEC_ALIGN == 0):
        ext = _rms_norm_pack_ext()
        if ext is not None:
            ext.rms_norm_pack(out, kv, weight, latent_dim, eps)
            return out

    _rms_norm_pack_kernel[(num_tokens,)](
        kv,
        weight,
        out,
        kv.stride(0),
        out.stride(0),
        eps,
        LATENT=latent_dim,
        ROPE=rope_dim,
        LATENT_BLOCK=triton.next_power_of_2(latent_dim),
        ROPE_BLOCK=triton.next_power_of_2(rope_dim),
        num_warps=4,
    )
    return out


def _weight_identity(param: torch.Tensor) -> tuple:
    """Composite identity of a weight tensor, for caching anything derived from it.

    Every component earns its place against measured behaviour of the harness's
    own weight-preparation order:

    * ``load_state_dict`` copies in place, so it preserves ``data_ptr`` and the
      Parameter object while bumping ``_version`` -- which is why ``data_ptr``
      alone is not enough;
    * ``module.to(dtype)`` (and any ``param.data = ...`` rebind) preserves the
      Parameter object *and* its ``_version``, and changes ``data_ptr`` and
      ``dtype`` -- which is why ``_version`` alone is not enough either.
      Measured on torch 2.11: version 6 before and after, pointer moved;
    * rebinding the attribute to a fresh ``nn.Parameter`` changes ``id``.

    Residual hole, accepted rather than assumed away: any write that reaches the
    storage without going through the Parameter's own version counter is
    invisible here -- ``param.data.copy_(...)``, a write through a separate
    tensor aliasing the same storage, or a ``param.data`` rebind that only
    reinterprets it. The transposed views would still observe such a write, since
    they share the storage; only the copied packed matrix can go stale. It does
    not arise on the benchmark path, where every derived tensor is built on the
    first forward -- after the harness has finished preparing weights.
    """
    return (
        id(param),
        param.data_ptr(),
        param._version,
        param.shape,
        param.stride(),
        param.dtype,
        param.device,
    )


class KimiMLAAttention(_BaselineKimiMLAAttention):
    """Kimi MLA attention with a guarded six-dispatch dense-prefill path."""

    # Diagnostic switches, not tuning knobs: the defaults are the fast
    # configuration, and a bench harness flips them one at a time so a latency or
    # numerics change can be attributed to a single structural change rather than
    # to the combination. ``_fold_rope_into_kv_b`` requires
    # ``_fuse_norm_and_pack``, which is what produces the contiguous operand the
    # packed GEMM consumes.
    _fuse_norm_and_pack: bool = True
    _fold_rope_into_kv_b: bool = True
    # ``_RAGGED_PREFILL`` or ``_FLASH_VARLEN``; anything else falls through to the
    # FlashAttention wrapper, which is also the fallback if trtllm-gen is
    # unusable, so an unrecognized value degrades rather than raising.
    _prefill_backend: str = _RAGGED_PREFILL

    # Lazily populated on the first fast-path forward. Nothing derived from a
    # weight may be built in ``__init__``: the harness randomizes and then copies
    # weights in *after* construction, so anything precomputed there would
    # silently encode the uninitialized values.
    _derived_weights: tuple | None = None
    _single_seq_lens: dict | None = None
    # Held in ``__dict__`` via ``object.__setattr__`` rather than assigned
    # normally: ``nn.Module.__setattr__`` files an ``nn.Module`` value under
    # ``_modules``, where this class-level ``None`` default then shadows it on
    # every read -- so each forward would build a fresh wrapper. The baseline
    # stores its own ``_kv_b_proj`` reference the same way, for the same reason.
    _ragged_prefill_op: TrtllmRaggedPrefill | None = None

    # A per-token-count cache of one-element device tensors, bounded so a long
    # run over many distinct shapes cannot grow it without limit.
    _SEQ_LENS_CACHE_LIMIT = 64

    # ------------------------------------------------------------------
    # Fast-path guard
    # ------------------------------------------------------------------

    def _fast_path_applies(self, hidden_states: torch.Tensor, ctx) -> bool:
        """Whether the observed state is the one the fast path is validated for.

        Read entirely from host-side metadata -- no device synchronization and no
        ``.item()`` on a device tensor -- because this runs on a launch budget of
        a few tens of microseconds. Several conditions are less obvious than the
        rest:

        * ``o_proj.tp_size == 1``: the fast path replaces
          ``RowParallelLinear.forward``, the one projection that all-reduces its
          output when tensor parallelism is on, so a sharded module must not take
          it.
        * the ``use_fp8`` reads: the baseline never stores ``quant_config``, so
          the only way to see a quantized layer from here is the boolean each
          ``parallel_linear`` class derives from it at construction.
        * the ``bias is None`` reads: this operator constructs every projection
          with ``bias=False``, so they are unreachable by construction -- but the
          fast path's ``torch.mm`` would silently drop a bias that some caller
          attached later, where ``F.linear`` would apply it.
        * ``norm.weight.device``: the baseline's ``RMSNorm.forward`` quietly moves
          a mismatched weight onto the input's device, whereas the fused kernel
          takes ``weight.data_ptr()`` as-is under a guard set from the *input's*
          device -- an off-device weight there is a bad access, not an error.
        * ``torch.is_grad_enabled()``: the fused norm has no registered backward,
          so the fast path is inference-only. The benchmark runs every forward
          under ``torch.no_grad``, and the baseline's own vendored norm drops
          gradient too, but delegating is the honest answer for a grad-enabled
          caller.
        """
        attn = self.attn
        norm = self.kv_a_layernorm
        dtype = hidden_states.dtype

        return (
            # Only the dense-prefill path, with a single fully-new batch.
            ctx.is_prefill
            and not ctx.is_mixed
            and ctx.chunked_context is None
            and ctx.slot_mapping is None
            and ctx.cu_seqlens_q is not None
            # An empty KV cache and a dense layer: nothing to store or gather.
            and attn.k_cache.numel() == 0
            and not attn.is_sparse
            # Custom-op / compiled / graph dispatch keeps the inherited path.
            and not attn._use_custom_op
            and not torch.compiler.is_compiling()
            and not torch.is_grad_enabled()
            and ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and not ctx.is_cuda_graph_replay
            # A plain 2-D bf16 CUDA batch, with weights in the same dtype.
            and hidden_states.dim() == 2
            and hidden_states.shape[0] > 0
            and hidden_states.is_cuda
            and dtype is torch.bfloat16
            and hidden_states.is_contiguous()
            and self.o_proj.tp_size == 1
            and not self.q_proj.use_fp8
            and not self.kv_a_proj_with_mqa.use_fp8
            and not self.kv_b_proj.use_fp8
            and not self.o_proj.use_fp8
            and self.q_proj.bias is None
            and self.kv_a_proj_with_mqa.bias is None
            and self.kv_b_proj.bias is None
            and self.o_proj.bias is None
            and self.q_proj.weight.dtype is dtype
            and self.kv_a_proj_with_mqa.weight.dtype is dtype
            and self.kv_b_proj.weight.dtype is dtype
            and self.o_proj.weight.dtype is dtype
            # The fused kernel applies the general weighted norm, so it needs the
            # weight to exist and to round-trip like the vendored kernel's.
            and norm.elementwise_affine
            and norm.weight.dtype is dtype
            and norm.weight.device == hidden_states.device
            and norm.weight.is_contiguous()
            and norm.weight.numel() == self.kv_lora_rank
        )

    # ------------------------------------------------------------------
    # Lazily derived weights
    # ------------------------------------------------------------------

    def _fast_path_weights(self) -> tuple:
        """``(q_t, kv_a_t, kv_b_t, packed_kv_b, o_t)``, cached behind one key.

        The four transposed weights must stay *views*. ``torch.mm`` costs
        measurably less host time than ``F.linear`` but wants the ``[in, out]``
        orientation; ``F.linear(x, W)`` and ``torch.mm(x, W.t())`` are
        bit-identical and hit the same cuBLAS layout, while
        ``torch.mm(x, W.t().contiguous())`` flips cuBLAS off the TN layout it
        prefers and costs throughput on the large shape, where ``q_proj`` and
        ``o_proj`` run at 97-98% tensor-pipe utilization.

        One combined key over all four parameters, rather than one key per
        derived tensor, so a forward validates the whole cache in a single pass.
        """
        w_q = self.q_proj.weight
        w_kv_a = self.kv_a_proj_with_mqa.weight
        w_kv_b = self.kv_b_proj.weight
        w_o = self.o_proj.weight
        # The layout ints go in the key as well: the packed matrix is laid out
        # per head, and ``kv_b_proj.weight``'s shape alone does not pin the split
        # (heads x (nope + value) has more than one factorization).
        key = (_weight_identity(w_q), _weight_identity(w_kv_a),
               _weight_identity(w_kv_b), _weight_identity(w_o),
               self.num_local_heads, self.kv_lora_rank, self.qk_rope_head_dim,
               self.qk_nope_head_dim, self.v_head_dim)
        cached = self._derived_weights
        if cached is not None and cached[0] == key:
            return cached[1]
        derived = (w_q.t(), w_kv_a.t(), w_kv_b.t(),
                   self._build_packed_kv_b(w_kv_b), w_o.t())
        self._derived_weights = (key, derived)
        return derived

    def _build_packed_kv_b(self, weight: torch.Tensor) -> torch.Tensor:
        """``[576, 32 * 320]`` matrix folding the ``k`` build into the ``kv_b`` GEMM.

        With ``weight`` viewed as ``[heads, nope + value, latent]`` -- the same
        decomposition ``compute_absorbed_weights`` uses -- each head gets a
        320-column block:

            P[    :512, h,   0:128] = W[h, :128, :].T   k_nope
            P[ 512:576, h, 128:192] = I(64)             k_pe passthrough
            P[    :512, h, 192:320] = W[h, 128:, :].T   v

        so ``mm(latent, P).view(N, heads, 320)`` yields ``k = [..., :192]`` and
        ``v = [..., 192:]`` as slices of one buffer, with no copy. The rope lane
        is exact for finite inputs -- one exact ``1.0 * k_pe`` term plus 512 exact
        zeros, accumulated in fp32 before the store, so reduction order cannot
        matter. The nope and value lanes are within about one bf16 ulp rather
        than bit-identical, because widening K from 512 to 576 with zero padding
        can change cuBLAS's tile and split-k choice.

        Zero padding does not isolate the lanes for non-finite values: a NaN or
        Inf anywhere in ``latent`` reaches every lane through ``NaN * 0``,
        whereas the baseline's separate GEMM and copy confine it to its own lane.
        Signed zero is likewise not preserved. The benchmark rejects any output
        containing NaN or Inf on either side, so this changes which of two
        already-failing answers appears, not whether the operator is correct.

        Costs ``576 x 320`` against ``512 x 256`` MACs per head, i.e. +41% on the
        smallest of the three projections (about +1.5% of total GPU time at
        N = 16384), and 11.8 MB of persistent bf16 weight.
        """
        heads = self.num_local_heads
        latent = self.kv_lora_rank
        rope = self.qk_rope_head_dim
        nope = self.qk_nope_head_dim
        value = self.v_head_dim
        qk = nope + rope
        assert weight.shape == (heads * (nope + value), latent), (
            f"kv_b_proj.weight is {tuple(weight.shape)}, expected "
            f"{(heads * (nope + value), latent)}"
        )

        blocks = weight.view(heads, nope + value, latent)
        packed = torch.zeros(
            latent + rope, heads, qk + value,
            dtype=weight.dtype, device=weight.device,
        )
        packed[:latent, :, :nope] = blocks[:, :nope, :].permute(2, 0, 1)
        packed[latent:, :, nope:qk] = torch.eye(
            rope, dtype=weight.dtype, device=weight.device,
        ).unsqueeze(1)
        packed[:latent, :, qk:] = blocks[:, nope:, :].permute(2, 0, 1)
        return packed.reshape(latent + rope, heads * (qk + value))

    def _prefill_seq_lens(
        self, cu_seqlens_q: torch.Tensor, max_seqlen_q: int, num_tokens: int,
    ) -> torch.Tensor:
        """Per-sequence KV lengths for the ragged kernel, without a new dispatch.

        For a single sequence -- which is every batch the benchmark builds -- the
        answer is host-known (``[num_tokens]``), and deriving it as
        ``cu_seqlens_q[1:] - cu_seqlens_q[:-1]`` would cost a device dispatch
        that is material against this launch budget. The cache is keyed on host
        integers only. It must not be keyed on the ``Context``: that is a
        non-frozen ``eq=True`` dataclass and therefore unhashable, every setter
        rebinds a fresh instance, and CPython recycles addresses, so an
        ``id``-keyed memo can report a false hit.

        Substituting ``[num_tokens]`` is only sound if that single segment really
        spans the whole batch. Reading ``cu_seqlens_q[1]`` to confirm would cost
        the device round trip this exists to avoid, but ``max_seqlen_q`` carries
        the same number on the host: for one segment it *is* the segment length.
        A context describing fewer tokens than the tensor holds therefore falls
        back to the subtraction rather than silently attending the padding.
        """
        if cu_seqlens_q.numel() != 2 or max_seqlen_q != num_tokens:
            return (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)

        cache = self._single_seq_lens
        if cache is None:
            cache = self._single_seq_lens = {}
        key = (num_tokens, cu_seqlens_q.device)
        seq_lens = cache.get(key)
        if seq_lens is None:
            if len(cache) >= self._SEQ_LENS_CACHE_LIMIT:
                cache.clear()
            seq_lens = torch.tensor(
                [num_tokens], dtype=torch.int32, device=cu_seqlens_q.device,
            )
            cache[key] = seq_lens
        return seq_lens

    # ------------------------------------------------------------------
    # Attention backend -- the single switch point
    # ------------------------------------------------------------------

    def _dense_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx,
        num_tokens: int,
    ) -> torch.Tensor:
        """Causal attention over the new prefill tokens, ``[N, heads, v_head_dim]``."""
        global _ragged_prefill_broken

        if self._prefill_backend == _RAGGED_PREFILL and _ragged_prefill_usable():
            # The trtllm-gen kernel *ignores* q's row stride, so a q whose rows
            # are not tightly packed comes back silently wrong rather than
            # raising. Assert instead of trusting the layout. This is also why
            # q_proj and kv_a_proj_with_mqa stay two GEMMs: fusing them would
            # make q a column slice of a wider output.
            assert q.stride() == (q.shape[1] * q.shape[2], q.shape[2], 1), (
                f"trtllm-gen ragged prefill needs a row-contiguous q, got "
                f"stride {q.stride()} for shape {tuple(q.shape)}"
            )
            # Take the scale from ``self.attn``, which is what the baseline's
            # attention actually uses, rather than from ``self.scaling`` -- they
            # are equal by construction, and reading the one that matters keeps
            # them from drifting apart. The cached wrapper bakes the scale in at
            # construction, so it is rebuilt if the scale ever changes.
            scale = self.attn.scale
            op = self._ragged_prefill_op
            if op is None or op.scale != scale:
                op = TrtllmRaggedPrefill(scale=scale)
                object.__setattr__(self, "_ragged_prefill_op", op)
            try:
                return op(
                    q, k, v,
                    seq_lens=self._prefill_seq_lens(
                        ctx.cu_seqlens_q, ctx.max_seqlen_q, num_tokens),
                    cu_seq_lens_q=ctx.cu_seqlens_q,
                    cu_seq_lens_kv=ctx.cu_seqlens_q,
                    max_q_len=ctx.max_seqlen_q,
                    max_kv_len=ctx.max_seqlen_q,
                    is_causal=True,
                    return_lse=False,
                )
            except Exception:
                _ragged_prefill_broken = True

        # The baseline's own FlashAttention wrapper, reused rather than
        # re-imported so it keeps passing the FA version the baseline selects. It
        # accepts both the strided k/v slices and a strided q.
        return self.attn.varlen_attn(
            q, k, v,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_q,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_q,
            softmax_scale=self.attn.scale,
            causal=True,
            return_softmax_lse=False,
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state_manager=None,
    ) -> torch.Tensor:
        ctx = get_context()
        if not self._fast_path_applies(hidden_states, ctx):
            return super().forward(
                hidden_states, positions=positions, state_manager=state_manager,
            )
        del positions, state_manager

        num_tokens = hidden_states.shape[0]
        heads = self.num_local_heads
        latent_dim = self.kv_lora_rank
        rope_dim = self.qk_rope_head_dim
        qk = self.qk_head_dim
        value = self.v_head_dim
        w_q, w_kv_a, w_kv_b, packed_kv_b, w_o = self._fast_path_weights()

        q = torch.mm(hidden_states, w_q).view(num_tokens, heads, qk)
        kv = torch.mm(hidden_states, w_kv_a)

        if self._fuse_norm_and_pack:
            latent = _fused_rms_norm_pack(
                kv, self.kv_a_layernorm.weight, self.kv_a_layernorm.eps,
                latent_dim,
            )
            kv_c = latent.narrow(1, 0, latent_dim)
            k_pe = latent.narrow(1, latent_dim, rope_dim).unsqueeze(1)
        else:
            kv_c, k_pe = kv.split([latent_dim, rope_dim], dim=-1)
            kv_c = self.kv_a_layernorm(kv_c)
            k_pe = k_pe.unsqueeze(1)
            latent = None

        if self._fold_rope_into_kv_b:
            assert latent is not None, (
                "the packed kv_b GEMM consumes the contiguous latent the fused "
                "norm produces; it cannot run with the norm unfused"
            )
            packed = torch.mm(latent, packed_kv_b).view(
                num_tokens, heads, qk + value,
            )
            k = packed.narrow(2, 0, qk)
            v = packed.narrow(2, qk, value)
        else:
            up = torch.mm(kv_c, w_kv_b).view(
                num_tokens, heads, self.qk_nope_head_dim + value,
            )
            k_nope, v = up.split([self.qk_nope_head_dim, value], dim=-1)
            k = self.attn._concat_k_nope_k_pe(k_nope, k_pe)

        attn_out = self._dense_prefill(q, k, v, ctx, num_tokens)
        return torch.mm(attn_out.reshape(num_tokens, heads * value), w_o)
