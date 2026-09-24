"""Rotary position embeddings with YaRN / YARN scaling.

Two variants:
  - ``YaRNRotaryEmbedding``: NeoX-style YaRN RoPE used by GPT-OSS. Applies
    magnitude correction via ``mscale``.
  - ``YarnRotaryEmbedding``: DeepSeek-style YARN RoPE (interleaved, NON-NeoX)
    used by DeepSeek V3.  Supports separate ``mscale`` / ``mscale_all_dim``
    knobs and exposes ``softmax_mscale`` as an attention scaling factor.

Both classes share the same L1 CUDA rotary kernel via
``torch.ops.fastkernels_rope.rotary_embedding``; the only differences are how
the ``cos_sin_cache`` is computed and whether NeoX layout is used.

References:
  - Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models"
  - vLLM: ``vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py``
  - vLLM: ``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Detect FlashInfer rotary op once at import time.  vLLM's
# ``torch.ops.vllm.flashinfer_rotary_embedding`` is a thin wrapper around
# ``flashinfer.rope.apply_rope_with_cos_sin_cache_inplace`` (see
# ``vllm/model_executor/layers/rotary_embedding/common.py``), so we call the
# FlashInfer package directly instead of importing vllm to register the op.
try:
    from flashinfer.rope import (
        apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope,
    )
    _USE_FLASHINFER_ROPE = True
except Exception:
    _flashinfer_apply_rope = None
    _USE_FLASHINFER_ROPE = False


# GLM-5.2's plain "default" rope is applied via the vendored vLLM rotary
# kernel (base RotaryEmbedding.forward_cuda), exposed as the
# ``torch.ops.fastkernels_rope.rotary_embedding`` custom op registered by
# ``rotary_emb``. Importing it here ensures the op is defined.
from .rotary_emb import RotaryEmbedding


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float,
    max_position_embeddings: int, truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale: float) -> float:
    """GPT-OSS style mscale (no explicit mscale parameter)."""
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def yarn_get_mscale(scale: float, mscale: float) -> float:
    """DeepSeek-style mscale with explicit parameter (matches vLLM)."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ---------------------------------------------------------------------------
# Fused NeoX rotary kernel (hand-written CUDA).
#
# Two things dominate this operator as the baseline writes it:
#
#   1. ``cos_sin_cache`` is built in fp32 with ``max_position_embeddings *
#      scaling_factor`` rows (GPT-OSS: 131072 * 32 = 4.2M rows x 64 = 1.07 GB).
#      The baseline re-casts that whole buffer to ``query.dtype`` on *every*
#      forward -- ~1.6 GB of HBM traffic (~230 us of the 280 us baseline) to
#      produce a table from which a 60-row batch reads 7.5 kB. Casting once at
#      first use and caching the result is bit-identical (fp32 -> bf16 rounds
#      the same whenever it happens) and removes that cost entirely.
#
#   2. The vendored vLLM kernel touches ``query``/``key`` two bytes at a time
#      (one scalar load + store per rotated element). The rotation is pure
#      streaming, so it wants the widest access the layout allows: with
#      ``head_size == rot_dim == 64`` each half of a head is 32 contiguous
#      elements = 64 B, so one thread can carry 8 elements (a 16 B vector) of
#      the x-half together with its 8 partners in the y-half.
#
# So the kernel assigns one *vector unit* -- (head, 8-element slice of the
# x-half) -- to each thread over a flat grid, moving query, key and the cos/sin
# row with 16 B accesses only.
#
# Query and key are covered by *disjoint* block ranges of a single launch
# rather than by the same threads. Letting one thread rotate both (the obvious
# formulation, since a 32-head/4-kv-head row makes only 1 in 8 threads do key
# work) costs 40 registers/thread and caps occupancy at 75%; splitting the two
# makes every block's work uniform, fits in 32 registers, reaches 100%
# theoretical / 82% achieved occupancy, and measured ~1.16x faster at 16k
# tokens. It still costs only one launch, which is what the small decode
# shapes -- the overwhelming majority of calls -- are bound by.
#
# Things that were tried and measured *slower*, for the record: pairing x/y
# across lanes with ``__shfl_xor`` so each warp access covers whole 128 B
# cache lines (halves L1TEX line touches but the extra shuffles and duplicated
# cos/sin conversions cost more than they save); staging cos/sin in shared
# memory; holding the table in fp32 to skip the bf16 unpack; and 2 or 4 heads
# per thread. bf16x2 SIMD math is faster still but loses ~6e-2 of accuracy, so
# it is not used -- every path here is bit-identical to the baseline.
# ---------------------------------------------------------------------------

_ROPE_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace fkrope {

// 16 B payload moved as a single vector access.
struct alignas(16) V16 { uint4 raw; };

template <typename T> struct Cvt;

template <> struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float to(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from(float v) { return __float2bfloat16(v); }
};
template <> struct Cvt<__half> {
  static __device__ __forceinline__ float to(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from(float v) { return __float2half(v); }
};
template <> struct Cvt<float> {
  static __device__ __forceinline__ float to(float v) { return v; }
  static __device__ __forceinline__ float from(float v) { return v; }
};

// ---------------------------------------------------------------------------
// One vector unit of one tensor: rotate the 16 B x-slice at (token, head,
// slice) against its y partner, NeoX layout.
//
// `i` enumerates units as token * units_per_token + unit, and
// unit = head * units_per_head + slice; both counts are powers of two (checked
// on the host) so the split is shifts and masks.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void rope_unit(
    int64_t i, const int64_t* __restrict__ positions, T* __restrict__ base,
    const T* __restrict__ cache, int64_t row_stride, int embed_dim,
    int64_t head_stride, int tok_shift, int unit_mask, int upn_shift,
    int slice_mask) {
  constexpr int NV = 16 / sizeof(T);
  const int token = (int)(i >> tok_shift);
  const int unit  = (int)(i & unit_mask);
  const int slice = unit & slice_mask;
  const int head  = unit >> upn_shift;

  // Issue the payload loads first: they do not depend on positions[], so they
  // overlap the positions -> cos/sin address dependency.
  T* rb = base + (int64_t)token * row_stride + head * head_stride + slice * NV;
  V16 xv = *(const V16*)rb;
  V16 yv = *(const V16*)(rb + embed_dim);

  const int64_t pos = positions[token];
  const T* crow = cache + pos * (int64_t)(2 * embed_dim);
  const V16 cv = *(const V16*)(crow + slice * NV);
  const V16 sv = *(const V16*)(crow + embed_dim + slice * NV);

  T* xs = (T*)&xv;
  T* ys = (T*)&yv;
  const T* cs = (const T*)&cv;
  const T* ss = (const T*)&sv;
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const float x = Cvt<T>::to(xs[j]), y = Cvt<T>::to(ys[j]);
    const float c = Cvt<T>::to(cs[j]), s = Cvt<T>::to(ss[j]);
    xs[j] = Cvt<T>::from(x * c - y * s);
    ys[j] = Cvt<T>::from(y * c + x * s);
  }
  *(V16*)rb = xv;
  *(V16*)(rb + embed_dim) = yv;
}

// Blocks [0, q_blocks) rotate query; the rest rotate key.
template <typename T, bool HAS_KEY>
__global__ void rope_neox_vec(
    const int64_t* __restrict__ positions, T* __restrict__ query,
    T* __restrict__ key, const T* __restrict__ cache, int64_t q_stride,
    int64_t k_stride, int embed_dim, int64_t head_stride, int q_tok_shift,
    int q_unit_mask, int k_tok_shift, int k_unit_mask, int upn_shift,
    int slice_mask, int64_t q_total, int64_t k_total, int q_blocks) {
  if (!HAS_KEY || blockIdx.x < (unsigned int)q_blocks) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < q_total)
      rope_unit<T>(i, positions, query, cache, q_stride, embed_dim, head_stride,
                   q_tok_shift, q_unit_mask, upn_shift, slice_mask);
  } else {
    const int64_t i = (int64_t)(blockIdx.x - q_blocks) * blockDim.x + threadIdx.x;
    if (i < k_total)
      rope_unit<T>(i, positions, key, cache, k_stride, embed_dim, head_stride,
                   k_tok_shift, k_unit_mask, upn_shift, slice_mask);
  }
}

// ---------------------------------------------------------------------------
// Scalar fallback for any head_size / alignment / layout the vector path
// rejects. One block per token, grid-stride over (head, rot_offset).
// ---------------------------------------------------------------------------
template <typename T, bool IS_NEOX>
__device__ __forceinline__ void rope_one(T* __restrict__ arr,
                                         const T* __restrict__ crow,
                                         int rot_offset, int embed_dim) {
  const int xi = IS_NEOX ? rot_offset : 2 * rot_offset;
  const int yi = IS_NEOX ? embed_dim + rot_offset : 2 * rot_offset + 1;
  const float c = Cvt<T>::to(crow[rot_offset]);
  const float s = Cvt<T>::to(crow[embed_dim + rot_offset]);
  const float x = Cvt<T>::to(arr[xi]), y = Cvt<T>::to(arr[yi]);
  arr[xi] = Cvt<T>::from(x * c - y * s);
  arr[yi] = Cvt<T>::from(y * c + x * s);
}

template <typename T, bool IS_NEOX>
__global__ void rope_scalar(const int64_t* __restrict__ positions,
                            T* __restrict__ query, T* __restrict__ key,
                            const T* __restrict__ cache, int64_t q_stride,
                            int64_t k_stride, int embed_dim, int64_t head_stride,
                            int num_heads, int num_kv_heads) {
  const int token = blockIdx.x;
  const int64_t pos = positions[token];
  const T* crow = cache + pos * (int64_t)(2 * embed_dim);
  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head = i / embed_dim;
    rope_one<T, IS_NEOX>(query + (int64_t)token * q_stride + head * head_stride,
                         crow, i - head * embed_dim, embed_dim);
  }
  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head = i / embed_dim;
      rope_one<T, IS_NEOX>(key + (int64_t)token * k_stride + head * head_stride,
                           crow, i - head * embed_dim, embed_dim);
    }
  }
}

static inline bool is_pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }
static inline int ilog2(int64_t v) { int n = 0; while ((int64_t(1) << n) < v) ++n; return n; }

// 128 threads/block measured best at 16k tokens and is indistinguishable from
// 64/256 on the decode shapes.
constexpr int kBlock = 128;

template <typename T>
static void launch(const int64_t* pos, T* q, T* k, const T* cache,
                   int64_t num_tokens, int64_t q_stride, int64_t k_stride,
                   int64_t head_stride, int embed_dim, int num_heads,
                   int num_kv_heads, bool is_neox, cudaStream_t stream) {
  constexpr int NV = 16 / sizeof(T);
  const int64_t upn = embed_dim / NV;             // 16 B units per head half
  const int64_t q_units = upn * num_heads;
  const int64_t k_units = upn * num_kv_heads;

  // Every 16 B access lands at base + {token*row_stride, head*head_stride,
  // slice*NV, embed_dim}, so each term must be a whole number of NV elements.
  const bool aligned =
      ((uintptr_t)q % 16 == 0) && (k == nullptr || (uintptr_t)k % 16 == 0) &&
      ((uintptr_t)cache % 16 == 0) && (q_stride % NV == 0) &&
      (k_stride % NV == 0) && (head_stride % NV == 0) && (embed_dim % NV == 0);

  if (is_neox && aligned && upn >= 1 && is_pow2(upn) && is_pow2(q_units) &&
      (k == nullptr || is_pow2(k_units))) {
    const int64_t q_total = num_tokens * q_units;
    const int64_t k_total = k ? num_tokens * k_units : 0;
    const int q_blocks = (int)((q_total + kBlock - 1) / kBlock);
    const int k_blocks = (int)((k_total + kBlock - 1) / kBlock);
    const int upn_shift = ilog2(upn);
    if (k == nullptr) {
      rope_neox_vec<T, false><<<q_blocks, kBlock, 0, stream>>>(
          pos, q, k, cache, q_stride, k_stride, embed_dim, head_stride,
          ilog2(q_units), (int)(q_units - 1), 0, 0, upn_shift, (int)(upn - 1),
          q_total, 0, q_blocks);
    } else {
      rope_neox_vec<T, true><<<q_blocks + k_blocks, kBlock, 0, stream>>>(
          pos, q, k, cache, q_stride, k_stride, embed_dim, head_stride,
          ilog2(q_units), (int)(q_units - 1), ilog2(k_units),
          (int)(k_units - 1), upn_shift, (int)(upn - 1), q_total, k_total,
          q_blocks);
    }
    return;
  }

  const int nthreads = (int)std::min<int64_t>(
      std::max<int64_t>((int64_t)num_heads * embed_dim, 32), 512);
  if (is_neox) {
    rope_scalar<T, true><<<(int)num_tokens, nthreads, 0, stream>>>(
        pos, q, k, cache, q_stride, k_stride, embed_dim, head_stride, num_heads,
        num_kv_heads);
  } else {
    rope_scalar<T, false><<<(int)num_tokens, nthreads, 0, stream>>>(
        pos, q, k, cache, q_stride, k_stride, embed_dim, head_stride, num_heads,
        num_kv_heads);
  }
}

}  // namespace fkrope

void fk_rope(const torch::Tensor& positions, torch::Tensor& query,
             std::optional<torch::Tensor> key, int64_t head_size,
             const torch::Tensor& cos_sin_cache, bool is_neox) {
  const int64_t num_tokens = positions.numel();
  if (num_tokens == 0) return;

  const int64_t num_heads = (query.numel() / num_tokens) / head_size;
  const bool has_key = key.has_value() && key->numel() > 0;
  const int64_t num_kv_heads =
      has_key ? (key->numel() / num_tokens) / head_size : num_heads;

  const int64_t rot_dim = cos_sin_cache.size(-1);
  TORCH_CHECK(rot_dim == head_size, "fk_rope expects rot_dim == head_size, got ",
              rot_dim, " vs ", head_size);
  TORCH_CHECK(positions.scalar_type() == at::kLong,
              "fk_rope expects int64 positions");
  TORCH_CHECK(cos_sin_cache.scalar_type() == query.scalar_type(),
              "fk_rope expects cos_sin_cache in query dtype");
  const int embed_dim = (int)(rot_dim / 2);

  const int seq_dim = (int)positions.dim() - 1;
  const int64_t q_stride = query.stride(seq_dim);
  const int64_t k_stride = has_key ? key->stride(seq_dim) : 0;
  // Flat [*, heads*head_size] rows keep heads head_size apart; an explicit
  // [*, heads, head_size] layout carries its own stride (matches vLLM).
  const int64_t head_stride =
      (query.dim() == positions.dim() + 2) ? query.stride(-2) : head_size;

  const c10::cuda::OptionalCUDAGuard guard(at::device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define FK_ROPE_DISPATCH(CT, TT)                                              \
  fkrope::launch<TT>(                                                         \
      positions.const_data_ptr<int64_t>(),                                    \
      reinterpret_cast<TT*>(query.data_ptr()),                                \
      has_key ? reinterpret_cast<TT*>(key->data_ptr()) : nullptr,             \
      reinterpret_cast<const TT*>(cos_sin_cache.const_data_ptr()), num_tokens,\
      q_stride, k_stride, head_stride, embed_dim, (int)num_heads,              \
      (int)num_kv_heads, is_neox, stream)

  switch (query.scalar_type()) {
    case at::kBFloat16: FK_ROPE_DISPATCH(kBFloat16, __nv_bfloat16); break;
    case at::kHalf:     FK_ROPE_DISPATCH(kHalf, __half); break;
    case at::kFloat:    FK_ROPE_DISPATCH(kFloat, float); break;
    default:
      TORCH_CHECK(false, "fk_rope: unsupported dtype ", query.scalar_type());
  }
#undef FK_ROPE_DISPATCH
}
"""

_ROPE_CPP = r"""
#include <torch/extension.h>
void fk_rope(const torch::Tensor& positions, torch::Tensor& query,
             std::optional<torch::Tensor> key, int64_t head_size,
             const torch::Tensor& cos_sin_cache, bool is_neox);
"""

_ROPE_EXT = None


def _rope_ext():
    """JIT-build (once per process) and return the rotary extension."""
    global _ROPE_EXT
    if _ROPE_EXT is None:
        # Importing cuda_ext pins TORCH_CUDA_ARCH_LIST to the local GPU.
        try:
            from ....infra import cuda_ext as _cuda_ext  # noqa: F401
        except Exception:
            pass
        from torch.utils.cpp_extension import load_inline
        _ROPE_EXT = load_inline(
            name="fk_yarn_rope_v3",
            cpp_sources=_ROPE_CPP,
            cuda_sources=_ROPE_CUDA,
            functions=["fk_rope"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    return _ROPE_EXT


class YaRNRotaryEmbedding(nn.Module):
    """YaRN RoPE with precomputed cos/sin cache.

    NeoX layout, as used by GPT-OSS. The rotation runs through the fused
    ``fk_rope`` kernel above, against a cos/sin table cast to the activation
    dtype once instead of on every call.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        rotary_dim = head_dim

        pos_freqs = rope_theta ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, rope_theta,
            original_max_position_embeddings, truncate,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
        )
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        mscale = _yarn_get_mscale(scaling_factor)

        max_t = int(max_position_embeddings * scaling_factor)
        t = torch.arange(max_t, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        # Activation-dtype view of the table, built on first use (the module is
        # on its device by then, so the cast runs on the GPU). Held in __dict__
        # rather than as a buffer so the hot path reads it without going
        # through nn.Module.__getattr__.
        self._cache_cast = None
        self._rope = _rope_ext().fk_rope

    def _cast_cache(self, query: torch.Tensor) -> torch.Tensor:
        cache = self.cos_sin_cache
        if cache.device != query.device:
            cache = cache.to(query.device)
            self.cos_sin_cache = cache
        cast = cache if cache.dtype == query.dtype else cache.to(query.dtype)
        self._cache_cast = cast
        return cast

    def forward(self, positions, query, key):
        if torch.compiler.is_compiling():
            c = self.cos_sin_cache
            if c.dtype != query.dtype:
                c = c.to(query.dtype)
            return RotaryEmbedding.forward_native(
                positions, query, key, self.head_dim, c,
            )
        cache = self._cache_cast
        if (
            cache is None
            or cache.dtype is not query.dtype
            or cache.device != query.device
        ):
            cache = self._cast_cache(query)
        self._rope(positions, query, key, self.head_dim, cache, True)
        return query, key


class YarnRotaryEmbedding(nn.Module):
    """DeepSeek-style YARN (Yet Another RoPE extensioN) RoPE.

    Uses NON-NeoX (interleaved) layout, matching vLLM's
    ``DeepseekScalingRotaryEmbedding``.  The cos/sin cache is scaled by
    ``softmax_mscale`` which folds the attention magnitude correction into
    the rotary cache (so attention scores do not need to multiply by
    ``softmax_mscale`` separately).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        scaling_factor: float,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        is_neox_style: bool = False,
        is_plain: bool = False,
        cache_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        # ``is_plain`` marks a degenerate (scaling_factor==1.0) instance that is
        # really standard RoPE — e.g. GLM-5.2's ``rope_type: "default"``. vLLM
        # maps a "default" rope to the base ``RotaryEmbedding``, which does NOT
        # use the FlashInfer kernel and casts the cos/sin cache to the model
        # dtype (bf16). DeepSeek-V3.2 YARN (scaling_factor>1) keeps FlashInfer +
        # fp32 cache. Threading this flag lets both match vLLM exactly.
        self.is_plain = is_plain
        rotary_dim = head_dim
        base = rope_theta

        softmax_mscale = (
            yarn_get_mscale(scaling_factor, mscale)
            / yarn_get_mscale(scaling_factor, mscale_all_dim)
            * attn_factor
        )
        self.softmax_mscale = softmax_mscale

        pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rotary_dim, base, max_position_embeddings,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
        ) * extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        t = torch.arange(max_position_embeddings * scaling_factor, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * softmax_mscale
        sin = freqs.sin() * softmax_mscale
        cache = torch.cat((cos, sin), dim=-1).float()
        # Plain "default" rope (GLM-5.2): vLLM's base ``RotaryEmbedding`` stores
        # the cos/sin cache in the model compute dtype (bf16) once at init, so
        # its forward never re-casts. Match that — computing in fp32 then
        # casting to bf16 here is bit-identical to casting per-forward, and
        # skips a full-cache dtype conversion on every rope call. YARN
        # (is_plain=False) keeps the fp32 cache for the FlashInfer path.
        if self.is_plain and cache_dtype is not None:
            cache = cache.to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        # vLLM's ``DeepseekScalingRotaryEmbedding.forward_cuda`` prefers the
        # FlashInfer fused kernel when available (see
        # ``vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:181-198``).
        # FlashInfer keeps ``cos_sin_cache`` in float32; only the fastkernels
        # CUDA kernel needs the cache cast to query.dtype.
        if _USE_FLASHINFER_ROPE and not self.is_plain \
                and query.dtype in (torch.float16, torch.bfloat16) \
                and self.head_dim in (64, 128, 256, 512):
            # Mirrors vLLM's ``flashinfer_rotary_embedding`` custom op, which
            # just forwards to this FlashInfer entry point in-place.
            _flashinfer_apply_rope(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_dim,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
            )
            return query, key
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        # GLM-5.2 plain "default" rope and the scaled path both go through the
        # vendored vLLM rotary kernel. Call it via the registered
        # ``fastkernels_rope`` custom op (whose CUDA impl is exactly
        # ``_C.rotary_embedding``) rather than the raw pybind function, so
        # ``torch.compile`` / cudagraph capture can trace it. Numerically
        # identical to calling ``_C.rotary_embedding`` directly.
        torch.ops.fastkernels_rope.rotary_embedding(
            positions, query, key, self.head_dim, cache, self.is_neox_style,
        )
        return query, key
