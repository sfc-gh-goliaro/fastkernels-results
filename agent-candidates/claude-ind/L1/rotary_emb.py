"""Rotary position embeddings (RoPE), with optional Llama 3.1-style frequency scaling.

Optimized over the vLLM-derived baseline in two places:

* **Host side.** The baseline re-casts its fp32 ``cos_sin_cache`` to the
  activation dtype on *every* forward (for Llama-3.1 that is a 67 MB read plus a
  34 MB write -- ~13 us of pure HBM traffic that dwarfs the rotation itself for
  decode-sized batches). Here the cast happens once and is memoized, and the
  kernel is reached through a single pybind call instead of the
  ``torch.library`` dispatcher.

* **Device side.** A NeOX-specialized kernel moves 16 bytes per instruction:
  each thread owns one ``(x, x + embed_dim)`` vector pair of eight 16-bit
  elements, so a head's 256 bytes are read and written as fully-coalesced
  128-byte segments. The launch geometry splits each token's pairs across
  several blocks when the token count alone would not fill the GPU, which is the
  common decode case. A scalar kernel covers everything the fast path does not
  (interleaved/GPT-J style, partial rotary dims, fp32 activations, odd strides).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Importing the op loader pins TORCH_CUDA_ARCH_LIST to the local arch, which is
# what load_inline() below picks up.
from fastkernels.infra import cuda_ext as _cuda_ext  # noqa: F401
from torch.utils.cpp_extension import load_inline

_IS_COMPILING = torch.compiler.is_compiling

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <torch/all.h>

namespace fkrope {

// ---------------------------------------------------------------------------
// Fast path: NeOX style, rot_dim == head_size, 16-bit activations, cache dtype
// == activation dtype, embed_dim a multiple of 8 with embed_dim/8 a power of
// two.  One thread handles one vector pair: 8 elements at x and 8 at
// x + embed_dim, i.e. two 16-byte loads and two 16-byte stores.
//
// grid = (num_tokens, pair_chunks), block = (pairs_per_block).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void rope_neox_v8_kernel(
    const int64_t* __restrict__ positions,
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    const int64_t query_stride,
    const int64_t key_stride,
    const int rot_dim,
    const int embed_dim,
    const int nq_pairs,
    const int tot_pairs,
    const int pph_shift) {
  int p = blockIdx.y * blockDim.x + threadIdx.x;
  if (p >= tot_pairs) return;

  const int token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];
  const T* __restrict__ cos_ptr = cos_sin_cache + pos * rot_dim;

  T* base;
  if (p < nq_pairs) {
    base = query + token_idx * query_stride;
  } else {
    p -= nq_pairs;
    base = key + token_idx * key_stride;
  }

  const int j = p & ((1 << pph_shift) - 1);
  const int x_off = (p >> pph_shift) * rot_dim + (j << 3);

  T* xp = base + x_off;
  T* yp = xp + embed_dim;

  uint4 xv = *reinterpret_cast<const uint4*>(xp);
  uint4 yv = *reinterpret_cast<const uint4*>(yp);
  const uint4 cv = *reinterpret_cast<const uint4*>(cos_ptr + (j << 3));
  const uint4 sv =
      *reinterpret_cast<const uint4*>(cos_ptr + embed_dim + (j << 3));

  T* xe = reinterpret_cast<T*>(&xv);
  T* ye = reinterpret_cast<T*>(&yv);
  const T* ce = reinterpret_cast<const T*>(&cv);
  const T* se = reinterpret_cast<const T*>(&sv);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float xf = static_cast<float>(xe[i]);
    const float yf = static_cast<float>(ye[i]);
    const float cf = static_cast<float>(ce[i]);
    const float sf = static_cast<float>(se[i]);
    xe[i] = static_cast<T>(xf * cf - yf * sf);
    ye[i] = static_cast<T>(yf * cf + xf * sf);
  }

  *reinterpret_cast<uint4*>(xp) = xv;
  *reinterpret_cast<uint4*>(yp) = yv;
}

// ---------------------------------------------------------------------------
// General path: scalar, mirrors the vLLM reference semantics exactly.
// ---------------------------------------------------------------------------
template <typename T, bool IS_NEOX>
__global__ void rope_generic_kernel(
    const int64_t* __restrict__ positions,
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const int rot_dim,
    const int num_heads,
    const int num_kv_heads) {
  const int token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];
  const T* cache_ptr = cos_sin_cache + pos * rot_dim;
  const int embed_dim = rot_dim / 2;
  const T* cos_ptr = cache_ptr;
  const T* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i - head_idx * embed_dim;
    T* arr = query + token_idx * query_stride + head_idx * head_stride;
    const float cf = static_cast<float>(cos_ptr[rot_offset]);
    const float sf = static_cast<float>(sin_ptr[rot_offset]);
    const int xi = IS_NEOX ? rot_offset : 2 * rot_offset;
    const int yi = IS_NEOX ? embed_dim + rot_offset : 2 * rot_offset + 1;
    const float xf = static_cast<float>(arr[xi]);
    const float yf = static_cast<float>(arr[yi]);
    arr[xi] = static_cast<T>(xf * cf - yf * sf);
    arr[yi] = static_cast<T>(yf * cf + xf * sf);
  }

  if (key == nullptr) return;
  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i - head_idx * embed_dim;
    T* arr = key + token_idx * key_stride + head_idx * head_stride;
    const float cf = static_cast<float>(cos_ptr[rot_offset]);
    const float sf = static_cast<float>(sin_ptr[rot_offset]);
    const int xi = IS_NEOX ? rot_offset : 2 * rot_offset;
    const int yi = IS_NEOX ? embed_dim + rot_offset : 2 * rot_offset + 1;
    const float xf = static_cast<float>(arr[xi]);
    const float yf = static_cast<float>(arr[yi]);
    arr[xi] = static_cast<T>(xf * cf - yf * sf);
    arr[yi] = static_cast<T>(yf * cf + xf * sf);
  }
}

}  // namespace fkrope

#define FKROPE_DISPATCH(ST, ...)                                         \
  switch (ST) {                                                          \
    case at::kBFloat16: { using T = at::BFloat16; __VA_ARGS__; break; }  \
    case at::kHalf:     { using T = at::Half;     __VA_ARGS__; break; }  \
    case at::kFloat:    { using T = float;        __VA_ARGS__; break; }  \
    default:                                                             \
      TORCH_CHECK(false, "rope_fast: unsupported activation dtype");     \
  }

void rope_fast(const at::Tensor& positions, const at::Tensor& query,
               const std::optional<at::Tensor>& key_opt, int64_t head_size,
               const at::Tensor& cos_sin_cache, bool is_neox) {
  const int64_t num_tokens = positions.numel();
  if (num_tokens == 0) return;

  const at::Tensor* key = key_opt.has_value() ? &key_opt.value() : nullptr;
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  const int embed_dim = rot_dim / 2;
  const int pos_dim = static_cast<int>(positions.dim());
  TORCH_CHECK(pos_dim == 1 || pos_dim == 2,
              "rope_fast: positions must be [num_tokens] or [batch, seq_len]");
  const int seq_dim = pos_dim - 1;

  const int64_t query_stride = query.stride(seq_dim);
  const int64_t key_stride = key ? key->stride(seq_dim) : 0;
  const int64_t q_hidden = query.numel() / num_tokens;
  const int64_t k_hidden = key ? key->numel() / num_tokens : 0;
  TORCH_CHECK(q_hidden % head_size == 0 && k_hidden % head_size == 0,
              "rope_fast: hidden size not divisible by head_size");
  const int num_heads = static_cast<int>(q_hidden / head_size);
  const int num_kv_heads =
      key ? static_cast<int>(k_hidden / head_size) : num_heads;
  const int64_t head_stride =
      (query.dim() == pos_dim + 2) ? query.stride(-2) : head_size;

  const at::ScalarType st = query.scalar_type();
  TORCH_CHECK(cos_sin_cache.scalar_type() == st,
              "rope_fast: cache dtype must match activation dtype");
  TORCH_CHECK(positions.scalar_type() == at::kLong,
              "rope_fast: positions must be int64");

  const c10::cuda::OptionalCUDAGuard device_guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int pph = embed_dim >> 3;  // vector pairs per head
  const bool pph_pow2 = pph > 0 && (pph & (pph - 1)) == 0;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(query.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(cos_sin_cache.data_ptr()) % 16 == 0) &&
      (!key || reinterpret_cast<uintptr_t>(key->data_ptr()) % 16 == 0);

  const bool fast =
      is_neox && key != nullptr && pos_dim == 1
      && rot_dim == head_size && head_stride == rot_dim
      && (st == at::kBFloat16 || st == at::kHalf) && (rot_dim % 16 == 0)
      && pph_pow2 && aligned && (query_stride % 8 == 0) && (key_stride % 8 == 0)
      && query.stride(query.dim() - 1) == 1
      && key->stride(key->dim() - 1) == 1 && num_tokens <= 0x7fffffffLL;

  if (fast) {
    int pph_shift = 0;
    while ((1 << pph_shift) < pph) ++pph_shift;
    const int nq_pairs = num_heads * pph;
    const int tot_pairs = nq_pairs + num_kv_heads * pph;

    // 128 threads per block measured fastest across the whole shape range on
    // sm_100: it keeps enough blocks in flight to hide HBM latency at 16K
    // tokens (where wider blocks lose ~3%) without paying extra block-launch
    // overhead at decode sizes. Pairs beyond one block spill onto grid.y.
    int ppb = 128;
    if (tot_pairs < ppb) ppb = ((tot_pairs + 31) / 32) * 32;
    const int gy = (tot_pairs + ppb - 1) / ppb;

    const dim3 grid(static_cast<unsigned>(num_tokens),
                    static_cast<unsigned>(gy));
    FKROPE_DISPATCH(st, (fkrope::rope_neox_v8_kernel<T><<<grid, ppb, 0, stream>>>(
        positions.const_data_ptr<int64_t>(),
        static_cast<T*>(query.data_ptr()),
        static_cast<T*>(key->data_ptr()),
        static_cast<const T*>(cos_sin_cache.const_data_ptr()),
        query_stride, key_stride, rot_dim, embed_dim, nq_pairs, tot_pairs,
        pph_shift)));
    return;
  }

  const dim3 grid(static_cast<unsigned>(num_tokens));
  const int block = static_cast<int>(
      std::min<int64_t>(static_cast<int64_t>(num_heads) * embed_dim, 512));
  FKROPE_DISPATCH(st, {
    T* qp = static_cast<T*>(query.data_ptr());
    T* kp = key ? static_cast<T*>(key->data_ptr()) : nullptr;
    const T* cp = static_cast<const T*>(cos_sin_cache.const_data_ptr());
    if (is_neox) {
      fkrope::rope_generic_kernel<T, true><<<grid, block, 0, stream>>>(
          positions.const_data_ptr<int64_t>(), qp, kp, cp, query_stride,
          key_stride, head_stride, rot_dim, num_heads, num_kv_heads);
    } else {
      fkrope::rope_generic_kernel<T, false><<<grid, block, 0, stream>>>(
          positions.const_data_ptr<int64_t>(), qp, kp, cp, query_stride,
          key_stride, head_stride, rot_dim, num_heads, num_kv_heads);
    }
  });
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>

void rope_fast(const at::Tensor& positions, const at::Tensor& query,
               const std::optional<at::Tensor>& key_opt, int64_t head_size,
               const at::Tensor& cos_sin_cache, bool is_neox);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_fast", &rope_fast, "fused RoPE (CUDA)");
}
"""

# Bound to the compiled entry point on first use; a plain module global so the
# hot path is one LOAD_GLOBAL instead of an attribute chain.
_ROPE = None


def _build_ext():
    global _ROPE
    ext = load_inline(
        name="fk_rope_neox_v8",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "--expt-relaxed-constexpr"],
        verbose=False,
    )
    _ROPE = ext.rope_fast
    return _ROPE


def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wl - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = torch.zeros_like(inv_freq)
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )


class RotaryEmbedding(nn.Module):
    """RoPE with optional Llama 3.1-style frequency scaling.

    When rope_scaling_factor == 1.0 (default), behaves as standard RoPE.
    When rope_scaling_factor != 1.0, applies the Llama 3.1 piecewise
    frequency scaling controlled by low/high freq factors.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
        is_neox_style: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))

        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self._cs = None

    # Any device/dtype move invalidates the memoized activation-dtype cache.
    def _apply(self, *args, **kwargs):
        self._cs = None
        return super()._apply(*args, **kwargs)

    def _cast_cache(self, query: torch.Tensor) -> torch.Tensor:
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        cache = cache.contiguous()
        self._cs = cache
        if _ROPE is None:
            _build_ext()
        return cache

    @staticmethod
    def forward_native(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch NeOX-style RoPE matching the CUDA kernel.

        The cache stores [cos, sin] each with embed_dim = head_dim/2 entries.
        Rotation pairs elements (i, i + embed_dim) across the full head,
        exactly matching the CUDA kernel's IS_NEOX=true path:
          out[i]            = x[i]*cos[i] - x[i+embed_dim]*sin[i]
          out[i+embed_dim]  = x[i+embed_dim]*cos[i] + x[i]*sin[i]
        """
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim]
        sin = cos_sin[..., embed_dim:]

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q1, q2 = q[..., :embed_dim], q[..., embed_dim:]
        k1, k2 = k[..., :embed_dim], k[..., embed_dim:]

        query = torch.cat([q1 * cos - q2 * sin,
                           q2 * cos + q1 * sin], dim=-1).view(q_shape)
        key = torch.cat([k1 * cos - k2 * sin,
                         k2 * cos + k1 * sin], dim=-1).view(k_shape)
        return query, key

    @staticmethod
    def forward_native_interleaved(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch GPT-J/interleaved RoPE matching CUDA IS_NEOX=false."""
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim].unsqueeze(1)
        sin = cos_sin[..., embed_dim:].unsqueeze(1)

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_rot = torch.stack(
            (q_even * cos - q_odd * sin,
             q_odd * cos + q_even * sin),
            dim=-1,
        ).flatten(-2)
        k_rot = torch.stack(
            (k_even * cos - k_odd * sin,
             k_odd * cos + k_even * sin),
            dim=-1,
        ).flatten(-2)
        return q_rot.view(q_shape), k_rot.view(k_shape)

    def forward_cuda(self, positions, query, key):
        """CUDA kernel path for eager mode."""
        cache = self._cs
        if cache is None or cache.dtype != query.dtype:
            cache = self._cast_cache(query)
        _ROPE(positions, query, key, self.head_dim, cache, self.is_neox_style)
        return query, key

    def forward(self, positions, query, key):
        cache = self._cs
        if cache is None or cache.dtype != query.dtype:
            cache = self._cast_cache(query)
        if _IS_COMPILING():
            if self.is_neox_style:
                return self.forward_native(
                    positions, query, key, self.head_dim, cache,
                )
            return self.forward_native_interleaved(
                positions, query, key, self.head_dim, cache,
            )
        _ROPE(positions, query, key, self.head_dim, cache, self.is_neox_style)
        return query, key


class Gemma4ProportionalRotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE.

    Gemma4 full-attention layers use a partial rotary factor, but the
    frequency exponents are divided by the full head dimension and the
    non-rotated angle pairs are represented as identity rotation.  This
    matches HF/vLLM's proportional RoPE instead of rotating a compact
    leading slice with ``rotary_dim`` as the denominator.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
    ):
        nn.Module.__init__(self)
        self.head_dim = head_dim
        self.is_neox_style = True
        rope_angles = rotary_dim // 2
        nope_angles = (head_dim // 2) - rope_angles

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim
            )
        )
        if nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq, torch.zeros(nope_angles, dtype=torch.float)],
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self._cs = None
