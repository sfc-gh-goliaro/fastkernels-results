"""Kernels for storing key/value into a paged KV cache.

Supports two layouts:
  NHD: [num_blocks, block_size, num_kv_heads, head_dim]  (flash_attn path)
  HND: [num_blocks, num_kv_heads, block_size, head_dim]  (TRTLLM path)

The HND store is a hand-written CUDA kernel.  Against the reference Triton
version it wins on three fronts:

* **one launch instead of two** -- the ``int32 -> int64`` widening of
  ``slot_mapping`` happens inside the kernel instead of as a separate
  ``aten::to`` copy (every captured call site passes int32).  A launch costs
  ~2 us of device time here, which is the *whole* cost of the decode shapes.
* **16-byte (``uint4``) accesses** -- the Triton kernel maps one bf16 *element*
  per lane, so a warp only moves 64 B per instruction; at 16 B/lane one warp
  covers two whole ``head_dim=128`` rows.  This is what takes the 16k-token
  prefill shape from 121 us to 32 us, against a ~30 us roofline for
  131k scattered 256 B page writes (~2.2 TB/s measured on B200).
* **k and v in the same thread** -- the slot lookup, the page/offset
  arithmetic and the grid bookkeeping are paid once for both tensors.

The grid is ``(ceil(N / tokens_per_block), num_kv_heads)`` and the block is
``(head_dim_vecs, tokens_per_block)``, so no thread ever divides: the head
comes from ``blockIdx.y`` and the token from ``blockIdx.x``/``threadIdx.y``.
Reads from ``key``/``value`` stay coalesced because consecutive
``threadIdx.x``/``threadIdx.y`` walk the contiguous ``[N, H, D]`` source.

Layout validation lives in C++ so the hot path is a single pybind call; if any
assumption fails (non-contiguous cache, exotic dtype, odd ``head_dim``) the
kernel reports it and we fall back to the reference Triton path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:  # package-relative when loaded as fastkernels.tasks.candidate.L1.store_kvcache
    from ....infra.cuda_ext import _pin_build_arch
except ImportError:  # pragma: no cover - direct/standalone import
    from fastkernels.infra.cuda_ext import _pin_build_arch


@triton.jit
def _store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
):
    idx = tl.program_id(0)
    # int64: Hopper hybrid pages are large.  ``slot * D`` in int32
    # overflows at slot >= 2^31/D (bid >= 65536 when D=2048).
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    if slot < 0:
        return
    offsets = tl.arange(0, D_PAD)
    mask = offsets < D
    key = tl.load(key_ptr + idx * key_stride + offsets, mask=mask)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask=mask)
    dst = slot * D + offsets
    tl.store(k_cache_ptr + dst, key, mask=mask)
    tl.store(v_cache_ptr + dst, value, mask=mask)


@triton.jit
def _store_kvcache_hnd_kernel(
    key_ptr, key_stride_n, value_ptr, value_stride_n,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Store KV into HND layout [num_blocks, num_kv_heads, block_size, head_dim]."""
    idx = tl.program_id(0)
    head = tl.program_id(1)
    # int64: same overflow as the NHD store.  ``block_idx * H * page * D``
    # in int32 wraps once block_idx >= 2^31 / (H * page * D) (131072 for
    # Jamba Mini H=8, page=16, D=128 -- under the 201k-block B200 pool).
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    if slot < 0:
        return
    block_idx = slot // PAGE_SIZE
    slot_in_block = slot % PAGE_SIZE
    src_k_offset = idx * key_stride_n + head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    src_v_offset = idx * value_stride_n + head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    dst_offset = (
        block_idx * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
        + head * PAGE_SIZE * HEAD_DIM
        + slot_in_block * HEAD_DIM
        + tl.arange(0, HEAD_DIM)
    )
    k = tl.load(key_ptr + src_k_offset)
    v = tl.load(value_ptr + src_v_offset)
    tl.store(k_cache_ptr + dst_offset, k)
    tl.store(v_cache_ptr + dst_offset, v)


_HND_CPP_DECL = r"""
#include <torch/extension.h>
bool store_kvcache_hnd(const at::Tensor& key, const at::Tensor& value,
                       const at::Tensor& k_cache, const at::Tensor& v_cache,
                       const at::Tensor& slot_mapping, int64_t page_size);
"""

_HND_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// One thread moves sizeof(V) bytes of k *and* of v.  blockIdx.y is the kv
// head; blockDim.y tokens are packed per block so the loads from the
// contiguous [N, H, D] key/value stay coalesced.  Each (token, head)
// destination is D*sizeof(elem) contiguous bytes inside its page.
template <typename V, typename S>
__global__ __launch_bounds__(256) void store_kv_hnd_kernel(
    const V* __restrict__ key,
    const V* __restrict__ value,
    V* __restrict__ k_cache,
    V* __restrict__ v_cache,
    const S* __restrict__ slot_mapping,
    long key_stride_v,
    long value_stride_v,
    int n_tokens,
    int num_heads,
    int dv,
    int page_size,
    int page_shift) {
  const int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= n_tokens) return;
  const long slot = static_cast<long>(slot_mapping[n]);
  if (slot < 0) return;  // padded / dropped token

  long blk, sib;
  if (page_shift >= 0) {  // power-of-two page size: shift + mask
    blk = slot >> page_shift;
    sib = slot & static_cast<long>(page_size - 1);
  } else {
    blk = slot / page_size;
    sib = slot - blk * page_size;
  }

  const int h = blockIdx.y;
  const long dst = ((blk * num_heads + h) * page_size + sib) * dv;
  const long ks = static_cast<long>(n) * key_stride_v + static_cast<long>(h) * dv;
  const long vs = static_cast<long>(n) * value_stride_v + static_cast<long>(h) * dv;

  for (int d = threadIdx.x; d < dv; d += blockDim.x) {
    const V a = key[ks + d];
    const V b = value[vs + d];
    k_cache[dst + d] = a;
    v_cache[dst + d] = b;
  }
}

#define FK_LAUNCH(V, S)                                                        \
  store_kv_hnd_kernel<V, S><<<grid, block, 0, stream>>>(                       \
      reinterpret_cast<const V*>(key.data_ptr()),                              \
      reinterpret_cast<const V*>(value.data_ptr()),                            \
      reinterpret_cast<V*>(k_cache.data_ptr()),                                \
      reinterpret_cast<V*>(v_cache.data_ptr()),                                \
      reinterpret_cast<const S*>(slot_mapping.data_ptr()),                     \
      key_stride_v, value_stride_v, n_tokens, num_heads, dv,                   \
      static_cast<int>(page_size), page_shift)

#define FK_LAUNCH_V(V)                                                         \
  if (slot_i64) { FK_LAUNCH(V, long long); } else { FK_LAUNCH(V, int); }

// Returns false (without launching) when the captured layout assumptions do
// not hold, so the caller can fall back to the reference Triton kernel.
bool store_kvcache_hnd(const at::Tensor& key, const at::Tensor& value,
                       const at::Tensor& k_cache, const at::Tensor& v_cache,
                       const at::Tensor& slot_mapping, int64_t page_size) {
  if (key.dim() != 3 || value.dim() != 3 || k_cache.dim() != 4 ||
      v_cache.dim() != 4 || slot_mapping.dim() != 1)
    return false;

  const int n_tokens = static_cast<int>(key.size(0));
  const int num_heads = static_cast<int>(key.size(1));
  const int64_t head_dim = key.size(2);

  const auto st = slot_mapping.scalar_type();
  const bool slot_i64 = (st == at::kLong);
  if (!slot_i64 && st != at::kInt) return false;
  if (key.scalar_type() != k_cache.scalar_type() ||
      value.scalar_type() != k_cache.scalar_type() ||
      v_cache.scalar_type() != k_cache.scalar_type())
    return false;
  if (slot_mapping.size(0) != n_tokens || !slot_mapping.is_contiguous())
    return false;

  // key/value: [N, H, D] with the (H, D) tile contiguous (stride(0) free).
  if (key.stride(2) != 1 || key.stride(1) != head_dim ||
      value.stride(2) != 1 || value.stride(1) != head_dim)
    return false;
  // caches: contiguous [num_blocks, H, page_size, D].
  if (!k_cache.is_contiguous() || !v_cache.is_contiguous()) return false;
  if (k_cache.size(1) != num_heads || k_cache.size(2) != page_size ||
      k_cache.size(3) != head_dim || !k_cache.sizes().equals(v_cache.sizes()))
    return false;

  if (n_tokens == 0 || num_heads == 0 || head_dim == 0) return true;

  // Widest power-of-two access every operand can sustain.
  const int64_t esz = key.element_size();
  const int64_t row_bytes = head_dim * esz;
  int64_t vec_bytes = 16;
  while (vec_bytes > 1 &&
         (row_bytes % vec_bytes != 0 ||
          (key.stride(0) * esz) % vec_bytes != 0 ||
          (value.stride(0) * esz) % vec_bytes != 0 ||
          (reinterpret_cast<uintptr_t>(key.data_ptr()) % vec_bytes) != 0 ||
          (reinterpret_cast<uintptr_t>(value.data_ptr()) % vec_bytes) != 0 ||
          (reinterpret_cast<uintptr_t>(k_cache.data_ptr()) % vec_bytes) != 0 ||
          (reinterpret_cast<uintptr_t>(v_cache.data_ptr()) % vec_bytes) != 0))
    vec_bytes >>= 1;

  const int dv = static_cast<int>(row_bytes / vec_bytes);
  const long key_stride_v = key.stride(0) * esz / vec_bytes;
  const long value_stride_v = value.stride(0) * esz / vec_bytes;

  int page_shift = -1;
  if (page_size > 0 && (page_size & (page_size - 1)) == 0) {
    page_shift = 0;
    while ((int64_t(1) << page_shift) < page_size) ++page_shift;
  }

  const int tx = dv < 64 ? dv : 64;
  const int ty = 256 / tx;
  const dim3 block(tx, ty);
  const dim3 grid((n_tokens + ty - 1) / ty, num_heads);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (vec_bytes) {
    case 16: FK_LAUNCH_V(uint4);          break;
    case 8:  FK_LAUNCH_V(uint2);          break;
    case 4:  FK_LAUNCH_V(unsigned int);   break;
    case 2:  FK_LAUNCH_V(unsigned short); break;
    default: FK_LAUNCH_V(unsigned char);  break;
  }
  return true;
}
"""


class _LazyHndExt:
    """Defer the one-time JIT build to the first HND store."""

    def __init__(self):
        self._fn = None

    def __call__(self, *args):
        fn = self._fn
        if fn is None:
            from torch.utils.cpp_extension import load_inline
            _pin_build_arch()
            mod = load_inline(
                name="fk_store_kvcache_hnd_cand",
                cpp_sources=_HND_CPP_DECL,
                cuda_sources=_HND_CUDA_SRC,
                functions=["store_kvcache_hnd"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
            fn = self._fn = mod.store_kvcache_hnd
        return fn(*args)


_hnd_store = _LazyHndExt()


class StoreKVCache(nn.Module):
    """NHD layout store: [num_blocks, block_size, num_kv_heads, head_dim]."""
    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        D_PAD = triton.next_power_of_2(D)
        if slot_mapping.dtype != torch.int64:
            slot_mapping = slot_mapping.to(torch.int64)
        _store_kvcache_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D, D_PAD,
        )


class StoreKVCacheHND(nn.Module):
    """HND layout store: [num_blocks, num_kv_heads, block_size, head_dim]."""
    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        if _hnd_store(key, value, k_cache, v_cache, slot_mapping, self.page_size):
            return

        # Fallback: reference Triton path for layouts the CUDA kernel rejects.
        N, num_kv_heads, head_dim = key.shape
        if slot_mapping.dtype != torch.int64:
            slot_mapping = slot_mapping.to(torch.int64)
        _store_kvcache_hnd_kernel[(N, num_kv_heads)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping,
            PAGE_SIZE=self.page_size,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
        )
