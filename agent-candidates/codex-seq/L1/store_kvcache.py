"""Triton kernel for storing key/value into paged KV cache.

Supports two layouts:
  NHD: [num_blocks, block_size, num_kv_heads, head_dim]  (flash_attn path)
  HND: [num_blocks, num_kv_heads, block_size, head_dim]  (TRTLLM path)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


_CPP_SOURCE = r"""
#include <torch/extension.h>

void store_kvcache_hnd_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <int WARPS, int HEADS, int DIM_VECS>
__global__ void store_kvcache_hnd_kernel(
    const uint4* __restrict__ key,
    const uint4* __restrict__ value,
    uint4* __restrict__ k_cache,
    uint4* __restrict__ v_cache,
    const int* __restrict__ slot_mapping,
    int num_tokens) {
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * WARPS + warp;
  if (token >= num_tokens) {
    return;
  }

  const int slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }

  const int lane = threadIdx.x & 31;
  constexpr int PAGE_SIZE = 16;
  constexpr int VECS_PER_TOKEN = HEADS * DIM_VECS;
  const long long block = slot >> 4;
  const int position = slot & 15;
  const long long block_base =
      block * HEADS * PAGE_SIZE * DIM_VECS;
  const int source_base = token * VECS_PER_TOKEN;

  #pragma unroll
  for (int vec = lane; vec < VECS_PER_TOKEN; vec += 32) {
    const int head = vec / DIM_VECS;
    const int column = vec % DIM_VECS;
    const long long destination =
        block_base
        + static_cast<long long>(head) * PAGE_SIZE * DIM_VECS
        + position * DIM_VECS
        + column;
    k_cache[destination] = key[source_base + vec];
    v_cache[destination] = value[source_base + vec];
  }
}

template <int WARPS, int HEADS, int DIM_VECS>
void launch_store(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping) {
  const int num_tokens = key.size(0);
  store_kvcache_hnd_kernel<WARPS, HEADS, DIM_VECS>
      <<<(num_tokens + WARPS - 1) / WARPS,
         WARPS * 32,
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          static_cast<const uint4*>(key.data_ptr()),
          static_cast<const uint4*>(value.data_ptr()),
          static_cast<uint4*>(k_cache.data_ptr()),
          static_cast<uint4*>(v_cache.data_ptr()),
          slot_mapping.data_ptr<int>(),
          num_tokens);
}

template <int HEADS, int DIM_VECS>
void select_warps(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping) {
  if (key.size(0) >= 1024) {
    launch_store<8, HEADS, DIM_VECS>(
        key, value, k_cache, v_cache, slot_mapping);
  } else {
    launch_store<4, HEADS, DIM_VECS>(
        key, value, k_cache, v_cache, slot_mapping);
  }
}

void store_kvcache_hnd_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping) {
  const int heads = key.size(1);
  const int dim_vecs = key.size(2) / 8;
  if (heads == 1 && dim_vecs == 16) {
    select_warps<1, 16>(key, value, k_cache, v_cache, slot_mapping);
  } else if (heads == 1 && dim_vecs == 32) {
    select_warps<1, 32>(key, value, k_cache, v_cache, slot_mapping);
  } else if (heads == 4 && dim_vecs == 8) {
    select_warps<4, 8>(key, value, k_cache, v_cache, slot_mapping);
  } else if (heads == 8 && dim_vecs == 16) {
    select_warps<8, 16>(key, value, k_cache, v_cache, slot_mapping);
  }
}
"""


_old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
try:
    _cuda_ext = load_inline(
        name="fk_store_kvcache_hnd_b200_v1",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=["store_kvcache_hnd_cuda"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
finally:
    if _old_arch_list is None:
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = _old_arch_list


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
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    if slot < 0:
        return
    block_idx = slot // PAGE_SIZE
    slot_in_block = slot % PAGE_SIZE
    offsets = tl.arange(0, NUM_KV_HEADS * HEAD_DIM)
    head = offsets // HEAD_DIM
    dim = offsets % HEAD_DIM
    src_k_offset = idx * key_stride_n + offsets
    src_v_offset = idx * value_stride_n + offsets
    dst_offset = (
        block_idx * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
        + head * PAGE_SIZE * HEAD_DIM
        + slot_in_block * HEAD_DIM
        + dim
    )
    k = tl.load(key_ptr + src_k_offset)
    v = tl.load(value_ptr + src_v_offset)
    tl.store(k_cache_ptr + dst_offset, k)
    tl.store(v_cache_ptr + dst_offset, v)


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
        N, num_kv_heads, head_dim = key.shape
        if (
            self.page_size == 16
            and slot_mapping.dtype == torch.int32
            and key.dtype == torch.bfloat16
            and value.dtype == torch.bfloat16
            and k_cache.dtype == torch.bfloat16
            and v_cache.dtype == torch.bfloat16
            and key.is_contiguous()
            and value.is_contiguous()
            and k_cache.is_contiguous()
            and v_cache.is_contiguous()
            and slot_mapping.is_contiguous()
            and (num_kv_heads, head_dim) in (
                (1, 128),
                (1, 256),
                (4, 64),
                (8, 128),
            )
        ):
            _cuda_ext.store_kvcache_hnd_cuda(
                key, value, k_cache, v_cache, slot_mapping
            )
            return

        num_warps = 4 if (
            (num_kv_heads == 4 and N >= 1024)
            or (num_kv_heads == 1 and head_dim >= 256)
        ) else 1
        _store_kvcache_hnd_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping,
            PAGE_SIZE=self.page_size,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            num_warps=num_warps,
            num_stages=1,
        )
