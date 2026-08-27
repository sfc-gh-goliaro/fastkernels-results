You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.


        Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is:

        ```
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return a + b

        def get_inputs():
            # randomly generate input tensors based on the model architecture
            a = torch.randn(1, 128).cuda()
            b = torch.randn(1, 128).cuda()
            return [a, b]

        def get_init_inputs():
            # randomly generate tensors required for initialization based on the model architecture
            return []
        ```

        The example new arch with custom Triton kernels looks like this:
        ```
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(
            x_ptr,  # Pointer to first input
            y_ptr,  # Pointer to second input
            out_ptr,  # Pointer to output
            n_elements,  # Total number of elements in input/output
            BLOCK_SIZE: tl.constexpr,
        ):
            # Each program handles a contiguous block of data of size BLOCK_SIZE
            block_start = tl.program_id(0) * BLOCK_SIZE
            # Create a range of offsets [0..BLOCK_SIZE-1]
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            # Mask to ensure we don't go out of bounds
            mask = offsets < n_elements
            # Load input values
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            # Perform the elementwise addition
            out = x + y
            # Store the result
            tl.store(out_ptr + offsets, out, mask=mask)

        def triton_add(x: torch.Tensor, y: torch.Tensor):
            """
            This function wraps the Triton kernel call. It:
              1. Ensures the inputs are contiguous on GPU.
              2. Calculates the grid (blocks) needed.
              3. Launches the Triton kernel.
            """
            assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
            x = x.contiguous()
            y = y.contiguous()

            # Prepare output tensor
            out = torch.empty_like(x)

            # Number of elements in the tensor
            n_elements = x.numel()
            BLOCK_SIZE = 128  # Tunable parameter for block size

            # Determine the number of blocks needed
            grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

            # Launch the Triton kernel
            add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return out

        class ModelNew(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                # Instead of "return a + b", call our Triton-based addition
                return triton_add(a, b)
        ```
        
    You are given the following architecture:
    ```

"""Triton kernel for storing key/value into paged KV cache.

Supports two layouts:
  NHD: [num_blocks, block_size, num_kv_heads, head_dim]  (flash_attn path)
  HND: [num_blocks, num_kv_heads, block_size, head_dim]  (TRTLLM path)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


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


class Model(nn.Module):
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
        if slot_mapping.dtype != torch.int64:
            slot_mapping = slot_mapping.to(torch.int64)
        _store_kvcache_hnd_kernel[(N, num_kv_heads)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping,
            PAGE_SIZE=self.page_size,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
        )

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### StoreKVCacheHND

| count | args |
|------:|------|
| 86010 | `key:bfloat16[1000, 1, 128] value:bfloat16[1000, 1, 128] k_cache:bfloat16[114541, 1, 16, 128] v_cache:bfloat16[114541, 1, 16, 128] slot_mapping:int32[1000]` |
| 23876 | `key:bfloat16[1, 1, 128] value:bfloat16[1, 1, 128] k_cache:bfloat16[114541, 1, 16, 128] v_cache:bfloat16[114541, 1, 16, 128] slot_mapping:int32[1]` |
| 6876 | `key:bfloat16[60, 4, 64] value:bfloat16[60, 4, 64] k_cache:bfloat16[216987, 4, 16, 64] v_cache:bfloat16[216987, 4, 16, 64] slot_mapping:int32[60]` |
| 6144 | `key:bfloat16[60, 8, 128] value:bfloat16[60, 8, 128] k_cache:bfloat16[70892, 8, 16, 128] v_cache:bfloat16[70892, 8, 16, 128] slot_mapping:int32[60]` |
| 4982 | `key:bfloat16[16384, 1, 128] value:bfloat16[16384, 1, 128] k_cache:bfloat16[114541, 1, 16, 128] v_cache:bfloat16[114541, 1, 16, 128] slot_mapping:int32[16384]` |
| 4680 | `key:bfloat16[1, 4, 64] value:bfloat16[1, 4, 64] k_cache:bfloat16[216987, 4, 16, 64] v_cache:bfloat16[216987, 4, 16, 64] slot_mapping:int32[1]` |
| 4128 | `key:bfloat16[1, 8, 128] value:bfloat16[1, 8, 128] k_cache:bfloat16[70892, 8, 16, 128] v_cache:bfloat16[70892, 8, 16, 128] slot_mapping:int32[1]` |
| 3096 | `key:bfloat16[16384, 4, 64] value:bfloat16[16384, 4, 64] k_cache:bfloat16[216987, 4, 16, 64] v_cache:bfloat16[216987, 4, 16, 64] slot_mapping:int32[16384]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
