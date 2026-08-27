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

"""Variable-length Flash Attention (no KV cache lookup).

Thin ``nn.Module`` wrapper around ``flash_attn_varlen_func`` from vLLM's
bundled FlashAttention build, at the version vLLM would select for this
device (see :mod:`fa_utils`).

Used by MLA prefill and chunked-context paths where Q, K, V are dense
``[total_tokens, num_heads, head_dim]`` tensors (no paged cache lookup,
no ``block_table``).  Supports ``return_softmax_lse`` for MLA chunked
prefix merging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.fa_utils import FA_VERSION, flash_attn_varlen_func


class Model(nn.Module):
    """Variable-length Flash Attention without paged KV cache lookup."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            fa_version=FA_VERSION,
        )

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### FlashAttnVarlen

| count | args |
|------:|------|
| 329 | `q:bfloat16[16384, 16, 192] k:bfloat16[16384, 16, 192] v:bfloat16[16384, 16, 128] cu_seqlens_q:int32[2] cu_seqlens_k:int32[2]` |
| 217 | `q:bfloat16[16384, 16, 192] k:bfloat16[16384, 16, 192] v:bfloat16[16384, 16, 128] cu_seqlens_q:int32[3] cu_seqlens_k:int32[3]` |
| 112 | `q:bfloat16[16384, 16, 192] k:bfloat16[16384, 16, 192] v:bfloat16[16384, 16, 128] cu_seqlens_q:int32[4] cu_seqlens_k:int32[4]` |
| 77 | `q:bfloat16[16384, 16, 192] k:bfloat16[65536, 16, 192] v:bfloat16[65536, 16, 128] cu_seqlens_q:int32[2] cu_seqlens_k:int32[2]` |
| 28 | `q:bfloat16[16384, 16, 192] k:bfloat16[65536, 16, 192] v:bfloat16[65536, 16, 128] cu_seqlens_q:int32[3] cu_seqlens_k:int32[3]` |
| 24 | `q:float16[2048, 16, 64] k:float16[2048, 16, 64] v:float16[2048, 16, 64] cu_seqlens_q:int32[33] cu_seqlens_k:int32[33]` |
| 24 | `q:float16[512, 16, 64] k:float16[512, 16, 64] v:float16[512, 16, 64] cu_seqlens_q:int32[9] cu_seqlens_k:int32[9]` |
| 24 | `q:float16[64, 16, 64] k:float16[64, 16, 64] v:float16[64, 16, 64] cu_seqlens_q:int32[2] cu_seqlens_k:int32[2]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
