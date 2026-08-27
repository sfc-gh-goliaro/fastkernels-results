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

"""Flash attention prefill kernel (variable-length sequences).

Routes through vLLM's bundled FlashAttention build at the version vLLM
itself would select for this device (FA3 on Hopper, FA4 on Blackwell,
FA2 otherwise) -- see :mod:`fa_utils`.
"""

import torch
import torch.nn as nn

from fastkernels.infra.fa_utils import fa3_scheduler_metadata, fa_version_for_head_size, flash_attn_varlen_func


class Model(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        # Hopper FA3 cannot run head_dim>256; vLLM upgrades those layers to FA4.
        self.fa_version = fa_version_for_head_size(head_dim)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        # vLLM's wrapper takes keyword args in a different order than the
        # standard flash_attn signature.  With a ``block_table`` the
        # kernel needs per-sequence ``seqused_k`` rather than cumulative
        # ``cu_seqlens_k``.
        fa_kw = dict(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            fa_version=self.fa_version,
        )
        if kwargs.get("block_table") is not None:
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            fa_kw["seqused_k"] = seqused_k
            if (
                self.fa_version == 3
                and not torch.cuda.is_current_stream_capturing()
            ):
                page_size = k.shape[1] if k.dim() >= 2 else None
                meta = fa3_scheduler_metadata(
                    batch_size=int(seqused_k.shape[0]),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    num_heads_q=self.num_heads,
                    num_heads_kv=self.num_kv_heads,
                    headdim=self.head_dim,
                    cache_seqlens=seqused_k,
                    qkv_dtype=q.dtype,
                    cu_seqlens_q=cu_seqlens_q,
                    page_size=page_size,
                    causal=kwargs.get("causal", True),
                    window_size=kwargs.get("window_size", (-1, -1)),
                    num_splits=0,
                )
                if meta is not None:
                    fa_kw["scheduler_metadata"] = meta
                fa_kw["num_splits"] = 0
        else:
            fa_kw["cu_seqlens_k"] = cu_seqlens_k
            # Dense prefill is compute-bound, so KV-splitting buys nothing, but
            # the FA4 (SM100 CuTe) auto heuristic still picks the split-KV kernel
            # for mid-size seqlens -- and that variant fails to compile in this
            # vLLM build (TYPE_UNSTABLE_JOIN on ``n_block_first``).  Pin
            # ``num_splits=1`` so the unsplit kernel is used.
            fa_kw["num_splits"] = 1
        fa_kw.update(kwargs)
        return flash_attn_varlen_func(q, k, v, **fa_kw)

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### FlashAttnPrefill

| count | args |
|------:|------|
| 108 | `q:bfloat16[20680, 4, 72] k:bfloat16[20680, 4, 72] v:bfloat16[20680, 4, 72] cu_seqlens_q:int32[25] cu_seqlens_k:int32[25]` |
| 81 | `q:bfloat16[23760, 4, 72] k:bfloat16[23760, 4, 72] v:bfloat16[23760, 4, 72] cu_seqlens_q:int32[29] cu_seqlens_k:int32[29]` |
| 54 | `q:bfloat16[3072, 4, 72] k:bfloat16[3072, 4, 72] v:bfloat16[3072, 4, 72] cu_seqlens_q:int32[2] cu_seqlens_k:int32[2]` |
| 54 | `q:bfloat16[23248, 4, 72] k:bfloat16[23248, 4, 72] v:bfloat16[23248, 4, 72] cu_seqlens_q:int32[29] cu_seqlens_k:int32[29]` |
| 54 | `q:bfloat16[24200, 4, 72] k:bfloat16[24200, 4, 72] v:bfloat16[24200, 4, 72] cu_seqlens_q:int32[29] cu_seqlens_k:int32[29]` |
| 54 | `q:bfloat16[23320, 4, 72] k:bfloat16[23320, 4, 72] v:bfloat16[23320, 4, 72] cu_seqlens_q:int32[29] cu_seqlens_k:int32[29]` |
| 54 | `q:bfloat16[26400, 4, 72] k:bfloat16[26400, 4, 72] v:bfloat16[26400, 4, 72] cu_seqlens_q:int32[31] cu_seqlens_k:int32[31]` |
| 54 | `q:bfloat16[23960, 4, 72] k:bfloat16[23960, 4, 72] v:bfloat16[23960, 4, 72] cu_seqlens_q:int32[31] cu_seqlens_k:int32[31]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
