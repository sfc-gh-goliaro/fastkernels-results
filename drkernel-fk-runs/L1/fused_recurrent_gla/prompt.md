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

"""Fused recurrent GLA — Triton-accelerated decode kernel.

Thin ``nn.Module`` wrapper around ``fla.ops.gla.fused_recurrent_gla``.
This is the SOTA decode-step path for GLA / RetNet (RetNet uses the
same kernel with a constant-in-time gk).

Tensor layout matches FLA's convention (``[B, T, H, K]``) so we don't
insert a transpose on the hot path. State is ``[N, H, K, V]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.ops.gla import fused_recurrent_gla


class Model(nn.Module):
    """Triton fused-recurrent GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        gk: torch.Tensor | None = None,  # [B, T, H, K]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return fused_recurrent_gla(
            q=q, k=k, v=v, gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### FusedRecurrentGLA

| count | args |
|------:|------|
| 46816 | `q:bfloat16[256, 1, 5, 256] k:bfloat16[256, 1, 5, 256] v:bfloat16[256, 1, 5, 512] gk:bfloat16[256, 1, 5, 256] scale:None initial_state:float32[256, 5, 256, 512] cu_seqlens:None` |
| 7264 | `q:bfloat16[1, 1, 5, 256] k:bfloat16[1, 1, 5, 256] v:bfloat16[1, 1, 5, 512] gk:bfloat16[1, 1, 5, 256] scale:None initial_state:float32[1, 5, 256, 512] cu_seqlens:None` |
| 3168 | `q:bfloat16[4, 1, 5, 256] k:bfloat16[4, 1, 5, 256] v:bfloat16[4, 1, 5, 512] gk:bfloat16[4, 1, 5, 256] scale:None initial_state:float32[4, 5, 256, 512] cu_seqlens:None` |
| 2656 | `q:bfloat16[64, 1, 5, 256] k:bfloat16[64, 1, 5, 256] v:bfloat16[64, 1, 5, 512] gk:bfloat16[64, 1, 5, 256] scale:None initial_state:float32[64, 5, 256, 512] cu_seqlens:None` |
| 2592 | `q:bfloat16[60, 1, 5, 256] k:bfloat16[60, 1, 5, 256] v:bfloat16[60, 1, 5, 512] gk:bfloat16[60, 1, 5, 256] scale:None initial_state:float32[60, 5, 256, 512] cu_seqlens:None` |
| 2592 | `q:bfloat16[8, 1, 5, 256] k:bfloat16[8, 1, 5, 256] v:bfloat16[8, 1, 5, 512] gk:bfloat16[8, 1, 5, 256] scale:None initial_state:float32[8, 5, 256, 512] cu_seqlens:None` |
| 1472 | `q:bfloat16[31, 1, 5, 256] k:bfloat16[31, 1, 5, 256] v:bfloat16[31, 1, 5, 512] gk:bfloat16[31, 1, 5, 256] scale:None initial_state:float32[31, 5, 256, 512] cu_seqlens:None` |
| 1376 | `q:bfloat16[5, 1, 5, 256] k:bfloat16[5, 1, 5, 256] v:bfloat16[5, 1, 5, 512] gk:bfloat16[5, 1, 5, 256] scale:None initial_state:float32[5, 5, 256, 512] cu_seqlens:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
