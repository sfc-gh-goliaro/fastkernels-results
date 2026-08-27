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

"""GemmaRMSNorm: RMSNorm where the stored weight is an offset from 1.0.

Ports vLLM's ``GemmaRMSNorm`` (``vllm/model_executor/layers/layernorm.py``)
to avoid a runtime dependency on ``sgl_kernel``.

Differences from a vanilla RMSNorm:
  1. The scale applied at runtime is ``(1 + weight)`` rather than ``weight``
     (the checkpoint stores values near zero).
  2. The cast back to the original dtype happens *after* the weight multiply
     -- ``(x * w).to(orig_dtype)`` instead of ``x.to(orig_dtype) * w``.
     See https://github.com/huggingface/transformers/pull/29402.

Implementation strategy mirrors vLLM:
  - ``forward_native``: pure PyTorch (f32 promotion + variance + rsqrt + scale).
  - ``forward_cuda``: same pure-PyTorch helpers but lazily wrapped with
    ``torch.compile`` so Inductor can fuse them. Skipped when already inside
    a compiled region (``torch.compiler.is_compiling()``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """RMSNorm with weight stored as offset from 1.0 (Gemma convention).

    The checkpoint stores values near zero; the scale ``(1 + weight)`` is
    materialized at runtime, matching vLLM's GemmaRMSNorm semantics exactly.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @staticmethod
    def _forward_static_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype)

    @staticmethod
    def _forward_static_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # Match vLLM: promote to f32 only when the residual add would lose
        # precision (i.e. fp16 inputs); otherwise add in the input dtype.
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x.to(orig_dtype) if x.dtype != orig_dtype else x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype), residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._forward_static_no_residual(
                self.weight.data, self.variance_epsilon, x,
            )
        return self._forward_static_with_residual(
            self.weight.data, self.variance_epsilon, x, residual,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self._forward_static_no_residual = torch.compile(  # type: ignore[method-assign]
                self._forward_static_no_residual,
            )
            self._forward_static_with_residual = torch.compile(  # type: ignore[method-assign]
                self._forward_static_with_residual,
            )
            self._is_compiled = True
        return self.forward_native(x, residual)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### GemmaRMSNorm

| count | args |
|------:|------|
| 255 | `x:bfloat16[60, 2048] residual:None` |
| 127 | `x:bfloat16[1, 2048] residual:None` |
| 85 | `x:bfloat16[16384, 2048] residual:None` |
| 40 | `x:bfloat16[26, 2048] residual:None` |
| 34 | `x:bfloat16[31, 2048] residual:None` |
| 26 | `x:bfloat16[69, 2048] residual:None` |
| 19 | `x:bfloat16[29, 2048] residual:None` |
| 16 | `x:bfloat16[30, 2048] residual:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
