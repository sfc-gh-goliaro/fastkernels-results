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

"""Linear (matrix multiply) kernels.

Matmul: pure functional op — F.linear(input, weight, bias).
BMM: batch matrix multiply — torch.matmul(a, b).
Linear: parametric op — holds weight/bias as nn.Parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Model(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### BMM

| count | args |
|------:|------|
| 96 | `a:bfloat16[1, 64, 512, 64] b:bfloat16[1, 64, 64, 512]` |
| 96 | `a:bfloat16[1, 64, 512, 512] b:bfloat16[1, 64, 512, 64]` |
| 48 | `a:float32[1, 12, 77, 64] b:float32[1, 12, 64, 77]` |
| 48 | `a:float32[1, 12, 77, 77] b:float32[1, 12, 77, 64]` |

### Linear

| count | args |
|------:|------|
| 375991 | `input:bfloat16[256, 1, 2560]` |
| 58797 | `input:bfloat16[1, 1, 2560]` |
| 46816 | `input:bfloat16[256, 1, 16]` |
| 46816 | `input:bfloat16[256, 1, 6912]` |
| 25524 | `input:bfloat16[4, 1, 2560]` |
| 24800 | `input:bfloat16[1, 16, 16, 128]` |
| 21338 | `input:bfloat16[64, 1, 2560]` |
| 20850 | `input:bfloat16[8, 1, 2560]` |

### Matmul

| count | args |
|------:|------|
| 140448 | `input:bfloat16[256, 1, 2560] weight:bfloat16[2560, 2560] bias:None` |
| 93632 | `input:bfloat16[256, 1, 2560] weight:bfloat16[1280, 2560] bias:None` |
| 93632 | `input:bfloat16[256, 1, 2560] weight:bfloat16[6912, 2560] bias:None` |
| 46816 | `input:bfloat16[256, 1, 2560] weight:bfloat16[16, 2560] bias:None` |
| 46816 | `input:bfloat16[256, 1, 16] weight:bfloat16[1280, 16] bias:bfloat16[1280]` |
| 46816 | `input:bfloat16[256, 1, 6912] weight:bfloat16[2560, 6912] bias:None` |
| 21792 | `input:bfloat16[1, 1, 2560] weight:bfloat16[2560, 2560] bias:None` |
| 18312 | `input:bfloat16[1, 16, 16, 128] weight:bfloat16[128, 128] bias:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
