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

"""RMSNorm with dual dispatch: CUDA custom op (eager) and pure-PyTorch (compiled).

Mirrors vLLM's ``CustomOp`` dispatch pattern:
  - ``forward_cuda``: calls vLLM's ``torch.ops._C.rms_norm`` /
    ``torch.ops._C.fused_add_rms_norm`` CUDA kernels for bitwise-identical
    numerics with vLLM.
  - ``forward_native``: pure PyTorch implementation (f32 promotion, variance,
    rsqrt, weight multiply).  Used when torch.compile is active so Inductor
    can inline, fuse, and optimise the norm with adjacent ops — this is the
    key mechanism that enables RMSNorm+FP8-quant fusion.

The ``forward`` method dispatches based on ``torch.compiler.is_compiling()``.

Known limitations of the CUDA kernel (forward_cuda path):
  - Produces incorrect output for hidden sizes that aren't multiples of 32
    (verified empirically: hidden=16 and hidden=80 give max-abs error ~1e3
    on random unit-variance input vs the reference math; hidden=32, 64, 128
    are correct).
  - Has no ``torch.autograd`` backward registered, so the norm silently
    drops gradient under ``torch.func.grad``.

Use :class:`L1.rms_norm_native.RMSNormNative` instead when you need either
of those properties — odd head_dims (e.g. TTT-E2E qk_norm at head_dim=16,
or any model with head_dim that isn't a multiple of 32), or autograd /
torch.func.grad support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("rms_norm", "rms_norm.cu")

# ---------------------------------------------------------------------------
# Register _C ops as torch.library custom ops for torch.compile compatibility.
# Used by ``rms_norm_native`` (not by this module's eager path, which goes
# through the vendored vLLM ``_C.rms_norm``).
# ---------------------------------------------------------------------------

_lib = torch.library.Library("fastkernels_norm", "DEF")

_lib.define("rmsnorm(Tensor! result, Tensor input, Tensor weight, float eps) -> ()")

def _rmsnorm_impl(result, input, weight, eps):
    _C.rmsnorm(result, input, weight, eps)

_lib.impl("rmsnorm", _rmsnorm_impl, "CUDA")

@torch.library.impl(_lib, "rmsnorm", "Meta")
def _rmsnorm_meta(result, input, weight, eps):
    pass

_lib.define(
    "fused_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual, "
    "Tensor weight, float eps) -> ()"
)

def _fused_add_rmsnorm_impl(input, residual, weight, eps):
    _C.fused_add_rmsnorm(input, residual, weight, eps)

_lib.impl("fused_add_rmsnorm", _fused_add_rmsnorm_impl, "CUDA")

@torch.library.impl(_lib, "fused_add_rmsnorm", "Meta")
def _fused_add_rmsnorm_meta(input, residual, weight, eps):
    pass


# ---------------------------------------------------------------------------
# RMSNorm module
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            # Match vLLM's has_weight=False path: use the same CUDA RMSNorm
            # kernel with a non-persistent unit scale instead of falling back
            # to torch.nn.functional.rms_norm in eager/CUDA-graph decode.
            self.register_buffer(
                "_unit_weight",
                torch.ones(hidden_size),
                persistent=False,
            )

    # -- Pure PyTorch path (used under torch.compile so Inductor can fuse) --

    @staticmethod
    def forward_native(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        hidden_size: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch RMSNorm matching vLLM's forward_static."""
        orig_dtype = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(orig_dtype)
        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        return x, residual

    # -- CUDA kernel path (used in eager mode / CUDA graph replay) --

    @staticmethod
    def forward_cuda(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if weight is not None:
            # The CUDA rms_norm / fused_add_rms_norm kernels assume the row
            # dimension is contiguous (row stride == hidden size). A strided
            # input — e.g. the K slice of a fused QKV output when num_kv_heads
            # collapses to a single head under tensor parallelism, so the
            # per-head reshape yields a non-contiguous view — makes the kernel
            # read the wrong memory for every row past the first, silently
            # corrupting the output. Force contiguity here (a no-op when the
            # tensor is already contiguous) so every caller is safe.
            x = x.contiguous()
            if residual is not None:
                residual = residual.contiguous()
            if residual is None:
                out = torch.empty_like(x)
                _C.rms_norm(out, x, weight, eps)
                return out
            _C.fused_add_rms_norm(x, residual, weight, eps)
            return x, residual
        if residual is None:
            return F.rms_norm(x, (x.size(-1),), eps=eps)
        x = x + residual
        residual = x
        return F.rms_norm(x, (x.size(-1),), eps=eps), residual

    def forward(self, x, residual=None):
        if torch.compiler.is_compiling():
            return self.forward_native(
                x, self.weight if self.elementwise_affine else None,
                self.eps, self.hidden_size, residual,
            )
        weight = self.weight if self.elementwise_affine else self._unit_weight
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        return self.forward_cuda(
            x, weight, self.eps, residual,
        )

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### RMSNorm

| count | args |
|------:|------|
| 172020 | `x:bfloat16[1000, 4096] residual:bfloat16[1000, 4096]` |
| 95095 | `x:bfloat16[256, 2560] residual:None` |
| 86010 | `x:bfloat16[1000, 16, 128] residual:None` |
| 86010 | `x:bfloat16[1000, 1, 128] residual:None` |
| 56008 | `x:bfloat16[1, 4096] residual:bfloat16[1, 4096]` |
| 46816 | `x:bfloat16[1280, 512] residual:None` |
| 23876 | `x:bfloat16[1, 16, 128] residual:None` |
| 23876 | `x:bfloat16[1, 1, 128] residual:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
