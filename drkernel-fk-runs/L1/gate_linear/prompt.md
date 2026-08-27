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

"""DeepSeek MoE router gate matmul (BF16 x BF16 -> FP32) with vLLM parity.

Mirrors vLLM's
``vllm/model_executor/layers/fused_moe/router/gate_linear.py:GateLinear``
which has a three-tier dispatch:

1. **DSV3 specialized kernel** — Hopper/Blackwell, ``num_experts in {256, 384}``,
   ``hidden_size == 7168``, batch ``<= 16``. BF16 x BF16 -> FP32 fused kernel
   that internally accumulates in FP32. Routes to
   ``_C.dsv3_router_gemm`` (verbatim port of vLLM's CUDA kernel — see
   the amalgamated ``tasks/baseline/L1/gate_linear.cu``).
2. **cuBLAS BF16 -> FP32** — Hopper/Blackwell + BF16 weight + FP32 out. Routes
   to ``_C.router_gemm_bf16_fp32`` (verbatim port of vLLM's cuBLAS wrapper —
   see the amalgamated ``tasks/baseline/L1/gate_linear.cu``).
3. **PyTorch fallback** — vanilla ``F.linear`` at the input dtype, with cast
   back to FP32 at the end.

Matching vLLM exactly here matters for **correctness** of the grouped-topk
router: fastkernels's previous "promote both to FP32 then matmul" path was
strictly more precise but used a different accumulation order than vLLM's
specialized kernels, which flipped near-tie expert / group selections in
the noaux_tc path (see audit notes).
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("gate_linear", "gate_linear.cu")

# The pybind11 entry points below are raw C++ extension functions, not torch
# ops, so Dynamo cannot trace them: compiling a model that calls one fails with
# "Dynamo does not know how to trace the builtin ... router_gemm_bf16_fp32".
# That is what blocked compiling Kimi-Linear (which in turn cost it the
# AR+RMSNorm post-grad fusion and all Inductor elementwise fusion).
#
# ``torch._dynamo.disable`` is NOT sufficient here: it graph-breaks, and
# ``compile_model`` uses fullgraph=True, which rejects breaks outright
# ("Skip calling torch.compiler.disable()'d function"). Registering real custom
# ops makes them opaque *without* breaking the graph -- the same reason
# ``fastkernels::unified_attention`` exists for attention.


@torch.library.custom_op("fastkernels::router_gemm_bf16_fp32", mutates_args=())
def _router_gemm_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _C.router_gemm_bf16_fp32(x, weight)


@_router_gemm_bf16_fp32.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


@torch.library.custom_op(
    "fastkernels::dsv3_router_gemm", mutates_args={"output"})
def _dsv3_router_gemm_op(
    output: torch.Tensor, hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> None:
    _C.dsv3_router_gemm(output, hidden_states, router_weight)


@_dsv3_router_gemm_op.register_fake
def _(output: torch.Tensor, hidden_states: torch.Tensor,
      router_weight: torch.Tensor) -> None:
    return None


@functools.cache
def _is_hopper_or_blackwell() -> bool:
    """Same gate vLLM uses (see ``GateLinear.__init__``):
    ``current_platform.is_device_capability((9, 0))`` (Hopper) or
    ``current_platform.is_device_capability_family(100)`` (Blackwell)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return (cap[0], cap[1]) == (9, 0) or cap[0] == 10


@functools.cache
def _dsv3_max_batch() -> int:
    """Max ``num_tokens`` routed to the DSV3 kernel: ``16`` on Hopper, ``8``
    otherwise (Blackwell). Mirrors vLLM ``GateLinear._dsv3_max_batch``
    (``16 if is_hopper else 8``; see vLLM PR #44217)."""
    if not torch.cuda.is_available():
        return 16
    cap = torch.cuda.get_device_capability()
    return 16 if (cap[0], cap[1]) == (9, 0) else 8


def _dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Allocates the output and dispatches to the DSV3 specialized kernel.

    Mirrors vLLM's ``_custom_ops.dsv3_router_gemm`` Python wrapper: the
    underlying CUDA op takes ``output`` as an in/out parameter, so the
    allocation lives on the Python side. ``output_dtype=None`` mirrors vLLM's
    ``torch.empty(dtype=None)``, which resolves to FP32 at inference time —
    fastkernels sets a global bf16 default dtype, so we pin FP32 explicitly
    to keep the decode router logits at FP32 like vLLM.
    """
    output = torch.empty(
        hidden_states.shape[0],
        router_weight.shape[0],
        device=hidden_states.device,
        dtype=output_dtype if output_dtype is not None else torch.float32,
    )
    _dsv3_router_gemm_op(output, hidden_states, router_weight)
    return output


class Model(nn.Module):
    """DeepSeek MoE router gate matmul with vLLM-parity three-tier dispatch.

    Mirrors the SOTA name (``vllm.../gate_linear.py:GateLinear``).  Stateless;
    the router weight is owned by the parent module and passed through
    ``forward``.
    """

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        """Compute router logits with vLLM-parity dispatch.

        Args:
            x: ``(num_tokens, hidden_size)`` activations (BF16).
            weight: ``(num_experts, hidden_size)`` gate weight (BF16).
            out_dtype: Desired output dtype. ``None`` mirrors vLLM's
                ``GateLinear.out_dtype is None`` (its CUDA default): the DSV3
                kernel emits FP32 (decode) and the F.linear fallback emits the
                weight dtype / BF16 (prefill), which is what DeepSeek-V3.2 and
                GLM-5.2 use on CUDA (``set_out_dtype`` is only called on ROCm).

        Returns:
            ``(num_tokens, num_experts)`` router logits.
        """
        num_tokens = x.shape[0]
        num_experts = weight.shape[0]
        hidden_size = weight.shape[1]

        is_hopper_or_blackwell = _is_hopper_or_blackwell()
        bf16_input = x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16

        # Tier 1: DSV3 specialized kernel. Matches vLLM's ``GateLinear.forward``
        # / ``allow_dsv3_router_gemm`` eligibility: SM90+, BF16 in, batch <=
        # ``_dsv3_max_batch`` (16 Hopper / 8 Blackwell), and a supported
        # (hidden_size, num_experts) shape — (7168, 256/384) [DeepSeek/Kimi] or
        # (6144, 256) [GLM-5.2]; (6144, 384) is unsupported. Dispatches for any
        # ``out_dtype`` (incl. None -> FP32 output), matching vLLM.
        dsv3_shape_ok = (
            (hidden_size == 7168 and num_experts in (256, 384))
            or (hidden_size == 6144 and num_experts == 256)
        )
        if (
            is_hopper_or_blackwell
            and bf16_input
            and num_tokens <= _dsv3_max_batch()
            and dsv3_shape_ok
        ):
            return _dsv3_router_gemm(x, weight, out_dtype)

        # Tier 2: cuBLAS BF16 x BF16 -> FP32. Only when FP32 output is requested
        # (vLLM's ``allow_cublas_router_gemm`` requires ``out_dtype == float32``;
        # with out_dtype None this tier is skipped, matching vLLM).
        if (
            is_hopper_or_blackwell
            and bf16_input
            and out_dtype == torch.float32
        ):
            return _router_gemm_bf16_fp32(x, weight)

        # Tier 3: F.linear fallback. Match vLLM's behaviour: cast input to
        # weight dtype, and cast the output to out_dtype only when a concrete
        # dtype was requested (out_dtype None -> keep weight/BF16 dtype, as vLLM
        # does when ``self.out_dtype is None``).
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        out = torch.nn.functional.linear(x, weight)
        if out_dtype is not None and out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### GateLinear

| count | args |
|------:|------|
| 6812 | `x:bfloat16[64, 2304] weight:bfloat16[256, 2304]` |
| 3302 | `x:bfloat16[1, 2304] weight:bfloat16[256, 2304]` |
| 2912 | `x:bfloat16[16384, 2304] weight:bfloat16[256, 2304]` |
| 1066 | `x:bfloat16[26, 2304] weight:bfloat16[256, 2304]` |
| 780 | `x:bfloat16[31, 2304] weight:bfloat16[256, 2304]` |
| 442 | `x:bfloat16[30, 2304] weight:bfloat16[256, 2304]` |
| 442 | `x:bfloat16[88, 2304] weight:bfloat16[256, 2304]` |
| 390 | `x:bfloat16[29, 2304] weight:bfloat16[256, 2304]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
