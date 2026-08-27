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

"""TRTLLM-gen MXFP4 fused MoE (via FlashInfer, Blackwell only).

This is the kernel vLLM 0.26 actually runs for gpt-oss on SM100. Its oracle logs

    Using 'FLASHINFER_TRTLLM_MXFP4_BF16' Mxfp4 MoE backend.
    Using TrtLlmMxfp4ExpertsMonolithic

and then autotunes ``flashinfer::trtllm_fp4_block_scale_moe``. The OAI Triton
``matmul_ogs`` path in :mod:`mxfp4_moe` is a *different* kernel for the same
math and is far slower here: a decode-step profile of gpt-oss-120b attributed
48% of all CUDA time to the two ``_p_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128``
launches, which is most of the 0.35x end-to-end gap against vLLM.

Two things the trtllm-gen kernel needs that the Triton one does not:

* **256-element alignment.** ``mxfp4_round_up_hidden_size_and_intermediate_size``
  rounds both ``hidden_size`` and ``intermediate_size_per_partition`` up to 256
  for the TRTLLM backends, so gpt-oss runs its experts at hidden 3072 (from
  2880) and, at tp=2, intermediate 1536 (from 1440). The activation is
  zero-padded into that width; the kernel writes an unpadded output
  (``has_unpadded_output``), so nothing has to be sliced afterwards.
* **A shuffled weight/scale layout** for the transposed MMA epilogue, plus a
  gate/up row swap because trtllm-gen defines SwiGLU with the two halves in the
  opposite order. :func:`prepare_trtllm_mxfp4_weights` is a port of vLLM's
  ``convert_gpt_oss_weight_to_mxfp4_moe_kernel_format`` TRTLLM branch.

Mirrors ``TrtLlmMxfp4ExpertsMonolithic.apply``
(``vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py``).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from flashinfer import trtllm_fp4_block_scale_moe


# vLLM's TRTLLM_BACKENDS branch of
# ``mxfp4_round_up_hidden_size_and_intermediate_size``.
TRTLLM_MXFP4_ALIGN = 256

# ``get_routing_method_type("softmax", renormalize=True, has_e_score_bias=False)``
# for gpt-oss -> RenormalizeNaive. Softmax->TopK->renormalize is the same
# function as TopK->softmax (softmax is monotonic, so the top-k set is
# identical and renormalizing the k values equals a softmax over just those
# logits), which is why the expert class accepts either spelling.
ROUTING_RENORMALIZE_NAIVE = 4

# gpt-oss SwiGLU-OAI constants; vLLM passes these as gemm1_alpha / gemm1_beta /
# gemm1_clamp_limit (``Mxfp4MoEMethod.get_fused_moe_quant_config``).
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0

# vLLM passes ``tune_max_num_tokens=max(moe_config.max_capture_size, 1)``, i.e.
# ``compilation_config.max_cudagraph_capture_size``, which for gpt-oss is 1024
# (the same value our own ``capture_cudagraph`` uses as ``max_capture_limit``).
# The autotuner tunes every token count up to this bound, so a too-small value
# leaves the larger batches on a tactic chosen for one token.
DEFAULT_TUNE_MAX_NUM_TOKENS = 1024

_MXFP4_SF_BLOCK = 32
_EPILOGUE_TILE_M = 128


def trtllm_mxfp4_moe_supported() -> bool:
    """True when the trtllm-gen MXFP4 MoE kernel can run on this device.

    Same gate as vLLM's ``TrtLlmMxfp4ExpertsBase._supports_current_device``:
    CUDA, SM100 family, FlashInfer present.

    ``FASTKERNELS_TRTLLM_MXFP4_MOE=0`` forces the Triton path instead. This
    kernel segfaults inside ``flashinfer::FP4BlockScaleLauncher::run`` on the
    first autotune profile for gpt-oss-120b at tp=1 with a 16384-token chunk
    (fine at tp=2, and fine at tp=1 with short prompts), so a switch is needed
    to isolate it and to keep that configuration runnable.
    """
    if os.environ.get("FASTKERNELS_TRTLLM_MXFP4_MOE", "1") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _swap_every_two_rows(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Swap adjacent pairs along ``axis`` (trtllm-gen's SwiGLU half order)."""
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    new_shape = list(shape)
    new_shape[axis] = shape[axis] // 2
    new_shape.insert(axis + 1, 2)
    x = x.reshape(*new_shape)
    x = x.flip(axis + 1)
    return x.reshape(*shape)


def prepare_trtllm_mxfp4_weights(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    permute_cache: dict | None = None,
) -> tuple[torch.Tensor, ...]:
    """Shuffle loaded MXFP4 expert weights into the trtllm-gen layout.

    Expects the *padded* shapes the kernel is configured for:
      ``w13_weight``       ``[E, 2*I, H // 2]``   uint8
      ``w13_weight_scale`` ``[E, 2*I, H // 32]``  uint8 (E8M0)
      ``w13_bias``         ``[E, 2*I]``
      ``w2_weight``        ``[E, H, I // 2]``     uint8
      ``w2_weight_scale``  ``[E, H, I // 32]``    uint8 (E8M0)
      ``w2_bias``          ``[E, H]``

    Returns the same six tensors in kernel layout, with the scales viewed as
    ``float8_e4m3fn`` and the biases in float32.
    """
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    if permute_cache is None:
        permute_cache = {}

    num_experts = w13_weight.shape[0]
    intermediate_size = w13_weight.shape[1] // 2
    hidden_size = w13_weight.shape[2] * 2

    w13_bias = w13_bias.to(torch.float32)
    w2_bias = w2_bias.to(torch.float32)

    # trtllm-gen's SwiGLU takes the two halves in the opposite order.
    w13_weight_scale = _swap_every_two_rows(w13_weight_scale, -2)
    w13_weight = _swap_every_two_rows(w13_weight, -2)
    w13_bias = _swap_every_two_rows(w13_bias, -1)

    g1_w, g1_s, g1_b = [], [], []
    g2_w, g2_s, g2_b = [], [], []
    for i in range(num_experts):
        idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_weight[i].view(torch.uint8), _EPILOGUE_TILE_M,
        )
        g1_w.append(
            w13_weight[i].view(torch.uint8)[idx.to(w13_weight.device)].contiguous()
        )
        sf_idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_weight_scale[i].view(torch.uint8),
            _EPILOGUE_TILE_M, num_elts_per_sf=16,
        )
        g1_s.append(
            nvfp4_block_scale_interleave(
                w13_weight_scale[i]
                .view(torch.uint8)[sf_idx.to(w13_weight_scale.device)]
                .contiguous()
            )
        )
        b_idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_bias[i].clone().reshape(-1, 1), _EPILOGUE_TILE_M,
        )
        g1_b.append(
            w13_bias[i].clone().reshape(-1, 1)[b_idx.to(w13_bias.device)].contiguous()
        )

        idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_weight[i].view(torch.uint8), _EPILOGUE_TILE_M,
        )
        g2_w.append(
            w2_weight[i].view(torch.uint8)[idx.to(w2_weight.device)].contiguous()
        )
        sf_idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_weight_scale[i].view(torch.uint8),
            _EPILOGUE_TILE_M, num_elts_per_sf=16,
        )
        g2_s.append(
            nvfp4_block_scale_interleave(
                w2_weight_scale[i]
                .view(torch.uint8)[sf_idx.to(w2_weight_scale.device)]
                .contiguous()
            )
        )
        b_idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_bias[i].clone().reshape(-1, 1), _EPILOGUE_TILE_M,
        )
        g2_b.append(
            w2_bias[i].clone().reshape(-1, 1)[b_idx.to(w2_bias.device)].contiguous()
        )

    w13_weight = torch.stack(g1_w)
    w13_weight_scale = (
        torch.stack(g1_s)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // _MXFP4_SF_BLOCK)
        .view(torch.float8_e4m3fn)
    )
    w2_weight = torch.stack(g2_w)
    w2_weight_scale = (
        torch.stack(g2_s)
        .reshape(num_experts, hidden_size, intermediate_size // _MXFP4_SF_BLOCK)
        .view(torch.float8_e4m3fn)
    )
    w13_bias = torch.stack(g1_b).reshape(num_experts, -1)
    w2_bias = torch.stack(g2_b).reshape(num_experts, -1)
    return (
        w13_weight, w13_weight_scale, w13_bias,
        w2_weight, w2_weight_scale, w2_bias,
    )


class Model(nn.Module):
    """Router + experts in one trtllm-gen launch.

    ``hidden_states`` arrives at the *padded* hidden width; the returned tensor
    is ``hidden_size_unpadded`` wide, matching vLLM's ``has_unpadded_output``.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        hidden_size_unpadded: int,
        max_capture_size: int = DEFAULT_TUNE_MAX_NUM_TOKENS,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.hidden_size_unpadded = hidden_size_unpadded
        self.max_capture_size = max(int(max_capture_size), 1)
        dev = torch.cuda.current_device()
        # Per-expert scalars, exactly as TrtLlmMxfp4ExpertsBase builds them.
        self.register_buffer(
            "gemm1_alpha",
            torch.full((num_experts,), SWIGLU_ALPHA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_beta",
            torch.full((num_experts,), SWIGLU_BETA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_clamp_limit",
            torch.full((num_experts,), SWIGLU_LIMIT, dtype=torch.float32, device=dev),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16
        output = torch.empty(
            *hidden_states.shape[:-1],
            self.hidden_size_unpadded,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16),
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_weights=w13_weight,
            gemm1_weights_scale=w13_weight_scale,
            gemm1_bias=w13_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2_weight,
            gemm2_weights_scale=w2_weight_scale,
            gemm2_bias=w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            routing_method_type=ROUTING_RENORMALIZE_NAIVE,
            do_finalize=True,
            tune_max_num_tokens=self.max_capture_size,
            output=output,
        )
        return output

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### TrtLlmMxfp4MoE

| count | args |
|------:|------|
| 6876 | `hidden_states:bfloat16[60, 3072] router_logits:bfloat16[60, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |
| 4680 | `hidden_states:bfloat16[1, 3072] router_logits:bfloat16[1, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |
| 3096 | `hidden_states:bfloat16[16384, 3072] router_logits:bfloat16[16384, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |
| 1152 | `hidden_states:bfloat16[31, 3072] router_logits:bfloat16[31, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |
| 1116 | `hidden_states:bfloat16[26, 3072] router_logits:bfloat16[26, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |
| 1080 | `hidden_states:bfloat16[29, 3072] router_logits:bfloat16[29, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |
| 684 | `hidden_states:bfloat16[30, 3072] router_logits:bfloat16[30, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |
| 576 | `hidden_states:bfloat16[67, 3072] router_logits:bfloat16[67, 128] w13_weight:uint8[128, 3072, 1536] w13_weight_scale:float8_e4m3fn[128, 3072, 96] w13_bias:float32[128, 3072] w2_weight:uint8[128, 3072, 768] w2_weight_scale:float8_e4m3fn[128, 3072, 48] w2_bias:float32[128, 3072]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
