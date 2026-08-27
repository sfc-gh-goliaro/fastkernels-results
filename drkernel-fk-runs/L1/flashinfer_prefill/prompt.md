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
import torch
import torch.nn as nn

def prime_trtllm_sinks(module: nn.Module, sinks: torch.Tensor | None) -> None:
    """Materialize the FP32 attention-sink copy the trtllm-gen kernels need.

    ``trtllm_batch_decode_with_kv_cache`` /
    ``trtllm_batch_context_with_kv_cache`` hard-check
    ``attention_sinks.dtype == float32``, while the FlashAttention build vLLM
    bundles asserts the opposite for the same weights
    (``learnable_sink must be bfloat16``).  So the conversion cannot live on the
    layer -- only the op knows which kernel it is about to call.  vLLM does the
    same conversion once per layer in
    ``FlashInferImpl.process_weights_after_loading``; call this from the owning
    attention layer's post-load hook so the copy never lands inside a forward
    or a CUDA-graph capture.
    """
    if sinks is None:
        module._sinks_fp32 = None
    elif sinks.dtype == torch.float32:
        module._sinks_fp32 = sinks
    else:
        module._sinks_fp32 = sinks.detach().to(torch.float32)
    module._sinks_src = sinks

def trtllm_sinks(module: nn.Module, s_aux: torch.Tensor | None):
    """Return the FP32 view of ``s_aux``, priming the cache if needed."""
    if s_aux is None or s_aux.dtype == torch.float32:
        return s_aux
    if module._sinks_fp32 is None or module._sinks_src is not s_aux:
        prime_trtllm_sinks(module, s_aux)
    return module._sinks_fp32

"""TRTLLM-gen paged attention prefill kernel (via FlashInfer, Blackwell only).

Accepts the same cu_seqlens-based interface as FlashAttnPrefill so that
LlamaAttention can dispatch to either backend without branch logic.
"""

import torch
import torch.nn as nn
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

from fastkernels.infra.fa_utils import FA_VERSION, flash_attn_varlen_func


class Model(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        if workspace is None:
            workspace = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self._workspace = workspace
        self._sinks_fp32: torch.Tensor | None = None
        self._sinks_src: torch.Tensor | None = None

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        prime_trtllm_sinks(self, sinks)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, s_aux=None,
                window_size=None, **kwargs):
        if block_table is not None:
            q = q.contiguous()
            seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            batch_size = seq_lens.shape[0]
            # trtllm-gen reads the page table as a dense row-major tensor; a
            # non-contiguous block_table (e.g. a column slice of a wider buffer)
            # makes every row > 0 read wrong page ids. Match vLLM, which asserts
            # is_strictly_contiguous here. See TRTLLMDecode for the full story.
            block_table = block_table.contiguous()
            seq_lens = seq_lens.contiguous()
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=(k, v),
                workspace_buffer=self._workspace,
                block_tables=block_table,
                seq_lens=seq_lens,
                max_q_len=max_seqlen_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
                bmm2_scale=1.0,
                batch_size=batch_size,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                # See TRTLLMDecode: sinks and the sliding window arrive under
                # their FlashAttention names and must be translated, not
                # swallowed by **kwargs -- dropping them corrupts numerics
                # silently.
                window_left=(
                    window_size[0] if window_size is not None
                    and window_size[0] >= 0 else -1
                ),
                sinks=trtllm_sinks(self, s_aux),
                kv_layout="HND",
            )
        # Dense (unpaged) fallback: same FlashAttention build/version vLLM
        # would use for this device.  Sinks/window must be carried across here
        # too, under FlashAttention's own parameter names.
        fa_extra = {}
        if s_aux is not None:
            fa_extra["s_aux"] = s_aux
        if window_size is not None:
            fa_extra["window_size"] = window_size
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            causal=causal,
            fa_version=FA_VERSION,
            # Compute-bound dense prefill gains nothing from KV-splitting, but the
            # FA4 (SM100 CuTe) auto heuristic still picks the split-KV kernel for
            # mid-size seqlens -- which fails to compile in this vLLM build
            # (TYPE_UNSTABLE_JOIN on ``n_block_first``).  Pin the unsplit kernel.
            num_splits=1,
            **fa_extra,
        )

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### TRTLLMPrefill

| count | args |
|------:|------|
| 188 | `q:bfloat16[804, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[2] cu_seqlens_k:int32[2] block_table:int32[1, 51] s_aux:None window_size:None` |
| 94 | `q:bfloat16[16384, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[71] cu_seqlens_k:int32[71] block_table:int32[70, 163] s_aux:None window_size:None` |
| 94 | `q:bfloat16[16315, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[58] cu_seqlens_k:int32[58] block_table:int32[57, 258] s_aux:None window_size:None` |
| 94 | `q:bfloat16[16260, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[51] cu_seqlens_k:int32[51] block_table:int32[50, 151] s_aux:None window_size:None` |
| 94 | `q:bfloat16[16211, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[52] cu_seqlens_k:int32[52] block_table:int32[51, 146] s_aux:None window_size:None` |
| 94 | `q:bfloat16[16161, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[64] cu_seqlens_k:int32[64] block_table:int32[63, 183] s_aux:None window_size:None` |
| 94 | `q:bfloat16[16099, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[23] cu_seqlens_k:int32[23] block_table:int32[22, 211] s_aux:None window_size:None` |
| 94 | `q:bfloat16[16080, 16, 128] k:bfloat16[114541, 1, 16, 128] v:bfloat16[114541, 1, 16, 128] cu_seqlens_q:int32[50] cu_seqlens_k:int32[50] block_table:int32[49, 326] s_aux:None window_size:None` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
