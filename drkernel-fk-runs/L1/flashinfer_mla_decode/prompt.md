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

"""FlashInfer trtllm-gen MLA decode kernel (Blackwell).

FlashMLA's dense decode kernel is **SM90a-only**::

    RuntimeError: dense_attn_decode_interface,
    flashmla-src/csrc/api/dense_decode.h:29,
    Dense decode MLA is only supported on SM90a architecture

so on Blackwell it cannot run at all. vLLM selects ``FLASHINFER_MLA`` there
(logged as ``Using FLASHINFER_MLA attention backend`` /
``Using HND KV cache layout for FLASHINFER_MLA``) and dispatches decode through
``trtllm_batch_decode_with_kv_cache_mla``. This module wraps that same entry
point with the argument shape our :class:`MLAAttention` decode path already
produces, so the two run the same kernel.

Mirrors ``FlashInferMLAImpl.forward_mqa``
(``vllm/v1/attention/backends/mla/flashinfer_mla.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla


def flashinfer_mla_decode_supported() -> bool:
    """True when the trtllm-gen MLA decode kernel can run on this device."""
    if not torch.cuda.is_available():
        return False
    # trtllm-gen MLA is a Blackwell (SM100) path.
    return torch.cuda.get_device_capability()[0] >= 10


class Model(nn.Module):
    """trtllm-gen MLA decode over a paged latent KV cache.

    Parameters mirror the FlashMLA decode op this stands in for, so the
    call site does not need to branch beyond choosing the module.
    """

    # vLLM keeps one workspace per (return_lse) variant; a single shared
    # buffer is enough here since we never request the LSE.
    _WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self._workspace = workspace

    @property
    def available(self) -> bool:
        return flashinfer_mla_decode_supported()

    def ensure_workspaces(self, device: torch.device) -> None:
        """Materialize the trtllm-gen MLA decode workspace before graph capture.

        See ``TopKPerRow.ensure_workspaces``: a workspace first allocated inside
        a capture region belongs to that graph's private pool, and later graphs
        replaying against it fault.
        """
        self._get_workspace(device)

    def _get_workspace(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.zeros(
                self._WORKSPACE_BYTES, dtype=torch.uint8, device=device,
            )
        return self._workspace

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        max_seq_len: int,
        bmm2_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        q : ``[num_decodes, q_len, num_heads, qk_head_dim]``
            Same layout the FlashMLA decode op receives (``q.unsqueeze(1)``).
        kv_cache : ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``
            Paged latent cache. vLLM passes ``kv_c_and_k_pe_cache.unsqueeze(1)``,
            i.e. a singleton head dim, which we add here if absent.
        block_table, cache_seqlens
            Per-request page table and context lengths.

        Returns ``(out, None)`` to match the FlashMLA op's ``(out, lse)``.
        """
        if kv_cache.dim() == 3:
            kv_cache = kv_cache.unsqueeze(1)

        # trtllm-gen walks the page table in 128-token strides, so it rejects a
        # width that is not a multiple of ceil(128 / page_size). Callers size
        # the table from max_model_len, so pad here as a backstop; the extra
        # columns are never read (the walk is bounded by seq_lens).
        page_size = kv_cache.shape[2]
        gran = max(1, -(-128 // page_size))
        if block_table.shape[-1] % gran:
            pad = gran - block_table.shape[-1] % gran
            block_table = torch.nn.functional.pad(block_table, (0, pad))

        # trtllm-gen reads the page table and seq lens as dense row-major
        # tensors; a sliced view would make every row > 0 read at the wrong
        # stride. vLLM asserts strict contiguity for the same reason.
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q.contiguous(),
            kv_cache=kv_cache,
            workspace_buffer=self._get_workspace(q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_table.contiguous(),
            seq_lens=cache_seqlens.contiguous(),
            max_seq_len=int(max_seq_len),
            bmm1_scale=softmax_scale,
            bmm2_scale=bmm2_scale,
            return_lse=False,
        )
        return out, None

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### FlashInferMLADecode

| count | args |
|------:|------|
| 896 | `q:bfloat16[64, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[64, 2046] cache_seqlens:int32[64]` |
| 658 | `q:bfloat16[64, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[64, 2044] cache_seqlens:int32[64]` |
| 581 | `q:bfloat16[1, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[1, 2] cache_seqlens:int32[1]` |
| 308 | `q:bfloat16[1, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[1, 4] cache_seqlens:int32[1]` |
| 231 | `q:bfloat16[64, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[64, 2048] cache_seqlens:int32[64]` |
| 210 | `q:bfloat16[31, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[31, 42] cache_seqlens:int32[31]` |
| 154 | `q:bfloat16[26, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[26, 42] cache_seqlens:int32[26]` |
| 133 | `q:bfloat16[26, 1, 16, 576] kv_cache:bfloat16[182699, 64, 576] block_table:int32[26, 44] cache_seqlens:int32[26]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
