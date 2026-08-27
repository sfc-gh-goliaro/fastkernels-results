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

"""MLA KV cache store and gather.

Supports two cache layouts via ``kv_cache_dtype``:

* ``"auto"`` (default, matches vLLM): BF16 KV cache with shape
  ``[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]``
  (e.g. 576 BF16 elements = 1152 bytes/token for DeepSeek-V3.2).
  vLLM's ``concat_and_cache_mla`` with ``kv_cache_dtype="auto"`` writes
  ``kv_c_normed`` and ``k_pe`` directly as BF16 — no quantization.
* ``"fp8_ds_mla"``: FP8 KV cache (656 bytes/token):

  * ``[0:512]`` — ``kv_c_normed`` as FP8 (``float8_e4m3fn``).
  * ``[512:528]`` — four per-group FP32 UE8M0 scales (128 dims per group).
  * ``[528:656]`` — ``k_pe`` as 64 ``bfloat16`` values (128 bytes).

  Cache tensor shape: ``[num_blocks, block_size, 656]`` with ``dtype=torch.uint8``.

The default is BF16 to match vLLM's stock behaviour (``kv_cache_dtype=auto``
on DeepSeek-V3.2 selects BF16 KV cache). Use the ``FASTKERNELS_KV_CACHE_DTYPE``
env var to force ``fp8_ds_mla`` for extra memory savings at the cost of
numerical drift vs. vLLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("store_kvcache_fp8_mla", "store_kvcache_fp8_mla.cu")

_KV_C_DIM = 512
_K_PE_DIM = 64
_FP8_BYTES_PER_TOKEN = 656
_BF16_ELEMS_PER_TOKEN = _KV_C_DIM + _K_PE_DIM  # 576


class Model(nn.Module):
    """Store ``kv_c_normed`` and ``k_pe`` into MLA paged cache.

    Wraps vendored ``_C.concat_and_cache_mla``. Dispatches on
    ``kv_cache_dtype``:

    * ``"auto"``: expects a BF16 cache of shape
      ``[num_blocks, block_size, 576]``; the kernel writes the
      concatenation of ``kv_c_normed`` (512) and ``k_pe`` (64) directly.
    * ``"fp8_ds_mla"``: expects a uint8 cache of shape
      ``[num_blocks, block_size, 656]``; the kernel fuses per-block
      UE8M0 FP8 quantization of ``kv_c_normed`` with BF16 ``k_pe`` storage.
    * ``"fp8_e4m3"``: expects a ``float8_e4m3fn`` cache of shape
      ``[num_blocks, block_size, 576]``; the kernel divides both halves by
      ``k_scale`` and casts to fp8. This is the layout vLLM's
      FLASHINFER_MLA_SPARSE backend uses -- plain per-tensor fp8, no block
      scales -- and it is NOT interchangeable with ``fp8_ds_mla``.

    Args:
        kv_c_normed: ``[N, 512]`` BF16 — compressed KV after layernorm.
        k_pe: ``[N, 1, 64]`` or ``[N, 64]`` BF16 — RoPE key component.
        kv_cache: ``[num_blocks, block_size, 576|656]`` (BF16 / fp8 / uint8).
        slot_mapping: ``[N]`` int64 — linear slot index per token (``-1`` skips).
    """

    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"StoreKVCacheFP8MLA: unsupported kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ``k_scale`` is the per-tensor dequant scale the kernel divides by on
        # the ``fp8_e4m3`` path (it is ignored for ``auto`` and for the
        # block-scaled ``fp8_ds_mla`` layout). vLLM initialises ``layer._k_scale``
        # to 1.0 and only overwrites it from a checkpoint's calibration scales,
        # which nvidia/GLM-5.2-NVFP4 does not ship -- so ONE, not zero. A zero
        # here silently turned every stored KV element into inf/nan.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        k_pe_2d = k_pe.reshape(k_pe.shape[0], -1)
        _C.concat_and_cache_mla(
            kv_c_normed, k_pe_2d, kv_cache, slot_mapping,
            self.kv_cache_dtype, self._k_scale,
        )


class GatherKVCacheFP8MLA(nn.Module):
    """Gather and upconvert KV from FP8 MLA paged cache to BF16.

    Wraps vendored ``_C.cp_gather_and_upconvert_fp8_kv_cache``
    which gathers FP8-quantized kv_c_normed and BF16 k_pe from paged cache,
    dequantizes the FP8 portion, and writes the result as a contiguous
    BF16 workspace tensor.

    Returns:
        ``workspace``: ``[total_tokens, 576]`` BF16 — dequantized kv_c_normed
        (512 dims) concatenated with k_pe (64 dims).
    """

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        num_seqs: int,
        workspace: torch.Tensor,
    ) -> None:
        # ``seq_lens`` is unused by the kernel; lengths are implied by
        # ``workspace_starts`` + ``workspace.size(0)``. Kept in the Module API
        # for call-site compatibility.
        del seq_lens
        _C.cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache, workspace, block_table, workspace_starts, num_seqs, None,
        )


class GatherAndDequantKVCacheMLA(nn.Module):
    """Gather MLA KV cache into a BF16 workspace using the
    ``gather_and_maybe_dequant_cache`` kernel (vLLM's chunked-context helper).

    Required arguments match the kernel's signature:
        ``kv_cache``: ``[num_blocks, block_size, 576]`` BF16 (``"auto"``) or
                      ``[num_blocks, block_size, 656]`` uint8 (``fp8_ds_mla``).
        ``workspace``: ``[total_tokens, 576]`` BF16 output buffer.
        ``block_table``: ``[num_seqs, max_blocks]`` int32.
        ``cu_seq_lens``: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        ``token_to_seq``: ``[total_tokens]`` int32 mapping.
        ``total_tokens``: scalar int.
        ``workspace_starts``: ``[num_seqs]`` int32 — starting workspace row
                             per sequence (for chunked context gathers).

    ``kv_cache_dtype`` selects the source layout and must match the cache the
    owning ``MLAAttention`` allocated; vLLM likewise forwards its own
    ``self.kv_cache_dtype`` here, and passing ``"fp8_ds_mla"`` for a BF16
    cache reinterprets the bytes and silently corrupts the gathered context.
    """

    def __init__(self, kv_cache_dtype: str = "fp8_ds_mla"):
        super().__init__()
        assert kv_cache_dtype in ("auto", "fp8_ds_mla", "fp8_e4m3"), (
            f"GatherAndDequantKVCacheMLA: unsupported "
            f"kv_cache_dtype={kv_cache_dtype!r}"
        )
        self.kv_cache_dtype = kv_cache_dtype
        # ONE, not zero: on the ``fp8_e4m3`` path the kernel MULTIPLIES the
        # gathered fp8 values by this scale to dequantize (vLLM passes
        # ``layer._k_scale``, default 1.0). Zero would blank the gathered
        # context. Ignored for ``auto`` and for the block-scaled ``fp8_ds_mla``.
        self.register_buffer(
            "_k_scale", torch.ones(1, dtype=torch.float32), persistent=False,
        )

    def forward(
        self,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        total_tokens: int,
        workspace_starts: torch.Tensor,
    ) -> None:
        _C.gather_and_maybe_dequant_cache(
            kv_cache, workspace,
            block_table, cu_seq_lens, token_to_seq,
            total_tokens,
            self.kv_cache_dtype,
            self._k_scale,
            workspace_starts,
        )

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### GatherAndDequantKVCacheMLA

| count | args |
|------:|------|
| 28 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[2, 2026] cu_seq_lens:int32[3] token_to_seq:int32[65536] workspace_starts:int32[2]` |
| 14 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[1, 1468] cu_seq_lens:int32[2] token_to_seq:int32[65536] workspace_starts:int32[1]` |
| 14 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[1, 1724] cu_seq_lens:int32[2] token_to_seq:int32[65536] workspace_starts:int32[1]` |
| 14 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[1, 1980] cu_seq_lens:int32[2] token_to_seq:int32[65536] workspace_starts:int32[1]` |
| 14 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[1, 1508] cu_seq_lens:int32[2] token_to_seq:int32[65536] workspace_starts:int32[1]` |
| 14 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[1, 1764] cu_seq_lens:int32[2] token_to_seq:int32[65536] workspace_starts:int32[1]` |
| 14 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[2, 1976] cu_seq_lens:int32[3] token_to_seq:int32[65536] workspace_starts:int32[2]` |
| 14 | `kv_cache:bfloat16[182699, 64, 576] workspace:bfloat16[65536, 576] block_table:int32[1, 1402] cu_seq_lens:int32[2] token_to_seq:int32[65536] workspace_starts:int32[1]` |

### StoreKVCacheFP8MLA

| count | args |
|------:|------|
| 1834 | `kv_c_normed:bfloat16[64, 512] k_pe:bfloat16[64, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[64]` |
| 889 | `kv_c_normed:bfloat16[1, 512] k_pe:bfloat16[1, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[1]` |
| 784 | `kv_c_normed:bfloat16[16384, 512] k_pe:bfloat16[16384, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[16384]` |
| 287 | `kv_c_normed:bfloat16[26, 512] k_pe:bfloat16[26, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[26]` |
| 210 | `kv_c_normed:bfloat16[31, 512] k_pe:bfloat16[31, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[31]` |
| 119 | `kv_c_normed:bfloat16[30, 512] k_pe:bfloat16[30, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[30]` |
| 119 | `kv_c_normed:bfloat16[88, 512] k_pe:bfloat16[88, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[88]` |
| 105 | `kv_c_normed:bfloat16[29, 512] k_pe:bfloat16[29, 1, 64] kv_cache:bfloat16[182699, 64, 576] slot_mapping:int64[29]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
