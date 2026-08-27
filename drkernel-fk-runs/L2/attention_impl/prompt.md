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
from fastkernels.infra.fa_utils import FA3_CUDA_GRAPH_MAX_NUM_SPLITS, fa3_scheduler_metadata, fa3_scheduler_metadata_size, fa_version_for_head_size, flash_attn_varlen_func
from fastkernels.infra.fa_utils import FA_VERSION as _FA_VERSION, flash_attn_varlen_func as _VLLM_FA_VARLEN_FUNC
from fastkernels.infra.fa_utils import FA_VERSION, flash_attn_varlen_func
from fastkernels.infra.fa_utils import fa3_scheduler_metadata, fa_version_for_head_size, flash_attn_varlen_func
from fastkernels.infra.kv_quant_mode import KVQuantMode
from fastkernels.infra.triton_attention_helpers import apply_alibi_to_score, apply_softcap, cdiv_fn, compute_kv_seq_mask, compute_tile_loop_bounds, find_seq_idx, init_softmax_M, load_qq_bias_tile, resolve_seq_and_query_len, softmax_step, store_segm_reduce_scalars
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from typing import Any
from typing import Optional
from typing import Optional, Tuple
import os
import torch
import torch.nn as nn
import triton
import triton.language as tl

class FlashAttnDecode(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int,
                 page_size: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.fa_version = fa_version_for_head_size(head_dim)
        self.page_size = page_size
        self._cu_seqlens_q = None
        # Persistent FA3 scheduler metadata.  vLLM builds this in
        # ``FlashAttentionMetadataBuilder`` *outside* the CUDA graph.
        self._sched_buf: torch.Tensor | None = None
        self._sched_meta: torch.Tensor | None = None
        self._graph_num_splits = FA3_CUDA_GRAPH_MAX_NUM_SPLITS
        # Jamba Hopper capture/pin: keep FA3 on num_splits=32 so eager
        # warmup cannot shrink the process-wide split-KV scratch below
        # the largest captured graph.  B200 never sets this (FA4 / TRTLLM).
        self._force_graph_splits = False
        self._window_size = (-1, -1)
        self._qkv_dtype = torch.bfloat16

    def _get_cu_seqlens_q(self, n: int, device: torch.device) -> torch.Tensor:
        needed = n + 1
        if self._cu_seqlens_q is None or self._cu_seqlens_q.numel() < needed:
            self._cu_seqlens_q = torch.arange(
                needed, dtype=torch.int32, device=device,
            )
        return self._cu_seqlens_q[:needed]

    def update_scheduler_metadata(
        self,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        qkv_dtype: torch.dtype | None = None,
        window_size: tuple[int, int] = (-1, -1),
        max_seqlen_q: int = 1,
    ) -> None:
        """Recompute FA3 tile-scheduler metadata into the persistent buffer."""
        if self.fa_version != 3:
            self._sched_meta = None
            return
        if qkv_dtype is not None:
            self._qkv_dtype = qkv_dtype
        self._window_size = window_size
        batch_size = int(cache_seqlens.shape[0])
        cu_seqlens_q = self._get_cu_seqlens_q(batch_size, cache_seqlens.device)
        meta = fa3_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self._qkv_dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=self.page_size,
            causal=True,
            window_size=window_size,
            num_splits=self._graph_num_splits,
        )
        if meta is None:
            self._sched_meta = None
            return
        n = int(meta.shape[0])
        need = max(n, fa3_scheduler_metadata_size(batch_size))
        if self._sched_buf is None or self._sched_buf.numel() < need:
            cap = max(need, fa3_scheduler_metadata_size(max(batch_size, 1024)))
            self._sched_buf = torch.zeros(
                cap, dtype=torch.int32, device=cache_seqlens.device,
            )
        self._sched_buf[:n].copy_(meta)
        self._sched_buf[n:].zero_()
        self._sched_meta = self._sched_buf[:n]

    def preallocate(self, max_batch_size: int, device: torch.device) -> None:
        """Allocate graph-stable cu_seqlens + scheduler buffers."""
        needed = max_batch_size + 1
        if self._cu_seqlens_q is None or self._cu_seqlens_q.numel() < needed:
            self._cu_seqlens_q = torch.arange(
                needed, dtype=torch.int32, device=device,
            )
        if self.fa_version == 3:
            cap = fa3_scheduler_metadata_size(max(max_batch_size, 1024))
            if self._sched_buf is None or self._sched_buf.numel() < cap:
                self._sched_buf = torch.zeros(
                    cap, dtype=torch.int32, device=device,
                )
            self._sched_meta = self._sched_buf[:fa3_scheduler_metadata_size(1)]

    def forward(self, q, k_cache, v_cache, cache_seqlens=None, **kwargs):
        max_seq_len = kwargs.pop("max_seq_len", None)
        block_table = kwargs.pop("block_table", None)
        softmax_scale = kwargs.pop("softmax_scale", None)
        kwargs.pop("causal", None)
        window_size = kwargs.get("window_size", self._window_size)

        n = q.shape[0]
        cu_seqlens_q = self._get_cu_seqlens_q(n, q.device)
        if max_seq_len is not None:
            max_seqlen_k = max_seq_len
        else:
            max_seqlen_k = int(cache_seqlens.max().item()) if cache_seqlens.numel() > 0 else 0

        capturing = torch.cuda.is_current_stream_capturing()
        use_graph_splits = (
            self.fa_version == 3
            and (capturing or self._force_graph_splits)
        )
        if use_graph_splits:
            if self._sched_meta is None:
                raise RuntimeError(
                    "FA3 CUDA-graph capture requires "
                    "update_scheduler_metadata() outside the graph first"
                )
            meta = self._sched_meta
            num_splits = self._graph_num_splits
        elif self.fa_version == 3 and cache_seqlens is not None:
            page_size = self.page_size
            if page_size is None and k_cache.dim() >= 2:
                page_size = k_cache.shape[1]
            meta = fa3_scheduler_metadata(
                batch_size=int(cache_seqlens.shape[0]),
                max_seqlen_q=1,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=self.num_heads,
                num_heads_kv=self.num_kv_heads,
                headdim=self.head_dim,
                cache_seqlens=cache_seqlens,
                qkv_dtype=q.dtype,
                cu_seqlens_q=cu_seqlens_q,
                page_size=page_size,
                causal=True,
                window_size=window_size,
                num_splits=0,
            )
            num_splits = 0
        else:
            meta = None
            num_splits = 0

        fa_kw = dict(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=block_table,
            fa_version=self.fa_version,
            num_splits=num_splits,
        )
        if meta is not None:
            fa_kw["scheduler_metadata"] = meta
        fa_kw.update(kwargs)
        return flash_attn_varlen_func(**fa_kw)

class FlashAttnPrefill(nn.Module):
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

class TRTLLMDecode(nn.Module):
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

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, s_aux=None, window_size=None, **kwargs):
        if max_seq_len is None:
            max_seq_len = int(cache_seqlens.max().item())
        # trtllm-gen requires a contiguous query: with a batched (multi-request)
        # decode, the query view is non-contiguous and the TMA load reads later
        # rows at the wrong stride -> only row 0 is correct, the rest are garbage.
        # vLLM's FlashInfer backend and our own TRTLLMPrefill both do this; the
        # decode path was missing it.
        q = q.contiguous()
        # block_tables / seq_lens MUST be contiguous: the trtllm-gen kernel
        # reads the page table assuming a dense [batch, max_pages] row-major
        # layout. The engine's eager/CUDA-graph decode buffers hand us a column
        # slice (``_eager_block_tables[:n, :bt_cols]``) whose row stride is the
        # full ``max_num_blocks``, not ``bt_cols`` -> every row > 0 would read
        # its page ids from the wrong offset (garbage pages), so only row 0
        # stayed correct and all other sequences in the batch were corrupted.
        # vLLM likewise asserts is_strictly_contiguous(block_tables/seq_lens).
        block_table = block_table.contiguous()
        if cache_seqlens is not None:
            cache_seqlens = cache_seqlens.contiguous()
        # Attention sinks and the sliding window must be forwarded explicitly.
        # The caller names them ``s_aux`` / ``window_size`` (the FlashAttention
        # spelling); trtllm-gen calls them ``sinks`` / ``window_left``. Letting
        # them fall into **kwargs silently dropped both, which is a *numerical*
        # bug, not a crash: gpt-oss-120b (sinks + alternating sliding window)
        # scored 0.8 of 385 matching tokens against vLLM. vLLM passes both here
        # (flashinfer.py: window_left=self.window_left, sinks=self.sinks).
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            bmm2_scale=1.0,
            window_left=(
                window_size[0] if window_size is not None
                and window_size[0] >= 0 else -1
            ),
            sinks=trtllm_sinks(self, s_aux),
            kv_layout="HND",
        )

class TRTLLMPrefill(nn.Module):
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

def _merge_state_kernel(
    output,         # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,     # [NUM_TOKENS, NUM_HEADS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,     # [NUM_TOKENS, NUM_HEADS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,     # [NUM_TOKENS, NUM_HEADS]
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    LSE_HEAD_MAJOR: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    if LSE_HEAD_MAJOR:
        lse_offset = head_idx * num_tokens + token_idx
    else:
        lse_offset = token_idx * num_heads + head_idx
    p_lse = tl.load(prefix_lse + lse_offset)
    s_lse = tl.load(suffix_lse + lse_offset)
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    out_se = tl.exp(p_lse) + tl.exp(s_lse)

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + token_idx * num_heads + head_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * HEAD_SIZE
        + head_idx * HEAD_SIZE
        + head_arange,
        mask=head_mask,
    )

    p_scale = tl.exp(p_lse) / out_se
    s_scale = tl.exp(s_lse) / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out,
        mask=head_mask,
    )

def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
    lse_head_major: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge ``(prefix_output, prefix_lse)`` and ``(suffix_output, suffix_lse)``.

    Shapes (token-major):
      - ``*_output``: ``[num_tokens, num_heads, head_size]``
      - ``*_lse``:    ``[num_tokens, num_heads]`` (NOT FA3's transposed layout)

    Returns ``(merged_output, merged_lse_or_None)``.
    """
    if output is None:
        output = torch.empty_like(prefix_output)
    write_lse = output_lse is not None
    if output_lse is None:
        # Dummy pointer: the Triton kernel does not touch it when
        # ``OUTPUT_LSE`` is false. Avoid allocating/writing LSE for tree
        # attention, which only consumes the merged output.
        output_lse = prefix_lse

    num_tokens = output.shape[0]
    num_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)

    _merge_state_kernel[(num_tokens, num_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        head_size,
        padded_head_size,
        write_lse,
        lse_head_major,
    )
    return output, output_lse if write_lse else None

def _vllm_fa_paged(q, k_cache, v_cache, cu_seqlens_q, seqused_k,
                   max_seqlen_q, max_seqlen_k, block_table, softmax_scale):
    out, lse = _VLLM_FA_VARLEN_FUNC(
        q, k_cache, v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=False,
        return_softmax_lse=True,
        fa_version=_FA_VERSION,
    )
    return out, lse

class TreeAttnPrefill(nn.Module):
    """Verify-step attention via cascade (two batched calls + LSE merge)."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table_prefix: torch.Tensor,
        cache_seqlens_prefix: torch.Tensor,
        cu_seqlens_q_prefix: torch.Tensor,
        max_seqlen_q_prefix: int,
        max_seqlen_k_prefix: int,
        page_table_expand: torch.Tensor,
        cache_seqlens_expand: torch.Tensor,
        cu_seqlens_q_expand: torch.Tensor,
        max_seqlen_k_expand: int,
        block_size: int,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q : [B*N, H_q, D]
            Verify queries (N = num_draft_tokens) flattened across batch.
        k_cache, v_cache : [num_blocks, block_size, H_kv, D]
            Paged KV cache (NHD layout). Prefix + draft tokens already written.
        block_table_prefix : [B, max_pages] int32
            Block-level page table for the prefix pass.
        cache_seqlens_prefix : [B] int32
            Prefix length per sequence (== ``t_committed_len[i]``).
        cu_seqlens_q_prefix : [B+1] int32 = [0, N, 2N, ..., B*N]
        max_seqlen_q_prefix : int = N
        max_seqlen_k_prefix : int = max(prefix lengths)
        page_table_expand : [B*N, N] int32
            Token-level slot indices for the expand pass. Each row i contains
            up to N slots; the first ``cache_seqlens_expand[i]`` are the draft
            tokens this query is allowed to attend to (sorted "live" first).
        cache_seqlens_expand : [B*N] int32
            Number of attended draft tokens per query.
        cu_seqlens_q_expand : [B*N+1] int32 = arange(B*N+1)
        max_seqlen_k_expand : int  (== N)
        block_size : int
            Page size of the underlying KV cache.
        """
        scale = softmax_scale if softmax_scale is not None else self.sm_scale
        H_kv = self.num_kv_heads
        D = self.head_dim

        kc_blk = k_cache.view(-1, block_size, H_kv, D)
        vc_blk = v_cache.view(-1, block_size, H_kv, D)

        o_prefix, lse_prefix = _vllm_fa_paged(
            q, kc_blk, vc_blk,
            cu_seqlens_q=cu_seqlens_q_prefix,
            seqused_k=cache_seqlens_prefix,
            max_seqlen_q=max_seqlen_q_prefix,
            max_seqlen_k=max_seqlen_k_prefix,
            block_table=block_table_prefix,
            softmax_scale=scale,
        )

        kc_tok = k_cache.view(-1, 1, H_kv, D)
        vc_tok = v_cache.view(-1, 1, H_kv, D)
        o_expand, lse_expand = _vllm_fa_paged(
            q, kc_tok, vc_tok,
            cu_seqlens_q=cu_seqlens_q_expand,
            seqused_k=cache_seqlens_expand,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k_expand,
            block_table=page_table_expand,
            softmax_scale=scale,
        )

        out, _ = merge_state(
            o_prefix, lse_prefix,
            o_expand, lse_expand,
            lse_head_major=True,
        )
        return out

_FP8_DTYPE = torch.float8_e4m3fn

float8_info = torch.finfo(_FP8_DTYPE)

def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)

def _is_device_capability_family(capability: int) -> bool:
    """Mirror vLLM's ``current_platform.is_device_capability_family``.

    Returns True if the current CUDA device capability is any <major>.x
    matching ``capability`` (e.g. 100 matches all 10.x / Blackwell parts).
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return ((major * 10 + minor) // 10) == (capability // 10)

def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)

def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32

def _load_q_td(
    query_ptr,
    q_block_local_len,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    cur_batch_in_all_start_index,
    q_block_local_idx,
    kv_head_idx,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load Q via a 2D tensor descriptor.

    Caller guarantees (via the wrapper's ``use_td_qo`` gate):
      * ``HEAD_SIZE == HEAD_SIZE_PADDED`` (head_size is a power of 2),
      * ``num_queries_per_kv`` is a power of 2,
      * the ``num_queries_per_kv`` heads of the current KV group are
        contiguous in memory (``query_stride_1 == HEAD_SIZE``, which is
        the default vLLM query layout).

    Under those preconditions the inner two axes are flattened into one
    row of size ``num_queries_per_kv * HEAD_SIZE`` with stride 1, which
    avoids the non-power-of-2 ``block_shape`` error from the Triton
    tensor-descriptor validator.  Returns (BLOCK_M, HEAD_SIZE_PADDED).
    """
    q_base = (
        query_ptr
        + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
        + (kv_head_idx * num_queries_per_kv) * query_stride_1
    )
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
        strides=(query_stride_0, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED),
    )
    return q_desc.load([0, 0]).reshape(BLOCK_M, HEAD_SIZE_PADDED)

def _load_kv_tile_td(
    cache_ptr,
    physical_block_idx_scalar,
    kv_head_idx,
    offset_in_block,
    stride_cache_0: tl.int64,
    stride_cache_1: tl.int64,
    stride_cache_2: tl.int64,
    stride_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load a KV cache tile via tensor descriptor.

    Returns shape (TILE_SIZE, HEAD_SIZE_PADDED). Caller transposes for K.
    Tensor descriptors zero-pad reads beyond the shape boundary, so
    ``HEAD_SIZE_PADDED > HEAD_SIZE`` is handled correctly.
    """
    base = (
        cache_ptr
        + physical_block_idx_scalar * stride_cache_0
        + kv_head_idx * stride_cache_2
    )
    desc = tl.make_tensor_descriptor(
        base=base,
        shape=(BLOCK_SIZE, HEAD_SIZE),
        strides=(stride_cache_1, stride_cache_3),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
    )
    return desc.load([offset_in_block, 0])

def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Modes handled inside the core kernel:

    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
      their scales separately on S/P inside the loop.
    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
      tensor-wide scale, unless Q is also FP8 and the caller folds the scales
      into the attention score and output accumulator.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)

def _store_output_td(
    base_ptr,
    acc,
    q_block_local_len,
    stride_token: tl.int64,
    stride_head: tl.int64,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Store an output tile via a tensor descriptor.

    The 2D and 3D epilogues differ only in ``base_ptr`` and the
    ``(stride_token, stride_head)`` pair: 2D writes directly to the
    flat output buffer, 3D writes to a single per-segment slice of
    ``segm_output_ptr``.  Descriptor shape / block_shape / reshape
    are the same in both modes, so share one helper.
    """
    acc = acc.to(base_ptr.dtype.element_ty)
    output_desc = tl.make_tensor_descriptor(
        base=base_ptr,
        shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(stride_token, stride_head, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )
    output_desc.store(
        [0, 0, 0],
        acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )

def kernel_unified_attention(
    # Output destination for the 2D path.  In 3D mode per-segment partials
    # go to the ``segm_*`` tensors (see bottom of signature) and
    # ``output_ptr`` is unused (callers may pass any non-null pointer).
    output_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Scalars
    scale,
    q_scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_CAUSAL: tl.constexpr,  # bool
    USE_PER_SEQ_CAUSAL: tl.constexpr,  # bool
    per_seq_causal_ptr,  # [num_seqs] bool, or None
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    rswa_prefix_lens_ptr,
    R_SWA_WINDOW: tl.constexpr,  # int
    USE_R_SWA: tl.constexpr,  # bool
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    # Parameters below default to None so Triton can skip materialising them
    # on call sites where the corresponding constexpr branch is dead.
    # Credit: @quinnlp identified this as a perf regression source in
    # intel/intel-xpu-backend-for-triton#6758 (review comment r3204641104).
    # Per-segment outputs: used in 3D mode; unused in 2D (IS_3D=False).
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    # ``tl.int64`` cannot be combined with a ``None`` default — Triton's JIT
    # rejects ``Optional[tl.int64]`` / ``tl.int64 | None`` at trace time, and
    # plain ``tl.int64 = None`` raises ``TypeError: 'NoneType' object cannot
    # be interpreted as an integer`` when callers omit these arguments.
    # ``int | None`` is the only annotation that lets the wrapper pass
    # ``None`` here so Triton can skip materialising the strides when the
    # ``USE_PER_TOKEN_HEAD_SCALES`` branch is dead.
    stride_ks_blk: int | None = None,
    stride_ks_slot: int | None = None,
    stride_ks_head: int | None = None,
    stride_vs_blk: int | None = None,
    stride_vs_slot: int | None = None,
    stride_vs_head: int | None = None,
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3). Sub-byte INT4 (4) uses its own
    # int4_per_token_head kernel, not this one.
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    # Tensor-descriptor load/store for HW 2D block reads on Intel Xe2/Xe3.
    # ``USE_TD`` gates KV tile loads; ``USE_TD_QO`` separately gates Q/output
    # (see ``unified_attention`` wrapper for the gating rules).
    USE_TD: tl.constexpr = False,
    USE_TD_QO: tl.constexpr = False,
    Q_IS_FP8: tl.constexpr = False,
    # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window
    # instead of letting them override it. Default False preserves the
    # original (causal AND SW) OR mm_prefix behavior for all other models.
    MM_PREFIX_CLAMP_SW: tl.constexpr = False,
):
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = (KV_QUANT_MODE >= 2) and (
        KV_QUANT_MODE <= 3
    )
    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    # Number of valid query rows in this block (used by TD descriptor
    # shapes, but always computed so the variable stays in scope).
    q_block_local_len = tl.minimum(
        BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    if USE_TD_QO:
        Q = _load_q_td(
            query_ptr,
            q_block_local_len,
            query_stride_0,
            query_stride_1,
            cur_batch_in_all_start_index,
            q_block_local_idx,
            kv_head_idx,
            num_queries_per_kv,
            BLOCK_Q,
            BLOCK_M,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    score_scale = scale
    value_scale = 1.0
    if USE_FP8_Q_DESCALE:
        score_scale = scale * tl.load(q_scale) * tl.load(k_scale)
        value_scale = tl.load(v_scale)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX or USE_R_SWA,
        IS_3D,
        USE_CAUSAL,
        USE_PER_SEQ_CAUSAL,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        if USE_TD:
            # All TILE_SIZE slots within a single KV tile map to one
            # physical block (guaranteed by ``BLOCK_SIZE % TILE_SIZE == 0``
            # from the static_assert above), so load the block index as
            # a scalar instead of a broadcast reduction.
            offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE
            physical_block_scalar = tl.load(
                block_tables_ptr + block_table_offset + (j * TILE_SIZE) // BLOCK_SIZE
            ).to(tl.int64)
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = _load_kv_tile_td(
                key_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            ).T
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = _load_kv_tile_td(
                value_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            USE_CAUSAL,
            USE_PER_SEQ_CAUSAL,
            per_seq_causal_ptr,
            rswa_prefix_lens_ptr,
            R_SWA_WINDOW,
            USE_R_SWA,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            MM_PREFIX_CLAMP_SW,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (score_scale * k_token_head_scales[None, :])
        else:
            S += score_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_PER_SEQ_CAUSAL:
                is_causal_seq = tl.load(per_seq_causal_ptr + seq_idx)
                sw_mask_v = tl.where(
                    is_causal_seq,
                    dist < SLIDING_WINDOW,
                    (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW),
                )
            elif USE_CAUSAL:
                sw_mask_v = dist < SLIDING_WINDOW
            else:
                sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
            V = tl.where(sw_mask_v, V, 0.0)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        # Store per-segment partials; finalized by ``reduce_segments``.
        if USE_TD_QO:
            # 3D target: segm_output[token, head, segm_idx, :].  Advance
            # the base to the correct (token-start, head-start, segm)
            # slice; strides step between tokens / heads of the flattened
            # (T, H, SEGS, PAD) layout.
            segm_base = (
                segm_output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q).to(
                    tl.int64
                )
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + (kv_head_idx * num_queries_per_kv)
                * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
            )
            _store_output_td(
                segm_base,
                acc,
                q_block_local_len,
                num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            segm_output_offset = (
                query_offset_0[:, None].to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
                + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
            )
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        if USE_TD_QO:
            # 2D target: flat output[token, head, :].  Strides come
            # straight from the caller (``output_stride_0`` per token,
            # ``output_stride_1`` per head).
            output_base = (
                output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
                * output_stride_0
                + (kv_head_idx * num_queries_per_kv) * output_stride_1
            )
            _store_output_td(
                output_base,
                acc,
                q_block_local_len,
                output_stride_0,
                output_stride_1,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            output_offset = (
                query_offset_0[:, None] * output_stride_0
                + query_offset_1[:, None] * output_stride_1
                + offs_d[None, :]
            )
            tl.store(
                output_ptr + output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )

is_batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"

def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    # R-SWA support: prefix tokens stay globally visible, generated tokens use
    # a fixed sliding window.
    rswa_prefix_lens=None,
    rswa_window: int | None = None,
    use_alibi_sqrt=False,
    # KV cache quantization mode and per-token-head scale caches.
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    # Chunked attention: restrict attention to aligned blocks with lookback.
    chunk_lookback=-1,
    # Tensor-descriptor mode: use ``tl.make_tensor_descriptor`` for Q/K/V
    # loads and output stores.  Enables HW 2D block reads on Intel Xe2/Xe3.
    # The non-TD branch is dead-code-eliminated at Triton compile time so
    # disabling this flag costs nothing.
    use_td: bool = False,
    # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window.
    # Default False keeps the original behavior for every other model.
    mm_prefix_clamp_sliding_window: bool = False,
):
    # Resolve causal: bool or per-seq tensor.
    use_per_seq_causal = isinstance(causal, torch.Tensor)
    use_causal = bool(causal) if not use_per_seq_causal else True
    per_seq_causal_ptr = causal if use_per_seq_causal else None

    # Sub-byte packed mode (INT4) needs a bespoke kernel (split-dot +
    # sub-byte unpack); everything else goes through the core kernel below.
    if kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        assert use_causal and not use_per_seq_causal, (
            "INT4_PER_TOKEN_HEAD only supports causal attention"
        )
        # INT4_PER_TOKEN_HEAD uses a bespoke sub-byte kernel that is not
        # vendored here (fastkernels only exercises KVQuantMode.NONE).
        raise NotImplementedError(
            "INT4_PER_TOKEN_HEAD KV cache mode is not vendored in fastkernels; "
            "only KVQuantMode.NONE is supported by this Triton path."
        )

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_per_token_head_scales = kv_quant_mode in (
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
        )

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_rswa = rswa_window is not None and rswa_prefix_lens is not None

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Tuned launch parameters; ``None`` lets Triton pick its defaults.
    launch_num_warps: int | None = None
    launch_num_stages: int | None = None

    # head_size 256 with many query rows per sequence (e.g. diffusion-gemma
    # bidirectional canvas passes) is prefill-shaped, but the decode-oriented
    # defaults (BLOCK_Q=8, TILE=32, 4 warps) under-tile it. A wider KV tile +
    # more query rows per block + 8 warps is ~2x faster on B200.
    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
        and _is_device_capability_family(100)
    )
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    # Wider KV tile for the tuned large-head path (see above). Only the 2D
    # path (used when max_seqlen_q > 1) reads TILE_SIZE_PREFILL.
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128

    # USE_TD requires BLOCK_SIZE % TILE_SIZE == 0 (enforced by a
    # ``tl.static_assert`` in the kernel).  The default prefill tile
    # size (32) is larger than a common ``block_size=16``, so clamp it
    # down when TD is enabled.  Zero overhead when disabled.
    if use_td:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, block_size)

    # Tensor descriptors for Q load / output store require every element
    # of ``block_shape`` to be a power of 2.  ``num_queries_per_kv`` is
    # not always pow2 (e.g. Qwen2-7B: 28 / 4 = 7), so gate the Q/O paths
    # separately from the KV tile loads (whose ``block_shape`` does not
    # include ``num_queries_per_kv``).
    #
    # The Q/O descriptors also encode ``HEAD_SIZE_PADDED`` on the inner
    # axis while the backing buffers (both flat output and per-segment
    # output) are laid out with ``HEAD_SIZE``.  When they differ (e.g.
    # Phi-3's head_size=96 → HEAD_SIZE_PADDED=128) the store would spill
    # padded lanes into neighbouring heads because tensor-descriptor
    # stores don't mask the padded tail.  Fall back to the pointer path
    # for Q/O in that case — KV tile loads are unaffected because their
    # ``shape`` already matches ``block_shape`` on the inner axis.
    head_size_padded = triton.next_power_of_2(head_size)
    _is_pow2_nq = (num_queries_per_kv & (num_queries_per_kv - 1)) == 0
    _is_pow2_hs = head_size == head_size_padded
    use_td_qo = use_td and _is_pow2_nq and _is_pow2_hs

    # ``_load_q_td`` / ``_store_output_td`` flatten ``(num_queries_per_kv,
    # HEAD_SIZE)`` into a single contiguous inner axis.  That's only
    # equivalent to the pointer path when the ``num_queries_per_kv`` heads
    # for this KV group start at ``kv_head_idx * num_queries_per_kv`` and
    # lie exactly HEAD_SIZE apart — i.e. ``query_stride_1 == HEAD_SIZE``
    # and ``output_stride_1 == head_size``.  This is the default vLLM
    # query/output layout; assert it explicitly so we fail fast if a
    # future caller passes a non-contiguous query tensor.
    if use_td_qo:
        assert q.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous query heads "
            f"(q.stride(1) = {q.stride(1)} != head_size = {head_size}); "
            f"set VLLM_TRITON_USE_TD=0 or pad the query layout."
        )
        assert out.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous output heads "
            f"(out.stride(1) = {out.stride(1)} != head_size = {head_size})."
        )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are passed as ``None`` when the
    # ``USE_PER_TOKEN_HEAD_SCALES`` branch is dead so Triton can skip
    # materialising those arguments and the associated registers.
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = None
        vs_blk = vs_slot = vs_head = None
        k_scale_ptr = None
        v_scale_ptr = None
    # 3D needs real segm tensors; 2D never touches them.  Pass ``None`` in
    # 2D mode so Triton can skip materialising these pointer arguments.
    segm_output_ptr = softmax_segm_output if use_3d else None
    segm_max_ptr = softmax_segm_max if use_3d else None
    segm_expsum_ptr = softmax_segm_expsum if use_3d else None
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    launch_kwargs: dict[str, int] = {}
    if launch_num_warps is not None:
        launch_kwargs["num_warps"] = launch_num_warps
    if launch_num_stages is not None:
        launch_kwargs["num_stages"] = launch_num_stages

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        scale=softmax_scale,
        q_scale=q_descale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_CAUSAL=use_causal,
        USE_PER_SEQ_CAUSAL=use_per_seq_causal,
        per_seq_causal_ptr=per_seq_causal_ptr,
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        rswa_prefix_lens_ptr=rswa_prefix_lens if use_rswa else seqused_k,
        R_SWA_WINDOW=rswa_window or 0,
        USE_R_SWA=use_rswa,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        KV_QUANT_MODE=kv_quant_mode,
        Q_IS_FP8=(q.dtype == _FP8_DTYPE),
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        USE_TD=use_td,
        USE_TD_QO=use_td_qo,
        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,
        **launch_kwargs,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )

def _store_kvcache_hnd_kernel(
    key_ptr, key_stride_n, value_ptr, value_stride_n,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Store KV into HND layout [num_blocks, num_kv_heads, block_size, head_dim]."""
    idx = tl.program_id(0)
    head = tl.program_id(1)
    # int64: same overflow as the NHD store.  ``block_idx * H * page * D``
    # in int32 wraps once block_idx >= 2^31 / (H * page * D) (131072 for
    # Jamba Mini H=8, page=16, D=128 -- under the 201k-block B200 pool).
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    if slot < 0:
        return
    block_idx = slot // PAGE_SIZE
    slot_in_block = slot % PAGE_SIZE
    src_k_offset = idx * key_stride_n + head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    src_v_offset = idx * value_stride_n + head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    dst_offset = (
        block_idx * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
        + head * PAGE_SIZE * HEAD_DIM
        + slot_in_block * HEAD_DIM
        + tl.arange(0, HEAD_DIM)
    )
    k = tl.load(key_ptr + src_k_offset)
    v = tl.load(value_ptr + src_v_offset)
    tl.store(k_cache_ptr + dst_offset, k)
    tl.store(v_cache_ptr + dst_offset, v)

class StoreKVCacheHND(nn.Module):
    """HND layout store: [num_blocks, num_kv_heads, block_size, head_dim]."""
    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        N, num_kv_heads, head_dim = key.shape
        if slot_mapping.dtype != torch.int64:
            slot_mapping = slot_mapping.to(torch.int64)
        _store_kvcache_hnd_kernel[(N, num_kv_heads)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping,
            PAGE_SIZE=self.page_size,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
        )

def _store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
):
    idx = tl.program_id(0)
    # int64: Hopper hybrid pages are large.  ``slot * D`` in int32
    # overflows at slot >= 2^31/D (bid >= 65536 when D=2048).
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    if slot < 0:
        return
    offsets = tl.arange(0, D_PAD)
    mask = offsets < D
    key = tl.load(key_ptr + idx * key_stride + offsets, mask=mask)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask=mask)
    dst = slot * D + offsets
    tl.store(k_cache_ptr + dst, key, mask=mask)
    tl.store(v_cache_ptr + dst, value, mask=mask)

class StoreKVCache(nn.Module):
    """NHD layout store: [num_blocks, block_size, num_kv_heads, head_dim]."""
    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        D_PAD = triton.next_power_of_2(D)
        if slot_mapping.dtype != torch.int64:
            slot_mapping = slot_mapping.to(torch.int64)
        _store_kvcache_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D, D_PAD,
        )

"""vLLM-aligned Attention layer with paged KV cache.

Mirrors vLLM's ``Attention`` class (from
``vllm/model_executor/layers/attention/attention.py``):

    forward(query, key, value) -> torch.Tensor

Inputs and outputs are **flat** ``[N, num_heads * head_dim]`` tensors.
KV cache metadata is obtained from the global ``Context`` (via
``get_context()``), matching vLLM's ``get_forward_context()`` pattern.

Backend selection (flash_attn vs TRTLLM-gen) is handled at init time
via ``AttnBackendConfig``.  The engine discovers this module for KV cache
assignment through duck-typing (``hasattr(module, "k_cache")``).

TODO(tech-debt): CUDA graph capture is incompatible with chunked local
attention because the metadata remapping (cu_seqlens, block_tables) varies
per batch.  vLLM disables CUDA graphs when chunked local attention is
active.  If/when we add CUDA graph support, we need to handle this case.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.infra.context import get_context, get_attn_backend_config

_TRITON_MIN_LAUNCH_GRID_SIZE_2D = 128
_TRITON_NUM_PAR_SOFTMAX_SEGMENTS = 16

# Largest head size the trtllm-gen paged kernels advertise.  Above this,
# SM100 layers drop to TRITON_ATTN (FA4 TMEM-rejects 512).  Hopper uses FA4
# for those heads instead; see ``_triton_only`` below.
_TRTLLM_MAX_HEAD_SIZE = 256

from fastkernels.infra.kv_quant_mode import KVQuantMode as _VllmKVQuantMode
import inspect as _inspect

_TRITON_UNIFIED_ACCEPTS_KV_QUANT = (
    "kv_quant_mode" in _inspect.signature(_triton_unified_attention).parameters
)


def _chunked_prefill_remap(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
    """Remap prefill metadata into chunked local-attention virtual batches.

    Follows vLLM's ``make_local_attention_virtual_batches`` algorithm: each
    original sequence is split into ``attention_chunk_size``-wide chunks that
    the kernel sees as independent sequences.

    Returns (cu_seqlens_q', cu_seqlens_k', max_seqlen_q', max_seqlen_k',
             block_tables').
    """
    device = cu_seqlens_q.device
    cu_q_np = cu_seqlens_q.cpu().numpy()
    cu_k_np = cu_seqlens_k.cpu().numpy()

    q_seqlens = cu_q_np[1:] - cu_q_np[:-1]
    k_seqlens = cu_k_np[1:] - cu_k_np[:-1]
    batch_size = len(q_seqlens)

    q_tokens_in_first_block = np.minimum(
        attention_chunk_size - ((k_seqlens - q_seqlens) % attention_chunk_size),
        q_seqlens,
    ).astype(np.int32)
    tokens_in_last_block = (
        attention_chunk_size + (k_seqlens % -attention_chunk_size)
    ).astype(np.int32)

    local_blocks = (
        1 + np.ceil(
            np.maximum(q_seqlens - q_tokens_in_first_block, 0) / attention_chunk_size
        ).astype(np.int32)
    )

    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = int(cu_num_blocks[-1])

    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1

    seqlens_q_local = np.repeat(
        q_seqlens - q_tokens_in_first_block, local_blocks,
    ).astype(np.int32)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attention_chunk_size * (arange - 1),
        attention_chunk_size,
    )[arange > 0]

    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    seqlens_k_local = np.full(virtual_batches, attention_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    cu_seqlens_k_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_k_local, out=cu_seqlens_k_local[1:])
    cu_seqlens_k_local[0] = 0

    max_seqlen_q = int(seqlens_q_local.max()) if virtual_batches > 0 else 0
    max_seqlen_k = int(seqlens_k_local.max()) if virtual_batches > 0 else 0

    cu_q_out = torch.from_numpy(cu_seqlens_q_local).to(device=device)
    cu_k_out = torch.from_numpy(cu_seqlens_k_local).to(device=device)

    block_tables_out = None
    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size

        k_seqstarts_absolute = np.repeat(k_seqlens, local_blocks) - (
            rarange * attention_chunk_size
            + np.repeat(tokens_in_last_block, local_blocks)
        )
        block_starts = k_seqstarts_absolute // block_size

        block_indices = (
            block_starts[:, None]
            + np.arange(pages_per_chunk, dtype=np.int32)
        )
        block_indices = block_indices.reshape(-1).clip(
            max=block_tables.shape[1] - 1,
        )
        batch_indices = np.repeat(
            np.arange(batch_size, dtype=np.int32),
            local_blocks * pages_per_chunk,
        )

        bi_torch = torch.from_numpy(batch_indices)
        bk_torch = torch.from_numpy(block_indices)
        block_tables_out = block_tables[bi_torch, bk_torch].view(
            virtual_batches, -1,
        )

    return cu_q_out, cu_k_out, max_seqlen_q, max_seqlen_k, block_tables_out


def _chunked_decode_remap(
    cache_seqlens: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Remap decode metadata so the kernel only attends within the last chunk.

    Returns (cache_seqlens', block_tables', max_context_len').
    """
    local_seqlens = torch.clamp(cache_seqlens, max=attention_chunk_size)
    max_context_len = int(local_seqlens.max().item()) if local_seqlens.numel() > 0 else 0

    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size
        chunk_start_page = (cache_seqlens - local_seqlens) // block_size
        offsets = torch.arange(pages_per_chunk, device=block_tables.device)
        page_indices = chunk_start_page.unsqueeze(1) + offsets
        page_indices = page_indices.clamp(max=block_tables.shape[1] - 1)
        block_tables = torch.gather(block_tables, 1, page_indices)

    return local_seqlens, block_tables, max_context_len


class Model(nn.Module):

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None,
                 attention_chunk_size: int | None = None,
                 prefer_triton: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.sinks = sinks
        self.attention_chunk_size = attention_chunk_size

        # TODO(tech-debt): For chunked local attention layers the KV cache
        # could be limited to ``attention_chunk_size`` tokens per layer instead
        # of ``max_seq_len``, following vLLM's ``ChunkedLocalAttentionSpec``.
        # This is not needed for correctness but would reduce memory usage.
        self.k_cache = self.v_cache = torch.tensor([])

        attn_cfg = get_attn_backend_config()
        self._block_size = attn_cfg.block_size

        # Per-layer backend selection, mirroring vLLM's per-KV-cache-group
        # choice rather than one global backend.  Reproduced by running
        # ``CudaPlatform.get_valid_backends`` for each config; on SM100:
        #
        #   head_size 128/256, DECODER      -> FLASHINFER  (trtllm-gen)
        #   ENCODER_ONLY / ENCODER_DECODER  -> FLASH_ATTN  ("attention type
        #       not supported" excludes FlashInfer) -- see whisper_attention
        #   PrefixLM bidirectional (Gemma4 sliding, 256) -> TRITON_ATTN
        #       (FlashInfer rejects mm_prefix; FA3 fails "mm_prefix requires
        #       FA4").  Gemma4 global layers (512) stay Triton on SM100
        #       because FA4 TMEM-rejects head_size>128; on Hopper they run
        #       FLASH_ATTN FA4 (FA3 caps at 256, FA4 is valid on SM90).
        #
        # ``prefer_triton`` opts a layer into the mm_prefix / FA3-reject case.
        # ``head_size > 256`` is only forced to Triton on the trtllm (SM100)
        # path; Hopper uses FA4 for those heads.
        #
        # The Triton unified kernel indexes the cache as
        # ``[num_blocks, block_size, num_kv_heads, head_size]`` (NHD), as does
        # the SDPA fallback's ``_cache_seq``, so a layer routed away from
        # trtllm-gen must also be *allocated* NHD -- hence the layout is a
        # per-layer property the engine reads back when sizing the cache.
        # Note this does not depend on whether the Triton kernel imported: an
        # HND cache would silently transpose the head and block dims for
        # either consumer.
        self._triton_only = prefer_triton or (
            head_size > _TRTLLM_MAX_HEAD_SIZE and attn_cfg.use_trtllm
        )
        self._use_trtllm = attn_cfg.use_trtllm and not self._triton_only
        self.kv_layout = "HND" if self._use_trtllm else "NHD"

        # Native FA3/TRTLLM path: sinks -> s_aux, sliding window -> window_size.
        # Held as a plain attribute (not a submodule parameter): the sinks
        # Parameter is already owned by the enclosing attention block, and
        # ``process_weights_after_loading`` may swap in an FP32 copy for the
        # trtllm-gen kernels, which nn.Module would reject on a parameter slot.
        object.__setattr__(self, "_fa3_sinks", sinks)
        self._fa3_window_size = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )

        self._use_custom_op = False
        self._layer_name = ""
        self.register_buffer(
            "_triton_kv_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        self._decode_cu_seqlens_q: torch.Tensor | None = None
        self._triton_seq_threshold_3d = max(
            1, _TRITON_MIN_LAUNCH_GRID_SIZE_2D // self.num_kv_heads,
        )
        self._triton_softmax_segm_output: torch.Tensor | None = None
        self._triton_softmax_segm_max: torch.Tensor | None = None
        self._triton_softmax_segm_expsum: torch.Tensor | None = None

        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            self.prefill_op = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, head_size,
            )
        else:
            self.store_kvcache = StoreKVCache()
            self.prefill_op = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, head_size,
                page_size=self._block_size,
            )
            self.decode_op._window_size = self._fa3_window_size

        self.tree_attn_op = TreeAttnPrefill(
            self.num_heads, self.num_kv_heads, head_size,
        )

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        if self._use_trtllm:
            self.decode_op._workspace = workspace
            self.prefill_op._workspace = workspace

    def process_weights_after_loading(self) -> None:
        """Prime the FP32 attention-sink copy the trtllm-gen kernels need.

        The two kernels this layer can dispatch to disagree on the sink dtype:
        ``trtllm_batch_{decode,context}_with_kv_cache`` reject anything but
        float32 (``attention_sinks must be a float tensor``) while the
        FlashAttention build vLLM bundles rejects anything but the model dtype
        (``learnable_sink must be bfloat16``) -- and a trtllm layer still falls
        back to FlashAttention for unpaged prefill.  So the layer keeps the
        checkpoint parameter and each trtllm op holds its own converted copy,
        materialized here (as vLLM does in
        ``FlashInferImpl.process_weights_after_loading``) rather than inside a
        forward or a graph capture.
        """
        if self.sinks is None or not self._use_trtllm:
            return
        self.prefill_op.prime_sinks(self.sinks)
        self.decode_op.prime_sinks(self.sinks)

    def forward_impl(self, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        """Core attention logic, callable from both eager and custom-op paths."""
        ctx = get_context()
        N = query.shape[0]

        q = query.view(N, self.num_heads, self.head_size)
        k = key.view(N, self.num_kv_heads, self.head_size)
        v = value.view(N, self.num_kv_heads, self.head_size)

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, self._group_slot_mapping(ctx))

        if getattr(ctx, "is_tree_verify", False):
            o = self.tree_attn_op(
                q, k_cache, v_cache,
                block_table_prefix=ctx.tree_block_table_prefix,
                cache_seqlens_prefix=ctx.tree_cache_seqlens_prefix,
                cu_seqlens_q_prefix=ctx.tree_cu_seqlens_q_prefix,
                max_seqlen_q_prefix=ctx.tree_max_seqlen_q_prefix,
                max_seqlen_k_prefix=ctx.tree_max_seqlen_k_prefix,
                page_table_expand=ctx.tree_page_table_expand,
                cache_seqlens_expand=ctx.tree_cache_seqlens_expand,
                cu_seqlens_q_expand=ctx.tree_cu_seqlens_q_expand,
                max_seqlen_k_expand=ctx.tree_num_verify_tokens,
                block_size=self._block_size,
                softmax_scale=self.scale,
            )
        elif ctx.is_mixed:
            if self._triton_only:
                can_use_triton = (
                    self._can_use_triton_unified(k_cache, self._group_prefill_block_tables(ctx))
                    and (ctx.num_decode_tokens == 0 or self._group_decode_block_tables(ctx) is not None)
                )
                if can_use_triton:
                    o = self._forward_mixed_triton(q, k_cache, v_cache, ctx)
                else:
                    o = self._forward_mixed_torch(q, k_cache, v_cache, ctx)
                return o.reshape(N, self.num_heads * self.head_size)
            o = self._forward_mixed(q, k_cache, v_cache, ctx)
        else:
            if self._triton_only:
                if self._can_use_triton_unified(k_cache, self._group_block_tables(ctx)):
                    o = self._forward_pure_triton(q, k_cache, v_cache, ctx)
                else:
                    o = self._forward_pure_torch(q, k, v, k_cache, v_cache, ctx)
                return o.reshape(N, self.num_heads * self.head_size)
            o = self._forward_pure(q, k, v, k_cache, v_cache, ctx)

        return o.reshape(N, self.num_heads * self.head_size)

    def _sliding_group_tensor(self, tensor):
        if tensor is None:
            return None
        gid = getattr(self, "_sliding_group_id", None)
        if gid is None:
            return tensor
        return tensor[gid].contiguous()

    def _group_slot_mapping(self, ctx):
        if self.sliding_window and ctx.sliding_slot_mapping is not None:
            return self._sliding_group_tensor(ctx.sliding_slot_mapping)
        return ctx.slot_mapping

    def _group_block_tables(self, ctx):
        if self.sliding_window and ctx.sliding_block_tables is not None:
            return self._sliding_group_tensor(ctx.sliding_block_tables)
        return ctx.block_tables

    def _group_prefill_block_tables(self, ctx):
        if self.sliding_window and ctx.sliding_prefill_block_tables is not None:
            return self._sliding_group_tensor(ctx.sliding_prefill_block_tables)
        return ctx.prefill_block_tables

    def _group_decode_block_tables(self, ctx):
        if self.sliding_window and ctx.sliding_decode_block_tables is not None:
            return self._sliding_group_tensor(ctx.sliding_decode_block_tables)
        return ctx.decode_block_tables

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            return torch.ops.fastkernels.unified_attention(
                query, key, value, self._layer_name,
            )
        return self.forward_impl(query, key, value)

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        if ctx.is_prefill:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k
            msq = ctx.max_seqlen_q
            msk = ctx.max_seqlen_k
            bt = self._group_block_tables(ctx)

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            if bt is not None:
                return self.prefill_op(
                    q, k_cache, v_cache,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=msq, max_seqlen_k=msk,
                    softmax_scale=self.scale, causal=True,
                    block_table=bt, **fa_extra,
                )
            return self.prefill_op(
                q, k, v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                **fa_extra,
            )

        cache_seqlens = ctx.context_lens
        bt = self._group_block_tables(ctx)
        max_ctx = ctx.max_context_len

        if self.attention_chunk_size is not None:
            cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                cache_seqlens, bt, self.attention_chunk_size, self._block_size,
            )

        return self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, block_table=bt,
            softmax_scale=self.scale, causal=True,
            max_seq_len=max_ctx, **fa_extra,
        )

    def _can_use_triton_unified(
        self,
        k_cache: torch.Tensor,
        block_tables: torch.Tensor | None,
    ) -> bool:
        return (
            # The kernel reads the cache as [num_blocks, block_size,
            # num_kv_heads, head_size]; an HND-allocated layer would have its
            # head and block dims transposed.
            self.kv_layout == "NHD"
            and self.attention_chunk_size is None
            and k_cache.numel() > 0
            and block_tables is not None
        )

    def _get_decode_cu_seqlens_q(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        needed = num_tokens + 1
        cached = self._decode_cu_seqlens_q
        if cached is None or cached.device != device or cached.numel() < needed:
            cached = torch.arange(needed, dtype=torch.int32, device=device)
            self._decode_cu_seqlens_q = cached
        return cached[:needed]

    def _triton_kv_descale(
        self,
        num_seqs: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        return self._triton_kv_scale.expand(num_seqs, num_kv_heads)

    def _get_triton_3d_buffers(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self._triton_softmax_segm_output
        if output is None or output.device != device:
            threshold = self._triton_seq_threshold_3d
            segments = _TRITON_NUM_PAR_SOFTMAX_SEGMENTS
            head_dim_padded = 1 << (self.head_size - 1).bit_length()
            self._triton_softmax_segm_output = torch.empty(
                (threshold, self.num_heads, segments, head_dim_padded),
                dtype=torch.float32,
                device=device,
            )
            self._triton_softmax_segm_max = torch.empty(
                (threshold, self.num_heads, segments),
                dtype=torch.float32,
                device=device,
            )
            self._triton_softmax_segm_expsum = torch.empty(
                (threshold, self.num_heads, segments),
                dtype=torch.float32,
                device=device,
            )
        return (
            self._triton_softmax_segm_output,
            self._triton_softmax_segm_max,
            self._triton_softmax_segm_expsum,
        )

    def _forward_paged_triton(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = int(seqused_k.shape[0])
        kv_descale = self._triton_kv_descale(num_seqs, k_cache.shape[2])
        triton_extra = {}
        if max_seqlen_q == 1 and num_seqs <= self._triton_seq_threshold_3d:
            segm_output, segm_max, segm_expsum = self._get_triton_3d_buffers(
                q.device,
            )
            triton_extra = {
                "seq_threshold_3D": self._triton_seq_threshold_3d,
                "num_par_softmax_segments": _TRITON_NUM_PAR_SOFTMAX_SEGMENTS,
                "softmax_segm_output": segm_output,
                "softmax_segm_max": segm_max,
                "softmax_segm_expsum": segm_expsum,
            }
        if _TRITON_UNIFIED_ACCEPTS_KV_QUANT:
            triton_extra["kv_quant_mode"] = _VllmKVQuantMode.NONE
        _triton_unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self._fa3_window_size,
            block_table=block_tables,
            softcap=0.0,
            q_descale=None,
            k_descale=kv_descale,
            v_descale=kv_descale,
            sinks=self._fa3_sinks,
            **triton_extra,
        )
        return out

    def _forward_pure_triton(self, q, k_cache, v_cache, ctx):
        if ctx.is_prefill:
            seqused_k = ctx.cu_seqlens_k[1:] - ctx.cu_seqlens_k[:-1]
            return self._forward_paged_triton(
                q,
                k_cache,
                v_cache,
                ctx.cu_seqlens_q,
                seqused_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                self._group_block_tables(ctx),
            )

        cu_q = self._get_decode_cu_seqlens_q(q.shape[0], q.device)
        return self._forward_paged_triton(
            q,
            k_cache,
            v_cache,
            cu_q,
            ctx.context_lens,
            1,
            ctx.max_context_len,
            self._group_block_tables(ctx),
        )

    def _repeat_kv_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _cache_seq(self, cache: torch.Tensor, block_table: torch.Tensor,
                   length: int) -> torch.Tensor:
        pages = (length + self._block_size - 1) // self._block_size
        block_ids = block_table[:pages].to(torch.long)
        return cache[block_ids].reshape(
            -1, self.num_kv_heads, self.head_size,
        )[:length]

    def _sdpa_one(self, q_seq: torch.Tensor, k_seq: torch.Tensor,
                  v_seq: torch.Tensor, key_offset: int = 0) -> torch.Tensor:
        q_len = q_seq.size(0)
        k_len = k_seq.size(0)
        k_seq = self._repeat_kv_for_heads(k_seq)
        v_seq = self._repeat_kv_for_heads(v_seq)
        q4 = q_seq.transpose(0, 1).unsqueeze(0)
        k4 = k_seq.transpose(0, 1).unsqueeze(0)
        v4 = v_seq.transpose(0, 1).unsqueeze(0)
        q_pos = key_offset + torch.arange(q_len, device=q_seq.device)
        k_pos = torch.arange(k_len, device=q_seq.device)
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask,
            dropout_p=0.0, scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1)

    def _prefill_torch_from_tensors(self, q, k, v, cu_q, cu_k) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = cu_q.numel() - 1
        for i in range(num_seqs):
            qs = int(cu_q[i].item())
            qe = int(cu_q[i + 1].item())
            ks = int(cu_k[i].item())
            ke = int(cu_k[i + 1].item())
            key_offset = (ke - ks) - (qe - qs)
            out[qs:qe] = self._sdpa_one(
                q[qs:qe], k[ks:ke], v[ks:ke], key_offset=key_offset,
            )
        return out

    def _prefill_torch_from_cache(self, q, k_cache, v_cache, cu_q, cu_k,
                                  block_tables) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = cu_q.numel() - 1
        for i in range(num_seqs):
            qs = int(cu_q[i].item())
            qe = int(cu_q[i + 1].item())
            k_len = int((cu_k[i + 1] - cu_k[i]).item())
            q_len = qe - qs
            k_seq = self._cache_seq(k_cache, block_tables[i], k_len)
            v_seq = self._cache_seq(v_cache, block_tables[i], k_len)
            out[qs:qe] = self._sdpa_one(
                q[qs:qe], k_seq, v_seq, key_offset=k_len - q_len,
            )
        return out

    def _decode_torch(self, q, k_cache, v_cache, cache_seqlens,
                      block_tables) -> torch.Tensor:
        out = torch.empty_like(q)
        for i in range(q.size(0)):
            k_len = int(cache_seqlens[i].item())
            k_seq = self._cache_seq(k_cache, block_tables[i], k_len)
            v_seq = self._cache_seq(v_cache, block_tables[i], k_len)
            out[i:i + 1] = self._sdpa_one(
                q[i:i + 1], k_seq, v_seq, key_offset=k_len - 1,
            )
        return out

    def _forward_pure_torch(self, q, k, v, k_cache, v_cache, ctx):
        bt = self._group_block_tables(ctx)
        if ctx.is_prefill:
            if bt is not None and k_cache.numel():
                return self._prefill_torch_from_cache(
                    q, k_cache, v_cache,
                    ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                    bt,
                )
            return self._prefill_torch_from_tensors(
                q, k, v, ctx.cu_seqlens_q, ctx.cu_seqlens_k,
            )
        return self._decode_torch(
            q, k_cache, v_cache, ctx.context_lens, bt,
        )

    def _forward_mixed(self, q, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
            msq = ctx.prefill_max_seqlen_q
            msk = ctx.prefill_max_seqlen_k
            bt = self._group_prefill_block_tables(ctx)

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            pq = q[:np_].contiguous() if self._use_trtllm else q[:np_]
            out[:np_] = self.prefill_op(
                pq, k_cache, v_cache,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                block_table=bt, **fa_extra,
            )

        if nd > 0:
            cache_seqlens = ctx.decode_context_lens
            bt = self._group_decode_block_tables(ctx)
            max_ctx = ctx.decode_max_context_len

            if self.attention_chunk_size is not None:
                cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                    cache_seqlens, bt,
                    self.attention_chunk_size, self._block_size,
                )

            out[np_:] = self.decode_op(
                q[np_:], k_cache, v_cache,
                cache_seqlens=cache_seqlens, block_table=bt,
                softmax_scale=self.scale, causal=True,
                max_seq_len=max_ctx, **fa_extra,
            )
        return out

    def _forward_mixed_triton(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            prefill_seqused_k = (
                ctx.prefill_cu_seqlens_k[1:] - ctx.prefill_cu_seqlens_k[:-1]
            )
            out[:np_] = self._forward_paged_triton(
                q[:np_],
                k_cache,
                v_cache,
                ctx.prefill_cu_seqlens_q,
                prefill_seqused_k,
                ctx.prefill_max_seqlen_q,
                ctx.prefill_max_seqlen_k,
                self._group_prefill_block_tables(ctx),
            )

        if nd > 0:
            cu_q = self._get_decode_cu_seqlens_q(nd, q.device)
            out[np_:] = self._forward_paged_triton(
                q[np_:],
                k_cache,
                v_cache,
                cu_q,
                ctx.decode_context_lens,
                1,
                ctx.decode_max_context_len,
                self._group_decode_block_tables(ctx),
            )
        return out

    def _forward_mixed_torch(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)
        if np_ > 0:
            out[:np_] = self._prefill_torch_from_cache(
                q[:np_],
                k_cache,
                v_cache,
                ctx.prefill_cu_seqlens_q,
                ctx.prefill_cu_seqlens_k,
                self._group_prefill_block_tables(ctx),
            )
        if nd > 0:
            out[np_:] = self._decode_torch(
                q[np_:],
                k_cache,
                v_cache,
                ctx.decode_context_lens,
                self._group_decode_block_tables(ctx),
            )
        return out

    ```

ModelNew must use the same `__init__` arguments and `forward` signature as `Model`.

The kernel will be evaluated on these captured input shapes:

### Attention

| count | args |
|------:|------|
| 86010 | `query:bfloat16[1000, 2048] key:bfloat16[1000, 128] value:bfloat16[1000, 128]` |
| 23876 | `query:bfloat16[1, 2048] key:bfloat16[1, 128] value:bfloat16[1, 128]` |
| 6876 | `query:bfloat16[60, 2048] key:bfloat16[60, 256] value:bfloat16[60, 256]` |
| 6144 | `query:bfloat16[60, 4096] key:bfloat16[60, 1024] value:bfloat16[60, 1024]` |
| 4982 | `query:bfloat16[16384, 2048] key:bfloat16[16384, 128] value:bfloat16[16384, 128]` |
| 4680 | `query:bfloat16[1, 2048] key:bfloat16[1, 256] value:bfloat16[1, 256]` |
| 4128 | `query:bfloat16[1, 4096] key:bfloat16[1, 1024] value:bfloat16[1, 1024]` |
| 3096 | `query:bfloat16[16384, 2048] key:bfloat16[16384, 256] value:bfloat16[16384, 256]` |

    
Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
