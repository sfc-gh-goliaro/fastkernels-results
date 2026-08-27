import math
import torch
import torch.nn as nn

# Try to import Triton; if unavailable, we'll fallback gracefully.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# Simple, fast elementwise scaling kernel: y = x * scale
if _HAS_TRITON:
    @triton.jit
    def _scale_q_kernel(x_ptr, y_ptr,
                        total_elems,
                        scale,
                        BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_elems
        x = tl.load(x_ptr + offs, mask=mask)
        y = x * scale
        tl.store(y_ptr + offs, y, mask=mask)


def _choose_launch_config(total: int):
    """
    Heuristic launch configuration based on problem size.
    Returns (BLOCK: int, num_warps: int, num_stages: int).
    """
    # Larger blocks for large problems to reduce grid size and improve throughput.
    if total >= (1 << 24):      # ~16M elements and above
        BLOCK = 4096
        num_warps = 8
    elif total >= (1 << 22):    # ~4M elements and above
        BLOCK = 4096
        num_warps = 4
    elif total >= (1 << 20):    # ~1M elements and above
        BLOCK = 2048
        num_warps = 4
    else:
        BLOCK = 1024
        num_warps = 4
    num_stages = 2
    return BLOCK, num_warps, num_stages


def _triton_scale_q(q: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scale q using a Triton kernel when possible.
    Falls back to torch if Triton is not available or tensor is on CPU.

    Returns a tensor q_scaled with the same shape/dtype/device.
    """
    if (not _HAS_TRITON) or (not q.is_cuda):
        return q * scale

    # Avoid copying if already contiguous
    if q.is_contiguous():
        q_scaled = torch.empty_like(q)  # preserve layout
    else:
        # Make a contiguous copy to ensure coalesced accesses (simplest/fastest kernel).
        q = q.contiguous()
        q_scaled = torch.empty_like(q)

    total = q.numel()
    BLOCK, num_warps, num_stages = _choose_launch_config(total)
    grid = (triton.cdiv(total, BLOCK),)

    _scale_q_kernel[grid](
        q, q_scaled,
        total_elems=total,
        scale=scale,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q_scaled


def prime_trtllm_sinks(module: nn.Module, sinks: torch.Tensor | None) -> None:
    """Materialize the FP32 attention-sink copy the trtllm-gen kernels need."""
    if sinks is None:
        module._sinks_fp32 = None
        module._sinks_src = None
    elif sinks.dtype == torch.float32:
        module._sinks_fp32 = sinks
        module._sinks_src = sinks
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


# Keep the same imports
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from fastkernels.infra.fa_utils import FA_VERSION, flash_attn_varlen_func


class ModelNew(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # PyTorch scaling: 1/sqrt(head_dim)
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
        # If block_table is provided: use the paged TRTLLM path as before.
        if block_table is not None:
            # Minimal contiguity checks
            if not q.is_contiguous():
                q = q.contiguous()
            block_table = block_table.contiguous()
            seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            if not seq_lens.is_contiguous():
                seq_lens = seq_lens.contiguous()

            batch_size = seq_lens.shape[0]
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
                window_left=(
                    window_size[0] if isinstance(window_size, (list, tuple))
                    and len(window_size) >= 1 and window_size[0] >= 0
                    else -1
                ),
                sinks=trtllm_sinks(self, s_aux),
                kv_layout="HND",
            )

        # Dense (unpaged) fallback: use a real Triton kernel to scale q,
        # then call FlashAttention with scale=1.0.
        if softmax_scale is None:
            scale = self.sm_scale
        else:
            scale = float(softmax_scale)

        # Scale q using Triton if possible; fallback to torch otherwise.
        q_scaled = _triton_scale_q(q, scale)

        fa_extra = {}
        if s_aux is not None:
            fa_extra["s_aux"] = s_aux
        if window_size is not None:
            fa_extra["window_size"] = window_size

        return flash_attn_varlen_func(
            q_scaled, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=1.0,      # already scaled q
            causal=causal,
            fa_version=FA_VERSION,
            num_splits=1,
            **fa_extra,
        )

TRTLLMPrefill = ModelNew
