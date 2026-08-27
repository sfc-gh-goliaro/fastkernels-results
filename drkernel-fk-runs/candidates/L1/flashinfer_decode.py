import torch
import torch.nn as nn
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


class ModelNew(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5

        if workspace is None:
            workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self._workspace = workspace

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, s_aux=None, window_size=None, **kwargs):
        # Ensure minimal layout assumptions for the kernel
        q = q if q.is_contiguous() else q.contiguous()
        block_table = block_table if block_table.is_contiguous() else block_table.contiguous()
        if cache_seqlens is not None:
            cache_seqlens = cache_seqlens if cache_seqlens.is_contiguous() else cache_seqlens.contiguous()

        # Softmax scale: default 1/sqrt(head_dim); pass as Python float
        bmm1_scale = float(softmax_scale) if softmax_scale is not None else float(self.sm_scale)

        # Determine max_seq_len if not provided
        if max_seq_len is None:
            if cache_seqlens is None:
                raise ValueError("cache_seqlens must be provided if max_seq_len is None")
            # .item() sync is fine and tiny; using Python int is fastest
            max_seq_len = int(cache_seqlens.max().item())

        # Sliding window left: interpret window_size
        window_left = -1
        if window_size is not None:
            if isinstance(window_size, (list, tuple)):
                if len(window_size) != 2:
                    window_left = -1
                else:
                    left, right = window_size
                    window_left = int(left) if left >= 0 else -1
            else:
                # scalar: treat as right; no left constraint
                window_left = -1

        # Sinks: convert to float32 if needed (kernel requires float32); keep it simple and fast
        sinks = None
        if s_aux is not None:
            if s_aux.dtype != torch.float32:
                # One-time or per-call convert; .to runs on device and is fine
                sinks = s_aux.to(torch.float32)
            else:
                sinks = s_aux

        # Launch the TRTLLM decode kernel
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,   # Python float -- fast, no tensor creation
            bmm2_scale=1.0,          # Python float
            window_left=window_left,
            sinks=sinks,
            kv_layout="HND",
        )

TRTLLMDecode = ModelNew
