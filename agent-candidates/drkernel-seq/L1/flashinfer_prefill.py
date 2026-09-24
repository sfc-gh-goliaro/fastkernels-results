import math
import torch
import torch.nn as nn

# Original helpers kept for API compatibility (not used in dense path)
def prime_trtllm_sinks(module: nn.Module, sinks: torch.Tensor | None) -> None:
    if sinks is None:
        module._sinks_fp32 = None
    elif sinks.dtype == torch.float32:
        module._sinks_fp32 = sinks
    else:
        module._sinks_fp32 = sinks.detach().to(torch.float32)
    module._sinks_src = sinks

def trtllm_sinks(module: nn.Module, s_aux: torch.Tensor | None):
    if s_aux is None or s_aux.dtype == torch.float32:
        return s_aux
    if module._sinks_fp32 is None or module._sinks_src is not s_aux:
        prime_trtllm_sinks(module, s_aux)
    return module._sinks_fp32

# Paged path remains using FlashInfer
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

import triton
import triton.language as tl


@triton.jit
def _varlen_attn_fused_kernel(
    Q, K, V,  # pointers
    Out,  # pointer to float32 [MQ, H, D]
    cu_q: tl.pointer_type,  # int32*
    cu_k: tl.pointer_type,  # int32*
    # strides
    stride_q_m, stride_q_h, stride_q_d,
    stride_k_m, stride_k_h, stride_k_l, stride_k_d,
    stride_v_m, stride_v_h, stride_v_l, stride_v_d,
    stride_o_m, stride_o_h, stride_o_d,
    # meta
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_mh = tl.program_id(0)  # over MQ*H
    pid_kt = tl.program_id(1)  # over K-tiles (will loop)

    # derive m, h from pid_mh
    h = pid_mh % H
    m = pid_mh // H

    # Q info
    q_start = tl.load(cu_q + m)          # int32
    L_q = tl.load(cu_q + m + 1) - q_start

    # K info
    k_start = tl.load(cu_k + m)
    L_k = tl.load(cu_k + m + 1) - k_start

    # D indices
    d = tl.arange(0, BLOCK_D)

    # Load Q tile [BD] once; convert to f32
    q_ptr = Q + m * stride_q_m + h * stride_q_h + d * stride_q_d
    q = tl.load(q_ptr, mask=d < D, other=0.0).to(tl.float32)  # [BD]

    # Prepare output accumulator o [BD]
    o = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Running max m and sum s for softmax normalization
    m_val = tl.full((), -float("inf"), dtype=tl.float32)
    s_val = tl.zeros((), dtype=tl.float32)

    # Loop over K tiles
    num_tiles = (L_k + BLOCK_K - 1) // BLOCK_K
    for t in range(0, num_tiles):
        k_off = t * BLOCK_K
        k_idx = k_off + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = k_idx < L_k

        # Causal mask: invalidate k >= L_q
        causal = k_idx < L_q
        valid = mask_k & causal

        # Load K tile [BK, BD]
        abs_k = k_start + k_idx
        k_ptr = (
            K
            + m * stride_k_m
            + h * stride_k_h
            + abs_k[:, None] * stride_k_l
            + d[None, :] * stride_k_d
        )
        k_tile = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(tl.float32)  # [BK, BD]

        # Compute s_tile = sum_d Q[d] * K[k,d] -> [BK]
        prod = k_tile * q[None, :, None]  # [BK, BD, 1] -> view as [BK, BD]
        s_tile = tl.sum(prod, axis=1)     # [BK]

        # Apply masks: out-of-range -> -inf, causal invalid -> -inf
        neg_inf = -float("inf")
        s_tile = tl.where(valid, s_tile, neg_inf)

        # Online softmax update
        tile_max = tl.max(s_tile, axis=0)  # scalar
        new_m = tl.maximum(m_val, tile_max)
        # Scale previous s to new_m
        scale_prev = tl.exp(m_val - new_m)
        # Contributions from this tile, scaled to new_m
        contrib = tl.exp(s_tile - new_m)  # [BK]
        # Sum contrib
        sum_contrib = tl.sum(contrib, axis=0)  # scalar
        # Update s
        s_val = s_val * scale_prev + sum_contrib
        m_val = new_m

        # Accumulate output: o += sum( exp(s_tile - m_val) * V, over d )
        # Load V tile [BK, BD]
        v_ptr = (
            V
            + m * stride_v_m
            + h * stride_v_h
            + abs_k[:, None] * stride_v_l
            + d[None, :] * stride_v_d
        )
        v_tile = tl.load(v_ptr, mask=valid[:, None], other=0.0).to(tl.float32)  # [BK, BD]

        # weight = exp(s_tile - m_val) [BK]
        wt = tl.exp(s_tile - m_val)
        # Multiply and reduce over K: [BK, BD] -> sum over BK -> [BD]
        weighted_v = v_tile * wt[:, None]
        o += tl.sum(weighted_v, axis=0)

    # Final normalize: o /= s
    o = o / s_val

    # Store Out[m, h, d]
    out_ptr = Out + m * stride_o_m + h * stride_o_h + d * stride_o_d
    tl.store(out_ptr, o, mask=d < D)


class ModelNew(nn.Module):
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

    def _triton_dense_varlen_attn_fused(self, q, k, v, cu_q, cu_k, softmax_scale=None, causal=True):
        """
        q: [MQ, H, D], k: [MQ, H, L, D], v: [MQ, H, L, D]
        cu_q: [MQ+1] int32, cu_k: [MQ+1] int32  (we assume same batch for q&k)
        Returns: out [MQ, H, D] float32
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernel requires CUDA tensors"
        device = q.device
        MQ = cu_q.shape[0] - 1
        H = q.shape[1]
        D = q.shape[2]
        assert k.shape[0] == MQ and v.shape[0] == MQ, "q/k/v batch mismatch"
        assert k.shape[1] == H and v.shape[1] == H, "Head mismatch"
        assert k.shape[3] == D and v.shape[3] == D, "Dim mismatch"
        assert q.dtype in (torch.bfloat16, torch.float16) and k.dtype == q.dtype and v.dtype == q.dtype, "Use BF16/FP16"

        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out = torch.empty((MQ, H, D), device=device, dtype=torch.float32)

        # Strides
        stride_q_m, stride_q_h, stride_q_d = q.stride(0), q.stride(1), q.stride(2)
        stride_k_m, stride_k_h, stride_k_l, stride_k_d = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        stride_v_m, stride_v_h, stride_v_l, stride_v_d = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
        stride_o_m, stride_o_h, stride_o_d = out.stride(0), out.stride(1), out.stride(2)

        # Tiling
        BLOCK_D = 128 if D >= 128 else (64 if D >= 64 else 32)
        BLOCK_K = 256 if D >= 128 else 128

        # Grid: over rows (MQ*H) and K-tiles (max over m); we'll loop t inside kernel, but grid dim1 can be max tiles
        # Compute max L over m
        L_per_m = cu_k[1:] - cu_k[:-1]
        max_L = int(torch.max(L_per_m).item()) if MQ > 1 else int(L_per_m[0].item())
        num_max_tiles = (max_L + BLOCK_K - 1) // BLOCK_K
        grid = (MQ * H, num_max_tiles)

        _varlen_attn_fused_kernel[grid](
            q, k, v,
            out,
            cu_q, cu_k,
            stride_q_m, stride_q_h, stride_q_d,
            stride_k_m, stride_k_h, stride_k_l, stride_k_d,
            stride_v_m, stride_v_h, stride_v_l, stride_v_d,
            stride_o_m, stride_o_h, stride_o_d,
            H=H, D=D,
            BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Apply softmax_scale if provided (FlashAttention applies scale after softmax;
        # here we've computed o = P @ V. To match, multiply by scale.)
        if softmax_scale is not None:
            out *= softmax_scale
        else:
            out *= self.sm_scale

        return out

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, s_aux=None,
                window_size=None, **kwargs):
        # Device check
        if not q.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors; got CPU.")

        # Ensure expected dtypes
        if q.dtype not in (torch.bfloat16, torch.float16):
            q = q.to(torch.bfloat16)
        if k.dtype not in (torch.bfloat16, torch.float16):
            k = k.to(torch.bfloat16)
        if v.dtype not in (torch.bfloat16, torch.float16):
            v = v.to(torch.bfloat16)

        if block_table is not None:
            # Paged TRTLLM path: keep as-is for performance and complexity
            seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            batch_size = seq_lens.shape[0]
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
                window_left=(
                    window_size[0] if isinstance(window_size, (list, tuple))
                    and window_size[0] >= 0 else -1
                ),
                sinks=trtllm_sinks(self, s_aux),
                kv_layout="HND",
            )
        else:
            # Dense varlen path: use our fused Triton kernel
            return self._triton_dense_varlen_attn_fused(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                softmax_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
                causal=True,
            )

TRTLLMPrefill = ModelNew
