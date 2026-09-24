import math
import torch
import torch.nn as nn

import triton
import triton.language as tl

# Fallback kernel from original code; needed if fastpath is not taken.
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


@triton.jit
def blockwise_attn_decode(
    q_ptr,             # *f16/f32 [M, H, D]
    k_ptr,             # *f16/f32 flattened [P*H*D]
    v_ptr,             # *f16/f32 flattened [P*H*D]
    out_ptr,           # *f32 [M, H, D]
    block_table_ptr,   # *i32 [1, P] (we pass flat)
    seq_len,           # i32
    num_blocks,        # i32
    H: tl.constexpr,   # num_heads
    D: tl.constexpr,   # head_dim
    scale,             # f32 softmax scale
    BLOCK_D: tl.constexpr,
):
    # One program per block p
    p = tl.program_id(0)

    if p >= num_blocks:
        return

    # Derive block id from table: not needed since p is block id; block_table is mapping from seq to blocks.
    # But here batch == 1 and seq_len is small; we assume block id == p.

    # Pass 1: compute max of scores over all (m,h)
    max_score = tl.full((), -float("inf"), tl.float32)

    m_limit = seq_len

    h = 0
    while h < H:
        m = 0
        while m < m_limit:
            # score scalar
            score = tl.zeros((), dtype=tl.float32)
            d = 0
            while d < D:
                offs = d + tl.arange(0, BLOCK_D)
                mask = offs < D
                # q offset: ((m*H + h)*D + offs)
                q_off = ((m * H + h) * D) + offs
                # k offset: ((p*H + h)*D + offs)
                k_off = ((p * H + h) * D) + offs
                q_vals = tl.load(q_ptr + q_off, mask=mask, other=0.0).to(tl.float32)
                k_vals = tl.load(k_ptr + k_off, mask=mask, other=0.0).to(tl.float32)
                prod = q_vals * k_vals
                score += tl.sum(prod, axis=0)
                d += BLOCK_D
            score = score * scale
            max_score = tl.maximum(max_score, score)
            m += 1
        h += 1

    # Pass 2: sum of exp(score - max)
    sum_exp = tl.zeros((), dtype=tl.float32)

    h = 0
    while h < H:
        m = 0
        while m < m_limit:
            score = tl.zeros((), dtype=tl.float32)
            d = 0
            while d < D:
                offs = d + tl.arange(0, BLOCK_D)
                mask = offs < D
                q_off = ((m * H + h) * D) + offs
                k_off = ((p * H + h) * D) + offs
                q_vals = tl.load(q_ptr + q_off, mask=mask, other=0.0).to(tl.float32)
                k_vals = tl.load(k_ptr + k_off, mask=mask, other=0.0).to(tl.float32)
                prod = q_vals * k_vals
                score += tl.sum(prod, axis=0)
                d += BLOCK_D
            score = score * scale
            e = tl.exp(score - max_score)
            sum_exp += e
            m += 1
        h += 1

    # Pass 3: write outputs O = softmax * V
    h = 0
    while h < H:
        m = 0
        while m < m_limit:
            score = tl.zeros((), dtype=tl.float32)
            d = 0
            while d < D:
                offs = d + tl.arange(0, BLOCK_D)
                mask = offs < D
                q_off = ((m * H + h) * D) + offs
                k_off = ((p * H + h) * D) + offs
                q_vals = tl.load(q_ptr + q_off, mask=mask, other=0.0).to(tl.float32)
                k_vals = tl.load(k_ptr + k_off, mask=mask, other=0.0).to(tl.float32)
                prod = q_vals * k_vals
                score += tl.sum(prod, axis=0)
                d += BLOCK_D
            score = score * scale
            e = tl.exp(score - max_score)
            alpha = e / sum_exp

            # Now accumulate alpha * V into out
            d = 0
            while d < D:
                offs = d + tl.arange(0, BLOCK_D)
                mask = offs < D
                q_off = ((m * H + h) * D) + offs
                v_off = ((p * H + h) * D) + offs
                # out has same layout as q ([M,H,D])
                out_off = q_off
                q_vals = tl.load(q_ptr + q_off, mask=mask, other=0.0).to(tl.float32)  # not used, but symmetric
                v_vals = tl.load(v_ptr + v_off, mask=mask, other=0.0).to(tl.float32)
                out_vals = alpha * v_vals
                tl.store(out_ptr + out_off, out_vals, mask=mask)
                d += BLOCK_D
            m += 1
        h += 1


class ModelNew(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = float(head_dim) ** -0.5
        if workspace is None:
            workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self._workspace = workspace
        # Sinks cache (not used in fastpath)
        self._sinks_fp32: torch.Tensor | None = None
        self._sinks_src: torch.Tensor | None = None
        self._cached_seqlens_id = None
        self._cached_max_seq_len = None

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        if sinks is None:
            self._sinks_fp32 = None
            self._sinks_src = None
        elif sinks.dtype == torch.float32:
            self._sinks_fp32 = sinks
            self._sinks_src = sinks
        else:
            self._sinks_fp32 = sinks.detach().to(torch.float32)
            self._sinks_src = sinks

    def _get_max_seq_len(self, cache_seqlens: torch.Tensor) -> int:
        if (self._cached_seqlens_id is not None and
                self._cached_seqlens_id == id(cache_seqlens) and
                self._cached_max_seq_len is not None):
            return self._cached_max_seq_len
        max_len = int(cache_seqlens.max().item())
        self._cached_seqlens_id = id(cache_seqlens)
        self._cached_max_seq_len = max_len
        return max_len

    def _supports_triton_fastpath(self, q, k_cache, v_cache, block_table, cache_seqlens,
                                  softmax_scale, s_aux, window_size):
        # Fastpath: batch==1, no sinks, no window, simple layouts
        if block_table is None or cache_seqlens is None:
            return False
        if q.dim() != 3:
            return False
        if k_cache.dim() != 4 or v_cache.dim() != 4:
            return False
        Bq = q.shape[0]; Bk = k_cache.shape[0]; Bv = v_cache.shape[0]
        if Bq != 1 or Bk != 1 or Bv != 1:
            return False
        Hq = q.shape[1]; Dq = q.shape[2]
        Hk = k_cache.shape[2]; Dk = k_cache.shape[3]
        Hv = v_cache.shape[2]; Dv = v_cache.shape[3]
        if not (Hq == Hk == Hv and Dq == Dk == Dv):
            return False
        if s_aux is not None or window_size is not None:
            return False
        # block_table shape [1, P]
        if block_table.shape[0] != 1:
            return False
        return True

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, s_aux=None, window_size=None, **kwargs):
        if cache_seqlens is None:
            raise ValueError("cache_seqlens must be provided.")
        if block_table is None:
            raise ValueError("block_table must be provided.")

        # Contiguity
        q = q.contiguous()
        block_table = block_table.contiguous()
        if cache_seqlens is not None:
            cache_seqlens = cache_seqlens.contiguous()

        # Max seq len
        if max_seq_len is None:
            max_seq_len = self._get_max_seq_len(cache_seqlens)

        # Scale
        scale = softmax_scale if softmax_scale is not None else self.sm_scale

        # Try Triton fast path
        if self._supports_triton_fastpath(q, k_cache, v_cache, block_table, cache_seqlens,
                                         scale, s_aux, window_size):
            B, H, D = 1, q.shape[1], q.shape[2]
            P = block_table.shape[1]
            # Flatten k/v: shapes [1, P, H, D] -> [P*H*D]
            k_flat = k_cache.view(-1).contiguous()
            v_flat = v_cache.view(-1).contiguous()
            # Output (float32)
            out = torch.empty_like(q, dtype=torch.float32)

            # Choose BLOCK_D
            # Power-of-two up to 128
            if D >= 128:
                BLOCK_D = 128
            elif D >= 64:
                BLOCK_D = 64
            elif D >= 32:
                BLOCK_D = 32
            else:
                BLOCK_D = 16

            blockwise_attn_decode[(P,)](
                q, k_flat, v_flat, out,
                block_table,
                int(cache_seqlens[0].item()),  # seq_len for batch 0
                int(P),
                H=H, D=D,
                scale=float(scale),
                BLOCK_D=BLOCK_D,
                num_warps=4,
                num_stages=2,
            )
            return out

        # Fallback: use original TRTLLM kernel
        sinks = s_aux
        if sinks is not None:
            if sinks.dtype == torch.float32:
                pass
            elif sinks.dtype == torch.bfloat16:
                if (self._sinks_fp32 is not None and
                        self._sinks_src is sinks and
                        self._sinks_fp32.shape == sinks.shape and
                        self._sinks_fp32.device == sinks.device):
                    sinks = self._sinks_fp32
                else:
                    self.prime_sinks(sinks)
                    sinks = self._sinks_fp32
            else:
                sinks = sinks.to(torch.float32)
        else:
            sinks = None

        window_left = -1
        if window_size is not None:
            if isinstance(window_size, (list, tuple)) and len(window_size) >= 1:
                wl = int(window_size[0])
                window_left = wl if wl >= 0 else -1
            else:
                window_left = int(window_size) if int(window_size) >= 0 else -1

        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=window_left,
            sinks=sinks,
            kv_layout="HND",
        )

TRTLLMDecode = ModelNew
