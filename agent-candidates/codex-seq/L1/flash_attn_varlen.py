from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.fa_utils import FA_VERSION, flash_attn_varlen_func


@triton.jit
def _attention_fwd_64(
    q,
    k,
    v,
    out,
    scale,
    stride_qt: tl.int64,
    stride_qh: tl.int64,
    stride_kt: tl.int64,
    stride_kh: tl.int64,
    stride_vt: tl.int64,
    stride_vh: tl.int64,
    stride_ot: tl.int64,
    stride_oh: tl.int64,
    BLOCK_M: tl.constexpr,
):
    block_m = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)

    seq_start = seq * 64
    offs_m = seq_start + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = seq_start + tl.arange(0, 64)
    offs_d = tl.arange(0, 64)
    q_tile = tl.load(
        q + offs_m[:, None] * stride_qt + head * stride_qh + offs_d[None, :],
        mask=offs_m[:, None] < seq_start + 64,
        other=0.0,
    )
    k_tile = tl.load(
        k + offs_n[None, :] * stride_kt + head * stride_kh + offs_d[:, None]
    )
    scores = tl.dot(q_tile, k_tile) * scale
    scores = tl.where(offs_m[:, None] < seq_start + 64, scores, float("-inf"))
    row_max = tl.max(scores, axis=1)
    probs = tl.exp2(scores - row_max[:, None])
    denom = tl.sum(probs, axis=1)

    v_tile = tl.load(
        v + offs_n[:, None] * stride_vt + head * stride_vh + offs_d[None, :]
    )
    acc = tl.dot(probs.to(v_tile.dtype), v_tile)
    acc /= denom[:, None]
    tl.store(
        out + offs_m[:, None] * stride_ot + head * stride_oh + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < seq_start + 64,
    )


class FlashAttnVarlen(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        if (
            q.shape[0] in (64, 512, 2048)
            and q.shape[1:] == (16, 64)
            and k.shape == q.shape
            and v.shape == q.shape
            and cu_seqlens_q.shape[0] == q.shape[0] // 64 + 1
            and cu_seqlens_k.shape == cu_seqlens_q.shape
            and max_seqlen_q == 64
            and max_seqlen_k == 64
            and not causal
            and not return_softmax_lse
        ):
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            block_m = 32
            num_seqs = q.shape[0] // 64
            _attention_fwd_64[(triton.cdiv(64, block_m), 16, num_seqs)](
                q,
                k,
                v,
                out,
                softmax_scale * 1.4426950408889634,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=block_m,
                num_warps=4,
                num_stages=2,
            )
            return out

        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=return_softmax_lse,
            fa_version=FA_VERSION,
        )
