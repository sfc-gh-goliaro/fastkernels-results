"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear


@triton.jit
def _qk_rotary_kernel(
    qk,
    qkv,
    cos,
    sin,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_half: tl.constexpr,
    WRITE_CONTIGUOUS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    q_or_k = tl.program_id(2)

    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, triton.next_power_of_2(rotary_half))
    mask = (
        (rh[:, None, None] < num_heads)
        & (rm[None, :, None] < seq_len)
        & (rd[None, None, :] < rotary_half)
    )

    cs_mask = (rm[:, None] < seq_len) & (rd[None, :] < rotary_half)
    c = tl.load(cos + rm[:, None] * rotary_half + rd[None, :],
                mask=cs_mask, other=1.0).to(tl.float32)
    s = tl.load(sin + rm[:, None] * rotary_half + rd[None, :],
                mask=cs_mask, other=0.0).to(tl.float32)

    src = (
        qkv
        + rm[None, :, None] * (3 * num_heads * head_dim)
        + q_or_k * (num_heads * head_dim)
        + rh[:, None, None] * head_dim
        + rd[None, None, :]
    )
    x0 = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(src + rotary_half, mask=mask, other=0.0).to(tl.float32)

    if WRITE_CONTIGUOUS:
        dst = (
            qk
            + q_or_k * (seq_len * num_heads * head_dim)
            + rm[None, :, None] * (num_heads * head_dim)
            + rh[:, None, None] * head_dim
            + rd[None, None, :]
        )
    else:
        dst = src
    tl.store(dst, x0 * c - x1 * s, mask=mask)
    tl.store(dst + rotary_half, x0 * s + x1 * c, mask=mask)


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder (Qwen2-VL / Qwen2.5-VL / Qwen3-VL).

    All heads are attention heads (no GQA). Uses full (non-causal) attention.
    Supports TP: QKV is sharded, then gathered for RoPE, then re-sharded.
    """

    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        tp = _tp_size()
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.head_dim = projection_size // num_heads
        self.num_heads = num_heads // tp

        self.qkv = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, num_heads, bias=True,
        )
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)
        self.attn = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv(x)

        q_size = self.num_heads * self.head_dim
        # Dense Q/K strides repay the copy once attention has enough rows.
        write_contiguous = seq_len >= 8192
        qk = (
            torch.empty(
                (2, seq_len, self.num_heads, self.head_dim),
                device=qkv.device,
                dtype=qkv.dtype,
            )
            if write_contiguous
            else qkv
        )
        grid = (
            triton.cdiv(self.num_heads, 2),
            triton.cdiv(seq_len, 8),
            2,
        )
        _qk_rotary_kernel[grid](
            qk,
            qkv,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            seq_len,
            self.num_heads,
            self.head_dim,
            rotary_pos_emb_cos.shape[-1],
            WRITE_CONTIGUOUS=write_contiguous,
            BLOCK_M=8,
            BLOCK_H=2,
        )

        if write_contiguous:
            q, k = qk[0], qk[1]
        else:
            q, k = (
                qkv[..., i * q_size:(i + 1) * q_size]
                .view(seq_len, batch_size, self.num_heads, self.head_dim)
                .transpose(0, 1)
                .reshape(-1, self.num_heads, self.head_dim)
                for i in range(2)
            )
        v = (qkv[..., 2 * q_size:]
             .view(seq_len, batch_size, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .reshape(-1, self.num_heads, self.head_dim))

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            # Disable split-KV. With ``num_splits=0`` (auto) FA4's CuTeDSL
            # kernel runs ``num_splits_heuristic`` and, for the few m-blocks a
            # TP-sharded encoder produces (num_heads // tp, e.g. 16 // 4 = 4)
            # at moderate seqlens, picks ``num_splits > 1``. That enables the
            # ``is_split_kv`` path in ``flash_fwd_sm100.py``, whose
            # ``n_block_first`` is typed ``None`` on one branch and ``Int32``
            # on another -- a TYPE_UNSTABLE_JOIN CuTe compile error on
            # Blackwell (SM100).  Encoder self-attention is balanced
            # (q_len == k_len) so split-KV never helps here; forcing 1 is
            # numerically identical and sidesteps the kernel bug.  The paged
            # LLM prefill path (block_table/seqused_k) keeps auto-splitting,
            # where short-q-over-long-KV chunks do benefit.
            num_splits=1,
        )

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)
