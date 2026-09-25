"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from flash_attn.ops.triton.rotary import apply_rotary

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear


@triton.jit
def _rotary_qk_inplace(
    qkv,
    cos,
    sin,
    n_tokens,
    token_stride: tl.constexpr,
    q_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    rotary = tl.arange(0, BLOCK_D)[None, :]
    mask = (token < n_tokens) & (rotary < half_dim)
    c = tl.load(cos + token * half_dim + rotary, mask=mask).to(tl.float32)
    s = tl.load(sin + token * half_dim + rotary, mask=mask).to(tl.float32)

    for qk in tl.static_range(2):
        for head in tl.static_range(num_heads):
            base = (
                token * token_stride
                + qk * q_size
                + head * head_dim
                + rotary
            )
            x0 = tl.load(qkv + base, mask=mask).to(tl.float32)
            x1 = tl.load(qkv + base + half_dim, mask=mask).to(tl.float32)
            tl.store(qkv + base, x0 * c - x1 * s, mask=mask)
            tl.store(qkv + base + half_dim, x0 * s + x1 * c, mask=mask)


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
        if batch_size == 1:
            # FlashAttention accepts arbitrary token strides. Rotate Q and K in
            # their projection buffer instead of materializing a packed copy.
            token_stride = qkv.stride(0)
            qk = qkv.as_strided(
                (2, seq_len, self.num_heads, self.head_dim),
                (q_size, token_stride, self.head_dim, 1),
            )
            if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                half_dim = rotary_pos_emb_cos.shape[-1]
                _rotary_qk_inplace[(triton.cdiv(seq_len, 8),)](
                    qkv,
                    rotary_pos_emb_cos,
                    rotary_pos_emb_sin,
                    seq_len,
                    token_stride,
                    q_size,
                    self.num_heads,
                    self.head_dim,
                    half_dim,
                    BLOCK_T=8,
                    BLOCK_D=triton.next_power_of_2(half_dim),
                    num_warps=4,
                )
            q = qk[0]
            k = qk[1]
            v = qkv[:, 0, 2 * q_size:].view(
                seq_len, self.num_heads, self.head_dim,
            )
        else:
            qk = qkv[..., : 2 * q_size].view(
                seq_len, batch_size, 2, self.num_heads, self.head_dim,
            )
            qk = qk.permute(2, 1, 0, 3, 4).contiguous()
            if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                flat = qk.view(
                    2 * batch_size, seq_len, self.num_heads, self.head_dim,
                )
                apply_rotary(
                    flat, rotary_pos_emb_cos, rotary_pos_emb_sin, inplace=True,
                )
            q = qk[0].reshape(-1, self.num_heads, self.head_dim)
            k = qk[1].reshape(-1, self.num_heads, self.head_dim)
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
