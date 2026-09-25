"""Vision transformer block for Qwen VL models.

Unified across Qwen2-VL and Qwen3-VL:
  - act_fn: Qwen2 uses QuickGELU (default), Qwen3 uses SiLU.
  - norm_eps: configurable LayerNorm epsilon.

Uses LayerNorm (not RMSNorm) with pre-norm residual connections,
encoder-only attention, and vision MLP.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.quickgelu import QuickGELU
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import VisionMLP


@triton.jit
def _rotary_qk_kernel(
    qkv_ptr,
    cos_ptr,
    sin_ptr,
    q_size: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_half: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    num_heads = q_size // head_dim
    pairs_per_qk = num_heads * rotary_half
    mask = offsets < 2 * pairs_per_qk
    qk = offsets // pairs_per_qk
    within_qk = offsets % pairs_per_qk
    head = within_qk // rotary_half
    dim = within_qk % rotary_half
    row_base = row * 3 * q_size
    pair_offset = qk * q_size + head * head_dim + dim
    x0 = tl.load(qkv_ptr + row_base + pair_offset, mask=mask, other=0.0)
    x1 = tl.load(
        qkv_ptr + row_base + pair_offset + rotary_half,
        mask=mask,
        other=0.0,
    )
    trig_offset = row * rotary_half + dim
    cos = tl.load(cos_ptr + trig_offset, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + trig_offset, mask=mask, other=0.0).to(tl.float32)
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    tl.store(
        qkv_ptr + row_base + pair_offset,
        x0 * cos - x1 * sin,
        mask=mask,
    )
    tl.store(
        qkv_ptr + row_base + pair_offset + rotary_half,
        x0 * sin + x1 * cos,
        mask=mask,
    )


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    normed_ptr,
    n_elements: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_elements
    base = row * n_elements + offsets

    values = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / n_elements
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_elements
    normalized = centered * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    tl.store(normed_ptr + base, normalized * weight + bias, mask=mask)


@triton.jit
def _add_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    normed_ptr,
    n_elements: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_elements
    base = row * n_elements + offsets

    # Match eager's bf16 residual boundary before accumulating LN statistics.
    residual = (
        tl.load(x_ptr + base, mask=mask, other=0.0)
        + tl.load(y_ptr + base, mask=mask, other=0.0)
    ).to(tl.bfloat16)
    tl.store(residual_ptr + base, residual, mask=mask)

    values = residual.to(tl.float32)
    mean = tl.sum(values, axis=0) / n_elements
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_elements
    normalized = centered * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        normed_ptr + base,
        normalized * weight + bias,
        mask=mask,
    )


def _add_layer_norm(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = torch.empty_like(x)
    normed = torch.empty_like(x)
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    _add_layer_norm_kernel[(rows,)](
        x, y, weight, bias, residual, normed,
        n_elements=width,
        eps=eps,
        block_size=triton.next_power_of_2(width),
        num_warps=4,
    )
    return residual, normed


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    normed = torch.empty_like(x)
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    _layer_norm_kernel[(rows,)](
        x, weight, bias, normed,
        n_elements=width,
        eps=eps,
        block_size=triton.next_power_of_2(width),
        num_warps=4,
    )
    return normed


def _rotary_qk(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_size: int,
    head_dim: int,
) -> None:
    rotary_half = cos.shape[-1]
    _rotary_qk_kernel[(qkv.shape[0],)](
        qkv, cos, sin,
        q_size=q_size,
        head_dim=head_dim,
        rotary_half=rotary_half,
        block_size=triton.next_power_of_2(
            2 * (q_size // head_dim) * rotary_half,
        ),
        num_warps=4,
    )


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        # promote_fp32=False to match vLLM, whose vision blocks use a plain
        # ``nn.LayerNorm`` on the bf16 activations (qwen3_vl.py:
        # ``norm_layer = partial(nn.LayerNorm, eps=1e-6)``). Our default promotes
        # to fp32 for the reduction, which exists for the DeepSeek-V3.2 indexer's
        # k_norm and is wrong to apply here: it costs an ``x.float()`` and a
        # ``.to(bf16)`` -- two full-tensor copies -- on every norm, and a Qwen3-VL
        # encoder pass runs 54 of them. Profiled against vLLM's encoder, that was
        # 11.7ms/call of aten::copy_ in ``unrolled_elementwise<direct_copy>``
        # that vLLM never emits. PyTorch's bf16 layer_norm already accumulates in
        # fp32 internally, so the reduction precision is unchanged.
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)
        self._approximate_gelu = (
            act_fn.__class__.__name__ == "GELU"
            and getattr(act_fn, "approximate", None) == "none"
        )

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        normed = _layer_norm(
            x, self.norm1.weight, self.norm1.bias, self.norm1.eps,
        )
        qkv = self.attn.qkv(normed)
        q_size = self.attn.num_heads * self.attn.head_dim
        qk = qkv[..., : 2 * q_size].view(
            seq_len, batch_size, 2, self.attn.num_heads, self.attn.head_dim,
        ).permute(2, 1, 0, 3, 4)

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            _rotary_qk(
                qkv,
                rotary_pos_emb_cos, rotary_pos_emb_sin,
                q_size=q_size,
                head_dim=self.attn.head_dim,
            )

        q = qk[0].reshape(-1, self.attn.num_heads, self.attn.head_dim)
        k = qk[1].reshape(-1, self.attn.num_heads, self.attn.head_dim)
        v = qkv[..., 2 * q_size:].view(
            -1, self.attn.num_heads, self.attn.head_dim,
        )
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn.attn(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            softmax_scale=self.attn.head_dim ** -0.5,
            causal=False,
            num_splits=1,
        )
        out = self.attn.proj(out.view(seq_len, batch_size, -1))
        x, normed = _add_layer_norm(
            x, out, self.norm2.weight, self.norm2.bias, self.norm2.eps,
        )
        if self._approximate_gelu:
            hidden = F.gelu(self.mlp.fc1(normed), approximate="tanh")
            x = x + self.mlp.fc2(hidden)
        else:
            x = x + self.mlp(normed)
        return x
