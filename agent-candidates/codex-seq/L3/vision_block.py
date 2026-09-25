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
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.quickgelu import QuickGELU
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import VisionMLP, _gelu_inplace_kernel


@triton.jit
def _residual_layer_norm_kernel(
    x_ptr,
    residual_ptr,
    norm_ptr,
    weight_ptr,
    bias_ptr,
    output_bias_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    FUSE_OUTPUT_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    update = tl.load(residual_ptr + row * n_cols + cols, mask=mask, other=0.0)
    residual = (x + update).to(tl.bfloat16)
    if FUSE_OUTPUT_BIAS:
        output_bias = tl.load(output_bias_ptr + cols, mask=mask)
        tl.store(
            residual_ptr + row * n_cols + cols,
            residual + output_bias,
            mask=mask,
        )
    else:
        tl.store(residual_ptr + row * n_cols + cols, residual, mask=mask)

    residual_f32 = residual.to(tl.float32)
    mean = tl.sum(residual_f32, axis=0) / n_cols
    centered = tl.where(mask, residual_f32 - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    norm = centered * tl.rsqrt(variance + eps)
    norm *= tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    norm += tl.load(bias_ptr + cols, mask=mask).to(tl.float32)
    tl.store(norm_ptr + row * n_cols + cols, norm, mask=mask)


@triton.jit
def _residual_add_inplace_kernel(
    out_ptr,
    residual_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    out = tl.load(out_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, out + residual, mask=mask)


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

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.norm1(x), cu_seqlens,
            rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and attn_out.is_contiguous()
        ):
            n_cols = x.shape[-1]
            n_rows = x.numel() // n_cols
            normed = torch.empty_like(x)
            fused_fc2 = (
                self.mlp.act_fn.__class__.__name__ == "GELU"
                and self.mlp.fc1.bias is not None
                and self.mlp.fc2.bias is not None
                and self.mlp.fc2.tp_size == 1
            )
            _residual_layer_norm_kernel[(n_rows,)](
                x,
                attn_out,
                normed,
                self.norm2.weight,
                self.norm2.bias,
                self.mlp.fc2.bias,
                n_cols=n_cols,
                eps=self.norm2.eps,
                FUSE_OUTPUT_BIAS=fused_fc2,
                BLOCK_SIZE=triton.next_power_of_2(n_cols),
                num_warps=4,
            )
            if fused_fc2:
                normed_2d = normed.reshape(-1, n_cols)
                hidden = torch.addmm(
                    self.mlp.fc1.bias,
                    normed_2d,
                    self.mlp.fc1.weight.t(),
                )
                _gelu_inplace_kernel[
                    (triton.cdiv(hidden.numel(), 2048),)
                ](
                    hidden,
                    hidden.numel(),
                    BLOCK_SIZE=2048,
                    num_warps=2,
                )
                out = attn_out.reshape(-1, n_cols)
                torch.addmm(
                    out,
                    hidden,
                    self.mlp.fc2.weight.t(),
                    out=out,
                )
                return attn_out

            mlp_out = self.mlp(normed)
            _residual_add_inplace_kernel[
                (triton.cdiv(mlp_out.numel(), 65536),)
            ](
                mlp_out,
                attn_out,
                mlp_out.numel(),
                BLOCK_SIZE=65536,
                num_warps=8,
            )
            return mlp_out

        x = x + attn_out
        return x + self.mlp(self.norm2(x))
