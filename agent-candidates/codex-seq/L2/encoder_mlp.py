"""Feed-forward blocks for encoder models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.gelu import GELU, _gelu_kernel
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


@triton.jit
def _residual_layer_norm_kernel(
    x_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    # Materialize the fp16 residual sum exactly as the unfused torch.add does.
    x = (
        tl.load(x_ptr + offsets, mask=mask, other=0.0)
        + tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    ).to(tl.float16).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    output = centered * tl.rsqrt(variance + eps)
    output *= tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    output += tl.load(bias_ptr + cols, mask=mask).to(tl.float32)
    tl.store(output_ptr + offsets, output, mask=mask)


class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = F.linear(
            hidden_states, self.dense.weight, self.dense.bias,
        )
        if not projected.is_cuda or not projected.is_contiguous():
            return self.intermediate_act_fn(projected)

        n_elements = projected.numel()
        if 4_000_000 < n_elements <= 32_000_000:
            block_size, num_warps = 2048, 2
        else:
            block_size, num_warps = 1024, 2
        _gelu_kernel[(triton.cdiv(n_elements, block_size),)](
            projected,
            projected,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return projected


class EncoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        # promote_fp32=False: vLLM's bert.py / roberta.py use a plain
        # nn.LayerNorm here (see encoder_embeddings for the full rationale).
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                   promote_fp32=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        projected = F.linear(
            hidden_states, self.dense.weight, self.dense.bias,
        )
        if not projected.is_cuda or not projected.is_contiguous():
            return self.LayerNorm(projected + input_tensor)

        n_cols = self.LayerNorm.normalized_shape[0]
        n_rows = projected.numel() // n_cols
        output = torch.empty_like(projected)
        block_size = triton.next_power_of_2(n_cols)
        _residual_layer_norm_kernel[(n_rows,)](
            projected,
            input_tensor,
            output,
            self.LayerNorm.weight,
            self.LayerNorm.bias,
            n_cols=n_cols,
            eps=self.LayerNorm.eps,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return output
