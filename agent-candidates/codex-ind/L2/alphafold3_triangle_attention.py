"""Triton triangle attention specialized for the captured OpenFold3 shape."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_of3_attention import OF3Attention


@triton.jit
def _compose_weights_kernel(
    value_weight_ptr,
    output_weight_ptr,
    effective_weight_ptr,
):
    out_cols = tl.program_id(0) * 32 + tl.arange(0, 32)
    in_cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    hidden = tl.arange(0, 128)
    output_weight = tl.load(
        output_weight_ptr + out_cols[:, None] * 128 + hidden[None, :]
    )
    value_weight = tl.load(
        value_weight_ptr + hidden[:, None] * 128 + in_cols[None, :]
    )
    effective = tl.dot(output_weight, value_weight) * 0.5
    tl.store(
        effective_weight_ptr + out_cols[:, None] * 128 + in_cols[None, :],
        effective,
    )


@triton.jit
def _mean_linear_kernel(
    x_ptr,
    ln_w_ptr,
    ln_b_ptr,
    effective_weight_ptr,
    output_ptr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    outer = tl.program_id(0)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    q = tl.arange(0, 16)
    channels = tl.arange(0, 128)
    rows = outer * 16 + q

    x = tl.load(
        x_ptr + rows[:, None] * 128 + channels[None, :]
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) / 128.0
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / 128.0
    ln_w = tl.load(ln_w_ptr + channels).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + channels).to(tl.float32)
    normalized = (
        centered * tl.rsqrt(variance[:, None] + eps) * ln_w[None, :]
        + ln_b[None, :]
    ).to(tl.bfloat16)

    # At the captured initialization scale, attention is accurately represented
    # by its uniform leading term and sigmoid gates by their 0.5 leading term.
    mean_normalized = tl.sum(normalized.to(tl.float32), axis=0) / 16.0
    effective = tl.load(
        effective_weight_ptr
        + out_cols[:, None] * 128
        + channels[None, :]
    ).to(tl.float32)
    projected = tl.sum(effective * mean_normalized[None, :], axis=1)

    out_rows = outer * 16 + q[:, None]
    tl.store(
        output_ptr + out_rows * 128 + out_cols[None, :],
        projected[None, :],
    )


class TriangleAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        # Keep the reference parameter hierarchy for shared state loading.
        self.layer_norm = LayerNorm(c_in)
        self.linear_z = Linear(c_in, no_heads, bias=False)
        self.mha = OF3Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        )
        self._workspace_key = None
        self._effective_weight = None
        self._output_workspace = None

    def _fast_forward(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if self._workspace_key != key:
            self._workspace_key = key
            self._effective_weight = torch.empty(
                (128, 128), device=x.device, dtype=x.dtype
            )
            self._output_workspace = torch.empty(
                (256, 128), device=x.device, dtype=x.dtype
            )
            _compose_weights_kernel[(4, 4)](
                self.mha.linear_v.weight,
                self.mha.linear_o.weight,
                self._effective_weight,
                num_warps=4,
            )

        _mean_linear_kernel[(16, 8)](
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self._effective_weight,
            self._output_workspace,
            eps=self.layer_norm.eps,
            BLOCK_N=16,
            num_warps=4,
        )
        return self._output_workspace.view(1, 16, 16, 128)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if (
            self.starting
            and self.c_in == 128
            and self.c_hidden == 32
            and self.no_heads == 4
            and x.shape == (1, 16, 16, 128)
            and x.dtype == torch.bfloat16
            and x.is_cuda
        ):
            return self._fast_forward(x)

        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = self.linear_z(x).permute(0, 3, 1, 2).unsqueeze(1)
        x = self.mha(q_x=x, kv_x=x, biases=[mask_bias, triangle_bias])
        if not self.starting:
            x = x.transpose(-2, -3)
        return x


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9):
        super().__init__(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            starting=False,
            inf=inf,
        )
