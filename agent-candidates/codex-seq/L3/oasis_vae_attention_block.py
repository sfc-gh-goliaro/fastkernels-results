"""Oasis VAE attention block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_vae_attention import OasisVAEAttention


@triton.jit
def _gelu_inplace_kernel(
    x_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask if not EVEN else None).to(tl.float32)
    x2 = x * x
    p = 3.2125361354614e-3
    p = -5.050443434847393e-2 + x2 * p
    p = 3.884417170626461e-1 + x2 * p
    cdf = tl.maximum(0.0, tl.minimum(1.0, 0.5 + x * p))
    tl.store(x_ptr + offsets, x * cdf, mask=mask if not EVEN else None)


@triton.jit
def _residual_layer_norm_kernel(
    x_ptr,
    residual_ptr,
    norm_ptr,
    weight_ptr,
    bias_ptr,
    n_cols: tl.constexpr,
    n_rows_per_batch: tl.constexpr,
    x_stride_batch: tl.constexpr,
    x_stride_row: tl.constexpr,
    x_stride_col: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    batch = row // n_rows_per_batch
    row_in_batch = row - batch * n_rows_per_batch
    x_offsets = (
        batch * x_stride_batch
        + row_in_batch * x_stride_row
        + cols * x_stride_col
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    update = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    residual = (x + update).to(tl.float16)
    values = residual.to(tl.float32)
    mean = tl.sum(values, axis=0) / n_cols
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = centered * tl.rsqrt(variance + eps)
    normalized *= tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    normalized += tl.load(bias_ptr + cols, mask=mask).to(tl.float32)

    tl.store(residual_ptr + offsets, residual, mask=mask)
    tl.store(norm_ptr + offsets, normalized, mask=mask)


@triton.jit
def _strided_layer_norm_kernel(
    x_ptr,
    norm_ptr,
    weight_ptr,
    bias_ptr,
    n_cols: tl.constexpr,
    n_rows_per_batch: tl.constexpr,
    x_stride_batch: tl.constexpr,
    x_stride_row: tl.constexpr,
    x_stride_col: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    batch = row // n_rows_per_batch
    row_in_batch = row - batch * n_rows_per_batch
    x_offsets = (
        batch * x_stride_batch
        + row_in_batch * x_stride_row
        + cols * x_stride_col
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = centered * tl.rsqrt(variance + eps)
    normalized *= tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    normalized += tl.load(bias_ptr + cols, mask=mask).to(tl.float32)
    tl.store(norm_ptr + row * n_cols + cols, normalized, mask=mask)


class OasisVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self._graph_input_ptr = None

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > 1:
            return self.mlp(x)
        hidden = self.mlp.fc1(x)
        n_elements = hidden.numel()
        block_size = 8192
        _gelu_inplace_kernel[(triton.cdiv(n_elements, block_size),)](
            hidden,
            n_elements,
            BLOCK_SIZE=block_size,
            EVEN=n_elements % block_size == 0,
            num_warps=8,
        )
        return self.mlp.fc2(hidden)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        n_rows_per_batch = x.shape[-2]
        block_size = triton.next_power_of_2(n_cols)
        if x.is_contiguous():
            normalized = self.norm1(x)
        else:
            normalized = torch.empty(
                x.shape, dtype=x.dtype, device=x.device
            )
            _strided_layer_norm_kernel[(n_rows,)](
                x,
                normalized,
                self.norm1.weight,
                self.norm1.bias,
                n_cols=n_cols,
                n_rows_per_batch=n_rows_per_batch,
                x_stride_batch=x.stride(0),
                x_stride_row=x.stride(-2),
                x_stride_col=x.stride(-1),
                eps=self.norm1.eps,
                BLOCK_SIZE=block_size,
                num_warps=4,
            )

        residual = self.attn(normalized)
        _residual_layer_norm_kernel[(n_rows,)](
            x,
            residual,
            normalized,
            self.norm2.weight,
            self.norm2.bias,
            n_cols=n_cols,
            n_rows_per_batch=n_rows_per_batch,
            x_stride_batch=x.stride(0),
            x_stride_row=x.stride(-2),
            x_stride_col=x.stride(-1),
            eps=self.norm2.eps,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return residual.add_(self._mlp_forward(normalized))

    def _capture_graph(self, x: torch.Tensor) -> None:
        direct_input = not x.is_contiguous()
        if direct_input:
            graph_input = x
        else:
            graph_input = torch.empty(
                x.shape, dtype=x.dtype, device=x.device
            )
            graph_input.copy_(x)

        warmup_stream = torch.cuda.Stream(device=x.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(warmup_stream):
            self._eager_forward(graph_input)
        torch.cuda.current_stream(x.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(x.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = self._eager_forward(graph_input)

        self._graph = graph
        self._graph_input = graph_input
        self._graph_output = graph_output
        self._graph_input_ptr = x.data_ptr() if direct_input else None
        self._graph.replay()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or torch.is_grad_enabled():
            return self._eager_forward(x)

        direct_input = not x.is_contiguous()
        needs_capture = self._graph is None
        if direct_input:
            needs_capture = needs_capture or self._graph_input_ptr != x.data_ptr()
        if needs_capture:
            self._capture_graph(x)
            return self._graph_output

        if not direct_input:
            self._graph_input.copy_(x)
        self._graph.replay()
        return self._graph_output
