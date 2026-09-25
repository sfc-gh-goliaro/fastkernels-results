"""Oasis DiT blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L1.silu import SiLU
from ..L2.oasis_mlp import OasisMLP, _gelu_inplace_kernel
from ..L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
from ..L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


@triton.jit
def _norm_mod_kernel(
    x_ptr,
    modulation_ptr,
    out_ptr,
    shift_offset: tl.constexpr,
    scale_offset: tl.constexpr,
    n_cols: tl.constexpr,
    spatial: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = (centered * tl.rsqrt(variance + eps)).to(tl.float16)

    mod_row = row // spatial
    mod_base = modulation_ptr + mod_row * (6 * n_cols)
    shift = tl.load(mod_base + shift_offset + cols, mask=mask)
    scale = tl.load(mod_base + scale_offset + cols, mask=mask)
    scaled = (1.0 + scale).to(tl.float16)
    out = (normalized * scaled).to(tl.float16)
    out = (out + shift).to(tl.float16)
    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


@triton.jit
def _residual_norm_mod_kernel(
    x_ptr,
    branch_ptr,
    gate_modulation_ptr,
    norm_modulation_ptr,
    x_out_ptr,
    modulated_out_ptr,
    gate_offset: tl.constexpr,
    shift_offset: tl.constexpr,
    scale_offset: tl.constexpr,
    n_cols: tl.constexpr,
    spatial: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offset = row * n_cols + cols
    mod_row = row // spatial

    gate_base = gate_modulation_ptr + mod_row * (6 * n_cols)
    gate = tl.load(gate_base + gate_offset + cols, mask=mask)
    x = tl.load(x_ptr + offset, mask=mask)
    branch = tl.load(branch_ptr + offset, mask=mask)
    gated = (gate * branch).to(tl.float16)
    residual = (x + gated).to(tl.float16)
    tl.store(x_out_ptr + offset, residual, mask=mask)

    residual_f32 = tl.where(mask, residual.to(tl.float32), 0.0)
    mean = tl.sum(residual_f32, axis=0) / n_cols
    centered = tl.where(mask, residual_f32 - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = (centered * tl.rsqrt(variance + eps)).to(tl.float16)

    norm_base = norm_modulation_ptr + mod_row * (6 * n_cols)
    shift = tl.load(norm_base + shift_offset + cols, mask=mask)
    scale = tl.load(norm_base + scale_offset + cols, mask=mask)
    scaled = (1.0 + scale).to(tl.float16)
    modulated = (normalized * scaled).to(tl.float16)
    modulated = (modulated + shift).to(tl.float16)
    tl.store(modulated_out_ptr + offset, modulated, mask=mask)


@triton.jit
def _residual_kernel(
    x_ptr,
    branch_ptr,
    modulation_ptr,
    out_ptr,
    gate_offset: tl.constexpr,
    n_cols: tl.constexpr,
    spatial: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offset = row * n_cols + cols
    mod_row = row // spatial
    mod_base = modulation_ptr + mod_row * (6 * n_cols)

    x = tl.load(x_ptr + offset, mask=mask)
    branch = tl.load(branch_ptr + offset, mask=mask)
    gate = tl.load(mod_base + gate_offset + cols, mask=mask)
    gated = (gate * branch).to(tl.float16)
    out = (x + gated).to(tl.float16)
    tl.store(out_ptr + offset, out, mask=mask)


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding,
        temporal_rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.hidden_size = hidden_size
        self._graph_cache = {}

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.float16
            and c.is_contiguous()
            and self.hidden_size == 1024
            and x.shape[0] == 1
            and x.shape[2:4] == (9, 16)
            and c.shape == x.shape[:2] + (1024,)
        ):
            key = (x.device.index, x.shape[1])
            cached = self._graph_cache.get(key)
            if cached is None:
                static_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)
                static_c = torch.empty_like(c)
                static_x.copy_(x)
                static_c.copy_(c)

                current = torch.cuda.current_stream(x.device)
                capture_stream = torch.cuda.Stream(device=x.device)
                capture_stream.wait_stream(current)
                with torch.cuda.stream(capture_stream):
                    warmup_out = self._forward_eager(static_x, static_c)
                capture_stream.synchronize()
                del warmup_out

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    graph_out = self._forward_eager(static_x, static_c)
                current.wait_stream(capture_stream)
                cached = (graph, static_x, static_c, graph_out)
                self._graph_cache[key] = cached
                graph.replay()
                return graph_out

            graph, static_x, static_c, graph_out = cached
            static_x.copy_(x)
            static_c.copy_(c)
            graph.replay()
            return graph_out

        return self._forward_eager(x, c)

    def _forward_eager(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.float16
            and c.is_contiguous()
            and self.hidden_size == 1024
            and x.shape[0] == 1
            and x.shape[2:4] == (9, 16)
            and c.shape == x.shape[:2] + (1024,)
        ):
            n_rows = x.numel() // 1024
            conditioned = self.s_adaLN_modulation[0](c)
            s_modulation = self.s_adaLN_modulation[1](conditioned)
            t_modulation = self.t_adaLN_modulation[1](conditioned)
            residual_input = x.contiguous()
            modulated = torch.empty_like(residual_input)
            state = torch.empty_like(modulated)

            _norm_mod_kernel[(n_rows,)](
                residual_input,
                s_modulation,
                modulated,
                shift_offset=0,
                scale_offset=1024,
                n_cols=1024,
                spatial=144,
                eps=1e-6,
                BLOCK_SIZE=1024,
                num_warps=4,
                launch_pdl=False,
            )
            branch = self.s_attn(modulated)
            _residual_norm_mod_kernel[(n_rows,)](
                residual_input,
                branch,
                s_modulation,
                s_modulation,
                state,
                modulated,
                gate_offset=2048,
                shift_offset=3072,
                scale_offset=4096,
                n_cols=1024,
                spatial=144,
                eps=1e-6,
                BLOCK_SIZE=1024,
                num_warps=4,
                launch_pdl=False,
            )
            hidden = self.s_mlp.fc1(modulated)
            _gelu_inplace_kernel[(triton.cdiv(hidden.numel(), 8192),)](
                hidden,
                hidden.numel(),
                BLOCK_SIZE=8192,
                EVEN=True,
                num_warps=8,
                launch_pdl=False,
            )
            branch = self.s_mlp.fc2(hidden)
            _residual_norm_mod_kernel[(n_rows,)](
                state,
                branch,
                s_modulation,
                t_modulation,
                state,
                modulated,
                gate_offset=5120,
                shift_offset=0,
                scale_offset=1024,
                n_cols=1024,
                spatial=144,
                eps=1e-6,
                BLOCK_SIZE=1024,
                num_warps=4,
                launch_pdl=False,
            )
            branch = self.t_attn(modulated)
            _residual_norm_mod_kernel[(n_rows,)](
                state,
                branch,
                t_modulation,
                t_modulation,
                state,
                modulated,
                gate_offset=2048,
                shift_offset=3072,
                scale_offset=4096,
                n_cols=1024,
                spatial=144,
                eps=1e-6,
                BLOCK_SIZE=1024,
                num_warps=4,
                launch_pdl=False,
            )
            hidden = self.t_mlp.fc1(modulated)
            _gelu_inplace_kernel[(triton.cdiv(hidden.numel(), 8192),)](
                hidden,
                hidden.numel(),
                BLOCK_SIZE=8192,
                EVEN=True,
                num_warps=8,
                launch_pdl=False,
            )
            branch = self.t_mlp.fc2(hidden)
            _residual_kernel[(n_rows,)](
                state,
                branch,
                t_modulation,
                state,
                gate_offset=5120,
                n_cols=1024,
                spatial=144,
                BLOCK_SIZE=1024,
                num_warps=4,
                launch_pdl=False,
            )
            return state

        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
            self.s_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.s_attn(_modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + _gate(self.s_mlp(_modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            self.t_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.t_attn(_modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + _gate(self.t_mlp(_modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)
        return x
