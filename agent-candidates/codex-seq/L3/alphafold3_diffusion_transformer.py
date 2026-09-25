"""Diffusion transformer for AlphaFold3.

24-block transformer used inside the diffusion module. Each block:
AttentionPairBias + ConditionedTransitionBlock (AdaLN-Zero).

Reference: openfold3/core/model/layers/diffusion_transformer.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias, CrossAttentionPairBias
from ..L2.alphafold3_attention_pair_bias import (
    _ada_projection_kernel,
    _cross_attention_kernel,
    _cross_gather_norm_kernel,
    _gated_output_kernel,
    _self_ada_projection_kernel,
    _self_attention_kernel,
)
from ..L2.alphafold3_swiglu_transition import (
    ConditionedTransitionBlock,
    _swiglu,
)


__targets__ = ["DiffusionTransformer"]


@triton.jit
def _copy_graph_inputs(
    a,
    s,
    z,
    mask,
    a_out,
    s_out,
    z_out,
    mask_out,
    N_A: tl.constexpr,
    N_S: tl.constexpr,
    N_Z: tl.constexpr,
    N_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a_mask = offsets < N_A
    tl.store(a_out + offsets, tl.load(a + offsets, mask=a_mask), mask=a_mask)
    s_mask = offsets < N_S
    tl.store(s_out + offsets, tl.load(s + offsets, mask=s_mask), mask=s_mask)
    z_mask = offsets < N_Z
    tl.store(z_out + offsets, tl.load(z + offsets, mask=z_mask), mask=z_mask)
    m_mask = offsets < N_MASK
    tl.store(
        mask_out + offsets,
        tl.load(mask + offsets, mask=m_mask),
        mask=m_mask,
    )


@triton.jit
def _stack_norm_s(
    s,
    attention_weights,
    transition_weights,
    attention_out,
    transition_out,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    layer = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    values = tl.load(
        s + row * K + cols, mask=col_mask, other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / K
    centered = tl.where(col_mask, values - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / K + 1e-5)
    normalized = centered * rstd
    base = (layer * M + row) * K + cols
    aw = tl.load(
        attention_weights + layer * K + cols,
        mask=col_mask,
        other=0.0,
    )
    tw = tl.load(
        transition_weights + layer * K + cols,
        mask=col_mask,
        other=0.0,
    )
    tl.store(attention_out + base, normalized * aw, mask=col_mask)
    tl.store(transition_out + base, normalized * tw, mask=col_mask)


@triton.jit
def _stack_linear(
    x,
    weights,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SIGMOID: tl.constexpr,
    SHARED_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    layer = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, 16)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((16, BLOCK_N), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + ks
        x_layer = 0 if SHARED_X else layer * M * K
        xv = tl.load(
            x + x_layer + rows[:, None] * K + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            weights
            + (layer * N + cols[None, :]) * K
            + k[:, None],
            mask=(cols[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(xv, w)
    if HAS_BIAS:
        acc += tl.load(
            bias + layer * N + cols, mask=cols < N, other=0.0,
        )[None, :]
    value = acc.to(tl.bfloat16)
    if APPLY_SIGMOID:
        value = tl.sigmoid(value.to(tl.float32)).to(tl.bfloat16)
    tl.store(
        out + (layer * M + rows[:, None]) * N + cols[None, :],
        value,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _stack_pair_bias(
    z,
    norm_weights,
    linear_weights,
    out,
    M: tl.constexpr,
    C_Z: tl.constexpr,
    N_HEAD: tl.constexpr,
):
    layer = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.arange(0, C_Z)
    heads = tl.arange(0, N_HEAD)
    values = tl.load(z + row * C_Z + cols).to(tl.float32)
    mean = tl.sum(values, axis=0) / C_Z
    values -= mean
    values *= tl.rsqrt(tl.sum(values * values, axis=0) / C_Z + 1e-5)
    values *= tl.load(norm_weights + layer * C_Z + cols).to(tl.float32)
    weights = tl.load(
        linear_weights
        + layer * N_HEAD * C_Z
        + heads[:, None] * C_Z
        + cols[None, :]
    )
    projected = tl.sum(values[None, :].to(weights.dtype) * weights, axis=1)
    tl.store(out + (layer * M + row) * N_HEAD + heads, projected)


@triton.jit
def _stack_cross_pair_bias(
    z,
    norm_weight,
    linear_weights,
    out,
    N_ROWS: tl.constexpr,
    N_HEAD: tl.constexpr,
    N_LAYER: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, 16)
    outputs = tl.arange(0, 16)
    valid_rows = rows < N_ROWS
    values = tl.load(
        z + rows[:, None] * 16 + cols[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=1) / 16
    centered = values - mean[:, None]
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) / 16 + 1e-5)
    scale = tl.load(norm_weight + cols).to(tl.float32)
    normalized = (centered * rstd[:, None] * scale[None, :]).to(tl.bfloat16)
    valid_outputs = outputs < N_LAYER * N_HEAD
    weights = tl.load(
        linear_weights + outputs[None, :] * 16 + cols[:, None],
        mask=valid_outputs[None, :],
        other=0.0,
    )
    projected = tl.dot(normalized, weights)
    layer = outputs // N_HEAD
    head = outputs % N_HEAD
    offsets = (
        layer[None, :] * N_ROWS * N_HEAD
        + rows[:, None] * N_HEAD
        + head[None, :]
    )
    tl.store(
        out + offsets,
        projected,
        mask=valid_rows[:, None] & valid_outputs[None, :],
    )


@triton.jit
def _normalize_a(
    a,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    values = tl.load(
        a + row * N + cols, mask=col_mask, other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / N
    centered = tl.where(col_mask, values - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / N + 1e-5)
    tl.store(out + row * N + cols, centered * rstd, mask=col_mask)


@triton.jit
def _attention_output_residual(
    attended,
    residual,
    weight,
    gate,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((16, BLOCK_N), tl.float32)
    for start in range(0, N, BLOCK_K):
        k = start + ks
        x = tl.load(
            attended + rows[:, None] * N + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < N),
            other=0.0,
        )
        w = tl.load(
            weight + cols[None, :] * N + k[:, None],
            mask=(cols[None, :] < N) & (k[:, None] < N),
            other=0.0,
        )
        acc += tl.dot(x, w)
    projected = acc.to(tl.bfloat16)
    gate_value = tl.load(
        gate + rows[:, None] * N + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < N),
        other=0.0,
    )
    update = (projected * gate_value).to(tl.bfloat16).to(tl.float32)
    old = tl.load(
        residual + rows[:, None] * N + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        old + update,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _cross_output_residual(
    attended,
    residual,
    s,
    out_weight,
    gate_weight,
    gate_bias,
    out,
    M: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, C, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        x = tl.load(
            attended + rows[:, None] * C + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < C),
            other=0.0,
        )
        w = tl.load(
            out_weight + cols[None, :] * C + k[:, None],
            mask=(cols[None, :] < C) & (k[:, None] < C),
            other=0.0,
        )
        acc += tl.dot(x, w)
        sv = tl.load(
            s + rows[:, None] * C + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < C),
            other=0.0,
        )
        gw = tl.load(
            gate_weight + cols[None, :] * C + k[:, None],
            mask=(cols[None, :] < C) & (k[:, None] < C),
            other=0.0,
        )
        gate_acc += tl.dot(sv, gw)

    projected = acc.to(tl.bfloat16)
    gate_acc += tl.load(
        gate_bias + cols, mask=cols < C, other=0.0,
    )[None, :]
    gate_value = tl.sigmoid(
        gate_acc.to(tl.bfloat16).to(tl.float32),
    ).to(tl.bfloat16)
    update = (projected * gate_value).to(tl.bfloat16).to(tl.float32)
    old = tl.load(
        residual + rows[:, None] * C + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < C),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out + rows[:, None] * C + cols[None, :],
        old + update,
        mask=(rows[:, None] < M) & (cols[None, :] < C),
    )


@triton.jit
def _apply_adaln_params(
    a,
    ada_params,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    values = tl.load(
        a + row * N + cols, mask=col_mask, other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / N
    centered = tl.where(col_mask, values - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / N + 1e-5)
    normalized = (centered * rstd).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(ada_params + row * (2 * N) + cols).to(tl.float32)
    gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    shift = tl.load(
        ada_params + row * (2 * N) + N + cols,
    ).to(tl.float32)
    shifted = (normalized + shift).to(tl.bfloat16).to(tl.float32)
    tl.store(out + row * N + cols, gate * shifted, mask=col_mask)


@triton.jit
def _apply_attention_params(
    a,
    ada_params,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    values = tl.load(
        a + row * N + cols, mask=col_mask, other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / N
    centered = tl.where(col_mask, values - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / N + 1e-5)
    normalized = (centered * rstd).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(ada_params + row * (2 * N) + cols).to(tl.float32)
    gate = tl.sigmoid(gate)
    shift = tl.load(
        ada_params + row * (2 * N) + N + cols,
    ).to(tl.float32)
    tl.store(
        out + row * N + cols,
        gate * (normalized + shift),
        mask=col_mask,
    )


@triton.jit
def _transition_output_residual(
    hidden,
    residual,
    weight,
    gate,
    mask,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((16, BLOCK_N), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + ks
        x = tl.load(
            hidden + rows[:, None] * K + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            weight + cols[None, :] * K + k[:, None],
            mask=(cols[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x, w)
    projected = acc.to(tl.bfloat16)
    gate_value = tl.load(
        gate + rows[:, None] * N + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < N),
        other=0.0,
    )
    update = (projected * gate_value).to(tl.bfloat16).to(tl.float32)
    row_mask = tl.load(mask + rows, mask=rows < M, other=0.0)
    update *= row_mask[:, None]
    update = update.to(tl.bfloat16).to(tl.float32)
    old = tl.load(
        residual + rows[:, None] * N + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        old + update,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


class DiffusionTransformerBlock(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer block.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
    ):
        super().__init__()
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = AttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                gating=True,
                inf=inf,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                gating=True,
                inf=inf,
            )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=n_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class DiffusionTransformer(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer stack.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
        blocks_per_ckpt: int | None = None,
        **kwargs,
    ):
        super().__init__()
        from ..L1.layer_norm import LayerNorm

        self.use_cross_attention = n_query is not None
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.no_heads = no_heads
        self.no_blocks = no_blocks
        if self.use_cross_attention:
            self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a, c_s=c_s, c_z=c_z,
                c_hidden=c_hidden, no_heads=no_heads,
                n_transition=n_transition,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])
        self._cuda_graph = None
        self._graph_inputs = None
        self._graph_output = None
        self._attention_norm_weights = None
        self._transition_norm_weights = None
        self._attention_ada_weights = None
        self._attention_ada_bias = None
        self._transition_ada_weights = None
        self._transition_ada_bias = None
        self._attention_output_gate_weights = None
        self._attention_output_gate_bias = None
        self._transition_output_gate_weights = None
        self._transition_output_gate_bias = None
        self._pair_norm_weights = None
        self._pair_linear_weights = None
        self._cross_pair_weights = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(
            state_dict, strict=strict, assign=assign,
        )
        # Recursive state loading bypasses the L2 module's public load hook,
        # which assembles its fused projection weights.
        for block in self.blocks:
            attention = block.attention_pair_bias
            attention.load_state_dict(attention.state_dict())
        if (
            not self.use_cross_attention
            and self.c_a == 768
            and self.c_s == 384
            and self.no_heads == 16
        ):
            attentions = [block.attention_pair_bias for block in self.blocks]
            transitions = [block.conditioned_transition for block in self.blocks]
            self._attention_norm_weights = torch.stack([
                module.layer_norm_a.layer_norm_s.weight
                for module in attentions
            ]).contiguous()
            self._transition_norm_weights = torch.stack([
                module.layer_norm.layer_norm_s.weight
                for module in transitions
            ]).contiguous()
            self._attention_ada_weights = torch.stack([
                module._ada_weight for module in attentions
            ]).contiguous()
            self._attention_ada_bias = torch.stack([
                module._ada_bias for module in attentions
            ]).contiguous()
            self._transition_ada_weights = torch.stack([
                torch.cat([
                    module.layer_norm.linear_g.weight,
                    module.layer_norm.linear_s.weight,
                ])
                for module in transitions
            ]).contiguous()
            self._transition_ada_bias = torch.stack([
                torch.cat([
                    module.layer_norm.linear_g.bias,
                    torch.zeros_like(module.layer_norm.linear_g.bias),
                ])
                for module in transitions
            ]).contiguous()
            self._attention_output_gate_weights = torch.stack([
                module.linear_ada_out.weight for module in attentions
            ]).contiguous()
            self._attention_output_gate_bias = torch.stack([
                module.linear_ada_out.bias for module in attentions
            ]).contiguous()
            self._transition_output_gate_weights = torch.stack([
                module.linear_g.weight for module in transitions
            ]).contiguous()
            self._transition_output_gate_bias = torch.stack([
                module.linear_g.bias for module in transitions
            ]).contiguous()
            self._pair_norm_weights = torch.stack([
                module.layer_norm_z.weight for module in attentions
            ]).contiguous()
            self._pair_linear_weights = torch.stack([
                module.linear_z.weight for module in attentions
            ]).contiguous()
        elif (
            self.use_cross_attention
            and self.c_a == 128
            and self.c_z == 16
            and self.no_heads == 4
        ):
            self._cross_pair_weights = torch.cat([
                block.attention_pair_bias.linear_z.weight
                for block in self.blocks
            ]).contiguous()
        self._cuda_graph = None
        self._graph_inputs = None
        self._graph_output = None
        return result

    def _forward_cross_optimized(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        c = 128
        n_atom = 368
        n_blocks = 12
        n_query = 32
        n_key = 128
        q_rows = n_blocks * n_query
        k_rows = n_blocks * n_key
        z_rows = n_blocks * n_query * n_key
        pair_bias = torch.empty(
            (self.no_blocks, z_rows, self.no_heads),
            device=z.device,
            dtype=z.dtype,
        )
        _stack_cross_pair_bias[(triton.cdiv(z_rows, 32),)](
            z,
            self.layer_norm_z.weight,
            self._cross_pair_weights,
            pair_bias,
            N_ROWS=z_rows,
            N_HEAD=self.no_heads,
            N_LAYER=self.no_blocks,
            BLOCK_ROWS=32,
            num_warps=4,
        )

        for layer, block in enumerate(self.blocks):
            attention = block.attention_pair_bias
            aq_norm = torch.empty(
                (q_rows, c), device=a.device, dtype=a.dtype,
            )
            sq_norm = torch.empty_like(aq_norm)
            ak_norm = torch.empty(
                (k_rows, c), device=a.device, dtype=a.dtype,
            )
            sk_norm = torch.empty_like(ak_norm)
            _cross_gather_norm_kernel[(k_rows,)](
                a,
                s,
                aq_norm,
                sq_norm,
                ak_norm,
                sk_norm,
                attention.layer_norm_a_q.layer_norm_s.weight,
                attention.layer_norm_a_k.layer_norm_s.weight,
                C=c,
                N_ATOM=n_atom,
                N_QUERY=n_query,
                N_KEY=n_key,
                EPS=1e-5,
                num_warps=4,
            )
            q_ada = F.linear(
                sq_norm, attention._ada_q_weight, attention._ada_q_bias,
            )
            k_ada = F.linear(
                sk_norm, attention._ada_k_weight, attention._ada_k_bias,
            )
            qg = torch.empty(
                (q_rows, 2 * c), device=a.device, dtype=a.dtype,
            )
            kv = torch.empty(
                (k_rows, 2 * c), device=a.device, dtype=a.dtype,
            )
            _ada_projection_kernel[(
                triton.cdiv(q_rows, 16), triton.cdiv(2 * c, 64),
            )](
                aq_norm,
                q_ada,
                attention._qg_weight,
                attention._qg_bias,
                qg,
                M=q_rows,
                C=c,
                OUT_C=2 * c,
                HAS_BIAS=True,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=32,
                num_warps=4,
            )
            _ada_projection_kernel[(
                triton.cdiv(k_rows, 16), triton.cdiv(2 * c, 64),
            )](
                ak_norm,
                k_ada,
                attention._kv_weight,
                attention._qg_bias,
                kv,
                M=k_rows,
                C=c,
                OUT_C=2 * c,
                HAS_BIAS=False,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=32,
                num_warps=4,
            )
            attended = torch.empty(
                (q_rows, c), device=a.device, dtype=a.dtype,
            )
            _cross_attention_kernel[(n_blocks, self.no_heads)](
                qg,
                kv,
                pair_bias[layer],
                attended,
                N_QUERY=n_query,
                N_KEY=n_key,
                C=c,
                HEAD_DIM=attention.mha.c_hidden,
                N_HEAD=self.no_heads,
                num_warps=8,
                num_stages=3,
            )
            attention_residual = torch.empty_like(a)
            _cross_output_residual[(
                triton.cdiv(n_atom, 32), triton.cdiv(c, 32),
            )](
                attended,
                a,
                s,
                attention.mha.linear_o.weight,
                attention.linear_ada_out.weight,
                attention.linear_ada_out.bias,
                attention_residual,
                M=n_atom,
                C=c,
                BLOCK_M=32,
                BLOCK_N=32,
                BLOCK_K=32,
                num_warps=4,
            )
            a = attention_residual
            a = a + block.conditioned_transition(a=a, s=s, mask=mask)
        return a

    def _forward_self_optimized(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        layers = self.no_blocks
        m = 16
        c = 768
        c_s = 384
        normalized_attention_s = torch.empty(
            (layers, m, c_s), device=s.device, dtype=s.dtype,
        )
        normalized_transition_s = torch.empty_like(normalized_attention_s)
        _stack_norm_s[(layers, m)](
            s,
            self._attention_norm_weights,
            self._transition_norm_weights,
            normalized_attention_s,
            normalized_transition_s,
            M=m,
            K=c_s,
            BLOCK_K=512,
            num_warps=4,
        )

        attention_params = torch.empty(
            (layers, m, 2 * c), device=s.device, dtype=s.dtype,
        )
        transition_params = torch.empty_like(attention_params)
        _stack_linear[(layers, triton.cdiv(2 * c, 64))](
            normalized_attention_s,
            self._attention_ada_weights,
            self._attention_ada_bias,
            attention_params,
            M=m,
            N=2 * c,
            K=c_s,
            HAS_BIAS=True,
            APPLY_SIGMOID=False,
            SHARED_X=False,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
        _stack_linear[(layers, triton.cdiv(2 * c, 64))](
            normalized_transition_s,
            self._transition_ada_weights,
            self._transition_ada_bias,
            transition_params,
            M=m,
            N=2 * c,
            K=c_s,
            HAS_BIAS=True,
            APPLY_SIGMOID=False,
            SHARED_X=False,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
        attention_output_gates = torch.empty(
            (layers, m, c), device=s.device, dtype=s.dtype,
        )
        transition_output_gates = torch.empty_like(attention_output_gates)
        _stack_linear[(layers, triton.cdiv(c, 64))](
            s,
            self._attention_output_gate_weights,
            self._attention_output_gate_bias,
            attention_output_gates,
            M=m,
            N=c,
            K=c_s,
            HAS_BIAS=True,
            APPLY_SIGMOID=True,
            SHARED_X=True,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
            num_stages=3,
        )
        _stack_linear[(layers, triton.cdiv(c, 64))](
            s,
            self._transition_output_gate_weights,
            self._transition_output_gate_bias,
            transition_output_gates,
            M=m,
            N=c,
            K=c_s,
            HAS_BIAS=True,
            APPLY_SIGMOID=True,
            SHARED_X=True,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )

        pair_bias = torch.empty(
            (layers, m * m, self.no_heads),
            device=z.device,
            dtype=z.dtype,
        )
        _stack_pair_bias[(layers, m * m)](
            z,
            self._pair_norm_weights,
            self._pair_linear_weights,
            pair_bias,
            M=m * m,
            C_Z=self.c_z,
            N_HEAD=self.no_heads,
            num_warps=4,
        )

        for layer, block in enumerate(self.blocks):
            attention = block.attention_pair_bias
            transition = block.conditioned_transition

            adapted_a = torch.empty_like(a)
            _apply_attention_params[(m,)](
                a,
                attention_params[layer],
                adapted_a,
                M=m,
                N=c,
                BLOCK_N=1024,
                num_warps=4,
            )
            qkvg = torch.empty(
                (m, 4 * c), device=a.device, dtype=a.dtype,
            )
            _stack_linear[(1, triton.cdiv(4 * c, 16))](
                adapted_a,
                attention._qkvg_weight,
                attention._qkvg_bias,
                qkvg,
                M=m,
                N=4 * c,
                K=c,
                HAS_BIAS=True,
                APPLY_SIGMOID=False,
                SHARED_X=True,
                BLOCK_N=16,
                BLOCK_K=32,
                num_warps=4,
                num_stages=3,
            )
            attended = torch.empty_like(a)
            _self_attention_kernel[(self.no_heads,)](
                qkvg,
                pair_bias[layer],
                attended,
                C=c,
                HEAD_DIM=attention.mha.c_hidden,
                HEAD_BLOCK=64,
                N_HEAD=self.no_heads,
                N_TOKEN=m,
                num_warps=4,
            )
            attention_residual = torch.empty_like(a)
            _attention_output_residual[(triton.cdiv(c, 16),)](
                attended,
                a,
                attention.mha.linear_o.weight,
                attention_output_gates[layer],
                attention_residual,
                M=m,
                N=c,
                BLOCK_N=16,
                BLOCK_K=32,
                num_warps=4,
                num_stages=3,
            )

            transition_input = torch.empty_like(a)
            _apply_adaln_params[(m,)](
                attention_residual,
                transition_params[layer],
                transition_input,
                M=m,
                N=c,
                BLOCK_N=1024,
                num_warps=4,
            )
            hidden = _swiglu(transition_input, transition.swiglu)
            next_a = torch.empty_like(a)
            _transition_output_residual[(triton.cdiv(c, 64),)](
                hidden,
                attention_residual,
                transition.linear_out.weight,
                transition_output_gates[layer],
                mask,
                next_a,
                M=m,
                N=c,
                K=2 * c,
                BLOCK_N=64,
                BLOCK_K=64,
                num_warps=4,
                num_stages=4,
            )
            a = next_a
        return a

    def _forward_impl(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        if (
            self.use_cross_attention
            and a.is_cuda
            and a.dtype == torch.bfloat16
            and a.numel() == 368 * 128
            and self._cross_pair_weights is not None
            and mask is not None
        ):
            return self._forward_cross_optimized(a, s, z, mask)

        if (
            not self.use_cross_attention
            and a.is_cuda
            and a.dtype == torch.bfloat16
            and a.numel() == 16 * 768
            and self._attention_ada_weights is not None
            and mask is not None
        ):
            return self._forward_self_optimized(a, s, z, mask)

        if self.use_cross_attention:
            z = self.layer_norm_z(z)

        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask)

        return a

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        if not a.is_cuda or torch.is_grad_enabled() or mask is None:
            return self._forward_impl(
                a, s, z, mask, use_deepspeed_evo_attention,
                use_cueq_triangle_kernels, use_lma,
                use_high_precision_attention, _mask_trans,
            )

        if self._cuda_graph is None:
            static_inputs = (a.clone(), s.clone(), z.clone(), mask.clone())
            # Compile and initialize all lazy library state before capture.
            self._forward_impl(*static_inputs)
            torch.cuda.synchronize(a.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = self._forward_impl(*static_inputs)
            self._cuda_graph = graph
            self._graph_inputs = static_inputs
            self._graph_output = graph_output
        else:
            static_a, static_s, static_z, static_mask = self._graph_inputs
            max_size = max(a.numel(), s.numel(), z.numel(), mask.numel())
            _copy_graph_inputs[(triton.cdiv(max_size, 1024),)](
                a,
                s,
                z,
                mask,
                static_a,
                static_s,
                static_z,
                static_mask,
                N_A=a.numel(),
                N_S=s.numel(),
                N_Z=z.numel(),
                N_MASK=mask.numel(),
                BLOCK=1024,
                num_warps=8,
            )

        self._cuda_graph.replay()
        return self._graph_output
