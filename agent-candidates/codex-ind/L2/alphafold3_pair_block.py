"""PairBlock for AlphaFold3.

Shared pair-representation update block used by PairFormer, MSA module,
and template embedder. Sequence: TriMulOut -> TriMulIn -> TriAttStart ->
TriAttEnd -> SwiGLUTransition.

Reference: openfold3/core/model/latent/base_blocks.py PairBlock
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .alphafold3_triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .alphafold3_triangle_attention import TriangleAttention
from .alphafold3_swiglu_transition import SwiGLUTransition


@triton.jit
def _copy_graph_inputs(
    input_ptr,
    mask_ptr,
    graph_input_ptr,
    graph_mask_ptr,
    block: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block + tl.arange(0, block)
    values = tl.load(input_ptr + offsets)
    tl.store(graph_input_ptr + offsets, values)
    mask_offsets = tl.arange(0, 256)
    mask_values = tl.load(mask_ptr + mask_offsets)
    tl.store(
        graph_mask_ptr + mask_offsets,
        mask_values,
        mask=pid == 0,
    )


@triton.jit
def _ln_projection(
    x_ptr,
    weight_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    n_out: tl.constexpr,
    ending: tl.constexpr,
    scale_q: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """LayerNorm(128) followed by a packed projection."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = pid_n * block_n + tl.arange(0, block_n)
    k = tl.arange(0, 128)

    if ending:
        outer = rows // 16
        inner = rows % 16
        source_rows = inner * 16 + outer
    else:
        source_rows = rows

    x = tl.load(x_ptr + source_rows[:, None] * 128 + k[None, :])
    x32 = x.to(tl.float32)
    mean = tl.sum(x32, axis=1) * (1.0 / 128.0)
    centered = x32 - mean[:, None]
    var = tl.sum(centered * centered, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    ln_w = tl.load(ln_w_ptr + k).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + k).to(tl.float32)
    norm = ((centered * rstd[:, None]) * ln_w[None, :] +
            ln_b[None, :]).to(tl.bfloat16)

    w = tl.load(
        weight_ptr + cols[None, :] * 128 + k[:, None],
        mask=cols[None, :] < n_out,
        other=0.0,
    )
    projected = tl.dot(norm, w)
    if scale_q:
        is_q = cols[None, :] < 128
        rounded = projected.to(tl.bfloat16)
        projected = tl.where(
            is_q,
            (rounded * 0.1767766952966369).to(tl.bfloat16),
            projected,
        )
    tl.store(
        out_ptr + rows[:, None] * n_out + cols[None, :],
        projected,
        mask=(rows[:, None] < 256) & (cols[None, :] < n_out),
    )


@triton.jit
def _triangle_contract(
    projections_ptr,
    mask_ptr,
    out_ptr,
    outgoing: tl.constexpr,
    block_k: tl.constexpr,
    block_c: tl.constexpr,
):
    outer = tl.program_id(0)
    k_block = tl.program_id(1)
    c_block = tl.program_id(2)
    out_k = k_block * block_k + tl.arange(0, block_k)
    red = tl.arange(0, 16)
    channels = c_block * block_c + tl.arange(0, block_c)

    if outgoing:
        a_rows = outer * 16 + red
        b_rows = out_k[:, None] * 16 + red[None, :]
    else:
        a_rows = red * 16 + outer
        b_rows = red[None, :] * 16 + out_k[:, None]

    a_p = tl.load(
        projections_ptr + a_rows[:, None] * 640 + channels[None, :],
    )
    a_g = tl.load(
        projections_ptr + a_rows[:, None] * 640 + 128 + channels[None, :],
    )
    a_mask = tl.load(mask_ptr + a_rows)
    a_gate = tl.sigmoid(a_g.to(tl.float32)).to(tl.bfloat16)
    a = (a_mask[:, None] * a_gate).to(tl.bfloat16)
    a = (a * a_p).to(tl.bfloat16)

    b_p = tl.load(
        projections_ptr
        + b_rows[:, :, None] * 640
        + 256
        + channels[None, None, :],
    )
    b_g = tl.load(
        projections_ptr
        + b_rows[:, :, None] * 640
        + 384
        + channels[None, None, :],
    )
    b_mask = tl.load(mask_ptr + b_rows)
    b_gate = tl.sigmoid(b_g.to(tl.float32)).to(tl.bfloat16)
    b = (b_mask[:, :, None] * b_gate).to(tl.bfloat16)
    b = (b * b_p).to(tl.bfloat16)

    values = tl.sum(
        a[None, :, :].to(tl.float32) * b.to(tl.float32),
        axis=1,
    )
    out_rows = outer * 16 + out_k
    tl.store(
        out_ptr + out_rows[:, None] * 128 + channels[None, :],
        values,
        mask=out_k[:, None] < 16,
    )


@triton.jit
def _triangle_output(
    x_ptr,
    projections_ptr,
    weight_ptr,
    ln_w_ptr,
    ln_b_ptr,
    residual_ptr,
    out_ptr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = pid_n * block_n + tl.arange(0, block_n)
    k = tl.arange(0, 128)

    x = tl.load(x_ptr + rows[:, None] * 128 + k[None, :])
    x32 = x.to(tl.float32)
    mean = tl.sum(x32, axis=1) * (1.0 / 128.0)
    centered = x32 - mean[:, None]
    var = tl.sum(centered * centered, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(var + 1.0e-5)
    ln_w = tl.load(ln_w_ptr + k).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + k).to(tl.float32)
    norm = ((centered * rstd[:, None]) * ln_w[None, :] +
            ln_b[None, :]).to(tl.bfloat16)
    w = tl.load(weight_ptr + cols[None, :] * 128 + k[:, None])
    projected = tl.dot(norm, w).to(tl.bfloat16)

    gate_raw = tl.load(
        projections_ptr + rows[:, None] * 640 + 512 + cols[None, :],
    )
    gate = tl.sigmoid(gate_raw.to(tl.float32)).to(tl.bfloat16)
    update = (projected * gate).to(tl.bfloat16)
    residual = tl.load(residual_ptr + rows[:, None] * 128 + cols[None, :])
    tl.store(
        out_ptr + rows[:, None] * 128 + cols[None, :],
        (residual + update).to(tl.bfloat16),
        mask=rows[:, None] < 256,
    )


@triton.jit
def _triangle_attention(
    projections_ptr,
    mask_ptr,
    attention_out_ptr,
    ending: tl.constexpr,
):
    outer = tl.program_id(0)
    head = tl.program_id(1)
    q_idx = tl.arange(0, 16)
    k_idx = tl.arange(0, 16)
    d = tl.arange(0, 32)

    q_rows = outer * 16 + q_idx
    k_rows = outer * 16 + k_idx
    hd = head * 32 + d
    q = tl.load(projections_ptr + q_rows[:, None] * 516 + hd[None, :])
    k = tl.load(projections_ptr + k_rows[:, None] * 516 + 128 + hd[None, :])
    v = tl.load(projections_ptr + k_rows[:, None] * 516 + 256 + hd[None, :])

    scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
    if ending:
        mask_rows = k_idx * 16 + outer
    else:
        mask_rows = outer * 16 + k_idx
    mask = tl.load(mask_ptr + mask_rows)
    mask_delta = (mask - 1.0).to(tl.bfloat16)
    mask_bias = (mask_delta * 1.0e9).to(tl.bfloat16)
    scores = (scores + mask_bias[None, :]).to(tl.bfloat16)

    bias_rows = q_idx[:, None] * 16 + k_idx[None, :]
    triangle_bias = tl.load(
        projections_ptr + bias_rows * 516 + 512 + head,
    )
    scores = (scores + triangle_bias).to(tl.bfloat16).to(tl.float32)
    scores = scores - tl.max(scores, axis=1)[:, None]
    probabilities = tl.exp(scores)
    probabilities = probabilities / tl.sum(probabilities, axis=1)[:, None]
    probabilities = probabilities.to(tl.bfloat16)
    attended = tl.dot(probabilities, v).to(tl.bfloat16)

    gate_raw = tl.load(
        projections_ptr + q_rows[:, None] * 516 + 384 + hd[None, :],
    )
    gate = tl.sigmoid(gate_raw.to(tl.float32)).to(tl.bfloat16)
    attended = (attended * gate).to(tl.bfloat16)
    tl.store(
        attention_out_ptr + q_rows[:, None] * 128 + hd[None, :],
        attended,
    )


@triton.jit
def _attention_output(
    attention_ptr,
    weight_ptr,
    residual_ptr,
    out_ptr,
    ending: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = pid_n * block_n + tl.arange(0, block_n)
    k = tl.arange(0, 128)
    x = tl.load(attention_ptr + rows[:, None] * 128 + k[None, :])
    w = tl.load(weight_ptr + cols[None, :] * 128 + k[:, None])
    projected = tl.dot(x, w).to(tl.bfloat16)

    if ending:
        outer = rows // 16
        inner = rows % 16
        dest_rows = inner * 16 + outer
    else:
        dest_rows = rows
    residual = tl.load(residual_ptr + dest_rows[:, None] * 128 + cols[None, :])
    tl.store(
        out_ptr + dest_rows[:, None] * 128 + cols[None, :],
        (residual + projected).to(tl.bfloat16),
        mask=rows[:, None] < 256,
    )


@triton.jit
def _transition_output(
    projections_ptr,
    weight_ptr,
    mask_ptr,
    residual_ptr,
    out_ptr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = pid_n * block_n + tl.arange(0, block_n)
    accumulator = tl.zeros((block_m, block_n), tl.float32)
    for start in range(0, 512, block_k):
        k = start + tl.arange(0, block_k)
        a = tl.load(projections_ptr + rows[:, None] * 1024 + k[None, :])
        b = tl.load(
            projections_ptr + rows[:, None] * 1024 + 512 + k[None, :],
        )
        silu = (a * tl.sigmoid(a.to(tl.float32))).to(tl.bfloat16)
        hidden = (silu * b).to(tl.bfloat16)
        w = tl.load(weight_ptr + cols[None, :] * 512 + k[:, None])
        accumulator += tl.dot(hidden, w)
    projected = accumulator.to(tl.bfloat16)
    pair_mask = tl.load(mask_ptr + rows)
    update = (projected * pair_mask[:, None]).to(tl.bfloat16)
    residual = tl.load(residual_ptr + rows[:, None] * 128 + cols[None, :])
    tl.store(
        out_ptr + rows[:, None] * 128 + cols[None, :],
        (residual + update).to(tl.bfloat16),
        mask=rows[:, None] < 256,
    )


class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template.

    Args:
        c_z: Pair embedding channel dimension
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Per-head hidden dim for triangle attention
        no_heads_pair: Number of heads in triangle attention
        transition_n: Scale of pair transition hidden dimension
        pair_dropout: Dropout rate (unused in inference baseline)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)

        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )

        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)
        self._packed_weights = None
        self._graph_warmed = False
        self._cuda_graph = None
        self._graph_input = None
        self._graph_mask = None
        self._graph_scratch = None
        self._graph_output = None

    def _pack_projection_weights(self):
        if self._packed_weights is not None:
            return self._packed_weights

        def triangle_weights(module):
            return torch.cat((
                module.linear_a_p.weight,
                module.linear_a_g.weight,
                module.linear_b_p.weight,
                module.linear_b_g.weight,
                module.linear_g.weight,
            ), dim=0).contiguous()

        def attention_weights(module):
            return torch.cat((
                module.mha.linear_q.weight,
                module.mha.linear_k.weight,
                module.mha.linear_v.weight,
                module.mha.linear_g.weight,
                module.linear_z.weight,
            ), dim=0).contiguous()

        transition_weights = torch.cat((
            self.pair_transition.swiglu.linear_a.weight,
            self.pair_transition.swiglu.linear_b.weight,
        ), dim=0).contiguous()
        self._packed_weights = (
            triangle_weights(self.tri_mul_out),
            triangle_weights(self.tri_mul_in),
            attention_weights(self.tri_att_start),
            attention_weights(self.tri_att_end),
            transition_weights,
        )
        return self._packed_weights

    def _execute_fused(self, z, pair_mask, scratch, result, packed_weights):
        (tri_out_w, tri_in_w, att_start_w, att_end_w,
         transition_w) = packed_weights

        def triangle_update(current, module, packed_weight, outgoing, output):
            projections = scratch[:256 * 640].view(256, 640)
            contracted = scratch[256 * 640:256 * 640 + 256 * 128].view(256, 128)
            _ln_projection[(8, 10)](
                current, packed_weight,
                module.layer_norm_in.weight, module.layer_norm_in.bias,
                projections, 640, False, False, 32, 64,
                num_warps=4,
            )
            _triangle_contract[(16, 4, 4)](
                projections, pair_mask, contracted, outgoing, 4, 32,
                num_warps=4,
            )
            _triangle_output[(8, 2)](
                contracted, projections, module.linear_z.weight,
                module.layer_norm_out.weight, module.layer_norm_out.bias,
                current, output, 32, 64, num_warps=4,
            )

        triangle_update(z, self.tri_mul_out, tri_out_w, True, result)
        triangle_update(result, self.tri_mul_in, tri_in_w, False, result)

        def attention_update(current, module, packed_weight, ending):
            projections = scratch[:256 * 516].view(256, 516)
            attention_out = scratch[
                256 * 516:256 * 516 + 256 * 128
            ].view(256, 128)
            _ln_projection[(16, 17)](
                current, packed_weight,
                module.layer_norm.weight, module.layer_norm.bias,
                projections, 516, ending, True, 16, 32,
                num_warps=4,
            )
            _triangle_attention[(16, 4)](
                projections, pair_mask, attention_out, ending,
                num_warps=4,
            )
            _attention_output[(8, 2)](
                attention_out, module.mha.linear_o.weight,
                current, current, ending, 32, 64, num_warps=4,
            )

        attention_update(result, self.tri_att_start, att_start_w, False)
        attention_update(result, self.tri_att_end, att_end_w, True)

        transition_projection = scratch[:256 * 1024].view(256, 1024)
        transition = self.pair_transition
        _ln_projection[(8, 16)](
            result, transition_w,
            transition.layer_norm.weight, transition.layer_norm.bias,
            transition_projection, 1024, False, False, 32, 64,
            num_warps=4,
        )
        _transition_output[(16, 2)](
            transition_projection, transition.linear_out.weight,
            pair_mask, result, result, 16, 64, 32,
            num_warps=4,
        )
        return result

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:         [*, N, N, C_z] pair embedding
            pair_mask: [*, N, N] pair mask

        Returns:
            [*, N, N, C_z] updated pair embedding
        """
        # The captured AF3 shape is fixed at [1, 16, 16, 128]. Keep the
        # reference path available for unsupported calls.
        if (z.shape != (1, 16, 16, 128)
                or z.dtype != torch.bfloat16
                or pair_mask.shape != (1, 16, 16)
                or not _mask_trans):
            pair_trans_mask = pair_mask if _mask_trans else None
            z = z + self.tri_mul_out(z, mask=pair_mask)
            z = z + self.tri_mul_in(z, mask=pair_mask)
            z = z + self.tri_att_start(z, mask=pair_mask)
            z = z + self.tri_att_end(z, mask=pair_mask)
            return z + self.pair_transition(z, mask=pair_trans_mask)

        packed_weights = self._pack_projection_weights()
        if self._cuda_graph is not None:
            _copy_graph_inputs[(32,)](
                z, pair_mask, self._graph_input, self._graph_mask, 1024,
                num_warps=4,
            )
            self._cuda_graph.replay()
            return self._graph_output.clone()

        if not self._graph_warmed:
            scratch = torch.empty(262144, device=z.device, dtype=z.dtype)
            result = torch.empty_like(z)
            result = self._execute_fused(
                z, pair_mask, scratch, result, packed_weights,
            )
            self._graph_warmed = True
            return result

        self._graph_input = torch.empty_like(z)
        self._graph_mask = torch.empty_like(pair_mask)
        self._graph_scratch = torch.empty(
            262144, device=z.device, dtype=z.dtype,
        )
        self._graph_output = torch.empty_like(z)
        self._graph_input.copy_(z)
        self._graph_mask.copy_(pair_mask)
        torch.cuda.synchronize(z.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._execute_fused(
                self._graph_input,
                self._graph_mask,
                self._graph_scratch,
                self._graph_output,
                packed_weights,
            )
        self._cuda_graph = graph
        graph.replay()
        return self._graph_output.clone()
