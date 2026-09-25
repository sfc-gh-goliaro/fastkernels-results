"""Triangle attention for AlphaFold3 (L2).

Implements AF3 Algorithms 14 (starting node) and 15 (ending node).
Self-attention over one dimension of the pair representation with a
learned triangle bias from the other dimension.

Reference: openfold3/core/model/layers/triangular_attention.py TriangleAttention
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_of3_attention import OF3Attention


@triton.jit
def _norm_project_kernel(
    x_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    wg_ptr,
    wz_ptr,
    projections_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    projection = tl.program_id(1)
    pid_n = tl.program_id(2)

    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, C)
    x = tl.load(
        x_ptr + rows[:, None] * C + cols[None, :],
        mask=rows[:, None] < M,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / C
    normed = centered * tl.rsqrt(variance[:, None] + 1e-5)
    normed = normed * tl.load(norm_weight_ptr + cols)[None, :]
    normed += tl.load(norm_bias_ptr + cols)[None, :]
    # The standalone LayerNorm writes BF16 before each baseline projection.
    normed = normed.to(tl.bfloat16)

    out_cols = pid_n * BN + tl.arange(0, BN)
    valid_cols = tl.where(projection == 4, out_cols < 4, out_cols < C)
    weight_offsets = out_cols[None, :] * C + cols[:, None]

    # Runtime control flow keeps all projections in one launch without
    # redundantly loading the four full weight matrices.
    if projection == 0:
        weight = tl.load(wq_ptr + weight_offsets, mask=valid_cols[None, :])
    elif projection == 1:
        weight = tl.load(wk_ptr + weight_offsets, mask=valid_cols[None, :])
    elif projection == 2:
        weight = tl.load(wv_ptr + weight_offsets, mask=valid_cols[None, :])
    elif projection == 3:
        weight = tl.load(wg_ptr + weight_offsets, mask=valid_cols[None, :])
    else:
        weight = tl.load(wz_ptr + weight_offsets, mask=valid_cols[None, :])

    projected = tl.dot(normed, weight)
    if projection == 0:
        # Match the explicit BF16 q projection followed by BF16 scaling.
        projected = projected.to(tl.bfloat16).to(tl.float32) * 0.1767766952966369

    out_offsets = (
        projection * M * C + rows[:, None] * C + out_cols[None, :]
    )
    tl.store(
        projections_ptr + out_offsets,
        projected,
        mask=(rows[:, None] < M) & valid_cols[None, :],
    )


@triton.jit
def _attention_gate_kernel(
    projections_ptr,
    mask_ptr,
    gated_ptr,
    inf: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
):
    outer = tl.program_id(0)
    head = tl.program_id(1)
    q_idx = tl.arange(0, N)
    k_idx = tl.arange(0, N)
    d_idx = tl.arange(0, D)

    slab = N * N * C
    row_base = outer * N
    q = tl.load(
        projections_ptr
        + (row_base + q_idx[:, None]) * C
        + head * D
        + d_idx[None, :]
    )
    k = tl.load(
        projections_ptr
        + slab
        + (row_base + k_idx[:, None]) * C
        + head * D
        + d_idx[None, :]
    )
    v = tl.load(
        projections_ptr
        + 2 * slab
        + (row_base + k_idx[:, None]) * C
        + head * D
        + d_idx[None, :]
    )

    scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
    mask_bias = (
        (tl.load(mask_ptr + outer * N + k_idx).to(tl.float32) - 1.0) * inf
    ).to(tl.bfloat16)
    scores = (scores + mask_bias[None, :]).to(tl.bfloat16)

    # Triangle bias is indexed by (query, key, head), independent of `outer`.
    z_offsets = (
        4 * slab
        + (q_idx[:, None] * N + k_idx[None, :]) * C
        + head
    )
    scores = (scores + tl.load(projections_ptr + z_offsets)).to(tl.bfloat16)
    scores_f32 = scores.to(tl.float32)
    scores_f32 -= tl.max(scores_f32, axis=1)[:, None]
    numerator = tl.exp(scores_f32)
    probabilities = (
        numerator / tl.sum(numerator, axis=1)[:, None]
    ).to(tl.bfloat16)

    attended = tl.dot(probabilities, v).to(tl.bfloat16)
    gate = tl.load(
        projections_ptr
        + 3 * slab
        + (row_base + q_idx[:, None]) * C
        + head * D
        + d_idx[None, :]
    )
    gate = tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
    attended = (attended * gate).to(tl.bfloat16)
    tl.store(
        gated_ptr
        + (row_base + q_idx[:, None]) * C
        + head * D
        + d_idx[None, :],
        attended,
    )


@triton.jit
def _output_project_kernel(
    gated_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    out_cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, C)
    gated = tl.load(
        gated_ptr + rows[:, None] * C + inner[None, :],
        mask=rows[:, None] < M,
        other=0.0,
    )
    weight = tl.load(
        weight_ptr + out_cols[None, :] * C + inner[:, None],
        mask=out_cols[None, :] < C,
        other=0.0,
    )
    output = tl.dot(gated, weight)
    tl.store(
        output_ptr + rows[:, None] * C + out_cols[None, :],
        output,
        mask=(rows[:, None] < M) & (out_cols[None, :] < C),
    )


@triton.jit
def _attention_gate_output_kernel(
    projections_ptr,
    mask_ptr,
    output_weight_ptr,
    output_ptr,
    inf: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BN: tl.constexpr,
):
    outer = tl.program_id(0)
    out_cols = tl.program_id(1) * BN + tl.arange(0, BN)
    q_idx = tl.arange(0, N)
    k_idx = tl.arange(0, N)
    d_idx = tl.arange(0, D)
    slab = N * N * C
    row_base = outer * N
    output = tl.zeros((N, BN), tl.float32)

    mask_bias = (
        (tl.load(mask_ptr + outer * N + k_idx).to(tl.float32) - 1.0) * inf
    ).to(tl.bfloat16)
    for head in range(0, H):
        q = tl.load(
            projections_ptr
            + (row_base + q_idx[:, None]) * C
            + head * D
            + d_idx[None, :]
        )
        k = tl.load(
            projections_ptr
            + slab
            + (row_base + k_idx[:, None]) * C
            + head * D
            + d_idx[None, :]
        )
        v = tl.load(
            projections_ptr
            + 2 * slab
            + (row_base + k_idx[:, None]) * C
            + head * D
            + d_idx[None, :]
        )
        scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        scores = (scores + mask_bias[None, :]).to(tl.bfloat16)
        z_offsets = (
            4 * slab
            + (q_idx[:, None] * N + k_idx[None, :]) * C
            + head
        )
        scores = (scores + tl.load(projections_ptr + z_offsets)).to(tl.bfloat16)
        scores_f32 = scores.to(tl.float32)
        scores_f32 -= tl.max(scores_f32, axis=1)[:, None]
        numerator = tl.exp(scores_f32)
        probabilities = (
            numerator / tl.sum(numerator, axis=1)[:, None]
        ).to(tl.bfloat16)
        attended = tl.dot(probabilities, v).to(tl.bfloat16)
        gate = tl.load(
            projections_ptr
            + 3 * slab
            + (row_base + q_idx[:, None]) * C
            + head * D
            + d_idx[None, :]
        )
        gate = tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
        attended = (attended * gate).to(tl.bfloat16)
        weight = tl.load(
            output_weight_ptr
            + out_cols[None, :] * C
            + head * D
            + d_idx[:, None],
            mask=out_cols[None, :] < C,
            other=0.0,
        )
        output += tl.dot(attended, weight)

    tl.store(
        output_ptr
        + (row_base + q_idx[:, None]) * C
        + out_cols[None, :],
        output,
        mask=out_cols[None, :] < C,
    )


@triton.jit
def _norm_vg_kernel(
    x_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    wv_ptr,
    wg_ptr,
    projections_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    projection = tl.program_id(1)
    out_cols = tl.program_id(2) * BN + tl.arange(0, BN)
    cols = tl.arange(0, C)
    x = tl.load(
        x_ptr + rows[:, None] * C + cols[None, :],
        mask=rows[:, None] < M,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / C
    normed = centered * tl.rsqrt(variance[:, None] + 1e-5)
    normed *= tl.load(norm_weight_ptr + cols)[None, :]
    normed += tl.load(norm_bias_ptr + cols)[None, :]
    normed = normed.to(tl.bfloat16)
    weight_offsets = out_cols[None, :] * C + cols[:, None]
    if projection == 0:
        weight = tl.load(wv_ptr + weight_offsets)
    else:
        weight = tl.load(wg_ptr + weight_offsets)
    projected = tl.dot(normed, weight)
    tl.store(
        projections_ptr
        + projection * M * C
        + rows[:, None] * C
        + out_cols[None, :],
        projected,
        mask=rows[:, None] < M,
    )


@triton.jit
def _dominant_mask_output_kernel(
    projections_ptr,
    mask_ptr,
    output_weight_ptr,
    output_ptr,
    inf: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    BN: tl.constexpr,
):
    outer = tl.program_id(0)
    out_cols = tl.program_id(1) * BN + tl.arange(0, BN)
    q_idx = tl.arange(0, N)
    k_idx = tl.arange(0, N)
    inner = tl.arange(0, C)

    # Compare the rounded values actually added to the BF16 score tensor.
    mask_bias = (
        (tl.load(mask_ptr + outer * N + k_idx).to(tl.float32) - 1.0) * inf
    ).to(tl.bfloat16)
    selected = tl.argmax(mask_bias, axis=0)
    value = tl.load(
        projections_ptr + (outer * N + selected) * C + inner
    )
    gate = tl.load(
        projections_ptr
        + N * N * C
        + (outer * N + q_idx[:, None]) * C
        + inner[None, :]
    )
    gated = (
        value[None, :] * tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
    ).to(tl.bfloat16)
    weight = tl.load(
        output_weight_ptr
        + out_cols[None, :] * C
        + inner[:, None]
    )
    output = tl.dot(gated, weight)
    tl.store(
        output_ptr
        + (outer * N + q_idx[:, None]) * C
        + out_cols[None, :],
        output,
    )


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class TriangleAttention(nn.Module):
    """AF3 Algorithms 14/15: Triangle attention.

    Args:
        c_in: Input channel dimension
        c_hidden: Overall hidden channel dimension (not per-head)
        no_heads: Number of attention heads
        starting: If True, starting node (Alg 14); else ending node (Alg 15)
        inf: Large constant for masking
    """

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

        self.layer_norm = LayerNorm(c_in)
        self.linear_z = Linear(c_in, no_heads, bias=False)

        self.mha = OF3Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        )

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
        """
        Args:
            x: [*, I, J, C_in] input tensor (pair representation)

        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if (
            self.starting
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.shape == (1, 16, 16, 128)
            and mask.shape == (1, 16, 16)
            and mask.dtype == torch.bfloat16
            and x.is_contiguous()
            and mask.is_contiguous()
            and self.c_in == 128
            and self.c_hidden == 32
            and self.no_heads == 4
            and chunk_size is None
            and not use_deepspeed_evo_attention
            and not use_cueq_triangle_kernels
            and not use_lma
        ):
            projections = torch.empty(
                (5, 256, 128), device=x.device, dtype=x.dtype
            )
            _norm_project_kernel[(16, 5, 4)](
                x,
                self.layer_norm.weight,
                self.layer_norm.bias,
                self.mha.linear_q.weight,
                self.mha.linear_k.weight,
                self.mha.linear_v.weight,
                self.mha.linear_g.weight,
                self.linear_z.weight,
                projections,
                M=256,
                C=128,
                BM=16,
                BN=32,
                num_warps=4,
            )
            output = torch.empty_like(x)
            _attention_gate_output_kernel[(16, 4)](
                projections,
                mask,
                self.mha.linear_o.weight,
                output,
                inf=self.inf,
                N=16,
                H=4,
                D=32,
                C=128,
                BN=32,
                num_warps=4,
            )
            return output

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J] -> [*, 1, H, I, J]
        triangle_bias = _permute_final_dims(self.linear_z(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """AF3 Algorithm 15."""

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9):
        super().__init__(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads,
                         starting=False, inf=inf)
