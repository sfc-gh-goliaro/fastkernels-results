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
def _normalized_rows(x_ptr, w_ptr, b_ptr, rows, cols, K: tl.constexpr):
    x = tl.load(x_ptr + rows[:, None] * K + cols[None, :]).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    centered = x - mean[:, None]
    var = tl.sum(centered * centered, axis=1) / K
    x = centered * tl.rsqrt(var[:, None] + 1e-5)
    w = tl.load(w_ptr + cols).to(tl.float32)
    b = tl.load(b_ptr + cols).to(tl.float32)
    return (x * w[None, :] + b[None, :]).to(tl.bfloat16)


@triton.jit
def _linear_block(x, weight_ptr, out_cols, K: tl.constexpr):
    k = tl.arange(0, K)
    weight = tl.load(
        weight_ptr + out_cols[None, :] * K + k[:, None]
    )
    return tl.dot(x, weight)


@triton.jit
def _triangle_project_kernel(
    z_ptr, mask_ptr, ln_w_ptr, ln_b_ptr,
    ap_w_ptr, ag_w_ptr, bp_w_ptr, bg_w_ptr, g_w_ptr,
    a_ptr, b_ptr, g_ptr,
    M: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, K)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = _normalized_rows(z_ptr, ln_w_ptr, ln_b_ptr, rows, cols, K)
    ap = _linear_block(x, ap_w_ptr, out_cols, K).to(tl.bfloat16)
    ag = _linear_block(x, ag_w_ptr, out_cols, K).to(tl.bfloat16)
    bp = _linear_block(x, bp_w_ptr, out_cols, K).to(tl.bfloat16)
    bg = _linear_block(x, bg_w_ptr, out_cols, K).to(tl.bfloat16)
    gate = _linear_block(x, g_w_ptr, out_cols, K).to(tl.bfloat16)
    mask = tl.load(mask_ptr + rows).to(tl.float32)
    offsets = rows[:, None] * K + out_cols[None, :]
    tl.store(a_ptr + offsets, ap * tl.sigmoid(ag.to(tl.float32)) * mask[:, None])
    tl.store(b_ptr + offsets, bp * tl.sigmoid(bg.to(tl.float32)) * mask[:, None])
    tl.store(g_ptr + offsets, gate)


@triton.jit
def _triangle_contract_kernel(
    a_ptr, b_ptr, out_ptr,
    INCOMING: tl.constexpr,
    H: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pairs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    hs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    js = tl.arange(0, 16)
    i = pairs // 16
    k = pairs % 16
    if INCOMING:
        a_offsets = ((js[None, :, None] * 16 + i[:, None, None]) * H
                     + hs[None, None, :])
        b_offsets = ((js[None, :, None] * 16 + k[:, None, None]) * H
                     + hs[None, None, :])
    else:
        a_offsets = ((i[:, None, None] * 16 + js[None, :, None]) * H
                     + hs[None, None, :])
        b_offsets = ((k[:, None, None] * 16 + js[None, :, None]) * H
                     + hs[None, None, :])
    a = tl.load(a_ptr + a_offsets).to(tl.float32)
    b = tl.load(b_ptr + b_offsets).to(tl.float32)
    value = tl.sum(a * b, axis=1)
    tl.store(out_ptr + pairs[:, None] * H + hs[None, :], value)


@triton.jit
def _triangle_finish_kernel(
    z_ptr, product_ptr, gate_ptr, ln_w_ptr, ln_b_ptr, out_w_ptr,
    out_ptr,
    M: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, K)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = _normalized_rows(product_ptr, ln_w_ptr, ln_b_ptr, rows, cols, K)
    update = _linear_block(x, out_w_ptr, out_cols, K).to(tl.bfloat16)
    offsets = rows[:, None] * K + out_cols[None, :]
    gate = tl.load(gate_ptr + offsets)
    z = tl.load(z_ptr + offsets)
    update = (update * tl.sigmoid(gate.to(tl.float32))).to(tl.bfloat16)
    tl.store(out_ptr + offsets, z + update)


@triton.jit
def _attention_project_kernel(
    z_ptr, ln_w_ptr, ln_b_ptr,
    q_w_ptr, k_w_ptr, v_w_ptr, g_w_ptr, bias_w_ptr,
    q_ptr, k_ptr, v_ptr, g_ptr, bias_ptr,
    TRANSPOSE: tl.constexpr,
    M: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    logical_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if TRANSPOSE:
        src_rows = (logical_rows % 16) * 16 + logical_rows // 16
    else:
        src_rows = logical_rows
    cols = tl.arange(0, K)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = _normalized_rows(z_ptr, ln_w_ptr, ln_b_ptr, src_rows, cols, K)
    q = (_linear_block(x, q_w_ptr, out_cols, K)
         * 0.1767766952966369).to(tl.bfloat16)
    k = _linear_block(x, k_w_ptr, out_cols, K).to(tl.bfloat16)
    v = _linear_block(x, v_w_ptr, out_cols, K).to(tl.bfloat16)
    g = _linear_block(x, g_w_ptr, out_cols, K).to(tl.bfloat16)
    offsets = logical_rows[:, None] * K + out_cols[None, :]
    tl.store(q_ptr + offsets, q)
    tl.store(k_ptr + offsets, k)
    tl.store(v_ptr + offsets, v)
    tl.store(g_ptr + offsets, g)

    if pid_n == 0:
        bias_cols = tl.arange(0, 16)
        bias = _linear_block(x, bias_w_ptr, bias_cols, K)
        bias_offsets = logical_rows[:, None] * 4 + bias_cols[None, :]
        tl.store(
            bias_ptr + bias_offsets, bias,
            mask=bias_cols[None, :] < 4,
        )


@triton.jit
def _attention_output_kernel(
    z_ptr, mask_ptr, q_ptr, k_ptr, v_ptr, g_ptr, bias_ptr, out_w_ptr,
    out_ptr,
    TRANSPOSE: tl.constexpr,
    K_TOTAL: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    qs = tl.arange(0, 16)
    ks = tl.arange(0, 16)
    ds = tl.arange(0, HEAD_DIM)
    acc = tl.zeros((16, BLOCK_N), tl.float32)

    for head in tl.static_range(0, 4):
        q_offsets = ((i * 16 + qs[:, None]) * K_TOTAL
                     + head * HEAD_DIM + ds[None, :])
        k_offsets = ((i * 16 + ks[None, :]) * K_TOTAL
                     + head * HEAD_DIM + ds[:, None])
        q = tl.load(q_ptr + q_offsets)
        key = tl.load(k_ptr + k_offsets)
        scores = tl.dot(q, key)
        logical_k = i * 16 + ks
        if TRANSPOSE:
            mask_rows = ks * 16 + i
        else:
            mask_rows = logical_k
        mask = tl.load(mask_ptr + mask_rows).to(tl.float32)
        bias = tl.load(bias_ptr + logical_k * 4 + head).to(tl.float32)
        scores += (1.0e9 * (mask[None, :] - 1.0) + bias[None, :])
        scores -= tl.max(scores, axis=1)[:, None]
        probs = tl.exp(scores)
        probs /= tl.sum(probs, axis=1)[:, None]
        v_offsets = ((i * 16 + ks[:, None]) * K_TOTAL
                     + head * HEAD_DIM + ds[None, :])
        value = tl.load(v_ptr + v_offsets)
        attended = tl.dot(probs.to(tl.bfloat16), value)
        gate = tl.load(g_ptr + q_offsets)
        attended = (
            attended.to(tl.bfloat16) * tl.sigmoid(gate.to(tl.float32))
        ).to(tl.bfloat16)
        w_offsets = (
            out_cols[None, :] * K_TOTAL
            + head * HEAD_DIM + ds[:, None]
        )
        weight = tl.load(out_w_ptr + w_offsets)
        acc += tl.dot(attended, weight)

    logical_rows = i * 16 + qs
    if TRANSPOSE:
        dst_rows = qs * 16 + i
    else:
        dst_rows = logical_rows
    offsets = dst_rows[:, None] * K_TOTAL + out_cols[None, :]
    update = acc.to(tl.bfloat16)
    z = tl.load(z_ptr + offsets)
    tl.store(out_ptr + offsets, z + update)


@triton.jit
def _transition_project_kernel(
    z_ptr, ln_w_ptr, ln_b_ptr, a_w_ptr, b_w_ptr, hidden_ptr,
    M: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, K)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = _normalized_rows(z_ptr, ln_w_ptr, ln_b_ptr, rows, cols, K)
    a = _linear_block(x, a_w_ptr, out_cols, K).to(tl.bfloat16)
    b = _linear_block(x, b_w_ptr, out_cols, K).to(tl.bfloat16)
    hidden = (a * tl.sigmoid(a.to(tl.float32)) * b).to(tl.bfloat16)
    tl.store(hidden_ptr + rows[:, None] * (4 * K) + out_cols[None, :], hidden)


@triton.jit
def _transition_finish_kernel(
    z_ptr, mask_ptr, hidden_ptr, out_w_ptr, out_ptr,
    M: tl.constexpr, K: tl.constexpr, HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_cols = tl.arange(0, HIDDEN)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    hidden = tl.load(hidden_ptr + rows[:, None] * HIDDEN + hidden_cols[None, :])
    weight = tl.load(
        out_w_ptr + out_cols[None, :] * HIDDEN + hidden_cols[:, None]
    )
    update = tl.dot(hidden, weight).to(tl.bfloat16)
    offsets = rows[:, None] * K + out_cols[None, :]
    mask = tl.load(mask_ptr + rows).to(tl.float32)
    z = tl.load(z_ptr + offsets)
    tl.store(out_ptr + offsets, z + update * mask[:, None])


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
        if (
            z.is_cuda
            and z.dtype == torch.bfloat16
            and z.shape == (1, 16, 16, 128)
            and pair_mask.shape == (1, 16, 16)
            and self.tri_mul_out.c_hidden == 128
            and self.tri_att_start.c_hidden == 32
            and self.tri_att_start.no_heads == 4
            and self.pair_transition.n == 4
            and _mask_trans
        ):
            return self._forward_16(z, pair_mask)

        pair_trans_mask = pair_mask if _mask_trans else None

        # Triangle multiplicative updates
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)

        # Triangle attention (start)
        z = z + self.tri_att_start(z, mask=pair_mask)

        # Triangle attention (end) -- uses transposed mask internally
        z = z + self.tri_att_end(z, mask=pair_mask)

        # Pair transition
        z = z + self.pair_transition(z, mask=pair_trans_mask)

        return z

    def _triangle_update(self, z, pair_mask, module, incoming):
        a = torch.empty_like(z)
        b = torch.empty_like(z)
        gate = torch.empty_like(z)
        product = torch.empty_like(z)
        out = torch.empty_like(z)
        _triangle_project_kernel[(16, 4)](
            z, pair_mask,
            module.layer_norm_in.weight, module.layer_norm_in.bias,
            module.linear_a_p.weight, module.linear_a_g.weight,
            module.linear_b_p.weight, module.linear_b_g.weight,
            module.linear_g.weight,
            a, b, gate,
            M=256, K=128, BLOCK_M=16, BLOCK_N=32,
            num_warps=4,
        )
        _triangle_contract_kernel[(64, 4)](
            a, b, product,
            INCOMING=incoming, H=128, BLOCK_P=4, BLOCK_H=32,
            num_warps=4,
        )
        _triangle_finish_kernel[(16, 4)](
            z, product, gate,
            module.layer_norm_out.weight, module.layer_norm_out.bias,
            module.linear_z.weight, out,
            M=256, K=128, BLOCK_M=16, BLOCK_N=32,
            num_warps=4,
        )
        return out

    def _attention_update(self, z, pair_mask, module, transpose):
        q = torch.empty_like(z)
        k = torch.empty_like(z)
        v = torch.empty_like(z)
        gate = torch.empty_like(z)
        bias = torch.empty((256, 4), dtype=z.dtype, device=z.device)
        out = torch.empty_like(z)
        _attention_project_kernel[(16, 4)](
            z, module.layer_norm.weight, module.layer_norm.bias,
            module.mha.linear_q.weight, module.mha.linear_k.weight,
            module.mha.linear_v.weight, module.mha.linear_g.weight,
            module.linear_z.weight,
            q, k, v, gate, bias,
            TRANSPOSE=transpose,
            M=256, K=128, BLOCK_M=16, BLOCK_N=32,
            num_warps=4,
        )
        _attention_output_kernel[(16, 4)](
            z, pair_mask, q, k, v, gate, bias,
            module.mha.linear_o.weight, out,
            TRANSPOSE=transpose, K_TOTAL=128, HEAD_DIM=32, BLOCK_N=32,
            num_warps=4,
        )
        return out

    def _forward_16(self, z, pair_mask):
        z = self._triangle_update(z, pair_mask, self.tri_mul_out, False)
        z = self._triangle_update(z, pair_mask, self.tri_mul_in, True)
        z = self._attention_update(z, pair_mask, self.tri_att_start, False)
        z = self._attention_update(z, pair_mask, self.tri_att_end, True)

        hidden = torch.empty(
            (256, 512), dtype=z.dtype, device=z.device,
        )
        out = torch.empty_like(z)
        transition = self.pair_transition
        _transition_project_kernel[(16, 8)](
            z, transition.layer_norm.weight, transition.layer_norm.bias,
            transition.swiglu.linear_a.weight,
            transition.swiglu.linear_b.weight,
            hidden,
            M=256, K=128, BLOCK_M=16, BLOCK_N=64,
            num_warps=4,
        )
        _transition_finish_kernel[(16, 4)](
            z, pair_mask, hidden, transition.linear_out.weight, out,
            M=256, K=128, HIDDEN=512, BLOCK_M=16, BLOCK_N=32,
            num_warps=4,
        )
        return out
