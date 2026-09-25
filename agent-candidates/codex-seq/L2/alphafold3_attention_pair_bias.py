"""Attention with pair bias for AlphaFold3.

AttentionPairBias: Used in PairFormer and diffusion transformer. Uses a single
    layer_norm_a for both Q and K (AdaLN or LayerNorm).
CrossAttentionPairBias: Used in atom attention (sequence-local). Uses separate
    layer_norm_a_q and layer_norm_a_k, no layer_norm_z.

Reference: openfold3/core/model/layers/attention_pair_bias.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN
from .alphafold3_of3_attention import OF3Attention


def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _attention_from_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
    c_hidden: int,
) -> torch.Tensor:
    q = q / (c_hidden ** 0.5)
    scores = torch.einsum("...qc,...kc->...qk", q, k)
    for bias in biases:
        scores = scores + bias
    probs = F.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("...qk,...kc->...qc", probs, v)


@triton.jit
def _cross_gather_norm_kernel(
    a_ptr,
    s_ptr,
    aq_ptr,
    sq_ptr,
    ak_ptr,
    sk_ptr,
    sq_weight_ptr,
    sk_weight_ptr,
    C: tl.constexpr,
    N_ATOM: tl.constexpr,
    N_QUERY: tl.constexpr,
    N_KEY: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, C)

    block = row // N_KEY
    key_col = row % N_KEY
    first = block * N_QUERY - (N_KEY - N_QUERY) // 2
    shift_left = tl.maximum(-first, 0)
    shift_right = tl.maximum(first + N_KEY - N_ATOM, 0)
    shift = tl.where(shift_left > 0, shift_left, -shift_right)
    key_idx = first + shift + key_col

    a_k = tl.load(a_ptr + key_idx * C + cols).to(tl.float32)
    s_k = tl.load(s_ptr + key_idx * C + cols).to(tl.float32)
    a_k_mean = tl.sum(a_k, axis=0) / C
    s_k_mean = tl.sum(s_k, axis=0) / C
    a_k = a_k - a_k_mean
    s_k = s_k - s_k_mean
    a_k *= tl.rsqrt(tl.sum(a_k * a_k, axis=0) / C + EPS)
    s_k *= tl.rsqrt(tl.sum(s_k * s_k, axis=0) / C + EPS)
    s_k *= tl.load(sk_weight_ptr + cols).to(tl.float32)
    tl.store(ak_ptr + row * C + cols, a_k)
    tl.store(sk_ptr + row * C + cols, s_k)

    if row < ((N_ATOM + N_QUERY - 1) // N_QUERY) * N_QUERY:
        query_idx = row
        valid = query_idx < N_ATOM
        a_q = tl.load(
            a_ptr + query_idx * C + cols, mask=valid, other=0.0,
        ).to(tl.float32)
        s_q = tl.load(
            s_ptr + query_idx * C + cols, mask=valid, other=0.0,
        ).to(tl.float32)
        a_q_mean = tl.sum(a_q, axis=0) / C
        s_q_mean = tl.sum(s_q, axis=0) / C
        a_q = a_q - a_q_mean
        s_q = s_q - s_q_mean
        a_q *= tl.rsqrt(tl.sum(a_q * a_q, axis=0) / C + EPS)
        s_q *= tl.rsqrt(tl.sum(s_q * s_q, axis=0) / C + EPS)
        s_q *= tl.load(sq_weight_ptr + cols).to(tl.float32)
        tl.store(aq_ptr + row * C + cols, a_q)
        tl.store(sq_ptr + row * C + cols, s_q)


@triton.jit
def _cross_attention_kernel(
    qg_ptr,
    kv_ptr,
    z_ptr,
    out_ptr,
    N_QUERY: tl.constexpr,
    N_KEY: tl.constexpr,
    C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_HEAD: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    q_rows = tl.arange(0, N_QUERY)
    k_rows = tl.arange(0, N_KEY)
    dims = tl.arange(0, HEAD_DIM)

    q_base = block * N_QUERY * (2 * C)
    q_offsets = q_base + q_rows[:, None] * (2 * C) + head * HEAD_DIM + dims[None, :]
    q = tl.load(qg_ptr + q_offsets)
    q = (q * (HEAD_DIM ** -0.5)).to(q.dtype)

    kv_base = block * N_KEY * (2 * C)
    k_offsets = kv_base + k_rows[None, :] * (2 * C) + head * HEAD_DIM + dims[:, None]
    k = tl.load(kv_ptr + k_offsets)
    scores = tl.dot(q, k)
    z_offsets = (
        block * N_QUERY * N_KEY * N_HEAD
        + q_rows[:, None] * N_KEY * N_HEAD
        + k_rows[None, :] * N_HEAD
        + head
    )
    scores += tl.load(z_ptr + z_offsets)
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs /= tl.sum(probs, axis=1)[:, None]

    v_offsets = (
        kv_base + k_rows[:, None] * (2 * C)
        + C + head * HEAD_DIM + dims[None, :]
    )
    v = tl.load(kv_ptr + v_offsets)
    acc = tl.dot(probs.to(v.dtype), v)

    gate_offsets = q_offsets + C
    gate = tl.load(qg_ptr + gate_offsets).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    out_offsets = (
        block * N_QUERY * C
        + q_rows[:, None] * C + head * HEAD_DIM + dims[None, :]
    )
    tl.store(out_ptr + out_offsets, acc * gate)


@triton.jit
def _cross_pair_kernel(
    z_ptr,
    weight_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_HEAD: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    z_cols = tl.arange(0, 16)
    heads = tl.arange(0, 16)
    z = tl.load(
        z_ptr + rows[:, None] * 16 + z_cols[None, :],
        mask=rows[:, None] < N_ROWS,
        other=0.0,
    )
    weight = tl.load(
        weight_ptr + heads[None, :] * 16 + z_cols[:, None],
        mask=heads[None, :] < N_HEAD,
        other=0.0,
    )
    projected = tl.dot(z, weight)
    tl.store(
        out_ptr + rows[:, None] * N_HEAD + heads[None, :],
        projected,
        mask=(rows[:, None] < N_ROWS) & (heads[None, :] < N_HEAD),
    )


@triton.jit
def _pair_bias_kernel_v2(
    z_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    linear_weight_ptr,
    out_ptr,
    C_Z: tl.constexpr,
    N_HEAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, C_Z)
    heads = tl.arange(0, N_HEAD)
    z = tl.load(z_ptr + row * C_Z + cols).to(tl.float32)
    mean = tl.sum(z, axis=0) / C_Z
    z -= mean
    z *= tl.rsqrt(tl.sum(z * z, axis=0) / C_Z + EPS)
    z *= tl.load(norm_weight_ptr + cols).to(tl.float32)
    if HAS_BIAS:
        z += tl.load(norm_bias_ptr + cols).to(tl.float32)
    weights = tl.load(
        linear_weight_ptr + heads[:, None] * C_Z + cols[None, :]
    )
    projected = tl.sum(z[None, :].to(weights.dtype) * weights, axis=1)
    tl.store(out_ptr + row * N_HEAD + heads, projected)


@triton.jit
def _self_ada_norm_kernel(
    a_ptr,
    s_ptr,
    a_norm_ptr,
    s_norm_ptr,
    s_weight_ptr,
    C_A: tl.constexpr,
    C_S: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    a_cols = tl.arange(0, BLOCK_A)
    a_mask = a_cols < C_A
    a = tl.load(
        a_ptr + row * C_A + a_cols, mask=a_mask, other=0.0,
    ).to(tl.float32)
    a_mean = tl.sum(a, axis=0) / C_A
    a_centered = tl.where(a_mask, a - a_mean, 0.0)
    a_norm = a_centered * tl.rsqrt(
        tl.sum(a_centered * a_centered, axis=0) / C_A + EPS
    )
    tl.store(a_norm_ptr + row * C_A + a_cols, a_norm, mask=a_mask)

    s_cols = tl.arange(0, BLOCK_S)
    s_mask = s_cols < C_S
    s = tl.load(
        s_ptr + row * C_S + s_cols, mask=s_mask, other=0.0,
    ).to(tl.float32)
    s_mean = tl.sum(s, axis=0) / C_S
    s_centered = tl.where(s_mask, s - s_mean, 0.0)
    s_norm = s_centered * tl.rsqrt(
        tl.sum(s_centered * s_centered, axis=0) / C_S + EPS
    )
    s_norm *= tl.load(s_weight_ptr + s_cols, mask=s_mask)
    tl.store(s_norm_ptr + row * C_S + s_cols, s_norm, mask=s_mask)


@triton.jit
def _self_ada_projection_kernel(
    a_norm_ptr,
    ada_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    OUT_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    out_cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, C, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        gate_linear = tl.load(
            ada_ptr + rows[:, None] * (2 * C) + k[None, :]
        ).to(tl.float32)
        shift = tl.load(
            ada_ptr + rows[:, None] * (2 * C) + C + k[None, :]
        ).to(tl.float32)
        a_norm = tl.load(
            a_norm_ptr + rows[:, None] * C + k[None, :]
        )
        gate = 1.0 / (1.0 + tl.exp(-gate_linear))
        x = (gate * (a_norm + shift)).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + out_cols[None, :] * C + k[:, None],
            mask=out_cols[None, :] < OUT_C,
            other=0.0,
        )
        acc += tl.dot(x, weight)
    acc += tl.load(
        bias_ptr + out_cols[None, :],
        mask=out_cols[None, :] < OUT_C,
        other=0.0,
    )
    tl.store(
        out_ptr + rows[:, None] * OUT_C + out_cols[None, :],
        acc,
        mask=out_cols[None, :] < OUT_C,
    )


@triton.jit
def _ada_projection_kernel(
    a_norm_ptr,
    ada_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    OUT_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, C, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        gate_linear = tl.load(
            ada_ptr + rows[:, None] * (2 * C) + k[None, :],
            mask=rows[:, None] < M,
            other=0.0,
        ).to(tl.float32)
        shift = tl.load(
            ada_ptr + rows[:, None] * (2 * C) + C + k[None, :],
            mask=rows[:, None] < M,
            other=0.0,
        ).to(tl.float32)
        a_norm = tl.load(
            a_norm_ptr + rows[:, None] * C + k[None, :],
            mask=rows[:, None] < M,
            other=0.0,
        )
        gate = 1.0 / (1.0 + tl.exp(-gate_linear))
        x = (gate * (a_norm + shift)).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + out_cols[None, :] * C + k[:, None],
            mask=out_cols[None, :] < OUT_C,
            other=0.0,
        )
        acc += tl.dot(x, weight)
    if HAS_BIAS:
        acc += tl.load(
            bias_ptr + out_cols[None, :],
            mask=out_cols[None, :] < OUT_C,
            other=0.0,
        )
    tl.store(
        out_ptr + rows[:, None] * OUT_C + out_cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (out_cols[None, :] < OUT_C),
    )


@triton.jit
def _self_attention_kernel(
    qkvg_ptr,
    pair_ptr,
    out_ptr,
    C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    N_HEAD: tl.constexpr,
    N_TOKEN: tl.constexpr,
):
    head = tl.program_id(0)
    rows = tl.arange(0, N_TOKEN)
    dims = tl.arange(0, HEAD_BLOCK)
    dim_mask = dims < HEAD_DIM

    q_offsets = (
        rows[:, None] * (4 * C) + head * HEAD_DIM + dims[None, :]
    )
    q = tl.load(qkvg_ptr + q_offsets, mask=dim_mask[None, :], other=0.0)
    q = (q * (HEAD_DIM ** -0.5)).to(q.dtype)
    k_offsets = (
        rows[None, :] * (4 * C) + C
        + head * HEAD_DIM + dims[:, None]
    )
    k = tl.load(qkvg_ptr + k_offsets, mask=dim_mask[:, None], other=0.0)
    scores = tl.dot(q, k)
    pair_offsets = (
        rows[:, None] * N_TOKEN * N_HEAD
        + rows[None, :] * N_HEAD + head
    )
    scores += tl.load(pair_ptr + pair_offsets)
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs /= tl.sum(probs, axis=1)[:, None]

    v_offsets = (
        rows[:, None] * (4 * C) + 2 * C
        + head * HEAD_DIM + dims[None, :]
    )
    v = tl.load(qkvg_ptr + v_offsets, mask=dim_mask[None, :], other=0.0)
    acc = tl.dot(probs.to(v.dtype), v)
    gate = tl.load(
        qkvg_ptr + v_offsets + C, mask=dim_mask[None, :], other=0.0,
    ).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    out_offsets = rows[:, None] * C + head * HEAD_DIM + dims[None, :]
    tl.store(out_ptr + out_offsets, acc * gate, mask=dim_mask[None, :])


@triton.jit
def _gated_output_kernel(
    attended_ptr,
    s_ptr,
    out_weight_ptr,
    gate_weight_ptr,
    gate_bias_ptr,
    out_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    C_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, C, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            attended_ptr + rows[:, None] * C + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < C),
            other=0.0,
        )
        w = tl.load(
            out_weight_ptr + cols[None, :] * C + k[:, None],
            mask=(cols[None, :] < C) & (k[:, None] < C),
            other=0.0,
        )
        acc += tl.dot(x, w)

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, C_S, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            s_ptr + rows[:, None] * C_S + k[None, :],
            mask=(rows[:, None] < M) & (k[None, :] < C_S),
            other=0.0,
        )
        w = tl.load(
            gate_weight_ptr + cols[None, :] * C_S + k[:, None],
            mask=(cols[None, :] < C) & (k[:, None] < C_S),
            other=0.0,
        )
        gate_acc += tl.dot(x, w)
    gate_acc += tl.load(
        gate_bias_ptr + cols[None, :],
        mask=cols[None, :] < C,
        other=0.0,
    )

    projected = acc.to(tl.bfloat16)
    gate_linear = gate_acc.to(tl.bfloat16).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-gate_linear))).to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * C + cols[None, :],
        projected * gate,
        mask=(rows[:, None] < M) & (cols[None, :] < C),
    )


class AttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Attention with pair bias.

    When use_ada_layer_norm is True, uses two separate AdaLN instances
    (layer_norm_a_q, layer_norm_a_k) for query and key normalization,
    plus a linear_ada_out for output gating.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )
        self._qkvg_weight = None
        self._qkvg_bias = None
        self._ada_weight = None
        self._ada_bias = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        linears = [
            self.mha.linear_q, self.mha.linear_k,
            self.mha.linear_v, self.mha.linear_g,
        ]
        self._qkvg_weight = torch.cat([linear.weight for linear in linears])
        zero = torch.zeros_like(self.mha.linear_q.bias)
        self._qkvg_bias = torch.cat([
            self.mha.linear_q.bias, zero, zero, zero,
        ])
        if self.use_ada_layer_norm:
            self._ada_weight = torch.cat([
                self.layer_norm_a.linear_g.weight,
                self.layer_norm_a.linear_s.weight,
            ])
            self._ada_bias = torch.cat([
                self.layer_norm_a.linear_g.bias,
                torch.zeros_like(self.layer_norm_a.linear_g.bias),
            ])
        return result

    def _forward_fixed(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None,
    ) -> torch.Tensor:
        n_token = 16
        n_head = self.mha.no_heads
        if self.use_ada_layer_norm:
            a_norm = torch.empty(
                (n_token, self.c_q), device=a.device, dtype=a.dtype,
            )
            s_norm = torch.empty(
                (n_token, self.c_s), device=a.device, dtype=a.dtype,
            )
            _self_ada_norm_kernel[(n_token,)](
                a,
                s,
                a_norm,
                s_norm,
                self.layer_norm_a.layer_norm_s.weight,
                C_A=self.c_q,
                C_S=self.c_s,
                BLOCK_A=triton.next_power_of_2(self.c_q),
                BLOCK_S=triton.next_power_of_2(self.c_s),
                EPS=1e-5,
                num_warps=4,
            )
            ada = F.linear(s_norm, self._ada_weight, self._ada_bias)
            qkvg = torch.empty(
                (n_token, 4 * self.c_q), device=a.device, dtype=a.dtype,
            )
            _self_ada_projection_kernel[(
                triton.cdiv(4 * self.c_q, 64),
            )](
                a_norm,
                ada,
                self._qkvg_weight,
                self._qkvg_bias,
                qkvg,
                M=n_token,
                C=self.c_q,
                OUT_C=4 * self.c_q,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=32,
                num_warps=4,
            )
        else:
            normalized = self.layer_norm_a(a).reshape(n_token, self.c_q)
            qkvg = F.linear(normalized, self._qkvg_weight, self._qkvg_bias)
        pair = torch.empty(
            (n_token * n_token, n_head), device=z.device, dtype=z.dtype,
        )
        norm_bias = (
            self.layer_norm_z.bias
            if self.layer_norm_z.bias is not None
            else self.layer_norm_z.weight
        )
        _pair_bias_kernel_v2[(n_token * n_token,)](
            z,
            self.layer_norm_z.weight,
            norm_bias,
            self.linear_z.weight,
            pair,
            C_Z=self.c_z,
            N_HEAD=n_head,
            HAS_BIAS=self.layer_norm_z.bias is not None,
            EPS=self.layer_norm_z.eps,
            num_warps=4,
        )
        attended = torch.empty(
            (n_token, self.c_q), device=a.device, dtype=a.dtype,
        )
        _self_attention_kernel[(n_head,)](
            qkvg,
            pair,
            attended,
            C=self.c_q,
            HEAD_DIM=self.mha.c_hidden,
            HEAD_BLOCK=triton.next_power_of_2(self.mha.c_hidden),
            N_HEAD=n_head,
            N_TOKEN=n_token,
            num_warps=4,
        )
        if not self.use_ada_layer_norm:
            projected = F.linear(attended, self.mha.linear_o.weight)
            return projected.view_as(a)

        out = torch.empty_like(a)
        _gated_output_kernel[(
            triton.cdiv(n_token, 16),
            triton.cdiv(self.c_q, 32),
        )](
            attended,
            s,
            self.mha.linear_o.weight,
            self.linear_ada_out.weight,
            self.linear_ada_out.bias,
            out,
            M=n_token,
            C=self.c_q,
            C_S=self.c_s,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
        )
        return out

    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = _permute_final_dims(z, [2, 0, 1])
        biases.append(z)

        return biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        if (
            a.is_cuda
            and a.shape[-2] == 16
            and a.numel() == 16 * self.c_q
            and z.numel() == 16 * 16 * self.c_z
            and self.mha.no_heads == 16
        ):
            return self._forward_fixed(a, z, s)

        biases = self._prep_bias(a=a, z=z, mask=mask)

        if self.use_ada_layer_norm:
            s_norm = self.layer_norm_a.layer_norm_s(s)
            ada = F.linear(s_norm, self._ada_weight, self._ada_bias)
            ada_g, ada_s = ada.split(self.c_q, dim=-1)
            a_norm = self.layer_norm_a.layer_norm_a(a)
            a = self.sigmoid(ada_g) * (a_norm + ada_s)
        else:
            a = self.layer_norm_a(a)

        qkvg = F.linear(a, self._qkvg_weight, self._qkvg_bias)
        q, k, v, g = qkvg.split(self.c_q, dim=-1)
        head_shape = q.shape[:-1] + (self.mha.no_heads, self.mha.c_hidden)
        q = q.view(head_shape).transpose(-2, -3)
        k = k.view(head_shape).transpose(-2, -3)
        v = v.view(head_shape).transpose(-2, -3)
        a = _attention_from_qkv(q, k, v, biases, self.mha.c_hidden)
        g = torch.sigmoid(g).view(head_shape)
        a = (a.transpose(-2, -3) * g).reshape(g.shape[:-2] + (-1,))
        a = F.linear(a, self.mha.linear_o.weight)

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a


class CrossAttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Uses separate layer_norm_a_q and layer_norm_a_k for query/key, and
    does NOT apply layer_norm_z (pair bias goes through linear_z directly).
    Handles sequence-local blocked inputs.

    Reference: openfold3/core/model/layers/attention_pair_bias.py CrossAttentionPairBias

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )
        self._qg_weight = None
        self._qg_bias = None
        self._kv_weight = None
        self._ada_q_weight = None
        self._ada_q_bias = None
        self._ada_k_weight = None
        self._ada_k_bias = None
        self._block_indices = {}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        zero = torch.zeros_like(self.mha.linear_q.bias)
        self._qg_weight = torch.cat([
            self.mha.linear_q.weight, self.mha.linear_g.weight,
        ])
        self._qg_bias = torch.cat([self.mha.linear_q.bias, zero])
        self._kv_weight = torch.cat([
            self.mha.linear_k.weight, self.mha.linear_v.weight,
        ])
        if self.use_ada_layer_norm:
            q = self.layer_norm_a_q
            k = self.layer_norm_a_k
            self._ada_q_weight = torch.cat([q.linear_g.weight, q.linear_s.weight])
            self._ada_q_bias = torch.cat([
                q.linear_g.bias, torch.zeros_like(q.linear_g.bias),
            ])
            self._ada_k_weight = torch.cat([k.linear_g.weight, k.linear_s.weight])
            self._ada_k_bias = torch.cat([
                k.linear_g.bias, torch.zeros_like(k.linear_g.bias),
            ])
        return result

    def _forward_fixed(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        c = 128
        n_atom = 368
        n_blocks = 12
        n_query = 32
        n_key = 128
        q_rows = n_blocks * n_query
        k_rows = n_blocks * n_key

        aq_norm = torch.empty((q_rows, c), device=a.device, dtype=a.dtype)
        sq_norm = torch.empty_like(aq_norm)
        ak_norm = torch.empty((k_rows, c), device=a.device, dtype=a.dtype)
        sk_norm = torch.empty_like(ak_norm)
        _cross_gather_norm_kernel[(k_rows,)](
            a,
            s,
            aq_norm,
            sq_norm,
            ak_norm,
            sk_norm,
            self.layer_norm_a_q.layer_norm_s.weight,
            self.layer_norm_a_k.layer_norm_s.weight,
            C=c,
            N_ATOM=n_atom,
            N_QUERY=n_query,
            N_KEY=n_key,
            EPS=1e-5,
            num_warps=4,
        )

        q_ada = F.linear(sq_norm, self._ada_q_weight, self._ada_q_bias)
        k_ada = F.linear(sk_norm, self._ada_k_weight, self._ada_k_bias)
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
            self._qg_weight,
            self._qg_bias,
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
            self._kv_weight,
            self._qg_bias,
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
        z_rows = n_blocks * n_query * n_key
        z_bias = torch.empty(
            (z_rows, self.mha.no_heads), device=z.device, dtype=z.dtype,
        )
        _cross_pair_kernel[(triton.cdiv(z_rows, 32),)](
            z,
            self.linear_z.weight,
            z_bias,
            N_ROWS=z_rows,
            N_HEAD=self.mha.no_heads,
            BLOCK_ROWS=32,
            num_warps=4,
        )
        attended = torch.empty(
            (q_rows, c), device=a.device, dtype=a.dtype,
        )
        _cross_attention_kernel[(n_blocks, self.mha.no_heads)](
            qg,
            kv,
            z_bias,
            attended,
            N_QUERY=n_query,
            N_KEY=n_key,
            C=c,
            HEAD_DIM=self.mha.c_hidden,
            N_HEAD=self.mha.no_heads,
            num_warps=8,
            num_stages=3,
        )

        out = torch.empty_like(a)
        _gated_output_kernel[(
            triton.cdiv(n_atom, 32),
            triton.cdiv(c, 32),
        )](
            attended,
            s,
            self.mha.linear_o.weight,
            self.linear_ada_out.weight,
            self.linear_ada_out.bias,
            out,
            M=n_atom,
            C=c,
            C_S=c,
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
        )
        return out

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """
        if (
            a.is_cuda
            and self.use_ada_layer_norm
            and self.c_q == 128
            and self.n_query == 32
            and self.n_key == 128
            and a.numel() == 368 * 128
            and z.numel() == 12 * 32 * 128 * 16
            and s is not None
        ):
            return self._forward_fixed(a, z, s)

        from .alphafold3_atom_attention import _convert_single_rep_to_blocks, _apply_block_indices

        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        a_query, a_key, block_mask = _convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
        )

        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        z_bias = self.linear_z(z)
        z_bias = _permute_final_dims(z_bias, [2, 0, 1])
        biases.append(z_bias)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = _apply_block_indices(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
            )
            s_q_norm = self.layer_norm_a_q.layer_norm_s(s_q)
            q_ada = F.linear(s_q_norm, self._ada_q_weight, self._ada_q_bias)
            q_g, q_s = q_ada.split(self.c_q, dim=-1)
            a_q = self.sigmoid(q_g) * (
                self.layer_norm_a_q.layer_norm_a(a_query) + q_s
            )
            s_k_norm = self.layer_norm_a_k.layer_norm_s(s_k)
            k_ada = F.linear(s_k_norm, self._ada_k_weight, self._ada_k_bias)
            k_g, k_s = k_ada.split(self.c_q, dim=-1)
            a_k = self.sigmoid(k_g) * (
                self.layer_norm_a_k.layer_norm_a(a_key) + k_s
            )
        else:
            a_q = self.layer_norm_a_q(a_query)
            a_k = self.layer_norm_a_k(a_key)

        qg = F.linear(a_q, self._qg_weight, self._qg_bias)
        q, g = qg.split(self.c_q, dim=-1)
        kv = F.linear(a_k, self._kv_weight)
        k, v = kv.split(self.c_q, dim=-1)
        q_shape = q.shape[:-1] + (self.mha.no_heads, self.mha.c_hidden)
        k_shape = k.shape[:-1] + (self.mha.no_heads, self.mha.c_hidden)
        q = q.view(q_shape).transpose(-2, -3)
        k = k.view(k_shape).transpose(-2, -3)
        v = v.view(k_shape).transpose(-2, -3)
        a_out = _attention_from_qkv(q, k, v, biases, self.mha.c_hidden)
        g = torch.sigmoid(g).view(q_shape)
        a_out = (a_out.transpose(-2, -3) * g).reshape(
            g.shape[:-2] + (-1,)
        )
        a_out = F.linear(a_out, self.mha.linear_o.weight)

        a_out = a_out.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a_out = self.sigmoid(self.linear_ada_out(s)) * a_out

        return a_out
