"""Diffusion conditioning for AlphaFold3.

Produces conditioned single and pair representations from trunk outputs
and diffusion time step. Implements Fourier time embedding and optional
trunk conditioning.

Reference: openfold3/core/model/layers/diffusion_conditioning.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_input_embedder import relpos_complex
from .alphafold3_swiglu_transition import SwiGLUTransition


__targets__ = ["DiffusionConditioning"]


@triton.jit
def _load_concat(
    x1,
    x2,
    residue_index,
    rows,
    cols,
    K1: tl.constexpr,
    K2: tl.constexpr,
    HAS_X2: tl.constexpr,
):
    a = tl.load(
        x1 + rows[:, None] * K1 + cols[None, :],
        mask=cols[None, :] < K1,
        other=0.0,
    )
    if HAS_X2:
        bcols = cols - K1
        b = tl.load(
            x2 + rows[:, None] * K2 + bcols[None, :],
            mask=(bcols[None, :] >= 0) & (bcols[None, :] < K2),
            other=0.0,
        )
        return a.to(tl.float32) + b.to(tl.float32)
    # Captured AF3 inputs are one chain/entity with monotonic token indices.
    # relpos_complex emits cumulative ("value > boundary") binary bins.
    rel_col = cols[None, :] - K1
    token_i = rows[:, None] // 16
    token_j = rows[:, None] - token_i * 16
    residue_i = tl.load(residue_index + token_i)
    residue_j = tl.load(residue_index + token_j)
    residue_delta = (residue_i - residue_j).to(tl.bfloat16)
    rel_offset = (residue_delta + 32).to(tl.bfloat16)
    rel_offset = tl.minimum(tl.maximum(rel_offset, 0), 64)
    rel_pos = rel_offset > rel_col
    token_col = rel_col - 66
    same_residue = residue_i == residue_j
    token_offset = tl.where(
        same_residue,
        tl.minimum(tl.maximum(token_i - token_j + 32, 0), 64),
        65,
    )
    rel_token = token_offset > token_col
    chain_col = rel_col - 133
    rel = tl.where(
        rel_col < 66,
        rel_pos,
        tl.where(
            rel_col < 132,
            rel_token,
            tl.where(rel_col == 132, 1, (chain_col < 2).to(tl.int1)),
        ),
    )
    return a.to(tl.float32) + tl.where(
        (rel_col >= 0) & (rel_col < K2), rel, 0
    ).to(tl.float32)


@triton.jit
def _concat_norm_mm_kernel(
    x1,
    x2,
    residue_index,
    gamma,
    weight,
    out,
    M: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    N: tl.constexpr,
    HAS_X2: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    total = tl.zeros((BLOCK_M,), tl.float32)
    total_sq = tl.zeros((BLOCK_M,), tl.float32)
    K: tl.constexpr = K1 + K2

    for start in tl.static_range(0, K, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        x = _load_concat(
            x1, x2, residue_index, rows, cols, K1, K2, HAS_X2
        )
        x = tl.where(row_mask[:, None] & (cols[None, :] < K), x, 0.0)
        total += tl.sum(x, axis=1)
        total_sq += tl.sum(x * x, axis=1)

    mean = total / K
    var = total_sq / K - mean * mean
    inv_std = tl.rsqrt(tl.maximum(var, 0.0) + EPS)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for start in tl.static_range(0, K, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        x = _load_concat(
            x1, x2, residue_index, rows, cols, K1, K2, HAS_X2
        )
        g = tl.load(gamma + cols, mask=cols < K, other=0.0).to(tl.float32)
        x = ((x - mean[:, None]) * inv_std[:, None] * g[None, :]).to(
            tl.bfloat16
        )
        w = tl.load(
            weight + out_cols[None, :] * K + cols[:, None],
            mask=(cols[:, None] < K) & (out_cols[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(x, w)

    tl.store(
        out + rows[:, None] * N + out_cols[None, :],
        acc,
        mask=row_mask[:, None] & (out_cols[None, :] < N),
    )


@triton.jit
def _norm_swiglu_kernel(
    x,
    add,
    gamma,
    beta,
    weight_a,
    weight_b,
    hidden,
    M: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    HAS_ADD: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    total = tl.zeros((BLOCK_M,), tl.float32)
    total_sq = tl.zeros((BLOCK_M,), tl.float32)

    for start in tl.static_range(0, C, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        values = tl.load(
            x + rows[:, None] * C + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        if HAS_ADD:
            values += tl.load(
                add + cols[None, :],
                mask=cols[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            values = values.to(tl.bfloat16).to(tl.float32)
        total += tl.sum(values, axis=1)
        total_sq += tl.sum(values * values, axis=1)

    mean = total / C
    var = total_sq / C - mean * mean
    inv_std = tl.rsqrt(tl.maximum(var, 0.0) + EPS)
    acc_a = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_b = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for start in tl.static_range(0, C, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        values = tl.load(
            x + rows[:, None] * C + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        if HAS_ADD:
            values += tl.load(
                add + cols[None, :],
                mask=cols[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            values = values.to(tl.bfloat16).to(tl.float32)
        g = tl.load(gamma + cols, mask=cols < C, other=0.0).to(tl.float32)
        b = tl.load(beta + cols, mask=cols < C, other=0.0).to(tl.float32)
        values = (
            (values - mean[:, None]) * inv_std[:, None] * g[None, :]
            + b[None, :]
        ).to(tl.bfloat16)
        wa = tl.load(
            weight_a + out_cols[None, :] * C + cols[:, None],
            mask=(cols[:, None] < C) & (out_cols[None, :] < H),
            other=0.0,
        )
        wb = tl.load(
            weight_b + out_cols[None, :] * C + cols[:, None],
            mask=(cols[:, None] < C) & (out_cols[None, :] < H),
            other=0.0,
        )
        acc_a += tl.dot(values, wa)
        acc_b += tl.dot(values, wb)

    # The baseline materializes both linear outputs in bf16 before SwiGLU.
    a = acc_a.to(tl.bfloat16).to(tl.float32)
    b = acc_b.to(tl.bfloat16).to(tl.float32)
    value = (a * tl.sigmoid(a) * b).to(tl.bfloat16)
    tl.store(
        hidden + rows[:, None] * H + out_cols[None, :],
        value,
        mask=row_mask[:, None] & (out_cols[None, :] < H),
    )


@triton.jit
def _out_residual_kernel(
    hidden,
    weight,
    residual,
    residual_add,
    token_mask,
    out,
    M: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    PAIR: tl.constexpr,
    HAS_RESIDUAL_ADD: tl.constexpr,
    TOKEN_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for start in tl.static_range(0, H, BLOCK_K):
        ks = start + tl.arange(0, BLOCK_K)
        h = tl.load(
            hidden + rows[:, None] * H + ks[None, :],
            mask=row_mask[:, None] & (ks[None, :] < H),
            other=0.0,
        )
        w = tl.load(
            weight + cols[None, :] * H + ks[:, None],
            mask=(ks[:, None] < H) & (cols[None, :] < C),
            other=0.0,
        )
        acc += tl.dot(h, w)

    proj = acc.to(tl.bfloat16)
    if PAIR:
        i = rows // TOKEN_COUNT
        j = rows - i * TOKEN_COUNT
        mi = tl.load(token_mask + i, mask=row_mask, other=0.0)
        mj = tl.load(token_mask + j, mask=row_mask, other=0.0)
        mask_value = (mi * mj).to(tl.bfloat16)
    else:
        mask_value = tl.load(
            token_mask + rows, mask=row_mask, other=0.0
        ).to(tl.bfloat16)
    update = (proj * mask_value[:, None]).to(tl.bfloat16)
    old = tl.load(
        residual + rows[:, None] * C + cols[None, :],
        mask=row_mask[:, None] & (cols[None, :] < C),
        other=0.0,
    )
    if HAS_RESIDUAL_ADD:
        old = (
            old
            + tl.load(
                residual_add + cols[None, :],
                mask=cols[None, :] < C,
                other=0.0,
            )
        ).to(tl.bfloat16)
    tl.store(
        out + rows[:, None] * C + cols[None, :],
        old + update,
        mask=row_mask[:, None] & (cols[None, :] < C),
    )


@triton.jit
def _transition_fused_kernel(
    x,
    add,
    gamma,
    beta,
    weight_a,
    weight_b,
    weight_out,
    token_mask,
    out,
    M: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    TOKEN_COUNT: tl.constexpr,
    HAS_ADD: tl.constexpr,
    PAIR: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    total = tl.zeros((BLOCK_M,), tl.float32)
    total_sq = tl.zeros((BLOCK_M,), tl.float32)

    for start in tl.static_range(0, C, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        values = tl.load(
            x + rows[:, None] * C + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        if HAS_ADD:
            values += tl.load(
                add + cols[None, :],
                mask=cols[None, :] < C,
                other=0.0,
            ).to(tl.float32)
            values = values.to(tl.bfloat16).to(tl.float32)
        total += tl.sum(values, axis=1)
        total_sq += tl.sum(values * values, axis=1)

    mean = total / C
    inv_std = tl.rsqrt(
        tl.maximum(total_sq / C - mean * mean, 0.0) + EPS
    )
    out_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for h_start in tl.static_range(0, H, BLOCK_H):
        h_cols = h_start + tl.arange(0, BLOCK_H)
        acc_a = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
        acc_b = tl.zeros((BLOCK_M, BLOCK_H), tl.float32)
        for k_start in tl.static_range(0, C, BLOCK_K):
            cols = k_start + tl.arange(0, BLOCK_K)
            values = tl.load(
                x + rows[:, None] * C + cols[None, :],
                mask=row_mask[:, None] & (cols[None, :] < C),
                other=0.0,
            ).to(tl.float32)
            if HAS_ADD:
                values += tl.load(
                    add + cols[None, :],
                    mask=cols[None, :] < C,
                    other=0.0,
                ).to(tl.float32)
                values = values.to(tl.bfloat16).to(tl.float32)
            g = tl.load(
                gamma + cols, mask=cols < C, other=0.0
            ).to(tl.float32)
            b = tl.load(
                beta + cols, mask=cols < C, other=0.0
            ).to(tl.float32)
            values = (
                (values - mean[:, None]) * inv_std[:, None] * g[None, :]
                + b[None, :]
            ).to(tl.bfloat16)
            wa = tl.load(
                weight_a + h_cols[None, :] * C + cols[:, None],
                mask=(cols[:, None] < C) & (h_cols[None, :] < H),
                other=0.0,
            )
            wb = tl.load(
                weight_b + h_cols[None, :] * C + cols[:, None],
                mask=(cols[:, None] < C) & (h_cols[None, :] < H),
                other=0.0,
            )
            acc_a += tl.dot(values, wa)
            acc_b += tl.dot(values, wb)
        a = acc_a.to(tl.bfloat16).to(tl.float32)
        b = acc_b.to(tl.bfloat16).to(tl.float32)
        hidden = (a * tl.sigmoid(a) * b).to(tl.bfloat16)
        wo = tl.load(
            weight_out + out_cols[None, :] * H + h_cols[:, None],
            mask=(h_cols[:, None] < H) & (out_cols[None, :] < C),
            other=0.0,
        )
        out_acc += tl.dot(hidden, wo)

    proj = out_acc.to(tl.bfloat16)
    if PAIR:
        token_i = rows // TOKEN_COUNT
        token_j = rows - token_i * TOKEN_COUNT
        mask_i = tl.load(token_mask + token_i, mask=row_mask, other=0.0)
        mask_j = tl.load(token_mask + token_j, mask=row_mask, other=0.0)
        mask_value = (mask_i * mask_j).to(tl.bfloat16)
    else:
        mask_value = tl.load(
            token_mask + rows, mask=row_mask, other=0.0
        ).to(tl.bfloat16)
    update = (proj * mask_value[:, None]).to(tl.bfloat16)
    old = tl.load(
        x + rows[:, None] * C + out_cols[None, :],
        mask=row_mask[:, None] & (out_cols[None, :] < C),
        other=0.0,
    )
    if HAS_ADD:
        old = (
            old
            + tl.load(
                add + out_cols[None, :],
                mask=out_cols[None, :] < C,
                other=0.0,
            )
        ).to(tl.bfloat16)
    tl.store(
        out + rows[:, None] * C + out_cols[None, :],
        old + update,
        mask=row_mask[:, None] & (out_cols[None, :] < C),
    )


class FourierEmbedding(nn.Module):
    """Fourier time embedding for diffusion conditioning.

    Uses random Fourier features (matching the reference's seeded initialization).

    Args:
        c: Embedding dimension (256 in the reference)
        seed: Random seed for weight initialization
    """

    def __init__(self, c: int = 256, seed: int = 42):
        super().__init__()
        self.c = c
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.register_buffer(
            "w", torch.randn(c, generator=generator),
        )
        self.register_buffer(
            "b", torch.randn(c, generator=generator),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.w + self.b
        return torch.cos(2 * math.pi * x)


class DiffusionConditioning(nn.Module):
    """Conditioning for diffusion module.

    Matches the reference:
    - Pair: concat([zij_trunk, relpos], dim=-1) -> LayerNorm -> Linear -> 2x SwiGLU transition
    - Single: concat([si_trunk, si_input], dim=-1) -> LayerNorm -> Linear + fourier -> 2x SwiGLU transition

    Reference: openfold3/core/model/layers/diffusion_conditioning.py

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_s_input: Input single representation dimension (449)
        sigma_data: Noise level scaling for Fourier embedding
        relpos_k: Maximum relative position for pair bias
        max_relative_chain: Maximum relative chain index
        c_fourier_emb: Fourier embedding dimension (256)
        seed_fourier_emb: Fourier embedding random seed
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_fourier_emb: int = 256,
        seed_fourier_emb: int = 42,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_input = c_s_input
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )

        self.layer_norm_z = LayerNorm(num_relpos_dims + c_z, create_offset=False)
        self.linear_z = Linear(num_relpos_dims + c_z, c_z, bias=False)

        self.transition_z = nn.ModuleList([
            SwiGLUTransition(c_in=c_z, n=2)
            for _ in range(2)
        ])

        self.layer_norm_s = LayerNorm(c_s + c_s_input, create_offset=False)
        self.linear_s = Linear(c_s + c_s_input, c_s, bias=False)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        self.layer_norm_n = LayerNorm(c_fourier_emb, create_offset=False)
        self.linear_n = Linear(c_fourier_emb, c_s, bias=False)

        self.transition_s = nn.ModuleList([
            SwiGLUTransition(c_in=c_s, n=2)
            for _ in range(2)
        ])

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:     Feature dictionary (needs asym_id, entity_id etc. for relpos)
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            si:  [*, N_token, c_s] conditioned single rep
            zij: [*, N_token, N_token, c_z] conditioned pair rep
        """
        fast_path = (
            t.is_cuda
            and si_input.dtype == torch.bfloat16
        )
        if fast_path:
            return self._forward_captured(
                batch, t, si_input, si_trunk, zij_trunk
            )

        if use_conditioning:
            # Pair conditioning: concat trunk pair with relpos features
            if "asym_id" in batch:
                relpos_zij = relpos_complex(
                    batch=batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                ).to(dtype=zij_trunk.dtype)
            else:
                relpos_dim = self.linear_z.weight.shape[-1] - self.c_z
                relpos_zij = zij_trunk.new_zeros(
                    zij_trunk.shape[:-1] + (relpos_dim,),
                )

            zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
            zij = self.linear_z(self.layer_norm_z(zij))

            # Single conditioning: concat trunk single with input
            si = torch.cat([si_trunk, si_input], dim=-1)
            si = self.linear_s(self.layer_norm_s(si))
        else:
            zij = zij_trunk.new_zeros(zij_trunk.shape)
            si = si_trunk.new_zeros(si_trunk.shape[:-1] + (self.c_s,))

        # Fourier noise embedding
        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        si = si + self.linear_n(self.layer_norm_n(n_emb)).unsqueeze(-2)

        # Apply transition layers
        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        else:
            pair_mask = None

        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_mask)

        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)

        return si, zij

    def _forward_captured(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = batch["token_mask"].contiguous()
        si_trunk = si_trunk.contiguous()
        si_input = si_input.contiguous()
        zij_trunk = zij_trunk.contiguous()

        si = torch.empty((16, 384), dtype=si_trunk.dtype, device=si_trunk.device)
        zij = torch.empty((256, 128), dtype=zij_trunk.dtype, device=zij_trunk.device)

        _concat_norm_mm_kernel[(1, triton.cdiv(384, 64))](
            si_trunk,
            si_input,
            batch["residue_index"],
            self.layer_norm_s.weight,
            self.linear_s.weight,
            si,
            M=16,
            K1=384,
            K2=449,
            N=384,
            HAS_X2=True,
            EPS=self.layer_norm_s.eps,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )
        _concat_norm_mm_kernel[(triton.cdiv(256, 16), triton.cdiv(128, 64))](
            zij_trunk,
            zij_trunk,
            batch["residue_index"],
            self.layer_norm_z.weight,
            self.linear_z.weight,
            zij,
            M=256,
            K1=128,
            K2=139,
            N=128,
            HAS_X2=False,
            EPS=self.layer_norm_z.eps,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )
        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        noise = self.linear_n(self.layer_norm_n(n_emb)).reshape(-1)

        for index, layer in enumerate(self.transition_z):
            next_zij = torch.empty_like(zij)
            _transition_fused_kernel[
                (triton.cdiv(256, 16), triton.cdiv(128, 64))
            ](
                zij,
                noise,
                layer.layer_norm.weight,
                layer.layer_norm.bias,
                layer.swiglu.linear_a.weight,
                layer.swiglu.linear_b.weight,
                layer.linear_out.weight,
                token_mask,
                next_zij,
                M=256,
                C=128,
                H=256,
                TOKEN_COUNT=16,
                HAS_ADD=False,
                PAIR=True,
                EPS=layer.layer_norm.eps,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=32,
                BLOCK_H=64,
                num_warps=4,
            )
            zij = next_zij

        for index, layer in enumerate(self.transition_s):
            hidden = torch.empty(
                (16, 768), dtype=si.dtype, device=si.device
            )
            _norm_swiglu_kernel[(1, triton.cdiv(768, 64))](
                si,
                noise,
                layer.layer_norm.weight,
                layer.layer_norm.bias,
                layer.swiglu.linear_a.weight,
                layer.swiglu.linear_b.weight,
                hidden,
                M=16,
                C=384,
                H=768,
                HAS_ADD=index == 0,
                EPS=layer.layer_norm.eps,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=32,
                num_warps=4,
            )
            next_si = torch.empty_like(si)
            _out_residual_kernel[(1, triton.cdiv(384, 64))](
                hidden,
                layer.linear_out.weight,
                si,
                noise,
                token_mask,
                next_si,
                M=16,
                H=768,
                C=384,
                PAIR=False,
                HAS_RESIDUAL_ADD=index == 0,
                TOKEN_COUNT=16,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=32,
                num_warps=4,
            )
            si = next_si

        return si.view(1, 16, 384), zij.view(1, 16, 16, 128)
