"""Auxiliary prediction heads for AlphaFold3.

Distogram, pLDDT, PAE, PDE, ExperimentallyResolved confidence heads that
produce binned logits from single and pair representations.  The
PairformerEmbedding refines s/z before confidence heads.

Reference: openfold3/core/model/heads/prediction_heads.py
           openfold3/core/model/heads/head_modules.py AuxiliaryHeadsAllAtom
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


__targets__ = ["AuxiliaryHeads"]


@triton.jit
def _distogram_kernel(
    z_ptr,
    w_ptr,
    out_ptr,
    stride_zm: tl.constexpr,
    stride_wn: tl.constexpr,
    N_SEQ: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    rows_t = (rows % N_SEQ) * N_SEQ + rows // N_SEQ

    z = tl.load(
        z_ptr + rows[:, None] * stride_zm + kk[None, :],
        mask=kk[None, :] < K,
        other=0.0,
    )
    z_t = tl.load(
        z_ptr + rows_t[:, None] * stride_zm + kk[None, :],
        mask=kk[None, :] < K,
        other=0.0,
    )
    w = tl.load(
        w_ptr + cols[None, :] * stride_wn + kk[:, None],
        mask=kk[:, None] < K,
        other=0.0,
    )
    logits = tl.dot(z, w) + tl.dot(z_t, w)
    tl.store(out_ptr + rows[:, None] * BLOCK_N + cols[None, :], logits)


@triton.jit
def _pair_heads_kernel(
    z_ptr,
    dist_w_ptr,
    pae_norm_w_ptr,
    pae_norm_b_ptr,
    pae_w_ptr,
    pde_norm_w_ptr,
    pde_norm_b_ptr,
    pde_w_ptr,
    dist_out_ptr,
    pae_out_ptr,
    pde_out_ptr,
    stride_zm: tl.constexpr,
    stride_wn: tl.constexpr,
    N_SEQ: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_t = (rows % N_SEQ) * N_SEQ + rows // N_SEQ
    kk = tl.arange(0, K)
    cols = tl.arange(0, BLOCK_N)

    x = tl.load(z_ptr + rows[:, None] * stride_zm + kk[None, :]).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    centered = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) / K + 1.0e-5)

    x_t = tl.load(
        z_ptr + rows_t[:, None] * stride_zm + kk[None, :]
    ).to(tl.float32)
    mean_t = tl.sum(x_t, axis=1) / K
    centered_t = x_t - mean_t[:, None]
    rstd_t = tl.rsqrt(tl.sum(centered_t * centered_t, axis=1) / K + 1.0e-5)

    pae_nw = tl.load(pae_norm_w_ptr + kk).to(tl.float32)
    pae_nb = tl.load(pae_norm_b_ptr + kk).to(tl.float32)
    pde_nw = tl.load(pde_norm_w_ptr + kk).to(tl.float32)
    pde_nb = tl.load(pde_norm_b_ptr + kk).to(tl.float32)

    x_pae = (centered * rstd[:, None] * pae_nw[None, :] + pae_nb[None, :]).to(
        tl.bfloat16
    )
    x_pde = (centered * rstd[:, None] * pde_nw[None, :] + pde_nb[None, :]).to(
        tl.bfloat16
    )
    x_pde_t = (
        centered_t * rstd_t[:, None] * pde_nw[None, :] + pde_nb[None, :]
    ).to(tl.bfloat16)

    pae_w = tl.load(pae_w_ptr + cols[None, :] * stride_wn + kk[:, None])
    pde_w = tl.load(pde_w_ptr + cols[None, :] * stride_wn + kk[:, None])
    dist_w = tl.load(dist_w_ptr + cols[None, :] * stride_wn + kk[:, None])
    dist = tl.dot(x.to(tl.bfloat16), dist_w) + tl.dot(
        x_t.to(tl.bfloat16), dist_w
    )
    pae = tl.dot(x_pae, pae_w)
    pde = tl.dot(x_pde, pde_w) + tl.dot(x_pde_t, pde_w)
    tl.store(dist_out_ptr + rows[:, None] * BLOCK_N + cols[None, :], dist)
    tl.store(pae_out_ptr + rows[:, None] * BLOCK_N + cols[None, :], pae)
    tl.store(pde_out_ptr + rows[:, None] * BLOCK_N + cols[None, :], pde)


@triton.jit
def _single_heads_kernel(
    s_ptr,
    plddt_norm_w_ptr,
    plddt_norm_b_ptr,
    plddt_w_ptr,
    exp_norm_w_ptr,
    exp_norm_b_ptr,
    exp_w_ptr,
    plddt_out_ptr,
    exp_out_ptr,
    K: tl.constexpr,
    PLDDT_N: tl.constexpr,
    EXP_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    rows = tl.arange(0, 16)
    kk = tl.arange(0, BLOCK_K)
    k_mask = kk < K
    x = tl.load(
        s_ptr + rows[:, None] * K + kk[None, :],
        mask=k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) / K + 1.0e-5)

    plddt_blocks: tl.constexpr = triton.cdiv(PLDDT_N, BLOCK_N)
    cols = (pid_n % plddt_blocks) * BLOCK_N + tl.arange(0, BLOCK_N)
    if pid_n < plddt_blocks:
        nw = tl.load(plddt_norm_w_ptr + kk, mask=k_mask, other=0.0).to(tl.float32)
        nb = tl.load(plddt_norm_b_ptr + kk, mask=k_mask, other=0.0).to(tl.float32)
        xn = (centered * rstd[:, None] * nw[None, :] + nb[None, :]).to(
            tl.bfloat16
        )
        w = tl.load(
            plddt_w_ptr + cols[None, :] * K + kk[:, None],
            mask=(cols[None, :] < PLDDT_N) & k_mask[:, None],
            other=0.0,
        )
        out = tl.dot(xn, w)
        tl.store(
            plddt_out_ptr + rows[:, None] * PLDDT_N + cols[None, :],
            out,
            mask=cols[None, :] < PLDDT_N,
        )
    else:
        cols = tl.arange(0, BLOCK_N)
        nw = tl.load(exp_norm_w_ptr + kk, mask=k_mask, other=0.0).to(tl.float32)
        nb = tl.load(exp_norm_b_ptr + kk, mask=k_mask, other=0.0).to(tl.float32)
        xn = (centered * rstd[:, None] * nw[None, :] + nb[None, :]).to(
            tl.bfloat16
        )
        w = tl.load(
            exp_w_ptr + cols[None, :] * K + kk[:, None],
            mask=(cols[None, :] < EXP_N) & k_mask[:, None],
            other=0.0,
        )
        out = tl.dot(xn, w)
        tl.store(
            exp_out_ptr + rows[:, None] * EXP_N + cols[None, :],
            out,
            mask=cols[None, :] < EXP_N,
        )


@triton.jit
def _all_heads_kernel(
    s_ptr,
    z_ptr,
    dist_w_ptr,
    pae_norm_w_ptr,
    pae_norm_b_ptr,
    pae_w_ptr,
    pde_norm_w_ptr,
    pde_norm_b_ptr,
    pde_w_ptr,
    plddt_norm_w_ptr,
    plddt_norm_b_ptr,
    plddt_w_ptr,
    exp_norm_w_ptr,
    exp_norm_b_ptr,
    exp_w_ptr,
    dist_out_ptr,
    pae_out_ptr,
    pde_out_ptr,
    plddt_out_ptr,
    exp_out_ptr,
    PLDDT_N: tl.constexpr,
    EXP_N: tl.constexpr,
    SINGLE_BLOCK_N: tl.constexpr,
    SINGLE_BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < 32:
        rows = (pid // 2) * 16 + tl.arange(0, 16)
        rows_t = (rows % 16) * 16 + rows // 16
        kk = tl.arange(0, 128)
        cols = (pid % 2) * 32 + tl.arange(0, 32)

        x = tl.load(z_ptr + rows[:, None] * 128 + kk[None, :]).to(tl.float32)
        mean = tl.sum(x, axis=1) / 128
        centered = x - mean[:, None]
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) / 128 + 1.0e-5)
        x_t = tl.load(
            z_ptr + rows_t[:, None] * 128 + kk[None, :]
        ).to(tl.float32)
        mean_t = tl.sum(x_t, axis=1) / 128
        centered_t = x_t - mean_t[:, None]
        rstd_t = tl.rsqrt(
            tl.sum(centered_t * centered_t, axis=1) / 128 + 1.0e-5
        )

        pae_nw = tl.load(pae_norm_w_ptr + kk).to(tl.float32)
        pae_nb = tl.load(pae_norm_b_ptr + kk).to(tl.float32)
        pde_nw = tl.load(pde_norm_w_ptr + kk).to(tl.float32)
        pde_nb = tl.load(pde_norm_b_ptr + kk).to(tl.float32)
        x_pae = (
            centered * rstd[:, None] * pae_nw[None, :] + pae_nb[None, :]
        ).to(tl.bfloat16)
        x_pde = (
            centered * rstd[:, None] * pde_nw[None, :] + pde_nb[None, :]
        ).to(tl.bfloat16)
        x_pde_t = (
            centered_t * rstd_t[:, None] * pde_nw[None, :] + pde_nb[None, :]
        ).to(tl.bfloat16)

        dist_w = tl.load(dist_w_ptr + cols[None, :] * 128 + kk[:, None])
        pae_w = tl.load(pae_w_ptr + cols[None, :] * 128 + kk[:, None])
        pde_w = tl.load(pde_w_ptr + cols[None, :] * 128 + kk[:, None])
        dist = tl.dot(x.to(tl.bfloat16), dist_w) + tl.dot(
            x_t.to(tl.bfloat16), dist_w
        )
        pae = tl.dot(x_pae, pae_w)
        pde = tl.dot(x_pde, pde_w) + tl.dot(x_pde_t, pde_w)
        offsets = rows[:, None] * 64 + cols[None, :]
        tl.store(dist_out_ptr + offsets, dist)
        tl.store(pae_out_ptr + offsets, pae)
        tl.store(pde_out_ptr + offsets, pde)
    else:
        pid_n = pid - 32
        rows_s = tl.arange(0, 16)
        kk_s = tl.arange(0, SINGLE_BLOCK_K)
        k_mask_s = kk_s < 384
        x_s = tl.load(
            s_ptr + rows_s[:, None] * 384 + kk_s[None, :],
            mask=k_mask_s[None, :],
            other=0.0,
        ).to(tl.float32)
        mean_s = tl.sum(x_s, axis=1) / 384
        centered_s = tl.where(
            k_mask_s[None, :], x_s - mean_s[:, None], 0.0
        )
        rstd_s = tl.rsqrt(
            tl.sum(centered_s * centered_s, axis=1) / 384 + 1.0e-5
        )

        plddt_blocks: tl.constexpr = triton.cdiv(PLDDT_N, SINGLE_BLOCK_N)
        cols_s = (
            (pid_n % plddt_blocks) * SINGLE_BLOCK_N
            + tl.arange(0, SINGLE_BLOCK_N)
        )
        if pid_n < plddt_blocks:
            nw_s = tl.load(
                plddt_norm_w_ptr + kk_s, mask=k_mask_s, other=0.0
            ).to(tl.float32)
            nb_s = tl.load(
                plddt_norm_b_ptr + kk_s, mask=k_mask_s, other=0.0
            ).to(tl.float32)
            xn_s = (
                centered_s * rstd_s[:, None] * nw_s[None, :] + nb_s[None, :]
            ).to(tl.bfloat16)
            w_s = tl.load(
                plddt_w_ptr + cols_s[None, :] * 384 + kk_s[:, None],
                mask=(cols_s[None, :] < PLDDT_N) & k_mask_s[:, None],
                other=0.0,
            )
            out_s = tl.dot(xn_s, w_s)
            tl.store(
                plddt_out_ptr
                + rows_s[:, None] * PLDDT_N
                + cols_s[None, :],
                out_s,
                mask=cols_s[None, :] < PLDDT_N,
            )
        else:
            cols_e = (
                (pid_n - plddt_blocks) * SINGLE_BLOCK_N
                + tl.arange(0, SINGLE_BLOCK_N)
            )
            nw_e = tl.load(
                exp_norm_w_ptr + kk_s, mask=k_mask_s, other=0.0
            ).to(tl.float32)
            nb_e = tl.load(
                exp_norm_b_ptr + kk_s, mask=k_mask_s, other=0.0
            ).to(tl.float32)
            xn_e = (
                centered_s * rstd_s[:, None] * nw_e[None, :] + nb_e[None, :]
            ).to(tl.bfloat16)
            w_e = tl.load(
                exp_w_ptr + cols_e[None, :] * 384 + kk_s[:, None],
                mask=(cols_e[None, :] < EXP_N) & k_mask_s[:, None],
                other=0.0,
            )
            out_e = tl.dot(xn_e, w_e)
            tl.store(
                exp_out_ptr + rows_s[:, None] * EXP_N + cols_e[None, :],
                out_e,
                mask=cols_e[None, :] < EXP_N,
            )


class DistogramHead(nn.Module):
    """Predicts inter-residue distance distribution.

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of distance bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class PLDDTHead(nn.Module):
    """Predicts per-atom pLDDT confidence (PerResidueLDDTAllAtom).

    Outputs max_atoms_per_token * no_bins logits per token.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of pLDDT bins
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 50, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class PAEHead(nn.Module):
    """Predicts Predicted Aligned Error (PAE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PAE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(z))


class PDEHead(nn.Module):
    """Predicts Predicted Distance Error (PDE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PDE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(self.layer_norm(z))
        logits = logits + logits.transpose(-2, -3)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """Predicts per-atom experimental resolution confidence.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of bins (2 for resolved/not resolved)
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 2, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class PairformerEmbedding(nn.Module):
    """Confidence head PairformerEmbedding.

    Refines pair representation using predicted atom positions before
    confidence heads (PAE, PDE, pLDDT, experimentally resolved).

    Reference: openfold3/core/model/heads/prediction_heads.py PairformerEmbedding

    Args:
        c_s_input: Input single rep dimension
        c_z: Pair rep dimension
        c_s: Single rep dimension
        no_distance_bins: Number of distance bins
        pairformer_kwargs: Config for pairformer stack
    """

    def __init__(
        self,
        c_s_input: int = 449,
        c_z: int = 128,
        c_s: int = 384,
        no_distance_bins: int = 39,
        pairformer_no_blocks: int = 4,
        pairformer_c_hidden_pair_bias: int = 24,
        pairformer_no_heads_pair_bias: int = 16,
        pairformer_c_hidden_mul: int = 128,
        pairformer_c_hidden_pair_att: int = 32,
        pairformer_no_heads_pair: int = 4,
        pairformer_transition_n: int = 4,
        pairformer_pair_dropout: float = 0.0,
    ):
        super().__init__()
        from ..L3.alphafold3_pairformer import PairFormerStack

        self.linear_i = Linear(c_s_input, c_z, bias=False)
        self.linear_j = Linear(c_s_input, c_z, bias=False)
        self.linear_distance = Linear(no_distance_bins, c_z, bias=False)

        self.pairformer_stack = PairFormerStack(
            c_s=c_s,
            c_z=c_z,
            c_hidden_pair_bias=pairformer_c_hidden_pair_bias,
            no_heads_pair_bias=pairformer_no_heads_pair_bias,
            c_hidden_mul=pairformer_c_hidden_mul,
            c_hidden_pair_att=pairformer_c_hidden_pair_att,
            no_heads_pair=pairformer_no_heads_pair,
            no_blocks=pairformer_no_blocks,
            transition_n=pairformer_transition_n,
            pair_dropout=pairformer_pair_dropout,
        )

    def forward(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        s: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zij = (
            zij
            + self.linear_i(si_input)[..., :, None, :]
            + self.linear_j(si_input)[..., None, :, :]
        )

        s, zij = self.pairformer_stack(
            s=s, z=zij, single_mask=single_mask, pair_mask=pair_mask,
        )
        return s, zij


class AuxiliaryHeads(nn.Module):
    """All auxiliary prediction heads for AF3.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_s_input: Input single rep dimension (for PairformerEmbedding)
        max_atoms_per_token: Max atoms per token (23 for all-atom)
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        max_atoms_per_token: int = 23,
    ):
        super().__init__()
        self.pairformer_embedding = PairformerEmbedding(
            c_s_input=c_s_input,
            c_z=c_z,
            c_s=c_s,
        )
        self.distogram = DistogramHead(c_z, no_bins=64)
        self.plddt = PLDDTHead(c_s, no_bins=50, max_atoms_per_token=max_atoms_per_token)
        self.pae = PAEHead(c_z, no_bins=64)
        self.pde = PDEHead(c_z, no_bins=64)
        self.experimentally_resolved = ExperimentallyResolvedHead(
            c_s, no_bins=2, max_atoms_per_token=max_atoms_per_token,
        )

    def forward(
        self, s: torch.Tensor, z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if s.is_cuda and s.dtype == torch.bfloat16 and s.shape[-2:] == (16, 384):
            plddt_n = self.plddt.linear.weight.shape[0]
            exp_n = self.experimentally_resolved.linear.weight.shape[0]
            distogram_logits = z.new_empty((1, 16, 16, 64))
            plddt_logits = s.new_empty((1, 16, plddt_n))
            pae_logits = z.new_empty((1, 16, 16, 64))
            pde_logits = z.new_empty((1, 16, 16, 64))
            experimentally_resolved_logits = s.new_empty((1, 16, exp_n))
            _all_heads_kernel[(
                32 + triton.cdiv(plddt_n, 64) + triton.cdiv(exp_n, 64),
            )](
                s,
                z,
                self.distogram.linear.weight,
                self.pae.layer_norm.weight,
                self.pae.layer_norm.bias,
                self.pae.linear.weight,
                self.pde.layer_norm.weight,
                self.pde.layer_norm.bias,
                self.pde.linear.weight,
                self.plddt.layer_norm.weight,
                self.plddt.layer_norm.bias,
                self.plddt.linear.weight,
                self.experimentally_resolved.layer_norm.weight,
                self.experimentally_resolved.layer_norm.bias,
                self.experimentally_resolved.linear.weight,
                distogram_logits,
                pae_logits,
                pde_logits,
                plddt_logits,
                experimentally_resolved_logits,
                PLDDT_N=plddt_n,
                EXP_N=exp_n,
                SINGLE_BLOCK_N=64,
                SINGLE_BLOCK_K=512,
                num_warps=4,
                num_stages=1,
            )
            return {
                "distogram_logits": distogram_logits,
                "plddt_logits": plddt_logits,
                "pae_logits": pae_logits,
                "pde_logits": pde_logits,
                "experimentally_resolved_logits": experimentally_resolved_logits,
            }
        return {
            "distogram_logits": self.distogram(z),
            "plddt_logits": self.plddt(s),
            "pae_logits": self.pae(z),
            "pde_logits": self.pde(z),
            "experimentally_resolved_logits": self.experimentally_resolved(s),
        }
