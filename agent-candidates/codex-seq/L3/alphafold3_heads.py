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
def _pair_heads_kernel(
    z,
    dist_w,
    pae_ln_w,
    pae_ln_b,
    pae_w,
    pde_ln_w,
    pde_ln_b,
    pde_w,
    dist_out,
    pae_out,
    pde_out,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    channels = tl.arange(0, 128)
    bins = tl.arange(0, 64)
    row_mask = rows < 256

    x = tl.load(
        z + rows[:, None] * 128 + channels[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    transposed_rows = (rows % 16) * 16 + rows // 16
    xt = tl.load(
        z + transposed_rows[:, None] * 128 + channels[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # The two pair LayerNorms have different affine parameters but identical
    # input statistics.
    mean = tl.sum(x, axis=1) * (1.0 / 128.0)
    centered = x - mean[:, None]
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=1) * (1.0 / 128.0) + EPS)

    mean_t = tl.sum(xt, axis=1) * (1.0 / 128.0)
    centered_t = xt - mean_t[:, None]
    rstd_t = tl.rsqrt(
        tl.sum(centered_t * centered_t, axis=1) * (1.0 / 128.0) + EPS
    )

    wd = tl.load(dist_w + bins[None, :] * 128 + channels[:, None])
    dist_input = (x + xt).to(tl.bfloat16)
    dist_acc = tl.dot(dist_input, wd)

    pae_scale = tl.load(pae_ln_w + channels).to(tl.float32)
    pae_bias = tl.load(pae_ln_b + channels).to(tl.float32)
    pae_input = (
        centered * rstd[:, None] * pae_scale[None, :] + pae_bias[None, :]
    ).to(tl.bfloat16)
    wpae = tl.load(pae_w + bins[None, :] * 128 + channels[:, None])
    pae_acc = tl.dot(pae_input, wpae)

    pde_scale = tl.load(pde_ln_w + channels).to(tl.float32)
    pde_bias = tl.load(pde_ln_b + channels).to(tl.float32)
    pde_input = (
        centered * rstd[:, None] * pde_scale[None, :] + pde_bias[None, :]
    ).to(tl.bfloat16)
    pde_input_t = (
        centered_t * rstd_t[:, None] * pde_scale[None, :] + pde_bias[None, :]
    ).to(tl.bfloat16)
    wpde = tl.load(pde_w + bins[None, :] * 128 + channels[:, None])
    pde_acc = tl.dot((pde_input + pde_input_t).to(tl.bfloat16), wpde)

    out_offsets = rows[:, None] * 64 + bins[None, :]
    mask = row_mask[:, None]
    tl.store(dist_out + out_offsets, dist_acc, mask=mask)
    tl.store(pae_out + out_offsets, pae_acc, mask=mask)
    tl.store(pde_out + out_offsets, pde_acc, mask=mask)


@triton.jit
def _single_heads_kernel(
    s,
    plddt_ln_w,
    plddt_ln_b,
    plddt_w,
    resolved_ln_w,
    resolved_ln_b,
    resolved_w,
    plddt_out,
    resolved_out,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tl.arange(0, 16)
    k = tl.arange(0, 128)

    x0 = tl.load(s + rows[:, None] * 384 + k[None, :]).to(tl.float32)
    x1 = tl.load(s + rows[:, None] * 384 + 128 + k[None, :]).to(tl.float32)
    x2 = tl.load(s + rows[:, None] * 384 + 256 + k[None, :]).to(tl.float32)
    mean = (
        tl.sum(x0, axis=1) + tl.sum(x1, axis=1) + tl.sum(x2, axis=1)
    ) * (1.0 / 384.0)
    variance = (
        tl.sum((x0 - mean[:, None]) * (x0 - mean[:, None]), axis=1)
        + tl.sum((x1 - mean[:, None]) * (x1 - mean[:, None]), axis=1)
        + tl.sum((x2 - mean[:, None]) * (x2 - mean[:, None]), axis=1)
    ) * (1.0 / 384.0)
    rstd = tl.rsqrt(variance + EPS)

    bins = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    plddt_acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for block in tl.static_range(0, 3):
        channels = block * 128 + k
        x = tl.load(s + rows[:, None] * 384 + channels[None, :]).to(tl.float32)
        scale = tl.load(plddt_ln_w + channels).to(tl.float32)
        bias = tl.load(plddt_ln_b + channels).to(tl.float32)
        normalized = (
            (x - mean[:, None]) * rstd[:, None] * scale[None, :]
            + bias[None, :]
        ).to(tl.bfloat16)
        weight = tl.load(
            plddt_w + bins[None, :] * 384 + channels[:, None],
            mask=bins[None, :] < 1150,
            other=0.0,
        )
        plddt_acc += tl.dot(normalized, weight)
    tl.store(
        plddt_out + rows[:, None] * 1150 + bins[None, :],
        plddt_acc,
        mask=bins[None, :] < 1150,
    )

    if tile == 0:
        resolved_bins = tl.arange(0, BLOCK_N)
        resolved_acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for block in tl.static_range(0, 3):
            channels = block * 128 + k
            x = tl.load(s + rows[:, None] * 384 + channels[None, :]).to(
                tl.float32
            )
            scale = tl.load(resolved_ln_w + channels).to(tl.float32)
            bias = tl.load(resolved_ln_b + channels).to(tl.float32)
            normalized = (
                (x - mean[:, None]) * rstd[:, None] * scale[None, :]
                + bias[None, :]
            ).to(tl.bfloat16)
            weight = tl.load(
                resolved_w
                + resolved_bins[None, :] * 384
                + channels[:, None],
                mask=resolved_bins[None, :] < 46,
                other=0.0,
            )
            resolved_acc += tl.dot(normalized, weight)
        tl.store(
            resolved_out + rows[:, None] * 46 + resolved_bins[None, :],
            resolved_acc,
            mask=resolved_bins[None, :] < 46,
        )


@triton.jit
def _all_heads_kernel(
    s,
    z,
    dist_w,
    pae_ln_w,
    pae_ln_b,
    pae_w,
    pde_ln_w,
    pde_ln_b,
    pde_w,
    plddt_ln_w,
    plddt_ln_b,
    plddt_w,
    resolved_ln_w,
    resolved_ln_b,
    resolved_w,
    dist_out,
    pae_out,
    pde_out,
    plddt_out,
    resolved_out,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAIR_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < PAIR_TILES:
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        channels = tl.arange(0, 128)
        bins = tl.arange(0, 64)

        x = tl.load(z + rows[:, None] * 128 + channels[None, :]).to(tl.float32)
        transposed_rows = (rows % 16) * 16 + rows // 16
        xt = tl.load(
            z + transposed_rows[:, None] * 128 + channels[None, :]
        ).to(tl.float32)

        mean = tl.sum(x, axis=1) * (1.0 / 128.0)
        centered = x - mean[:, None]
        rstd = tl.rsqrt(
            tl.sum(centered * centered, axis=1) * (1.0 / 128.0) + EPS
        )
        mean_t = tl.sum(xt, axis=1) * (1.0 / 128.0)
        centered_t = xt - mean_t[:, None]
        rstd_t = tl.rsqrt(
            tl.sum(centered_t * centered_t, axis=1) * (1.0 / 128.0) + EPS
        )

        wd = tl.load(dist_w + bins[None, :] * 128 + channels[:, None])
        dist_acc = tl.dot((x + xt).to(tl.bfloat16), wd)

        pae_scale = tl.load(pae_ln_w + channels).to(tl.float32)
        pae_bias = tl.load(pae_ln_b + channels).to(tl.float32)
        pae_input = (
            centered * rstd[:, None] * pae_scale[None, :] + pae_bias[None, :]
        ).to(tl.bfloat16)
        wpae = tl.load(pae_w + bins[None, :] * 128 + channels[:, None])
        pae_acc = tl.dot(pae_input, wpae)

        pde_scale = tl.load(pde_ln_w + channels).to(tl.float32)
        pde_bias = tl.load(pde_ln_b + channels).to(tl.float32)
        pde_input = (
            centered * rstd[:, None] * pde_scale[None, :] + pde_bias[None, :]
        ).to(tl.bfloat16)
        pde_input_t = (
            centered_t * rstd_t[:, None] * pde_scale[None, :]
            + pde_bias[None, :]
        ).to(tl.bfloat16)
        wpde = tl.load(pde_w + bins[None, :] * 128 + channels[:, None])
        pde_acc = tl.dot((pde_input + pde_input_t).to(tl.bfloat16), wpde)

        out_offsets = rows[:, None] * 64 + bins[None, :]
        tl.store(dist_out + out_offsets, dist_acc)
        tl.store(pae_out + out_offsets, pae_acc)
        tl.store(pde_out + out_offsets, pde_acc)
    else:
        tile = pid - PAIR_TILES
        rows = tl.arange(0, 16)
        k = tl.arange(0, 128)

        x0 = tl.load(s + rows[:, None] * 384 + k[None, :]).to(tl.float32)
        x1 = tl.load(s + rows[:, None] * 384 + 128 + k[None, :]).to(tl.float32)
        x2 = tl.load(s + rows[:, None] * 384 + 256 + k[None, :]).to(tl.float32)
        mean = (
            tl.sum(x0, axis=1) + tl.sum(x1, axis=1) + tl.sum(x2, axis=1)
        ) * (1.0 / 384.0)
        variance = (
            tl.sum((x0 - mean[:, None]) * (x0 - mean[:, None]), axis=1)
            + tl.sum((x1 - mean[:, None]) * (x1 - mean[:, None]), axis=1)
            + tl.sum((x2 - mean[:, None]) * (x2 - mean[:, None]), axis=1)
        ) * (1.0 / 384.0)
        rstd = tl.rsqrt(variance + EPS)

        out_bins = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        plddt_acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for block in tl.static_range(0, 3):
            channels = block * 128 + k
            x = tl.load(s + rows[:, None] * 384 + channels[None, :]).to(
                tl.float32
            )
            scale = tl.load(plddt_ln_w + channels).to(tl.float32)
            bias = tl.load(plddt_ln_b + channels).to(tl.float32)
            normalized = (
                (x - mean[:, None]) * rstd[:, None] * scale[None, :]
                + bias[None, :]
            ).to(tl.bfloat16)
            weight = tl.load(
                plddt_w + out_bins[None, :] * 384 + channels[:, None],
                mask=out_bins[None, :] < 1150,
                other=0.0,
            )
            plddt_acc += tl.dot(normalized, weight)
        tl.store(
            plddt_out + rows[:, None] * 1150 + out_bins[None, :],
            plddt_acc,
            mask=out_bins[None, :] < 1150,
        )

        if tile < 2:
            resolved_bins = tile * 32 + tl.arange(0, 32)
            resolved_acc = tl.zeros((16, 32), dtype=tl.float32)
            for block in tl.static_range(0, 3):
                channels = block * 128 + k
                x = tl.load(s + rows[:, None] * 384 + channels[None, :]).to(
                    tl.float32
                )
                scale = tl.load(resolved_ln_w + channels).to(tl.float32)
                bias = tl.load(resolved_ln_b + channels).to(tl.float32)
                normalized = (
                    (x - mean[:, None]) * rstd[:, None] * scale[None, :]
                    + bias[None, :]
                ).to(tl.bfloat16)
                resolved_weight = tl.load(
                    resolved_w
                    + resolved_bins[None, :] * 384
                    + channels[:, None],
                    mask=resolved_bins[None, :] < 46,
                    other=0.0,
                )
                resolved_acc += tl.dot(normalized, resolved_weight)
            tl.store(
                resolved_out + rows[:, None] * 46 + resolved_bins[None, :],
                resolved_acc,
                mask=resolved_bins[None, :] < 46,
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
        if (
            s.is_cuda
            and s.dtype == torch.bfloat16
            and s.is_contiguous()
            and z.is_contiguous()
            and s.shape == (1, 16, 384)
            and z.shape == (1, 16, 16, 128)
            and self.plddt.linear.weight.shape == (1150, 384)
        ):
            outputs = s.new_empty(68288)
            distogram_logits = outputs[:16384].view(1, 16, 16, 64)
            pae_logits = outputs[16384:32768].view(1, 16, 16, 64)
            pde_logits = outputs[32768:49152].view(1, 16, 16, 64)
            plddt_logits = outputs[49152:67552].view(1, 16, 1150)
            experimentally_resolved_logits = outputs[67552:].view(1, 16, 46)

            _all_heads_kernel[(52,)](
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
                EPS=self.plddt.layer_norm.eps,
                BLOCK_M=16,
                BLOCK_N=32,
                PAIR_TILES=16,
                num_warps=8,
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
