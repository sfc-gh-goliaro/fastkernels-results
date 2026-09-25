"""MSA module embedder for AlphaFold3 (Algorithm 8, lines 1-4).

Embeds MSA features and adds projected s_input.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           MSAModuleEmbedder
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear


@triton.jit
def _msa_embed_scalar_kernel(
    msa,
    has_deletion,
    deletion_value,
    s_input,
    weight_m,
    weight_s,
    output,
    BLOCK_N: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    block_idx = tl.program_id(0)
    blocks_per_token: tl.constexpr = 64 // BLOCK_N
    col_block = block_idx % blocks_per_token
    token_seq_block = block_idx // blocks_per_token
    token = token_seq_block % 16
    seq_block = token_seq_block // 16
    seqs = seq_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    rows = seqs * 16 + token
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    msa_k = tl.arange(0, 32)
    msa_values = tl.load(
        msa + rows[:, None, None] * 32 + msa_k[None, None, :]
    ).to(tl.float32)
    msa_weights = tl.load(
        weight_m + cols[None, :, None] * 34 + msa_k[None, None, :]
    ).to(tl.float32)
    acc = tl.sum(msa_values * msa_weights, axis=2)
    has = tl.load(has_deletion + rows).to(tl.float32)
    deletion = tl.load(deletion_value + rows).to(tl.float32)
    acc += has[:, None] * tl.load(weight_m + cols[None, :] * 34 + 32).to(
        tl.float32
    )
    acc += deletion[:, None] * tl.load(weight_m + cols[None, :] * 34 + 33).to(
        tl.float32
    )

    s_k = tl.arange(0, 512)
    s_values = tl.load(
        s_input + token * 449 + s_k[None, :],
        mask=s_k[None, :] < 449,
        other=0.0,
    ).to(tl.float32)
    s_weights = tl.load(
        weight_s + cols[:, None] * 449 + s_k[None, :],
        mask=s_k[None, :] < 449,
        other=0.0,
    ).to(tl.float32)
    acc += tl.sum(s_values * s_weights, axis=1)[None, :]
    tl.store(output + rows[:, None] * 64 + cols[None, :], acc)


class MSAModuleEmbedder(nn.Module):
    """AF3 Algorithm 8, lines 1-4: MSA feature embedding.

    Args:
        c_m_feats: MSA input features channel dimension (34 = 32 msa + has_deletion + deletion_value)
        c_m: MSA channel dimension
        c_s_input: Single (s_input) channel dimension
    """

    def __init__(
        self,
        c_m_feats: int = 34,
        c_m: int = 64,
        c_s_input: int = 449,
    ):
        super().__init__()
        self.linear_m = Linear(c_m_feats, c_m, bias=False)
        self.linear_s_input = Linear(c_s_input, c_m, bias=False)

    def forward(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: needs msa [*, N_msa, N_token, 32],
                   has_deletion [*, N_msa, N_token],
                   deletion_value [*, N_msa, N_token],
                   msa_mask [*, N_msa, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
        """
        msa = batch["msa"]
        msa_mask = batch["msa_mask"]
        m = torch.empty(
            (*msa.shape[:-1], self.linear_m.weight.shape[0]),
            device=msa.device,
            dtype=msa.dtype,
        )
        _msa_embed_scalar_kernel[(m.numel() // 2,)](
            msa,
            batch["has_deletion"],
            batch["deletion_value"],
            s_input,
            self.linear_m.weight,
            self.linear_s_input.weight,
            m,
            BLOCK_N=2,
            BLOCK_SEQ=1,
            num_warps=2,
        )

        return m, msa_mask
