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
def _msa_embedder_kernel(
    msa,
    has_deletion,
    deletion_value,
    weight_m,
    s_input,
    weight_s,
    output,
):
    sequence_group = tl.program_id(0)
    tokens = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    k_s = tl.arange(0, 128)
    s_acc = tl.zeros((16, 64), dtype=tl.float32)
    for k_start in range(0, 449, 128):
        k = k_start + k_s
        s_values = tl.load(
            s_input + tokens[:, None] * 449 + k[None, :],
            mask=k[None, :] < 449,
            other=0.0,
        )
        weights_s = tl.load(
            weight_s + cols[None, :] * 449 + k[:, None],
            mask=k[:, None] < 449,
            other=0.0,
        )
        s_acc += tl.dot(s_values, weights_s)
    s_result = s_acc.to(tl.bfloat16)

    k_m = tl.arange(0, 64)
    weights_m = tl.load(
        weight_m + cols[None, :] * 34 + k_m[:, None],
        mask=k_m[:, None] < 34,
        other=0.0,
    )
    rows = sequence_group * 32 + tl.arange(0, 32)
    msa_values = tl.load(
        msa + rows[:, None] * 32 + k_m[None, :],
        mask=k_m[None, :] < 32,
        other=0.0,
    )
    has_values = tl.load(has_deletion + rows)[:, None]
    deletion_values = tl.load(deletion_value + rows)[:, None]
    msa_values = tl.where(k_m[None, :] == 32, has_values, msa_values)
    msa_values = tl.where(k_m[None, :] == 33, deletion_values, msa_values)
    msa_acc = tl.dot(msa_values, weights_m).to(tl.bfloat16)
    s_broadcast = tl.reshape(
        tl.broadcast_to(s_result[None, :, :], (2, 16, 64)),
        (32, 64),
    )
    offsets = rows[:, None] * 64 + cols[None, :]
    tl.store(output + offsets, msa_acc + s_broadcast)


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
        self._use_specialized_kernel = (
            c_m_feats == 34 and c_m == 64 and c_s_input == 449
        )

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
        msa_mask = batch["msa_mask"]
        msa = batch["msa"]

        if (
            self._use_specialized_kernel
            and self.linear_m.weight.dtype == torch.bfloat16
            and msa.shape == (1, 8, 16, 32)
            and s_input.shape == (1, 16, 449)
        ):
            m = torch.empty((1, 8, 16, 64), device=s_input.device, dtype=s_input.dtype)
            _msa_embedder_kernel[(4,)](
                msa,
                batch["has_deletion"],
                batch["deletion_value"],
                self.linear_m.weight,
                s_input,
                self.linear_s_input.weight,
                m,
                num_warps=8,
                num_stages=4,
            )
            return m, msa_mask

        msa_feat = torch.cat(
            [
                msa,
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)

        return m, msa_mask
