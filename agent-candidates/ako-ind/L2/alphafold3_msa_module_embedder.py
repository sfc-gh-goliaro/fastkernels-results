"""MSA module embedder for AlphaFold3 (Algorithm 8, lines 1-4).

Embeds MSA features and adds projected s_input.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           MSAModuleEmbedder

The captured shape is tiny (msa [1,8,16,32], s_input [1,16,449] -> m
[1,8,16,64], ~86 KB of traffic), so the operator is pure launch/dispatch
latency: the reference path spends four eager kernel launches plus a cat and an
unsqueeze on ~0.7 MFMA of work. Everything is fused into a single hand-written
CUDA kernel (``msa_embed.cu``, compiled once at import), launched with
Programmatic Dependent Launch so the block prologue overlaps the producer
kernel's tail, and ``msa_mask`` is returned by aliasing the input tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ....infra.cuda_ext import load_op

_fused = load_op("af3_msa_module_embedder_fused", "msa_embed.cu").msa_embed


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
                   msa_mask [*, N_seq, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
        """
        try:
            wm, ws = self._w
        except AttributeError:
            wm, ws = self.linear_m.weight, self.linear_s_input.weight
            self._w = (wm, ws)
        try:
            return _fused(batch, s_input, wm, ws)
        except RuntimeError:
            # Shape / dtype / layout the fused kernel does not cover.
            return self._reference(batch, s_input)

    def _reference(self, batch, s_input):
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        m = self.linear_m(msa_feat)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)
        return m, batch["msa_mask"]
