"""CLIP MLP and text embeddings (L2).

CLIPMLP: Linear -> QuickGELU -> Linear (no TP, frozen encoder).
CLIPTextEmbeddings: token + position embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import CLIPTextConfig

from ..L1.embedding import Embedding
from ..L1.linear import Linear
from ..L1.quickgelu import QuickGELU


@triton.jit
def _text_embeddings_kernel(
    input_ids,
    token_weight,
    position_weight,
    output,
    num_rows,
    seq_length: tl.constexpr,
    hidden_size: tl.constexpr,
    block_rows: tl.constexpr,
    block_hidden: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    cols = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
    row_mask = rows < num_rows
    token_ids = tl.load(input_ids + rows, mask=row_mask)
    positions = rows % seq_length
    mask = row_mask[:, None] & (cols[None, :] < hidden_size)
    token = tl.load(
        token_weight + token_ids[:, None] * hidden_size + cols[None, :],
        mask=mask,
    )
    position = tl.load(
        position_weight + positions[:, None] * hidden_size + cols[None, :],
        mask=mask,
    )
    tl.store(
        output + rows[:, None] * hidden_size + cols[None, :],
        token + position,
        mask=mask,
    )


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        output = torch.empty(
            (*input_ids.shape, self.token_embedding.embedding_dim),
            device=input_ids.device,
            dtype=self.token_embedding.emb.weight.dtype,
        )
        num_rows = input_ids.numel()
        hidden_size = self.token_embedding.embedding_dim
        block_hidden = triton.next_power_of_2(hidden_size)
        block_rows = 2
        _text_embeddings_kernel[
            (
                triton.cdiv(num_rows, block_rows),
                triton.cdiv(hidden_size, block_hidden),
            )
        ](
            input_ids,
            self.token_embedding.emb.weight,
            self.position_embedding.emb.weight,
            output,
            num_rows,
            seq_length,
            hidden_size,
            block_rows,
            block_hidden,
            num_warps=8,
        )
        return output
