"""Embedding lookup kernel."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _embedding_rows(input_ids, weight, output, embedding_dim: tl.constexpr,
                    block_dim: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block_dim)
    index = tl.load(input_ids + row)
    values = tl.load(weight + index * embedding_dim + cols,
                     mask=cols < embedding_dim)
    tl.store(output + row * embedding_dim + cols, values,
             mask=cols < embedding_dim)


@triton.jit
def _embedding_tiles(input_ids, weight, output, num_rows,
                     embedding_dim: tl.constexpr, block_dim: tl.constexpr,
                     block_rows: tl.constexpr):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    cols = tl.arange(0, block_dim)
    row_mask = rows < num_rows
    indices = tl.load(input_ids + rows, mask=row_mask)
    offsets = rows[:, None] * embedding_dim + cols[None, :]
    weight_offsets = indices[:, None] * embedding_dim + cols[None, :]
    mask = row_mask[:, None] & (cols[None, :] < embedding_dim)
    values = tl.load(weight + weight_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=padding_idx)
        self.embedding_dim = embedding_dim

    def forward(self, input_ids):
        output_shape = (*input_ids.shape, self.embedding_dim)
        output = torch.empty(output_shape, device=input_ids.device,
                             dtype=self.emb.weight.dtype)
        num_rows = input_ids.numel()
        if self.embedding_dim <= 128:
            block_dim = triton.next_power_of_2(self.embedding_dim)
            block_rows = 8192 // block_dim
            _embedding_tiles[(triton.cdiv(num_rows, block_rows),)](
                input_ids, self.emb.weight, output, num_rows,
                self.embedding_dim, block_dim, block_rows, num_warps=8)
        else:
            block_dim = triton.next_power_of_2(self.embedding_dim)
            num_warps = 8 if block_dim >= 2048 else 4
            _embedding_rows[(num_rows,)](
                input_ids, self.emb.weight, output, self.embedding_dim,
                block_dim, num_warps=num_warps)
        return output
