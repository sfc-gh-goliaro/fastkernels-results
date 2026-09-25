"""TP-aware embedding and LM head (L2 operators)."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.context import get_context
from ....infra.tp import _tp_size, _tp_rank
from ..L1.linear import Matmul
from ..L1.embedding import Embedding
from ..L1.allreduce import AllReduce


@triton.jit
def _lm_head_matmul(x, weight, output, m: tl.constexpr, n: tl.constexpr,
                    k: tl.constexpr, block_m: tl.constexpr,
                    block_n: tl.constexpr, block_k: tl.constexpr,
                    group_m: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(m, block_m)
    num_n = tl.cdiv(n, block_n)
    group_width = group_m * num_n
    group_id = pid // group_width
    first_m = group_id * group_m
    group_size = tl.minimum(num_m - first_m, group_m)
    pid_m = first_m + ((pid % group_width) % group_size)
    pid_n = (pid % group_width) // group_size

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    x_ptrs = x + offs_m[:, None] * k + offs_k[None, :]
    w_ptrs = weight + offs_n[:, None] * k + offs_k[None, :]
    accumulator = tl.zeros((block_m, block_n), tl.float32)
    for kk in range(0, k, block_k):
        a = tl.load(x_ptrs, mask=offs_m[:, None] < m, other=0.0)
        b = tl.load(w_ptrs)
        accumulator += tl.dot(a, tl.trans(b))
        x_ptrs += block_k
        w_ptrs += block_k
    out_ptrs = output + offs_m[:, None] * n + offs_n[None, :]
    tl.store(out_ptrs, accumulator,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def _project_small(x, weight):
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    block_m = 64
    block_n = 128
    block_k = 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _lm_head_matmul[grid](
        x, weight, output, m, n, k, block_m, block_n, block_k, 8,
        num_warps=8, num_stages=3)
    return output


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__()
        tp, rank = _tp_size(), _tp_rank()
        assert num_embeddings % tp == 0
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.padding_size = padding_size
        self.embedding_dim = embedding_dim
        self.per_partition = num_embeddings // tp
        self.vocab_start = self.per_partition * rank
        self.vocab_end = self.vocab_start + self.per_partition
        self.tp_size = tp
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.embedding_op = Embedding(self.per_partition, embedding_dim)
        self.embedding_op.emb.weight.weight_loader = self._weight_loader
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, rank * shard, shard))

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start) & (x < self.vocab_end)
            x = mask * (x - self.vocab_start)
        y = self.embedding_op(x)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            y = self.allreduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 bias: bool = False,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64):
        super().__init__(num_embeddings, embedding_dim,
                         params_dtype=params_dtype,
                         org_num_embeddings=org_num_embeddings,
                         padding_size=padding_size)
        self.linear_op = Matmul()

    def project(self, x):
        """Linear projection only (no gather). Used inside CUDA graph."""
        ctx = get_context()
        if ctx.is_mixed:
            x = x[ctx.logit_indices].contiguous()
        elif ctx.is_prefill:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        weight = self.embedding_op.emb.weight
        if (x.ndim == 2 and x.shape == (64, 2304)
                and x.dtype == torch.bfloat16
                and weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()):
            return _project_small(x, weight)
        return self.linear_op(x, weight)

    def gather_logits(self, logits):
        """Gather partial logits from all ranks. Used outside CUDA graph."""
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if _tp_rank() == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if _tp_rank() == 0 else logits
        return logits

    def gather_greedy(self, logits):
        """Fast path for greedy: local argmax + small allgather.

        Instead of gathering full vocab logits (~31MB/rank), gather only
        the (max_val, max_idx) per sequence (~2KB/rank).
        Returns token IDs directly on rank 0, None on other ranks.
        """
        if self.tp_size <= 1:
            return None

        rank = _tp_rank()
        local_max_vals, local_max_idxs = logits.max(dim=-1)
        local_max_idxs = local_max_idxs + self.vocab_start

        info = torch.stack([local_max_vals, local_max_idxs.float()], dim=-1)
        gathered = [torch.empty_like(info) for _ in range(self.tp_size)]
        dist.all_gather(gathered, info)
        if rank == 0:
            all_info = torch.stack(gathered, dim=0)
            all_vals = all_info[:, :, 0]
            all_idxs = all_info[:, :, 1].long()
            best_rank = all_vals.argmax(dim=0)
            bs = logits.size(0)
            token_ids = all_idxs[best_rank, torch.arange(bs, device=logits.device)]
            return token_ids
        return None

    def forward(self, x):
        logits = self.project(x)
        return self.gather_logits(logits)
