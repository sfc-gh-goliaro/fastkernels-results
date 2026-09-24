"""TP-aware embedding and LM head (L2 operators).

Two hand-written Triton kernels replace the two hot ops:

* ``_gather_rows`` -- the vocab-embedding lookup.  ``torch.embedding`` leaves
  20-28% on the table for these shapes; a 2-D grid of (row, D-block) with one
  128-bit-wide store per thread lands at the store-bandwidth wall.
* ``_lmh_splitn`` -- the logits projection for tiny M.  At M=1 the op is pure
  weight streaming (the 0.6 GB vocab table read once), so a split-N kernel with
  no K-split and an fp32 accumulator beats cuBLAS's GEMV path.

Everything outside those two windows (larger M, non-CUDA, odd dtypes/layouts,
the tp>1 collectives) falls through to the original torch implementation.
"""

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

_HIGH_PREC = (torch.float16, torch.bfloat16, torch.float32)


# ---------------------------------------------------------------------------
# Embedding gather.
# ---------------------------------------------------------------------------
@triton.jit
def _gather_rows(IDX, W, Y, n_rows, V, D, BLOCK_D: tl.constexpr):
    """Y[r, :] = W[IDX[r], :] over a (row, D-block) grid.

    Out-of-range ids yield zeros instead of an illegal access; the reference
    never produces them (they would be UB there too), this only keeps a bad
    input from taking the process down.
    """
    row = tl.program_id(0)
    offd = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    md = offd < D
    i = tl.load(IDX + row)
    ok = (i >= 0) & (i < V)
    v = tl.load(W + i * D + offd, mask=md & ok, other=0.0)
    tl.store(Y + row * D + offd, v, mask=md)


def _gather(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """nn.Embedding-equivalent lookup, or None if this input isn't supported."""
    if (weight.dim() != 2 or not weight.is_cuda or weight.dtype not in _HIGH_PREC
            or not weight.is_contiguous()):
        return None
    if idx.dtype not in (torch.int32, torch.int64) or idx.numel() == 0:
        return None
    V, D = weight.shape
    flat = idx.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    n = flat.numel()
    out = torch.empty((n, D), device=weight.device, dtype=weight.dtype)
    block_d = min(triton.next_power_of_2(D), 4096)
    _gather_rows[(n, triton.cdiv(D, block_d))](
        flat, weight, out, n, V, D, BLOCK_D=block_d, num_warps=4, num_stages=2)
    return out.view(*idx.shape, D)


# ---------------------------------------------------------------------------
# Logits projection for tiny M:  y = x @ W.T, split over N only.
# ---------------------------------------------------------------------------
@triton.jit
def _lmh_splitn(X, W, Y, M, N, K, sx, sw, sy,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    offn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offm = tl.arange(0, BLOCK_M)
    offk = tl.arange(0, BLOCK_K)
    mm = offm < M
    mn = offn < N
    xp = X + offm[:, None] * sx + offk[None, :]
    wp = W + offn[:, None] * sw + offk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mk = offk < K - k
        a = tl.load(xp, mask=mm[:, None] & mk[None, :], other=0.0)
        b = tl.load(wp, mask=mn[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(a, b.T, acc)
        xp += BLOCK_K
        wp += BLOCK_K
    tl.store(Y + offm[:, None] * sy + offn[None, :], acc.to(Y.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


# Beyond this many rows cuBLAS wins (measured on B200: 1.06x at M=1, 1.00x by
# M=60), so the custom path deliberately covers only the GEMV-like regime.
_LMH_MAX_M = 16


def _lmh_matmul(x: torch.Tensor, weight: torch.Tensor):
    """x @ weight.T for tiny M, or None if this input isn't supported."""
    if x.dim() != 2 or weight.dim() != 2 or not x.is_cuda:
        return None
    if x.dtype != weight.dtype or x.dtype not in (torch.float16, torch.bfloat16):
        return None
    M, K = x.shape
    N, KW = weight.shape
    if M > _LMH_MAX_M or K != KW or K % 8 != 0:
        return None
    if x.stride(1) != 1 or weight.stride(1) != 1:
        return None
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    _lmh_splitn[(triton.cdiv(N, 64),)](
        x, weight, out, M, N, K, x.stride(0), weight.stride(0), out.stride(0),
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=256, num_warps=4, num_stages=3)
    return out


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

    def _lookup(self, x):
        y = _gather(self.embedding_op.emb.weight, x)
        return self.embedding_op(x) if y is None else y

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start) & (x < self.vocab_end)
            x = mask * (x - self.vocab_start)
        y = self._lookup(x)
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
        y = _lmh_matmul(x, weight)
        return self.linear_op(x, weight) if y is None else y

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
