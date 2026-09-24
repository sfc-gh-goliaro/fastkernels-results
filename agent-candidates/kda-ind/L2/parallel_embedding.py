"""TP-aware embedding and LM head (L2 operators).

`VocabParallelEmbedding` replaces `F.embedding` with a Triton row-gather on the
single-rank path. The writes dominate: only a handful of distinct rows are ever
read, so the read side stays in L2 while the whole `[N, D]` result streams out.
The kernel is therefore a plain masked copy, and the block width and warp count
are the only knobs that moved the number.

Profiling qualifies that picture rather than confirming it. The store is the
hottest instruction and is already at the maximum width (`STG.E.128`, 16
sectors/request, no excess), but the kernel is not pinned at the DRAM write
ceiling -- memory speed-of-light is ~70 %, and the dominant stall is an L1TEX
dependency: the scalar index load must return before any weight address can be
formed. Details and counters in
`profile/gather_p1_triton_vs_index_select/REPORT.md`.

`ParallelLMHead` is unchanged: cuBLASLt already runs the projection at 79 % of
HBM peak on the skinny shapes and 64 % of bf16 FLOP peak on the large one, and
hand-written Triton `tl.dot` prototypes measured 0.94-1.05x against it, so there
is nothing to win by replacing it and a second numerical path to lose by trying.

Accepted behavioural deviation: in a release build the gather does no
device-side bounds check, so an out-of-range index reads out of bounds where
`nn.Embedding`'s `index_select` raises a device-side assert. Every caller here
passes valid ids -- the multi-rank path masks out-of-partition ids to 0 before
the lookup -- and a per-element range test would cost every call to catch a case
none of them hit. Clamping instead of checking is deliberately not done: it would
turn a loud failure into a wrong answer.

The check is written, not merely described: `_gather_rows` carries a
`tl.device_assert` against the row count. Triton emits code for it only when
`TRITON_DEBUG` is set to a non-zero value, so setting `TRITON_DEBUG=1` turns an
out-of-range id into a device-side assertion failure at no cost to the release
path. `profile/debug_assert_p2.py` measures both halves of that claim.

That is the only deviation. Everything else the kernel does not model -- a CPU
tensor, a non-integer index dtype, a gradient-tracking weight, a padding row,
norm renormalisation, a weight that is not a contiguous matrix, or a missing
Triton -- takes the `F.embedding` path, so those cases keep `nn.Embedding`'s
exact semantics including autograd and its own error messages.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context
from ....infra.tp import _tp_size, _tp_rank
from ..L1.linear import Matmul
from ..L1.embedding import Embedding
from ..L1.allreduce import AllReduce

try:
    import triton
    import triton.language as tl
except ImportError:  # fall back to F.embedding rather than fail to import
    triton = None


# Fixed by an offline block-width sweep at N=16384, D=4096 (median of
# bench-style timings): 2048 and 4096 tie at 29.66 us, 1024 costs 29.73 us and
# 512 costs 31.74 us. 2048 is the smaller of the two tied widths, so it wastes
# less work on the D=2048 shape. Not autotuned -- a config search inside the
# benchmarked path would move compilation into the measured region.
_GATHER_BLOCK = 2048
_GATHER_WARPS = 4


if triton is not None:

    @triton.jit
    def _gather_rows(weight_ptr, index_ptr, out_ptr, dim, num_rows,
                     BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.program_id(1)
        # int64 row addressing: free on every shape benched here (the captured
        # index tensors are already int64) and it keeps `src * dim` from
        # wrapping for a vocabulary large enough to overflow int32.
        src = tl.load(index_ptr + row).to(tl.int64)
        # Generates no code unless TRITON_DEBUG is set, so the release path pays
        # nothing; with it set, an out-of-range id fails here instead of reading
        # out of bounds.
        tl.device_assert((src >= 0) & (src < num_rows),
                         "embedding index out of range")
        offs = col * BLOCK + tl.arange(0, BLOCK)
        mask = offs < dim
        vals = tl.load(weight_ptr + src * dim + offs, mask=mask)
        tl.store(out_ptr + row.to(tl.int64) * dim + offs, vals, mask=mask)


# Exactly what `F.embedding` accepts. Widening this would make the gather answer
# calls that `nn.Embedding` rejects, which is a behaviour change disguised as
# leniency: int16/int8/uint8/bool all raise there and must raise here too.
_INDEX_DTYPES = (torch.int64, torch.int32)


def _gather(emb: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    """Row lookup of `x` into `emb.weight`, bit-identical to `emb(x)`.

    Anything the kernel does not model falls back to `F.embedding` instead of
    being approximated: a CPU tensor, a non-integer index dtype, a weight that
    needs a gradient, a padding row, norm renormalisation, a weight that is not
    a contiguous matrix, or a missing Triton.
    """
    weight = emb.weight
    if (triton is None or not weight.is_cuda or x.device != weight.device
            or x.dtype not in _INDEX_DTYPES
            or emb.padding_idx is not None or emb.max_norm is not None
            or weight.dim() != 2 or not weight.is_contiguous()
            or (torch.is_grad_enabled() and weight.requires_grad)):
        return F.embedding(x, weight, emb.padding_idx, emb.max_norm,
                           emb.norm_type, emb.scale_grad_by_freq, emb.sparse)

    dim = weight.shape[1]
    if x.numel() == 0:
        return torch.empty((*x.shape, dim), dtype=weight.dtype, device=weight.device)

    index = x.reshape(-1)
    if not index.is_contiguous():
        index = index.contiguous()
    rows = index.numel()
    out = torch.empty((rows, dim), dtype=weight.dtype, device=weight.device)
    block = min(triton.next_power_of_2(dim), _GATHER_BLOCK)
    _gather_rows[(rows, triton.cdiv(dim, block))](
        weight, index, out, dim, weight.shape[0],
        BLOCK=block, num_warps=_GATHER_WARPS)
    return out.view(*x.shape, dim)


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
            y = self.embedding_op(mask * (x - self.vocab_start))
            return self.allreduce(mask.unsqueeze(-1) * y)
        # The weight is read through `self.embedding_op.emb` on every call, not
        # cached: the harness swaps `.data` when it casts params to bf16.
        return _gather(self.embedding_op.emb, x)


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
        # The full parameter, never a view: cuBLASLt falls off a cliff below
        # 16-byte weight alignment (a 1-element bf16 offset measured 11.4x
        # worse at M=16384).
        return self.linear_op(x, self.embedding_op.emb.weight)

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
