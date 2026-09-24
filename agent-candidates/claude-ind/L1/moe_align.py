"""MoE token-to-expert alignment with block padding.

The whole alignment runs in a *single* kernel launch (see ``moe_align_fused.cu``
for the two strategies it picks between), instead of the usual count kernel
followed by a scatter kernel.  At these sizes a launch costs more than the work
it carries, so kernel count is what sets the latency.

The host side is shaped the same way: every per-shape quantity (output views,
expert count, block size) is resolved once and cached, and the ``naive`` path
reuses its device scalar instead of allocating a fresh one -- a ``torch.full``
per call is another launch, which is the entire cost of that path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("moe_align_fused", "moe_align_fused.cu")


class MoeAlign(nn.Module):
    """MoE token-to-expert alignment.

    Pre-allocates output buffers for reuse and CUDA graph compatibility.
    """

    def __init__(self):
        super().__init__()
        self._sorted_token_ids = None
        self._expert_ids = None
        self._num_tokens_post_padded = None
        self._cumsum_buffer = None
        self._plans = {}
        self._naive = {}

    def _ensure_buffers(self, max_padded, max_blocks, num_experts, device):
        if (self._sorted_token_ids is None
                or self._sorted_token_ids.size(0) < max_padded):
            self._sorted_token_ids = torch.empty(
                max_padded, dtype=torch.int32, device=device,
            )
        if (self._expert_ids is None
                or self._expert_ids.size(0) < max_blocks):
            self._expert_ids = torch.empty(
                max_blocks, dtype=torch.int32, device=device,
            )
        if (self._num_tokens_post_padded is None
                or self._num_tokens_post_padded.device != device):
            self._num_tokens_post_padded = torch.zeros(
                1, dtype=torch.int32, device=device,
            )
        # Unused by the fused kernel (the scan lives in shared memory), but
        # kept so the buffer allocation sequence -- and therefore the
        # allocator state behind the never-written tail of ``expert_ids``,
        # which the baseline also leaves uninitialized -- matches the baseline.
        if (self._cumsum_buffer is None
                or self._cumsum_buffer.size(0) < num_experts + 1):
            self._cumsum_buffer = torch.zeros(
                num_experts + 1, dtype=torch.int32, device=device,
            )

    def _build_plan(self, key, topk_ids, block_size, num_experts):
        numel = topk_ids.numel()
        if numel < num_experts:
            max_padded = numel * block_size
        else:
            max_padded = numel + num_experts * (block_size - 1)
        max_blocks = (max_padded + block_size - 1) // block_size

        prev = (id(self._sorted_token_ids), id(self._expert_ids))
        self._ensure_buffers(max_padded, max_blocks, num_experts,
                             topk_ids.device)
        if prev != (id(self._sorted_token_ids), id(self._expert_ids)):
            self._plans.clear()  # old views point at the outgrown buffers
        plan = (
            self._sorted_token_ids[:max_padded],
            self._expert_ids[:max_blocks],
            self._num_tokens_post_padded,
            int(num_experts),
            int(block_size),
        )
        self._plans[key] = plan
        return plan

    def _naive_forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Fast path: skip full alignment when tokens * top_k is very small."""
        numel = topk_ids.numel()
        expert_ids = topk_ids.reshape(-1)
        if expert_ids.dtype is not torch.int32:
            expert_ids = expert_ids.to(torch.int32)
        key = (numel, block_size, topk_ids.device)
        num_tokens_post_padded = self._naive.get(key)
        if num_tokens_post_padded is None:
            # Materialized once per (shape, block_size): a fresh device scalar
            # per call is a whole extra kernel launch, which dominates this path.
            num_tokens_post_padded = torch.full(
                (1,), numel * block_size, dtype=torch.int32,
                device=topk_ids.device,
            )
            self._naive[key] = num_tokens_post_padded
        return None, expert_ids, num_tokens_post_padded

    def forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        naive: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if naive:
            return self._naive_forward(topk_ids, block_size)

        key = (topk_ids.shape, block_size, num_experts, topk_ids.device)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._build_plan(key, topk_ids, block_size, num_experts)
        sorted_token_ids, expert_ids, npp, n_exp, bs = plan

        flat = topk_ids.reshape(-1)
        if flat.dtype is not torch.int32:
            flat = flat.to(torch.int32)
        _C.moe_align_fused(flat, n_exp, bs, sorted_token_ids, expert_ids, npp)
        return sorted_token_ids, expert_ids, npp
