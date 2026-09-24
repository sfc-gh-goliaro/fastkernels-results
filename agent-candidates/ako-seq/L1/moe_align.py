"""MoE token-to-expert alignment with block padding.

One fused CUDA launch (``moe_align_fast.cu``) replaces the stock two-launch /
two-block design, and the whole host side is memoised per (numel, block_size,
num_experts) so a steady-state call is a dict lookup plus one extension call.
Both matter: at these shapes the measured cost is dominated by launch count and
host-side aten calls, not by memory traffic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("moe_align_fast", "moe_align_fast.cu")


class MoeAlign(nn.Module):
    """MoE token-to-expert alignment.

    Pre-allocates output buffers for reuse and CUDA graph compatibility.
    """

    def __init__(self):
        super().__init__()
        self._sorted_token_ids = None
        self._expert_ids = None
        self._num_tokens_post_padded = None
        # (numel, block_size, num_experts) -> (sorted_view, expert_view, npad, ws).
        # A plan pins the buffer views that existed when it was built.  If a later,
        # larger shape grows a buffer, older plans keep using the older (still
        # correctly sized) allocation rather than re-slicing every call -- each
        # plan is internally consistent, which is all the caller can observe.
        self._plans = {}
        # num_experts -> kernel workspace.  Keyed, not shared: the layout below
        # depends on num_experts, so one buffer per expert count.
        self._ws = {}
        # (numel * block_size, device) -> constant length tensor for `naive`
        self._naive_lens = {}

    # ------------------------------------------------------------------ setup
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
        # [arrive, epoch, counts[E], prefix[E+1], gap_start[E]] -- see
        # moe_align_fast.cu.  Only the barrier path (large numel) uses it.  Must
        # start zeroed; the kernel restores the zeros it depends on before exit.
        if num_experts not in self._ws:
            self._ws[num_experts] = torch.zeros(
                3 * num_experts + 3, dtype=torch.int32, device=device,
            )

    def _plan(self, key, device):
        numel, block_size, num_experts = key
        if numel < num_experts:
            max_padded = numel * block_size
        else:
            max_padded = numel + num_experts * (block_size - 1)
        max_blocks = (max_padded + block_size - 1) // block_size
        self._ensure_buffers(max_padded, max_blocks, num_experts, device)
        plan = (
            self._sorted_token_ids[:max_padded],
            self._expert_ids[:max_blocks],
            self._num_tokens_post_padded,
            self._ws[num_experts],
        )
        self._plans[key] = plan
        return plan

    # ---------------------------------------------------------------- forward
    def _naive_forward(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Fast path: skip full alignment when tokens * top_k is very small.

        ``num_tokens_post_padded`` is a pure function of (numel, block_size), so
        it is memoised instead of re-materialised (a fresh ``torch.full`` is a
        whole kernel launch, ~30% of this path's cost).  It is never written
        in-place, so no inference-tensor hazard: the constant is only ever read.
        """
        numel = topk_ids.numel()
        total = numel * block_size
        key = (total, topk_ids.device)
        num_tokens_post_padded = self._naive_lens.get(key)
        if num_tokens_post_padded is None:
            num_tokens_post_padded = torch.full(
                (1,), total, dtype=torch.int32, device=topk_ids.device,
            )
            self._naive_lens[key] = num_tokens_post_padded
        expert_ids = topk_ids.view(-1)
        if expert_ids.dtype is not torch.int32:
            expert_ids = expert_ids.to(torch.int32)
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

        # The kernel is int32-only (one dtype keeps the template count down);
        # every captured call is already int32, so this is an identity check on
        # the hot path, not a dispatch.
        if topk_ids.dtype is not torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        numel = topk_ids.numel()
        key = (numel, block_size, num_experts)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plan(key, topk_ids.device)

        _C.moe_align_fast(
            topk_ids, plan[0], plan[1], plan[2], plan[3],
            num_experts, block_size,
        )
        return plan[0], plan[1], plan[2]
