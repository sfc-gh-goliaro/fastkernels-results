"""Fused top-k + softmax routing for Mixture-of-Experts.

Uses a custom CUDA kernel for fused top-k selection and softmax
normalization with optional renormalization. Pre-allocates output buffers
for reuse and CUDA graph compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("topk_softmax_fast", "topk_softmax_fast.cu")


class TopKSoftmax(nn.Module):
    """Fused top-k selection with softmax normalization.

    Pre-allocates topk_weights and topk_ids buffers for CUDA graph
    compatibility. The output views are memoized per ``(M, top_k)`` so a steady
    state call does no slicing work at all.
    """

    def __init__(self):
        super().__init__()
        self._topk_weights = None
        self._topk_ids = None
        self._views = {}

    def _views_for(self, M, top_k, device):
        buf = self._topk_weights
        if buf is None or buf.size(0) < M or buf.size(1) != top_k:
            self._topk_weights = torch.empty(
                M, top_k, device=device, dtype=torch.float32,
            )
            self._topk_ids = torch.empty(
                M, top_k, device=device, dtype=torch.int32,
            )
            self._views = {}
        view = (self._topk_weights[:M], self._topk_ids[:M])
        self._views[(M, top_k)] = view
        return view

    @staticmethod
    def _reference(topk_weights, topk_ids, router_logits, top_k, renormalize):
        """Fallback for shapes/dtypes the fused kernel does not specialize for.

        A stable descending sort reproduces the kernel's tie break (equal
        probabilities keep increasing expert order); ``torch.topk`` does not.
        """
        probs = torch.softmax(router_logits.float(), dim=-1)
        values, ids = torch.sort(probs, dim=-1, descending=True, stable=True)
        weights = values[..., :top_k]
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        topk_weights.copy_(weights)
        topk_ids.copy_(ids[..., :top_k])

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts with softmax weights.

        Args:
            router_logits: [M, num_experts] router scores
            top_k: number of experts per token
            renormalize: renormalize weights to sum to 1

        Returns:
            topk_weights: [M, top_k] float32
            topk_ids: [M, top_k] int32
        """
        M = router_logits.size(0)
        view = self._views.get((M, top_k))
        if view is None:
            view = self._views_for(M, top_k, router_logits.device)
        topk_weights, topk_ids = view
        if not _C.topk_softmax_fast(topk_weights, topk_ids, router_logits,
                                    renormalize):
            self._reference(topk_weights, topk_ids, router_logits, top_k,
                            renormalize)
        return topk_weights, topk_ids
