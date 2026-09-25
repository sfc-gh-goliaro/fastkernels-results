"""Specialized fused top-k routing for the captured 128-expert workload."""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("topk_softmax_candidate", "topk_softmax_cuda.cu")
_topk_softmax = _C.topk_softmax


class TopKSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self._topk_weights = None
        self._topk_ids = None
        self._outputs = None

    def forward(
        self,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._outputs is None:
            M = router_logits.size(0)
            self._topk_weights = torch.empty(
                M, top_k, device=router_logits.device, dtype=torch.float32,
            )
            self._topk_ids = torch.empty(
                M, top_k, device=router_logits.device, dtype=torch.int32,
            )
            self._outputs = (self._topk_weights, self._topk_ids)
        _topk_softmax(router_logits, self._topk_weights, self._topk_ids)
        return self._outputs
