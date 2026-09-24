"""Fused top-k + softmax routing for Mixture-of-Experts.

Every captured call of this op is ``bfloat16[M, 128]`` with ``top_k=8`` and
``renormalize=True``, and 7 of 8 captured M values read <= 256 KB of logits --
so the op is launch/critical-path bound, never bandwidth bound. ``tks2.cu``
holds a kernel specialized to exactly that regime; anything else falls back to
the generic vendored kernel (compiled lazily, so the fallback costs nothing
unless it is actually reached).

Host side: output views are cached per M so the steady-state call does one
dict lookup and one pybind call -- no allocation, no tensor slicing, no
dtype/expert-count dispatch ladder. Buffers are pre-allocated and stable, so
the call sequence stays CUDA-graph capturable.

The specialized kernel launches with Programmatic Dependent Launch, so its grid
is dispatched while whatever produced ``router_logits`` is still draining; it
waits on that producer with ``cudaGridDependencySynchronize()`` before its first
load, so every load and store still happens after the producer completes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

# Specialized bf16 / num_experts=128 / top_k=8 / renormalize kernel.
_C = lazy_op("tks2_fast_e128k8", "tks2.cu")
# Generic path (any dtype, any power-of-2 expert count, any k, softcapping,
# correction bias). Lazy: never compiled unless a non-specialized call arrives.
_C_GEN = lazy_op("tks_generic", "topk_softmax.cu")

_NUM_EXPERTS_FAST = 128
_TOP_K_FAST = 8


class TopKSoftmax(nn.Module):
    """Fused top-k selection with softmax normalization.

    Pre-allocates topk_weights and topk_ids buffers for CUDA graph
    compatibility.
    """

    def __init__(self):
        super().__init__()
        self._topk_weights = None
        self._topk_ids = None
        self._cap = 0
        self._cap_k = 0
        # M -> (weights_view, ids_view); rebuilt whenever the buffers grow.
        self._views = {}
        self._out_aligned = False

    def _alloc(self, M, top_k, device):
        self._cap = max(M, 1)
        self._cap_k = top_k
        self._topk_weights = torch.empty(
            self._cap, top_k, device=device, dtype=torch.float32,
        )
        self._topk_ids = torch.empty(
            self._cap, top_k, device=device, dtype=torch.int32,
        )
        self._views = {}
        self._out_aligned = (
            (self._topk_weights.data_ptr() | self._topk_ids.data_ptr()) & 15
        ) == 0

    def _make_view(self, M, top_k, device):
        if (self._topk_weights is None or M > self._cap
                or top_k != self._cap_k
                or self._topk_weights.device != device):
            self._alloc(M, top_k, device)
        v = (self._topk_weights[:M], self._topk_ids[:M])
        self._views[M] = v
        return v

    def _ensure_buffers(self, M, top_k, device):
        # Kept for API compatibility with the reference module.
        if (self._topk_weights is None or M > self._cap
                or top_k != self._cap_k):
            self._alloc(M, top_k, device)

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
        got = self._views.get(M)
        if got is None or top_k != self._cap_k:
            got = self._make_view(M, top_k, router_logits.device)
        topk_weights, topk_ids = got

        if (top_k == _TOP_K_FAST and renormalize
                and router_logits.dtype is torch.bfloat16
                and router_logits.size(1) == _NUM_EXPERTS_FAST
                and self._out_aligned
                and (router_logits.data_ptr() & 15) == 0):
            _C.tks2(topk_weights, topk_ids, router_logits, 0)
        else:
            _C_GEN.topk_softmax(topk_weights, topk_ids, router_logits,
                                renormalize, 0.0, None)
        return topk_weights, topk_ids
