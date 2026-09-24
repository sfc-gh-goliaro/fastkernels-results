"""Multi-head attention with bias list support for AlphaFold3 (L2).

Composes QKV projections + SDPA + gated output.

Reference: openfold3/core/model/primitives/attention.py Attention

Why this is one CUDA kernel
---------------------------
The reference forward is ~35 torch dispatches (5 GEMMs, two einsums, softmax,
sigmoid, a gate multiply, and a pile of view/transpose/reshape) around a problem
that is *tiny*: the captured shapes are 16-128 tokens by 128-768 channels, i.e.
at most ~80 MMAC of arithmetic.  Under the scorer's timing loop on this B200 the
op costs ~200us, essentially all of it launch and dispatch: measured there, one
extra kernel in the stream is worth ~4.1us and one device-wide barrier inside a
kernel ~2.05us, while the arithmetic is under 2us total.

So the whole op is fused into a single kernel launch (``of3_attn_fused.cu``):
projections, biased softmax attention, gating and the output projection, with two
in-kernel device barriers for the two real data dependencies.  Anything the
kernel does not cover (non-bf16 inputs, a bias whose broadcast pattern is not an
affine function of (batch, head, query, key), head dims that are not a multiple
of 16, gating off) falls back to the reference path below, which is the baseline
code verbatim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.infra.cuda_ext import lazy_op

from ..L1.linear import Linear
from ..L1.softmax import Softmax

_C = lazy_op("of3_attn_fused", "of3_attn_fused.cu")

_NOBIAS: list[torch.Tensor] = []


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

        # Fast-path handle + its bound weight tuple, resolved on the first
        # forward: the harness moves and re-dtypes the module before then.  Kept in
        # __dict__ so the per-call lookup is a plain dict hit, not
        # nn.Module.__getattr__.
        d = self.__dict__
        d["_fkfn"] = None
        d["_fka"] = None
        d["_fktried"] = False

    def _fk_setup(self):
        d = self.__dict__
        d["_fktried"] = True
        fn = None
        try:
            fn = _C.of3_forward
        except Exception:  # noqa: BLE001 - no nvcc / build failure: stay on torch
            return None
        d["_fka"] = (
            self.linear_q.weight,
            self.linear_q.bias,
            self.linear_k.weight,
            self.linear_v.weight,
            self.linear_g.weight if self.linear_g is not None else None,
            self.linear_o.weight,
            self.no_heads,
            self.c_hidden,
        )
        d["_fkfn"] = fn
        return fn

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def _reference(self, q_x, kv_x, biases):
        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)
        return self._wrap_up(o, q_x)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = _NOBIAS

        fn = self._fkfn
        if fn is None:
            if self._fktried:
                return self._reference(q_x, kv_x, biases)
            fn = self._fk_setup()
            if fn is None:
                return self._reference(q_x, kv_x, biases)

        try:
            out = fn(q_x, kv_x, biases, *self._fka)
        except TypeError:
            # Something in the call the extension cannot even accept (e.g. a
            # ``None`` in the bias list); anything it merely does not *support* it
            # reports by returning None instead.
            return self._reference(q_x, kv_x, biases)
        if out is None:
            return self._reference(q_x, kv_x, biases)
        return out
