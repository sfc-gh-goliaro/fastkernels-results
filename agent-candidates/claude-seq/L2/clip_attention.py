"""CLIP self-attention (L2) -- fastkernels candidate.

The baseline spells the block out as eleven eager ops (three projections, three
``view``/``transpose`` pairs, a BMM, a scale, a mask add, a softmax, a second
BMM, a ``contiguous`` copy and the output projection). At the captured shape
(B=1, S=77, D=768, H=12, fp32) that is ~396 MFLOP -- roughly 2 us of B200
tensor-core work -- against ~100 us measured, so essentially all of the baseline
is per-op overhead, not arithmetic.

``clip_attention_fk.cu`` therefore collapses the whole block into a *single*
cooperative kernel with three stages separated by grid barriers:

* one GEMM computes q, k and v together (N = 3*768) and writes them straight into
  the ``[3, H, S, 64]`` layout attention reads, so the projections, views and
  transposes cost nothing extra;
* one block per (head, 16-row band) does QK^T, the scale, the mask add, the softmax
  and PV, storing to ``[S, H*64]`` -- the layout the baseline builds with
  ``transpose(1, 2).contiguous().reshape(...)``;
* one GEMM does the output projection.

A grid barrier costs ~1.1 us against ~2.0 us for a kernel boundary and, unlike a
launch, no host time at all -- which matters, because at ~25 us of GPU work the
host side is close to binding.

The kernels round their operands to TF32 the way the reference's cuBLAS path does
(``torch.backends.cuda.matmul.fp32_precision == 'tf32'``) and accumulate in fp32;
the tie rule turns out to matter, see ``to_tf32`` in the ``.cu``. The weights are
concatenated and pre-rounded once, on the first forward, so the kernel never
converts one; the cache is dropped whenever the parameters are reloaded or cast.

The module keeps the baseline's submodules -- and therefore its ``state_dict``
keys -- and falls back to the eager formulation for any input the kernel does not
cover (non-fp32, batch > 1, S > 128, a head dim other than 64, a hidden size that
is not a multiple of 256, an oddly strided mask).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPTextConfig

from fastkernels.infra.cuda_ext import load_op

from ..L1.linear import BMM, Linear
from ..L1.softmax import Softmax

_fused = load_op("clip_attention_fk", "clip_attention_fk.cu").clip_attention


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        # Packed kernel arguments, built on first use (weights are frozen by then).
        self._fused_args = None

    # -- weight cache ------------------------------------------------------
    @staticmethod
    def _to_tf32(t: torch.Tensor) -> torch.Tensor:
        """Round to TF32 the way cuBLAS does on the way into the tensor cores:
        nearest, ties to even. Adding ``0x0FFF + kept_lsb`` (not ``0x1000 +
        kept_lsb``, which rounds ties away from zero) is what makes it ties-to-even
        -- see the tie probe described in ``clip_attention_fk.cu``.

        Done once, when the weights are packed, so the kernel never converts a
        weight: the result is TF32-exact, so the mma's truncation is a no-op.
        """
        i = t.contiguous().view(torch.int32)
        i = i + 0x0FFF + ((i >> 13) & 1)
        return (i & -8192).view(torch.float32)

    def _build_fused_args(self):
        if self.q_proj.weight.dtype is not torch.float32:
            # Low-precision weights have no TF32 packing; forward stays on the eager
            # path. An empty tuple is the "do not try the kernel" marker.
            self._fused_args = ()
            return ()
        with torch.no_grad():
            w = self._to_tf32(torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), 0))
            b = torch.cat(
                (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), 0
            ).contiguous()
            wo = self._to_tf32(self.out_proj.weight)
        args = (w, b, wo, self.out_proj.bias, self.num_heads, self.scale)
        self._fused_args = args
        return args

    def _apply(self, *args, **kwargs):  # .to() / .cuda() / dtype casts
        self._fused_args = None
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._fused_args = None
        return super()._load_from_state_dict(*args, **kwargs)

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        args = self._fused_args
        if args is None:
            args = self._build_fused_args()
        if args:
            # The kernel vets its own preconditions -- cheaper than a dozen guards in
            # the interpreter -- and returns None when it cannot take the input.
            out = _fused(hidden_states, attention_mask, *args)
            if out is not None:
                return out
        return self._forward_eager(hidden_states, attention_mask)

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = self.bmm(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = self.softmax(attn_weights.float()).to(queries.dtype)

        attn_output = self.bmm(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)
