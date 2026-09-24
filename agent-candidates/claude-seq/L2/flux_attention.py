"""FLUX attention module (L2 composite) -- fastkernels candidate.

Same public surface as the baseline: joint attention for the dual-stream blocks
(``added_kv_proj_dim`` set, text stream projected by ``add_kv_proj``) and
self-attention for the single-stream blocks (``pre_only=True``).

What differs is the *prologue*.  The reference path between the QKV GEMM and the
attention kernel is five or eight separate passes over the same ~57 MB of q/k::

    qkv GEMM -> norm_q -> norm_k -> [cat_q, cat_k, cat_v] -> rope(q) -> rope(k)

every one of which reads and writes all of q and k, plus an fp64->bf16 cast of
``image_rotary_emb`` on every call.  Two changes collapse that to a single pass:

* **The concatenation is done by the GEMMs.**  The image and text QKV
  projections write disjoint row ranges of one joint ``[S_text + S_img, qkv]``
  buffer through ``addmm(out=...)`` (which keeps cuBLASLt's fused-bias epilogue
  and is bit-identical to ``F.linear``), so the three ``cat``s disappear.  The
  ``cat`` of the *value* stream was the single most expensive kernel in the
  dual-stream blocks -- its inputs are strided views of the two QKV buffers, so
  ``CatArrayBatchedCopy`` ran unvectorized at ~0.3 TB/s.

* **q/k-norm and RoPE are one in-place kernel** over the q|k region of that
  buffer (``flux_attention_fused.cu``), with the per-stream norm weights selected
  per row and cos/sin read (and rounded) straight from their captured fp64.

q, k and v are then *views* of the joint buffer.  cuDNN's sm100 flash kernel
takes them strided, and it lays its output out like the query -- so the
``(B, S, H, D) -> (B, S, H*D)`` flatten after it is free, and the dual-stream
split feeds the two output projections contiguous slices.  Net effect per call:
one GEMM epilogue, one elementwise pass, one attention kernel.

Anything the fast path does not cover -- TP > 1, fp8 quantized projections, a
head_dim other than 128, partial rotation, a missing ``image_rotary_emb``,
batch > 1 -- falls through to :meth:`FluxAttention._forward_reference`, which is
the baseline implementation verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from fastkernels.infra.cuda_ext import load_op
# Fused RMSNorm for q/k-norm: head_dim (128) is a multiple of 32 so the kernel is
# valid.  Kept as the module type the reference uses -- ``load_state_dict`` and
# the fallback path both go through it -- even though the fast path reads its
# weight directly.
from ..L1.rms_norm import RMSNorm as FP32RMSNorm
from ..L1.diffusion_rope import DiffusionRoPE
from ..L1.dense_attention import DenseAttention
from .parallel_linear import (
    QKVParallelLinear,
    RowParallelLinear,
)

_C = load_op("flux_attention_fused", "flux_attention_fused.cu")
_qk_norm_rope_ = _C.qk_norm_rope_


def _tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather tensor across TP ranks along the given dimension."""
    import torch.distributed as dist
    tp = _tp_size()
    if tp <= 1:
        return tensor
    gather_list = [torch.empty_like(tensor) for _ in range(tp)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)


def _norm_weight(norm: nn.Module) -> torch.Tensor | None:
    """The affine scale an RMSNorm applies, or None if it has no usable one."""
    w = getattr(norm, "weight", None)
    if w is None:
        w = getattr(norm, "_unit_weight", None)
    return w


class FluxAttention(nn.Module):
    """Multi-head attention for FLUX diffusion transformer.

    Supports two modes controlled by constructor args:
    - Dual-stream (``added_kv_proj_dim is not None``): separate QKV for image
      and text streams, concatenated before attention, split after.
    - Single-stream / pre-only (``pre_only=True``): standard self-attention,
      no output projection (caller handles it).
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = FP32RMSNorm(dim_head, eps=eps)
        self.norm_k = FP32RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            total_num_kv_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias,
                                  quant_config=quant_config),
                nn.Dropout(dropout),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = FP32RMSNorm(dim_head, eps=eps)
            self.norm_added_k = FP32RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                total_num_kv_heads=self.heads,
                bias=added_proj_bias if added_proj_bias is not None else True,
                quant_config=quant_config,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
                quant_config=quant_config,
            )

        self.rope = DiffusionRoPE(is_neox_style=False)
        self.attn = DenseAttention()

        # Static half of the fast-path predicate: nothing here can change between
        # calls, so the per-call check only has to look at the arguments.
        self._fused_static = (
            self.head_dim == 128
            and not self.to_qkv.use_fp8
            and (added_kv_proj_dim is None or not self.add_kv_proj.use_fp8)
            # The fused kernel covers one q and one k head segment per 16-lane
            # group, eight groups to a block.
            and self.to_qkv.num_heads == self.to_qkv.num_kv_heads
            and self.to_qkv.num_heads % 8 == 0
            and (added_kv_proj_dim is None
                 or (self.to_qkv.num_heads == self.add_kv_proj.num_heads
                     and self.to_qkv.num_kv_heads == self.add_kv_proj.num_kv_heads))
        )
        # Filled on the first fused call (see ``_fused_plan``): the weight/bias
        # handles and scalars the fast path needs, hoisted out of the per-call
        # attribute walks.  At ~150 us of Python per call against ~180 us of GPU
        # work for the smallest captured shape, the launch thread is close to
        # being the limit, so the fast path keeps its host work to the tensor ops
        # themselves.
        object.__setattr__(self, "_plan", None)

    # -- fast path ---------------------------------------------------------

    def _fused_plan(self):
        """(Re)build the cached handles the fused path reads.

        The transposed weight views are the only entries that can go stale, and
        they do: casting a module to the run dtype rebinds ``p.data`` to fresh
        storage while keeping the ``Parameter`` object, so a view taken before
        that still addresses the old buffer.  ``_forward_fused`` therefore
        revalidates by data pointer, not by object identity.
        """
        qkv_lin = self.to_qkv
        add_w_t = add_b = None
        w_aq = w_ak = None
        if self.added_kv_proj_dim is not None:
            add_w_t = self.add_kv_proj.weight.t()
            add_b = self.add_kv_proj.bias
            w_aq = _norm_weight(self.norm_added_q)
            w_ak = _norm_weight(self.norm_added_k)
        plan = (
            qkv_lin.weight.t(), qkv_lin.bias, add_w_t, add_b,
            _norm_weight(self.norm_q), _norm_weight(self.norm_k), w_aq, w_ak,
            qkv_lin.num_heads, qkv_lin.num_kv_heads,
            qkv_lin.num_heads + 2 * qkv_lin.num_kv_heads,
            # Spelled exactly as the reference does: ``head_dim ** -0.5`` differs
            # from it by one ulp at head_dim=128, which the attention would carry
            # into every score.
            float(self.norm_q.eps), 1.0 / (self.head_dim ** 0.5),
        )
        object.__setattr__(self, "_plan", plan)
        return plan

    def _fused_ok(self, hidden_states, encoder_hidden_states, image_rotary_emb) -> bool:
        if not self._fused_static or image_rotary_emb is None:
            return False
        if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
            return False
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            return False
        if _tp_size() > 1:
            return False
        if (encoder_hidden_states is None) != (self.added_kv_proj_dim is None):
            return False
        if encoder_hidden_states is not None and (
                encoder_hidden_states.dim() != 3 or encoder_hidden_states.size(0) != 1
                or encoder_hidden_states.dtype != hidden_states.dtype):
            return False
        cos, sin = image_rotary_emb
        if cos.dim() != 2 or sin.dim() != 2 or cos.size(1) != self.head_dim // 2:
            return False
        return True

    def _forward_fused(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        plan = self._plan
        if plan is None or plan[0].data_ptr() != self.to_qkv.weight.data_ptr():
            plan = self._fused_plan()
        (w_t, bias, add_w_t, add_bias, w_q, w_k, w_aq, w_ak,
         num_heads, num_kv_heads, total_heads, eps, scale) = plan

        x = hidden_states.view(-1, hidden_states.size(-1))
        if encoder_hidden_states is None:
            qkv = (torch.mm(x, w_t) if bias is None
                   else torch.addmm(bias, x, w_t))
            n_enc = 0
            w_aq = w_ak = None
        else:
            # Both projections write disjoint row ranges of one buffer, in the
            # order the reference's three ``cat``s produce -- text, then image.
            ctx = encoder_hidden_states.view(-1, encoder_hidden_states.size(-1))
            n_enc = ctx.size(0)
            qkv = torch.empty((n_enc + x.size(0), w_t.size(1)),
                              dtype=x.dtype, device=x.device)
            head, tail = qkv[:n_enc], qkv[n_enc:]
            if add_bias is None:
                torch.mm(ctx, add_w_t, out=head)
            else:
                torch.addmm(add_bias, ctx, add_w_t, out=head)
            if bias is None:
                torch.mm(x, w_t, out=tail)
            else:
                torch.addmm(bias, x, w_t, out=tail)

        cos, sin = image_rotary_emb
        # Raises for any layout the kernel does not cover -> reference chain.
        _qk_norm_rope_(qkv, cos, sin, w_q, w_k, w_aq, w_ak,
                       n_enc, num_heads, num_kv_heads, eps)

        seq = qkv.size(0)
        qkv = qkv.view(1, seq, total_heads, self.head_dim)
        out = self.attn(qkv[:, :, :num_heads],
                        qkv[:, :, num_heads:num_heads + num_kv_heads],
                        qkv[:, :, num_heads + num_kv_heads:],
                        softmax_scale=scale, causal=False)
        # cuDNN lays its output out like the query, so this is a view, not a copy.
        out = out.flatten(2, 3)

        if encoder_hidden_states is None:
            return out
        image_out = self.to_out[0](out.narrow(1, n_enc, seq - n_enc))
        if self.dropout:
            image_out = self.to_out[1](image_out)
        return image_out, self.to_add_out(out.narrow(1, 0, n_enc))

    # -- reference path (baseline, verbatim) --------------------------------

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)
        return query, key

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_heads = self.to_qkv.num_heads
        num_kv_heads = self.to_qkv.num_kv_heads

        qkv = self.to_qkv(hidden_states)
        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (num_heads, -1))
        key = key.unflatten(-1, (num_kv_heads, -1))
        value = value.unflatten(-1, (num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            add_num_heads = self.add_kv_proj.num_heads
            add_num_kv_heads = self.add_kv_proj.num_kv_heads

            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            add_q_size = add_num_heads * self.head_dim
            add_kv_size = add_num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (add_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (add_num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (add_num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query, key = self._apply_rope(query, key, image_rotary_emb)

        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = self.attn(query, key, value, softmax_scale=softmax_scale, causal=False)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        else:
            if _tp_size() > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._fused_ok(hidden_states, encoder_hidden_states, image_rotary_emb):
            try:
                return self._forward_fused(
                    hidden_states, encoder_hidden_states, image_rotary_emb)
            except RuntimeError:
                pass
        return self._forward_reference(
            hidden_states, encoder_hidden_states, image_rotary_emb)
