"""FLUX attention module (L2 composite).

Joint attention for dual-stream blocks (with added_kv_proj for text stream)
and self-attention for single-stream blocks (pre_only=True).

Mirrors vllm-omni's ``FluxAttention`` in
``vllm_omni/diffusion/models/flux/flux_transformer.py``.

Everything between the projection GEMMs and the attention kernel is streaming
glue, and the reference chain re-reads the 28 MB q and k tensors six-plus times
to do it: split/unflatten, ``norm_q``/``norm_k`` (and their ``_added``
counterparts), three ``torch.cat`` calls to join the text and image streams,
two ``cos/sin.to(dtype)`` casts, and two rope launches.  Measured on B200 at the
captured dual-stream shape that glue is ~190 us of a ~610 us call, and the
``value`` concatenation alone is ~107 us of it -- ``value`` is never normalized,
so it stays a strided view of the QKV projection and ``torch.cat`` takes its
non-vectorized path over it.

This implementation collapses the whole path into one packed buffer plus one
launch:

* **One buffer, no concatenation.**  ``[B, S_text + S_image, 3, H, D]`` is
  allocated once and the two QKV GEMMs are pointed straight at their row ranges
  with ``torch.addmm(out=...)`` -- text rows first, then image rows, the order
  the output ``split_with_sizes`` and the two output projections depend on.  Per
  token the ``3 * H * D`` projection outputs are contiguous, so each GEMM's
  destination is a dense 2-D block and the ``out=`` form costs exactly what the
  unfused GEMM did (measured identical).  ``q``, ``k`` and ``v`` are then
  strided views of that buffer, which the cuDNN sm100 flash kernel consumes
  as-is: bit-identical output, same time as contiguous inputs.  The single-stream
  path needs no allocation at all -- its QKV output *is* this layout.
* **One elementwise launch.**  ``flux_qk_fused.cu`` reads each q/k head row
  once, reduces in fp32, scales by the norm weight, rotates in registers and
  stores once, selecting ``norm_added_q/k`` vs ``norm_q/k`` from the token index
  so the two streams are covered by one launch over one buffer.  It consumes the
  captured fp64 ``(cos, sin)`` directly -- staged once per block through shared
  memory and rounded through the storage dtype there -- so the two cast kernels
  disappear rather than being cached.

Numerics are unchanged: the norm still accumulates in fp32 and is still rounded
to bf16 *before* the rotation (RoPE is norm-preserving but does not commute with
the affine weight), and cos/sin are still rounded to the storage dtype first.
Anything the packed path does not cover -- GQA, a head_dim other than 64/128, an
fp8 projection, batch > 1 with a text stream, a build failure -- falls through to
``_forward_reference``, which is the original chain verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from ....infra.cuda_ext import lazy_op
# Fused RMSNorm (torch.ops._C.rms_norm) for q/k-norm: head_dim (128) is a multiple
# of 32 so the CUDA kernel is valid. The divergence diagnostic showed our T5LayerNorm
# qk-norm is numerically identical to vllm-omni's RMSNorm (attention output cos=1.0),
# and this fused wrapper is that same kernel -- so it's bit-identical and replaces
# ~6 fp32 up/down-cast kernels per call with one fused kernel.
from ..L1.rms_norm import RMSNorm as FP32RMSNorm
from ..L1.diffusion_rope import DiffusionRoPE
from ..L1.dense_attention import DenseAttention
from .parallel_linear import (
    QKVParallelLinear,
    RowParallelLinear,
)

# Fused per-head RMSNorm + interleaved RoPE over the packed [B,S,3,H,D] buffer.
_C = lazy_op("flux_qk_fused", "flux_qk_fused.cu")

# The extension is built on first use, and any failure to build it is permanent
# and silent: the reference chain still produces the right answer.
_FUSED = {"fn": None, "tried": False}


def _fused_norm_rope():
    if not _FUSED["tried"]:
        _FUSED["tried"] = True
        try:
            _FUSED["fn"] = _C.flux_qk_norm_rope
        except Exception:  # noqa: BLE001 -- no nvcc, no GPU, compile error, ...
            _FUSED["fn"] = None
    return _FUSED["fn"]


_PACKED_DTYPES = (torch.bfloat16, torch.float16)
# head_dim -> most heads one block can walk: a lane group of 8 lanes owns one head
# row (two 16-byte vectors per lane at head_dim 128, one at 64), so 192 threads are
# 24 groups at either head_dim, and a group walks at most 8 rows.
_PACKED_MAX_HEADS = {64: 24 * 8 // 2, 128: 24 * 8 // 2}


def _tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather tensor across TP ranks along the given dimension."""
    import torch.distributed as dist
    tp = _tp_size()
    if tp <= 1:
        return tensor
    gather_list = [torch.empty_like(tensor) for _ in range(tp)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)


def _mm_into(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
             bias: torch.Tensor | None) -> None:
    """``out[:] = x @ weight.T + bias`` with no intermediate tensor.

    ``out`` is a dense 2-D row block of the packed buffer, which is what the
    cuBLASLt bias-epilogue path in ``addmm`` requires -- so this is the same
    kernel and the same time as the unfused projection, just writing elsewhere.
    """
    if bias is None:
        torch.mm(x, weight.t(), out=out)
    else:
        torch.addmm(bias, x, weight.t(), out=out)


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

    # ------------------------------------------------------------------
    # Packed pre-attention path
    # ------------------------------------------------------------------
    # At the two smaller captured shapes the call is host-bound, not
    # GPU-bound: the whole forward enqueues in ~176 us of CPU time against
    # ~119 us of GPU work, and ~56 us of that CPU time is the four GEMM
    # dispatches alone.  So the eligibility test is priced like a hot path --
    # everything that cannot change after construction is resolved once and
    # cached, and everything the kernel itself validates is left to the kernel
    # (its `false` return lands on `_norm_rope_reference`, which is correct for
    # any input the packed layout mis-describes).  Submodule and parameter
    # lookups go through ``nn.Module.__getattr__`` at ~0.24-0.34 us each, which
    # is why they are counted here rather than repeated.
    def _pack_cfg(self):
        cfg = self.__dict__.get("_pk_cfg")
        if cfg is None:
            qkv = self.to_qkv
            num_heads = qkv.num_heads
            head_dim = self.head_dim
            max_heads = _PACKED_MAX_HEADS.get(head_dim)
            ok = (max_heads is not None and num_heads <= max_heads
                  # A packed [S, 3, H, D] token row only describes q|k|v when
                  # the three projections have the same head count.
                  and qkv.num_kv_heads == num_heads and not qkv.use_fp8)
            if ok and self.added_kv_proj_dim is not None:
                add = self.add_kv_proj
                ok = (not add.use_fp8 and add.num_heads == num_heads
                      and add.num_kv_heads == num_heads)
            cfg = (ok, num_heads, head_dim, 3 * num_heads * head_dim,
                   1.0 / (head_dim ** 0.5), _tp_size())
            self.__dict__["_pk_cfg"] = cfg
        return cfg

    @staticmethod
    def _norm_weight(norm: FP32RMSNorm) -> torch.Tensor:
        return norm.weight if norm.elementwise_affine else norm._unit_weight

    def _norm_rope_reference(self, packed, s_split, image_rotary_emb):
        """Reference norm+rope over the packed buffer (fused launch declined).

        Reached when the kernel rejects the call -- an unsupported dtype, a norm
        weight in the wrong dtype, a cos/sin pair it cannot address.  Slower than
        the reference chain, but it keeps every such case correct instead of
        having to re-validate all of it on the host per call.
        """
        query, key = packed[:, :, 0], packed[:, :, 1]
        spans = ((0, s_split, self.norm_added_q, self.norm_added_k),
                 (s_split, packed.shape[1], self.norm_q, self.norm_k))
        for lo, hi, nq, nk in spans:
            if hi <= lo:
                continue
            query[:, lo:hi] = nq(query[:, lo:hi])
            key[:, lo:hi] = nk(key[:, lo:hi])
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query.copy_(self.rope(query.contiguous(), cos, sin))
            key.copy_(self.rope(key.contiguous(), cos, sin))

    def _forward_packed(self, fn, cfg, hidden_states, encoder_hidden_states,
                        image_rotary_emb, dual: bool):
        _, num_heads, head_dim, row, softmax_scale, tp_size = cfg
        s_image = hidden_states.shape[1]

        if dual:
            s_split = encoder_hidden_states.shape[1]
            seq = s_split + s_image
            packed = torch.empty(1, seq, 3, num_heads, head_dim,
                                 dtype=hidden_states.dtype, device=hidden_states.device)
            flat = packed.view(seq, row)
            # Text rows first, then image rows: the concatenation order the
            # output split and the to_out / to_add_out projections assume.
            _mm_into(flat[:s_split], encoder_hidden_states.view(-1, self.added_kv_proj_dim),
                     self.add_kv_proj.weight, self.add_kv_proj.bias)
            _mm_into(flat[s_split:], hidden_states.view(-1, self.query_dim),
                     self.to_qkv.weight, self.to_qkv.bias)
            added_q = self._norm_weight(self.norm_added_q)
            added_k = self._norm_weight(self.norm_added_k)
        else:
            s_split = 0
            seq = s_image
            packed = self.to_qkv.forward(hidden_states).view(-1, seq, 3, num_heads, head_dim)
            added_q = added_k = None

        if image_rotary_emb is None:
            cos = sin = None
        else:
            cos, sin = image_rotary_emb

        if not fn(packed, cos, sin, self._norm_weight(self.norm_q),
                  self._norm_weight(self.norm_k), added_q, added_k,
                  s_split, self.norm_q.eps):
            self._norm_rope_reference(packed, s_split, image_rotary_emb)

        # `unbind` builds the three views in one call; three `packed[:, :, i]`
        # subscripts cost 2.6 us more of host time.  Calling `forward` directly
        # skips ``nn.Module.__call__``'s hook dispatch (~2 us per submodule) --
        # the same trade the frozen DiffusionRoPE winner makes with
        # ``__call__ = forward``; none of these submodules has hooks.
        query, key, value = torch.unbind(packed, 2)
        hidden_states = self.attn.forward(query, key, value,
                                          softmax_scale=softmax_scale, causal=False)
        # cuDNN/SDPA hands back (head, seq) strides that make this permuted
        # flatten a view rather than a 28 MB copy.
        hidden_states = hidden_states.flatten(2, 3)
        if hidden_states.dtype is not packed.dtype:
            hidden_states = hidden_states.to(packed.dtype)

        if not dual:
            if tp_size > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states
        encoder_out, hidden_states = hidden_states.split_with_sizes(
            [s_split, hidden_states.shape[1] - s_split], dim=1)
        hidden_states = self.to_out[0].forward(hidden_states.contiguous())
        # F.dropout is the identity when training is False, so the eval-mode
        # call is skipped rather than dispatched.
        if self.training:
            hidden_states = self.to_out[1](hidden_states)
        return hidden_states, self.to_add_out.forward(encoder_out.contiguous())

    # ------------------------------------------------------------------
    def _project_out(self, hidden_states, encoder_hidden_states):
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1],
                 hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        if _tp_size() > 1:
            hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
        return hidden_states

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dual = self.added_kv_proj_dim is not None
        fn = _FUSED["fn"] if _FUSED["tried"] else _fused_norm_rope()
        # A mismatch between the projection config and the arguments (an
        # ``added_kv_proj_dim`` module called without an encoder stream, or the
        # reverse) is left to the reference chain, so it fails exactly where the
        # reference fails instead of being reinterpreted here.
        if fn is not None and (encoder_hidden_states is not None) is dual:
            cfg = self._pack_cfg()
            x = hidden_states
            # Only the incoming tensors are re-checked; everything else was
            # settled once in _pack_cfg or is the kernel's own to reject.
            if (cfg[0] and x.dim() == 3 and x.dtype in _PACKED_DTYPES
                    and x.is_cuda and x.is_contiguous()
                    and (not dual
                         # With a text stream the two GEMM destinations are row
                         # ranges of one buffer, dense 2-D blocks only at batch 1.
                         or (x.shape[0] == 1
                             and encoder_hidden_states.is_contiguous()
                             and encoder_hidden_states.dtype is x.dtype))):
                return self._forward_packed(fn, cfg, hidden_states,
                                            encoder_hidden_states,
                                            image_rotary_emb, dual)
        return self._forward_reference(hidden_states, encoder_hidden_states,
                                       image_rotary_emb)

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
        return self._project_out(hidden_states, encoder_hidden_states)
