"""FLUX attention module (L2 composite).

Joint attention for dual-stream blocks (with added_kv_proj for text stream)
and self-attention for single-stream blocks (pre_only=True).

Mirrors vllm-omni's ``FluxAttention`` in
``vllm_omni/diffusion/models/flux/flux_transformer.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_size
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


@triton.jit
def _norm_rope_qkv_kernel(
    QKV,
    QUERY,
    KEY,
    VALUE,
    NORM_Q,
    NORM_K,
    COS,
    SIN,
    qkv_stride,
    output_token_offset,
    eps,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WRITE_VALUE: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // NUM_HEADS
    head = row - token * NUM_HEADS
    d = tl.arange(0, HEAD_DIM)
    qkv_base = token * qkv_stride + head * HEAD_DIM

    q = tl.load(QKV + qkv_base + d).to(tl.float32)
    k = tl.load(
        QKV + qkv_base + NUM_HEADS * HEAD_DIM + d
    ).to(tl.float32)
    q_scale = tl.rsqrt(tl.sum(q * q, axis=0) / HEAD_DIM + eps)
    k_scale = tl.rsqrt(tl.sum(k * k, axis=0) / HEAD_DIM + eps)

    # Quantize at the same point as the standalone BF16 RMSNorm. RoPE then
    # operates on that rounded value, matching the baseline's kernel boundary.
    q = (q * q_scale * tl.load(NORM_Q + d)).to(tl.bfloat16)
    k = (k * k_scale * tl.load(NORM_K + d)).to(tl.bfloat16)

    partner_d = d + tl.where(d % 2 == 0, 1, -1)
    q_partner = tl.gather(q, partner_d, axis=0)
    k_partner = tl.gather(k, partner_d, axis=0)

    output_token = token + output_token_offset
    rotary_col = d // 2
    cos = tl.load(COS + output_token * (HEAD_DIM // 2) + rotary_col)
    sin = tl.load(SIN + output_token * (HEAD_DIM // 2) + rotary_col)
    # The baseline casts the captured FP64 tables to BF16 before applying RoPE.
    cos = cos.to(tl.bfloat16).to(tl.float32)
    sin = sin.to(tl.bfloat16).to(tl.float32)
    sign = tl.where(d % 2 == 0, -1.0, 1.0)

    out_offset = (
        output_token * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + d
    )
    tl.store(
        QUERY + out_offset,
        q.to(tl.float32) * cos + sign * q_partner.to(tl.float32) * sin,
    )
    tl.store(
        KEY + out_offset,
        k.to(tl.float32) * cos + sign * k_partner.to(tl.float32) * sin,
    )
    if WRITE_VALUE:
        value = tl.load(
            QKV + qkv_base + 2 * NUM_HEADS * HEAD_DIM + d
        )
        tl.store(VALUE + out_offset, value)


def _tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather tensor across TP ranks along the given dimension."""
    import torch.distributed as dist
    tp = _tp_size()
    if tp <= 1:
        return tensor
    gather_list = [torch.empty_like(tensor) for _ in range(tp)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)


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
        num_heads = self.to_qkv.num_heads
        qkv = self.to_qkv(hidden_states)

        if self.added_kv_proj_dim is not None:
            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            encoder_tokens = encoder_hidden_states.shape[1]
            hidden_tokens = hidden_states.shape[1]
            total_tokens = encoder_tokens + hidden_tokens
            shape = (hidden_states.shape[0], total_tokens, num_heads, self.head_dim)
            query = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            key = torch.empty_like(query)
            value = torch.empty_like(query)

            cos, sin = image_rotary_emb
            _norm_rope_qkv_kernel[(encoder_tokens * num_heads,)](
                encoder_qkv, query, key, value,
                self.norm_added_q.weight, self.norm_added_k.weight, cos, sin,
                encoder_qkv.stride(-2), 0, self.norm_added_q.eps,
                NUM_HEADS=num_heads,
                HEAD_DIM=self.head_dim,
                WRITE_VALUE=True,
                num_warps=2,
            )
            _norm_rope_qkv_kernel[(hidden_tokens * num_heads,)](
                qkv, query, key, value,
                self.norm_q.weight, self.norm_k.weight, cos, sin,
                qkv.stride(-2), encoder_tokens, self.norm_q.eps,
                NUM_HEADS=num_heads,
                HEAD_DIM=self.head_dim,
                WRITE_VALUE=True,
                num_warps=2,
            )
        else:
            hidden_tokens = hidden_states.shape[1]
            shape = (hidden_states.shape[0], hidden_tokens, num_heads, self.head_dim)
            query = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
            key = torch.empty_like(query)
            cos, sin = image_rotary_emb
            _norm_rope_qkv_kernel[(hidden_tokens * num_heads,)](
                qkv, query, key, query,
                self.norm_q.weight, self.norm_k.weight, cos, sin,
                qkv.stride(-2), 0, self.norm_q.eps,
                NUM_HEADS=num_heads,
                HEAD_DIM=self.head_dim,
                WRITE_VALUE=False,
                num_warps=2,
            )
            value = qkv[..., 2 * self.inner_dim:].unflatten(
                -1, (num_heads, self.head_dim)
            )

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
