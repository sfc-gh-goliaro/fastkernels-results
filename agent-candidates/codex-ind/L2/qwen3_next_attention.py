"""Qwen3-Next full attention with per-head QK-norm, partial RoPE, output gating, KV cache (L2).

GQA attention: 16 query heads, 2 KV heads, head_dim=256.
Q projection outputs 2x: [Q, gate] interleaved per head.
Partial RoPE (25% of head_dim = 64 dims rotated).
Output: attn_output * sigmoid(gate).

KV cache is stored in the engine's paged state manager so Qwen3-Next can
run batched prefill/decode instead of one Python call per sequence.

Uses the existing flash-attention prefill/decode wrappers, ``GemmaRMSNorm``,
``StoreKVCache``, and the canonical TP linears in ``parallel_linear``.

Weight names match HuggingFace checkpoint:
  self_attn.q_proj.weight   [2 * num_heads * head_dim, hidden_size]  (Q + gate)
  self_attn.k_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.v_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.o_proj.weight   [hidden_size, num_heads * head_dim]
  self_attn.q_norm.weight   [head_dim]
  self_attn.k_norm.weight   [head_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.context import get_attn_backend_config, get_context
from ....infra.tp import _tp_size
from ..L1.flash_attn_decode import FlashAttnDecode
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND
from .fused_qk_norm_rope import (
    fused_qk_rmsnorm_rope_gate as _vllm_fused_qk_rmsnorm_rope_gate,
)
from .parallel_linear import QKVParallelLinear, RowParallelLinear


@triton.jit
def _gate_mul_inplace_kernel(
    out_ptr,
    gate_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    out = tl.load(out_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offsets, out * gate, mask=mask)


def _gate_mul_inplace(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block = 1024
    _gate_mul_inplace_kernel[(triton.cdiv(n_elements, block),)](
        out,
        gate,
        n_elements,
        BLOCK=block,
    )
    return out


@triton.jit
def _single_token_kv_gate_kernel(
    q_gate_ptr,
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    out_ptr,
    cache_stride_p: tl.constexpr,
    EPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    head = tl.program_id(0)
    offsets = tl.arange(0, HEAD_DIM)
    if head < NUM_Q_HEADS:
        value_head = head // GROUP_SIZE
        value = tl.load(value_ptr + value_head * HEAD_DIM + offsets)
        gate = tl.load(
            q_gate_ptr + head * 2 * HEAD_DIM + HEAD_DIM + offsets,
        ).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp(-gate))
        tl.store(out_ptr + head * HEAD_DIM + offsets, value * gate)
    else:
        kv_head = head - NUM_Q_HEADS
        key_base = key_ptr + kv_head * HEAD_DIM
        x = tl.load(key_base + offsets).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / HEAD_DIM
        inv_rms = tl.rsqrt(variance + EPS)
        weight = tl.load(k_weight_ptr + offsets).to(tl.float32)
        x_norm = (x * inv_rms * weight).to(INPUT_DTYPE).to(tl.float32)

        half_rotary = ROTARY_DIM // 2
        rotary_offset = offsets % half_rotary
        partner_offset = tl.where(
            offsets < half_rotary,
            offsets + half_rotary,
            offsets - half_rotary,
        )
        partner = tl.load(
            key_base + partner_offset,
            mask=offsets < ROTARY_DIM,
            other=0.0,
        ).to(tl.float32)
        partner_weight = tl.load(
            k_weight_ptr + partner_offset,
            mask=offsets < ROTARY_DIM,
            other=0.0,
        ).to(tl.float32)
        partner_norm = (
            partner * inv_rms * partner_weight
        ).to(INPUT_DTYPE).to(tl.float32)

        position = tl.load(positions_ptr).to(tl.int64)
        cache_offset = position * cache_stride_p
        cos = tl.load(
            cos_sin_cache_ptr + cache_offset + rotary_offset,
            mask=offsets < ROTARY_DIM,
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + cache_offset + half_rotary + rotary_offset,
            mask=offsets < ROTARY_DIM,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(
            offsets < half_rotary,
            x_norm * cos - partner_norm * sin,
            x_norm * cos + partner_norm * sin,
        )
        key_out = tl.where(offsets < ROTARY_DIM, rotated, x_norm)

        slot = tl.load(slot_mapping_ptr).to(tl.int64)
        if slot < 0:
            return
        block = slot // PAGE_SIZE
        slot_in_block = slot % PAGE_SIZE
        dst = (
            block * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
            + kv_head * PAGE_SIZE * HEAD_DIM
            + slot_in_block * HEAD_DIM
            + offsets
        )
        tl.store(k_cache_ptr + dst, key_out)
        value = tl.load(value_ptr + kv_head * HEAD_DIM + offsets)
        tl.store(v_cache_ptr + dst, value)


def _single_token_kv_gate(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    page_size: int,
) -> torch.Tensor:
    out = torch.empty(
        (1, num_heads, head_dim), dtype=value.dtype, device=value.device,
    )
    _single_token_kv_gate_kernel[(num_heads + num_kv_heads,)](
        q_gate,
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        k_weight,
        cos_sin_cache,
        positions,
        out,
        cos_sin_cache.stride(0),
        EPS=eps,
        NUM_Q_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        PAGE_SIZE=page_size,
        INPUT_DTYPE=tl.bfloat16 if value.dtype == torch.bfloat16 else tl.float16,
        num_warps=4,
    )
    return out


@triton.jit
def _fused_qk_rope_gate_store_kernel(
    q_gate_ptr,
    key_ptr,
    value_ptr,
    q_out_ptr,
    key_out_ptr,
    gate_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    key_stride_t,
    value_stride_t,
    q_out_stride_t,
    key_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    EPS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    STORE_KEY_OUT: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_key = head >= NUM_Q_HEADS
    local_head = tl.where(is_key, head - NUM_Q_HEADS, head)
    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
    block = slot // PAGE_SIZE
    slot_in_block = slot % PAGE_SIZE
    key_out_base = (
        key_out_ptr + token * key_out_stride_t + local_head * HEAD_DIM
    )

    if is_key:
        in_base = key_ptr + token * key_stride_t + local_head * HEAD_DIM
        weight_ptr = k_weight_ptr
        out_base = (
            k_cache_ptr
            + block * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
            + local_head * PAGE_SIZE * HEAD_DIM
            + slot_in_block * HEAD_DIM
        )
    else:
        in_base = (
            q_gate_ptr
            + token * q_gate_stride_t
            + local_head * 2 * HEAD_DIM
        )
        weight_ptr = q_weight_ptr
        out_base = (
            q_out_ptr + token * q_out_stride_t + local_head * HEAD_DIM
        )

    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(in_base + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    x_norm = (x * inv_rms * weight).to(INPUT_DTYPE).to(tl.float32)
    valid_store = (~is_key) | (slot >= 0)
    tl.store(out_base + offsets, x_norm, mask=valid_store)
    if STORE_KEY_OUT:
        tl.store(key_out_base + offsets, x_norm, mask=is_key)

    half_rotary = ROTARY_DIM // 2
    rotary_offset = tl.arange(0, ROT_HALF_BLOCK)
    rotary_mask = rotary_offset < half_rotary
    x1 = tl.load(in_base + rotary_offset, mask=rotary_mask).to(tl.float32)
    x2 = tl.load(
        in_base + half_rotary + rotary_offset, mask=rotary_mask,
    ).to(tl.float32)
    w1 = tl.load(weight_ptr + rotary_offset, mask=rotary_mask).to(tl.float32)
    w2 = tl.load(
        weight_ptr + half_rotary + rotary_offset, mask=rotary_mask,
    ).to(tl.float32)
    x1 = (x1 * inv_rms * w1).to(INPUT_DTYPE).to(tl.float32)
    x2 = (x2 * inv_rms * w2).to(INPUT_DTYPE).to(tl.float32)

    position = tl.load(positions_ptr + token).to(tl.int64)
    cache_offset = position * cache_stride_p
    cos = tl.load(
        cos_sin_cache_ptr + cache_offset + rotary_offset,
        mask=rotary_mask,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rotary_offset,
        mask=rotary_mask,
    ).to(tl.float32)
    tl.store(
        out_base + rotary_offset,
        x1 * cos - x2 * sin,
        mask=valid_store & rotary_mask,
    )
    if STORE_KEY_OUT:
        tl.store(
            key_out_base + rotary_offset,
            x1 * cos - x2 * sin,
            mask=is_key & rotary_mask,
        )
    tl.store(
        out_base + half_rotary + rotary_offset,
        x2 * cos + x1 * sin,
        mask=valid_store & rotary_mask,
    )
    if STORE_KEY_OUT:
        tl.store(
            key_out_base + half_rotary + rotary_offset,
            x2 * cos + x1 * sin,
            mask=is_key & rotary_mask,
        )

    if is_key:
        value = tl.load(
            value_ptr + token * value_stride_t + local_head * HEAD_DIM + offsets,
        )
        value_out = (
            v_cache_ptr
            + block * NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM
            + local_head * PAGE_SIZE * HEAD_DIM
            + slot_in_block * HEAD_DIM
        )
        tl.store(value_out + offsets, value, mask=slot >= 0)
    else:
        gate = tl.load(in_base + HEAD_DIM + offsets)
        gate_out = (
            gate_out_ptr
            + token * gate_out_stride_t
            + local_head * HEAD_DIM
        )
        tl.store(gate_out + offsets, gate)


def _fused_qk_rope_gate_store(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    page_size: int,
    store_key_out: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    n_tokens = q_gate.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim),
        dtype=q_gate.dtype,
        device=q_gate.device,
    )
    key_out = (
        torch.empty(
            (n_tokens, num_kv_heads * head_dim),
            dtype=key.dtype,
            device=key.device,
        )
        if store_key_out
        else None
    )
    gate_out = torch.empty_like(q_out)
    _fused_qk_rope_gate_store_kernel[
        (n_tokens, num_q_heads + num_kv_heads)
    ](
        q_gate,
        key,
        value,
        q_out,
        key_out if key_out is not None else q_out,
        gate_out,
        k_cache,
        v_cache,
        slot_mapping,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        key.stride(0),
        value.stride(0),
        q_out.stride(0),
        key_out.stride(0) if key_out is not None else q_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        EPS=eps,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        ROT_HALF_BLOCK=triton.next_power_of_2(rotary_dim // 2),
        PAGE_SIZE=page_size,
        INPUT_DTYPE=(
            tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16
        ),
        STORE_KEY_OUT=store_key_out,
        num_warps=4,
        num_stages=2,
    )
    return q_out, key_out, gate_out


@triton.jit
def _small_causal_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    out_ptr,
    q_stride_t,
    k_stride_t,
    v_stride_t,
    gate_stride_t,
    out_stride_t,
    N: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    q_head = tl.program_id(0)
    kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr
        + rows[:, None] * q_stride_t
        + q_head * HEAD_DIM
        + dims[None, :],
        mask=rows[:, None] < N,
        other=0.0,
    )
    key = tl.load(
        k_ptr
        + cols[None, :] * k_stride_t
        + kv_head * HEAD_DIM
        + dims[:, None],
        mask=cols[None, :] < N,
        other=0.0,
    )
    scores = tl.dot(q, key) * SCALE
    causal_mask = (
        (rows[:, None] < N)
        & (cols[None, :] < N)
        & (cols[None, :] <= rows[:, None])
    )
    scores = tl.where(causal_mask, scores, -float("inf"))
    scores = scores - tl.max(scores, axis=1)[:, None]
    probabilities = tl.exp(scores)
    probabilities = probabilities / tl.sum(probabilities, axis=1)[:, None]

    value = tl.load(
        v_ptr
        + cols[:, None] * v_stride_t
        + kv_head * HEAD_DIM
        + dims[None, :],
        mask=cols[:, None] < N,
        other=0.0,
    )
    result = tl.dot(probabilities.to(INPUT_DTYPE), value)
    gate = tl.load(
        gate_ptr
        + rows[:, None] * gate_stride_t
        + q_head * HEAD_DIM
        + dims[None, :],
        mask=rows[:, None] < N,
        other=0.0,
    ).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(
        out_ptr
        + rows[:, None] * out_stride_t
        + q_head * HEAD_DIM
        + dims[None, :],
        result * gate,
        mask=rows[:, None] < N,
    )


def _small_causal_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    n, num_q_heads, head_dim = q.shape
    num_kv_heads = key.shape[1]
    out = torch.empty_like(q)
    _small_causal_attention_kernel[(num_q_heads,)](
        q,
        key,
        value,
        gate,
        out,
        q.stride(0),
        key.stride(0),
        value.stride(0),
        gate.stride(0),
        out.stride(0),
        N=n,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=triton.next_power_of_2(n),
        SCALE=scale,
        INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        num_warps=8,
        num_stages=2,
    )
    return out


class Qwen3NextAttention(nn.Module):
    """Full attention with per-head QK-norm, partial RoPE, output gating, and KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        rms_norm_eps: float = 1e-6,
        reduce_output: bool = True,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp if num_key_value_heads % tp == 0 else num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # QKV projection: Q outputs 2x heads (Q + gate)
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads * 2,  # doubled for output gate
            num_key_value_heads,
        )

        # ``reduce_output=False`` defers the all-reduce to the decoder layer's
        # next norm, which fuses the two.
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            reduce_results=reduce_output,
        )

        # Per-head QK norms (GemmaRMSNorm)
        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)

        # Qwen3-Next's full-attention layers use head_dim=256.  vLLM 0.26 runs
        # them on FlashInfer with an HND cache ("Using FLASHINFER attention
        # backend" / "Using HND KV cache layout for FLASHINFER" on B200), so
        # follow the same per-device backend selection the generic
        # ``Attention`` layer uses.  FlashAttention is not a substitute here:
        # FA4's SM100 head_dim=256 forward rejects seqused_k/seqused_q, which
        # the paged decode path requires.
        attn_cfg = get_attn_backend_config()
        self.kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm
        # vLLM collapses the gated split + QK-RMSNorm + partial NeoX RoPE +
        # gate copy into one Triton launch
        # (``Qwen3NextAttention.use_fused_qk_norm_rope_gate``). Unfused that is
        # nine kernels per attention layer -- two gate/q slices, two norms, two
        # rotary slices, the rotary op and two cats -- which at batch 1 is pure
        # launch overhead.
        self._fused_qk_rope_gate = True
        self._norm_gain_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._use_custom_op = False
        self._layer_name = ""
        self.rotary_emb = None
        if self._use_trtllm:
            from ..L1.flashinfer_decode import TRTLLMDecode
            from ..L1.flashinfer_prefill import TRTLLMPrefill

            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            self.flash_attn_prefill = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
        else:
            self.store_kvcache = StoreKVCache()
            self.flash_attn_prefill = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )

    def set_trtllm_workspace(self, workspace: torch.Tensor) -> None:
        """Adopt the engine's single shared trtllm-gen workspace.

        Without this each layer keeps the 512 MiB buffer it allocated in
        ``__init__``; Qwen3-Next has one MHA layer per 4 decoder layers, so
        that would waste several GiB.
        """
        if self._use_trtllm:
            self.flash_attn_decode._workspace = workspace
            self.flash_attn_prefill._workspace = workspace

    def _norm_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``1 + weight`` for the QK norms, in fp32, materialized once.

        vLLM recomputes ``q_norm.weight.float() + 1.0`` per call and lets
        Inductor hoist it; in eager that would be two extra launches on every
        one of the 12 attention layers. The values are constants after weight
        loading, so caching them is exact.
        """
        if self._norm_gain_cache is None:
            self._norm_gain_cache = (
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
            )
        return self._norm_gain_cache

    def forward(self, hidden_states, rotary_emb=None, positions=None,
                state_manager=None):
        if rotary_emb is not None:
            self.rotary_emb = rotary_emb
        if self._use_custom_op:
            return torch.ops.fastkernels.qwen3_next_attention(
                hidden_states, positions, self._layer_name,
            )
        return self.forward_impl(hidden_states, positions, state_manager)

    def forward_impl(self, hidden_states, positions=None, state_manager=None):
        md = get_context().kda_metadata
        if state_manager is None:
            state_manager = get_context().kda_state
        rotary_emb = self.rotary_emb
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        N = x.shape[0]
        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills
        empty_single_prefill = (
            nd == 0
            and np_ == 1
            and md.max_query_len == md.max_seq_len == N
        )
        use_dense_prefill = empty_single_prefill and 128 < N < 1024
        use_small_attention = empty_single_prefill and 1 < N <= 128

        qkv = self.qkv_proj(x)
        q_gate_size = self.num_heads * 2 * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

        use_fused = (
            self._fused_qk_rope_gate
            and rotary_emb is not None
            and positions is not None
            and getattr(rotary_emb, "is_neox_style", False)
        )
        k_cache = state_manager.k_cache[self.layer_idx]
        v_cache = state_manager.v_cache[self.layer_idx]
        if (
            N == 1
            and empty_single_prefill
            and use_fused
            and self._use_trtllm
        ):
            _, k_gain = self._norm_gains()
            o = _single_token_kv_gate(
                q_gate,
                k,
                v,
                k_cache,
                v_cache,
                md.slot_mapping,
                k_gain,
                rotary_emb.cos_sin_cache,
                positions.reshape(-1),
                self.k_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                rotary_emb.head_dim,
                self.store_kvcache.page_size,
            )
            return self.o_proj(o.reshape(1, self.num_heads * self.head_dim))

        stored_kv = False
        if (
            use_fused
            and self._use_trtllm
            and empty_single_prefill
            and N < 1024
        ):
            q_gain, k_gain = self._norm_gains()
            q, dense_k, gate = _fused_qk_rope_gate_store(
                q_gate,
                k,
                v,
                q_gain,
                k_gain,
                rotary_emb.cos_sin_cache,
                positions.reshape(-1),
                md.slot_mapping,
                k_cache,
                v_cache,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                rotary_emb.head_dim,
                self.store_kvcache.page_size,
                use_dense_prefill or use_small_attention,
            )
            q = q.view(N, self.num_heads, self.head_dim)
            if dense_k is not None:
                k = dense_k.view(N, self.num_kv_heads, self.head_dim)
            gate = gate.view(N, self.num_heads, self.head_dim)
            stored_kv = True
        elif use_fused:
            q_gain, k_gain = self._norm_gains()
            q, k, gate = _vllm_fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                q_gain,
                k_gain,
                rotary_emb.cos_sin_cache,
                positions.reshape(-1),
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                rotary_emb.head_dim,
            )
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_kv_heads, self.head_dim)
            gate = gate.view(N, self.num_heads, self.head_dim)
        else:
            # Split Q and gate
            q_gate = q_gate.view(N, self.num_heads, 2 * self.head_dim)
            q = q_gate[:, :, :self.head_dim].contiguous()
            gate = q_gate[:, :, self.head_dim:].contiguous()

            k = k.view(N, self.num_kv_heads, self.head_dim)

            # Per-head QK-norm (applied before RoPE)
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(
                N, self.num_heads, self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(
                N, self.num_kv_heads, self.head_dim)

            # Partial RoPE (only rotates first rotary_dim dimensions)
            if rotary_emb is not None and positions is not None:
                pos_flat = (
                    positions.reshape(-1) if positions.dim() > 1 else positions
                )
                rotary_dim = rotary_emb.head_dim
                q_rot, q_pass = (
                    q[..., :rotary_dim].contiguous(), q[..., rotary_dim:],
                )
                k_rot, k_pass = (
                    k[..., :rotary_dim].contiguous(), k[..., rotary_dim:],
                )
                q_rot, k_rot = rotary_emb(pos_flat, q_rot, k_rot)
                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1)

        v = v.view(N, self.num_kv_heads, self.head_dim)

        if not stored_kv:
            self.store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)

        output_is_gated = False
        if nd > 0:
            out = torch.empty(
                N,
                self.num_heads,
                self.head_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            out[:ndt] = self.flash_attn_decode(
                q[:ndt],
                k_cache,
                v_cache,
                cache_seqlens=md.seq_lens[:nd].to(torch.int32),
                block_table=md.block_tables[:nd],
                softmax_scale=self.scaling,
                causal=True,
                max_seq_len=md.max_seq_len,
            )

        if np_ > 0:
            if empty_single_prefill:
                # Empty-context, one-sequence prefill. The query cumulative
                # lengths already describe both Q and K.
                cu_pf = md.query_start_loc
                cu_k_pf = cu_pf
            else:
                cu_pf = (md.query_start_loc[nd:] - md.query_start_loc[nd]).to(
                    torch.int32,
                )
                seqs_k = md.seq_lens[nd:]
                cu_k_pf = torch.zeros(
                    np_ + 1, dtype=torch.int32, device=q.device,
                )
                cu_k_pf[1:] = torch.cumsum(seqs_k.to(torch.int32), dim=0)
            if use_small_attention:
                prefill_out = _small_causal_attention(
                    q, k, v, gate, self.scaling,
                )
                output_is_gated = True
            else:
                prefill_out = self.flash_attn_prefill(
                    q if ndt == 0 else q[ndt:],
                    k if use_dense_prefill else k_cache,
                    v if use_dense_prefill else v_cache,
                    cu_seqlens_q=cu_pf,
                    cu_seqlens_k=cu_k_pf,
                    max_seqlen_q=md.max_query_len,
                    max_seqlen_k=md.max_seq_len,
                    softmax_scale=self.scaling,
                    causal=True,
                    block_table=(
                        None
                        if use_dense_prefill
                        else md.block_tables if nd == 0 else md.block_tables[nd:]
                    ),
                )
            if nd == 0:
                out = prefill_out
            else:
                out[ndt:] = prefill_out

        # Fuse sigmoid and multiply and reuse the attention output allocation.
        o = out if output_is_gated else _gate_mul_inplace(out, gate)

        # Output projection
        o = o.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(o)
