"""Qwen3-Next full attention with per-head QK-norm, partial RoPE, output gating, KV cache (L2).

GQA attention: 16 query heads, 2 KV heads, head_dim=256.
Q projection outputs 2x: [Q, gate] interleaved per head.
Partial RoPE (25% of head_dim = 64 dims rotated).
Output: attn_output * sigmoid(gate).

The baseline spends fifteen launches per layer -- the fused split/QK-norm/RoPE/
gate-copy kernel, a paged KV store, the trtllm-gen paged prefill, a sigmoid, a
multiply, an output copy, and the int32/cumsum/zeros metadata churn the
FlashInfer wrapper needs -- around two GEMMs.  At the captured shapes (one
prefill sequence of 1, 26, 60, 445 or 16384 tokens) the short ones are pure
per-launch overhead: 36 us of GPU work inside a 110-270 us step.

This candidate keeps the same math and collapses the layer to four kernels, and
for short sequences to a **single host call** (``qwen3_next_attn_fused.cu``):

  1. ``qkv_proj``               one GEMM (as the baseline: torch ``mm``)
  2. ``qk_norm_rope_store``     per-head QK-RMSNorm + partial NeoX RoPE, with K
     and V written straight into the paged HND cache -- no separate store pass,
     and no gate copy at all: the gate stays in the QKV buffer where the
     attention epilogue reads it
  3. ``attn``                   paged causal GQA flash attention (bf16 WMMA)
     whose epilogue divides by the softmax denominator and multiplies by
     sigmoid(gate)
  4. ``o_proj``                 one GEMM

Routing (``_CUDA_ATTN_MAX_SEQ`` / ``_TRITON_ATTN_MAX_TOKENS``):

  * seq_len <= 256    -> ``fused_forward``: all four steps in one host call.
  * seq_len <= 4096   -> ``prep`` (GEMM + step 2) then the Triton flash kernel
    below, which reuses each K/V tile across 64 queries instead of 16, then
    ``proj``.  Three host calls.
  * longer            -> same, but the attention is the frozen L1 trtllm-gen
    paged prefill (it beats both of the above past a few thousand tokens), with
    the output gate folded into one Triton pass.
  * mixed decode/prefill batches, non-NeoX RoPE, non-Blackwell backends, or any
    unexpected dtype/layout fall back to the baseline sequence unchanged.

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
import torch.nn.functional as F
import triton
import triton.language as tl

from ....infra.context import get_attn_backend_config, get_context
from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_size
from ..L1.flash_attn_decode import FlashAttnDecode
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND
from .fused_qk_norm_rope import (
    fused_qk_rmsnorm_rope_gate as _vllm_fused_qk_rmsnorm_rope_gate,
)
from .parallel_linear import QKVParallelLinear, RowParallelLinear

# Whole-layer CUDA extension: the two projections plus the two fused kernels are
# launched from one host call, so a decode-shaped step costs a single Python
# round trip instead of four (the baseline needs fifteen).
_C = lazy_op("qwen3_next_attn_fused", "qwen3_next_attn_fused.cu")


# ---------------------------------------------------------------------------
# 1) QK-RMSNorm + partial NeoX RoPE + paged K/V store, one launch.
#
# Program (BLOCK_T tokens, one "head slot"):  head slots [0, HQ) are Q heads
# (normed + rotated into ``q_out``), [HQ, HQ + HKV) are K heads (normed +
# rotated straight into the paged K cache) and the last HKV slots copy V into
# the paged V cache.  The gate half of each Q head is left untouched in the
# QKV buffer -- the attention epilogue reads it from there.
# ---------------------------------------------------------------------------
@triton.jit
def _qk_norm_rope_store_kernel(
    qkv_ptr,            # [N, qkv_stride] input dtype
    q_out_ptr,          # [N, HQ * D] input dtype
    k_cache_ptr,        # [num_blocks, HKV, PAGE, D]
    v_cache_ptr,
    slot_ptr,           # [N] int32/int64
    q_gain_ptr,         # [D] fp32   (1 + q_norm.weight)
    k_gain_ptr,         # [D] fp32
    cos_sin_ptr,        # [P, 2 * HALF] fp32
    pos_ptr,            # [N] int64
    N,
    qkv_stride,
    cos_stride,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    HALF: tl.constexpr,
    PAGE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    IN_DTYPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    slot_id = tl.program_id(1)

    ts = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = ts < N
    d = tl.arange(0, D)

    if slot_id >= HQ + HKV:
        # ---- V: straight copy into the paged cache -------------------------
        kvh = slot_id - HQ - HKV
        src = qkv_ptr + ts[:, None] * qkv_stride + (HQ * 2 * D + HKV * D + kvh * D) + d[None, :]
        v = tl.load(src, mask=tmask[:, None], other=0.0)
        slot = tl.load(slot_ptr + ts, mask=tmask, other=-1).to(tl.int64)
        dst = (
            (slot // PAGE) * (HKV * PAGE * D)
            + kvh * (PAGE * D)
            + (slot % PAGE) * D
        )
        tl.store(v_cache_ptr + dst[:, None] + d[None, :], v,
                 mask=(tmask & (slot >= 0))[:, None])
        return

    is_k = slot_id >= HQ
    if is_k:
        kvh = slot_id - HQ
        base = qkv_ptr + ts[:, None] * qkv_stride + (HQ * 2 * D + kvh * D)
        gain_ptr = k_gain_ptr
    else:
        kvh = 0
        base = qkv_ptr + ts[:, None] * qkv_stride + slot_id * 2 * D
        gain_ptr = q_gain_ptr

    x = tl.load(base + d[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
    w = tl.load(gain_ptr + d)
    var = tl.sum(x * x, axis=1) * (1.0 / D)
    inv_rms = tl.rsqrt(var + EPS)[:, None]
    xn = (x * inv_rms * w[None, :]).to(IN_DTYPE).to(tl.float32)

    # partial NeoX RoPE on [0, ROT): pairs (i, i + HALF)
    in_rot = d < ROT
    cidx = d % HALF
    pidx = (d + HALF) % ROT
    pos = tl.load(pos_ptr + ts, mask=tmask, other=0).to(tl.int64)
    cs = pos[:, None] * cos_stride
    cos = tl.load(cos_sin_ptr + cs + cidx[None, :],
                  mask=tmask[:, None] & in_rot[None, :], other=1.0)
    sin = tl.load(cos_sin_ptr + cs + HALF + cidx[None, :],
                  mask=tmask[:, None] & in_rot[None, :], other=0.0)
    xp = tl.load(base + pidx[None, :],
                 mask=tmask[:, None] & in_rot[None, :], other=0.0).to(tl.float32)
    wp = tl.load(gain_ptr + pidx, mask=in_rot, other=0.0)
    xpn = (xp * inv_rms * wp[None, :]).to(IN_DTYPE).to(tl.float32)
    sign = tl.where(d < HALF, -1.0, 1.0)[None, :]
    y = (xn * cos + sign * xpn * sin).to(IN_DTYPE)

    if is_k:
        slot = tl.load(slot_ptr + ts, mask=tmask, other=-1).to(tl.int64)
        dst = (
            (slot // PAGE) * (HKV * PAGE * D)
            + kvh * (PAGE * D)
            + (slot % PAGE) * D
        )
        tl.store(k_cache_ptr + dst[:, None] + d[None, :], y,
                 mask=(tmask & (slot >= 0))[:, None])
    else:
        tl.store(q_out_ptr + ts[:, None] * (HQ * D) + slot_id * D + d[None, :], y,
                 mask=tmask[:, None])


def _qk_norm_rope_store(qkv, q_out, k_cache, v_cache, slot_mapping,
                        q_gain, k_gain, cos_sin_cache, positions,
                        num_heads, num_kv_heads, head_dim, rotary_dim,
                        page_size, eps):
    n = qkv.shape[0]
    block_t = 8 if n >= 8 else 1
    grid = (triton.cdiv(n, block_t), num_heads + 2 * num_kv_heads)
    _qk_norm_rope_store_kernel[grid](
        qkv, q_out, k_cache, v_cache, slot_mapping,
        q_gain, k_gain, cos_sin_cache, positions,
        n, qkv.stride(0), cos_sin_cache.stride(0),
        HQ=num_heads, HKV=num_kv_heads, D=head_dim,
        ROT=rotary_dim, HALF=rotary_dim // 2, PAGE=page_size, EPS=eps,
        BLOCK_T=block_t,
        IN_DTYPE=tl.bfloat16 if qkv.dtype == torch.bfloat16 else tl.float16,
        num_warps=4 if block_t >= 8 else 1,
        num_stages=2,
    )


# ---------------------------------------------------------------------------
# 2) Paged causal flash attention with the sigmoid output gate fused in.
# ---------------------------------------------------------------------------
@triton.jit
def _paged_attn_kernel(
    q_ptr,              # [N, HQ * D]
    k_cache_ptr,        # [num_blocks, HKV, PAGE, D]
    v_cache_ptr,
    gate_ptr,           # [N, qkv_stride]  (raw QKV; gate at head * 2D + D)
    out_ptr,            # [N, HQ * D]
    block_table_ptr,    # [S, max_blocks] int32
    seq_lens_ptr,       # [S] int32
    cu_q_ptr,           # [S + 1] int32
    cu_q_base,
    qkv_stride,
    bt_stride,
    sm_scale,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    q_start = tl.load(cu_q_ptr + s) - cu_q_base
    q_len = tl.load(cu_q_ptr + s + 1) - cu_q_base - q_start
    m0 = pid_m * BLOCK_M
    if m0 >= q_len:
        return

    seq_len = tl.load(seq_lens_ptr + s)
    ctx = seq_len - q_len

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    qm = offs_m < q_len
    q = tl.load(
        q_ptr + (q_start + offs_m)[:, None].to(tl.int64) * (HQ * D) + h * D + offs_d[None, :],
        mask=qm[:, None], other=0.0,
    )

    kvh = h // GQA
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    hi = tl.minimum(seq_len, ctx + m0 + BLOCK_M)
    for start_n in range(0, hi, BLOCK_N):
        kv = start_n + tl.arange(0, BLOCK_N)
        kmask = kv < hi
        page = tl.load(block_table_ptr + s * bt_stride + kv // PAGE,
                       mask=kmask, other=0).to(tl.int64)
        koff = page * (HKV * PAGE * D) + kvh * (PAGE * D) + (kv % PAGE) * D
        k = tl.load(k_cache_ptr + koff[:, None] + offs_d[None, :],
                    mask=kmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(kmask[None, :] & ((ctx + offs_m)[:, None] >= kv[None, :]),
                      qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_cache_ptr + koff[:, None] + offs_d[None, :],
                    mask=kmask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    gate = tl.load(
        gate_ptr + (q_start + offs_m)[:, None].to(tl.int64) * qkv_stride
        + h * 2 * D + D + offs_d[None, :],
        mask=qm[:, None], other=0.0,
    ).to(tl.float32)
    acc = acc / (1.0 + tl.exp(-gate))
    tl.store(
        out_ptr + (q_start + offs_m)[:, None].to(tl.int64) * (HQ * D) + h * D + offs_d[None, :],
        acc.to(OUT_DTYPE), mask=qm[:, None],
    )


def _paged_attn(q, k_cache, v_cache, qkv, out, block_table, seq_lens, cu_q,
                cu_q_base, num_seqs, max_query_len, num_heads, num_kv_heads,
                head_dim, page_size, sm_scale, block_m, block_n,
                num_warps=8, num_stages=2):
    if block_table.stride(-1) != 1:
        block_table = block_table.contiguous()
    grid = (triton.cdiv(max_query_len, block_m), num_heads, num_seqs)
    _paged_attn_kernel[grid](
        q, k_cache, v_cache, qkv, out, block_table, seq_lens, cu_q,
        cu_q_base, qkv.stride(0), block_table.stride(0), sm_scale,
        HQ=num_heads, HKV=num_kv_heads, D=head_dim, PAGE=page_size,
        GQA=num_heads // num_kv_heads, BLOCK_M=block_m, BLOCK_N=block_n,
        OUT_DTYPE=tl.bfloat16 if out.dtype == torch.bfloat16 else tl.float16,
        num_warps=num_warps, num_stages=num_stages,
    )


# ---------------------------------------------------------------------------
# Fallback helpers (decode / non-fast paths), kept from the baseline.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_mul_kernel(
    out_ptr,
    gate_ptr,
    n_elements,
    HD: tl.constexpr,     # num_heads * head_dim
    D: tl.constexpr,      # head_dim
    gate_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    out = tl.load(out_ptr + offsets, mask=mask)
    # gate lives inside the raw QKV buffer: row t, head h, [D, 2D)
    t = offsets // HD
    r = offsets % HD
    h = r // D
    c = r % D
    g = tl.load(gate_ptr + t * gate_stride + h * 2 * D + D + c, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, out / (1.0 + tl.exp(-g)), mask=mask)


def _gate_mul_inplace(out: torch.Tensor, qkv: torch.Tensor,
                      num_heads: int, head_dim: int) -> torch.Tensor:
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block = 1024
    _gate_mul_kernel[(triton.cdiv(n_elements, block),)](
        out, qkv, n_elements,
        HD=num_heads * head_dim, D=head_dim, gate_stride=qkv.stride(0),
        BLOCK=block, num_warps=4,
    )
    return out


# Longest ``seq_len`` routed through the one-call CUDA path.  Its attention
# kernel re-stages every K/V tile per 16 queries, so past a few hundred tokens
# the Triton kernel (better tile reuse) and then trtllm-gen win by more than the
# extra Python round trips cost.
_CUDA_ATTN_MAX_SEQ = 256


class Qwen3NextAttention(nn.Module):
    """Full attention with per-head QK-norm, partial RoPE, output gating, and KV cache."""

    # Above this many prefill tokens the frozen L1 trtllm-gen paged prefill
    # kernel is faster than the Triton flash kernel below by more than the
    # ~45 us of host-side metadata + wrapper work its path costs (measured
    # crossover: 445 tokens 12 vs 24 us, 1024 tokens 20 vs 57 us, 2048 tokens
    # 54 vs 171 us).
    _TRITON_ATTN_MAX_TOKENS = 1024

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

        attn_cfg = get_attn_backend_config()
        self.kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm
        self.page_size = attn_cfg.block_size
        self._fused_qk_rope_gate = True
        self._norm_gain_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cuda_static: bool | None = None
        # Resolve (and JIT-build) the extension once, here rather than on the
        # first forward: a build failure then degrades to the Triton path below
        # instead of taking the layer down, and forward skips the lazy-handle
        # indirection.
        try:
            self._ext = _C._load()
        except Exception:  # noqa: BLE001 - no CUDA toolchain / arch mismatch
            self._ext = None
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
        """Adopt the engine's single shared trtllm-gen workspace."""
        if self._use_trtllm:
            self.flash_attn_decode._workspace = workspace
            self.flash_attn_prefill._workspace = workspace

    def _norm_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``1 + weight`` for the QK norms, in fp32, materialized once."""
        if self._norm_gain_cache is None:
            self._norm_gain_cache = (
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
            )
        return self._norm_gain_cache

    def _cuda_ok(self, rotary_emb, positions, md, k_cache) -> bool:
        """Whether the CUDA kernels apply to this layer + step.

        The static half (head/page geometry, weight dtypes) is answered once;
        the per-step half is a handful of dtype identity checks.
        """
        static = self._cuda_static
        if static is None:
            static = (
                self._ext is not None
                and self.head_dim == 256
                and self.page_size == 16
                and rotary_emb.head_dim == 64
                and self.num_heads % self.num_kv_heads == 0
                and self.qkv_proj.weight.dtype == torch.bfloat16
                and self.o_proj.weight.dtype == torch.bfloat16
                and self.qkv_proj.weight.is_contiguous()
                and self.o_proj.weight.is_contiguous()
                and getattr(self.qkv_proj, "bias", None) is None
                and getattr(self.o_proj, "bias", None) is None
                and rotary_emb.cos_sin_cache.dtype == torch.float32
                and rotary_emb.cos_sin_cache.is_contiguous()
                and k_cache.dim() == 4
                and k_cache.shape[1] == self.num_kv_heads
                and k_cache.shape[2] == 16
                and k_cache.shape[3] == 256
                and k_cache.is_contiguous()
            )
            self._cuda_static = static
        return (
            static
            and positions.dtype == torch.int64
            and md.slot_mapping.dtype in (torch.int32, torch.int64)
            and md.block_tables.dtype == torch.int32
            and md.block_tables.stride(-1) == 1
            and md.seq_lens.dtype == torch.int32
            and md.query_start_loc.dtype == torch.int32
        )

    def forward(self, hidden_states, rotary_emb=None, positions=None,
                state_manager=None):
        if rotary_emb is not None:
            self.rotary_emb = rotary_emb
        if self._use_custom_op:
            return torch.ops.fastkernels.qwen3_next_attention(
                hidden_states, positions, self._layer_name,
            )
        return self.forward_impl(hidden_states, positions, state_manager)

    # ------------------------------------------------------------------
    def forward_impl(self, hidden_states, positions=None, state_manager=None):
        ctx = get_context()
        md = ctx.kda_metadata
        if state_manager is None:
            state_manager = ctx.kda_state
        rotary_emb = self.rotary_emb
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        x = hidden_states if hidden_states.dim() == 2 else hidden_states.reshape(
            -1, hidden_states.shape[-1])
        N = x.shape[0]
        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim

        fast = (
            md.num_decodes == 0
            and rotary_emb is not None
            and positions is not None
            and getattr(rotary_emb, "is_neox_style", False)
            and self._use_trtllm
            and N > 0
        )
        if not fast:
            return self._forward_generic(x, positions, state_manager, md)

        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]
        q_gain, k_gain = self._norm_gains()

        cuda_ok = self._cuda_ok(rotary_emb, positions, md, k_cache)

        # ---- one-call CUDA path: the whole layer in a single host call ----
        if cuda_ok and md.max_seq_len <= _CUDA_ATTN_MAX_SEQ:
            return self._ext.fused_forward(
                x, self.qkv_proj.weight, self.o_proj.weight, q_gain, k_gain,
                rotary_emb.cos_sin_cache, positions, k_cache, v_cache,
                md.slot_mapping, md.block_tables, md.seq_lens, md.query_start_loc,
                md.num_prefills, md.max_query_len, md.max_seq_len,
                self.num_heads, self.num_kv_heads,
                self.q_norm.variance_epsilon, self.scaling,
            )

        if cuda_ok:
            qkv, q = self._ext.prep(
                x, self.qkv_proj.weight, q_gain, k_gain, rotary_emb.cos_sin_cache,
                positions, k_cache, v_cache, md.slot_mapping, H, HKV,
                self.q_norm.variance_epsilon,
            )
        else:
            qkv = self.qkv_proj(x)
            q = torch.empty(N, H * D, device=x.device, dtype=x.dtype)
            _qk_norm_rope_store(
                qkv, q, k_cache, v_cache, md.slot_mapping, q_gain, k_gain,
                rotary_emb.cos_sin_cache, positions, H, HKV, D,
                rotary_emb.head_dim, self.page_size,
                self.q_norm.variance_epsilon,
            )

        if N <= self._TRITON_ATTN_MAX_TOKENS:
            o = torch.empty(N, H * D, device=x.device, dtype=x.dtype)
            mq = md.max_query_len
            block_m = 16 if mq <= 16 else (64 if mq < 512 else 128)
            _paged_attn(
                q, k_cache, v_cache, qkv, o, md.block_tables, md.seq_lens,
                md.query_start_loc, 0, md.num_prefills, md.max_query_len,
                H, HKV, D, self.page_size, self.scaling, block_m, 64,
            )
        else:
            np_ = md.num_prefills
            cu_pf = md.query_start_loc
            if cu_pf.dtype != torch.int32:
                cu_pf = cu_pf.to(torch.int32)
            seqs_k = md.seq_lens
            if seqs_k.dtype != torch.int32:
                seqs_k = seqs_k.to(torch.int32)
            cu_k_pf = torch.zeros(np_ + 1, dtype=torch.int32, device=q.device)
            torch.cumsum(seqs_k, dim=0, out=cu_k_pf[1:])
            o = self.flash_attn_prefill(
                q.view(N, H, D), k_cache, v_cache,
                cu_seqlens_q=cu_pf, cu_seqlens_k=cu_k_pf,
                max_seqlen_q=md.max_query_len, max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling, causal=True,
                block_table=md.block_tables,
            )
            o = _gate_mul_inplace(o.view(N, H * D), qkv, H, D)

        o = o.view(N, H * D)
        if cuda_ok:
            return self._ext.proj(o, self.o_proj.weight)
        return self.o_proj(o)

    # ------------------------------------------------------------------
    def _forward_generic(self, x, positions, state_manager, md):
        """Baseline path: mixed decode/prefill batches, non-NeoX RoPE, ..."""
        rotary_emb = self.rotary_emb
        N = x.shape[0]

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
        if use_fused:
            q_gain, k_gain = self._norm_gains()
            q, k, gate = _vllm_fused_qk_rmsnorm_rope_gate(
                q_gate, k, q_gain, k_gain, rotary_emb.cos_sin_cache,
                positions.reshape(-1), self.q_norm.variance_epsilon,
                self.num_heads, self.num_kv_heads, self.head_dim,
                rotary_emb.head_dim,
            )
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_kv_heads, self.head_dim)
            gate = gate.view(N, self.num_heads, self.head_dim)
        else:
            q_gate = q_gate.view(N, self.num_heads, 2 * self.head_dim)
            q = q_gate[:, :, :self.head_dim].contiguous()
            gate = q_gate[:, :, self.head_dim:].contiguous()
            k = k.view(N, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(
                N, self.num_heads, self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(
                N, self.num_kv_heads, self.head_dim)
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

        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]
        self.store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)

        out = torch.empty(
            N, self.num_heads, self.head_dim,
            device=x.device, dtype=x.dtype,
        )

        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills

        if nd > 0:
            out[:ndt] = self.flash_attn_decode(
                q[:ndt], k_cache, v_cache,
                cache_seqlens=md.seq_lens[:nd].to(torch.int32),
                block_table=md.block_tables[:nd],
                softmax_scale=self.scaling, causal=True,
                max_seq_len=md.max_seq_len,
            )

        if np_ > 0:
            cu_pf = (md.query_start_loc[nd:] - md.query_start_loc[nd]).to(
                torch.int32,
            )
            seqs_k = md.seq_lens[nd:]
            cu_k_pf = torch.zeros(np_ + 1, dtype=torch.int32, device=q.device)
            cu_k_pf[1:] = torch.cumsum(seqs_k.to(torch.int32), dim=0)
            out[ndt:] = self.flash_attn_prefill(
                q[ndt:], k_cache, v_cache,
                cu_seqlens_q=cu_pf, cu_seqlens_k=cu_k_pf,
                max_seqlen_q=md.max_query_len, max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling, causal=True,
                block_table=md.block_tables[nd:],
            )

        o = out * torch.sigmoid(gate)
        o = o.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(o)
