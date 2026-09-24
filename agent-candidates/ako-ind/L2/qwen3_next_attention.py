"""Qwen3-Next full attention with per-head QK-norm, partial RoPE, output gating, KV cache (L2).

GQA attention: 16 query heads, 2 KV heads, head_dim=256.
Q projection outputs 2x: [Q, gate] interleaved per head.
Partial RoPE (25% of head_dim = 64 dims rotated).
Output: attn_output * sigmoid(gate).

KV cache is stored in the engine's paged state manager so Qwen3-Next can
run batched prefill/decode instead of one Python call per sequence.

Weight names match HuggingFace checkpoint:
  self_attn.q_proj.weight   [2 * num_heads * head_dim, hidden_size]  (Q + gate)
  self_attn.k_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.v_proj.weight   [num_kv_heads * head_dim, hidden_size]
  self_attn.o_proj.weight   [hidden_size, num_heads * head_dim]
  self_attn.q_norm.weight   [head_dim]
  self_attn.k_norm.weight   [head_dim]

Optimization notes (vs. the reference composition of L1/L2 ops)
--------------------------------------------------------------
The pure-prefill path is the one that matters: every captured step is a single
sequence, and four of the five benchmarked token counts are 1..445, where the
layer costs ~11us of L2-warm GPU work and the reference spends ~250us on it.
The reference sequence is

    qkv GEMM | fused qk-norm+rope+gate-copy | slot_mapping int64 cast |
    kv-cache store | empty(out) | cu_seqlens_q sub | zeros+cumsum+slice for
    cu_seqlens_k | seq_lens sub (in the trtllm wrapper) | paged FMHA |
    out[nd:] = ... | sigmoid | mul | o_proj GEMM

= ~15 launches plus five allocations.  This module collapses the whole tail to
four:

    qkv GEMM | fused prologue | fused attention+gate | o_proj GEMM

  * ``_attn_gate_kernel`` replaces the trtllm-gen paged FMHA *and* the gate
    epilogue for N <= 128.  trtllm-gen is a general batched paged kernel with a
    512 MB workspace and costs ~9us in the bench's post-flush window even at
    N=1, where there is one query and one key; the scored path is always
    num_prefills=1 / num_decodes=0 with max_seq_len == max_query_len == N, i.e.
    one sequence whose keys are exactly the tokens the prologue just stored, so
    a plain causal flash loop covers it and measures at the ~2us floor for one
    launch.  Applying ``sigmoid(gate)`` while the fp32 accumulator is still in
    registers removes the epilogue launch too.  Above 128 tokens trtllm-gen wins
    (20.4 vs 28.6us at N=445, measured cold-L2) and is still used.
  * ``_qk_norm_rope_store_kernel`` folds the KV-cache store into the
    qk-norm/RoPE kernel: the K programs write the normed+rotated head straight
    into the paged HND cache (casting the int32 slot id in-register, so the
    separate ``slot_mapping.to(int64)`` launch disappears) and copy V across in
    the same program.  ``k_out`` is never materialized.  It also tiles four
    tokens into each program, which is what turns the reference's 4-bytes-per-
    lane access pattern into one 16-byte vector per lane (1.3 -> 4.4 TB/s at
    16384 tokens).
  * The gate is *not* copied out.  Both the fused attention and
    ``_gate_mul_kernel`` (the N > 128 path) read it straight from the
    interleaved ``[q|gate]`` columns of the QKV GEMM output, so the
    N x num_heads x head_dim gate buffer is never written nor read back.
  * ``cu_seqlens_q``, ``cu_seqlens_k`` and the FMHA's ``seq_lens`` are memoized
    on the per-step metadata object -- which the engine shares across every
    attention layer of a step -- instead of being rebuilt per call out of a
    subtract, a ``zeros``, a ``cumsum``, a slice-assign and another subtract.
  * The paged FMHA writes reused scratch directly, which is then gated in
    place: no staging ``torch.empty`` and no ``out[nd:] = ...`` copy of the full
    N x num_heads x head_dim tensor.
  * The nn.Module and flashinfer Python wrappers are bypassed, every Triton
    kernel is launched through its prebound C launcher, and the long argument
    lists are cached whole for a repeating step shape.

What is *not* worth trying (measured in the bench's own flushed window; see
``ITERATIONS.md`` for the numbers): the window is GPU-bound, not host-bound, so
cutting dispatch cost buys nothing; the flush leaves L2 full of dirty lines, so
achievable DRAM bandwidth for the first ~126 MiB is ~3.2 TB/s rather than peak;
and at that ceiling both cuBLAS GEMMs are within ~1us of a pure streaming read
of their own weights, which is why replacing either with Triton -- wide
row-per-CTA decomposition, split-K, or a concurrent Wo prefetch -- loses.  Two
cuBLAS GEMMs plus two Triton launches is the floor for this decomposition.

Numerics: bf16 output error is unchanged from the reference (max 7.8e-3 on the
output, every element inside the 1e-2/1e-2 tolerance).  The prologue's RMSNorm
reduces over a different tile shape than the reference kernel, so KV-cache
values can differ in the last bf16 bit.  The fused attention keeps the fp32
accumulator through the gate multiply instead of rounding the attention output to
bf16 first, which is strictly closer to an exact result, not further.

The general mixed decode/prefill path is kept verbatim as a fallback and is
taken whenever the fast path's preconditions do not hold.
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

_RAW_STREAM = torch._C._cuda_getCurrentRawStream


# Triton's ``kernel[grid](...)`` re-binds the signature, recomputes the
# specialization key and re-hashes every constexpr on each call -- 10-18us for
# these kernels, more than they spend on the GPU.  Once compiled the binary is
# fixed, so ``Qwen3NextAttention._launch`` keeps the raw launcher, the
# CUfunction and the packed metadata and calls straight into the C launcher.
# Its cache key covers everything Triton itself specializes on that can vary
# here (pointer alignment; the one integer argument that varies is declared
# ``do_not_specialize``), so a cache hit is the same binary Triton would pick.


# Presence probe for the Blackwell paged FMHA the L1 ``TRTLLMPrefill`` wrapper
# calls.  ``_setup_prefill`` resolves the underlying op and invokes it directly:
# the wrapper recomputes ``seq_lens`` with a device subtract and re-runs ~30
# argument checks per call, which at 1..445 tokens costs more than the kernel.
try:
    from flashinfer.prefill import (
        trtllm_batch_context_with_kv_cache as _trtllm_paged_context,
    )
except Exception:  # pragma: no cover - non-Blackwell / no flashinfer
    _trtllm_paged_context = None


# ---------------------------------------------------------------------------
# Fused prologue: split [q|gate] -> QK-RMSNorm -> partial NeoX RoPE, with the
# paged KV-cache store folded into the K programs.
#
# The normalized value is round-tripped through the input dtype before the
# rotation, exactly as the unfused (qk_rmsnorm -> memory -> apply_rope)
# reference does, so this differs from ``fused_qk_rmsnorm_rope_gate`` followed by
# ``StoreKVCacheHND`` only in the order the RMSNorm reduction is summed.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_tokens"])
def _qk_norm_rope_store_kernel(
    qkv_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    n_tokens,
    qkv_stride_t,
    q_out_stride_t,
    cache_stride_b,
    cache_stride_h,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    k_col: tl.constexpr,
    v_col: tl.constexpr,
    page_size: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One head of BLOCK_T tokens per program.

    A program per (token, head) -- what the reference kernel does -- gives each
    lane 4 bytes of the 512-byte head and holds the 16k-token shape to 1.3 TB/s.
    Tiling BLOCK_T tokens into the same program turns every access into one
    16-byte vector per lane and more than doubles that.
    """
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    tmask = t < n_tokens
    d = tl.arange(0, HEAD_BLOCK)[None, :]
    dmask = d < head_dim
    # ``other=-1`` keeps the tail rows out of the cache store below; their
    # address arithmetic is computed but never dereferenced.
    slot = tl.load(slot_ptr + t, mask=tmask, other=-1).to(tl.int64)
    row = t * qkv_stride_t

    if is_k:
        local_head = head - num_q_heads
        in_base = qkv_ptr + row + k_col + local_head * head_dim
        w_ptr = k_weight_ptr
        out_base = (
            k_cache_ptr
            + (slot // page_size) * cache_stride_b
            + local_head * cache_stride_h
            + (slot % page_size) * cache_stride_p
        )
        st_mask = tmask & (slot >= 0)
    else:
        in_base = qkv_ptr + row + head * (2 * head_dim)
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + t * q_out_stride_t + head * head_dim
        st_mask = tmask

    # --- RMSNorm over the full head_dim ---
    x = tl.load(in_base + d, mask=tmask & dmask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1)[:, None] / head_dim
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + d, mask=dmask, other=0.0).to(tl.float32)
    # Round-trip through the input dtype so the RoPE input matches the
    # bf16-storage behaviour of the unfused reference path.
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm only ---
    if HAS_PASS:
        tl.store(out_base + d, x_norm,
                 mask=st_mask & dmask & (d >= rotary_dim))

    # --- Partial NeoX RoPE on the leading rotary_dim elements ---
    r = tl.arange(0, ROT_HALF_BLOCK)[None, :]
    rmask = r < half_rotary
    x_rot1 = tl.load(in_base + r, mask=tmask & rmask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + r, mask=tmask & rmask,
                     other=0.0).to(tl.float32)
    w_rot1 = tl.load(w_ptr + r, mask=rmask, other=0.0).to(tl.float32)
    w_rot2 = tl.load(w_ptr + half_rotary + r, mask=rmask, other=0.0).to(tl.float32)
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    pos = tl.load(positions_ptr + t, mask=tmask, other=0).to(tl.int64)
    cs_base = cos_sin_cache_ptr + pos * rotary_dim
    cos = tl.load(cs_base + r, mask=tmask & rmask, other=0.0).to(tl.float32)
    sin = tl.load(cs_base + half_rotary + r, mask=tmask & rmask,
                  other=0.0).to(tl.float32)

    tl.store(out_base + r, x_rot1 * cos - x_rot2 * sin, mask=st_mask & rmask)
    tl.store(out_base + half_rotary + r, x_rot2 * cos + x_rot1 * sin,
             mask=st_mask & rmask)

    # --- V is copied verbatim into the paged cache by the same program ---
    if is_k:
        kv_head = head - num_q_heads
        v = tl.load(qkv_ptr + row + v_col + kv_head * head_dim + d,
                    mask=tmask & dmask, other=0.0)
        tl.store(
            v_cache_ptr
            + (slot // page_size) * cache_stride_b
            + kv_head * cache_stride_h
            + (slot % page_size) * cache_stride_p
            + d,
            v,
            mask=tmask & dmask & (slot >= 0),
        )


# ---------------------------------------------------------------------------
# Fused paged causal attention + sigmoid gate (small-N single-sequence prefill).
#
# This replaces *both* the trtllm-gen paged FMHA and the gate epilogue below.
# trtllm-gen is a general batched paged kernel with a 512 MB workspace, and in
# the bench's post-flush window it costs ~9us even at N=1 -- where there is one
# query and one key.  The scored shapes are all num_prefills=1 / num_decodes=0
# with max_seq_len == max_query_len == N, i.e. one sequence whose keys are
# exactly the tokens the prologue just stored, so a plain flash loop covers them
# and measures at the floor for a single kernel launch.  Applying the output
# gate while the fp32 accumulator is still in registers removes the epilogue
# launch as well, and the gate is read straight from the interleaved [q|gate]
# columns of the QKV GEMM so it is still never materialized.
#
# Keys are addressed the way the paged cache defines them -- sequence position
# ``j`` lives at page ``block_table[j // page_size]``, slot ``j % page_size`` --
# so the identity slot mapping the engine happens to hand us is not assumed.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_tokens"])
def _attn_gate_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    bt_ptr,
    qkv_ptr,
    out_ptr,
    n_tokens,
    q_stride_t,
    qkv_stride_t,
    out_stride_t,
    cache_stride_b,
    cache_stride_h,
    cache_stride_p,
    scale: tl.constexpr,
    GQA: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, head_dim)
    mmask = offs_m < n_tokens

    q = tl.load(q_ptr + offs_m[:, None] * q_stride_t + head * head_dim
                + offs_d[None, :], mask=mmask[:, None], other=0.0)

    m_i = tl.full((BM,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, head_dim), dtype=tl.float32)

    # Causal, so this block never needs a key past the last token it owns.
    hi = tl.minimum(pid_m * BM + BM, n_tokens)
    kv_h_off = (head // GQA) * cache_stride_h
    for n0 in range(0, hi, BN):
        toks = n0 + tl.arange(0, BN)
        kmask = toks < hi
        page = tl.load(bt_ptr + toks // page_size, mask=kmask, other=0)
        base = (page.to(tl.int64) * cache_stride_b + kv_h_off
                + (toks % page_size) * cache_stride_p)
        k = tl.load(k_cache_ptr + base[:, None] + offs_d[None, :],
                    mask=kmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(kmask[None, :] & (offs_m[:, None] >= toks[None, :]),
                      qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        p = tl.exp2((qk - m_new[:, None]) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_cache_ptr + base[:, None] + offs_d[None, :],
                    mask=kmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = acc / l_i[:, None]
    g = tl.load(qkv_ptr + offs_m[:, None] * qkv_stride_t
                + head * (2 * head_dim) + head_dim + offs_d[None, :],
                mask=mmask[:, None], other=0.0).to(tl.float32)
    o = o * (1.0 / (1.0 + tl.exp2(-g * 1.4426950408889634)))
    tl.store(out_ptr + offs_m[:, None] * out_stride_t + head * head_dim
             + offs_d[None, :], o.to(out_ptr.dtype.element_ty),
             mask=mmask[:, None])


# ---------------------------------------------------------------------------
# Fused gate epilogue: out *= sigmoid(gate), reading the raw gate straight out
# of the interleaved [q|gate] columns of the QKV projection.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_tokens"])
def _gate_mul_kernel(
    out_ptr,
    qkv_ptr,
    n_tokens,
    out_stride_t,
    qkv_stride_t,
    head_dim: tl.constexpr,
    HD_TOTAL: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    t = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)[:, None]
    j = tl.arange(0, HD_TOTAL)[None, :]
    mask = t < n_tokens
    o_ptrs = out_ptr + t * out_stride_t + j
    g_ptrs = (
        qkv_ptr
        + t * qkv_stride_t
        + (j // head_dim) * (2 * head_dim)
        + head_dim
        + (j % head_dim)
    )
    o = tl.load(o_ptrs, mask=mask)
    g = tl.load(g_ptrs, mask=mask).to(tl.float32)
    tl.store(o_ptrs, o * (1.0 / (1.0 + tl.exp(-g))), mask=mask)


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


# Scratch is reused only up to this many tokens: beyond it the allocator round
# trip is noise next to the FLOPs and pinning the buffers would cost hundreds of
# MB.  Reused scratch is also what makes a cached step plan possible.
_SCRATCH_MAX_TOKENS = 1024

# Above this token count trtllm-gen's paged FMHA is faster than the fused
# attention above: measured cold-L2 (fmha + gate epilogue vs one fused launch)
# 14.4 -> 8.2us at N=1, 16.4 -> 10.2 at 26, 16.4 -> 10.3 at 60, but 20.5 -> 28.7
# at N=445, where the O(N^2) score matrix starts to matter and trtllm's
# hand-tuned tiling wins.  All the captured shapes are either <= 69 or >= 445.
_FUSED_ATTN_MAX_TOKENS = 128
# 16 query rows per program: the smallest MMA tile, which maximizes the CTA
# count (ceil(N/16) x num_heads) and measured best at every N <= 60.
_FUSED_ATTN_BM = 16
# Keys per softmax step.  head_dim=256 makes each key tile 2 x BN x 512 bytes of
# shared memory for k and v, so 64 is the largest that still compiles at
# num_stages=2; beyond N=64 it is clamped by N anyway.
_FUSED_ATTN_BN_MAX = 64


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
        # them on FlashInfer with an HND cache, so follow the same per-device
        # backend selection the generic ``Attention`` layer uses.
        attn_cfg = get_attn_backend_config()
        self.kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm
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

        # ---- config-derived constants, hoisted out of the per-call path ----
        self._q_gate_size = self.num_heads * 2 * self.head_dim
        self._kv_size = self.num_kv_heads * self.head_dim
        self._qkv_split = [self._q_gate_size, self._kv_size, self._kv_size]
        self._k_col = self._q_gate_size
        self._v_col = self._q_gate_size + self._kv_size
        self._q_flat = self.num_heads * self.head_dim
        self._head_block = triton.next_power_of_2(head_dim)
        # BLOCK_T x head_dim per program with 32 bytes per lane: measured
        # 13.3 -> 9.2us at 445 tokens and 216 -> 77us at 16384.
        self._prologue_tokens = 4
        self._prologue_warps = 2
        self._rot_half_block = 0
        # Pure-prefill plan, rebuilt only when the rotary module or the paged
        # cache tensors it was validated against change identity.
        self._pf_ok: bool | None = None
        self._pf_rope_box: tuple = (None,)
        self._pf_kc_box: tuple = (None, None)
        self._pf_cos_sin: torch.Tensor | None = None
        self._pf_rot = 0
        self._pf_half_rot = 0
        self._pf_has_pass = True
        self._pf_in_dtype = None
        self._pf_in_dtype_torch = None
        self._pf_cs = (0, 0, 0)
        self._pf_page = 0
        self._pf_eps = 0.0
        self._pf_gain = None
        self._pf_wqkv_t: torch.Tensor | None = None
        self._pf_wqkv_bias: torch.Tensor | None = None
        self._pf_wo_t: torch.Tensor | None = None
        self._pf_workspace: torch.Tensor | None = None
        self._pf_ws_bytes = 0
        self._pf_sm_count = 0
        self._pf_pdl = False
        self._pf_fmha = None
        self._pf_qkv_stride = 0
        self._pf_dev = 0
        self._gqa = (
            self.num_heads // self.num_kv_heads if self.num_kv_heads else 0
        )
        self._pf_fused_attn = False
        # Prebound Triton launches, keyed on what Triton specializes on.
        self._pro_launch: dict = {}
        self._gate_launch: dict = {}
        self._attn_launch: dict = {}
        self._fast_launch = True
        # Reused scratch for the small-N shapes, where a caching-allocator
        # round trip is a measurable slice of the whole layer.  Every element
        # is overwritten before it is read, so this is scratch, not state.
        self._pf_bufs: dict = {}
        # Fully resolved launch arguments for one repeated step shape,
        # invalidated wholesale by ``_pf_gen``.
        self._step: list | None = None
        self._pf_gen = 0

    def set_trtllm_workspace(self, workspace: torch.Tensor) -> None:
        """Adopt the engine's single shared trtllm-gen workspace."""
        if self._use_trtllm:
            self.flash_attn_decode._workspace = workspace
            self.flash_attn_prefill._workspace = workspace
            self._pf_workspace = workspace
            self._pf_ws_bytes = workspace.numel() * workspace.element_size()

    def _norm_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``1 + weight`` for the QK norms, in fp32, materialized once."""
        if self._norm_gain_cache is None:
            self._norm_gain_cache = (
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
            )
        return self._norm_gain_cache

    # -- prefill cu_seqlens, memoized on the per-step metadata object ---------
    @staticmethod
    def _prefill_meta(md, nd: int, np_: int):
        """``(cu_seqlens_q, cu_seqlens_k, seq_lens)`` as int32, built once per step.

        The reference rebuilds these on every call: a subtract for
        ``cu_seqlens_q``, then ``zeros`` + ``cumsum`` + slice-assign for
        ``cu_seqlens_k``, then another subtract inside the FMHA wrapper to
        recover ``seq_lens`` -- half a dozen two-element launches per attention
        layer, all recomputing the same values.  The engine builds one metadata
        object per step and shares it across every attention layer, so cache the
        result there (vLLM's own trick).

        The cache key holds strong references to the source buffers plus their
        version counters, so it cannot survive an in-place metadata update or be
        confused by a recycled allocation.
        """
        qsl = md.query_start_loc
        sl = md.seq_lens
        c = getattr(md, "_q3n_pf_meta", None)
        if (
            c is not None
            and c[0] is qsl
            and c[1] is sl
            and c[2] == qsl._version
            and c[3] == sl._version
            and c[4] == nd
            and c[5] == np_
        ):
            return c[6]

        seqs_k = sl[nd:] if nd else sl
        if seqs_k.dtype != torch.int32:
            seqs_k = seqs_k.to(torch.int32)
        seqs_k = seqs_k.contiguous()
        cu_q = qsl[nd:] if nd else qsl
        if nd:
            cu_q = cu_q - qsl[nd]
        if cu_q.dtype != torch.int32:
            cu_q = cu_q.to(torch.int32)
        cu_k = torch.zeros(np_ + 1, dtype=torch.int32, device=sl.device)
        cu_k[1:] = torch.cumsum(seqs_k, dim=0)
        res = (cu_q, cu_k, seqs_k)
        try:
            md._q3n_pf_meta = (
                qsl, sl, qsl._version, sl._version, nd, np_, res,
            )
        except (AttributeError, TypeError):
            pass
        return res

    def forward(self, hidden_states, rotary_emb=None, positions=None,
                state_manager=None):
        if rotary_emb is not None and rotary_emb is not self._modules.get(
                "rotary_emb"):
            # ``nn.Module.__setattr__`` is a few hundred nanoseconds of dict
            # surgery, and the engine hands us the same module every call.
            self.rotary_emb = rotary_emb
        if self._use_custom_op:
            return torch.ops.fastkernels.qwen3_next_attention(
                hidden_states, positions, self._layer_name,
            )
        return self.forward_impl(hidden_states, positions, state_manager)

    # ------------------------------------------------------------------
    # Fast path: pure prefill (num_decodes == 0) on a paged HND cache with
    # NeoX partial RoPE -- five launches end to end.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Pure-prefill fast path (num_decodes == 0) on a paged HND cache with
    # NeoX partial RoPE: five launches end to end.
    #
    # At 1..445 tokens this layer is entirely host-bound -- the five kernels
    # cost ~15us on the GPU and the *dispatch* costs several times that -- so
    # everything that can be resolved once is resolved once: weight transposes,
    # the FMHA's device constants, the Triton binaries, and the whole
    # precondition check (revalidated only when the rotary module or the cache
    # tensors change identity).
    # ------------------------------------------------------------------
    def _setup_prefill(self, rope, k_cache, v_cache, x) -> bool:
        """Validate the fast path and resolve everything it needs. Once."""
        self._pf_rope_box = (rope,)
        self._pf_kc_box = (k_cache, v_cache)
        self._pf_ok = False
        self._pf_gen += 1
        self._step = None
        if not (self._use_trtllm and self._fused_qk_rope_gate):
            return False
        if rope is None or not getattr(rope, "is_neox_style", False):
            return False
        cos_sin = getattr(rope, "cos_sin_cache", None)
        rot = getattr(rope, "head_dim", 0)
        if cos_sin is None or not (0 < rot <= self.head_dim) or rot % 2:
            return False
        if cos_sin.stride(0) != rot or cos_sin.stride(1) != 1:
            return False
        if self.qkv_proj.use_fp8 or self.o_proj.use_fp8:
            return False
        if self.o_proj.bias is not None:
            return False
        if self.o_proj.reduce_results and self.o_proj.tp_size > 1:
            return False
        if k_cache is None or v_cache is None or k_cache.dim() != 4:
            return False
        if k_cache.shape[1] != self.num_kv_heads:
            return False
        if k_cache.shape != v_cache.shape or k_cache.stride() != v_cache.stride():
            return False
        if k_cache.stride(3) != 1 or k_cache.dtype != x.dtype:
            return False
        page = k_cache.shape[2]
        if page != getattr(self.store_kvcache, "page_size", -1):
            return False
        ws = self.flash_attn_prefill._workspace
        if ws is None:
            return False
        gains = self._norm_gains()
        w_qkv = self.qkv_proj.weight
        w_o = self.o_proj.weight
        if x.dtype not in (torch.bfloat16, torch.float16):
            return False
        # 16-byte alignment of every tensor that is fixed for the lifetime of
        # the plan, so the per-call prebind key only has to cover the two that
        # the engine hands us fresh (``slot_mapping`` and ``positions``).
        fixed = (
            k_cache.data_ptr() | v_cache.data_ptr() | cos_sin.data_ptr()
            | gains[0].data_ptr() | gains[1].data_ptr()
            | w_qkv.data_ptr() | w_o.data_ptr()
        )
        if fixed & 15:
            return False

        fmha = None
        if _trtllm_paged_context is not None:
            try:
                from flashinfer.prefill import get_trtllm_gen_fmha_module
                from flashinfer.utils import device_support_pdl, get_device_sm_count

                fmha = get_trtllm_gen_fmha_module().trtllm_paged_attention_context
                self._pf_sm_count = get_device_sm_count(x.device)
                self._pf_pdl = device_support_pdl(x.device)
            except Exception:
                fmha = None
        if fmha is None:
            return False

        self._pf_cos_sin = cos_sin
        self._pf_rot = rot
        self._pf_half_rot = rot // 2
        self._rot_half_block = triton.next_power_of_2(rot // 2)
        self._pf_has_pass = rot < self.head_dim
        self._pf_in_dtype = (
            tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
        )
        self._pf_in_dtype_torch = x.dtype
        self._pf_cs = (k_cache.stride(0), k_cache.stride(1), k_cache.stride(2))
        self._pf_page = page
        self._pf_eps = self.q_norm.variance_epsilon
        self._pf_gain = gains
        self._pf_wqkv_t = w_qkv.t()
        self._pf_wqkv_bias = self.qkv_proj.bias
        self._pf_wo_t = w_o.t()
        self._pf_workspace = ws
        self._pf_ws_bytes = ws.numel() * ws.element_size()
        self._pf_fmha = fmha
        # The fused attention indexes the paged cache as
        # ``block_table[j // page_size] , j % page_size`` and tiles head_dim in
        # one power-of-two block, and it needs an integral GQA ratio.
        self._pf_fused_attn = (
            page > 0
            and (page & (page - 1)) == 0
            and self.num_kv_heads > 0
            and self.num_heads % self.num_kv_heads == 0
            and self.head_dim == triton.next_power_of_2(self.head_dim)
        )
        self._pf_qkv_stride = w_qkv.shape[0]
        self._pf_dev = x.device.index or 0
        self._pf_ok = True
        self._pf_gen += 1
        self._step = None
        return True

    def _pf_scratch(self, N: int, dtype, device):
        """``(qkv, q, out)`` scratch for *N* tokens.

        Small-N steps re-run with the same token count over and over, and three
        caching-allocator round trips is ~8us of the ~60us this layer costs
        there.  Each buffer is fully written before it is read.  Large chunks
        are not cached -- the allocation is noise next to their FLOPs and the
        memory is not worth pinning.
        """
        if N > _SCRATCH_MAX_TOKENS:
            qkv = torch.empty((N, self._pf_qkv_stride), dtype=dtype, device=device)
            q = torch.empty((N, self.num_heads, self.head_dim), dtype=dtype,
                            device=device)
            return qkv, q, torch.empty_like(q)
        key = (N, dtype)
        bufs = self._pf_bufs.get(key)
        if bufs is None:
            qkv = torch.empty((N, self._pf_qkv_stride), dtype=dtype, device=device)
            q = torch.empty((N, self.num_heads, self.head_dim), dtype=dtype,
                            device=device)
            bufs = (qkv, q, torch.empty_like(q))
            if len(self._pf_bufs) > 32:
                self._pf_bufs.clear()
                self._pf_gen += 1
                self._step = None
            self._pf_bufs[key] = bufs
        return bufs

    def _launch(self, cache, key, jit_fn, grid, vals, warps):
        """Launch *jit_fn*, reusing the prebound binary when one is cached."""
        entry = cache.get(key)
        if entry is not None:
            try:
                entry[0](grid[0], grid[1], 1, _RAW_STREAM(self._pf_dev),
                         entry[1], entry[2], None, None, None, *vals)
                return
            except TypeError:  # launcher convention differs from what we assume
                cache.clear()
                self._fast_launch = False
        ck = jit_fn[grid](*vals, num_warps=warps, num_stages=2)
        if self._fast_launch:
            ck._init_handles()
            cache[key] = (ck._run, ck.function, ck.packed_metadata)

    def _forward_prefill(self, x, positions, md, N, np_):
        """Run the step from a fully resolved plan when the shape repeats.

        Rebuilding the argument lists is a real cost at this size: the prologue
        takes 26 arguments and the paged FMHA 31, and marshalling them is a
        measurable slice of a ~45us layer.  Every one of them is config-derived,
        weight-derived, reused scratch, or a per-step metadata buffer -- only
        ``positions`` changes between calls -- so cache the lists and overwrite
        that one slot.  The plan is dropped the moment any metadata buffer, the
        token count, a baked-in host scalar or the plan generation changes.
        """
        st = self._step
        if (
            st is not None
            and st[0] is md
            and st[1] == N
            and st[2] == np_
            and st[3] == md.max_query_len
            and st[4] == md.max_seq_len
            and st[5] is md.slot_mapping
            and st[6] is md.block_tables
            and st[7] is md.query_start_loc
            and st[8] is md.seq_lens
            and st[9] == self._pf_gen
            and st[10] == positions.data_ptr() & 15
        ):
            vals = st[11]
            vals[8] = positions
            stream = _RAW_STREAM(self._pf_dev)
            pro = st[13]
            attn = st[20]
            try:
                torch.mm(x, self._pf_wqkv_t, out=st[16])
                pro[0](st[19], st[18], 1, stream, pro[1], pro[2],
                       None, None, None, *vals)
                if attn is not None:
                    attn[0](attn[3], attn[4], 1, stream, attn[1], attn[2],
                            None, None, None, *attn[5])
                else:
                    gate = st[14]
                    self._pf_fmha(*st[12])
                    gate[0](N, 1, 1, stream, gate[1], gate[2],
                            None, None, None, *st[15])
            except TypeError:
                # The raw launcher's calling convention is not what we assume on
                # this Triton build. Fall back for good; ``_prefill_eager``
                # recomputes the whole step from ``x``, so a partial run above
                # is harmless.
                self._fast_launch = False
                self._pro_launch.clear()
                self._gate_launch.clear()
                self._attn_launch.clear()
                self._step = None
                return self._prefill_eager(x, positions, md, N, np_)
            return torch.mm(st[17], self._pf_wo_t)
        return self._prefill_eager(x, positions, md, N, np_)

    def _prefill_eager(self, x, positions, md, N, np_):
        dtype = x.dtype
        device = x.device
        qkv, q, out = self._pf_scratch(N, dtype, device)

        bias = self._pf_wqkv_bias
        if bias is None:
            torch.mm(x, self._pf_wqkv_t, out=qkv)
        else:
            torch.addmm(bias, x, self._pf_wqkv_t, out=qkv)

        slot = md.slot_mapping
        gains = self._pf_gain
        cs = self._pf_cs
        vals = (
            qkv, q, self._pf_kc_box[0], self._pf_kc_box[1], slot,
            gains[0], gains[1], self._pf_cos_sin, positions, N,
            self._pf_qkv_stride, self._q_flat, cs[0], cs[1], cs[2],
            self.num_heads, self.head_dim, self._pf_rot, self._pf_half_rot,
            self._k_col, self._v_col, self._pf_page, self._pf_eps,
            self._pf_in_dtype, self._head_block, self._rot_half_block,
            self._pf_has_pass, self._prologue_tokens,
        )
        pro_key = (slot.data_ptr() | positions.data_ptr()) & 15
        self._launch(
            self._pro_launch,
            pro_key,
            _qk_norm_rope_store_kernel,
            (triton.cdiv(N, self._prologue_tokens),
             self.num_heads + self.num_kv_heads),
            vals,
            self._prologue_warps,
        )

        bt = md.block_tables
        # One sequence whose keys are exactly the tokens just stored is the only
        # case the fused attention covers; ``max_seq_len``/``max_query_len`` are
        # maxima over ``np_`` prefills, so with np_ == 1 they *are* this
        # sequence's key and query lengths -- no device read needed to check it.
        use_fa = (
            self._pf_fused_attn
            and N <= _FUSED_ATTN_MAX_TOKENS
            and np_ == 1
            and md.max_seq_len == N
            and md.max_query_len == N
            and bt.dim() == 2
            and bt.shape[0] >= 1
            and bt.shape[1] * self._pf_page >= N
        )
        fmha_args = gvals = avals = None
        attn = gate = None
        if use_fa:
            bn = min(_FUSED_ATTN_BN_MAX, max(16, triton.next_power_of_2(N)))
            avals = (
                q, self._pf_kc_box[0], self._pf_kc_box[1], bt, qkv, out, N,
                self._q_flat, self._pf_qkv_stride, self._q_flat,
                cs[0], cs[1], cs[2], self.scaling, self._gqa, self.head_dim,
                self._pf_page, _FUSED_ATTN_BM, bn,
            )
            agrid = (triton.cdiv(N, _FUSED_ATTN_BM), self.num_heads)
            akey = (bn, (q.data_ptr() | qkv.data_ptr() | out.data_ptr()
                         | bt.data_ptr()) & 15)
            try:
                # 8 warps only once the key tile is wide enough to keep them
                # busy: measured end to end, 4 warps is 40.0 vs 42.0us at N=1
                # (BN=16) and the two are within noise at BN=32.
                self._launch(self._attn_launch, akey, _attn_gate_kernel, agrid,
                             avals, 8 if bn >= 64 else 4)
            except Exception:
                # A head_dim / tile combination this GPU cannot fit in shared
                # memory (Required > limit) surfaces here as OutOfResources at
                # *compile* time, before anything has run.  Retire the fused
                # attention for good and let the trtllm paged FMHA below take
                # over rather than failing the layer.
                self._pf_fused_attn = False
                self._attn_launch.pop(akey, None)
                self._pf_gen += 1
                self._step = None
                use_fa = False
            else:
                attn = self._attn_launch.get(akey)
                if attn is not None:
                    attn = (attn[0], attn[1], attn[2], agrid[0], agrid[1],
                            list(avals))
        if not use_fa:
            cu_q, cu_k, seq_lens = self._prefill_meta(md, 0, np_)

            # The paged FMHA writes straight into ``out``; the reference lets
            # the wrapper allocate, then slice-copies into a staging buffer.
            fmha_args = (
                out, None, q, self._pf_kc_box[0], self._pf_kc_box[1],
                self._pf_workspace, bt, seq_lens,
                md.max_query_len, md.max_seq_len, self.scaling, 1.0, -1.0,
                -1, 0, np_, -1, cu_q, cu_k, self._pf_sm_count, self._pf_pdl,
                self._pf_ws_bytes, None, None, None, None, True, True, None,
                0, 0,
            )
            self._pf_fmha(*fmha_args)

            gvals = (out, qkv, N, self._q_flat, self._pf_qkv_stride,
                     self.head_dim, self._q_flat, 1)
            self._launch(self._gate_launch, 0, _gate_mul_kernel, (N, 1),
                         gvals, 8)
            gate = self._gate_launch.get(0)

        o2 = out.view(N, self._q_flat)
        pro = self._pro_launch.get(pro_key)
        if (
            pro is not None and (attn is not None or gate is not None)
            and N <= _SCRATCH_MAX_TOKENS and bias is None
        ):
            self._step = [
                md, N, np_, md.max_query_len, md.max_seq_len,
                slot, bt, md.query_start_loc, md.seq_lens,
                self._pf_gen, positions.data_ptr() & 15,
                list(vals), fmha_args, pro, gate, gvals, qkv, o2,
                self.num_heads + self.num_kv_heads,
                triton.cdiv(N, self._prologue_tokens), attn,
            ]
        return torch.mm(o2, self._pf_wo_t)

    def forward_impl(self, hidden_states, positions=None, state_manager=None):
        ctx = get_context()
        md = ctx.kda_metadata
        if state_manager is None:
            state_manager = ctx.kda_state
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        # ``rotary_emb`` is a registered submodule, so plain attribute access
        # goes through ``nn.Module.__getattr__``; the module dict is a direct
        # hit.  ``None`` lives in ``__dict__`` instead, hence the fallback.
        rotary_emb = self._modules.get("rotary_emb")
        if rotary_emb is None:
            rotary_emb = self.rotary_emb

        x = hidden_states
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        N = x.shape[0]

        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]

        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills

        if nd == 0 and np_ > 0 and N > 0 and positions is not None:
            kc_box = self._pf_kc_box
            if not (
                self._pf_rope_box[0] is rotary_emb
                and kc_box[0] is k_cache
                and kc_box[1] is v_cache
            ):
                self._setup_prefill(rotary_emb, k_cache, v_cache, x)
            if (
                self._pf_ok
                and x.dtype is self._pf_in_dtype_torch
                and positions.dim() == 1
                and positions.numel() == N
                and md.slot_mapping is not None
                and md.slot_mapping.numel() >= N
                and md.block_tables is not None
                and md.block_tables.is_contiguous()
                and md.query_start_loc is not None
                and md.seq_lens is not None
            ):
                return self._forward_prefill(x, positions, md, N, np_)

        # ------------------------------------------------------------------
        # General path: mixed decode/prefill, non-Blackwell backends, or any
        # configuration the fused prologue above does not cover.
        # ------------------------------------------------------------------
        qkv = self.qkv_proj(x)
        q_gate, k, v = qkv.split(self._qkv_split, dim=-1)

        use_fused = (
            self._fused_qk_rope_gate
            and rotary_emb is not None
            and positions is not None
            and getattr(rotary_emb, "is_neox_style", False)
        )
        if use_fused:
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

        self.store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)

        out = torch.empty(
            N,
            self.num_heads,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if nd > 0:
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
            cu_pf = (md.query_start_loc[nd:] - md.query_start_loc[nd]).to(
                torch.int32,
            )
            seqs_k = md.seq_lens[nd:]
            cu_k_pf = torch.zeros(np_ + 1, dtype=torch.int32, device=q.device)
            cu_k_pf[1:] = torch.cumsum(seqs_k.to(torch.int32), dim=0)
            out[ndt:] = self.flash_attn_prefill(
                q[ndt:],
                k_cache,
                v_cache,
                cu_seqlens_q=cu_pf,
                cu_seqlens_k=cu_k_pf,
                max_seqlen_q=md.max_query_len,
                max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                block_table=md.block_tables[nd:],
            )

        o = _gate_mul_inplace(out, gate)

        # Output projection
        o = o.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(o)
