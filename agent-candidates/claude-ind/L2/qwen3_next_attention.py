"""Qwen3-Next full attention with per-head QK-norm, partial RoPE, output gating, KV cache (L2).

GQA attention: 16 query heads, 2 KV heads, head_dim=256.
Q projection outputs 2x: [Q, gate] interleaved per head.
Partial RoPE (25% of head_dim = 64 dims rotated).
Output: attn_output * sigmoid(gate).

Candidate notes
---------------
Per layer the baseline runs about a dozen launches: the QKV GEMM, vLLM's fused
QK-norm/RoPE/gate Triton kernel, the paged KV store, a ``cu_seqlens``/``cumsum``
metadata prologue, the trtllm-gen paged attention call, ``sigmoid`` + multiply,
and the output GEMM.  At the captured decode / small-chunk token counts
(1..445) the device work is a few microseconds and the layer is entirely
host-bound -- most of it inside FlashInfer's Python wrapper -- while at the
16k-token chunk the QK-norm kernel and the gate multiply push the full Q tensor
through HBM several extra times.

This keeps the two projections as plain ``F.linear`` (as the reference does) and
rewrites everything between them as two Triton kernels:

1. split -> QK-RMSNorm -> partial NeoX RoPE -> paged KV store in a single pass.
   Q is written contiguously and the gate is left where the GEMM put it.
2. paged causal attention with the sigmoid gate folded into the epilogue, read
   straight out of the QKV buffer so it is never materialized.  One kernel
   covers decode and prefill: a varlen query block attends to the end of a
   paged context, so ``q_len == 1`` degenerates to decode.

The metadata prologue is gone -- ``query_start_loc`` / ``seq_lens`` /
``block_tables`` are read on the device, so no cumsum/zeros/sub launches and no
host syncs -- and for launch-bound token counts the remaining four launches are
replayed from a CUDA graph, which is what actually removes the host cost.

Measured against the baseline on B200 (``fastkernels bench``): 3.4-3.9x at
1/26/60 tokens, 2.2x at 445, 0.47x at 16384.  That last case is the attention
kernel itself.  head_dim=256 forces a 128x256 fp32 accumulator, which pins the
kernel to 8 warps per SM (12.5% occupancy, 197 KiB of shared memory) and leaves
it latency-bound at ~305 TFLOP/s, against roughly 1 PFLOP/s for trtllm-gen's
warp-specialized tcgen05 kernel.  Sweeping tile shapes, stage counts, TMA
descriptor gathers, Triton's automatic warp specialization, 2-CTA clusters,
register caps, a two-pass (non-online) softmax and dropping the accumulator
rescale altogether all stayed within a few percent of this, so the remaining gap
is structural rather than a tuning miss.

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
from ....infra.tp import _tp_size
from ..L1.gemma_rms_norm import GemmaRMSNorm
from .parallel_linear import QKVParallelLinear, RowParallelLinear

_LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# 1) split -> QK-RMSNorm -> partial NeoX RoPE -> paged KV store
#
# ``program_id(1)`` selects the slot this program owns:
#   [0, NH)                 a Q head  -> normalize, rotate, write to ``q_out``
#   [NH, NH + NKV)          a K head  -> normalize, rotate, write to ``k_cache``
#   [NH + NKV, NH + 2*NKV)  a V head  -> copy to ``v_cache``
# The gate half of each Q head is left in the QKV buffer; the attention
# epilogue reads it from there, so it is never copied.
# ---------------------------------------------------------------------------
@triton.jit
def _qk_norm_rope_store_kernel(
    qkv_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_gain_ptr,
    k_gain_ptr,
    n_tokens,
    qkv_stride,
    cos_stride,
    kv_stride_blk,
    kv_stride_head,
    kv_stride_tok,
    eps,
    NH: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    HALF: tl.constexpr,
    PAGE: tl.constexpr,
    BT: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    slot_id = tl.program_id(1)
    offs_t = pid_t * BT + tl.arange(0, BT)
    tmask = offs_t < n_tokens
    d = tl.arange(0, D)

    if slot_id >= NH + NKV:
        # ---- V: straight copy into the paged cache ----
        h = slot_id - NH - NKV
        src = (
            qkv_ptr
            + offs_t[:, None] * qkv_stride
            + (NH * 2 * D + NKV * D + h * D)
            + d[None, :]
        )
        val = tl.load(src, mask=tmask[:, None], other=0.0)
        slot = tl.load(slot_ptr + offs_t, mask=tmask, other=-1).to(tl.int64)
        dst = (
            v_cache_ptr
            + ((slot // PAGE) * kv_stride_blk
               + h * kv_stride_head
               + (slot % PAGE) * kv_stride_tok)[:, None]
            + d[None, :]
        )
        tl.store(dst, val, mask=(tmask & (slot >= 0))[:, None])
        return

    if slot_id < NH:
        in_base = slot_id * 2 * D
        gain_ptr = q_gain_ptr
    else:
        in_base = NH * 2 * D + (slot_id - NH) * D
        gain_ptr = k_gain_ptr

    src = qkv_ptr + offs_t[:, None] * qkv_stride + in_base
    x = tl.load(src + d[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) * (1.0 / D)
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(gain_ptr + d).to(tl.float32)
    # Round-trip through the storage dtype so RoPE sees the same bf16 values the
    # unfused (norm -> memory -> rope) reference path would.
    x_norm = (x * inv_rms[:, None] * w[None, :]).to(DTYPE).to(tl.float32)

    if slot_id < NH:
        dst_base = q_out_ptr + (offs_t * NH + slot_id)[:, None] * D
        valid = tmask[:, None]
    else:
        slot = tl.load(slot_ptr + offs_t, mask=tmask, other=-1).to(tl.int64)
        dst_base = k_cache_ptr + (
            (slot // PAGE) * kv_stride_blk
            + (slot_id - NH) * kv_stride_head
            + (slot % PAGE) * kv_stride_tok
        )[:, None]
        valid = (tmask & (slot >= 0))[:, None]

    # Pass-through tail [2*HALF, D): normalized but not rotated.  Masked off the
    # rotary head so the two stores below never race with this one.
    if not HAS_ROPE:
        tl.store(dst_base + d[None, :], x_norm, mask=valid)
        return
    if HAS_PASS:
        tl.store(dst_base + d[None, :], x_norm,
                 mask=valid & (d >= 2 * HALF)[None, :])

    dh = tl.arange(0, HALF)
    x1 = tl.load(src + dh[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(src + HALF + dh[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
    w1 = tl.load(gain_ptr + dh).to(tl.float32)
    w2 = tl.load(gain_ptr + HALF + dh).to(tl.float32)
    x1 = (x1 * inv_rms[:, None] * w1[None, :]).to(DTYPE).to(tl.float32)
    x2 = (x2 * inv_rms[:, None] * w2[None, :]).to(DTYPE).to(tl.float32)

    pos = tl.load(pos_ptr + offs_t, mask=tmask, other=0).to(tl.int64)
    cs = cos_sin_ptr + pos[:, None] * cos_stride
    cos = tl.load(cs + dh[None, :], mask=tmask[:, None], other=1.0).to(tl.float32)
    sin = tl.load(cs + HALF + dh[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)

    vmask = tl.broadcast_to(valid, (BT, HALF))
    tl.store(dst_base + dh[None, :], x1 * cos - x2 * sin, mask=vmask)
    tl.store(dst_base + HALF + dh[None, :], x2 * cos + x1 * sin, mask=vmask)


# ---------------------------------------------------------------------------
# 2) paged causal attention + sigmoid gate epilogue
#
# One program per (query tile, Q head, request).  A request contributes
# ``query_start_loc[i+1] - query_start_loc[i]`` query tokens that sit at the end
# of a ``seq_lens[i]``-long context, so query row j sees keys
# ``[0, seq_len - q_len + j]``.  q_len == 1 is the decode case.
#
# The K/V tiles are gathered out of the paged cache with one base offset per
# token and a contiguous head_dim, so a page size of 16 costs essentially the
# same as a contiguous cache (measured: 3%).
# ---------------------------------------------------------------------------
@triton.jit
def _attn_tile(q, k_cache_ptr, v_cache_ptr, bt_ptr, acc, m_i, l_i,
               req, bt_stride, kv_stride_blk, kv_base, kv_stride_tok,
               qk_scale, lo, hi, qpos,
               MASKED: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DTYPE: tl.constexpr):
    """Accumulate ``lo:hi`` of the key axis into (acc, m_i, l_i).

    ``MASKED=False`` is the strictly-below-diagonal part: no causal mask and no
    bounds mask, which keeps the hot loop free of predication.
    """
    d = tl.arange(0, D)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if MASKED:
            nmask = offs_n < hi
            pg = tl.load(bt_ptr + req * bt_stride + offs_n // PAGE,
                         mask=nmask, other=0).to(tl.int64)
        else:
            pg = tl.load(bt_ptr + req * bt_stride + offs_n // PAGE).to(tl.int64)
        kv_off = pg * kv_stride_blk + kv_base + (offs_n % PAGE) * kv_stride_tok
        # [D, BLOCK_N]: the dot's B operand wants head_dim-major, which is the
        # cache's own layout, so no transpose is needed.
        if MASKED:
            k = tl.load(k_cache_ptr + kv_off[None, :] + d[:, None],
                        mask=nmask[None, :], other=0.0)
        else:
            k = tl.load(k_cache_ptr + kv_off[None, :] + d[:, None])
        qk = tl.dot(q, k) * qk_scale
        if MASKED:
            qk = tl.where(nmask[None, :] & (qpos[:, None] >= offs_n[None, :]),
                          qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if MASKED:
            v = tl.load(v_cache_ptr + kv_off[:, None] + d[None, :],
                        mask=nmask[:, None], other=0.0)
        else:
            v = tl.load(v_cache_ptr + kv_off[:, None] + d[None, :])
        acc = tl.dot(p.to(DTYPE), v, acc)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _paged_attn_gate_kernel(
    q_ptr,
    gate_ptr,
    out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    bt_ptr,
    seqlens_ptr,
    cuq_ptr,
    n_m_tiles,
    gate_stride,
    bt_stride,
    kv_stride_blk,
    kv_stride_head,
    kv_stride_tok,
    qk_scale,
    NH: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Causal work grows linearly with the query-tile index and CUDA issues
    # blocks in program_id(0)-major order, so a straight mapping would put every
    # expensive tile in the last wave.  Pair cheap tiles with expensive ones.
    tid = tl.program_id(0)
    pid_m = tl.where(tid % 2 == 0, tid // 2, n_m_tiles - 1 - tid // 2)
    h = tl.program_id(1)
    req = tl.program_id(2)

    q_start = tl.load(cuq_ptr + req).to(tl.int32)
    q_len = tl.load(cuq_ptr + req + 1).to(tl.int32) - q_start
    m0 = pid_m * BLOCK_M
    if m0 >= q_len:
        return
    seq_len = tl.load(seqlens_ptr + req).to(tl.int32)
    ctx = seq_len - q_len

    offs_m = m0 + tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    qm = offs_m < q_len
    rows = q_start + offs_m
    q = tl.load(q_ptr + (rows * NH + h)[:, None] * D + d[None, :],
                mask=qm[:, None], other=0.0)

    kv_base = (h // (NH // NKV)) * kv_stride_head
    qpos = ctx + offs_m
    hi = tl.minimum(seq_len, ctx + m0 + BLOCK_M)
    # Every row of this tile attends to all keys below ``ctx + m0``, so that
    # prefix needs no causal mask.
    lo_full = (ctx + m0 + 1) // BLOCK_N * BLOCK_N

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # ``SPLIT`` peels the unmasked prefix into its own loop.  That is a win when
    # there are many key blocks, and a loss for a handful of them: it doubles the
    # instruction footprint of a kernel whose whole runtime is start-up.
    if SPLIT:
        acc, m_i, l_i = _attn_tile(
            q, k_cache_ptr, v_cache_ptr, bt_ptr, acc, m_i, l_i, req, bt_stride,
            kv_stride_blk, kv_base, kv_stride_tok, qk_scale, 0, lo_full, qpos,
            False, D, PAGE, BLOCK_M, BLOCK_N, DTYPE)
        acc, m_i, l_i = _attn_tile(
            q, k_cache_ptr, v_cache_ptr, bt_ptr, acc, m_i, l_i, req, bt_stride,
            kv_stride_blk, kv_base, kv_stride_tok, qk_scale, lo_full, hi, qpos,
            True, D, PAGE, BLOCK_M, BLOCK_N, DTYPE)
    else:
        acc, m_i, l_i = _attn_tile(
            q, k_cache_ptr, v_cache_ptr, bt_ptr, acc, m_i, l_i, req, bt_stride,
            kv_stride_blk, kv_base, kv_stride_tok, qk_scale, 0, hi, qpos,
            True, D, PAGE, BLOCK_M, BLOCK_N, DTYPE)

    acc = acc / l_i[:, None]
    # Output gate: read the raw gate straight out of the fused QKV buffer (per
    # head it is the second half of the 2*head_dim Q slot), so the baseline's
    # gate copy, sigmoid and multiply all disappear.
    g = tl.load(gate_ptr + rows[:, None] * gate_stride + (h * 2 * D + D) + d[None, :],
                mask=qm[:, None], other=0.0).to(tl.float32)
    g = (1.0 / (1.0 + tl.exp2(-g * 1.4426950408889634))).to(DTYPE).to(tl.float32)
    o = acc.to(DTYPE).to(tl.float32) * g
    tl.store(out_ptr + (rows * NH + h)[:, None] * D + d[None, :],
             o.to(DTYPE), mask=qm[:, None])


def _kv_strides(cache: torch.Tensor, kv_layout: str) -> tuple[int, int, int, int]:
    """(block, head, token, page_size) element strides for an HND/NHD cache."""
    if kv_layout == "HND":  # [blocks, heads, page, dim]
        return cache.stride(0), cache.stride(1), cache.stride(2), cache.shape[2]
    return cache.stride(0), cache.stride(2), cache.stride(1), cache.shape[1]


def _attn_config(max_q: int, max_kv: int, nreq: int,
                 nh: int) -> tuple[int, int, int, int, bool]:
    """(BLOCK_M, BLOCK_N, num_warps, num_stages, split) for the attention launch.

    ``BLOCK_M=128`` is the tile shape tcgen05 wants and it halves the K/V traffic
    per unit of work, so it wins once the query axis is long enough to fill the
    148 SMs.  Below that the kernel is almost entirely start-up cost, and the
    cheapest tile wins: a small ``BLOCK_M``/``BLOCK_N`` keeps the shared-memory
    footprint down, one pipeline stage beats two when there is a single key block
    to consume, and the unmasked-prefix loop is not worth the extra instruction
    footprint.  Values from sweeping the whole layer at the captured token counts
    (1 / 60 / 445 / 16384).
    """
    if max_q >= 2048:
        bm = 128
    elif max_q >= 128:
        bm = 64
    else:
        bm = 16
    bn = min(64, max(16, triton.next_power_of_2(max_kv)))
    warps = 8 if (bn >= 64 or bm >= 64) else 4
    n_kv_blocks = triton.cdiv(max_kv, bn)
    return bm, bn, warps, (2 if n_kv_blocks > 2 else 1), n_kv_blocks > 4


class _Graph:
    __slots__ = ("graph", "x", "pos", "out", "dsts")


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

        attn_cfg = get_attn_backend_config()
        self.kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm
        self._norm_gain_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._use_custom_op = False
        self._layer_name = ""
        self.rotary_emb = None
        self._graphs: dict[tuple, _Graph] = {}

    # -- engine hooks ------------------------------------------------------
    def set_trtllm_workspace(self, workspace: torch.Tensor) -> None:
        """No-op: this implementation owns no trtllm-gen scratch buffer."""
        return None

    def _norm_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``1 + weight`` for the QK norms, in fp32, materialized once.

        The values are constants after weight loading, so caching them is exact
        and saves two launches per layer per step.
        """
        if self._norm_gain_cache is None:
            self._norm_gain_cache = (
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
            )
        return self._norm_gain_cache

    # -- forward -----------------------------------------------------------
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
        ctx = get_context()
        md = ctx.kda_metadata
        if state_manager is None:
            state_manager = ctx.kda_state
        rotary_emb = self.rotary_emb
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        x = hidden_states
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        N = x.shape[0]

        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]
        pos = positions
        if pos is not None and pos.dim() != 1:
            pos = pos.reshape(-1)

        nreq = md.num_decodes + md.num_prefills
        if rotary_emb is not None and not getattr(rotary_emb, "is_neox_style", True):
            raise RuntimeError(
                "Qwen3NextAttention: only NeoX-style partial RoPE is supported",
            )
        if nreq == 0 or N == 0:
            return self.o_proj(
                torch.zeros((N, self.num_heads * self.head_dim),
                            device=x.device, dtype=x.dtype),
            )

        # At these token counts the layer is a handful of microseconds of device
        # work behind a host that needs longer than that to enqueue it, so the
        # four launches are replayed from a CUDA graph instead.  The graph bakes
        # in every buffer address and the launch geometry, so it is keyed on both;
        # ``hidden_states``/``positions`` move each step and are staged in.
        use_graph = (
            N <= _GRAPH_MAX_TOKENS
            and not torch.cuda.is_current_stream_capturing()
        )
        if use_graph:
            key = (N, nreq, md.max_query_len, md.max_seq_len,
                   k_cache.data_ptr(), v_cache.data_ptr(),
                   md.slot_mapping.data_ptr(), md.block_tables.data_ptr(),
                   md.seq_lens.data_ptr(), md.query_start_loc.data_ptr(),
                   rotary_emb.cos_sin_cache.data_ptr(),
                   x.dtype, pos.dtype)
            entry = self._graphs.get(key, _MISSING)
            if entry is _MISSING:
                entry = self._capture(key, x, pos, md, k_cache, v_cache, nreq)
            if entry is not None:
                torch._foreach_copy_(entry.dsts, (x, pos))
                entry.graph.replay()
                return entry.out

        return self._run(x, pos, md, k_cache, v_cache, nreq, N)

    # -- implementation ----------------------------------------------------
    def _run(self, x, pos, md, k_cache, v_cache, nreq, N):
        nh = self.num_heads
        nkv = self.num_kv_heads
        dim = self.head_dim
        q_gain, k_gain = self._norm_gains()
        rope = self.rotary_emb
        has_rope = rope is not None and pos is not None
        cos_sin = rope.cos_sin_cache if has_rope else q_gain
        half = (rope.head_dim // 2) if has_rope else 1
        pos_arg = pos if has_rope else md.slot_mapping

        qkv = F.linear(x, self.qkv_proj.weight)
        q = torch.empty((N, nh, dim), device=x.device, dtype=x.dtype)
        out = torch.empty((N, nh, dim), device=x.device, dtype=x.dtype)

        blk, hd, tok, page = _kv_strides(k_cache, self.kv_layout)
        dt = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16

        bt = 16 if N >= 512 else (4 if N > 4 else 1)
        _qk_norm_rope_store_kernel[(triton.cdiv(N, bt), nh + 2 * nkv)](
            qkv, q, k_cache, v_cache, md.slot_mapping, cos_sin, pos_arg,
            q_gain, k_gain,
            N, qkv.stride(0), cos_sin.stride(0), blk, hd, tok,
            self.q_norm.variance_epsilon,
            NH=nh, NKV=nkv, D=dim, HALF=half, PAGE=page, BT=bt,
            HAS_PASS=(2 * half < dim), HAS_ROPE=has_rope, DTYPE=dt,
            num_warps=8 if bt >= 16 else 4, num_stages=2,
        )

        bm, bn, warps, stages, split = _attn_config(
            md.max_query_len, md.max_seq_len, nreq, nh)
        n_m = triton.cdiv(md.max_query_len, bm)
        _paged_attn_gate_kernel[(n_m, nh, nreq)](
            q, qkv, out, k_cache, v_cache, md.block_tables, md.seq_lens,
            md.query_start_loc, n_m,
            qkv.stride(0), md.block_tables.stride(0), blk, hd, tok,
            self.scaling * _LOG2E,
            NH=nh, NKV=nkv, D=dim, PAGE=page, BLOCK_M=bm, BLOCK_N=bn,
            SPLIT=split, DTYPE=dt, num_warps=warps, num_stages=stages,
        )

        return self.o_proj(out.view(N, nh * dim))

    def _capture(self, key, x, pos, md, k_cache, v_cache, nreq):
        """Capture the four-kernel sequence for this (shape, buffer) signature.

        Each graph owns a private memory pool, so a producer that hands out fresh
        buffers every step would otherwise grow without bound; past the cap the
        layer just runs eagerly.  Any capture failure is cached as "no graph" so
        it is not retried on every call.
        """
        if len(self._graphs) >= _MAX_GRAPHS:
            self._graphs[key] = None
            return None
        try:
            static_x = torch.empty_like(x)
            static_pos = torch.empty_like(pos)
            static_x.copy_(x)
            static_pos.copy_(pos)
            N = x.shape[0]
            self._norm_gains()
            # Warm up (Triton JIT + cuBLAS workspace) outside the capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._run(static_x, static_pos, md, k_cache, v_cache, nreq, N)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run(static_x, static_pos, md, k_cache, v_cache, nreq, N)
        except Exception:
            self._graphs[key] = None
            return None
        entry = _Graph()
        entry.graph = graph
        entry.x = static_x
        entry.pos = static_pos
        entry.out = out
        entry.dsts = [static_x, static_pos]
        self._graphs[key] = entry
        return entry


# Above this token count the host is no longer the bottleneck and the staging
# copy a graph replay needs would cost more than the launches it saves.
_GRAPH_MAX_TOKENS = 1024
_MAX_GRAPHS = 64
_MISSING = object()
