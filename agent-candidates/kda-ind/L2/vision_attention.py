"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from flash_attn.ops.triton.rotary import apply_rotary

from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear

# Largest linear element offset the rotary indexing may reach. Triton derives
# its offsets from ``tl.arange``, which is int32, so the whole ``q|k`` region
# has to be addressable in int32. At the largest benched case (S=64680,
# row stride 3456) the top offset is ~2.24e8 -- comfortably inside the range,
# but asserted rather than assumed because it scales with S.
_MAX_LINEAR_OFFSET = 2 ** 31

# Tiling for the rotary kernel below, hard-coded rather than autotuned: the
# benchmark times 50 iterations after 10 warmups, so anything that can compile
# inside the timed region shows up as latency. Chosen by sweeping 14
# BLOCK_H x BLOCK_M x num_warps combinations over the five benched sequence
# lengths (``tests/sweep_rope.py``) and re-timing the shortlist three times
# (``tests/confirm_rope_config.py``). This wins on four of the five and costs
# 2.4% against the per-shape best on the fifth, which is well inside the margin
# that would justify branching on sequence length.
_ROPE_BLOCK_H = 4
_ROPE_BLOCK_M = 8
_ROPE_NUM_WARPS = 2


@triton.jit(do_not_specialize_on_alignment=["seqlen"])
def _rope_qk_inplace(
    X, COS, SIN,
    stride_row, seqlen, nslots,
    HALF: tl.constexpr,
    STRIDE_SLOT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Rotate-half rotary embedding, in place over ``(token, slot)`` tiles.

    flash_attn's ``rotary_kernel`` tiles ``BLOCK_H=2, BLOCK_M=8``, which is
    sized for ``head_dim`` 64 or 128. At ``head_dim = 72`` that leaves 41360
    tiny blocks whose two half-loads cover 36 of 64 lanes as 72-byte runs, and
    it profiles ALU-bound at 2.22 sectors per request rather than
    bandwidth-bound. Tiling over slots as well as tokens is the whole point of
    replacing it.

    Writing in place is safe because a program owns an entire ``(token, slot)``
    tile and loads *both* halves of every rotate-half pair before storing
    either, so no pair is ever split across two programs. A flat power-of-two
    tiling over the same region does split pairs -- no power of two divides 72
    -- and races.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    # ``tl.arange`` needs a power-of-two extent and HALF is 36, so the upper
    # lanes are masked off. Every load and store carries the same predicate.
    rk = tl.arange(0, BLOCK_K)
    lane = rk < HALF

    # cos/sin are a row-major [seqlen, HALF] table indexed by the absolute
    # token position, and one row is shared by every slot in the tile.
    cs = rm[:, None] * HALF + rk[None, :]
    cs_mask = (rm[:, None] < seqlen) & lane[None, :]
    cos = tl.load(COS + cs, mask=cs_mask, other=1.0).to(tl.float32)[:, None, :]
    sin = tl.load(SIN + cs, mask=cs_mask, other=0.0).to(tl.float32)[:, None, :]

    P = X + (rm[:, None, None] * stride_row
             + rh[None, :, None] * STRIDE_SLOT
             + rk[None, None, :])
    mask = ((rm[:, None, None] < seqlen)
            & (rh[None, :, None] < nslots)
            & lane[None, None, :])
    # fp32 intermediates, matching rotary_kernel, so the result is bit-identical
    # rather than merely close.
    x0 = tl.load(P, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(P + HALF, mask=mask, other=0.0).to(tl.float32)
    tl.store(P, (x0 * cos - x1 * sin).to(X.dtype.element_ty), mask=mask)
    tl.store(P + HALF, (x0 * sin + x1 * cos).to(X.dtype.element_ty), mask=mask)


def _apply_rope_inplace(qk, cos, sin, *, block_h=_ROPE_BLOCK_H,
                        block_m=_ROPE_BLOCK_M, num_warps=_ROPE_NUM_WARPS):
    """Launch :func:`_rope_qk_inplace` over a ``(seqlen, nslots, head_dim)`` view.

    ``seqlen`` and ``nslots`` stay runtime arguments so the benched sequence
    lengths share one compiled specialization instead of triggering a compile
    each. Keeping them runtime is necessary but not sufficient: Triton also
    derives a specialization from whether an integer argument is divisible by
    16, and the benched lengths are mixed (20680 and 64680 are 8 mod 16, the
    other three are 0 mod 16), which would split them across two variants. Hence
    ``do_not_specialize_on_alignment=["seqlen"]`` on the kernel -- the alignment
    hint buys nothing here because ``seqlen`` is only ever compared against, and
    never used as a base offset whose vectorization could depend on it.

    Dimensions at and above ``2 * HALF`` are left untouched, which is free here
    and is what a partial-rotary caller needs.
    """
    seqlen, nslots, _ = qk.shape
    half = cos.shape[-1]
    grid = (triton.cdiv(seqlen, block_m), triton.cdiv(nslots, block_h))
    # Without the device guard Triton launches from cuda:0 and rejects a pointer
    # it cannot reach, exactly as ``apply_rotary`` documents.
    with torch.cuda.device(qk.device.index):
        _rope_qk_inplace[grid](
            qk, cos, sin,
            qk.stride(0), seqlen, nslots,
            HALF=half,
            STRIDE_SLOT=qk.stride(1),
            BLOCK_H=block_h,
            BLOCK_M=block_m,
            BLOCK_K=triton.next_power_of_2(half),
            num_warps=num_warps,
        )
    return qk


def _rotary_gate_ok(qkv, cos, sin, seq_len, head_dim) -> bool:
    """Whether ``cos``/``sin`` are shaped for rotary on the strided ``q|k`` view.

    Only called with both present. Everything here is host-side tensor metadata,
    so the whole gate is decided before ``qkv`` is touched: an in-place rotary
    that has already run cannot be undone, so falling back after a failed launch
    is not an option.
    """
    if cos.dim() != 2 or sin.dim() != 2:
        return False
    if cos.shape != sin.shape:
        return False
    if cos.dtype is not qkv.dtype or sin.dtype is not qkv.dtype:
        return False
    if cos.device != qkv.device or sin.device != qkv.device:
        return False
    # Indexed as a row-major [seqlen, rotary_dim // 2] table, which is what
    # ``apply_rotary`` reduces its own inputs to via ``.contiguous()``.
    if not cos.is_contiguous() or not sin.is_contiguous():
        return False
    half = cos.shape[-1]
    if half <= 0 or 2 * half > head_dim:
        return False
    return cos.shape[0] >= seq_len


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder (Qwen2-VL / Qwen2.5-VL / Qwen3-VL).

    All heads are attention heads (no GQA). Uses full (non-causal) attention.
    Supports TP: QKV is sharded, then gathered for RoPE, then re-sharded.
    """

    def __init__(self, embed_dim: int, num_heads: int, projection_size: int | None = None):
        super().__init__()
        if projection_size is None:
            projection_size = embed_dim
        tp = _tp_size()
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.head_dim = projection_size // num_heads
        self.num_heads = num_heads // tp

        self.qkv = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, num_heads, bias=True,
        )
        self.proj = RowParallelLinear(projection_size, embed_dim, bias=True)
        self.attn = FlashAttnPrefill(self.num_heads, self.num_heads, self.head_dim)

    def _qkv_views(self, qkv, seq_len, cos, sin):
        """Rotate ``q|k`` in place and return ``(q, k, v)`` as strided views.

        ``self.qkv`` is a plain ``F.linear``, so ``qkv`` is contiguous with
        ``[q | k | v]`` on the last dim. q and k are therefore *adjacent*, which
        makes the whole ``q|k`` region one slice of stride ``(3*q_size, 1)``;
        splitting a dim whose own stride is 1 is a legal ``view``, so
        ``(seq_len, 2*num_heads, head_dim)`` costs no copy.

        That view is what removes the permute. Rotary is per-token -- the same
        ``cos[t]``/``sin[t]`` is broadcast across the head axis -- so treating
        q's heads and k's heads as ``2*num_heads`` head slots of one tensor
        computes exactly what the previous
        ``permute(2, 1, 0, 3, 4).contiguous()`` into ``(2*batch, seq, heads,
        dim)`` computed, minus the ~190 MB round trip through DRAM that the copy
        cost at a full encoder batch.
        """
        q_size = self.num_heads * self.head_dim
        qk = qkv[:, 0, : 2 * q_size].view(
            seq_len, 2 * self.num_heads, self.head_dim,
        )
        if cos is not None:
            _apply_rope_inplace(qk, cos, sin)
        q = qk[:, : self.num_heads, :]
        k = qk[:, self.num_heads:, :]
        v = qkv[:, 0, 2 * q_size:].view(
            seq_len, self.num_heads, self.head_dim,
        )
        return q, k, v

    def _qkv_views_via_copy(self, qkv, seq_len, batch_size, cos, sin):
        """The layout-copy path, for inputs the strided view cannot serve."""
        q_size = self.num_heads * self.head_dim
        qk = qkv[..., : 2 * q_size].view(
            seq_len, batch_size, 2, self.num_heads, self.head_dim,
        )
        # -> (2, batch, seq, heads, dim), one copy
        qk = qk.permute(2, 1, 0, 3, 4).contiguous()

        if cos is not None and sin is not None:
            flat = qk.view(2 * batch_size, seq_len, self.num_heads,
                           self.head_dim)
            apply_rotary(flat, cos, sin, inplace=True)

        q = qk[0].reshape(-1, self.num_heads, self.head_dim)
        k = qk[1].reshape(-1, self.num_heads, self.head_dim)
        # Keep v in the same (batch, seq) order as q/k. batch_size is 1 on every
        # current caller, but ordering v seq-major would silently disagree with
        # q/k if that ever changed.
        v = (qkv[..., 2 * q_size:]
             .view(seq_len, batch_size, self.num_heads, self.head_dim)
             .transpose(0, 1)
             .reshape(-1, self.num_heads, self.head_dim))
        return q, k, v

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = x.shape
        qkv = self.qkv(x)

        # A single one of cos/sin is not a rotary table. The copy path skips
        # rotary entirely in that case, so the strided path does too rather than
        # paying for a copy it has no use for.
        has_rotary = rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None

        # Decided entirely from host-side metadata, before anything is written:
        # the rotary below runs in place, and a launch that has already started
        # cannot be taken back, so there is no recovering by falling back after
        # the fact. batch_size == 1 is load-bearing rather than defensive -- the
        # strided view indexes ``qkv[:, 0, :]``, which simply is not q|k for a
        # second batch element, and every captured call has batch_size == 1.
        use_strided = (
            batch_size == 1
            and seq_len > 0
            and qkv.is_cuda
            and qkv.dim() == 3
            and qkv.is_contiguous()
            and seq_len * qkv.stride(0) < _MAX_LINEAR_OFFSET
            and (not has_rotary or _rotary_gate_ok(
                qkv, rotary_pos_emb_cos, rotary_pos_emb_sin, seq_len, self.head_dim))
        )

        if use_strided:
            q, k, v = self._qkv_views(
                qkv, seq_len,
                rotary_pos_emb_cos if has_rotary else None,
                rotary_pos_emb_sin if has_rotary else None,
            )
        else:
            q, k, v = self._qkv_views_via_copy(
                qkv, seq_len, batch_size, rotary_pos_emb_cos, rotary_pos_emb_sin,
            )

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            # Disable split-KV. With ``num_splits=0`` (auto) FA4's CuTeDSL
            # kernel runs ``num_splits_heuristic`` and, for the few m-blocks a
            # TP-sharded encoder produces at moderate seqlens, picks
            # ``num_splits > 1``. That enables the ``is_split_kv`` path in
            # ``flash_fwd_sm100.py``, whose ``n_block_first`` is typed ``None``
            # on one branch and ``Int32`` on another -- a TYPE_UNSTABLE_JOIN
            # CuTe compile error on Blackwell (SM100).
            num_splits=1,
        )

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)
