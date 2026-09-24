"""Encoder-only attention for Qwen vision transformer blocks.

Non-causal, no KV cache. Uses FlashAttnPrefill L1 op with cu_seqlens
for variable-length sequence support within the vision encoder.

Two changes over the baseline, both aimed at what a profile says the layer
actually spends its time on (per layer, 20680 tokens, B200):

* The baseline makes two full bandwidth-bound passes over the q|k half of the
  fused qkv activation -- a ``.contiguous()`` that re-lays it out as
  ``(2, seq, heads, dim)`` (0.173ms) and an in-place ``apply_rotary`` over the
  result (0.108ms).  ``vision_attention_rope.cu`` does both in one pass
  (0.062ms): each q/k row is fetched from DRAM once and the rotated, contiguous
  result is written straight out.  ``v`` is handed to attention as a strided view of
  ``qkv``, exactly as the baseline does, so it is never copied at all.

* That kernel also emits q and k at a head_dim rounded up to a multiple of 32.
  FlashAttention's SM100 forward has tuned tiles at those sizes and falls onto a
  slower path at head_dim 72 (0.349ms vs 0.269ms measured, i.e. padding to 96
  wins despite the extra 33% of q@k^T work).  Padding q/k with zero channels
  cannot change ``q @ k^T``; ``v`` keeps its real width, so the attention output
  and ``proj`` are untouched.

Output is bitwise identical to the baseline on every captured shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from flash_attn.ops.triton.rotary import apply_rotary

from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_size, _tp_rank
from ..L1.flash_attn_prefill import FlashAttnPrefill
from .parallel_linear import QKVParallelLinear, RowParallelLinear

_C = lazy_op("fk_l2_vision_attention_rope", "vision_attention_rope.cu")


@triton.jit
def _qk_rope_kernel(
    QKV, COS, SIN, Q, K,
    n_tokens, stride_qkv_row,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr, HALF: tl.constexpr, QSIZE: tl.constexpr,
    PAD_DIM: tl.constexpr, PAD_SIZE: tl.constexpr,
):
    """Triton port of ``vision_attention_rope.cu``, for shapes it can't launch.

    Each program owns a ``BLOCK_R x BLOCK_C`` tile of the (token, channel) grid
    of the *output* and emits it for both q and k.  Channels are indexed in the
    destination layout (head stride ``PAD_DIM`` >= ``HEAD_DIM``) so the stores
    stay affine; the padding channels are skipped, which is why the destination
    has to be allocated zeroed on this path.

    For channel ``d`` the rotary partner is ``d +/- HALF`` inside the same head,
    so the source row is read twice: once at ``d`` and once at the partner.  The
    partner read hits cache lines a neighbouring tile already pulled in, so it
    costs L1/L2 hits rather than DRAM traffic.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_tokens
    oc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    head = oc // PAD_DIM
    d = oc - head * PAD_DIM
    is_lo = d < HALF
    c = head * HEAD_DIM + d
    partner = tl.where(is_lo, c + HALF, c - HALF)
    rot = tl.where(is_lo, d, d - HALF)
    mask = rmask[:, None] & (d < HEAD_DIM)[None, :]

    cs = tl.load(COS + rows[:, None] * HALF + rot[None, :], mask=mask,
                 other=0.0).to(tl.float32)
    sn = tl.load(SIN + rows[:, None] * HALF + rot[None, :], mask=mask,
                 other=0.0).to(tl.float32)
    # o_lo = x_lo * cos - x_hi * sin ; o_hi = x_lo * sin + x_hi * cos.  Folding
    # the sign into ``sin`` makes both halves the same expression.
    sn = tl.where(is_lo[None, :], -sn, sn)

    src = QKV + rows[:, None] * stride_qkv_row
    dst = rows[:, None] * PAD_SIZE + oc[None, :]

    q_self = tl.load(src + c[None, :], mask=mask, other=0.0).to(tl.float32)
    q_part = tl.load(src + partner[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(Q + dst, (q_self * cs + q_part * sn).to(Q.dtype.element_ty), mask=mask)

    k_self = tl.load(src + QSIZE + c[None, :], mask=mask, other=0.0).to(tl.float32)
    k_part = tl.load(src + QSIZE + partner[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(K + dst, (k_self * cs + k_part * sn).to(K.dtype.element_ty), mask=mask)


def _block_c(pad_size: int) -> int:
    """Largest power-of-two tile width that divides ``pad_size`` (<= 128)."""
    bc = 128
    while bc > 1 and pad_size % bc:
        bc //= 2
    return bc


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

        # Width q/k are emitted at; see the module docstring.
        self.pad_dim = -(-self.head_dim // 32) * 32
        # The CUDA kernel addresses (pair-group, head, q-or-k) by thread
        # coordinate, so it needs 8-byte-aligned channel groups and a block that
        # fits; anything else uses the Triton port.
        self._cuda_rope = (
            self.head_dim % 8 == 0
            and self.head_dim // 8 <= 32
            and (self.head_dim // 8) * self.num_heads * 2 <= 1024
        )

    def _split_rope(self, qkv, seq_len, cos, sin):
        """Rotated q/k at the padded head_dim, plus a strided view of ``v``."""
        q_size = self.num_heads * self.head_dim
        pad_dim = self.pad_dim
        flat = qkv.view(seq_len, 3 * q_size)
        shape = (2, seq_len, self.num_heads, pad_dim)
        buf = None
        if self._cuda_rope and flat.dtype in (torch.bfloat16, torch.float16):
            # The kernel writes the padding channels too, so the buffer does not
            # need pre-zeroing (and writing them is itself cheaper than leaving
            # partially-dirty cache lines behind -- see the .cu).
            buf = torch.empty(shape, device=flat.device, dtype=flat.dtype)
            # Enough tokens per block to amortize the per-token cos/sin staging,
            # but not so many that the grid stops filling the GPU.
            rows = 8 if seq_len > 32768 else (4 if seq_len > 4096 else 2)
            try:
                _C.qk_rope(flat, cos, sin, buf[0], buf[1], rows)
            except Exception:
                # No usable nvcc, or a layout the kernel rejects: fall back to
                # the Triton port for the rest of this module's life rather
                # than failing the layer.
                self._cuda_rope = False
                buf = None
        if buf is None:
            # The Triton kernel skips the padding channels, so they must already
            # be zero.
            buf = torch.zeros(shape, device=flat.device, dtype=flat.dtype)
            pad_size = self.num_heads * pad_dim
            block_c = _block_c(pad_size)
            block_r = 16
            _qk_rope_kernel[(triton.cdiv(seq_len, block_r), pad_size // block_c)](
                flat, cos, sin, buf[0], buf[1], seq_len, flat.stride(0),
                BLOCK_R=block_r, BLOCK_C=block_c,
                HEAD_DIM=self.head_dim, HALF=self.head_dim // 2, QSIZE=q_size,
                PAD_DIM=pad_dim, PAD_SIZE=pad_size,
                num_warps=4, num_stages=1,
            )
        v = flat[:, 2 * q_size:].view(seq_len, self.num_heads, self.head_dim)
        return buf[0], buf[1], v

    def _split_rope_ref(self, qkv, seq_len, batch_size, cos, sin):
        """Baseline layout path: one permute-copy plus an in-place rotary."""
        q_size = self.num_heads * self.head_dim
        qk = qkv[..., : 2 * q_size].view(
            seq_len, batch_size, 2, self.num_heads, self.head_dim,
        )
        qk = qk.permute(2, 1, 0, 3, 4).contiguous()
        if cos is not None and sin is not None:
            apply_rotary(qk.view(2 * batch_size, seq_len, self.num_heads,
                                 self.head_dim), cos, sin, inplace=True)
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

        # The fused kernel assumes one packed batch and a rotary that covers the
        # whole head; anything else takes the baseline layout path.
        fused = (
            batch_size == 1
            and rotary_pos_emb_cos is not None
            and rotary_pos_emb_sin is not None
            and self.head_dim % 2 == 0
            and 2 * rotary_pos_emb_cos.shape[-1] == self.head_dim
            and rotary_pos_emb_cos.is_contiguous()
            and rotary_pos_emb_sin.is_contiguous()
            and rotary_pos_emb_cos.shape[0] >= seq_len
            and rotary_pos_emb_cos.dtype == x.dtype
            and rotary_pos_emb_sin.dtype == x.dtype
            and _block_c(self.num_heads * self.pad_dim) >= 8
        )
        if fused:
            q, k, v = self._split_rope(qkv, seq_len, rotary_pos_emb_cos,
                                       rotary_pos_emb_sin)
        else:
            q, k, v = self._split_rope_ref(qkv, seq_len, batch_size,
                                           rotary_pos_emb_cos, rotary_pos_emb_sin)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        out = self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            # q/k may be wider than head_dim now; the scale follows the real one.
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            # Disable split-KV. With ``num_splits=0`` (auto) FA4's CuTeDSL
            # kernel runs ``num_splits_heuristic`` and, for the few m-blocks a
            # TP-sharded encoder produces (num_heads // tp, e.g. 16 // 4 = 4)
            # at moderate seqlens, picks ``num_splits > 1``. That enables the
            # ``is_split_kv`` path in ``flash_fwd_sm100.py``, whose
            # ``n_block_first`` is typed ``None`` on one branch and ``Int32``
            # on another -- a TYPE_UNSTABLE_JOIN CuTe compile error on
            # Blackwell (SM100).  Encoder self-attention is balanced
            # (q_len == k_len) so split-KV never helps here; forcing 1 is
            # numerically identical and sidesteps the kernel bug.
            num_splits=1,
        )

        out = out.view(seq_len, batch_size, -1)
        return self.proj(out)
