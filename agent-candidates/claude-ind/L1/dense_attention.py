"""Dense (non-paged) multi-head attention -- Triton fused forward.

Same public surface as the baseline (``DenseAttention(backend=...)``, identical
``forward`` signature).  The maskless fp16/bf16 shapes that the library kernels
handle badly are served by two hand-written Triton kernels; everything else
falls through to the baseline SDPA/cuDNN logic, which is preserved verbatim.

``_attn_tiny_fwd`` -- *group-packed* attention for very short sequences
    (``S <= 16``).  The captured ``[144, 2..6, 16, 64]`` shapes are 2304
    independent 6x6 attentions; dispatched one-at-a-time they cost ~116 us of
    pure launch/setup.  Here many ``(batch, head)`` groups are flattened into a
    single ``BLOCK_R``-row tile and *one* ``BLOCK_R x BLOCK_R`` score matrix
    serves all of them, with cross-group entries masked out.  Masked-out
    probabilities are exactly zero, so the same trick makes the following
    ``P @ V`` matmul produce per-group results with no extra bookkeeping --
    the whole operator becomes a single wave of small CTAs.

``_attn_fwd`` -- a flash-attention forward (online softmax, fp32 accumulators)
    over ``(batch, head, seq, dim)`` *strided* views, so the ``(B, S, H, D)``
    captures never need a transpose or a contiguous copy.  Used for the short
    ``S <= 256`` / ``head_dim <= 64`` shapes, where SDPA's kernels are
    launch-bound rather than compute-bound.

Both kernels fold ``log2(e)`` into the softmax scale and use ``exp2`` directly,
saving one multiply per score element.

Long sequences and ``head_dim > 64`` are left to the baseline: cuDNN's
``sdpa_sm100_flash_*`` kernels already run the big bf16 joint-attention shapes
(``[1, 4608, 24, 128]``) at ~1 PFLOP/s, which a Triton ``tl.dot`` pipeline does
not reach on sm100.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

# cuDNN's SDPA kernels are limited to head_dim <= 128 ("head_dim should be no
# more than 128" in sdp_utils.cpp); larger heads must use EFFICIENT/MATH.
_CUDNN_MAX_HEAD_DIM = 128

_LOG2E = 1.4426950408889634


def _resolve_flash_attn_func():
    """Return the flash-attention callable for Ampere/Hopper."""
    for mod in ("fa3_fwd_interface", "flash_attn_interface"):
        try:
            return __import__(mod, fromlist=["flash_attn_func"]).flash_attn_func
        except (ImportError, ModuleNotFoundError):
            pass
    from flash_attn import flash_attn_func
    return flash_attn_func


# ###########################################################################
# Triton kernels
# ###########################################################################

@triton.jit
def _attn_tiny_fwd(
    Q, K, V, O,
    sqb, sqh, sqs, skb, skh, sks, svb, svh, svs, sob, soh, sos,
    qk_scale, NG,
    H: tl.constexpr,
    S: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_R: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Short-sequence attention, ``BLOCK_G`` (batch, head) groups per CTA.

    Tile rows are ``(group, position)`` pairs; scores between different groups
    are driven to -1e30 so their probabilities vanish and the single ``P @ V``
    matmul stays per-group correct.  grid = (cdiv(NG, BLOCK_G),).
    """
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    g = pid * BLOCK_G + r // S
    s = r % S
    valid = (r < BLOCK_G * S) & (g < NG)
    b = g // H
    h = g % H
    d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + (b * sqb + h * sqh + s * sqs)[:, None] + d[None, :],
                mask=valid[:, None], other=0.0)
    kt = tl.load(K + (b * skb + h * skh + s * sks)[None, :] + d[:, None],
                 mask=valid[None, :], other=0.0)
    # V is independent of the scores: issue its load before the first dot so
    # the three cold HBM reads overlap instead of serialising.
    v = tl.load(V + (b * svb + h * svh + s * svs)[:, None] + d[None, :],
                mask=valid[:, None], other=0.0)

    qk = tl.dot(q, kt) * qk_scale
    keep = (g[:, None] == g[None, :]) & valid[None, :]
    if IS_CAUSAL:
        keep = keep & (s[None, :] <= s[:, None])
    qk = tl.where(keep, qk, -1.0e30)

    m_i = tl.max(qk, 1)
    p = tl.exp2(qk - m_i[:, None])
    l_i = tl.sum(p, 1)

    acc = tl.dot(p.to(v.dtype), v) / l_i[:, None]
    tl.store(O + (b * sob + h * soh + s * sos)[:, None] + d[None, :],
             acc.to(O.dtype.element_ty), mask=valid[:, None])


@triton.jit
def _attn_fwd(
    Q, K, V, O,
    sqb, sqh, sqs, skb, skh, sks, svb, svh, svs, sob, soh, sos,
    qk_scale,
    SEQ_Q, SEQ_K,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Flash-attention forward.  grid = (cdiv(SEQ_Q, BLOCK_M), B * H)."""
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + (b * sqb + h * sqh) + offs_m[:, None] * sqs + offs_d[None, :]
    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_Q, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    kt_base = K + (b * skb + h * skh) + offs_d[:, None]
    v_base = V + (b * svb + h * svh) + offs_d[None, :]

    if IS_CAUSAL:
        hi = tl.minimum(SEQ_K, (start_m + 1) * BLOCK_M)
    else:
        hi = SEQ_K

    for start_n in tl.range(0, hi, BLOCK_N):
        offs_nn = start_n + offs_n
        if EVEN_N:
            kt = tl.load(kt_base + offs_nn[None, :] * sks)
            v = tl.load(v_base + offs_nn[:, None] * svs)
        else:
            nmask = offs_nn < SEQ_K
            kt = tl.load(kt_base + offs_nn[None, :] * sks,
                         mask=nmask[None, :], other=0.0)
            v = tl.load(v_base + offs_nn[:, None] * svs,
                        mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, kt) * qk_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_nn[None, :], qk, -1.0e30)
        elif not EVEN_N:
            qk = tl.where(offs_nn[None, :] < SEQ_K, qk, -1.0e30)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + (b * sob + h * soh) + offs_m[:, None] * sos + offs_d[None, :]
    if EVEN_M:
        tl.store(o_ptrs, acc.to(O.dtype.element_ty))
    else:
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < SEQ_Q)


# ###########################################################################
# Dispatch
# ###########################################################################
# Row tile for the group-packed kernel.  A BLOCK_R x BLOCK_R score tile holds
# BLOCK_R/S groups, so it computes BLOCK_R*S scores per group where only S*S
# are wanted -- the waste is BLOCK_R/S and is minimised by the smallest tile
# ``tl.dot`` accepts (16).  Measured best across S = 2..6 on B200.
_TINY_BLOCK_R = 16
_TINY_WARPS = 4

# (BLOCK_M, BLOCK_N, num_warps, num_stages) for the flash kernel.  Measured on
# B200 for the captured short shapes: one warp per CTA and the smallest useful
# row tile win, because these launches are latency- rather than throughput
# bound and extra warps only add cross-warp softmax reductions.
_FWD_CFG = (16, 32, 1, 3)

_FAST_DTYPES = (torch.float16, torch.bfloat16)
# Head dims the Triton path serves; past 64, and past ~256 keys, the baseline's
# cuDNN flash kernels are comfortably faster and take over.
_FAST_DIMS = (16, 32, 64)
_FAST_MAX_SEQ = 256


def _triton_attention(q, k, v, scale, causal):
    """q/k/v: (B, H, S, D) strided views with an innermost stride of 1."""
    B, H, SQ, D = q.shape
    SK = k.shape[2]
    o = torch.empty((B, H, SQ, D), dtype=q.dtype, device=q.device)
    strides = (
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
    )
    qk_scale = scale * _LOG2E
    if SQ == SK and SQ <= 16:
        ng = B * H
        br = _TINY_BLOCK_R
        bg = br // SQ
        _attn_tiny_fwd[((ng + bg - 1) // bg,)](
            q, k, v, o, *strides, qk_scale, ng,
            H=H, S=SQ, HEAD_DIM=D, BLOCK_G=bg, BLOCK_R=br,
            IS_CAUSAL=causal, num_warps=_TINY_WARPS, num_stages=1,
        )
        return o
    bm, bn, warps, stages = _FWD_CFG
    _attn_fwd[((SQ + bm - 1) // bm, B * H)](
        q, k, v, o, *strides, qk_scale, SQ, SK,
        H=H, HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn, IS_CAUSAL=causal,
        EVEN_M=(SQ % bm == 0), EVEN_N=(SK % bn == 0),
        num_warps=warps, num_stages=stages,
    )
    return o


class DenseAttention(nn.Module):
    """Dense multi-head attention.  Input layout: (batch, seq_len, num_heads, head_dim)."""

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None

        if backend == "sdpa":
            return

        if backend == "cudnn":
            self.use_cudnn_kernel = True
            return

        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            return

        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True

    # -- Triton fast path --------------------------------------------------
    def _try_triton(self, query, key, value, softmax_scale, causal):
        """Return the attention output, or None if this shape isn't covered."""
        shape = query.shape
        if len(shape) != 4:
            return None
        D = shape[3]
        S = shape[1]
        if S == 0 or S > _FAST_MAX_SEQ or shape[0] == 0 or shape[2] == 0:
            return None
        if D not in _FAST_DIMS:
            return None
        if query.dtype not in _FAST_DTYPES:
            return None
        if key.shape != shape or value.shape != shape:
            return None
        if key.dtype is not query.dtype or value.dtype is not query.dtype:
            return None
        if not query.is_cuda:
            return None
        if query.stride(3) != 1 or key.stride(3) != 1 or value.stride(3) != 1:
            return None
        scale = float(softmax_scale) if softmax_scale is not None else D ** -0.5
        out = _triton_attention(
            query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3), scale, bool(causal),
        )
        return out.permute(0, 2, 1, 3)

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask: torch.Tensor | None = None,
    ):
        if attn_mask is None and not self.use_flex_kernel:
            out = self._try_triton(query, key, value, softmax_scale, causal)
            if out is not None:
                return out

        if self.fa_func is not None and attn_mask is None and query.dtype != torch.float32:
            out = self.fa_func(
                query, key, value,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        if self.use_flex_kernel:
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            out = self._flex_fn(
                q, k, v,
                block_mask=attn_mask,
                scale=softmax_scale,
            )
        elif self.use_cudnn_kernel:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            if attn_mask is not None and causal:
                raise ValueError(
                    "DenseAttention: pass either attn_mask or causal=True, not both "
                    "(an explicit mask must already encode causality). Got "
                    f"attn_mask={tuple(attn_mask.shape)} with causal=True."
                )
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
            else:
                try:
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
                except RuntimeError:
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
        else:
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False if attn_mask is not None else causal,
                scale=softmax_scale,
            )
        return out.permute(0, 2, 1, 3)
