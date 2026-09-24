"""Vision transformer block for Qwen VL models.

Unified across Qwen2-VL and Qwen3-VL:
  - act_fn: Qwen2 uses QuickGELU (default), Qwen3 uses SiLU.
  - norm_eps: configurable LayerNorm epsilon.

Uses LayerNorm (not RMSNorm) with pre-norm residual connections,
encoder-only attention, and vision MLP.

The four GEMMs stay on cuBLAS and attention stays on FlashAttention; what is
hand-written here is everything *around* them.  At the captured shapes
(seq ~20.7k-26.4k, d=1152, 16 heads, head_dim=72, mlp_hidden=4304) the block
pushes ~630 GFLOP through the GEMMs but ~1.4 GB/call through plumbing --
a permute/copy, rotary, activation, two residual adds, two LayerNorms -- which
profiled at 508 us of a 1.26 ms block on B200, 40% of it, and is almost all
avoidable.  Worse, that plumbing ran at 1.6-3.9 TB/s where a plain ``add``
reaches 6.7 TB/s on the same GPU, so the *efficiency* was as wrong as the
traffic.  Four changes, in payoff order:

  1. One Triton pass reads the qkv GEMM output and writes rotary-applied q and
     k straight into FlashAttention's layout, replacing a
     ``permute().contiguous()`` plus an in-place ``apply_rotary`` (207 -> 50 us).
  2. The activation moves into fc1's cuBLASLt epilogue, so the ~199 MB
     [seq, 4304] intermediate is never re-read just to be activated (-71 us).
  3. One kernel emits the updated residual *and* the normalized activation,
     replacing an add plus a LayerNorm (94 -> 42 us); the trailing
     ``x + mlp(x)`` becomes a beta=1 accumulate inside fc2 (-24 us).
  4. The ``max_seqlen`` readback is issued before the block's first launch and
     joined only after everything independent of it is in flight, so the
     device->host stall costs no GPU idle (90 -> 12 us of gap).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.quickgelu import QuickGELU
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import VisionMLP


# ---------------------------------------------------------------------------
# Activations.  ``act_fn`` is a caller-supplied Callable, so a fused activation
# is only reachable for the ones we recognise; anything else keeps calling
# ``act_fn`` itself on the materialized intermediate.
# ---------------------------------------------------------------------------
# (rows, num_warps) for norm1 and for the fused add+norm2; (block_m, block_h,
# num_warps) for the rotary pass.
_LN1_LAUNCH = (4, 4)
_LN2_LAUNCH = (4, 4)
_ROPE_LAUNCH = (1, 32, 4)

_ACT_GELU_TANH = 0
_ACT_QUICKGELU = 1
_ACT_SILU = 2
_ACT_GELU_ERF = 3


def _act_code(fn) -> int | None:
    name = type(fn).__name__
    if name == "QuickGELU":
        return _ACT_QUICKGELU
    if name in ("GELU", "GELUActivation", "NewGELUActivation"):
        return (_ACT_GELU_TANH if getattr(fn, "approximate", "none") == "tanh"
                else _ACT_GELU_ERF)
    if name in ("SiLU", "SiLUActivation"):
        return _ACT_SILU
    if getattr(fn, "__name__", None) == "silu":
        return _ACT_SILU
    if getattr(fn, "__name__", None) == "gelu":
        return _ACT_GELU_ERF
    return None


@triton.jit
def _act_kernel(X, Y, n, ACT: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(X + off, mask=m, other=0.0).to(tl.float32)
    # ``0.5 * (1 + tanh(u)) == sigmoid(2u)``, so tanh-GELU needs only the
    # (numerically stable) sigmoid rather than a libdevice tanh.
    if ACT == 0:  # GELU(tanh)
        y = x * tl.sigmoid(2.0 * 0.7978845608028654 * (x + 0.044715 * x * x * x))
    elif ACT == 1:  # QuickGELU
        y = x * tl.sigmoid(1.702 * x)
    elif ACT == 2:  # SiLU
        y = x * tl.sigmoid(x)
    else:  # exact GELU
        y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


def _act_(x, act, block=4096, num_warps=8):
    n = x.numel()
    _act_kernel[(triton.cdiv(n, block),)](x, x, n, ACT=act, BLOCK=block,
                                          num_warps=num_warps)
    return x


# ---------------------------------------------------------------------------
# LayerNorm, optionally folding in the residual branch.
#
# bf16 in / bf16 out with an fp32 reduction, matching ``F.layer_norm`` on bf16
# (whose accumulator type is already float) -- *not* ``promote_fp32=True``, which
# would add two full-tensor casts per norm.  The variance is taken about the
# measured mean rather than as E[x^2]-E[x]^2; the one-pass form measured the same
# speed here and is the one that loses precision when |mean| >> sigma.
#
# With RESID the kernel also emits ``x + a`` (plus *resid_bias*), so the residual
# stream materializes in the pass that normalizes it instead of needing a
# separate 3-tensor add: 219 MB of traffic at 5.2 TB/s in place of an add at
# 6.7 TB/s plus a norm at 1.6 TB/s.
# ---------------------------------------------------------------------------
@triton.jit
def _layer_norm_kernel(
    X, A, R, Y, W, B, RB,
    stride_x, stride_a, stride_r, stride_y,
    M, N, eps,
    ROWS: tl.constexpr, BLOCK_N: tl.constexpr,
    RESID: tl.constexpr, HAS_B: tl.constexpr, HAS_RB: tl.constexpr,
):
    rm = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    mask = (rm[:, None] < M) & cmask[None, :]

    x = tl.load(X + rm[:, None] * stride_x + cols[None, :], mask=mask,
                other=0.0).to(tl.float32)
    if RESID:
        x += tl.load(A + rm[:, None] * stride_a + cols[None, :], mask=mask,
                     other=0.0).to(tl.float32)
        # *resid_bias* is fc2's bias, pushed one stage upstream so fc2 can
        # accumulate straight into the residual (see ``_fc2_add``).  It must not
        # reach the normalized branch, which normalizes x + attn alone.
        if HAS_RB:
            r = x + tl.load(RB + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        else:
            r = x
        tl.store(R + rm[:, None] * stride_r + cols[None, :],
                 r.to(R.dtype.element_ty), mask=mask)

    mean = tl.sum(x, 1) / N
    xc = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, 1) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :]
    if HAS_B:
        y += tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + rm[:, None] * stride_y + cols[None, :],
             y.to(Y.dtype.element_ty), mask=mask)


def _layer_norm(x, weight, bias, eps, out=None, resid=None, resid_out=None,
                resid_bias=None, rows=4, num_warps=4):
    """``LayerNorm(x + resid)``, also writing ``x + resid + resid_bias`` to
    *resid_out*."""
    M, N = x.shape
    if out is None:
        out = torch.empty_like(x)
    grid = (triton.cdiv(M, rows),)
    _layer_norm_kernel[grid](
        x, resid, resid_out, out, weight, bias, resid_bias,
        x.stride(0), resid.stride(0) if resid is not None else 0,
        resid_out.stride(0) if resid_out is not None else 0, out.stride(0),
        M, N, eps,
        ROWS=rows, BLOCK_N=triton.next_power_of_2(N),
        RESID=resid is not None, HAS_B=bias is not None,
        HAS_RB=resid_bias is not None, num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Rotary + q/k extraction, one pass over the qkv GEMM output.
#
# The reference path is ``qkv[..., :2q].permute(...).contiguous()`` followed by
# an in-place ``apply_rotary``: two full round-trips of q and k through HBM,
# ~440 MB/call at the captured shapes, where one suffices.  Here q and k are
# read once, rotated, and written straight into the ``(total_tokens, heads,
# head_dim)`` layout FlashAttention wants.  v is not touched at all -- a
# row-strided view of the qkv buffer is already a legal FA input, and FA
# measured identical (0.249 ms) on strided and contiguous v.
#
# The head axis spans q *and* k: ``hh`` in ``[0, 2*heads)`` indexes column
# ``hh*head_dim``, which lands in the k half once ``hh >= heads`` because q
# occupies exactly ``heads*head_dim`` columns.  cos/sin are indexed by token and
# rotary offset only, so they load as a [rows, 1, rot_half] tile and broadcast
# across heads -- the flat-column alternative needs a gather per element for the
# rotation partner and for cos/sin, and measured 3.74 TB/s against 4.36 here.
# ---------------------------------------------------------------------------
@triton.jit
def _rope_split_kernel(
    QKV, OUT, COS, SIN,
    stride_qkv, stride_out, stride_cos,
    M, HH,
    HEAD_DIM: tl.constexpr, ROT_HALF: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_R: tl.constexpr,
):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    r = tl.arange(0, BLOCK_R)
    row_ok = (rm < M)[:, None, None]
    rot_ok = (r < ROT_HALF)[None, None, :]
    mask = row_ok & (hh < HH)[None, :, None] & rot_ok

    head = hh[None, :, None] * HEAD_DIM + r[None, None, :]
    src = QKV + rm[:, None, None] * stride_qkv + head
    x0 = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(src + ROT_HALF, mask=mask, other=0.0).to(tl.float32)

    coff = rm[:, None, None] * stride_cos + r[None, None, :]
    cmask = row_ok & rot_ok
    cs = tl.load(COS + coff, mask=cmask, other=1.0).to(tl.float32)
    sn = tl.load(SIN + coff, mask=cmask, other=0.0).to(tl.float32)

    dst = OUT + rm[:, None, None] * stride_out + head
    tl.store(dst, (x0 * cs - x1 * sn).to(OUT.dtype.element_ty), mask=mask)
    tl.store(dst + ROT_HALF, (x0 * sn + x1 * cs).to(OUT.dtype.element_ty),
             mask=mask)


def _rope_split(qkv, cos, sin, num_heads, head_dim, block_m=1, block_h=32,
                num_warps=4):
    M = qkv.shape[0]
    q_size = num_heads * head_dim
    out = torch.empty((M, 2 * q_size), dtype=qkv.dtype, device=qkv.device)
    rot_half = cos.shape[-1]
    block_r = triton.next_power_of_2(rot_half)
    grid = (triton.cdiv(M, block_m), triton.cdiv(2 * num_heads, block_h))
    _rope_split_kernel[grid](
        qkv, out, cos, sin,
        qkv.stride(0), out.stride(0), cos.stride(0),
        M, 2 * num_heads,
        HEAD_DIM=head_dim, ROT_HALF=rot_half,
        BLOCK_M=block_m, BLOCK_H=block_h, BLOCK_R=block_r,
        num_warps=num_warps,
    )
    if 2 * rot_half < head_dim:
        # rotary_dim < head_dim: the kernel wrote only the rotated prefix of each
        # head, so the remaining columns still have to pass through untouched.
        # Never taken at the captured shapes (2*rot_half == head_dim), so it is a
        # plain strided copy rather than another branch inside the kernel.
        heads2 = 2 * num_heads
        src = qkv[:, :heads2 * head_dim].view(M, heads2, head_dim)
        out.view(M, heads2, head_dim)[:, :, 2 * rot_half:] = (
            src[:, :, 2 * rot_half:])
    return out


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        # promote_fp32=False to match vLLM, whose vision blocks use a plain
        # ``nn.LayerNorm`` on the bf16 activations (qwen3_vl.py:
        # ``norm_layer = partial(nn.LayerNorm, eps=1e-6)``). Our default promotes
        # to fp32 for the reduction, which exists for the DeepSeek-V3.2 indexer's
        # k_norm and is wrong to apply here: it costs an ``x.float()`` and a
        # ``.to(bf16)`` -- two full-tensor copies -- on every norm, and a Qwen3-VL
        # encoder pass runs 54 of them. Profiled against vLLM's encoder, that was
        # 11.7ms/call of aten::copy_ in ``unrolled_elementwise<direct_copy>``
        # that vLLM never emits. PyTorch's bf16 layer_norm already accumulates in
        # fp32 internally, so the reduction precision is unchanged.
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)

        self.embed_dim = embed_dim
        self.norm_eps = norm_eps
        self._act = _act_code(act_fn)
        self._gelu_epilogue = (
            self._act in (_ACT_GELU_TANH, _ACT_GELU_ERF)
            and self.mlp.fc1.bias is not None
            and hasattr(torch, "_addmm_activation"))
        self._fc2_acc = self.mlp.fc2.bias is not None
        # Launch shapes for the two hand-written kernels, tuned in-block (their
        # standalone optima differ -- microbenchmarks re-touch the same buffers
        # every iteration and so overstate cache reuse).
        self._ln1 = _LN1_LAUNCH
        self._ln2 = _LN2_LAUNCH
        self._rope = _ROPE_LAUNCH
        # Pinned staging buffer + event for the max_seqlen readback.
        self._cu_host: torch.Tensor | None = None
        self._cu_event: torch.cuda.Event | None = None

    # -- max_seqlen without a mid-forward stall -----------------------------
    #
    # ``max_seqlen`` arrives as None from every captured caller, and FA needs it
    # as a host int -- it sizes the tile scheduler.  The reference path computes
    # ``cu_seqlens.diff().max().item()`` *between* the qkv GEMM and attention,
    # draining the stream at the one point in the block with nothing queued
    # behind it: ~90 us/call of GPU idle on a 1.26 ms block, paid 27 times per
    # encoder pass.
    #
    # So the readback is enqueued first, ahead of every other launch, and joined
    # only once all the work that does not depend on it is in flight.  The event
    # covers just a 124-byte copy sitting at the front of the queue, so the host
    # wait finds it long since done and the GPU never idles.  A sync-free upper
    # bound (total_tokens) exists but would oversize FA's grid by the segment
    # count -- ~30x mostly-empty tiles here.
    def _start_max_seqlen(self, cu_seqlens: torch.Tensor):
        n = cu_seqlens.numel()
        host = self._cu_host
        if host is None or host.numel() < n or host.dtype != cu_seqlens.dtype:
            host = torch.empty(max(n, 64), dtype=cu_seqlens.dtype,
                               device="cpu", pin_memory=True)
            self._cu_host = host
        if self._cu_event is None:
            self._cu_event = torch.cuda.Event()
        host = host[:n]
        host.copy_(cu_seqlens, non_blocking=True)
        self._cu_event.record()
        return host

    def _finish_max_seqlen(self, host: torch.Tensor) -> int:
        self._cu_event.synchronize()
        return int((host[1:] - host[:-1]).max())

    # -- fc1 with the activation folded into the GEMM epilogue ---------------
    #
    # cuBLASLt can apply bias+GELU as a matmul epilogue, reachable from ATen as
    # ``_addmm_activation``.  That keeps the [seq, 4304] fc1 output -- ~199 MB
    # here -- from being read and rewritten purely to activate it: 0.168 ms
    # against 0.247 ms for linear-then-GELU at seq=23760.  The epilogue's GELU is
    # the tanh approximation, but tanh- and erf-GELU agree to ~3e-4 absolute, two
    # orders below the bf16 comparison tolerance, so it serves the exact variant
    # too.  QuickGELU and SiLU have no epilogue and keep the activation kernel.
    def _fc1_act(self, normed):
        fc1 = self.mlp.fc1
        if self._gelu_epilogue:
            try:
                return torch._addmm_activation(fc1.bias, normed,
                                               fc1.weight.t(), use_gelu=True)
            except Exception:  # noqa: BLE001 - no epilogue in this build
                self._gelu_epilogue = False
        hid = F.linear(normed, fc1.weight, fc1.bias)
        _act_(hid, self._act)
        return hid

    # -- fc2, accumulating into the residual instead of adding afterwards ----
    #
    # ``resid.addmm_(hid, W2.T)`` is a beta=1 GEMM: cuBLAS reads the residual as
    # C and writes the sum, so the trailing ``x + mlp(x)`` costs no kernel and no
    # extra tensor.  Measured free (0.196 ms against 0.195 for the plain fc2)
    # because fc2 is compute-bound and the extra C stream hides under it, where
    # the separate add cost 0.021 ms.  fc2's bias rode upstream into the residual
    # (see ``_layer_norm``), a beta=1 GEMM having no bias epilogue left to use.
    def _fc2_add(self, hid, resid):
        fc2 = self.mlp.fc2
        if self._fc2_acc:
            try:
                resid.addmm_(hid, fc2.weight.t())
                return resid
            except Exception:  # noqa: BLE001
                self._fc2_acc = False
        resid += F.linear(hid, fc2.weight, fc2.bias)
        return resid

    # -- eligibility --------------------------------------------------------
    def _fused_ok(self, x, cos, sin) -> bool:
        attn = self.attn
        if self._act is None:
            return False
        if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
            return False
        if x.dim() != 3 or x.shape[1] != 1:
            return False
        if attn.tp_size > 1 or self.mlp.fc2.tp_size > 1:
            return False
        if (attn.qkv.use_fp8 or attn.proj.use_fp8
                or self.mlp.fc1.use_fp8 or self.mlp.fc2.use_fp8):
            return False
        if attn.qkv.num_kv_heads != attn.num_heads:
            return False
        if self.norm1.weight is None or self.norm2.weight is None:
            return False
        if cos is not None:
            if sin is None or cos.dim() != 2 or cos.shape[0] < x.shape[0]:
                return False
            if 2 * cos.shape[-1] > attn.head_dim:
                return False
        return True

    def _forward_ref(self, x, cu_seqlens, cos, sin, max_seqlen):
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin, max_seqlen)
        return x + self.mlp(self.norm2(x))

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if not self._fused_ok(x, rotary_pos_emb_cos, rotary_pos_emb_sin):
            return self._forward_ref(x, cu_seqlens, rotary_pos_emb_cos,
                                     rotary_pos_emb_sin, max_seqlen)

        seq_len, batch_size, dim = x.shape
        attn, mlp = self.attn, self.mlp
        nheads, head_dim = attn.num_heads, attn.head_dim
        q_size = nheads * head_dim

        pending = None
        if max_seqlen is None and cu_seqlens.is_cuda:
            pending = self._start_max_seqlen(cu_seqlens)

        x2 = x.reshape(seq_len, dim)
        h = _layer_norm(x2, self.norm1.weight, self.norm1.bias, self.norm_eps,
                        rows=self._ln1[0], num_warps=self._ln1[1])
        qkv = F.linear(h, attn.qkv.weight, attn.qkv.bias)
        qkv3 = qkv.view(seq_len, 3, nheads, head_dim)

        if rotary_pos_emb_cos is not None:
            qk = _rope_split(qkv, rotary_pos_emb_cos, rotary_pos_emb_sin,
                             nheads, head_dim, *self._rope)
            q = qk[:, :q_size].view(seq_len, nheads, head_dim)
            k = qk[:, q_size:].view(seq_len, nheads, head_dim)
        else:
            q, k = qkv3[:, 0], qkv3[:, 1]
        v = qkv3[:, 2]

        if pending is not None:
            max_seqlen = self._finish_max_seqlen(pending)
        elif max_seqlen is None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

        out = attn.attn(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            softmax_scale=head_dim ** -0.5, causal=False,
            # See VisionAttention: num_splits=0 (auto) picks FA4's split-KV
            # kernel at these seqlens, which fails to compile on SM100.
            num_splits=1,
        )
        proj = F.linear(out.reshape(seq_len, q_size), attn.proj.weight,
                        attn.proj.bias)

        resid = torch.empty_like(x2)
        normed = _layer_norm(
            x2, self.norm2.weight, self.norm2.bias, self.norm_eps,
            resid=proj, resid_out=resid,
            resid_bias=mlp.fc2.bias if self._fc2_acc else None,
            rows=self._ln2[0], num_warps=self._ln2[1])
        resid = self._fc2_add(self._fc1_act(normed), resid)
        return resid.view(seq_len, batch_size, dim)
