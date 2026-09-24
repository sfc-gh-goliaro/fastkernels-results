"""FLUX transformer blocks (L3 composites) -- fused Triton implementation.

Same module tree and parameter names as the baseline (so shared weights load
verbatim), but the forward passes replace the baseline's long chain of small
norm / rope / cat / elementwise kernels with a handful of fused Triton kernels,
and overlap the two independent halves of each block on a second CUDA stream.

Fused kernels
-------------
``_ln_mod_kernel``           LayerNorm(no affine) + (1 + scale) * x + shift
``_qk_rope_kernel``          per-head RMSNorm(q), RMSNorm(k) + interleaved RoPE
                             read straight out of the fused QKV GEMM output --
                             no ``.contiguous()`` clones and no q/k/v cats
``_attn_flat_kernel``        attention output -> row-major [S, H*D]
``_gelu_kernel``             tanh-GELU with a strided destination
``_gate_res_kernel``         out = residual + gate * x
``_gate_res2_kernel``        the same for a K-split projection (two partials)
``_gate_res_ln_mod_kernel``  residual add fused with the LayerNorm+modulate
                             that follows it

Numerics follow the baseline op for op, *including* every point where the
baseline materializes a bf16 tensor (``F.layer_norm`` output, the vLLM
``rms_norm`` store, each bf16 elementwise result). At bf16 ULP (~0.4%) against
the benchmark's 1% tolerance those roundings have to be reproduced rather than
skipped, and ``.to(bf16).to(f32)`` is folded away by the compiler, so ``_rnd``
does the round-to-nearest-even on the bit pattern instead.

Scheduling
----------
Both blocks contain two chains that only meet at the block output:

* dual stream: the 512-token text chain vs the image chain (they are joined only
  by the joint attention). The text GEMMs are M=512, barely one wave, so running
  them beside the image chain costs almost nothing.
* single stream: the parallel MLP branch vs the attention branch. Splitting
  ``proj_out`` along K lets the MLP's share of it run while attention is still
  in flight, instead of waiting for the ``cat``.

GEMMs stay on the same cuBLAS path the baseline uses (already ~1.7 PFLOP/s, 75%
of peak here) and attention stays on the cuDNN SDPA kernel ``DenseAttention``
selects on Blackwell.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L2.ada_layer_norm import AdaLayerNormZero, AdaLayerNormZeroSingle
from ..L2.flux_attention import FluxAttention
from ..L2.flux_feedforward import FeedForward
from ..L2.parallel_linear import ReplicatedLinear

# Debug switch: run everything on the current stream (used to get clean
# per-kernel profiles, which concurrent streams make unreadable).
_NO_STREAM = bool(os.environ.get("FK_NOSTREAM"))


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _rnd(x):
    """Round an fp32 value to bfloat16 precision, staying in fp32.

    ``x.to(tl.bfloat16).to(tl.float32)`` is folded away by the compiler, so the
    round-to-nearest-even is done on the bit pattern.
    """
    i = x.to(tl.int32, bitcast=True)
    r = i + 0x7FFF + ((i >> 16) & 1)
    return (r & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _tanh(x):
    # 1 - 2 / (exp(2x) + 1): saturates cleanly for large |x| and tracks tanhf to
    # far below a bf16 ULP over the gelu range.
    return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)


@triton.jit
def _cast_kernel(A, B, OA, OB, N, BLOCK: tl.constexpr):
    """Down-cast the fp64 rope (cos, sin) pair to the activation dtype."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(OA + off, tl.load(A + off, mask=m, other=0.0).to(OA.dtype.element_ty), mask=m)
    tl.store(OB + off, tl.load(B + off, mask=m, other=0.0).to(OB.dtype.element_ty), mask=m)


@triton.jit
def _ln_mod_kernel(X, Y, SCALE, SHIFT, stride_x, stride_y,
                   N: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
    """Y = LayerNorm(X) * (1 + SCALE) + SHIFT, one row per program."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(X + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(m, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    y = _rnd(xc * tl.rsqrt(var + EPS))
    sc = tl.load(SCALE + cols, mask=m, other=0.0).to(tl.float32)
    sh = tl.load(SHIFT + cols, mask=m, other=0.0).to(tl.float32)
    y = _rnd(y * _rnd(1.0 + sc))
    tl.store(Y + row * stride_y + cols, (y + sh).to(Y.dtype.element_ty), mask=m)


@triton.jit
def _qk_rope_kernel(QKV, Q, K, COS, SIN, WQ, WK, WQT, WKT,
                    S, n_text, stride_qkv, stride_q, stride_cs,
                    H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
                    EPS: tl.constexpr, HAS_ROPE: tl.constexpr,
                    TWO_W: tl.constexpr):
    """RMSNorm + interleaved RoPE on the q/k halves of a fused QKV buffer.

    ``TWO_W`` selects the text-stream norm weights for rows below ``n_text``,
    which is how the dual block's separate ``norm_added_q`` / ``norm_added_k``
    are applied without splitting the buffer.
    """
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BS + tl.arange(0, BS)
    rm = rows < S
    d = tl.arange(0, D)
    base = QKV + rows[:, None] * stride_qkv + (h * D + d[None, :])
    q = tl.load(base, mask=rm[:, None], other=0.0).to(tl.float32)
    k = tl.load(base + H * D, mask=rm[:, None], other=0.0).to(tl.float32)

    wq = tl.load(WQ + d).to(tl.float32)[None, :]
    wk = tl.load(WK + d).to(tl.float32)[None, :]
    if TWO_W:
        wqt = tl.load(WQT + d).to(tl.float32)[None, :]
        wkt = tl.load(WKT + d).to(tl.float32)[None, :]
        istxt = (rows < n_text)[:, None]
        wq = tl.where(istxt, wqt, wq)
        wk = tl.where(istxt, wkt, wk)

    rq = tl.rsqrt(tl.sum(q * q, axis=1) / D + EPS)[:, None]
    rk = tl.rsqrt(tl.sum(k * k, axis=1) / D + EPS)[:, None]
    q = _rnd(q * rq * wq)
    k = _rnd(k * rk * wk)

    if HAS_ROPE:
        hd = tl.arange(0, D // 2)
        cs_off = rows[:, None] * stride_cs + hd[None, :]
        cos = tl.load(COS + cs_off, mask=rm[:, None], other=1.0).to(tl.float32)
        sin = tl.load(SIN + cs_off, mask=rm[:, None], other=0.0).to(tl.float32)
        q0, q1 = tl.split(tl.reshape(q, (BS, D // 2, 2)))
        k0, k1 = tl.split(tl.reshape(k, (BS, D // 2, 2)))
        q = tl.reshape(tl.join(q0 * cos - q1 * sin, q0 * sin + q1 * cos), (BS, D))
        k = tl.reshape(tl.join(k0 * cos - k1 * sin, k0 * sin + k1 * cos), (BS, D))

    off = rows[:, None] * stride_q + (h * D + d[None, :])
    dt = Q.dtype.element_ty
    tl.store(Q + off, q.to(dt), mask=rm[:, None])
    tl.store(K + off, k.to(dt), mask=rm[:, None])


@triton.jit
def _attn_flat_kernel(SRC, DST, S, stride_src_h, stride_src_s, stride_dst,
                      D: tl.constexpr, BS: tl.constexpr):
    """[1, H, S, D] attention output -> [S, H*D] rows (DST may be a wide slice)."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    rows = pid * BS + tl.arange(0, BS)
    rm = rows < S
    d = tl.arange(0, D)
    v = tl.load(SRC + h * stride_src_h + rows[:, None] * stride_src_s + d[None, :],
                mask=rm[:, None])
    tl.store(DST + rows[:, None] * stride_dst + (h * D + d[None, :]), v,
             mask=rm[:, None])


@triton.jit
def _gelu_kernel(X, Y, N, stride_x, stride_y, BN: tl.constexpr):
    """Y = gelu_tanh(X), row-strided so Y can be a slice of a wider buffer."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    m = cols < N
    x = tl.load(X + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + _tanh(inner))
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=m)


@triton.jit
def _gate_res_kernel(SRC, P, OUT, GATE, stride_s, stride_p, stride_o,
                     N: tl.constexpr, BLOCK: tl.constexpr):
    """OUT = SRC + GATE * P (GATE broadcast over rows)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    s = tl.load(SRC + row * stride_s + cols, mask=m, other=0.0).to(tl.float32)
    p = tl.load(P + row * stride_p + cols, mask=m, other=0.0).to(tl.float32)
    g = tl.load(GATE + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + row * stride_o + cols,
             (_rnd(p * g) + s).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gate_res2_kernel(SRC, P1, P2, OUT, GATE, stride_s, stride_p, stride_o,
                      N: tl.constexpr, BLOCK: tl.constexpr):
    """OUT = SRC + GATE * (P1 + P2), for a K-split output projection."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    s = tl.load(SRC + row * stride_s + cols, mask=m, other=0.0).to(tl.float32)
    p1 = tl.load(P1 + row * stride_p + cols, mask=m, other=0.0).to(tl.float32)
    p2 = tl.load(P2 + row * stride_p + cols, mask=m, other=0.0).to(tl.float32)
    g = tl.load(GATE + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + row * stride_o + cols,
             (_rnd(_rnd(p1 + p2) * g) + s).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gate_res_ln_mod_kernel(SRC, P, RES, NRM, GATE, SCALE, SHIFT,
                            stride_s, stride_p, stride_r, stride_n,
                            N: tl.constexpr, BLOCK: tl.constexpr,
                            EPS: tl.constexpr):
    """RES = SRC + GATE * P ; NRM = LayerNorm(RES) * (1 + SCALE) + SHIFT."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    dt = RES.dtype.element_ty
    s = tl.load(SRC + row * stride_s + cols, mask=m, other=0.0).to(tl.float32)
    p = tl.load(P + row * stride_p + cols, mask=m, other=0.0).to(tl.float32)
    g = tl.load(GATE + cols, mask=m, other=0.0).to(tl.float32)
    x = _rnd(_rnd(p * g) + s)
    tl.store(RES + row * stride_r + cols, x.to(dt), mask=m)

    x = tl.where(m, x, 0.0)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(m, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    y = _rnd(xc * tl.rsqrt(var + EPS))
    sc = tl.load(SCALE + cols, mask=m, other=0.0).to(tl.float32)
    sh = tl.load(SHIFT + cols, mask=m, other=0.0).to(tl.float32)
    y = _rnd(y * _rnd(1.0 + sc))
    tl.store(NRM + row * stride_n + cols, (y + sh).to(dt), mask=m)


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _ln_mod(x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor,
            shift: torch.Tensor, eps: float) -> None:
    rows, n = x.shape
    _ln_mod_kernel[(rows,)](
        x, out, scale, shift, x.stride(0), out.stride(0),
        N=n, BLOCK=_next_pow2(n), EPS=eps, num_warps=4,
    )


def _gate_res(src: torch.Tensor, p: torch.Tensor, out: torch.Tensor,
              gate: torch.Tensor) -> None:
    rows, n = src.shape
    _gate_res_kernel[(rows,)](
        src, p, out, gate, src.stride(0), p.stride(0), out.stride(0),
        N=n, BLOCK=_next_pow2(n), num_warps=8,
    )


def _gate_res2(src: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
               out: torch.Tensor, gate: torch.Tensor) -> None:
    rows, n = src.shape
    _gate_res2_kernel[(rows,)](
        src, p1, p2, out, gate, src.stride(0), p1.stride(0), out.stride(0),
        N=n, BLOCK=_next_pow2(n), num_warps=8,
    )


def _gate_res_ln_mod(src, p, res, nrm, gate, scale, shift, eps) -> None:
    rows, n = src.shape
    _gate_res_ln_mod_kernel[(rows,)](
        src, p, res, nrm, gate, scale, shift,
        src.stride(0), p.stride(0), res.stride(0), nrm.stride(0),
        N=n, BLOCK=_next_pow2(n), EPS=eps, num_warps=4,
    )


def _gelu_into(x: torch.Tensor, out: torch.Tensor) -> None:
    rows, n = x.shape
    bn = 2048 if n >= 2048 else _next_pow2(n)
    _gelu_kernel[(rows, triton.cdiv(n, bn))](
        x, out, n, x.stride(0), out.stride(0), BN=bn, num_warps=4,
    )


def _attn_flat(out: torch.Tensor, dst: torch.Tensor, heads: int,
               head_dim: int) -> None:
    s = out.shape[2]
    _attn_flat_kernel[(triton.cdiv(s, 16), heads)](
        out, dst, s, out.stride(1), out.stride(2), dst.stride(0),
        D=head_dim, BS=16, num_warps=4,
    )


def _attn_rows(out: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """SDPA output as ``[S, H*D]`` rows; a free view when the layout allows.

    cuDNN returns the output in the query's own (B, S, H, D) arrangement, so the
    ``[S, H, D]`` transpose is already contiguous and no copy is needed.
    """
    t = out[0].transpose(0, 1)
    s = t.shape[0]
    if t.is_contiguous():
        return t.view(s, heads * head_dim)
    flat = torch.empty((s, heads * head_dim), device=out.device, dtype=out.dtype)
    _attn_flat(out, flat, heads, head_dim)
    return flat


def _qk_norm_rope(qkv: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
                  rope, wq, wk, wqt, wkt, n_text: int, heads: int,
                  head_dim: int, eps: float) -> None:
    s = qkv.shape[0]
    bs = 4 if s >= 4 else 1
    has_rope = rope is not None
    cos = rope[0] if has_rope else qkv
    sin = rope[1] if has_rope else qkv
    two_w = wqt is not None
    _qk_rope_kernel[(triton.cdiv(s, bs), heads)](
        qkv, q, k, cos, sin, wq, wk,
        wqt if two_w else wq, wkt if two_w else wk,
        s, n_text, qkv.stride(0), q.stride(0),
        cos.stride(0) if has_rope else 0,
        H=heads, D=head_dim, BS=bs, EPS=eps, HAS_ROPE=has_rope, TWO_W=two_w,
        num_warps=2,
    )


_SDPA_DIRECT = True


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    """cuDNN flash SDPA -- the backend ``DenseAttention`` pins on Blackwell."""
    global _SDPA_DIRECT
    if _SDPA_DIRECT:
        try:
            return torch.ops.aten._scaled_dot_product_cudnn_attention(
                q, k, v, None, False, 0.0, False, scale=scale)[0]
        except Exception:  # noqa: BLE001 - fall back to the dispatcher
            _SDPA_DIRECT = False
    from torch.nn.attention import sdpa_kernel, SDPBackend
    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, scale=scale)
    except RuntimeError:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, scale=scale)


class _Unsupported(Exception):
    pass


def _rope_check(image_rotary_emb, head_dim: int):
    """Validate the (cos, sin) pair. No kernels launched."""
    if image_rotary_emb is None:
        return None
    cos, sin = image_rotary_emb
    if cos.dim() == 3:
        cos, sin = cos[0], sin[0]
    if cos.dim() != 2 or sin.dim() != 2 or sin.shape != cos.shape:
        raise _Unsupported
    if cos.shape[1] * 2 != head_dim:
        raise _Unsupported
    return cos, sin


def _rope_cast(rope, dtype: torch.dtype):
    """Down-cast (cos, sin) to the activation dtype, as the baseline does."""
    if rope is None:
        return None
    cos, sin = rope
    if not cos.is_contiguous() or not sin.is_contiguous():
        cos, sin = cos.contiguous(), sin.contiguous()
    if cos.dtype != dtype:
        n = cos.numel()
        oc = torch.empty_like(cos, dtype=dtype)
        os_ = torch.empty_like(sin, dtype=dtype)
        _cast_kernel[(triton.cdiv(n, 256),)](cos, sin, oc, os_, n, BLOCK=256,
                                             num_warps=2)
        cos, sin = oc, os_
    return cos, sin


# ---------------------------------------------------------------------------
# Stream plumbing
# ---------------------------------------------------------------------------

_SIDE_STREAM = None


def _side_stream(device):
    """A second stream for the block's shorter, independent chain."""
    global _SIDE_STREAM
    if _NO_STREAM:
        return torch.cuda.current_stream(device)
    if _SIDE_STREAM is None or _SIDE_STREAM.device != device:
        _SIDE_STREAM = torch.cuda.Stream(device=device)
    return _SIDE_STREAM


class _Sync:
    """Reusable cross-stream events (``wait_stream`` allocates a fresh one)."""

    __slots__ = ("pre", "mid", "post")

    def __init__(self):
        self.pre = torch.cuda.Event()
        self.mid = torch.cuda.Event()
        self.post = torch.cuda.Event()


def _fast_ok(module, hidden_states, encoder_hidden_states, temb,
             joint_attention_kwargs) -> bool:
    """True when the fused path covers this call exactly."""
    if module._quantized or joint_attention_kwargs:
        return False
    if hidden_states is None or encoder_hidden_states is None or temb is None:
        return False
    if hidden_states.dtype is not torch.bfloat16:
        return False
    return (
        hidden_states.dim() == 3
        and encoder_hidden_states.dim() == 3
        and temb.dim() == 2
        and hidden_states.is_cuda
        and hidden_states.shape[0] == 1
        and encoder_hidden_states.shape[0] == 1
        and temb.shape[0] == 1
        and hidden_states.shape[2] == module.dim
        and encoder_hidden_states.shape[2] == module.dim
        and hidden_states.dtype == encoder_hidden_states.dtype == temb.dtype
        and hidden_states.is_contiguous()
        and encoder_hidden_states.is_contiguous()
        and module.head_dim in (32, 64, 128, 256)
        and module.heads * module.head_dim == module.dim
    )


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class FluxTransformerBlock(nn.Module):
    """Dual-stream DiT block: joint attention over text+image, then separate FFNs."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, promote_fp32=False)
        self.norm1_context = AdaLayerNormZero(dim, promote_fp32=False)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.dim = dim
        self.heads = self.attn.heads
        self.head_dim = self.attn.head_dim
        self._quantized = quant_config is not None
        self._sync = None
        self._adaln_key = None
        self._adaln = None

    def _adaln_fused(self):
        """norm1 + norm1_context projections as one [12*dim, dim] GEMM.

        Both read the same ``silu(temb)`` and both are M=1, i.e. pure weight
        bandwidth, so one launch over a concatenated weight beats two. Rebuilt
        whenever a source parameter is replaced or mutated in place.
        """
        w1, w2 = self.norm1.linear.weight, self.norm1_context.linear.weight
        b1, b2 = self.norm1.linear.bias, self.norm1_context.linear.bias
        if b1 is None or b2 is None:
            return None
        key = tuple((t.data_ptr(), t._version, t.dtype) for t in (w1, w2, b1, b2))
        if self._adaln_key != key:
            self._adaln = (torch.cat([w1, w2], dim=0), torch.cat([b1, b2], dim=0))
            self._adaln_key = key
        return self._adaln

    # -- baseline reference path (for shapes/dtypes the fast path rejects) ----
    def _forward_ref(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        ip_attn_output = None
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if ip_attn_output is not None:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _fast_ok(self, hidden_states, encoder_hidden_states, temb,
                        joint_attention_kwargs):
            return self._forward_ref(hidden_states, encoder_hidden_states, temb,
                                     image_rotary_emb, joint_attention_kwargs)
        proj = self._adaln_fused()
        if proj is None:
            return self._forward_ref(hidden_states, encoder_hidden_states, temb,
                                     image_rotary_emb, joint_attention_kwargs)
        try:
            rope_raw = _rope_check(image_rotary_emb, self.head_dim)
        except _Unsupported:
            return self._forward_ref(hidden_states, encoder_hidden_states, temb,
                                     image_rotary_emb, joint_attention_kwargs)

        C = self.dim
        H, D = self.heads, self.head_dim
        n_img = hidden_states.shape[1]
        n_txt = encoder_hidden_states.shape[1]
        S = n_img + n_txt
        dt = hidden_states.dtype
        dev = hidden_states.device
        attn = self.attn
        img = hidden_states.view(n_img, C)
        txt = encoder_hidden_states.view(n_txt, C)

        main = torch.cuda.current_stream(dev)
        side = _side_stream(dev)
        sync = self._sync
        if sync is None:
            sync = self._sync = _Sync()

        # --- adaLN-Zero conditioning (one silu, one fused projection) --------
        both = F.linear(F.silu(temb), proj[0], proj[1])[0]
        e, ec = both[:6 * C], both[6 * C:]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = e.split(C)
        (c_shift_msa, c_scale_msa, c_gate_msa,
         c_shift_mlp, c_scale_mlp, c_gate_mlp) = ec.split(C)

        # --- norm1 / norm1_context + fused QKV into one [text; image] buffer --
        qkv = torch.empty((S, 3 * H * D), device=dev, dtype=dt)
        norm_txt = torch.empty((n_txt, C), device=dev, dtype=dt)
        for t in (qkv, norm_txt, txt, both):
            t.record_stream(side)
        sync.pre.record(main)
        sync.pre.wait(side)
        with torch.cuda.stream(side):
            rope = _rope_cast(rope_raw, dt)
            _ln_mod(txt, norm_txt, c_scale_msa, c_shift_msa,
                    self.norm1_context.norm.eps)
            torch.addmm(attn.add_kv_proj.bias, norm_txt,
                        attn.add_kv_proj.weight.t(), out=qkv[:n_txt])
            sync.mid.record(side)
        if rope is not None:
            for t in rope:
                t.record_stream(main)

        norm_img = torch.empty((n_img, C), device=dev, dtype=dt)
        _ln_mod(img, norm_img, scale_msa, shift_msa, self.norm1.norm.eps)
        torch.addmm(attn.to_qkv.bias, norm_img, attn.to_qkv.weight.t(),
                    out=qkv[n_txt:])
        sync.mid.wait(main)

        # --- joint attention --------------------------------------------------
        q = torch.empty((S, H, D), device=dev, dtype=dt)
        k = torch.empty((S, H, D), device=dev, dtype=dt)
        _qk_norm_rope(qkv, q, k, rope,
                      attn.norm_q.weight, attn.norm_k.weight,
                      attn.norm_added_q.weight, attn.norm_added_k.weight,
                      n_txt, H, D, attn.norm_q.eps)
        v = qkv[:, 2 * H * D:].view(1, S, H, D)
        out = _sdpa(q.view(1, S, H, D).transpose(1, 2),
                    k.view(1, S, H, D).transpose(1, 2),
                    v.transpose(1, 2), D ** -0.5)
        attn_flat = _attn_rows(out, H, D)

        # --- text chain (side stream) -----------------------------------------
        cf0, cf2 = self.ff_context.net[0].proj, self.ff_context.net[2]
        res_txt = torch.empty((n_txt, C), device=dev, dtype=dt)
        o_txt = torch.empty((n_txt, C), device=dev, dtype=dt)
        for t in (res_txt, o_txt, attn_flat):
            t.record_stream(side)
        sync.pre.record(main)
        sync.pre.wait(side)
        with torch.cuda.stream(side):
            a_txt = F.linear(attn_flat[:n_txt], attn.to_add_out.weight,
                             attn.to_add_out.bias)
            _gate_res_ln_mod(txt, a_txt, res_txt, norm_txt, c_gate_msa,
                             c_scale_mlp, c_shift_mlp, self.norm2_context.eps)
            h_txt = F.linear(norm_txt, cf0.weight, cf0.bias)
            _gelu_into(h_txt, h_txt)
            f_txt = F.linear(h_txt, cf2.weight, cf2.bias)
            _gate_res(res_txt, f_txt, o_txt, c_gate_mlp)
            sync.post.record(side)

        # --- image chain (main stream) ----------------------------------------
        ff0, ff2 = self.ff.net[0].proj, self.ff.net[2]
        a_img = F.linear(attn_flat[n_txt:], attn.to_out[0].weight,
                         attn.to_out[0].bias)
        res_img = torch.empty((n_img, C), device=dev, dtype=dt)
        _gate_res_ln_mod(img, a_img, res_img, norm_img, gate_msa,
                         scale_mlp, shift_mlp, self.norm2.eps)
        h_img = F.linear(norm_img, ff0.weight, ff0.bias)
        _gelu_into(h_img, h_img)
        f_img = F.linear(h_img, ff2.weight, ff2.bias)
        o_img = torch.empty((n_img, C), device=dev, dtype=dt)
        _gate_res(res_img, f_img, o_img, gate_mlp)
        sync.post.wait(main)

        return o_txt.view(1, n_txt, C), o_img.view(1, n_img, C)


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream DiT block: text+image concatenated, self-attention + MLP in parallel."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, promote_fp32=False)
        self.proj_mlp = ReplicatedLinear(dim, self.mlp_hidden_dim, bias=True,
                                         quant_config=quant_config)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + self.mlp_hidden_dim, dim, bias=True,
                                         quant_config=quant_config)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )

        self.dim = dim
        self.heads = self.attn.heads
        self.head_dim = self.attn.head_dim
        self._quantized = quant_config is not None
        self._sync = None
        self._po_key = None
        self._po = None

    def _proj_out_split(self):
        """``proj_out`` weight split into its attention and MLP column blocks.

        ``proj_out`` consumes ``cat([attn_out, gelu(proj_mlp(x))], -1)``, so as
        one GEMM it cannot start until both branches are done. Split along K, the
        MLP term (4/5 of the work) runs on the side stream while attention is
        still in flight and only the small attention term stays on the critical
        path. Rebuilt whenever the parameter is replaced or mutated in place.
        """
        w = self.proj_out.weight
        key = (w.data_ptr(), w._version, w.dtype, tuple(w.shape))
        if self._po_key != key:
            c = self.dim
            self._po = (w[:, :c].contiguous(), w[:, c:].contiguous())
            self._po_key = key
        return self._po

    def _forward_ref(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb=None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ):
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (self.norm.linear.bias is None or self.proj_out.bias is None
                or not _fast_ok(self, hidden_states, encoder_hidden_states, temb,
                                joint_attention_kwargs)):
            return self._forward_ref(hidden_states, encoder_hidden_states, temb,
                                     image_rotary_emb, joint_attention_kwargs)
        try:
            rope_raw = _rope_check(image_rotary_emb, self.head_dim)
        except _Unsupported:
            return self._forward_ref(hidden_states, encoder_hidden_states, temb,
                                     image_rotary_emb, joint_attention_kwargs)

        C = self.dim
        H, D = self.heads, self.head_dim
        n_img = hidden_states.shape[1]
        n_txt = encoder_hidden_states.shape[1]
        S = n_img + n_txt
        dt = hidden_states.dtype
        dev = hidden_states.device
        attn = self.attn
        img = hidden_states.view(n_img, C)
        txt = encoder_hidden_states.view(n_txt, C)

        main = torch.cuda.current_stream(dev)
        side = _side_stream(dev)
        sync = self._sync
        if sync is None:
            sync = self._sync = _Sync()
        w_attn, w_mlp = self._proj_out_split()

        # The rope down-cast is independent of everything else; get it out of the
        # way on the side stream.
        sync.pre.record(main)
        sync.pre.wait(side)
        with torch.cuda.stream(side):
            rope = _rope_cast(rope_raw, dt)
            sync.mid.record(side)
        if rope is not None:
            for t in rope:
                t.record_stream(main)

        e = F.linear(F.silu(temb), self.norm.linear.weight,
                     self.norm.linear.bias)[0]
        shift, scale, gate = e.split(C)

        normed = torch.empty((S, C), device=dev, dtype=dt)
        eps = self.norm.norm.eps
        _ln_mod(txt, normed[:n_txt], scale, shift, eps)
        _ln_mod(img, normed[n_txt:], scale, shift, eps)

        # MLP branch + its share of proj_out on the side stream.
        normed.record_stream(side)
        sync.pre.record(main)
        sync.pre.wait(side)
        with torch.cuda.stream(side):
            mlp = F.linear(normed, self.proj_mlp.weight, self.proj_mlp.bias)
            _gelu_into(mlp, mlp)
            p2 = F.linear(mlp, w_mlp, self.proj_out.bias)
            sync.post.record(side)
        p2.record_stream(main)

        # Attention branch + its share of proj_out on the main stream.
        qkv = F.linear(normed, attn.to_qkv.weight, attn.to_qkv.bias)
        q = torch.empty((S, H, D), device=dev, dtype=dt)
        k = torch.empty((S, H, D), device=dev, dtype=dt)
        sync.mid.wait(main)
        _qk_norm_rope(qkv, q, k, rope, attn.norm_q.weight, attn.norm_k.weight,
                      None, None, 0, H, D, attn.norm_q.eps)
        v = qkv[:, 2 * H * D:].view(1, S, H, D)
        out = _sdpa(q.view(1, S, H, D).transpose(1, 2),
                    k.view(1, S, H, D).transpose(1, 2),
                    v.transpose(1, 2), D ** -0.5)
        p1 = F.linear(_attn_rows(out, H, D), w_attn)
        sync.post.wait(main)

        o_txt = torch.empty((n_txt, C), device=dev, dtype=dt)
        o_img = torch.empty((n_img, C), device=dev, dtype=dt)
        _gate_res2(txt, p1[:n_txt], p2[:n_txt], o_txt, gate)
        _gate_res2(img, p1[n_txt:], p2[n_txt:], o_img, gate)

        return o_txt.view(1, n_txt, C), o_img.view(1, n_img, C)
