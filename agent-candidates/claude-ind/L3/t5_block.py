"""T5 encoder block: self-attention + FFN with pre-norm residuals (L3).

Fused Triton implementation.  The module tree, parameter names and shapes are
the baseline's; only ``forward`` changes -- the eager op-by-op path is replaced
by four hand-written Triton kernels:

``_pre_kernel``
    T5 RMSNorm for the attention input, *plus* the position-bias preparation,
    in one launch.  The two jobs are independent, so the kernel simply
    concatenates their grids (first ``NROW`` programs normalize a row, the rest
    prepare bias) and they overlap for free.  Bias preparation is either a
    relayout of the captured head-minor bias or, when none is supplied, direct
    construction of the T5 relative-position bias from the bucket formula --
    replacing an arange/log/embedding-gather chain.
``_attn_kernel``
    Flash-style attention reading Q/K/V straight out of the packed ``qkv_proj``
    output, never materializing the ``[1, 64, 512, 512]`` score matrix.
``_add_rms_kernel``
    Attention residual add fused with the FFN RMSNorm.
``_geglu_kernel``
    gate/up split + NewGELU + multiply in one pass.
``_add_kernel``
    FFN residual add.

Every kernel reproduces the baseline's *rounding sequence*, not just its maths.
That matters more than usual here: T5 omits the 1/sqrt(d) attention scale, so
with the benchmark's random weights the pre-softmax scores have a standard
deviation of ~13 and the softmax is nearly one-hot.  A single-ulp difference in
a bf16 score then moves that key's weight by ~10%, so anything upstream of the
scores (the norm, the QK product) has to agree with the baseline bit for bit,
and the softmax has to round its *normalized* weights to bf16 exactly where the
baseline does.

The dense projections stay on ``F.linear``, exactly as in the baseline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config

import triton
import triton.language as tl

from ..L1.t5_layer_norm import T5LayerNorm
from ..L2.t5_attention import T5SelfAttention
from ..L2.t5_dense import T5DenseActDense, T5DenseGatedActDense


__targets__ = ["T5Block"]

# Bias-prep mode for _pre_kernel's second grid segment.  Triton needs the value
# it compares MODE against to be a tl.constexpr global.
_PRE_NONE = 0      # nothing to prepare (RMSNorm only)
_PRE_RELAY = 1     # relay a captured head-minor bias into head-major order
_PRE_BUILD = 2     # build the T5 relative-position bias from scratch
_RELAY = tl.constexpr(_PRE_RELAY)


@triton.jit
def _rms_row(X, W, Y, row, eps, H: tl.constexpr):
    """out = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), variance in fp32.

    Mirrors ``T5LayerNorm`` exactly, including both intermediate bf16 rounds.
    """
    cols = tl.arange(0, H)
    x = tl.load(X + row * H + cols).to(tl.float32)
    var = tl.sum(x * x, 0) / H
    t = (x * tl.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    tl.store(Y + row * H + cols, (w * t).to(tl.bfloat16))


@triton.jit
def _pre_kernel(X, W, Y, PB, PBT, WEMB, eps, LOGDEN,
                S: tl.constexpr, NH: tl.constexpr, H: tl.constexpr,
                NROW: tl.constexpr, BR: tl.constexpr, MODE: tl.constexpr,
                HALF: tl.constexpr, MAXEXACT: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NROW:
        _rms_row(X, W, Y, pid, eps, H)
    else:
        blk = pid - NROW
        h = tl.arange(0, NH)
        if MODE == _RELAY:
            # [S*S, NH] -> [NH, S*S]: a plain 2D relayout, coalesced both ways.
            r = blk * BR + tl.arange(0, BR)
            tile = tl.load(PB + r[:, None] * NH + h[None, :])
            tl.store(PBT + h[:, None] * (S * S) + r[None, :], tl.trans(tile))
        else:
            # T5 relative-position bucket bias, straight into head-major order.
            i = blk // (S // BR)
            j = (blk % (S // BR)) * BR + tl.arange(0, BR)
            rel = j - i
            arel = tl.abs(rel)
            relf = tl.maximum(arel, 1).to(tl.float32) / MAXEXACT
            big = MAXEXACT + (tl.log(relf) / LOGDEN * (HALF - MAXEXACT)).to(tl.int32)
            bucket = (tl.where(rel > 0, HALF, 0)
                      + tl.where(arel < MAXEXACT, arel, tl.minimum(big, HALF - 1)))
            vals = tl.load(WEMB + bucket[:, None] * NH + h[None, :])
            tl.store(PBT + h[:, None] * (S * S) + i * S + j[None, :], tl.trans(vals))


_PRE_BR = 64
_PRE_WARPS = 8   # the RMSNorm reduction tree must match torch's mean(-1) bitwise


def _prologue(x2d, weight, eps, pb, wemb, seq, nh, num_buckets, max_distance):
    """RMSNorm(x) and the head-major position bias, in one launch.

    Returns ``(normed, pbt, position_bias)``.  ``pbt`` is ``[NH, S, S]``
    contiguous; for the built bias ``position_bias`` is a view of it (only the
    *values* are part of the contract -- the baseline's own return is a
    non-contiguous permute of a dense buffer).
    """
    rows, h = x2d.shape
    normed = torch.empty_like(x2d)
    dev, dt = x2d.device, x2d.dtype
    half = num_buckets // 2

    if pb is None:
        mode, nblk = _PRE_BUILD, seq * seq // _PRE_BR
        pbt = torch.empty((nh, seq, seq), device=dev, dtype=dt)
        position_bias = pbt.unsqueeze(0)
        src = pbt
    elif (pb.stride(1) == 1 and pb.stride(3) == nh
          and pb.stride(2) == seq * nh and pb.shape[3] == seq):
        # Captured layout: permute(2, 0, 1) of a dense [S, S, NH] buffer.  Read
        # in that order the attention kernel would waste 15/16 of every memory
        # sector, so relay it into head-major once.
        mode, nblk = _PRE_RELAY, seq * seq // _PRE_BR
        pbt = torch.empty((nh, seq, seq), device=dev, dtype=dt)
        position_bias, src = pb, pb
    else:
        mode, nblk = _PRE_NONE, 0
        pbt = pb[0].contiguous()
        position_bias, src = pb, pb

    _pre_kernel[(rows + nblk,)](
        x2d, weight, normed, src, pbt, wemb, eps,
        math.log(max_distance / (half // 2)),
        S=seq, NH=nh, H=h, NROW=rows, BR=_PRE_BR, MODE=mode,
        HALF=half, MAXEXACT=half // 2, num_warps=_PRE_WARPS,
    )
    return normed, pbt, position_bias


@triton.jit
def _add_rms_kernel(X, R, W, Y, HOUT, eps, H: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    x = tl.load(X + row * H + cols).to(tl.float32)
    r = tl.load(R + row * H + cols).to(tl.float32)
    h = (x + r).to(tl.bfloat16)
    tl.store(HOUT + row * H + cols, h)
    xf = h.to(tl.float32)
    var = tl.sum(xf * xf, 0) / H
    t = (xf * tl.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    tl.store(Y + row * H + cols, (w * t).to(tl.bfloat16))


def _add_rms(x2d, res2d, weight, eps):
    rows, h = x2d.shape
    y = torch.empty_like(x2d)
    hout = torch.empty_like(x2d)
    _add_rms_kernel[(rows,)](x2d, res2d, weight, y, hout, eps, H=h, num_warps=8)
    return hout, y


@triton.jit
def _attn_kernel(QKV, PBT, OUT, HAS_BIAS: tl.constexpr, S: tl.constexpr,
                 NH: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
    """Fused T5 attention: one program per (query block, head).

    Two passes over the key axis rather than one streaming pass.  The first
    finds the row max and the exponential sum; the second rebuilds the scores
    and forms the *normalized* bf16 weights, which is what the baseline feeds
    its PV matmul.  A streaming softmax would instead round unnormalized
    probabilities, and with these near-one-hot distributions that difference is
    visible in the block output.  Keeping only a [BM, BN] tile live also stays
    clear of the register ceiling a single full-row pass runs into.

    Scores are rounded to bf16 before *and* after the bias add, matching
    ``matmul`` (bf16 out) followed by ``scores += position_bias`` (bf16 add).
    """
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    ss = 3 * NH * D
    offm = pid_m * BM + tl.arange(0, BM)
    offd = tl.arange(0, D)
    q = tl.load(QKV + h * D + offm[:, None] * ss + offd[None, :])
    kbase = QKV + NH * D + h * D
    vbase = QKV + 2 * NH * D + h * D
    pbase = PBT + h * S * S

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    for n0 in tl.range(0, S, BN):
        offn = n0 + tl.arange(0, BN)
        s = tl.dot(q, tl.load(kbase + offd[:, None] + offn[None, :] * ss)).to(tl.bfloat16)
        if HAS_BIAS:
            s = s + tl.load(pbase + offm[:, None] * S + offn[None, :])
        s = s.to(tl.float32)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        l_i = (l_i * tl.exp2((m_i - m_new) * 1.4426950408889634)
               + tl.sum(tl.exp2((s - m_new[:, None]) * 1.4426950408889634), 1))
        m_i = m_new

    inv = 1.0 / l_i
    acc = tl.zeros([BM, D], tl.float32)
    for n0 in tl.range(0, S, BN):
        offn = n0 + tl.arange(0, BN)
        s = tl.dot(q, tl.load(kbase + offd[:, None] + offn[None, :] * ss)).to(tl.bfloat16)
        if HAS_BIAS:
            s = s + tl.load(pbase + offm[:, None] * S + offn[None, :])
        s = s.to(tl.float32)
        w = (tl.exp2((s - m_i[:, None]) * 1.4426950408889634) * inv[:, None]).to(tl.bfloat16)
        acc = tl.dot(w, tl.load(vbase + offn[:, None] * ss + offd[None, :]), acc)

    tl.store(OUT + offm[:, None] * (NH * D) + h * D + offd[None, :],
             acc.to(tl.bfloat16))


_ATTN_BM = 64
_ATTN_BN = 64
_ATTN_WARPS = 4
_ATTN_STAGES = 2


def _attention(qkv, pbt, seq, nh, d):
    out = torch.empty((seq, nh * d), device=qkv.device, dtype=qkv.dtype)
    _attn_kernel[(seq // _ATTN_BM, nh)](
        qkv, pbt if pbt is not None else qkv, out,
        HAS_BIAS=pbt is not None, S=seq, NH=nh, D=d,
        BM=_ATTN_BM, BN=_ATTN_BN,
        num_warps=_ATTN_WARPS, num_stages=_ATTN_STAGES,
    )
    return out


@triton.jit
def _tanh(x):
    """``tanh.approx.f32`` -- one MUFU op in place of a polynomial call.

    Its ~1e-5 relative error is far finer than bf16's 2^-9 spacing, so after the
    round back to bf16 the result was bit-identical to ``torch.tanh`` on every
    one of 21M test inputs, at under half the cost.
    """
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=r,r", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _geglu_kernel(GU, OUT, FF: tl.constexpr, BLOCK: tl.constexpr):
    """act = NewGELU(gate) * up, keeping the eager path's bf16 intermediates.

    HuggingFace's ``NewGELUActivation`` runs on bf16 tensors, so every step
    rounds; folding the chain into fp32 would be *more* accurate but would not
    match, so the rounds are reproduced one for one.
    """
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + row * 2 * FF + cols).to(tl.float32)
    u = tl.load(GU + row * 2 * FF + FF + cols).to(tl.float32)

    c = (g * g).to(tl.bfloat16).to(tl.float32)
    c = (c * g).to(tl.bfloat16).to(tl.float32)
    c = (c * 0.044715).to(tl.bfloat16).to(tl.float32)
    c = (g + c).to(tl.bfloat16).to(tl.float32)
    c = (c * 0.7978845608028654).to(tl.bfloat16).to(tl.float32)
    c = _tanh(c).to(tl.bfloat16).to(tl.float32)
    c = (1.0 + c).to(tl.bfloat16).to(tl.float32)
    act = ((0.5 * g).to(tl.bfloat16).to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * FF + cols, (act * u).to(tl.bfloat16))


def _geglu(gate_up):
    rows, two_ff = gate_up.shape
    ff = two_ff // 2
    out = torch.empty((rows, ff), device=gate_up.device, dtype=gate_up.dtype)
    block = 1024 if ff % 1024 == 0 else 512
    _geglu_kernel[(rows, ff // block)](gate_up, out, FF=ff, BLOCK=block, num_warps=4)
    return out


@triton.jit
def _add_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + offs, mask=mask).to(tl.float32)
    b = tl.load(B + offs, mask=mask).to(tl.float32)
    tl.store(OUT + offs, (a + b).to(tl.bfloat16), mask=mask)


def _add(a, b):
    out = torch.empty_like(a)
    n = a.numel()
    _add_kernel[(triton.cdiv(n, 2048),)](a, b, out, n, BLOCK=2048, num_warps=4)
    return out


# ---------------------------------------------------------------------------
# Modules -- same tree, names and shapes as the baseline so the benchmark's
# ``load_state_dict`` weight sharing lines up.
# ---------------------------------------------------------------------------
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, mask=mask, position_bias=position_bias,
        )
        hidden_states = hidden_states + attn_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed)
        hidden_states = hidden_states + ff_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])
        self._fast_ok = (
            config.is_gated_act
            and config.dense_act_fn == "gelu_new"
            and config.d_kv * config.num_heads == config.d_model
            and config.relative_attention_num_buckets % 4 == 0
        )

    def _forward_fast(self, hidden_states, position_bias):
        attn_layer = self.layer[0]
        sa = attn_layer.SelfAttention
        ff_layer = self.layer[1]
        dense = ff_layer.DenseReluDense

        b, s, d_model = hidden_states.shape
        nh = sa.n_heads_per_partition
        x = hidden_states.reshape(s, d_model)

        if position_bias is None and not sa.has_relative_attention_bias:
            position_bias = torch.zeros((1, nh, s, s), device=x.device, dtype=x.dtype)

        wemb = (sa.relative_attention_bias.emb.weight
                if sa.has_relative_attention_bias else x)
        normed, pbt, position_bias = _prologue(
            x, attn_layer.layer_norm.weight, attn_layer.layer_norm.variance_epsilon,
            position_bias, wemb, s, nh,
            sa.relative_attention_num_buckets, sa.relative_attention_max_distance,
        )
        qkv = F.linear(normed, sa.qkv_proj.weight)
        attn = F.linear(_attention(qkv, pbt, s, nh, sa.d_kv), sa.o.weight)

        h1, normed2 = _add_rms(x, attn, ff_layer.layer_norm.weight,
                               ff_layer.layer_norm.variance_epsilon)
        ff = F.linear(_geglu(F.linear(normed2, dense.wi.weight)), dense.wo.weight)
        return _add(h1, ff).view(b, s, d_model), position_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (self._fast_ok and mask is None and hidden_states.dtype == torch.bfloat16
                and hidden_states.is_cuda and hidden_states.dim() == 3
                and hidden_states.shape[0] == 1 and hidden_states.is_contiguous()
                and hidden_states.shape[1] % _ATTN_BM == 0
                and hidden_states.shape[1] % _ATTN_BN == 0
                and hidden_states.shape[1] % _PRE_BR == 0
                and (position_bias is None or position_bias.dtype == torch.bfloat16)):
            return self._forward_fast(hidden_states, position_bias)

        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias
