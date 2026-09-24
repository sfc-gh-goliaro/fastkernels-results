"""PairFormer stack for AlphaFold3 -- fused, graph-captured implementation.

48-block PairFormer: each block runs a PairBlock on pair (z) then
AttentionPairBias + SwiGLUTransition on single (s).

The captured workload is N_token=16, c_z=128, c_s=384 over 48 blocks: ~150M
parameters of weights driving a chain of ~5000 tiny dependent ops, so the eager
baseline is almost pure launch overhead.  This implementation keeps the
parameter tree (and hence the state_dict) identical to the baseline, but runs
the stack as ~19 fused Triton kernels per block replayed from a CUDA graph.

The interesting constraint is numerical.  The residual stack is chaotic: a
single 1-ulp perturbation of one input element moves ~59% of the output
elements outside the comparison tolerance by block 48.  The fused kernels
therefore reproduce the eager op-by-op arithmetic *bit for bit*, which the
kernels below go to some length for -- see ``_rb`` (rounding points the
compiler likes to fold away), the KC-chunked dots (cuBLAS's accumulation
order), ``_ln_stats`` (ATen's Welford layer norm), ``_exp_parts`` (CUDA's
expf) and ``_tree_sum16`` (ATen's warp-softmax reduction order).

Reference: openfold3/core/model/latent/pairformer.py PairFormerStack
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition


__targets__ = ["PairFormerStack"]

_EPS = 1e-5


class PairFormerBlock(nn.Module):
    """Single block of AF3 Algorithm 17 (parameter container + eager path)."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

        self.attn_pair_bias = AttentionPairBias(
            c_q=c_s, c_k=c_s, c_v=c_s,
            c_s=c_s, c_z=c_z,
            c_hidden=c_hidden_pair_bias,
            no_heads=no_heads_pair_bias,
            use_ada_layer_norm=False,
            gating=True,
            inf=inf,
        )

        self.single_transition = SwiGLUTransition(c_in=c_s, n=transition_n)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_trans_mask = single_mask if _mask_trans else None
        z = self.pair_stack(z=z, pair_mask=pair_mask)
        s = s + self.attn_pair_bias(a=s, z=z, s=None, mask=single_mask)
        s = s + self.single_transition(s, mask=single_trans_mask)
        return s, z


# ---------------------------------------------------------------------------
# Weight packing: per-block views of the parameter tree with the independent
# projections of each op concatenated along the output dim (one GEMM instead
# of four).  Concatenation is exact -- no folding of LayerNorm scales into the
# weights, so every bf16 rounding point of the baseline is preserved.
# ---------------------------------------------------------------------------
class _BlockWeights:
    __slots__ = ("tmo", "tmi", "tas", "tae", "pt", "ab", "st")


def _ln(mod):
    return (mod.weight.float(), mod.bias.float())


def _trimul_pack(m):
    w = {}
    w["ln_in"] = _ln(m.layer_norm_in)
    w["ln_out"] = _ln(m.layer_norm_out)
    w["cat"] = torch.cat([m.linear_a_p.weight, m.linear_a_g.weight,
                          m.linear_b_p.weight, m.linear_b_g.weight,
                          m.linear_g.weight], 0).contiguous()
    w["wz"] = m.linear_z.weight.contiguous()
    return w


def _triatt_pack(m):
    mha = m.mha
    w = {}
    w["ln"] = _ln(m.layer_norm)
    w["cat"] = torch.cat([mha.linear_q.weight, mha.linear_k.weight,
                          mha.linear_v.weight, mha.linear_g.weight,
                          m.linear_z.weight], 0).contiguous()
    w["wo"] = mha.linear_o.weight.contiguous()
    w["nh"] = m.no_heads
    w["ch"] = m.c_hidden
    return w


def _swiglu_pack(m):
    w = {}
    w["ln"] = _ln(m.layer_norm)
    w["cat"] = torch.cat([m.swiglu.linear_a.weight, m.swiglu.linear_b.weight], 0).contiguous()
    w["wo"] = m.linear_out.weight.contiguous()
    return w


def _apb_pack(m):
    mha = m.mha
    w = {}
    w["ln_z"] = _ln(m.layer_norm_z)
    w["wlz"] = m.linear_z.weight.contiguous()
    w["ln_a"] = _ln(m.layer_norm_a)
    w["cat"] = torch.cat([mha.linear_q.weight, mha.linear_k.weight,
                          mha.linear_v.weight, mha.linear_g.weight], 0).contiguous()
    nq = mha.linear_q.weight.shape[0]
    w["bq"] = torch.cat([mha.linear_q.bias,
                         mha.linear_q.bias.new_zeros(3 * nq)]).contiguous()
    w["wo"] = mha.linear_o.weight.contiguous()
    w["nh"] = mha.no_heads
    w["ch"] = mha.c_hidden
    return w


def _pack_block(blk) -> _BlockWeights:
    ps = blk.pair_stack
    w = _BlockWeights()
    w.tmo = _trimul_pack(ps.tri_mul_out)
    w.tmi = _trimul_pack(ps.tri_mul_in)
    w.tas = _triatt_pack(ps.tri_att_start)
    w.tae = _triatt_pack(ps.tri_att_end)
    w.pt = _swiglu_pack(ps.pair_transition)
    w.ab = _apb_pack(blk.attn_pair_bias)
    w.st = _swiglu_pack(blk.single_transition)
    return w


# ---------------------------------------------------------------------------
# Reference fused path in torch ops -- numerically mirrors the baseline
# (identical bf16 rounding points), used to validate the kernels and as the
# fallback for shapes/dtypes the kernels do not cover.
# ---------------------------------------------------------------------------
def _layer_norm(x, wb, dt):
    return F.layer_norm(x.float(), (x.shape[-1],), wb[0], wb[1], _EPS).to(dt)


def _trimul_ref(z, mask, w, outgoing: bool):
    dt = z.dtype
    c = w["wz"].shape[0]
    zl = _layer_norm(z, w["ln_in"], dt)
    proj = F.linear(zl, w["cat"])
    ap, ag, bp, bg, gg = proj.split([c, c, c, c, c], -1)
    m = mask.unsqueeze(-1)
    a = m * torch.sigmoid(ag) * ap
    b = m * torch.sigmoid(bg) * bp
    if outgoing:
        x = torch.einsum("ijc,kjc->ikc", a, b)
    else:
        x = torch.einsum("jic,jkc->ikc", a, b)
    x = _layer_norm(x, w["ln_out"], dt)
    x = F.linear(x, w["wz"])
    return x * torch.sigmoid(gg)


def _triatt_ref(z, mask, w, starting: bool, inf: float):
    dt = z.dtype
    if not starting:
        z = z.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
    n, _, cz = z.shape
    nh, ch = w["nh"], w["ch"]
    x = _layer_norm(z, w["ln"], dt)
    proj = F.linear(x, w["cat"])
    q, k, v, g, lz = proj.split([nh * ch, nh * ch, nh * ch, nh * ch, nh], -1)
    q = (q.view(n, n, nh, ch).transpose(-2, -3) / math.sqrt(ch))
    k = k.view(n, n, nh, ch).transpose(-2, -3)
    v = v.view(n, n, nh, ch).transpose(-2, -3)
    scores = torch.einsum("ihqc,ihkc->ihqk", q, k)
    scores = scores + (inf * (mask - 1))[:, None, None, :]
    scores = scores + lz.permute(2, 0, 1).unsqueeze(-4)
    p = F.softmax(scores, dim=-1)
    o = torch.einsum("ihqk,ihkc->ihqc", p, v).transpose(-2, -3)
    o = o * torch.sigmoid(g).view(n, n, nh, ch)
    out = F.linear(o.reshape(n, n, nh * ch), w["wo"])
    if not starting:
        out = out.transpose(-2, -3)
    return out


def _swiglu_ref(x, mask, w):
    dt = x.dtype
    xl = _layer_norm(x, w["ln"], dt)
    h = F.linear(xl, w["cat"])
    a, b = h.chunk(2, -1)
    out = F.linear(F.silu(a) * b, w["wo"])
    return out * mask.unsqueeze(-1)


def _apb_ref(s, z, single_mask, w, inf: float):
    dt = s.dtype
    n, cs = s.shape
    nh, ch = w["nh"], w["ch"]
    zb = F.linear(_layer_norm(z, w["ln_z"], dt), w["wlz"]).permute(2, 0, 1)
    al = _layer_norm(s, w["ln_a"], dt)
    proj = F.linear(al, w["cat"], w["bq"])
    q, k, v, g = proj.split([nh * ch] * 4, -1)
    q = (q.view(n, nh, ch).transpose(0, 1) / math.sqrt(ch))
    k = k.view(n, nh, ch).transpose(0, 1)
    v = v.view(n, nh, ch).transpose(0, 1)
    scores = torch.einsum("hqc,hkc->hqk", q, k)
    scores = scores + (inf * (single_mask - 1))[None, None, :]
    scores = scores + zb
    p = F.softmax(scores, dim=-1)
    o = torch.einsum("hqk,hkc->hqc", p, v).transpose(0, 1)
    o = o * torch.sigmoid(g).view(n, nh, ch)
    return F.linear(o.reshape(n, nh * ch), w["wo"])


# ---------------------------------------------------------------------------
# Fused kernels (see the module docstring for the bit-exactness rules).
# ---------------------------------------------------------------------------


@triton.jit
def _rb(x):
    """Round fp32 to bf16 precision, keeping the value in fp32."""
    return (x.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(
        tl.float32, bitcast=True)


# ---- transcendentals, bit-compatible with CUDA's expf -------------------
# tl.exp lowers to a bare ex2.approx and is up to 2 ulp off from ATen's expf;
# these mirror the exact instruction sequence nvcc emits for expf(), and the
# way ATen spells sigmoid (rcp of a fused 1+exp) and silu (x / (1+exp(-x))).
@triton.jit
def _sat01(x):
    return tl.inline_asm_elementwise("cvt.sat.f32.f32 $0, $1;", "=f,f", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _ex2(x):
    return tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise("rcp.rn.f32 $0, $1;", "=f,f", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _fma_rm(a, b, c):
    return tl.inline_asm_elementwise("fma.rm.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                     [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _exp_parts(x):
    """exp(x) == ex2.approx(y) * 2**q, with expf's argument reduction."""
    t = _sat01(tl.math.fma(x, 0.005724980030208826, 0.5))
    q = _fma_rm(t, 252.0, 12582913.0)
    y = tl.math.fma(x, 1.4426950216293335, -(q + (-12583039.0)))
    y = tl.math.fma(x, 1.925963033500011e-08, y)
    return _ex2(y), (q.to(tl.int32, bitcast=True) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _expf(x):
    r, scale = _exp_parts(x)
    return r * scale


@triton.jit
def _sig(x):
    r, scale = _exp_parts(-x)
    return _rcp(tl.math.fma(r, scale, 1.0))


@triton.jit
def _silu(x):
    r, scale = _exp_parts(-x)
    return tl.math.div_rn(x, tl.math.fma(r, scale, 1.0))


# ---- layer norm statistics (bit-compatible with ATen) ---------------------
@triton.jit
def _wsum(val, mean, s2, RECIP: tl.constexpr):
    # exactly the instruction sequence nvcc emits for ATen's cuWelfordOnlineSum:
    # the two multiply-adds are fused, so spell them out (letting the compiler
    # choose makes the result depend on the tile layout).
    delta = val - mean
    new_mean = tl.math.fma(delta, RECIP, mean)
    return new_mean, tl.math.fma(delta, val - new_mean, s2)


@triton.jit
def _wcomb(mean_b, s2_b, mean_a, s2_a, NA: tl.constexpr, NB: tl.constexpr,
           CNTA: tl.constexpr):
    # ATen's cuWelfordCombine, in nvcc's fusion: mean = fma(m_b, nB, nA*m_a) and
    # sigma2 = fma(delta*delta*count_a, nB, s2_a + s2_b).
    delta = mean_b - mean_a
    return (tl.math.fma(mean_a, NA, NB * mean_b),
            tl.math.fma(delta * delta * CNTA, NB, s2_a + s2_b))


@triton.jit
def _wstep(mean, s2, HALF: tl.constexpr, CN: tl.constexpr, CNT: tl.constexpr):
    mb, ma = tl.split(tl.permute(tl.reshape(mean, (2, HALF, CN)), (1, 2, 0)))
    sb, sa = tl.split(tl.permute(tl.reshape(s2, (2, HALF, CN)), (1, 2, 0)))
    return _wcomb(mb, sb, ma, sa, 0.5, 0.5, CNT)


@triton.jit
def _group_stats(X, R, offn, nmask, W: tl.constexpr, CN: tl.constexpr):
    """Welford stats of channels [W*128, W*128+128) for each column of offn."""
    base = W * 128 + tl.arange(0, 32) * 4
    m = tl.zeros((32, CN), dtype=tl.float32)
    v = tl.zeros((32, CN), dtype=tl.float32)
    for ii in tl.static_range(4):
        x = tl.load(X + (base + ii)[:, None] * R + offn[None, :],
                    mask=nmask[None, :], other=0.0).to(tl.float32)
        if ii == 0:
            m, v = _wsum(x, m, v, 1.0)
        elif ii == 1:
            m, v = _wsum(x, m, v, 0.5)
        elif ii == 2:
            m, v = _wsum(x, m, v, 1.0 / 3.0)
        else:
            m, v = _wsum(x, m, v, 0.25)
    m, v = _wstep(m, v, 16, CN, 4.0)
    m, v = _wstep(m, v, 8, CN, 8.0)
    m, v = _wstep(m, v, 4, CN, 16.0)
    m, v = _wstep(m, v, 2, CN, 32.0)
    m, v = _wstep(m, v, 1, CN, 64.0)
    return tl.reshape(m, (CN,)), tl.reshape(v, (CN,))


@triton.jit
def _ln_stats(X, R, offn, nmask, C: tl.constexpr, CN: tl.constexpr):
    m0, v0 = _group_stats(X, R, offn, nmask, 0, CN)
    if C > 256:
        m1, v1 = _group_stats(X, R, offn, nmask, 1, CN)
        m2, v2 = _group_stats(X, R, offn, nmask, 2, CN)
        ma, va = _wcomb(m0, v0, m2, v2, 0.5, 0.5, 128.0)
        # nA / nB as the fp32 values ATen computes (128/384 and 256/384) -- python
        # doubles would round elsewhere.
        mean, s2 = _wcomb(ma, va, m1, v1, 0.3333333432674408, 0.6666666865348816, 128.0)
    elif C > 128:
        m1, v1 = _group_stats(X, R, offn, nmask, 1, CN)
        mean, s2 = _wcomb(m0, v0, m1, v1, 0.5, 0.5, 128.0)
    else:
        mean, s2 = m0, v0
    return mean, tl.rsqrt(tl.math.div_rn(s2, C * 1.0) + 1e-5)


@triton.jit
def _pair_add(x, NQ: tl.constexpr, HALF: tl.constexpr):
    a, b = tl.split(tl.permute(tl.reshape(x, (NQ, 2, HALF)), (0, 2, 1)))
    return a + b


@triton.jit
def _tree_sum16(x, NQ: tl.constexpr):
    """Sum 16 elements in the order ATen's warp softmax uses: a butterfly
    (i, i+8), (i, i+4), (i, i+2), (i, i+1) -- not the order tl.sum picks."""
    y = _pair_add(x, NQ, 8)
    y = _pair_add(y, NQ, 4)
    y = _pair_add(y, NQ, 2)
    y = _pair_add(y, NQ, 1)
    return tl.reshape(y, (NQ,))


@triton.jit
def _ln_chunk(X, LNW, LNB, R, offn, nmask, mu, rstd, kidx):
    x = tl.load(X + kidx[:, None] * R + offn[None, :], mask=nmask[None, :],
                other=0.0).to(tl.float32)
    lw = tl.load(LNW + kidx)
    lb = tl.load(LNB + kidx)
    return tl.math.fma(rstd[None, :] * (x - mu[None, :]), lw[:, None],
                       tl.broadcast_to(lb[:, None], x.shape)).to(tl.bfloat16)


# --------------------------------------------------------------------------
# layout conversion
# --------------------------------------------------------------------------
@triton.jit
def k_to_cm(SRC, DST, C: tl.constexpr, CP: tl.constexpr, R: tl.constexpr,
            BR: tl.constexpr):
    """[R, C] row-major -> [C, R] channel-major."""
    offr = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = offr < R
    offc = tl.arange(0, CP)
    cm = offc < C
    x = tl.load(SRC + offr[:, None] * C + offc[None, :],
                mask=rm[:, None] & cm[None, :], other=0.0)
    tl.store(DST + offc[None, :] * R + offr[:, None], x, mask=rm[:, None] & cm[None, :])


@triton.jit
def k_from_cm(SRC, DST, C: tl.constexpr, CP: tl.constexpr, R: tl.constexpr,
              BR: tl.constexpr):
    """[C, R] channel-major -> [R, C] row-major."""
    offr = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = offr < R
    offc = tl.arange(0, CP)
    cm = offc < C
    x = tl.load(SRC + offc[None, :] * R + offr[:, None],
                mask=rm[:, None] & cm[None, :], other=0.0)
    tl.store(DST + offr[:, None] * C + offc[None, :], x, mask=rm[:, None] & cm[None, :])


# --------------------------------------------------------------------------
# triangle multiplicative update
# --------------------------------------------------------------------------
@triton.jit
def k_trimul_pre(Z, PM, W, LNW, LNB, A, B, G,
                 C: tl.constexpr, R: tl.constexpr,
                 CM: tl.constexpr, CN: tl.constexpr, KC: tl.constexpr):
    """pid0: 0 -> A = m*sig(a_g)*a_p, 1 -> B = m*sig(b_g)*b_p, 2 -> G = sig(g)."""
    which = tl.program_id(0)
    offm = tl.program_id(1) * CM + tl.arange(0, CM)
    offn = tl.program_id(2) * CN + tl.arange(0, CN)
    nmask = offn < R
    base = tl.where(which == 0, 0, tl.where(which == 1, 2 * C, 4 * C))
    wp = W + (base + offm)[:, None] * C
    wg = W + (base + C + offm)[:, None] * C
    mu, rstd = _ln_stats(Z, R, offn, nmask, C, CN)
    accp = tl.zeros((CM, CN), dtype=tl.float32)
    accg = tl.zeros((CM, CN), dtype=tl.float32)
    for kk in range(C // KC):
        kidx = kk * KC + tl.arange(0, KC)
        zl = _ln_chunk(Z, LNW, LNB, R, offn, nmask, mu, rstd, kidx)
        accp = tl.dot(tl.load(wp + kidx[None, :]), zl, accp)
        if which != 2:
            accg = tl.dot(tl.load(wg + kidx[None, :]), zl, accg)
    if which == 2:
        tl.store(G + offm[:, None] * R + offn[None, :], _sig(_rb(accp)).to(tl.bfloat16),
                 mask=nmask[None, :])
    else:
        p = _rb(accp)
        m = tl.load(PM + offn, mask=nmask, other=0.0).to(tl.float32)
        t = _rb(m[None, :] * _rb(_sig(_rb(accg))))
        out = (t * p).to(tl.bfloat16)
        if which == 0:
            tl.store(A + offm[:, None] * R + offn[None, :], out, mask=nmask[None, :])
        else:
            tl.store(B + offm[:, None] * R + offn[None, :], out, mask=nmask[None, :])


@triton.jit
def k_trimul_mid(A, B, X, N: tl.constexpr, R: tl.constexpr,
                 CC: tl.constexpr, OUTGOING: tl.constexpr):
    """x[c] = a[c] @ b[c]^T  (outgoing) or a[c]^T @ b[c]  (incoming).

    One 16x16x16 dot per channel: torch reaches this einsum through a batched
    matmul, and a tensor-core dot reproduces its accumulation exactly while a
    hand-rolled fp32 FMA chain over j does not.  Both operands are loaded in
    the orientation the dot wants, so no register transpose is needed.
    """
    c0 = tl.program_id(0) * CC
    r = tl.arange(0, N)
    for cc in tl.static_range(CC):
        base = (c0 + cc) * R
        if OUTGOING:
            a = tl.load(A + base + r[:, None] * N + r[None, :])
            b = tl.load(B + base + r[None, :] * N + r[:, None])
        else:
            a = tl.load(A + base + r[None, :] * N + r[:, None])
            b = tl.load(B + base + r[:, None] * N + r[None, :])
        tl.store(X + base + r[:, None] * N + r[None, :], tl.dot(a, b).to(tl.bfloat16))


# --------------------------------------------------------------------------
# attention projections (triangle attention and attention-pair-bias)
# --------------------------------------------------------------------------
@triton.jit
def k_qkvg(X, W, BQ, LNW, LNB, Q, K, V, GA, LZ,
           N: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
           NH: tl.constexpr, CQ: tl.constexpr, SCALE: tl.constexpr,
           CM: tl.constexpr, CN: tl.constexpr, KC: tl.constexpr,
           TRANS: tl.constexpr, HASBQ: tl.constexpr, HASLZ: tl.constexpr):
    """pid0 in {q, k, v, gate}; the narrow linear_z head rides along with q."""
    which = tl.program_id(0)
    offm = tl.program_id(1) * CM + tl.arange(0, CM)
    offn = tl.program_id(2) * CN + tl.arange(0, CN)
    nmask = offn < R
    offh = tl.arange(0, 16)
    dolz = HASLZ and (which == 0) and (tl.program_id(1) == 0)
    wp = W + (which * CQ + offm)[:, None] * C
    wl = W + (4 * CQ + offh)[:, None] * C
    if TRANS:
        rd = (offn % N) * N + (offn // N)
    else:
        rd = offn
    mu, rstd = _ln_stats(X, R, rd, nmask, C, CN)
    acc = tl.zeros((CM, CN), dtype=tl.float32)
    accl = tl.zeros((16, CN), dtype=tl.float32)
    for kk in range(C // KC):
        kidx = kk * KC + tl.arange(0, KC)
        zl = _ln_chunk(X, LNW, LNB, R, rd, nmask, mu, rstd, kidx)
        acc = tl.dot(tl.load(wp + kidx[None, :]), zl, acc)
        if dolz:
            accl = tl.dot(tl.load(wl + kidx[None, :], mask=(offh < NH)[:, None],
                                  other=0.0), zl, accl)
    if HASBQ:
        acc += tl.load(BQ + which * CQ + offm).to(tl.float32)[:, None]
    ob = _rb(acc)
    om = nmask[None, :]
    if which == 0:
        tl.store(Q + offm[:, None] * R + offn[None, :], (ob * SCALE).to(tl.bfloat16), mask=om)
    elif which == 1:
        tl.store(K + offm[:, None] * R + offn[None, :], ob.to(tl.bfloat16), mask=om)
    elif which == 2:
        tl.store(V + offm[:, None] * R + offn[None, :], ob.to(tl.bfloat16), mask=om)
    else:
        tl.store(GA + offm[:, None] * R + offn[None, :], _sig(ob).to(tl.bfloat16), mask=om)
    if dolz:
        tl.store(LZ + offh[:, None] * R + offn[None, :], accl.to(tl.bfloat16),
                 mask=(offh < NH)[:, None] & om)


@triton.jit
def k_att(Q, K, V, GA, BIAS, MASK, O,
          N: tl.constexpr, R: tl.constexpr, RB: tl.constexpr,
          CH: tl.constexpr, CHP: tl.constexpr, INF: tl.constexpr,
          TRANS: tl.constexpr, PAIRMASK: tl.constexpr):
    """One program per (i-group, head): softmax(q.k + mask_bias + bias) @ v."""
    i = tl.program_id(0)
    h = tl.program_id(1)
    offd = h * CH + tl.arange(0, CHP)
    dm = tl.arange(0, CHP) < CH
    offj = tl.arange(0, N)
    cols = i * N + offj
    q = tl.load(Q + offd[:, None] * R + cols[None, :], mask=dm[:, None], other=0.0)
    k = tl.load(K + offd[:, None] * R + cols[None, :], mask=dm[:, None], other=0.0)
    s = _rb(tl.dot(tl.trans(q), k))
    if PAIRMASK:
        if TRANS:
            mi = offj * N + i
        else:
            mi = i * N + offj
        m = tl.load(MASK + mi).to(tl.float32)
    else:
        m = tl.load(MASK + offj).to(tl.float32)
    mb = _rb(_rb(m - 1.0) * INF)
    s = _rb(s + mb[None, :])
    bz = tl.load(BIAS + h * RB + offj[:, None] * N + offj[None, :]).to(tl.float32)
    s = _rb(s + bz)
    e = _expf(s - tl.max(s, 1)[:, None])
    p = tl.math.div_rn(e, _tree_sum16(e, N)[:, None]).to(tl.bfloat16)
    v = tl.load(V + offd[:, None] * R + cols[None, :], mask=dm[:, None], other=0.0)
    o = _rb(tl.dot(v, tl.trans(p)))
    g = tl.load(GA + offd[:, None] * R + cols[None, :], mask=dm[:, None], other=0.0)
    tl.store(O + offd[:, None] * R + cols[None, :], (o * g.to(tl.float32)).to(tl.bfloat16),
             mask=dm[:, None])


# --------------------------------------------------------------------------
# output projection + residual; optionally layer-norms its input (closing a
# triangle multiplicative update) and/or applies a mask or gate
# --------------------------------------------------------------------------
@triton.jit
def k_proj_add(X, W, Z, MASK, GATE, LNW, LNB,
               N: tl.constexpr, R: tl.constexpr, KD: tl.constexpr, KC: tl.constexpr,
               CM: tl.constexpr, CN: tl.constexpr, TRANS: tl.constexpr,
               USEMASK: tl.constexpr, USEGATE: tl.constexpr, USELN: tl.constexpr):
    """z[:, r] += bf16(bf16(W @ [LN](X)) [* mask] [* gate])."""
    offm = tl.program_id(0) * CM + tl.arange(0, CM)
    offn = tl.program_id(1) * CN + tl.arange(0, CN)
    nmask = offn < R
    wp = W + offm[:, None] * KD
    if USELN:
        mu, rstd = _ln_stats(X, R, offn, nmask, KD, CN)
    acc = tl.zeros((CM, CN), dtype=tl.float32)
    for kk in range(KD // KC):
        kidx = kk * KC + tl.arange(0, KC)
        if USELN:
            x = _ln_chunk(X, LNW, LNB, R, offn, nmask, mu, rstd, kidx)
        else:
            x = tl.load(X + kidx[:, None] * R + offn[None, :], mask=nmask[None, :],
                        other=0.0)
        acc = tl.dot(tl.load(wp + kidx[None, :]), x, acc)
    upd = _rb(acc)
    if USEMASK:
        upd = _rb(upd * tl.load(MASK + offn, mask=nmask, other=0.0).to(tl.float32)[None, :])
    if USEGATE:
        upd = _rb(upd * tl.load(GATE + offm[:, None] * R + offn[None, :],
                                mask=nmask[None, :], other=0.0).to(tl.float32))
    if TRANS:
        rd = (offn % N) * N + (offn // N)
    else:
        rd = offn
    zp = Z + offm[:, None] * R + rd[None, :]
    z = tl.load(zp, mask=nmask[None, :], other=0.0).to(tl.float32)
    tl.store(zp, (z + upd).to(tl.bfloat16), mask=nmask[None, :])


@triton.jit
def k_zbias(Z, LNZW, LNZB, WLZ, ZB, C: tl.constexpr, R: tl.constexpr,
            NHB: tl.constexpr, CN: tl.constexpr, KC: tl.constexpr):
    """The pair bias linear_z(layer_norm_z(z)) that AttentionPairBias consumes."""
    offn = tl.program_id(0) * CN + tl.arange(0, CN)
    nmask = offn < R
    offh = tl.arange(0, 16)
    wl = WLZ + offh[:, None] * C
    mu, rstd = _ln_stats(Z, R, offn, nmask, C, CN)
    acc = tl.zeros((16, CN), dtype=tl.float32)
    for kk in range(C // KC):
        kidx = kk * KC + tl.arange(0, KC)
        zl = _ln_chunk(Z, LNZW, LNZB, R, offn, nmask, mu, rstd, kidx)
        acc = tl.dot(tl.load(wl + kidx[None, :], mask=(offh < NHB)[:, None], other=0.0),
                     zl, acc)
    tl.store(ZB + offh[:, None] * R + offn[None, :], acc.to(tl.bfloat16),
             mask=(offh < NHB)[:, None] & nmask[None, :])


# --------------------------------------------------------------------------
# SwiGLU transition: LN -> silu(a) * b
# --------------------------------------------------------------------------
@triton.jit
def k_swiglu_pre(X, W, LNW, LNB, H,
                 C: tl.constexpr, R: tl.constexpr, HID: tl.constexpr,
                 CM: tl.constexpr, CN: tl.constexpr, KC: tl.constexpr):
    offm = tl.program_id(0) * CM + tl.arange(0, CM)
    offn = tl.program_id(1) * CN + tl.arange(0, CN)
    nmask = offn < R
    wa = W + offm[:, None] * C
    wb = W + (HID + offm)[:, None] * C
    mu, rstd = _ln_stats(X, R, offn, nmask, C, CN)
    acca = tl.zeros((CM, CN), dtype=tl.float32)
    accb = tl.zeros((CM, CN), dtype=tl.float32)
    for kk in range(C // KC):
        kidx = kk * KC + tl.arange(0, KC)
        xl = _ln_chunk(X, LNW, LNB, R, offn, nmask, mu, rstd, kidx)
        acca = tl.dot(tl.load(wa + kidx[None, :]), xl, acca)
        accb = tl.dot(tl.load(wb + kidx[None, :]), xl, accb)
    a = _rb(acca)
    b = _rb(accb)
    sa = _rb(_silu(a))
    tl.store(H + offm[:, None] * R + offn[None, :], (sa * b).to(tl.bfloat16),
             mask=nmask[None, :])


# ---------------------------------------------------------------------------
# Launch sequence: one persistent set of channel-major buffers, 19 kernels
# per block, all captured into a single CUDA graph.
# ---------------------------------------------------------------------------
class Runner:
    """Buffer set + launch sequence for the fused stack (channel-major).

    The tile constants below (32-wide output/column tiles, 64-wide K chunks,
    4 warps) are not free parameters: they are the shape for which every kernel
    was verified bit-identical to the eager baseline.  Layer-norm tiles in
    particular must be >= 32 columns wide, and wider output tiles change the
    generated dot -- so a change here needs re-verification, not just a
    re-benchmark.
    """

    def __init__(self, weights, N, CZ, CS, dev="cuda"):
        self.w = weights
        self.N, self.CZ, self.CS = N, CZ, CS
        R = N * N
        self.R = R
        e = lambda *sh: torch.empty(*sh, device=dev, dtype=torch.bfloat16)
        self.Z = e(CZ, R)
        self.A, self.B, self.G, self.X = (e(CZ, R) for _ in range(4))
        self.Q, self.K, self.V, self.GA, self.O = (e(CZ, R) for _ in range(5))
        self.LZ = e(16, R)
        w0 = weights[0]
        self.HID_PT = w0.pt["wo"].shape[1]
        self.HID_ST = w0.st["wo"].shape[1]
        self.H = e(self.HID_PT, R)
        self.ZB = e(16, R)
        self.S = e(CS, N)
        self.SQ, self.SK, self.SV, self.SG, self.SO = (e(CS, N) for _ in range(5))
        self.SH = e(self.HID_ST, N)
        self.zout = e(R, CZ)
        self.sout = e(N, CS)
        self.NH, self.CH = w0.tas["nh"], w0.tas["ch"]
        self.NHB, self.CHB = w0.ab["nh"], w0.ab["ch"]
        self.CSP = 1 << (CS - 1).bit_length()
        self.kw = dict(num_warps=4)

    def _trimul(self, w, outgoing, PM):
        N, C, R, kw = self.N, self.CZ, self.R, self.kw
        k_trimul_pre[(3, C // 32, R // 32)](
            self.Z, PM, w["cat"], w["ln_in"][0], w["ln_in"][1],
            self.A, self.B, self.G, C, R, 32, 32, min(64, C), **kw)
        k_trimul_mid[(C // 2,)](
            self.A, self.B, self.X, N, R, 2, outgoing, **kw)
        k_proj_add[(C // 32, R // 32)](
            self.X, w["wz"], self.Z, PM, self.G, w["ln_out"][0], w["ln_out"][1],
            N, R, C, min(64, C), 32, 32, False, False, True, True, **kw)

    def _triatt(self, w, trans, PM):
        N, C, R, kw = self.N, self.CZ, self.R, self.kw
        k_qkvg[(4, (self.NH * self.CH) // 32, R // 32)](
            self.Z, w["cat"], w["cat"], w["ln"][0], w["ln"][1],
            self.Q, self.K, self.V, self.GA, self.LZ,
            N, C, R, self.NH, self.NH * self.CH, 1.0 / math.sqrt(self.CH),
            32, 32, min(64, C), trans, False, True, **kw)
        k_att[(N, self.NH)](
            self.Q, self.K, self.V, self.GA, self.LZ, PM, self.O,
            N, R, R, self.CH, 1 << (self.CH - 1).bit_length(), 1e9, trans, True,
            num_warps=4)
        k_proj_add[(C // 32, R // 32)](
            self.O, w["wo"], self.Z, PM, self.G, PM, PM, N, R, C, min(64, C),
            32, 32, trans, False, False, False, **kw)

    def _pt(self, w, wab, PM):
        N, C, R, kw = self.N, self.CZ, self.R, self.kw
        HID = self.HID_PT
        k_swiglu_pre[(HID // 32, R // 32)](
            self.Z, w["cat"], w["ln"][0], w["ln"][1], self.H,
            C, R, HID, 32, 32, min(64, C), **kw)
        k_proj_add[(C // 32, R // 32)](
            self.H, w["wo"], self.Z, PM, self.G, PM, PM, N, R, HID, min(64, HID),
            32, 32, False, True, False, False, **kw)
        k_zbias[(R // 32,)](
            self.Z, wab["ln_z"][0], wab["ln_z"][1], wab["wlz"], self.ZB,
            C, R, self.NHB, 32, min(64, C), **kw)

    def _apb(self, w, SM):
        N, CS, R, kw = self.N, self.CS, self.R, self.kw
        k_qkvg[(4, CS // 32, 1)](
            self.S, w["cat"], w["bq"], w["ln_a"][0], w["ln_a"][1],
            self.SQ, self.SK, self.SV, self.SG, self.LZ,
            N, CS, N, self.NHB, CS, 1.0 / math.sqrt(self.CHB),
            32, 32, min(64, CS), False, True, False, **kw)
        k_att[(1, self.NHB)](
            self.SQ, self.SK, self.SV, self.SG, self.ZB, SM, self.SO,
            N, N, R, self.CHB, 1 << (self.CHB - 1).bit_length(), 1e9, False, False,
            num_warps=4)
        k_proj_add[(CS // 32, 1)](
            self.SO, w["wo"], self.S, SM, self.SG, SM, SM, N, N, CS, min(64, CS),
            32, N, False, False, False, False, **kw)

    def _st(self, w, SM):
        N, CS, kw = self.N, self.CS, self.kw
        HID = self.HID_ST
        k_swiglu_pre[(HID // 32, 1)](
            self.S, w["cat"], w["ln"][0], w["ln"][1], self.SH,
            CS, N, HID, 32, 32, min(64, CS), **kw)
        k_proj_add[(CS // 32, 1)](
            self.SH, w["wo"], self.S, SM, self.SG, SM, SM, N, N, HID, min(64, HID),
            32, N, False, True, False, False, **kw)

    def run(self, s, z, sm, pm, nblocks=None):
        """s: [N, CS] bf16 row-major, z: [R, CZ] row-major, pm: [R], sm: [N]."""
        N, C, R = self.N, self.CZ, self.R
        k_to_cm[(R // 64,)](z, self.Z, C, C, R, 64, num_warps=4)
        k_to_cm[(1,)](s, self.S, self.CS, self.CSP, N, N, num_warps=4)
        ws = self.w if nblocks is None else self.w[:nblocks]
        for w in ws:
            self._trimul(w.tmo, True, pm)
            self._trimul(w.tmi, False, pm)
            self._triatt(w.tas, False, pm)
            self._triatt(w.tae, True, pm)
            self._pt(w.pt, w.ab, pm)
            self._apb(w.ab, sm)
            self._st(w.st, sm)
        k_from_cm[(R // 64,)](self.Z, self.zout, C, C, R, 64, num_warps=4)
        k_from_cm[(1,)](self.S, self.sout, self.CS, self.CSP, N, N, num_warps=4)
        return self.sout, self.zout


class PairFormerStack(nn.Module):
    """AF3 Algorithm 17: PairFormer stack."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.inf = inf
        self.blocks = nn.ModuleList([
            PairFormerBlock(
                c_s=c_s, c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])
        self._packed = None
        self._graph = None
        self._fast = False
        self._runner = None
        self._shapes = None

    # -- packed weights (built lazily, after load_state_dict) ---------------
    def _weights(self):
        if self._packed is None:
            with torch.no_grad():
                self._packed = [_pack_block(b) for b in self.blocks]
        return self._packed

    def _fused(self, s4, z4, single_mask4, pair_mask4):
        inf = self.inf
        s, z = s4[0], z4[0]
        single_mask, pair_mask = single_mask4[0], pair_mask4[0]
        for w in self._weights():
            z = z + _trimul_ref(z, pair_mask, w.tmo, True)
            z = z + _trimul_ref(z, pair_mask, w.tmi, False)
            z = z + _triatt_ref(z, pair_mask, w.tas, True, inf)
            z = z + _triatt_ref(z, pair_mask, w.tae, False, inf)
            z = z + _swiglu_ref(z, pair_mask, w.pt)
            s = s + _apb_ref(s, z, single_mask, w.ab, inf)
            s = s + _swiglu_ref(s, single_mask, w.st)
        return s.unsqueeze(0), z.unsqueeze(0)

    # -- fast path eligibility --------------------------------------------
    def _kernels_ok(self, s, z) -> bool:
        """Whether the fused kernels cover this geometry.

        The layer-norm reduction mirrors ATen's vectorized kernel (4 channels
        per lane, 128 per group) and the softmax reduction its 16-lane warp
        tree, so the channel counts and token count are constrained; anything
        else falls back to the (also graph-captured) op-by-op path.
        """
        n = z.shape[1]
        cz, cs = z.shape[-1], s.shape[-1]
        blk = self.blocks[0]
        ps = blk.pair_stack
        hid_pt = ps.pair_transition.linear_out.weight.shape[1]
        hid_st = blk.single_transition.linear_out.weight.shape[1]
        ta, ab = ps.tri_att_start, blk.attn_pair_bias.mha
        return (n == 16 and z.shape[2] == n
                and cz in (128, 256, 384) and cs in (128, 256, 384)
                and hid_pt % 32 == 0 and hid_st % 32 == 0
                and ta.no_heads * ta.c_hidden == cz
                and ab.no_heads * ab.c_hidden == cs
                and ta.no_heads <= 16 and ab.no_heads <= 16
                and ps.tri_mul_out.c_hidden == cz
                and self.inf == 1e9)

    # -- CUDA graph -------------------------------------------------------
    # The stack is a chain of thousands of tiny dependent ops; on the CPU side
    # it is pure launch overhead.  Capture the whole chain once and replay it,
    # copying the (freshly allocated every call) inputs into the captured
    # buffers first.
    def _build_graph(self, s, z, single_mask, pair_mask):
        ws = self._weights()
        n, cz, cs = z.shape[1], z.shape[-1], s.shape[-1]
        self._fast = self._kernels_ok(s, z)
        if self._fast:
            self._runner = Runner(ws, n, cz, cs)
            # clone (not contiguous()) -- these are the buffers the graph reads
            # from, so they must be ours, not views of a caller tensor that the
            # allocator may hand to something else later.
            self._sin = s[0].clone()
            self._zin = z[0].reshape(n * n, cz).clone()
            self._smin = single_mask[0].clone()
            self._pmin = pair_mask[0].reshape(-1).clone()
            body = lambda: self._runner.run(self._sin, self._zin, self._smin, self._pmin)
        else:
            self._sin, self._zin = s.clone(), z.clone()
            self._smin, self._pmin = single_mask.clone(), pair_mask.clone()
            body = lambda: self._fused(self._sin, self._zin, self._smin, self._pmin)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            so, zo = body()
        self._sout = so.reshape(1, n, cs) if self._fast else so
        self._zout = zo.reshape(1, n, n, cz) if self._fast else zo
        self._shapes = (tuple(s.shape), tuple(z.shape))
        self._graph = g

    def _replay(self, s, z, single_mask, pair_mask):
        if self._graph is None:
            self._build_graph(s, z, single_mask, pair_mask)
        if self._fast:
            n, cz = z.shape[1], z.shape[-1]
            self._sin.copy_(s[0])
            self._zin.copy_(z[0].reshape(n * n, cz))
            self._smin.copy_(single_mask[0])
            self._pmin.copy_(pair_mask.reshape(-1))
        else:
            self._sin.copy_(s)
            self._zin.copy_(z)
            self._smin.copy_(single_mask)
            self._pmin.copy_(pair_mask)
        self._graph.replay()
        return self._sout, self._zout

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        # The graph bakes in shapes and addresses, so replay only for the
        # geometry it was captured with; anything else runs eagerly.
        if (s.dim() == 3 and s.shape[0] == 1 and z.dim() == 4 and s.is_cuda
                and s.dtype == torch.bfloat16 and _mask_trans
                and not torch.is_grad_enabled()
                and (self._graph is None
                     or self._shapes == (tuple(s.shape), tuple(z.shape)))):
            return self._replay(s, z, single_mask, pair_mask)

        for block in self.blocks:
            s, z = block(s=s, z=z, single_mask=single_mask, pair_mask=pair_mask)
        return s, z
