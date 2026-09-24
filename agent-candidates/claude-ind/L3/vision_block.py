"""Vision transformer block for Qwen VL models (fused Triton implementation).

Unified across Qwen2-VL and Qwen3-VL:
  - act_fn: Qwen2 uses QuickGELU (default), Qwen3 uses SiLU / GELU.
  - norm_eps: configurable LayerNorm epsilon.

Uses LayerNorm (not RMSNorm) with pre-norm residual connections,
encoder-only attention, and vision MLP.

Why this is not just the reference composition
----------------------------------------------
The reference block (``LayerNorm`` -> ``VisionAttention`` -> residual ->
``LayerNorm`` -> ``VisionMLP`` -> residual) spends more time in memory-bound
glue than in its four GEMMs.  Measured on B200 at the hot capture shape
(N=20680 tokens, d=1152, 16 heads x 72, mlp 4304), per block call::

    flash-attention                  336 us      qkv permute+contiguous  173 us
    fc1 + fc2 GEMM                   305 us      aten activation         168 us
    2x aten vectorized_layer_norm    196 us      flash_attn apply_rotary 101 us
    qkv GEMM                         185 us      proj GEMM                60 us
                                                 2x residual add          57 us

The GEMMs are already at ~91% of this device's clock-locked bf16 peak (1559
TFLOPS on the qkv GEMM against a ~1717 TFLOPS ceiling at 1500 MHz), so they are
left to cuBLAS.  Everything else -- 44% of the block -- is traffic that does not
need to exist, and is replaced here by three Triton kernels:

``_layer_norm_kernel``
    bf16 LayerNorm with the fp32 reduction aten also uses, optionally fusing the
    *incoming* residual add and writing the updated residual out.  That removes a
    whole extra full-tensor add per block.  36.9 us vs 102.4 us, and 49.2 us vs
    134.0 us for the add+norm pair.

``_rope_kernel``
    One pass over the fused qkv GEMM output that rotates q and k and writes them
    as ``[M, 2, heads, dim]``.  The reference instead materializes a permuted
    contiguous ``(2, seq, heads, dim)`` copy and *then* runs ``flash_attn``'s
    ``apply_rotary`` over it -- two full passes plus a 95 MB temporary where one
    pass over the same bytes does.  v is never materialized at all: it is handed
    to flash-attention as a stride-3456 view of the qkv buffer, which costs the
    attention kernel nothing (measured identical to a contiguous v).
    48 us vs 272 us, against a 40 us pure-copy floor for the same bytes.

    The kernel reads each head's 72 values as one contiguous 16B-aligned run and
    splits it into the two rotary halves with ``tl.split``, which needs those
    halves *interleaved* rather than blocked.  ``_interleaved_qkv`` bakes that
    permutation into the q/k rows of the qkv weight once, at the first forward.
    A rotary permutation applied identically to q and k leaves ``q @ k^T``
    unchanged, so attention is bit-identical; v is left alone, so the attention
    output stays in the basis ``proj`` expects.  Doing it the blocked way instead
    costs 54 us -- the second half then starts 72 B into the head and vectorizes
    only 8 B at a time.

``_act_kernel``
    The MLP activation in one vectorized pass.  For exact (erf) GELU, Triton's
    ``tl.erf`` lowers to a libdevice call that costs more than the memory does,
    so this rewrites it as ``relu(x) - 0.5|x|*erfc(|x|/sqrt 2)`` with erfc folded
    into a single ``exp2`` (max error 7.4e-5, ~50x under a bf16 ulp here).  83 us
    vs 169 us, against a 69 us copy floor.

Attention itself stays FlashAttention, but is given a padded QK head dim -- see
:func:`_pad_qk_head_dim`.
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
# LayerNorm (+ optional fused residual add)
# ---------------------------------------------------------------------------
# One program per BLOCK_M rows.  BLOCK_N is next_pow2(d); the tail lanes are
# masked so they move no memory (they only widen an fp32 reduction, which is
# free at these bandwidths).  The reduction is fp32 to match aten's bf16
# LayerNorm, which promotes its accumulator.


@triton.jit
def _layer_norm_kernel(
    X, RES, RB, XOUT, Y, W, B,
    M, N: tl.constexpr, eps,
    HAS_RES: tl.constexpr,
    HAS_RB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    mask = (rows < M)[:, None] & cmask[None, :]
    offs = rows[:, None] * N + cols[None, :]

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        x = x + tl.load(RES + offs, mask=mask, other=0.0).to(tl.float32)
        xo = x
        if HAS_RB:
            # A row-broadcast bias folded into the residual only -- never into
            # the normalized branch.  Lets the consumer of the residual finish
            # with one ``addmm_`` instead of a GEMM-with-bias plus a full-tensor
            # add (see _fast_forward).
            xo = x + tl.load(RB + cols, mask=cmask, other=0.0).to(tl.float32)
        tl.store(XOUT + offs, xo.to(XOUT.dtype.element_ty), mask=mask)

    mean = tl.sum(x, axis=1) / N
    xc = tl.where(mask, x - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=1) / N + eps)

    w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def _layer_norm(x, weight, bias, eps, residual=None, res_bias=None):
    """LayerNorm over the last dim of a 2-D ``x``.

    With ``residual``, computes ``x + residual`` first and returns
    ``(that_sum [+ res_bias], normed)``; otherwise returns ``normed``.
    """
    M, N = x.shape
    y = torch.empty_like(x)
    has_res = residual is not None
    xout = torch.empty_like(x) if has_res else x
    BLOCK_M = 2
    _layer_norm_kernel[(triton.cdiv(M, BLOCK_M),)](
        x, residual, res_bias, xout, y, weight, bias,
        M, N, eps,
        HAS_RES=has_res,
        HAS_RB=has_res and res_bias is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=triton.next_power_of_2(N),
        num_warps=2, num_stages=2,
    )
    return (xout, y) if has_res else y


# ---------------------------------------------------------------------------
# Rotary embedding + q/k extraction out of a fused qkv buffer
# ---------------------------------------------------------------------------
# ``QKV`` is ``[M, 3*D]`` with ``D = heads*head_dim``.  Each head's head_dim
# values are laid out as rotary *pairs* (see ``_interleave_perm``), so one
# contiguous aligned load per head splits into the two rotary halves:
#     out[2j]   = x[2j] * cos[j] - x[2j+1] * sin[j]
#     out[2j+1] = x[2j] * sin[j] + x[2j+1] * cos[j]
# The output head dim ``HDO`` may exceed ``HD`` (see _pad_qk_head_dim); lanes
# past HD read as 0 against cos=1 / sin=0, so they fall out of the rotation as
# the zero padding that needs.


@triton.jit
def _rope_kernel(
    QKV, COS, SIN, QK,
    M, D: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr, RH: tl.constexpr,
    HDO: tl.constexpr,
    BLOCK_M: tl.constexpr, NHP: tl.constexpr, HDP: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    hh = tl.arange(0, NHP)
    kk = tl.arange(0, HDP)
    jj = tl.arange(0, HDP // 2)
    rm = rows[:, None, None] < M
    hm = (hh < NH)[None, :, None]
    smask = rm & hm & (kk < HD)[None, None, :]
    dmask = rm & hm & (kk < HDO)[None, None, :]
    soff = (rows[:, None, None] * (3 * D) + hh[None, :, None] * HD
            + kk[None, None, :])
    doff = (rows[:, None, None] * (2 * NH * HDO) + hh[None, :, None] * HDO
            + kk[None, None, :])

    q = tl.load(QKV + soff, mask=smask, other=0.0)
    k = tl.load(QKV + soff + D, mask=smask, other=0.0)
    if HAS_ROPE:
        jmask = rm & (jj < RH)[None, None, :]
        cof = rows[:, None, None] * RH + jj[None, None, :]
        cs = tl.load(COS + cof, mask=jmask, other=1.0).to(tl.float32)
        sn = tl.load(SIN + cof, mask=jmask, other=0.0).to(tl.float32)
        q0, q1 = tl.split(tl.reshape(q.to(tl.float32),
                                     [BLOCK_M, NHP, HDP // 2, 2]))
        q = tl.reshape(tl.join(q0 * cs - q1 * sn, q0 * sn + q1 * cs),
                       [BLOCK_M, NHP, HDP]).to(QK.dtype.element_ty)
        k0, k1 = tl.split(tl.reshape(k.to(tl.float32),
                                     [BLOCK_M, NHP, HDP // 2, 2]))
        k = tl.reshape(tl.join(k0 * cs - k1 * sn, k0 * sn + k1 * cs),
                       [BLOCK_M, NHP, HDP]).to(QK.dtype.element_ty)
    tl.store(QK + doff, q, mask=dmask)
    tl.store(QK + doff + NH * HDO, k, mask=dmask)


def _pad_qk_head_dim(head_dim: int) -> int:
    """QK head dim to hand flash-attention.

    FA4 on SM100 has no good tile for head_dim 72: it runs the captured shapes
    ~25% *slower* there than at 96, even though 96 is a third more work (348 us
    vs 273 us at N=20680, and the same ordering at every captured shape).
    Zero-padding q and k leaves ``q @ k^T`` bit-identical, and V keeps its own
    head dim so the attention output -- and so ``proj`` -- is untouched.
    Rounding up to a multiple of 32 is the general form of that.
    """
    return -(-head_dim // 32) * 32


def _rope_split_qk(qkv, cos, sin, num_heads, head_dim, hdo):
    """Rotate q/k in the fused qkv buffer into a fresh ``[M, 2, heads, hdo]``.

    Returns ``(q, k)`` as strided views of that one buffer.
    """
    M = qkv.shape[0]
    has_rope = cos is not None and sin is not None
    qk = torch.empty((M, 2, num_heads, hdo), device=qkv.device, dtype=qkv.dtype)
    _rope_kernel[(M,)](
        qkv, cos, sin, qk,
        M, num_heads * head_dim, num_heads, head_dim,
        cos.shape[-1] if has_rope else 0, hdo,
        BLOCK_M=1,
        NHP=triton.next_power_of_2(num_heads),
        HDP=triton.next_power_of_2(max(head_dim, hdo)),
        HAS_ROPE=has_rope,
        num_warps=4, num_stages=2,
    )
    return qk[:, 0], qk[:, 1]


def _interleave_perm(num_heads: int, head_dim: int, rot_half: int, device):
    """Column permutation taking a blocked rotary layout to a pairwise one.

    Blocked (what the weights ship in): ``[x_0 .. x_{r-1} | y_0 .. y_{r-1}]`` per
    head, where the rotary mixes ``x_j`` with ``y_j``.  Pairwise (what
    ``_rope_kernel`` wants): ``[x_0 y_0 x_1 y_1 ...]``.  Columns at or past
    ``2*rot_half`` are not rotated and stay put.
    """
    idx = torch.arange(num_heads * head_dim, device=device)
    h, r = idx // head_dim, idx % head_dim
    j, hi = r // 2, (r % 2) * rot_half
    return torch.where(r < 2 * rot_half, h * head_dim + hi + j, idx)


# ---------------------------------------------------------------------------
# MLP activation
# ---------------------------------------------------------------------------
_ACT_GELU_ERF = 0
_ACT_GELU_TANH = 1
_ACT_SILU = 2
_ACT_QUICKGELU = 3


@triton.jit
def _act_kernel(X, Y, n_elements, KIND: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    if KIND == 0:
        # Exact (erf) GELU, written as relu minus a positive tail correction:
        #     gelu(x) = relu(x) - |x|/2 * erfc(|x|/sqrt 2)
        # and erfc folded into a single exp2 -- erfc(a/sqrt2) = exp(-(a^2/2 +
        # Q(a))) with Q a degree-6 minimax fit of -ln(erfc(a/sqrt2)) - a^2/2 on
        # [0, 9].  Max error 7.4e-5 in gelu, ~50x under a bf16 ulp here, and
        # ~5x tighter than the Abramowitz & Stegun rational form -- which also
        # needs a second transcendental (a reciprocal) and measures 101 us
        # against this one's 83 us.  |x| is clamped at 8 only to keep the
        # polynomial from overflowing the exponent; the correction there is
        # already 5e-15.
        ax = tl.minimum(tl.abs(x), 8.0)
        q = (0.00110293737603 + ax * (0.790821467351 + ax * (
             -0.170956255095 + ax * (0.0300606851554 + ax * (
             -0.00350515360236 + ax * (0.00023231369602
             + ax * -6.53248136463e-06))))))
        y = tl.maximum(x, 0.0) - (0.5 * ax) * tl.exp2(
            -0.7213475204444817 * ax * ax - 1.4426950408889634 * q)
    elif KIND == 1:
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        y = x / (1.0 + tl.exp2(-2.885390081777927 * inner))
    elif KIND == 2:
        y = x / (1.0 + tl.exp2(-1.4426950408889634 * x))
    else:
        y = x / (1.0 + tl.exp2(-2.4554629595604 * x))
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def _activation(x, kind):
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    _act_kernel[(triton.cdiv(n, BLOCK),)](
        x, y, n, KIND=kind, BLOCK=BLOCK, num_warps=4, num_stages=4)
    return y


def _act_kind(act_fn) -> int | None:
    """Classify the block's activation module into a Triton kernel variant."""
    name = type(act_fn).__name__
    if name == "QuickGELU":
        return _ACT_QUICKGELU
    if name == "SiLU":
        return _ACT_SILU
    if name == "GELU":
        return (_ACT_GELU_TANH
                if getattr(act_fn, "approximate", "none") == "tanh"
                else _ACT_GELU_ERF)
    return None


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
        # encoder pass runs 54 of them. PyTorch's bf16 layer_norm already
        # accumulates in fp32 internally, so the reduction precision is unchanged.
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)

        self.norm_eps = norm_eps
        self._act_kind = _act_kind(act_fn)
        # qkv weight/bias with the rotary halves interleaved, derived lazily on
        # the first forward (weight loading finishes before that) and re-derived
        # if the parameter is replaced, moved, cast or written to.
        self._il_key: tuple | None = None
        self._il_w: torch.Tensor | None = None
        self._il_b: torch.Tensor | None = None

    # -- interleaved qkv weight cache --------------------------------------
    def _interleaved_qkv(self, rot_half: int):
        qkv = self.attn.qkv
        w, b = qkv.weight, qkv.bias
        key = (w.data_ptr(), w._version, w.dtype, w.device, rot_half,
               None if b is None else (b.data_ptr(), b._version))
        if self._il_key != key:
            heads, hd = self.attn.num_heads, self.attn.head_dim
            d = heads * hd
            src = _interleave_perm(heads, hd, rot_half, w.device)
            # Only the q and k row blocks are rotated; v is left in place.
            rows = torch.cat([src, d + src,
                              torch.arange(2 * d, 3 * d, device=w.device)])
            self._il_w = w.detach()[rows].contiguous()
            self._il_b = None if b is None else b.detach()[rows].contiguous()
            self._il_key = key
        return self._il_w, self._il_b

    # -- fast path ---------------------------------------------------------
    def _fast_forward(self, x, cu_seqlens, cos, sin, max_seqlen):
        attn, mlp = self.attn, self.mlp
        seq_len, batch_size, d = x.shape
        m = seq_len * batch_size
        heads, hd = attn.num_heads, attn.head_dim
        xf = x.reshape(m, d)

        h = _layer_norm(xf, self.norm1.weight, self.norm1.bias, self.norm_eps)

        if cos is not None:
            w_qkv, b_qkv = self._interleaved_qkv(cos.shape[-1])
        else:
            w_qkv, b_qkv = attn.qkv.weight, attn.qkv.bias
        qkv = F.linear(h, w_qkv, b_qkv)
        q, k = _rope_split_qk(qkv, cos, sin, heads, hd, _pad_qk_head_dim(hd))
        v = qkv.view(m, 3, heads, hd)[:, 2]

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        o = attn.attn(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
            softmax_scale=hd ** -0.5, causal=False, num_splits=1,
        )
        a = F.linear(o.reshape(m, heads * hd), attn.proj.weight, attn.proj.bias)

        # fc2's bias rides along in the residual so the block can finish with a
        # single ``addmm_``: cuBLAS accumulates the fc2 GEMM straight into the
        # residual buffer (beta=1) instead of writing a separate output that a
        # third full-tensor pass then adds to.
        res, h2 = _layer_norm(a, self.norm2.weight, self.norm2.bias,
                              self.norm_eps, residual=xf,
                              res_bias=mlp.fc2.bias)

        u = F.linear(h2, mlp.fc1.weight, mlp.fc1.bias)
        g = _activation(u, self._act_kind)
        res.addmm_(g, mlp.fc2.weight.t())
        return res.view(seq_len, batch_size, d)

    def _can_fast(self, x, cos, sin) -> bool:
        attn = self.attn
        if self._act_kind is None or not x.is_cuda or x.dim() != 3:
            return False
        # The flat rotary indexing assumes one packed sequence dimension, i.e.
        # cos/sin row i belongs to token i.  With batch > 1 the reference
        # restarts positions per batch element, so fall back.
        if x.shape[1] != 1 or not x.is_contiguous():
            return False
        if attn.tp_size != 1 or self.mlp.fc2.tp_size != 1:
            return False
        # The fused qkv buffer is indexed as three equal [heads, head_dim]
        # blocks, which VisionAttention always builds (no GQA here).
        if attn.qkv.num_kv_heads != attn.qkv.num_heads:
            return False
        for mod in (self.norm1, self.norm2):
            if mod.weight is None or mod.bias is None:
                return False
        for lin in (attn.qkv, attn.proj, self.mlp.fc1, self.mlp.fc2):
            if lin.use_fp8:
                return False
        if cos is not None or sin is not None:
            # The rotary path wants both halves, contiguous, covering the head.
            if cos is None or sin is None:
                return False
            if not (cos.is_contiguous() and sin.is_contiguous()):
                return False
            if cos.dim() != 2 or cos.shape != sin.shape:
                return False
            if (2 * cos.shape[-1] != attn.head_dim
                    or cos.shape[0] < x.shape[0]):
                return False
        return True

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if self._can_fast(x, rotary_pos_emb_cos, rotary_pos_emb_sin):
            return self._fast_forward(x, cu_seqlens, rotary_pos_emb_cos,
                                      rotary_pos_emb_sin, max_seqlen)
        x = x + self.attn(
            self.norm1(x), cu_seqlens,
            rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x
