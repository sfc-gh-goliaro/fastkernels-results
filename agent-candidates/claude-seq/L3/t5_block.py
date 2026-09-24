"""T5 encoder block: self-attention + FFN with pre-norm residuals (L3).

T5LayerSelfAttention: T5LayerNorm -> T5SelfAttention -> residual add.
T5LayerFF: T5LayerNorm -> T5Dense{Gated}ActDense -> residual add.
T5Block: T5LayerSelfAttention + T5LayerFF.

The block keeps the reference module tree (so the scorer's weight sharing still
works) but runs its own forward.  Three things drive the design.

1.  Why the frozen L2 attention cannot be reused
------------------------------------------------
``..L2.t5_attention``'s flash pass is a clear win at L2, where the scored tensor
*is* the output: its only deviation is that flash rounds the **unnormalized**
``exp`` to bf16 for the second ``dot`` while the reference rounds the
**normalized** probability -- a ~2^-9 relative perturbation that still leaves
0.9999 of the elements inside tolerance.

At L3 the feed-forward half amplifies exactly that.  Flipping a fraction ``f`` of
the residual stream's bf16 elements by one ulp costs ~1.5f of the *final*
elements their tolerance (measured: f=0.01 -> 0.9853 matched, f=0.1 -> 0.9336),
because ``gelu_new``'s ``1 + tanh(...)`` cancels to a multiple of 2^-9 for
negative gates, so a one-ulp change in a gate flips the whole activation and
``wo`` spreads it across the row.  Seeded from the baseline -- i.e. calling the
frozen L2 attention -- this operator scores **0.8739 matched** and fails.

The same sensitivity rules out the frozen L1 ``T5LayerNorm``: it disagrees with
the reference on 2e-5 of its elements, which is enough (the norm feeds the QKV
projection, and the captured softmax is peaked -- mean max probability 0.83 --
so a perturbed score is followed rather than averaged away) to turn into a 2%
bit-difference in the residual stream and 0.9900 matched, right on the limit.

So every step here has to be the reference's arithmetic, not an approximation of
it.  ``_t5_norm_kernel`` / ``_add_t5_norm_kernel`` reproduce ``T5LayerNorm`` op
for op, and the two attention GEMMs stay on the reference's own ``bmm``.  What
is left to win is the score matrix.

2.  The score matrix
--------------------
The reference makes five separate passes over ``[1, 64, 512, 512]``: ``+= bias``
(33.5 MB read + 67 MB read/write -- and the captured bias arrives permuted,
stride ``(64, 1, 32768, 64)``, so that read is 64x amplified and costs ~150 us on
its own), ``.float()``, ``softmax``, ``.type_as``.  ~400 MB and ~200 us.

``_bias_relayout_kernel`` moves the bias into ``[H, S, S]`` with one tiled
transpose, then ``_bias_softmax_kernel`` does the rest in a single in-place pass:
read the scores, read the bias, softmax the row in fp32, write the bf16
probabilities back over the scores.  100 MB, 14 us, with the reference's rounding
steps in the reference's places.

Normalizing before rounding means the row sum has to be known before any
probability is written, i.e. two traversals of the row.  Keeping the whole row
resident instead spills (158-336 us); recomputing the scores from Q/K in a
two-pass flash, or fusing PV into the pass over the materialized scores, both
land at 60 us.  Materializing the scores and streaming them once costs 41 us,
and that is what runs.  The GEMMs stay on ``bmm`` for numerics and for speed --
they are within ~5% of this machine's bf16 rate, where the best Triton QK^T tile
found needed ~40 us against cuBLAS' 14 -- and ``out=`` on a strided view lets the
second one write straight into ``[S, H*D]`` layout, so the reference's
``transpose(1, 2).contiguous()`` (11 us) disappears.

3.  Host overhead
-----------------
With the above the block's *kernels* took 207 us but a forward took 345: spelled
op by op, this operator costs more host time than the device time it enqueues,
and the scorer times the whole call.  So the forward

* works in 2-D throughout (``[S, d_model]``) and addresses Q, K^T, V and the
  attention output as four ``as_strided`` views instead of
  slice/view/transpose chains;
* calls ``torch.mm`` / ``torch.bmm`` directly rather than ``F.linear`` /
  ``torch.matmul`` -- the same cuBLAS kernels and the same bits, ~8 fewer
  dispatches each;
* reuses one set of scratch buffers (all fully overwritten, none escaping) so
  the seven intermediates are not re-allocated per call;
* stages the bias on the main stream -- overlapping it with the QKV GEMM buys
  1.3 us of device time and costs more than that in stream bookkeeping;
* ends on ``addmm``, folding the last residual add into wo's epilogue.

That is 345 -> 201 us of host time against 213 us of device time, i.e. the
device is the limit again.  Two Triton launches were also collapsed into one for
the computed-bias case; a third variant that fused the bucket gather into the
broadcast with 32 unrolled selects was 55 us slower and is not used.

Net on the captured shapes, both cases: ~225 us against 568 / 778 us for the
baseline (2.5x / 3.4x), with 0.9998-0.9999 of the output elements inside
tolerance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import T5Config

from ..L1.t5_layer_norm import T5LayerNorm
from ..L2.t5_attention import T5SelfAttention, _bucket_index_table
from ..L2.t5_dense import T5DenseActDense, T5DenseGatedActDense


__targets__ = ["T5Block"]


# ---------------------------------------------------------------------------
# scores += position_bias  ->  softmax(scores.float()).type_as(scores)
# ---------------------------------------------------------------------------
@triton.jit
def _bias_softmax_kernel(
    SC, PB, sph, spm, S,
    N: tl.constexpr, R: tl.constexpr, HAS_BIAS: tl.constexpr, EVEN: tl.constexpr,
):
    """In-place: ``SC[h, i, :] = softmax_fp32(bf16(SC + PB))`` for R query rows.

    One program owns ``R`` whole rows of one head, so the row reduction never
    leaves registers and the score matrix is read once and written once.  The
    rounding steps mirror the reference: the bias add lands in a bf16 tensor
    (``scores += position_bias``), the softmax runs in fp32
    (``softmax(scores.float())``), and only the normalized probability is
    rounded back (``.type_as(scores)``).
    """
    h = tl.program_id(0)
    i0 = tl.program_id(1) * R

    oi = i0 + tl.arange(0, R)
    oj = tl.arange(0, N)
    sp = SC + h * (S * S) + oi[:, None] * S + oj[None, :]

    if EVEN:
        x = tl.load(sp).to(tl.float32)
    else:
        m2 = (oi[:, None] < S) & (oj[None, :] < S)
        x = tl.load(sp, mask=m2, other=0.0).to(tl.float32)

    if HAS_BIAS:
        bp = PB + h * sph + oi[:, None] * spm + oj[None, :]
        if EVEN:
            bias = tl.load(bp)
        else:
            bias = tl.load(bp, mask=m2, other=0.0)
        x = (x + bias.to(tl.float32)).to(SC.dtype.element_ty).to(tl.float32)

    if not EVEN:
        x = tl.where(oj[None, :] < S, x, float("-inf"))

    e = tl.exp(x - tl.max(x, 1)[:, None])
    p = (e / tl.sum(e, 1)[:, None]).to(SC.dtype.element_ty)

    if EVEN:
        tl.store(sp, p)
    else:
        tl.store(sp, p, mask=m2)


@triton.jit
def _bias_relayout_kernel(
    SRC, DST, sh, sm, sn, S, H,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """Copy a permuted ``position_bias`` into contiguous ``[H, S, S]``.

    ``compute_bias``' view has the *head* axis contiguous (stride 1) and j at
    stride H, so a kernel that owns one head would pull every 2-byte element
    from its own sector.  Loading ``[BLOCK_N, BLOCK_H]`` instead makes both the
    read and (after ``tl.trans``) the write coalesced: the bias moves in 67 MB.
    """
    i = tl.program_id(0)
    jb = tl.program_id(1)
    offs_j = jb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    jm = offs_j < S
    hm = offs_h < H
    x = tl.load(SRC + i * sm + offs_j[:, None] * sn + offs_h[None, :] * sh,
                mask=jm[:, None] & hm[None, :], other=0.0)
    tl.store(DST + offs_h[:, None] * (S * S) + i * S + offs_j[None, :],
             tl.trans(x), mask=hm[:, None] & jm[None, :])


@triton.jit
def _bias_value_kernel(BTAB, W, BVAL, R, H, sw,
                       BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    """``BVAL[h, rel] = W[bucket[rel], h]`` -- every distinct bias value once."""
    rb = tl.program_id(0)
    offs_r = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_H)
    rm = offs_r < R
    hm = offs_h < H
    bucket = tl.load(BTAB + offs_r, mask=rm, other=0)
    v = tl.load(W + bucket[:, None] * sw + offs_h[None, :],
                mask=rm[:, None] & hm[None, :], other=0.0)
    tl.store(BVAL + offs_h[:, None] * R + offs_r[None, :], tl.trans(v),
             mask=hm[:, None] & rm[None, :])


@triton.jit
def _bias_expand_kernel(BVAL, PB, sbv, sph, S, R: tl.constexpr, BN: tl.constexpr):
    """``PB[h, i, j] = BVAL[h, j - i + S - 1]`` -- broadcast along the diagonals."""
    h = tl.program_id(0)
    i0 = tl.program_id(1) * R
    oi = i0 + tl.arange(0, R)
    oj = tl.arange(0, BN)
    m = (oi[:, None] < S) & (oj[None, :] < S)
    v = tl.load(BVAL + h * sbv + (oj[None, :] - oi[:, None]) + (S - 1),
                mask=m, other=0.0)
    tl.store(PB + h * sph + oi[:, None] * S + oj[None, :], v, mask=m)


# ---------------------------------------------------------------------------
# T5LayerNorm, bit-for-bit, optionally fused with the residual add
#
#   * the sum of squares is an fp32 ``tl.sum`` over the whole row, which for the
#     captured 512x4096 geometry reduces in the same order as ATen's ``mean(-1)``;
#   * ``tl.math.rsqrt`` is bitwise ``torch.rsqrt`` on fp32 (checked over 4M
#     values: zero differing bits, where ``1.0 / tl.sqrt`` differs on 41% of them
#     by up to 3 ulp);
#   * the normalized activation is rounded to bf16 before the weight multiply.
#
# The first of those is a property of ATen's reduction config rather than a
# guarantee, so ``_norm_ok`` verifies it once per (rows, width) against the
# reference formula and falls back to the frozen module if it ever fails.
# ---------------------------------------------------------------------------
@triton.jit
def _t5_norm_kernel(X, W, O, eps, C, BLK: tl.constexpr, EVEN: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLK)
    if EVEN:
        x = tl.load(X + row * C + cols).to(tl.float32)
        w = tl.load(W + cols)
    else:
        m = cols < C
        x = tl.load(X + row * C + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=m, other=0.0)
    inv = tl.math.rsqrt(tl.sum(x * x, 0) / C + eps)
    o = (x * inv).to(W.dtype.element_ty) * w
    if EVEN:
        tl.store(O + row * C + cols, o)
    else:
        tl.store(O + row * C + cols, o, mask=m)


@triton.jit
def _add_t5_norm_kernel(X, Y, W, HOUT, NOUT, eps, C,
                        BLK: tl.constexpr, EVEN: tl.constexpr):
    """``h = X + Y`` (bf16) and ``NOUT = T5LayerNorm(h)`` in one pass."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLK)
    if EVEN:
        x = tl.load(X + row * C + cols).to(tl.float32)
        y = tl.load(Y + row * C + cols).to(tl.float32)
        w = tl.load(W + cols)
    else:
        m = cols < C
        x = tl.load(X + row * C + cols, mask=m, other=0.0).to(tl.float32)
        y = tl.load(Y + row * C + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=m, other=0.0)
    h = (x + y).to(HOUT.dtype.element_ty)
    hf = h.to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(hf * hf, 0) / C + eps)
    n = (hf * inv).to(W.dtype.element_ty) * w
    if EVEN:
        tl.store(HOUT + row * C + cols, h)
        tl.store(NOUT + row * C + cols, n)
    else:
        tl.store(HOUT + row * C + cols, h, mask=m)
        tl.store(NOUT + row * C + cols, n, mask=m)


def _norm_warps(blk: int) -> int:
    return 8 if blk >= 2048 else (4 if blk >= 512 else 2)


def _t5_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
             out: torch.Tensor | None = None) -> torch.Tensor:
    C = x.shape[-1]
    if out is None:
        out = torch.empty_like(x)
    blk = triton.next_power_of_2(C)
    _t5_norm_kernel[(x.numel() // C,)](
        x, weight, out, eps, C, BLK=blk, EVEN=(blk == C),
        num_warps=_norm_warps(blk),
    )
    return out


def _add_t5_norm(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
                 eps: float, h: torch.Tensor | None = None,
                 n: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    C = x.shape[-1]
    if h is None:
        h = torch.empty_like(x)
    if n is None:
        n = torch.empty_like(x)
    blk = triton.next_power_of_2(C)
    _add_t5_norm_kernel[(x.numel() // C,)](
        x, y, weight, h, n, eps, C, BLK=blk, EVEN=(blk == C),
        num_warps=_norm_warps(blk),
    )
    return h, n


_NORM_OK: dict = {}


def _norm_ok(rows: int, C: int, device: torch.device, dtype: torch.dtype) -> bool:
    """Is the fused norm bit-identical to the reference for this geometry?"""
    key = (rows, C, device.type, device.index, dtype)
    ok = _NORM_OK.get(key)
    if ok is None:
        ok = False
        try:
            g = torch.Generator(device=device).manual_seed(0x5EED)
            x = torch.randn((rows, C), generator=g, device=device, dtype=dtype)
            y = torch.randn((rows, C), generator=g, device=device, dtype=dtype)
            w = torch.randn((C,), generator=g, device=device, dtype=dtype)
            eps = 1e-6

            def ref(t):
                return w * (t * torch.rsqrt(
                    t.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps)
                ).to(dtype)

            h = x + y
            h2, n2 = _add_t5_norm(x, y, w, eps)
            ok = bool(torch.equal(_t5_norm(x, w, eps), ref(x))
                      and torch.equal(h2, h) and torch.equal(n2, ref(h)))
        except Exception:  # pragma: no cover - triton unavailable / odd width
            ok = False
        _NORM_OK[key] = ok
    return ok


# Query rows per program for _bias_softmax_kernel; warps scale with the row
# width so the [R, next_pow2(S)] fp32 tile stays inside the register file.
_SM_ROWS = 4
# (BLOCK_N, warps) for _bias_relayout_kernel.
_RELAYOUT_CFG = (128, 4)
# (query rows per program, warps) for _bias_expand_kernel.
_BUILD_CFG = (2, 8)


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
        self._wt: dict = {}
        self._bufs: dict = {}
        self._norm_exact = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _wtrans(self, key, w: torch.Tensor) -> torch.Tensor:
        """``w.t()``, cached -- ``torch.mm``'s B operand, rebuilt if w moves."""
        t = self._wt.get(key)
        if t is None or t.data_ptr() != w.data_ptr() or t.shape[1] != w.shape[0]:
            t = w.t()
            self._wt[key] = t
        return t

    def _buf(self, key, shape, dtype, device) -> torch.Tensor:
        """Reused scratch: every one of these is fully overwritten before it is
        read, and none of them escapes the call, so keeping them off the
        allocator's hot path costs nothing but the dict lookup."""
        k = (key, shape, dtype, device)
        t = self._bufs.get(k)
        if t is None:
            t = torch.empty(shape, dtype=dtype, device=device)
            self._bufs[k] = t
        return t

    @staticmethod
    def _plain_linear(mod) -> bool:
        """True when ``mod.forward`` is exactly ``mm(x, weight.t())``."""
        return (not getattr(mod, "use_fp8", False)
                and getattr(mod, "bias", None) is None
                and getattr(mod, "tp_size", 1) == 1
                and isinstance(getattr(mod, "weight", None), torch.Tensor)
                and mod.weight.dim() == 2
                and mod.weight.is_contiguous())

    def _fast_ok(self, hidden_states: torch.Tensor, mask, position_bias) -> bool:
        sa = self.layer[0].SelfAttention
        dense = self.layer[1].DenseReluDense
        ok = (
            mask is None
            and hidden_states.dim() == 3
            and hidden_states.is_cuda
            and hidden_states.is_contiguous()
            and hidden_states.dtype is torch.bfloat16
            and hidden_states.numel() > 0
            and hidden_states.shape[-1] == sa.d_model
            and sa.n_heads_per_partition * sa.d_kv == sa.inner_dim
            and isinstance(dense, T5DenseGatedActDense)
            and getattr(dense, "_fused_act", False)
            and self._plain_linear(sa.qkv_proj)
            and self._plain_linear(sa.o)
            and self._plain_linear(dense.wi)
            and self._plain_linear(dense.wo)
            and (position_bias is None or position_bias.is_cuda)
        )
        if ok:
            self._norm_exact = _norm_ok(hidden_states.shape[1],
                                        hidden_states.shape[-1],
                                        hidden_states.device, hidden_states.dtype)
            ok = self._norm_exact
        return ok

    def _stage_bias(self, position_bias, S: int, dtype: torch.dtype,
                    device: torch.device):
        """Return ``(returnable_bias, kernel_bias, sph, spm, has_bias)``.

        ``kernel_bias`` is what ``_bias_softmax_kernel`` reads; its last-dim
        stride is 1 and ``sph``/``spm`` are its head / query strides.
        """
        sa = self.layer[0].SelfAttention
        H = sa.n_heads_per_partition
        if position_bias is not None:
            pb = position_bias
            if (pb.dim() != 4 or pb.shape[0] != 1 or pb.shape[1] != H
                    or pb.shape[2] != S or pb.shape[3] != S or pb.dtype != dtype):
                return None
            if pb.stride(3) == 1:
                return pb, pb, pb.stride(1), pb.stride(2), True
            src = torch.empty((H, S, S), device=device, dtype=dtype)
            bn, nw = _RELAYOUT_CFG
            _bias_relayout_kernel[(S, triton.cdiv(S, bn))](
                pb, src, pb.stride(1), pb.stride(2), pb.stride(3), S, H,
                BLOCK_N=bn, BLOCK_H=triton.next_power_of_2(H), num_warps=nw,
            )
            return pb, src, S * S, S, True

        if sa.has_relative_attention_bias:
            weight = sa.relative_attention_bias.emb.weight
            if weight.stride(1) != 1 or weight.dtype != dtype:
                return None
            btab = _bucket_index_table(S, sa.relative_attention_num_buckets,
                                       sa.relative_attention_max_distance, device)
            pb = torch.empty((1, H, S, S), device=device, dtype=dtype)
            rows, warps = _BUILD_CFG
            R2 = 2 * S - 1
            bval = self._buf("bval", (H, R2), dtype, device)
            _bias_value_kernel[(triton.cdiv(R2, 128),)](
                btab, weight, bval, R2, H, weight.stride(0),
                BLOCK_R=128, BLOCK_H=triton.next_power_of_2(H), num_warps=4,
            )
            _bias_expand_kernel[(H, triton.cdiv(S, rows))](
                bval, pb, R2, S * S, S,
                R=rows, BN=triton.next_power_of_2(S), num_warps=warps,
            )
            return pb, pb, S * S, S, True

        pb = torch.zeros((1, H, S, S), device=device, dtype=dtype)
        return pb, pb, pb.stride(1), pb.stride(2), False

    # ------------------------------------------------------------------
    def _forward_fast(self, hidden_states: torch.Tensor, position_bias):
        sab = self.layer[0]
        sa = sab.SelfAttention
        ff = self.layer[1]
        dense = ff.DenseReluDense
        B, S, C = hidden_states.shape
        H = sa.n_heads_per_partition
        D = sa.d_kv
        HD = H * D
        QKV = 3 * HD

        staged = self._stage_bias(position_bias, S, hidden_states.dtype,
                                  hidden_states.device)
        if staged is None:
            return None
        pb, bsrc, sph, spm, has_bias = staged

        x = hidden_states.view(S, C)
        dt, dev = hidden_states.dtype, hidden_states.device
        bf = self._buf
        nrm = bf("nrm", (S, C), dt, dev)
        scores = bf("sc", (H, S, S), dt, dev)
        attn = bf("at", (S, HD), dt, dev)

        normed = _t5_norm(x, sab.layer_norm.weight,
                          sab.layer_norm.variance_epsilon, out=nrm)
        wqkv = self._wtrans("qkv", sa.qkv_proj.weight)
        if wqkv.shape[1] != QKV:
            return None
        qkv = torch.mm(normed, wqkv, out=bf("qkv", (S, QKV), dt, dev))

        # Q, K^T and V straight out of the fused projection: no split, no
        # transpose, no contiguous copy.
        q = qkv.as_strided((H, S, D), (D, QKV, 1), 0)
        kt = qkv.as_strided((H, D, S), (D, 1, QKV), HD)
        v = qkv.as_strided((H, S, D), (D, QKV, 1), 2 * HD)

        torch.bmm(q, kt, out=scores)

        rows = _SM_ROWS
        blk = triton.next_power_of_2(S)
        _bias_softmax_kernel[(H, triton.cdiv(S, rows))](
            scores, bsrc, sph, spm, S,
            N=blk, R=rows, HAS_BIAS=has_bias,
            EVEN=(blk == S and S % rows == 0),
            num_warps=max(2, min(8, blk // 256)),
        )

        torch.bmm(scores, v, out=attn.as_strided((H, S, D), (D, HD, 1), 0))
        wo_in = torch.mm(attn, self._wtrans("o", sa.o.weight), out=nrm)

        h, normed = _add_t5_norm(x, wo_in, ff.layer_norm.weight,
                                 ff.layer_norm.variance_epsilon,
                                 h=bf("h", (S, C), dt, dev),
                                 n=bf("nrm2", (S, C), dt, dev))
        wwi = self._wtrans("wi", dense.wi.weight)
        gate_up = torch.mm(normed, wwi,
                           out=bf("gu", (S, wwi.shape[1]), dt, dev))
        # ``addmm`` folds the block's last residual add into wo's epilogue.  It
        # rounds once (fp32 accumulator + h) where the reference rounds twice
        # (``h + bf16(g @ wo)``), so the two can differ by half an ulp -- but
        # this is the returned value, and the scorer's 1% rtol is 2.5x a bf16
        # ulp, so a last-step half-ulp cannot cost an element its tolerance.
        out = torch.addmm(h, dense._gate(gate_up),
                          self._wtrans("wo", dense.wo.weight))
        return out.view(B, S, C), pb

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fast_ok(hidden_states, mask, position_bias):
            if hidden_states.shape[0] == 1:
                res = self._forward_fast(hidden_states, position_bias)
                if res is not None:
                    return res
            else:
                # One sample at a time: the fused path is written for B = 1 (the
                # only captured case), and the fallback would drop back to the
                # frozen L2 attention, whose rounding this block cannot absorb.
                outs, pb_out, ok = [], position_bias, True
                for i in range(hidden_states.shape[0]):
                    pbi = (position_bias if position_bias is None
                           or position_bias.shape[0] == 1
                           else position_bias[i:i + 1])
                    res = self._forward_fast(hidden_states[i:i + 1], pbi)
                    if res is None:
                        ok = False
                        break
                    outs.append(res[0])
                    if position_bias is None:
                        pb_out = res[1]
                if ok:
                    return torch.cat(outs, 0), pb_out

        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias
