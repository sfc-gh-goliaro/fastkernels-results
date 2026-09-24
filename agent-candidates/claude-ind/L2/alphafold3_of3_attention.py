"""Multi-head attention with bias list support for AlphaFold3 (L2).

Fused Triton implementation.

The captured shapes are tiny (16-384 query rows, head dims of 24/32/48, key
lengths of 16/128) and every projection is square, so the baseline's ~16 separate
CUDA kernels cost far more in per-launch dispatch -- on both the CPU and the GPU
side -- than in arithmetic.  Two things therefore matter, in this order:

* **Host cost.**  The harness times with CUDA events while the CPU races ahead of
  the GPU, so a forward that is slow to *enqueue* is slow to measure.  Everything
  shape-derived (tile configs, grids, scratch offsets, bias stride plans, the
  packed weight copy) is computed once per input signature and cached as a
  closure in ``_plans``; the steady-state forward only builds an output tensor and
  fires three kernels.
* **Kernel count.**  The composition is collapsed into three launches:

  1. ``_proj_kernel`` -- q / g / k / v projections in one launch, with the
     ``1/sqrt(c_hidden)`` scale, the q bias and the gate's sigmoid folded into the
     epilogue.  q/g read ``q_x`` while k/v read ``kv_x`` and the two groups
     generally have different row counts, so the tile-id space is laid out as
     ``[q | g | k | v]`` and decoded per CTA -- no CTA is launched for a tile that
     does not exist.
  2. ``_attn_kernel`` -- scores + bias list + softmax + PV + the gate multiply,
     one CTA per (batch, head, q-tile).  Keys are capped at a single tile so the
     softmax is one pass rather than an online rescale.
  3. ``_out_kernel`` -- the output projection, reusing the projections' tile
     routine.

  Folding the output projection into the attention kernel was measurably worse:
  it forces the per-head attention to be recomputed once per output-column tile,
  and that costs more than the launch it saves.

Intermediate values are rounded to bfloat16 at exactly the points the reference
composition rounds them, so results track the baseline closely.

Anything the fused path does not cover (non-bf16 inputs, very long keys, a bias
whose broadcast cannot be expressed as flat strides) falls back to the reference
composition in :meth:`OF3Attention._reference`.

Reference: openfold3/core/model/primitives/attention.py Attention
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.softmax import Softmax

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if _HAVE_TRITON:

    @triton.jit
    def _gemm_tile(
        X, W, O, M, C, N, pm, pn, BIAS, scale,
        KIND: tl.constexpr, HAS_BIAS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """One [BM, BN] tile of ``O = epilogue(X @ W^T)``.

        ``X`` is [M, C] row-major, ``W`` is [N, C] row-major, ``O`` is [M, N].
        KIND 0 = q (bias, then the 1/sqrt(d) scale), 1 = gate (sigmoid),
        2 = plain.
        """
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        mm = rm < M
        mn = rn < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, C, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < C
            a = tl.load(X + rm[:, None] * C + rk[None, :],
                        mask=mm[:, None] & mk[None, :], other=0.0)
            b = tl.load(W + rn[:, None] * C + rk[None, :],
                        mask=mn[:, None] & mk[None, :], other=0.0)
            acc = tl.dot(a, tl.trans(b), acc)
        if KIND == 0 and HAS_BIAS:
            acc += tl.load(BIAS + rn, mask=mn, other=0.0).to(tl.float32)[None, :]
        o = acc.to(tl.bfloat16)
        if KIND == 0:
            o = (o.to(tl.float32) * scale).to(tl.bfloat16)
        if KIND == 1:
            o = tl.sigmoid(o.to(tl.float32)).to(tl.bfloat16)
        tl.store(O + rm[:, None] * N + rn[None, :], o,
                 mask=mm[:, None] & mn[None, :])

    @triton.jit
    def _proj_kernel(
        QX, KVX, W, BQB, SC,
        M0, M1, C, HD,
        nt_m0, nt_m1, nt_n, t0, scale,
        off_q, off_g, off_k, off_v,
        w_q, w_g, w_k, w_v,
        HAS_BQ: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """q/g from ``QX``, k/v from ``KVX``, all four packed into ``SC``."""
        pid = tl.program_id(0)
        if pid < t0:
            nt = nt_m0 * nt_n
            sub = pid // nt
            r = pid % nt
            pm = r // nt_n
            pn = r % nt_n
            if sub == 0:
                _gemm_tile(QX, W + w_q, SC + off_q, M0, C, HD, pm, pn, BQB,
                           scale, 0, HAS_BQ, BM, BN, BK)
            else:
                _gemm_tile(QX, W + w_g, SC + off_g, M0, C, HD, pm, pn, BQB,
                           scale, 1, HAS_BQ, BM, BN, BK)
        else:
            p = pid - t0
            nt = nt_m1 * nt_n
            sub = p // nt
            r = p % nt
            pm = r // nt_n
            pn = r % nt_n
            if sub == 0:
                _gemm_tile(KVX, W + w_k, SC + off_k, M1, C, HD, pm, pn, BQB,
                           scale, 2, HAS_BQ, BM, BN, BK)
            else:
                _gemm_tile(KVX, W + w_v, SC + off_v, M1, C, HD, pm, pn, BQB,
                           scale, 2, HAS_BQ, BM, BN, BK)

    @triton.jit
    def _out_kernel(
        OG, off_o, WO, OUT, M, HD, CQ, nt_n,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """The output projection ``OUT = OG @ WO^T``."""
        pid = tl.program_id(0)
        _gemm_tile(OG + off_o, WO, OUT, M, HD, CQ, pid // nt_n, pid % nt_n,
                   OG, 0.0, 2, False, BM, BN, BK)

    @triton.jit
    def _attn_kernel(
        SC, B0, B1, off_q, off_g, off_k, off_v, off_o,
        Qn, Kn, D, HD, nt_q,
        b0_sb, b0_sh, b0_sq, b0_sk,
        b1_sb, b1_sh, b1_sq, b1_sk,
        H: tl.constexpr, GATING: tl.constexpr, NBIAS: tl.constexpr,
        BQ: tl.constexpr, BK: tl.constexpr, DP: tl.constexpr,
    ):
        """Scores + biases + softmax + PV + gate for one (batch, head, q-tile)."""
        pid = tl.program_id(0)
        pq = pid % nt_q
        t = pid // nt_q
        h = t % H
        b = t // H

        rq = pq * BQ + tl.arange(0, BQ)
        rk = tl.arange(0, BK)
        rd = tl.arange(0, DP)
        mq = rq < Qn
        mk = rk < Kn
        md = rd < D
        hd = h * D + rd

        qm = mq[:, None] & md[None, :]
        km = mk[:, None] & md[None, :]
        qrow = (b * Qn + rq)[:, None] * HD
        krow = (b * Kn + rk)[:, None] * HD

        q = tl.load(SC + off_q + qrow + hd[None, :], mask=qm, other=0.0)
        k = tl.load(SC + off_k + krow + hd[None, :], mask=km, other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        sm = mq[:, None] & mk[None, :]
        if NBIAS >= 1:
            bb = tl.load(B0 + b * b0_sb + h * b0_sh
                         + rq[:, None] * b0_sq + rk[None, :] * b0_sk,
                         mask=sm, other=0.0)
            s = (s.to(tl.float32) + bb.to(tl.float32)).to(tl.bfloat16)
        if NBIAS >= 2:
            bb = tl.load(B1 + b * b1_sb + h * b1_sh
                         + rq[:, None] * b1_sq + rk[None, :] * b1_sk,
                         mask=sm, other=0.0)
            s = (s.to(tl.float32) + bb.to(tl.float32)).to(tl.bfloat16)
        s32 = tl.where(mk[None, :], s.to(tl.float32), float("-inf"))
        p = tl.exp(s32 - tl.max(s32, 1)[:, None])
        p = (p / tl.sum(p, 1)[:, None]).to(tl.bfloat16)
        v = tl.load(SC + off_v + krow + hd[None, :], mask=km, other=0.0)
        o = tl.dot(p, v).to(tl.bfloat16)
        if GATING:
            g = tl.load(SC + off_g + qrow + hd[None, :], mask=qm, other=0.0)
            o = (o.to(tl.float32) * g.to(tl.float32)).to(tl.bfloat16)
        tl.store(SC + off_o + qrow + hd[None, :], o, mask=qm)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------
def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pick_bn(n: int, rows: int, target: int) -> int:
    """Largest power-of-two column tile of ``n`` still giving ``target`` CTAs."""
    best = 16
    for bn in (128, 64, 32, 16):
        if bn > n:
            continue
        if rows * ((n + bn - 1) // bn) >= target:
            return bn
        best = bn
    return best


def _gemm_cfg(rows_per_group, groups, n, c, target=150):
    """(BM, BN, BK, num_warps) for a skinny ``[rows, c] @ [c, n]`` tile GEMM.

    These GEMMs are all short and fat, so the useful knob is CTA count rather
    than tile efficiency: small row tiles plus the narrowest column tile that
    still reaches ``target`` CTAs measured fastest across the captured shapes.
    """
    mx = max(rows_per_group)
    bm = 16 if mx <= 16 else 32
    rows = sum(-(-m // bm) for m in rows_per_group) * groups
    bn = _pick_bn(n, rows, target)
    bk = max(16, min(_pow2(c), 256))  # tl.dot requires K >= 16
    return bm, bn, bk, max(2, min(8, _pow2(max(1, (bm * bn) // 512))))


def _bias_plan(bias: torch.Tensor, batch: tuple[int, ...], H: int, Q: int, K: int):
    """Collapse *bias* to flat (batch, head, q, k) element strides.

    Returns ``(sb, sh, sq, sk)`` such that the bias element broadcast to score
    position ``(b_flat, h, i, j)`` lives at ``sb*b_flat + sh*h + sq*i + sk*j``,
    or ``None`` when the broadcast cannot be expressed that way.
    """
    full = batch + (H, Q, K)
    r = bias.dim()
    if r > len(full):
        return None
    pad = len(full) - r
    strides = [0] * len(full)
    for i in range(r):
        d = pad + i
        sz = bias.shape[i]
        if sz == full[d]:
            strides[d] = bias.stride(i)
        elif sz != 1:
            return None
    nb = len(batch)
    # The flattened batch index has to act through a single linear stride.
    sb = 0
    mult = 1
    for i in range(nb - 1, -1, -1):
        if batch[i] != 1:
            if sb == 0:
                sb = strides[i] // mult
            if strides[i] != sb * mult:
                return None
        mult *= batch[i]
    return sb, strides[nb], strides[nb + 1], strides[nb + 2]


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

        # Per-input-signature launch plans, the packed weight copy they close
        # over, and a reusable scratch buffer.  Invalidated whenever the weights
        # change: storage swaps via _apply, load_state_dict via the hook below,
        # and in-place writes via the _versions() check in forward.
        self._plans: dict = {}
        self._packed = None
        self._scratch = None
        self._wver = None
        hook = getattr(self, "register_load_state_dict_post_hook", None) \
            or self._register_load_state_dict_post_hook
        hook(lambda mod, incompatible_keys: mod._invalidate())

    # -- reference path ----------------------------------------------------
    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def _reference(self, q_x, kv_x, biases):
        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)
        return self._wrap_up(o, q_x)

    # -- plan / weight bookkeeping -----------------------------------------
    def _invalidate(self):
        self._plans.clear()
        self._packed = None
        self._wver = None

    def _apply(self, *args, **kwargs):
        # Covers .to(device) / .to(dtype) / .float(), which swap param storage.
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _versions(self):
        """Version counters of every weight the packed copy is derived from.

        Cheap enough to check on each call, and it catches in-place mutation
        that neither ``_apply`` nor the load-state-dict hook would see.
        """
        g = self.linear_g
        b = self.linear_q.bias
        return (self.linear_q.weight._version, self.linear_k.weight._version,
                self.linear_v.weight._version, self.linear_o.weight._version,
                g.weight._version if g is not None else 0,
                b._version if b is not None else 0)

    def _pack_weights(self):
        """Concatenate the projection weights once into [n, H*D, C]."""
        parts = [self.linear_q.weight]
        if self.linear_g is not None:
            parts.append(self.linear_g.weight)
        parts += [self.linear_k.weight, self.linear_v.weight]
        w = torch.cat(parts, 0).contiguous()
        wo = self.linear_o.weight.contiguous()
        bq = self.linear_q.bias
        if bq is not None:
            bq = bq.contiguous()
        self._packed = (w, wo, bq)
        self._wver = self._versions()
        return self._packed

    def _get_scratch(self, n):
        sc = self._scratch
        if sc is None or sc.numel() < n:
            sc = torch.empty(n, dtype=torch.bfloat16,
                             device=self.linear_q.weight.device)
            self._scratch = sc
        return sc

    # -- fused path --------------------------------------------------------
    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []
        # The plan bakes in shapes, dtypes, device and every bias's broadcast
        # strides, so all of those have to select it.
        n = len(biases)
        if n == 0:
            key = (q_x.shape, kv_x.shape, q_x.dtype, kv_x.dtype, q_x.device)
        elif n == 1:
            b0 = biases[0]
            key = (q_x.shape, kv_x.shape, q_x.dtype, kv_x.dtype, q_x.device,
                   b0.shape, b0.stride(), b0.dtype)
        elif n == 2:
            b0, b1 = biases
            key = (q_x.shape, kv_x.shape, q_x.dtype, kv_x.dtype, q_x.device,
                   b0.shape, b0.stride(), b0.dtype,
                   b1.shape, b1.stride(), b1.dtype)
        else:
            return self._reference(q_x, kv_x, biases)

        plan = self._plans.get(key)
        if plan is None or self._wver != self._versions():
            if plan is not None:
                self._invalidate()
            plan = self._build_plan(q_x, kv_x, biases, key)
        if plan is False:
            return self._reference(q_x, kv_x, biases)
        return plan(q_x, kv_x, biases)

    def _build_plan(self, q_x, kv_x, biases, key):
        """Resolve tiles, grids, offsets and bias strides once for a signature."""
        plan = self._make_plan(q_x, kv_x, biases)
        self._plans[key] = plan
        return plan

    def _make_plan(self, q_x, kv_x, biases):
        self._wver = self._versions()
        if not _HAVE_TRITON or not q_x.is_cuda:
            return False
        H, D = self.no_heads, self.c_hidden
        HD = H * D
        batch = tuple(q_x.shape[:-2])
        if (len(batch) > 8 or tuple(kv_x.shape[:-2]) != batch
                or q_x.shape[-1] != self.c_q or kv_x.shape[-1] != self.c_k
                or self.c_k != self.c_q or self.c_v != self.c_q
                or q_x.dtype is not torch.bfloat16
                or kv_x.dtype is not torch.bfloat16
                or self.linear_q.weight.dtype is not torch.bfloat16):
            return False
        Q = q_x.shape[-2]
        K = kv_x.shape[-2]
        C = self.c_q
        CQ = self.linear_o.weight.shape[0]
        if K > 256 or Q > 128:
            return False

        z = (0, 0, 0, 0)
        plans = []
        for b in biases:
            if b.dtype is not torch.bfloat16:
                return False
            p = _bias_plan(b, batch, H, Q, K)
            if p is None:
                return False
            plans.append(p)
        nbias = len(plans)
        p0 = plans[0] if nbias > 0 else z
        p1 = plans[1] if nbias > 1 else z

        packed = self._packed
        if packed is None:
            packed = self._pack_weights()
        w, wo, bq = packed

        B = 1
        for s in batch:
            B *= s
        M0 = B * Q
        M1 = B * K
        gating = self.linear_g is not None
        n0 = 2 if gating else 1

        n_sc = ((n0 + 1) * M0 + 2 * M1) * HD
        off_q = 0
        off_g = M0 * HD if gating else 0
        off_k = n0 * M0 * HD
        off_v = off_k + M1 * HD
        off_o = off_v + M1 * HD

        BM, BN, BK, nw = _gemm_cfg((M0, M1), 2, HD, C)
        nt_m0 = -(-M0 // BM)
        nt_m1 = -(-M1 // BM)
        nt_n = -(-HD // BN)
        t0 = n0 * nt_m0 * nt_n
        proj_grid = (t0 + 2 * nt_m1 * nt_n,)
        proj_args = [
            q_x, kv_x, w, bq if bq is not None else w, None,
            M0, M1, C, HD, nt_m0, nt_m1, nt_n, t0, 1.0 / math.sqrt(D),
            off_q, off_g, off_k, off_v,
            0, HD * C if gating else 0, n0 * HD * C, (n0 + 1) * HD * C,
        ]
        proj_kw = dict(HAS_BQ=bq is not None, BM=BM, BN=BN, BK=BK,
                       num_warps=nw, num_stages=3)

        # A 16-row q tile maximises CTA count; every tl.dot extent needs >= 16.
        BQ = 16
        BKa = max(16, _pow2(K))
        DP = max(16, _pow2(D))
        nt_q = -(-Q // BQ)
        attn_grid = (B * H * nt_q,)
        attn_args = [
            None, None, None, off_q, off_g, off_k, off_v, off_o,
            Q, K, D, HD, nt_q, *p0, *p1,
        ]
        attn_kw = dict(H=H, GATING=gating, NBIAS=nbias, BQ=BQ, BK=BKa, DP=DP,
                       num_warps=max(1, min(8, _pow2(BQ * max(BKa, DP) // 256))),
                       num_stages=2)

        BM2, BN2, BK2, nw2 = _gemm_cfg((M0,), 1, CQ, HD)
        nt_n2 = -(-CQ // BN2)
        out_grid = (-(-M0 // BM2) * nt_n2,)
        out_args = [None, off_o, wo, None, M0, HD, CQ, nt_n2]
        out_kw = dict(BM=BM2, BN=BN2, BK=BK2, num_warps=nw2, num_stages=3)

        out_shape = batch + (Q, CQ)
        proj = _proj_kernel
        attn = _attn_kernel
        outk = _out_kernel
        get_scratch = self._get_scratch
        dev = q_x.device

        def run(q_x, kv_x, biases):
            if not q_x.is_contiguous():
                q_x = q_x.contiguous()
            if not kv_x.is_contiguous():
                kv_x = kv_x.contiguous()
            sc = get_scratch(n_sc)
            out = torch.empty(out_shape, dtype=torch.bfloat16, device=dev)
            proj_args[0] = q_x
            proj_args[1] = kv_x
            proj_args[4] = sc
            proj[proj_grid](*proj_args, **proj_kw)
            attn_args[0] = sc
            attn_args[1] = biases[0] if nbias > 0 else sc
            attn_args[2] = biases[1] if nbias > 1 else sc
            attn[attn_grid](*attn_args, **attn_kw)
            out_args[0] = sc
            out_args[3] = out
            outk[out_grid](*out_args, **out_kw)
            return out

        return run
