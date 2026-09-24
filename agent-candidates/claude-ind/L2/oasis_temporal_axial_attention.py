"""Oasis temporal axial attention -- fused Triton implementation.

The captured shapes are tiny (x: [1, T<=6, 9, 16, 1024], ~7 GFLOP total) so the
eager baseline is dominated by per-op overhead: ~40 kernels for the qkv GEMM,
three permute copies, ~20 elementwise rotary ops, SDPA with its own layout
copies, an output permute copy and the out GEMM.

Here a forward is two or three kernels, replayed from a per-shape CUDA graph so
the whole call costs one graph launch on the CPU side:

  ``_qkv_attn_kernel``    x @ W_qkv^T, rotary, and causal attention along the
                          *time* axis, fused.  One program owns S spatial
                          positions x all T timesteps (BM = S*T rows) of one
                          head, so every sequence it touches is complete and the
                          attention happens in-register -- no round trip through
                          a [tokens, 3*dim] scratch buffer.
  ``_out_proj_kernel``    attn_out @ W_out^T + bias.

Shapes the fused kernel does not suit (its spatial blocking can leave too many
idle rows) fall back to the unfused pair instead:

  ``_qkv_kernel``         x @ W_qkv^T -> qkv[tokens, 3*dim]
  ``_attn_kernel``        rotary + causal attention, PK spatial positions of one
                          head packed into a single tile with a block-diagonal
                          mask so one MMA pair serves all of them.

Two notes on why things sit where they do.  Rotary lives in the attention step
rather than the GEMM epilogue because a [rows, dim_head] cos/sin tile is a
contiguous load there, while the same table indexed by a GEMM output tile is a
2-D gather.  And the token axis of every buffer is padded up to a whole number
of row-tiles, which lets both GEMMs drop their M mask for free -- a masked
partial tile costs the same MMA work as a full one.

Inputs the kernels do not cover (fp32, time > 8, odd dim_head) fall through to
``_ref_forward``, which is the baseline implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

_BT_MAX = 8  # rotary table rows / packed sequence length (captured time <= 6)


if _HAS_TRITON:

    @triton.jit
    def _swizzle(pid, M, N, BM: tl.constexpr, BN: tl.constexpr, GM: tl.constexpr):
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        in_group = GM * num_pid_n
        group_id = pid // in_group
        first_m = group_id * GM
        group_m = min(num_pid_m - first_m, GM)
        return first_m + ((pid % in_group) % group_m), (pid % in_group) // group_m

    @triton.jit
    def _qkv_kernel(
        X, W, O, M, N, K,
        stride_xm, stride_wn, stride_om,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
        WS: tl.constexpr,
    ):
        pid_m, pid_n = _swizzle(tl.program_id(0), M, N, BM, BN, GM)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :]
        w_ptrs = W + offs_n[:, None] * stride_wn + offs_k[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in tl.range(0, K, BK, warp_specialize=WS):
            acc = tl.dot(tl.load(x_ptrs), tl.trans(tl.load(w_ptrs)), acc)
            x_ptrs += BK
            w_ptrs += BK
        tl.store(O + offs_m[:, None] * stride_om + offs_n[None, :],
                 acc.to(O.dtype.element_ty))

    @triton.jit
    def _out_proj_kernel(
        A, W, B, C, M, N, K,
        stride_am, stride_wn, stride_cm,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
        WS: tl.constexpr,
    ):
        pid_m, pid_n = _swizzle(tl.program_id(0), M, N, BM, BN, GM)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
        w_ptrs = W + offs_n[:, None] * stride_wn + offs_k[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in tl.range(0, K, BK, warp_specialize=WS):
            acc = tl.dot(tl.load(a_ptrs), tl.trans(tl.load(w_ptrs)), acc)
            a_ptrs += BK
            w_ptrs += BK
        acc += tl.load(B + offs_n)[None, :].to(tl.float32)
        tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :],
                 acc.to(C.dtype.element_ty))

    @triton.jit
    def _attn_kernel(
        QKV, O, COS, SIN,
        T, NPOS, BNP, NBLK, stride_qm, stride_om, SCALE,
        H: tl.constexpr, DH: tl.constexpr, ND: tl.constexpr,
        BT: tl.constexpr, PK: tl.constexpr, IS_CAUSAL: tl.constexpr,
    ):
        """Rotary + causal attention over the time axis for PK spatial
        positions of one head, packed into one (BT*PK) x DH tile."""
        pid = tl.program_id(0)
        per_batch = NBLK * H
        b = pid // per_batch
        rem = pid % per_batch
        h = rem % H
        n0 = (rem // H) * PK

        r = tl.arange(0, BT * PK)
        od = tl.arange(0, DH)
        tt = r % BT
        gg = r // BT
        nn = n0 + gg
        live = (tt < T) & (nn < NPOS)
        same = gg[:, None] == gg[None, :]
        if IS_CAUSAL:
            valid = same & (tt[:, None] >= tt[None, :]) & live[None, :]
        else:
            valid = same & live[None, :]

        rowm = b * BNP + tt * NPOS + nn
        base = rowm[:, None] * stride_qm + (h * DH + od[None, :])
        # rotary: interleaved pairs (2i, 2i+1).  COS/SIN are [BT_MAX, DH] tables
        # already expanded over the pair axis (SIN carries the rotate_half sign),
        # so both are contiguous loads; the paired element is fetched with a
        # second, L1-resident load at column d^1 instead of an in-register
        # de-interleave.
        rot_ofs = tt[:, None] * DH + od[None, :]
        cs = tl.load(COS + rot_ofs)
        sn = tl.load(SIN + rot_ofs)
        swp = (od ^ 1)[None, :] - od[None, :]
        q = tl.load(QKV + base, mask=live[:, None], other=0.0).to(tl.float32)
        qs = tl.load(QKV + base + swp, mask=live[:, None], other=0.0).to(tl.float32)
        k = tl.load(QKV + base + ND, mask=live[:, None], other=0.0).to(tl.float32)
        ks = tl.load(QKV + base + ND + swp, mask=live[:, None], other=0.0).to(tl.float32)
        v = tl.load(QKV + base + 2 * ND, mask=live[:, None], other=0.0)
        q = (q * cs + qs * sn).to(QKV.dtype.element_ty)
        k = (k * cs + ks * sn).to(QKV.dtype.element_ty)

        s = tl.dot(q, tl.trans(k)) * SCALE
        s = tl.where(valid, s, -1.0e30)
        p = tl.exp(s - tl.max(s, 1)[:, None])
        p = p / tl.sum(p, 1)[:, None]
        o = tl.dot(p.to(QKV.dtype.element_ty), v)

        tl.store(O + rowm[:, None] * stride_om + (h * DH + od[None, :]),
                 o.to(O.dtype.element_ty), mask=live[:, None])

    @triton.jit
    def _rot_pairs(z, cs, sn, R: tl.constexpr, DH: tl.constexpr):
        """Interleaved-pair rotary on a [R, DH] tile (SIN carries the sign)."""
        z1, z2 = tl.split(tl.reshape(z, (R, DH // 2, 2)))
        sw = tl.reshape(tl.join(z2, z1), (R, DH))
        return z * cs + sw * sn

    @triton.jit
    def _qkv_attn_kernel(
        X, W, O, COS, SIN,
        T, NPOS, S, NBLK, BNP, K, SCALE,
        H: tl.constexpr, DH: tl.constexpr, ND: tl.constexpr,
        BM: tl.constexpr, BK: tl.constexpr, IS_CAUSAL: tl.constexpr,
    ):
        """qkv projection + rotary + causal-time attention, fused.

        One program owns S spatial positions x T timesteps (BM = S*T rows, so
        every sequence it touches is complete) for a single head, which is what
        lets the attention happen in-register right after the projection --
        no round trip through a [tokens, 3*dim] scratch buffer.
        """
        pid = tl.program_id(0)
        per_batch = NBLK * H
        b = pid // per_batch
        rem = pid % per_batch
        h = rem % H
        sb = rem // H
        r = tl.arange(0, BM)
        od = tl.arange(0, DH)
        ok = tl.arange(0, BK)
        s_idx = r // T
        t_idx = r % T
        n = sb * S + s_idx
        live = (s_idx < S) & (n < NPOS)
        xrow = tl.where(live, b * BNP + t_idx * NPOS + n, 0)

        xp = X + xrow[:, None] * K + ok[None, :]
        wq = W + (h * DH + od)[:, None] * K + ok[None, :]
        wk = wq + ND * K
        wv = wq + 2 * ND * K
        aq = tl.zeros((BM, DH), dtype=tl.float32)
        ak = tl.zeros((BM, DH), dtype=tl.float32)
        av = tl.zeros((BM, DH), dtype=tl.float32)
        for _ in tl.range(0, K, BK):
            a = tl.load(xp, mask=live[:, None], other=0.0)
            aq = tl.dot(a, tl.trans(tl.load(wq)), aq)
            ak = tl.dot(a, tl.trans(tl.load(wk)), ak)
            av = tl.dot(a, tl.trans(tl.load(wv)), av)
            xp += BK
            wq += BK
            wk += BK
            wv += BK

        rof = t_idx[:, None] * DH + od[None, :]
        cs = tl.load(COS + rof)
        sn = tl.load(SIN + rof)
        q = _rot_pairs(aq, cs, sn, BM, DH).to(X.dtype.element_ty)
        k = _rot_pairs(ak, cs, sn, BM, DH).to(X.dtype.element_ty)
        v = av.to(X.dtype.element_ty)

        sc = tl.dot(q, tl.trans(k)) * SCALE
        same = s_idx[:, None] == s_idx[None, :]
        if IS_CAUSAL:
            valid = same & (t_idx[:, None] >= t_idx[None, :]) & live[None, :]
        else:
            valid = same & live[None, :]
        # Masked-out entries get a large finite sentinel rather than -inf: rows
        # past the end of the spatial block have no valid key at all, and
        # -inf - (-inf) would make their softmax NaN.  With a finite sentinel
        # those rows come out uniform (and are dropped by the store mask), while
        # live rows are unaffected -- exp(-1e30 - max) underflows to 0 either way.
        sc = tl.where(valid, sc, -1.0e30)
        p = tl.exp(sc - tl.max(sc, 1)[:, None])
        p = p / tl.sum(p, 1)[:, None]
        o = tl.dot(p.to(X.dtype.element_ty), v)
        tl.store(O + xrow[:, None] * ND + (h * DH + od[None, :]),
                 o.to(O.dtype.element_ty), mask=live[:, None])


# Spatial positions packed into one attention tile.  BT (>=8) * _ATTN_PK must be
# at least 16, the minimum MMA tile; PK need not divide the position count -- the
# kernel bounds-checks it.
_ATTN_PK = 2


# Tile configs, chosen by timing the whole forward (not each kernel alone:
# the token axis is padded up to a whole number of row-tiles, so a kernel that
# looks faster in isolation can lose by forcing more padding on its neighbour).
#
# (BM, BK, num_warps, num_stages) for the fused qkv+attention kernel, keyed by
# token count.  A shape absent here runs the unfused pair instead -- at 720
# tokens the fused kernel's spatial blocking wastes too many rows to pay off.
_FUSED_CFG: dict = {
    288: (32, 64, 4, 3),
    432: (64, 64, 8, 3),
    576: (64, 64, 8, 3),
    864: (128, 64, 8, 3),
}
# (BM, BN, BK, GROUP_M, num_warps, num_stages, warp_specialize) for the GEMMs.
_QKV_CFG_DEFAULT = (128, 128, 64, 8, 8, 4, False)
_OUT_CFG_DEFAULT = (64, 64, 128, 8, 4, 3, False)
_QKV_CFG: dict = {}
_OUT_CFG: dict = {}


class _Plan:
    """Pre-allocated buffers (plus a CUDA graph) for one input shape."""

    __slots__ = ("x", "qkv", "att", "out", "xv", "outv", "graph", "shape", "mpad")

    def __init__(self, x, qkv, att, out, xv, outv, shape, mpad):
        self.x, self.qkv, self.att, self.out = x, qkv, att, out
        self.xv, self.outv, self.shape, self.mpad = xv, outv, shape, mpad
        self.graph = None


class OasisTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
        *,
        is_causal: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")

        self.dim = dim
        self.dim_head = dim_head
        self._rope_cache: dict = {}
        self._plans: dict = {}
        self._use_graph = True

    # ------------------------------------------------------------------ #
    # reference (eager) path -- used when the fast path does not apply
    # ------------------------------------------------------------------ #
    def _ref_forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)

        q = q.reshape(bsz * height * width, self.heads, time, -1)
        k = k.reshape(bsz * height * width, self.heads, time, -1)
        v = v.reshape(bsz * height * width, self.heads, time, -1)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, causal=self.is_causal)
        out = out.reshape(bsz, height, width, time, self.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))

    # ------------------------------------------------------------------ #
    def _rope_tables(self, dtype: torch.dtype):
        """cos/sin of shape [_BT_MAX, dim_head], expanded over rotary pairs.

        Built exactly the way ``OasisRotaryEmbedding.rotate_queries_or_keys``
        does (positions in the activation dtype, cos/sin in the freqs dtype)
        so the values match the baseline bit for bit.
        """
        key = dtype
        got = self._rope_cache.get(key)
        if got is not None:
            return got
        freqs = self.rotary_emb.freqs
        positions = torch.arange(_BT_MAX, device=freqs.device, dtype=dtype)
        sf = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        sf = sf.repeat_interleave(2, dim=-1)
        pad = self.dim_head - sf.shape[-1]
        if pad > 0:  # partial-rotary: identity on the untouched tail
            sf = torch.nn.functional.pad(sf, (0, pad))
            mask = torch.zeros(self.dim_head, device=sf.device, dtype=torch.float32)
            mask[: self.dim_head - pad] = 1.0
        else:
            mask = None
        cos = sf.cos().float()
        sgn = torch.where(
            torch.arange(sf.shape[-1], device=sf.device) % 2 == 0, -1.0, 1.0
        )
        sin = sf.sin().float() * sgn
        if mask is not None:
            cos = torch.where(mask.bool(), cos, torch.ones_like(cos))
            sin = sin * mask
        got = (cos.contiguous(), sin.contiguous())
        self._rope_cache[key] = got
        return got

    def _launch(self, plan: _Plan) -> None:
        bsz, time, height, width, _ = plan.shape
        dim, dh, h = self.dim, self.dim_head, self.heads
        nd = h * dh
        npos = height * width
        mreal = bsz * time * npos
        m = plan.mpad
        cos, sin = self._rope_tables(plan.x.dtype)
        xf = plan.x
        qkv, att, out = plan.qkv, plan.att, plan.out
        wq, wo, bo = self.to_qkv.weight, self.to_out.weight, self.to_out.bias

        bnp = time * npos           # rows per batch element
        fcfg = _FUSED_CFG.get(mreal)
        if fcfg is not None:
            BM, BK, nw, ns = fcfg
            spat = BM // time
            nblk = -(-npos // spat)
            _qkv_attn_kernel[(bsz * nblk * h,)](
                xf, wq, att, cos, sin,
                time, npos, spat, nblk, bnp, dim, dh ** -0.5,
                H=h, DH=dh, ND=nd, BM=BM, BK=BK, IS_CAUSAL=self.is_causal,
                num_warps=nw, num_stages=ns,
            )
        else:
            n_qkv = 3 * nd
            BM, BN, BK, GM, nw, ns, ws = _QKV_CFG.get(mreal, _QKV_CFG_DEFAULT)
            _qkv_kernel[(triton.cdiv(m, BM) * triton.cdiv(n_qkv, BN),)](
                xf, wq, qkv, m, n_qkv, dim,
                xf.stride(0), wq.stride(0), qkv.stride(0),
                BM=BM, BN=BN, BK=BK, GM=GM, WS=ws, num_warps=nw, num_stages=ns,
            )
            pk = _ATTN_PK
            nblk = -(-npos // pk)
            _attn_kernel[(bsz * nblk * h,)](
                qkv, att, cos, sin,
                time, npos, bnp, nblk, qkv.stride(0), att.stride(0), dh ** -0.5,
                H=h, DH=dh, ND=nd, BT=_BT_MAX, PK=pk, IS_CAUSAL=self.is_causal,
                num_warps=1, num_stages=1,
            )

        BM, BN, BK, GM, nw, ns, ws = _OUT_CFG.get(mreal, _OUT_CFG_DEFAULT)
        _out_proj_kernel[(triton.cdiv(m, BM) * triton.cdiv(dim, BN),)](
            att, wo, bo, out, m, dim, nd,
            att.stride(0), wo.stride(0), out.stride(0),
            BM=BM, BN=BN, BK=BK, GM=GM, WS=ws, num_warps=nw, num_stages=ns,
        )

    def _make_plan(self, x: torch.Tensor) -> _Plan:
        bsz, time, height, width, _ = x.shape
        nd = self.heads * self.dim_head
        m = bsz * time * height * width
        # Pad the token axis out to a whole number of GEMM row-tiles.  A masked
        # partial tile costs the same MMA work as a full one, so this is free --
        # it just lets both GEMMs drop their M mask.
        bm = _OUT_CFG.get(m, _OUT_CFG_DEFAULT)[0]
        if m in _FUSED_CFG:
            bm = max(bm, _FUSED_CFG[m][0])
        else:
            bm = max(bm, _QKV_CFG.get(m, _QKV_CFG_DEFAULT)[0])
        mpad = -(-m // bm) * bm
        dev, dt = x.device, x.dtype
        xb = torch.zeros((mpad, self.dim), dtype=dt, device=dev)
        ob = torch.zeros((mpad, self.dim), dtype=dt, device=dev)
        plan = _Plan(
            xb,
            None if m in _FUSED_CFG else torch.zeros((mpad, 3 * nd), dtype=dt, device=dev),
            torch.zeros((mpad, nd), dtype=dt, device=dev),
            ob,
            xb[:m].view(bsz, time, height, width, self.dim),
            ob[:m].view(bsz, time, height, width, self.dim),
            tuple(x.shape),
            mpad,
        )
        plan.xv.copy_(x)
        if self._use_graph:
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._launch(plan)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._launch(plan)
                plan.graph = g
            except Exception:  # noqa: BLE001 - fall back to plain launches
                plan.graph = None
                self._use_graph = False
        return plan

    def _fast_forward(self, x: torch.Tensor) -> torch.Tensor:
        key = tuple(x.shape)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._make_plan(x)
            self._plans[key] = plan
        else:
            plan.xv.copy_(x)
        if plan.graph is not None:
            plan.graph.replay()
        else:
            self._launch(plan)
        return plan.outv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not _HAS_TRITON
            or not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.dim() != 5
            or x.shape[-1] != self.dim
            or x.shape[1] > _BT_MAX
            or self.dim_head % 2
            or self.to_qkv.weight.dtype != x.dtype
        ):
            return self._ref_forward(x)
        return self._fast_forward(x)
