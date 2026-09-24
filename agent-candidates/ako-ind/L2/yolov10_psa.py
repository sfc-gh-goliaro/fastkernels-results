"""YOLOv10 PSA (Partial Self-Attention) block.

The captured workloads are tiny (fp16, C=256, 20x20 spatial, B in {1,4}): ~1 GFLOP
of real math spread over ~30 eager launches, which makes the reference module
purely CPU-launch-bound (measured: 425us of Python/cuDNN dispatch per forward vs
425us wall).  This implementation collapses the block into **three** Triton
kernels -- one per global barrier the data flow actually requires:

  stage 1  cv1 (1x1 conv + folded BN + SiLU) -> split(a, b) -> attn.qkv, plus
           cv2's ``a`` contribution hoisted forward
  stage 2  flash-style attention (fp32 online softmax, never materializing the
           400x400 score matrix) and the depthwise 3x3 positional encoding on v
  stage 3  attn.proj + residual + ffn + residual + cv2's ``b`` contribution

Only the token-mixing attention needs all tokens, so stages 1 and 3 are pure
per-token GEMM chains; everything else (BatchNorm, SiLU, the splits, the two
residual adds and the concat) is folded into GEMM epilogues and never becomes a
kernel of its own.

**The cost model.**  Under the benchmark's timing loop (a 264 MB L2 memset before
every iteration, so the CPU runs far ahead and the event pair measures pure GPU
time) everything on this operator costs an integer number of ~2.05us quanta:

    total ~= 3.15us + 2.05us x (#kernels + #dependent global round trips)

Fitting that to the parent (3 kernels, 4 serialized round trips each) gives
3.15 + 2.05x15 = 33.9us against 33.8us measured.  A quantum is one dependent
global-memory round trip -- ~4000 cycles, inflated well past normal HBM latency
by the memset's writeback -- and a kernel's weight tile for GEMM level k+1 cannot
be staged into shared memory until level k has finished reading it, so a
d-deep dependent GEMM chain costs d quanta no matter how it is tiled.  (Verified
directly: with the inter-level data dependency removed but the same tiles and the
same bytes, stage 3 drops from 15.3us to 9.2us.)

Round trips, not FLOPs, occupancy or bytes, are therefore what this code
minimises.  Three measured levers do that:

* **Job-axis packing.**  Independent CTAs added to an *existing* launch are free
  -- doubling a launch's CTA count costs nothing, while running the same work as
  a second launch costs a full quantum -- so mutually independent parts of the
  DAG become extra grid-z jobs instead of extra kernels.  cv1's ``a`` and ``b``
  halves and (in stage 2) the attention and the depthwise pe are split this way.
  The budget is *total work*, not CTA count: at B=4 a job that recomputes a cv1
  half costs a quantum, so nothing is recomputed.
* **Hoisting whole GEMMs off the longest chain.**  ``ya = a @ Wc2[:C]`` is cv2's
  ``a`` contribution; it only needs cv1, so stage 1 computes it (in fp32) and
  stage 3's chain is four dependent GEMMs instead of five.
* **Pipelined reduction loops.**  ``tl.range(..., num_stages=n)`` multi-buffers
  the k-th and k+1-th weight tiles, which is the only way found to overlap
  otherwise-serialized dependent loads; stage 1 uses BK < C1 purely to have
  something to pipeline.

Layouts are picked per consumer, because the cost is memory *transactions*:
cv1's output, ``v``, the attention output and the pe output are token-major so a
[BM, C] tile has 256 B contiguous per row, while ``q``/``k`` stay channel-major
because the score GEMM wants a [KD, BJ] tile with the key index contiguous.
``x`` and the output are NCHW by contract, so only those stay strided.  Tile
shapes are chosen per token count, and stages 2/3 use PDL so each overlaps its
predecessor's drain.

Weights are BN-folded, relayouted and packed once into a single fp16 blob plus a
single fp32 scale/bias blob, cached on the module and invalidated on
``load_state_dict`` or ``fuse()``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


from .yolov10_attention import YOLOAttention
from .yolov10_conv import YOLOConv


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
@triton.jit
def _stage1(X, WB, SB, WS, WS32, OQK, OV,
            C1: tl.constexpr, C: tl.constexpr, NH: tl.constexpr,
            KD: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr, NSK: tl.constexpr):
    """cv1 + one [C,C] follow-on GEMM, replicated over a *job* grid axis.

    Independent CTAs in the same launch are free (measured: doubling the CTA
    count via a job axis costs nothing, while the same work as a second launch
    costs a full launch quantum).  So instead of one CTA doing
    cv1(a) + cv1(b) + qkv, four jobs each do one cv1 half plus one 128-wide
    follow-on GEMM, recomputing the cv1 half rather than sharing it:

      job 0  b = silu(cv1_b(x)) -> store b ; qk = b @ Wqk
      job 1  b                              ; v  = b @ Wv
      job 2  a = silu(cv1_a(x))             ; ya = a @ Wc2[:C, :128]
      job 3  a                              ; ya = a @ Wc2[:C, 128:]

    ``ya`` is cv2's contribution from ``a``; hoisting it here takes a whole GEMM
    off stage 3's dependent chain, and ``a`` itself never has to be stored.
    """
    QKC: tl.constexpr = 2 * NH * KD
    OWQK: tl.constexpr = C1 * (2 * C)
    OWV: tl.constexpr = OWQK + C * QKC
    OWP: tl.constexpr = OWV + C * C
    OWPE: tl.constexpr = OWP + C * C
    OWF1: tl.constexpr = OWPE + 9 * C
    OWF2: tl.constexpr = OWF1 + C * (2 * C)
    OWC2: tl.constexpr = OWF2 + (2 * C) * C
    OS1: tl.constexpr = 0
    OB1: tl.constexpr = 2 * C
    OSQK: tl.constexpr = 4 * C
    OBQK: tl.constexpr = OSQK + QKC
    OSV: tl.constexpr = OBQK + QKC
    OBV: tl.constexpr = OSV + C

    pm = tl.program_id(0)
    pb = tl.program_id(1)
    job = tl.program_id(2)
    m = pm * BM + tl.arange(0, BM)
    mm = m < N
    dt = X.dtype.element_ty
    co = tl.arange(0, C)

    # cv1 half: jobs 0/1 take b (output cols C..2C-1), jobs 2/3 take a (0..C-1)
    hi = job < 2
    cofs = tl.where(hi, C, 0)
    xb = X + pb * (C1 * N)
    acc = tl.zeros((BM, C), tl.float32)
    for k0 in tl.range(0, C1, BK, num_stages=NSK):
        kk = k0 + tl.arange(0, BK)
        xt = tl.load(xb + kk[None, :] * N + m[:, None], mask=mm[:, None], other=0.0)
        acc = tl.dot(xt, tl.load(WB + kk[:, None] * (2 * C) + (cofs + co)[None, :]), acc)
    h = acc * tl.load(SB + OS1 + cofs + co)[None, :] + tl.load(SB + OB1 + cofs + co)[None, :]
    h = h * tl.sigmoid(h)
    hh = h.to(dt)
    if job == 0:
        tl.store(WS + (pb * N + m)[:, None] * C + co[None, :], hh, mask=mm[:, None])

    # follow-on GEMM: [C, C] for every job, only the weight block differs
    wb = tl.where(job == 0, OWQK, tl.where(job == 1, OWV,
                  tl.where(job == 2, OWC2, OWC2 + C)))
    wstride = tl.where(job < 2, C, 2 * C)
    o2 = tl.dot(hh, tl.load(WB + wb + co[:, None] * wstride + co[None, :]))
    if job == 0:
        o2 = o2 * tl.load(SB + OSQK + co)[None, :] + tl.load(SB + OBQK + co)[None, :]
        tl.store(WS + OQK + pb * (QKC * N) + co[None, :] * N + m[:, None],
                 o2.to(dt), mask=mm[:, None])
    elif job == 1:
        o2 = o2 * tl.load(SB + OSV + co)[None, :] + tl.load(SB + OBV + co)[None, :]
        tl.store(WS + OV + (pb * N + m)[:, None] * C + co[None, :],
                 o2.to(dt), mask=mm[:, None])
    else:
        tl.store(WS32 + (pb * N + m)[:, None] * C1 + (job - 2) * C + co[None, :], o2,
                 mask=mm[:, None])
    gdc_launch_dependents()


@triton.jit
def _stage2(WS, OQK, OV, OXA, OPE, WB, SB,
            C: tl.constexpr, C1: tl.constexpr, NH: tl.constexpr,
            KD: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
            SCALE: tl.constexpr, BM: tl.constexpr, BJ: tl.constexpr,
            NSJ: tl.constexpr):
    """Flash attention and the depthwise pe as two *independent jobs*.

    Both read only stage 1's output, so they are mutually independent and can be
    separate CTAs of the same launch; the launch then costs max(attn, pe) instead
    of attn + pe (measured: the fused form costs one extra quantum at B=4).
    The head is also a grid axis, so a CTA carries a single [BM, HD] tile.
    """
    N: tl.constexpr = H * W
    HD: tl.constexpr = C // NH
    QKC: tl.constexpr = 2 * NH * KD
    OWQK: tl.constexpr = C1 * (2 * C)
    OWV: tl.constexpr = OWQK + C * QKC
    OWP: tl.constexpr = OWV + C * C
    OWPE: tl.constexpr = OWP + C * C
    OSPE: tl.constexpr = 4 * C + 2 * QKC + 2 * C + 2 * C
    OBPE: tl.constexpr = OSPE + C

    pm = tl.program_id(0)
    pb = tl.program_id(1)
    hj = tl.program_id(2)
    h = hj % NH
    m = pm * BM + tl.arange(0, BM)
    mm = m < N
    dt = WS.dtype.element_ty
    dk = tl.arange(0, KD)
    dv = tl.arange(0, HD)

    qp = WS + OQK + pb * (QKC * N) + (h * 2 * KD) * N
    vh = WS + OV + pb * (N * C) + h * HD
    orow = (pb * N + m)[:, None] * C + (h * HD + dv)[None, :]

    if hj < NH:
        gdc_wait()
        q = tl.load(qp + dk[None, :] * N + m[:, None], mask=mm[:, None], other=0.0)
        acco = tl.zeros((BM, HD), tl.float32)
        mi = tl.full((BM,), -float("inf"), tl.float32)
        li = tl.zeros((BM,), tl.float32)
        for j0 in tl.range(0, N, BJ, num_stages=NSJ):
            j = j0 + tl.arange(0, BJ)
            jm = j < N
            k = tl.load(qp + KD * N + dk[:, None] * N + j[None, :], mask=jm[None, :], other=0.0)
            s = tl.dot(q, k) * SCALE
            s = tl.where(jm[None, :], s, -float("inf"))
            mn = tl.maximum(mi, tl.max(s, 1))
            p = tl.exp(s - mn[:, None])
            al = tl.exp(mi - mn)
            li = li * al + tl.sum(p, 1)
            v = tl.load(vh + j[:, None] * C + dv[None, :], mask=jm[:, None], other=0.0)
            acco = acco * al[:, None] + tl.dot(p.to(dt), v)
            mi = mn
        tl.store(WS + OXA + orow, (acco / li[:, None]).to(dt), mask=mm[:, None])
    else:
        # pe weights/scales come from the packed blob, so they can be fetched
        # while stage 1 is still draining (PDL overlap window).
        spe = tl.load(SB + OSPE + h * HD + dv)[None, :]
        bpe = tl.load(SB + OBPE + h * HD + dv)[None, :]
        wt0 = tl.load(WB + OWPE + 0 * C + h * HD + dv)[None, :].to(tl.float32)
        gdc_wait()
        yq = m // W
        xq = m % W
        p0 = tl.zeros((BM, HD), tl.float32)
        p1 = tl.zeros((BM, HD), tl.float32)
        p2 = tl.zeros((BM, HD), tl.float32)
        for t in tl.static_range(9):
            dy = t // 3 - 1
            dx = t % 3 - 1
            ok = (yq + dy >= 0) & (yq + dy < H) & (xq + dx >= 0) & (xq + dx < W) & mm
            vv = tl.load(vh + (m + dy * W + dx)[:, None] * C + dv[None, :],
                         mask=ok[:, None], other=0.0)
            wt = wt0 if t == 0 else tl.load(
                WB + OWPE + t * C + h * HD + dv)[None, :].to(tl.float32)
            if t % 3 == 0:
                p0 += vv.to(tl.float32) * wt
            elif t % 3 == 1:
                p1 += vv.to(tl.float32) * wt
            else:
                p2 += vv.to(tl.float32) * wt
        pe = ((p0 + p1) + p2) * spe + bpe
        tl.store(WS + OPE + orow, pe.to(dt), mask=mm[:, None])
    gdc_launch_dependents()


@triton.jit
def _stage3(WS, OXA, OPE, Y, WB, SB, WS32,
            C1: tl.constexpr, C: tl.constexpr, NH: tl.constexpr, KD: tl.constexpr,
            N: tl.constexpr, BM: tl.constexpr):
    """attn.proj + residual + ffn + residual + cv2 (b half only).

    cv2's ``a`` half arrives as the fp32 partial ``ya`` from stage 1, so the
    dependent chain here is four GEMMs instead of five.
    """
    QKC: tl.constexpr = 2 * NH * KD
    OWQK: tl.constexpr = C1 * (2 * C)
    OWV: tl.constexpr = OWQK + C * QKC
    OWP: tl.constexpr = OWV + C * C
    OWPE: tl.constexpr = OWP + C * C
    OWF1: tl.constexpr = OWPE + 9 * C
    OWF2: tl.constexpr = OWF1 + C * (2 * C)
    OWC2: tl.constexpr = OWF2 + (2 * C) * C
    OSP: tl.constexpr = 4 * C + 2 * QKC + 2 * C
    OBP: tl.constexpr = OSP + C
    OSF1: tl.constexpr = OBP + C + 2 * C
    OBF1: tl.constexpr = OSF1 + 2 * C
    OSF2: tl.constexpr = OBF1 + 2 * C
    OBF2: tl.constexpr = OSF2 + C
    OSC2: tl.constexpr = OBF2 + C
    OBC2: tl.constexpr = OSC2 + C1

    pm = tl.program_id(0)
    pb = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM)
    mm = m < N
    dt = Y.dtype.element_ty
    co = tl.arange(0, C)
    o2 = tl.arange(0, 2 * C)
    oc = tl.arange(0, C1)

    wp = tl.load(WB + OWP + co[:, None] * C + co[None, :])
    gdc_wait()
    row = (pb * N + m)[:, None]
    bres = tl.load(WS + row * C + co[None, :], mask=mm[:, None], other=0.0)
    xa = (tl.load(WS + OXA + row * C + co[None, :], mask=mm[:, None], other=0.0)
          + tl.load(WS + OPE + row * C + co[None, :], mask=mm[:, None], other=0.0))
    ya = tl.load(WS32 + row * C1 + oc[None, :], mask=mm[:, None], other=0.0)

    accp = tl.dot(xa, wp)
    b1 = bres.to(tl.float32) + (accp * tl.load(SB + OSP + co)[None, :]
                                + tl.load(SB + OBP + co)[None, :])
    f1 = tl.dot(b1.to(dt), tl.load(WB + OWF1 + co[:, None] * (2 * C) + o2[None, :]))
    f1 = f1 * tl.load(SB + OSF1 + o2)[None, :] + tl.load(SB + OBF1 + o2)[None, :]
    f1 = f1 * tl.sigmoid(f1)
    f2 = tl.dot(f1.to(dt), tl.load(WB + OWF2 + o2[:, None] * C + co[None, :]))
    b2 = b1 + (f2 * tl.load(SB + OSF2 + co)[None, :] + tl.load(SB + OBF2 + co)[None, :])
    accy = tl.dot(b2.to(dt), tl.load(WB + OWC2 + (C + co)[:, None] * C1 + oc[None, :]), ya)
    y = accy * tl.load(SB + OSC2 + oc)[None, :] + tl.load(SB + OBC2 + oc)[None, :]
    y = y * tl.sigmoid(y)
    tl.store(Y + pb * (C1 * N) + oc[None, :] * N + m[:, None], y.to(dt), mask=mm[:, None])


# ---------------------------------------------------------------------------
# Host side: BN folding + weight packing (done once, cached)
# ---------------------------------------------------------------------------
def _fold_bn(mod: YOLOConv):
    """(weight, scale[f32], bias[f32]) with any eval-mode BatchNorm folded in."""
    w = mod.conv.weight.detach()
    cout = w.shape[0]
    f32 = dict(dtype=torch.float32, device=w.device)
    if getattr(mod, "_is_fused", False) or not hasattr(mod, "bn"):
        s = torch.ones(cout, **f32)
        beta = (mod.conv.bias.detach().float() if mod.conv.bias is not None
                else torch.zeros(cout, **f32))
    else:
        bn = mod.bn
        s = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
        beta = bn.bias.detach().float() - bn.running_mean.detach().float() * s
        if mod.conv.bias is not None:
            beta = beta + mod.conv.bias.detach().float() * s
    return w, s, beta


class _Packed:
    __slots__ = ("key", "WB", "SB", "cfg", "ws", "ws32", "meta")


class YOLOPSA(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )
        self._c1 = c1
        self._pk: _Packed | None = None
        self.register_load_state_dict_post_hook(lambda *a, **k: setattr(self, "_pk", None))

    # -- reference (training / CPU) -----------------------------------------
    def _forward_ref(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

    # -- packing ------------------------------------------------------------
    @torch.no_grad()
    def _pack(self, x: torch.Tensor) -> _Packed:
        C1 = self._c1
        C = self.c
        at = self.attn
        NH, KD, HD = at.num_heads, at.key_dim, at.head_dim
        QKC = 2 * NH * KD
        H, W = int(x.shape[2]), int(x.shape[3])
        dt = x.dtype
        dev = x.device

        w1, s1, b1 = _fold_bn(self.cv1)
        wq, sq, bq = _fold_bn(at.qkv)
        wp, sp, bp = _fold_bn(at.proj)
        wpe, spe, bpe = _fold_bn(at.pe)
        wf1, sf1, bf1 = _fold_bn(self.ffn[0])
        wf2, sf2, bf2 = _fold_bn(self.ffn[1])
        wc2, sc2, bc2 = _fold_bn(self.cv2)

        # The baseline views qkv's output as [B, NH, 2*KD+HD, N] and splits
        # (q, k, v).  Relayout the output channels into [q|k per head] followed
        # by [v per head] so stage 2 reads contiguous runs, and split it into two
        # weight blocks (q/k is stored channel-major, v token-major).
        blk = 2 * KD + HD
        pqk, pv = [], []
        for h in range(NH):
            for d in range(KD):
                pqk.append(h * blk + d)
            for d in range(KD):
                pqk.append(h * blk + KD + d)
        for h in range(NH):
            for d in range(HD):
                pv.append(h * blk + 2 * KD + d)
        pqk = torch.as_tensor(pqk, dtype=torch.long, device=dev)
        pv = torch.as_tensor(pv, dtype=torch.long, device=dev)

        def t1(w):  # [out, in, 1, 1] -> [in, out]
            return w.reshape(w.shape[0], w.shape[1]).t().contiguous().to(dt)

        WB = torch.cat([
            t1(w1).reshape(-1),
            t1(wq.index_select(0, pqk)).reshape(-1),
            t1(wq.index_select(0, pv)).reshape(-1),
            t1(wp).reshape(-1),
            wpe.reshape(C, 9).t().contiguous().reshape(-1).to(dt),
            t1(wf1).reshape(-1),
            t1(wf2).reshape(-1),
            t1(wc2).reshape(-1),
        ])
        SB = torch.cat([
            s1, b1,
            sq.index_select(0, pqk), bq.index_select(0, pqk),
            sq.index_select(0, pv), bq.index_select(0, pv),
            sp, bp, spe, bpe, sf1, bf1, sf2, bf2, sc2, bc2,
        ])

        pk = _Packed()
        pk.key = (H, W, dt, self.cv1._is_fused)
        pk.WB = WB
        pk.SB = SB
        pk.ws = None
        pk.ws32 = None
        pk.meta = (C1, C, NH, KD, H, W, float(at.scale), H * W, QKC)
        pk.cfg = {}
        self._pk = pk
        return pk

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or not x.is_cuda:
            return self._forward_ref(x)
        pk = self._pk
        if pk is None or pk.key != (x.shape[2], x.shape[3], x.dtype, self.cv1._is_fused):
            pk = self._pack(x)
        C1, C, NH, KD, H, W, scale, N, QKC = pk.meta
        B = x.shape[0]
        cfg = pk.cfg.get(B)
        if cfg is None:
            cfg = pk.cfg[B] = _launch_cfg(B, N)
        (g1, BM1, BK1, NSK1, NW1, NS1), (g2, BM2, BJ2, NSJ2, NW2, NS2), (g3, BM3, NW3, NS3) = cfg
        if not x.is_contiguous():
            x = x.contiguous()

        oqk = B * N * C
        ov = oqk + B * QKC * N
        oxa = ov + B * N * C
        ope = oxa + B * N * C
        need = ope + B * N * C
        ws = pk.ws
        if ws is None or ws.numel() < need:
            ws = torch.empty(need, dtype=x.dtype, device=x.device)
            pk.ws = ws
        ws32 = pk.ws32
        if ws32 is None or ws32.numel() < B * N * C1:
            ws32 = torch.empty(B * N * C1, dtype=torch.float32, device=x.device)
            pk.ws32 = ws32
        y = torch.empty_like(x)

        # NB: stage 1 is deliberately launched *without* launch_pdl.  The
        # attribute lets a kernel begin before its predecessor on the stream has
        # finished, and ordering is then the kernel's own responsibility via
        # gdc_wait(); stage 1 has no gdc_wait (its input is x, not a predecessor's
        # output), so allowing it to start early would race with whatever wrote x.
        # stages 2 and 3 do wait, so they may overlap the previous stage's drain.
        _stage1[(g1, B, 4)](x, pk.WB, pk.SB, ws, ws32, oqk, ov,
                            C1, C, NH, KD, N, BM1, BK1, NSK1, num_warps=NW1, num_stages=NS1)
        _stage2[(g2, B, 2 * NH)](ws, oqk, ov, oxa, ope, pk.WB, pk.SB,
                                 C, C1, NH, KD, H, W, scale, BM2, BJ2, NSJ2,
                                 num_warps=NW2, num_stages=NS2, launch_pdl=True)
        _stage3[(g3, B)](ws, oxa, ope, y, pk.WB, pk.SB, ws32,
                         C1, C, NH, KD, N, BM3, num_warps=NW3, num_stages=NS3,
                         launch_pdl=True)
        return y


# Tile configs, swept on B200 for the two captured token counts.  These kernels
# are latency-bound on dependent global loads, not throughput-bound, so the tile
# choice is really a choice of how much redundant per-CTA work the launch can
# afford before it costs another quantum -- which makes it a function of B.  The
# reduction-loop num_stages entries are load-pipelining depth (BK < C1 exists
# only so there is something to pipeline), not the usual occupancy knob.  Values:
#   stage1 (BM, BK, k-loop num_stages, num_warps, num_stages)
#   stage2 (BM, BJ, j-loop num_stages, num_warps, num_stages)
#   stage3 (BM, num_warps, num_stages)
_CFG_BIG = ((16, 64, 3, 4, 1), (32, 512, 1, 4, 1), (16, 16, 1))
_CFG_SMALL = ((16, 64, 4, 8, 1), (8, 512, 1, 8, 1), (8, 16, 1))


def _launch_cfg(B: int, N: int):
    """Resolve the tile config for this batch and bake in the three grids."""
    c1, c2, c3 = _CFG_BIG if B * N >= 800 else _CFG_SMALL
    return ((triton.cdiv(N, c1[0]),) + c1,
            (triton.cdiv(N, c2[0]),) + c2,
            (triton.cdiv(N, c3[0]),) + c3)
