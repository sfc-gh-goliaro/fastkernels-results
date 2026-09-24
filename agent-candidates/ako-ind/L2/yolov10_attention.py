"""YOLOv10 spatial attention block -- three fused Triton kernels.

The captured init is ``YOLOAttention(dim=128, num_heads=2, attn_ratio=0.5)``, so
the geometry is key_dim=32, head_dim=64, n=20*20=400, fp16, b in {1, 4}.  The
whole block is under 0.5 GFLOP and the eager baseline is launch-bound (~0.17 ms,
identical at b=1 and b=4), so everything here is about latency and parallelism,
not arithmetic.

  1. ``_qkv_kernel``       -- 1x1 qkv conv, BatchNorm folded into a pre-transposed
     weight.  Emits the result twice, pixel-major ``[b, n, 256]`` and
     channel-major ``[b, 256, n]``, so kernel 2 gets every operand in the layout
     its ``tl.dot`` wants with no in-kernel transpose.
  2. ``_attn_part_kernel`` -- one CTA per (batch, query-tile, head, key-split).
     Flash-style attention over its slice of the key axis, with the 3x3 depthwise
     ``pe`` conv folded in as 9 constant-shift loads of the same v rows (a flat
     query tile shifts by dr*W+dc, so every tap is a contiguous load and only the
     border mask varies).  Stores the *normalised* partial and its (m, l).
  3. ``_combine_proj_kernel`` -- combines the key-splits and applies the 1x1
     output projection, back to NCHW.

Why the key axis is split across CTAs, and why the combine is free
-----------------------------------------------------------------
This operator is parallelism-starved, not overhead-bound: with the key axis
unsplit, the attention kernel takes 8.2 us at b=4 (200 CTAs) and 6.2 us at b=1
(50 CTAs) -- 4x the work for 1.3x the time, i.e. almost pure per-CTA critical
path with most of the machine idle.  Splitting the key axis shortens that path
and adds CTAs.  Splitting normally costs an extra reduction stage, but here it
does not:

* each split stores ``avbar_i = (sum_j p_ij v_j) / l_i`` rather than the raw
  numerator.  ``avbar`` is bounded by max|v|, so it stays in fp16 and each split's
  buffer is the same size ``u`` used to be;
* the combine is a weighted average with ``w_i = l_i * exp(m_i - max_i m_i)``, and
  a weighted average is affine, so adding ``pe`` into *every* split's ``avbar``
  reproduces pe exactly once:
  ``sum_i w_i (avbar_i + pe) / sum_i w_i == sum_i w_i avbar_i / sum_i w_i + pe``.
  So pe needs no separate pass and no special-casing of one split;
* the combine then folds into the projection kernel, which already had to read a
  [pixels, C] tile -- so the structure stays at three launches.

All accumulation is fp32.  Max abs error vs the baseline is ~3e-05 (tolerance is
atol=rtol=1e-2 at 99% match).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:  # Blackwell/Hopper programmatic dependent launch (Triton 3.5+)
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _PDL = 1
except ImportError:  # pragma: no cover - no PDL intrinsics available
    _PDL = 0

# Tuned on B200 under the harness's L2-flush regime (see ITERATIONS.md).  Two
# facts drive these numbers.  (a) head_dim is 64, so the PV dot is
# [TQ,BK]x[BK,64] and extra warps *can* split its N axis: num_warps=4 on the
# attention kernel is worth ~4 us over num_warps=2.  (b) Measured GPU time is
# billed in ~2.05 us quanta, so only whole-quantum savings are visible and the
# tile sweeps below are flat plateaus rather than smooth optima.
_PT1, _BO1, _NW1 = 16, 64, 8       # qkv projection
_TQ, _KS, _NW2 = 16, 2, 4          # attention: query tile, key splits, warps
_PT3, _BO3, _NW3 = 16, 64, 8       # combine + output projection

# Programmatic dependent launch.  Every stage has a grid far below one wave per
# SM, so each launch's CTA-dispatch ramp would be pure exposed latency; PDL lets
# the consumer's CTAs become resident while the producer drains.  Each consumer
# waits (gdc_wait) only just before its first load of producer-written data, and
# each producer triggers (gdc_launch_dependents) only after its last store.
# Worth ~7 us of the ~20 us total at b=4 and ~5 us at b=1.


@triton.jit
def _qkv_kernel(X, WT, BQ, PX, CH, n_pix, NT, NB,
                C: tl.constexpr, OC: tl.constexpr, PT: tl.constexpr,
                BO: tl.constexpr, PDL: tl.constexpr):
    """qkv[b, p, oc] = sum_ic WT[ic, oc] * x[b, ic, p] + BQ[oc]; emitted twice."""
    pid = tl.program_id(0)
    ob = pid % NB
    rest = pid // NB
    bi = rest // NT
    p = (rest % NT) * PT + tl.arange(0, PT)
    pm = p < n_pix
    ic = tl.arange(0, C)
    oc = ob * BO + tl.arange(0, BO)
    if PDL:  # x is written by the caller's copy; wait before the first read
        gdc_wait()
    x = tl.load(X + bi * C * n_pix + ic[:, None] * n_pix + p[None, :],
                mask=pm[None, :], other=0.0)
    acc = tl.dot(tl.trans(x), tl.load(WT + ic[:, None] * OC + oc[None, :]))
    acc += tl.load(BQ + oc)[None, :]
    h = acc.to(tl.float16)
    tl.store(PX + bi * n_pix * OC + p[:, None] * OC + oc[None, :], h, mask=pm[:, None])
    tl.store(CH + bi * OC * n_pix + oc[:, None] * n_pix + p[None, :],
             tl.trans(h), mask=pm[None, :])
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _attn_part_kernel(PX, CH, PEW, PEB, AVB, ML, scale, n_pix, NT, GH, GW,
                      KD: tl.constexpr, KDP: tl.constexpr, HD: tl.constexpr,
                      C: tl.constexpr, OC: tl.constexpr, NH: tl.constexpr,
                      TQ: tl.constexpr, BKS: tl.constexpr, KS: tl.constexpr,
                      PDL: tl.constexpr):
    """One key-split of softmax(q'k * scale) v + depthwise3x3(v) + pe_bias.

    Writes the split's normalised partial and its (m, l) so the projection kernel
    can combine the splits with a weighted average.
    """
    pid = tl.program_id(0)
    ks = pid % KS
    rest = pid // KS
    hh = rest % NH
    rest = rest // NH
    bi = rest // NT
    qi = (rest % NT) * TQ + tl.arange(0, TQ)
    qm = qi < n_pix
    row = qi // GW
    col = qi - row * GW
    e = tl.arange(0, KDP)      # key axis, zero-padded from KD up to a power of 2
    ev = e < KD
    d = tl.arange(0, HD)
    off = hh * (2 * KD + HD)
    cb = hh * HD
    pxb = PX + bi * n_pix * OC
    vcol = (off + 2 * KD + d)[None, :]
    krow = CH + bi * OC * n_pix + (off + KD + e)[:, None] * n_pix
    if PDL:  # everything above is index arithmetic only
        gdc_wait()
    q = tl.load(pxb + qi[:, None] * OC + (off + e)[None, :],
                mask=qm[:, None] & ev[None, :], other=0.0)
    j = ks * BKS + tl.arange(0, BKS)
    jm = j < n_pix
    k = tl.load(krow + j[None, :], mask=ev[:, None] & jm[None, :], other=0.0)
    s = tl.dot(q, k) * scale
    s = tl.where(jm[None, :], s, -float("inf"))
    m_i = tl.max(s, 1)
    pr = tl.exp(s - m_i[:, None])
    l_i = tl.sum(pr, 1)
    v = tl.load(pxb + j[:, None] * OC + vcol, mask=jm[:, None], other=0.0)
    av = tl.dot(pr.to(tl.float16), v)
    # A split whose whole key range is padding contributes nothing; keep its
    # divisor finite here and give it zero weight in the combine.
    live = l_i > 0.0
    u = av / tl.where(live, l_i, 1.0)[:, None]

    # pe goes into every split: the combine is an affine weighted average, so it
    # lands in the result exactly once (see the module docstring).
    for t in tl.static_range(9):
        dr = t // 3 - 1
        dc = t % 3 - 1
        ok = ((row + dr >= 0) & (row + dr < GH) & (col + dc >= 0)
              & (col + dc < GW) & qm)
        nb = tl.maximum(tl.minimum(qi + dr * GW + dc, n_pix - 1), 0)
        vt = tl.load(pxb + nb[:, None] * OC + vcol, mask=ok[:, None], other=0.0)
        u += vt.to(tl.float32) * tl.load(PEW + t * C + cb + d)[None, :]
    u += tl.load(PEB + cb + d)[None, :]

    base = (bi * KS + ks) * n_pix
    tl.store(AVB + base * C + qi[:, None] * C + (cb + d)[None, :],
             u.to(tl.float16), mask=qm[:, None])
    tl.store(ML + (base + qi) * (2 * NH) + 2 * hh,
             tl.where(live, m_i, -float("inf")), mask=qm)
    tl.store(ML + (base + qi) * (2 * NH) + 2 * hh + 1, l_i, mask=qm)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _combine_proj_kernel(AVB, ML, WT, BP, OUT, n_pix, NT, NB,
                         C: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
                         KS: tl.constexpr, PT: tl.constexpr, BO: tl.constexpr,
                         PDL: tl.constexpr):
    """u = sum_i w_i avbar_i / sum_i w_i (per head), then out = proj(u), NCHW."""
    pid = tl.program_id(0)
    ob = pid % NB
    rest = pid // NB
    bi = rest // NT
    p = (rest % NT) * PT + tl.arange(0, PT)
    pm = p < n_pix
    c = tl.arange(0, C)
    hidx = c // HD             # channel c belongs to head c // HD
    oc = ob * BO + tl.arange(0, BO)
    w = tl.load(WT + c[:, None] * C + oc[None, :])          # not producer-written
    bias = tl.load(BP + oc)[None, :]
    if PDL:
        gdc_wait()
    # Every intermediate is [PT, C]: a per-head scalar becomes a channel tile by
    # a masked select over the static head loop.  Doing this with a 3-D
    # where/sum instead costs a full 2.05 us quantum.
    ninf = -float("inf")
    mx = tl.full([PT, C], ninf, tl.float32)
    for i in tl.static_range(KS):
        r = (bi * KS + i) * n_pix + p
        for hh in tl.static_range(NH):
            mi = tl.load(ML + r * (2 * NH) + 2 * hh, mask=pm, other=ninf)
            mx = tl.maximum(mx, tl.where(hidx == hh, mi[:, None], ninf))
    num = tl.zeros([PT, C], tl.float32)
    den = tl.zeros([PT, C], tl.float32)
    for i in tl.static_range(KS):
        base = (bi * KS + i) * n_pix
        r = base + p
        wc = tl.zeros([PT, C], tl.float32)
        for hh in tl.static_range(NH):
            mi = tl.load(ML + r * (2 * NH) + 2 * hh, mask=pm, other=ninf)
            li = tl.load(ML + r * (2 * NH) + 2 * hh + 1, mask=pm, other=0.0)
            # mx already holds this head's max broadcast over its own channels,
            # so no reduction back to [PT] is needed.
            wc += tl.where(hidx == hh, li[:, None] * tl.exp(mi[:, None] - mx), 0.0)
        den += wc
        av = tl.load(AVB + base * C + p[:, None] * C + c[None, :],
                     mask=pm[:, None], other=0.0).to(tl.float32)
        num += av * wc
    u = num / den
    acc = tl.dot(u.to(tl.float16), w) + bias
    tl.store(OUT + bi * C * n_pix + oc[:, None] * n_pix + p[None, :],
             tl.trans(acc).to(tl.float16), mask=pm[None, :])


@torch.no_grad()
def _fuse_bn(conv, bn) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold an eval-mode BatchNorm into its conv weight/bias (fp32 math)."""
    w = conv.weight.detach().float()
    b0 = (conv.bias.detach().float() if conv.bias is not None
          else w.new_zeros(w.shape[0]))
    if bn is None:
        return w, b0
    scale = bn.weight.detach().float() / torch.sqrt(
        bn.running_var.detach().float() + bn.eps)
    fw = w * scale.view(-1, *([1] * (w.dim() - 1)))
    fb = (b0 - bn.running_mean.detach().float()) * scale + bn.bias.detach().float()
    return fw, fb


from .yolov10_conv import YOLOConv  # noqa: E402  (kept for state_dict parity)


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)
        self._dim = dim
        self._oc = h
        # tl.dot needs a power-of-two contraction of at least 16.
        self._kdp = max(16, 1 << (self.key_dim - 1).bit_length())
        self._packed = False
        self._buf: dict = {}

    # -- one-time weight packing (real weights only land via load_state_dict) --
    @torch.no_grad()
    def _pack(self) -> None:
        dev = self.qkv.conv.weight.device
        C, OC = self._dim, self._oc
        wq, bq = _fuse_bn(self.qkv.conv, getattr(self.qkv, "bn", None))
        wp, bp = _fuse_bn(self.pe.conv, getattr(self.pe, "bn", None))
        wo, bo = _fuse_bn(self.proj.conv, getattr(self.proj, "bn", None))
        self._wt_qkv = wq.reshape(OC, C).t().contiguous().half().to(dev)   # [C, OC]
        self._bq = bq.contiguous().to(dev)
        self._pew = wp.reshape(C, 9).t().contiguous().to(dev)              # [9, C]
        self._peb = bp.contiguous().to(dev)
        self._wt_proj = wo.reshape(C, C).t().contiguous().half().to(dev)   # [c, oc]
        self._bp = bo.contiguous().to(dev)
        self._packed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, gh, gw = x.shape
        n = gh * gw
        if not self._packed:
            self._pack()
        key = (b, n, gh, x.device)
        got = self._buf.get(key)
        if got is None:
            oc, nh = self._oc, self.num_heads
            # Each split spans a power-of-two key block covering ceil(n / KS).
            bks = 1 << max(4, ((-(-n // _KS)) - 1).bit_length())
            nt1 = -(-n // _PT1)
            nt2 = -(-n // _TQ)
            nt3 = -(-n // _PT3)
            nb1 = -(-oc // _BO1)
            nb3 = -(-c // _BO3)
            got = (
                torch.empty((b, n, oc), device=x.device, dtype=torch.float16),
                torch.empty((b, oc, n), device=x.device, dtype=torch.float16),
                torch.empty((b * _KS, n, c), device=x.device, dtype=torch.float16),
                torch.empty((b * _KS * n, 2 * nh), device=x.device, dtype=torch.float32),
                torch.empty((b, c, gh, gw), device=x.device, dtype=torch.float16),
                (b * nt1 * nb1,), (b * nt2 * nh * _KS,), (b * nt3 * nb3,),
                nt1, nt2, nt3, nb1, nb3, bks,
            )
            self._buf[key] = got
        px, ch, avb, ml, out, g1, g2, g3, nt1, nt2, nt3, nb1, nb3, bks = got

        _qkv_kernel[g1](x, self._wt_qkv, self._bq, px, ch, n, nt1, nb1,
                        C=c, OC=self._oc, PT=_PT1, BO=_BO1, PDL=_PDL,
                        num_warps=_NW1, launch_pdl=bool(_PDL))
        _attn_part_kernel[g2](px, ch, self._pew, self._peb, avb, ml, self.scale,
                              n, nt2, gh, gw, KD=self.key_dim, KDP=self._kdp,
                              HD=self.head_dim, C=c, OC=self._oc,
                              NH=self.num_heads, TQ=_TQ, BKS=bks, KS=_KS, PDL=_PDL,
                              num_warps=_NW2, launch_pdl=bool(_PDL))
        _combine_proj_kernel[g3](avb, ml, self._wt_proj, self._bp, out, n, nt3, nb3,
                                 C=c, HD=self.head_dim, NH=self.num_heads, KS=_KS,
                                 PT=_PT3, BO=_BO3, PDL=_PDL,
                                 num_warps=_NW3, launch_pdl=bool(_PDL))
        return out
