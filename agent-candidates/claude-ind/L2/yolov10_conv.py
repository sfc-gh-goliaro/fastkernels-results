"""YOLOv10 Conv-BN-Act building block, fused into a single Triton kernel.

The baseline runs three kernels per call (``F.conv2d`` -> ``F.batch_norm`` ->
``F.silu``).  At the captured shapes every one of them is launch-latency bound:
the fixed cost of getting a kernel onto the GPU dwarfs the few microseconds of
arithmetic, and cuDNN's NCHW fp16 convolutions at these tiny channel counts are
themselves far from roofline.  So the whole block collapses into one
implicit-GEMM kernel that folds the (eval-mode) BatchNorm affine and the
activation into its epilogue.

Three things carry most of the speed:

* The GEMM is oriented as ``out[co, pixel] = weight[co, k] @ patch[k, pixel]``,
  so both operands *and* the store are contiguous along their last axis.  The
  conv weight ``(CO, CI, KH, KW)`` is already ``(CO, K)`` row-major and
  consecutive output pixels are consecutive addresses in NCHW, which lets Triton
  emit wide vector loads instead of per-element gathers.
* The reduction length is rounded up by zero-padding the weight, and ``k`` is
  mapped to an input address through a small precomputed table, so the inner
  loop needs neither a mask on ``k`` nor integer division to recover the tap.
* Zero padding is applied by a register select on an *unmasked* load rather than
  by a predicated one, since a predicated gather does not vectorise -- and for
  blocks whose whole window lies inside the input (the large majority, once the
  output is tiled a row at a time) even that select is skipped, chosen by a
  block-uniform branch.

Tile shapes are chosen per problem by timing a short candidate list the first
time a given (shape, stride, dtype) is seen -- outside any measured region, and
memoised process-wide.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


# ===========================================================================
# Kernels
# ===========================================================================
@triton.jit
def _conv_fused_kernel(
    x_ptr, w_ptr, s_ptr, b_ptr, o_ptr, kt_ptr,
    CO, KP, IH, IW, OW, ohow, co_ohow, ntw, glo, ghi, xnum,
    rlo, rhi, tlo, thi, sn, sc, sh,
    KH: tl.constexpr, KW: tl.constexpr,
    ST: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr, DL: tl.constexpr,
    ACT: tl.constexpr, ROWT: tl.constexpr, SPMASK: tl.constexpr,
    PMASK: tl.constexpr, CMASK: tl.constexpr,
    BCO: tl.constexpr, BP: tl.constexpr, BK: tl.constexpr,
):
    """``out[n, co, oh, ow] = act(scale[co] * conv2d(x, w)[n, co, oh, ow] + shift[co])``

    ``ROWT`` picks the output decomposition: 0 tiles the flattened output plane
    (only valid when the output plane matches the input plane at stride 1, so an
    input address is ``output_pixel + const``), 1 tiles one output row at a time
    and handles any stride.

    ``kt_ptr`` holds three precomputed ``int32`` rows of length ``KP``: the input
    address delta per reduction index, and the row/column offsets of its tap.
    Looking those up is far cheaper than recovering ``(ci, kh, kw)`` from ``k``
    with integer division inside the loop.
    """
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    nb = tl.program_id(2)

    offs_c = pid_c * BCO + tl.arange(0, BCO)
    cvalid = offs_c < CO

    if ROWT:
        orow = pid_p // ntw
        tw = pid_p - orow * ntw
        ih0 = orow * ST - PH
        base = nb * sn + ih0 * sh + tw * BP * ST - PW
        lane = tl.arange(0, BP)
        loff = lane * ST
        ow = tw * BP + lane
        pvalid = ow < OW
        iw0 = tw * BP * ST - PW + loff
        p = orow * OW + ow
    else:
        p = pid_p * BP + tl.arange(0, BP)
        pvalid = p < ohow
        base = nb * sn + pid_p * BP
        loff = tl.arange(0, BP)
        if SPMASK:
            orow = p // OW
            iw0 = p - orow * OW - PW
            ih0 = orow - PH

    safe = (base >= glo) & (base <= ghi)
    if ROWT:
        # whole window inside the input and every lane a real output pixel:
        # no per-element masking needed at all for this block
        fast = safe & (orow >= rlo) & (orow <= rhi) & (tw >= tlo) & (tw <= thi)
    else:
        fast = False
    acc = tl.zeros((BCO, BP), dtype=tl.float32)
    for k0 in range(0, KP, BK):
        kk = k0 + tl.arange(0, BK)
        off = (base + tl.load(kt_ptr + kk))[:, None] + loff[None, :]
        if SPMASK or PMASK:
            if fast:
                xt = tl.load(x_ptr + off)
            else:
                if SPMASK:
                    khd = tl.load(kt_ptr + KP + kk)
                    kwd = tl.load(kt_ptr + 2 * KP + kk)
                    ihh = ih0 + khd if ROWT else ih0[None, :] + khd[:, None]
                    hv = (ihh >= 0) & (ihh < IH)
                    iww = iw0[None, :] + kwd[:, None]
                    keep = (hv[:, None] if ROWT else hv) & (iww >= 0) & (iww < IW)
                    if PMASK:
                        keep = keep & pvalid[None, :]
                else:
                    keep = tl.broadcast_to(pvalid[None, :], (BK, BP))
                if safe:
                    # every address this block touches is inside the tensor, so
                    # the load can be unmasked (hence vectorised) and the padding
                    # applied as a register select
                    xt = tl.where(keep, tl.load(x_ptr + off),
                                  tl.zeros((BK, BP), x_ptr.dtype.element_ty))
                else:
                    # edge block: predicate on the address too, so no tile shape
                    # can read outside the tensor
                    xt = tl.load(x_ptr + off, mask=keep & (off >= 0) & (off < xnum),
                                 other=0.0)
        else:
            xt = tl.load(x_ptr + off)
        wp = w_ptr + offs_c[:, None] * KP + kk[None, :]
        if CMASK:
            wt = tl.load(wp, mask=cvalid[:, None], other=0.0)
        else:
            wt = tl.load(wp)
        acc = tl.dot(wt, xt, acc)

    if CMASK:
        scale = tl.load(s_ptr + offs_c, mask=cvalid, other=0.0)
        shift = tl.load(b_ptr + offs_c, mask=cvalid, other=0.0)
    else:
        scale = tl.load(s_ptr + offs_c)
        shift = tl.load(b_ptr + offs_c)
    acc = acc * scale[:, None] + shift[:, None]
    if ACT == 1:
        acc = acc * tl.sigmoid(acc)

    op = o_ptr + nb * co_ohow + offs_c[:, None] * ohow + p[None, :]
    val = acc.to(o_ptr.dtype.element_ty)
    if PMASK and CMASK:
        tl.store(op, val, mask=cvalid[:, None] & pvalid[None, :])
    elif PMASK:
        tl.store(op, val, mask=pvalid[None, :])
    elif CMASK:
        tl.store(op, val, mask=cvalid[:, None])
    else:
        tl.store(op, val)


@triton.jit
def _affine_act_kernel(o_ptr, s_ptr, b_ptr, n_elem, ohow, CO,
                       ACT: tl.constexpr, BLK: tl.constexpr):
    """In-place ``o = act(scale[co] * o + shift[co])`` over an NCHW tensor."""
    offs = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = offs < n_elem
    co = (offs // ohow) % CO
    v = tl.load(o_ptr + offs, mask=m, other=0.0).to(tl.float32)
    v = v * tl.load(s_ptr + co, mask=m, other=1.0)
    v = v + tl.load(b_ptr + co, mask=m, other=0.0)
    if ACT == 1:
        v = v * tl.sigmoid(v)
    tl.store(o_ptr + offs, v.to(o_ptr.dtype.element_ty), mask=m)


# ===========================================================================
# Tile selection
# ===========================================================================
_BCO = (16, 32, 64, 128)
_BP = (16, 32, 64, 128)
_WARPS = (1, 2, 4, 8)
_SMEM = 190 * 1024
_MAX_CFGS = 60
_CFG_CACHE: dict = {}
_NSM = None


def _nsm(device) -> int:
    global _NSM
    if _NSM is None:
        _NSM = torch.cuda.get_device_properties(device).multi_processor_count
    return _NSM


def _bk_choices(K: int) -> list[int]:
    """Reduction block lengths worth trying: powers of two up to ~2K.

    A block length that does not divide K is fine -- the weight is zero-padded
    to ``ceil(K/bk)*bk`` -- but the padding is wasted work, so anything that
    would more than double the reduction is dropped.
    """
    out = [v for v in (16, 32, 64, 128, 256) if v <= K or v < 2 * K]
    return out or [16]


def _candidate_cfgs(K, CO, ohow, OW, OH, nimg, modes, nsm):
    """A short ranked list of ``(rowt, BCO, BP, BK, warps, stages)`` tiles to time.

    Which decomposition and tile shape wins is not predictable from the shape
    alone -- masking cost, wave quantisation and reduction-loop latency all pull
    different ways -- so the list deliberately spreads over pixel-tile widths
    rather than taking the globally cheapest few by the cost model.
    """
    per_mode = max(10, _MAX_CFGS // max(1, len(modes)))
    picked = []
    for rowt in modes:
        buckets: dict = {}
        for bk in _bk_choices(K):
            kp = -(-K // bk) * bk
            niter = kp // bk
            waste = kp / K
            for bco in _BCO:
                if bco > 2 * CO:
                    continue
                for bp in _BP:
                    ntp = OH * -(-OW // bp) if rowt else -(-ohow // bp)
                    blocks = ntp * -(-CO // bco) * nimg
                    if blocks * 4 < nsm:
                        continue
                    waves = -(-blocks // nsm)
                    for warps in _WARPS:
                        if bco * bp < warps * 32 or bco * bp > warps * 32 * 16:
                            continue
                        for stages in (1, 2, 3, 4):
                            if stages > niter + 1:
                                continue
                            if 2 * bk * (bco + bp) * max(1, stages - 1) > _SMEM:
                                continue
                            steps = -(-niter // (stages - 1)) if stages > 1 else niter
                            cost = (waves * (220.0 * steps + 120.0)
                                    + waves * niter * bco * bp * bk / 2048.0)
                            cost *= waste ** 0.5
                            if (OW if rowt else ohow) % bp:
                                cost *= 1.05
                            cfg = (rowt, bco, bp, bk, warps, stages)
                            buckets.setdefault((bp, bco, bk), []).append((cost, cfg))
        for slot in buckets.values():
            slot.sort(key=lambda t: t[0])
        chosen: list = []
        seen = set()
        flat = sorted((e for slot in buckets.values() for e in slot), key=lambda t: t[0])
        for _, cfg in flat[:per_mode // 2]:
            if cfg not in seen:
                seen.add(cfg)
                chosen.append(cfg)
        # then round-robin over tile shapes so no shape is crowded out
        order = sorted(buckets, key=lambda k: buckets[k][0][0])
        depth = 0
        while len(chosen) < per_mode:
            progressed = False
            for key in order:
                slot = buckets[key]
                if depth < len(slot):
                    progressed = True
                    cfg = slot[depth][1]
                    if cfg not in seen:
                        seen.add(cfg)
                        chosen.append(cfg)
                    if len(chosen) >= per_mode:
                        break
            if not progressed:
                break
            depth += 1
        picked.extend(chosen)
    return picked or [(modes[0], 16, 16, 16, 4, 1)]


def _flush_buffer(device):
    """A buffer big enough to evict L2, so tuning sees the same cold caches a
    latency benchmark does -- a tile that wins with a warm L2 is often not the
    one that wins from DRAM.  ``None`` if it will not fit; tuning then just
    compares warm-cache timings."""
    try:
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
    except Exception:
        l2 = 64 << 20
    try:
        return torch.empty(2 * int(l2), dtype=torch.int8, device=device)
    except Exception:
        return None


def _shot(launch, src, feed, out):
    """One timed unit: refresh the input (as a previous layer would) then run."""
    def go():
        feed.copy_(src)
        launch(feed, out)
    return go


def _time_calls(fn, flush, iters: int = 15) -> float:
    """Median per-call latency (ms), with L2 flushed before each call."""
    for _ in range(3):
        if flush is not None:
            flush.zero_()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


# ===========================================================================
# Module
# ===========================================================================
class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False
        self._plan = None

    # -- folded BatchNorm --------------------------------------------------
    def _affine(self, device):
        """(scale, shift) applied to the raw conv output, folding eval-mode BN."""
        co = self.conv.weight.shape[0]
        ones = torch.ones(co, device=device, dtype=torch.float32)
        zeros = torch.zeros(co, device=device, dtype=torch.float32)
        bn = getattr(self, "bn", None)
        if self._is_fused or bn is None:
            scale = ones
            shift = self.conv.bias.detach().float() if self.conv.bias is not None else zeros
        else:
            inv = torch.rsqrt(bn.running_var.detach().float() + bn.eps)
            gamma = bn.weight.detach().float() if bn.weight is not None else ones
            beta = bn.bias.detach().float() if bn.bias is not None else zeros
            scale = gamma * inv
            shift = beta - scale * bn.running_mean.detach().float()
            if self.conv.bias is not None:
                shift = shift + scale * self.conv.bias.detach().float()
        return scale.contiguous(), shift.contiguous()

    def _act_code(self) -> int:
        a = self.act
        if isinstance(a, (SiLU, nn.SiLU)):
            return 1
        if isinstance(a, nn.Identity):
            return 0
        return -1

    # -- plan construction -------------------------------------------------
    def _build_plan(self, x: torch.Tensor):
        conv = self.conv
        w = conv.weight.detach()
        act = self._act_code()
        if act < 0:
            bn = getattr(self, "bn", None)
            if self._is_fused or bn is None:
                return lambda inp: self.act(conv(inp))
            return lambda inp: self.act(bn(conv(inp)))

        scale, shift = self._affine(w.device)
        st_h, st_w = conv.stride
        ph, pw = conv.padding
        dh, dw = conv.dilation
        co, cig, kh, kw = w.shape
        n, _, ih, iw = x.shape
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // st_h + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // st_w + 1

        fusable = (
            x.is_cuda and x.dim() == 4 and w.dtype == x.dtype
            and x.dtype in (torch.float16, torch.bfloat16)
            and conv.groups == 1 and x.stride(3) == 1
            and st_h == st_w and dh == dw and ph == pw
            and oh > 0 and ow > 0 and w.is_contiguous()
            and 2 * ph <= dh * (kh - 1) and 2 * pw <= dw * (kw - 1)
        )
        if not fusable:
            return self._fallback_plan(scale, shift, act)

        K = cig * kh * kw
        ohow = oh * ow
        flat_ok = (st_h == 1 and oh == ih and ow == iw and x.stride(2) == iw)
        if flat_ok and kh == 1 and kw == 1:
            modes = [0]          # row tiling can only be a coarser version of this
        else:
            modes = ([0] if flat_ok else []) + [1]
        spmask = 0 if (ph == 0 and pw == 0) else 1
        wflat = w.reshape(co, K)
        sn, sc, sh = x.stride(0), x.stride(1), x.stride(2)
        # One past the largest offset from ``x``'s base that is still inside its
        # storage.  ``x`` can be a strided view -- a channel slice of a wider
        # tensor -- where offsets beyond ``x.numel()`` are still legal reads, and
        # bounding by the element count would wrongly mask out real elements.
        xspan = (n - 1) * sn + (cig - 1) * sc + (ih - 1) * sh + iw
        out_shape = (n, co, oh, ow)
        dtype, dev = x.dtype, x.device
        kern = _conv_fused_kernel
        cache: dict = {}

        # (ci, kh, kw) triple for every reduction index, in weight order
        ar = torch.arange(K, device=dev, dtype=torch.int32)
        t_kw = ar % kw
        t_kh = (ar // kw) % kh
        t_ci = ar // (kh * kw)

        def tables(bk, rowt):
            """Zero-padded weight plus the per-k address/tap-offset lookup rows."""
            key = (bk, rowt)
            got = cache.get(key)
            if got is not None:
                return got
            kp = -(-K // bk) * bk
            if kp == K:
                wt = wflat
            else:
                wt = torch.zeros(co, kp, device=dev, dtype=dtype)
                wt[:, :K] = wflat
            if rowt:
                delta = t_ci * sc + (t_kh * dh) * sh + t_kw * dw
            else:
                delta = t_ci * sc + (t_kh * dh - ph) * sh + (t_kw * dw - pw)
            kt = torch.zeros(3, kp, device=dev, dtype=torch.int32)
            kt[0, :K] = delta
            kt[1, :K] = t_kh * dh
            kt[2, :K] = t_kw * dw
            got = (wt, kt, kp)
            cache[key] = got
            return got

        def build(cfg):
            rowt, bco, bp, bk, warps, stages = cfg
            wt, kt, kp = tables(bk, rowt)
            ntw = -(-ow // bp)
            cmask = co % bco != 0
            grid = (oh * ntw if rowt else -(-ohow // bp), -(-co // bco), n)
            if rowt:
                lstep = (bp - 1) * st_h
                xdmax = (cig - 1) * sc + (kh - 1) * dh * sh + (kw - 1) * dw
                # rows / column-tiles whose whole window lies inside the input
                rlo = -(-ph // st_h)
                rhi = (ih - 1 - (kh - 1) * dh + ph) // st_h
                tlo = -(-pw // (bp * st_h))
                span = (bp - 1) * st_h + (kw - 1) * dw - pw
                thi = min((iw - 1 - span) // (bp * st_h),
                          (ow - bp) // bp if bp <= ow else -1)
                args = (co, kp, ih, iw, ow, ohow, co * ohow, ntw,
                        pw, xspan - 1 - xdmax - lstep, xspan,
                        rlo, rhi, tlo, thi, sn, sc, sh)
                pm = ow % bp != 0
            else:
                lstep = bp - 1
                xdmax = (cig - 1) * sc + ((kh - 1) * dh - ph) * sh + ((kw - 1) * dw - pw)
                args = (co, kp, ih, iw, ow, ohow, co * ohow, ntw,
                        ph * sh + pw, xspan - 1 - xdmax - lstep, xspan,
                        0, 0, 0, 0, sn, sc, sh)
                pm = ohow % bp != 0

            def launch(inp, out):
                kern[grid](
                    inp, wt, scale, shift, out, kt, *args,
                    KH=kh, KW=kw, ST=st_h, PH=ph, PW=pw, DL=dh,
                    ACT=act, ROWT=rowt, SPMASK=spmask, PMASK=pm, CMASK=cmask,
                    BCO=bco, BP=bp, BK=bk,
                    num_warps=warps, num_stages=stages,
                )

            return launch

        sig = (spmask, cig, co, kh, kw, st_h, ph, dh, ih, iw, n, act, dtype)
        launch = None
        cfg = _CFG_CACHE.get(sig)
        if cfg is not None:
            try:
                launch = build(cfg)
            except Exception:
                launch = None
        if launch is None:
            cands = _candidate_cfgs(K, co, ohow, ow, oh, n, modes, _nsm(dev))
            probe = torch.empty(out_shape, device=dev, dtype=dtype)
            flush = _flush_buffer(dev)
            # In use the input has just been written by the preceding layer, so
            # it is L2-resident even after a cache flush.  Reproduce that here or
            # the timings rank tiles for the wrong cache state.  The stand-in has
            # to carry *x*'s exact strides: the kernel is built around them, and
            # ``empty_like`` silently compacts a non-dense view.
            feed = torch.empty_strided(x.shape, x.stride(), dtype=dtype, device=dev)
            best = None
            dbg = os.environ.get("FK_TUNE_DEBUG")
            if dbg:
                print(f"  tune {tuple(x.shape)} c{cig}->{co} k{kh} s{st_h} "
                      f"K={K} ohow={ohow} OW={ow} modes={modes}", file=sys.stderr)
            scored = []
            for c in cands:
                try:
                    fn = build(c)
                    fn(x, probe)
                    torch.cuda.synchronize()
                    t = _time_calls(_shot(fn, x, feed, probe), flush)
                except Exception as exc:
                    if dbg:
                        print(f"    cfg {c} failed: {exc!r:.90}", file=sys.stderr)
                    continue
                if dbg:
                    print(f"    cfg {c}  {t * 1000:.2f}us", file=sys.stderr)
                scored.append((t, c, fn))
            # The spread between good tiles is a couple of microseconds, which a
            # short timing run cannot resolve; re-time the leaders for longer.
            scored.sort(key=lambda e: e[0])
            for t, c, fn in scored[:4]:
                t2 = _time_calls(_shot(fn, x, feed, probe), flush, iters=49)
                if dbg:
                    print(f"    confirm {c}  {t2 * 1000:.2f}us", file=sys.stderr)
                if best is None or t2 < best[0]:
                    best = (t2, c, fn)
            del probe, flush, feed
            if best is None:
                return self._fallback_plan(scale, shift, act)
            if dbg:
                print(f"  ==> chose {best[1]}  {best[0] * 1000:.2f}us", file=sys.stderr)
            _CFG_CACHE[sig] = best[1]
            launch = best[2]

        def run(inp, launch=launch):
            out = torch.empty(out_shape, device=dev, dtype=dtype)
            launch(inp, out)
            return out

        return run

    def _fallback_plan(self, scale, shift, act):
        conv = self.conv
        w = conv.weight.detach()
        st, pd, dl, gr = conv.stride, conv.padding, conv.dilation, conv.groups
        co = w.shape[0]

        def run(inp):
            out = F.conv2d(inp, w, None, st, pd, dl, gr)
            ohow = out.shape[-1] * out.shape[-2]
            n_elem = out.numel()
            _affine_act_kernel[(-(-n_elem // 1024),)](
                out, scale, shift, n_elem, ohow, co, ACT=act, BLK=1024, num_warps=4)
            return out

        return run

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        key = (x.shape, x.stride(), x.dtype, x.device)
        if plan is None or plan[0] != key:
            plan = (key, self._build_plan(x))
            self._plan = plan
        return plan[1](x)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        self._plan = None
        return self


def fuse_module(module: nn.Module) -> nn.Module:
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
