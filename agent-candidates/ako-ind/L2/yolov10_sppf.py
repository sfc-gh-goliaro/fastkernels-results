"""YOLOv10 Spatial Pyramid Pooling - Fast (fused Triton).

The op is tiny -- fp16 [4,256,20,20] / [1,256,20,20], ~0.5 GFLOP, well under a
microsecond of real math on B200 -- so the eager baseline is dominated by
per-kernel cost, not FLOPs: conv+bn+silu, 3 maxpools, cat, conv+bn+silu is ~10
launches, and torch's fp16 stride-1 `max_pool2d` alone burns 24us of GPU time
for the three pools.  This collapses the whole block into three Triton kernels:

  K1  cv1: 1x1 conv == GEMM over channels, + BN + SiLU  -> concat segment 0
  K2  all three pools, row pass and column pass fused    -> concat segments 1..3
  K3  cv2: GEMM over the 512-deep concat, + BN + SiLU    -> output, in NCHW

Structural facts exploited:

  * Both convs are 1x1, i.e. out[Cout,HW] = W[Cout,Cin] @ in[Cin,HW].  NCHW makes
    HW contiguous, which suits both the GEMM and the pooling window, and cv2's
    output lands directly in NCHW with no transpose.
  * BN in eval is a per-output-channel affine, so it is applied to the fp32
    accumulator *after* the dot.  That is exact, needs no fused weight (conv's
    fp16 weight is used bit-identically and is already contiguous as
    [Cout, Cin]), and leaves forward with zero host-side work: no `.item()`, no
    `torch.cat`, no `contiguous()`, no allocation beyond the output.
  * stride=1 with -inf padding makes the pool chain collapse: y1 = P2(x),
    y2 = P4(x), y3 = P6(x), all reading cv1's output directly and independent of
    each other.
  * A pool window intersected with a rectangular plane factorizes into
    (row range) x (col range), so each P_r is *exactly* separable into a row
    (j) sliding max then a column (i) sliding max, with every intermediate
    centre in-plane.  The r=2 and r=4 row windows are prefixes of the r=6 one,
    so 13 row-load offsets feed all three radii, then 27 column-load offsets
    produce all three segments -- 40 offsets rather than the 169 a direct 13x13
    window would need.
  * Every window offset is a *load offset*, never a register shuffle.
    `tl.gather` can shift a register tile (and is correct), but measured ~30us
    on a [BC,512] fp16 tile versus re-reading L1 -- see ITERATIONS.md.

The three kernels are chained with **programmatic dependent launch** (PDL): each
one triggers its dependents after its last store and waits for its producer's
trigger just before its first load of produced data.  Because each kernel here
occupies well under one wave (112 blocks of 148 SMs for the two GEMMs) or drains
unevenly (512 blocks for the pool), the idle SMs let the next kernel's prologue
- address arithmetic, weight loads - overlap the previous kernel's tail instead
of paying full launch latency three times.  Worth ~28% end-to-end here, the
single largest win after the initial fusion: 25.6 -> 18.4us at N=4 and
21.5 -> 16.4us at N=1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv

_NEG_INF = tl.constexpr(float("-inf"))

# Tile configs, chosen by graph-amortized GPU time (a wall-clock sweep here just
# measures Python dispatch, so REP launches are captured into one CUDA graph and
# replayed -- see ITERATIONS.md).  Only ~1600 (N=4) or 400 (N=1) output rows
# exist to fill 148 SMs, so occupancy beats tile efficiency, and batch 1 wants
# strictly smaller tiles than batch 4 to keep enough blocks in flight.
_CFG = {
    1: (dict(BC=16, BP=64, BK=64, nw=4, ns=4),      # cv1
        dict(BC=1, nw=4, ns=2),                     # pool
        dict(BM=32, BP=32, BK=64, nw=4, ns=4)),     # cv2
    4: (dict(BC=32, BP=64, BK=256, nw=8, ns=3),
        dict(BC=1, nw=4, ns=2),
        dict(BM=64, BP=64, BK=128, nw=4, ns=3)),
}


def _cfg(n: int):
    return _CFG[1] if n == 1 else _CFG[4]


@triton.jit
def _act(acc, mu, sc, bi, dt):
    """BN (eval) + SiLU, reproducing the eager dataflow exactly.

    Eager rounds three times -- conv->fp16, batch_norm in fp32->fp16, silu in
    fp32->fp16 -- and ATen's `batch_norm_transform_input_kernel` evaluates
    `(x - mean) * (gamma * invstd) + beta` in fp32, subtracting the mean
    *before* scaling.  Rounding at the same three points and keeping that same
    association makes the whole op bit-exact against the baseline
    (err 0.00e+00) rather than merely inside tolerance.
    """
    v = (acc.to(dt).to(tl.float32) - mu) * sc + bi
    v = v.to(dt).to(tl.float32)
    return (v * tl.sigmoid(v)).to(dt)


@triton.jit
def _cv1_kernel(
    X, W, MU, SC, BI, CAT, NPT, sx_n, scat_n,
    HW: tl.constexpr, CIN: tl.constexpr, COUT: tl.constexpr,
    BC: tl.constexpr, BP: tl.constexpr, BK: tl.constexpr,
):
    """cv1: out[BC, BP] = W1[BC, CIN] @ x[n, CIN, HW], + BN + SiLU."""
    pid = tl.program_id(0)
    n = pid // NPT
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    p = (pid % NPT) * BP + tl.arange(0, BP)
    ok = (p < HW)[None, :]

    acc = tl.zeros((BC, BP), dtype=tl.float32)
    xb = X + n * sx_n
    # x is written by whatever precedes this kernel in the stream (in the bench,
    # the harness's own input copy).  Since this kernel carries launch_pdl, it
    # may start before that predecessor finishes, so wait for its trigger --
    # implicit at completion for a non-PDL producer -- before the first load.
    gdc_wait()
    for k0 in range(0, CIN, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(W + c[:, None] * CIN + k[None, :])
        b = tl.load(xb + k[:, None] * HW + p[None, :], mask=ok, other=0.0)
        acc = tl.dot(a, b, acc)

    dt = CAT.dtype.element_ty
    tl.store(CAT + n * scat_n + c[:, None] * HW + p[None, :],
             _act(acc, tl.load(MU + c)[:, None], tl.load(SC + c)[:, None],
                  tl.load(BI + c)[:, None], dt),
             mask=ok)
    gdc_launch_dependents()


@triton.jit
def _pool_kernel(
    CAT, RM, scat_n, srm_n,
    H: tl.constexpr, WD: tl.constexpr, HW: tl.constexpr, COUT: tl.constexpr,
    BC: tl.constexpr, BP: tl.constexpr, R: tl.constexpr,
):
    """All three pools: row (j) sliding max, then column (i) sliding max.

    The column pass reaches +-3R rows, so fusing the two passes into one kernel
    forces a block to own all H rows.  For a GEMM-bearing kernel that would
    crush occupancy (measured 14.6us when cv1 was fused in too), but this kernel
    has no GEMM, so BC can be a *single* channel: the block owns the whole
    20x20 plane for one channel and there are still N*Cout = 512 blocks at N=4.

    Every shifted read is therefore of data this same block just wrote, so the
    row pass can spill to a global scratch buffer and read it back across a
    `tl.debug_barrier()` (`barrier.sync` is a CTA-scope memory fence).  No
    cross-block ordering is involved.  Verified bit-exact against chained
    `F.max_pool2d` over 90 random / tied / single-spike planes.
    """
    n = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    p = tl.arange(0, BP)
    ok = (p < HW)[None, :]
    i = (p // WD)[None, :]
    j = (p % WD)[None, :]

    src = CAT + n * scat_n + c[:, None] * HW
    rmb = RM + n * srm_n + c[:, None] * HW
    neg = tl.full((BC, BP), _NEG_INF, dtype=CAT.dtype.element_ty)

    # row pass: r=R and r=2R windows are prefixes of r=3R, so 6R+1 loads serve all
    r1 = neg
    r2 = neg
    r3 = neg
    gdc_wait()
    for d in tl.static_range(-3 * R, 3 * R + 1):
        v = tl.load(src + (p + d)[None, :],
                    mask=ok & (j + d >= 0) & (j + d < WD), other=_NEG_INF)
        r3 = tl.maximum(r3, v)
        if (d >= -2 * R) & (d <= 2 * R):
            r2 = tl.maximum(r2, v)
            if (d >= -R) & (d <= R):
                r1 = tl.maximum(r1, v)
    tl.store(rmb + p[None, :], r1, mask=ok)
    tl.store(rmb + COUT * HW + p[None, :], r2, mask=ok)
    tl.store(rmb + 2 * COUT * HW + p[None, :], r3, mask=ok)
    tl.debug_barrier()

    # column pass -> concat segments 1..3
    y1 = neg
    y2 = neg
    y3 = neg
    for d in tl.static_range(-3 * R, 3 * R + 1):
        m = ok & (i + d >= 0) & (i + d < H)
        off = (p + d * WD)[None, :]
        y3 = tl.maximum(y3, tl.load(rmb + 2 * COUT * HW + off, mask=m,
                                    other=_NEG_INF))
        if (d >= -2 * R) & (d <= 2 * R):
            y2 = tl.maximum(y2, tl.load(rmb + COUT * HW + off, mask=m,
                                        other=_NEG_INF))
            if (d >= -R) & (d <= R):
                y1 = tl.maximum(y1, tl.load(rmb + off, mask=m, other=_NEG_INF))

    tl.store(src + COUT * HW + p[None, :], y1, mask=ok)
    tl.store(src + 2 * COUT * HW + p[None, :], y2, mask=ok)
    tl.store(src + 3 * COUT * HW + p[None, :], y3, mask=ok)
    gdc_launch_dependents()


@triton.jit
def _cv2_kernel(
    CAT, W, MU, SC, BI, OUT, NPT, scat_n, so_n,
    HW: tl.constexpr, CIN: tl.constexpr,
    BM: tl.constexpr, BP: tl.constexpr, BK: tl.constexpr,
):
    """cv2: out[BM, BP] = W2[BM, 4*Cout] @ concat[n, 4*Cout, HW], + BN + SiLU.

    The concat is a real buffer here rather than a K-split over four separate
    operands: materializing it costs one extra 400KB store plus a load, but lets
    this stay a plain contiguous GEMM, and lets the pools be computed exactly
    once instead of once per Cout tile.
    """
    pid = tl.program_id(0)
    n = pid // NPT
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    p = (pid % NPT) * BP + tl.arange(0, BP)
    ok = (p < HW)[None, :]

    acc = tl.zeros((BM, BP), dtype=tl.float32)
    cb = CAT + n * scat_n
    gdc_wait()
    for k0 in range(0, CIN, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(W + m[:, None] * CIN + k[None, :])
        b = tl.load(cb + k[:, None] * HW + p[None, :], mask=ok, other=0.0)
        acc = tl.dot(a, b, acc)

    dt = OUT.dtype.element_ty
    tl.store(OUT + n * so_n + m[:, None] * HW + p[None, :],
             _act(acc, tl.load(MU + m)[:, None], tl.load(SC + m)[:, None],
                  tl.load(BI + m)[:, None], dt),
             mask=ok)


# ---------------------------------------------------------------------------
# Host side.  At these sizes the Python path is a first-class cost, so anything
# decidable once is decided when the plan is built and the steady-state call
# does only: dict lookup, contiguity check, one allocation, three launches.
# Measured per-launch cost of one kernel, B200 / Triton 3.6:
#     kernel[grid](...)        JIT dispatch           8.2-10.5 us
#     CompiledKernel.run(...)  bound launcher         ~4-5 us
# and torch.cuda.current_stream(d).cuda_stream 2.07us vs
# torch._C._cuda_getCurrentRawStream(i) 0.07us, torch.empty 2.26us vs
# x.new_empty 1.59us.  Reading self.cv1.conv.weight through
# nn.Module.__getattr__ is ~1us, so validation is hoisted out of the hot path.
# ---------------------------------------------------------------------------
_raw_stream = torch._C._cuda_getCurrentRawStream

# The plan caches the BN affine as separate (scale, bias) tensors, so it is
# rebuilt when weights are loaded (load_state_dict hook), when the module is
# moved or recast (_apply), when .fuse() changes either conv, or when the input
# shape/device changes.  In-place mutation of bn.running_var etc. *without* any
# of those is not detected -- the same caveat YOLOConv.fuse() itself carries.
# This is an eval-only path (self.training defers to eager).


class _Bound:
    """A compiled Triton kernel plus its pre-built launch argument list.

    ``args[3]`` is the stream slot; ``args[9 + i]`` is declared parameter *i*
    (the launcher takes the grid, stream, function, packed metadata, launch
    metadata and the two hooks ahead of the kernel's own parameters).
    """

    __slots__ = ("run", "args")

    def __init__(self, jit_fn, grid, params, num_warps, num_stages):
        ck = jit_fn.warmup(*params, grid=grid, num_warps=num_warps,
                           num_stages=num_stages, launch_pdl=True)
        ck._init_handles()
        self.run = ck.run
        self.args = [grid[0], grid[1], grid[2], 0, ck.function,
                     ck.packed_metadata, None, None, None, *params]

    def set(self, i, v):
        self.args[9 + i] = v

    def __call__(self, stream):
        a = self.args
        a[3] = stream
        self.run(*a)


def _affine(conv, bn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-output-channel (mean, scale, bias) of eval-mode BN for this conv.

    Deliberately *not* folded into the weight: leaving BN as a post-dot affine
    keeps the fp16 conv weight bit-identical to the baseline's, so the only
    remaining difference would be association order -- hence this mirrors
    ATen's `batch_norm_transform_input_kernel`: `scale = gamma * rsqrt(var+eps)`
    in fp32, with the mean kept separate so the kernel subtracts it *before*
    scaling.  (Evaluating rsqrt in double instead measured 2.4e-4 off, i.e. one
    fp32 ulp of scale, so ATen's own fp32 rsqrt is what has to be matched.)
    """
    w = conv.weight
    cout = w.shape[0]
    f32 = dict(device=w.device, dtype=torch.float32)
    if bn is None:                                   # already folded by .fuse()
        bias = (conv.bias.to(torch.float32) if conv.bias is not None
                else torch.zeros(cout, **f32))
        return torch.zeros(cout, **f32), torch.ones(cout, **f32), bias
    invstd = torch.rsqrt(bn.running_var.to(torch.float32) + bn.eps)
    scale = bn.weight.to(torch.float32) * invstd
    mean = bn.running_mean.to(torch.float32)
    bias = bn.bias.to(torch.float32)
    if conv.bias is not None:      # conv bias lands before BN, so shift mean
        mean = mean - conv.bias.to(torch.float32)
    return mean, scale, bias


class _Plan:
    __slots__ = ("ks", "out_shape", "dev", "fused", "keep")

    def __init__(self, ks, out_shape, dev, fused, keep):
        self.ks = ks
        self.out_shape = out_shape
        self.dev = dev
        self.fused = fused          # (cv1, cv2) .fuse() state the plan assumes
        self.keep = keep            # holds scratch + staged weights alive


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.c1, self.c2, self.c_, self.k = c1, c2, c_, k
        self._plans = {}
        # Weights are only correct after the caller's load_state_dict, so the
        # plan (which stages them) is built lazily and dropped when they change.
        self.register_load_state_dict_post_hook(lambda *a, **kw: self._reset())

    def _reset(self, *_):
        self._plans = {}

    def _apply(self, *args, **kwargs):        # .to() / .half() / .cuda()
        self._reset()
        return super()._apply(*args, **kwargs)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        """Reference path for anything the fused path does not cover."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

    def _supported(self, x: torch.Tensor) -> bool:
        a, b, c = _cfg(x.shape[0])
        return (x.is_cuda and x.dim() == 4 and x.dtype == torch.float16
                and x.shape[1] == self.c1 and not self.training
                # the pool holds one whole plane per block, so H*W must fit a
                # tile; 20x20 is the captured shape and keeps it at 512 lanes
                and x.shape[2] == 20 and x.shape[3] == 20
                and self.k == 5 and self.m.kernel_size == 5
                and self.m.stride == 1 and self.m.padding == 2
                and not self.m.ceil_mode
                and self.cv1.conv.weight.dtype == torch.float16
                and self.cv2.conv.weight.dtype == torch.float16
                and self.cv1.conv.stride == (1, 1) and self.cv1.conv.groups == 1
                and self.cv2.conv.stride == (1, 1) and self.cv2.conv.groups == 1
                and self.c_ % a["BC"] == 0 and self.c_ % b["BC"] == 0
                and self.c2 % c["BM"] == 0)

    @torch.no_grad()
    def _build(self, x: torch.Tensor) -> _Plan:
        c_, c2, cin = self.c_, self.c2, self.c1
        n, _, h, w = x.shape
        hw = h * w
        a, b, c = _cfg(n)
        m1, s1, b1 = _affine(self.cv1.conv, getattr(self.cv1, "bn", None))
        m2, s2, b2 = _affine(self.cv2.conv, getattr(self.cv2, "bn", None))
        w1 = self.cv1.conv.weight.reshape(c_, -1).contiguous()
        w2 = self.cv2.conv.weight.reshape(c2, -1).contiguous()
        m1, s1, b1, m2, s2, b2 = (t.contiguous()
                                  for t in (m1, s1, b1, m2, s2, b2))
        cat = torch.empty((n, 4 * c_, hw), device=x.device, dtype=x.dtype)
        rm = torch.empty((n, 3 * c_, hw), device=x.device, dtype=x.dtype)
        out = torch.empty((n, c2, h, w), device=x.device, dtype=x.dtype)

        npt = triton.cdiv(hw, a["BP"])
        k1 = _Bound(_cv1_kernel, (n * npt, c_ // a["BC"], 1),
                    (x, w1, m1, s1, b1, cat, npt, cin * hw, cat.stride(0), hw,
                     cin, c_, a["BC"], a["BP"], a["BK"]), a["nw"], a["ns"])
        k2 = _Bound(_pool_kernel, (n, c_ // b["BC"], 1),
                    (cat, rm, cat.stride(0), rm.stride(0), h, w, hw, c_,
                     b["BC"], triton.next_power_of_2(hw), self.k // 2),
                    b["nw"], b["ns"])
        npt = triton.cdiv(hw, c["BP"])
        k3 = _Bound(_cv2_kernel, (n * npt, c2 // c["BM"], 1),
                    (cat, w2, m2, s2, b2, out, npt, cat.stride(0), c2 * hw, hw,
                     4 * c_, c["BM"], c["BP"], c["BK"]), c["nw"], c["ns"])
        return _Plan((k1, k2, k3), (n, c2, h, w), x.device.index,
                     (self.cv1._is_fused, self.cv2._is_fused),
                     (cat, rm, w1, w2, m1, s1, b1, m2, s2, b2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device.index
        plan = self._plans.get(x.shape)
        if (plan is None or plan.dev != dev
                or plan.fused != (self.cv1._is_fused, self.cv2._is_fused)):
            if not self._supported(x) or not x.is_contiguous():
                return self._eager(x)
            try:
                plan = self._build(x)
            except Exception:
                # e.g. PDL unsupported below sm_90, or a tile that will not fit
                return self._eager(x)
            self._plans[x.shape] = plan
        elif not x.is_contiguous():
            return self._eager(x)

        k1, k2, k3 = plan.ks
        out = x.new_empty(plan.out_shape)
        k1.set(0, x)
        k3.set(5, out)
        s = _raw_stream(dev)
        k1(s)
        k2(s)
        k3(s)
        return out
