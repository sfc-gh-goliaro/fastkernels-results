"""Conv2d with two fused Triton implicit-GEMM paths, both NCHW in and NCHW out.

    out[n, co, oh, ow] = sum_{c,i,j} x[n, c, oh*sh - ph + i, ow*sw - pw + j]
                                     * w[co, c, i, j]   (+ bias[co])

Both paths run this as one GEMM with ``M = OH*OW``, ``N = Cout``,
``K = Cin*kh*kw``, gathering the A tile straight out of NCHW inside the k loop --
no im2col buffer, no permute pass, no padding pass, fp32 accumulation, one launch
per call.  They differ only in how the gather addresses are formed:

* ``_patch_conv_kernel`` -- non-overlapping patches (``stride == kernel_size``,
  ``padding == 0``), where patch extraction is a pure strided read.  This is the
  oasis-500m patchify.
* ``_pad_conv_kernel`` -- overlapping, padded 3x3 (``padding == 1``, stride 1 or
  2), where the halo is handled by masked loads inside the k loop.  This is the
  yolov10n fp16 backbone conv.

Everything else -- fp32 non-patch, 1x1, grouped/depthwise, dilated -- falls back
to ``F.conv2d``, which is already at its launch floor there.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused patch-conv kernel: C[co, m] = sum_k Wt[co, k] * A[k, m] (+ bias[co])
#
# ``acc`` is laid out [BLOCK_CO, BLOCK_P] so the store runs along the output's
# contiguous (oh, ow) axis.  Every problem constant is a ``constexpr``: the
# shapes are fixed per module instance, so specializing buys magic-number
# division for the (c, i, j) / (oh, ow) decompositions and a fully unrolled K
# loop, which matters when the op is this small.
# ---------------------------------------------------------------------------
@triton.jit
def _patch_conv_kernel(
    X, Wt, Bias, Y,
    P: tl.constexpr,          # OH*OW
    COUT: tl.constexpr,
    K: tl.constexpr,          # Cin*KH*KW
    OW: tl.constexpr,
    IMW: tl.constexpr,        # input width (== stride between input rows)
    STRIDE_XN: tl.constexpr,
    STRIDE_XC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PREC: tl.constexpr,       # 0 = fp32 via 3-way tf32 split, 1 = native (fp16/bf16)
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_P: tl.constexpr,
):
    pid_co = tl.program_id(0)
    pid_p = tl.program_id(1)
    n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    m_co = offs_co < COUT
    m_p = offs_p < P

    oh = offs_p // OW
    ow = offs_p % OW
    base_a = n * STRIDE_XN + oh * (SH * IMW) + ow * SW

    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
    for kb in tl.static_range(NUM_K):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        c = offs_k // (KH * KW)
        r = offs_k % (KH * KW)
        a_off = (c[:, None] * STRIDE_XC
                 + (r // KW)[:, None] * IMW
                 + (r % KW)[:, None]
                 + base_a[None, :])
        w_off = offs_co[:, None] * K + offs_k[None, :]

        if EVEN_K:
            a_mask = None if EVEN_P else m_p[None, :]
            w_mask = None if EVEN_CO else m_co[:, None]
        else:
            m_k = offs_k < K
            a_mask = m_k[:, None] if EVEN_P else (m_k[:, None] & m_p[None, :])
            w_mask = m_k[None, :] if EVEN_CO else (m_co[:, None] & m_k[None, :])

        if a_mask is None:
            a = tl.load(X + a_off)
        else:
            a = tl.load(X + a_off, mask=a_mask, other=0.0)
        if w_mask is None:
            w = tl.load(Wt + w_off)
        else:
            w = tl.load(Wt + w_off, mask=w_mask, other=0.0)

        if PREC == 0:
            acc = tl.dot(w, a, acc=acc, input_precision="tf32x3")
        else:
            acc = tl.dot(w, a, acc=acc)

    if HAS_BIAS:
        if EVEN_CO:
            acc += tl.load(Bias + offs_co)[:, None].to(tl.float32)
        else:
            acc += tl.load(Bias + offs_co, mask=m_co, other=0.0)[:, None].to(tl.float32)

    y_off = n * (COUT * P) + offs_co[:, None] * P + offs_p[None, :]
    if EVEN_CO and EVEN_P:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty))
    else:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty), mask=m_co[:, None] & m_p[None, :])


# ---------------------------------------------------------------------------
# Fused padded/overlapping implicit-GEMM kernel (fp16/bf16 NCHW, any stride).
#
#   out[n, co, oh, ow] = sum_{c,i,j} x[n, c, oh*SH-PH+i, ow*SW-PW+j] * w[co,c,i,j]
#
# Blocking is over M = OH*OW (and Cout, and N) only: for these shapes the whole
# filter is a few KB (9 KB / 73 KB) so the K loop is a fully unrolled static
# nest over (tap, c-chunk) and the weight tile stays resident in L1.  The pad-1
# halo is handled by masked loads *inside* the k-loop -- no padding pass, no
# im2col buffer, one launch, NCHW in and NCHW out.
#
# ``acc`` is [BLOCK_CO, BLOCK_P] so the store walks the output's contiguous
# (oh, ow) axis; the A-tile load walks the input's contiguous w axis.
# ---------------------------------------------------------------------------
@triton.jit
def _pad_k_step(acc, xn, Wt, kb, offs_co, m_co, m_p, ih0, iw0,
                C: tl.constexpr, IMH: tl.constexpr, IMW: tl.constexpr,
                K: tl.constexpr, KW: tl.constexpr,
                ROW_TILED: tl.constexpr, BLOCK_K: tl.constexpr,
                BLOCK_P: tl.constexpr, EVEN_K: tl.constexpr,
                EVEN_CO: tl.constexpr, EVEN_P: tl.constexpr):
    """One k block: gather the A tile, load the weight tile, accumulate."""
    offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    tap = offs_k // C
    c = offs_k - tap * C
    # ih is [BLOCK_K, 1] when row-tiled (oh is a scalar there) and
    # [BLOCK_K, BLOCK_P] otherwise; iw is always 2D and broadcasts against it.
    if ROW_TILED:
        ih = ih0 + (tap // KW)[:, None]
    else:
        ih = ih0[None, :] + (tap // KW)[:, None]
    iw = iw0[None, :] + (tap % KW)[:, None]
    ok = (ih >= 0) & (ih < IMH) & (iw >= 0) & (iw < IMW)
    if not EVEN_K:
        ok = ok & (offs_k < K)[:, None]
    if not EVEN_P:
        ok = ok & m_p[None, :]
    a = tl.load(xn + c[:, None] * (IMH * IMW) + ih * IMW + iw, mask=ok, other=0.0)
    w_off = offs_co[:, None] * K + offs_k[None, :]
    if EVEN_CO and EVEN_K:
        w = tl.load(Wt + w_off)
    else:
        w_mask = m_co[:, None] if EVEN_K else m_co[:, None] & (offs_k < K)[None, :]
        w = tl.load(Wt + w_off, mask=w_mask, other=0.0)
    return tl.dot(w, a, acc=acc)


@triton.jit
def _pad_conv_kernel(
    X, Wt, Bias, Y,
    C: tl.constexpr,
    IMH: tl.constexpr,
    IMW: tl.constexpr,
    COUT: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,            # OH*OW
    TPR: tl.constexpr,          # p tiles per output row (ROW_TILED only)
    K: tl.constexpr,            # C*KH*KW
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROW_TILED: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    LOOP_STAGES: tl.constexpr,  # 0 = fully unroll the k loop, >=2 = pipeline it
    EVEN_K: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)
    n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    m_co = offs_co < COUT

    if ROW_TILED:
        # One p tile never crosses an output row: ``oh`` is a scalar and ``ow``
        # is an arange, so every input address is affine in the lane index
        # (stride SW).  Deriving oh/ow from a flat p (below) hides that from the
        # compiler, which then has to emit a general per-element gather.
        oh = pid_p // TPR
        ow = (pid_p - oh * TPR) * BLOCK_P + tl.arange(0, BLOCK_P)
        offs_p = oh * OW + ow
        m_p = ow < OW
    else:
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        oh = offs_p // OW
        ow = offs_p - oh * OW
        m_p = offs_p < P
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    xn = X + n * (C * IMH * IMW)
    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

    # K = (tap, c) with c innermost, matching ``_pad_weight``'s [COUT, tap, c]
    # transpose -- so the weight tile is BLOCK_K *contiguous* halves per output
    # channel.  Reading the natural [COUT, C, KH, KW] layout here instead strides
    # by KH*KW and turns every weight tile into a 9x-amplified 2-byte gather
    # (measured 0.73x on case #4, i.e. worse than cudnn).
    #
    # BLOCK_K spans whole taps: with BLOCK_K >= C one iteration covers several
    # taps at once, so the A-tile gather, the weight tile and the MMA are all
    # issued as one batch of independent loads.  Problems with only a couple of
    # hundred CTAs (0.5-2 warps per scheduler) have nothing else to hide memory
    # latency with, so that MLP inside the CTA is the whole game there.
    if LOOP_STAGES == 0:
        for kb in tl.static_range(NUM_K):
            acc = _pad_k_step(acc, xn, Wt, kb, offs_co, m_co, m_p, ih0, iw0,
                              C, IMH, IMW, K, KW, ROW_TILED, BLOCK_K, BLOCK_P,
                              EVEN_K, EVEN_CO, EVEN_P)
    else:
        for kb in tl.range(NUM_K, num_stages=LOOP_STAGES):
            acc = _pad_k_step(acc, xn, Wt, kb, offs_co, m_co, m_p, ih0, iw0,
                              C, IMH, IMW, K, KW, ROW_TILED, BLOCK_K, BLOCK_P,
                              EVEN_K, EVEN_CO, EVEN_P)

    if HAS_BIAS:
        if EVEN_CO:
            acc += tl.load(Bias + offs_co)[:, None].to(tl.float32)
        else:
            acc += tl.load(Bias + offs_co, mask=m_co, other=0.0)[:, None].to(tl.float32)

    y_off = n * (COUT * P) + offs_co[:, None] * P + offs_p[None, :]
    if EVEN_CO and EVEN_P:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty))
    else:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty),
                 mask=m_co[:, None] & m_p[None, :])


# ---------------------------------------------------------------------------
# Host side: plan (constexpr bundle + launch config) and launch.
# ---------------------------------------------------------------------------
_MAX_FUSED_FLOPS = 256 << 20
def _pick_cfg(p: int, cout: int, k: int) -> dict:
    """Tile shape for these launch-bound problems.

    Swept on the oasis patchify case (P=144, COUT=1024, K=64): a narrow patch
    tile wins by a wide margin -- 16 patches is one 64 B store row per output
    channel, and it keeps the CTA count at a couple of waves instead of
    serializing a few fat CTAs (BLOCK_P=128 measured 2x worse).  Warp count and
    stage count are flat within noise once the tile is right.
    """
    block_p = 16
    block_co = min(64, max(16, triton.next_power_of_2(cout)))
    block_k = min(triton.next_power_of_2(k), 64)
    return {"BLOCK_CO": block_co, "BLOCK_P": block_p, "BLOCK_K": block_k,
            "num_warps": 2, "num_stages": 1}


def _plan(x: torch.Tensor, weight: torch.Tensor, has_bias: bool,
          stride, padding, dilation, groups) -> dict | None:
    """Return a launch plan when *x* matches the fused pattern, else ``None``."""
    if x.dim() != 4 or weight.dim() != 4:
        return None
    if groups != 1 or padding != (0, 0) or dilation != (1, 1):
        return None
    kh, kw = weight.shape[2], weight.shape[3]
    if stride != (kh, kw):
        return None
    # 1x1 stride-1 also matches "stride == kernel_size", but cudnn is already at
    # the ~7 us single-launch floor there (measured 0.82x / 1.01x for the two
    # captured 1x1 cases), so only take genuine patch convs.
    if kh * kw == 1:
        return None
    if x.dtype != weight.dtype or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    if not weight.is_contiguous():
        return None

    n, c, h, w = (int(v) for v in x.shape)
    cout = int(weight.shape[0])
    if int(weight.shape[1]) != c:
        return None
    oh, ow = h // kh, w // kw
    if oh <= 0 or ow <= 0:
        return None

    p = oh * ow
    k = c * kh * kw
    # Launch-bound regime only.  The tile shape below is chosen for problems
    # whose whole GEMM is smaller than a single kernel launch is expensive
    # (case #1 is 38 MFLOP); on a genuinely compute-bound patchify -- the
    # [1,3,360,640] k20s20 variant, 1.4 GFLOP -- cudnn's tuned GEMM wins
    # (measured 0.94x), so leave that to the fallback.
    if 2 * n * p * cout * k > _MAX_FUSED_FLOPS:
        return None
    cfg = _pick_cfg(p, cout, k)
    block_k = cfg["BLOCK_K"]
    return {
        "kind": "patch",
        "out_shape": (n, cout, oh, ow),
        "grid": (triton.cdiv(cout, cfg["BLOCK_CO"]), triton.cdiv(p, cfg["BLOCK_P"]), n),
        "cfg": cfg,
        "consts": {
            "P": p, "COUT": cout, "K": k, "OW": ow, "IMW": w,
            "STRIDE_XN": c * h * w, "STRIDE_XC": h * w,
            "KH": kh, "KW": kw, "SH": kh, "SW": kw,
            "HAS_BIAS": has_bias,
            "PREC": 0 if x.dtype is torch.float32 else 1,
            "NUM_K": triton.cdiv(k, block_k),
            "EVEN_K": k % block_k == 0,
            "EVEN_CO": cout % cfg["BLOCK_CO"] == 0,
            "EVEN_P": p % cfg["BLOCK_P"] == 0,
        },
    }


def _run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
         plan: dict) -> torch.Tensor:
    y = torch.empty(plan["out_shape"], dtype=x.dtype, device=x.device)
    _patch_conv_kernel[plan["grid"]](
        x, weight, bias, y, **plan["consts"], **plan["cfg"])
    return y


# ---------------------------------------------------------------------------
# Host side (padded/overlapping path): gate + plan + launch.
#
# The gate is deliberately narrow: exactly the pattern that was benched (fp16 or
# bf16, 3x3, padding 1, stride 1 or 2, groups 1, dilation 1, contiguous weight).
# cudnn runs the two scored shapes at 0.02 and 0.38 TB/s -- nowhere near any
# hardware limit -- which is what makes one fused launch worth it.  Everything
# else (fp32 non-patch, 1x1, grouped/depthwise, dilated, other kernel sizes)
# stays on ``F.conv2d``: a pattern match is not evidence of a win, and 1x1 in
# particular is a measured dead end (0.82x / 1.01x through Triton in round 1).
# ``_MAX_PAD_FUSED_FLOPS`` is a safety bound rather than a measured boundary --
# it keeps a much larger, compute-bound conv (where a tuned cudnn GEMM should
# win, as it did for the 1.4 GFLOP patchify in round 1) off an untested path.
# ---------------------------------------------------------------------------
_MAX_PAD_FUSED_FLOPS = 4 << 30


@functools.lru_cache(maxsize=None)
def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _pick_pad_cfg(p: int, cout: int, c: int, k: int, n: int) -> dict:
    """Tile shape for the padded implicit GEMM.

    Only M (= OH*OW) and Cout are blocked; the k loop is a static nest over
    ceil(K / BLOCK_K) chunks of the (tap, channel) axis.  The two scored shapes
    sit in opposite regimes and a 400-config sweep under the faithful loop puts
    the split at roughly one CTA per SM:

    * case #4 ``[1,64,40,40]`` 64->64 s1 -- 200 CTAs on 148 SMs, i.e. ~1.4 warps
      per scheduler.  Nothing outside the CTA hides memory latency, so the win is a
      wide k block: the whole 73 KB filter and its A tile become one batch of
      independent loads feeding one dot (BLOCK_K 64 -> 256: 17.4 -> 13.4 us).
    * case #2 ``[4,16,320,320]`` 16->32 s2 -- 3200 CTAs.  Latency is already
      hidden, so a wide k block only buys masked-off lanes: every BLOCK_K > C
      measured strictly worse (52.2 -> 56-58 us), and the wide p tile wins
      instead.
    """
    ctas = triton.cdiv(p, 32) * triton.cdiv(cout, 16) * n
    if ctas < 2 * _num_sms():
        return {"BLOCK_CO": min(triton.next_power_of_2(cout), 16),
                "BLOCK_P": 32, "BLOCK_K": min(triton.next_power_of_2(k), 256),
                "ROW_TILED": False, "LOOP_STAGES": 0,
                "num_warps": 4, "num_stages": 1}
    return {"BLOCK_CO": min(triton.next_power_of_2(cout), 32),
            "BLOCK_P": 128, "BLOCK_K": min(triton.next_power_of_2(c), 32),
            "ROW_TILED": False, "LOOP_STAGES": 0,
            "num_warps": 4, "num_stages": 1}


def _plan_pad(x: torch.Tensor, weight: torch.Tensor, has_bias: bool,
              stride, padding, dilation, groups) -> dict | None:
    """Return a launch plan when *x* matches the padded fused pattern."""
    if x.dim() != 4 or weight.dim() != 4:
        return None
    if groups != 1 or dilation != (1, 1):
        return None
    if x.dtype != weight.dtype or x.dtype not in (torch.float16, torch.bfloat16):
        return None
    if not weight.is_contiguous():
        return None

    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    ph, pw = int(padding[0]), int(padding[1])
    sh, sw = int(stride[0]), int(stride[1])
    # 1x1 is at cudnn's single-launch floor already (measured 0.82x / 1.01x in
    # round 1) -- confirmed dead end, leave it on the fallback.
    if kh * kw <= 1:
        return None
    # Only the measured pattern: 3x3, pad 1, stride 1 or 2.  Wider gates were
    # not benched, and a pattern match alone is not evidence of a win.
    if (kh, kw) != (3, 3) or (ph, pw) != (1, 1) or sh not in (1, 2) or sw != sh:
        return None

    n, c, h, w = (int(v) for v in x.shape)
    cout = int(weight.shape[0])
    if int(weight.shape[1]) != c:
        return None
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        return None

    p = oh * ow
    k = c * kh * kw
    if 2 * n * p * cout * k > _MAX_PAD_FUSED_FLOPS:
        return None

    cfg = _pick_pad_cfg(p, cout, c, k, n)
    cfg = dict(cfg)
    block_k, block_p = cfg["BLOCK_K"], cfg["BLOCK_P"]
    row_tiled = cfg.pop("ROW_TILED")
    loop_stages = cfg.pop("LOOP_STAGES")
    tpr = triton.cdiv(ow, block_p)
    return {
        "kind": "pad",
        "out_shape": (n, cout, oh, ow),
        "grid": ((tpr * oh) if row_tiled else triton.cdiv(p, block_p),
                 triton.cdiv(cout, cfg["BLOCK_CO"]), n),
        "cfg": cfg,
        "consts": {
            "C": c, "IMH": h, "IMW": w, "COUT": cout, "OW": ow, "P": p,
            "TPR": tpr, "K": k, "KW": kw, "SH": sh, "SW": sw, "PH": ph, "PW": pw,
            "HAS_BIAS": has_bias, "ROW_TILED": row_tiled,
            "LOOP_STAGES": loop_stages,
            "NUM_K": triton.cdiv(k, block_k),
            "EVEN_K": k % block_k == 0,
            "EVEN_CO": cout % cfg["BLOCK_CO"] == 0,
            "EVEN_P": (ow % block_p == 0) if row_tiled else (p % block_p == 0),
        },
    }


def _pad_weight(weight: torch.Tensor) -> torch.Tensor:
    """``[COUT, Cin, KH, KW] -> [COUT, KH*KW, Cin]`` so the k-loop's weight tile
    is contiguous along the channel axis it blocks over."""
    return weight.permute(0, 2, 3, 1).reshape(weight.shape[0], -1).contiguous()


def _run_pad(x: torch.Tensor, wt: torch.Tensor, bias: torch.Tensor | None,
             plan: dict) -> torch.Tensor:
    y = torch.empty(plan["out_shape"], dtype=x.dtype, device=x.device)
    _pad_conv_kernel[plan["grid"]](
        x, wt, bias, y, **plan["consts"], **plan["cfg"])
    return y


class Conv2d(nn.Module):
    """Parametric 2D convolution: stores weight and bias internally."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self._plans: dict = {}
        self._wt: tuple | None = None

    def _get_plan(self, x: torch.Tensor):
        key = (x.shape, x.dtype, self.weight.dtype)
        plan = self._plans.get(key, False)
        if plan is False:
            args = (x, self.weight, self.bias is not None,
                    self.stride, self.padding, self.dilation, self.groups)
            plan = _plan(*args) or _plan_pad(*args)
            self._plans[key] = plan
        return plan

    def _padded_weight(self) -> torch.Tensor:
        """Transposed weight for the padded path, rebuilt when the weight is."""
        w = self.weight
        key = (w.data_ptr(), w._version, w.dtype)
        if self._wt is None or self._wt[0] != key:
            self._wt = (key, _pad_weight(w))
        return self._wt[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and self.weight.is_cuda:
            plan = self._get_plan(x)
            if plan is not None:
                xc = x if x.is_contiguous() else x.contiguous()
                if plan["kind"] == "patch":
                    return _run(xc, self.weight, self.bias, plan)
                return _run_pad(xc, self._padded_weight(), self.bias, plan)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
