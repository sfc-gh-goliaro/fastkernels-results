"""Vision MLP for Qwen vision transformer blocks.

Unified across Qwen2-VL (QuickGELU) and Qwen3-VL (SiLU) activations.

The captured workload is ``x: bf16[M, 1, 1152]`` with ``in_features=1152`` and
``hidden_features=4304``; on the Qwen3-VL capture that scores this operator the
activation handed in is ``L1.gelu.GELU(approximate="none")``.

The baseline runs three device kernels -- fc1, the activation, fc2 -- and passes
over the hidden tensor (M x 4304 bf16; 204 MB at M=23760, 556 MB at M=64680)
four times: fc1 writes it, the activation reads and rewrites it, fc2 reads it.
This file runs two kernels and passes over it twice. fc2 stays on the reference
GEMM, which on B200 is a Blackwell-native ``nvjet_sm100`` kernel with 2-CTA
clusters that nothing here beats: fc2's shape (M x 4304 -> 1152) is the less
favourable of the two -- N=1152 caps the useful tile width, so it runs at
1.0 PFLOP/s where fc1 gets 1.4 for the same FLOP count -- and the best of 11
Triton configs on it still measures 0.790x cuBLAS, which is the exact-N-fit
[256, 128] tile. Tile fit is worth the ~9% the arithmetic predicts and Triton's
own deficit eats it three times over.

Why the activation is the thing to attack (measured on B200, per call, us):

    M       fc1    F.gelu   fused act    fc2
    1760   21.5     15.4       11.3     25.7
    20680  140.4   126.0       60.4    142.3
    23760  138.3   121.8       66.6    185.5
    25168  173.3   156.7       70.6    204.0
    64680  390.2   404.3      169.0    416.9

``F.gelu(x, approximate="none")`` evaluates ``erf`` at ~40 instructions per
element, so it is *compute* bound, not bandwidth bound: at M=64680 it costs
404us -- as much as either GEMM -- while the 1.1 GB it has to move is only 169us
of B200 HBM. One fused pass with ``erf`` refactored onto a single hardware
``tanh.approx.f32`` turns the whole activation back into pure bandwidth. That
is step one; step two folds that same expression into the fc1 epilogue so the
hidden tensor is written already-activated and the middle two passes disappear
entirely.

There are two ways to get that epilogue and they are not equally cheap. The
hand-written Triton kernel (``_fc1_act_kernel``) fuses the activation but runs
the GEMM itself at 0.78x of cuBLAS, because Triton 3.6 cannot emit the 2-CTA
cluster the Blackwell-native ``nvjet_sm100`` kernel uses, so most of the saved
traffic goes straight back out as lost GEMM efficiency. cuBLASLt has a GELU
epilogue of its own -- reachable as ``torch._addmm_activation`` -- which fuses
the activation onto *that same* nvjet kernel and so gives up nothing. Measured
with alternating A/B/B/A timing on the five scored M, the cuBLASLt epilogue
beats the Triton one by 1.041/1.081/1.070/1.099/1.120 and the un-fused
three-kernel path by 1.086 geomean, so it is the preferred fc1 whenever it is
admissible -- see ``_LtFc1``. Its epilogue menu is only {GELU, ReLU}, so
QuickGELU and SiLU callers still need ``_fc1_act_kernel``, which is kept for
exactly that case (see ``_G_MIN_WAVES`` for where it pays).

Dispatch is by *measured equivalence*, not by name. On the first call each
candidate fused activation is run against the ``act_fn`` the caller actually
passed, in the live dtype, over a fixed magnitude sweep, and is adopted only if
it agrees to the harness' own per-dtype tolerance with margin. Anything
unrecognised, mismatching, TP-sharded, fp8-quantised, or of an untested dtype
falls back to ``fc2(act_fn(fc1(x)))`` exactly as the baseline writes it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# Frozen L1 winner. The bf16/fp16 kernel below is bit-identical to it and the
# same speed (measured: |diff| = 0, 11.2/58.4/66.6/70.7/168.8us against
# 11.2/58.4/66.6/70.6/168.5us), so its measured grid constants are adopted
# wholesale below rather than re-derived. It is kept as a probe candidate for
# fp32/fp64 GELU, where the harness' (1e-5, 1e-3) is tighter than a fitted
# polynomial and this winner has an accurate-intrinsic path.
try:
    from ..L1.gelu import GELU as _L1GELU
except Exception:  # pragma: no cover - no triton / build failure
    _L1GELU = None

try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:  # pragma: no cover
    triton = None
    TensorDescriptor = None

if triton is not None:
    try:
        from triton.language.extra.cuda import gdc_wait as _gdc_wait
    except Exception:  # pragma: no cover - older triton without PDL intrinsics
        _gdc_wait = None
else:  # pragma: no cover
    _gdc_wait = None


# ---------------------------------------------------------------------------
# Fused elementwise activations.
#
# All four activations this operator can be handed are the same shape --
# ``0.5*x*(1 + tanh(g(x)))`` -- which is one hardware ``tanh.approx.f32`` (SFU)
# plus a couple of FMAs, so the pass runs at copy bandwidth:
#
#   exact gelu   g(x) = x*(A1 + A3 x^2 + A5 x^4)     (fitted, see below)
#   tanh gelu    g(x) = x*(T1 + T3 x^2)              (torch's own polynomial)
#   quickgelu    g(x) = 0.851*x                      (sigmoid(1.702x) half-angle)
#   silu         g(x) = 0.5*x                        (sigmoid(x) half-angle)
#
# The exact-mode constants are the L1 GELU winner's least-squares fit of
# ``logit(Phi(x))``, already halved so they feed ``tanh`` directly; worst-case
# error against ``x*Phi(x)`` over x in [-8, 8] is 3.0e-5, below one bf16 ULP.
# ``x^2`` is capped because the fitted quintic turns over at |x| ~ 11 (A5 < 0),
# past which g would swing negative and the activation would collapse to 0
# instead of approaching x.
# ---------------------------------------------------------------------------
_K_GELU, _K_GELU_TANH, _K_QUICKGELU, _K_SILU = 0, 1, 2, 3

if triton is not None:
    # Triton reads these at trace time, so they have to be constexpr globals.
    _A1 = tl.constexpr(0.79745782)
    _A3 = tl.constexpr(0.037051035)
    _A5 = tl.constexpr(-0.000358865)
    _T1 = tl.constexpr(0.7978845608028654)
    _T3 = tl.constexpr(0.035677408136300125)
    _UCAP = tl.constexpr(64.0)
    _QG_HALF = tl.constexpr(0.851)  # 1.702 / 2

    @triton.jit
    def _tanh_approx(v):
        return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [v],
                                        dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def act_fp32(xf, KIND: tl.constexpr):
        """Activation on an fp32 value/tile. Shared by the elementwise pass."""
        h = 0.5 * xf
        if KIND == 0:
            u = tl.minimum(xf * xf, _UCAP)
            g = xf * ((_A5 * u + _A3) * u + _A1)
        elif KIND == 1:
            g = xf * (_T3 * (xf * xf) + _T1)
        elif KIND == 2:
            g = _QG_HALF * xf
        else:
            g = h
        return h * _tanh_approx(g) + h

    @triton.jit
    def _act_kernel(X, Y, n, KIND: tl.constexpr, EXACT: tl.constexpr,
                    BLOCK: tl.constexpr, REVERSE: tl.constexpr):
        if REVERSE:
            # X was just written by fc1, so L2 holds X's *tail*. Walking tiles
            # high-to-low reads the most-recently-used end first while our own
            # stores evict from the cold (low) end, instead of the read front
            # forever chasing lines its own stores just evicted.
            pid = tl.num_programs(0) - 1 - tl.program_id(0)
        else:
            pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        _gdc_wait()  # fc1 may still be draining; wait before reading its output
        if EXACT:
            x = tl.load(X + off)
            tl.store(Y + off, act_fp32(x.to(tl.float32), KIND).to(X.dtype.element_ty))
        else:
            m = off < n
            x = tl.load(X + off, mask=m)
            tl.store(Y + off, act_fp32(x.to(tl.float32), KIND).to(X.dtype.element_ty),
                     mask=m)


# Grid, taken from the L1 GELU winner's B200 measurements: 16 elements/thread
# (2048/(32*4)) is the bandwidth plateau, and short passes that hide entirely
# behind the producer instead want the same bytes spread over 4x the CTAs.
_BLOCK, _WARPS = 2048, 4
_SMALL_BLOCK, _SMALL_WARPS = 512, 4
_SMALL_N = 524288
_FAST_DTYPES = (torch.float16, torch.bfloat16)

_l2_bytes: dict[int, int] = {}


def _l2_capacity(device: torch.device) -> int:
    dev = device.index if device.index is not None else 0
    cap = _l2_bytes.get(dev)
    if cap is None:
        try:
            cap = int(torch.cuda.get_device_properties(dev).L2_cache_size)
        except Exception:
            cap = 1 << 30  # unknown: never reverse
        _l2_bytes[dev] = cap
    return cap


class _FusedAct:
    """One fused elementwise activation pass over a dense fp16/bf16 tensor."""

    __slots__ = ("kind",)

    def __init__(self, kind: int):
        self.kind = kind

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        if x.stride() != y.stride():
            x = x.contiguous()
        n = x.numel()
        if n == 0:
            return y
        block, warps = ((_SMALL_BLOCK, _SMALL_WARPS) if n <= _SMALL_N
                        else (_BLOCK, _WARPS))
        _act_kernel[((n + block - 1) // block,)](
            x, y, n, self.kind, n % block == 0, block,
            2 * n * x.element_size() > _l2_capacity(x.device),
            num_warps=warps, launch_pdl=True)
        return y


def _fused_act(kind: int, dtype: torch.dtype) -> Callable | None:
    if triton is None or _gdc_wait is None or dtype not in _FAST_DTYPES:
        return None
    return _FusedAct(kind)




# ---------------------------------------------------------------------------
# fc1 with the activation fused into the GEMM epilogue.
#
# The hidden tensor is ~4x the input (M x 4304 bf16, 556 MB at M=64680) and the
# baseline touches it four times: fc1 writes it, the activation reads and
# rewrites it, fc2 reads it. Computing the activation in the fc1 epilogue, on
# the fp32 accumulator before it is ever stored, removes two of those four
# passes outright.
#
# The catch on B200 is that cuBLAS' bf16 GEMM here is a Blackwell-native
# ``nvjet_sm100`` kernel with 2-CTA clusters, and Triton 3.6 cannot express
# that: ``num_ctas=2`` fails to compile together with ``warp_specialize=True``
# (PassManager failure in ConvertTritonGPUToLLVM). So this kernel gives up raw
# GEMM efficiency and has to pay for it out of the two saved passes. Measured
# on B200 at M=20680, K=1152, N=4304 (fraction of cuBLAS' un-fused GEMM time):
#
#   plain tl.dot, masked loads, flat grid            0.52x
#   + host TMA descriptors, persistent grid          0.57x
#   + warp_specialize=True                           0.74x
#   + swizzle group 4 instead of 8                   0.77x
#   + subtiled epilogue                              0.78x
#
# 0.78x is a structural ceiling, not a tuning shortfall. The [128, 256] fp32
# accumulator is 512 tcgen05 tensor-memory columns, which is the *entire* B200
# tmem budget: [256, 256] and [128, 512] both fail with "out of resource: tensor
# memory, Required: 1024", and every wider operand tile fails on the 227 KB smem
# limit instead. cuBLAS' nvjet kernel gets 2x512 columns because it runs a 2-CTA
# cluster, and that is exactly the thing Triton 3.6 will not compile here:
# ``num_ctas=2`` plus ``warp_specialize=True`` dies in ConvertTritonGPUToLLVM,
# and ``num_ctas=2`` without warp specialization measures 0.54x.
#
# At 0.78x the fused GEMM costs 150.5us where cuBLAS' GEMM plus the fused
# activation pass costs 115.7 + 58.4 = 174.1us, so the fusion nets out ahead --
# but only once the grid is deep enough to amortise the persistent kernel's
# per-tile pipeline prologue, which is what ``_G_MIN_WAVES`` gates on.
# ---------------------------------------------------------------------------
_G_BM, _G_BN, _G_BK = 128, 256, 64
_G_GROUP = 4        # M-tiles per swizzle group; 4 beat 1/2/8/16/32
_G_STAGES, _G_WARPS = 3, 8
# Tiles per SM below which the persistent grid loses more to load imbalance and
# un-amortised pipeline prologues than the fused epilogue saves. Measured
# crossover on B200 (gain vs the 3-kernel path): 1.6 waves 0.92x, 2.8 waves
# 0.93x, 4.1 waves 1.02x, 5.5 waves 1.00x, 11 waves 1.01x, 18.6 waves 1.04x.
_G_MIN_WAVES = 4

if triton is not None:

    @triton.jit
    def _fc1_act_kernel(AD, BD, CD, BIAS, M, N, K, KIND: tl.constexpr,
                        HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                        BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
                        NSMS: tl.constexpr):
        """C[M, N] = act(A[M, K] @ B[N, K].T + bias[N]).

        Both operands are K-contiguous (A is the collapsed activation, B is
        ``F.linear``'s ``[N, K]`` weight), which is the layout tcgen05 wants, so
        no transpose lands in the pipeline. TMA zero-fills out-of-bounds reads
        and clips out-of-bounds writes, so ragged M, N and K need no masking:
        zeros in the operand tile contribute nothing to the accumulator.
        """
        nm = tl.cdiv(M, BM)
        nn = tl.cdiv(N, BN)
        # Address setup is all that precedes the wait, so the producing kernel's
        # tail overlaps this grid's dispatch.
        _gdc_wait()
        for tile in tl.range(tl.program_id(0), nm * nn, NSMS, flatten=True,
                             warp_specialize=True):
            # Group GM consecutive M-tiles against the same N-tile so the B
            # tiles they share stay resident in L2.
            gsz = GM * nn
            base = (tile // gsz) * GM
            gm = tl.minimum(nm - base, GM)
            pm = base + ((tile % gsz) % gm)
            pn = (tile % gsz) // gm
            om, on = pm * BM, pn * BN
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BK)):
                a = AD.load([om, k * BK])
                b = BD.load([on, k * BK])
                acc = tl.dot(a, b.T, acc)
            if HAS_BIAS:
                rn = on + tl.arange(0, BN)
                acc += tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
            # Subtiled epilogue: converting and storing the [BM, BN] fp32
            # accumulator in two [BM, BN/2] halves keeps the store staging
            # buffer small enough that the next tile's MMA can start earlier.
            # Worth 3% over storing the full tile.
            half = tl.permute(tl.reshape(acc, (BM, 2, BN // 2)), (0, 2, 1))
            c0, c1 = tl.split(half)
            CD.store([om, on], act_fp32(c0, KIND).to(CD.dtype))
            CD.store([om, on + BN // 2], act_fp32(c1, KIND).to(CD.dtype))


class _FusedFc1:
    """fc1 + activation as one kernel. ``None`` from :func:`_fused_fc1` when the
    shape, dtype, layout or GPU is outside what was measured."""

    __slots__ = ("kind", "wdesc", "n", "k", "nsms", "nn")

    def __init__(self, kind: int, weight: torch.Tensor, nsms: int):
        self.kind = kind
        self.n, self.k = weight.shape
        self.wdesc = TensorDescriptor.from_tensor(weight, [_G_BN, _G_BK])
        self.nsms = nsms
        self.nn = -(-self.n // _G_BN)

    def waves(self, m: int) -> float:
        return (-(-m // _G_BM)) * self.nn / self.nsms

    def __call__(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        m = x.shape[0]
        out = torch.empty(m, self.n, dtype=x.dtype, device=x.device)
        grid = (min(self.nsms, (-(-m // _G_BM)) * self.nn),)
        _fc1_act_kernel[grid](
            TensorDescriptor.from_tensor(x, [_G_BM, _G_BK]), self.wdesc,
            TensorDescriptor.from_tensor(out, [_G_BM, _G_BN // 2]), bias,
            m, self.n, self.k, self.kind, bias is not None,
            _G_BM, _G_BN, _G_BK, _G_GROUP, self.nsms,
            num_stages=_G_STAGES, num_warps=_G_WARPS, launch_pdl=True)
        return out


def _aligned(t: torch.Tensor) -> bool:
    if t.data_ptr() % 16:
        return False
    return all(s * t.element_size() % 16 == 0 for s in t.stride()[:-1])


def _fused_fc1(kind: int | None, weight: torch.Tensor, bias: torch.Tensor | None,
               dtype: torch.dtype) -> "_FusedFc1 | None":
    """Build the fused fc1, or None if anything is outside the tuned envelope."""
    if kind is None or triton is None or TensorDescriptor is None or _gdc_wait is None:
        return None
    if dtype not in _FAST_DTYPES:
        return None
    try:
        # tcgen05 + warp specialization is the entire margin, and the
        # cuBLAS-vs-Triton trade-off behind _G_MIN_WAVES was measured on
        # Blackwell only.
        if torch.cuda.get_device_capability(weight.device)[0] < 10:
            return None
        if weight.dim() != 2 or not weight.is_contiguous():
            return None
        n, k = weight.shape
        # A tile smaller than one block would need a TMA box larger than the
        # tensor; nothing in the captured set is that small.
        if n < _G_BN or k < _G_BK:
            return None
        if not _aligned(weight) or (bias is not None and not _aligned(bias)):
            return None
        if bias is not None and (bias.dim() != 1 or bias.numel() != n
                                 or not bias.is_contiguous()):
            return None
        nsms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        return _FusedFc1(kind, weight, nsms)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# fc1 with the activation in cuBLASLt's own epilogue.
#
# ``torch._addmm_activation(bias, x, W.T, use_gelu=...)`` is cuBLASLt's
# ``EPILOGUE_GELU_BIAS`` / ``EPILOGUE_RELU_BIAS``: the activation is applied to
# the fp32 accumulator in the GEMM's own epilogue, exactly like
# ``_fc1_act_kernel`` does, except the GEMM underneath is still cuBLAS' own
# Blackwell ``nvjet_sm100`` kernel with its 2-CTA cluster. So this fusion is
# free where the Triton one costs 22% of the GEMM. Measured on B200 with
# alternating A/B/B/A timing (ratio of full-pipeline time, five scored M
# 1760/20680/23760/25168/64680):
#
#   cuBLASLt epilogue vs the Triton epilogue    1.041 1.081 1.070 1.099 1.120
#   cuBLASLt epilogue vs one fused act pass     0.977 1.103 1.072 1.106 1.182
#   cuBLASLt epilogue vs exact F.gelu, 3 kernels 1.083 1.345 1.273 1.317 1.380
#
# fc1 alone is not what improves -- ``_addmm_activation`` picks the same GEMM as
# plain ``addmm`` -- the win is the hidden tensor's separate write+read going
# away, which is why the gain tracks M and vanishes once that tensor is small
# enough to stay in L2 (see ``_LT_MIN_BYTES``).
#
# Two limits matter. cuBLASLt only has GELU and ReLU epilogues, so a QuickGELU
# or SiLU caller cannot use this at all and falls through to
# ``_fc1_act_kernel``; and it is *addmm*, so the bias operand is mandatory --
# a bias-free layer gets a cached zero vector, which is exact (adding 0.0 does
# not round) and measures the same as a real bias.
# ---------------------------------------------------------------------------
_HAS_LT_ACT = hasattr(torch, "_addmm_activation")


def _lt_min_m(weight: torch.Tensor) -> int:
    """Smallest M at which the cuBLASLt epilogue is worth its GEMM penalty.

    The epilogue deletes a write and a read of the M x N hidden tensor, but the
    epilogue-fused GEMM itself measures 1.04-1.14x the time of the plain one
    (same shapes, alternating timing), so the trade only pays once that
    write+read is real HBM traffic instead of an L2 hit. Measured ratio of the
    un-fused three-kernel path to this one, hidden = M x 4304 bf16 on a 132.6 MB
    L2:

        M      1760  2560  3072  3584  4096  5120  6144  7705 10240 15410 20680
        ratio  0.94  1.02  0.92  1.04  1.04  0.99  1.07  1.08  1.14  1.11  1.12

    It alternates on kernel-selection luck while the activation's working set is
    L2-resident and turns into a consistent 7-14% win once ``2 x hidden bytes``
    stops fitting -- which is the same predicate ``_FusedAct`` already uses to
    decide whether its tile order matters. So that is the threshold, rather than
    a fitted M.
    """
    row = 2 * weight.shape[0] * weight.element_size()
    return max(1, _l2_capacity(weight.device) // row + 1)


class _LtFc1:
    """fc1 + activation as a single cuBLASLt call."""

    __slots__ = ("use_gelu", "wt", "bias", "n")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None,
                 use_gelu: bool):
        self.use_gelu = use_gelu
        self.n = weight.shape[0]
        # [N, K] row-major transposed to [K, N] column-major: the TN layout
        # cuBLAS wants, so the transpose is a view and never a kernel.
        self.wt = weight.t()
        self.bias = (bias if bias is not None else
                     torch.zeros(self.n, dtype=weight.dtype, device=weight.device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch._addmm_activation(self.bias, x, self.wt,
                                       use_gelu=self.use_gelu)


class _LtActProbe:
    """The cuBLASLt epilogue's activation on its own, as an elementwise callable.

    A ``1 x 8 @ 8 x n`` ``_addmm_activation`` whose two GEMM operands are zero
    accumulates exactly 0.0, so the output is the epilogue applied to the bias
    vector alone -- the same fp32 code path, on the same upcast-add-downcast
    sequence, as in the real call. That makes the epilogue testable by
    :func:`_equivalent` against whatever the caller passed, with no assumption
    about which GELU formula cuBLASLt implements.
    """

    __slots__ = ("use_gelu",)

    def __init__(self, use_gelu: bool):
        self.use_gelu = use_gelu

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        flat = v.reshape(-1)
        a = torch.zeros(1, 8, dtype=flat.dtype, device=flat.device)
        b = torch.zeros(8, flat.numel(), dtype=flat.dtype, device=flat.device)
        out = torch._addmm_activation(flat, a, b, use_gelu=self.use_gelu)
        return out.reshape(v.shape)


def _resolve_lt(act_fn: Callable, weight: torch.Tensor, bias: torch.Tensor | None,
                dtype: torch.dtype, device: torch.device) -> "_LtFc1 | None":
    """Build the cuBLASLt fused fc1, or None if its epilogue is not the
    caller's activation (to the harness' own tolerance) or the layer is outside
    what a plain ``addmm`` would accept."""
    # fp32/fp64 only: measured, not assumed. The epilogue's activation alone
    # clears the (1e-5, 1e-3) fp32 bound for tanh-GELU, but the *GEMM* under it
    # does not -- ``_addmm_activation`` accumulates fp32 differently enough from
    # ``F.linear`` that at M=7705, K=1152 only 98.67% of elements land inside
    # that bound, under the harness' required 99%. bf16/fp16 are two orders of
    # magnitude looser (1e-2, 1e-2) and clear it with room to spare, and they
    # are the only dtypes this operator is captured in, so the epilogue is
    # restricted to them rather than gated on a probe that a small M passes and
    # a large M fails.
    if not _HAS_LT_ACT or dtype not in _FAST_DTYPES:
        return None
    try:
        if weight.dim() != 2 or weight.stride(1) != 1:
            return None
        if bias is not None and (bias.dim() != 1
                                 or bias.numel() != weight.shape[0]
                                 or bias.stride(0) != 1):
            return None
        # GELU first: it is the epilogue the captured activation maps to, and
        # ReLU can never match a GELU-shaped function anyway.
        for use_gelu in (True, False):
            if _equivalent(act_fn, _LtActProbe(use_gelu), dtype, device):
                return _LtFc1(weight, bias, use_gelu)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Equivalence probe: which fused activation, if any, matches the caller's.
# ---------------------------------------------------------------------------
# Per-dtype (atol, rtol), mirroring the harness' comparison bounds. The probe
# additionally demands 99.9% of a 4k-point sweep inside them, where the harness
# asks 99% of the real tensor.
_PROBE_TOL = {
    torch.float64: (1e-5, 1e-3),
    torch.float32: (1e-5, 1e-3),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
}


def _probe_points(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """A deterministic magnitude sweep for the probe.

    Deterministic on purpose: the harness reseeds the global RNG immediately
    before each forward so baseline and candidate see identical inputs, and a
    probe that drew from it would be a side effect on that contract.
    """
    pts = torch.linspace(-40.0, 40.0, 4001, device=device, dtype=torch.float32)
    extra = torch.tensor(
        [0.0, 1e-6, -1e-6, 1e-3, -1e-3, 0.5, -0.5, 1.702, -1.702,
         88.0, -88.0, 1.0e3, -1.0e3, 1.0e4, -1.0e4],
        device=device, dtype=torch.float32)
    return torch.cat([pts, extra]).to(dtype)


def _equivalent(ref: Callable, fast: Callable, dtype: torch.dtype,
                device: torch.device) -> bool:
    """True if *fast* reproduces *ref* on the sweep in shape, dtype and value."""
    atol, rtol = _PROBE_TOL.get(dtype, (1e-2, 1e-2))
    try:
        x = _probe_points(dtype, device)
        with torch.no_grad():
            a = ref(x)
            b = fast(x)
        if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                or b.dtype is not a.dtype or b.shape != a.shape):
            return False
        af, bf = a.float(), b.float()
        if not (torch.isfinite(af).all() and torch.isfinite(bf).all()):
            return False
        good = (af - bf).abs() <= atol + rtol * af.abs()
        return bool(good.float().mean().item() >= 0.999)
    except Exception:
        return False


def _act_order(act_fn: Callable) -> list[int]:
    """Fused activation kinds worth probing, best guess first.

    The name/type hints only order the list -- :func:`_equivalent` decides. Every
    kind is probed as a last resort, so an unrecognised wrapper around a known
    formula is still caught.
    """
    order = [_K_GELU, _K_GELU_TANH, _K_QUICKGELU, _K_SILU]
    tag = f"{type(act_fn).__name__} {getattr(act_fn, '__name__', '')}".lower()
    approx = getattr(act_fn, "approximate", None)
    if "quick" in tag:
        first = _K_QUICKGELU
    elif "silu" in tag or "swish" in tag:
        first = _K_SILU
    elif "gelu" in tag:
        first = _K_GELU_TANH if approx == "tanh" else _K_GELU
    else:
        return order
    return [first] + [k for k in order if k != first]


def _resolve_act(act_fn: Callable, dtype: torch.dtype, device: torch.device):
    """Return ``(act, kind)``: a fused elementwise replacement for *act_fn* and
    the kind index the GEMM epilogue can reuse. Either may be None."""
    if dtype not in _PROBE_TOL:
        return None, None
    order = _act_order(act_fn)
    for kind in order:
        fn = _fused_act(kind, dtype)
        if fn is not None and _equivalent(act_fn, fn, dtype, device):
            return fn, kind
    # fp32/fp64 GELU: the harness' (1e-5, 1e-3) is far tighter than the fitted
    # polynomial above, and the frozen L1 winner has an accurate-intrinsic path
    # for exactly this case.
    if _L1GELU is not None:
        for mode in (("tanh", "none") if order[0] == _K_GELU_TANH
                     else ("none", "tanh")):
            fn = _L1GELU(mode)
            if _equivalent(act_fn, fn, dtype, device):
                return fn, None
    return None, None


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn
        # Resolved on the first forward: __init__ runs before the harness moves
        # the module to the GPU and casts it to the workload dtype, so neither
        # the device nor the dtype the activation will see is known here. Nothing
        # here is a Parameter or a submodule, so state_dict is untouched.
        self._act: Callable | None = None
        self._lt: "_LtFc1 | None" = None
        self._lt_min_m = 0
        self._fused: "_FusedFc1 | None" = None
        self._fast = False
        self._resolved = False

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))

    def _validate_fc1(self, call: Callable, w: torch.Tensor,
                      b: torch.Tensor | None, m: int):
        """Run a fused-fc1 candidate once against ``act_fn(F.linear(...))`` --
        the reference this operator is scored against -- and keep it only if it
        agrees to the harness' own per-dtype bound.

        The probe is deliberately a *ragged* M, and one large enough that the
        GEMM underneath makes the same kernel-selection decision it will make in
        anger -- a 135-row probe passes shapes that a 7705-row one rejects, so
        the caller passes the M the path is actually gated to. Its rows are
        scaled geometrically so the pre-activation values sweep magnitudes from
        ~0 into the tanh-saturated tail rather than clustering where every
        activation agrees. Deterministic: the harness reseeds the global RNG
        immediately before each forward, so drawing from it here would be a side
        effect on that contract.
        """
        try:
            k = w.shape[1]
            dev = w.device
            v = torch.sin(torch.arange(m * k, device=dev, dtype=torch.float32)
                          * 0.7071).view(m, k)
            rows = torch.logspace(-2.0, 1.0, m, device=dev).view(m, 1)
            sgn = torch.where(torch.arange(m, device=dev) % 2 == 0, 1.0, -1.0).view(m, 1)
            probe = (v * rows * sgn).to(w.dtype)
            with torch.no_grad():
                ref = self.act_fn(F.linear(probe, w, b)).float()
                got = call(probe).float()
            if got.shape != ref.shape or not torch.isfinite(got).all():
                return None
            if not torch.isfinite(ref).all():
                return None
            atol, rtol = _PROBE_TOL[w.dtype]
            good = (got - ref).abs() <= atol + rtol * ref.abs()
            return call if good.float().mean().item() >= 0.999 else None
        except Exception:
            return None

    def _resolve(self, x: torch.Tensor) -> bool:
        """First-call dispatch. Returns True if any fast path was adopted."""
        self._resolved = True
        fc1, fc2 = self.fc1, self.fc2
        # TP sharding adds an all-reduce and fp8 swaps in a block-scaled GEMM;
        # neither is on the captured path, so both stay eager.
        if getattr(fc1, "use_fp8", False) or getattr(fc2, "use_fp8", False):
            return False
        if getattr(fc2, "tp_size", 1) != 1 or not getattr(fc2, "reduce_results", True):
            return False
        if not (x.is_cuda and x.dtype is fc1.weight.dtype
                and x.dtype is fc2.weight.dtype):
            return False
        act, kind = _resolve_act(self.act_fn, x.dtype, x.device)
        self._act = act
        # cuBLASLt's own epilogue first: same nvjet GEMM as the reference plus a
        # free activation, so where it is admissible it beats both other paths.
        lt = _resolve_lt(self.act_fn, fc1.weight, fc1.bias, x.dtype, x.device)
        if lt is not None:
            # Validate at (a ragged M just over) the threshold the path is gated
            # to, capped so a narrow hidden dim cannot ask for a huge probe.
            min_m = _lt_min_m(fc1.weight)
            self._lt = self._validate_fc1(
                lt, fc1.weight, fc1.bias, min(max(_G_BM, min_m), 8192) + 7)
        if self._lt is not None:
            self._lt_min_m = min_m
        elif act is not None and kind is not None:
            # No cuBLASLt epilogue for this activation (QuickGELU / SiLU): fall
            # back to fusing it into the Triton GEMM, which is still ahead of
            # the un-fused path once the grid is deep enough.
            fused = _fused_fc1(kind, fc1.weight, fc1.bias, x.dtype)
            if fused is not None:
                self._fused = self._validate_fc1(
                    lambda t, _f=fused, _b=fc1.bias: _f(t, _b),
                    fc1.weight, fc1.bias, _G_BM + 7)
                if self._fused is not None:
                    self._fused = fused
        self._fast = (act is not None) or (self._lt is not None)
        return self._fast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fast:
            if self._resolved or not self._resolve(x):
                return self._eager(x)
        # Collapse [M, 1, K] -> [M, K] once; everything downstream is 2-D and
        # the trailing view back is free.
        shape = x.shape
        x2 = x if x.dim() == 2 else x.reshape(-1, shape[-1])
        fc1, fc2 = self.fc1, self.fc2
        m = x2.shape[0]
        lt = self._lt
        fused = self._fused
        if lt is not None and m >= self._lt_min_m and x2.stride(-1) == 1:
            h = lt(x2)
        elif (fused is not None and x2.is_contiguous()
                and fused.waves(m) >= _G_MIN_WAVES
                and m >= _G_BM and _aligned(x2)):
            h = fused(x2, fc1.bias)
        else:
            h = (self._act or self.act_fn)(F.linear(x2, fc1.weight, fc1.bias))
        y = F.linear(h, fc2.weight, fc2.bias)
        return y if x.dim() == 2 else y.view(*shape[:-1], y.shape[-1])
