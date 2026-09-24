"""GELU activation as a single fused 1-D elementwise Triton kernel.

The captured workloads are all dense fp16/bf16 tensors, so the tensor is treated
as a flat 1-D buffer: every access is a full-width vectorized load/store
regardless of the logical 3-D/5-D layout, the math runs in fp32, and the result
is stored back in the input dtype with no fp32 intermediate.

Three things make this faster than ``F.gelu`` on B200:

* **The transcendental is one instruction.** ``libdevice.erf`` costs ~40
  instructions/element and ``libdevice.tanh`` ~32, which makes exact GELU
  *compute* bound on the large bf16 shapes (measured 3.3 TB/s, where a plain
  copy of the same bytes gets 5.1 TB/s). Both modes are rewritten onto the
  hardware ``tanh.approx.f32`` (1 SFU op) via ``0.5x(1+tanh(x*P(x^2)))``. After
  that the kernel measures exactly as fast as one that only copies the bytes, so
  the arithmetic is entirely hidden behind memory.
* **PDL hides the launch latency.** A dependent kernel launch costs a fixed
  ~2 us of otherwise-idle GPU time here, which is the whole cost of the small
  shapes (they move only 0.5-3.5 MB). GELU always runs downstream of whatever
  produced its input, so ``launch_pdl=True`` lets this grid be staged while the
  producer drains; ``gdc_wait()`` before the first load keeps the dependency.
* **Tiles are walked high-to-low once the working set outgrows L2.** The input
  was just written by this op's producer, so L2 holds the input's *tail*.
  Descending means the read front starts on resident lines and our own stores
  evict from the far (already-cold) end, instead of forever evicting the lines
  the read front is about to touch. Worth 2-4% on the 139 MB shape; below L2
  capacity everything is resident anyway, and descending only costs the
  prefetcher, so it is gated on the working set.

Numerics: tolerances are (1e-2, 1e-2) for fp16/bf16, and the fitted exact-mode
polynomial is accurate to 3e-5 -- below one output ULP -- so the rounded result
is indistinguishable from exact. fp32/fp64 have far tighter tolerances
(1e-5, 1e-3) and take an accurate-intrinsic path instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait, libdevice

# Exact mode. gelu(x) = x*Phi(x) is refactored as x*sigmoid(y) = 0.5x(1+tanh(y/2))
# with y = x*(C1 + C3 x^2 + C5 x^4) least-squares fitted to logit(Phi(x)),
# weighted by d(gelu)/dy so the error is flat in the output. The constants below
# are y/2, i.e. the fitted C's already halved, so they feed tanh directly.
# Worst-case |error| vs x*Phi(x) over x in [-8, 8] is 3.0e-5.
_A1 = tl.constexpr(0.79745782)
_A3 = tl.constexpr(0.037051035)
_A5 = tl.constexpr(-0.000358865)
# The fitted quintic turns over at |x| ~ 11.02 (C5 < 0), past which y would swing
# negative and gelu would collapse to 0 instead of approaching x. Capping x^2
# keeps the tanh argument large and positive in the tail, giving the correct
# gelu(x) -> x limit for every finite input.
_UCAP = tl.constexpr(64.0)

# tanh mode: torch's own argument sqrt(2/pi)*(x + 0.044715 x^3), expanded. These
# are NOT halved -- torch's formula already writes the half-angle explicitly as
# 0.5*x*(1 + tanh(...)).
_T1 = tl.constexpr(0.7978845608028654)
_T3 = tl.constexpr(0.035677408136300125)

_RSQRT2 = tl.constexpr(0.7071067811865476)

# Two regimes, and they want opposite things:
#
# * Bandwidth-bound (the default). 2048/(32*4) = 16 elements per thread, i.e. two
#   128-bit accesses in flight per thread per direction. 16 elem/thread is the
#   measured plateau: 8 costs 6% and 4 costs 24% on the 139 MB shape.
# * Latency-bound (<= ~0.5 Mi elements, so under ~1 MB moved). Here the pass is
#   short enough to disappear entirely behind the producer, and what matters is
#   finishing fast, not per-thread memory-level parallelism: 512/(32*4) = 4
#   elem/thread spreads the same bytes over 4x the CTAs. Measured on the 262144
#   -element shape over 6 independent processes, BLOCK=512 landed at 7.2 us
#   (1.56-1.58x) every time while BLOCK=2048 landed at 9.0 us (1.24x) every time.
#
# The boundary is not sharp -- every size from ~393k to ~1M elements measures the
# same under both configs -- so it sits in the middle of that insensitive band.
_BLOCK = 2048
_WARPS = 4
_SMALL_BLOCK = 512
_SMALL_WARPS = 4
_SMALL_N = 524288

_FAST_DTYPES = (torch.float16, torch.bfloat16)

# Tile order. The read set (X) and the write set (Y) are each ``n*itemsize``
# bytes, so L2 only comes under eviction pressure once ``2*n*itemsize`` exceeds
# its capacity -- below that the whole working set stays resident and the order
# is irrelevant (measured: identical at 72 MB, and reverse *loses* ~6% at 89-102
# MB where forward already hits in L2). Above it, reverse wins 2-4%. The
# crossover measured on B200 (126.5 MB L2) sits between a 102 MB and a 126 MB
# working set, i.e. at capacity.
_l2_bytes: dict[int, int] = {}


def _reverse_tiles(x: torch.Tensor, n: int) -> bool:
    dev = x.device.index if x.device.index is not None else 0
    cap = _l2_bytes.get(dev)
    if cap is None:
        try:
            cap = int(torch.cuda.get_device_properties(dev).L2_cache_size)
        except Exception:
            cap = 1 << 30  # unknown: never reverse
        _l2_bytes[dev] = cap
    return 2 * n * x.element_size() > cap


@triton.jit
def _gelu_fast(x):
    """fp16/bf16 exact mode: one SFU op, error well inside an output ULP."""
    u = tl.minimum(x * x, _UCAP)
    p = (_A5 * u + _A3) * u + _A1
    h = 0.5 * x
    t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x * p],
                                  dtype=tl.float32, is_pure=True, pack=1)
    return h * t + h


@triton.jit
def _gelu_fast_tanh(x):
    """fp16/bf16 tanh mode: torch's polynomial, same single SFU op."""
    p = _T3 * (x * x) + _T1
    h = 0.5 * x
    t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x * p],
                                  dtype=tl.float32, is_pure=True, pack=1)
    return h * t + h


@triton.jit
def _gelu(x, TANH: tl.constexpr):
    xf = x.to(tl.float32)
    if TANH:
        return _gelu_fast_tanh(xf)
    return _gelu_fast(xf)


@triton.jit
def _gelu_kernel(X, Y, n, TANH: tl.constexpr, EXACT_TILES: tl.constexpr,
                 BLOCK: tl.constexpr, REVERSE: tl.constexpr):
    """Hot path: fp16/bf16."""
    if REVERSE:
        # X was just written by this op's producer, so L2 holds X's *tail*.
        # Walking tiles high-to-low reads the most-recently-used end first while
        # our own stores evict from the least-recently-used (low) end, so the
        # two fronts meet in the middle; walking low-to-high instead means the
        # read front is forever chasing lines its own stores just evicted.
        # `tl.num_programs` keeps this out of the specialization key, so all
        # shapes still share one compile.
        pid = tl.num_programs(0) - 1 - tl.program_id(0)
    else:
        pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    # The producer of X may still be draining; wait before reading its output.
    gdc_wait()
    if EXACT_TILES:
        x = tl.load(X + off)
        tl.store(Y + off, _gelu(x, TANH).to(X.dtype.element_ty))
    else:
        m = off < n
        x = tl.load(X + off, mask=m)
        tl.store(Y + off, _gelu(x, TANH).to(X.dtype.element_ty), mask=m)


@triton.jit
def _gelu_kernel_hp(X, Y, n, TANH: tl.constexpr, BLOCK: tl.constexpr):
    """fp32/fp64: (1e-5, 1e-3) is far tighter than the fitted polynomial above,
    so use the accurate intrinsics. Kept as a separate kernel because Triton
    unifies the return type across a helper's branches, and because no captured
    shape reaches it -- only correctness matters here, not speed."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    gdc_wait()
    x = tl.load(X + off, mask=m)
    if TANH:
        h = 0.5 * x
        y = h * libdevice.tanh(_T1 * x + _T3 * x * x * x) + h
    else:
        y = 0.5 * x * (1.0 + libdevice.erf(x * _RSQRT2))
    tl.store(Y + off, y.to(X.dtype.element_ty), mask=m)


class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        self._tanh = approximate == "tanh"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # empty_like reproduces exactly the layout F.gelu returns: the input's
        # own strides when it is non-overlapping and dense (contiguous,
        # transposed, channels_last), and contiguous otherwise.
        y = torch.empty_like(x)
        if x.stride() != y.stride():
            # Overlapping or gappy (broadcast view, strided slice): y is
            # contiguous, so densify x into matching logical order.
            x = x.contiguous()
        n = x.numel()
        if n == 0:
            return y
        if x.dtype in _FAST_DTYPES:
            block, warps = ((_SMALL_BLOCK, _SMALL_WARPS) if n <= _SMALL_N
                            else (_BLOCK, _WARPS))
            _gelu_kernel[((n + block - 1) // block,)](
                x, y, n, self._tanh, n % block == 0, block,
                _reverse_tiles(x, n), num_warps=warps, launch_pdl=True)
            return y
        if x.dtype not in (torch.float32, torch.float64):
            # Match F.gelu, which has no kernel for integer or fp8 inputs.
            raise NotImplementedError(
                f'"GeluCUDAKernelImpl" not implemented for {x.dtype}')
        _gelu_kernel_hp[((n + _BLOCK - 1) // _BLOCK,)](
            x, y, n, self._tanh, _BLOCK, num_warps=_WARPS, launch_pdl=True)
        return y
