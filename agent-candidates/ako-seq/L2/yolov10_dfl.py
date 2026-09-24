"""YOLOv10 Distribution Focal Loss layer -- the whole layer as one Triton launch.

``YOLODFL`` is a softmax over ``c1`` "distribution" bins followed by a frozen
1x1 convolution whose weight is ``arange(c1)``, i.e. an *expectation of the bin
index* under the softmax::

    y[b, j, a] = sum_k w[k] * softmax_k( x[b, j*c1 + k, a] )

Composing the two library ops costs three trips over HBM and two launches on a
problem that is only ~4.3 MB in / ~268 KB out: the softmax materializes a full
permuted intermediate and the 1x1 conv (which falls back to ``F.conv2d``)
re-reads it.  This module fuses everything into a single kernel:

* **One pass, no intermediate.**  For a fixed ``(b, j)`` the ``c1`` bins are
  ``c1`` rows strided by ``stride_c`` while ``a`` stays the contiguous axis, so
  the ``view(b, 4, c1, a).transpose(2, 1)`` of the reference is pure index
  arithmetic -- nothing is permuted or copied.  A program owns a ``BLOCK_A``
  slab of ``a`` for one ``(b, j)`` pair and walks the ``c1`` bins as ``c1``
  fully coalesced ``BLOCK_A``-wide row loads.
* **Softmax and the conv in one accumulation.**  The bin loop keeps three
  fp32 vectors -- running max, ``sum_k e_k`` and ``sum_k w_k e_k`` -- and the
  output is a single divide per *output* element.  The ``c1`` probabilities are
  never normalized individually, and the ``c1``-wide dot is not a GEMM: it is a
  register reduction that belongs next to the exp.
* **Vector-shaped, not tile-shaped.**  The bin loop is fully unrolled over 1-D
  ``BLOCK_A`` vectors rather than expressed as one ``(c1, BLOCK_A)`` tile.  Both
  read the same bytes, but the tile form makes the reduction axis cross threads
  (Triton lands it in shared memory) and is 2-3x slower here; with 1-D vectors
  every max / exp / fma is thread-local.
* **Nothing per call.**  The launch plan is memoized per
  ``(dtype, shape, stride, weight)``, so steady-state ``forward`` is a dict hit,
  one allocation and one launch.  ``c1``, ``a`` and the strides are
  ``constexpr``, which is what lets the bin loop unroll and folds every address
  into a constant multiply; ``w == arange(c1)`` is checked once at plan time and
  then folded in as immediates instead of being re-read per program.
* **Programmatic dependent launch.**  The launch is issued with PDL and opens
  with ``griddepcontrol.wait`` once its own addressing is done, so the grid is
  staged while the producer is still draining.  Measured worth ~2 us here --
  about a fifth of the total, at this traffic volume.

Accumulation is fp32 (fp64 for fp64 in) with the row max subtracted; loads and
stores use the tensor's native dtype.  Anything the fused path does not cover --
a channel count that is not a multiple of ``c1``, a tensor that is not
contiguous along ``a``, a ``c1`` too large to unroll -- falls back to the
reference composition of the L1 kernels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda.gdc import gdc_wait

from ..L1.conv2d import Conv2d
from ..L1.softmax import Softmax

_LOG2E = tl.constexpr(1.4426950408889634)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _dfl_kernel(X, Y, W,
                A: tl.constexpr,          # spatial extent (contiguous axis)
                C1: tl.constexpr,         # DFL bins == conv in-channels
                G: tl.constexpr,          # bin groups (4 for a box)
                SB: tl.constexpr,         # x stride over batch
                SC: tl.constexpr,         # x stride over channel
                BLOCK_A: tl.constexpr,
                EXACT_A: tl.constexpr,
                ARANGE: tl.constexpr,     # weight is arange(C1) -> fold it in
                I64: tl.constexpr,
                EP: tl.constexpr,
                GDC: tl.constexpr,
                ACC: tl.constexpr):
    """One program: the ``BLOCK_A`` slab of ``a`` for one ``(b, j)`` pair.

    ``pid(0)`` is the ``a`` tile, so concurrently-dispatched programs walk
    adjacent addresses.  Pass 1 takes the max over the ``C1`` bins, pass 2
    re-reads them out of L1 and accumulates the softmax denominator and the
    weighted numerator together.  Two passes over registers-worth of L1 beat
    one pass over an online rescale (which needs a second ``exp2`` per bin) and
    beat holding all ``C1`` vectors live.
    """
    pid_a = tl.program_id(0)
    pid_bj = tl.program_id(1)
    off = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    if I64:
        xb = (pid_bj // G).to(tl.int64) * SB + (pid_bj % G).to(tl.int64) * (C1 * SC)
        yb = pid_bj.to(tl.int64) * A
    else:
        xb = (pid_bj // G) * SB + (pid_bj % G) * (C1 * SC)
        yb = pid_bj * A
    xp = X + xb + off

    # Everything above is arithmetic on our own constants; only now wait for the
    # producer, so its tail drains under our prologue.
    if GDC:
        gdc_wait()

    mx = tl.full((BLOCK_A,), float("-inf"), ACC)
    if EXACT_A:
        for k in tl.static_range(C1):
            mx = tl.maximum(mx, tl.load(xp + k * SC, eviction_policy=EP).to(ACC))
        num = tl.zeros((BLOCK_A,), ACC)
        den = tl.zeros((BLOCK_A,), ACC)
        for k in tl.static_range(C1):
            e = tl.exp2((tl.load(xp + k * SC, eviction_policy=EP).to(ACC) - mx) * _LOG2E)
            den += e
            num += (k if ARANGE else tl.load(W + k).to(ACC)) * e
        tl.store(Y + yb + off, (num / den).to(Y.dtype.element_ty))
    else:
        m = off < A
        for k in tl.static_range(C1):
            mx = tl.maximum(mx, tl.load(xp + k * SC, mask=m, other=0.0,
                                        eviction_policy=EP).to(ACC))
        num = tl.zeros((BLOCK_A,), ACC)
        den = tl.zeros((BLOCK_A,), ACC)
        for k in tl.static_range(C1):
            e = tl.exp2((tl.load(xp + k * SC, mask=m, other=0.0,
                                 eviction_policy=EP).to(ACC) - mx) * _LOG2E)
            den += e
            num += (k if ARANGE else tl.load(W + k).to(ACC)) * e
        tl.store(Y + yb + off, (num / den).to(Y.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------
# Host-side launch planning (memoized)
# ---------------------------------------------------------------------------
# ``x`` is read once per pass and the producer has just written it, so there is
# nothing L2 should retain on our behalf.
_EVICT = "evict_first"

# Largest ``c1`` we are willing to unroll; beyond it the reference composition
# is both correct and a better use of compile time.
_MAX_UNROLL = 64

_PDL: bool | None = None


def _pdl_ok(device) -> bool:
    """PDL needs Hopper+; ``gdc_wait`` is what makes the early launch safe."""
    global _PDL
    if _PDL is None:
        try:
            _PDL = torch.cuda.get_device_capability(device)[0] >= 9
        except Exception:
            _PDL = False
    return _PDL


def _cfg(bj: int, a: int) -> tuple[int, int]:
    """(BLOCK_A, num_warps).

    Total work is fixed at ``bj * a`` outputs, so ``BLOCK_A`` only trades grid
    width against per-program width.  Measured on B200: anything that lands the
    grid in the low hundreds of programs sits at the floor, and both ends fall
    off -- too few programs leaves SMs idle, too many (tiny ``BLOCK_A``) makes
    each of the ``c1`` row loads narrower than a sector.  Aim for >= 256
    programs with the widest ``BLOCK_A`` that still gets there.
    """
    block = 512
    while block > 32 and bj * ((a + block - 1) // block) < 256:
        block //= 2
    return block, 8 if block >= 512 else 4


def _acc_dtype(dtype):
    return tl.float64 if dtype is torch.float64 else tl.float32


def _build_plan(dtype, shape, stride, c1, weight):
    """Return ``run(x) -> y``, or ``None`` if the fused path cannot take it."""
    if len(shape) != 3 or stride[2] != 1:
        return None
    b, ch, a = shape
    if c1 <= 0 or c1 > _MAX_UNROLL or ch % c1 != 0 or a <= 0 or b <= 0:
        return None
    g = ch // c1

    w = weight.detach().reshape(-1)
    if w.numel() != c1:
        return None
    arange = bool(torch.equal(w.to(torch.float32),
                              torch.arange(c1, dtype=torch.float32,
                                           device=w.device)))
    wflat = None if arange else w.contiguous()

    bj = b * g
    block, warps = _cfg(bj, a)
    grid = ((a + block - 1) // block, bj)
    numel = b * ch * a
    kwargs = dict(A=a, C1=c1, G=g, SB=stride[0], SC=stride[1],
                  BLOCK_A=block, EXACT_A=(a % block == 0), ARANGE=arange,
                  I64=numel > 0x7FFFFFFF, EP=_EVICT, GDC=_pdl_ok(weight.device),
                  ACC=_acc_dtype(dtype), launch_pdl=_pdl_ok(weight.device),
                  num_warps=warps, num_stages=1)
    oshape = (b, g, a)

    def run(x, _g=grid, _kw=kwargs, _s=oshape, _w=wflat):
        y = torch.empty(_s, dtype=x.dtype, device=x.device)
        _dfl_kernel[_g](x, y, _w, **_kw)
        return y

    return run


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)
        self._cache: dict = {}

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(
            self._softmax(x.view(b, 4, self.c1, a).transpose(2, 1))).view(b, 4, a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``_version`` / ``data_ptr`` in the key means a re-loaded weight
        # (``load_state_dict``) invalidates the folded-in ``arange`` decision.
        w = self.conv.weight
        key = (x.dtype, tuple(x.shape), x.stride(), w.dtype, w.data_ptr(),
               w._version)
        run = self._cache.get(key)
        if run is None:
            try:
                run = _build_plan(x.dtype, tuple(x.shape), x.stride(), self.c1, w)
            except Exception:
                run = None
            if run is None:
                run = self._reference
            self._cache[key] = run
        return run(x)
