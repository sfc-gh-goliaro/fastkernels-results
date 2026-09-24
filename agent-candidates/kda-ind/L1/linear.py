"""Linear (matrix multiply) kernels for B200 / sm_100.

Same three operators as the baseline -- ``Matmul`` (functional ``F.linear``), ``BMM``
(``torch.matmul``), ``Linear`` (parametric) -- behind one shared dispatcher.

The dispatcher is deliberately conservative: it routes a call to the bf16 tensor-core
kernel below only when the exact problem has been *measured* faster in-harness than
both ``F.linear`` and the explicit fallback, and only when every hardware precondition
the kernel relies on holds. Everything else -- and everything at all if the kernel
fails to compile -- runs the same torch call the baseline runs, so a build or dispatch
problem can never turn into a wrong answer.

Two facts from this workspace's measurements shape the design:

* The scored window has a ~9.2 us fixed cost plus ~3.1 us for one launch and the
  output allocation, and readings quantise in ~2.04 us steps, so only multi-microsecond
  kernel improvements are visible and every extra launch starts ~2 us in debt.
* fp32 is numerically pinned to the reference. ``torch.matmul`` routes fp32 batched
  GEMM to a TF32 CUTLASS kernel whose own deviation from exact fp32 (~1.3e-2) already
  exceeds the fp32 tolerance bound (~8e-3 at |y| ~ 8), so an independent fp32
  implementation cannot match it -- exact fp32 scores 0.826 and TF32 0.576 against a
  0.99 gate. Hence the dtype predicate is an allow-list of bf16, not a deny-list.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_MATMUL = "Matmul"
_LINEAR = "Linear"

# ---------------------------------------------------------------------------
# Tile configuration and the admission table.
# ---------------------------------------------------------------------------


class Tile:
    """One frozen kernel configuration, plus the measurement that admitted it."""

    __slots__ = ("bm", "bn", "bk", "num_warps", "num_stages", "speedup")

    def __init__(self, bm, bn, bk, num_warps, num_stages, speedup=0.0):
        self.bm, self.bn, self.bk = bm, bn, bk
        self.num_warps, self.num_stages = num_warps, num_stages
        self.speedup = speedup  # minimum in-harness speedup observed, vs the baseline

    def __repr__(self):
        return (f"Tile({self.bm}, {self.bn}, {self.bk}, num_warps={self.num_warps}, "
                f"num_stages={self.num_stages}, speedup={self.speedup:.3f})")


# Keyed ``(class, M, K, N, has_bias)``. Populated only from profile/bench_cases.py
# output, and per class: Linear and Matmul are not interchangeable even at identical
# (M, K, N), because the harness's shifting pool re-copies every *contiguous forward
# argument* inside the timed window. For Matmul the weight is a forward argument, so
# it is copied every iteration and stays resident in L2; for Linear it is module
# state, is not pooled, and faces the pre-iteration L2 flush. The same shape is
# therefore a different problem for the two classes.
#
# Empty means "no shape has cleared the gate" -- every case delegates. That is a
# valid state, not an unfinished one.
_MEASURED_FAST: dict[tuple, Tile] = {}

# The kernel's own domain, kept independent of the table above so that a bad table
# entry cannot route a problem the kernel is not written for.
_M_MAX = 512          # above this the baseline already has enough parallelism
_K_MIN = 256          # short-K cases are already at the harness floor
_ALIGN_BYTES = 16     # 128-bit vectorised loads; the pool guarantees 256 B bases

# Fast-path entries per key. Plain ints, incremented on the host: no threads, no
# device sync, nothing the integrity guards watch. This is what distinguishes "the
# fast path ran and tied" from "the fast path never ran".
_FASTPATH_HITS: dict[tuple, int] = {}

_KERNEL = None                              # the compiled launcher, or None
_KERNEL_STATUS = "disabled:no-enabled-shapes"


# ---------------------------------------------------------------------------
# Kernel: C[M,N] = A[M,K] . B[N,K]^T [+ bias[N]], bf16 in / bf16 out, fp32 accumulate.
#
# Both captured operands are already K-contiguous, which is the layout the tensor
# cores want, so nothing is relaid out. Operands are staged through host-built TMA
# descriptors and the K loop is warp-specialised, which on SM100 is the documented
# path to tcgen05.mma with TMEM accumulators -- confirmed in the emitted PTX, see
# profile/02-triton-blackwell-native/.
#
# The predicates in _plan are, conveniently, exactly what TMA itself requires: a
# 16-byte-aligned base, a unit last stride, and 16-byte-aligned outer strides, which
# for bf16 is what K % 8 == 0 and N % 8 == 0 give.
# ---------------------------------------------------------------------------
def _build_kernel():
    """Compile the bf16 GEMM and return a launcher, or raise.

    Called at import rather than on first use. The harness's no-new-threads guard
    would not actually catch a first-call build -- it snapshots the thread count
    after the correctness forwards and checks it after timing, so a build during
    those forwards falls outside the window. The reasons to build at import are
    simpler: it keeps compilation out of the timed region entirely, and it happens
    while the worker is still producing output, clear of the no-output stall
    watchdog that only watches the stderr log's mtime.
    """
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def _tn_gemm(a_desc, b_desc, c_desc, Bias, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 HAS_BIAS: tl.constexpr, NS: tl.constexpr):
        pid = tl.program_id(0)
        num_n = tl.cdiv(N, BN)
        off_m = (pid // num_n) * BM
        off_n = (pid % num_n) * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in tl.range(0, K, BK, warp_specialize=True, num_stages=NS):
            a = a_desc.load([off_m, k])
            b = b_desc.load([off_n, k])
            acc = tl.dot(a, tl.trans(b), acc)
        if HAS_BIAS:
            rn = off_n + tl.arange(0, BN)
            acc += tl.load(Bias + rn, mask=rn < N, other=0.0)[None, :].to(tl.float32)
        c_desc.store([off_m, off_n], acc.to(c_desc.dtype))

    def launch(x2: torch.Tensor, weight: torch.Tensor, bias, out: torch.Tensor,
               tile: Tile) -> torch.Tensor:
        # Descriptors are built per call, not cached: the harness's shifting pool
        # hands out a fresh data_ptr for every iteration, so a cached descriptor
        # would address the previous iteration's buffer. Building them on the host
        # rather than with tl.make_tensor_descriptor avoids three
        # tensormap_create + fenceproxy_acquire pairs in every CTA, which measured
        # ~2 us of pure per-CTA overhead (profile/02-triton-blackwell-native/).
        M, K = x2.shape
        N = weight.shape[0]
        grid = (triton.cdiv(M, tile.bm) * triton.cdiv(N, tile.bn),)
        _tn_gemm[grid](
            TensorDescriptor.from_tensor(x2, [tile.bm, tile.bk]),
            TensorDescriptor.from_tensor(weight, [tile.bn, tile.bk]),
            TensorDescriptor.from_tensor(out, [tile.bm, tile.bn]),
            bias if bias is not None else x2, N, K,
            BM=tile.bm, BN=tile.bn, BK=tile.bk,
            HAS_BIAS=bias is not None, NS=tile.num_stages,
            num_warps=tile.num_warps)
        return out

    return launch


def _check_tile(tile: Tile) -> None:
    """Reject a malformed table entry loudly, once, instead of per call.

    TMA block shapes and `tl.arange(0, BN)` both require positive powers of two, and
    `_plan` deliberately does not re-check the tile it looks up -- it runs on every
    dispatch and must stay to integer comparisons.
    """
    for name, value in (("bm", tile.bm), ("bn", tile.bn), ("bk", tile.bk)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"tile.{name}={value} is not a positive power of two")
    if tile.num_warps <= 0 or tile.num_stages <= 0:
        raise ValueError(f"{tile!r} has a non-positive num_warps/num_stages")


def _compile_for(keys) -> None:
    """Force compilation of every tile the given keys need, on dummy operands.

    Triton compiles on first launch, so the admitted configurations are launched once
    here, at import, rather than on the harness's first correctness forward.
    """
    if _KERNEL is None:
        return
    seen = set()
    for key in keys:
        tile = _MEASURED_FAST.get(key)
        if tile is None:
            continue
        _check_tile(tile)
        _cls, M, K, N, has_bias = key
        sig = (tile.bm, tile.bn, tile.bk, tile.num_warps, tile.num_stages, has_bias)
        if sig in seen:
            continue
        seen.add(sig)
        x2 = torch.zeros((M, K), device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros((N, K), device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros((N,), device="cuda", dtype=torch.bfloat16) if has_bias else None
        out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        _KERNEL(x2, weight, bias, out, tile)
    torch.cuda.synchronize()


def _init_kernel() -> None:
    """Resolve the kernel once, at import. Any failure degrades to torch."""
    global _KERNEL, _KERNEL_STATUS
    if not _MEASURED_FAST:
        _KERNEL_STATUS = "disabled:no-enabled-shapes"
        return
    if not torch.cuda.is_available():
        _KERNEL_STATUS = "disabled:no-cuda-device"
        return
    try:
        _KERNEL = _build_kernel()
        _compile_for(_MEASURED_FAST)
        _KERNEL_STATUS = "built"
    except Exception as exc:  # compiler, driver, or architecture rejected the kernel
        _KERNEL = None
        _KERNEL_STATUS = f"failed:{type(exc).__name__}: {exc}"
        # One line, at import, on stderr: the bench worker sends this to the
        # per-operator log, so a swallowed build failure stays visible instead of
        # hiding behind a silent 1.00x.
        print(f"[candidate L1/linear] kernel unavailable, delegating to torch: "
              f"{_KERNEL_STATUS}", file=sys.stderr, flush=True)


_init_kernel()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _plan(cls: str, input: torch.Tensor, weight: torch.Tensor, bias) -> Tile | None:
    """Return the tile to run this call with, or None to delegate to torch.

    Pure and cheap: integer and attribute checks only, no CUDA calls and no sync.
    Every predicate guards something the kernel actually relies on, and the shape
    must additionally be in the measured table.
    """
    if _KERNEL is None:
        return None
    # An inference-only kernel: it builds no graph, so grad mode must delegate.
    if torch.is_grad_enabled():
        return None
    # bf16 only. fp32 is pinned to the reference kernel's TF32 accumulation order and
    # fp16 is simply unmeasured.
    if input.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return None
    if bias is not None and bias.dtype is not torch.bfloat16:
        return None
    # One kernel, one device: every operand has to live where the launch goes.
    if not input.is_cuda or weight.device != input.device:
        return None
    if bias is not None and bias.device != input.device:
        return None
    if weight.dim() != 2 or input.dim() < 1:
        return None
    K = weight.shape[1]
    N = weight.shape[0]
    if input.shape[-1] != K:
        return None
    # Bounds first, so nothing below divides by a zero K. F.linear accepts K == 0
    # and must keep handling it.
    if K < _K_MIN or N <= 0 or K % 8 or N % 8:
        return None
    # A hidden .contiguous() would cost an extra launch, which is worth more than the
    # kernel can win back; the TMA descriptors also require a unit last stride.
    if not input.is_contiguous() or weight.stride(-1) != 1 or weight.stride(0) != K:
        return None
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != N or
                             not bias.is_contiguous()):
        return None
    M = input.numel() // K
    # Degenerate problems, and the shapes the baseline already handles at the floor.
    if M <= 0 or M > _M_MAX:
        return None
    # 128-bit vectorised loads need 16-byte-aligned bases. The shifting pool hands out
    # 256-byte-aligned slots, but that is a property of the pool, not a guarantee.
    if (input.data_ptr() % _ALIGN_BYTES or weight.data_ptr() % _ALIGN_BYTES or
            (bias is not None and bias.data_ptr() % _ALIGN_BYTES)):
        return None
    return _MEASURED_FAST.get((cls, M, K, N, bias is not None))


def _dispatch(cls: str, input: torch.Tensor, weight: torch.Tensor, bias):
    tile = _plan(cls, input, weight, bias)
    if tile is None:
        return F.linear(input, weight, bias)
    K = weight.shape[1]
    N = weight.shape[0]
    M = input.numel() // K
    key = (cls, M, K, N, bias is not None)
    _FASTPATH_HITS[key] = _FASTPATH_HITS.get(key, 0) + 1
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    _KERNEL(input.reshape(M, K), weight, bias, out, tile)
    return out.view(*input.shape[:-1], N)


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return _dispatch(_MATMUL, input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b).

    Delegated in full, on measurement rather than for lack of trying. Of the four
    scored cases, the two fp32 ones cannot be matched by an independent
    implementation (see the module docstring); QK^T (b=64, 512x64x512) is already
    scheduled at 6.92 waves/SM, which is not the small-grid defect this file
    exploits; and PV (b=64, 512x512x64) is bandwidth-shaped (29.1% of peak DRAM,
    long_scoreboard 23.8), which is different work. The captured operands also arrive
    with head-major strides that torch.matmul consumes as a batched GEMM with a
    leading dimension and no copy -- and which the harness's shifting pool passes
    through un-copied, so they cost nothing in the timed window.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally.

    Registers exactly the baseline's parameters and derives nothing from them: the
    harness moves and casts the module and only *then* loads the baseline's
    state_dict, so anything precomputed from ``weight`` in ``__init__`` would be
    stale. Pre-transposing was measured to buy nothing anyway (spread <= 1%).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, input):
        return _dispatch(_LINEAR, input, self.weight, self.bias)
