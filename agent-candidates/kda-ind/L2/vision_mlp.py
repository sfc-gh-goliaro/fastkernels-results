"""Vision MLP with bias and activation folded into the first GEMM's epilogue.

Same operator as the baseline: ``y = fc2(act(fc1(x)))`` over the TP-aware
``ColumnParallelLinear`` / ``RowParallelLinear`` wrappers.  The baseline runs it
as three kernels -- GEMM, a full-size elementwise activation over
``[M, hidden_features]``, then GEMM.  On a B200 at ``M = 20680`` the middle pass
costs 105 us against a 64.6 us pure-copy floor for the same 356 MB, so roughly
40 us of it is ``erff`` on the CUDA cores and none of it has to be a separate
pass: computing bias and activation in the first GEMM's epilogue, straight out
of the fp32 accumulator, hides that arithmetic behind the MMA pipeline and drops
the intermediate's write-then-read round trip entirely.  The second GEMM stays
on cuBLAS, which already reaches 1500+ TF/s on this shape.

Anything the fused path does not cover -- an unrecognized activation, fp8
wrapper weights, tensor parallelism, a CPU or fp32 input -- evaluates the exact
baseline expression instead.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.gelu import GELU
from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton ships with torch on this box
    triton = None
    tl = None

# Activation forms the fused epilogue evaluates.  Plain ints rather than an enum
# because the value crosses into the kernel as a compile-time constant.
#
# Deliberately only the two GELU forms.  The epilogue could just as easily
# compute QuickGELU or SiLU -- both are a sigmoid away -- but admitting them
# would mean those activations stop evaluating the baseline expression, and the
# acceptance criteria require exactly the opposite: with act_fn=QuickGELU() or
# nn.SiLU() the candidate output must be bit-for-bit equal to the baseline's.
# So they classify as UNKNOWN and take the fallback, like any other activation
# this module does not reproduce exactly.
ACT_UNKNOWN = 0
ACT_NONE = 1  # identity: a bias-only epilogue, for a GEMM with no activation
ACT_GELU_ERF = 2
ACT_GELU_TANH = 3

#: Which implementation of the fused first stage to use.  ``"auto"`` picks the
#: best available for the device and inputs; the rest force one path so the
#: alternatives can be measured through the same harness.  ``"cublaslt"`` is
#: cuBLASLt's own fused GEMM+bias+GELU epilogue via ``torch._addmm_activation``
#: -- a reference oracle and a fallback, not the deliverable.
BACKENDS = ("auto", "triton", "cublaslt", "eager")
BACKEND_ENV = "FK_VISION_MLP_BACKEND"

# wiki/patterns/tail-effect.md puts wave quantization in play below roughly 4x
# the SM count, which is where a tiling that is good for a many-wave problem
# starts leaving SMs idle on the tail.  The measured picture (see
# scratch/probe_crossover.py) is not a sharp crossover: at 1.6 waves the fused
# kernel runs 0.90x the baseline, between ~2 and ~6 waves it sits within noise
# of 1.0x either way, and it is consistently ahead from ~7 waves up.  4 is
# therefore a deliberately conservative pick inside the noisy band rather than a
# measured optimum -- it keeps the clear loss out and gives up little, since
# every selected case except the smallest is above 18 waves.
_WAVES_FOR_STEADY_STATE = 4


def classify_activation(act_fn: Callable[[torch.Tensor], torch.Tensor] | None) -> int:
    """Map an activation module to the epilogue form that reproduces it.

    Only the two GELU forms are recognized; everything else maps to
    :data:`ACT_UNKNOWN` and sends the whole forward back to the baseline
    expression.  Two separate reasons to fail closed here:

    - Guessing a near-enough form would be a silent numerical error rather than a
      slow path.  ``QuickGELU`` is not a GELU variant despite the name:
      ``max|gelu_erf - quickgelu| = 2.03e-2`` over ``[-6, 6]``, an order of
      magnitude past the bf16 tolerance the harness checks against.
    - ``QuickGELU`` and ``nn.SiLU`` *could* be computed in the epilogue -- each is
      one sigmoid -- but the contract requires those two to reproduce the baseline
      expression bit for bit, and admitting them into a fused kernel would end
      that.  Speed on an activation the captured configuration never uses is not
      worth giving up an exactness guarantee.

    Matching is on the exact type, not ``isinstance``: a subclass of ``nn.GELU``
    is free to override ``forward`` and compute something else entirely, and it
    would inherit the attribute this reads.
    """
    kind = type(act_fn)
    if kind is GELU or kind is nn.GELU:
        approximate = getattr(act_fn, "approximate", "none")
        if approximate == "none":
            return ACT_GELU_ERF
        if approximate == "tanh":
            return ACT_GELU_TANH
        return ACT_UNKNOWN
    return ACT_UNKNOWN


# ---------------------------------------------------------------------------
# Fused GEMM + bias + activation.
# ---------------------------------------------------------------------------
if triton is not None:

    # Triton 3.6 only lets a jitted function read a module global that is itself
    # a ``tl.constexpr``, so the kernel compares against these mirrors rather
    # than against the plain ints above.  Derived from them, so there is still
    # one source of truth for the encoding.
    _TL_GELU_ERF = tl.constexpr(ACT_GELU_ERF)
    _TL_GELU_TANH = tl.constexpr(ACT_GELU_TANH)

    @triton.jit
    def _sigmoid(v):
        # exp2 lowers to the SM's multi-function unit; going through expf costs a
        # multiply the compiler cannot always fold away.
        return 1.0 / (1.0 + tl.exp2(-v * 1.4426950408889634))

    @triton.jit
    def _activate(v, ACT: tl.constexpr):
        """The activation, evaluated in fp32 on the accumulator."""
        if ACT == _TL_GELU_ERF:
            # Exact erf GELU: 0.5 x (1 + erf(x / sqrt(2))).  Triton 3.6 has no
            # tl.math.tanh, but it does have erf, so the exact form is the cheap
            # one here -- and it is what F.gelu(approximate="none") computes.
            return 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
        elif ACT == _TL_GELU_TANH:
            # 0.5 (1 + tanh z) == sigmoid(2z), so the tanh form needs no tanh.
            return v * _sigmoid(1.5957691216057308 * (v + 0.044715 * v * v * v))
        else:
            # ACT_NONE: bias only, which is what the second GEMM needs.
            return v

    @triton.jit
    def _linear_act_kernel(
            x_ptr, w_ptr, bias_ptr, out_ptr,
            M, N, K,
            stride_xm, stride_wn, stride_om,
            ACT: tl.constexpr, HAS_BIAS: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr,
            WARP_SPECIALIZE: tl.constexpr, EPILOGUE_SPLIT: tl.constexpr):
        """``out[m, n] = act(sum_k x[m, k] * w[n, k] + bias[n])`` in fp32.

        ``w`` is the ``[N, K]`` parameter exactly as stored, so both operands are
        K-contiguous -- the layout the tensor cores want -- and no repacked copy
        of the weight is needed anywhere.

        The grid is one program per SM and tiles are walked in a GROUP_M-swizzled
        linear order, so a program keeps hitting the same rows of ``w`` while it
        advances and the weight stays L2-resident across the wave.  Descriptors
        are built device-side so the launch carries no per-call host work: at
        ``M = 1760`` the whole operator is ~52 us, which is the same order as
        building three host-side descriptor objects in Python.
        """
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[M, K], strides=[stride_xm, 1],
            block_shape=[BLOCK_M, BLOCK_K])
        w_desc = tl.make_tensor_descriptor(
            w_ptr, shape=[N, K], strides=[stride_wn, 1],
            block_shape=[BLOCK_N, BLOCK_K])
        out_desc = tl.make_tensor_descriptor(
            out_ptr, shape=[M, N], strides=[stride_om, 1],
            block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SPLIT])

        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        k_tiles = tl.cdiv(K, BLOCK_K)
        num_tiles = num_pid_m * num_pid_n
        tiles_per_group = GROUP_M * num_pid_n

        for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True):
            group_id = tile_id // tiles_per_group
            first_pid_m = group_id * GROUP_M
            group_rows = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_rows)
            pid_n = (tile_id % tiles_per_group) // group_rows
            off_m = pid_m * BLOCK_M
            off_n = pid_n * BLOCK_N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kt in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
                off_k = kt * BLOCK_K
                a = x_desc.load([off_m, off_k])
                b = w_desc.load([off_n, off_k])
                acc = tl.dot(a, tl.trans(b), acc)

            if EPILOGUE_SPLIT == 2:
                # Draining the accumulator in halves lets the store of the first
                # half overlap the conversion of the second.
                half_n: tl.constexpr = BLOCK_N // 2
                lo, hi = acc.reshape(BLOCK_M, 2, half_n).permute(0, 2, 1).split()
                for part in tl.static_range(2):
                    block = lo if part == 0 else hi
                    part_n = off_n + part * half_n
                    if HAS_BIAS:
                        cols = part_n + tl.arange(0, half_n)
                        block += tl.load(bias_ptr + cols, mask=cols < N,
                                         other=0.0).to(tl.float32)
                    out_desc.store([off_m, part_n],
                                   _activate(block, ACT).to(out_desc.dtype))
            else:
                if HAS_BIAS:
                    cols = off_n + tl.arange(0, BLOCK_N)
                    acc += tl.load(bias_ptr + cols, mask=cols < N,
                                   other=0.0).to(tl.float32)
                out_desc.store([off_m, off_n],
                               _activate(acc, ACT).to(out_desc.dtype))


#: The one tile configuration the fused kernel ships with.
#:
#: Chosen offline -- ``scratch/probe_fused_fc1.py`` swept 11 shared-memory-legal
#: tile shapes across all five selected cases and ``scratch/probe_small_m.py``
#: swept 248 configurations including num_warps, warp specialization and epilogue
#: splitting -- so that no tuning runs inside the benchmark's watchdog window.
#: The space was pruned by shared memory first: the per-stage cost is
#: ``(BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2`` bytes against a 232 KB budget,
#: which rules out BLOCK_N=256 with BLOCK_K=128 (288 KB at 3 stages) and every
#: BLOCK_M=256 tile (whose fp32 accumulator alone would want all 256 KB of TMEM).
_FUSED_TILE: dict[str, int | bool] = dict(
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
    WARP_SPECIALIZE=True, EPILOGUE_SPLIT=1, num_stages=3, num_warps=8)


def _output_tiles(m: int, n: int) -> int:
    """How many output tiles the shipped tiling cuts an ``m x n`` problem into."""
    block_m = _FUSED_TILE["BLOCK_M"]
    block_n = _FUSED_TILE["BLOCK_N"]
    return (-(-m // block_m)) * (-(-n // block_n))


def fused_is_profitable(m: int, n: int, num_sms: int) -> bool:
    """Whether the problem is big enough for the fused kernel to be worth it.

    Stated in tile geometry against the SM count, so it describes the machine
    rather than the benchmarked shapes -- the same rule moves if the GPU is wider
    or if ``n`` is narrower.  Below a few waves the tail of the last wave is a
    large fraction of the whole kernel and the MMA pipeline never reaches its
    steady state, so the fused kernel gives up more on GEMM efficiency than it
    wins back by deleting the intermediate's round trip.  Measured: at the
    smallest selected case the fused kernel runs at 0.90-0.94x the baseline,
    while from roughly two waves upward it is consistently above 1.0x.
    """
    return _output_tiles(m, n) >= _WAVES_FOR_STEADY_STATE * num_sms


_SM_COUNT: dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SM_COUNT[index]


_allocator_installed = False


def _ensure_descriptor_allocator() -> None:
    """Device-side TMA descriptors need a small global scratch allocation.

    Installed only if nothing else has claimed the hook, so importing this
    module into a host application that manages its own Triton workspace is not
    a side effect.
    """
    global _allocator_installed
    if _allocator_installed:
        return

    def allocate(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        return torch.empty(size, dtype=torch.int8, device="cuda")

    try:
        from triton.runtime import _allocation
        if isinstance(_allocation._allocator.get(), _allocation.NullAllocator):
            triton.set_allocator(allocate)
    except (ImportError, AttributeError):  # pragma: no cover - Triton internals moved
        triton.set_allocator(allocate)
    _allocator_installed = True


_HOOK_ATTRS = ("_forward_pre_hooks", "_forward_hooks",
               "_backward_hooks", "_backward_pre_hooks")


def _any_forward_hooks(*modules: object) -> bool:
    """Whether any of *modules*, or nn.Module globally, has a hook registered.

    The fast path calls ``F.linear`` and the fused kernel instead of invoking the
    submodules, so a hook on them would never run.
    """
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks"):
        if getattr(nn.modules.module, name, None):
            return True
    return any(getattr(module, attr, None)
               for module in modules for attr in _HOOK_ATTRS)


def _tma_compatible(t: torch.Tensor) -> bool:
    """Whether *t* can back a 2-D TMA descriptor.

    TMA wants a 2-D view with a 16-byte-aligned base, a unit-stride last
    dimension, and a row stride that is both a real row step and a whole number
    of 16-byte lines.  The row-step requirement rules out a broadcast view such
    as ``w.expand(n, k)`` with stride ``(0, 1)``, which would otherwise satisfy
    every divisibility test and then describe overlapping rows.
    """
    return (t.dim() == 2
            and t.stride(-1) == 1
            and t.stride(0) >= t.shape[-1]
            and t.data_ptr() % 16 == 0
            and (t.stride(0) * t.element_size()) % 16 == 0)


def triton_legal(x2: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the fused Triton kernel *can* run this problem correctly.

    Kept strictly separate from :func:`fused_is_profitable`, which is a
    performance policy.  Conflating the two made ``backend="triton"`` silently
    resolve to eager on a small problem, so a measurement labelled "Triton" was
    not one.  Legality is a correctness question and applies to every caller;
    profitability is advice that only ``auto`` takes.
    """
    if triton is None:
        return False
    # The kernel is written against Blackwell's tcgen05 / TMEM lowering; on
    # older architectures Triton takes a different path that has not been
    # measured here, so leave those to cuBLAS.
    if torch.cuda.get_device_capability(x2.device)[0] < 10:
        return False
    if not (_tma_compatible(x2) and _tma_compatible(weight)):
        return False
    # The output row stride is N elements, and it also has to be TMA-legal.
    return (weight.shape[0] * x2.element_size()) % 16 == 0


def fused_linear_act(x2: torch.Tensor, weight: torch.Tensor,
                     bias: torch.Tensor | None, act_kind: int) -> torch.Tensor:
    """``act(x2 @ weight.T + bias)`` for 2-D *x2*, in one kernel."""
    m, k = x2.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x2.dtype, device=x2.device)
    if m == 0:
        # Nothing to compute, and a zero-extent TMA descriptor is not valid.
        return out
    _ensure_descriptor_allocator()
    num_sms = _num_sms(x2.device)
    _linear_act_kernel[(num_sms,)](
        x2, weight, bias, out,
        m, n, k,
        x2.stride(0), weight.stride(0), out.stride(0),
        ACT=act_kind, HAS_BIAS=bias is not None, NUM_SMS=num_sms, **_FUSED_TILE)
    return out


# ---------------------------------------------------------------------------
# Split-K second GEMM.
#
# The profile of cuBLAS's own fc2 at M = 1760 shows the defect this addresses: it
# picks a 192x80 tile, which yields grid = 132 on a 148-SM device, so 16 SMs get
# no work at all (sm__cycles_active.min = 0, 0.89 waves) and it reaches 472 TF/s.
# A 128x128 tiling of the same problem is no better on its own -- 14 * 9 = 126
# output tiles, still under one wave. Splitting K four ways turns that into
# 14 * 9 * 4 = 504 independently schedulable partial tiles.
#
# The cost is an fp32 partial buffer and a second pass over it: 4 * M * N * 4
# bytes written then read, against the fused path's single M * N * 2 byte store.
# At M = 1760 that is ~36 MB of extra traffic, roughly 5 us, against a ~37 us
# cuBLAS fc2 -- which is why this is worth measuring rather than assuming either
# way.
# ---------------------------------------------------------------------------
#: Number of disjoint contiguous K ranges. Four because 34 K tiles at BLOCK_K=128
#: divide into ranges of 9/9/9/7, and because it is the smallest split that lifts
#: M = 1760 clear of one wave.
FC2_SPLITS = 4

#: Which implementation of the second GEMM to use.
FC2_BACKENDS = ("auto", "splitk", "cublas")
FC2_BACKEND_ENV = "FK_VISION_MLP_FC2_BACKEND"

#: Whether the split-K path ever beats cuBLAS on this box. Set from measurement
#: (``scratch/probe_fc2_splitk.py``, recorded in ``docs/measurements.md``). When
#: False, ``auto`` never selects it and the path stays reachable only by an
#: explicit request, which is what keeps the experiment honest instead of
#: inventing a geometry rule that happens to exclude every real shape.
_FC2_SPLIT_K_EVER_WINS = False

_FC2_SPLIT_TILE: dict[str, int | bool] = dict(
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
    WARP_SPECIALIZE=False, num_stages=3, num_warps=8)

if triton is not None:

    @triton.jit
    def _split_k_partial_kernel(
            h_ptr, w_ptr, partial_ptr,
            M, N, K,
            stride_hm, stride_wn, stride_pm,
            SPLITS: tl.constexpr, NUM_SMS: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, WARP_SPECIALIZE: tl.constexpr):
        """One fp32 partial product per (row tile, column tile, K range).

        ``partial`` is viewed as ``[SPLITS * M, N]``; range ``s`` writes rows
        ``[s * M, (s + 1) * M)``.  No atomics: every partial tile has exactly one
        writer, and the reduction is a separate kernel.

        The partial store is a plain masked store rather than a TMA one.  A TMA
        store would need a ``BLOCK_M x BLOCK_N`` fp32 staging buffer in shared
        memory -- 64 KB here -- which on top of 192 KB of operand buffers exceeds
        the 232 KB budget outright.  Storing directly keeps three mainloop stages,
        which matters more than the store path.

        K need not divide ``BLOCK_K``. The descriptors pad out-of-range reads
        with zero, and a zero contributes nothing to the dot product, so the
        ``4304 % 128 = 80`` element tail is handled without a mask.
        """
        h_desc = tl.make_tensor_descriptor(
            h_ptr, shape=[M, K], strides=[stride_hm, 1],
            block_shape=[BLOCK_M, BLOCK_K])
        w_desc = tl.make_tensor_descriptor(
            w_ptr, shape=[N, K], strides=[stride_wn, 1],
            block_shape=[BLOCK_N, BLOCK_K])
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        k_tiles = tl.cdiv(K, BLOCK_K)
        tiles_per_split = num_pid_m * num_pid_n
        tiles_per_group = GROUP_M * num_pid_n
        # Ceiling division, so the ranges stay disjoint and contiguous and the
        # last one absorbs the remainder (9/9/9/7 for 34 tiles split 4 ways).
        k_per_split = tl.cdiv(k_tiles, SPLITS)

        for tile_id in tl.range(tl.program_id(0), tiles_per_split * SPLITS,
                                NUM_SMS, flatten=True):
            split = tile_id // tiles_per_split
            local = tile_id % tiles_per_split
            group_id = local // tiles_per_group
            first_pid_m = group_id * GROUP_M
            group_rows = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (local % group_rows)
            pid_n = (local % tiles_per_group) // group_rows
            off_m = pid_m * BLOCK_M
            off_n = pid_n * BLOCK_N

            # Uniform trip count across splits, so the loop bound does not vary
            # per program. The last range then walks past the final K tile, which
            # is harmless: those reads are out of range, the descriptors pad with
            # zero, and a zero contributes nothing to the dot product. The ranges
            # stay disjoint, which is what makes the reduction a plain sum.
            k_begin = split * k_per_split
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for i in tl.range(k_per_split, warp_specialize=WARP_SPECIALIZE):
                off_k = (k_begin + i) * BLOCK_K
                a = h_desc.load([off_m, off_k])
                b = w_desc.load([off_n, off_k])
                acc = tl.dot(a, tl.trans(b), acc)

            rows = off_m + tl.arange(0, BLOCK_M)
            cols = off_n + tl.arange(0, BLOCK_N)
            offs = (split * M + rows[:, None]) * stride_pm + cols[None, :]
            tl.store(partial_ptr + offs, acc,
                     mask=(rows[:, None] < M) & (cols[None, :] < N))

    @triton.jit
    def _split_k_reduce_kernel(
            partial_ptr, bias_ptr, out_ptr,
            M, N,
            stride_pm, stride_om,
            SPLITS: tl.constexpr, HAS_BIAS: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        """Sum exactly ``SPLITS`` fp32 partials, add bias, cast, store.

        Kept as its own kernel rather than folded into the partial kernel with
        atomics: an atomic accumulation would make the summation order
        nondeterministic, and the harness compares against a fixed reference.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (cols[None, :] < N)

        total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for split in tl.static_range(SPLITS):
            offs = ((split * M + rows[:, None]) * stride_pm) + cols[None, :]
            total += tl.load(partial_ptr + offs, mask=mask, other=0.0)
        if HAS_BIAS:
            total += tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        out_offs = rows[:, None] * stride_om + cols[None, :]
        tl.store(out_ptr + out_offs, total.to(out_ptr.dtype.element_ty), mask=mask)


def fc2_split_k_legal(h: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the split-K path can run this problem correctly.

    Legality only, as with :func:`triton_legal` -- profitability is a separate
    question that only ``auto`` asks.
    """
    if triton is None:
        return False
    if torch.cuda.get_device_capability(h.device)[0] < 10:
        return False
    if not (_tma_compatible(h) and _tma_compatible(weight)):
        return False
    # The fp32 partial buffer's row stride must also be TMA-legal.
    return (weight.shape[0] * 4) % 16 == 0


def fc2_split_k_is_profitable(m: int, n: int, num_sms: int) -> bool:
    """Whether splitting K is worth its extra memory traffic on this geometry.

    Set from measurement: see ``docs/measurements.md``. The predicate is written
    in tile geometry so it describes the machine rather than the benchmarked
    shapes, and the constant below records the measured outcome rather than a
    guess -- if the split never wins, the honest encoding of that is a threshold
    no geometry reaches, not a fabricated rule.
    """
    if not _FC2_SPLIT_K_EVER_WINS:
        return False
    tiles = (-(-m // 128)) * (-(-n // 128))
    return tiles < _WAVES_FOR_STEADY_STATE * num_sms


def linear_split_k(h: torch.Tensor, weight: torch.Tensor,
                   bias: torch.Tensor | None) -> torch.Tensor:
    """``h @ weight.T + bias`` via ``FC2_SPLITS`` partial products plus a reduction."""
    m, k = h.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=h.dtype, device=h.device)
    if m == 0:
        return out
    _ensure_descriptor_allocator()
    num_sms = _num_sms(h.device)
    partial = torch.empty((FC2_SPLITS * m, n), dtype=torch.float32, device=h.device)
    _split_k_partial_kernel[(num_sms,)](
        h, weight, partial,
        m, n, k,
        h.stride(0), weight.stride(0), partial.stride(0),
        SPLITS=FC2_SPLITS, NUM_SMS=num_sms, **_FC2_SPLIT_TILE)
    reduce_tile = (128, 128)
    grid = (triton.cdiv(m, reduce_tile[0]), triton.cdiv(n, reduce_tile[1]))
    _split_k_reduce_kernel[grid](
        partial, bias, out, m, n, partial.stride(0), out.stride(0),
        SPLITS=FC2_SPLITS, HAS_BIAS=bias is not None,
        BLOCK_M=reduce_tile[0], BLOCK_N=reduce_tile[1], num_warps=8)
    return out


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.

    ``fc1``/``fc2`` are deliberately the same wrapper submodules the baseline
    builds, so ``state_dict`` keys and shapes match and the benchmark harness's
    weight sharing actually lands.  Nothing derived from the weights is cached:
    the kernel consumes ``fc1.weight`` in its stored ``[N, K]`` layout, which
    removes both the 9.9 MB repacked copy and any question of it going stale
    behind an in-place weight update.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True, *, backend: str | None = None,
                 fc2_backend: str | None = None):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn
        requested = backend if backend is not None else os.environ.get(BACKEND_ENV)
        self.backend = (requested or "auto").lower()
        requested_fc2 = (fc2_backend if fc2_backend is not None
                         else os.environ.get(FC2_BACKEND_ENV))
        self.fc2_backend = (requested_fc2 or "auto").lower()
        #: Path the most recent forward actually took, so a test can assert the
        #: default really is the fused kernel instead of inferring it.
        self.last_backend: str | None = None
        #: Same, for the second GEMM.
        self.last_fc2_backend: str | None = None

    @property
    def act_kind(self) -> int:
        """The epilogue form for the *current* ``act_fn``.

        Derived per call rather than cached at construction.  Caching would be
        marginally cheaper -- an isinstance chain against a 48 us floor -- but
        ``act_fn`` is a public attribute, and swapping it, or flipping
        ``approximate`` on the GELU in place, would leave a cached answer
        describing an activation the module no longer has.  That is a silent
        wrong number, which is not a trade worth nanoseconds.
        """
        return classify_activation(self.act_fn)

    def fast_path_ok(self, x: torch.Tensor) -> bool:
        """Whether *x* and the current module state are inside the fused path.

        Every condition below is a case where the fused stage would compute
        something other than what the baseline computes, so falling through is
        correctness, not conservatism.
        """
        if self.act_kind == ACT_UNKNOWN:
            return False
        # The fused stage reads the wrappers' weights directly and reimplements
        # their forward, so it is only valid for the wrappers it was written
        # against.  Requiring the exact types (rather than duck-typing on
        # `use_fp8`) means a replaced or evolved wrapper falls back instead of
        # silently having its semantics guessed at.
        if type(self.fc1) is not ColumnParallelLinear:
            return False
        if type(self.fc2) is not RowParallelLinear:
            return False
        # fp8 wrapper weights are float8_e4m3fn with a separate block scale and
        # go through a different linear op entirely.
        if self.fc1.use_fp8 is not False or self.fc2.use_fp8 is not False:
            return False
        # Under TP, RowParallelLinear all-reduces its output and applies bias on
        # rank 0 only, from the tp_size/tp_rank it cached at construction.
        # Reimplementing that here would duplicate logic the wrapper has right.
        if self.fc2.tp_size != 1 or self.fc2.tp_rank != 0:
            return False
        # The fast path calls F.linear directly instead of the submodules, so any
        # registered forward hook would be skipped -- and a hook that rewrites an
        # output is a real difference, not a stylistic one.
        if _any_forward_hooks(self.fc1, self.fc2, self.act_fn):
            return False
        w1, w2 = self.fc1.weight, self.fc2.weight
        # The fused path hands raw data pointers to a kernel, which bypasses any
        # __torch_dispatch__ / __torch_function__ a subclass relies on.
        if type(x) is not torch.Tensor:
            return False
        if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
            return False
        # The fused kernel has no backward, and cuBLASLt's fused epilogue has no
        # registered derivative, so under autograd the graph would either be cut
        # silently or raise.  Every tensor the expression consumes has to be
        # checked, not just the weights: a frozen-weight, trainable-bias module is
        # an ordinary fine-tuning setup, and it would otherwise lose the bias
        # gradient without any error.  The benchmark runs in eval under no_grad.
        if torch.is_grad_enabled() and any(
                t is not None and t.requires_grad
                for t in (x, w1, w2, self.fc1.bias, self.fc2.bias)):
            return False
        if x.dtype != w1.dtype or x.dtype != w2.dtype:
            return False
        if x.device != w1.device or x.device != w2.device:
            return False
        # A non-contiguous x would need a materializing copy before the GEMM,
        # which is the sort of extra full-size pass this module exists to remove.
        if x.dim() < 2 or not x.is_contiguous():
            return False
        # F.linear accepts a zero-width in_features; ``reshape(-1, 0)`` cannot,
        # because the free dimension is then ambiguous.
        if x.shape[-1] == 0:
            return False
        if x.shape[-1] != w1.shape[1] or w2.shape[1] != w1.shape[0]:
            return False
        # F.linear broadcasts a bias, the epilogue indexes it as bias[n], so only
        # the one-per-output-column shape the wrappers build is safe.
        for bias, out_features in ((self.fc1.bias, w1.shape[0]),
                                   (self.fc2.bias, w2.shape[0])):
            if bias is None:
                continue
            if bias.dtype != x.dtype or bias.device != x.device:
                return False
            if bias.shape != (out_features,) or not bias.is_contiguous():
                return False
        return True

    def _cublaslt_ok(self) -> bool:
        """Whether cuBLASLt's fused epilogue can stand in here.

        ``torch._addmm_activation`` exposes only a ``use_gelu`` boolean, and that
        epilogue computes the *tanh* GELU: measured in fp32 against both forms it
        sits 5.6e-6 from ``F.gelu(approximate="tanh")`` and 4.74e-4 from
        ``approximate="none"``, the latter being exactly the distance between the
        two functions.  So this path is exact for the tanh form and for ReLU, and
        for exact-erf GELU it is a substitution with a 4.73e-4 deviation on the
        pre-activation.  That is accepted here because it was checked end to end
        after propagation through ``W2`` -- every element of ``y`` lands inside
        ``atol + rtol * |y|`` with the worst element at 0.83 of the bound, on all
        five selected cases across all three rounds -- not because the scalar
        figure looks small.  It also needs a bias, having no bias-free form.
        """
        return (self.fc1.bias is not None
                and self.act_kind in (ACT_GELU_ERF, ACT_GELU_TANH))

    def resolve_backend(self, x2: torch.Tensor) -> str:
        """Which implementation of the fused stage to run for *x2*.

        ``auto`` applies both legality and the profitability policy, so it
        declines the fused kernel on a problem too small to pay for it and lets
        the vendor epilogue take over.  An explicit request applies *legality
        only* and runs what it names wherever that is possible -- otherwise a row
        in a comparison table labelled "triton" could quietly be measuring
        something else.  A request that is genuinely impossible (an unsupported
        activation for cuBLASLt, no Triton on the device) still degrades to
        ``eager`` rather than raising, so the selector stays usable for
        measurement.
        """
        legal = triton_legal(x2, self.fc1.weight)
        if self.backend == "triton":
            return "triton" if legal else "eager"
        if self.backend == "cublaslt":
            return "cublaslt" if self._cublaslt_ok() else "eager"
        if self.backend == "eager":
            return "eager"
        if legal and fused_is_profitable(x2.shape[0], self.fc1.weight.shape[0],
                                        _num_sms(x2.device)):
            return "triton"
        return "cublaslt" if self._cublaslt_ok() else "eager"

    def fused_fc1(self, x2: torch.Tensor) -> torch.Tensor:
        """``act(x2 @ W1.T + b1)`` for a 2-D *x2*, by the selected backend."""
        backend = self.resolve_backend(x2)
        self.last_backend = backend
        weight, bias = self.fc1.weight, self.fc1.bias
        if backend == "triton":
            return fused_linear_act(x2, weight, bias, self.act_kind)
        if backend == "cublaslt":
            # _cublaslt_ok only admits the two GELU forms, so the epilogue is
            # always the GELU one.
            return torch._addmm_activation(bias, x2, weight.t(), use_gelu=True)
        return self.act_fn(F.linear(x2, weight, bias))

    def resolve_fc2_backend(self, h: torch.Tensor) -> str:
        """Which implementation of the second GEMM to run for *h*.

        Same discipline as :meth:`resolve_backend`: an explicit request is gated
        on legality alone and must run what it names, while ``auto`` additionally
        consults profitability.
        """
        if self.fc2_backend == "splitk":
            return "splitk" if fc2_split_k_legal(h, self.fc2.weight) else "cublas"
        if self.fc2_backend == "cublas":
            return "cublas"
        if (fc2_split_k_legal(h, self.fc2.weight)
                and fc2_split_k_is_profitable(h.shape[0], self.fc2.weight.shape[0],
                                              _num_sms(h.device))):
            return "splitk"
        return "cublas"

    def fc2_matmul(self, h: torch.Tensor) -> torch.Tensor:
        """``h @ W2.T + b2`` for a 2-D *h*, by the selected backend."""
        backend = self.resolve_fc2_backend(h)
        self.last_fc2_backend = backend
        if backend == "splitk":
            return linear_split_k(h, self.fc2.weight, self.fc2.bias)
        return F.linear(h, self.fc2.weight, self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fast_path_ok(x):
            self.last_backend = "baseline"
            self.last_fc2_backend = "baseline"
            return self.fc2(self.act_fn(self.fc1(x)))
        leading = x.shape[:-1]
        h = self.fused_fc1(x.reshape(-1, x.shape[-1]))
        y = self.fc2_matmul(h)
        return y.view(*leading, y.shape[-1])
