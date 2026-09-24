"""Vision MLP for Qwen vision transformer blocks -- fused Triton implementation.

Unified across Qwen2-VL (QuickGELU) and Qwen3-VL (SiLU) activations.

The eager module makes three passes over the hidden tensor: ``fc1`` writes
``h[M, hidden]``, ``act_fn`` reads it and writes ``g``, ``fc2`` reads ``g``. At
the captured sizes (M = 1.8k-65k, in=1152, hidden=4304) that tensor is 15-560 MB,
so the activation's round-trip is the single biggest line item in the op --
profiled on a B200 at M=20680 the baseline spends 183 us in fc1, **168 us in the
activation** and 168 us in fc2. The activation moves no bytes the GEMMs don't
already move; it just moves them two more times.

So it is folded into the first GEMM's epilogue, while the fp32 accumulator tile
is still in registers. Three kernels become two and two full passes over
``hidden`` disappear from the memory traffic.

Both GEMMs run through one Blackwell persistent-TMA kernel: one program per SM
walking a grouped tile order, ``tcgen05`` MMA behind a warp-specialized
producer/consumer pipeline, and the epilogue of tile *i* overlapping the
prologue of tile *i+1*. The ``B`` operand is fed in whichever of the two
K-contiguous orientations measured faster for that GEMM's shape (see the
``_FC1_*`` / ``_FC2_*`` tables), which for ``fc2`` means a transposed copy of the
weight -- built once and cached, since weights are constant.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# ---------------------------------------------------------------------------
# Activation codes. Every one of them is a sigmoid in disguise, so the epilogue
# evaluates them all as ``x / (1 + exp2(e))`` -- one ``ex2.approx.f32`` plus a
# ``div.full.f32``, which is the cheapest form available:
#
#   x*sigmoid(a*x)                 == x / (1 + exp2(-a*log2(e) * x))
#   0.5x*(1 + tanh(x*(c1+c3*x^2))) == x / (1 + exp2(-2*log2(e)*x*(c1+c3*x^2)))
#
# with ``0.5(1+tanh(p)) == sigmoid(2p)`` linking the gelu forms to the first.
# The cubic's coefficients are the textbook ones for ``approximate="tanh"`` and
# a minimax refit of ``erf(x/sqrt 2)`` for ``approximate="none"`` (max |gelu err|
# 2.7e-4 in fp32 -- inside one bf16 ulp of the activation, and independent per
# hidden unit, so the 4304-term sum in ``fc2`` averages it down further). Both
# cubics increase monotonically with positive coefficients, so the tails
# saturate correctly: large +x drives the exponent to -inf (result -> x) and
# large -x to +inf (exp2 -> inf, x/inf -> 0), with no NaN on either side.
#
# Calling ``erf`` here instead would cost ~25 instructions per element rather
# than ~8, and at 89M hidden elements per call that is the difference between an
# epilogue that hides under the MMA pipeline and one that does not. The two other
# forms that look cheaper are not: ``libdevice.tanh`` does *not* lower to
# ``tanh.approx.f32`` on sm_100 (it expands to ~20 ops), and swapping the divide
# for ``fast_dividef``/``div.approx.f32`` measured 1.4% *slower* on fc1 at both
# M=20680 and M=64680 -- so the plain, correctly-rounded divide stays.
# ---------------------------------------------------------------------------
_ACT_NONE = 0
_ACT_GELU = 1        # F.gelu(approximate="none")
_ACT_GELU_TANH = 2   # F.gelu(approximate="tanh")
_ACT_SILU = 3        # x * sigmoid(x)
_ACT_QUICKGELU = 4   # x * sigmoid(1.702 x)


# Triton builds its device-side TMA descriptors in a global scratch buffer that
# it asks for on *every* launch. At the small end of the captured shapes a kernel
# is only ~40 us, so a fresh allocation per launch is measurable; hand back a
# per-(device, stream) buffer instead. Reuse is safe because the scratch is
# written and consumed inside the one kernel, and kernels on a stream serialize.
_SCRATCH: dict[tuple[int, int], torch.Tensor] = {}


def _tl_alloc(size: int, alignment: int, stream):
    key = (torch.cuda.current_device(), int(stream) if stream is not None else 0)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < size:
        buf = torch.empty(max(size, 1 << 14), dtype=torch.int8, device="cuda")
        _SCRATCH[key] = buf
    return buf


triton.set_allocator(_tl_alloc)

_NUM_SMS: int | None = None


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _NUM_SMS


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _tile_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M: tl.constexpr):
    """Grouped (L2-friendly) tile order: GROUP_M row-tiles per column sweep."""
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    return (first_pid_m + (tile_id % group_size_m),
            (tile_id % num_pid_in_group) // group_size_m)


@triton.jit
def _activate(x, ACT: tl.constexpr):
    """One of the ``_ACT_*`` codes above, spelled as a literal: a @jit'ed body
    may only read module globals that are declared ``tl.constexpr``."""
    if ACT == 1:      # _ACT_GELU
        return x / (1.0 + tl.exp2(x * (-2.30876537 - 0.10012500 * x * x)))
    elif ACT == 2:    # _ACT_GELU_TANH
        return x / (1.0 + tl.exp2(x * (-2.30220820 - 0.10292110 * x * x)))
    elif ACT == 3:    # _ACT_SILU
        return x / (1.0 + tl.exp2(-1.44269504 * x))
    elif ACT == 4:    # _ACT_QUICKGELU
        return x / (1.0 + tl.exp2(-2.45561981 * x))
    else:             # _ACT_NONE, or an act_fn we did not recognize
        return x


@triton.jit
def _gemm_act_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M,
                     N: tl.constexpr, K: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                     ACT: tl.constexpr, HAS_BIAS: tl.constexpr,
                     B_KMAJOR: tl.constexpr, GRID: tl.constexpr):
    """C[M,N] = act(A[M,K] @ B + bias).

    ``N`` and ``K`` are constexpr: they are fixed by the module's dimensions, and
    pinning them makes the k-loop trip count and the tile counts compile-time
    constants (better pipelining, and a couple of fewer runtime args to hash on
    every launch, which is visible at the small end of the captured shapes).

    ``B_KMAJOR`` picks the weight orientation: ``B`` is ``[N, K]``
    (``nn.Linear``'s own layout, contracted via ``tl.dot(a, b.T)``) when set, and
    ``[K, N]`` otherwise. Both are K-contiguous for TMA; which one wins depends
    on the shape, so the caller chooses per GEMM.
    """
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    # TMA handles the ragged edges: M, N and K need not divide the block shape.
    # The out-of-bounds part of a load box reads as zero -- exactly right for a
    # dot product -- and of a store box is dropped.
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    if B_KMAJOR:
        b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BLOCK_N, BLOCK_K])
    else:
        b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N // 2])

    # The epilogue trails the mainloop by one tile, so storing tile i overlaps
    # the first TMA loads of tile i+1 instead of serializing with them.
    tile_id_c = start_pid - GRID

    for tile_id in tl.range(start_pid, num_tiles, GRID, flatten=True,
                            warp_specialize=True):
        pid_m, pid_n = _tile_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_M)
        off_am = pid_m * BLOCK_M
        off_bn = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            off_k = ki * BLOCK_K
            a = a_desc.load([off_am, off_k])
            if B_KMAJOR:
                acc = tl.dot(a, b_desc.load([off_bn, off_k]).T, acc)
            else:
                acc = tl.dot(a, b_desc.load([off_k, off_bn]), acc)

        tile_id_c += GRID
        pid_m, pid_n = _tile_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_M)
        off_cm = pid_m * BLOCK_M
        off_cn = pid_n * BLOCK_N
        if HAS_BIAS:
            idx = off_cn + tl.arange(0, BLOCK_N)
            acc += tl.load(bias_ptr + idx, mask=idx < N,
                           other=0.0).to(tl.float32)[None, :]
        # Split the accumulator so only half of it is live as fp32 while the
        # other half is already converted and on its way out through TMA.
        halves = tl.permute(tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2)), (0, 2, 1))
        acc0, acc1 = tl.split(halves)
        c_desc.store([off_cm, off_cn], _activate(acc0, ACT).to(dtype))
        c_desc.store([off_cm, off_cn + BLOCK_N // 2],
                     _activate(acc1, ACT).to(dtype))


# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages, num_warps, B_KMAJOR),
# measured on a B200 at the captured shapes (in=1152, hidden=4304).
_FC1_BIG = (128, 256, 64, 8, 3, 8, True)
_FC1_SMALL = (128, 128, 64, 8, 6, 4, True)
_FC2_BIG = (256, 128, 64, 8, 4, 4, False)
_FC2_SMALL = (128, 128, 64, 8, 6, 4, False)
_SMALL_M = 4096


def _gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None,
          act: int, cfg) -> torch.Tensor:
    BM, BN, BK, GM, stages, warps, kmajor = cfg
    M, K = a.shape
    N = b.shape[0] if kmajor else b.shape[1]
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = min(_num_sms(), triton.cdiv(M, BM) * triton.cdiv(N, BN))
    _gemm_act_kernel[(grid,)](
        a, b, out, bias, M, N, K, BM, BN, BK, GM, act, bias is not None,
        kmajor, grid, num_stages=stages, num_warps=warps)
    return out


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
_OK_DTYPES = (torch.bfloat16, torch.float16)


def _act_code(act_fn) -> int:
    """Map ``act_fn`` onto an epilogue code, or ``-1`` if we can't identify it."""
    if isinstance(act_fn, QuickGELU) or type(act_fn).__name__ == "QuickGELU":
        return _ACT_QUICKGELU
    if isinstance(act_fn, nn.SiLU) or act_fn is F.silu:
        return _ACT_SILU
    if isinstance(act_fn, nn.GELU) or type(act_fn).__name__ == "GELU":
        return (_ACT_GELU_TANH if getattr(act_fn, "approximate", "none") == "tanh"
                else _ACT_GELU)
    if act_fn is F.gelu:
        return _ACT_GELU
    return -1


def _row_pitch_ok(t: torch.Tensor) -> bool:
    """TMA needs the row pitch to be a multiple of 16 bytes."""
    return t.shape[-1] % (16 // t.element_size()) == 0


def _tma_ok(t: torch.Tensor) -> bool:
    return t.is_contiguous() and t.data_ptr() % 16 == 0 and _row_pitch_ok(t)


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
        self._act = _act_code(act_fn)
        self._fusable = not (self.fc1.use_fp8 or self.fc2.use_fp8)
        self._w2t_key = None
        self._w2t = None

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._w2t_key = None  # weights changed under us; rebuild the transpose

    def _fc2_weight(self, w2: torch.Tensor) -> torch.Tensor:
        """``[hidden, in]`` copy of ``fc2.weight``, rebuilt only when it changes.

        Keyed on storage identity *and* version counter, so both an in-place
        update and a re-cast that moves the tensor invalidate the cache.
        """
        key = (w2.data_ptr(), w2._version)
        if self._w2t_key != key:
            self._w2t = w2.t().contiguous()
            self._w2t_key = key
        return self._w2t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, w2 = self.fc1.weight, self.fc2.weight
        if (self._fusable and x.is_cuda and x.dtype in _OK_DTYPES
                and w1.dtype == x.dtype and w2.dtype == x.dtype
                and x.ndim >= 2 and x.shape[-1] == w1.shape[1] and x.numel() > 0
                # Five tensors go through TMA; between them these three checks
                # cover every row pitch involved. ``x``/``w1`` give in_features
                # (which is also ``y``'s width and ``w1``-as-B's pitch) and
                # ``w2``'s last dim gives hidden (which is ``h``'s width and the
                # transposed ``w2``'s pitch). Everything but ``x`` and the
                # weights is freshly allocated, hence 16B-aligned.
                and _tma_ok(x) and _tma_ok(w1) and _row_pitch_ok(w2)):
            a = x.reshape(-1, w1.shape[1])
            small = a.shape[0] <= _SMALL_M
            h = _gemm(a, w1, self.fc1.bias, self._act,
                      _FC1_SMALL if small else _FC1_BIG)
            if self._act < 0:            # unrecognized act_fn: run it eagerly
                h = self.act_fn(h)
            b2 = self.fc2.bias if self.fc2.tp_rank == 0 else None
            y = _gemm(h, self._fc2_weight(w2), b2, _ACT_NONE,
                      _FC2_SMALL if small else _FC2_BIG)
            y = y.view(*x.shape[:-1], w2.shape[0])
            if self.fc2.reduce_results and self.fc2.tp_size > 1:
                y = self.fc2.allreduce(y)
            return y
        # Fallback: anything the fused path does not cover (fp8 weights, fp32,
        # unaligned or odd-width shapes) goes through the eager module.
        return self.fc2(self.act_fn(self.fc1(x)))
