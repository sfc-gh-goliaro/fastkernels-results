"""Llama SwiGLU MLP block: gate_up_proj -> SiluAndMul -> down_proj.

The submodules and their parameters are the baseline's, unchanged -- only
``forward`` is overridden -- so ``state_dict`` keys, the fp8 path, bias and TP all
keep working and the fallback is one call away.

Two Triton kernels replace parts of the sequence, each only where it measured
faster than what it replaces on a B200 under the harness's L2-flushed timing:

* thin token counts run one fused kernel that streams ``gate_up_proj.weight``
  once, computing ``silu(x @ Wg^T) * (x @ Wu^T)`` with a single shared ``x`` tile,
  in place of a cuBLAS GEMM plus the vLLM ``silu_and_mul``;
* wider token counts keep cuBLAS for the projection and run a Triton SwiGLU over
  its ``[M, 2I]`` result.

The down projection stays on ``F.linear``. Every Triton variant tried for it came
out 1.5-1.6x slower than cuBLAS: ``down_proj.weight`` is ``[H, I]``, so tiling
``H`` leaves only 32-64 tiles for 148 SMs, and split-K then costs an fp32 partial
round trip plus a reduce kernel -- more than it saves.

Launch configurations come from a static table rather than ``triton.autotune``:
autotune benchmarks on first call, and the harness checks for injected threads and
patched globals around the timing window, so nothing here may benchmark itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear
from ..L1.silu_and_mul import SiluAndMul

# ``tl.dot`` needs at least 16 rows, so a single-token call still pays for a
# 16-row tile; the row mask keeps the other 15 from being loaded or stored.
_MIN_DOT_M = 16
_MAX_TILE_M = 128

# Token count at or below which the fused kernel is the faster of the two, per
# (hidden, intermediate). Measured by dev/sweep_thin_m.py at nine token counts on each
# configuration, as balanced randomised AB/BA pairs of the two paths with an exact sign
# test -- ten of ten pairs and p = 0.002 at every point. The pairing matters: measuring
# one whole path and then the other cannot separate a real difference from clock drift
# over the run, and drift always favours whichever path went second.
#
# At hidden=2304 the relation is monotonic -- the fused kernel leads through 64 tokens and
# the SwiGLU path from 96 -- so 64 is simply the crossover.
#
# At hidden=4096 it is not monotonic, and 32 is a deliberate choice rather than a
# crossover. The fused kernel leads at 16 and 32, *loses* at 48 and 64, and leads again
# at 96 and 128 before losing from 192. dev/tune_window.py searched 72 correct
# configurations at each of those token counts -- including row blocks smaller than the
# token count, which splits M across several blocks and which no earlier search reached
# -- and nothing closes the 48-64 gap. The mechanism is structural rather than a bad
# tile: the fused kernel's entire advantage is reading gate_up_proj.weight exactly once,
# which pins the block count at intermediate/BN. With BN=128 that is 112 blocks against
# 148 SMs, so 36 sit idle, and buying them by splitting M costs a whole extra pass over
# a 224 MiB weight -- measured 89.1 us at one row block, 99.4 at two, 134.2 at four,
# against 87.0 for the SwiGLU path. At 96-128 tokens the activation pass the fused kernel
# saves outweighs the idle SMs; at 48-64 it does not.
#
# So no single threshold makes this shape's dispatch monotone. 32 is the largest value
# that never routes a token count onto the slower path, which is what protects the
# 60-token case the harness benches: raising it to 128 to capture the 96-128 window would
# regress that case from parity to 0.978x. Closing the gap properly needs the block count
# raised without a second weight pass -- split-K with an in-kernel combine, or a two-SM
# cooperative tile -- which profile/llama_mlp_v2_live_guard/REPORT.md recommends first.
_THIN_M: dict[tuple[int, int], int] = {
    (2304, 9216): 64,
    (4096, 14336): 32,
}

# Token count above which the Triton SwiGLU stops paying for itself end-to-end and the
# baseline activation is kept instead. There is no such point up to 16384 tokens on
# either configuration, so the Triton kernel is used throughout.
#
# The margin narrows with size and is smallest exactly where the earlier probe had
# claimed a regression: by paired randomised measurement (dev/paired_ab.py) the Triton
# path is 1.055x ahead at 313 tokens (12/12 pairs), 1.053x at 2048 (11/12), 1.016x at
# 16384 (10/12, p = 0.039), and 1.040x at 16384 on the smaller configuration (12/12).
# The block-at-a-time measurement in dev/measure_wide_swiglu.py put the 16384 margin
# at 6%; the paired figure of 1.6% is the trustworthy one, since the block ordering
# there measured the Triton path before the baseline activation and so absorbed any
# clock drift in its favour. The sign of the decision is unchanged either way.
_SWIGLU_MAX_M: dict[tuple[int, int], int] = {
    (2304, 9216): 1 << 30,
    (4096, 14336): 1 << 30,
}

_FAST_DTYPES = (torch.bfloat16, torch.float16)

# (BM, BN, BK, num_warps, num_stages) for the fused gate/up kernel, keyed by
# (BM tile, hidden, intermediate). One entry per tile bucket the thresholds above can
# actually reach -- the check below enforces that -- so no configuration here is
# unreachable and none that is reachable is missing.
#
# Chosen by dev/tune_end_to_end.py, which ranks by the *assembled module's* time under
# the harness's own timing loop. Ranking by the kernel's own time, as
# dev/tune_gate_up.py does, picks differently and wrongly: at these token counts the
# case is bounded by streaming the weights once plus the harness's L2 flush, so a
# kernel 6 us faster in isolation can be no faster in place. At hidden=4096 with one
# token that mistake cost 2.5 percentage points, and BN=128 -- 112 CTAs, inside a
# single wave of 148 SMs -- beat the BN=64 configuration the kernel-level search
# preferred, which needs 224 CTAs and so leaves half a second wave idle.
#
# dev/tune_gate_up.py also screens the combinations that do not fit in shared memory:
# BM=128 with BK=128 and 4 stages needs 256 KiB against a 227 KiB limit.
_GATE_UP_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {
    (16, 2304, 9216): (16, 64, 128, 4, 5),
    (32, 2304, 9216): (32, 64, 128, 4, 4),
    (64, 2304, 9216): (64, 64, 128, 4, 4),
    (16, 4096, 14336): (16, 128, 128, 4, 4),
    (32, 4096, 14336): (32, 128, 128, 8, 4),
}

# (BM, BD, num_warps, num_stages) for the standalone SwiGLU, keyed by intermediate
# size. Searched offline by dev/tune_swiglu.py; one entry per admitted shape.
_SWIGLU_CONFIGS: dict[int, tuple[int, int, int, int]] = {
    9216: (1, 1024, 4, 2),
    14336: (2, 512, 4, 1),
}

# Only these (hidden, intermediate) pairs have measured launch configurations; any
# other pair takes the baseline sequence rather than an unmeasured fast path.
_SUPPORTED_SHAPES = frozenset(
    (h, i) for _bm, h, i in _GATE_UP_CONFIGS if i in _SWIGLU_CONFIGS
)

# There is deliberately no clause rejecting a zero-row input. A zero-row grid launches
# no CTA and the empty result is already correct (dev/test_fallback.py checks the
# launcher directly), so such a clause would be unfalsifiable dead weight.
#
# Token tiles go on the grid's second axis, which CUDA limits to 65535 blocks, so the
# largest token count each kernel can cover is that limit times its row-block size.
# Derived from the tables rather than written down, so it cannot drift from them: the
# binding case is the SwiGLU kernel, whose row block is as small as one. Well above
# the 16384 tokens the captures reach, but a larger input has to take the fallback
# rather than launch an illegal grid.
_CUDA_MAX_GRID_Y = 65535
_MAX_FAST_TOKENS = _CUDA_MAX_GRID_Y * min(
    [bm for bm, _bd, _w, _s in _SWIGLU_CONFIGS.values()]
    + [bm for bm, _bn, _bk, _w, _s in _GATE_UP_CONFIGS.values()]
)


def _tile_m(tokens: int) -> int:
    """``tokens`` rounded up to a power of two, clamped to the usable tile range."""
    return min(_MAX_TILE_M, max(_MIN_DOT_M, 1 << (tokens - 1).bit_length()))


def _check_tables() -> None:
    """Every tile bucket the thresholds can reach must have a configuration.

    Run at import, so a missing entry is a clear failure here rather than a
    ``KeyError`` from inside a launcher on some token count nobody tried.
    """
    for shape in _SUPPORTED_SHAPES:
        if shape not in _THIN_M or shape not in _SWIGLU_MAX_M:
            raise RuntimeError(f"no dispatch threshold for {shape}")
        reachable = {_tile_m(m) for m in range(1, _THIN_M[shape] + 1)}
        missing = [b for b in sorted(reachable)
                   if (b, *shape) not in _GATE_UP_CONFIGS]
        if missing:
            raise RuntimeError(
                f"{shape} can dispatch tile buckets {missing} with no configuration")
    unreachable = [key for key in _GATE_UP_CONFIGS
                   if key[0] > _tile_m(_THIN_M.get(key[1:], 0))]
    if unreachable:
        raise RuntimeError(f"unreachable launch configurations: {unreachable}")


_check_tables()


@triton.jit
def _silu_times(gate, up, out_dtype: tl.constexpr):
    """``silu(gate) * up``, rounded exactly the way ``act_and_mul_kernel`` rounds.

    The vLLM kernel widens the gate to fp32, evaluates ``x / (1 + expf(-x))``, casts
    the *activation alone* back to the scalar type, and only then multiplies by the
    up half in fp32 before one final cast. Both casts matter: the harness scores the
    candidate against that kernel's output, not against an exact result, so an
    epilogue that rounds less often ends up further away rather than closer.
    """
    silu = (gate / (1.0 + tl.exp(-gate))).to(out_dtype).to(tl.float32)
    return (silu * up).to(out_dtype)


@triton.jit
def _swiglu_gate_up_kernel(X, W, OUT, M, N, K,
                           stride_xm, stride_wn, stride_om,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           EVEN_N: tl.constexpr, EVEN_K: tl.constexpr):
    """``OUT = silu(X @ W[:N].T) * (X @ W[N:].T)`` for a thin ``X``.

    One program owns ``BN`` columns of ``OUT`` and walks the whole ``K``
    reduction, so ``W`` -- which dominates the byte count at these token counts --
    is read exactly once. The gate and up row blocks share the same ``X`` tile, so
    the second ``tl.dot`` costs no extra traffic on ``X``.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    row_ok = offs_m[:, None] < M
    col_ok = offs_n[:, None] < N

    x_ptr = X + offs_m[:, None] * stride_xm + offs_k[None, :]
    g_ptr = W + offs_n[:, None] * stride_wn + offs_k[None, :]
    u_ptr = W + (offs_n + N)[:, None] * stride_wn + offs_k[None, :]

    acc_g = tl.zeros((BM, BN), tl.float32)
    acc_u = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        if EVEN_K:
            x = tl.load(x_ptr, mask=row_ok, other=0.0)
            if EVEN_N:
                w_g = tl.load(g_ptr)
                w_u = tl.load(u_ptr)
            else:
                w_g = tl.load(g_ptr, mask=col_ok, other=0.0)
                w_u = tl.load(u_ptr, mask=col_ok, other=0.0)
        else:
            k_ok = (k + offs_k)[None, :] < K
            x = tl.load(x_ptr, mask=row_ok & k_ok, other=0.0)
            w_ok = k_ok if EVEN_N else col_ok & k_ok
            w_g = tl.load(g_ptr, mask=w_ok, other=0.0)
            w_u = tl.load(u_ptr, mask=w_ok, other=0.0)
        acc_g = tl.dot(x, tl.trans(w_g), acc_g)
        acc_u = tl.dot(x, tl.trans(w_u), acc_u)
        x_ptr += BK
        g_ptr += BK
        u_ptr += BK

    # Round both accumulators where ``F.linear`` would have stored them, then run
    # the activation the way the baseline does. Carrying fp32 all the way through
    # the epilogue is *more* accurate against an exact result but disagrees with
    # the baseline the harness scores against -- see dev/probe_epilogue.py, where
    # skipping this rounding drops the match ratio to 0.94 on the larger config.
    out_dtype = OUT.dtype.element_ty
    gate = acc_g.to(out_dtype).to(tl.float32)
    up = acc_u.to(out_dtype).to(tl.float32)
    h = _silu_times(gate, up, out_dtype)
    store_ok = row_ok if EVEN_N else row_ok & (offs_n[None, :] < N)
    tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :],
             h, mask=store_ok)


@triton.jit
def _swiglu_kernel(Y, OUT, M, D, stride_ym, stride_om,
                   BM: tl.constexpr, BD: tl.constexpr, EVEN_D: tl.constexpr):
    """``OUT = silu(Y[:, :D]) * Y[:, D:]`` on a 2-D grid over rows and columns.

    Blocking rows and columns separately keeps the gate and up loads 16-byte
    aligned and, unlike a flat 1-D grid, needs no runtime ``//`` or ``%`` to
    recover the row -- the integer division showed up in the probe version's
    end-to-end time at large ``M``.
    """
    pid_d = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = pid_d * BD + tl.arange(0, BD)

    mask = offs_m[:, None] < M
    if not EVEN_D:
        mask = mask & (offs_d[None, :] < D)

    gate_ptr = Y + offs_m[:, None] * stride_ym + offs_d[None, :]
    gate = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_ptr + D, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + offs_m[:, None] * stride_om + offs_d[None, :],
             _silu_times(gate, up, OUT.dtype.element_ty), mask=mask)


def _fused_gate_up(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Fused gate/up projection with a SwiGLU epilogue, for thin ``x``."""
    tokens, hidden = x.shape
    inter = w.shape[0] // 2
    out = torch.empty((tokens, inter), dtype=x.dtype, device=x.device)
    bm, bn, bk, warps, stages = _GATE_UP_CONFIGS[(_tile_m(tokens), hidden, inter)]
    _swiglu_gate_up_kernel[(triton.cdiv(inter, bn), triton.cdiv(tokens, bm))](
        x, w, out, tokens, inter, hidden,
        x.stride(0), w.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk,
        EVEN_N=(inter % bn == 0), EVEN_K=(hidden % bk == 0),
        num_warps=warps, num_stages=stages,
    )
    return out


def _swiglu(y: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` over a ``[tokens, 2 * inter]`` projection result."""
    tokens, width = y.shape
    inter = width // 2
    out = torch.empty((tokens, inter), dtype=y.dtype, device=y.device)
    bm, bd, warps, stages = _SWIGLU_CONFIGS[inter]
    _swiglu_kernel[(triton.cdiv(inter, bd), triton.cdiv(tokens, bm))](
        y, out, tokens, inter, y.stride(0), out.stride(0),
        BM=bm, BD=bd, EVEN_D=(inter % bd == 0),
        num_warps=warps, num_stages=stages,
    )
    return out


class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config: dict | None = None,
                 hidden_size: int | None = None,
                 intermediate_size: int | None = None,
                 reduce_results: bool = True):
        super().__init__()
        h = hidden_size if hidden_size is not None else config.hidden_size
        i = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            h, [i] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            i, h,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def _fast_path_ok(self, x: torch.Tensor, gate_up: nn.Module, down: nn.Module,
                      w_gate_up: torch.Tensor) -> bool:
        """Whether this call is one the Triton path handles.

        Every fact is read from live state on each call. An earlier version settled
        the module-side half once in ``__init__``, on the argument that quantization,
        bias, TP width and weight layout cannot change afterwards. They can: nothing
        stops a caller attaching a bias to a linear after construction, and a cached
        decision then sends a biased module down a bias-free path. Measured on such a
        module, the cached guard produced ``max_abs_error`` 13.1 and a match ratio of
        0.0053 against the baseline. The checks below are cheap enough that there is
        no reason to trade correctness for them -- ``use_fp8``, ``bias`` when it is
        ``None``, and ``tp_size`` are plain instance attributes, so reading them is a
        dict lookup, not a descriptor call.

        Only what the Triton kernels themselves rely on is checked. ``down.weight``
        is deliberately *not* validated for shape, dtype, device or layout: the final
        ``F.linear`` consumes it exactly as the baseline's ``down_proj`` would, so it
        raises or succeeds identically on both paths and a check here could only
        duplicate that.

        Rank and contiguity are checked directly rather than by trying ``reshape``:
        on a non-contiguous input ``reshape`` silently allocates and copies, which
        inside the timed window would be a large unbudgeted cost. A contiguous 2-D
        ``x`` whose last dimension is the hidden size needs no reshape at all, on the
        way in or out.

        ``x`` is matched against the weight for dtype and device rather than checked
        against a dtype allowlist alone: an fp16 input to a bf16 module makes
        ``F.linear`` raise a clear error where the kernel would fail inside Triton.
        The device must also be the *current* one, because a Triton launch goes to
        whatever device is current rather than to the tensor's.

        There is deliberately no clause on ``self.training`` or ``x.requires_grad``:
        the baseline's activation writes into a fresh ``torch.empty`` from a CUDA
        extension, so it already breaks the autograd graph at the same point, and
        ``gate_up_proj.weight.grad`` comes back ``None`` either way (measured in
        dev/test_fallback.py). Such a clause would cost work on every call without
        restoring anything the baseline provided.
        """
        if (gate_up.use_fp8 or down.use_fp8
                or gate_up.bias is not None or down.bias is not None
                # TP shards both weights and the row-parallel half then needs an
                # all-reduce; neither is modelled here.
                or down.tp_size != 1
                or torch.compiler.is_compiling()):
            return False
        if (x.dim() != 2 or x.dtype not in _FAST_DTYPES or not x.is_cuda
                or not x.is_contiguous() or x.shape[0] > _MAX_FAST_TOKENS):
            return False
        # Rank first, then the dimensions. Checking ``shape[1]`` without checking
        # ``dim()`` accepts a weight of any higher rank whose first two extents happen
        # to match -- a contiguous ``[2I, H, 1]`` view passed this and the kernel read
        # its storage as 2-D, while ``F.linear`` on the same weight raises. The gate
        # and up halves are the two halves of the row axis, so an odd row count has no
        # such split.
        if (w_gate_up.dim() != 2 or w_gate_up.shape[0] % 2
                or w_gate_up.shape[1] != x.shape[1]
                or not w_gate_up.is_contiguous()):
            return False
        if (x.shape[1], w_gate_up.shape[0] >> 1) not in _SUPPORTED_SHAPES:
            return False
        device = x.get_device()
        return (x.dtype == w_gate_up.dtype and device == w_gate_up.get_device()
                and device == torch.cuda.current_device())

    def forward(self, x):
        gate_up = self.gate_up_proj
        down = self.down_proj
        w_gate_up = gate_up.weight
        if not self._fast_path_ok(x, gate_up, down, w_gate_up):
            x = gate_up(x)
            x = self.act_fn(x)
            return down(x)

        # The guard established that this shape has an entry in both threshold
        # tables, so neither lookup can miss.
        tokens, hidden = x.shape
        shape = (hidden, w_gate_up.shape[0] >> 1)
        if tokens <= _THIN_M[shape]:
            h = _fused_gate_up(x, w_gate_up)
        elif tokens <= _SWIGLU_MAX_M[shape]:
            h = _swiglu(F.linear(x, w_gate_up))
        else:
            h = self.act_fn(F.linear(x, w_gate_up))
        # No all-reduce: the fast path only runs at tp_size == 1, where the
        # baseline's row-parallel reduction is a no-op.
        return F.linear(h, down.weight)
