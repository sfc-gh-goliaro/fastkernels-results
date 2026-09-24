"""Llama SwiGLU MLP block with a fused fast path for measured shapes.

The baseline is three kernels deep — ``gate_up`` projection, SiLU-and-mul, ``down``
projection — and materializes a ``[rows, 2*intermediate]`` tensor between the first two.
This module keeps the baseline's parameter surface exactly (the same two submodules
under the same names, so ``state_dict()`` matches key for key and the benchmark's
weight sharing takes effect) and, for row counts and weight geometries measured to
benefit, replaces all three kernels with **one fused MLP kernel plus one finalization
kernel** — two device kernels per forward.

Column ``j`` of the activation depends only on rows ``j`` and ``j + intermediate`` of
``W1`` and column ``j`` of ``W2``, so the intermediate dimension is sliced across CTAs:
each CTA computes its own slice of the activation in registers, consumes it against the
matching columns of ``W2`` without the activation ever reaching memory, and atomically
adds its ``[rows, hidden]`` partial into an fp32 workspace. Every weight element is
therefore read by exactly one CTA exactly once, and the per-CTA weight footprint is
``3 * cols_per_tile * hidden * 2`` bytes. The finalization kernel converts the workspace
to the output dtype and re-zeros it, which both keeps the launch count at two and makes
"the workspace is zero on entry" an invariant carried across calls.

An epilogue-fused gate-up projection followed by cuBLAS measured *faster* than this on
some shapes (1.062x and 1.093x of the fallback at 26 and 64 rows, against 0.871x and
0.706x for the fused form), but it is three device kernels — cuBLAS contributes a GEMM
and a split-K reduction — so it is not shipped. Those shapes run the baseline
composition instead. The comparison lives in ``experiments/phase1_config_sweep.py`` and
its recorded output.

Nothing derived from the weights is stored. The fp32 reduction workspace is allocated
lazily inside ``forward`` from the input's device and keyed by shape, device, dtype and
stream; it is never registered as a buffer, so it cannot change the ``state_dict()`` key
set, and ``nn.Module.to`` never has to migrate it.

Every input that fails :meth:`LlamaMLP._fused_config` — a quantized module, a compiled
trace, tensor parallelism, a bias, autograd, a non-bf16 dtype, a weight layout the
kernels' pointer arithmetic does not assume, an unmeasured shape — runs the baseline
composition unchanged.

Two details of the fused arithmetic are load-bearing rather than incidental, because the
sum over the intermediate dimension amplifies any difference from the reference: the
reference rounds to bf16 twice on the way through, and both roundings are reproduced
here. See ``_swiglu``.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear
from ..L1.silu_and_mul import SiluAndMul


class _Isplit(NamedTuple):
    """Launch geometry for the fused MLP plus its finalization kernel."""

    cols_per_tile: int
    hidden_per_step: int
    out_per_step: int
    num_warps: int
    num_stages: int
    final_block: int
    final_warps: int


# Shapes the fast path claims, keyed by ``(rows, hidden, intermediate)``, with the
# configuration measured best for each. Membership is decided by
# ``experiments/phase1_config_sweep.py``, which requires the complete path to beat the
# composition this module would otherwise run — already faster than the benchmark's
# baseline, because the fallback reaches the frozen L1 activation — by at least 2% with
# a margin above the run-to-run spread. The sweep prefers ``_Isplit`` whenever it
# qualifies, because only that structure is a single fused MLP kernel.
_FUSED_SHAPES: dict[tuple[int, int, int], _Isplit] = {
    (1, 2304, 9216): _Isplit(64, 128, 256, 4, 4, 1024, 4),
    (1, 4096, 14336): _Isplit(128, 128, 256, 4, 4, 1024, 4),
}

# bf16 only. fp16 would work arithmetically but was never measured or validated on any
# shape, and the benchmark only ever runs bf16; admitting it would put an untested dtype
# on the fast path.
_FUSED_DTYPE = torch.bfloat16

# Reproduction switch, unset in normal use. ``FK_LLAMA_MLP_FUSED=0 python validate.py``
# runs this same file with the fast path off, which is how each claimed shape is checked
# against the composition it would otherwise take, inside one benchmark run rather than
# across two.
_FUSED_ENABLED = os.environ.get("FK_LLAMA_MLP_FUSED", "1") != "0"


@triton.jit
def _swiglu(acc_gate, acc_up, dtype):
    """SwiGLU rounded exactly where the reference rounds it.

    The reference rounds to bf16 twice between the two projections, and both roundings
    have to be reproduced or the sum over the intermediate dimension amplifies the
    difference past the benchmark's tolerance:

    1. ``gate_up`` is a bf16 tensor, so the activation sees values already rounded once.
       Applying the activation to the fp32 accumulator instead left only 95.0% of the
       output inside tolerance at ``hidden=4096, intermediate=14336``, against the 99%
       the benchmark requires.
    2. The reference's ``silu_kernel`` returns ``scalar_t``, so ``silu(gate)`` is
       rounded to bf16 *before* being multiplied by ``up``. Keeping that product in fp32
       left 98.4%, and made 27% of the activation entries differ.

    With both in place the only remaining difference is fp32 summation order. On the
    claimed shapes that happens to leave the result bitwise equal to the reference's,
    but that is an observation and not an invariant — it can change with a tile size, a
    row count or a cuBLAS heuristic — so correctness is asserted against the benchmark's
    tolerance rule and never against bitwise equality.
    """
    gate = acc_gate.to(dtype).to(tl.float32)
    up = acc_up.to(dtype).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(dtype).to(tl.float32)
    return (silu * up).to(dtype)


@triton.jit
def _fused_mlp_kernel(X, W1, W2, WS, rows, hidden, inter,
                      stride_x, stride_w1, stride_w2, stride_ws,
                      ROWS_PER_TILE: tl.constexpr, COLS_PER_TILE: tl.constexpr,
                      HIDDEN_PER_STEP: tl.constexpr, OUT_PER_STEP: tl.constexpr):
    """The whole MLP in one kernel, sliced across the intermediate dimension.

    CTA ``c`` owns columns ``[c*COLS_PER_TILE, (c+1)*COLS_PER_TILE)`` of the activation.
    It needs only rows ``rj`` and ``rj + inter`` of ``W1`` and columns ``rj`` of ``W2``,
    so there is no cross-CTA dependency and no global barrier, and each weight element
    is read by exactly one CTA exactly once: the per-CTA weight footprint is
    ``3 * COLS_PER_TILE * hidden * 2`` bytes and the CTAs partition the weights.

    The activation never reaches memory — it is consumed against ``W2`` in registers
    immediately after the epilogue — so no ``[rows, 2*intermediate]`` or
    ``[rows, intermediate]`` intermediate is written.

    The ``[rows, hidden]`` partials are summed across CTAs by fp32 atomics into ``WS``,
    which the caller guarantees is zero on entry. The conversion to the output dtype
    happens in ``_finalize_kernel``, whose kernel boundary is the device-wide barrier
    that makes every partial visible; doing it in this kernel instead would need a
    device-scope publication step that Triton's ``tl.*`` surface cannot express for all
    lanes of a block.

    The weight loads carry ``evict_first`` and the re-read ``x`` tile ``evict_last``:
    the weights are a one-shot stream of hundreds of megabytes through a 133 MB L2 and
    must not evict the data that is reused.

    Both index ranges are masked. Neither the row count nor the intermediate size is
    assumed to divide its tile — the shared-expert recipe has ``intermediate=1024``,
    and captured row counts include 26 and 31.
    """
    cta = tl.program_id(0)
    j0 = cta * COLS_PER_TILE
    rj = j0 + tl.arange(0, COLS_PER_TILE)
    rm = tl.arange(0, ROWS_PER_TILE)
    col_ok = rj < inter
    row_ok = rm < rows

    acc_gate = tl.zeros((ROWS_PER_TILE, COLS_PER_TILE), dtype=tl.float32)
    acc_up = tl.zeros((ROWS_PER_TILE, COLS_PER_TILE), dtype=tl.float32)
    for k0 in range(0, hidden, HIDDEN_PER_STEP):
        rk = k0 + tl.arange(0, HIDDEN_PER_STEP)
        k_ok = rk < hidden
        x = tl.load(X + rm[:, None] * stride_x + rk[None, :],
                    mask=row_ok[:, None] & k_ok[None, :], other=0.0,
                    eviction_policy="evict_last")
        w_gate = tl.load(W1 + rj[:, None] * stride_w1 + rk[None, :],
                         mask=col_ok[:, None] & k_ok[None, :], other=0.0,
                         eviction_policy="evict_first")
        w_up = tl.load(W1 + (rj + inter)[:, None] * stride_w1 + rk[None, :],
                       mask=col_ok[:, None] & k_ok[None, :], other=0.0,
                       eviction_policy="evict_first")
        acc_gate = tl.dot(x, tl.trans(w_gate), acc_gate)
        acc_up = tl.dot(x, tl.trans(w_up), acc_up)

    a = _swiglu(acc_gate, acc_up, W2.dtype.element_ty)

    for n0 in range(0, hidden, OUT_PER_STEP):
        rn = n0 + tl.arange(0, OUT_PER_STEP)
        out_ok = rn < hidden
        w_down = tl.load(W2 + rn[:, None] * stride_w2 + rj[None, :],
                         mask=out_ok[:, None] & col_ok[None, :], other=0.0,
                         eviction_policy="evict_first")
        partial = tl.dot(a, tl.trans(w_down))
        tl.atomic_add(WS + rm[:, None] * stride_ws + rn[None, :], partial,
                      mask=row_ok[:, None] & out_ok[None, :], sem="relaxed")


@triton.jit
def _finalize_kernel(WS, OUT, total, hidden, stride_ws, stride_out,
                     BLOCK: tl.constexpr):
    """Convert the fp32 reduction workspace to the output dtype, and re-zero it.

    Re-zeroing here is what makes "the workspace is zero on entry" an invariant carried
    across calls, so the fused kernel needs no per-call memset. That is the difference
    between two device kernels per forward and three: measured on this operator, a
    separate ``zero_()`` costs 5.3 us and a separate cast 6.0 us, both launch-latency
    bound on a few kilobytes.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    ok = idx < total
    row = idx // hidden
    col = idx % hidden
    src = row * stride_ws + col
    value = tl.load(WS + src, mask=ok, other=0.0)
    tl.store(OUT + row * stride_out + col, value.to(OUT.dtype.element_ty), mask=ok)
    tl.store(WS + src, tl.zeros((BLOCK,), dtype=tl.float32), mask=ok)


def _row_tile(rows: int) -> int:
    """Smallest ``tl.dot``-compatible row tile that covers *rows* in one tile."""
    return max(16, triton.next_power_of_2(rows))


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

        self.hidden_size = h
        self.intermediate_size = i
        # A quantized module keeps fp8 weights and a block scale the fused kernels know
        # nothing about, so it is excluded once here rather than on every call.
        self._fusable = quant_config is None
        # Reduction workspaces, allocated lazily from the input's device. A plain dict
        # attribute, never a buffer: a buffer would change the state_dict key set, and a
        # tensor created at construction time would stay on the CPU because
        # nn.Module.to migrates only parameters and buffers.
        self._workspaces: dict[tuple, torch.Tensor] = {}

    def _fused_config(self, x: torch.Tensor) -> _Isplit | None:
        """Return the launch geometry for *x*, or ``None`` to run the composition.

        The shape lookup comes first, and everything else only runs on a hit. That
        ordering is not cosmetic: at 60 us per call the benchmark's timing loop is close
        enough to the CPU/GPU crossover that the predicate's own cost is measurable, and
        running the full chain before a lookup that was going to miss cost about 2% on
        every shape the table does not claim -- a regression on shapes the fast path
        never even touches. A miss now costs one integer divide and one dict probe.

        Every condition below a hit is something the fused path does not implement, so
        failing any one of them is a correctness requirement rather than a heuristic.
        """
        if not self._fusable or not _FUSED_ENABLED:
            return None
        h = self.hidden_size
        if x.shape[-1] != h:
            return None
        config = _FUSED_SHAPES.get((x.numel() // h, h, self.intermediate_size))
        if config is None:
            return None

        # Inductor traces the pure-PyTorch composition; a custom kernel would either
        # break the trace or be baked in with the wrong assumptions.
        if torch.compiler.is_compiling():
            return None
        # The fused path builds no autograd graph, so a training caller has to get the
        # composition rather than a silently non-differentiable result.
        if torch.is_grad_enabled() or x.requires_grad:
            return None
        if not x.is_cuda or x.dtype is not _FUSED_DTYPE or not x.is_contiguous():
            return None
        w_gate_up = self.gate_up_proj.weight
        w_down = self.down_proj.weight
        if w_gate_up.dtype is not _FUSED_DTYPE or w_down.dtype is not _FUSED_DTYPE:
            return None
        # The kernel indexes the weights as ``row * stride(0) + col``, which assumes a
        # unit inner stride, and it takes pointers to all three tensors in one launch,
        # which assumes one device.
        if not w_gate_up.is_contiguous() or not w_down.is_contiguous():
            return None
        if w_gate_up.device != x.device or w_down.device != x.device:
            return None
        # The fused path applies neither bias, and skips the row-parallel all-reduce.
        if self.gate_up_proj.bias is not None or self.down_proj.bias is not None:
            return None
        if self.down_proj.tp_size != 1:
            return None
        return config

    def _workspace(self, rows: int, hidden: int, device) -> torch.Tensor:
        """A zeroed fp32 ``[rows, hidden]`` reduction buffer for this stream.

        Keyed by stream as well as shape and device because the buffer carries the
        "zero on entry" invariant across calls, and two streams sharing one buffer
        could interleave a fused kernel with the other's finalization kernel. Allocated
        with ``torch.zeros`` so the invariant holds on first use.
        """
        key = (rows, hidden, device, torch.cuda.current_stream(device).cuda_stream)
        ws = self._workspaces.get(key)
        if ws is None:
            ws = torch.zeros((rows, hidden), dtype=torch.float32, device=device)
            self._workspaces[key] = ws
        return ws

    def _run_isplit(self, flat: torch.Tensor, cfg: _Isplit) -> torch.Tensor:
        rows, h = flat.shape
        i = self.intermediate_size
        workspace = self._workspace(rows, h, flat.device)
        out = torch.empty((rows, h), dtype=flat.dtype, device=flat.device)
        _fused_mlp_kernel[(triton.cdiv(i, cfg.cols_per_tile),)](
            flat, self.gate_up_proj.weight, self.down_proj.weight, workspace,
            rows, h, i,
            flat.stride(0), self.gate_up_proj.weight.stride(0),
            self.down_proj.weight.stride(0), workspace.stride(0),
            ROWS_PER_TILE=_row_tile(rows), COLS_PER_TILE=cfg.cols_per_tile,
            HIDDEN_PER_STEP=cfg.hidden_per_step, OUT_PER_STEP=cfg.out_per_step,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )
        total = rows * h
        _finalize_kernel[(triton.cdiv(total, cfg.final_block),)](
            workspace, out, total, h, workspace.stride(0), out.stride(0),
            BLOCK=cfg.final_block, num_warps=cfg.final_warps,
        )
        return out

    def forward(self, x):
        config = self._fused_config(x)
        if config is None:
            x = self.gate_up_proj(x)
            x = self.act_fn(x)
            return self.down_proj(x)

        h = self.hidden_size
        out = self._run_isplit(x.reshape(-1, h), config)
        return out.reshape(*x.shape[:-1], h)
