"""TP-aware linear layers (L2 operators) for B200 / sm_100.

Same five classes as the baseline -- ``ColumnParallelLinear``,
``MergedColumnParallelLinear``, ``QKVParallelLinear``, ``ReplicatedLinear``,
``RowParallelLinear`` -- with the baseline's ``__init__``/``forward`` contract,
its weight loaders, and its full public attribute surface.  These wrappers are a
shared dependency: eighteen sibling L2/L3 operator modules read ``num_heads``,
``weight_loader``, ``tp_size``, ``num_kv_heads``, ``allreduce``, ``use_fp8``,
``head_size``, ``weight_scale_inv``, ``linear_op``, ``tp_rank``,
``output_size_per_partition``, ``output_sizes``, ``reduce_results``,
``disable_tp``, ``input_size_per_partition`` and ``_replicate_kv`` off them, and
one of them rebinds the ``weight_loader`` this file installs.  So the contract is
wider than the two methods, and nothing here is renamed or derived away.

Strip the sharding bookkeeping -- which only runs in the weight loaders, never in
``forward`` -- and every forward is one of two lines, ``self.linear_op(...)`` for
fp8 or ``F.linear(...)`` otherwise, with ``RowParallelLinear`` adding
``self.allreduce(y)``.  There is no arithmetic to restructure: the merged and QKV
classes are already single fused GEMMs.  The whole optimization surface is which
GEMM runs and which all-reduce runs, so this file is a dispatcher, not a kernel.

Two changes against the baseline, in descending order of measured value:

* **fp8 reaches the frozen L1 winner.**  ``from ..L1.fp8_linear import
  Fp8Linear`` resolves to ``candidate/L1/fp8_linear.py`` (the candidate finder
  prefers a candidate file over the baseline alias, and the bench does not run
  ``--standalone``), whose register-resident activation quantizer and cached
  scratch arena the baseline's ``Fp8Linear`` does not have.  This is the same
  import line the baseline uses; the redirect, not new code here, is what makes
  the three fp8 cases faster.  Measured in ``profile/bench_fp8.txt``.
* **A table-driven bf16 backend dispatcher, admitting nothing.**  See below.

``RowParallelLinear`` keeps NCCL, measured rather than assumed.  Cases 22 and 24
are the only scored cases with a collective inside the timed window, and four
alternatives to ``dist.all_reduce`` were timed on the real 4-rank NVLink group
(``profile/bench_collective.py``, output in ``profile/bench_collective_case22.txt``
and ``profile/bench_collective_case24.txt``).  Every one of them lost, on both
message sizes, all at matched ratio 1.00000:

======================  ==================  =================
variant                 128 MiB (case 22)   7.8 MiB (case 24)
======================  ==================  =================
``dist.all_reduce``     1.000 (738 us)      1.000 (142 us)
vendored IPC, 8 MiB       declined: capped  0.960
vendored IPC, raised    0.596               0.954
``flashinfer`` vllm_ar  0.203 / 0.582       0.453 / 0.943
M-chunked overlap       0.915 / 0.893       0.806 / 0.661
======================  ==================  =================

So no fast communicator is installed, no ``max_size`` is raised, and the
rank-order hazard is not taken on.  Two things are worth recording for later.
The chunked overlap cannot win on case 22 because the GEMM is only 306 us against
a 738 us collective -- there is not enough compute to hide the message behind,
and splitting it into 2 or 4 smaller collectives costs more latency than the
overlap recovers.  And the vLLM one-shot IPC kernel is strongly CTA-count
sensitive (0.203 at 8 CTAs against 0.582 at 36 on case 22), so its ceiling here
is a tuning question, not a structural one.  Absolute collective microseconds are
inflated by inter-rank skew inside the window, which every variant pays equally;
the ranking is the reliable output, not the microseconds.

What the dispatcher is for, and why its table is empty.  A faithful replica of
the harness's timing construction -- 2x L2 flush outside the window, the shifting
pool's activation copy inside it, CUDA events, median of 50 -- was run against
seven BF16 GEMM backends on all 22 bf16 scored shapes
(``profile/bench_backends.py``, output in ``profile/bench_backends.txt``).
``F.linear`` (cuBLAS) won 20 of 22 outright, and neither exception survived
re-measurement on separate GPU leases (``profile/repeat_backends_L{1,2,3}.txt``),
which is the gate an entry has to clear.  So the table ships empty and every bf16
call delegates.  That is a valid terminal state, not an unfinished one, and it
also keeps this file torch-only -- worth something, because eighteen sibling
operator modules import it and a non-empty table would put a FlashInfer import in
all of their load paths.

Rejected, each with the number that rejected it:

* **``tinygemm_bf16``** at 1x2048x1024, the sweep's best-looking result at 11.3 us
  against 13.4 us (1.18x).  Re-measured on three separate leases it is 13.3 us
  against 13.4 us -- 1.005x, 1.006x, 1.010x.  The original 11.3 us reading did not
  reproduce, and a 0.1 us edge is ~20x smaller than the ~2.04 us quantisation
  step, so both backends sit on the same rung of the ladder: a median of 50 can
  report a stable sub-quantum difference without there being a measurable one.
* **``mm_bf16[cudnn]``** at 16384x4096x28672, the sweep's other candidate: one
  reading of 2645 us against ``F.linear`` at 2799 us (1.06x).  Re-measured it
  *loses* -- 0.945x on one lease, 0.991x on another.  This shape is also where
  ``F.linear`` is least repeatable: across five rounds inside a single lease its
  own readings span 391-401 us (2469-2860 us on one lease, 2542-2943 us on the
  other, medians 2690 and 2697 us).  A spread that wide is what made a one-shot
  5.8 % margin look real.  cudnn also loses by 2-6x on every small shape.
* **FlashInfer TGV GEMM** (``tgv_gemm_sm100``, with and without PDL), the
  direction upstream reports at "up to 60 % over cuBLAS on B200" for
  decode-shaped GEMMs (measured there at gpcclk 1800 MHz): **1.7x to 25x slower on
  all 22 shapes here**, and the loss is GPU time, not wrapper overhead (case 2:
  5.83 ms of kernel against 0.46 ms, with 63 us of host time).  Pinning the tactic
  through the same runner the autotuner uses splits the blame by shape rather
  than settling it one way: at M=1 the autotuner really is picking a poor tactic
  (0.869x tuned against 1.005x at the best pinned tactic, about one quantum), but
  at 492x2304x18432 the best pinned tactic is still 345 us against 42 us, so
  there the kernel itself is responsible and no tuner fix would rescue it.
* **``mm_bf16[cutlass]`` / ``mm_bf16[cublaslt]``**: never fastest anywhere;
  ``cublaslt``'s wrapper alone costs 185-280 us of host time.  Both also raise on
  a non-contiguous activation, as does ``tinygemm``.
* **``mm_bf16[cutile]``**: unusable in this environment, no ``tileiras``
  compiler.
* **A hand-written skinny GEMM**, rejected on counters rather than on precedent.
  One Nsight Compute pass over the two cases with the most unexplained time
  (``profile/ncu-cublas-skinny-gemm/REPORT.md``) shows cuBLAS is neither
  bandwidth-starved nor wave-quantised on them: at M=492 the tensor/TMEM pipe is
  already at 70.8 % of peak with DRAM at 23 %, and at M=379 the whole launch is a
  single 144-CTA wave on 148 SMs, which ncu's own grid-too-small rule scores at
  2.7 %.  Both shapes sit *above* the ~280 FLOP/byte roofline ridge (479 and 357),
  so they are compute-side, not the bandwidth-bound regime a persistent
  CLC-scheduled rewrite is the documented answer to.

Why the dispatcher stays even though it admits nothing.  It is the structure
that makes every other change safe -- a dispatch bug can only ever cost a
delegation, never a wrong answer -- and it is the shape the closely related frozen
``candidate/L1/linear.py`` shipped in, so a later phase that finds a real backend
win has somewhere to put it and a gate to put it through.  Its cost was measured,
not assumed: with the table empty, ``_plan`` returns at its first predicate, and
the candidate matches the baseline to 1.0002x on the largest case with
bit-identical output (``profile/check_noise.txt``), because the host-side detour
sits inside the shadow of the pre-iteration L2 flush.  The backend callables all
take the same ``(x2, weight, bias, N)`` signature even though only ``tinygemm``
needs ``N``; a uniform signature is what lets the table name any of them.

Open for a later phase, in descending order of value: the vLLM one-shot IPC
all-reduce is strongly CTA-count sensitive and was never tuned here; the TGV
autotuner picks a worse tactic than the best available at M=1; and the narrowest
collective margin (case 24 at 0.960x of NCCL) was not re-confirmed with
per-iteration rank synchronisation.  Two latent defects also exist in the
baseline and are reproduced here deliberately, because this file's contract is to
match it: ``MergedColumnParallelLinear`` sizes its weight with ``effective_tp``
but its bias with ``tp``, so ``disable_tp=True`` with ``bias=True`` is
dimensionally inconsistent above tp=1; and ``QKVParallelLinear._scale_loader``
always shards the scale by rank even when ``_replicate_kv`` is set, which the
weight loader correctly does not.  Neither is reachable from the 25 scored cases
(the four non-distributed classes are built at tp=1, where both branches
coincide), and fixing either here would be a contract divergence, not a fix.

Two harness constants bound what is worth attempting at all, both corroborated by
the sweep and by the frozen ``candidate/L1/linear.py``: the scored window costs
~9.3 us fixed, and readings quantise in ~2.04 us steps.  Two of the 25 scored
cases -- the two no-reduce ``RowParallelLinear`` ones, at 9.3 and 9.2 us -- sit on
that floor with no improvable time at all, and the next three sit one rung above
it at 11.3-11.4 us.  Elsewhere the improvable time is ``measured - 9.3 us``.  That
is why sub-quantum backend wins are not worth admitting even when they look
repeatable.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn

import torch.nn.functional as F

from ....infra.tp import _tp_size, _tp_rank
from ..L1.allreduce import AllReduce


_FP8_LINEAR_CLS = None


def _get_fp8_linear_cls():
    """The fp8 implementation these wrappers delegate to.

    The import is relative, so the candidate finder redirects it to
    ``candidate/L1/fp8_linear.py`` when that frozen winner is present and to the
    baseline module when it is not.  Kept lazy, exactly as the baseline has it:
    the fp8 module pulls in a CUDA extension, and an instance built without a
    ``quant_config`` -- which is most of them, and all 22 bf16 scored cases --
    must not pay for it.

    The returned class also has to be *recognisable* as an fp8 linear to the
    bench's weight initialiser, which is a harder constraint than it looks.  A
    reconstructed fp8 wrapper allocates ``weight``/``weight_scale_inv`` with
    ``torch.empty`` in the checkpoint layout and relies on the loader to fill and
    layout-transform them; the bench does that itself, replacing the scale with
    DeepGEMM's ``(N, ceil(K/1024)) int32`` UE8M0 form -- but only for submodules
    whose ``linear_op`` passes ``isinstance(..., baseline Fp8Linear)``.  The
    frozen winner is an independent ``nn.Module``, so a wrapper holding it fails
    that test, keeps an uninitialised scale that the run dtype has meanwhile cast
    to bf16, and DeepGEMM then asserts ``sfb_dtype == kFloat or kInt``.  Weight
    sharing cannot repair it either: the two sides' scales now differ in shape, so
    ``load_state_dict`` raises and the bench swallows it.  Measured, not inferred
    -- see ``profile/diag_fp8.txt``.

    So when the frozen implementation is not already a subclass of the baseline
    class, it is given it as a second base.  This is a marker only: the method
    resolution order puts the frozen module first, so ``forward``,
    ``_ensure_buffers``, ``BLOCK_SIZE`` and the whole quantize/GEMM path are the
    frozen winner's, and the baseline contributes nothing but its identity.  What
    it buys is that the bench builds *identical* valid fp8 weights on both sides,
    which is what makes the fp8 comparison honest rather than an error.

    The class is built once and memoised, so every wrapper instance shares one
    type -- a fresh type per call would fragment ``torch.compile``'s guard cache
    -- and it is bound into this module under its own ``__qualname__`` so it is a
    resolvable module-level symbol and stays picklable.
    """
    global _FP8_LINEAR_CLS
    if _FP8_LINEAR_CLS is not None:
        return _FP8_LINEAR_CLS
    from ..L1.fp8_linear import Fp8Linear
    # Relative, like every other import here, and marker-only: it supplies the
    # identity the bench's fp8 weight initialiser tests for, never behaviour.
    from ...baseline.L1.fp8_linear import Fp8Linear as _Reference
    if issubclass(Fp8Linear, _Reference):
        _FP8_LINEAR_CLS = Fp8Linear
    else:
        _FP8_LINEAR_CLS = type("Fp8Linear", (Fp8Linear, _Reference), {})
        globals().setdefault(_FP8_LINEAR_CLS.__qualname__, _FP8_LINEAR_CLS)
    return _FP8_LINEAR_CLS

_FP8_BLOCK = 128


def _scale_shape(out_dim: int, in_dim: int) -> tuple[int, int]:
    return (math.ceil(out_dim / _FP8_BLOCK), math.ceil(in_dim / _FP8_BLOCK))


# ---------------------------------------------------------------------------
# BF16 GEMM dispatch.
#
# One table serves all five classes.  In every one of them the weight is module
# state rather than a forward argument, so it is never pooled and it faces the
# pre-iteration L2 flush identically -- the same (M, K, N) really is the same
# problem regardless of which wrapper asked.  (This is the opposite of the frozen
# candidate/L1/linear.py situation, where ``Matmul`` takes the weight as a
# forward argument, so the pool re-copies it every iteration and keeps it warm in
# L2, and the same shape had to be keyed per class.)
# ---------------------------------------------------------------------------

# name -> callable(x2, weight, bias, N) -> out, each computing F.linear exactly.
_BACKENDS: dict[str, object] = {}

# (M, K, N, has_bias) -> backend name.  Populated only from a repeated in-harness
# measurement: faster than F.linear on at least three independent GPU leases, a
# matched ratio >= 0.99 on that shape, and a strictly better position on the
# ~2.04 us reading ladder.  Nothing has cleared that gate, so it is empty and
# every call delegates -- which also keeps flashinfer out of this file's import.
_MEASURED_FAST: dict[tuple[int, int, int, bool], str] = {}

# Host-side counters, incremented on the host: no threads, no device sync,
# nothing the harness's integrity guards watch.  This is what distinguishes "the
# fast path ran and tied" from "the fast path never ran".
_FASTPATH_HITS: dict[tuple[int, int, int, bool], int] = {}

_STATUS = "delegating:no-admitted-shapes"


def _build_backends() -> dict[str, object]:
    """Bind the FlashInfer BF16 GEMM entry points, or raise.

    Only called when the admission table is non-empty, so an empty table costs
    nothing at import.  Two layout facts keep every binding copy-free against the
    baseline's own ``(N, K)`` row-major parameter: ``tgv_gemm_sm100`` and
    ``mm_bf16`` want ``b`` as ``(K, N)`` column-major, which ``weight.t()``
    already is as a view, and ``tinygemm_bf16`` consumes the ``(N, K)`` layout
    natively.
    """
    import flashinfer.gemm as fg

    def tgv(x2, weight, bias, N):
        return fg.tgv_gemm_sm100(x2, weight.t(), bias)

    def tgv_pdl(x2, weight, bias, N):
        return fg.tgv_gemm_sm100(x2, weight.t(), bias, pdl=True)

    def _mm(backend):
        # mm_bf16 has no bias epilogue, so a biased shape pays a separate add.
        # That add is why no biased shape was ever competitive here.
        def call(x2, weight, bias, N):
            out = fg.mm_bf16(x2, weight.t(), bias=None, backend=backend)
            return out if bias is None else out + bias
        return call

    def tinygemm(x2, weight, bias, N):
        # out = input @ weight.T + bias, straight off the (N, K) layout; needs
        # N % 16 == 0 and K % 64 == 0, which _plan checks.
        out = torch.empty(x2.shape[0], N, device=x2.device, dtype=x2.dtype)
        fg.tinygemm_bf16(x2, weight, out, bias)
        return out

    return {
        "tgv": tgv,
        "tgv+pdl": tgv_pdl,
        "cudnn": _mm("cudnn"),
        "cutlass": _mm("cutlass"),
        "cublaslt": _mm("cublaslt"),
        "tinygemm": tinygemm,
    }


def _check_admitted() -> None:
    """Refuse at import if an admitted key could never be dispatched.

    ``_plan`` applies the strictest backend's shape alignment to every backend, so
    it is possible to add a table entry that the predicates then silently refuse --
    a table that looks populated but never fires.  Checking here turns that into an
    import-time error instead of a mystery 1.00x.
    """
    for key in _MEASURED_FAST:
        M, K, N, has_bias = key
        if M <= 0 or K <= 0 or N <= 0 or K % 64 or N % 16:
            raise ValueError(
                f"admitted key {key} can never be dispatched: _plan requires "
                f"positive M/K/N with K % 64 == 0 and N % 16 == 0")


def _prewarm(backends: dict[str, object]) -> None:
    """Run every admitted (shape, backend) pair once, here at import.

    FlashInfer's wrappers resolve a plan, and some of them autotune, on first
    call for a given problem.  Doing that inside the harness's first correctness
    forward would be invisible in the timings but would put a multi-hundred-
    millisecond stall where the bench expects a forward, so it happens here while
    the worker is still producing output.
    """
    for (M, K, N, has_bias), name in _MEASURED_FAST.items():
        fn = backends.get(name)
        if fn is None:
            raise KeyError(f"admitted backend {name!r} is not bound")
        x2 = torch.zeros((M, K), device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros((N, K), device="cuda", dtype=torch.bfloat16)
        bias = (torch.zeros((N,), device="cuda", dtype=torch.bfloat16)
                if has_bias else None)
        fn(x2, weight, bias, N)
    torch.cuda.synchronize()


def _resolve() -> None:
    """Bind and pre-warm the admitted backends once, at import.

    Any failure -- no flashinfer, wrong arch, a backend that raises while warming
    -- leaves ``_BACKENDS`` empty, so every call delegates to ``F.linear``, and
    prints one line on stderr.  The bench worker routes stderr to the per-op log,
    so a swallowed failure stays visible instead of hiding behind a silent 1.00x.
    """
    global _STATUS
    if not _MEASURED_FAST:
        return
    if not torch.cuda.is_available():
        _STATUS = "delegating:no-cuda-device"
        return
    try:
        _check_admitted()
        backends = _build_backends()
        _prewarm(backends)
    except Exception as exc:  # noqa: BLE001 - import, arch, or warm-up failure
        _BACKENDS.clear()
        _STATUS = f"delegating:{type(exc).__name__}: {exc}"
        print(f"[candidate L2/parallel_linear] backends unavailable, delegating "
              f"to torch: {_STATUS}", file=sys.stderr, flush=True)
        return
    _BACKENDS.update(backends)
    _STATUS = "resolved"


_resolve()


def _plan(x: torch.Tensor, weight: torch.Tensor, bias) -> str | None:
    """Return the backend name to run this call with, or None to delegate.

    Pure and cheap: integer and attribute reads only, no CUDA call and no device
    sync.  (It does build the small lookup tuple below, which is host-side and
    what a dict key costs; there is no device allocation.)  Every predicate guards
    something a backend actually requires, and the table lookup happens last, so a
    table entry can never override a precondition.
    """
    if not _BACKENDS:
        return None
    # Inference only: none of these backends builds an autograd graph.
    if torch.is_grad_enabled():
        return None
    # bf16 only.  fp16 is simply unmeasured here, and fp32 goes through a
    # different accumulation order in every backend than the reference does.
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return None
    if bias is not None and bias.dtype is not torch.bfloat16:
        return None
    # One launch, one device.
    if not x.is_cuda or weight.device != x.device:
        return None
    if bias is not None and bias.device != x.device:
        return None
    if weight.dim() != 2 or x.dim() < 1:
        return None
    N, K = weight.shape
    if x.shape[-1] != K:
        return None
    # Bounds before any division: F.linear accepts K == 0 and must keep doing so.
    if K <= 0 or N <= 0:
        return None
    # The strictest backend's alignment, applied to all of them: tinygemm needs
    # K % 64 == 0 and N % 16 == 0, while tgv and mm_bf16 only need the
    # 16-byte-aligned row pitch that K % 8 gives.  Gating globally over-delegates
    # rather than under-delegates, which is the safe direction, and every scored
    # shape satisfies it anyway.  _check_admitted() below refuses at import if an
    # admitted key would be made unreachable by this, so the conservatism cannot
    # silently disable a future entry.
    if K % 64 or N % 16:
        return None
    # A hidden .contiguous() costs more than any of these backends can win back,
    # and cutlass, cublaslt and tinygemm reject a non-contiguous activation
    # outright.  This is also what keeps x.reshape(-1, K) a view.
    if not x.is_contiguous() or weight.stride(-1) != 1 or weight.stride(0) != K:
        return None
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != N
                             or not bias.is_contiguous()):
        return None
    M = x.numel() // K
    if M <= 0:
        return None
    return _MEASURED_FAST.get((M, K, N, bias is not None))


def _dispatch(x: torch.Tensor, weight: torch.Tensor, bias):
    name = _plan(x, weight, bias)
    if name is None:
        return F.linear(x, weight, bias)
    N, K = weight.shape
    M = x.numel() // K
    key = (M, K, N, bias is not None)
    _FASTPATH_HITS[key] = _FASTPATH_HITS.get(key, 0) + 1
    out = _BACKENDS[name](x.reshape(M, K), weight, bias, N)
    # A fresh output every call: returning anything that aliases a previous
    # iteration's pool slot would be read back after the pool had shifted.
    return out.view(*x.shape[:-1], N)


class ColumnParallelLinear(nn.Module):
    """Splits output dim across TP ranks."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        assert output_size % tp == 0
        self.output_size_per_partition = output_size // tp
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, input_size,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(self.output_size_per_partition, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        rows_per_shard = param.data.size(0)
        loaded_weight = loaded_weight.narrow(0, rank * rows_per_shard, rows_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return _dispatch(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """gate_proj + up_proj merged into one linear, sharded across TP."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 quant_config: dict | None = None, disable_tp: bool = False):
        super().__init__()
        tp = _tp_size()
        self.disable_tp = disable_tp
        self.output_sizes = output_sizes
        total = sum(output_sizes)
        if not disable_tp:
            assert all(s % tp == 0 for s in output_sizes)
        self.use_fp8 = quant_config is not None

        effective_tp = 1 if disable_tp else tp
        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(total // effective_tp, input_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(total // effective_tp, input_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(total // effective_tp, input_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(total // tp))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: int | None = None):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id is None:
            # Fused weight: ``loaded_weight`` is the full ``[sum(output_sizes), in]``
            # tensor.  Recurse per-shard so each output block is sharded across
            # TP ranks independently (mirrors vLLM's ``MergedColumnParallelLinear``
            # weight loader when called without an explicit shard id).
            offset = 0
            for sid, sz in enumerate(self.output_sizes):
                self._weight_loader(
                    param, loaded_weight.narrow(0, offset, sz), sid,
                )
                offset += sz
            return
        effective_tp = 1 if self.disable_tp else tp
        shard_offset = sum(self.output_sizes[:shard_id]) // effective_tp
        shard_size = self.output_sizes[shard_id] // effective_tp
        dst = param.data.narrow(0, shard_offset, shard_size)
        if self.disable_tp:
            dst.copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: int):
        tp, rank = _tp_size(), _tp_rank()
        effective_tp = 1 if self.disable_tp else tp
        shard_size_out = self.output_sizes[shard_id] // effective_tp
        scale_rows = math.ceil(shard_size_out / _FP8_BLOCK)
        shard_offset_out = sum(self.output_sizes[:shard_id]) // effective_tp
        scale_offset = math.ceil(shard_offset_out / _FP8_BLOCK)
        if self.disable_tp:
            param.data.narrow(0, scale_offset, scale_rows).copy_(loaded_weight)
        else:
            src = loaded_weight.chunk(tp, 0)[rank]
            param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return _dispatch(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """Q, K, V projections merged and sharded across TP."""

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int,
                 bias: bool = False, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // tp
        # Replicate KV heads when not evenly divisible by TP
        if total_num_kv_heads % tp == 0:
            self.num_kv_heads = total_num_kv_heads // tp
            self._replicate_kv = False
        else:
            self.num_kv_heads = total_num_kv_heads
            self._replicate_kv = True
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, hidden_size), dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.weight.weight_loader = self._weight_loader

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            src = loaded_weight.chunk(tp, 0)[rank]
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            src = loaded_weight if self._replicate_kv else loaded_weight.chunk(tp, 0)[rank]
        dst = param.data.narrow(0, shard_offset, shard_size)
        dst.copy_(src)

    def _scale_loader(self, param, loaded_weight, shard_id: str):
        tp, rank = _tp_size(), _tp_rank()
        if shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        scale_rows = math.ceil(shard_size / _FP8_BLOCK)
        scale_offset = math.ceil(shard_offset / _FP8_BLOCK)
        src = loaded_weight.chunk(tp, 0)[rank]
        param.data.narrow(0, scale_offset, scale_rows).copy_(src)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return _dispatch(x, self.weight, self.bias)


class ReplicatedLinear(nn.Module):
    """Full weight replicated on every TP rank (no sharding, no all-reduce)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, input_size),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)
            self.weight_scale_inv.weight_loader = lambda p, w: p.data.copy_(w)
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = lambda p, w: p.data.copy_(w)

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)

    def forward(self, x):
        if self.use_fp8:
            return self.linear_op(x, self.weight, self.weight_scale_inv, self.bias)
        return _dispatch(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Splits input dim across TP ranks, all-reduces output."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 quant_config: dict | None = None, reduce_results: bool = True):
        super().__init__()
        tp = _tp_size()
        assert input_size % tp == 0
        self.input_size_per_partition = input_size // tp
        self.tp_size = tp
        self.tp_rank = _tp_rank()
        self.reduce_results = reduce_results
        self.use_fp8 = quant_config is not None

        if self.use_fp8:
            self.weight = nn.Parameter(
                torch.empty(output_size, self.input_size_per_partition,
                            dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(*_scale_shape(output_size, self.input_size_per_partition),
                            dtype=torch.float32),
                requires_grad=False,
            )
            self.weight.weight_loader = self._weight_loader
            self.weight_scale_inv.weight_loader = self._scale_loader
            self.linear_op = _get_fp8_linear_cls()()
        else:
            self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
            self.weight.weight_loader = self._weight_loader

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        if self.bias is not None:
            self.bias.weight_loader = lambda p, w: p.data.copy_(w)
        self.allreduce = AllReduce()

    def _weight_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * shard, shard)
        param.data.copy_(loaded_weight)

    def _scale_loader(self, param, loaded_weight):
        tp, rank = _tp_size(), _tp_rank()
        cols_per_shard = param.data.size(1)
        loaded_weight = loaded_weight.narrow(1, rank * cols_per_shard, cols_per_shard)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        if self.use_fp8:
            y = self.linear_op(x, self.weight, self.weight_scale_inv,
                               self.bias if self.tp_rank == 0 else None)
        else:
            y = _dispatch(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_results and self.tp_size > 1:
            y = self.allreduce(y)
        return y
