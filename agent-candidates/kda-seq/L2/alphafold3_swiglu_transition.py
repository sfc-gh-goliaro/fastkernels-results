"""SwiGLU transition composites for AlphaFold3 on B200 (sm_100).

``SwiGLUTransition`` (LayerNorm -> SwiGLU -> Linear -> mask) and
``ConditionedTransitionBlock`` (AdaLN -> SwiGLU -> sigmoid-gated Linear -> mask)
are not GPU-throughput-bound at the captured shapes. Measured in-harness on the
leased B200 (``profile/p1-baseline-survey/results.md``): the scored window is
99-107 us for ``SwiGLUTransition`` and 210-266 us for
``ConditionedTransitionBlock``, while the sum of GPU kernel time is 11-24% of
that. The rest is CPU -- 9-10 and 20 kernel launches issued through about a dozen
nested ``nn.Module.__call__`` frames (AdaLN / SwiGLU / Sigmoid / SiLU / LayerNorm
/ Linear wrappers, each with its own dispatch).

The same measurements calibrate what a fix is worth::

    window ~ 15 us floor                        (pool re-copy + event records)
           + ~4 us for one torch-level call, its first kernel launch included
           + ~4 us per additional kernel launch inside that call
           + the GPU tail, which nothing overlaps

so kernel *count* is the budget, not FLOPs and not bandwidth. Argument count is
free (eleven arguments measured identical to one), and a GEMM dispatch costs the
same ~4 us as an elementwise one.

Each class therefore collapses to one custom operator backed by the fewest
kernels the data flow allows: two for ``SwiGLUTransition`` and three for
``ConditionedTransitionBlock``. The rule that fixes those counts is that a stage
folds into its consumer only when the consumer's CTA already owns the whole
reduction extent that stage produces:

* A LayerNorm/AdaLN prologue folds. The CTA re-reads its own ``BM x K`` rows and
  reduces them itself, which is O(rows) extra work and no replication.
* The up-projection does *not* fold into the down-projection: it produces the
  down-projection's *reduction* dimension, so folding replicates it once per
  output tile -- roughly 3.7x the chain FLOPs, about 6 us of added GPU time on
  the c_a=768 case, against the ~4 us launch it would remove.
* The outer sigmoid gate *does* fold into the down-projection epilogue: it is an
  elementwise factor of the output, so the CTA needs exactly the ``[BM, BN]``
  gate tile matching the output tile it already owns. The mask folds there too,
  for free.

Correctness is an accumulation-order argument rather than a tolerance argument.
Every point at which the baseline materializes a bf16 tensor is a rounding point
here, and everything between two of them stays fp32. See ``_ROUNDING_BOUNDARIES``
below for the enumerated list.

The claim that stops there is deliberate. It says what *this* code does; it does
not say ATen is fp32 between the same points. ATen's CUDA LayerNorm is free to use
a Welford or tree reduction rather than the mean-then-squared-deviations form used
here, and ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` is
on by default, so cuBLAS may reduce a bf16 GEMM at reduced precision. Neither was
observable at the captured reduction extents on this device -- the fused output is
bitwise equal to the baseline on most scored cases and within one output ulp on
the rest -- but the residual is "whatever ATen chose, plus reduction order", not
"reduction order" alone. The boundary-list reference in
``profile/p1-numerics/`` is what pins the part of the claim this file controls.

Anything outside the declared fast-path domain -- another dtype, a mask that does
not flatten row-for-row onto the activation, a non-contiguous activation, a
gradient-enabled or autocast call, a reduction extent that is not a multiple of
16 -- runs the baseline module composition instead. ``_FALLBACK_CALLS`` counts
those, so "the fast path ran" is distinguishable from "everything quietly
delegated and tied".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN, SwiGLU

# The fallback is deliberately *not* uniform, and it matters for reading the
# skeleton's measured speedup. The relative imports above resolve through the
# candidate finder: ``..L1.*`` finds the frozen L1 winners, while
# ``.alphafold3_swiglu`` has no candidate file and aliases to the baseline L2
# module -- whose own ``..L1`` imports were already bound to baseline L1. So
# ``AdaLN``/``SwiGLU`` run baseline Linear/SiLU/Sigmoid/LayerNorm internally,
# while ``self.layer_norm`` (SwiGLUTransition), ``self.linear_out``,
# ``self.linear_g`` and ``self.sigmoid`` are frozen winners. A pure-fallback
# candidate therefore measures slightly *above* 1.0x, not at it.

__all__ = ["ConditionedTransitionBlock", "SwiGLUTransition"]

# Every bf16 materialization point in the baseline, in evaluation order. The
# fused kernels round at exactly these points and keep fp32 everywhere between,
# which is what makes the difference against the baseline an accumulation-order
# difference rather than a precision one.
#
# SwiGLUTransition:
#   xn = LayerNorm(x)                      -> bf16
#   ha = xn @ Wa^T                         -> bf16
#   silu(ha)                               -> bf16   (fp32 opmath, then round)
#   hb = xn @ Wb^T                         -> bf16
#   hh = silu(ha) * hb                     -> bf16
#   d  = hh @ Wout^T                       -> bf16
#   out = d * mask                         -> bf16
#
# ConditionedTransitionBlock:
#   sn = LayerNorm_s(s)                    -> bf16   (weight-only affine, eps_s)
#   g1 = sn @ Wag^T + bag                  -> bf16
#   sigmoid(g1)                            -> bf16
#   an = LayerNorm_a(a)                    -> bf16   (no affine, eps_a)
#   ls = sn @ Was^T                        -> bf16
#   t  = an + ls                           -> bf16
#   a1 = sigmoid(g1) * t                   -> bf16
#   ha = a1 @ Wa^T                         -> bf16
#   silu(ha)                               -> bf16
#   hb = a1 @ Wb^T                         -> bf16
#   hh = silu(ha) * hb                     -> bf16
#   g2 = s @ Wg^T + bg                     -> bf16   (raw s, not sn)
#   sigmoid(g2)                            -> bf16
#   d  = hh @ Wout^T                       -> bf16
#   p  = sigmoid(g2) * d                   -> bf16
#   out = p * mask                         -> bf16
#
# The rounding *before* the mask multiply is not optional even though the
# harness materializes an all-ones mask: the declared fast-path domain accepts
# other masks, and for those the two orders differ.
_ROUNDING_BOUNDARIES = __doc__  # documentation anchor; see the comment above

# Reduction extents are stepped 16 at a time (one mma K tile), so a K that is not
# a multiple of 16 would need a masked tail this kernel does not write.
_K_MULTIPLE = 16
# Above this row count the small-M design loses its point: the grid is wide
# enough that cuBLAS is already efficient and the baseline is not launch-bound.
_M_MAX = 4096
# 128-bit loads need 16-byte-aligned bases. The harness's shifting pool hands out
# 256-byte-aligned slots, but that is the pool's property, not a guarantee.
_ALIGN_BYTES = 16

# Incremented only when a call takes the baseline composition. The fast path adds
# no work here at all, so this is free on the path that matters, and it is what
# separates "fused and equal" from "silently delegated and equal".
_FALLBACK_CALLS: dict[str, int] = {"SwiGLUTransition": 0, "ConditionedTransitionBlock": 0}


def fallback_calls() -> dict[str, int]:
    """Fallback counts per class. Zero on every captured case, by construction."""
    return dict(_FALLBACK_CALLS)


def reset_fallback_calls() -> None:
    for key in _FALLBACK_CALLS:
        _FALLBACK_CALLS[key] = 0


# ---------------------------------------------------------------------------
# Fast-path predicate: the specification.
#
# These helpers define the declared domain, in readable form, and are what the
# unit tests exercise. The *enforcement* is the identical predicate in
# ``_af3_swiglu_transition.cu``, which the operator applies to its own arguments
# and which returns None for anything outside the domain. Two implementations of
# one predicate is a drift risk, so ``profile/p1-contract/check_fastpath.py``
# asserts they agree case by case over every positive and negative input it knows
# about.
#
# The split is not gratuitous. Running this in Python on every call measured 7.5 us
# for SwiGLUTransition and 13.8 us for ConditionedTransitionBlock
# (``profile/p1-predicate/results.txt``) against a ~15 us window floor, and the
# verdict cannot be cached: a lazy negation view or a same-storage reshape leaves a
# parameter's identity and address untouched, so a cached verdict outlives the
# metadata it validated and the kernel reads storage that does not represent the
# tensor. In C++ the same comparisons cost nothing.
#
# Integer and attribute comparisons only, in both implementations: no CUDA call, no
# device synchronization, nothing that could turn the guard into a stall. Everything
# rejected reaches the baseline composition, which is the point -- a predicate that
# admits an input the kernel is not written for produces a wrong answer, whereas one
# that rejects an input it could have handled only costs speed.
# ---------------------------------------------------------------------------
def _flattens_row_for_row(shape: tuple, leading: tuple) -> bool:
    """True iff a contiguous tensor of *shape* addresses *leading* row-for-row.

    The kernels index rows of a flat ``[M, K]`` view, so an operand with fewer
    dimensions than the activation is only usable if the baseline's broadcast of
    it would neither replicate a row nor change the output shape.

    Suffix equality alone is not enough. ``mask[3, 4]`` against ``x[2, 3, 4, 8]``
    has the right suffix, but the baseline broadcast replicates the mask over the
    leading 2, so a flat kernel would read mask elements 12..23 past the end and
    silently mis-gate. The element-count clause is what forces every leading
    dimension of *leading* not covered by *shape* to be 1, which is exactly the
    condition under which the broadcast is a no-op.
    """
    rank = len(shape)
    if rank == 0 or rank > len(leading):
        return False
    if shape != leading[-rank:]:
        return False
    rows = 1
    for dim in leading:
        rows *= dim
    elems = 1
    for dim in shape:
        elems *= dim
    return elems == rows


def _materialize(t: torch.Tensor | None) -> torch.Tensor | None:
    """Turn a lazy negation/conjugation view into a real tensor.

    Used on the *fallback* path only, for the forward *arguments*. The affine
    parameters need the same treatment but cannot get it this way -- resolving them
    would mean writing to ``p.data``, which is the module's state, not ours. See
    ``SwiGLUTransition._normalize`` for how those are handled instead.

    The baseline's own ``F.layer_norm`` handles such a view correctly, but the frozen
    ``candidate/L1/layer_norm.py`` this module composes with takes ``data_ptr()``
    behind a predicate that does not test ``is_neg``, so delegating the view
    unresolved would reproduce that file's answer rather than the baseline's. The
    frozen winners cannot be edited, so the resolve happens here. It costs one
    attribute test per fallback call and nothing at all on the fast path, which has
    already rejected these inputs.
    """
    if t is None:
        return None
    if t.is_neg() or t.is_conj():
        return t.resolve_conj().resolve_neg()
    return t


def _resolved(t: torch.Tensor) -> bool:
    """Reject tensors whose value differs from the bytes in their storage.

    A lazy negation view -- ``torch._neg_view(x)`` -- is contiguous, bf16, CUDA,
    correctly shaped and correctly aligned, and reads as ``-x`` to every ATen op,
    while its storage still holds ``+x``. A kernel that takes ``data_ptr()`` and
    indexes raw memory therefore returns the wrong *sign*, silently, on an input
    the rest of the predicate is happy with. ``is_conj`` is the same hazard for
    complex tensors and is checked for the same reason even though this operator
    only admits bf16.
    """
    return not t.is_neg() and not t.is_conj()


def _param_ok(p: torch.Tensor | None, shape: tuple, dev: torch.device) -> bool:
    """A read parameter must match its expected rank, shape, dtype, layout,
    contiguity, device and alignment, and must not be a lazy view.

    This runs on **every** eligible call, for every parameter, and the verdict is
    deliberately not cached. Caching it was a wrong-answer bug: ``p.data`` can be
    replaced with a lazy negation view or reshaped onto the same storage, which
    keeps both the ``Parameter`` object's identity and its address, so no key built
    from identity, address, dtype and device can notice. Two reproductions, both on
    ``linear_out.weight`` after one warm forward:

      ``p.data = torch._neg_view(p.data)``  -> 33.5% of elements matched
      ``p.data = p.data.view(512, 128)``    -> no fallback, wrong answer, right shape

    ``None`` is the first clause so an absent parameter short-circuits before
    anything dereferences it.
    """
    return (p is not None
            and p.dtype is torch.bfloat16
            and p.layout is torch.strided
            and p.device == dev
            and p.shape == shape
            and p.is_contiguous()
            and _resolved(p)
            and p.data_ptr() % _ALIGN_BYTES == 0)


def _activation_ok(t: torch.Tensor, k: int) -> bool:
    return (t.dtype is torch.bfloat16
            and t.is_cuda
            and t.is_contiguous()
            and _resolved(t)
            and t.dim() >= 2
            and t.shape[-1] == k
            and t.data_ptr() % _ALIGN_BYTES == 0)


def _mask_ok(mask: torch.Tensor | None, leading: tuple, dev: torch.device) -> bool:
    """``mask is None`` is admissible and cheaper than the baseline's own path.

    The baseline materializes ``x.new_ones(x.shape[:-1])`` and multiplies by it,
    which is a launch plus an elementwise pass. Multiplying by an exact 1.0 is
    the identity on the rounding chain for every normal, subnormal, signed zero
    and infinity, so the fast path simply omits the factor -- the same answer,
    one launch fewer. The one exception is a NaN payload, which the multiply
    canonicalizes to ``0x7fff`` and omitting it preserves; the harness rejects any
    NaN output either way, so no case can distinguish them.

    A present mask must share the activation's dtype. ``out * mask`` promotes, so
    an fp32 mask would change the output *dtype*, not merely its values, and the
    harness checks dtype on the first correctness round.
    """
    if mask is None:
        return True
    return (mask.dtype is torch.bfloat16
            and mask.is_cuda
            and mask.device == dev
            and mask.is_contiguous()
            and _resolved(mask)
            and _flattens_row_for_row(tuple(mask.shape), leading)
            and mask.data_ptr() % _ALIGN_BYTES == 0)


def _rows_ok(m: int, *ks: int) -> bool:
    if m < 1 or m > _M_MAX:
        return False
    return all(k > 0 and k % _K_MULTIPLE == 0 for k in ks)


def _ambient_ok() -> bool:
    """Grad mode being *enabled* is the test, not whether an operand currently
    requires grad: the fused kernels allocate with ``at::empty`` and launch raw
    kernels, so they record nothing for autograd, and a caller who has not
    entered ``no_grad`` may attach ``requires_grad`` later in the same graph.
    Autocast would insert its own casts around the composition and change which
    dtype the baseline actually computes in."""
    return not torch.is_grad_enabled() and not torch.is_autocast_enabled("cuda")


class SwiGLUTransition(nn.Module):
    """AF3 Algorithm 11: SwiGLU-based transition.

    Args:
        c_in: Input channel dimension
        n: Factor multiplied to c_in for hidden dimension
    """

    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

        # Nothing is derived from a weight here, and nothing is cached from one.
        # The harness moves the module to the device and casts its parameters
        # *before* loading the baseline's state dict, so anything precomputed from
        # ``weight`` in ``__init__`` would be provably stale by the first forward.

    def _op_inputs(self) -> tuple:
        """The read-only operator arguments, gathered but **not** validated here.

        Validation lives in the operator (see the predicate at the top of
        ``_af3_swiglu_transition.cu``), which returns ``None`` for anything outside
        the declared domain. Two reasons it is there and not here:

        * It cannot be cached. Replacing ``p.data`` with a lazy negation view, or
          reshaping it onto the same storage, changes neither the ``Parameter``
          object's identity nor its address, so any key built from those serves a
          stale verdict -- measured at 33.5% of elements matching in the negation
          case, and a wrong answer with a plausible output shape in the reshape case.
        * Run uncached in Python it costs 7.5 us per call for these five parameters
          and 13.8 us for ``ConditionedTransitionBlock``'s nine
          (``profile/p1-predicate/results.txt``), against a ~15 us window floor. The
          same comparisons in C++ are free.

        ``_param_ok`` and friends below remain the readable specification and are
        unit-tested; ``check_fastpath.py`` asserts they agree with the operator case
        by case, so the two cannot drift.

        Nothing here is derived from a weight, because the harness moves and casts
        the module *before* loading the state dict.
        """
        ln, sg = self.layer_norm, self.swiglu
        return (ln.weight, ln.bias, sg.linear_a.weight, sg.linear_b.weight,
                self.linear_out.weight, float(ln.eps))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm for the fallback path, avoiding a hazard in a frozen winner.

        Normally this is ``self.layer_norm(x)`` -- the frozen
        ``candidate/L1/layer_norm.py``. But that file's predicate checks shape,
        contiguity and alignment and then takes ``data_ptr()`` *without* testing
        ``is_neg``, so a lazy-negated ``weight`` or ``bias`` makes it read the
        un-negated storage. Measured on a rejected-input fallback: ``matched=0.9752``
        against the harness's 0.99 bar for a negated ``weight``, and ``max_rel=0.32``
        for a negated ``bias``. The operator rejects such a parameter, which is why
        the call lands here -- but reaching the fallback is only half of what AC-3
        asks for; the fallback also has to be right.

        The frozen file cannot be edited, and resolving the view in place would mean
        writing to ``p.data`` -- mutating module state this class does not own. So
        when an affine parameter is lazy, the baseline's own formula is evaluated
        here on resolved local copies, ``promote_fp32`` semantics included: fp32
        input, affine promoted to fp32, ``F.layer_norm``, back to the input dtype.
        Nothing is mutated and nothing is cached.

        Only ``SwiGLUTransition`` needs this. A scan of every parameter position in
        both classes (``check_fastpath.py::check_lazy_every_parameter``) shows every
        other one already agrees with the baseline: the ``Linear`` submodules all
        delegate to ``F.linear``, which honours the view, and ``ConditionedTransition
        Block``'s ``AdaLN`` resolves to the *baseline* L1 LayerNorm. That test is the
        guarantee for the rest, rather than defensive code in every position.
        """
        ln = self.layer_norm
        w, b = ln.weight, ln.bias
        if (w is None or _resolved(w)) and (b is None or _resolved(b)):
            return ln(x)

        w = _materialize(w)
        b = _materialize(b)
        if not ln.promote_fp32:
            return F.layer_norm(x, ln.normalized_shape, w, b, ln.eps)
        return F.layer_norm(
            x.float(), ln.normalized_shape,
            None if w is None else w.float(),
            None if b is None else b.float(),
            ln.eps,
        ).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        # Grad mode and autocast stay here: two cheap calls, and both are ambient
        # state the operator would need version-sensitive ATen APIs to read. Every
        # tensor-metadata clause is enforced inside the operator, which returns None
        # rather than a wrong answer for anything it will not handle.
        if _OP_SWIGLU is not None and _ambient_ok():
            out = _OP_SWIGLU(x, mask, *self._op_inputs(), _VARIANT)
            if out is not None:
                return out

        _FALLBACK_CALLS["SwiGLUTransition"] += 1
        x = _materialize(x)
        mask = _materialize(mask)
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        mask = mask.unsqueeze(-1)

        x = self._normalize(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        x = x * mask

        return x


class ConditionedTransitionBlock(nn.Module):
    """AF3 Algorithm 25: SwiGLU transition with AdaLN-Zero conditioning.

    Submodule names match the reference checkpoint:
    - layer_norm: AdaLN
    - swiglu: SwiGLU
    - linear_g: output gate Linear(c_s, c_a, bias=True)
    - linear_out: Linear(n*c_a, c_a, bias=False)

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
        n: Factor for hidden dimension
    """

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)

        # The baseline stores no c_a/c_s/n attributes on this class, so they are
        # read back off the submodule the kernel needs them from rather than
        # added here -- the state dict has to stay key-for-key identical, and an
        # extra plain attribute is harmless but an extra buffer would not be.
        self._c_a, self._c_s, self._n = c_a, c_s, n

    def _op_inputs(self) -> tuple:
        """As ``SwiGLUTransition._op_inputs``: gathered here, validated in the
        operator. The two ``eps`` values are read independently -- they are separate
        mutable attributes, each defaulting to 1e-5, and using one for both
        normalizations is a real difference the moment a caller changes either."""
        ada, sg = self.layer_norm, self.swiglu
        return (ada.layer_norm_s.weight, ada.linear_g.weight, ada.linear_g.bias,
                ada.linear_s.weight, sg.linear_a.weight, sg.linear_b.weight,
                self.linear_g.weight, self.linear_g.bias, self.linear_out.weight,
                float(ada.layer_norm_a.eps), float(ada.layer_norm_s.eps))

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if _OP_CONDITIONED is not None and _ambient_ok():
            out = _OP_CONDITIONED(a, s, mask, *self._op_inputs(), _VARIANT)
            if out is not None:
                return out

        _FALLBACK_CALLS["ConditionedTransitionBlock"] += 1
        a = _materialize(a)
        s = _materialize(s)
        mask = _materialize(mask)
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        mask = mask.unsqueeze(-1)

        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        a = a * mask

        return a


# ---------------------------------------------------------------------------
# Operator binding.
#
# Compilation happens here, at import, so nothing is deferred into a timed
# forward. The translation unit registers through ``TORCH_LIBRARY`` and is built
# with ``is_python_module=False`` and ``no_implicit_headers=True``: pybind would
# pull ``<torch/extension.h>``, which pushed this workspace's build probe past
# 120 s where the lean unit builds in seconds.
#
# ``_VARIANT`` selects the inner product: 0 = mma.sync.m16n8k16, 1 = the explicit
# fp32 fused-multiply-add reference. Both are compiled, so the two can be
# compared inside one process on identical inputs -- the mma path's diff oracle
# is a runtime switch rather than a rebuild.
# ---------------------------------------------------------------------------
_LIBRARY_NAME = "fk_af3_swiglu_transition_cand"

_VARIANT_MMA = 0
_VARIANT_FMA = 1


def _cuda_source() -> tuple[str, str]:
    """``(source, include_dir)``.

    The device code sits in ``_af3_swiglu_kernels.cuh`` beside this file so a
    standalone nvcc harness can compile the identical kernels with ``-lineinfo``
    for Nsight Compute; ``load_inline`` writes the source into its own build
    directory, so that directory has to be on the include path.
    """
    import pathlib

    here = pathlib.Path(__file__).parent
    return ((here / "_af3_swiglu_transition.cu").read_text(encoding="utf-8"),
            str(here))


def _load_ops():
    """Build the extension and return ``(swiglu_op, conditioned_op)``."""
    import os
    import sys

    from torch.utils.cpp_extension import load_inline

    # cpp_extension otherwise honours the ambient TORCH_CUDA_ARCH_LIST, which in
    # this environment names six architectures -- six nvcc passes over every
    # template instantiation, for five targets that will never run this kernel.
    # Narrowing is the difference between a build that fits inside the harness's
    # wall-clock bound and one that does not, and it is restored afterwards so no
    # later build in this process inherits the narrowed value.
    #
    # It is also a correctness requirement, not only a speed one. The kernels use
    # `mma.sync.aligned.m16n8k16...bf16`, which does not exist below sm_80, so an
    # unfiltered ambient list containing 7.5 makes nvcc fail outright. When a
    # device is visible its own capability is used (never hardcoded, so it cannot
    # name the wrong arch); with no device visible -- a CPU-only import, as some of
    # the checks under profile/ do -- the ambient list is filtered to the
    # architectures this code can actually target.
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    else:
        kept = []
        for entry in (previous or "").split():
            head = entry.split("+", 1)[0]
            try:
                if float(head) >= 8.0:
                    kept.append(entry)
            except ValueError:
                continue
        os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(kept) if kept else "8.0"
    # One line on stderr before the build starts. The harness's stall watchdog
    # watches stderr mtime, so a long cold build has to look alive rather than
    # hung -- and a build that fails must leave a trace, since a swallowed
    # failure would present as a silent 1.00x.
    print(f"[{_LIBRARY_NAME}] building fused SwiGLU transition extension "
          f"(arch={os.environ.get('TORCH_CUDA_ARCH_LIST')})",
          file=sys.stderr, flush=True)
    source, include_dir = _cuda_source()
    try:
        load_inline(
            name=_LIBRARY_NAME,
            cpp_sources="",
            cuda_sources=source,
            extra_include_paths=[include_dir],
            # No -use_fast_math: it is off by default, and turning it on would
            # swap expf for the reduced-accuracy intrinsic and enable
            # contractions the fp32 reference inner product does not perform.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            is_python_module=False,
            no_implicit_headers=True,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    print(f"[{_LIBRARY_NAME}] extension ready", file=sys.stderr, flush=True)
    # Bind the overloads, not the packets: a packet re-resolves the overload from
    # the argument types on every call, and forward is launch-latency bound.
    lib = getattr(torch.ops, _LIBRARY_NAME)
    return lib.swiglu_transition.default, lib.conditioned_transition.default


try:
    _OP_SWIGLU, _OP_CONDITIONED = _load_ops()
except Exception as exc:  # noqa: BLE001 - degrade loudly, do not take the module
    # down with it: an import failure would cost every case at once, and a
    # silently swallowed one would look like a candidate that simply tied.
    import sys as _sys
    import traceback as _traceback

    print(f"[{_LIBRARY_NAME}] BUILD FAILED, falling back to the baseline "
          f"composition on every call: {type(exc).__name__}: {exc}",
          file=_sys.stderr, flush=True)
    _traceback.print_exc(file=_sys.stderr)
    _sys.stderr.flush()
    _OP_SWIGLU = None
    _OP_CONDITIONED = None

def _default_variant() -> int:
    """Which inner product the operator uses.

    The tensor-core path by default; ``FK_AF3_SWIGLU_INNER=fma`` selects the fp32
    reference so the two can be compared in-harness and in the scripts under
    ``profile/``. Read once here rather than per call -- ``forward`` is
    launch-latency bound and has no business touching the environment.
    """
    import os

    return (_VARIANT_FMA
            if os.environ.get("FK_AF3_SWIGLU_INNER", "mma").lower() == "fma"
            else _VARIANT_MMA)


_VARIANT = _default_variant()
