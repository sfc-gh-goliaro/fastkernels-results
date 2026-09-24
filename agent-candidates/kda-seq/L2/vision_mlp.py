"""Vision MLP with the activation folded into the fc1 GEMM epilogue, for B200 (sm_100).

Drop-in for ``baseline.py``'s ``self.fc2(self.act_fn(self.fc1(x)))``, which evaluates as
three device kernels: a GEMM for ``fc1``, a standalone elementwise activation over the
[M, 4304] hidden tensor, and a GEMM for ``fc2``.

The middle kernel is the one worth removing, and not for the usual reason. At the hottest
scored shape (M = 20680) it moves 20680 * 4304 = 89.0 M elements, or 356.0 MB read plus
written, and takes 0.199 ms measured under the benchmark's own conditions. Net of the 31 us
the benchmark's input copy costs inside the same timed region, that is 0.168 ms, so about
2.12 TB/s on a part that sustains roughly 8 TB/s -- it is arithmetic-bound on the exact
``erf`` evaluation, not bandwidth-bound. Widening its memory access would buy nothing.
Folding it into the ``fc1`` epilogue deletes it instead: cuBLASLt applies bias and
activation to the accumulator in registers after the TMEM load and before the store, so
the hidden tensor is never written unactivated and never re-read. Measured cost of the
fold is 0.248 - 0.227 = 21 us added to ``fc1``, against 0.199 ms removed -- both timings
include the input copy, so the difference is theirs alone. Three kernels become two, which
a profiler capture confirms directly.

``torch._addmm_activation(bias, a, b, use_gelu=True)`` is the ATen entry to that epilogue;
``torch.addmm`` reaches the plain bias epilogue for ``fc2``, as ``F.linear`` does.

Which GELU the epilogue computes
--------------------------------
It is the **tanh approximation**, not the exact-``erf`` GELU the benchmark supplies as
``act_fn``. This was measured, not assumed: against fp32 references rounded to bfloat16 the
epilogue result is 99.936 % bit-identical to ``approximate="tanh"`` and 96.327 %
bit-identical to ``approximate="none"``, with mean |delta| 5.568e-07 versus 4.868e-05 -- a
factor of 87. Evidence: ``profile/phase1_probe/probe3.out``. The flavour must be established
on CUDA: the CPU implementation of the same operator is bit-exact erf, so a host-side check
reports the opposite.

So this module substitutes one activation formula for another. That is legitimate here, and
only at bfloat16. The largest erf-vs-tanh GELU gap is 4.7324e-04, at |x| = 2.6989, and one
ULP in the binade containing that point is:

    bfloat16  1.5625e-02   gap is   33x BELOW one ULP   <- admitted
    float16   1.9531e-03   gap is  4.1x BELOW one ULP   <- not admitted, see below
    float32   2.3842e-07   gap is  1985x ABOVE one ULP  <- must never be admitted

float32 is excluded because its tolerance is atol 1e-5 / rtol 1e-3 and the substitution
would be plainly visible against it. float16 is excluded for a weaker but deliberate
reason: its margin is 8x thinner than bfloat16's, no float16 correctness data was ever
measured for this operator, and float16 is not a benchmarked dtype here, so admitting it
would trade an unmeasured risk for no gain. The dtype test below is an allowlist, so
adding float16 later is a one-line change gated on measuring it.

What the residual difference actually is
---------------------------------------
The fused path applies the activation to the fp32 accumulator and rounds once, where the
baseline rounds the pre-activation to bfloat16 first and then activates. That missing
intermediate rounding -- not the choice of flavour -- is the whole of the remaining
difference, which was isolated by measurement rather than argued
(``profile/phase1_probe/probe5.out``). Over 1.79 billion element comparisons at M = 64680
against the benchmark's own tolerance:

    fused, addmm for fc2      2 elements outside tolerance   rate 1.1e-09
    fused, F.linear for fc2   2 elements outside tolerance   rate 1.1e-09
    three kernels, tanh GELU  0                              rate 0
    three kernels, erf GELU   0                              rate 0

The third row is the informative one. Rounding the pre-activation to bfloat16 and *then*
applying the tanh flavour reproduces the baseline everywhere, which is direct evidence that
the flavour substitution is invisible exactly as the ULP margins above predict, and that the
residual belongs to the rounding order alone. The first two rows are identical, so the choice
between ``addmm`` and ``F.linear`` for fc2 -- which select different kernels and accumulate
in different orders -- has no bearing on it either; that choice is settled on speed alone.

So the only way to remove the last ~1e-9 of disagreement is to materialize the rounded
pre-activation, which means writing and re-reading the hidden tensor: the three-kernel form
this module exists to avoid. The trade is deliberate and the direction is not close -- one
element in roughly a billion, against a benchmark requirement of 99 % of elements. Nobody
should "fix" this by inserting a rounding.

Why two GEMMs and not one fused kernel
--------------------------------------
Fusing fc1 + activation + fc2 into a single kernel would delete the intermediate entirely
(178.0 MB written and 178.0 MB read at M = 20680). It does not fit the hardware, for three
independent reasons:

* A CTA owning ``BLOCK_M`` rows must hold a ``[BLOCK_M, 1152]`` fp32 accumulator while it
  streams the 4304-wide hidden dimension. TMEM on sm_100 is 128 lanes x 512 fp32 columns =
  256 KB per SM, so 1152 columns cannot live there at all -- the column budget, not the
  byte count, is the binding constraint.
* The same accumulator in shared memory is 128 x 1152 x 4 B = 576 KiB against ~227 KiB
  usable per SM.
* The whole register file is 256 KiB, so it cannot hold it either.

Splitting the second GEMM's N across CTAs would make each CTA recompute the entire hidden
tile, multiplying fc1's work by the split factor. Shrinking ``BLOCK_M`` to about 32 so the
accumulator fits destroys weight reuse instead: 19.83 MB of weights re-read once per 32
rows is roughly 12.8 GB of L2 traffic at M = 20680, against roughly 0.47 GB moved in total
by the two-kernel form. The intermediate therefore stays, and the activation fold is the
win that is actually available.

Deliberately not done
---------------------
* fp8 / nvfp4 GEMMs -- e4m3 rounding of both operands gives ~4-6 % relative error per
  element against a 1 % rtol gate, and because each output is itself a random-sign sum of
  4304 terms the error does not average out.
* CUDA graphs -- the benchmark hands out a fresh ``data_ptr`` every iteration, so a
  static-input graph would need an extra 47.6 MB input copy inside the timed window at
  M = 20680.
* Side streams -- moving work outside the timed region is not a speedup.
* ``torch.compile`` -- compiles inside the measured process, and hides the design.
* Any cache keyed on the input data pointer -- a fresh pointer per iteration by
  construction, so such a cache would address the previous iteration's buffer.
* Chunking M so the intermediate stays L2-resident. 178.0 MB does not fit a 133 MB L2, so
  this looked like the obvious next win. Measured, it loses at every scored shape and every
  chunk size: at M = 20680 the whole-tensor form is 0.406 ms against 0.445 ms for the best
  chunk (16384), 0.466 ms for 8192 and 0.540 ms for 4096. The gap narrows monotonically as
  the chunk grows, i.e. the best chunk is no chunk -- the extra launches and the loss of one
  large GEMM's scheduling outweigh the residency gain. A measured rejection, not an open
  question.
* Replacing either GEMM with a hand-written one. cuBLAS reaches 947.7 TFLOP/s on fc1+GELU
  and 944.1 TFLOP/s on fc2 at M = 20680, rising to 1190.4 and 1092.1 at M = 64680, against
  a calibrated ceiling of 1275-1342 TFLOP/s on this part -- so 82-89 % of achievable at the
  large shapes. The frozen ``..L1.linear`` candidate's own admission table is empty above
  M = 512 because cuBLAS won every larger shape it measured, and hand-written Blackwell
  GEMMs are reported reaching 86 % of cuBLAS without persistent scheduling and 98 % with
  persistent scheduling plus CLC -- i.e. approaching it, not beating it.

Both GEMMs were profiled to settle this rather than argued
(``profile/vision_mlp_gemm_headroom_m1760_vs_m64680/REPORT.md``). At M = 64680 fc1 sits at
95.7 % of tensor-pipe peak and fc2 at 86.3 %, with DRAM at 3-12 % of peak -- compute-bound and
near the roofline, nothing to chase.

The smallest scored shape is the exception, and the reason is occupancy rather than
arithmetic. At M = 1760 fc2's grid is 132 CTAs on 148 SMs, which is **0.89 waves per SM** --
the grid is smaller than the machine. Its tensor pipe reads 37.2 % over elapsed cycles but
55.4 % while active, and the gap between those two is idle SMs, not idle pipes; issue-active
is 5.4 % with long-scoreboard the dominant stall, so it is latency-bound with nothing resident
to hide the loads. cuBLAS does not split K anywhere -- ``grid_dim_z == 1`` in all four
profiled kernels -- so it leaves that parallelism on the table.

Acting on that, the forms available here were measured. Two of them test the remedy the
profile actually points at -- raising the CTA count -- and one tests the other free variable,
the GEMM orientation (``profile/phase1_probe/probe6.out``, ``probe8.out``):

    torch.addmm for fc2 (ships)   M=1760 1.109x   geomean 1.3189x
    F.linear for fc2              M=1760 1.090x   geomean 1.1502x
    torch.mm + bias add           M=1760 0.929x   geomean 0.9706x
    fc2 as one batched baddbmm    M=1760 1.026x   geomean 1.2073x
    transposed: W1 @ X.T, W2 @ H  M=1760 0.805x   geomean 0.8082x

The batched form is the honest occupancy test, and it does raise occupancy: profiled at
M = 1760 its single fc2 launch has **144 CTAs against 132**, 0.973 waves per SM against 0.892,
tensor pipe 45.7 % against 37.4 %, and the fc2 kernel alone drops from 41.73 us to 36.26 us --
cuBLAS even selects a finer 128x112 tile for it instead of 192x80
(``profile/vision_mlp_m1760_batched_fc2_occupancy/``). It still loses end to end, at every
shape, under the benchmark's L2-flushed timing. A faster kernel in isolation is not a faster
operator.

The transposed orientation loses outright at 0.808x geomean: the column-major layout costs far
more than any accumulation-order difference could return, so it is excluded on speed before its
numerics matter.

Filling that grid properly needs a kernel that tiles differently from the inside, not a
different way of calling cuBLAS.

Everything this module cannot serve safely runs the baseline expression verbatim -- same
values, and the same observable behaviour including exception types: a non-CUDA tensor, any
dtype outside the allowlist, a non-contiguous or empty input, a wrong trailing dimension, a
bias-less or fp8 or tensor-parallel configuration, a tensor subclass, a forward-mode dual
tensor, anything needing autograd, an activation the probe below does not recognize as
GELU-shaped, an activation object replaced or mutated after construction, and a build whose
``_addmm_activation`` does not pass the capability check. Rejection is decided before the
launch, never recovered from after one.

The scope of that guarantee is worth stating exactly rather than rounding up, because a
one-time verdict about an arbitrary callable can only ever be conditional.

What is checked. The activation must have no parameters and no buffers, and every public
attribute it holds must be an immutable scalar, so its observable state can be captured by
value at admission. That snapshot is compared on every call, alongside the identity of the
object itself, so reassigning ``act_fn``, mutating a flag on it, adding an attribute to it, or
advancing a counter inside it all retire the verdict and route back to the baseline. Being
probed must also leave the activation unchanged, which is what refuses a callable that answers
GELU for exactly as many calls as the probe makes.

What is not checked, and cannot be. A finite probe cannot prove that a callable will keep
computing the same function -- only that it did at the points sampled, twice, at two shapes.
Two holes follow from that and are left open deliberately rather than hidden: a plain function
reading a module-level global has no state this can capture, and the fingerprint is taken on
the host, so an activation that agrees with GELU on the CPU and computes something else on
CUDA is not screened either. Both are inherent to fingerprinting behaviour, which is the
approach this operator is built around; neither is reachable by any host-side check.

Within that scope -- an activation whose behaviour is a function of its captured state and not
of the device -- a declined input is served by the baseline and an admitted one by an epilogue
indistinguishable from it at bfloat16, so the module is not incorrect, only sometimes not
faster.

Measured result
---------------
All five scored shapes pass, at speedups 1.084x (M = 1760), 1.438x (20680), 1.263x (23760),
1.479x (25168) and 1.500x (64680), for a
geometric mean of 1.3427x against a requirement of 1.0.

Four of the five agree with the baseline on every single element. At M = 64680 exactly one
element out of 74,511,360 falls outside atol + rtol*|ref| at 1e-2/1e-2, which is
0.99999998658 against a 0.99 requirement -- the residual described above, reproduced
identically on every run because the benchmark's weights are drawn deterministically. It is
not removable without giving up the fold; see that section.

Ratios are stable across runs but absolute latencies are not -- an instrumented re-run put
the M = 20680 baseline 9 % lower while reporting the same 1.37x -- so nothing here encodes
an absolute latency as a threshold.

Evidence: ``profile/phase1_probe/`` (probe scripts, their captured outputs, the harness runs
with and without route tracing, and ``report.md`` recording device, driver and library
versions, the shapes measured, and the reconciliation of every number quoted above).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# Reached by attribute access rather than an import statement, to keep the module's import
# surface to torch, torch.nn, torch.nn.functional, os and the two relative imports.
# torch.autograd imports this submodule itself, so a plain ``import torch`` is enough.
_forward_ad = torch.autograd.forward_ad

# ---------------------------------------------------------------------------------------
# Capability and admission constants, all resolved once at import.
# ---------------------------------------------------------------------------------------

def _resolve_fused_addmm():
    """The cuBLASLt bias+activation entry point, or None if it cannot be trusted.

    A private ATen symbol, so its presence is not enough: a build could expose a different
    signature, a different argument order, or a different activation. Existence, callability,
    the keyword this code passes, the returned type and shape, the activation itself, and the
    1-D bias broadcast are all checked once here, on four host floats. Anything unexpected
    returns None, which leaves the activation decision False and routes every call to the
    baseline expression -- a launch is never attempted to find out.
    """
    op = getattr(torch, "_addmm_activation", None)
    if op is None or not callable(op):
        return None
    try:
        a = torch.tensor([[1.0, 0.5], [-0.5, 2.0]], dtype=torch.float32)
        b = torch.tensor([[2.0, 3.0], [4.0, -1.0]], dtype=torch.float32)
        for bias in (torch.zeros(2, dtype=torch.float32),
                     torch.tensor([1.0, -1.0], dtype=torch.float32)):
            got = op(bias, a, b, use_gelu=True)
            if type(got) is not torch.Tensor or got.dtype != a.dtype:
                return None
            if tuple(got.shape) != (a.shape[0], b.shape[1]):
                return None
            # A 1-D bias must broadcast across rows, and the epilogue must actually be a
            # GELU rather than a bias-only or ReLU one. Compared loosely: this establishes
            # which operation it is, not how it rounds.
            want = F.gelu(torch.addmm(bias, a, b))
            if not torch.allclose(got, want, atol=1e-3, rtol=1e-3):
                return None
        # use_gelu=False must differ from use_gelu=True, or the keyword is being ignored.
        plain = op(torch.zeros(2, dtype=torch.float32), a, b, use_gelu=False)
        if torch.allclose(plain, op(torch.zeros(2, dtype=torch.float32), a, b, use_gelu=True)):
            return None
    except Exception:
        return None
    return op


_ADDMM_ACTIVATION = _resolve_fused_addmm()

# An allowlist, never a denylist: an unlisted dtype must fall back, and the reason bfloat16
# is the only entry is the ULP margin recorded in the module docstring.
_FUSED_DTYPES = (torch.bfloat16,)

_BF16_INF = torch.tensor(float("inf"), dtype=torch.bfloat16)

# Sampled once at import, never per call. See the tracing section at the bottom.
_TRACE_FALLBACK = bool(os.environ.get("VISION_MLP_TRACE_FALLBACK"))


# ---------------------------------------------------------------------------------------
# Is this activation GELU-shaped?
#
# The benchmark builds the *baseline* GELU class, not this tree's, so an isinstance test
# against a frozen candidate class would be False exactly on the path that matters, and a
# type-name test is a guess about naming rather than a statement about behaviour. So the
# activation is fingerprinted by what it computes.
# ---------------------------------------------------------------------------------------


# Written out rather than imported from math, to keep the import surface to torch, os and
# the two relative imports. Both are correctly rounded float64 literals.
_SQRT_2 = 1.4142135623730951           # sqrt(2)
_SQRT_2_OVER_PI = 0.7978845608028654   # sqrt(2 / pi), ATen's own tanh-GELU constant


def _one_bf16_ulp(t: torch.Tensor) -> torch.Tensor:
    """Distance to the next representable bfloat16 above each |t|."""
    a = t.abs().to(torch.bfloat16)
    return (torch.nextafter(a, _BF16_INF) - a).float()


def _gelu_exact(f: torch.Tensor) -> torch.Tensor:
    """Exact-erf GELU, evaluated in float32."""
    return 0.5 * f * (1.0 + torch.erf(f / _SQRT_2))


def _gelu_tanh(f: torch.Tensor) -> torch.Tensor:
    """tanh-approximation GELU, evaluated in float32."""
    return 0.5 * f * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (f + 0.044715 * f * f * f)))


def _flavour_disagreement(x: torch.Tensor) -> torch.Tensor:
    """How far apart the two GELU flavours are at each point, in float32.

    This is the admission width the whole substitution rests on. The epilogue computes one
    flavour while the caller asked for the other, and that is defensible precisely because
    the two differ by at most 4.7324e-04 -- far below one bfloat16 ULP wherever the value is
    large enough to matter. Measuring that distance here rather than assuming a constant
    makes the width self-calibrating: it is generous only where the two admitted flavours
    are themselves generous, and it collapses toward zero in the saturating tails, where
    both flavours agree to many digits and an impostor has nowhere to hide.
    """
    f = x.float()
    gap = (_gelu_exact(f) - _gelu_tanh(f)).abs()
    # +-inf and NaN inputs make the difference undefined; those points are judged by class
    # agreement instead, so contribute no width here.
    return torch.where(torch.isfinite(gap), gap, torch.zeros_like(gap))


def _probe_points() -> torch.Tensor:
    """Where to interrogate the activation.

    A mid-range sweep alone would say nothing about the tails, the specials, or the
    saturating region, so all three are included. The sweep is 513 points because bfloat16
    holds only about 256 distinct values per binade -- past that density the samples repeat
    and the verdict stops changing.

    The largest magnitude is 1e38 rather than something nearer the bfloat16 maximum of
    3.39e38 on purpose: torch's erf-GELU kernel forms an intermediate around twice the
    input, so a probe point above ~1.7e38 overflows that reference to +inf and would then
    refuse every activation, including exact GELU itself.
    """
    sweep = torch.linspace(-8.0, 8.0, 513, dtype=torch.float32)
    specials = torch.tensor(
        [
            0.0, -0.0,                    # signed zeros
            1e-40, -1e-40,                # subnormal in bfloat16
            1e-38, -1e-38,                # just above the smallest normal
            20.0, -20.0, 1e3, -1e3,       # saturating tails
            1e4, -1e4, 1e38, -1e38,       # large magnitudes
            float("inf"), float("-inf"), float("nan"),
        ],
        dtype=torch.float32,
    )
    return torch.cat([sweep, specials]).to(torch.bfloat16)


def _gelu_references(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Every rendering of GELU an admitted activation may legitimately match.

    Two flavours, each in two forms: as torch's own kernel computes it, and as an exact
    float32 evaluation rounded to the input dtype. Both forms are needed because the two
    disagree at the edges in ways that are properties of torch's kernels rather than of
    GELU -- the erf kernel flushes the far negative tail to -0 where the exact value is
    around -1e-7, and returns NaN for +inf under vector evaluation where the exact form
    returns +inf. An activation matching any one of the four is indistinguishable from GELU
    at the precision the epilogue delivers.
    """
    f = x.float()
    return (
        F.gelu(x, approximate="none"),
        F.gelu(x, approximate="tanh"),
        _gelu_exact(f).to(x.dtype),
        _gelu_tanh(f).to(x.dtype),
    )


def _agrees_with(result: torch.Tensor, reference: torch.Tensor,
                 points: torch.Tensor) -> bool:
    """Is *result* the same function as *reference* at the epilogue's precision?

    Non-finite results are compared by exact class and sign: NaN, +inf and -inf are three
    different answers and are never interchangeable. Finite results are compared at the
    scale of the *output*, widened by the distance between the two GELU flavours at that
    point. Scaling by the input magnitude instead would be unsound -- one bfloat16 ULP at
    x = -20 is 0.125 while GELU there is zero to 80-odd digits, which would admit an
    activation returning 0.125 in a saturated tail.
    """
    for is_class in (torch.isnan, torch.isposinf, torch.isneginf):
        if not torch.equal(is_class(reference), is_class(result)):
            return False
    finite = torch.isfinite(reference)
    # One step of the output's own representation -- computed with nextafter, so it is the
    # true ULP right down into the subnormal range rather than a constant floor. An earlier
    # version clamped this to the smallest *normal* bfloat16, which is 128 ULPs wide at zero
    # and let an activation return the smallest normal where GELU returns -0. The four
    # references below are what legitimately absorb torch's kernel edge behaviour, so no
    # floor is needed and none is applied.
    tolerance = torch.maximum(
        _one_bf16_ulp(reference),
        _flavour_disagreement(points),     # plus however far the two flavours already are
    )
    deviation = (result.float() - reference.float()).abs()
    # Inclusive: two functions agreeing far below one ULP still round to neighbouring
    # bfloat16 values, which reads as a full ULP of difference. At |x| = 2.75 the exact and
    # tanh flavours do precisely that.
    return bool((deviation[finite] <= tolerance[finite]).all())


# Attribute types a snapshot can capture by value. Anything else -- a tensor, a list, a
# nested module -- is mutable or opaque, so an activation holding one is refused rather than
# admitted on a verdict that could silently stop being true.
_IMMUTABLE_SCALARS = (bool, int, float, str, bytes, type(None))
_NO_CLOSURE = object()
_MISSING = object()


def _activation_state(act_fn):
    """An immutable snapshot of everything about *act_fn* that is observable from the host and
    could change what it computes, or None if it holds state this cannot capture.

    "No parameters and no buffers" is not the same as "immutable behaviour": a module with a
    plain ``self.use_relu`` flag, or one counting its own calls, changes what it computes
    without touching a parameter. So the snapshot covers the public attributes, the parameter
    and buffer counts, and the size of the instance dictionary -- the last of those is what
    notices an attribute *appearing* after admission.

    A Python closure is refused outright: its cells can be rebound after admission and there
    is no way to capture them by value. What this still cannot see is a function reading a
    module-level global; see the module docstring for that limit.
    """
    closure = getattr(act_fn, "__closure__", _NO_CLOSURE)
    if closure is not _NO_CLOSURE and closure is not None:
        return None
    own = getattr(act_fn, "__dict__", None)
    attrs = []
    if own is None:
        dict_len = -1          # a C-level callable: no Python attribute state to drift
    else:
        dict_len = len(own)
        for key in sorted(own):
            value = own[key]
            if type(value) in _IMMUTABLE_SCALARS:
                if key == "training":
                    # Deliberately not snapshotted. The benchmark calls .eval() on the module
                    # after construction, so capturing this flag would retire every verdict
                    # the moment it does. Admission instead *proves* the activation ignores
                    # the flag, by evaluating it in both modes and requiring identical bits,
                    # so there is nothing left for a snapshot here to protect.
                    continue
                # Every other scalar is tracked, including an underscore-prefixed one: a
                # module keeping its mode in `self._mode` is holding behaviour, not
                # bookkeeping, and skipping it by name would let that mode change silently.
                attrs.append((key, value))
                continue
            # Not a scalar. Acceptable only as one of nn.Module's own containers, and only
            # while empty -- which the parameter, buffer and module checks already imply. A
            # registered hook, or any populated container, is real state that cannot be
            # captured by value, so it is refused rather than ignored.
            if (key.startswith("_") and isinstance(value, (dict, set, list, tuple))
                    and len(value) == 0):
                continue
            return None
    if isinstance(act_fn, nn.Module):
        return (tuple(attrs), dict_len, len(act_fn._parameters), len(act_fn._buffers))
    return (tuple(attrs), dict_len, -1, -1)


def _activation_state_unchanged(act_fn, snapshot) -> bool:
    """Is *act_fn* still in the state its verdict was taken in?

    Re-reads only what was captured, so this costs one length check and one comparison per
    public scalar attribute -- one or two of them for any real activation.
    """
    attrs, dict_len, n_parameters, n_buffers = snapshot
    if dict_len >= 0:
        own = act_fn.__dict__
        if len(own) != dict_len:
            return False
        for key, value in attrs:
            current = own.get(key, _MISSING)
            if current is value:
                continue
            # Type first, then value. `current` is read live, so it need not still be a
            # scalar: comparing a reassigned multi-element tensor with `!=` would yield a
            # tensor and raise when this `if` forced it to bool, turning a decline into a
            # crash. _MISSING also lands here, which is how a deleted attribute is caught.
            if type(current) not in _IMMUTABLE_SCALARS or current != value:
                return False
    if n_parameters >= 0:
        if len(act_fn._parameters) != n_parameters or len(act_fn._buffers) != n_buffers:
            return False
    return True


def _admits_gelu_epilogue(act_fn):
    """Can the cuBLASLt GELU epilogue stand in for *act_fn*? Returns its state snapshot if so,
    and None otherwise.

    Decided once, by the caller, at construction. Any failure -- including an exception from
    a hostile or unusual callable -- declines the fast path rather than breaking
    construction, because refusing to fuse is always a correct answer.

    The verdict is about behaviour on the host, which is where it can be taken cheaply and
    before the module has been moved to a device. An activation whose result depends on the
    device it runs on is therefore outside what this establishes; see the module docstring.
    """
    if _ADDMM_ACTIVATION is None:
        return None
    try:
        # Parameters and buffers are refused outright: a later load_state_dict or dtype cast
        # could rewrite them and change what the activation computes.
        if isinstance(act_fn, nn.Module):
            if any(True for _ in act_fn.parameters()) or any(True for _ in act_fn.buffers()):
                return None
        if not callable(act_fn):
            return None
        # Everything else observable about the activation, captured before it is called.
        before = _activation_state(act_fn)
        if before is None:
            return None

        points = _probe_points()

        # An activation that answers differently in train and eval mode cannot be admitted on
        # one verdict, because the benchmark switches the module to eval after construction.
        # Establish that the flag does not matter, restoring whatever it was.
        if isinstance(act_fn, nn.Module):
            was_training = act_fn.training
            try:
                act_fn.train(True)
                in_train = act_fn(points)
                act_fn.train(False)
                in_eval = act_fn(points)
            finally:
                act_fn.train(was_training)
            if type(in_train) is not torch.Tensor or type(in_eval) is not torch.Tensor:
                return None
            if in_train.shape != in_eval.shape or in_train.dtype != in_eval.dtype:
                return None
            if not torch.equal(in_train.view(torch.int16), in_eval.view(torch.int16)):
                return None

        # Two shapes, so an activation that only matches GELU at one of them is refused;
        # each shape is judged against references computed at that same shape.
        for shaped in (points, points.reshape(-1, 1)):
            first = act_fn(shaped)
            if type(first) is not torch.Tensor:
                return None
            if first.shape != shaped.shape or first.dtype != shaped.dtype:
                return None
            references = _gelu_references(shaped)
            if not any(_agrees_with(first, ref, shaped) for ref in references):
                return None
            # Twice, so a randomized activation is refused. Compared on the raw bits, since
            # "same values" is not the question -- "same function" is.
            second = act_fn(shaped)
            if type(second) is not torch.Tensor or second.shape != first.shape:
                return None
            if not torch.equal(first.view(torch.int16), second.view(torch.int16)):
                return None
        # Being probed must not itself have changed the activation. This is what catches a
        # callable that answers GELU for exactly as many calls as the probe makes and
        # something else afterwards, when it counts those calls in its own state.
        after = _activation_state(act_fn)
        if after is None or after != before:
            return None
        return before
    except Exception:
        return None


def _carries_forward_tangent(*tensors: torch.Tensor) -> bool:
    """Is any of these carrying a forward-mode tangent?

    A dual tensor has exact type ``torch.Tensor`` and ``requires_grad=False``, and
    ``torch.no_grad()`` does not disable forward-mode AD, so nothing else in the predicate
    would stop one; the fused path would silently drop the derivative. Testing the active
    dual level first makes this a single integer comparison when nobody is doing forward
    AD, which is the case throughout the benchmark.
    """
    if getattr(_forward_ad, "_current_level", -1) < 0:
        return False
    return any(_forward_ad.unpack_dual(t).tangent is not None for t in tensors)


# ---------------------------------------------------------------------------------------
# The operator.
# ---------------------------------------------------------------------------------------


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu. Neither is GELU-shaped, so
    both run the baseline expression; the benchmarked configuration supplies exact GELU and
    takes the fused path.
    """

    # act_fn is deliberately unannotated: the baseline spells its type with
    # collections.abc.Callable, which is outside this module's permitted imports, and an
    # annotation naming an unimported type would raise under any caller that resolves
    # annotations. The contract is the parameter name and default, both of which are kept.
    def __init__(self, in_features: int, hidden_features: int,
                 act_fn=QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn
        # The declared width, not one read off a weight: the state dict is loaded after
        # construction, so nothing here may be derived from a parameter or a buffer.
        self._in_features = in_features
        # The verdict is taken once, because a stateless activation's behaviour cannot
        # change -- but which object `self.act_fn` refers to certainly can. Remember the
        # exact object the verdict was taken about, so that `module.act_fn = something_else`
        # falls back instead of running a GELU epilogue for whatever was assigned.
        # object.__setattr__ bypasses nn.Module's attribute machinery on purpose: a plain
        # assignment would register an nn.Module activation as a second child, and the point
        # of this reference is identity, not ownership.
        state = _admits_gelu_epilogue(act_fn)
        self._fuse_activation = state is not None
        object.__setattr__(self, "_admitted_act_fn", act_fn if state is not None else None)
        object.__setattr__(self, "_admitted_act_state", state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `is`, not ==: the verdict belongs to one object, and reassigning or wrapping
        # self.act_fn must retire it rather than inherit it. The state check then covers the
        # same object being changed in place -- a flag flipped, an attribute appearing, a
        # counter advancing -- which identity alone cannot see.
        if (self._fuse_activation
                and self.act_fn is self._admitted_act_fn
                and _activation_state_unchanged(self.act_fn, self._admitted_act_state)):
            fc1, fc2 = self.fc1, self.fc2
            w1, b1, w2, b2 = fc1.weight, fc1.bias, fc2.weight, fc2.bias
            if (
                # The epilogue needs a bias operand; both linears must supply one.
                b1 is not None
                and b2 is not None
                # fp8 wants the quantized linear path, which this does not reproduce.
                and not fc1.use_fp8
                and not fc2.use_fp8
                # Above one rank the row-parallel half owes an all-reduce and applies its
                # bias on rank 0 only, neither of which the two-call form does.
                and fc2.tp_size == 1
                # Not isinstance: a subclass implementing only __torch_function__ is
                # unwrapped by pybind on the way into an ATen call, so it has to be
                # screened here in Python or not at all.
                and type(x) is torch.Tensor
                and x.dtype in _FUSED_DTYPES
                and x.is_cuda
                and x.dim() >= 2
                and x.shape[-1] == self._in_features
                # Not x.contiguous(): materializing a copy would add a kernel inside the
                # measured region that costs more than the fold saves.
                and x.is_contiguous()
                and x.numel() > 0
                # addmm will not mix dtypes or devices, and a mismatch here means the
                # module was half-moved or half-cast; the baseline expression reports that
                # in its own words rather than ours.
                and w1.dtype == x.dtype and b1.dtype == x.dtype
                and w2.dtype == x.dtype and b2.dtype == x.dtype
                and w1.device == x.device and b1.device == x.device
                and w2.device == x.device and b2.device == x.device
                # Reverse-mode: keep autograd on the plain composition, whose activation is
                # the one the caller asked for. Never fires under the benchmark, which
                # wraps every forward in no_grad, but it is what keeps callers correct.
                and not (torch.is_grad_enabled()
                         and (x.requires_grad or w1.requires_grad or b1.requires_grad
                              or w2.requires_grad or b2.requires_grad))
                and not _carries_forward_tangent(x, w1, b1, w2, b2)
            ):
                return _fused_mlp(x, w1, b1, w2, b2)
        return self.fc2(self.act_fn(self.fc1(x)))


def _fused_mlp(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
               w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """fc1 + bias + GELU in one kernel, then fc2 + bias in a second.

    ``reshape`` rather than ``view`` for the flatten: a contiguous [M, 1, K] input needs no
    copy either way, and reshape cannot raise on a legal layout. The trailing ``view`` is
    safe because addmm returns a contiguous 2-D result whose element count matches.
    """
    flat = x.reshape(-1, x.shape[-1])
    hidden = _ADDMM_ACTIVATION(b1, flat, w1.t(), use_gelu=True)
    fused = torch.addmm(b2, hidden, w2.t())
    return fused.view(*x.shape[:-1], fused.shape[-1])


# ---------------------------------------------------------------------------------------
# Optional route tracing.
#
# Answers "did the measured workload actually take the fused route, or did it fall back and
# tie at 1.00x?" -- and costs nothing when the question is not being asked. Setting
# VISION_MLP_TRACE_FALLBACK swaps in an instrumented forward once, here at import; the
# method above is left untouched, so the normal hot path carries no counter, no flag read
# and no branch belonging to this section.
#
# The predicate is restated below rather than shared, so keep the two in step. Only the
# diagnostic can drift: both routes compute through the same _fused_mlp, so a divergence
# here cannot change a result.
#
# Timings from a traced run describe the instrumented forward, not the shipped one, and are
# not a performance measurement.
# ---------------------------------------------------------------------------------------

_FALLBACK_REASON_ORDER = (
    "activation-not-gelu",
    "activation-replaced",
    "activation-state-changed",
    "no-bias",
    "fp8-linear",
    "tensor-parallel",
    "tensor-subclass",
    "unsupported-dtype",
    "not-cuda",
    "bad-rank",
    "wrong-in-features",
    "non-contiguous",
    "empty",
    "operand-dtype-mismatch",
    "operand-device-mismatch",
    "autograd",
    "forward-mode-dual",
)
_fallback_seen: dict[str, int] = {}
_fastpath_calls = 0


def _fallback_reason(module: VisionMLP, x: torch.Tensor) -> str | None:
    """Why this call cannot take the fused route, or None if it can."""
    fc1, fc2 = module.fc1, module.fc2
    w1, b1, w2, b2 = fc1.weight, fc1.bias, fc2.weight, fc2.bias
    if not module._fuse_activation:
        return "activation-not-gelu"
    if module.act_fn is not module._admitted_act_fn:
        return "activation-replaced"
    if not _activation_state_unchanged(module.act_fn, module._admitted_act_state):
        return "activation-state-changed"
    if b1 is None or b2 is None:
        return "no-bias"
    if fc1.use_fp8 or fc2.use_fp8:
        return "fp8-linear"
    if fc2.tp_size != 1:
        return "tensor-parallel"
    if type(x) is not torch.Tensor:
        return "tensor-subclass"
    if x.dtype not in _FUSED_DTYPES:
        return "unsupported-dtype"
    if not x.is_cuda:
        return "not-cuda"
    if x.dim() < 2:
        return "bad-rank"
    if x.shape[-1] != module._in_features:
        return "wrong-in-features"
    if not x.is_contiguous():
        return "non-contiguous"
    if x.numel() == 0:
        return "empty"
    if any(t.dtype != x.dtype for t in (w1, b1, w2, b2)):
        return "operand-dtype-mismatch"
    if any(t.device != x.device for t in (w1, b1, w2, b2)):
        return "operand-device-mismatch"
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (x, w1, b1, w2, b2)
    ):
        return "autograd"
    if _carries_forward_tangent(x, w1, b1, w2, b2):
        return "forward-mode-dual"
    return None


def _forward_traced(self: VisionMLP, x: torch.Tensor) -> torch.Tensor:
    global _fastpath_calls
    reason = _fallback_reason(self, x)
    if reason is None:
        _fastpath_calls += 1
        if _fastpath_calls == 1:
            print("[vision_mlp] fused route taken", flush=True)
        return _fused_mlp(x, self.fc1.weight, self.fc1.bias,
                          self.fc2.weight, self.fc2.bias)
    count = _fallback_seen.get(reason, 0) + 1
    _fallback_seen[reason] = count
    if count == 1:
        # Once per reason, not once per call: the point is to notice a route at all.
        print(f"[vision_mlp] fallback taken: {reason}", flush=True)
    return self.fc2(self.act_fn(self.fc1(x)))


def route_counts() -> dict[str, int]:
    """Fused-call count and per-reason fallback tally.

    With tracing off this is ``{"fused": 0}`` and nothing else, because the instrumented
    forward that maintains these counters was never installed.

    Reasons are reported in _FALLBACK_REASON_ORDER rather than in the order they happened to
    be hit first, so two runs are directly comparable.
    """
    counts = {"fused": _fastpath_calls}
    for reason in _FALLBACK_REASON_ORDER:
        if reason in _fallback_seen:
            counts[reason] = _fallback_seen[reason]
    return counts


if _TRACE_FALLBACK:
    VisionMLP.forward = _forward_traced
