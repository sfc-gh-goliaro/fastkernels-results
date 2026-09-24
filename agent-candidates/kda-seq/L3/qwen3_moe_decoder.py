"""Qwen3 MoE decoder layer: the frozen attention and MoE, with every stage that
feeds the MoE driven by the reference implementation.

This layer is eight lines of glue over three operators that all have frozen
winners, so a candidate whose imports are relative inherits all three by
construction.  That free composition is fast -- 2.25x at ``M = 1000``, 2.11x at
``M = 16384`` -- and **numerically wrong at every ``M >= 128`` shape**
(``matched`` 0.195 and 0.257 against a bound that needs 0.99).  No frozen winner
is at fault; each passes in isolation.  The composition fails by *amplification*.

The MoE is an FP8 pipeline whose activation scales are UE8M0 -- powers of two.  A
perturbation the size of one bfloat16 ULP in its *input* moves a group's scale
across a binade boundary, which flips FP8 bytes by a whole step, which in a
1536-term dot product is about 2% relative error: past ``rtol = 1e-2`` on most
elements.  The frozen MoE's own docstring makes this argument for its second
quantizer; the same argument applies to its first, and its first is what reads
this layer's post-attention norm output.

So the rule for this layer is a cut, not a preference: **the tensor entering
``mlp`` and the returned ``residual`` must be bitwise identical to the
reference's.**  The MoE output is the only leaf that may differ, because it is
terminal -- nothing reconverges after it.  Below the cut, bitwise means the
reference's exact bytes, so the stages that are not already bit-identical are
driven by the reference class.

What is already bitwise, measured stage by stage with the reference's own
intermediate fed in so a difference is local (``tools/diag_stages.py``,
``tools/diag_attn.py``):

    qkv_proj (frozen fp8 GEMM)              max_abs 0     kept frozen
    attention core (direct prefill op)      max_abs 0     kept frozen
    o_proj (frozen fp8 GEMM)                max_abs 0     kept frozen

What is not, and is therefore pinned to the reference here:

    input_layernorm / post_attention_layernorm   max_abs 0.0156 at M = 16384
    q_norm / k_norm (128-wide, strided)          max_abs 0.0078 / 0.0002
    the rope stand-in inside the frozen attention  max_abs 0.0625

Nothing that is bitwise is replaced, and nothing that is not is kept.  The
ablation that fixes the pin list (``tools/diag_pinall.py``,
``profile/phase1_pinall.json``) says each pin is load-bearing:

    frozen everywhere        M=1 PASS 1.50x   M=1000 FAIL 0.195   M=16384 FAIL 0.257
    + reference rotary       M=1 PASS 1.24x   M=1000 PASS 0.9949  M=16384 FAIL 0.872
    + reference QK norm      M=1 PASS 1.27x   M=1000 PASS 1.0     M=16384 FAIL 0.923
    + reference 4096 norms   M=1 PASS 1.28x   M=1000 PASS 1.0     M=16384 PASS 1.0

The two 4096-wide norms are the last pin and they are what closes 0.923 to 1.0, so
they are not optional -- and they are free: adding them moves the layer by 0.4% at
``M = 16384`` and 0.3% at ``M = 1000``, both inside noise.  The ratio is quoted
rather than the two latencies because the ratio is what holds across runs while the
absolute times drift with the machine; ``tools/diag_pinall.py`` asserts it stays
under 2%.  The pins that cost are
the other two: about 78 us for the QK norms and about 162 us for the rotary, at
``M = 1000``.  Where the QK-norm figure goes is visible in the layer's own
timeline at that shape: 66.0 us in the two norm kernels and 30.0 us in the
``.contiguous()`` copies the reference forces before them
(``profile/phase1_pinned_kernels/``).  ``..L1.rms_norm`` is still imported below -- it is the frozen
winner for this shape and the relative import is how this workspace reaches it --
and it is deliberately not the class used for the four norms whose output feeds
the MoE.  A later phase that builds a bit-exact fast norm can put it back on
evidence rather than on hope; the numbers above are what it has to beat, and
``tools/diag_pinall.py`` is kept re-runnable so removing a pin costs an
experiment rather than an argument.

What this layer wins back
-------------------------
The reference rotary's cost is not the rotation.  Its ``forward`` opens with
``cache = self.cos_sin_cache; if cache.dtype != query.dtype: cache =
cache.to(query.dtype)``, and for this model the table is ``[1048576, 128]``
float32 = 512 MiB: converted to bfloat16 and thrown away on every call, about
130 us of copy kernel plus a 256 MiB allocation, independent of token count
(``profile/phase1_kernels_M1.txt``).

That conversion is a pure function of a weight, so it is memoized here and the
*same reference class* is handed a table that is already bfloat16 -- which makes
that line a no-op and leaves every downstream byte identical.  The conversion is
elementwise and the ``cache[positions]`` that follows is a selection, so
converting-then-gathering and gathering-then-converting produce the same bytes.
Measured (``tools/probe_rope_cache.py``, ``profile/phase1_rope_cache.json``):

    M=1     reference 0.186 ms   pre-converted 0.069 ms   q, k bitwise equal
    M=1000  reference 0.200 ms   pre-converted 0.074 ms   q, k bitwise equal

About 117-126 us at both shapes, consistent with a cost that depends on the table
and not on token count.  (One run; ``tools/probe_rope_cache.py`` re-measures and
asserts both the bitwise equality and that the saving is positive, so what is pinned
is the property rather than the third digit.)  What that is worth *in the layer* is a different
question, and the answer is not the one the standalone number suggests.  Timed as
whole layers on the five scored cases, the same weights and inputs, the only
difference being whether the pre-converted table is handed over
(``tools/measure_standin.py``, ``profile/phase1_standin_integrated.json``):

    M=1     residual=None   0.6217 -> 0.6143 ms     +7 us    1.47x -> 1.49x
    M=1     residual        0.6276 -> 0.6152 ms    +12 us    1.29x -> 1.32x
    M=870   residual=None   1.0276 -> 0.8878 ms   +140 us    1.70x -> 1.96x
    M=1000  residual        1.0445 -> 0.9023 ms   +142 us    1.72x -> 1.99x
    M=16384 residual        9.2257 -> 8.9811 ms   +245 us    1.62x -> 1.66x

So it pays at every scored case, and it pays *least* where the standalone number
said it would pay most.  At ``M = 1`` the host is the wall -- about 0.70 ms of
enqueue against about 0.68 ms of GPU time, only about 330 us of which is kernel
time -- so 117 us of removed GPU work surfaces as 7-12 us of removed latency.  The
three prefill cases are where it lands, and at ``M = 1000`` it is the difference
between 1.72x and 1.99x.

The stand-in, and why it is shaped this way
-------------------------------------------
``_standin_for`` admits a module on its **exact type** and never through
``isinstance``.  A subclass is free to override ``forward`` and rotate by some
other law, and nothing about the class says whether it did, so admitting
subclasses would mean silently applying the base rotation to an unknown number
of them.  Only the fields that shape the rotation are copied -- ``head_dim``,
``mrope_section``, ``mrope_interleaved`` / ``is_neox_style`` -- and they are
re-checked against the live module on every call, because the module reads its
own live and a caller that rewires them means the rotation they now describe.
``rope_theta`` is not retained and is not needed: it only ever shaped the table,
and the live table is transplanted rather than rebuilt.  Hence
``max_position_embeddings=1``, whose own table is a throwaway.

The stand-in is held off the module tree with ``object.__setattr__``.  Registered
as a submodule, ``Module._apply`` would visit it and replace its buffers
independently, so ``.to(device)`` would hand it a second full-size table instead of
a shared view -- 256 MiB allocated twice.  It would *not* add ``state_dict`` keys:
it owns no parameter and its only buffer is registered ``persistent=False``, so a
registered stand-in is invisible to ``state_dict`` and visible only to
``modules()``, which is what the schema check therefore asserts.  Off the tree it is
never visited at all, which is why the table is transplanted on the first forward
rather than in ``__init__``: at construction the source buffer is still on the host,
and a reference taken then would stay there.

It is delivered as the frozen attention's **per-call** ``rotary_emb``, an
already-supported path in that module, so ``self.self_attn.rotary_emb`` is left
exactly as it was handed over.  Passing a per-call rotary is also what bypasses
the frozen attention's own stand-in -- faster, but the source of the 0.0625 rope
divergence in the table above.

Fallback is a decision, not a recovery.  When no stand-in could be built, or the
live module has been rewired, or a guard refuses, this layer's ``forward`` passes
the rotary it was given and the frozen attention calls it.  Nothing wraps a
kernel launch in ``try``/``except``.  For that fallback to reach the reference
rotation at all, the frozen attention's own ``_rope_bridge`` is set to ``None``
on the instance in ``__init__``: that module substitutes its stand-in whenever
the rotary it resolves *is* ``self.rotary_emb``, which is precisely the fallback
case, so leaving it in place would have quietly restored the divergence the pin
list exists to remove.  Setting an instance attribute is the same category of
change as replacing ``q_norm`` and ``k_norm`` below -- no frozen file is edited.

Retention, and what the key cannot see
--------------------------------------
The memo is a ``WeakKeyDictionary`` keyed on the source rotary module, holding
**one** converted table per live rotary.  When a rotary becomes unreachable, its
converted table does too -- and that takes two mechanisms, not one, because the
memo is not the table's only holder:

* the memo entry disappears with the weak key, releasing the strong references it
  held to the float32 source and to the conversion;
* the stand-in is pointing at that same conversion, so the weak reference to the
  source carries a callback that clears the stand-in's buffer when the source dies
  (``_watch_source``).

The second is what makes the bound real rather than eventual.  Releasing the
stand-in's copy on the *next* call would be releasing it at a moment that may
never arrive: the bench worker runs all five cases in one process, reconstructing
baseline and candidate init arguments -- a fresh rotary each -- per case, and it
fills a layer, scores it, drops it and moves on without calling it again.

The arithmetic: one 256 MiB bfloat16 table per live rotary, against a 256 MiB
transient allocation *per call* in the reference.  In a real deployment one rotary
module is shared by every layer, so one entry serves the whole model.

The entry serves exactly one ``(device, dtype)`` pair -- the one it was built
for.  A request for another is refused and the call falls back to the source
module rather than converting a second 512 MiB table, and rather than reusing
the first: a float32 source converted to bfloat16 and then to some third dtype
is not the reference's direct conversion, and handing that back would be a
near-miss instead of an equality.  Falling back is never wrong and never slower
than the reference, because the reference is what it falls back to.  Re-keying the
single entry -- discarding the bfloat16 table and converting the float32 source
directly to the new dtype -- would be equally exact and equally bounded; it is not
taken because it converts 512 MiB on a dtype flip to buy what the fallback
already gives correctly, and it would surrender the hot entry to whichever dtype
asked last.

A refusal also releases the table the stand-in is still holding, so the bound
stays one converted table per live rotary rather than two in the window after an
invalidation.

Invalidation covers rebinding ``cos_sin_cache`` (identity), an in-place write to
the buffer itself (version counter), moving it between devices (``Module._apply``
installs a fresh tensor, so identity again), its storage address, and the target
dtype.  It is lazy: a superseded entry is dropped on the next lookup rather than
when the source changes, so between a device move and the next call the old
device's allocation is still live.

Two holes it cannot see.  A ``param.data.copy_()`` from outside a state-dict load:
``Tensor.data`` hands out a view with its own version counter, so such a write is
invisible by construction -- the frozen MoE documents the same hole for its own
weight-derived cache.  And a mutation of the source concurrent with this layer's
use of it, or consumption of the converted table from a stream this layer does not
order against: nothing here synchronizes streams, because the layer it composes
does not either.

Ordinary ``state_dict`` loading does not refresh this buffer, and not because the
memo misses it -- ``cos_sin_cache`` is registered ``persistent=False``, so it is
absent from ``state_dict`` and a load never touches it.  That is also why, unlike
the frozen MoE's weight-scale memo, there is no ``_load_from_state_dict`` hook
here: there would be nothing for it to invalidate.

The documented alternative, if retention ever proves unacceptable, is to convert
only the gathered rope rows instead of the whole table.  That is bit-exact by the
same commutation argument and needs no persistent memory, but it means
reimplementing the reference ``forward`` rather than handing the same class a
different table -- forfeiting the "same class, identical downstream bytes,
identical aliasing behaviour" argument -- and gathering bfloat16 from a
pre-converted table moves half the bytes of gathering float32 and then casting,
so it is also faster at large ``M``.

Known failure modes
-------------------
* If the frozen attention's fp8 fast-path predicate stops holding, the projection
  goes through its own ``forward`` instead of the frozen GEMM.  What that costs the
  cut depends on *why* the predicate failed, and the two cases are not the same.
  An engine pre-sizing the linear op's activation buffer, or an observation-only
  hook, changes where the output is written and how many Python frames it takes to
  get there, not the arithmetic -- the projection calls the same quantized op with
  the same weights and scales.  A hook or an instance-level ``forward`` that
  *modifies* the projection's output changes the numerics by the caller's own
  intent, and the tensor entering the MoE moves with it.  Neither is something this
  layer can prevent, and the second is not something it should.  Note also that
  "slower" is not universal: under compilation or graph capture that route can be
  the faster one.  What is claimed here is only the first case, and it is claimed
  from reading the frozen projection rather than from a measurement -- the
  bit-exactness gate deliberately does not exercise this route, because installing
  the hook that would trigger it is what changes it.
* A hook installed on the rotary module, or a compiled call on it, makes the
  stand-in ineligible for the same reason the fp8 predicate exists: calling the
  stand-in's ``forward`` would skip work the caller installed.  The layer then
  pays the per-call table conversion again.
* Under autocast the stand-in is refused outright.  The guard reads the activation
  dtype, the rotary consumes the query dtype, and autocast is the one mechanism
  that makes those differ; the cost of refusing is the per-call conversion, and
  the cost of not refusing would be a rotation that is close instead of equal.
* ``M = 1`` is host-bound.  The fully pinned composition spends about 0.70 ms of
  host time enqueueing roughly 25 launches against about 0.68 ms of GPU time, of
  which only about 330 us is kernel time -- the rest is bubbles, concentrated in
  the attention (about 0.405 ms) and the MoE (about 0.251 ms)
  (``profile/phase1_decode_stages_M1.json``).  That 330 us describes the pinned
  composition *before* the memoized table, and so includes the 130 us per-call
  conversion this layer no longer pays; the layer's own figure is 179 us of kernel
  time in one forward, from a timeline trace, against a 570 us latency measured in
  the same process -- so 391 us, 69%, of the decode latency is outside kernel
  execution, with a gap after essentially every launch
  (``tools/measure_decode_gap.py``, ``profile/phase1_decode_gap.json``).  Removing
  GPU time there pays only until the host becomes the wall, which is why the
  memoized table is worth 7-12 us at ``M = 1`` and 140-245 us at the prefill shapes.
  How that 391 us divides between launch cost, driver work and queue starvation is
  not measured.
* Calling each submodule's bound ``forward`` instead of ``__call__`` where no hook
  is installed was measured and is deliberately *not* done here.  Alternating nine
  timings per variant on the two decode cases puts it at +4.2 us and +1.8 us
  against per-variant spreads of 35 us and 10 us
  (``tools/measure_shortcut.py``, ``profile/phase1_shortcut.json``): inside the
  noise, and it would trade a real risk of skipping a caller's hook for it.

Deferred, in the order they are worth doing
-------------------------------------------
These are precisely what the pinning costs -- the work needed to recover the free
composition's speed *with* the reference's bits.

1. A bit-exact stride-aware per-head QK RMSNorm: about 78 us at ``M = 1000`` and
   about 29 us at ``M = 1``, plus 4 allocations and 2 contiguity copies per
   layer.  The vendored norm kernel is reproducible instruction for instruction,
   and at ``head_dim = 128`` only 16 of its 1024 threads carry data while the
   rest add zeros, so a copy taking a row stride is bitwise identical while
   reading the strided qkv slice where it lies.  Profiled, the kernel is
   latency-bound rather than bandwidth- or compute-bound -- 3.5% of peak read
   bandwidth, 3.7% of samples on ``math_pipe_throttle``, issuing on only 13.4%, and
   a single instruction address carrying 29.8% of the stalled samples, almost all of
   it waiting on a global load it has no other warp to hide behind.  What the
   deferred kernel removes is not coalescing -- the loads are perfectly coalesced,
   32.0 of 32 bytes used per sector -- but the two things a stride-aware bit-exact
   norm can avoid: the ``.contiguous()`` copies the reference forces (30.0 us at
   M = 1000 in the layer's own timeline, and a layout penalty of 1.7x for q there
   against 3.9x at M = 16384), and the second read of every row (768 bytes of loads
   for a 256-byte row today: a variance pass, a normalize pass and the weight, with
   a 0.51% L2 hit rate saying the reread is not cached).  A strided layernorm taking the
   same shape of argument is upstream in vLLM (``csrc/layernorm_kernels.cu``, the
   fused MLA QKV work), so the layout question has a settled answer to compare
   against; the bit-exactness requirement is this workspace's own.
2. A fused bit-exact QK-norm plus M-RoPE in one launch, reading the strided qkv
   slice and writing contiguous q and k: removes the cos/sin gather, the chunk,
   the two contiguity copies and 3-4 launches -- the cached rope still costs
   0.069 ms at ``M = 1`` end to end, plus host time.  Needs the reference Triton
   kernel's bfloat16 rounding reproduced exactly, each product rounded to
   bfloat16 before the subtract and the section-masked cosine assembled by adding
   masked zeros.  This is where upstream went too -- SGLang carries a
   ``fused_qknorm_rope`` kernel and wires it into its own Qwen3-MoE model -- which
   is a reason to treat item 1 as the correctness stepping stone it is (a strided
   norm to validate the fused kernel against) rather than as something to ship on
   its own.
3. Decode-shape CUDA graph capture.  ``M = 1`` issues 27 GPU operations carrying
   179 us of kernel time inside a 570 us latency, so a replay collapses the host
   side; the highest potential of the four, since ``M = 1`` is 2 of 5 scored cases,
   and the only one of them whose prize is quantified -- 391 us, 69%, of the decode
   latency is not kernel execution.  Risks to settle first: returned tensors must be copied out of the
   graph pool so consecutive calls do not alias (the baseline's do not), inputs
   must be copied into static buffers, and the frozen MoE's stream-capture check
   must land on the path it takes today.
4. Fusing ``post_attention_layernorm`` into the MoE's first activation
   quantizer: one ``[M, 4096]`` bfloat16 write plus read, about 10 us at
   ``M = 1000`` and about 67 us at ``M = 16384`` -- and less room than that
   suggests, since the expert GEMMs it would sit beside already run at 67.3% of
   peak DRAM throughput.  The router GEMM still needs the
   normed bfloat16 rows, so the norm output cannot disappear entirely, and the
   quantizer input must stay bit-exact -- which is the whole difficulty, since
   fusing the norm into the quantizer changes where the rounding happens and this
   layer's entire thesis is that a bfloat16 ULP upstream of that quantizer is 2%
   downstream.  Norm-plus-quant fusion has upstream precedent (vLLM's
   ``csrc/layernorm_quant_kernels.cu``, FlashInfer's CuTe-DSL fused
   RMSNorm-plus-FP4), none of it under a bitwise constraint.  Least value against
   most complexity.

Not a target: the attention core is 5.78 ms of the composition's 9.80 ms at
``M = 16384`` and is already the vendored CuTe FlashAttention-4 kernel, and the MoE
is frozen.  Its largest expert GEMM is 443 us of the 1155.6 us of kernel time the
layer issues at ``M = 1000``, memory-bound at 67.3% of peak DRAM throughput -- 4.18 TB/s of reads,
58.8% of samples waiting on global loads, tensor cores busy 61.9% of active cycles.
Two-thirds of the way to the bandwidth ceiling rather than at it, and what is left
is a question for DeepGEMM rather than for this composition.
"""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn

# Absolute reference imports.  The candidate finder intercepts only
# ``fastkernels.tasks.candidate.*``, so these always resolve to the reference and
# are never redirected.  Used where a stage has to be bit-identical, which is
# every stage upstream of the MoE that is not already so.  The frozen MoE sets
# the precedent, importing ``SiluAndMul`` from the reference path for exactly
# this reason.
from fastkernels.tasks.baseline.L1.mrope import (
    MRotaryEmbedding as ReferenceMRotaryEmbedding,
)
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm as ReferenceRMSNorm
from fastkernels.tasks.baseline.L1.rotary_emb import (
    RotaryEmbedding as ReferenceRotaryEmbedding,
)

# The frozen winner for this shape, reached by the relative import this workspace
# uses, and deliberately not the class used for the four norms above the cut.
# See the module docstring for the measurement that decides that, and for what a
# later phase has to beat to put it back.
from ..L1.rms_norm import RMSNorm  # noqa: F401
from ..L2.attention import LlamaAttention, _would_only_call_forward
from ..L2.qwen3_moe import Qwen3MoE

# Shapes the throwaway table the stand-in is constructed with and never reads.
_THROWAWAY_ROPE_THETA = 10000.0

# source rotary module -> (source_table, version, data_ptr, converted_table).
#
# Weakly keyed so a converted table becomes unreachable exactly when the rotary
# it belongs to does.  The value holds the source table strongly, which is what
# makes the identity check in ``_converted_table`` sound -- a memo that kept only
# an address would happily hand back a table for a destroyed object whose address
# had been recycled.  Nothing in the value refers back to the key, so there is no
# cycle to keep a dead rotary alive; a rotary caught in some *other* cycle needs a
# collector pass before its entry goes, which is why the retention test forces one.
_CONVERTED_TABLES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _standin_for(rotary: nn.Module | None) -> nn.Module | None:
    """A reference rotary module able to stand in for *rotary*, or ``None``.

    Exact type only, and only the fields that shape the rotation.  See the module
    docstring for why both of those are rules rather than conveniences.
    """
    if rotary is None:
        return None
    kind = type(rotary)
    if kind is ReferenceMRotaryEmbedding:
        return ReferenceMRotaryEmbedding(
            head_dim=rotary.head_dim,
            max_position_embeddings=1,
            rope_theta=_THROWAWAY_ROPE_THETA,
            mrope_section=rotary.mrope_section,
            mrope_interleaved=rotary.mrope_interleaved,
        ).eval()
    if kind is ReferenceRotaryEmbedding:
        return ReferenceRotaryEmbedding(
            head_dim=rotary.head_dim,
            max_position_embeddings=1,
            rope_theta=_THROWAWAY_ROPE_THETA,
            is_neox_style=rotary.is_neox_style,
        ).eval()
    return None


def _standin_still_matches(standin: nn.Module, rotary: nn.Module) -> bool:
    """Whether *standin* still describes the same rotation as *rotary*.

    Compared rather than re-derived, so nothing is allocated on the hot path.
    ``mrope_section`` is compared by identity because the stand-in was handed the
    very same list -- an in-place edit is already shared with it, since the kernel
    reads the three section sizes per launch -- so only a rebind needs catching.
    """
    if standin.head_dim != rotary.head_dim:
        return False
    if type(standin) is ReferenceMRotaryEmbedding:
        return (standin.mrope_section is rotary.mrope_section
                and standin.mrope_interleaved == rotary.mrope_interleaved)
    return standin.is_neox_style == rotary.is_neox_style


def _converted_table(rotary, dtype, device):
    """*rotary*'s cos/sin table in *dtype* on *device*, converted once, or
    ``None`` when this call must not use a memoized one.

    ``None`` is returned when the live table is not on *device* -- converting
    would not move it and the reference does not move it either -- and when an
    entry already exists for a different ``(device, dtype)``.  Both send the
    caller to the source module, which is correct by construction.

    Any path that does not return the memoized table drops the entry first.
    Invalidation here is lazy, so an entry that has been superseded -- by a device
    move, or by a source that no longer needs converting -- would otherwise keep
    both its stale source and its stale conversion alive for as long as the rotary
    lives, on a device the layer has left.
    """
    source = rotary.cos_sin_cache
    entry = _CONVERTED_TABLES.get(rotary)
    if (entry is not None and entry[0] is source
            and entry[1] == source._version
            and entry[2] == source.data_ptr()):
        converted = entry[3]
        if converted.dtype is dtype and converted.device == device:
            return converted
        return None
    _CONVERTED_TABLES.pop(rotary, None)
    if source.device != device:
        return None
    if source.dtype is dtype:
        # What the reference would use unconverted: nothing to memoize, and no
        # entry to spend -- so a later request for a dtype that *does* need
        # converting still gets one.
        return source
    converted = source.to(dtype)
    _CONVERTED_TABLES[rotary] = (
        source, source._version, source.data_ptr(), converted)
    return converted


def _release(standin):
    """Drop the table a stand-in is holding.

    Reached on a refusal, which is cold, and from the source rotary's own death.
    Without it the stand-in would keep its last converted table alive next to
    whatever the memo now holds, so the bound would be two tables per live rotary
    in the window after an invalidation instead of one.  A stand-in that is not
    about to be used is not called, so the buffer being empty is never read.
    """
    if standin is not None and standin.cos_sin_cache is not None:
        standin.cos_sin_cache = None


def _watch_source(rotary, standin):
    """A weak reference to *rotary* that releases *standin*'s table when it dies.

    Retention has to be scoped to the source rotary and not to the layer holding
    the stand-in.  Weak-keying the memo gets half of that -- the entry disappears
    when the rotary is collected -- but the stand-in points at the *same* tensor,
    and releasing it on the next refusal is releasing it at a moment that may never
    arrive: the bench worker fills a layer, scores it, drops it, and moves to the
    next case without calling it again.  So the release is driven by the death
    itself rather than by the next call.

    The callback holds a weak reference to the stand-in and nothing else.  A strong
    one would keep the stand-in -- and through it the very table this exists to
    free -- alive for as long as the callback lives, which is as long as the layer.
    """
    if rotary is None:
        return None
    if standin is None:
        return weakref.ref(rotary)
    boxed = weakref.ref(standin)

    def release_on_death(_dead_ref, boxed=boxed):
        held = boxed()
        if held is not None:
            _release(held)

    return weakref.ref(rotary, release_on_death)


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        # Same class the baseline builds, same state_dict keys, same shapes --
        # replaced rather than reconstructed around, so the rest of the frozen
        # attention (the fp8 projection routing and the direct prefill call, both
        # bitwise) is untouched.
        self.self_attn.q_norm = ReferenceRMSNorm(
            config.head_dim, eps=config.rms_norm_eps)
        self.self_attn.k_norm = ReferenceRMSNorm(
            config.head_dim, eps=config.rms_norm_eps)
        # The frozen attention substitutes its own rope stand-in whenever the
        # rotary it resolves is the one it holds -- which is every call this layer
        # falls back on.  That stand-in is not bit-exact, so the fallback would
        # not have reached the reference rotation without this.
        object.__setattr__(self.self_attn, "_rope_bridge", None)

        self.mlp = Qwen3MoE(config, quant_config=quant_config)
        self.input_layernorm = ReferenceRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ReferenceRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        # Both off the module tree: the stand-in so ``Module._apply`` never gives
        # it a second full-size table and it adds no state_dict keys, the source
        # so a later rebind of ``self_attn.rotary_emb`` is caught by identity
        # rather than by hoping the copied fields still describe it.
        #
        # The source is held by *weak* reference, and that reference carries the
        # callback that frees the stand-in's table when the source dies.  A strong
        # reference would cost no memory while the attention still holds the same
        # module -- but it would outlive a rebind of ``self_attn.rotary_emb``, and a
        # rotary kept alive here is a rotary whose memo entry cannot be collected,
        # which is the accumulation the retention bound exists to prevent.  A dead
        # referent reads back as ``None``, which no live module compares equal to,
        # so the guard refuses and the call falls back.
        standin = _standin_for(rotary_emb)
        object.__setattr__(self, "_rope_standin", standin)
        object.__setattr__(self, "_rope_source", _watch_source(rotary_emb, standin))

    def _rope_for(self, hidden_states):
        """The stand-in, or the rotary this layer was given.

        Every test here is a comparison against live state, because everything it
        reads can change after ``__init__``: a caller rewires the rotation
        fields, an engine compiles the module, a profiler installs a hook.  A
        decision made once at construction would be exactly the wrong one.

        The target dtype is read off the activations, and what the rotary will
        actually see is the *query* dtype -- produced inside the frozen attention
        by the projection, one stage later.  For every path this layer can take
        those are the same dtype, because the projection carries its activation
        dtype through.  Autocast is what breaks that, and it is refused rather
        than assumed away: under autocast the projection emits the autocast dtype
        while the activations are still in theirs, and a table converted for the
        wrong one would be a near-miss rather than an equality.
        """
        standin = self._rope_standin
        source = self.self_attn.rotary_emb
        held = self._rope_source
        if (standin is None
                # A rotary rebound to ``None`` means the caller wants no rotation,
                # and the baseline simply skips it.  Tested before the identity
                # comparison because a dead weak referent also reads back as
                # ``None``: without this, both sides of that comparison would be
                # ``None``, it would pass, and the field re-check would then look
                # for ``head_dim`` on nothing.
                or source is None
                or held is None or held() is not source
                or not _standin_still_matches(standin, source)
                or not _would_only_call_forward(source)
                or torch.is_autocast_enabled(hidden_states.device.type)):
            _release(standin)
            return source
        table = _converted_table(source, hidden_states.dtype, hidden_states.device)
        if table is None:
            _release(standin)
            return source
        if standin.cos_sin_cache is not table:
            standin.cos_sin_cache = table
        return standin

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions, hidden_states, rotary_emb=self._rope_for(hidden_states))
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
