"""Qwen3-Next decoder layer: the frozen winners composed, then replayed.

Three things live in this file, each behind its own switch so a regression bisects
to one change:

**The composition.**  ``__init__`` mirrors the baseline's explicitly rather than
subclassing it, so which submodule comes from where is visible in the imports.  Every
submodule is the frozen lower-level winner, with two deliberate exceptions -- the two
``GemmaRMSNorm``s and the GDN attention -- both for the same reason and both measured.

The reason, once, because everything below follows from it: this layer feeds a top-10-of-512
softmax router whose gate logits are bf16, so two experts within one bf16 ulp are an *exact
tie* broken by implementation detail, and the two experts have unrelated weights.  A difference
far inside the bf16 comparison bound can therefore flip which expert wins, and a token whose
expert set flipped has all 2048 of its output elements outside that bound.  Nothing upstream of
the router may differ from the reference at all -- not "differ by little", but not differ.

**The GDN attention** runs the frozen winner's own *root* forward, through an L3 subclass, rather
than its fused path.  The fused path is not bitwise equal to the reference -- 0.47% of attention
output elements differ at 16384 tokens, which flips two expert sets -- and bisecting the frozen
winner's own increment flags finds no combination that reaches zero flips by construction except
the root.  This costs real speed; ``docs/notes/gdn_exactness.md`` has the table and the trade.

**The norms.**  The frozen L1 norm's own docstring records that
``residual is not None`` takes its pure-PyTorch path because "the residual variant is
not exercised by any captured shape" -- true at L1, and false here, where
``post_attention_layernorm`` takes that branch on every call and ``input_layernorm``
takes it on four of the five scored shapes.  Taking it costs 610 us against 59 us for
the reference's ``torch.compile``d kernel at 16384 tokens, which is a 1.1 ms
regression over two norms; and its fp32 reduction order differs from Inductor's by one
ulp on ~1.0% of elements, which is fatal for a different reason.  So the residual
branch here delegates to the reference module's *own* ``_forward_static_with_residual``
function object wrapped once in ``torch.compile``.  Dynamo keeps its guarded cache
entries and its automatic-shape ``FrameState`` on the wrapped function's code object,
not on the wrapper, and a lookup matches when the guards and the backend match -- both
wrappers pass the default ``inductor`` backend with no options, so the second wrapper
accepts the entry the first one compiled.  A retyped copy would have its own code
object and therefore its own promotion history, in lockstep with the reference's only by
coincidence of call order; sharing the function object makes the lockstep structural.
Because the candidate never compiles a second graph, the Inductor FX-graph cache and
Triton's autotuner are not consulted twice either -- which matters, since that cache is
content-addressed on the graph and the compiler configuration rather than on the code
object, so "same code object" alone would not have implied one generated kernel.

Why one ulp matters: this layer feeds a top-10-of-512 softmax router whose gate is a
``ReplicatedLinear`` over bf16 weights, so its logits are bf16.  Two experts whose
true logits differ by less than one bf16 ulp are an *exact tie* there, broken by
implementation detail, and the two experts have unrelated weights -- so a one-ulp
perturbation of the norm output changes the output by O(1), not by O(ulp).  A token
whose expert set flipped has all 2048 of its output elements outside the bf16
comparison bound, and the measured flip rate is the measured failure rate.  Bitwise
agreement of the norm outputs is therefore a correctness requirement here.

The no-residual branch goes to the reference's compiled helper too.  The frozen winner's
register-resident CUDA kernel is bitwise equal to the reference at one token -- the only
scored shape that reaches that branch -- but a 12-seed sweep across token counts finds it
drifts by one ulp on 1e-5 to 4e-5 of elements from 26 tokens up, because a warp-per-row
reduction and Inductor's looped reduction associate the fp32 sum differently once a row
spans more than one accumulator.  ``Qwen3NextModel``'s layer 0 takes that branch at every
token count, so the kernel is safe to score and unsafe to ship; it stays reachable behind
``Q3ND_NORM_KERNEL=1`` with the measurement recorded.

``forward`` keeps the reference's structure and drops what is dead at ``tp = 1``: the
``fused_ar_norm`` indirection is skipped when there is no collective to defer, and the
``layer_type`` string compare is resolved to a boolean once in ``__init__``.
``fuse_ar_norm`` remains an attribute and ``fused_ar_norm`` remains reachable, because
this class must still be a drop-in for ``Qwen3NextModel`` under tensor parallelism.

**The host trim, measured and not enabled** (``Q3ND_HOST_TRIM`` defaults off).  ``nn.Module.__getattr__`` is a Python call plus two dict lookups
per submodule access, and this forward makes four of them.  The trim caches the four
callables in a plain tuple in ``__dict__`` (not as registered submodules, which would
duplicate ``state_dict`` keys) and drops the cache from ``__setattr__`` when any of the
four names -- or ``layer_type`` or ``fuse_ar_norm`` -- is reassigned or deleted, holding the
instance lock across the mutation itself so a concurrent forward cannot rebuild against the old
value in between.  ``load_state_dict`` copies into parameters in place and never rebinds a
submodule, so weight reloading cannot stale it either.

**The graph replay.**  On four of the five scored shapes this layer is host-issue
bound at about 4x: the CPU spends ~1.7 ms issuing work the GPU finishes in ~0.4 ms,
most of it inside the trtllm-gen fused-MoE Python wrapper, which rebuilds a runner and
a tuning config on every call.  Replaying a captured graph of the whole layer removes
that issue cost.  It does not make a single kernel faster: it enqueues the very same kernel
launches the eager chain would, in the same order, from a recording instead of from Python.
Read the speedup that way.

Be precise about *why* replay is exact, because the obvious argument is no longer available.
It is not "these are the reference's own kernels": the composition keeps the frozen
``RMSNormGated`` inside the GDN submodule, which is a different -- and faster -- kernel than the
reference's (105.3 us against 132.2 us at 16384 tokens; see the census).  The claim that does
hold has two parts.  Replay reproduces **this candidate's eager composition**, which is
separately shown bitwise equal to the reference stage by stage and per case by
``tools/correctness.py --section router``.  And replay is checked against that eager chain
after every capture, mandatorily, on the same buffers and the same metadata object that will be
replayed -- so "the recording still matches what eager would do" is verified rather than
assumed.  Exactness with respect to the *reference* comes from the composition; exactness of
the *replay* comes from that check.

Replay is gated so that the module stays a drop-in and never merely a fast path.
Inputs are copied into static buffers.  Outputs are cloned by default -- without the
clone the ``residual is None`` case returns the static *input* buffer, which the next
call overwrites.  A validity key covering the caller's plan is recomputed and compared
before every replay, with a strong reference held to everything it compares by identity
or by pointer, so a stale plan rebuilds rather than replaying against freed or recycled
memory.  At most one graph is cached.  The graph is bound to the stream that built it
and to that stream's device.  And every condition the predicate rejects falls through to
the eager chain with a recorded reason rather than failing:

  - a capture already in progress on the current stream -- which is what happens when the
    serving engine captures this layer from the outside, and where a nested capture would
    be illegal;
  - an active ``torch.compile`` trace, grad mode enabled, a CPU or non-contiguous input;
  - a decode token in the batch, no prefill, or a token count above the threshold;
  - a non-zero initial recurrent state, because the pre-capture warmup has to really
    execute and would consume state the previous iteration overwrote;
  - a full-attention layer, always: no scored shape is one, so there is no measurement to
    justify capturing it, and its plan carries state this key does not cover;
  - ``tp > 1``, always: the fused collective's non-fused fallback is a custom IPC
    all-reduce with its own capture-registration protocol that the engine enters around
    its own capture and this layer cannot;
  - a stream other than the one the graph was built on, since the static buffers are one
    set per instance;
  - weights that moved since capture.

And after every capture the eager chain runs once more on the same static buffers and the same
metadata object that will be replayed, and the two are compared bitwise; a mismatch drops the
graph and serves eagerly.  That check is on by default rather than behind a test flag, because
"replay is bit-exact by construction" is a claim about somebody else's state staying put -- the
trtllm-gen MoE's autotuner picks its GEMM tactic per process and accumulates profiles as it sees
shapes, so a graph captured early can hold a tactic the eager path has since stopped choosing.
Capture itself is wrapped: a failure is recorded in ``LAST_CAPTURE_ERROR`` and served eagerly,
never half-captured.  One thing the fallback does *not* fix: a first-ever call
made from inside somebody else's capture region would do its first-time allocation and
compilation in that region's pool.  The reference has the same requirement, and the
engine satisfies it by warming up before it captures -- but "falls back safely" should be
read as "falls back to the eager chain", not as "is safe to call cold under capture".

``LAST_PATH`` and ``LAST_REJECTION`` record what served the most recent call, so "the
fast path never ran" is distinguishable from "the fast path ran and tied".

Numbers, B200, three correctness rounds, against the reference in the same process.
``validate.py`` on the shipped defaults reports 5/5 ``PASSED``, none skipped, with
**``max_abs_error = 0.00e+00`` and ``matched_ratio = 1.0`` on every case** -- bitwise equal to the
reference, not merely inside tolerance -- at **geomean 3.19x and 3.19x** on two back-to-back runs
(7.6x / 1.00x / 2.0x / 2.8x / 7.6x).  A seed sweep over 12 draws per case gives worst matched
exactly 1.0 everywhere, with the reference exactly deterministic against itself.

Read the geomean as a floor rather than a figure: it is host-load dependent, and the spread between
leases is structural rather than noise.  Replay costs ~200 us of almost pure host work, so a busier
CPU inflates it proportionally while barely moving a 1.7 ms reference that has ten times more host
work to hide the jitter in; the same configuration read 3.6-4.7x on quieter leases before the
exactness work.  The device-bound case is stable across leases at 1.00-1.01x, because clock drift
scales both sides equally.

The composition alone -- no graph, no trim -- is **1.027x** on the same harness, with every case
bitwise.  It was 1.149x before the GDN submodule was reduced to its exact root forward, and that
0.12x is the price of exactness, stated in ``docs/notes/gdn_exactness.md`` rather than buried.
``docs/notes/nodes_table.md`` carries the generated per-node table.

The one case the graph does not serve, ``[16384, 2048]``, is the only device-bound one and is off
by threshold: 41% of its device time is the MoE, 36% the GDN recurrence, 21% the projections,
and 2.27% the two norms -- and the norms already run at ~50% of peak DRAM throughput with
minimal read traffic, so even a perfect norm kernel is worth about 1% there.  See
``profile/decoder_16384_round1/REPORT.md``.
"""

from __future__ import annotations

import contextlib
import os
import threading
import weakref

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.tp import _tp_size
from ....tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm as _ReferenceNorm
from ..L1.gemma_rms_norm import GemmaRMSNorm as _FrozenNorm
from ..L2.flashinfer_allreduce_fusion import fused_allreduce_add_gemma_rmsnorm
from ..L2.qwen3_next_attention import Qwen3NextAttention
from ..L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention
from ..L2.shared_expert_moe import SharedExpertMoE

__all__ = ["Qwen3NextDecoderLayer", "DecoderGemmaRMSNorm", "fused_ar_norm"]


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


# --- Increment selection -------------------------------------------------------
# One switch per change, so every recorded node is reproducible from this file by
# naming its configuration.  Defaults are the configuration chosen on measurement;
# ``docs/notes/increments.md`` carries the per-node, per-shape latencies.
#
# ``Q3ND_COMPOSITION_ONLY=1`` is the correct root: the frozen winners composed with
# the reference-exact residual norm, and nothing else.  It is the fallback the other
# two increments rest on and the configuration to bisect to first.
_COMPOSITION_ONLY = _flag("Q3ND_COMPOSITION_ONLY", "0")

#: Cache the four submodule callables instead of re-resolving them per call.  **Off by
#: default: measured and dropped.**  The reasoning was sound -- four
#: ``nn.Module.__getattr__`` calls per forward, each a Python call plus two dict lookups --
#: and the estimate was 20-40 us of ~1500.  Measured in one process with the switch flipped
#: between five alternating timing loops per case, it is +1.6 us summed over the five scored
#: shapes, per-case -10.1 to +11.4 us against a run-to-run spread of 0.1 to 3.7%.  That is
#: zero.  Kept selectable so the measurement is reproducible; not enabled, because
#: plausibility is not evidence.
_HOST_TRIM_DEFAULT = "0"
_HOST_TRIM = _flag("Q3ND_HOST_TRIM", _HOST_TRIM_DEFAULT) and not _COMPOSITION_ONLY

#: Keyed whole-layer capture and replay.
_GRAPH = _flag("Q3ND_GRAPH", "1") and not _COMPOSITION_ONLY

#: Clone the replayed outputs before returning them.  On by default: the
#: ``residual is None`` case returns the static *input* buffer as the residual, so an
#: un-cloned return aliases a buffer the next call overwrites -- and the harness never holds
#: outputs across calls, so it cannot catch that.  Measured cost of the clone: 0.2% of geomean at
#: the shipped threshold (3.235x un-cloned against 3.229x, same lease); it has read as high as 8% on
#: other leases and thresholds, which makes it the least stable number in the record.  The argument
#: for cloning never depended on the cost, and does not now.
_GRAPH_CLONE = _flag("Q3ND_GRAPH_CLONE", "1")

#: After each capture, run the eager chain on the same static buffers and the same metadata
#: object that will be replayed, and compare bitwise; on a mismatch, drop the graph and serve
#: eagerly.  **On by default**, and it belongs in the shipped path rather than in a test flag:
#: replay is bit-exact *by construction* only as long as the captured kernels are the ones the
#: eager chain would run now, and that is an assumption about somebody else's state, not a
#: property of this file.  Concretely, the trtllm-gen MoE's autotuner picks its GEMM tactic per
#: process and accumulates profiles as it sees new shapes, so a graph captured early can hold a
#: tactic the eager path has since stopped choosing.  The cost is one extra eager run per
#: rebuild -- about four rebuilds per scored case, all of which land inside the harness's
#: correctness rounds or its untimed warmup, so none of it is in the timed window.
_GRAPH_VERIFY = _flag("Q3ND_GRAPH_VERIFY", "1")

#: Token count above which replay is not attempted.  2048 is the last count at which replay
#: measurably wins, measured **candidate-eager against candidate-replay in one process** -- which
#: is the only comparison that answers the question the threshold asks, "is replaying cheaper than
#: running my own eager chain".  Round 0 timed the *reference* as the eager arm, which folds the
#: composition's own speedup into the crossover, and that mistake produced both a wrong threshold
#: (4096) and a wrong conclusion (that replay never actually loses above it).  With the right
#: comparator: 2.00x at 301 tokens, 1.10x at 2048, then 0.98x at 3072 and 0.98-1.00x from there up,
#: while the host-to-replay ratio falls through 1.0 between 2048 and 3072.  So replay does become a
#: small loss, not merely a wash, once the host cost it removes stops dominating device time.
#: The scored mix cannot pick this: no scored shape lies strictly between 301 and 16384, so every
#: threshold in that interval makes identical decisions there.  ``docs/notes/threshold.md``.
_GRAPH_MAX_TOKENS = int(os.environ.get("Q3ND_GRAPH_MAX_TOKENS", "2048"))

#: Warmup iterations on the side stream before capture.  Three, matching the two L4
#: whole-model wrappers in this repository, and load-bearing rather than ritual: any
#: buffer first allocated inside the capture would come from the graph's private pool
#: and then be reused from eager code.
_GRAPH_WARMUP_ITERS = int(os.environ.get("Q3ND_GRAPH_WARMUP_ITERS", "3"))

#: Reduce the frozen GDN winner to its own root forward -- the reference's chain, reached
#: through the subclass the harness's prep hook requires -- instead of its fused path.
#: **On by default, and it costs real speed**, so here is the measurement that forces it.
#: The frozen winner's fused path is not bitwise equal to the reference: at 16384 tokens its
#: attention output differs on 0.47% of elements, which is far inside the bf16 bound but lands
#: on the wrong side of two near-ties in the top-10-of-512 router, and each flipped token puts
#: all 2048 of its output elements outside tolerance (``matched_ratio`` 0.99982, and 0.9979 at
#: 301 tokens on some seeds).  Bisecting its own increment flags shows this is not attributable
#: to one of them -- declining the fused prologue leaves 0.41% and two flips, declining the
#: fused gate leaves three flips, and no combination reaches zero except the root:
#:
#:     GDN configuration        worst matched   flips/16384   attn not bitwise
#:     frozen default             0.9998166        2            0.004690
#:     GDN_FUSE_PROLOGUE=0        0.9999306        2            0.004139
#:     GDN_FUSE_GATE=0            0.9998555        3            0.004981
#:     GDN_BF16_STATE=0           0.9998038        2            0.004943
#:     GDN_G_EXP=0                0.9999245        0            0.001044
#:     GDN_ZERO_STATE=0           0.9998556        2            0.004971
#:     prologue+gate off          0.9998860        0            0.004073
#:     root forward             1.0000000          0            0.000000
#:
#: Only the last row is zero flips *by construction* rather than by luck of the draw -- the two
#: rows above it reach zero on these seeds while still differing on 0.1-0.4% of elements, which
#: is the same "it happens to pass" the router makes unsafe.  So the exact root is the default
#: and the fused path is a recorded, faster, inexact node behind ``Q3ND_GDN_EXACT=0``.
#: ``docs/notes/gdn_exactness.md``.
_GDN_EXACT = _flag("Q3ND_GDN_EXACT", "1")

#: Rebind the MoE subtree's internals to the frozen L2/L1 winners.  ``SharedExpertMoE`` has no
#: candidate file, so the relative import resolves through the candidate finder's alias loader
#: to the *baseline module object*, whose own imports were resolved inside the baseline package
#: -- which is why the frozen ``trtllm_bf16_moe`` and ``silu_and_mul`` winners are invisible to
#: this layer even though both are on ``FASTKERNELS_CANDIDATE_DIR``.  Rebinding them from here is
#: the same mechanism the frozen GDN winner uses for its gated norm and needs no new L2 file.
#:
#: **Off by default, and the reason is a distinction worth keeping straight.**  It is worth about
#: 6% on the one scored shape where device time rather than host issue is the constraint (1.06x
#: against 1.00x at ``[16384, 2048]``), it flips no expert sets, and every scored case still reports
#: ``matched_ratio = 1.0``.  But it is *not bitwise*: ``max_abs_error`` at that shape goes
#: from ``0.00e+00`` to ``9.77e-04``.  That is harmless in a way the GDN divergence was not --
#: the MoE runs *downstream* of the router, so its differences cannot flip an expert set and are
#: never amplified from O(ulp) to O(1) -- but this operator's whole correctness argument is
#: bitwise agreement with the reference, and trading that for 6% of one case is the wrong
#: direction after a round spent establishing it.  Recorded as the ``shipped-moe-frozen`` node
#: with its numbers.  ``docs/notes/moe_opportunity.md``.
_MOE_FROZEN = _flag("Q3ND_MOE_FROZEN", "0")

#: Use the frozen L1 CUDA kernel on the no-residual branch instead of the reference's
#: compiled helper.  **Off by default**, which is not the obvious choice and is worth the
#: paragraph: the frozen kernel is bitwise equal to the reference at one token, which is
#: the only scored shape that reaches this branch, but not above it -- a 12-seed sweep
#: measures 12/12 exact at N=1, 6/12 at N=60, 2/12 at N=301 and 0/12 from N=512 up, on
#: 1e-5 to 4e-5 of elements.  Its warp-per-row reduction and Inductor's looped reduction
#: simply associate the fp32 sum differently once a row spans more than one accumulator.
#: Scoring the kernel would be safe; shipping it would not, because
#: ``Qwen3NextModel``'s layer 0 takes the no-residual branch at *every* token count, and
#: a 1-ulp perturbation there flips MoE expert sets.  The reference's own compiled helper
#: is 12/12 exact at every token count tested, so it serves both branches.  See
#: ``docs/notes/norm_numerics.md``.
_NORM_KERNEL = _flag("Q3ND_NORM_KERNEL", "0")


# --- Fault injection, for the negative tests -----------------------------------
# Each breaks exactly one thing the implementation depends on, so
# ``tools/correctness.py`` can show the corresponding positive check has teeth: a test
# that mutates nothing proves nothing.  All default off.
_MUT_NORM_EAGER = _flag("Q3ND_MUT_NORM_EAGER", "0")        # frozen eager residual path
_MUT_NORM_UNROUNDED = _flag("Q3ND_MUT_NORM_UNROUNDED", "0")  # skip the bf16 sum rounding
_MUT_KEY_NO_METADATA = _flag("Q3ND_MUT_KEY_NO_METADATA", "0")  # drop md identity from key
_MUT_KEY_NO_POINTERS = _flag("Q3ND_MUT_KEY_NO_POINTERS", "0")  # drop md/state data_ptrs
# The naive key: token count and dtype only.  This is the version the ``tools/ab.py``
# prototype effectively had, and setting it reproduces the failure the real key exists to
# prevent -- the harness republishes its plan, the tensors the graph baked in are freed,
# and the replay reads them anyway.  The two flags above are finer-grained and, on their
# own, each masked by what the other still covers; that redundancy is deliberate, and it
# is why demonstrating the hazard needs this flag rather than either of them.
_MUT_KEY_SHAPE_ONLY = _flag("Q3ND_MUT_KEY_SHAPE_ONLY", "0")
_MUT_WEAK_HOLD = _flag("Q3ND_MUT_WEAK_HOLD", "0")          # hold keyed objects weakly
_MUT_NO_WARMUP = _flag("Q3ND_MUT_NO_WARMUP", "0")          # capture with no warmup
_MUT_MULTI_GRAPH = _flag("Q3ND_MUT_MULTI_GRAPH", "0")      # keep every graph ever built
_MUT_CAPTURE_FAIL = _flag("Q3ND_MUT_CAPTURE_FAIL", "0")    # make capture raise
_MUT_NO_WEIGHT_CHECK = _flag("Q3ND_MUT_NO_WEIGHT_CHECK", "0")  # skip the weight snapshot
_MUT_VERIFY_MISMATCH = _flag("Q3ND_MUT_VERIFY_MISMATCH", "0")  # corrupt one replayed output
# Restore the exact pre-fix behaviour: take the lock *only if a graph is currently cached*, drop it,
# release the lock, and only then perform the assignment.  That conditional is what made it unsafe --
# ``_graph_build`` clears ``_g_graph`` before capturing, so a concurrent mutation mid-rebuild saw no
# graph, skipped the lock entirely, and could replace a submodule during capture.  Exists so the race
# test can show it has teeth: a concurrency test that passes against both the broken and the fixed
# code proves nothing.
_MUT_UNSAFE_MUTATION = _flag("Q3ND_MUT_UNSAFE_MUTATION", "0")
# Skip the completion event, so a drop can free buffers the device is still using.
_MUT_NO_DONE_EVENT = _flag("Q3ND_MUT_NO_DONE_EVENT", "0")
_MUT_RENAME_ATTN = _flag("Q3ND_MUT_RENAME_ATTN", "0")      # register attn under a new name
_MUT_WRAP_GDN = _flag("Q3ND_MUT_WRAP_GDN", "0")            # break the prep hook's isinstance


#: What served the most recent call.
#:   layer:             "replay" | "eager"
#:   norm_residual:     "compiled" | "traced" | "eager" | "unrounded" | "none"
#:   norm_no_residual:  "kernel" | "compiled" | "traced" | "none"
LAST_PATH: dict[str, str] = {
    "layer": "none",
    "norm_residual": "none",
    "norm_no_residual": "none",
}
#: Which condition sent the last call to the eager chain, or ``None`` when replay
#: served it.  ``None`` also when the graph increment is off -- there is no rejection
#: to report then, the path simply does not exist.
LAST_REJECTION: str | None = None
#: ``"ok"``, ``"mismatch"`` or ``None``: the result of the post-capture bitwise
#: self-check, when ``Q3ND_GRAPH_VERIFY`` is on.
LAST_VERIFY: str | None = None
#: Why the most recent capture attempt failed, if one did.  Kept rather than swallowed:
#: a candidate that silently serves eagerly scores about 1.00x, which is exactly what a
#: working candidate that ties scores.
LAST_CAPTURE_ERROR: str | None = None


# The reference's two static helpers, wrapped once each.  These are the reference
# module's own function objects, not copies, so the compiled artifact is shared with the
# reference's own lazily-created wrappers through Dynamo's code-object cache.  The
# arguments are deliberately identical to the reference's bare ``torch.compile(fn)``:
# a different backend, mode, options or ``dynamic`` setting would fail the cache entry's
# backend-equality guard and compile a second artifact, which is the one thing this
# delegation exists to prevent.
#
# Wrapping here rather than lazily does *not* compile anything -- ``torch.compile``
# builds a wrapper and traces nothing until the first call.  What keeps compilation out
# of the timed window is the harness's three correctness rounds, which run before it.
_REFERENCE_ADD_NORM = _ReferenceNorm._forward_static_with_residual
_REFERENCE_NORM = _ReferenceNorm._forward_static_no_residual
_ADD_NORM = torch.compile(_REFERENCE_ADD_NORM)
_NORM = torch.compile(_REFERENCE_NORM)

# Fail closed rather than silently: if either wrapper stops pointing at the reference's
# own function -- a reload under a different module name, a monkey-patch, a refactor that
# replaces it with a local copy -- the bitwise argument above is void, and a wrong answer
# here is a flipped MoE expert set rather than a rounding difference.
for _wrapper, _original, _which in (
    (_ADD_NORM, _REFERENCE_ADD_NORM, "_forward_static_with_residual"),
    (_NORM, _REFERENCE_NORM, "_forward_static_no_residual"),
):
    if getattr(_wrapper, "_torchdynamo_orig_callable", None) is not _original:
        raise RuntimeError(
            f"the compiled norm no longer wraps the reference's own GemmaRMSNorm.{_which}; "
            "bitwise agreement is not established and the MoE router is sensitive at one ulp",
        )

#: Dynamo's recompilation limits, recorded at import.  Read but deliberately not
#: written: an operator has no business mutating global compiler configuration.  Why it
#: is recorded at all -- at the limit, default non-``fullgraph`` Dynamo does not evict
#: and recompile, it marks the frame run-only and executes the eager Python from then
#: on, which on this path is the 1-ulp drift that flips the router.  This code object
#: accumulates two entries under the scored shapes (a static one-token variant and a
#: dynamic one), so the installed defaults leave a wide margin; anything that lowers
#: these below two turns a bitwise guarantee into a silent numerical change.
try:  # pragma: no cover - diagnostic only
    import torch._dynamo.config as _dynamo_config

    DYNAMO_CACHE_LIMITS = (
        getattr(_dynamo_config, "cache_size_limit", None),
        getattr(_dynamo_config, "accumulated_cache_size_limit", None),
    )
except Exception:  # noqa: BLE001
    DYNAMO_CACHE_LIMITS = (None, None)


def _add_norm_unrounded(
    weight: torch.Tensor,
    variance_epsilon: float,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The wrong fp32 association, for the negative test only.

    The reference rounds the residual sum to bf16 and computes the variance *from the
    rounded sum*; this normalises the unrounded fp32 sum instead.  It differs from the
    reference on ~24% of elements, which is what makes it a useful control on the
    bitwise assertion.
    """
    orig_dtype = x.dtype
    s = x.float() + residual.float()
    variance = s.pow(2).mean(dim=-1, keepdim=True)
    out = s * torch.rsqrt(variance + variance_epsilon)
    out = out * (1.0 + weight.float())
    return out.to(orig_dtype), s.to(orig_dtype)


class DecoderGemmaRMSNorm(_FrozenNorm):
    """The frozen norm winner, with the residual branch restored to the reference's.

    Nothing derived from ``weight`` is cached: the harness calls ``load_state_dict``
    after construction, and a precomputed ``(1 + weight)`` would silently go stale.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight.data
        eps = self.variance_epsilon
        compiling = torch.compiler.is_compiling()

        if residual is None:
            if compiling:
                # Inside an outer trace, hand the tracer the same eager math the
                # reference hands it; a nested ``torch.compile`` call would only be
                # inlined back to this.
                LAST_PATH["norm_no_residual"] = "traced"
                return _ReferenceNorm._forward_static_no_residual(weight, eps, x)
            if _NORM_KERNEL and self._kernel_eligible(x, None):
                LAST_PATH["norm_no_residual"] = "kernel"
                return _FrozenNorm.forward_cuda(self, x, None)
            # Not the frozen module's own eager fallback: that copy is bitwise equal
            # to the reference's *source*, not to what Inductor generates from it.
            LAST_PATH["norm_no_residual"] = "compiled"
            return _NORM(weight, eps, x)

        if _MUT_NORM_EAGER:
            LAST_PATH["norm_residual"] = "eager"
            return _FrozenNorm.forward_native(self, x, residual)
        if _MUT_NORM_UNROUNDED:
            LAST_PATH["norm_residual"] = "unrounded"
            return _add_norm_unrounded(weight, eps, x, residual)
        if compiling:
            LAST_PATH["norm_residual"] = "traced"
            return _ReferenceNorm._forward_static_with_residual(weight, eps, x, residual)
        LAST_PATH["norm_residual"] = "compiled"
        return _ADD_NORM(weight, eps, x, residual)


def _rebind_moe_to_frozen_winners(mlp) -> list[str]:
    """Point the MoE's internals at the frozen winners it cannot see for itself.

    Returns what was rebound, for the record.  Both replacements are parameter-free modules, so
    ``state_dict`` keys are unchanged; ``process_weights_after_loading`` stays on
    ``SharedExpertMoE`` and is untouched, which matters because it is what produces the
    BlockMajorK expert weights the trtllm path requires.
    """
    done: list[str] = []
    existing = getattr(mlp, "trtllm_moe", None)
    if existing is not None:
        try:
            from ..L2.trtllm_bf16_moe import TrtLlmBf16MoE as FrozenTrtLlmBf16MoE
        except ImportError:  # pragma: no cover - frozen winner absent
            FrozenTrtLlmBf16MoE = None
        if (FrozenTrtLlmBf16MoE is not None
                and type(existing) is not FrozenTrtLlmBf16MoE):
            # Built from the existing instance's own fields rather than from the config, so
            # the routing method, expert sharding and scaling cannot drift apart.
            mlp.trtllm_moe = FrozenTrtLlmBf16MoE(
                num_experts=existing.num_experts,
                top_k=existing.top_k,
                intermediate_size_per_partition=existing.intermediate_size_per_partition,
                routing_method_type=existing.routing_method_type,
                local_expert_offset=existing.local_expert_offset,
                local_num_experts=existing.local_num_experts,
                num_expert_group=existing.num_expert_group,
                topk_group=existing.topk_group,
                routed_scaling_factor=getattr(existing, "routed_scaling_factor", None),
            )
            done.append("trtllm_moe")

    shared = getattr(mlp, getattr(mlp, "shared_expert_attr_name", ""), None)
    act = getattr(shared, "act_fn", None)
    if act is not None:
        try:
            from ..L1.silu_and_mul import SiluAndMul as FrozenSiluAndMul
        except ImportError:  # pragma: no cover - frozen winner absent
            FrozenSiluAndMul = None
        if FrozenSiluAndMul is not None and type(act) is not FrozenSiluAndMul:
            shared.act_fn = FrozenSiluAndMul()
            done.append("shared_expert.act_fn")
    return done


def fused_ar_norm(norm, hidden_states, residual, fuse: bool):
    """``norm(all_reduce(hidden_states), residual)``, fused when ``fuse``.

    Kept at the reference's name and signature so a caller that imports it from the
    decoder module still finds it.  The layer itself only calls it when ``fuse`` is
    true; at ``tp = 1`` there is no collective to defer and the indirection is pure
    host cost.
    """
    if fuse:
        # Opaque under torch.compile: tracing into FlashInfer's fused collective hits
        # Python logging / datetime and aborts Dynamo.
        if torch.compiler.is_compiling():
            return torch.ops.fastkernels.fused_allreduce_add_gemma_rmsnorm(
                hidden_states, residual, norm.weight, float(norm.variance_epsilon),
            )
        return fused_allreduce_add_gemma_rmsnorm(hidden_states, residual, norm)
    return norm(hidden_states, residual)


def _tensor_sig(t):
    """A replay-relevant signature for one keyed tensor, or ``None``.

    Pointer alone is not enough.  Two different views can share an address while
    differing in shape or stride, which changes the host launch parameters the capture
    baked in without changing anything a pointer comparison would see.  Device is in
    here because two GPUs can hand out the same address.
    """
    if not isinstance(t, torch.Tensor):
        return None
    return (t.data_ptr(), t.device, t.dtype, t.shape, t.stride(),
            t.storage_offset())


#: Metadata fields the recurrent forward reads as device tensors.  A superset of what
#: the frozen GDN winner touches, deliberately: the key is cheap and being wrong here
#: means replaying against freed memory.
_MD_TENSORS = (
    "query_start_loc",
    "query_start_loc_int32",
    "state_indices",
    "state_indices_long",
    "non_spec_query_start_loc",
    "non_spec_state_indices_tensor",
    "has_initial_state",
    "seq_lens",
    "batch_ptr",
    "token_chunk_offset_ptr",
)

#: ``compute_causal_conv1d_metadata`` builds a chunk plan for exactly one token tile,
#: and the conv kernel reads the plan out of that tile's entry rather than off the
#: metadata.  Fixed by the producer, not a tuning knob.
_CONV_BLOCK_T = 8

#: Tensors inside the per-tile chunk-plan entry.  Keyed separately from the metadata's
#: own copies: the two are published independently, and the kernel reads the entry's.
_CONV_PLAN_TENSORS = (
    "batch_ptr",
    "token_chunk_offset_ptr",
    "nums",
    "mlist",
    "offsetlist",
)

#: Attributes whose reassignment or deletion must drop the cached call chain and any
#: captured graph: the four submodules the chain holds, and the two scalars that decide
#: which chain it is.
_INVALIDATING_NAMES = frozenset({
    "input_layernorm", "post_attention_layernorm", "linear_attn", "self_attn", "mlp",
    "layer_type", "fuse_ar_norm",
})

_MISS = object()


def _drop_graph_after_load(module, incompatible_keys):  # noqa: ARG001
    # ``load_state_dict`` has already written the parameters by the time a post-hook runs, so
    # there is no assignment left to bracket -- but the drop still has to wait for any in-flight
    # replay before releasing what it was reading.
    with module._g_lock:
        module._graph_drop()


class ExactQwen3NextGDNAttention(Qwen3NextGDNAttention):
    """The frozen GDN winner reduced to the reference's own forward.

    A subclass rather than an edit or an env flag: the frozen file stays frozen, the harness's
    prep hook still finds this through ``isinstance(sub, baseline.Qwen3NextGDNAttention)``
    because the frozen winner subclasses that class, and the choice is per instance rather than
    per process.

    Everything the frozen winner does in ``__init__`` is kept, including its rebinding of
    ``self.norm`` to the frozen ``RMSNormGated`` -- which the measurement says is bitwise equal
    to the reference on this path, since the root configuration comes out at exactly zero
    differing elements on every stage.  Only ``forward_impl`` is redirected.
    """

    def forward_impl(self, hidden_states, state_manager=None):
        # Skip the frozen winner's own ``forward_impl`` and call the reference's.
        return super(Qwen3NextGDNAttention, self).forward_impl(
            hidden_states, state_manager,
        )


class _Wrapped(nn.Module):
    """Fault injection: a wrapper that hides the GDN submodule's class.

    The harness's module-prep hook finds the recurrent attention with
    ``isinstance(sub, baseline.Qwen3NextGDNAttention)`` and raises ``_UnsupportedInput``
    when it cannot -- which marks the case ``SKIPPED`` rather than failed.  This exists
    so a test can demonstrate that hole instead of asserting against it blindly.
    """

    def __init__(self, inner):
        super().__init__()
        self._inner = inner

    def forward(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        # Before anything else that ``__setattr__`` might have to invalidate.  One instance owns
        # one graph, one set of static buffers and one output tuple, so every read *or write* of
        # that state is one critical section -- including the attribute assignments that
        # invalidate it, which have to be inside the same section as the invalidation or a
        # forward can rebuild against the old value in between.  Re-entrant because
        # ``_graph_build`` runs inside it and calls ``_graph_drop``.  A lock, not a thread: the
        # harness fails a run if a background thread appears during timing.
        object.__setattr__(self, "_g_lock", threading.RLock())
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Only worth deferring when there is a collective to defer.
        self.fuse_ar_norm = _tp_size() > 1

        if self.layer_type == "linear_attention":
            gdn_cls = ExactQwen3NextGDNAttention if _GDN_EXACT else Qwen3NextGDNAttention
            attn = gdn_cls(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                layer_idx=layer_idx,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
            if _MUT_WRAP_GDN:
                attn = _Wrapped(attn)
            if _MUT_RENAME_ATTN:
                self.attn = attn
            else:
                self.linear_attn = attn
        elif self.layer_type == "full_attention":
            attn = Qwen3NextAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
            if _MUT_RENAME_ATTN:
                self.attn = attn
            else:
                self.self_attn = attn
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")

        # MoE for all Qwen3-Next layers (every layer is sparse).
        self.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            routing="softmax",
            correction_bias=False,
            renormalize=config.norm_topk_prob,
            routed_scaling_factor=1.0,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            shared_expert_attr_name="shared_expert",
            shared_expert_gate=True,
            reduce_results=not self.fuse_ar_norm,
        )

        if _MOE_FROZEN:
            _rebind_moe_to_frozen_winners(self.mlp)

        self.input_layernorm = DecoderGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = DecoderGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

        # Resolved once rather than compared per call.  ``layer_type`` stays as the
        # attribute a caller may read.
        self._is_linear = self.layer_type == "linear_attention"

        # Graph state.  Plain ``__dict__`` entries: a tensor or a tuple assigned to a
        # fresh attribute name is not intercepted by ``nn.Module.__setattr__``, so none
        # of this reaches ``state_dict`` or ``named_modules``.
        self._g_key: tuple | None = None
        self._g_graph = None
        self._g_static: tuple | None = None      # (hidden_states, residual)
        self._g_out: tuple | None = None
        self._g_hold: tuple | None = None        # strong refs to everything keyed
        self._g_weights: tuple | None = None    # the snapshot the graph was captured against
        self._g_stream: int | None = None        # the stream the graph was captured on
        self._g_device: torch.device | None = None
        self._g_retired: list = []               # only ever non-empty under fault injection
        #: A completion event recorded on the owner device after each replay's output clone.
        #: ``_graph_drop`` waits on it before releasing anything the replay touched.  The lock
        #: alone is not enough for this: it is released as soon as the work is *enqueued*, and
        #: freeing a static buffer or an output buffer while the device is still reading it is a
        #: use-after-free that no amount of Python mutual exclusion prevents.
        self._g_done = None

        # A captured graph holds the weight *storages* it was captured against.  The
        # per-call pointer check below catches storage rebinds on the parameter objects
        # the graph holds (``.data =``, ``Module._apply``, an in-place quantization
        # pass) but cannot see ``load_state_dict(assign=True)``, which puts an entirely
        # new tensor into ``_parameters`` and leaves the held object untouched.  This
        # hook closes that without adding anything to the hot path.
        self.register_load_state_dict_post_hook(_drop_graph_after_load)

    # -- Submodule access -----------------------------------------------------

    def __setattr__(self, name, value):
        # The invalidation and the assignment are one critical section, not two.  Invalidating
        # first and assigning after -- which is what this did before -- leaves a window in which
        # a forward on another thread rebuilds a graph against the *old* submodule and then the
        # assignment lands, so the cached graph belongs to a chain that no longer exists.  And
        # taking the lock only when a graph happens to exist is not enough either: during a
        # rebuild ``_graph_build`` clears ``_g_graph`` first, so a concurrent invalidation would
        # see none, skip the lock, and replace a submodule mid-capture.
        if name in _INVALIDATING_NAMES:
            with self._graph_mutation():
                super().__setattr__(name, value)
                if name == "layer_type":
                    # ``_is_linear`` is this value resolved once.  The reference re-reads
                    # ``layer_type`` on every call, so a caller may reasonably expect changing
                    # it to take effect.
                    self.__dict__["_is_linear"] = value == "linear_attention"
            return
        super().__setattr__(name, value)

    def __delattr__(self, name):
        # ``del layer.mlp`` does not go through ``__setattr__``, and a cached chain would go on
        # calling the deleted submodule.
        if name in _INVALIDATING_NAMES:
            with self._graph_mutation():
                super().__delattr__(name)
            return
        super().__delattr__(name)

    @contextlib.contextmanager
    def _graph_mutation(self):
        """Hold the instance lock across a change that invalidates the cached graph.

        Unconditionally, whether or not a graph is currently cached, because "no graph right now"
        is also what the middle of a rebuild looks like.
        """
        lock = self.__dict__.get("_g_lock")
        if lock is None:                      # during ``__init__``, before the lock exists
            self.__dict__.pop("_chain_cache", None)
            yield
            return
        if _MUT_UNSAFE_MUTATION:
            # Fault injection: the conditional lock the pre-fix code used, then mutate outside it.
            self.__dict__.pop("_chain_cache", None)
            if self.__dict__.get("_g_graph") is not None:
                with lock:
                    self._graph_drop()
            yield
            return
        with lock:
            self.__dict__.pop("_chain_cache", None)
            self._graph_drop()
            yield

    def __getstate__(self):
        # A captured ``CUDAGraph``, the tensors it holds alive and a ``threading.Lock``
        # are all live runtime state that must not be copied or pickled.  Drop them; the
        # clone rebuilds on its first call.
        state = dict(self.__dict__)
        for key in ("_g_key", "_g_graph", "_g_static", "_g_out", "_g_hold",
                    "_g_weights", "_g_stream", "_g_device", "_g_done",
                    "_chain_cache", "_g_lock"):
            state.pop(key, None)
        state["_g_retired"] = []
        return state

    def __setstate__(self, state):
        # ``nn.Module.__setstate__`` fills in back-compat defaults for attributes older
        # checkpoints lack; keep it and add this module's own live-state slots.
        super().__setstate__(state)
        for key in ("_g_key", "_g_graph", "_g_static", "_g_out", "_g_hold",
                    "_g_weights", "_g_stream", "_g_device", "_g_done"):
            self.__dict__.setdefault(key, None)
        self.__dict__.setdefault("_g_retired", [])
        self.__dict__.setdefault("_g_lock", threading.RLock())

    def drop_graph(self) -> None:
        """Discard any captured graph.

        Public because one hole cannot be closed from inside: a caller that replaces a
        *nested* parameter object directly -- ``layer.mlp.gate.weight = nn.Parameter(...)``
        rather than through ``load_state_dict`` or ``.to()`` -- changes weights without
        touching anything this module observes.  Call this after doing that.

        Takes the instance lock and waits for any in-flight replay, so it is safe to call from
        another thread while a forward is running.
        """
        with self._g_lock:
            self._graph_drop()

    def _apply(self, *args, **kwargs):
        # ``.to()`` / ``.cuda()`` / ``.float()`` all land here and may either rebind or replace
        # parameter storages.  Dropping unconditionally is cheaper than reasoning about which,
        # and the drop and the move are one critical section for the same reason the attribute
        # hooks are.
        with self._graph_mutation():
            return super()._apply(*args, **kwargs)

    def _attn_module(self):
        if _MUT_RENAME_ATTN:
            return self.attn
        return self.linear_attn if self._is_linear else self.self_attn

    def _chain(self):
        """``(input_layernorm, attn, post_attention_layernorm, mlp)``, cached.

        Submodules do not live in ``__dict__``, so each ``self.<name>`` is an
        ``nn.Module.__getattr__`` call.  Four of them per forward is ~20-40 us of the
        ~1500 us this layer costs at the scored token counts.  The cache is dropped from
        ``__setattr__`` and ``__delattr__`` when any of those four names -- or
        ``layer_type`` or ``fuse_ar_norm``, which decide *which* chain this is -- changes.
        ``load_state_dict`` copies into parameters in place and never rebinds a submodule,
        so a weight reload cannot stale it; that is asserted by reloading weights between
        two forwards.
        """
        chain = self.__dict__.get("_chain_cache")
        if chain is None:
            chain = (
                self.input_layernorm,
                self._attn_module(),
                self.post_attention_layernorm,
                self.mlp,
            )
            self.__dict__["_chain_cache"] = chain
        return chain

    # -- The eager chain ------------------------------------------------------

    def _eager_forward(self, hidden_states, residual, positions=None,
                       rotary_emb=None, state_manager=None):
        if _HOST_TRIM:
            in_norm, attn, post_norm, mlp = self._chain()
        else:
            in_norm = self.input_layernorm
            attn = self._attn_module()
            post_norm = self.post_attention_layernorm
            mlp = self.mlp
        fuse = self.fuse_ar_norm

        if residual is None:
            # Layer 0: the input is the vocab-parallel embedding's output, which is
            # already reduced, and there is no residual stream yet.
            residual = hidden_states
            hidden_states = in_norm(hidden_states)
        elif fuse:
            hidden_states, residual = fused_ar_norm(
                in_norm, hidden_states, residual, True,
            )
        else:
            hidden_states, residual = in_norm(hidden_states, residual)

        if self._is_linear:
            hidden_states = attn(hidden_states, state_manager=state_manager)
        else:
            hidden_states = attn(
                hidden_states, rotary_emb=rotary_emb, positions=positions,
                state_manager=state_manager,
            )

        if fuse:
            hidden_states, residual = fused_ar_norm(
                post_norm, hidden_states, residual, True,
            )
        else:
            hidden_states, residual = post_norm(hidden_states, residual)
        hidden_states = mlp(hidden_states)

        return hidden_states, residual

    # -- The replay path ------------------------------------------------------

    def _graph_rejection(self, hidden_states, residual, positions, rotary_emb,
                         state_manager):
        """The first condition that disqualifies replay, or ``None``.

        Returned as a string rather than a bool so ``LAST_REJECTION`` can say which one
        it was; each is covered independently by ``tools/correctness.py``.
        """
        if not hidden_states.is_cuda:
            return "not_cuda", None, None
        if torch.compiler.is_compiling():
            # A ``torch.compile`` trace is active: capturing here would bake the
            # tracer's own tensors in, and the graph is not what the tracer wants
            # anyway.
            return "compile_trace_active", None, None
        if torch.cuda.is_current_stream_capturing():
            # In production this layer runs *inside* the engine's own capture region
            # (``ModelRunner.capture_kimi_cudagraph`` captures the whole decode step),
            # and a nested capture is illegal.
            return "stream_capturing", None, None
        if torch.is_grad_enabled():
            # Capture and replay record no autograd graph.  Rather than silently drop
            # gradients, decline; the harness and the serving engine both call under
            # ``no_grad``.
            return "grad_enabled", None, None
        if not self._is_linear:
            # The full-attention path is never captured.  Two reasons, and neither is
            # about difficulty: no scored shape is a full-attention layer, so there is no
            # measurement to justify it; and its plan carries state this key does not
            # cover (``slot_mapping``, the block tables, ``max_query_len`` /
            # ``max_seq_len``, and the rotary cache, which the frozen attention assigns
            # onto itself inside ``forward`` and which therefore is not in
            # ``self.parameters()`` when the graph is built).  A key that does not cover
            # what the capture baked in is worse than no fast path.
            return "full_attention_not_captured", None, None
        if self.fuse_ar_norm:
            # At ``tp > 1`` the two norms route through a fused collective whose non-fused
            # fallback is a custom IPC all-reduce, and that all-reduce has its own
            # capture-registration protocol -- the engine enters it around
            # ``ModelRunner``'s capture and this layer cannot.  Capturing the collective
            # without registering its graph buffers is a silent cross-rank corruption, and
            # rank-by-rank rebuild has no coordination either.  Replay is for ``tp = 1``.
            return "tensor_parallel_graph_disabled", None, None
        if not hidden_states.is_contiguous():
            return "input_not_contiguous", None, None
        if residual is not None and (
            residual.shape != hidden_states.shape
            or residual.dtype != hidden_states.dtype
            or not residual.is_contiguous()
        ):
            return "residual_layout_mismatch", None, None

        ctx = get_context()
        md = getattr(ctx, "kda_metadata", None)
        if md is None:
            return "no_recurrent_metadata", None, None
        if state_manager is None:
            state_manager = getattr(ctx, "kda_state", None)
        if state_manager is None:
            return "no_state_manager", None, None

        num_prefills = getattr(md, "num_prefills", None)
        num_decodes = getattr(md, "num_decodes", None)
        if not num_prefills:
            return "no_prefill", None, None
        if num_decodes:
            # A mixed or decode-only batch takes a different branch inside the
            # recurrence, with its own metadata; not this graph.
            return "decode_tokens_present", None, None
        if getattr(md, "any_have_initial_state", False):
            # Capture needs warmup iterations that really execute, and with a non-zero
            # initial recurrent state those iterations consume it and write the final
            # state back -- so the second warmup would read state the first one
            # overwrote, and the caller's state would be wrong before the real call
            # ever ran.  With an all-zero initial state each iteration recomputes the
            # same final state, which is what makes the warmup safe.
            return "nonzero_initial_recurrent_state", None, None

        tokens = hidden_states.numel() // hidden_states.shape[-1]
        if tokens > _GRAPH_MAX_TOKENS:
            return "above_token_threshold", None, None

        if self._state_buffers(state_manager) is None:
            return "no_recurrent_state_buffers", None, None

        # The unchanged-weight predicate is *not* here: it has to drop the cached graph, and
        # every mutation of graph state belongs inside the instance lock.  ``_graph_forward``
        # runs it there.
        return None, md, state_manager

    def _state_buffers(self, state_manager):
        """The conv and recurrent state tensors this layer's graph writes through."""
        li = self.layer_idx
        try:
            if self._is_linear:
                conv = state_manager.gdn_conv[li]
                rec = state_manager.recurrent[li]
            else:
                conv = state_manager.k_cache[li]
                rec = state_manager.v_cache[li]
        except (AttributeError, IndexError, TypeError):
            return None
        if not isinstance(conv, torch.Tensor) or not isinstance(rec, torch.Tensor):
            return None
        return conv, rec

    def _weight_snapshot(self) -> tuple:
        """``(name, id, pointer, dtype, shape)`` for every parameter and buffer, in order.

        Walked fresh on every call, deliberately.  The cheaper version -- re-reading
        ``data_ptr()`` from the parameter objects retained at capture -- cannot see a
        *replacement*: holding the objects pins their storages, so it observes mutation and
        never substitution, and ``layer.mlp.gate.weight = nn.Parameter(...)`` slips straight
        through it.  The hooks on ``load_state_dict``, ``_apply`` and ``__setattr__`` cover the
        ways a caller normally gets there, but what replay needs is a predicate on the weights
        themselves rather than a set of covered entry points -- a public ``drop_graph()`` only
        moves the obligation to the caller.  Measured: 39.8 us for the 15-entry walk in isolation,
        and -0.8 us (-0.13%) on the replay itself, because it is host work that overlaps the
        asynchronous copy and replay.  See ``docs/notes/contract.md``.
        """
        return tuple(
            (name, id(t), t.data_ptr(), t.dtype, t.shape)
            for name, t in (
                *self.named_parameters(recurse=True),
                *self.named_buffers(recurse=True),
            )
        )

    def _graph_key(self, hidden_states, residual, positions, rotary_emb, md,
                   state_manager) -> tuple:
        """Everything replay assumes about the caller's plan.

        Identities *and* pointers: the identity of the metadata object catches the
        harness publishing a fresh plan (which it does before every correctness round
        and again before timing), and the ``data_ptr()``s catch a plan that was
        rebuilt in place.  A strong reference to each keyed object is held for the
        graph's lifetime, so an ``id()`` freed and reissued to a different object
        cannot compare equal.
        """
        plan = getattr(md, "nums_dict", None)
        entry = plan.get(_CONV_BLOCK_T) if isinstance(plan, dict) else None
        if _MUT_KEY_SHAPE_ONLY:
            return (tuple(hidden_states.shape), hidden_states.dtype,
                    residual is None, positions is None)
        parts = [
            tuple(hidden_states.shape),
            hidden_states.dtype,
            residual is None,
            positions is None,
            id(rotary_emb),
            None if _MUT_KEY_NO_METADATA else id(md),
            id(state_manager),
            id(plan),
            # The chunk plan is published twice -- on the metadata and inside the
            # per-tile entry -- and the conv kernel reads the entry's copy.  Holding
            # only the outer dict does not retain an old entry after
            # ``nums_dict[8] = new_entry``, so the entry is keyed and held in its own
            # right.  ``tot`` is the captured Triton grid size and ``mlist_len`` is what
            # the reference conv fallback sizes itself from; neither is implied by the
            # token count, because different partitions of the same token count give
            # different sums of ``ceil(seq_len / 8)``.
            id(entry),
            entry.get("tot") if isinstance(entry, dict) else None,
            entry.get("mlist_len") if isinstance(entry, dict) else None,
        ]
        if not _MUT_KEY_NO_POINTERS:
            for name in _MD_TENSORS:
                parts.append(_tensor_sig(getattr(md, name, None)))
            if isinstance(entry, dict):
                for name in _CONV_PLAN_TENSORS:
                    parts.append(_tensor_sig(entry.get(name)))
            for t in self._state_buffers(state_manager):
                parts.append(_tensor_sig(t))
        parts += [
            getattr(md, "num_prefills", None),
            getattr(md, "num_decodes", None),
            getattr(md, "num_prefill_tokens", None),
            getattr(md, "num_decode_tokens", None),
            getattr(md, "num_actual_tokens", None),
            bool(getattr(md, "all_have_initial_state", False)),
            bool(getattr(md, "any_have_initial_state", False)),
        ]
        return tuple(parts)

    def _graph_hold(self, md, state_manager, rotary_emb) -> tuple:
        """Strong references to every object the key compares by identity or pointer.

        Without these the allocator is free to free a keyed tensor and hand its address
        to something else, at which point a stale key compares equal and the replay
        reads freed memory.  ``Q3ND_MUT_WEAK_HOLD=1`` swaps them for weak references so
        a test can show that.
        """
        plan = getattr(md, "nums_dict", None)
        entry = plan.get(_CONV_BLOCK_T) if isinstance(plan, dict) else None
        # The per-tile entry is held in its own right: holding only the outer dict does
        # not retain an old entry after ``nums_dict[8] = new_entry``, which would leave
        # its ``id()`` free to be reissued.
        held = [md, state_manager, rotary_emb, plan, entry]
        for name in _MD_TENSORS:
            t = getattr(md, name, None)
            if isinstance(t, torch.Tensor):
                held.append(t)
        if isinstance(entry, dict):
            for name in _CONV_PLAN_TENSORS:
                t = entry.get(name)
                if isinstance(t, torch.Tensor):
                    held.append(t)
        held.extend(self._state_buffers(state_manager))
        if _MUT_WEAK_HOLD:
            out = []
            for obj in held:
                try:
                    out.append(weakref.ref(obj))
                except TypeError:
                    out.append(None)
            return tuple(out)
        return tuple(held)

    def _graph_drop(self) -> None:
        """Release the cached graph and everything it holds alive.

        At most one graph is ever cached: a rebuild drops the previous one rather than
        accumulating a per-key cache, which would pin one private memory pool per key.

        Waits for the last replay to *finish on the device* first.  The instance lock is released
        as soon as a replay is enqueued, so by the time another thread gets here the device may
        still be reading the static input buffers and writing the output buffers this is about to
        free.  A lock cannot express that; an event can.  Only the recorded event is waited on --
        not the whole device -- so an unrelated stream is not stalled.
        """
        done = self.__dict__.get("_g_done")
        if done is not None:
            device = self.__dict__.get("_g_device")
            try:
                if device is not None:
                    with torch.cuda.device(device):
                        done.synchronize()
                else:  # pragma: no cover - only if a graph existed without a device
                    done.synchronize()
            except Exception:  # noqa: BLE001 - a torn-down context must not block the drop
                pass
            self._g_done = None
        if _MUT_MULTI_GRAPH and self._g_graph is not None:
            self._g_retired.append(
                (self._g_graph, self._g_static, self._g_out, self._g_hold),
            )
        self._g_key = None
        self._g_graph = None
        self._g_static = None
        self._g_out = None
        self._g_hold = None
        self._g_weights = None
        self._g_stream = None
        self._g_device = None

    def graph_cache_size(self) -> int:
        """How many captured graphs this layer currently holds."""
        return (1 if self._g_graph is not None else 0) + len(self._g_retired)

    def _graph_build(self, hidden_states, residual, positions, rotary_emb,
                     state_manager, md, key) -> bool:
        global LAST_CAPTURE_ERROR, LAST_VERIFY
        # Cleared up front so a capture that fails cannot leave an ``ok`` from an older graph
        # sitting there looking like this one was verified.
        LAST_VERIFY = None
        self._graph_drop()

        static_h = hidden_states.clone()
        static_r = None if residual is None else residual.clone()
        # No static buffer for ``positions``: nothing on the linear path reads it, and the
        # graph path never serves the full-attention layer.  Skipping the copy is worth
        # ~4 us at one token (10.12x against 9.05x with it).  ``positions is None`` stays
        # in the key regardless, so a caller that starts or stops passing it rebuilds.
        device = hidden_states.device

        try:
            # Warm up before capture, so that everything first-use and *persistent* on
            # this path is materialized outside the graph's private memory pool: the
            # FlashInfer SM100 GDN adapter's compiled kernel and its per-device
            # workspace, the trtllm-gen MoE extension load and its cubins, FlashInfer's
            # autotuner singleton and its warn-once state, Triton's process-global
            # allocator and both prologue kernels' compilation, and the compiled residual
            # norm's Dynamo/Inductor compilation -- which must not begin inside a capture
            # region at all.  A buffer first allocated inside the capture belongs to the
            # graph's pool, and this repository records reusing one from eager code
            # afterwards as a silent, traceback-free rank kill in
            # ``L2/flashinfer_allreduce_fusion.py``.
            #
            # Two things this warmup deliberately does *not* claim.  It is not a
            # substitute for ``process_weights_after_loading`` -- the GDN's merged input
            # projection and the MoE's block-major expert weights are prepared by the
            # caller before any forward, and the harness's module-prep hook does that.
            # And the trtllm launcher's per-call routing arrays, padded-token maps and
            # GEMM workspaces are reallocated on every call and cannot be hoisted; they
            # are captured intermediates that never escape the launcher, which is why
            # letting them come from the graph pool is correct rather than a leak.
            #
            # Capture runs on the same explicit stream as the warmup.  The frozen GDN
            # winner keys its optional scratch cache on the current stream, so warming on
            # one stream and capturing on another would leave capture creating a second,
            # graph-pool-owned entry that eager code could later be handed.  That cache
            # is off by default; capturing on the warmup stream makes the invariant hold
            # whether or not a caller turns it on.
            #
            # Every stream and graph object below belongs to the *input's* device, not to
            # the process's current device.  Two instances of this layer on two GPUs in one
            # process are an ordinary thing to have, and ``torch.cuda.Stream()`` with no
            # argument would put both captures on device 0.
            with torch.cuda.device(device):
                side = torch.cuda.Stream(device=device)
                side.wait_stream(torch.cuda.current_stream(device))
                iters = 0 if _MUT_NO_WARMUP else _GRAPH_WARMUP_ITERS
                with torch.cuda.stream(side), torch.no_grad():
                    for _ in range(iters):
                        self._eager_forward(
                            static_h, static_r, positions, rotary_emb, state_manager,
                        )
                torch.cuda.current_stream(device).wait_stream(side)
                torch.cuda.synchronize(device)
                if _MUT_CAPTURE_FAIL:
                    raise RuntimeError("fault injection: Q3ND_MUT_CAPTURE_FAIL=1")
                graph = torch.cuda.CUDAGraph()
                with torch.no_grad(), torch.cuda.graph(graph, stream=side):
                    out = self._eager_forward(
                        static_h, static_r, positions, rotary_emb, state_manager,
                    )
                torch.cuda.current_stream(device).wait_stream(side)
        except Exception as exc:  # noqa: BLE001 - degrade to eager, never half-captured
            LAST_CAPTURE_ERROR = f"{type(exc).__name__}: {exc}"
            self._graph_drop()
            return False

        LAST_CAPTURE_ERROR = None
        self._g_graph = graph
        self._g_static = (static_h, static_r)
        self._g_out = tuple(out)
        self._g_hold = self._graph_hold(md, state_manager, rotary_emb)
        self._g_weights = self._weight_snapshot()
        self._g_device = device
        # The graph replays on whichever stream calls ``replay()``, but the static buffers
        # and the output tuple are single-instance state.  Binding the graph to the stream
        # that built it, and declining every other stream, is what keeps two concurrent
        # forwards from overwriting each other's inputs and outputs.
        self._g_stream = torch.cuda.current_stream(device).cuda_stream
        self._g_key = key

        if _GRAPH_VERIFY:
            # Same static buffers, same metadata object that will be replayed.  The published
            # plan has ``has_initial_state`` all-False, so both runs start the recurrence from
            # zero and the comparison is meaningful.  Under the capture device's guard, and
            # synchronizing *that* device rather than whichever one is current.
            with torch.cuda.device(device), torch.no_grad():
                ref = tuple(
                    t.clone() for t in self._eager_forward(
                        static_h, static_r, positions, rotary_emb, state_manager,
                    )
                )
                graph.replay()
                torch.cuda.synchronize(device)
            if _MUT_VERIFY_MISMATCH and ref:
                # Fault injection: perturb the reference the check compares against, so a
                # passing verification cannot be passing vacuously.
                ref = (ref[0] + 1,) + ref[1:]
            ok = len(ref) == len(self._g_out) and all(
                a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
                for a, b in zip(self._g_out, ref)
            )
            LAST_VERIFY = "ok" if ok else "mismatch"
            del ref
            if not ok:
                self._graph_drop()
                return False
        return True

    def _graph_forward(self, hidden_states, residual, positions, rotary_emb,
                       state_manager):
        """Serve this call by replay, or return ``_MISS`` to let the eager chain have it.

        Everything that reads or writes graph state happens inside the instance lock, and the
        owner-stream check happens *before* any drop or rebuild.  Both matter for the same
        reason: there is one graph, one set of static input buffers and one output tuple per
        instance.  With the ownership check after the rebuild, a second stream carrying a
        different key would drop and re-capture the first stream's graph and only then be told
        it is not the owner -- so the non-owner would have destroyed the owner's fast path and
        left a graph bound to itself.  With the build outside the lock, two first calls can
        capture concurrently into the same slots.
        """
        global LAST_REJECTION
        reason, md, state_manager = self._graph_rejection(
            hidden_states, residual, positions, rotary_emb, state_manager,
        )
        if reason is not None:
            LAST_REJECTION = reason
            return _MISS

        with self._g_lock:
            # Ownership first, and only when a graph exists: a fresh instance has no owner yet,
            # so the first stream to arrive becomes it.
            if self._g_graph is not None and (
                torch.cuda.current_stream(self._g_device).cuda_stream != self._g_stream
            ):
                # Work on one stream is ordered against itself, which is what makes
                # copy-in / replay / clone-out safe there; across streams it is not.  Decline
                # without dropping the graph, so the owner keeps its fast path.
                LAST_REJECTION = "other_stream"
                return _MISS

            if (not _MUT_NO_WEIGHT_CHECK and self._g_graph is not None
                    and self._weight_snapshot() != self._g_weights):
                # Any weight moved, was replaced, or changed dtype or shape since capture --
                # including a nested ``Parameter`` assigned directly, which no hook sees.
                # Serve eagerly now and drop the graph so the next call rebuilds against the new
                # weights.  The hooks on ``load_state_dict`` / ``_apply`` / ``__setattr__`` are
                # kept as well: they drop the graph *at the moment* of the change, which beats
                # waiting for the next forward to notice.
                self._graph_drop()
                LAST_REJECTION = "weight_pointers_changed"
                return _MISS

            key = self._graph_key(
                hidden_states, residual, positions, rotary_emb, md, state_manager,
            )
            if key != self._g_key and not self._graph_build(
                hidden_states, residual, positions, rotary_emb, state_manager, md, key,
            ):
                LAST_REJECTION = "capture_failed"
                return _MISS

            device = self._g_device
            static_h, static_r = self._g_static
            with torch.cuda.device(device):
                static_h.copy_(hidden_states)
                if static_r is not None:
                    static_r.copy_(residual)
                self._g_graph.replay()
                out = self._g_out
                result = tuple(t.clone() for t in out) if _GRAPH_CLONE else out
                # Everything above is *enqueued*, not finished, and the lock is about to be
                # released.  This event is what a later drop or rebuild waits on before freeing
                # the buffers the device is still using.  Reused rather than reallocated per
                # call: re-recording an event is cheap and allocating one is not free.
                if not _MUT_NO_DONE_EVENT:
                    done = self._g_done
                    if done is None:
                        done = torch.cuda.Event()
                        self._g_done = done
                    done.record()
            LAST_REJECTION = None
            return result

    # -- Entry point ----------------------------------------------------------

    def forward(self, hidden_states, residual, positions=None,
                rotary_emb=None, state_manager=None):
        if _GRAPH:
            out = self._graph_forward(
                hidden_states, residual, positions, rotary_emb, state_manager,
            )
            if out is not _MISS:
                LAST_PATH["layer"] = "replay"
                return out
        LAST_PATH["layer"] = "eager"
        return self._eager_forward(
            hidden_states, residual, positions, rotary_emb, state_manager,
        )
