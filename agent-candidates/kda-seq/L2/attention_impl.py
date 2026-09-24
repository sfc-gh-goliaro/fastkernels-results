"""Paged attention layer, inherited, with one intercepted dispatch path.

The baseline is a ~790-line vLLM-style attention layer with seven dispatch paths
(tree-verify, mixed prefill+decode, pure prefill, pure decode, Triton-unified and
an SDPA fallback) over paged and unpaged KV, chunked local attention, attention
sinks and sliding windows. Every captured call reaches exactly **one** of them.

``_build_attention_prefill_inputs`` materialises ``query``/``key``/``value`` and
publishes a prefill ``Context`` with ``block_tables`` and ``slot_mapping`` left
unset, while the module's ``k_cache``/``v_cache`` stay at their ``__init__``
default of ``torch.tensor([])``. Inside ``forward_impl`` that means
``store_kvcache`` is skipped, the tree-verify / mixed / Triton-unified branches
are all bypassed, and ``is_prefill`` is true with ``block_tables is None`` --
landing on the last branch of ``_forward_pure``: an unpaged, single-segment,
causal variable-length prefill. The captured forward variants carry no ``_ctx``,
so that context is built with one segment, ``cu_seqlens_q == cu_seqlens_k ==
[0, N]`` and ``max_seqlen_q == max_seqlen_k == N`` as host Python ints.

On this device ``AttnBackendConfig.auto_detect()`` selects ``trtllm``, so
``prefill_op`` is ``TRTLLMPrefill``; with ``block_table is None`` that is its
dense fallback -- ``flash_attn_varlen_func(..., causal=True, fa_version=4)``.
``FA_VERSION == 4`` here, so the baseline on the scored path already **is**
FlashAttention-4, the hand-written CuTeDSL Blackwell kernel bundled in vLLM. The
frozen lower-level ``flashinfer_prefill`` winner has the identical fallback, so
there is nothing to inherit from it on this path either.

Why subclass instead of reproducing the tree
--------------------------------------------
The decision tree is ~790 lines with six paths no captured call reaches, plus two
metadata remappers. Copying them buys nothing and risks silent divergence on
every future baseline edit; inheriting them makes parity a property of the
language rather than of a transcription, and keeps the diff equal to the
optimisation. Exactly one method is overridden -- ``_forward_pure``, the tightest
enclosure of the one reachable path. Overriding it rather than ``forward_impl``
means the inherited ``forward_impl`` has already fetched the ``Context`` once,
done the ``view(N, num_heads, head_size)`` reshapes, run ``store_kvcache`` where a
cache exists, routed tree-verify / mixed / Triton-unified elsewhere, and owns the
``reshape(N, num_heads * head_size)`` on the way out -- so the flat-output
contract is structurally satisfied instead of re-implemented.

What the claim predicate decides
--------------------------------
``_claims_unpaged_causal_prefill`` is a conjunction of *sufficient* conditions,
in the style of the frozen lower-level winners' dispatch predicates. Anything it
declines runs the inherited path byte for byte, so parity is the floor by
construction and an untested configuration costs nothing but the opportunity.

It is a total function of **host-visible metadata**: shapes, dtypes, strides,
``numel()``, device identity, Python ints and ``None`` from the ``Context``, and
constructor-derived scalars. It never reads a device value -- no ``.item()``, no
``.cpu()``, no ``.tolist()``, no ``bool(tensor)``, no indexing of a CUDA tensor --
because a device-to-host copy would serialise against work already queued on the
stream and cost far more than the path it is choosing between can save, and
because routing must never depend on tensor *contents*.

Two consequences of that rule are worth naming:

* Single-segment identity comes from host integers, not from reading
  ``cu_seqlens``: ``max_seqlen_q == max_seqlen_k == N == q.shape[0] ==
  k.shape[0]`` together with ``cu_seqlens_q.numel() == cu_seqlens_k.numel() ==
  2``. The prefill context sets ``max_seqlen_* = max(segment sizes)``, so any
  multi-segment split makes ``max_seqlen_q < N`` and is declined for free.
* Paged-ness is read from ``ctx.block_tables`` and ``ctx.sliding_block_tables``
  **directly**, never through the inherited ``_group_block_tables`` helper. That
  helper routes through ``_sliding_group_tensor``, which for a sliding-window
  layer carrying a ``_sliding_group_id`` performs ``tensor[gid].contiguous()`` --
  real GPU work on a call the predicate is about to decline.

Documented preconditions, not checks
------------------------------------
``cu_seqlens[0] == 0`` and ``cu_seqlens[-1] == query.shape[0]`` -- the defining
invariant of the packed variable-length layout -- are **assumed**, exactly as the
frozen ``flash_attn_varlen`` winner documents them, because confirming them needs
a device-to-host read and that is precisely the synchronisation this path exists
to avoid.

"Assumed" needs to be stated more carefully than "a violation is undefined
anyway", because that is not true in general. The two entry points derive the
segment extent from different places -- the variable-length one from
``cu_seqlens[i+1] - cu_seqlens[i]``, the dense one from the physical token count --
so a caller who violates the invariant can make them disagree. If the *key*
extent alone is understated, the inherited path returns a fully **defined** answer
(bottom-right causal masking blanks the leading query rows) that the dense route
does not reproduce. That is a real difference, not two flavours of undefined.

So the predicate closes it without reading a value: it requires
``cu_seqlens_q`` and ``cu_seqlens_k`` to be the **same buffer**. A shared buffer
can only understate both extents together, and that leaves the inherited path's
trailing query rows unwritten -- undefined, with no defined answer to differ from.
Every producer in this tree publishes one tensor for both, so the restriction
costs nothing on any reachable call, and a caller passing two equal-valued buffers
loses the fast path rather than receiving a wrong answer.

The predicate is also careful never to *become* a synchronising read: the
``Context`` flags are compared by identity against ``True``/``False`` rather than
evaluated for truth, and the host integers have their types checked before they
are compared, because ``Context`` annotates these fields but does not enforce the
annotation and a device tensor in a boolean context would synchronise.

Which lower-level ops the inherited constructor composes
--------------------------------------------------------
``bench.py`` loads this class through ``_load_candidate_class`` and does not call
``apply_candidates``, so during scoring the inherited ``__init__`` resolves its
function-local ``from ..L1.flashinfer_prefill import TRTLLMPrefill`` under
``fastkernels.tasks.baseline.L1`` -- the baseline lower-level ops, not the frozen
winners. That is harmless here (the frozen ``flashinfer_prefill``'s dense
fallback is identical, both sides of any A/B see the same ops, and the fast path
bypasses ``prefill_op`` entirely), but it is a real difference from a transcribed
candidate, which would pick up ``candidate.L1``. It also means this module is
insensitive to ``--standalone``.

Under the full-engine path ``apply_candidates`` monkey-patches *module-level*
references to a baseline class with its candidate class, and the
``_BaselineAttention`` alias below is such a reference -- after patching that name
denotes this very class. Zero-argument ``super()`` is safe (it resolves through
the ``__class__`` cell and the unmodified method resolution order), so it is the
only form used; an explicit ``_BaselineAttention._forward_pure(self, ...)`` would
recurse forever.

Measured, and why the design looks like this
--------------------------------------------
All figures from the probes under ``profile/``, on this B200. The scored interval
is dominated by the harness: ``_time_module`` copies every *contiguous* forward
tensor into a shifting pool inside the timed region, so the four small rows pay
14-16 us of harness cost before any attention happens -- except the sink+window
row, whose q, k and v are all non-contiguous and therefore passed through
unchanged, at 3.6 us. Even a module that computes nothing is capped at
1.63x/2.85x/3.28x/4.88x there, and "one output-sized kernel" (1.31x/2.22x/2.30x/
1.55x) is the honest bound for anything memory-bound.

FA4 knob sweep, as end-to-end ratios on the five scored rows (A, C, D, E, B):

  dense batch-1 entry point  1.008x  1.109x  1.094x  1.138x  1.00x   <- shipped
  num_splits 0 / 2 / 4       fails to compile (TYPE_UNSTABLE_JOIN on n_block_first)
  pack_gqa=False             0.997x  0.910x  1.000x  1.000x  0.966x
  tile_mn=(128, 64)          1.078x  0.906x  0.851x  1.000x  0.834x
  tile_mn=(64, *), (192,128) rejected by the compiler (M-mode must be 64 or 128)

Three conclusions, recorded so they are not re-litigated: ``pack_gqa`` is already
on by default (it is ``qhead_per_kvhead > 1``), so there is no grouped-query
packing win left to collect; ``use_2cta_instrs`` requires ``not causal and not
local and cu_seqlens_q is None``, so two-SM cooperative MMA is unreachable on
every captured case regardless of entry point; and the dense re-route's benefit is
therefore *not* two-CTA scheduling but dropping the variable-length sequence
indirection and its per-tile bounds. The dense entry is bit-identical to the
varlen call on all five scored rows (max abs difference 0.00e+00) -- measured with
the corpus's all-zero sink, so that identity is not claimed for non-zero sinks,
which are covered by the numerical corpus under ``tests/`` instead.

The largest row is not a target. FA4 runs it at 1.26-1.33 PFLOPS, 78-83 % of the
1.605 PFLOPS FA4 itself reports on B200; the dense re-route measures 1.04x there
and every other knob is a loss. It passes the predicate and takes the dense
route, so its parity is a *measured* property rather than a structural one -- the
cheapest discriminators are ordered first to keep the predicate's host cost far
below the 0.1 % that row could notice.

The authored kernel, and where it is allowed to serve
-----------------------------------------------------
Above the dense re-route sits one authored Triton kernel, ``_seq_attn_fwd``: a
single-segment causal attention forward with fp32 online softmax in log2 space,
grouped-query heads read by index, attention sinks in FA4's own non-max-raising
form, an optional sliding window, runtime strides, and a fixed launch table. It
serves only the token counts and layer configurations that ``_AUTHORED_ROUTE``
names, and every entry there was admitted by a paired end-to-end measurement of
the whole module against the whole baseline module -- see that table's comment for
the admission rule, and ``profile/route_admission.csv`` for the 34-rung ladders.

Two of the five scored rows are routed to it and win far beyond the dense floor;
the other three are declined and take the dense route. That asymmetry is the
design working rather than a gap: at the mid token counts one program per
``(query-block, query head)`` is roughly 0.4x-0.7x of FA4, because FA4 packs the
query heads of a grouped-query layer into its M dimension and reads each key-value
tile once. Packing those heads is the obvious next variant and it is not attempted
here; the route table is what makes not attempting it safe.

What the whole design deliberately does not do: bypass FA4's ~6 us Python preamble
(measured small, and fragile); split KV (fails to compile in this build); de-page
anything (the benchmark never presents a page table on this path); or claim the two
largest rows.
"""

from __future__ import annotations

import inspect

import torch
import triton
import triton.language as tl

from ...baseline.L2.attention_impl import Attention as _BaselineAttention
from ....infra.fa_utils import FA_VERSION

# FA4's dense entry point. Resolved once here rather than per call so a build
# without the CuTeDSL path disables the fast route instead of raising on every
# forward. ``FA_VERSION == 4`` is required because the dense entry is only
# equivalent to what the baseline runs when the baseline is also FA4.
try:  # pragma: no cover - exercised by whichever build is installed
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as _fa4_dense_fwd
except Exception:  # noqa: BLE001 - any import failure means: serve from the baseline
    _fa4_dense_fwd = None

_DENSE_ROUTE_AVAILABLE = _fa4_dense_fwd is not None and FA_VERSION == 4

# ``num_splits`` is pinned for the same reason ``baseline.py`` pins it: FA4's
# split-KV heuristic picks a kernel that fails to compile in this build
# (TYPE_UNSTABLE_JOIN on ``n_block_first``) for mid-size sequences, and every
# other value was measured to fail or to lose.
_DENSE_NUM_SPLITS = 1

# Head sizes the fast route is validated for. Both captured sizes are here; a
# layer outside them runs the inherited path.
_CLAIMED_HEAD_SIZES = (64, 128)

# Input dtypes the fast route is validated for. fp32 and mixed-dtype calls run
# the inherited path.
_CLAIMED_DTYPES = (torch.bfloat16, torch.float16)

# Captured against the base method at import: a signature drift in the baseline
# should fail loudly here rather than silently mismatch at call time.
_BASE_FORWARD_PURE_SIG = inspect.signature(_BaselineAttention._forward_pure)

_LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Authored kernel: single-segment causal attention forward.
# ---------------------------------------------------------------------------
# ``do_not_specialize`` on the three integer scalars is a decision about *warming*,
# not about speed. Triton specialises a kernel on integer arguments equal to 1 and
# on divisibility by 16, so left alone, ``n_tokens=1`` and ``n_tokens=656`` compile
# two different programs -- and the reachable set would then depend on which token
# counts happen to occur, which warming cannot enumerate honestly. Pinning
# ``n_tokens`` collapses that; ``kv_group`` and ``window_left`` are fixed by the
# route key and the layer, so pinning them costs nothing and keeps the key small.
#
# The **strides are deliberately left to specialise**, which is the opposite
# choice, for a measured reason: pinning them costs 9 % of the kernel at one token
# and 18 % at sixty, and 1.7x at the mid token counts, because the lost hint is
# that a row stride is a multiple of 16 elements -- the hint that drives vectorised
# tile loads. That was measured by compiling this same function body under four
# policies; see ``profile/probe_specialisation_cost.py``.
#
# Leaving them free keeps the program set enumerable anyway, because only three of
# the eight stride arguments are the caller's to choose, and the claim predicate
# *checks* the rest rather than hoping. ``stride_qh``, ``stride_kh`` and
# ``stride_vh`` are required to equal the head size and the two output strides are
# this function's own allocation, so all five are fixed once the route key is -- and
# they are multiples of 16 for both claimed head sizes. That leaves the three row
# strides, which the predicate requires to be at least a whole row wide, so each can
# only be a multiple of 16 elements or not (never 1, which Triton would specialise as
# a constant), giving eight combinations per program. ``_warm_authored_kernel``
# launches all eight, and ``tests/probe_kernel.py`` then attacks the result with row
# pitches warming never used and asserts that nothing new compiles.
#
# Pointer alignment is also part of the key, and is pinned from the other side: the
# predicate requires 16-byte-aligned data pointers, so every reachable call is in the
# aligned class that warming builds.
#
# One limit worth naming: the cache is per device. Warming runs at import, on the
# device the process is using, and the predicate requires q, k and v to share a
# device -- so a process that later switched to a *different* device would compile
# there. ``bench.py`` gives each worker one device, so this cannot arise under
# scoring; it is a real limit outside it.
@triton.jit(do_not_specialize=["n_tokens", "kv_group", "window_left"])
def _seq_attn_fwd(
    Q, K, V, Out, Sink,
    stride_qm, stride_qh,
    stride_kn, stride_kh,
    stride_vn, stride_vh,
    stride_om, stride_oh,
    n_tokens, kv_group, window_left,
    scale_log2e,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
):
    """One program per ``(query-block, query-head)``.

    Query and key extents are equal -- the caller establishes that on the host --
    so bottom-right causal alignment collapses to ``j <= i`` and there is no
    sequence-offset arithmetic and no cumulative-length load anywhere in the body.
    Grouped-query heads are read by index rather than broadcast, so K and V are
    fetched at the key-value head and never materialised per query head.

    The online softmax runs in fp32 throughout, in log2 space: the scale is folded
    as ``scale * log2(e)`` on the host so the inner loop uses ``exp2`` directly.
    """
    pid_m = tl.program_id(0)
    off_h = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < n_tokens

    kv_h = off_h // kv_group

    q = tl.load(Q + offs_m[:, None] * stride_qm + off_h * stride_qh + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Causal: the last key this block can attend to is its own last query row.
    hi = tl.minimum(start_m + BLOCK_M, n_tokens)
    if HAS_WINDOW:
        # The first key the block's *first* row can attend to, floored to a tile so
        # the loop stays tile-aligned. Skipping whole tiles is the only reason the
        # window term is worth having in the loop bound rather than the mask alone.
        lo = tl.maximum(start_m - window_left, 0)
        lo = (lo // BLOCK_N) * BLOCK_N
    else:
        lo = 0

    for start_n in range(lo, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        mask_n = offs_n_cur < n_tokens
        k = tl.load(
            K + offs_n_cur[:, None] * stride_kn + kv_h * stride_kh + offs_d[None, :],
            mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale_log2e

        keep = mask_m[:, None] & mask_n[None, :]
        keep = keep & (offs_n_cur[None, :] <= offs_m[:, None])
        if HAS_WINDOW:
            keep = keep & (offs_n_cur[None, :] >= offs_m[:, None] - window_left)
        qk = tl.where(keep, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # A row still entirely masked carries ``m_new == -inf``; subtracting it
        # would give ``-inf - (-inf) = NaN``, so rescale against zero there.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + offs_n_cur[:, None] * stride_vn + kv_h * stride_vh + offs_d[None, :],
            mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    if HAS_SINK:
        # FA4's own form: the sink is a logit for a slot with no value vector, so it
        # enters the denominator and nothing else. The running max is deliberately
        # *not* raised to include it -- matching ``flash_fwd_sm100.py``, which
        # computes ``row_sum += exp2(sink * LOG2_E - row_max_scaled)`` -- which
        # means the value accumulator needs no rescale and cannot be left
        # inconsistent with the denominator. The sink arrives in natural-log units,
        # hence the log2(e) conversion into this kernel's log2 domain.
        #
        # A sink well above the row max simply dominates the denominator and drives
        # the output toward zero, which is the correct limit -- all the mass sits on
        # the null slot. It only reaches literal ``+inf`` when the *relative* logit
        # exceeds ln(FLT_MAX) ~ 88.7, and even then fp32 ``acc / +inf`` is zero
        # rather than NaN, which is what FA4's reciprocal of the same sum also
        # gives. So there is nothing to guard here.
        sink = tl.load(Sink + off_h).to(tl.float32)
        m_use = tl.where(m_i == float("-inf"), 0.0, m_i)
        l_i = l_i + tl.exp2(sink * 1.4426950408889634 - m_use)

    # With equal query and key extents the diagonal is always kept -- including at
    # ``window_left == 0``, where causality and the window intersect at exactly
    # ``j == i`` -- so no row with finite inputs ends fully masked. The guard still
    # earns its two instructions: it covers the inactive lanes of a final partial
    # block, and it covers non-finite input data, which can drive a row's max to
    # -inf however the mask is written. FA4 keeps the equivalent special case.
    empty = l_i == 0.0
    acc = acc / tl.where(empty, 1.0, l_i)[:, None]

    tl.store(Out + offs_m[:, None] * stride_om + off_h * stride_oh + offs_d[None, :],
             acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


# Token-count bands. The band a call lands in picks its launch configuration, and
# it is also the unit the route table is validated over, so the two cannot drift.
_BANDS = ((64, "tiny"), (256, "small"), (1024, "mid"), (1 << 62, "large"))


def _band(n: int) -> str:
    for bound, name in _BANDS:
        if n <= bound:
            return name
    return _BANDS[-1][1]


# Launch configurations, chosen offline by the sweep under ``profile/`` and fixed
# here: no autotuner is reachable from the call path, so nothing compiles and no
# thread is spawned inside the harness's guarded timing window. Keyed
# ``(head_size, band)`` -> ``(BLOCK_M, BLOCK_N, num_warps, num_stages)``.
#
# Each entry is the fastest of a 64-configuration sweep (BLOCK_M and BLOCK_N over
# {16, 32, 64, 128}, warps over {4, 8}, stages over {2, 3}) on that band's
# representative captured shape.
#
# Only the ``tiny`` band is reachable from a forward call -- the route table bounds
# both of its entries at 64 tokens -- and both ``tiny`` entries were re-swept under
# the *shipped* specialisation policy, which is what ``profile/launch_cfg.csv``
# records; the paired admission ladder then timed exactly these two configurations
# end to end. The other bands' entries come from the original nine-case sweep
# (``profile/launch_cfg_sweep.log``), which predates that policy, so they are the
# measured answer for an older compilation and may no longer be their band's
# optimum. They are unreachable from ``forward``, and are kept only because
# ``tests/probe_kernel.py`` exercises the kernel across every band, which is what
# keeps it trustworthy for a later phase. Any future route entry that extends past
# 64 tokens must re-sweep the band it extends into.
_LAUNCH_CFG: dict[tuple[int, str], tuple[int, int, int, int]] = {
    (64, "tiny"): (16, 64, 8, 2),
    (64, "small"): (16, 64, 8, 2),
    (64, "mid"): (128, 128, 8, 3),
    (64, "large"): (128, 32, 4, 3),
    (128, "tiny"): (16, 32, 4, 2),
    (128, "small"): (16, 64, 8, 2),
    (128, "mid"): (128, 128, 8, 2),
    (128, "large"): (128, 128, 8, 2),
}

# Where the authored kernel is allowed to serve, keyed
# ``(dtype, head_size, num_heads, num_kv_heads, window_left, has_sink)`` ->
# inclusive ``(n_min, n_max)``.
#
# The key carries the sliding window's *width*, not merely whether one exists,
# because the width sets the kernel's loop lower bound and changes FA4's competing
# workload too -- so a win measured at one width does not speak for another. Only
# the width the captures actually present is listed.
#
# Both entries were admitted by ``profile/route_admission.py``, which walks a
# 34-rung token-count ladder and, at every rung, times four arms round-robin --
# baseline, dense route, authored route, and a second baseline as the noise floor --
# then admits only the widest contiguous run of rungs where the authored route beat
# the **dense route** by at least 2 % in *every* repeat and beat the baseline in
# every repeat. The margin is taken against the dense route because that is the
# alternative actually on offer: it is bit-identical to the baseline and free.
#
# Two percent rather than the per-rung noise half-width, because within one process
# the harness is reproducible enough that the half-width collapses to ~0.001 while
# the timings are quantised near 0.03 us out of ~28 us -- so a rung clearing its own
# noise by 0.001 has shown nothing except that both routes are sitting on the
# harness floor. An earlier pass with the noise half-width as the threshold admitted
# exactly such rungs, which is what prompted the fixed margin.
#
# The key carries the head *counts* as well as the head size, so a win measured at
# one grouped-query ratio never speaks for another -- the ratio is what decides
# whether one program per query head can compete with FA4's packed key-value reuse,
# and at the mid token counts it decidedly cannot.
#
# The ladder ran to 129 and every rung of it won, so it never found either
# configuration's crossover -- the launch-configuration sweep puts the kernel far
# behind FA4 by 656 tokens, so it lies in between. The bound is nevertheless **64**,
# not 129, for two reasons that have nothing to do with where the wins stopped:
#
#  * Below 64 the ladder is dense -- 28 rungs, never a gap wider than four, and every
#    grid-size boundary of ``BLOCK_M = 16`` (17, 33, 49) bracketed. Above it the
#    ladder jumps 96 -> 129, so 97 and 113 -- both grid-size boundaries -- are
#    unmeasured, and a range containing unmeasured boundaries is interpolated rather
#    than validated.
#  * 64 is also the edge of the ``tiny`` band, so a routed call has exactly one
#    launch configuration per entry. Spanning two bands would put a config change at
#    n = 65 inside a claimed range.
#
# It costs nothing that is scored: the two routed rows are at 1 and 60 tokens, and no
# captured shape lies between 64 and 656. Anything not listed takes the dense route,
# which is bit-identical to the baseline, so the cost of an omission is the
# opportunity and never the correctness.
_AUTHORED_ROUTE: dict[tuple, tuple[int, int]] = {
    # Row A's configuration. Across 1..64: 1.18x-1.37x over the baseline module,
    # 1.17x-1.27x over the dense route.
    (torch.bfloat16, 128, 16, 1, None, False): (1, 64),
    # Row E's configuration, sink and a 128-wide sliding window both active. Across
    # 1..64: 1.18x-1.55x over the baseline module, 1.09x-1.36x over the dense route.
    (torch.bfloat16, 64, 32, 4, 127, True): (1, 64),
}


def _launch_seq_attn(q, k, v, sink, window_left, scale, cfg=None):
    """Allocation-light launcher: one output tensor, one kernel, no host reads."""
    n, h_q, d = q.shape
    h_kv = k.shape[1]
    block_m, block_n, num_warps, num_stages = (
        cfg if cfg is not None else _LAUNCH_CFG[(d, _band(n))])
    out = torch.empty((n, h_q, d), dtype=q.dtype, device=q.device)
    _seq_attn_fwd[(triton.cdiv(n, block_m), h_q)](
        q, k, v, out, sink,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        n, h_q // h_kv, window_left if window_left is not None else 0,
        scale * _LOG2E,
        BLOCK_M=block_m, BLOCK_N=block_n, D=d,
        HAS_SINK=sink is not None,
        HAS_WINDOW=window_left is not None,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _reachable_specialisations():
    """Every compiled program the route table can select, exactly.

    A program is identified by its constexprs and launch metaparameters --
    ``(dtype, head_size, BLOCK_M, BLOCK_N, num_warps, num_stages, has_sink,
    has_window)`` -- together with the alignment class of each argument Triton
    specialises on. The route key and the launch table fix the first group; of the
    second, only the three row strides are free, and each is either a multiple of
    16 elements or not. So the set below, crossed with those eight combinations, is
    the whole of what a call can reach.
    """
    specs = {}
    for (dtype, d, h_q, h_kv, has_window, has_sink), (n_lo, n_hi) in _AUTHORED_ROUTE.items():
        lo_prev = 0
        for bound, name in _BANDS:
            band_lo, band_hi = lo_prev + 1, bound
            lo_prev = bound
            n = max(band_lo, n_lo)
            if n > min(band_hi, n_hi):
                continue          # this band does not intersect the routed range
            cfg = _LAUNCH_CFG[(d, name)]
            specs.setdefault((dtype, d, cfg, has_sink, has_window),
                             (n, h_q, h_kv))
    return specs


def _row_strided(n: int, heads: int, d: int, dtype, aligned: bool) -> torch.Tensor:
    """A ``[n, heads, d]`` view in each of the two reachable row-stride classes.

    The route requires every row to start on a 16-byte boundary, so a reachable row
    stride is always a whole number of 16-byte units. Triton's own alignment class is
    coarser -- a multiple of 16 *elements* or not -- and both of its classes are
    reachable inside that: a row of ``heads * d`` elements is a multiple of 16, and
    adding one 16-byte unit (eight elements at two bytes each) lands outside it.
    """
    width = heads * d
    stride = width if aligned else width + 16 // torch.tensor([], dtype=dtype).element_size()
    buf = torch.zeros(n * stride, dtype=dtype, device="cuda")
    return buf.as_strided((n, heads, d), (stride, d, 1))


def _warm_authored_kernel() -> None:
    """Compile every reachable program once, at import.

    ``bench.py`` samples ``threading.active_count()`` immediately before timing the
    candidate and checks it immediately after, so a compile anywhere in the
    candidate's warmup *or* timed loop is reported as a reward hack rather than as
    a slow row. Import happens long before that sample is taken, which is the only
    place this work can safely go.

    The probe values matter. Each program is warmed at a token count inside the band
    it serves, with the route entry's real grouped-query ratio and a real window,
    and once for each of the eight ways the three row strides can be aligned or not
    -- because a stride's alignment class is part of the compile key, so an
    unforeseen layout would otherwise compile inside the timed window.
    """
    for (dtype, d, cfg, has_sink, has_window), (n, h_q, h_kv) in \
            _reachable_specialisations().items():
        sink = torch.zeros((h_q,), dtype=dtype, device="cuda") if has_sink else None
        window_left = (n // 2) if has_window else None
        for q_aligned in (True, False):
            for k_aligned in (True, False):
                for v_aligned in (True, False):
                    q = _row_strided(n, h_q, d, dtype, q_aligned)
                    k = _row_strided(n, h_kv, d, dtype, k_aligned)
                    v = _row_strided(n, h_kv, d, dtype, v_aligned)
                    _launch_seq_attn(q, k, v, sink, window_left, 0.125, cfg)
    torch.cuda.synchronize()


#: Why the authored route is unavailable, when it is. Degrading to the dense
#: re-route on a toolchain problem is right, but degrading *silently* hides real
#: breakage behind a plausible 1.07x, so the reason is kept.
_AUTHORED_ROUTE_ERROR: str | None = None

if not _AUTHORED_ROUTE:
    _AUTHORED_ROUTE_ERROR = "no configuration has a recorded paired win"
elif not torch.cuda.is_available():
    _AUTHORED_ROUTE_ERROR = "no CUDA device available at import"
else:
    try:
        _warm_authored_kernel()
    except Exception as exc:  # noqa: BLE001 - serve everything from the dense route
        _AUTHORED_ROUTE_ERROR = f"{type(exc).__name__}: {exc}"
        _AUTHORED_ROUTE = {}


def _as_batch1(t: torch.Tensor) -> torch.Tensor:
    """``[n, h, d] -> [1, n, h, d]`` as a pure view, carrying strides through.

    A single-segment variable-length call describes one logical sequence, which
    is exactly a dense batch-1 call. Nothing is copied, so the row strides the
    captures record (q, k and v are slices of a fused projection) survive.
    """
    n, h, d = t.shape
    s0, s1, s2 = t.stride()
    return t.as_strided((1, n, h, d), (n * s0, s0, s1, s2))


class Attention(_BaselineAttention):
    """The baseline layer with its unpaged causal prefill path intercepted.

    Only ``_forward_pure`` is overridden. Everything else -- construction,
    ``forward``, ``forward_impl``, the paged, decode, mixed, tree-verify,
    Triton-unified and SDPA paths, ``set_trtllm_workspace`` and
    ``process_weights_after_loading`` -- is inherited unchanged.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None,
                 attention_chunk_size: int | None = None,
                 prefer_triton: bool = False):
        super().__init__(
            num_heads, head_size, scale, num_kv_heads=num_kv_heads,
            sliding_window=sliding_window, sinks=sinks,
            attention_chunk_size=attention_chunk_size,
            prefer_triton=prefer_triton,
        )
        # The half of the claim predicate that depends only on constructor
        # arguments, so a call pays for the per-call half alone. This caches a
        # decision; it changes no inherited state.
        self._layer_claimable = bool(
            _DENSE_ROUTE_AVAILABLE
            # A Triton-unified layer never reaches ``_forward_pure`` anyway;
            # naming it here keeps the predicate readable as one conjunction.
            and not self._triton_only
            # Chunked local attention rewrites the sequence metadata into virtual
            # batches, which is a different problem than the one this path solves.
            and self.attention_chunk_size is None
            and self.head_size in _CLAIMED_HEAD_SIZES
            and self.num_kv_heads > 0
            and self.num_heads % self.num_kv_heads == 0
            # A tensor scale would have to be compared on the device to be used.
            and isinstance(self.scale, (int, float))
            and not isinstance(self.scale, bool)
            # A scale of exactly zero makes FA4's dense path return NaN, while the
            # authored kernel would return the uniform-attention answer -- so the two
            # routes would disagree, and the candidate would disagree with the
            # baseline on whichever one it took. Declined rather than reconciled: the
            # inherited path keeps producing exactly what it produces today. The
            # frozen lower-level ``flashinfer_prefill`` winner excludes it for the
            # same reason, and no non-zero scale down to 1e-12 is affected.
            and self.scale != 0
        )
        # ``window_size`` as the dense entry point spells it. The variable-length
        # wrapper maps ``(left, right)`` to ``window_size_left/right``, passing
        # ``None`` for either end that is negative; mirrored exactly so the two
        # entry points describe the same mask.
        left, right = self._fa3_window_size
        self._dense_window_left = left if left >= 0 else None
        self._dense_window_right = right if right >= 0 else None
        # The constructor-fixed part of the route key; the call supplies the dtype.
        self._authored_key_tail = (
            self.head_size, self.num_heads, self.num_kv_heads,
            self._dense_window_left, self._fa3_sinks is not None,
        )

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        out = self._unpaged_causal_prefill(q, k, v, k_cache, v_cache, ctx)
        if out is not None:
            return out
        return super()._forward_pure(q, k, v, k_cache, v_cache, ctx)

    # -- the claim predicate ------------------------------------------------
    def _claims_unpaged_causal_prefill(self, q, k, v, k_cache, v_cache, ctx) -> bool:
        """Is this call an unpaged, single-segment, causal prefill this module serves?

        Every term reads host-visible metadata only: a shape, a dtype, a stride, a
        ``numel()``, a ``data_ptr()``, a device identity, a Python int/bool/``None``,
        or a value cached from the constructor. Cheapest and most discriminating
        terms first.

        The three ``Context`` flags are compared by *identity* against ``True`` and
        ``False`` rather than read in a boolean context. ``Context`` annotates them
        as ``bool`` but does not enforce it -- ``set_forward_context`` forwards
        arbitrary keywords into the dataclass -- and a device tensor in a boolean
        context is a synchronising read. Identity comparison is exact for real
        bools and declines anything else, so no ``Context`` content can make this
        predicate touch the device. The same reasoning covers the host integers:
        their types are checked before they are compared.
        """
        if not self._layer_claimable:
            return False
        # ``_forward_pure`` serves prefill and decode; only prefill is claimed.
        # ``is_mixed`` and ``is_tree_verify`` cannot be set on a call routed here by
        # the inherited ``forward_impl``, which has already branched on both, but
        # this method is reachable directly too and declining costs two reads.
        if ctx.is_prefill is not True or ctx.is_mixed is not False:
            return False
        if getattr(ctx, "is_tree_verify", False) is not False:
            return False
        # Read the page tables directly. Going through ``_group_block_tables``
        # would launch a gather for a sliding-window layer, on a call that may be
        # about to be declined.
        if ctx.block_tables is not None or ctx.sliding_block_tables is not None:
            return False
        # A populated cache means the paged representation is the authoritative
        # one; ``numel()`` reads the shape, not the memory.
        if k_cache.numel() or v_cache.numel():
            return False

        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            return False
        n_q = q.shape[0]
        if n_q == 0:
            return False
        # Host integers force the single-segment reading and equal query and key
        # extents, which is what makes bottom-right causal masking reduce to
        # ``j <= i``. A multi-segment context reports ``max_seqlen_q < n_q``.
        msq, msk = ctx.max_seqlen_q, ctx.max_seqlen_k
        if type(msq) is not int or type(msk) is not int:
            return False
        if msq != n_q or msk != n_q:
            return False
        if k.shape[0] != n_q or v.shape[0] != n_q:
            return False

        if q.shape[1] != self.num_heads or q.shape[2] != self.head_size:
            return False
        if k.shape[1] != self.num_kv_heads or k.shape[2] != self.head_size:
            return False
        if k.shape != v.shape:
            return False

        dtype = q.dtype
        if dtype not in _CLAIMED_DTYPES or k.dtype is not dtype or v.dtype is not dtype:
            return False
        device = q.device
        if device.type != "cuda" or k.device != device or v.device != device:
            return False

        cu_q, cu_k = ctx.cu_seqlens_q, ctx.cu_seqlens_k
        if cu_q is None or cu_k is None:
            return False
        if cu_q.dim() != 1 or cu_k.dim() != 1:
            return False
        # One segment on both sides. The *values* are never read; see below and the
        # module docstring on the packed-layout preconditions.
        if cu_q.numel() != 2 or cu_k.numel() != 2:
            return False
        # The layout FA4's variable-length entry point requires of these two, so
        # that anything this accepts is also something the inherited path accepts:
        # int32, on the same device as the data, unit stride. Without these the
        # inherited path raises where this one would happily compute.
        if cu_q.dtype is not torch.int32 or cu_k.dtype is not torch.int32:
            return False
        if cu_q.device != device or cu_k.device != device:
            return False
        if cu_q.stride(0) != 1 or cu_k.stride(0) != 1:
            return False
        # Query and key cumulative lengths must be the *same buffer*. This is what
        # makes the unchecked packed-layout precondition safe rather than merely
        # documented, and it is the one term here that is about the consequences of
        # a violation rather than about the shape of a valid call.
        #
        # The two entry points read the segment extent from different places: the
        # variable-length one from ``cu_seqlens[i+1] - cu_seqlens[i]``, the dense
        # one from the physical token count. If a caller violates
        # ``cu_seqlens[-1] == tokens``, those disagree. With one shared buffer the
        # disagreement can only shorten *both* extents together, which leaves the
        # inherited path's trailing query rows unwritten -- undefined, so there is
        # no defined answer for this route to differ from. With two different
        # buffers a caller could shorten only the key extent, and then the
        # inherited path returns a fully defined result (bottom-right causal
        # masking blanks the leading query rows) that the dense route would not
        # reproduce. Declining that case is cheaper than reading the values.
        #
        # Every producer in this tree publishes one tensor for both, so the cost is
        # nothing here; a caller passing two equal-valued buffers loses the fast
        # path rather than getting a wrong answer.
        if cu_q.data_ptr() != cu_k.data_ptr():
            return False
        # Only the innermost stride matters: it is what both FA4 entry points
        # assume, and it makes the variable-length wrapper's ``maybe_contiguous``
        # a no-op, so the two entry points see identical tensors. The row stride
        # is free to be the width of the fused projection these views came from.
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            return False
        # FA4 *asserts* 16-byte alignment of the data pointer on both of its entry
        # points rather than checking it politely, so the inherited path raises on a
        # misaligned tensor. The authored kernel has no such requirement and would
        # compute an answer -- which is a behavioural difference dressed up as
        # leniency, and exactly the kind of thing this module promises not to do. So
        # the predicate declines what FA4 would reject, and the inherited path raises
        # as it does today.
        #
        # Measured rather than assumed: acceptance depends on the base pointer alone,
        # and the row stride's own alignment is irrelevant to it -- a row stride of
        # 2049 bfloat16 elements is accepted, a storage offset of one element is not
        # (``tests/probe_alignment.py`` enumerates the grid).
        if q.data_ptr() % 16 or k.data_ptr() % 16 or v.data_ptr() % 16:
            return False
        # The head stride must be the head size and the row stride must be at least a
        # whole row. Both hold for every ``view(n, heads, head_size)`` of a flat
        # ``[n, heads * head_size]`` tensor, which is what the inherited
        # ``forward_impl`` hands over -- so on the benched path these are free.
        #
        # They are checked rather than assumed because they are what makes the set of
        # compiled programs enumerable, and therefore what makes import-time warming
        # complete. An overlapping layout could present a row stride of 1, which
        # Triton specialises as a constant and which warming does not build; a 3-D
        # caller could present a head stride other than the head size, which warming
        # also does not build. Either would compile inside the timed window, and the
        # harness reports that as a reward hack rather than as a slow row.
        if q.stride(1) != self.head_size or k.stride(1) != self.head_size:
            return False
        if v.stride(1) != self.head_size:
            return False
        if q.stride(0) < self.num_heads * self.head_size:
            return False
        row_kv = self.num_kv_heads * self.head_size
        if k.stride(0) < row_kv or v.stride(0) < row_kv:
            return False

        sinks = self._fa3_sinks
        if sinks is not None:
            if sinks.dim() != 1 or sinks.shape[0] != self.num_heads:
                return False
            if sinks.stride(0) != 1 or sinks.device != device:
                return False
            # Same reasoning as for q/k/v: the dense route hands this tensor to FA4,
            # which asserts on its alignment, while the authored kernel reads one
            # element per program and would not.
            if sinks.data_ptr() % 16:
                return False
        return True

    # -- the route ----------------------------------------------------------
    def _unpaged_causal_prefill(self, q, k, v, k_cache, v_cache, ctx):
        """Serve a claimed call, or return ``None`` to defer to the inherited path.

        Returns ``[n, num_heads, head_size]``; the inherited ``forward_impl``
        flattens it.
        """
        if not self._claims_unpaged_causal_prefill(q, k, v, k_cache, v_cache, ctx):
            return None
        if self._routes_to_authored_kernel(q, k, v):
            return _launch_seq_attn(q, k, v, self._fa3_sinks,
                                    self._dense_window_left, self.scale)
        return self._dense_prefill(q, k, v)

    def _routes_to_authored_kernel(self, q, k, v) -> bool:
        """Was *this* configuration measured to beat the baseline module?

        A host-only lookup, like the predicate. The key is deliberately narrow --
        head counts as well as head size, and the presence of a window and a sink
        separately -- because the alternative is letting a win measured at one
        grouped-query ratio speak for another, and the ratio is exactly what
        decides whether one program per query head can compete with FA4's packed
        key-value reuse.
        """
        if not _AUTHORED_ROUTE:
            return False
        bounds = _AUTHORED_ROUTE.get((q.dtype,) + self._authored_key_tail)
        if bounds is None:
            return False
        if not bounds[0] <= q.shape[0] <= bounds[1]:
            return False
        # Every *row* must start on a 16-byte boundary, not just the tensor. The
        # kernel's tile loads are vectorised along the contiguous head dimension, and
        # a row stride that is not a whole number of 16-byte units puts later rows at
        # a 2-byte-aligned address, which those loads get wrong -- measured, not
        # theorised: at a row stride of 2049 bfloat16 elements the kernel disagrees
        # with the baseline beyond tolerance (``tests/probe_alignment.py``).
        #
        # This is a *route* condition rather than a claim condition, because FA4 has
        # no such requirement: it handles a 2049 stride correctly. So an odd stride
        # keeps the dense route, which is bit-identical to the baseline, instead of
        # losing the fast path altogether. Every captured stride -- 2048, 2304, 2560,
        # 6144 -- already satisfies it.
        unit = 16 // q.element_size()
        if q.stride(0) % unit or k.stride(0) % unit or v.stride(0) % unit:
            return False
        # The kernel reads the sink at the query head with no conversion, so its
        # element type is part of the compiled program. A sink in some other dtype
        # is a program that was never warmed, and takes the dense route.
        sinks = self._fa3_sinks
        return sinks is None or sinks.dtype is q.dtype

    def _dense_prefill(self, q, k, v):
        """FA4 through its dense batch-1 entry point.

        The same FA4 forward the baseline reaches, entered without the
        variable-length sequence indirection. Measured bit-identical to the
        variable-length call on every scored row, for 1.00x-1.14x.
        """
        out, _, _, _ = _fa4_dense_fwd(
            _as_batch1(q), _as_batch1(k), _as_batch1(v),
            softmax_scale=self.scale,
            causal=True,
            learnable_sink=self._fa3_sinks,
            window_size_left=self._dense_window_left,
            window_size_right=self._dense_window_right,
            num_splits=_DENSE_NUM_SPLITS,
        )
        return out.squeeze(0)


if inspect.signature(Attention._forward_pure) != _BASE_FORWARD_PURE_SIG:
    raise RuntimeError(
        "Attention._forward_pure has drifted from the baseline method it "
        f"overrides: {inspect.signature(Attention._forward_pure)} != "
        f"{_BASE_FORWARD_PURE_SIG}"
    )
