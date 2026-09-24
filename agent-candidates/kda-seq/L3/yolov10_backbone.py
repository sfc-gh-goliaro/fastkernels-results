"""YOLOv10 native backbone -- the eleven-block composition, replayed from one CUDA graph.

This operator has no arithmetic of its own. It is eleven blocks in sequence, and every one of
them is already a frozen L2 winner in this tree, so the only thing left to remove is the cost of
*asking* for the work. Composing the winners (``_eager`` below) already collapses the baseline's
~200 ``nn.Module.__call__`` frames and 160--180 device ops to eleven frames and roughly thirty
kernels: measured 2.551x at N=4 and 3.213x at N=1 against the harness baseline's 2.4946 ms /
2.3690 ms (``python validate.py``, first commit of this file). What remains after that is host
submission -- eleven module frames whose own guards are lean but not free, plus per-launch host
time the frozen ``yolov10_scdown`` measured at 8.8--16.8 us.

So ``forward`` replays the whole backbone from a single captured ``torch.cuda.CUDAGraph`` when it
can, and runs ``_eager`` when it cannot. ``_eager`` is ``baseline.py``'s own eleven lines over the
frozen winners, which makes the fallback the composition rather than a second implementation --
and both paths run the *same* eleven statements (``_stages``), so the fallback cannot drift away
from what capture recorded.

Nothing is derived in ``__init__``. The harness moves the module to the device, recasts every
parameter to fp16, re-randomises any it judges uninitialised and then loads the baseline's
weights, all *before* the first forward, so constants built at construction time would describe
weights that no longer exist. That is the argument every frozen L2 winner in this tree makes, and
it is why capture happens on a forward instead. Nothing derived is registered either: a new
``state_dict`` key would be loaded with ``strict=False`` and silently keep a candidate-local
value.

Capture happens on the **second** eligible call for a key, not the first. The first eligible call
is what runs each ``YOLOC2f``'s ``_measure_route``, which times three routes against each other
with ``torch.cuda.Event`` and six ``torch.cuda.synchronize()`` calls -- illegal under capture (and
from CUDA 13.0 an explicit failure rather than undefined behaviour), and its result decides *which
kernel sequence a capture would even record*. It is also what settles the Triton JIT
specializations, ``YOLOSPPF._derive_folded``, ``YOLOPSA._fold_weights`` and the C2f C++
``BlockPlan``. Capturing on call two rather than deferring to the timing warmup means correctness
rounds 2 and 3 both validate the replay path against the harness baseline, which is the cheapest
correctness evidence available.

Measured on the box (``tests/canary_capture.py``, one GPU leased through ``with_gpu.py``): the
whole eleven-block region captures and replays bitwise at N=4 and N=1, with ``YOLOPSA``'s two
``launch_pdl=True`` launches inside the captured region and no hang across repeated replays. The
``stem1..sppf`` region does too, and stays available as ``FK_YOLOV10_BACKBONE_ROUTE=partial``
rather than as a paragraph, so choosing it later is a measurement and not a rewrite. CUDA 13.0.2's
own documentation is why the PDL launches were expected to survive: capture does not keep the
launch attribute, it translates it into a ``cudaGraphDependencyTypeProgrammatic`` edge, which is
what the consumer's ``gdc_wait()`` waits on.

What the numbers are, and the one thing to know about reading them
-----------------------------------------------------------------
Every latency below is ``fastkernels bench``'s own median CUDA-event time, from a file in
``bench_results/ab/``. They come in two regimes, and the regime matters more than any difference
within it: this node runs several workspaces' benches at once, each on its own GPU but all on the
same host CPU.

======================================  ==============  ==============
configuration                           N=4             N=1
======================================  ==============  ==============
eager composition, node busy            1.2380-1.2432   0.8950-0.9503
graph replay, node busy                 1.0383-1.0423   0.7056-0.7066
graph replay, lighter load              0.6543-0.7108   0.4516-0.5254
harness baseline (every run)            2.3302-2.5289   2.2209-2.5514
======================================  ==============  ==============

**The one comparison this file actually claims** is the pair whose arms ran adjacently under the
same load: eager 1.2390 ms against replay 1.0405 ms at N=4, and 0.9247 against 0.7064 at N=1 --
replay 16% and 24% faster. The 0.65-0.71 ms replay figures are recorded because they are real, but
there is **no eager measurement taken under that lighter load**, so pairing them with a busy-node
eager number would be exactly the mistake ``.humanize/bitlesson.md`` records, and this file does
not do it.

Note also what the same-commit spread says: replay itself moved 0.7108 -> 1.0405 ms with node
load, a 46% increase, so "replay is insensitive to host load" would be wrong too. Both paths are
load-sensitive; the controlled pair says replay is the faster of the two under the same load, and
that is all it says.

The harness baseline is the one stable number here (2.33-2.53 ms across every run, whatever the
load), which is why per-case ``speedup`` cannot be used to normalize the candidate's load
sensitivity away: the baseline is device-bound and the candidate is not, so the two do not move
together.

Guard cost, host-side against the real tree (``tests/bench_guard_cost.py``): **102.7 us** per
admitted call end to end, of which the key is 0.37 us. Components, each measured separately: the
live re-fetch of 180 tensors 3.63 us, then identity 3.73, ``_version`` 7.94, ``data_ptr`` 7.81,
contiguity 8.52, dtype 7.38 and the negation bit 12.63 us over that list; the structural scan
32.02 us; the module re-fetch 3.28 us with identity 3.56 and ``training`` 4.16 us over it. The two
designs ruled out by measurement rather than by taste: re-walking ``parameters()`` + ``buffers()``
costs 218 us, and a recursive per-module hook scan costs 72 us. Both are affordable exactly once,
at capture time, and are used there.

**The guard is not on the critical path, and the evidence is an accident of history rather than an
argument.** The guard doubled between two commits -- 47.8 us to 102.7 us, when the review-driven
maps went in -- and the in-harness number did not move: 1.0405, 1.0383 and 1.0404 ms at N=4 across
three busy-node runs spanning both versions. A guard-stripped build (0.5 us of guard) measured
1.0382 ms in the same regime. Four numbers within 0.2% while the host-side cost varied by 200x.

The mechanism: the bench enqueues the next iteration's 2x-L2 flush immediately after the previous
iteration's end event, so the host runs a full iteration ahead of the device, and roughly a
millisecond of host work per iteration is hidden behind device work that is already queued. That is
why the complete guard ships and its price is recorded rather than traded away -- and it is also
the reason not to read the guard's cost as free in any other harness.

Output shapes, measured adjacently: the packed shape 0.7108 ms and returning the static tensors
0.7016 ms at N=4 -- 1.3% apart. The packed shape ships anyway, because returning the static
tensors makes two successive calls alias and the baseline never does. Three clones measured
1.0423 ms but under a different load, so that arm is not comparable and is recorded as such.

The graph's private pool: 38 MiB reserved at N=4, 24 MiB at N=1, reclaimed when the module is
destroyed (``tests/check_graph.py`` group ``memory``).

Two measured findings this file records but deliberately does not act on
-----------------------------------------------------------------------
Both are left to the next step because acting on either is a change to *what the frozen winners
compute with*, not to how this module submits it, and both want a quiet node to decide.

1. **The frozen C2f route ranking looks like it inverts under replay.**
   ``YOLOC2f._measure_route`` picks fused/aten/folded by eager timing, i.e. with launch overhead
   included, and under replay launch count is nearly free. Measured at N=4, all with the graph:
   ``aten`` 0.5984 ms, ``folded`` 0.6502 ms, the chosen ``fused`` route 0.7108 ms. Without the
   graph: ``fused`` 1.2380 ms, ``folded`` 1.3011 ms, ``aten`` 1.3196 ms -- the ranking the winner
   itself measured. The eager half is a controlled pair (baselines 2.4339 and 2.4152 ms); the
   replayed half is **not** -- those three runs were not adjacent, and load alone moves a single
   commit by more than the 9-19% that separates them. So this is a strong indication and not an
   established result, and what would settle it is three adjacent runs on a quiet node. The reason
   to believe it anyway: the frozen file's own docstring already measured cuDNN several times
   faster on the device (8.4 us against 89 us for one 3x3 stage), and the L3 shapes -- ``stage3``
   at 80x80, ``stage4`` at 40x40, N=4 -- are far larger than the L2 captures the fused FFMA
   kernels were sized for.
2. **Leaving PSA outside the graph measured faster than capturing it.** In the one *adjacent*
   pair, ``partial`` 0.9788 ms against the whole region's 1.0383 ms at N=4 (5.7%), with N=1 a wash
   (0.7106 against 0.7060). A second, non-adjacent pair pointed the same way by more (0.6085
   against 0.7108). ``partial`` pays one extra module frame and two extra launches and still wins,
   which is what makes it interesting: the plausible mechanism is PSA's two ``launch_pdl=True``
   stages losing their overlap once their ordering is a graph edge rather than stream
   serialization. One adjacent pair at 5.7%, with the other batch size flat, is not enough to
   flip the shipped region -- so the whole region stays the default, ``partial`` stays one
   environment variable away, and the next step should decide it from adjacent runs plus a
   kernel-level profile of the two PSA stages under both regimes.

The mutation contract
---------------------
Replay skips 140 module frames and every per-call check the frozen winners make, so the guard
re-establishes what they establish. What it **observes** after capture, each clause with its own
test in ``tests/check_guard.py``: the input's shape, dtype, device and exact type; ``training``,
``is_grad_enabled``, ``is_autocast_enabled("cuda")`` and an open forward-mode dual level; any
process-wide module hook, and -- through ``RemovableHandle.next_id`` -- any hook registered
anywhere in the process after capture, including on a descendant; object identity, ``_version``,
``data_ptr``, contiguity, dtype and the negation bit of all 180 float parameters and buffers,
re-fetched live from their owner ``_parameters`` / ``_buffers`` dicts; the identity of every module
binding in the tree, re-fetched live from its owner's ``_modules`` dict, so a *replaced* child is a
refusal rather than a replay of a module that is no longer there; the structural scalars of the nine
holders whose raw tensors the captured kernels read directly -- including the frozen ``YOLOConv``'s
``route``, ``tile`` and cached extents, which select the kernel -- plus the pool's full geometry;
and the ``training`` flag of every module in the tree.

What it does **not** observe, stated rather than implied:

* a value written **through** ``.data`` -- ``p.data.copy_(other)`` changes the bytes the captured
  kernels read while leaving the version counter at its old value, the pointer unchanged and the
  object identical, so all three freshness maps agree. Measured directly, not assumed: version
  0 -> 0, same ``data_ptr``, same ``id``, different values. This is the gap the frozen
  ``yolov10_sppf`` documents for its own signature, and it is the reason ``fuse()`` -- which
  writes through ``.data`` -- is caught there by ``_is_fused`` and a disappearing ``bn`` rather
  than by any tensor key. Nothing cheap closes it: detecting it needs the contents, not the
  metadata. ``p.data.mul_(2)`` is the same gap. What the metadata maps *do* catch is a ``.data``
  rebinding that changes shape, strides, dtype or the negation bit, which is why those three maps
  are there;
* a rebinding *inside* a child's derived state -- ``sppf._folded.w1 = other_tensor``. The keepalive
  holds the folded tensors themselves and not only their container, so the original cannot be
  freed under the graph and replay keeps computing the values it captured; but the guard does not
  inspect derived state, so it will not notice that the module would now compute something else
  eagerly;
* mutating ``mod._forward_hooks`` (or the other three dicts) directly instead of through
  ``register_forward_hook``, which bumps no ``next_id`` and touches no tensor;
* an ``eps`` or structural change *inside* ``stage2``/``stage3``/``stage4``/``stage5`` or ``psa``
  -- those blocks fold such scalars into their own cached plans and their own per-call guards do
  not re-read them either, so this module is no more wrong than the frozen winner it composes;
* host-side bookkeeping the eager path performs and no output depends on: ``YOLOSCDown.calls`` and
  ``last_route``, and ``yolov10_sppf``'s C++ ``g_fast_path_calls`` counter and its ``LAST_REFUSAL``
  module global. Replay does not update them. They are diagnostics, not semantics.

Environment switches, following the ``FK_YOLOV10_C2F_ROUTE`` precedent in this tree. All four are
read per construction, so a test may set one and then build a module:

* ``FK_YOLOV10_BACKBONE_ROUTE`` -- ``graph`` (force; a capture failure raises instead of falling
  back), ``eager`` (never capture, never replay), ``partial`` (capture ``stem1..sppf`` and run
  ``psa`` eagerly). Unset takes the normal path: capture on the second eligible call, replay
  when the guard admits, eager otherwise.
* ``FK_YOLOV10_BACKBONE_OUTPUT`` -- ``packed`` (default), ``clones``, ``static``. See
  :meth:`_Entry.unpack`.
* ``FK_YOLOV10_BACKBONE_GUARD`` -- ``full`` (default) or ``lean``, which keeps the key, the input
  type, the mode/dispatcher flags and the hook checks and drops the three freshness maps, the
  structural scan and the per-descendant ``training`` scan. Only for the guard-cost A/B; ``lean``
  is not a shipping configuration, because it can replay stale weights.
"""

from __future__ import annotations

import os
import sys
from operator import attrgetter, getitem

import torch
import torch.nn as nn
import torch.nn.modules.module as _module_hooks
from torch.autograd import forward_ad as _forward_ad
from torch.utils.hooks import RemovableHandle

from ..L2.yolov10_c2f import YOLOC2f
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_psa import YOLOPSA
from ..L2.yolov10_scdown import YOLOSCDown
from ..L2.yolov10_sppf import YOLOSPPF

# -- routes ---------------------------------------------------------------------------------
ROUTE_AUTO = "auto"
ROUTE_GRAPH = "graph"        # force the graph; a capture failure raises rather than falls back
ROUTE_EAGER = "eager"        # never capture, never replay
ROUTE_PARTIAL = "partial"    # capture stem1..sppf, run psa eagerly -- the canary's contingency
_ROUTE_ENV = "FK_YOLOV10_BACKBONE_ROUTE"
_ROUTES = {ROUTE_GRAPH: ROUTE_GRAPH, ROUTE_EAGER: ROUTE_EAGER, ROUTE_PARTIAL: ROUTE_PARTIAL}

# -- output shapes --------------------------------------------------------------------------
OUT_PACKED = "packed"        # three in-graph copy_s into one staging buffer, one clone outside
OUT_CLONES = "clones"        # three clones of the static outputs
OUT_STATIC = "static"        # the static tensors themselves: aliases across calls, measured only
_OUTPUT_ENV = "FK_YOLOV10_BACKBONE_OUTPUT"
_OUTPUTS = {OUT_PACKED: OUT_PACKED, OUT_CLONES: OUT_CLONES, OUT_STATIC: OUT_STATIC}

# -- guard variants -------------------------------------------------------------------------
GUARD_FULL = "full"
GUARD_LEAN = "lean"          # measurement only; can replay stale weights
_GUARD_ENV = "FK_YOLOV10_BACKBONE_GUARD"
_GUARDS = {GUARD_FULL: GUARD_FULL, GUARD_LEAN: GUARD_LEAN}

# One entry per module instance. The harness builds a fresh pair per case, so one key is all that
# is ever seen; a dict rather than a single slot so the cap is this one constant.
_CACHE_CAP = 1

# Three warm-up iterations on the stream capture runs on. One is enough for the Triton
# specializations (none of the 15 reachable @triton.jit kernels uses @triton.autotune -- grepped,
# and deliberate per two of the winners' own comments); three is cheap insurance for cuDNN's
# per-stream handle, algorithm heuristics and workspace, which the C2f aten/folded routes reach
# through F.conv2d and which YOLOConv's two-kernel route reaches for down3.
_WARMUP_ITERS = 3

# The dtype the graph is captured for. Both captured cases are fp16, every frozen fast path on
# this composition's hot route is fp16-gated (YOLOConv also takes bf16, but YOLOC2f's plan and
# YOLOSPPF's fold are fp16-only), and with a one-entry cache a second dtype would evict the entry
# that serves the scored run. bf16 and fp32 therefore stay eager by design, not by accident.
_GRAPH_DTYPE = torch.float16

_VERSION = attrgetter("_version")
_TRAINING = attrgetter("training")
_DTYPE = attrgetter("dtype")
_DATA_PTR = torch.Tensor.data_ptr
_IS_CONTIGUOUS = torch.Tensor.is_contiguous
_IS_NEG = torch.Tensor.is_neg

_GLOBAL_HOOKS = (
    _module_hooks._global_forward_hooks,
    _module_hooks._global_forward_pre_hooks,
    _module_hooks._global_backward_hooks,
    _module_hooks._global_backward_pre_hooks,
)


def _mode_from_env(env: str, table: dict, default: str) -> str:
    return table.get(os.environ.get(env, "").strip().lower(), default)


def _holder_scalars(holder) -> tuple:
    """Every non-tensor thing a ``conv -> bn -> act`` holder's captured kernels baked in.

    The graph froze the geometry these scalars describe and the pointers the tensors had. A
    caller can reassign any of them afterwards -- they are all plain attributes, and the eager
    path re-reads every one of them on every call -- so a replay after such a change computes the
    geometry it was compiled for on a module that no longer has it, which is a wrong answer rather
    than an error. ``eps`` is in here because the frozen winners fold it into the affine map at
    fold time; the instance-``forward`` clause is the cheap way to swap an implementation without
    changing a type, and it is invisible to every other clause.
    """
    conv = getattr(holder, "conv", None)
    bn = getattr(holder, "bn", None)
    act = getattr(holder, "act", None)
    parts = (
        type(holder), type(conv), type(bn), type(act),
        getattr(holder, "_is_fused", False), holder.training,
        "forward" in vars(holder),
        # The frozen ``YOLOConv`` decides which kernel to launch from these, and every one of
        # them is a plain attribute assigned in ``__init__`` and readable afterwards: ``route``
        # and ``tile`` select the kernel and its tile shape, ``_c1``/``_c2`` are the extents the
        # kernels take as their reduction and store bounds, and ``_route_config`` is the
        # configuration the route was chosen for. A caller who reassigns ``route`` gets a
        # different kernel from the eager path and the captured one from replay.
        getattr(holder, "route", None), getattr(holder, "tile", None),
        getattr(holder, "_c1", None), getattr(holder, "_c2", None),
        getattr(holder, "_route_config", None), getattr(holder, "_weight_shape", None),
        getattr(holder, "_bias_shape", None),
    )
    if conv is not None:
        parts += (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                  conv.padding, conv.dilation, conv.groups, conv.bias is None,
                  "forward" in vars(conv))
    if bn is not None:
        parts += (bn.training, bn.track_running_stats, bn.affine, bn.eps, bn.num_features,
                  "forward" in vars(bn))
    if act is not None:
        parts += ("forward" in vars(act),)
    return parts


def _pool_scalars(pool) -> tuple:
    """The SPPF pool's geometry. The radii 2/4/6 the folded kernels implement are the cascade of
    *this* same-shape 5x5 unit-stride pool; any other kernel size, stride, padding, dilation or
    ``return_indices`` is a different operator -- ``return_indices`` in particular makes the eager
    reference return a tuple where the fold returns a tensor -- and the fold that baked the radii
    in is cached against the pool's own signature, which replay never re-reads."""
    return (type(pool), pool.kernel_size, pool.stride, pool.padding, pool.ceil_mode,
            getattr(pool, "dilation", None), getattr(pool, "return_indices", None))


def _derived_tensors(obj) -> tuple:
    """Every tensor one level inside a child's derived-state container.

    The keepalive holds these *tensors*, not only the container. Holding the container is not
    enough: ``_Folded`` and ``_Stage`` both have mutable slots, so rebinding ``sppf._folded.w1``
    would free the tensor whose address the captured kernel carries while the container itself
    stayed alive. One level is the right depth because that is where the frozen winners keep their
    folded blobs; the guard does not inspect any of this, so the keepalive has to be the thing that
    makes it safe.
    """
    held = []
    for name in getattr(type(obj), "__slots__", ()) or ():
        value = getattr(obj, name, None)
        if isinstance(value, torch.Tensor):
            held.append(value)
        elif isinstance(value, (tuple, list)):
            held.extend(v for v in value if isinstance(v, torch.Tensor))
    for value in (getattr(obj, "__dict__", None) or {}).values():
        if isinstance(value, torch.Tensor):
            held.append(value)
        elif isinstance(value, (tuple, list)):
            held.extend(v for v in value if isinstance(v, torch.Tensor))
    return tuple(held)


class _Entry:
    """One captured graph plus everything that keeps it honest and everything that keeps it alive.

    ``keepalive`` is a correctness requirement, not hygiene. A ``CUDAGraph`` holds no Python
    references to the tensors its kernels address, and the frozen winners' folded constants are
    plain attributes that ``_apply`` / ``load_state_dict`` / a re-fold will drop. If they are
    freed the graph replays against recycled memory -- garbage or a fault, with no exception. So
    the entry holds every parameter and buffer of the module plus each child's derived state,
    opaquely, as objects it never interprets.
    """

    __slots__ = ("key", "graph", "x_static", "outputs", "staging", "spans", "keepalive",
                 "slot_dicts", "slot_names", "identity", "versions", "pointers",
                 "contiguity", "dtypes", "negations",
                 "module_dicts", "module_names", "module_identity", "training_flags",
                 "holders", "pool", "structural",
                 "hook_next_id", "output_mode", "through_psa", "pool_reserved_bytes")

    def unpack(self):
        """The three outputs, as plain tensors the caller owns.

        ``bench`` compares the outputs against the baseline's and then drops them, so returning
        the graph's static tensors would pass. It is still a real semantic deviation -- two
        successive calls would hand back aliases and the first result would change under the
        second, which the baseline never does -- so the default packs the three outputs into one
        flat staging buffer with three ``copy_``s *inside* the graph (free at replay) and clones
        that buffer once outside it, returning three contiguous, disjoint views. One launch
        instead of three. The claim is precisely "no *cross-call* aliasing, three **disjoint**
        views of one storage freshly allocated per call" -- they do share a storage with each
        other, disjointly.

        ``torch.cuda.make_graphed_callables`` is not a counter-example: it returns
        ``tuple(o.detach() for o in static_outputs)``, which shares the static storage.
        """
        mode = self.output_mode
        if mode is OUT_PACKED:
            flat = self.staging.clone()
            p3, p4, p5 = (flat[a:b].view(shape) for a, b, shape in self.spans)
        elif mode is OUT_CLONES:
            p3, p4, p5 = (t.clone() for t in self.outputs)
        else:
            p3, p4, p5 = self.outputs
        return p3, p4, p5


class YOLOv10Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # The baseline's eleven children, same names, same order, built from the frozen L2
        # winners. The names are load-bearing rather than cosmetic: the harness shares weights
        # with ``load_state_dict(baseline.state_dict(), strict=False)``, which reports nothing
        # when a key is missing -- a renamed child keeps its own random weights and then fails
        # numerically for a reason that looks like a kernel bug.
        self.stem1 = YOLOConv(3, 16, 3, 2)
        self.stem2 = YOLOConv(16, 32, 3, 2)
        self.stage2 = YOLOC2f(32, 32, n=1, shortcut=True)
        self.down3 = YOLOConv(32, 64, 3, 2)
        self.stage3 = YOLOC2f(64, 64, n=2, shortcut=True)
        self.down4 = YOLOSCDown(64, 128, 3, 2)
        self.stage4 = YOLOC2f(128, 128, n=2, shortcut=True)
        self.down5 = YOLOSCDown(128, 256, 3, 2)
        self.stage5 = YOLOC2f(256, 256, n=1, shortcut=True)
        self.sppf = YOLOSPPF(256, 256, 5)
        self.psa = YOLOPSA(256, 256)

        # Plain attributes, never buffers: a derived ``state_dict`` key would be loaded with
        # strict=False and silently keep a candidate-local value.
        self._graphs: dict = {}
        self._seen: set = set()
        self._uncapturable: dict = {}
        # Observable by tests, so that an "all green" run cannot quietly mean the graph never
        # engaged and the eager composition's own speed was what got measured.
        self.graph_replays = 0
        self.graph_generation = 0
        self.eager_calls = 0
        # Read per construction rather than at import, so a test can set one and then build.
        self._route = _mode_from_env(_ROUTE_ENV, _ROUTES, ROUTE_AUTO)
        self._output_mode = _mode_from_env(_OUTPUT_ENV, _OUTPUTS, OUT_PACKED)
        self._guard_mode = _mode_from_env(_GUARD_ENV, _GUARDS, GUARD_FULL)

    # -- the composition, which both paths run ----------------------------------------------
    def _stages(self, x: torch.Tensor, through_psa: bool = True):
        """The baseline's eleven statements. Capture records this; the fallback runs this.

        One function rather than two so the fallback cannot drift away from what was captured.
        ``through_psa=False`` is the canary's contingency region, ``stem1..sppf``.
        """
        x = self.stem1(x)
        x = self.stem2(x)
        p2 = self.stage2(x)
        x = self.down3(p2)
        p3 = self.stage3(x)
        x = self.down4(p3)
        p4 = self.stage4(x)
        x = self.down5(p4)
        p5 = self.stage5(x)
        p5 = self.sppf(p5)
        if through_psa:
            p5 = self.psa(p5)
        return p3, p4, p5

    def _eager(self, x: torch.Tensor):
        """The baseline's expression. Also the fallback, and what raises whatever it raises."""
        p3, p4, p5 = self._stages(x)
        return {"p3_backbone": p3, "p4_backbone": p4, "p5_backbone": p5}

    # -- forward ----------------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        route = self._route
        if route is ROUTE_EAGER:
            self.eager_calls += 1
            return self._eager(x)

        key = self._graph_key(x)
        if key is not None:
            entry = self._graphs.get(key)
            if entry is not None:
                if self._guard_admits(entry):
                    return self._replay(entry, x)
                if route is ROUTE_GRAPH:
                    # A forced route exists so that an A/B measures what it thinks it measures.
                    # A guard refusal is not a capture failure, but it produces eager results just
                    # the same, so it is just as fatal to a measurement and is raised too. The one
                    # eager call the lifecycle needs before capture is not this case: it happens
                    # before any entry exists.
                    raise RuntimeError(
                        f"{_ROUTE_ENV}={route} but the guard refused a captured key: the module "
                        f"or its weights changed after capture")
            elif key not in self._uncapturable and len(self._graphs) < _CACHE_CAP:
                if key in self._seen:
                    return self._capture_then_replay(x, key)
                self._seen.add(key)
        elif route is ROUTE_GRAPH:
            raise RuntimeError(
                f"{_ROUTE_ENV}={route} but this call is not capturable: "
                f"{self._ineligible_reason(x)}")

        self.eager_calls += 1
        return self._eager(x)

    # -- the key ----------------------------------------------------------------------------
    def _graph_key(self, x: torch.Tensor):
        """``(shape, dtype, device index, stream)`` for a call a graph could serve, else None.

        Cheapest first, and every clause is here because replay would otherwise be wrong rather
        than merely slow:

        * the geometry, because the graph is compiled for exactly one of them;
        * the current stream, because a replay on a different stream is not ordered against that
          stream's other work;
        * ``type(x) is torch.Tensor``, because a subclass -- a Parameter, a FakeTensor, a
          tensor-subclass wrapper -- carries semantics above the pointer the kernels read;
        * ``training`` / ``is_grad_enabled`` / autocast, because the folded BatchNorms the winners
          bake in are the affine map only in eval, a graph carries no autograd history, and under
          autocast the eager reference returns the autocast dtype;
        * an open forward-mode dual level, because ``no_grad`` does *not* disable forward-mode AD
          and the out-of-graph ``x_static.copy_(x)`` would drop a dual's tangent silently. The
          test is the level counter and not ``torch._C._is_fwd_grad_enabled()``, which is
          unconditionally true -- the same trap the frozen ``yolov10_conv`` documents.
        """
        if type(x) is not torch.Tensor:
            return None
        if x.dtype is not _GRAPH_DTYPE or not x.is_cuda or x.dim() != 4 or x.numel() == 0:
            return None
        if (self.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled("cuda") or _forward_ad._current_level >= 0):
            return None
        device = x.device
        return (tuple(x.shape), x.dtype, device.index,
                torch.cuda.current_stream(device).cuda_stream)

    def _ineligible_reason(self, x: torch.Tensor) -> str:
        """Why :meth:`_graph_key` refused, for the forced-``graph`` error message only."""
        if type(x) is not torch.Tensor:
            return f"input is a {type(x).__name__}, not exactly torch.Tensor"
        if not x.is_cuda:
            return "input is not on CUDA"
        if x.dtype is not _GRAPH_DTYPE:
            return f"input dtype {x.dtype} is not {_GRAPH_DTYPE}"
        if x.dim() != 4 or x.numel() == 0:
            return f"input is {x.dim()}-D with {x.numel()} elements"
        if self.training:
            return "module is in training mode"
        if torch.is_grad_enabled():
            return "gradients are enabled"
        if torch.is_autocast_enabled("cuda"):
            return "an autocast region is active"
        if _forward_ad._current_level >= 0:
            return "a forward-mode dual level is open"
        return "unknown"

    # -- the guard --------------------------------------------------------------------------
    def _guard_admits(self, entry: _Entry) -> bool:
        """Whether the captured graph may serve this call.

        The key clauses have already run in :meth:`_graph_key`; what is left is everything the
        replay path skips that a *later* mutation could invalidate. Ordered cheapest-first, with
        the host costs measured on this box against the real 140-module tree
        (``tests/bench_guard_cost.py``).

        The three freshness maps are three because three distinct mutations need three distinct
        detectors, verified rather than assumed: ``p.mul_(2)`` bumps ``_version`` and leaves
        ``data_ptr`` alone; ``p.data = p.data.clone()`` leaves ``_version`` alone and changes
        ``data_ptr``; and ``nn.Parameter(p.detach())`` changes **neither**, so only object
        identity sees it. Re-fetching each tensor from its owner ``_parameters`` / ``_buffers``
        dict rather than from a stored list is what makes the identity map possible at all -- a
        scan over stored objects compares a rebound parameter against the tensor it replaced.
        Re-walking the ``parameters()`` / ``buffers()`` generators instead costs 208 us on this
        tree and is ruled out by that measurement; it is affordable once, at capture time, and it
        is used there.
        """
        # Hooks fire once per module the eager path enters and never on replay. The four
        # process-wide registries are read directly; ``RemovableHandle.next_id`` is an O(1)
        # conservative detector of *any* hook registered anywhere in the process since capture,
        # which is what closes the descendant-hook gap for a fraction of a microsecond. It is
        # deliberately conservative: it also trips on an unrelated handle created elsewhere, and
        # it does not fall back when a hook is removed, because ``remove()`` does not decrement.
        if (_GLOBAL_HOOKS[0] or _GLOBAL_HOOKS[1] or _GLOBAL_HOOKS[2] or _GLOBAL_HOOKS[3]
                or RemovableHandle.next_id != entry.hook_next_id):
            return False
        if self._guard_mode is GUARD_LEAN:
            return True
        live = list(map(getitem, entry.slot_dicts, entry.slot_names))
        if (tuple(map(id, live)) != entry.identity
                or tuple(map(_VERSION, live)) != entry.versions
                or tuple(map(_DATA_PTR, live)) != entry.pointers):
            return False
        # Three more maps over the same list, because ``p.data = <a view>`` can keep the object,
        # the pointer *and* the version counter while changing what the bytes mean. The frozen
        # per-call guards read all three of these properties; replay reads raw memory and so
        # cannot. ``weight.data = weight.data.transpose(2, 3)`` on a square kernel is the case
        # that motivated them: identical identity, pointer and version, non-contiguous strides,
        # and an eager path that demotes to a different kernel while replay would not.
        if (tuple(map(_IS_CONTIGUOUS, live)) != entry.contiguity
                or tuple(map(_DTYPE, live)) != entry.dtypes
                or tuple(map(_IS_NEG, live)) != entry.negations):
            return False
        # Replacing a module leaves every tensor the guard re-fetches untouched, because the
        # re-fetch goes through the *old* module's ``_parameters`` dict. So the module bindings
        # themselves are re-fetched too, from their owners' ``_modules`` dicts, and compared by
        # identity: ``self.stem1 = YOLOConv(...)`` or ``stage3.cv1 = ...`` is then a refusal
        # rather than a replay of the module that is no longer there. One re-fetch serves both
        # this and the ``training`` scan below.
        live_mods = list(map(getitem, entry.module_dicts, entry.module_names))
        if tuple(map(id, live_mods)) != entry.module_identity:
            return False
        # ``nn.Module.train()`` recurses, so ``self.training`` (already checked) covers the usual
        # case -- but a single descendant can be switched on its own, and a descendant in training
        # mode is a different function inside the blocks whose BatchNorms the winners folded.
        if tuple(map(_TRAINING, live_mods)) != entry.training_flags:
            return False
        return self._structural_snapshot(entry) == entry.structural

    def _structural_snapshot(self, entry: _Entry) -> tuple:
        """The scalars the replay would otherwise compute stale, for the holders the captured
        kernels read as raw memory: ``stem1``, ``stem2``, ``down3``, ``down4.cv1/cv2``,
        ``down5.cv1/cv2``, ``sppf.cv1/cv2``, plus the SPPF pool. Read through references resolved
        once at capture time, so no ``nn.Module.__getattr__`` chain is walked per call."""
        parts = [_pool_scalars(entry.pool)]
        for holder in entry.holders:
            parts.append(_holder_scalars(holder))
        return tuple(parts)

    # -- replay -----------------------------------------------------------------------------
    def _replay(self, entry: _Entry, x: torch.Tensor):
        """One ``copy_`` in, one graph launch, one pack out.

        The input copy is not optional: the harness's ``_ShiftingPool`` hands every iteration a
        different ``data_ptr`` (a fresh slot per call), so the graph cannot address the caller's
        tensor. 19.7 MB at N=4 is roughly 2.5 us of device traffic, and it normalizes a
        non-contiguous, negated or conjugated input for free, which is why those replay rather
        than falling back.
        """
        entry.x_static.copy_(x)
        entry.graph.replay()
        self.graph_replays += 1
        p3, p4, p5 = entry.unpack()
        if not entry.through_psa:
            # The contingency region stops before PSA's two PDL launches; one extra host call.
            p5 = self.psa(p5)
        return {"p3_backbone": p3, "p4_backbone": p4, "p5_backbone": p5}

    # -- capture ----------------------------------------------------------------------------
    def _capture_then_replay(self, x: torch.Tensor, key):
        """Capture the region, then replay it for this call's answer.

        Capture executes nothing, so the capturing call has to replay to produce values.

        A failure here is not allowed to become a wrong answer or a silent slowdown. On the normal
        route the key is marked permanently uncapturable, one line naming the reason goes to
        stderr, and the module runs eager forever -- once per key, never retried, because a
        capture that failed for a structural reason will fail the same way every call and
        retrying it would cost a stream sync per forward. Under a forced ``graph`` route the
        exception propagates instead, so an A/B cannot quietly measure the eager path.
        """
        forced = self._route is ROUTE_GRAPH
        try:
            entry = self._capture(x, key)
        except Exception as exc:  # noqa: BLE001 - any capture failure is a fallback, not a crash
            if forced:
                raise
            self._uncapturable[key] = repr(exc)
            print(f"[yolov10_backbone] capture failed, running eager for "
                  f"{tuple(key[0])}/{key[1]}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            self.eager_calls += 1
            return self._eager(x)
        self._graphs[key] = entry
        self.graph_generation += 1
        return self._replay(entry, x)

    def _capture(self, x: torch.Tensor, key) -> _Entry:
        through_psa = self._route is not ROUTE_PARTIAL
        device = x.device
        # Restoring the stream is not hygiene either. ``torch.cuda.graph.__exit__`` calls
        # ``capture_end()`` *before* leaving its stream context, so when capture fails the
        # exception skips the stream restore and the process is left running on this side stream
        # for good. Measured consequence, from the induced-failure test: the next call's graph key
        # carried the side stream, so it read as a brand-new key, and the module captured again
        # against the very failure it had just recorded as permanent -- while every unrelated
        # allocation and launch in the process quietly moved to a stream the caller never chose.
        prior_stream = torch.cuda.current_stream(device)
        try:
            return self._capture_on_device(x, key, through_psa, device)
        finally:
            torch.cuda.set_stream(prior_stream)

    def _capture_on_device(self, x: torch.Tensor, key, through_psa: bool, device) -> _Entry:
        with torch.cuda.device(device):
            # A recursive hook scan and a full ``named_parameters`` walk cost 64 us and 224 us
            # respectively -- ruled out per call by that measurement, affordable exactly once,
            # and so they run here. Capture is refused outright if any hook is already registered
            # anywhere in the tree, because the eager path would fire it and replay never will.
            hooked = self._first_hooked_module()
            if hooked is not None:
                raise RuntimeError(f"a module hook is registered on {hooked!r}; the eager path "
                                   f"fires it and a replay never would")

            x_static = torch.empty(tuple(x.shape), dtype=x.dtype, device=device)
            x_static.copy_(x)

            # Warm on the stream capture runs on. ``torch.cuda.graph`` otherwise uses a
            # process-wide class-level default capture stream, so the documented
            # warm-on-a-side-stream idiom warms one stream and captures on another -- and several
            # reachable paths hold *per-stream* lazy state (cuDNN's handle, algorithm heuristics
            # and workspace, reached by down3's two-kernel route and by C2f's aten/folded
            # routes). Passing ``stream=side`` and warming on that same stream removes the
            # question for free. Correctness is identical either way; this is the cheaper
            # reasoning.
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(_WARMUP_ITERS):
                    warm = self._stages(x_static, through_psa)
            torch.cuda.current_stream(device).wait_stream(side)

            output_mode = self._output_mode
            spans, staging = (), None
            if output_mode is OUT_PACKED:
                offset, spans_list = 0, []
                for t in warm:
                    n = t.numel()
                    spans_list.append((offset, offset + n, tuple(t.shape)))
                    offset += n
                spans = tuple(spans_list)
                # Allocated *before* the capture region, so it lives in the ordinary allocator
                # rather than in the graph's private pool, and one clone of it is what the caller
                # gets. The three copies into it are inside the graph and free at replay.
                staging = torch.empty(offset, dtype=warm[0].dtype, device=device)
            del warm

            # ``torch.cuda.graph.__enter__`` synchronizes and then calls ``empty_cache()``
            # itself, so doing both here first is what makes the difference across the capture
            # attributable to the graph's private pool rather than to the cache it released.
            # Once per capture, outside any timed window, and the number is recorded rather than
            # used: nothing branches on it.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            reserved_before = torch.cuda.memory_reserved(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                outputs = self._stages(x_static, through_psa)
                if staging is not None:
                    for (a, b, _), t in zip(spans, outputs):
                        staging[a:b].copy_(t.reshape(-1))

            entry = _Entry()
            entry.key = key
            entry.graph = graph
            entry.x_static = x_static
            entry.outputs = tuple(outputs)
            entry.staging = staging
            entry.spans = spans
            entry.output_mode = output_mode
            entry.through_psa = through_psa
            entry.pool_reserved_bytes = torch.cuda.memory_reserved(device) - reserved_before
            self._snapshot_guard(entry)
            entry.keepalive = self._keepalive()
            return entry

    def _first_hooked_module(self):
        """The first module in the tree carrying any hook, or None. Capture time only."""
        for mod in self.modules():
            if (mod._forward_hooks or mod._forward_pre_hooks
                    or mod._backward_hooks or mod._backward_pre_hooks):
                return mod
        for registry in _GLOBAL_HOOKS:
            if registry:
                return "a process-wide registry"
        return None

    def _snapshot_guard(self, entry: _Entry) -> None:
        """Resolve, once, everything the per-call guard compares against.

        The slots are ``(owner dict, name)`` pairs rather than tensors, which is the whole point:
        the guard re-fetches through them and so sees a rebound parameter that a stored-object
        scan would miss. ``num_batches_tracked`` is excluded because nothing reads it in eval --
        that is 36 of the 216 tensors, leaving 180.
        """
        dicts, names = [], []
        for mod in self.modules():
            for owner in (mod._parameters, mod._buffers):
                for name, t in owner.items():
                    if t is not None and t.is_floating_point():
                        dicts.append(owner)
                        names.append(name)
        entry.slot_dicts = tuple(dicts)
        entry.slot_names = tuple(names)
        live = list(map(getitem, entry.slot_dicts, entry.slot_names))
        entry.identity = tuple(map(id, live))
        entry.versions = tuple(map(_VERSION, live))
        entry.pointers = tuple(map(_DATA_PTR, live))
        entry.contiguity = tuple(map(_IS_CONTIGUOUS, live))
        entry.dtypes = tuple(map(_DTYPE, live))
        entry.negations = tuple(map(_IS_NEG, live))

        # Module bindings, by owner dict and name, for the same reason the tensors are re-fetched
        # rather than stored: a replaced module is invisible to any scan that starts from the
        # objects capture happened to see.
        mod_dicts, mod_names = [], []
        for mod in self.modules():
            for name, child in mod._modules.items():
                if child is not None:
                    mod_dicts.append(mod._modules)
                    mod_names.append(name)
        entry.module_dicts = tuple(mod_dicts)
        entry.module_names = tuple(mod_names)
        live_mods = list(map(getitem, entry.module_dicts, entry.module_names))
        entry.module_identity = tuple(map(id, live_mods))
        entry.training_flags = tuple(map(_TRAINING, live_mods))

        entry.holders = (self.stem1, self.stem2, self.down3,
                         self.down4.cv1, self.down4.cv2,
                         self.down5.cv1, self.down5.cv2,
                         self.sppf.cv1, self.sppf.cv2)
        entry.pool = self.sppf.m
        entry.structural = self._structural_snapshot(entry)
        entry.hook_next_id = RemovableHandle.next_id

    def _keepalive(self) -> tuple:
        """Every object the captured kernels address, held opaquely.

        Held here and interpreted nowhere: the parameters and buffers themselves, and each
        child's derived state -- ``YOLOC2f._plan`` with its ``_FusedStepProgram`` (whose
        ``__del__`` destroys the C++ ``BlockPlan`` the graph baked addresses out of),
        ``YOLOSPPF._folded`` and ``YOLOPSA._folded``. Dropping ``stage3._plan`` and collecting
        between capture and replay is a real scenario -- ``_apply``, a ``load_state_dict``
        post-hook and a re-fold all do it -- and without these references the replay would run
        against recycled memory. With them, the folded tensors and the C++ plan stay at the
        addresses the graph baked in, holding the values they held, so the replay stays correct.
        """
        held = [tuple(self.parameters()), tuple(self.buffers())]
        for stage in (self.stage2, self.stage3, self.stage4, self.stage5):
            plan = getattr(stage, "_plan", None)
            if plan is not None:
                # The plan object itself, because ``_FusedStepProgram.__del__`` destroys the C++
                # ``BlockPlan`` whose addresses the graph baked in; and separately every tensor
                # inside the plan and its stages, because those slots are mutable and holding the
                # container would not stop a rebinding from freeing what it held.
                held.append(plan)
                held.append(getattr(plan, "fused", None))
                held.append(tuple(plan.stages))
                held.append(_derived_tensors(plan))
                for st in plan.stages:
                    held.append(_derived_tensors(st))
        for owner in (self.sppf, self.psa):
            folded = getattr(owner, "_folded", None)
            if folded is not None:
                held.append(folded)
                held.append(_derived_tensors(folded))
                if isinstance(folded, (tuple, list)):
                    for item in folded:
                        held.append(item if isinstance(item, (tuple, list))
                                    else _derived_tensors(item))
        return tuple(held)
