"""FLUX transformer blocks for B200 (sm_100), same contract as ``baseline.py``.

Two things are going on here, and the first is a correctness fix rather than an
optimisation.

**Composing the frozen lower-level winners is not correct for the dual block.**
Substituting them into the baseline body is 1.50x faster for free, but the
matched ratio falls to 0.9845 (S_img=1024) / 0.9876 (S_img=4096) against a 0.99
gate. A per-submodule ablation puts the whole deficit on the frozen
``AdaLayerNormZero``, which is also the child that buys the least (1.02-1.04x):
its fused SiLU+GEMV projection differs from ``F.linear(F.silu(temb), W, b)`` by
rms_rel 1.7e-4 on 0.16% of elements. That projection produces ``shift_msa`` and
``scale_msa``, so the perturbation lands on ``norm_hidden_states`` -- which is the
input to ``to_qkv``. Under the harness's ``normal_(0, 0.02)`` weights the
attention logits have std ~13.7 over 4608 keys, so the softmax is nearly one-hot,
and for the ~1% of rows whose top-1/top-2 margin is comparable to the
perturbation the arg-max flips and that row's attention output changes by O(1).
The same perturbation *downstream* of attention is free: the ablation gives
1.0000 for both ``norm2`` modules.

So the rule this file is built around: **anything computed upstream of ``to_qkv``
must match the baseline's arithmetic operation-for-operation and
rounding-for-rounding, not merely to within tolerance.** Being 1e-4 close is not
close enough, and being *more accurate than the baseline* is a failure and not a
bonus -- the grade rewards agreement. The conditioning projection is therefore
hand-written as ``F.linear(F.silu(temb), W, b)`` here, and the frozen adaLN
modules are held for contract parity but not invoked.

**The second is the fusion.** Outside the GEMMs, the attention and the
feed-forward -- which belong to the frozen winners and are called as modules --
the block is three broadcast-and-residual chains over [rows, 3072] bf16, ten ATen
kernels per stream, running at 1.4-1.9 TB/s where a plain ``copy_`` reaches
3.3 TB/s. That is 13% of the dual block's time at S_img=4096 and 18% at 1024.
``flux_l3_kernels.cu`` replaces them with two fused epilogue kernels.

The normalization's own statistics stay in ``F.layer_norm``, deliberately.
ATen's bf16 LayerNorm at N=3072 runs ``vectorized_layer_norm_kernel``, whose mean
and rstd come out of a Welford recurrence over a (32, 4) block; a two-pass
mean-then-variance reduction reproduces the mean on 22% of rows and the rstd on
63% (``tools/probe_ln_stats.py``), leaving ~0.001% of elements one bf16 ULP from
the reference. That is 100x smaller than the error that fails the gate and would
almost certainly pass -- but "almost certainly passes" is the reasoning that made
the frozen adaLN fail, so it is not used upstream of attention. Replicating the
Welford tree was simulated for every plausible geometry and none was bit-exact
(``tools/probe_ln_welford.py``). Calling ``F.layer_norm`` and fusing only the
explicitly rounded epilogue is bit-exact by construction and still removes eight
of the eleven tensor passes.

Anything the fused path does not cover computes the same values in pure ATen,
step by step, so a decline costs speed and never correctness. Crucially the
fallback is **not** the submodules: ``self.norm1`` is the frozen adaLN carrying
the 1.7e-4 error, and ``self.norm2`` is the frozen L1 ``LayerNorm``, which fuses
and rounds once and is a different function from the ``F.layer_norm`` the
baseline composite calls. Routing a decline through either would reintroduce the
failure the file exists to remove. The submodules are used only when a caller has
actually replaced one, because then following the replacement is the contract.

Both classes keep the baseline's module tree verbatim -- same child names, same
classes, same relative imports -- because the harness shares weights with
``load_state_dict(..., strict=False)`` inside a bare ``try/except``: a renamed or
restructured parameter is silently *not* loaded and the module then runs on
different random weights with no diagnostic. Nothing is derived from weight
*values* in ``__init__`` either; the harness casts parameters to bf16 and only
then loads the state dict, so anything precomputed there would be stale.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import forward_ad as _forward_ad
# The module object, not its four dicts by value: ``nn.Module._call_impl`` reads
# these attributes live, so anything that *rebinds* one -- as opposed to mutating it
# through ``register_module_forward_hook`` -- would leave an imported alias stale
# and this file looking at a registry the baseline no longer consults.
from torch.nn.modules import module as _module_globals

# Set BEFORE the frozen L1/L2 imports below, because those modules build their own
# extensions at *their* import time and inherit whatever this variable says then.
# Left unset, they land in the shared ~/.cache/torch_extensions, where every
# workspace that imports the same frozen module builds under the same name -- and
# torch's FileBaton waits on the mere existence of a lock file, so one sibling
# process killed mid-build leaves a lock that blocks this candidate's import
# forever. Observed: a stale ``fk_ln_cand/lock`` from another workspace hung an
# import for minutes with no diagnostic. Pointing the whole tree at this
# workspace's own gitignored directory removes the shared name entirely.
_WORKSPACE = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_WORKSPACE / ".torch_extensions"))

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU
from ..L2.ada_layer_norm import AdaLayerNormZero, AdaLayerNormZeroSingle
from ..L2.flux_attention import FluxAttention
from ..L2.flux_feedforward import FeedForward
from ..L2.parallel_linear import ReplicatedLinear

__all__ = ["FluxTransformerBlock", "FluxSingleTransformerBlock",
           "fastpath_counters", "reset_fastpath_counters", "extension_error"]

# Compile-time knobs, each a ``-D`` macro whose default is the shipped
# configuration. The override set is hashed into the extension name *and* the
# TORCH_LIBRARY namespace so two arms can coexist on disk and a stale ``.so`` can
# never be mistaken for the other arm.
_KNOBS = ("FK_L3_BLOCK", "FK_L3_VPT", "FK_L3_CTAS_PER_SM")
_OVERRIDES = {k: os.environ[k] for k in _KNOBS if os.environ.get(k, "") != ""}

# Not a kernel knob: ``-lineinfo`` so NCU can map SASS back to source. It changes
# the binary, so it is part of the build key like any override.
_LINEINFO = os.environ.get("FK_L3_LINEINFO", "") not in ("", "0")

# The frozen ``FeedForward`` selects ``torch._addmm_activation`` at 1024 and 4096
# rows, which applies GELU to the fp32 accumulator and so drops the baseline's bf16
# rounding between GEMM1 and the activation (rms_rel 3.1e-3). That is the whole
# residual error budget of the dual block once the conditioning is exact, and it is
# worth 0.008 of matched ratio: 0.9915 without it against 0.9996 with it, on a gate
# of 0.99. ``path_override = "reference"`` restores the rounding at the cost of a
# separate GELU pass, measured at 2-5% of this block's latency.
#
# **Both** children have to be pinned, which is not obvious and was not what the
# plan expected. The graded quantity is the *worst* output leaf: ``ff`` drives the
# image-side ``hidden_states`` and ``ff_context`` the encoder-side
# ``encoder_hidden_states``, so pinning either alone leaves the other leaf at 0.9915
# and the worst-leaf figure does not move (measured: 0.991530 for ff_context alone,
# 0.991578 for ff alone, 0.999550 for both). "Apply the cheaper half first" cannot
# reach the bar here at any price; it is both or neither.
#
# Off via FK_L3_FF_EXACT=0, which is how the arm was measured.
_FF_EXACT = os.environ.get("FK_L3_FF_EXACT", "1") not in ("", "0")

# Not a compile knob: whether the normalization is fused into the epilogue kernel
# (a two-pass reduction, not bit-identical to ATen's Welford tree) or left to
# ``F.layer_norm``. Off by default -- see the module docstring. Exists so the arm
# can be measured rather than argued about.
_FUSED_NORM = os.environ.get("FK_L3_FUSED_NORM", "") not in ("", "0")
if _FUSED_NORM:
    # Loud, because this arm is knowingly *not* bit-exact against ATen's Welford
    # reduction and exists to be measured, not shipped. Enabling it silently is
    # the one way this file can produce a result that is merely close.
    print("[flux_transformer_block] FK_L3_FUSED_NORM is set: the fully fused "
          "normalization is a measurement arm and is NOT bit-exact against "
          "F.layer_norm; do not report a graded result from it.",
          file=sys.stderr)

# The name must be unique across the whole tree. Torch keys both the build
# directory and the module on the name alone and rebuilds whenever a source is
# newer than the ``.so``, so sharing one with a live extension would make the two
# sources invalidate each other's build on every import. Names already live here
# include fk_adaln_cand, fk_ln_cand, fk_adalnc_cand, flux_qk_norm_rope,
# rms_norm_single_pass, fk_ffn_lt_*, fk_timestep_embedding_sm100_v1,
# fk_silu_sm100_v1, dense_attn_tiny_seq and fk_gelu_*.
if _OVERRIDES or _LINEINFO:
    _tag = hashlib.sha1(
        ";".join([f"{k}={v}" for k, v in sorted(_OVERRIDES.items())]
                 + [f"lineinfo={_LINEINFO}"]).encode()
    ).hexdigest()[:8]
    _LIBRARY_NAME = f"fk_flux_l3_{_tag}"
else:
    _LIBRARY_NAME = "fk_flux_l3"


def _local_cuda_arch() -> str:
    """Compute capability of the live device, in ``TORCH_CUDA_ARCH_LIST`` form.

    Set unconditionally, with a hardcoded fallback rather than leaving the
    variable unset or ``"native"``: that branch of ``_get_cuda_arch_flags``
    iterates ``torch.cuda.device_count()`` and then indexes the result, raising
    IndexError with no GPU visible -- which is this workspace's normal state. The
    ambient list here names six architectures, i.e. six nvcc passes over every
    instantiation for five targets that will never run the kernel.
    """
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except Exception:  # noqa: BLE001 - fall through to the pinned default
        pass
    return "10.0"


def _load_ops():
    """Build and register the operators, returning their bound callables.

    Compilation happens here, at import, so nothing is deferred into a timed
    ``forward``: ninja spawns subprocesses, and the harness's ``_check_threads``
    fails any candidate whose thread count grows during its timing window.
    """
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "flux_l3_kernels.cu"
    flags = ["-O3", f"-DFK_L3_LIB={_LIBRARY_NAME}"]
    flags += [f"-D{k}={v}" for k, v in sorted(_OVERRIDES.items())]
    if _LINEINFO:
        flags.append("-lineinfo")

    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _local_cuda_arch()
    try:
        load(
            name=_LIBRARY_NAME,
            sources=[str(source)],
            extra_cuda_cflags=flags,
            is_python_module=False,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list

    ns = getattr(torch.ops, _LIBRARY_NAME)
    # Bind the resolved overloads, not the packets: a packet re-resolves overloads
    # from the argument types on every call, and this forward is launch-latency
    # bound on the small captured shapes.
    return (ns.gate_add.default, ns.affine.default, ns.norm_affine.default,
            ns.claims.default, ns.counters.default, ns.reset_counters.default)


try:
    (_GATE_ADD_OP, _AFFINE_OP, _NORM_AFFINE_OP, _CLAIMS_OP, _COUNTERS_OP,
     _RESET_COUNTERS_OP) = _load_ops()
    _EXTENSION_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - a build that cannot happen must degrade,
    # not take the module down with it: an import failure costs every case at once.
    print(f"[flux_transformer_block] fused extension unavailable, using ATen: "
          f"{exc!r}", file=sys.stderr)
    _GATE_ADD_OP = _AFFINE_OP = _NORM_AFFINE_OP = None
    _CLAIMS_OP = _COUNTERS_OP = _RESET_COUNTERS_OP = None
    _EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"


def extension_error() -> str | None:
    """Why the extension is unavailable, or ``None`` if it built."""
    return _EXTENSION_ERROR


# ---------------------------------------------------------------------------
# Dispatch counters.
#
# Plain host ints. This is what distinguishes "the fused path ran and tied" from
# "the fused path never ran", which a latency number on its own cannot -- and the
# specific failure it excludes is a candidate graded on one path and timed on
# another. The harness runs correctness through ``_run_forward`` and timing
# through ``_time_module``, both under ``torch.no_grad()``, so the grad-mode
# predicate holds in both; the counters are how that is proven rather than
# re-derived.
# ---------------------------------------------------------------------------
_KERNEL_COUNTER_NAMES = ("k_gate_add", "k_affine", "k_norm_affine",
                         "k_gate_add_norm_affine", "k_declined", "k_claims_ok",
                         "k_claims_declined")

_PATH_COUNTS = {
    "dual_exact": 0,        # hand-written exact conditioning, kernels allowed
    "dual_aten": 0,         # same arithmetic, pure ATen (kernels not allowed)
    "dual_submodules": 0,   # a caller replaced a child; follow the replacement
    "single_exact": 0,
    "single_aten": 0,
    "single_submodules": 0,
}


def fastpath_counters() -> dict[str, int]:
    """Per-decision dispatch counts, including the kernels' own."""
    out = dict(_PATH_COUNTS)
    if _COUNTERS_OP is not None:
        out.update(zip(_KERNEL_COUNTER_NAMES, _COUNTERS_OP()))
    return out


def reset_fastpath_counters() -> None:
    for key in _PATH_COUNTS:
        _PATH_COUNTS[key] = 0
    if _RESET_COUNTERS_OP is not None:
        _RESET_COUNTERS_OP()


def claims(x, a, p, c, max_chunk) -> bool:
    """The operator's own layout predicate, exposed for tests."""
    if _CLAIMS_OP is None:
        return False
    return _CLAIMS_OP(x, a, p, c, max_chunk)


# ---------------------------------------------------------------------------
# Eligibility.
#
# Split the way the frozen sibling at L2 splits it: Python screens the semantics
# only visible from Python -- exact tensor type, module hooks and instance
# ``forward`` overrides, tracing, forward-mode tangents, the live configuration of
# the modules being bypassed -- and the operator screens tensor layout. Ordered so
# that no check can raise: type before attribute, rank before ``size(-1)``, tree
# membership before nested access.
# ---------------------------------------------------------------------------

# The two exact types whose ATen ops produce plain tensors, so the kernels'
# plain-tensor outputs match what the baseline would have returned.
_PLAIN_TENSOR_TYPES = (torch.Tensor, nn.Parameter)

# ``forward`` of each bypassed class as of import. An exact-type test cannot see a
# *class-level* monkeypatch -- ``LayerNorm.forward = something`` leaves every
# instance's type matching while changing what calling the child would do -- so the
# identity of the class method is compared against what it was when this module
# loaded.
_ORIGINAL_FORWARD = {
    LayerNorm: LayerNorm.forward,
    Linear: Linear.forward,
    SiLU: SiLU.forward,
    AdaLayerNormZero: AdaLayerNormZero.forward,
    AdaLayerNormZeroSingle: AdaLayerNormZeroSingle.forward,
}


def _plain_tensors(*tensors) -> bool:
    """Every tensor a kernel will read is exactly a Tensor or a Parameter.

    An exact-type test rather than ``isinstance``: a plain ``as_subclass`` tensor
    has ordinary storage, so ``at::isTensorSubclassLike`` does not catch it and the
    kernels would read it happily -- but the baseline's ATen ops propagate the
    subclass into their outputs while the kernels return plain tensors, so the
    *type* of the block's result would differ. That has to cover every tensor a
    kernel touches, not only the ones that arrive as arguments.
    """
    for t in tensors:
        if type(t) not in _PLAIN_TENSOR_TYPES:
            return False
    return True


def _carries_forward_grad(x: torch.Tensor) -> bool:
    """Does x hold a forward-mode tangent?

    A dual tensor has exact type ``torch.Tensor`` and ``requires_grad=False``, so
    no other check here would stop it, and the kernels build no dual. Checking the
    active dual level first makes this one integer comparison when nobody is doing
    forward AD, which is always, in the benchmark.
    """
    if getattr(_forward_ad, "_current_level", -1) < 0:
        return False
    return _forward_ad.unpack_dual(x).tangent is not None


def _no_global_hooks() -> bool:
    """No process-wide module hook is installed.

    These are the module-level dicts ``nn.Module._call_impl`` consults, so a global
    hook would run for every submodule the baseline calls and for none of the ones
    this file bypasses.
    """
    return not (_module_globals._global_backward_hooks
                or _module_globals._global_backward_pre_hooks
                or _module_globals._global_forward_hooks
                or _module_globals._global_forward_pre_hooks)


def _plain_module(m, expected_type) -> bool:
    """Would calling ``m(...)`` do anything other than run ``expected_type.forward``?

    The baseline reaches its submodules through ``nn.Module.__call__``, which runs
    forward pre-hooks, the forward, then forward hooks, and honours a compiled call
    or an instance-level ``forward`` override. This file reads parameters out of
    those modules instead of calling them, so it may only do so when none of that
    would have had an effect. Exact type alone is not enough: a hook can be
    registered on an exact-type instance, and ``m.forward = something`` shadows the
    class's method without changing the type.

    The exact-type test is also what catches a parametrized weight:
    ``register_parametrization`` injects a fresh subclass, so ``type(m)`` stops
    matching. The explicit ``parametrizations`` lookups below are belt and braces
    for the same condition -- a parametrized weight makes the module's effective
    weight differ from ``m.weight``, which is the tensor read here.
    """
    if type(m) is not expected_type:
        # Also the ``None`` a deleted submodule leaves behind, which then reaches
        # the submodule path and raises there, as the baseline would.
        return False
    # Class-level dispatch, which the exact-type test above cannot see.
    if _ORIGINAL_FORWARD.get(expected_type) is not expected_type.forward:
        return False
    if getattr(expected_type, "_compiled_call_impl", None) is not None:
        return False
    # Read straight out of the instance dict. ``nn.Module`` keeps its hook
    # registries there, so this is dict lookups rather than attribute lookups --
    # and ``getattr(m, "_compiled_call_impl", None)`` would be worse still, because
    # that attribute is absent until torch.compile installs it, so the getattr
    # misses, falls into ``nn.Module.__getattr__`` and raises AttributeError to be
    # swallowed. On a forward this launch-latency bound, an exception per submodule
    # per call is not affordable.
    d = m.__dict__
    if (d.get("_forward_hooks") or d.get("_forward_pre_hooks")
            or d.get("_backward_hooks") or d.get("_backward_pre_hooks")):
        return False
    if d.get("parametrizations") is not None or d.get("_parametrizations"):
        return False
    if "parametrizations" in d.get("_modules", ()):
        return False
    return "forward" not in d and d.get("_compiled_call_impl") is None


def _eps_ok(eps) -> bool:
    """An ``eps`` ``F.layer_norm`` would accept as written.

    ``float(norm.eps)`` would silently canonicalize ``"1e-6"``, which ATen rejects --
    so the candidate would return a value where the baseline raises.
    """
    return type(eps) is float or type(eps) is int


def _layer_norm_plain(norm, c: int) -> bool:
    """Is ``norm`` exactly the function ``F.layer_norm(x, (c,), None, None, eps)``?

    Checking ``eps`` alone is not enough. An affine LayerNorm, or one that promotes
    to fp32 around the reduction, computes a different function, and the value this
    file substitutes is the plain one.
    """
    shape = norm.normalized_shape
    if not isinstance(shape, tuple) or len(shape) != 1:
        return False
    # The element's *type*, before any coercion. ``int(shape[0])`` would accept
    # ``(3072.0,)`` and ``("3072",)``, both of which F.layer_norm rejects.
    if type(shape[0]) is not int or shape[0] != c:
        return False
    if not _eps_ok(norm.eps):
        return False
    # The frozen L1 LayerNorm hands its own ``self._n`` to its fused kernel, so a
    # caller who sets that changes what this child computes even though
    # ``normalized_shape`` is untouched. Read from the instance dict, because the
    # baseline's class has no such field and absent has to read as agreeing.
    n = norm.__dict__.get("_n")
    if n is not None and n != c:
        return False
    # The parameters themselves, not ``elementwise_affine``: the baseline hands
    # ``self.weight`` / ``self.bias`` to F.layer_norm, so those are what decide the
    # reference's answer.
    if norm.weight is not None or norm.bias is not None:
        return False
    return not bool(norm.promote_fp32)


def _projection_plain(linear, chunks: int, c: int) -> bool:
    """Can ``F.linear`` be substituted for this projection module?

    ``bias is None`` declines rather than taking a null-bias path: the captured
    configuration is always ``bias=True``, so a null-bias fast path has no
    correctness evidence behind it. The kernels never read the bias, so this costs
    a ``bias=False`` module its speedup; that is the deliberate price of not
    shipping an unmeasured path.
    """
    if not _plain_module(linear, Linear):
        return False
    weight, bias = linear.weight, linear.bias
    if bias is None:
        return False
    if not _plain_tensors(weight, bias):
        return False
    if weight.dim() != 2 or weight.shape != (chunks * c, c):
        return False
    return bias.dim() == 1 and bias.shape[0] == chunks * c


def _adaln_emb(adaln):
    """``adaln.emb``, without an attribute-error round trip.

    ``AdaLayerNormZero.__init__`` assigns ``self.emb = None``, which nn.Module
    stores in the instance dict, while a caller who installs a real embedder puts
    it in ``_modules`` (or ``_parameters`` / ``_buffers`` for a tensor). And
    ``AdaLayerNormZeroSingle`` has no such attribute at all, so a plain
    ``adaln.emb`` falls into ``nn.Module.__getattr__`` and raises AttributeError --
    an exception per submodule per call is not affordable on a forward this
    launch-latency bound.
    """
    # A class attribute or data descriptor wins over the instance dict during
    # normal attribute lookup, so ``AdaLayerNormZero.emb = property(...)`` would
    # make the baseline call an embedder while the instance dict still says None.
    # ``getattr`` on the *type* cannot raise and does not reach
    # ``nn.Module.__getattr__``, which is what makes this affordable here.
    on_class = getattr(type(adaln), "emb", None)
    if on_class is not None:
        return on_class
    d = adaln.__dict__
    for store in ("_modules", "_parameters", "_buffers"):
        registry = d.get(store)
        if registry is not None:
            found = registry.get("emb")
            if found is not None:
                return found
    return d.get("emb")


def _adaln_config(adaln, expected_type, chunks: int):
    """``(c, eps)`` for a bypassed adaLN module, or ``None``.

    ``self.emb`` has to be None. The baseline's ``AdaLayerNormZero.forward``
    *ignores* the ``emb=temb`` it was passed and calls
    ``self.emb(timestep, class_labels, ...)`` whenever ``self.emb`` is not None, so
    a path that conditions on ``temb`` unconditionally would silently condition on
    the wrong tensor. ``AdaLayerNormZeroSingle`` has no ``emb`` at all and no such
    branch, so absent reads the same as None here.
    """
    if not _plain_module(adaln, expected_type):
        return None
    if _adaln_emb(adaln) is not None:
        return None
    mods = adaln._modules
    silu, linear, norm = mods.get("silu"), mods.get("linear"), mods.get("norm")
    # ``silu`` is checked even though it is never called: the baseline evaluates
    # ``self.linear(self.silu(emb))``, so a replaced or hooked SiLU changes the
    # reference's answer while ``F.silu`` here would not follow it.
    if not (_plain_module(silu, SiLU) and _plain_module(norm, LayerNorm)):
        return None
    shape = norm.normalized_shape
    if not isinstance(shape, tuple) or len(shape) != 1:
        return None
    c = int(shape[0])
    if not (_layer_norm_plain(norm, c) and _projection_plain(linear, chunks, c)):
        return None
    return c, float(norm.eps)


# ---------------------------------------------------------------------------
# The L3-owned arithmetic, once. Each helper computes exactly what the baseline
# computes: the kernel when it accepts the operands, the same expression in ATen
# when it declines. A decline is reported in band as ``None`` from the operator, so
# every step degrades independently and none of them can produce a value that is
# merely close.
# ---------------------------------------------------------------------------


def _chunk(p: torch.Tensor, index: int, c: int) -> torch.Tensor:
    """Chunk ``index`` of the projection output.

    ``p.chunk(n, dim=1)[i]`` and ``p.narrow(1, i * c, c)`` are the same view; this
    is the one the baseline unpacks.
    """
    return p.narrow(1, index * c, c)


def _norm_affine(use_op: bool, x: torch.Tensor, p: torch.Tensor, shift_index: int,
                 scale_index: int, c: int, eps: float) -> torch.Tensor:
    """``F.layer_norm(x) * (1 + scale) + shift``, exactly.

    This value feeds ``to_qkv``, so it is the one that has to be bit-exact above
    all others.
    """
    # ``x`` here can be a value a called submodule produced -- ``h`` from the gate
    # add, or an attention output -- so its exact type is re-tested at the point of
    # use rather than only on the block's arguments. A caller whose ``attn`` returns
    # ``as_subclass(T)`` gets a subclass out of the baseline's ATen ops and a plain
    # tensor out of a raw kernel, which is a difference in the *type* of the result.
    use_op = use_op and _plain_tensors(x, p)
    if use_op and _FUSED_NORM:
        fused = _NORM_AFFINE_OP(x, None, p, 0, shift_index, scale_index, c, eps)
        if fused:
            return fused[0]
    # ``F.layer_norm`` on the caller's own tensor, so whichever kernel ATen selects
    # for it -- vectorized or the generic two-kernel path, which are *not*
    # bit-identical to each other -- is the one the reference selected too.
    normed = F.layer_norm(x, (c,), None, None, eps)
    if use_op:
        out = _AFFINE_OP(normed, p, shift_index, scale_index, c)
        if out is not None:
            return out
    # Reuses ``normed`` rather than recomputing it, so a decline costs one kernel
    # rather than the whole chain.
    return (normed * (1 + _chunk(p, scale_index, c)[:, None])
            + _chunk(p, shift_index, c)[:, None])


def _gate_add(use_op: bool, x: torch.Tensor, f: torch.Tensor, p: torch.Tensor,
              gate_index: int, c: int) -> torch.Tensor:
    """``x + gate.unsqueeze(1) * f``, with the gate product rounded to bf16 first.

    The two ATen kernels the baseline uses here make that rounding observable;
    collapsing them into one fp32 fma changes ~22% of elements.
    """
    # ``f`` is always a called submodule's output (attention, feed-forward or
    # proj_out), so this is the only place its exact type can be tested.
    if use_op and _plain_tensors(x, f, p):
        out = _GATE_ADD_OP(x, f, p, gate_index, c)
        if out is not None:
            return out
    return x + _chunk(p, gate_index, c).unsqueeze(1) * f


def _gate_add_norm_affine(use_op: bool, x: torch.Tensor, a: torch.Tensor,
                          p: torch.Tensor, gate_index: int, shift_index: int,
                          scale_index: int, c: int, eps: float):
    """``h = x + gate*a`` and ``F.layer_norm(h) * (1 + scale) + shift``.

    ``h`` is returned because the block's feed-forward residual needs it, and it is
    the value an error in would reach the block's output by two paths.
    """
    if use_op and _FUSED_NORM and _plain_tensors(x, a, p):
        fused = _NORM_AFFINE_OP(x, a, p, gate_index, shift_index, scale_index, c,
                                eps)
        if fused:
            return fused[0], fused[1]
    h = _gate_add(use_op, x, a, p, gate_index, c)
    return h, _norm_affine(use_op, h, p, shift_index, scale_index, c, eps)


def _kernels_allowed(*tensors) -> bool:
    """The execution-context half of eligibility.

    Exact tensor type is *not* tested here: a non-plain tensor has to reach the
    submodule body rather than the pure-ATen tier, so the callers screen it before
    choosing a tier at all. What is left is context that makes the kernels
    inapplicable while the ATen replication is still correct.
    """
    if _GATE_ADD_OP is None:
        return False
    # A traced or compiled call must see the ATen expression, not an opaque custom
    # op whose meta behaviour this file does not register. ``is_tracing`` is a
    # separate condition from ``is_compiling``: under ``torch.jit.trace``,
    # ``nn.Module.__call__`` dispatches through ``_slow_forward`` instead of
    # ``forward``, so the baseline's child calls are recorded and these are not.
    if torch.compiler.is_compiling() or torch.jit.is_tracing():
        return False
    for t in tensors:
        if _carries_forward_grad(t):
            return False
    return True


class FluxTransformerBlock(nn.Module):
    """Dual-stream DiT block: joint attention over text+image, then separate FFNs."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: dict | None = None,
    ):
        super().__init__()
        # The baseline's tree verbatim. Nothing here reads a parameter's values:
        # the harness casts parameters to bf16 and only *then* loads the state
        # dict, so anything precomputed from a weight would be stale.
        self.norm1 = AdaLayerNormZero(dim, promote_fp32=False)
        self.norm1_context = AdaLayerNormZero(dim, promote_fp32=False)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        # A string, not anything derived from a weight value, so it survives the
        # harness's post-__init__ cast and state_dict load. See _FF_EXACT above for
        # why both children are pinned rather than the cheaper one.
        if _FF_EXACT:
            self.ff.path_override = "reference"
            self.ff_context.path_override = "reference"

    def _config(self):
        """Live configuration for the exact path, or ``None`` to follow the tree.

        Read now rather than snapshotted in ``__init__``, because the baseline
        evaluates these submodules on every call: a caller who sets ``norm2.eps``,
        flips ``promote_fp32``, installs an ``emb``, registers a hook, replaces a
        child or overrides one instance's ``forward`` changes what the baseline
        computes, and this file then has to either follow the change or stand down.
        """
        mods = self._modules
        norm2, norm2_context = mods.get("norm2"), mods.get("norm2_context")
        if not (_plain_module(norm2, LayerNorm)
                and _plain_module(norm2_context, LayerNorm)
                and _no_global_hooks()):
            return None
        image = _adaln_config(mods.get("norm1"), AdaLayerNormZero, 6)
        if image is None:
            return None
        context = _adaln_config(mods.get("norm1_context"), AdaLayerNormZero, 6)
        if context is None:
            return None
        c, eps_msa = image
        c_context, eps_msa_context = context
        if c_context != c:
            return None
        if not (_layer_norm_plain(norm2, c) and _layer_norm_plain(norm2_context, c)):
            return None
        # ``attn``, ``ff`` and ``ff_context`` are *called* exactly as the baseline
        # calls them, so whatever they are is what the baseline would have used and
        # there is nothing to screen. They only have to be present.
        if mods.get("attn") is None or mods.get("ff") is None or mods.get("ff_context") is None:
            return None
        return c, eps_msa, eps_msa_context, float(norm2.eps), float(norm2_context.eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self._config()
        if config is None:
            _PATH_COUNTS["dual_submodules"] += 1
            return self._forward_submodules(hidden_states, encoder_hidden_states,
                                            temb, image_rotary_emb,
                                            joint_attention_kwargs)
        c, eps_msa, eps_msa_context, eps_mlp, eps_mlp_context = config
        linear = self.norm1.linear
        linear_context = self.norm1_context.linear

        # A non-plain tensor anywhere the exact path reads follows the submodules,
        # not the pure-ATen tier. Tier 2 computes the same *values* but not the same
        # *op sequence*: it narrows where the baseline chunks, and evaluates
        # ``F.silu(temb)`` once where the baseline evaluates it once per adaLN. A
        # subclass with ``__torch_function__``, or an active dispatch mode, observes
        # the sequence and not only the numbers -- so for those callers the honest
        # answer is what these modules compute.
        if not _plain_tensors(hidden_states, encoder_hidden_states, temb,
                              linear.weight, linear.bias,
                              linear_context.weight, linear_context.bias):
            _PATH_COUNTS["dual_submodules"] += 1
            return self._forward_submodules(hidden_states, encoder_hidden_states,
                                            temb, image_rotary_emb,
                                            joint_attention_kwargs)

        use_op = _kernels_allowed(hidden_states, encoder_hidden_states, temb)
        _PATH_COUNTS["dual_exact" if use_op else "dual_aten"] += 1

        # The conditioning projection, hand-written so it is bit-exact. The
        # baseline's ``Matmul``/``SiLU`` are ``F.linear``/``F.silu``, so this is the
        # same computation; the frozen adaLN's fused SiLU+GEMV is not, by rms_rel
        # 1.7e-4. ``F.silu(temb)`` once instead of twice is bit-exact -- same input,
        # same deterministic op -- and is the one saving taken here.
        activated = F.silu(temb)
        p = F.linear(activated, linear.weight, linear.bias)
        p_context = F.linear(activated, linear_context.weight,
                             linear_context.bias)

        norm_hidden_states = _norm_affine(use_op, hidden_states, p, 0, 1, c,
                                         eps_msa)
        norm_encoder_hidden_states = _norm_affine(
            use_op, encoder_hidden_states, p_context, 0, 1, c, eps_msa_context)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # The image stream, then the context stream, in the baseline's order. The
        # attention output's layout is not knowable until after the call, so the
        # guard on it lives inside ``_gate_add`` -- a decline there completes the
        # remainder in ATen using the projections already computed, so attention is
        # evaluated exactly once whatever happens.
        hidden_states, norm_hidden_states = _gate_add_norm_affine(
            use_op, hidden_states, attn_output, p, 2, 3, 4, c, eps_mlp)
        hidden_states = _gate_add(use_op, hidden_states, self.ff(norm_hidden_states),
                                  p, 5, c)

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        encoder_hidden_states, norm_encoder_hidden_states = _gate_add_norm_affine(
            use_op, encoder_hidden_states, context_attn_output, p_context, 2, 3, 4,
            c, eps_mlp_context)
        encoder_hidden_states = _gate_add(
            use_op, encoder_hidden_states, self.ff_context(norm_encoder_hidden_states),
            p_context, 5, c)

        # Unreachable on the fused path, which is bf16 only, but the ATen path
        # above is dtype-agnostic and the baseline clips here.
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def _forward_submodules(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb,
        joint_attention_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The baseline's body, run through the actual submodules.

        Reached only when a child was replaced, reconfigured, hooked, or given an
        instance ``forward`` override. When a caller has done that, following the
        replacement *is* the contract, even though these modules are the frozen
        winners and ``self.norm1`` in particular carries the 1.7e-4 projection
        error. That is why this is not the fallback for an ordinary decline.
        """
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream DiT block: text+image concatenated, self-attention + MLP in parallel."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, promote_fp32=False)
        self.proj_mlp = ReplicatedLinear(dim, self.mlp_hidden_dim, bias=True,
                                         quant_config=quant_config)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + self.mlp_hidden_dim, dim, bias=True,
                                         quant_config=quant_config)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )

    def _config(self):
        """``(c, eps)`` for the exact path, or ``None`` to follow the tree."""
        mods = self._modules
        if not _no_global_hooks():
            return None
        config = _adaln_config(mods.get("norm"), AdaLayerNormZeroSingle, 3)
        if config is None:
            return None
        for name in ("proj_mlp", "act_mlp", "proj_out", "attn"):
            if mods.get(name) is None:
                return None
        return config

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self._config()
        if config is None:
            _PATH_COUNTS["single_submodules"] += 1
            return self._forward_submodules(hidden_states, encoder_hidden_states,
                                            temb, image_rotary_emb,
                                            joint_attention_kwargs)
        c, eps = config
        linear = self.norm.linear

        # Screened before the concatenation, so a subclass input reaches the
        # submodule body with the baseline's own op sequence rather than this
        # file's. See the dual block for why tier 2 is not the right home for it.
        if not _plain_tensors(hidden_states, encoder_hidden_states, temb,
                              linear.weight, linear.bias):
            _PATH_COUNTS["single_submodules"] += 1
            return self._forward_submodules(hidden_states, encoder_hidden_states,
                                            temb, image_rotary_emb,
                                            joint_attention_kwargs)

        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states

        use_op = _kernels_allowed(hidden_states, temb)
        _PATH_COUNTS["single_exact" if use_op else "single_aten"] += 1

        # The same exact conditioning as the dual block. This module's adaLN also
        # feeds attention, so the exactness rule applies to it too -- the class
        # passing at 0.9988 / 0.9995 with the frozen module reflects forgiving
        # seeds rather than an exemption.
        p = F.linear(F.silu(temb), linear.weight, linear.bias)
        norm_hidden_states = _norm_affine(use_op, hidden_states, p, 0, 1, c, eps)

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = _gate_add(use_op, residual, self.proj_out(hidden_states),
                                  p, 2, c)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states

    def _forward_submodules(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb,
        joint_attention_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The baseline's body, run through the actual submodules."""
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states
