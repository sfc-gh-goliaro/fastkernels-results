"""CLIP text-encoder self-attention for B200 / sm_100.

The captured problem is tiny -- ``[1, 77, 768]`` fp32, 12 heads of 64, about 0.4 GFLOP --
and the baseline spends **10 kernel launches** on it: four projection GEMMs, two batched
GEMMs for QK^T and PV, two elementwise kernels for the scale and the mask add, the
softmax, and the copy behind ``transpose(1, 2).contiguous()``. Measured in the bench's
own timing loop (``profile/00-numerics/probe_timing.py``), the module costs 102.40 us
while an ``identity`` module costs 12.26 us, and the ten kernels account for only 64.0 us
of GPU time. Every launch bills roughly 3 us of GPU timeline plus roughly 2 us of exposed
host gap no matter how little work it does -- the ``* scale`` kernel touches 285 KB and
still costs 3.6 us. So the lever here is launch count, not arithmetic.

This file runs the operator in three launches:

    1. one fused QKV projection through cuBLAS, on ``cat([W_q, W_k, W_v])``
    2. one Triton kernel covering QK^T, the scale, the mask add, the fp32 softmax, PV,
       and the transposed store
    3. the output projection through cuBLAS, the call the baseline makes

Numerically the gate is *pinned*, not merely bounded. The bench compares against the
baseline with ``atol=1e-5 + rtol=1e-3*|ref|`` and needs 99% of elements inside it, and
with ``|out| ~ 0.033`` that bound is about 1.3e-3 relative -- tighter than the
reference's own 8.4e-5 deviation from exact arithmetic, because torch defaults to
``fp32_precision=tf32`` and the bench sets no precision knob. Being *more* accurate
therefore fails: exact fp32 scores 0.8670, ``sdpa`` 0.9225, fp16 0.9759, bf16 0.1990
(``profile/00-numerics/probe_numerics.py``, three seeds). The candidate has to mimic
cuBLAS's tf32, so two measurements shape the kernel:

* Keeping the projections on cuBLAS makes them *bitwise* identical to the baseline's --
  one 2304-wide GEMM on concatenated weights equals three 768-wide ones exactly, max abs
  error 0 on all three seeds. Q, K and V enter the attention kernel bit-for-bit, so the
  kernel's only deviation is its own.
* ``tl.dot(..., input_precision="tf32")`` **truncates** fp32 operands to tf32, while
  cuBLAS round-to-nearest-evens them. Left alone that is a systematic ~2.4e-4 relative
  bias on the logits, and since logits feed ``exp`` it reaches the output: the native dot
  scores 0.5867 on the QKV shape and 0.5793 on per-head QK^T, i.e. it fails outright.
  Rounding both operands to nearest-even *before* the dot makes the Triton result bitwise
  identical to cuBLAS on the projections and on QK^T, and equal to 6e-8 on PV, measured
  in the strided per-head layouts the reference really uses
  (``profile/00-numerics/probe_tf32.py``). The pre-rounding is idempotent, so it cannot
  compound with the hardware's own conversion.

Where the residual deviation enters is measured per stage
(``profile/02-attention-kernel/results-sweep.txt``): the fused projection and the logits
after the scale and the mask add are *bitwise* identical to the reference, the softmax
differs by 4.5e-8, and PV by 2.3e-5. That jump is not accumulation order -- with identical
probabilities the isolated PV dot agrees to 6e-8. It is the softmax difference crossing
tf32 rounding midpoints: at ``p ~ 1/77`` one tf32 ULP is ``2**-17 ~ 7.6e-6``, so a 4.5e-8
difference flips the rounding of a small fraction of the probabilities, and each flip moves
a PV term by about ``7.6e-6 * |v|``. The final output deviation is *smaller* (1.0e-5) than
PV's, which is expected rather than suspicious: the output projection's row norm is about
``0.02 * sqrt(768) ~ 0.55``, so uncorrelated PV errors attenuate, and at ``|out| ~ 0.03``
one tf32 ULP is already 1.5-3.1e-5, so many distinct PV values quantize to the same
operand -- 72-78% of output elements come out bitwise identical.

Anything the fast path is not written for runs the baseline's own sequence of torch ops.
Note that the fallback softmax is ``F.softmax`` rather than the frozen
``candidate/L1/softmax.py`` kernel: the baseline's ``..L1.softmax`` import resolves inside
the *baseline* package, so ``F.softmax`` is what "the reference" means here.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig

from ..L1.linear import Linear

_HEAD_DIM = 64
_KEY_BLOCK = 128                      # the kernel's key-block width, and so the S it can cover

_PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "out_proj")

# The fast path admits exactly the captured configuration and nothing else. The kernel
# handles 20 <= S <= 128 correctly -- that range was measured at every length, and the lower
# bound is where torch stops dispatching its BMMs to tf32 so that a faithful tf32 kernel
# becomes *more* accurate than the reference and fails for it (S in {1,2,3,5,...,19} score
# 0.79-0.92; see profile/03-dispatch/results-guards.txt). It is narrowed here anyway, because
# comparing one `torch.Size` against one constant is cheaper than unpacking a shape and
# bounds-checking it, and the per-call host budget in AC-5.1 is tight enough for that to
# matter. Everything outside these two shapes runs the reference path.
_CAPTURED_INPUT = torch.Size((1, 77, 768))
_CAPTURED_MASK = torch.Size((1, 1, 77, 77))
_CAPTURED_SEQ = 77
_CAPTURED_EMBED = 768
_CAPTURED_HEADS = 12

# tf32 is the reference's arithmetic, and the *effective cuBLAS policy* is what decides
# it -- not `torch.get_float32_matmul_precision()`. Those two can disagree: setting
# `torch.backends.cuda.matmul.fp32_precision = "ieee"` leaves the generic getter at
# "high" while cuBLAS switches to IEEE fp32, and a kernel still mimicking tf32 would then
# be measured against an exact-fp32 reference and score about 0.867. The generic setter
# propagates the other way ("highest" -> "ieee", "high" and "medium" -> "tf32"), so reading
# the backend value alone covers both APIs. Resolved once, here, because the public
# property costs 302 ns per read against 128 ns for the getter it forwards to.
try:
    _fp32_policy = torch._C._get_fp32_precision_getter
    _fp32_policy("cuda", "matmul")
except Exception:  # private getter gone or renamed: use the documented property
    def _fp32_policy(_backend: str, _op: str) -> str:
        return torch.backends.cuda.matmul.fp32_precision

# Selected by measurement over BM in {16, 32, 64, 128} x num_warps in {4, 8}, timed
# through the harness's own loop (profile/02-attention-kernel/results-sweep.txt). The
# sweep is about register pressure, not a compute peak: at 36 MFLOP the kernel is
# launch-latency bound, and the [BM, 128] fp32 logits tile on top of two [128, 64] operand
# tiles is what threatens to spill. BM=16 / 8 warps took 46.19 us end-to-end at 121
# registers and no spills; the widest tile, BM=128 / 4 warps, spilled 132 bytes and fell to
# 108.43 us -- slower than the baseline. There is no K loop to pipeline (S <= 128 is one key
# block), so num_stages is not a free parameter here.
_BM = 16
_NUM_WARPS = 8

# Host-side diagnostics: plain ints, no device sync, nothing the harness's integrity
# guards watch. Without them a failed build degrades to the fallback and validation
# "passes" at baseline speed, which reads as a win rather than as an untested run.
_FASTPATH_HITS: dict[str, int] = {}
_QKV_BUILDS: dict[str, int] = {}

_KERNEL = None
_KERNEL_STATUS = "disabled:not-built"


def _count(table: dict[str, int], key: str) -> None:
    table[key] = table.get(key, 0) + 1


class _Projection(Linear):
    """``..L1.linear.Linear``, plus flags it raises when a parameter object is replaced.

    ``m.q_proj.weight = nn.Parameter(...)`` changes neither the old object's version counter
    nor its storage pointer, so nothing a caller could poll would notice it, and a fused copy
    built from the old object would go on being used. Catching it in ``__setattr__`` costs
    nothing per forward.

    Each flag is a one-element list shared with an owner rather than a back-reference, so it
    survives ``deepcopy`` pointing at the copy's own owner. There is a *list* of them because
    a projection can be owned by more than one module at once: ``m.v_proj = other.v_proj``
    leaves both modules holding it, and both of their fused copies are derived from it.

    Two revisions of this got it wrong, in opposite directions. First the flag was bound at
    construction and never rebound, so the *receiving* module was never notified -- matched
    0.0003-0.0006 on all four projections. Then assignment *overwrote* the flag with the
    receiver's, so the *donating* module was never notified -- matched 0.0006-0.944. Owners
    are therefore added, never replaced, and de-duplicated by identity.

    Registers exactly ``weight`` and ``bias``, as ``Linear`` does, so state-dict keys are
    unchanged; ``_owners`` is a plain list and never appears in ``state_dict``.
    """

    def __init__(self, in_features: int, out_features: int, dirty: list):
        self._owners = [dirty]
        super().__init__(in_features, out_features, bias=True)

    def register_owner(self, dirty: list) -> None:
        """Add an owner's flag, de-duplicated by identity. Never removes one."""
        for existing in self._owners:
            if existing is dirty:
                return
        self._owners.append(dirty)

    def owned_by(self, dirty: list) -> bool:
        for existing in self._owners:
            if existing is dirty:
                return True
        return False

    def _mark_owners(self) -> None:
        for dirty in self.__dict__.get("_owners", ()):
            dirty[0] = True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("weight", "bias"):
            self._mark_owners()

    def _apply(self, *args, **kwargs):
        try:
            return super()._apply(*args, **kwargs)
        finally:
            self._mark_owners()

    def _load_from_state_dict(self, *args, **kwargs):
        try:
            super()._load_from_state_dict(*args, **kwargs)
        finally:
            self._mark_owners()


# ---------------------------------------------------------------------------
# The fused attention kernel.
#
# One program owns one (row tile, head) pair and does the whole chain: load this head's
# Q rows and all of its K and V rows out of the fused QKV buffer, QK^T, the scale, the
# mask add, an fp32 softmax over one 128-wide key block, PV, and a store straight into
# ``[S, embed_dim]`` layout -- which is what ``transpose(1, 2).contiguous()`` produces, so
# the baseline's copy kernel disappears instead of moving.
#
# Deliberately *not* flash-style: S <= 128 is a single key block, so there is nothing to
# tile over and no online rescaling to do, and a plain one-pass softmax keeps the
# arithmetic closer to the sequence the reference performs.
#
# Plain pointer arithmetic rather than TMA descriptors: the frozen candidate/L1/linear.py
# records device-side descriptor creation at about 2 us of per-CTA overhead, and host-side
# descriptors would have to be rebuilt every call because the harness's shifting pool
# hands out a fresh data_ptr each iteration.
# ---------------------------------------------------------------------------
def _build_kernel(bm: int, num_warps: int):
    """Compile the fused attention kernel and return a launcher, or raise."""
    import triton
    import triton.language as tl

    @triton.jit
    def _to_tf32_rne(x):
        """Round fp32 to tf32's 10-bit mantissa, to nearest even.

        The dot below truncates its operands; cuBLAS rounds to nearest even. Doing it
        here first makes the two agree -- the increment is (half - 1) plus the lowest
        kept bit, then the dropped 13 bits are cleared. Idempotent, because after one
        application those 13 bits are zero and the increment cannot carry into the
        mantissa a second time, so it composes safely with whatever the tensor core
        does to an already-representable operand.
        """
        b = x.to(tl.int32, bitcast=True)
        rounded = ((b + 0x0FFF + ((b >> 13) & 1)) & -8192).to(tl.float32, bitcast=True)
        # NaN has to survive. A NaN whose payload lives only in the dropped 13 bits --
        # 0x7f800001, say -- would round to infinity, and 0x7fffffff carries all the way
        # into the sign bit and lands on -0.0. Infinities and finite overflow are already
        # right (overflowing to infinity is what round-to-nearest does), so only NaN needs
        # the select.
        return tl.where(x == x, rounded, x)

    @triton.jit
    def _fused_attention(QKV, MASK, OUT, seq, embed, qkv_row, mask_row, out_row, scale,
                         HAS_MASK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         HD: tl.constexpr):
        head = tl.program_id(1)
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        feat = head * HD + tl.arange(0, HD)
        row_ok = rows < seq
        col_ok = cols < seq

        # Q rows past S load as 0.0, not -inf: a fully padded row tile has to stay
        # finite, because its softmax is still evaluated before the store drops it.
        q = tl.load(QKV + rows[:, None] * qkv_row + feat[None, :],
                    mask=row_ok[:, None], other=0.0)
        k = tl.load(QKV + cols[:, None] * qkv_row + (embed + feat)[None, :],
                    mask=col_ok[:, None], other=0.0)
        v = tl.load(QKV + cols[:, None] * qkv_row + (2 * embed + feat)[None, :],
                    mask=col_ok[:, None], other=0.0)

        logits = tl.dot(_to_tf32_rne(q), _to_tf32_rne(tl.trans(k)),
                        input_precision="tf32")
        # Scale after the dot and mask after the scale, the baseline's order. The mask is
        # all-ones here (the harness materializes any float arg named *mask* as ones) and
        # softmax is invariant to a uniform shift, but the fp32 addition still perturbs
        # low bits exactly as the reference's does, so it is performed rather than
        # algebraically dropped.
        logits = logits * scale
        if HAS_MASK:
            logits += tl.load(MASK + rows[:, None] * mask_row + cols[None, :],
                              mask=row_ok[:, None] & col_ok[None, :], other=0.0)
        # Only now the -inf tail, so padded columns cannot perturb the scale or the mask
        # add: exp(-inf - max) is exactly 0.0, so they contribute nothing to the sum.
        logits = tl.where(col_ok[None, :], logits, float("-inf"))
        prob = tl.exp(logits - tl.max(logits, 1)[:, None])
        prob = prob / tl.sum(prob, 1)[:, None]

        acc = tl.dot(_to_tf32_rne(prob), _to_tf32_rne(v), input_precision="tf32")
        tl.store(OUT + rows[:, None] * out_row + feat[None, :], acc, mask=row_ok[:, None])

    def launch(qkv, mask, out, seq, embed, num_heads, scale):
        _fused_attention[(triton.cdiv(seq, bm), num_heads)](
            qkv, mask if mask is not None else qkv, out,
            seq, embed, qkv.stride(0), mask.stride(2) if mask is not None else 0,
            out.stride(0), scale,
            HAS_MASK=mask is not None, BM=bm, BN=_KEY_BLOCK, HD=_HEAD_DIM,
            num_warps=num_warps)
        return out

    # Exposed so the configuration sweep and the NCU run can read the compiled kernel's
    # register count and spill count without reaching into a closure.
    launch.kernel = _fused_attention
    return launch


def _warm(launch, embed: int = 768, seq: int = _KEY_BLOCK) -> None:
    """Launch every configuration once, at import, then synchronize.

    Triton compiles on first launch. Doing it here rather than on the harness's first
    correctness forward keeps compilation out of the timed window entirely, and happens
    while the bench worker is still producing output -- clear of the watchdog that stalls
    on a quiet stderr log. Both mask variants are compiled because both are reachable.
    """
    heads = embed // _HEAD_DIM
    qkv = torch.zeros((seq, 3 * embed), device="cuda", dtype=torch.float32)
    out = torch.empty((seq, embed), device="cuda", dtype=torch.float32)
    mask = torch.zeros((1, 1, seq, seq), device="cuda", dtype=torch.float32)
    for arg in (mask, None):
        launch(qkv, arg, out, seq, embed, heads, _HEAD_DIM ** -0.5)
    torch.cuda.synchronize()


def _init_kernel() -> None:
    """Resolve the kernel once, at import. Any failure degrades to the reference path."""
    global _KERNEL, _KERNEL_STATUS
    if not torch.cuda.is_available():
        _KERNEL_STATUS = "disabled:no-cuda-device"
        return
    try:
        launch = _build_kernel(_BM, _NUM_WARPS)
        _warm(launch)
    except Exception as exc:  # compiler, driver, or architecture rejected the kernel
        _KERNEL = None
        # Flattened to a single physical line. A Triton or ptxas failure is routinely
        # multiline -- source excerpts, carets, a nested traceback -- and interpolating it
        # raw would break the one-line promise exactly when the message is most alarming.
        detail = " ".join(str(exc).split())
        _KERNEL_STATUS = f"failed:{type(exc).__name__}: {detail}"[:400]
        # One line, at import, on stderr: the bench worker routes this to the per-operator
        # log, so a swallowed build failure stays visible instead of hiding behind a silent
        # 1.00x.
        print(f"[candidate L2/clip_attention] kernel unavailable, delegating to torch: "
              f"{_KERNEL_STATUS}", file=sys.stderr, flush=True)
        return
    _KERNEL = launch
    _KERNEL_STATUS = "built"


_init_kernel()


# ---------------------------------------------------------------------------
class CLIPAttention(nn.Module):
    """Multi-head self-attention over a CLIP text sequence.

    Parameters live in four ``..L1.linear.Linear`` submodules, which register exactly
    ``weight`` and ``bias`` and derive nothing from them, so ``state_dict`` keys match the
    baseline's and the harness's ``load_state_dict(..., strict=False)`` shares weights.
    That load is wrapped in a bare ``except Exception: pass`` on the harness side, so a
    key mismatch would be *silent* -- hence the parity check in
    ``profile/01-contract/test_contract.py`` rather than trust.

    Nothing derived from parameter *values* is computed in ``__init__``: the harness
    constructs the module, moves it, casts it, and rewrites uninitialized parameters, and
    only *then* loads the baseline's state dict. Anything precomputed here would be built
    from ``torch.empty`` garbage.
    """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Config-derived, not weight-derived, and it cannot change after construction, so it
        # is not re-derived per call. The head geometry has to be checked separately from the
        # input shape: hidden_size=769 with 12 heads also gives head_dim 64, and a config with
        # 16 heads of 48 still presents a [1, 77, 768] input, so neither is caught by the
        # shape comparison alone.
        self._fast_ok = (self.embed_dim == _CAPTURED_EMBED
                         and self.num_heads == _CAPTURED_HEADS
                         and self.head_dim == _HEAD_DIM
                         and self.num_heads * self.head_dim == self.embed_dim)

        # Raised by the parameter holders when a parameter *object* is swapped out, and by
        # this module's __setattr__ when a whole projection is. Created before the
        # submodules, which capture it.
        self._dirty = [True]

        self.q_proj = _Projection(self.embed_dim, self.embed_dim, self._dirty)
        self.k_proj = _Projection(self.embed_dim, self.embed_dim, self._dirty)
        self.v_proj = _Projection(self.embed_dim, self.embed_dim, self._dirty)
        self.out_proj = _Projection(self.embed_dim, self.embed_dim, self._dirty)

        # Built on first use, never here. Plain attributes, so they stay out of
        # state_dict: registering the fused weight as a parameter or buffer would add a
        # key the baseline does not have, and the harness's weight sharing is keyed on
        # exact parity.
        self._cached = None

    # -- the fused weight, and keeping it coherent --------------------------------
    #
    # Coherence is carried entirely by *invalidation*, not by polling. Nothing is read per call
    # except one cached reference and one list index, which is what makes the per-call host
    # budget in AC-5.1 reachable. AC-5.1 offers exactly this choice -- "either a small number of
    # cheap version/pointer reads, or invalidation hooks on load_state_dict and _apply ... with
    # the residual staleness window documented" -- and an earlier revision took the first option
    # and could not fit the budget: six `_version` reads cost about 0.33 us per call, and the
    # `(data_ptr, _version)` variant about 0.88 us.
    #
    # What invalidates:
    #   * `_apply` and `_load_from_state_dict`, on this module and on each `_Projection`, both
    #     in a `finally` so a hook that mutates a weight and then raises cannot leave the cache
    #     both stale and trusted. These are the paths the harness uses: it moves and casts the
    #     module, then loads the baseline's state dict.
    #   * `_Projection.__setattr__`, for a replaced `weight` or `bias` object -- which bumps no
    #     version counter and changes no storage pointer, so nothing pollable would see it.
    #   * `CLIPAttention.__setattr__`, for a replaced projection module, which also registers
    #     this module as an owner of the incoming projection.
    #
    # What is left uncovered, and why each is acceptable here:
    #   * **Ordinary in-place parameter writes** -- `p.normal_()`, `p.copy_()` -- outside a
    #     state-dict load. Previously caught by the version stamp; now part of the residual
    #     window, which is the trade that buys the budget. The scored workload never does this
    #     after the first forward: the harness's own `p.normal_(0, 0.02)` sanitizer runs before
    #     any forward, and its weight sharing goes through `load_state_dict`, which invalidates.
    #   * In-place writes through the `.data` alias (`p.data.mul_(2)`), and `p.data = X`.
    #     Measured: these bump no version counter *and* preserve `data_ptr`, so no per-call
    #     check at any price detects them.
    #   * Publication across CUDA streams: the rebuild is enqueued on the calling stream and the
    #     cache is published to the host immediately, so a different stream consuming it
    #     establishes no dependency on the `cat`. Single-stream use, which is what the bench
    #     does, is ordered correctly.
    #   * Forward-mode AD, for which this build exposes no cheap predicate. Reverse mode is
    #     guarded by `torch.is_grad_enabled()`.
    def _invalidate(self) -> None:
        self._cached = None
        self._dirty[0] = True

    def __setattr__(self, name, value):
        """Notice a whole projection being replaced, not just its parameters.

        Marking this module dirty is only half of it: the incoming projection has to be
        rebound to *this* module's flag, or every later parameter replacement on it would
        notify its previous owner instead. A projection that cannot be rebound is caught in
        ``_rebuild``, which refuses to cache it.
        """
        super().__setattr__(name, value)
        if name in _PROJECTION_NAMES:
            dirty = self.__dict__.get("_dirty")
            if dirty is not None:
                dirty[0] = True
                if isinstance(value, _Projection):
                    # Added, never replaced: the donating module still owns it too, and its
                    # own fused copy is still derived from this projection.
                    value.register_owner(dirty)

    def _apply(self, *args, **kwargs):
        """Drop the cache across ``.to()`` / ``.cuda()`` / ``.float()``.

        ``_apply`` replaces every parameter's storage, so a cached concatenation of the
        old ones would address freed memory or the wrong device. Invalidated in a
        ``finally`` because a partially-applied module is exactly the case that must not
        keep using the cache.
        """
        try:
            return super()._apply(*args, **kwargs)
        finally:
            self._invalidate()

    def _load_from_state_dict(self, *args, **kwargs):
        # Also in a finally: a pre-hook that mutates a weight and then raises would
        # otherwise leave the cache both stale and trusted.
        try:
            super()._load_from_state_dict(*args, **kwargs)
        finally:
            self._invalidate()

    def _projections(self):
        """The cached projections, rebuilt if stale. The out-of-band accessor, for probes.

        ``forward`` does not call this -- it uses ``_fast_projections``, which folds the same
        three lines into the admission check so the timed path pays one Python call instead of
        two. The duplication is three lines and is deliberate.
        """
        cached = self._cached
        if cached is not None and not self._dirty[0]:
            return cached
        return self._rebuild()

    def _rebuild(self):  # noqa: C901 - a flat sequence of independent preconditions
        """Rebuild the fused weight, or return None if the parameters are not what the kernel
        assumes -- in which case the caller delegates rather than launching.

        The shape checks are not defensive noise. A ``q_proj`` replaced by a 767-output linear
        makes the concatenation 2303 columns wide while the kernel indexes 2304, which is an
        out-of-bounds read; an ``out_proj`` of a different output width makes the final
        ``view`` fail. Both are cheap to check here, because this runs once, not per call.
        """
        # Only cache from holders that participate in the invalidation protocol, and only if
        # *this* module is one of their registered owners. A plain `nn.Linear` assigned over a
        # projection, or a `_Projection` this module never registered with, cannot be relied
        # on to report a later parameter replacement -- so nothing is cached and every call
        # delegates. Four submodule lookups, once per rebuild.
        # The head geometry, checked here rather than per call: it is fixed at construction, so
        # a configuration the kernel is not shaped for simply never gets a cache and therefore
        # always delegates. It costs that configuration one wasted rebuild attempt per forward,
        # on a path that then issues ten kernel launches -- and it keeps one instance-attribute
        # read (~40 ns) off the accepting path, which the AC-5.1 budget needs.
        if not self._fast_ok:
            return None
        dirty = self._dirty
        projections = []
        for name in _PROJECTION_NAMES:
            proj = getattr(self, name, None)
            if not isinstance(proj, _Projection) or not proj.owned_by(dirty):
                return None
            projections.append(proj)
        q_proj, k_proj, v_proj, out_proj = projections

        wq, wk, wv = q_proj.weight, k_proj.weight, v_proj.weight
        bq, bk, bv = q_proj.bias, k_proj.bias, v_proj.bias
        wo, bo = out_proj.weight, out_proj.bias
        embed = self.embed_dim
        for tensor, shape in ((wq, (embed, embed)), (wk, (embed, embed)),
                              (wv, (embed, embed)), (wo, (embed, embed)),
                              (bq, (embed,)), (bk, (embed,)), (bv, (embed,)), (bo, (embed,))):
            if (tensor is None or tensor.shape != shape
                    or tensor.dtype is not torch.float32
                    or tensor.device != wq.device):
                return None
        with torch.no_grad():
            fused = (torch.cat([wq, wk, wv], 0), torch.cat([bq, bk, bv], 0))
        self._dirty[0] = False
        self._cached = fused + (wo, bo)
        _count(_QKV_BUILDS, "clip_attention")
        return self._cached

    # -- the one accepting-path helper --------------------------------------------
    def _fast_projections(self, hidden_states: torch.Tensor,
                          attention_mask: torch.Tensor | None):
        """Admission and the warm-cache lookup in one call: the projections, or None.

        This is the whole of the per-call host cost, and the acceptance criterion bounds it at
        about 1 us. It is one function rather than two because a Python call is ~70 ns and the
        timed path should pay for one. Ordered cheapest-first, and every predicate's cost was
        measured rather than guessed (profile/03-dispatch/results-guards.txt):

        * one `torch.Size` comparison against a constant (108 ns) instead of unpacking a shape
          and bounds-checking three values (158 ns) -- which is why admission is narrowed to
          the captured configuration even though the kernel handles 20 <= S <= 128;
        * `torch.is_autocast_enabled()` (46 ns) rather than the `("cuda")` form (122 ns),
          verified on device to return True inside `torch.autocast("cuda")`;
        * one `get_device()` per tensor (55 ns each) rather than `is_cuda` plus a
          `device`-object comparison (48 + 124 ns); it returns -1 for a CPU tensor, so it
          covers the not-on-CUDA case at the same time;
        * the cache check is two loads and a list index, because coherence is carried by
          invalidation hooks and dirty flags rather than by polling version counters -- see
          the note on the residual window below.

        The exact-`torch.Tensor` checks are on *both* arguments. An earlier revision dropped
        the one on the mask, reasoning that tracing makes both operands fake together; they are
        independent arguments, and a plain input paired with a `Tensor._make_subclass` mask was
        observed taking the fast path. A wrapper subclass may have no usable storage pointer at
        all, so its 36 ns stays paid.
        """
        if _KERNEL is None:
            return None
        if hidden_states.dtype is not torch.float32 or type(hidden_states) is not torch.Tensor:
            return None
        if hidden_states.shape != _CAPTURED_INPUT:
            return None
        device = hidden_states.get_device()          # -1 when not on CUDA
        if device < 0 or not hidden_states.is_contiguous():
            return None
        # An inference-only kernel: it builds no autograd graph. And under autocast the
        # reference would run in a different dtype entirely.
        if torch.is_grad_enabled() or torch.is_autocast_enabled():
            return None
        # Mimicking cuBLAS's tf32 is only correct while cuBLAS is on tf32, and the backend
        # value is the one that decides. An allow-list of exactly "tf32" rather than a
        # deny-list of "ieee", so a future backend precision this kernel has never been
        # measured against delegates instead of being assumed compatible. Wrapped because
        # mixing the legacy `allow_tf32` API with the new one makes the accessor itself raise,
        # and an unreadable policy is not one to bet on.
        try:
            if _fp32_policy("cuda", "matmul") != "tf32":
                return None
        except Exception:
            return None
        if attention_mask is not None:
            if (type(attention_mask) is not torch.Tensor
                    or attention_mask.dtype is not torch.float32
                    or attention_mask.shape != _CAPTURED_MASK
                    or not attention_mask.is_contiguous()
                    or attention_mask.get_device() != device):
                return None
        cached = self._cached
        if cached is not None and not self._dirty[0]:
            return cached
        return self._rebuild()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projections = self._fast_projections(hidden_states, attention_mask)
        if projections is None:
            return self._reference(hidden_states, attention_mask)
        qkv_weight, qkv_bias, out_weight, out_bias = projections
        seq, embed = _CAPTURED_SEQ, _CAPTURED_EMBED

        # Kernel 1: one 2304-wide cuBLAS GEMM in place of three 768-wide ones. Bitwise
        # identical to the baseline's, so Q, K and V reach the kernel bit-for-bit.
        qkv = F.linear(hidden_states.view(seq, embed), qkv_weight, qkv_bias)
        # Kernel 2: everything between the projections, storing in [S, embed] layout --
        # which is what transpose(1, 2).contiguous() produces, so the copy has nowhere to
        # go rather than moving. Allocated contiguous so cuBLAS takes it without a copy.
        attn = torch.empty((seq, embed), device=qkv.device, dtype=qkv.dtype)
        _KERNEL(qkv, attention_mask, attn, seq, embed, _CAPTURED_HEADS, self.scale)
        _count(_FASTPATH_HITS, "clip_attention")
        # Kernel 3: the baseline's own output projection, on the same weights. It adds no
        # deviation of its own, though its input carries the kernel's -- so the end-to-end
        # result is not bitwise identical, only the projection call is.
        return F.linear(attn, out_weight, out_bias).view(1, seq, embed)

    # -- fallback ----------------------------------------------------------------
    def _reference(self, hidden_states: torch.Tensor,
                   attention_mask: torch.Tensor | None) -> torch.Tensor:
        """The baseline's operation sequence, op for op.

        ``F.linear`` / ``torch.matmul`` / ``F.softmax`` are what the baseline's ``Linear``,
        ``BMM`` and ``Softmax`` submodules reduce to, so this is bit-identical to the
        reference for every input, at the reference's cost.
        """
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(shape).transpose(1, 2)
        keys = keys.view(shape).transpose(1, 2)
        values = values.view(shape).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(queries.dtype)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)
