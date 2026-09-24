"""Flash attention prefill kernel (variable-length sequences).

Same contract as ``baseline.py``: routes through vLLM's bundled FlashAttention
build at the version vLLM itself would select for this device (FA3 on Hopper,
FA4 on Blackwell, FA2 otherwise) -- see :mod:`fa_utils`.

The one deviation is a narrow fast path for the shape this operator actually
sees -- non-causal bf16 variable-length prefill with ``num_heads ==
num_kv_heads`` -- which compiles its own FlashAttention-4 CuTe kernel with a
tuned exp2-emulation frequency and launches it directly, skipping vLLM's Python
dispatch layers.

Of those two changes, **the tuned kernel is what makes it faster**: 1.09-1.18x
end to end on the five captured shapes (median of 6 paired runs), from 1.15-1.21x
of GPU kernel time.  Bypassing the dispatch layers cuts host cost per call from
40.4 us to ~17 us, but that is worth 1.00x under ``fastkernels.bench``, which
enqueues a ~72 us L2 flush before every timed iteration and so leaves the CPU
enough slack to stay ahead of the GPU either way.  The bypass is kept because it
is the mechanism that makes owning the kernel possible, not for its own sake.

The fast path is an allowlist.  Every input it does not positively recognise
runs the baseline body verbatim, so behaviour outside the window is unchanged.

Pinned assumptions, each re-checked per call rather than trusted:

* The fast path admits **only the captured domain**: bf16, ``head_dim == 72``,
  ``num_heads == num_kv_heads``, non-causal varlen, unsplit, unpaged,
  inference-only, compute capability exactly 10.0, and current
  ``fa_version == 4`` (a public attribute, so read every call).  fp16, other head
  dims and GQA run the baseline, which keeps the tuned kernel confined to the
  shape it was measured on and leaves every other input bit-identical to today.
* vLLM 0.26.0's ``vllm.vllm_flash_attn.cute`` internals: the
  ``FlashAttentionForwardSm100`` constructor arguments and the ``cute.compile``
  argument layout for this branch, the 19-argument runtime call, the ``AuxData``
  slot CuTe reports as unconvertible, and the per-instance ``_tune`` attribute
  the kernel reads its ex2 parameters from.
* **Nothing shared is read or written.** ``fastkernels.bench`` times the candidate
  and the baseline in one process and both read
  ``interface._flash_attn_fwd.compile_cache``; since ``compile_key`` covers
  neither the ex2-emulation parameters nor the register split, a tuned kernel
  reachable from that cache would silently become the baseline's kernel too.  So
  the kernel is constructed and compiled here with its tuning set on the
  instance, and this module does not even *import* ``interface`` or
  ``flash_fwd_sm100._TUNING_CONFIG`` -- there is no code path that could reach
  them.  Note that merely *calling* ``_flash_attn_fwd`` is a write: it inserts its
  own compiled callable into that cache whenever the key is cold.

Reproducing the compile arguments by hand is the price of that isolation, and a
silently different kernel would be a wrong answer rather than a crash.  So a
newly compiled kernel is validated against ground truth before it is adopted --
an untuned sibling against naive fp32 attention to catch a structurally wrong
kernel, then the tuned kernel against that sibling to check the tuning (see
``_validated_layout``).  Parity against the real baseline is covered from outside,
by ``scratch/candidate_check.py``, which is where ``_flash_attn_fwd`` can be
consulted freely.

Every failure mode -- an unrecognised input, a constructor or compile-argument
mismatch, a kernel that disagrees with ground truth, a launch error -- ends in the
baseline body: never in a wrong answer, and never in an exception the baseline
would not itself have raised.
"""


import torch
import torch.nn as nn

from ....infra.fa_utils import (
    fa3_scheduler_metadata,
    fa_version_for_head_size,
    flash_attn_varlen_func,
)

# Only these five names are imported from vLLM, and none of them is a mutable
# global: a kernel class, a tensor adapter, the log-level getter that
# ``compile_key`` covers, the aux NamedTuple, and CuTe itself.  Neither
# ``interface._flash_attn_fwd`` nor ``flash_fwd_sm100._TUNING_CONFIG`` is imported,
# so no code path here can reach them.
try:  # FA4 is only present in vLLM's bundled build, and only used on Blackwell.
    import cutlass.cute as _cute
    from vllm.vllm_flash_attn.cute.cute_dsl_utils import to_cute_tensor as _to_cute_tensor
    from vllm.vllm_flash_attn.cute.fa_logging import get_fa_log_level as _fa_log_level
    from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import (
        FlashAttentionForwardSm100 as _Sm100Fwd,
    )
    from vllm.vllm_flash_attn.cute.utils import AuxData as _AuxData
except Exception:  # pragma: no cover - no FA4 build available
    _cute = _to_cute_tensor = _AuxData = _Sm100Fwd = _fa_log_level = None

# ``AuxData`` is a NamedTuple, so one shared empty instance is safe to reuse and
# saves an allocation on every launch.
_NO_AUX = None if _AuxData is None else _AuxData(None, None)

# ex2-emulation tuning for the candidate's own kernel.
#
# The kernel's softmax leans on the transcendental pipe: ncu puts XU at 36-45% of
# peak active while FMA idles near 9-11%, which is the asymmetry FA4's FMA-based
# Cody-Waite emulation exists to fix.  ``_TUNING_CONFIG`` has no entry for
# ``head_dim_padded == 80``, so the untuned kernel takes the ``ex2_emu_freq=16``
# default -- emulating one fragment in sixteen -- and leaves FMA idle.  Halving
# that period moves about 12% of the kernel's total XU work onto FMA (+28% FMA)
# for ~9% more instructions, and because the kernel is latency-bound rather than
# instruction-bound that trade pays: ``long_scoreboard`` stalls drop 37-44% and
# warp eligibility rises ~1.45x, worth 1.205x / 1.153x / 1.150x / 1.182x of GPU
# kernel time on the four device-bound captured shapes at max|diff| = 1.95e-3,
# five times inside the bf16 tolerance.  Note XU *percentage* stays flat -- it is
# a rate, and the kernel got shorter; see profile/fa4_candidate_ex2freq8/REPORT.md.
# Timings were taken with the compile cache cleared between configurations,
# without which ``compile_key``'s omission of these parameters aliases every row
# onto the first kernel (``scratch/ex2_sweep.log``, ``scratch/ex2_minimal.log``).
#
# ``ex2_emu_freq`` is the only knob carried, because it is the only one with a
# demonstrated effect: ``ex2_emu_start_frg`` would be set to the 1 it already
# defaults to, and ``ex2_emu_res`` is read only by the dedicated hd256 2-CTA
# kernel, never by the one this shape compiles.  Both were confirmed inert by
# measurement as well as by reading the source.
_EX2_TUNING = {"ex2_emu_freq": 8}
_TUNING_FINGERPRINT = tuple(sorted(_EX2_TUNING.items()))

# The captured workload, and the only domain the direct launch admits: the
# Qwen3-VL vision tower at tp4, bf16, 4 heads, head_dim 72, non-causal varlen.
# Everything else -- other head dims, fp16, GQA -- runs the baseline body, which
# is what keeps the tuned kernel confined to the shape it was measured on and
# every other input bit-identical to today.
_CAPTURED_DTYPE = torch.bfloat16
_CAPTURED_HEAD_DIM = 72

# Compute capability the direct launch has been measured on.
_MEASURED_ARCH = 100

# FA4's SM100 default forward tile, from the ``FwdConfig(128, 128, ...)``
# fallthrough in ``cute/interface.py``; SM100 has no per-head-dim heuristic.
# ``q_stage`` is derived from ``tile_m``, and ``q_stage`` is part of ``compile_key``.
_SM100_DEFAULT_TILE_M = 128
_SM100_DEFAULT_TILE_N = 128

# Acceptance bounds for a freshly compiled kernel, both applied elementwise as
# ``|a - b| <= atol + rtol * |b|`` -- the same form ``bench._compare_tensor`` uses,
# rather than a single absolute number.  A flat absolute bound is the wrong shape
# here: bf16's rounding step scales with output magnitude, so a bound tight enough
# to be meaningful on one probe rejects legitimate rounding on another.  (Measured:
# tuned-vs-untuned is 1.95e-3 on a 256-token probe and 3.91e-3 on a 64-token one --
# both one bf16 ULP at their respective magnitudes, and both far inside the
# harness's own 1e-2.)
#
# The tuning bound is the harness's bf16 tolerance exactly, because that is the
# criterion the graded run applies.  The layout bound is looser: its job is to
# catch a *structurally* wrong kernel -- a misplaced compile argument or
# constructor flag yields garbage, not a few ULPs -- while tolerating the bf16-vs-
# fp32 gap against a naive reference that accumulates in a different order.
_TUNING_ATOL = _TUNING_RTOL = 1e-2
_LAYOUT_ATOL = _LAYOUT_RTOL = 5e-2

# Bound once: these are called on every fast-path dispatch, and the attribute
# lookups are a measurable share of a path whose whole point is host cost.
_is_grad_enabled = torch.is_grad_enabled
_is_capturing = torch.cuda.is_current_stream_capturing

_MISSING = object()


# A predicate below must be at least as strict as the condition under which the
# direct launch equals the baseline, which means testing a value the same way
# vLLM's own code tests it.  Truthiness is not a safe stand-in for a ``== 0``
# comparison, and iteration is not a safe stand-in for indexing: a type that
# disagrees with itself across those two protocols would be admitted here and
# interpreted differently downstream, which is the silent-wrong-answer class.
# The exact-type checks exclude such types outright rather than reasoning about
# them.
_NUMBER_TYPES = (bool, int, float)


def _is_none(value):
    return value is None


def _is_falsy(value):
    return not value


def _is_zero(value):
    """``0``/``False``/``0.0`` only.

    ``cute/interface.py`` disables softcap with ``if softcap == 0.0``, a value
    comparison, so a nonzero value that merely happens to be falsy would keep
    the baseline's score modifier while the fast path dropped it.  ``None`` is
    admitted too because ``None == 0.0`` is false there, which also disables it.
    """
    return value is None or (type(value) in _NUMBER_TYPES and value == 0)


def _is_disabled_causal(value):
    """``False`` or ``0``, but not ``None``.

    ``causal`` reaches ``compile_key`` verbatim, and ``None`` is a distinct key
    from ``False`` there even though the kernel treats both as non-causal, so
    admitting it would let one signature cover two compile keys.
    """
    return type(value) in _NUMBER_TYPES and value == 0


def _is_default_window(value):
    """``None`` or a real 2-sequence of ``-1``.

    ``flash_attn_varlen_func`` reads this as ``window_size[0]`` /
    ``window_size[1]`` after asserting ``len(...) == 2``, so it is indexed here
    the same way rather than iterated.
    """
    if value is None:
        return True
    return (
        type(value) in (tuple, list)
        and len(value) == 2
        and type(value[0]) is int and value[0] == -1
        and type(value[1]) is int and value[1] == -1
    )


def _is_one(value):
    return type(value) in _NUMBER_TYPES and value == 1


def _is_scale(value):
    return value is None or type(value) in (int, float)


# Every forward keyword the fast path knows how to interpret, mapped to the
# constraint under which the direct launch reproduces the baseline.  A keyword
# that is absent from this table routes to the baseline, which is what keeps an
# unrecognised future argument from being silently dropped -- including
# ``fa_version``, which ``baseline.forward`` lets the caller override via
# ``fa_kw.update(kwargs)``, and ``seqused_q``, which the baseline's wrapper does
# not accept at all and must keep rejecting.
_FAST_PATH_KWARGS = {
    # Features the 19-argument launch passes ``None`` for and therefore cannot
    # express.
    "block_table": _is_none,
    "seqused_k": _is_none,
    "scheduler_metadata": _is_none,
    "q_descale": _is_none,
    "k_descale": _is_none,
    "v_descale": _is_none,
    "alibi_slopes": _is_none,
    "s_aux": _is_none,
    "mask_mod": _is_none,
    "aux_tensors": _is_none,
    "output_scale": _is_none,
    "dynamic_causal": _is_none,
    "q_v": _is_none,
    "cp_tot_seqused_k": _is_none,
    # A caller-supplied ``out`` has its own dtype/shape/stride contract; let the
    # wrapper validate it.
    "out": _is_none,
    "causal": _is_disabled_causal,
    "softcap": _is_zero,
    # These two are read with plain truthiness downstream (``if requires_grad or
    # return_lse``), so truthiness is the faithful test.  The direct launch never
    # allocates an lse, and ``lse is None`` is part of ``compile_key``.
    "return_softmax_lse": _is_falsy,
    "return_attn_probs": _is_falsy,
    # Forward is deterministic either way, and neither flag reaches the FA4
    # branch at all, but keep the window narrow.
    "deterministic": _is_falsy,
    "dropout_p": _is_zero,
    "window_size": _is_default_window,
    # ``baseline.forward`` pins ``num_splits=1`` and then calls
    # ``fa_kw.update(kwargs)``, so a caller-supplied value wins.  Split-KV runs
    # a different kernel plus a combine pass (``cute/interface.py``), and
    # ``is_split_kv`` is part of ``compile_key``.
    "num_splits": _is_one,
    "cp_world_size": _is_one,
    "cp_rank": _is_zero,
    # Used, not merely tolerated: the fast path follows the wrapper's rule of
    # ``q.shape[-1] ** -0.5`` when this is absent or None.
    "softmax_scale": _is_scale,
}


# Compute capability per device index; ``torch.cuda.get_device_capability`` is
# not free enough to call on a path whose whole point is host cost.
_DEVICE_ARCH: "dict[int, int]" = {}


def _device_arch(index: int) -> int:
    """Compute capability of a CUDA device, by ``Tensor.get_device()`` index."""
    arch = _DEVICE_ARCH.get(index)
    if arch is None:
        major, minor = torch.cuda.get_device_capability(index)
        arch = major * 10 + minor
        _DEVICE_ARCH[index] = arch
    return arch


def _compile_kernel(q_stage: int, device: int, *, tuned: bool):
    """``cute.compile`` a candidate-owned SM100 forward kernel for the captured domain.

    Every argument below is the value ``cute/interface.py`` derives for bf16,
    ``head_dim == head_dim_v == 72``, varlen, non-causal, unsplit, unpaged, arch
    100, no optional features.  Reading it against that file is how this stays
    honest across a vLLM bump; ``_validated_layout`` is what catches it when it
    does not.
    """
    kernel = _Sm100Fwd(
        _CAPTURED_HEAD_DIM,
        _CAPTURED_HEAD_DIM,
        qhead_per_kvhead=1,        # guard pins num_heads == num_kv_heads
        is_causal=False,
        is_local=False,            # window_size is (-1, -1)
        is_split_kv=False,         # num_splits == 1
        pack_gqa=False,            # qhead_per_kvhead == 1
        m_block_size=_SM100_DEFAULT_TILE_M,
        n_block_size=_SM100_DEFAULT_TILE_N,
        q_stage=q_stage,
        is_persistent=False,       # varlen: `cu_seqlens_q is None` is False
        score_mod=None,
        mask_mod=None,
        has_aux_tensors=False,
        paged_kv_non_tma=False,    # no page table, so page_size is None
        is_varlen_q=True,
        q_subtile_factor=1,        # no block sparsity
        use_2cta_instrs=False,     # needs `cu_seqlens_q is None` and hd 128/192
        use_clc_scheduler=False,   # disabled for varlen MHA
        output_quant_key=None,     # no output_scale
    )
    if tuned:
        # ``__init__`` reads the module-level ``_TUNING_CONFIG`` into
        # ``self._tune``; overriding the *instance* attribute is what the global
        # mutation used to be for.  ``__call__`` reads ``self._tune`` at trace
        # time, so this is baked into the compiled kernel.  ``enable_ex2_emu`` is
        # already True from the ``head_dim_padded <= 128`` default, and the
        # register split is the fixed ``< 96`` branch either way.
        kernel._tune = dict(_EX2_TUNING)

    q, k, v, cu_q, cu_k, _ = _probe_inputs(q_stage, device)
    return _cute.compile(
        kernel,
        _to_cute_tensor(q),
        _to_cute_tensor(k),
        _to_cute_tensor(v),
        _to_cute_tensor(q.new_empty(q.shape)),
        None,                                        # lse
        _CAPTURED_HEAD_DIM ** -0.5,
        _to_cute_tensor(cu_q, assumed_align=4, leading_dim=0),
        _to_cute_tensor(cu_k, assumed_align=4, leading_dim=0),
        None, None,                                  # seqused_q, seqused_k
        None,                                        # dynamic_causal
        None,                                        # page_table
        None, None,                                  # window_size_left/right
        None,                                        # learnable_sink
        None,                                        # descale_tensors
        None,                                        # block-sparse tensors
        _NO_AUX,
        None,                                        # output_scale
        _cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _probe_inputs(q_stage: int, device: int):
    """A small varlen batch whose ``q_stage`` matches, for compiling and checking.

    Two sequences, long enough to land on the requested ``q_stage`` given
    ``tile_m``: ``q_stage`` is ``2 if max_seqlen_q > tile_m else 1``.  Small enough
    that a naive fp32 attention reference over it is trivial.
    """
    seqlen = 2 * _SM100_DEFAULT_TILE_M if q_stage == 2 else _SM100_DEFAULT_TILE_M // 2
    nseq, nh, hd = 2, 4, _CAPTURED_HEAD_DIM
    dev = torch.device("cuda", device)
    g = torch.Generator(device=dev).manual_seed(0x5EED)
    shape = (nseq * seqlen, nh, hd)
    q = torch.randn(shape, generator=g, device=dev, dtype=_CAPTURED_DTYPE)
    k = torch.randn(shape, generator=g, device=dev, dtype=_CAPTURED_DTYPE)
    v = torch.randn(shape, generator=g, device=dev, dtype=_CAPTURED_DTYPE)
    cu = torch.arange(0, nseq * seqlen + 1, seqlen, dtype=torch.int32, device=dev)
    return q, k, v, cu, cu, seqlen


def _within(actual, reference, atol, rtol):
    """``bench._compare_tensor``'s rule, requiring every element to be inside it."""
    a, r = actual.float(), reference.float()
    return bool(((a - r).abs() <= atol + rtol * r.abs()).all().item())


def _naive_attention(q, k, v, seqlen):
    """Reference attention in fp32, sharing no code with the CuTe path.

    Deliberately the dumbest possible implementation: this is the ground truth
    that catches a structurally wrong kernel, so it must not be able to be wrong
    in the same way.
    """
    nh, hd = q.shape[1], q.shape[2]
    out = torch.empty_like(q)
    scale = hd ** -0.5
    for start in range(0, q.shape[0], seqlen):
        sl = slice(start, start + seqlen)
        qs = q[sl].float().transpose(0, 1)            # (nh, seqlen, hd)
        ks = k[sl].float().transpose(0, 1)
        vs = v[sl].float().transpose(0, 1)
        probs = torch.softmax(qs @ ks.transpose(1, 2) * scale, dim=-1)
        out[sl] = (probs @ vs).transpose(0, 1).to(out.dtype)
    return out


# ``(q_stage, device)`` pairs whose compile-argument layout has been validated
# against ground truth in this process.  The layout is a property of the code, not
# of a module instance, so validating it once per process is enough -- and a
# compiled function is bound to the device it was compiled for, hence the device.
_LAYOUT_OK: "dict[tuple[int, int], bool]" = {}


def _validated_layout(q_stage: int, device: int) -> bool:
    """Has this kernel shape been shown to compute attention correctly?

    Two independent checks, because the hand-written compile arguments are the one
    thing here that could be silently wrong:

    * an **untuned** kernel against naive fp32 attention -- a misplaced argument or
      a wrong constructor flag produces garbage, which this catches decisively;
    * the **tuned** kernel against that untuned kernel -- which is what validates
      the ex2 tuning rather than the layout.

    Both references are candidate-owned.  ``_flash_attn_fwd`` would be the more
    authoritative reference, but consulting it inserts its own compiled callable
    into ``_flash_attn_fwd.compile_cache`` whenever the key is cold, and this
    module must not write that cache.  Parity against the real baseline is covered
    from outside, by ``scratch/candidate_check.py``.
    """
    cached = _LAYOUT_OK.get((q_stage, device))
    if cached is not None:
        return cached
    ok = False
    try:
        q, k, v, cu_q, cu_k, seqlen = _probe_inputs(q_stage, device)
        scale = _CAPTURED_HEAD_DIM ** -0.5
        untuned_out, tuned_out = q.new_empty(q.shape), q.new_empty(q.shape)
        _launch_kernel(_compile_kernel(q_stage, device, tuned=False),
                       q, k, v, untuned_out, cu_q, cu_k, scale)
        _launch_kernel(_compile_kernel(q_stage, device, tuned=True),
                       q, k, v, tuned_out, cu_q, cu_k, scale)
        torch.cuda.synchronize()
        ok = (torch.isfinite(tuned_out).all().item()
              and _within(untuned_out, _naive_attention(q, k, v, seqlen),
                          _LAYOUT_ATOL, _LAYOUT_RTOL)
              and _within(tuned_out, untuned_out, _TUNING_ATOL, _TUNING_RTOL))
    except Exception:
        ok = False
    _LAYOUT_OK[(q_stage, device)] = ok
    return ok


def _launch_kernel(compiled, q, k, v, out, cu_seqlens_q, cu_seqlens_k, softmax_scale):
    """The 19-argument runtime call ``cute/interface.py`` builds for the arch-100,
    non-qv, non-split-KV, non-hd256 branch.

    The stream is not among them: the kernel is compiled against
    ``make_fake_stream(use_tvm_ffi_env_stream=True)``, so it picks up the current
    torch stream from the TVM-FFI environment.
    """
    compiled(
        q, k, v, out,
        None,                 # lse
        softmax_scale,
        cu_seqlens_q, cu_seqlens_k,
        None, None,           # seqused_q, seqused_k
        None,                 # dynamic_causal (SM90 only)
        None,                 # page_table
        None, None,           # window_size_left, window_size_right
        None,                 # learnable_sink
        None,                 # descale_tensors (arch 10/11 slot)
        None,                 # block-sparse tensors
        _NO_AUX,
        None,                 # output_scale
    )


class FlashAttnPrefill(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        # Hopper FA3 cannot run head_dim>256; vLLM upgrades those layers to FA4.
        self.fa_version = fa_version_for_head_size(head_dim)
        # Compiled kernel per signature.  A ``None`` value records a signature
        # whose acquisition or launch failed, so it is attempted once and then
        # left alone.
        self._compiled: dict = {}

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        if _Sm100Fwd is not None:
            try:
                admitted = self._admits_direct_launch(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, kwargs
                )
            except Exception:
                # A value that will not answer one of the guard's predicates is
                # not a value the fast path recognises.  The baseline may well
                # accept it, so this must not become an exception of its own.
                admitted = False
            if admitted:
                out = self._direct_launch(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    kwargs.get("softmax_scale"),
                )
                if out is not None:
                    return out
        return self._baseline_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
        )

    # ------------------------------------------------------------------
    # Fast path
    # ------------------------------------------------------------------
    def _admits_direct_launch(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k, kwargs) -> bool:
        """True only for the exact captured call the direct launch reproduces."""
        # Read from ``self`` every call, not cached at construction: these are
        # public attributes that L2/L3/L4 callers can reach into, and a caller
        # that lowers ``fa_version`` to 3 would send the baseline to FA3 while a
        # frozen flag still sent the fast path to FA4.
        if (self.fa_version != 4
                or self.head_dim != _CAPTURED_HEAD_DIM
                or self.num_heads != self.num_kv_heads):
            return False

        for name, value in kwargs.items():
            check = _FAST_PATH_KWARGS.get(name)
            if check is None or not check(value):
                return False

        # Inference only.  FA4's forward allocates an lse whenever any input
        # requires grad, and ``lse is None`` is part of ``compile_key``.
        if _is_grad_enabled():
            return False
        # Acquisition compiles, which cannot happen mid-capture, and a captured
        # graph replays the pointers it was captured with.  Refusing outright
        # reproduces today's behaviour under capture exactly.
        if _is_capturing():
            return False

        if q is None or k is None or v is None:
            return False
        if (q.dtype is not _CAPTURED_DTYPE or k.dtype is not _CAPTURED_DTYPE
                or v.dtype is not _CAPTURED_DTYPE):
            return False
        # ``Tensor.shape`` builds a fresh ``torch.Size`` on every access, so bind
        # each one once rather than indexing the property repeatedly.
        qs, ks, vs = q.shape, k.shape, v.shape
        if len(qs) != 3 or len(ks) != 3 or len(vs) != 3:
            return False
        hd, nh = _CAPTURED_HEAD_DIM, self.num_heads
        if qs[2] != hd or ks[2] != hd or vs[2] != hd:
            return False
        if qs[1] != nh or ks[1] != nh or vs[1] != nh:
            return False
        if ks[0] != vs[0]:
            return False
        # Nothing to attend over.  ``cute/interface.py`` zero-fills and returns
        # from its Python prologue for this, but only after ``_validate_head_dims``
        # has had a chance to reject the shape, so reproducing the zero-fill here
        # would answer where the baseline raises.  The baseline does both, and
        # these shapes do no work, so hand them over.
        if qs[0] == 0 or vs[0] == 0:
            return False
        if q.requires_grad or k.requires_grad or v.requires_grad:
            return False
        # What ``maybe_contiguous`` in vLLM's wrapper checks before it would
        # rewrite the tensor; the captured ``v`` is strided but unit-stride in its
        # last dimension, so nothing is copied.
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            return False

        if cu_seqlens_q is None or cu_seqlens_k is None:
            return False
        if cu_seqlens_q.dtype is not torch.int32 or cu_seqlens_k.dtype is not torch.int32:
            return False
        cqs, cks = cu_seqlens_q.shape, cu_seqlens_k.shape
        if len(cqs) != 1 or len(cks) != 1 or cqs[0] != cks[0] or cqs[0] < 2:
            return False
        if cu_seqlens_q.stride(0) != 1 or cu_seqlens_k.stride(0) != 1:
            return False

        # ``q_stage`` -- and so ``compile_key`` -- is derived from ``max_seqlen_q``,
        # so it has to be a plain int rather than something the prologue would
        # substitute a default for.
        if type(max_seqlen_q) is not int or type(max_seqlen_k) is not int:
            return False

        # ``get_device`` returns an int, where comparing ``Tensor.device`` would
        # build and compare device objects.
        dev = q.get_device()
        if (dev < 0 or k.get_device() != dev or v.get_device() != dev
                or cu_seqlens_q.get_device() != dev or cu_seqlens_k.get_device() != dev):
            return False
        # ``arch`` is part of ``compile_key`` and only 10.0 is measured here;
        # SM103 and SM110 select a different tuning key.
        if _device_arch(dev) != _MEASURED_ARCH:
            return False
        return True

    def _direct_launch(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, softmax_scale):
        """The launch, or ``None`` to run the baseline instead.

        Never raises: an acquisition or launch failure disables the signature and
        hands the call back, because the module promises that anything it cannot
        do is done by the baseline rather than turned into an error the baseline
        would not have raised.
        """
        if softmax_scale is None:
            # ``flash_attn_varlen_func``'s rule.  The guard has pinned
            # ``q.shape[-1]`` to the captured head dim, so this equals
            # ``self.sm_scale``, but the wrapper's rule is the one to follow.
            softmax_scale = _CAPTURED_HEAD_DIM ** -0.5

        # ``2 if max_seqlen_q * qhead_per_kvhead > tile_m else 1`` in
        # ``cute/interface.py``, with the ratio pinned to 1 by the guard.  The
        # captured shapes all land on 2, so a signature without it would validate
        # cleanly and still be wrong for a caller with a short max_seqlen_q.
        q_stage = 2 if max_seqlen_q > _SM100_DEFAULT_TILE_M else 1
        # Everything else in ``compile_key`` is pinned by the guard or constant
        # for the process; what is left is ``q_stage``, the device a compiled
        # function is bound to, and the FA log level, which ``compile_key``
        # covers and ``fa_logging.set_fa_log_level`` can change at run time.
        signature = (q_stage, q.get_device(), _fa_log_level(), _TUNING_FINGERPRINT)

        compiled = self._compiled.get(signature, _MISSING)
        if compiled is _MISSING:
            try:
                compiled = self._acquire(q_stage, signature[1])
            except Exception:
                compiled = None
            self._compiled[signature] = compiled
        if compiled is None:
            return None

        # Contiguous, matching the prologue's own ``torch.empty``; ``empty_like``
        # would inherit q's strides, and the guard only pins q's last stride.
        out = q.new_empty(q.shape)
        try:
            self._launch(compiled, q, k, v, out, cu_seqlens_q, cu_seqlens_k, softmax_scale)
        except Exception:
            # An ABI or configuration mismatch costs one fallback per signature,
            # not an exception. (It cannot be relied on for correctness: CUDA
            # errors are asynchronous and an ABI-compatible but wrong callable
            # would not raise at all. That is what the guard, the signature and
            # the acquisition self-check are for.)
            self._compiled[signature] = None
            return None
        return out

    def _launch(self, compiled, q, k, v, out, cu_seqlens_q, cu_seqlens_k, softmax_scale):
        """Instance-level seam over the one runtime-call definition, so tests can
        wrap it to observe that a direct launch actually happened."""
        _launch_kernel(compiled, q, k, v, out, cu_seqlens_q, cu_seqlens_k, softmax_scale)

    def _acquire(self, q_stage, device):
        """Compile a kernel this module owns, or ``None`` to use the baseline.

        Nothing shared is read or written.  Two earlier designs did: one borrowed
        ``_flash_attn_fwd.compile_cache`` and ``_TUNING_CONFIG`` inside a
        ``try/finally`` (restoring a global cannot un-publish it), and one called
        ``_flash_attn_fwd`` for a reference answer -- which inserts its compiled
        callable into that same shared cache whenever the key is cold
        (``cute/interface.py``, the ``compile_key not in ...`` branch).  Neither is
        acceptable, so the kernel is constructed here, its tuning is set on the
        *instance*, and ``cute.compile`` is called directly.

        The price of not going through vLLM's prologue is that the constructor
        arguments and the compile-argument layout are reproduced by hand, and a
        silently different kernel would be a wrong answer rather than a crash.  So
        a new kernel is validated against ground truth before it is adopted, by
        ``_validated_layout`` below -- not against ``_flash_attn_fwd``, which cannot
        be consulted without writing to shared state.

        Only ever reached for the exact captured domain: bf16,
        ``head_dim == head_dim_v == 72``, ``num_heads == num_kv_heads``, varlen,
        non-causal, unsplit, unpaged, single-device, no optional features (see
        ``_admits_direct_launch``).  Every constructor value below is the value
        vLLM's prologue derives for that domain.
        """
        if _Sm100Fwd is None or not _validated_layout(q_stage, device):
            return None
        return _compile_kernel(q_stage, device, tuned=True)

    # ------------------------------------------------------------------
    # Fallback: the baseline body, unchanged
    # ------------------------------------------------------------------
    def _baseline_forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, **kwargs):
        # vLLM's wrapper takes keyword args in a different order than the
        # standard flash_attn signature.  With a ``block_table`` the
        # kernel needs per-sequence ``seqused_k`` rather than cumulative
        # ``cu_seqlens_k``.
        fa_kw = dict(
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            fa_version=self.fa_version,
        )
        if kwargs.get("block_table") is not None:
            seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            fa_kw["seqused_k"] = seqused_k
            if (
                self.fa_version == 3
                and not torch.cuda.is_current_stream_capturing()
            ):
                page_size = k.shape[1] if k.dim() >= 2 else None
                meta = fa3_scheduler_metadata(
                    batch_size=int(seqused_k.shape[0]),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    num_heads_q=self.num_heads,
                    num_heads_kv=self.num_kv_heads,
                    headdim=self.head_dim,
                    cache_seqlens=seqused_k,
                    qkv_dtype=q.dtype,
                    cu_seqlens_q=cu_seqlens_q,
                    page_size=page_size,
                    causal=kwargs.get("causal", True),
                    window_size=kwargs.get("window_size", (-1, -1)),
                    num_splits=0,
                )
                if meta is not None:
                    fa_kw["scheduler_metadata"] = meta
                fa_kw["num_splits"] = 0
        else:
            fa_kw["cu_seqlens_k"] = cu_seqlens_k
            # Dense prefill is compute-bound, so KV-splitting buys nothing, but
            # the FA4 (SM100 CuTe) auto heuristic still picks the split-KV kernel
            # for mid-size seqlens -- and that variant fails to compile in this
            # vLLM build (TYPE_UNSTABLE_JOIN on ``n_block_first``).  Pin
            # ``num_splits=1`` so the unsplit kernel is used.
            fa_kw["num_splits"] = 1
        fa_kw.update(kwargs)
        return flash_attn_varlen_func(q, k, v, **fa_kw)
