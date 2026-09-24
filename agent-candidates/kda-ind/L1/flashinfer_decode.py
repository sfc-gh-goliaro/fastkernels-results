"""TRTLLM-gen paged attention decode kernel (via FlashInfer, Blackwell only).

Same interface as ``FlashAttnDecode`` / the baseline ``TRTLLMDecode``, so
``LlamaAttention`` can dispatch to either backend without branch logic.

This version reaches the trtllm-gen decode cubin by two routes:

* a **direct dispatch** that calls
  ``get_trtllm_gen_fmha_module().trtllm_paged_attention_decode`` with the same 30
  positional arguments ``trtllm_batch_decode_with_kv_cache`` would pass, skipping
  the wrapper's validation and dispatch body. That removes 0.6-1.3 us of per-call
  Python dispatch, a measured 1.07-1.10x on host cost alone; the launched kernel
  and its output are identical. It is not a throughput win and does not show up in
  a benchmark whose timed region is dominated by input staging;
* the **wrapper itself**, called exactly the way the baseline calls it, for every
  input outside the direct route's validated envelope.

The two hazards this structure introduces, and how each is contained:

**The positional order is an unversioned ABI.** ``trtllm_paged_attention_decode``
is a ``tvm_ffi.core.Function``. It exposes no signature and no schema:
``inspect.signature`` reports ``(*args, **kwargs)``, there is no ``_schema`` /
``schema`` / ``__name__``, and ``dir()`` yields only ``release_gil`` and
``same_as``. So the argument count and the argument names cannot be verified at
all -- and even if the arity were readable, checking it could not detect the
failure that actually matters, a *reordering* of 30 arguments that keeps the same
count and silently corrupts every output. Two gates replace the unavailable
introspection, in this order:

1. ``flashinfer.__version__`` must be in ``_VALIDATED_FLASHINFER_VERSIONS`` -- a
   list of versions whose ``trtllm_batch_decode_with_kv_cache`` body has been read
   and whose positional order is the one encoded below. This is a string compare,
   so an unvetted release never gets a native call issued against it.
2. A one-time self-check runs the same decode through *both* routes and enables the
   direct one only if ``torch.equal`` holds. This catches silent divergence, which
   an arity check never could.

   **That self-check runs in a separate interpreter, not here.** Validating the ABI
   means deliberately exercising an argument order that might be wrong, and a bad
   launch can leave a sticky CUDA error behind -- the same reason ``forward`` never
   wraps the direct call in ``except``. Catching such a failure in this process and
   then continuing through the wrapper would be exactly the unsound recovery this
   module argues against elsewhere: the wrapper's own launch would fail too, at
   some later and unrelated synchronization point. So the parent resolves the
   handle but never calls it. A fresh child interpreter -- spawned, never forked
   after CUDA initialization, with its own scratch buffer -- runs the comparison,
   synchronizes, and reports one verdict. Mismatch, an exception, a crash, a
   signal, a timeout, or any unrecognized output all mean "direct route disabled",
   and in every one of those cases the poisoned context died with the child. Only a
   clean, bit-equal child exit lets this process issue the call.

   The verdict is memoized per (flashinfer version, device), so the child runs once
   per process rather than once per module.

Either gate failing, or the JIT build failing, leaves the module working through
the wrapper alone; it never makes construction fail, and it is logged once per
module naming the gate that rejected it.

**Path selection has to happen before the launch, not after a failure.** An
invalid launch can leave the CUDA context in an error state, so "call it and fall
back in ``except``" is not a recovery strategy -- the fallback call would fail
too, and the error surfaces later at an unrelated synchronization point. Every
predicate in ``_within_direct_dispatch_envelope`` therefore reads only
Python-visible metadata (dtype, rank, device capability, argument type), adds no
device-to-host synchronization of its own, and is fully evaluated before the
direct call is issued. Anything unrecognized -- an FP8 or NVFP4 cache, a 3-D page
table, a tensor-valued scale, a non-Blackwell device -- routes to the wrapper,
which then raises the same Python error the baseline would have raised, with no
kernel launched.

The four preprocessing statements that run *ahead* of the predicate do touch the
device -- resolving ``max_seq_len`` from ``cache_seqlens`` is a host sync, and the
three ``.contiguous()`` calls plus the sink conversion can each launch a copy --
but every one of them is a statement the baseline also executes, in the same
order, for the same reason. They are inherited deliberately: they are where the
baseline raises for a missing ``block_table`` or a missing ``cache_seqlens``, and
moving the predicate ahead of them would change which error a caller sees.

Two differences from the baseline are accepted rather than eliminated, because
removing either would defeat the purpose:

* construction does device work the baseline's does not -- the self-check runs two
  decodes on its own scratch buffer. It leaves the caller's workspace untouched,
  but it is not free and it is not silent;
* a forward that takes the direct route does not pass through FlashInfer's
  ``@flashinfer_api`` decorator, so with ``FLASHINFER_LOGLEVEL`` above 0 the
  baseline emits a log/trace record for the call and this module does not. At the
  default level 0 the decorator returns the wrapped function unchanged and there
  is nothing to miss.
"""

import logging
import os
import subprocess
import sys

import torch
import torch.nn as nn
from flashinfer import __version__ as _flashinfer_version
from flashinfer.decode import (
    get_trtllm_gen_fmha_module,
    trtllm_batch_decode_with_kv_cache,
)
from flashinfer.utils import device_support_pdl, get_device_sm_count

logger = logging.getLogger(__name__)

# Versions whose ``trtllm_batch_decode_with_kv_cache`` body was read and whose
# positional call order matches ``_direct_decode`` below. Checked before any
# native call, so a version bump falls back instead of misinterpreting arguments.
_VALIDATED_FLASHINFER_VERSIONS = frozenset({"0.6.14"})

# An allowlist, not a denylist of uint8 / float8: an unanticipated cache dtype
# must fall back rather than be launched with the wrong element size assumed.
_DIRECT_DISPATCH_DTYPES = (torch.bfloat16, torch.float16)

# trtllm-gen is the backend ``trtllm_batch_decode_with_kv_cache`` picks for
# compute capability 10.x; on anything else it dispatches to xqa instead, which
# takes different arguments entirely.
_BLACKWELL_CC_MAJOR = 10

# The launcher sub-allocates an 8 MiB ``trtllm_gen_counter_workspace`` out of the
# buffer it is handed, so the self-check needs its own comfortably above that.
_SELF_CHECK_WORKSPACE_BYTES = 32 * 1024 * 1024

# Set in the child interpreter that runs the ABI self-check, so a child can never
# recursively spawn another one.
_SELF_CHECK_CHILD_ENV = "FLASHINFER_DECODE_ABI_SELF_CHECK_CHILD"

# The child has to import torch and load the trtllm-gen module, and on a cold
# cubin cache it also downloads kernels. Generous on purpose: a spurious timeout
# costs the direct route for the whole process, while a real hang costs this wait
# exactly once.
_SELF_CHECK_TIMEOUT_S = float(os.environ.get("FLASHINFER_DECODE_SELF_CHECK_TIMEOUT", "600"))

_SELF_CHECK_OK = "ABI_SELF_CHECK_OK"
_SELF_CHECK_MISMATCH = "ABI_SELF_CHECK_MISMATCH"

# (flashinfer version, device index, device name) -> (validated, reason). The
# child is expensive; the ABI it validates does not change within a process.
_ABI_VERDICTS: dict[tuple, tuple[bool, str]] = {}


def prime_trtllm_sinks(module: nn.Module, sinks: torch.Tensor | None) -> None:
    """Materialize the FP32 attention-sink copy the trtllm-gen kernels need.

    ``trtllm_batch_decode_with_kv_cache`` /
    ``trtllm_batch_context_with_kv_cache`` hard-check
    ``attention_sinks.dtype == float32``, while the FlashAttention build vLLM
    bundles asserts the opposite for the same weights
    (``learnable_sink must be bfloat16``).  So the conversion cannot live on the
    layer -- only the op knows which kernel it is about to call.  vLLM does the
    same conversion once per layer in
    ``FlashInferImpl.process_weights_after_loading``; call this from the owning
    attention layer's post-load hook so the copy never lands inside a forward
    or a CUDA-graph capture.
    """
    if sinks is None:
        module._sinks_fp32 = None
    elif sinks.dtype == torch.float32:
        module._sinks_fp32 = sinks
    else:
        module._sinks_fp32 = sinks.detach().to(torch.float32)
    module._sinks_src = sinks


def trtllm_sinks(module: nn.Module, s_aux: torch.Tensor | None):
    """Return the FP32 view of ``s_aux``, priming the cache if needed."""
    if s_aux is None or s_aux.dtype == torch.float32:
        return s_aux
    if module._sinks_fp32 is None or module._sinks_src is not s_aux:
        prime_trtllm_sinks(module, s_aux)
    return module._sinks_fp32


def _direct_decode(op, out, q, k_cache, v_cache, workspace, block_table,
                   cache_seqlens, max_seq_len, bmm1_scale, window_left, sinks):
    """Invoke the trtllm-gen decode cubin the way FlashInfer itself invokes it.

    The 30 positional arguments, and the literals among them, mirror the
    ``run_func(...)`` call at the end of ``trtllm_batch_decode_with_kv_cache``'s
    ``trtllm-gen`` branch for every version in
    ``_VALIDATED_FLASHINFER_VERSIONS``. The literals are not arbitrary defaults:

    * ``out_scale_factor=None`` and ``o_sf_start_index=0`` are what the wrapper
      uses on the non-NVFP4 output path;
    * ``max_q_len=1`` and ``batch_size=q.shape[0]`` follow from
      ``q_len_per_req=1``, which is the wrapper's default;
    * ``o_sf_scale`` and ``o_sf_vec_size`` reach the op as ``o_sf_scale or -1.0``
      and ``o_sf_vec_size or -1``, i.e. a *float* -1.0 and an *int* -1. The two
      types are not interchangeable across the FFI boundary;
    * ``bmm1_scale`` is forwarded unscaled. The wrapper multiplies by ``log2e``
      only when the scale is a ``torch.Tensor``; applying it to a Python float
      would silently change every output value, which is why a tensor scale is
      kept out of the envelope rather than converted here;
    * ``sm_count`` and ``enable_pdl`` are read from ``q.device`` on every call,
      exactly as the wrapper reads them, so a module built on one device and
      called with a query on another still agrees with the baseline. Both
      underlying helpers are memoized, so this costs ~0.07 us;
    * ``workspace_bytes`` is likewise recomputed from the buffer in hand, so
      swapping ``_workspace`` between calls is picked up.
    """
    op(
        out,                    # out
        None,                   # out_scale_factor (NVFP4 output only)
        q,                      # query
        k_cache,
        v_cache,
        workspace,              # workspace_buffer
        block_table,            # block_tables
        cache_seqlens,          # seq_lens
        1,                      # max_q_len, from q_len_per_req=1
        max_seq_len,
        bmm1_scale,
        1.0,                    # bmm2_scale
        -1.0,                   # o_sf_scale, float
        -1,                     # o_sf_vec_size, int
        0,                      # o_sf_start_index
        q.shape[0],             # batch_size, from q_len_per_req=1
        window_left,
        0,                      # sparse_mla_top_k
        get_device_sm_count(q.device),
        device_support_pdl(q.device),
        workspace.numel() * workspace.element_size(),
        sinks,
        None,                   # cum_seq_lens_q
        None,                   # k_block_scales (NVFP4 cache only)
        None,                   # v_block_scales (NVFP4 cache only)
        None,                   # skip_softmax_threshold_scale_factor
        True,                   # uses_shared_paged_kv_idx
        None,                   # lse
        0,                      # lse_stride_tokens
        0,                      # lse_stride_heads
    )


def _within_direct_dispatch_envelope(q, k_cache, v_cache, block_table, bmm1_scale) -> bool:
    """Whether the direct call is known-equivalent to the wrapper for these inputs.

    Metadata only: no tensor element is read, and nothing here synchronizes with
    the device, so this is decided before anything is launched and adds no
    ordering the baseline does not already have. ``q.is_cuda`` guards the
    capability query rather than adding a condition -- a non-CUDA device has no
    compute capability to compare, and letting the wrapper reject it reproduces
    the baseline's error exactly.
    """
    return (
        q.is_cuda
        and torch.cuda.get_device_capability(q.device)[0] == _BLACKWELL_CC_MAJOR
        and k_cache.dtype == v_cache.dtype
        and k_cache.dtype in _DIRECT_DISPATCH_DTYPES
        and q.ndim == 3
        and q.dtype in _DIRECT_DISPATCH_DTYPES
        and block_table.ndim == 2
        and not isinstance(bmm1_scale, torch.Tensor)
    )


def _self_check_sample(device):
    """The decode the ABI self-check compares, as a keyword dict.

    Chosen so the positional slots hold *mutually distinct* values wherever two
    slots could be confused, because that is the only thing that gives a
    value-based check any power against a same-arity reordering:

    * ``batch=3`` separates ``batch_size`` (3) from ``max_q_len`` (1);
    * ``window_left=24`` separates it from ``o_sf_vec_size`` (-1) and from
      ``o_sf_scale`` (-1.0);
    * ``max_seq_len=40`` separates it from ``max_q_len`` (1);
    * a real FP32 ``sinks`` tensor separates that slot from the block of ``None``
      arguments after it, and exercises the sink path at the same time;
    * three unequal sequence lengths over a shared page table make the page walk
      order observable.

    What remains undetectable is a swap between two slots that hold *equal*
    values -- the five ``None`` arguments among themselves, the four zeros
    (``o_sf_start_index``, ``sparse_mla_top_k``, and the two LSE strides), or the
    two ``True`` flags. Such a swap is inert by construction, so nothing is lost.
    A reordering of any two slots whose values differ changes the result and is
    caught. This is strictly stronger than the arity check that
    ``tvm_ffi.core.Function`` makes impossible, which would have passed every one
    of these reorderings.

    Values come from a private ``torch.Generator`` rather than the global RNG, so
    running the check perturbs no caller's random stream.
    """
    num_qo_heads, head_dim, page_size, num_pages = 16, 128, 16, 4
    gen = torch.Generator(device=device).manual_seed(0x7A17)
    cache_shape = (num_pages, 1, page_size, head_dim)
    return dict(
        q=torch.randn(3, num_qo_heads, head_dim, dtype=torch.bfloat16,
                      device=device, generator=gen),
        k_cache=torch.randn(cache_shape, dtype=torch.bfloat16, device=device, generator=gen),
        v_cache=torch.randn(cache_shape, dtype=torch.bfloat16, device=device, generator=gen),
        sinks=torch.randn(num_qo_heads, dtype=torch.float32, device=device, generator=gen),
        block_table=torch.tensor([[0, 1, 2], [3, 0, 1], [2, 3, 0]],
                                 dtype=torch.int32, device=device),
        cache_seqlens=torch.tensor([40, 16, 33], dtype=torch.int32, device=device),
        max_seq_len=40,
        window_left=24,
        bmm1_scale=head_dim ** -0.5,
    )


def _compare_routes(op, device) -> bool:
    """Run :func:`_self_check_sample` both ways and report bitwise agreement.

    **Only ever called in a child interpreter** (see the module docstring): it
    invokes an argument order that has not been validated yet, which is precisely
    the call that can poison a CUDA context. Its scratch buffer is its own, so no
    caller's workspace is touched even in the child.

    The sample is checked against :func:`_within_direct_dispatch_envelope` before
    the direct call, so a device that could never take the direct route -- a CPU
    tensor, a Hopper or SM120 GPU -- reports failure without issuing a native call
    at all.
    """
    s = _self_check_sample(device)
    if not _within_direct_dispatch_envelope(
            s["q"], s["k_cache"], s["v_cache"], s["block_table"], s["bmm1_scale"]):
        return False
    workspace = torch.zeros(_SELF_CHECK_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    direct = torch.empty_like(s["q"])
    _direct_decode(op, direct, s["q"], s["k_cache"], s["v_cache"], workspace,
                   s["block_table"], s["cache_seqlens"], s["max_seq_len"],
                   s["bmm1_scale"], s["window_left"], s["sinks"])
    reference = trtllm_batch_decode_with_kv_cache(
        query=s["q"], kv_cache=(s["k_cache"], s["v_cache"]), workspace_buffer=workspace,
        block_tables=s["block_table"], seq_lens=s["cache_seqlens"],
        max_seq_len=s["max_seq_len"], bmm1_scale=s["bmm1_scale"], bmm2_scale=1.0,
        window_left=s["window_left"], sinks=s["sinks"], kv_layout="HND",
    )
    # Force the launches to complete here, in the child, so a fault surfaces as a
    # dead child rather than as a mystery at the parent's next synchronization.
    torch.cuda.synchronize(device)
    return torch.equal(direct, reference)


def _abi_self_check_command(device_index: int) -> list[str]:
    """The child interpreter that validates the positional ABI.

    A fresh ``sys.executable`` rather than a forked worker: forking a process that
    has already initialized CUDA gives the child a broken context, and the point
    here is to get a *clean* one whose death costs the parent nothing. Overridden
    in tests to inject each child failure mode deterministically.
    """
    return [sys.executable, os.path.abspath(__file__), "--abi-self-check", str(device_index)]


def _abi_validated_by_child(device_index: int) -> tuple[bool, str]:
    """Whether an isolated interpreter confirmed the positional ABI bit-for-bit.

    Anything other than a clean exit carrying the agreement token counts as "not
    validated": a mismatch, an exception, a crash or signal, a timeout, or output
    this function does not recognize. The parent never distinguishes "the ABI is
    wrong" from "the child died trying to find out", because both have the same
    consequence -- do not issue that call here.
    """
    if os.environ.get(_SELF_CHECK_CHILD_ENV):
        return False, "refusing to nest ABI self-check children"
    env = dict(os.environ)
    env[_SELF_CHECK_CHILD_ENV] = "1"
    try:
        child = subprocess.run(_abi_self_check_command(device_index), env=env,
                               capture_output=True, text=True,
                               timeout=_SELF_CHECK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"the isolated check timed out after {_SELF_CHECK_TIMEOUT_S:g}s"
    except OSError as exc:
        return False, f"could not start an isolated interpreter ({exc})"
    # The protocol is one token and nothing else, and it is enforced as written: a
    # membership test would accept "garbage ABI_SELF_CHECK_OK extra", and worse, a
    # child that printed the success token *and* the mismatch token. Either would
    # authorize the parent-process call this whole gate exists to withhold, so the
    # comparison is against the exact token list. Mismatch is examined first, so a
    # self-contradictory child is reported as a disagreement rather than as noise.
    tokens = child.stdout.split()
    if _SELF_CHECK_MISMATCH in tokens:
        return False, "the isolated check did not match the wrapper bitwise"
    if child.returncode < 0:
        return False, f"the isolated check died on signal {-child.returncode}"
    if child.returncode == 0 and tokens == [_SELF_CHECK_OK]:
        return True, "an isolated interpreter reproduced the wrapper bit-for-bit"
    detail = (child.stderr or "").strip().splitlines()
    tail = detail[-1] if detail else "no diagnostic output"
    if child.returncode != 0:
        return False, f"the isolated check exited {child.returncode}: {tail}"
    if tokens:
        return False, f"the isolated check produced unrecognized output: {tokens[:4]!r}"
    return False, "the isolated check produced no verdict"


def _validate_abi(device) -> tuple[bool, str]:
    """Memoized :func:`_abi_validated_by_child`, keyed by version and device."""
    if device.type != "cuda":
        return False, f"{device} cannot take the direct route"
    index = 0 if device.index is None else device.index
    key = (_flashinfer_version, index, torch.cuda.get_device_name(index))
    if key not in _ABI_VERDICTS:
        _ABI_VERDICTS[key] = _abi_validated_by_child(index)
    return _ABI_VERDICTS[key]


class TRTLLMDecode(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        if workspace is None:
            workspace = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self._workspace = workspace
        self._sinks_fp32: torch.Tensor | None = None
        self._sinks_src: torch.Tensor | None = None
        self._decode_op = self._resolve_decode_op(self._workspace.device)

    def _resolve_decode_op(self, device):
        """Return the trtllm-gen decode handle, or None to use the wrapper only.

        Three gates, cheapest first, and **this process issues no native call on
        any of these paths**:

        1. the version allowlist -- a string compare, so an unvetted flashinfer
           release is rejected without a JIT build and without a launch;
        2. handle resolution -- guarded, because the module must stay usable
           through the wrapper on a host where the JIT build cannot complete. This
           loads the cubin but does not call it;
        3. the isolated ABI comparison -- run in a child interpreter, whose death
           by any cause is reported as "not validated" and leaves this context
           untouched. See the module docstring for why catching that failure here
           instead would be unsound.

        Each rejection is reported once, here in construction, and names its gate;
        nothing is logged per forward.
        """
        if _flashinfer_version not in _VALIDATED_FLASHINFER_VERSIONS:
            logger.warning(
                "TRTLLMDecode: direct trtllm-gen dispatch disabled by the version "
                "allowlist (flashinfer %s is not one of %s); using "
                "trtllm_batch_decode_with_kv_cache. Read the positional argument "
                "order in this version's trtllm_batch_decode_with_kv_cache and add "
                "it to _VALIDATED_FLASHINFER_VERSIONS if it still matches.",
                _flashinfer_version, sorted(_VALIDATED_FLASHINFER_VERSIONS),
            )
            return None
        try:
            op = get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
        except Exception:
            logger.warning(
                "TRTLLMDecode: direct trtllm-gen dispatch disabled at op-handle "
                "resolution; using trtllm_batch_decode_with_kv_cache.",
                exc_info=True,
            )
            return None
        validated, reason = _validate_abi(device)
        if not validated:
            logger.warning(
                "TRTLLMDecode: direct trtllm-gen dispatch disabled by the isolated "
                "self-check on flashinfer %s (%s); using "
                "trtllm_batch_decode_with_kv_cache. The positional argument order "
                "in _direct_decode may no longer describe this build.",
                _flashinfer_version, reason,
            )
            return None
        return op

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        prime_trtllm_sinks(self, sinks)

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, s_aux=None, window_size=None, **kwargs):
        if max_seq_len is None:
            max_seq_len = int(cache_seqlens.max().item())
        # trtllm-gen requires a contiguous query: with a batched (multi-request)
        # decode, the query view is non-contiguous and the TMA load reads later
        # rows at the wrong stride -> only row 0 is correct, the rest are garbage.
        # vLLM's FlashInfer backend and our own TRTLLMPrefill both do this; the
        # decode path was missing it.
        q = q.contiguous()
        # block_tables / seq_lens MUST be contiguous: the trtllm-gen kernel
        # reads the page table assuming a dense [batch, max_pages] row-major
        # layout. The engine's eager/CUDA-graph decode buffers hand us a column
        # slice (``_eager_block_tables[:n, :bt_cols]``) whose row stride is the
        # full ``max_num_blocks``, not ``bt_cols`` -> every row > 0 would read
        # its page ids from the wrong offset (garbage pages), so only row 0
        # stayed correct and all other sequences in the batch were corrupted.
        # vLLM likewise asserts is_strictly_contiguous(block_tables/seq_lens).
        block_table = block_table.contiguous()
        if cache_seqlens is not None:
            cache_seqlens = cache_seqlens.contiguous()
        # These four statements come first, in this order, on both routes: they
        # are where the baseline raises for a missing block_table or a missing
        # cache_seqlens-with-no-max_seq_len, and testing the envelope ahead of
        # them would turn those errors into something else.

        # Attention sinks and the sliding window must be forwarded explicitly.
        # The caller names them ``s_aux`` / ``window_size`` (the FlashAttention
        # spelling); trtllm-gen calls them ``sinks`` / ``window_left``. Letting
        # them fall into **kwargs silently dropped both, which is a *numerical*
        # bug, not a crash: gpt-oss-120b (sinks + alternating sliding window)
        # scored 0.8 of 385 matching tokens against vLLM. vLLM passes both here
        # (flashinfer.py: window_left=self.window_left, sinks=self.sinks).
        # Evaluated in the order the baseline's call site evaluates them, so a
        # caller who passes two bad arguments still sees the same one reported.
        bmm1_scale = softmax_scale if softmax_scale is not None else self.sm_scale
        window_left = (
            window_size[0] if window_size is not None
            and window_size[0] >= 0 else -1
        )
        sinks = trtllm_sinks(self, s_aux)

        if self._decode_op is not None and _within_direct_dispatch_envelope(
                q, k_cache, v_cache, block_table, bmm1_scale):
            # Allocated from the normalized query, not the caller's: the wrapper
            # allocates after its own conversions, and taking empty_like off a
            # non-contiguous input would hand back non-baseline output strides.
            # A fresh buffer per call, never a reused one, so a caller may hold
            # on to an earlier result.
            out = torch.empty_like(q)
            _direct_decode(self._decode_op, out, q, k_cache, v_cache,
                           self._workspace, block_table, cache_seqlens,
                           max_seq_len, bmm1_scale, window_left, sinks)
            return out

        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=cache_seqlens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            window_left=window_left,
            sinks=sinks,
            kv_layout="HND",
        )


def _abi_self_check_main(argv: list[str]) -> int:
    """Child entry point: validate the positional ABI and report one verdict.

    Runs in its own interpreter with its own CUDA context, so if the argument order
    below turns out to be wrong for this build, the resulting fault kills this
    process instead of poisoning the caller's. Prints a single token the parent
    looks for and says nothing else on stdout.
    """
    if len(argv) != 1:
        print(f"usage: {os.path.basename(__file__)} --abi-self-check <device index>",
              file=sys.stderr)
        return 2
    device = torch.device("cuda", int(argv[0]))
    op = get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
    if _compare_routes(op, device):
        print(_SELF_CHECK_OK)
        return 0
    print(_SELF_CHECK_MISMATCH)
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--abi-self-check":
        raise SystemExit(_abi_self_check_main(sys.argv[2:]))
    raise SystemExit(
        f"{os.path.basename(__file__)} is an operator module, not a program; "
        f"its only entry point is --abi-self-check <device index>"
    )
