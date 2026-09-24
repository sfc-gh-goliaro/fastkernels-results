"""FP8 linear (block-scaled FP8 matrix multiply) with vLLM-parity dispatch.

Registers the FP8 GEMM as a ``torch.library`` custom op so it stays
**opaque** during ``torch.compile`` tracing (Inductor does not attempt
to inline or fuse it).  At runtime the real DeepGEMM kernel executes —
matching vLLM's approach of using ``torch.ops.vllm.fp8_gemm_nt_op``.

For DeepSeek-V3.2 dense FP8 linears (block-scaled, BF16-in, FP8 weight,
BF16-out), vLLM's ``W8A8BlockFp8LinearOp.apply`` uses a two-tier dispatch:
  * ``M < 32``: FlashInfer's ``fp8_blockscale_gemm_sm90`` (swapAB kernel
    with **internal** activation quantization).  Required for accuracy
    parity with vLLM in low-batch decode and small prefill.
  * ``M >= 32``: DeepGEMM ``fp8_gemm_nt`` with **external** activation
    quantization (``QuantFP8`` -> column-major scales -> ``fp8_gemm_nt``
    with ``disable_ue8m0_cast=not is_deep_gemm_e8m0_used()``).

Mirroring this dispatch is necessary for bit-equivalence with vLLM:
otherwise a single FP8 linear at layer 0 already drifts by ``max|Δ|≈0.25``
on a 16-token batch, which compounds into expert-selection mismatches in
the noaux_tc grouped-topk path of every subsequent MoE layer.
"""

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.infra.cuda_ext import lazy_op

# Specialized per-token-group FP8 quantizer (group_size=128, UE8M0 scales,
# register-resident, both scale layouts).  Bit-identical to the vendored vLLM
# ``per_token_group_quant_8bit`` kernel it replaces -- see fp8_quant_fast.cu.
_QUANT_EXT = lazy_op("fp8_quant_fast", "fp8_quant_fast.cu")

# Fused quantize+GEMV for the M == 1 (decode) row: one GPU op for the whole
# forward instead of quantize + GEMM.  See fp8_gemv_fused.cu.
_GEMV_EXT = lazy_op("fp8_gemv_fused", "fp8_gemv_fused.cu")

# DeepGEMM provides the FP8 fast path on Hopper+.
import deep_gemm

from .fp8_grouped_gemm_contiguous import _is_deep_gemm_e8m0_used

_DG_FP8_GEMM_NT = deep_gemm.fp8_gemm_nt

# ---------------------------------------------------------------------------
# Two DeepGEMM globals were measured this round and both are deliberately left
# alone (see ITERATIONS.md iters 02-04):
#
# * ``set_pdl(True)`` -- off by default, so the programmatic completion our
#   quantizer triggers goes unused, and turning it on is a real 2-4 us win in
#   absolute terms.  But it is process-global and the harness builds baseline and
#   candidate in one worker, so the baseline gets it too -- and gains *more*,
#   having one extra op to overlap: ratio 1.237 -> 1.175x at [344,4096],
#   1.276 -> 1.148x at [1000,2048].
# * ``set_block_size_multiple_of((64, 64))`` -- lets the largest row use 64-wide
#   M/N tiles (GEMM-only 105.5 -> 97.4 us), bit-identical output, and a +0.3%
#   geomean in a back-to-back A/B.  It did not survive a bench run (our own
#   [16384,4096] time went 177.2 -> 183.3 us), which is inside this harness's
#   run-to-run spread either way, so the fallback path is left exactly as r1
#   proved it.


_FP8_INFO = torch.finfo(torch.float8_e4m3fn)
_FP8_MIN = _FP8_INFO.min
_FP8_MAX = _FP8_INFO.max
_GROUP_SIZE: tl.constexpr = 128

# Hoisted out of the per-call path: attribute chains on ``torch``/``deep_gemm``
# and a ``torch.compiler`` module lookup are a measurable fraction of the
# few-microsecond shapes.
_is_compiling = torch.compiler.is_compiling

# Match vLLM's ``per_token_group_quant_fp8`` default:
# ``vllm/.../fp8_utils.py:860`` ``eps: float = 1e-10``.  The previous
# value (1e-12) caused the per-token scale of all-zero rows to differ
# from vLLM by exactly 100x, which propagates as a constant offset into
# the FP8 GEMM output and contributes to layer-0 divergence.
_QUANT_EPS = 1e-10


# ---------------------------------------------------------------------------
# FlashInfer FP8 blockscale GEMM (M < 32 swapAB) - resolved lazily
# ---------------------------------------------------------------------------

def _is_batch_invariant() -> bool:
    """vLLM's dynamic FP8 blockscale dispatch forces the DeepGEMM path (skips the
    FlashInfer swapAB kernel) for ALL M under batch-invariant determinism mode
    (``VLLM_BATCH_INVARIANT=1``) — see the early-out in
    ``scaled_mm/flashinfer.py`` and ``grouped_topk._is_batch_invariant``. Mirror
    it so fastkernels matches vLLM in that mode."""
    return os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"


_FLASHINFER_RESOLVED = False
_FLASHINFER_FN: object | None = None


def _maybe_get_flashinfer_fp8_gemm():
    """Return ``flashinfer.gemm.fp8_blockscale_gemm_sm90`` if importable +
    enabled by env, otherwise ``None``.

    Mirrors vLLM's enablement gate in
    ``vllm/utils/flashinfer.py:is_flashinfer_fp8_blockscale_gemm_supported``:
    ``VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER`` (default 1) AND
    ``has_flashinfer_fp8_blockscale_gemm()`` (Hopper + flashinfer wheel
    exposes ``fp8_blockscale_gemm_sm90``).
    """
    global _FLASHINFER_RESOLVED, _FLASHINFER_FN
    if _FLASHINFER_RESOLVED:
        return _FLASHINFER_FN
    _FLASHINFER_RESOLVED = True

    if os.environ.get("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "1") != "1":
        return None
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability()
    if cap[0] != 9:  # Hopper only — same gate as vLLM.
        return None
    from flashinfer.gemm import fp8_blockscale_gemm_sm90
    _FLASHINFER_FN = fp8_blockscale_gemm_sm90
    return _FLASHINFER_FN

# ---------------------------------------------------------------------------
# Register FP8 GEMM as torch.library custom ops (opaque to Inductor).
# This mirrors vLLM's direct_register_custom_op for fp8_gemm_nt_op.
# ---------------------------------------------------------------------------

# This module is loaded as ``fastkernels.tasks.candidate.L1.fp8_linear``
# alongside the baseline, which owns the ``fastkernels_fp8`` namespace; a
# second ``TORCH_LIBRARY`` on the same namespace is a hard error, so the
# candidate's opaque ops live in their own namespace.
_FK_NS = "fastkernels_fp8_cand"
_fp8_lib = torch.library.Library(_FK_NS, "DEF")

_fp8_lib.define(
    "fp8_gemm_nt(Tensor q_input, Tensor input_scale, "
    "Tensor weight, Tensor weight_scale, Tensor! output) -> ()"
)


def _fp8_gemm_nt_impl(q_input, input_scale, weight, weight_scale, output):
    # vLLM passes ``is_deep_gemm_e8m0_used`` explicitly so DeepGEMM's
    # ``disable_ue8m0_cast`` flag matches the SF format used at quantization
    # time (see ``vllm/utils/deep_gemm.py:fp8_gemm_nt`` -> forwards
    # ``disable_ue8m0_cast=not use_ue8m0`` to ``_fp8_gemm_nt_impl``). Without
    # this, DeepGEMM falls back to its module-level default which can re-cast
    # the SFs and silently change the GEMM result by ~1e-3 absolute.
    _DG_FP8_GEMM_NT(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        disable_ue8m0_cast=not _use_ue8m0(),
    )


_fp8_lib.impl("fp8_gemm_nt", _fp8_gemm_nt_impl, "CUDA")


@torch.library.impl(_fp8_lib, "fp8_gemm_nt", "Meta")
def _fp8_gemm_nt_meta(q_input, input_scale, weight, weight_scale, output):
    pass


# ---------------------------------------------------------------------------
# FlashInfer FP8 blockscale GEMM custom op (BF16 in, FP8 weight, BF16 out).
# Used for M < 32 so fastkernels picks the same swapAB kernel as vLLM's
# ``W8A8BlockFp8LinearOp.apply`` (see ``vllm/.../fp8_utils.py:402-407``).
# ---------------------------------------------------------------------------

_fp8_lib.define(
    "flashinfer_blockscale_gemm(Tensor input_bf16, Tensor weight_fp8, "
    "Tensor weight_scale, Tensor! output) -> ()"
)


def _flashinfer_blockscale_gemm_impl(input_bf16, weight_fp8, weight_scale,
                                     output):
    fn = _maybe_get_flashinfer_fp8_gemm()
    assert fn is not None, "FlashInfer FP8 blockscale GEMM not available"
    fn(
        input=input_bf16,
        weight=weight_fp8,
        input_scale=None,  # internal quantization
        weight_scale=weight_scale,
        out=output,
        out_dtype=torch.bfloat16,
    )


_fp8_lib.impl("flashinfer_blockscale_gemm", _flashinfer_blockscale_gemm_impl,
              "CUDA")


@torch.library.impl(_fp8_lib, "flashinfer_blockscale_gemm", "Meta")
def _flashinfer_blockscale_gemm_meta(input_bf16, weight_fp8, weight_scale,
                                     output):
    pass


_fp8_lib.define(
    "per_token_group_quant_fp8(Tensor input, Tensor! output_fp8, "
    "Tensor! output_scale, bool column_major_scales=False) -> ()"
)


def _per_token_group_quant_fp8_op_impl(input, output_fp8, output_scale,
                                       column_major_scales=False):
    _per_token_group_quant_fp8(
        input, output_fp8, output_scale,
        column_major_scales=column_major_scales,
    )


_fp8_lib.impl("per_token_group_quant_fp8", _per_token_group_quant_fp8_op_impl,
              "CUDA")


@torch.library.impl(_fp8_lib, "per_token_group_quant_fp8", "Meta")
def _per_token_group_quant_fp8_op_meta(input, output_fp8, output_scale,
                                       column_major_scales=False):
    pass


@triton.jit
def _fp8_group_quant_kernel(
    x_ptr, out_ptr, scale_ptr,
    stride_x_row, stride_out_row, stride_s_row, stride_s_group,
    num_cols,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_UE8M0: tl.constexpr = True,
):
    pid = tl.program_id(0)
    groups_per_row = num_cols // GROUP_SIZE
    row = pid // groups_per_row
    group = pid % groups_per_row

    x_base = x_ptr + row * stride_x_row + group * GROUP_SIZE
    cols = tl.arange(0, GROUP_SIZE)
    x = tl.load(x_base + cols).to(tl.float32)

    absmax = tl.max(tl.abs(x))
    # Match vLLM's ``per_token_group_quant_fp8`` epsilon (1e-10, see
    # ``vllm/.../fp8_utils.py:860``).  Differs from the previous 1e-12
    # only on all-zero rows but the bias compounds across layers.
    absmax = tl.maximum(absmax, 1e-10)
    # Multiply-by-reciprocal (not division) to match vLLM's
    # ``_per_token_group_quant_fp8`` (fp8_utils.py:144): fast-division for a
    # constexpr divisor introduces a 1-ULP error that flips FP8 quantization
    # at representable-value boundaries.
    scale_raw = absmax * (1.0 / fp8_max)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale_raw))) if USE_UE8M0 else scale_raw

    x_scaled = x / scale
    x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)
    x_fp8 = x_clamped.to(out_ptr.dtype.element_ty)

    out_base = out_ptr + row * stride_out_row + group * GROUP_SIZE
    tl.store(out_base + cols, x_fp8)

    scale_base = scale_ptr + row * stride_s_row + group * stride_s_group
    tl.store(scale_base, scale)


_USE_UE8M0: bool | None = None


def _use_ue8m0() -> bool:
    """``_is_deep_gemm_e8m0_used`` re-reads ``VLLM_USE_DEEP_GEMM_E8M0`` from the
    environment on every call; the answer is a property of the build + arch, so
    resolve it once."""
    global _USE_UE8M0
    if _USE_UE8M0 is None:
        _USE_UE8M0 = _is_deep_gemm_e8m0_used()
    return _USE_UE8M0


# FlashInfer eligibility that does not depend on the call's shapes. ``None``
# until first resolved; on non-Hopper it is ``False`` and the whole vLLM-parity
# swapAB branch folds out of the hot path.
_FI_AVAILABLE: bool | None = None


def _flashinfer_available() -> bool:
    global _FI_AVAILABLE
    if _FI_AVAILABLE is None:
        _FI_AVAILABLE = _maybe_get_flashinfer_fp8_gemm() is not None
    return _FI_AVAILABLE


_QUANT_FN = None
_GEMV_FN = None


def _resolve_gemv_fn():
    """Bind the fused GEMV once (``lazy_op`` JIT-compiles on first attribute
    access, so keep the bound function instead of paying an ``__getattr__`` per
    call)."""
    global _GEMV_FN
    if _GEMV_FN is None:
        _GEMV_FN = _GEMV_EXT.fp8_gemv_fused
    return _GEMV_FN


def _resolve_quant_fn():
    """Bind the specialized CUDA quantizer once (the ``lazy_op`` handle JIT
    compiles on first attribute access, so keep the bound function around
    instead of paying an ``__getattr__`` per call)."""
    global _QUANT_FN
    if _QUANT_FN is None:
        _QUANT_FN = _QUANT_EXT.per_token_group_quant_e4m3
    return _QUANT_FN


def _per_token_group_quant_fp8(x: torch.Tensor,
                               out_fp8: torch.Tensor,
                               out_scale: torch.Tensor,
                               use_ue8m0: bool = True,
                               column_major_scales: bool = False) -> None:
    """In-place per-token-group FP8 quantization.

    When use_ue8m0=True (default), scales are rounded to powers of two
    (UE8M0 format), matching DeepGEMM dense linear expectations.
    When use_ue8m0=False, scales are plain float32 (absmax / fp8_max),
    matching vLLM's Triton MoE activation quantization.

    When column_major_scales=True, the caller has already allocated
    ``out_scale`` with column-major strides (``out_scale.stride(0) == 1``),
    matching vLLM's DeepGEMM dense linear path.  Used for ``Fp8Linear``
    so the SF layout matches DeepGEMM's expectation without a
    post-quant transpose.

    Prefers the vendored CUDA C++ kernel when available for lower launch
    overhead; falls back to Triton.
    """
    M, K = x.shape
    if use_ue8m0 and x.is_cuda and x.is_contiguous() and K % _GROUP_SIZE == 0:
        # The specialized kernel reads the scale layout (row- vs column-major)
        # straight off ``out_scale``'s strides, so ``column_major_scales`` needs
        # no separate flag here.  ``eps=1e-10`` matches vLLM's
        # ``per_token_group_quant_fp8`` default.
        _resolve_quant_fn()(
            x, out_fp8, out_scale, _QUANT_EPS, _FP8_INFO.min, _FP8_INFO.max,
        )
        return

    groups_per_row = K // _GROUP_SIZE
    _fp8_group_quant_kernel[(M * groups_per_row,)](
        x, out_fp8, out_scale,
        x.stride(0), out_fp8.stride(0),
        out_scale.stride(0), out_scale.stride(1),
        K,
        fp8_max=_FP8_INFO.max,
        GROUP_SIZE=_GROUP_SIZE,
        USE_UE8M0=use_ue8m0,
    )


class PerTokenGroupQuantFp8(nn.Module):
    """In-place per-token-group FP8 quantization (single Triton/CUDA kernel).

    Public ``nn.Module`` wrapper around the registered
    ``getattr(torch.ops, _FK_NS).per_token_group_quant_fp8`` custom op.  L2
    callers that need activation quantization outside of ``Fp8Linear``
    (e.g. ``DeepSeekMoE``, ``SparseAttnIndexer``) should use this module
    instead of importing the private ``_per_token_group_quant_fp8``.
    """

    def forward(self, x: torch.Tensor, out_fp8: torch.Tensor,
                out_scale: torch.Tensor) -> None:
        # Eager calls the CUDA entry point directly: the ``torch.library``
        # dispatch, the two Python wrapper frames and the Python-side
        # contiguity re-check together cost more than the kernel does on the
        # small captured shapes (the C++ entry point re-materializes a
        # non-contiguous ``x`` itself).  Under tracing the call still goes
        # through the opaque custom op so Inductor cannot fuse into it.
        if _is_compiling():
            getattr(torch.ops, _FK_NS).per_token_group_quant_fp8(
                x, out_fp8, out_scale,
            )
            return
        fn = _QUANT_FN
        if fn is None:
            fn = _resolve_quant_fn()
        fn(x, out_fp8, out_scale, _QUANT_EPS, _FP8_MIN, _FP8_MAX)


def _alloc_colmajor_scale(M: int, num_groups: int,
                          device: torch.device) -> torch.Tensor:
    """Allocate a per-token-group scale tensor with column-major strides.

    Layout matches vLLM's DeepGEMM dense path
    (``vllm/model_executor/layers/quantization/utils/fp8_utils.py:914-918``):
    physical storage is ``(num_groups, M)`` row-major and we expose it as
    ``(M, num_groups)`` via ``.permute(-1, -2)`` so DeepGEMM's
    ``fp8_gemm_nt`` sees ``stride(0)==1`` SF — i.e. SF columns are
    contiguous, which the kernel expects for TMA loads.
    """
    return torch.empty(
        (num_groups, M), device=device, dtype=torch.float32,
    ).permute(-1, -2)


def _alloc_packed_scale(M: int, num_groups: int,
                        device: torch.device) -> torch.Tensor:
    """Allocate the *packed* UE8M0 activation-scale tensor DeepGEMM consumes
    when ``disable_ue8m0_cast=True``.

    Probed off ``deep_gemm.transform_sf_into_required_layout(is_sfa=True)``:
    physical storage is ``(num_groups // 4, align(M, 4))`` int32 row-major,
    exposed as ``(M, num_groups // 4)`` with ``stride(0) == 1``.  Each int32
    holds four consecutive groups' UE8M0 exponent bytes along K, low byte
    first.

    Emitting this directly is what lets ``Fp8Linear`` pass
    ``disable_ue8m0_cast=True`` and drop DeepGEMM's internal SF-cast kernel --
    one of only three GPU ops in the small-M critical path.  ``zeros`` (not
    ``empty``) so the up-to-3 alignment padding rows are deterministic; the
    buffer is cached, so this is paid once per shape.
    """
    aligned_m = (M + 3) & ~3
    return torch.zeros(
        (num_groups // 4, aligned_m), device=device, dtype=torch.int32,
    ).permute(-1, -2)[:M]


# ---------------------------------------------------------------------------
# torch.compile-safe M-based dispatch. vLLM registers the FlashInfer(M<32) vs
# DeepGEMM(M>=32) selection as a custom op that branches on the RUNTIME M
# (``dynamic_flashinfer_deepgemm_blockscale_gemm`` -> ``torch.cond`` over both
# branches), so CUDA-graph / torch.compile capture keeps the FlashInfer path.
# fastkernels' eager ``Fp8Linear.forward`` selects the branch in Python gated on
# ``not torch.compiler.is_compiling()``, which freezes to DeepGEMM at trace time
# and silently drops FlashInfer's low-batch (M<32) accuracy path under compile.
# This op performs the same selection INSIDE an opaque custom op (Inductor never
# inlines it), so the M<32 branch survives capture. ``flashinfer_ok`` folds the
# M-independent eligibility (dtype/N%64/K%128/availability) into a compile-time
# constant; only the M<32 test happens at runtime here.
_fp8_lib.define(
    "blockscale_gemm_dispatch(Tensor input_2d, Tensor weight_fp8, "
    "Tensor weight_scale, bool flashinfer_ok) -> Tensor"
)

# FlashInfer swapAB M threshold — hard-coded to 32 in vLLM (fp8_utils.py:308),
# same as ``Fp8Linear._FLASHINFER_M_THRESHOLD``.
_FLASHINFER_M_THRESHOLD = 32


def _blockscale_gemm_dispatch_impl(input_2d, weight_fp8, weight_scale,
                                   flashinfer_ok):
    N, K = weight_fp8.shape
    M = input_2d.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=input_2d.device)
    if flashinfer_ok and M < _FLASHINFER_M_THRESHOLD:
        # FlashInfer swapAB kernel: internal activation quant (BF16 -> FP8),
        # FP8 GEMM, BF16 out. Same path as eager ``use_flashinfer``.
        _flashinfer_blockscale_gemm_impl(input_2d, weight_fp8, weight_scale,
                                         output)
        return output
    # External per-token-group quant (column-major UE8M0 scales) + DeepGEMM
    # fp8_gemm_nt — identical math to the eager M>=32 path (fresh allocations,
    # matching the pre-existing compiled branch).
    num_groups = (K + _GROUP_SIZE - 1) // _GROUP_SIZE
    q_input = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=input_2d.device)
    input_scale = _alloc_colmajor_scale(M, num_groups, input_2d.device)
    _per_token_group_quant_fp8_op_impl(input_2d, q_input, input_scale, True)
    _fp8_gemm_nt_impl(q_input, input_scale, weight_fp8, weight_scale, output)
    return output


_fp8_lib.impl("blockscale_gemm_dispatch", _blockscale_gemm_dispatch_impl,
              "CUDA")


@torch.library.impl(_fp8_lib, "blockscale_gemm_dispatch", "Meta")
def _blockscale_gemm_dispatch_meta(input_2d, weight_fp8, weight_scale,
                                   flashinfer_ok):
    N = weight_fp8.shape[0]
    M = input_2d.shape[0]
    return input_2d.new_empty((M, N), dtype=torch.bfloat16)


class _Fp8PrefillBufs:
    """Shared prefill buffers for FP8 activation quantization.

    Since decoder layers execute sequentially, a single set of buffers
    (sized for max_num_batched_tokens) can be reused across all Fp8Linear
    instances, eliminating per-layer dynamic allocation during prefill.
    One instance per unique (K, N) weight shape.

    The scale buffer ``s`` is column-major (matches vLLM's DeepGEMM path).
    """
    __slots__ = ("a", "s", "o")

    def __init__(self, max_tokens: int, K: int, N: int, device: torch.device):
        num_groups = math.ceil(K / 128)
        self.a = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self.s = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self.o = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)


class Fp8Linear(nn.Module):
    """Block-scaled FP8 linear using deep_gemm.fp8_gemm_nt.

    Weights are stored in float8_e4m3fn with pre-processed UE8M0 block scales
    (transformed via deep_gemm.transform_sf_into_required_layout at load time).
    Activations are dynamically quantized to FP8 per-token-group (group=128)
    using in-place ops for CUDA graph compatibility.
    """

    BLOCK_SIZE = 128

    def __init__(self):
        super().__init__()
        self._a_buf: torch.Tensor | None = None
        self._s_buf: torch.Tensor | None = None
        self._o_buf: torch.Tensor | None = None
        self._pf: _Fp8PrefillBufs | None = None
        # M -> [K, fp8 activation, scale, scale_is_packed] scratch for the
        # DeepGEMM path.  Both tensors are pure temporaries consumed by the GEMM
        # within the same call, and re-creating them costs two ``torch.empty``
        # calls plus a ``permute`` (~5us of host time) per forward -- more than
        # the small-M GEMM itself.  Bounded so a workload with many distinct
        # token counts cannot grow it without limit.
        self._scratch: dict[int, tuple] = {}
        # Fused-GEMV state: eligibility of this layer's weights (resolved once,
        # they never change) and the (1, N) output buffer it writes.
        self._gemv_key: tuple | None = None
        self._gemv_ok = False
        self._gemv_out: torch.Tensor | None = None

    _MAX_SCRATCH = 8

    def _gemv_eligible(self, K: int, N: int, weight_fp8: torch.Tensor,
                       weight_scale_inv: torch.Tensor) -> bool:
        """Whether the fused GEMV can serve this layer, cached per weight shape.

        It needs DeepGEMM's *packed* UE8M0 weight SF -- int32 ``(N, K/512)`` with
        unit row stride, four block exponents per word along K -- which is what
        ``postprocess_fp8_weights`` produces on this arch, plus a K it can cut
        into whole 512-column chunks.  Anything else (fp32 scales, a non-UE8M0
        build, K % 512, a non-contiguous weight) falls through to the
        quantize + DeepGEMM path."""
        key = (K, N, weight_scale_inv.dtype, weight_scale_inv.shape,
               weight_scale_inv.stride(), weight_fp8.is_contiguous())
        if self._gemv_key != key:
            self._gemv_key = key
            self._gemv_ok = (
                K % 512 == 0
                and weight_fp8.is_contiguous()
                and weight_fp8.dtype == torch.float8_e4m3fn
                and weight_scale_inv.dtype == torch.int32
                and weight_scale_inv.dim() == 2
                and weight_scale_inv.stride(0) == 1
                and weight_scale_inv.shape == (N, K // 512)
                and _use_ue8m0()
            )
            self._gemv_out = None
        return self._gemv_ok

    def _quant_scratch(self, M: int, K: int, device: torch.device,
                       need_q: bool):
        cache = self._scratch
        ent = cache.get(M)
        if ent is None or ent[0] != K:
            if len(cache) >= self._MAX_SCRATCH:
                cache.clear()
            num_groups = -(-K // self.BLOCK_SIZE)
            # The packed SF layout needs whole 4-group words along K; anything
            # else keeps the fp32 column-major layout and lets DeepGEMM cast.
            packed = _use_ue8m0() and K % (4 * self.BLOCK_SIZE) == 0
            scale = (_alloc_packed_scale(M, num_groups, device) if packed
                     else _alloc_colmajor_scale(M, num_groups, device))
            cache[M] = ent = [K, None, scale, packed]
        # Only the callers without a pre-allocated activation buffer of their
        # own need the fp8 scratch, so it is filled in on demand.
        if need_q and ent[1] is None:
            ent[1] = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device)
        return ent[1], ent[2], ent[3]

    def _ensure_buffers(self, max_tokens: int, K: int, N: int, device: torch.device):
        """Pre-allocate activation FP8 buffers for CUDA graph capture.

        The scale buffer is **column-major** to match vLLM's DeepGEMM
        dense FP8 path (see ``_alloc_colmajor_scale``).
        """
        num_groups = math.ceil(K / self.BLOCK_SIZE)
        self._a_buf = torch.empty(max_tokens, K, dtype=torch.float8_e4m3fn, device=device)
        self._s_buf = _alloc_colmajor_scale(max_tokens, num_groups, device)
        self._o_buf = torch.empty(max_tokens, N, dtype=torch.bfloat16, device=device)

    # Threshold matching vLLM's ``W8A8BlockFp8LinearOp.apply`` /
    # ``_flashinfer_fp8_blockscale_gemm_impl``: below this M the swapAB
    # kernel inside FlashInfer's ``fp8_blockscale_gemm_sm90`` is used; above,
    # external-quant + DeepGEMM ``fp8_gemm_nt``.  The threshold is hard-coded
    # to 32 in vLLM (``fp8_utils.py:308``).
    _FLASHINFER_M_THRESHOLD = 32

    def forward(self, input_bf16: torch.Tensor,
                weight_fp8: torch.Tensor,
                weight_scale_inv: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """FP8 block-scaled GEMM with vLLM-parity dispatch.

        * ``M < 32`` and FlashInfer available → FlashInfer swapAB kernel
          (BF16 in, internal quant, FP8 GEMM, BF16 out).  Same path as
          ``vllm/.../fp8_utils.py:_run_flashinfer``.
        * Otherwise → external per-token-group quantization +
          ``deep_gemm.fp8_gemm_nt``, the same path as
          ``vllm/.../fp8_utils.py:_run_deepgemm``.  The quantizer emits the
          *packed* UE8M0 scale layout the GEMM consumes directly whenever it
          can (``disable_ue8m0_cast=True``), falling back to vLLM's fp32
          column-major layout otherwise.

        FP8 ops are routed through ``getattr(torch.ops, _FK_NS).*`` so they stay
        opaque to ``torch.compile`` (mirrors vLLM's
        ``torch.ops.vllm.fp8_gemm_nt_op`` / ``flashinfer_fp8_blockscale_gemm``).
        """
        N, K = weight_fp8.shape
        in_shape = input_bf16.shape
        if len(in_shape) == 2:
            input_2d = input_bf16
            M = in_shape[0]
        else:
            input_2d = input_bf16.reshape(-1, K)
            M = input_2d.shape[0]

        # The FlashInfer swapAB kernel is Hopper-only; where it is unavailable
        # (or batch-invariant mode forces DeepGEMM) the whole vLLM-parity
        # dispatch below collapses to the DeepGEMM branch, so resolve that once
        # and keep the shape-dependent eligibility tests off the hot path.
        if _flashinfer_available() and not _is_batch_invariant():
            # M-independent FlashInfer eligibility (mirrors vLLM
            # ``should_use_flashinfer_for_blockscale_fp8_gemm``: N % 64 == 0,
            # K % 128 == 0, plus the SM90+FlashInfer availability gate).
            flashinfer_ok = (
                input_bf16.dtype == torch.bfloat16
                and weight_fp8.dtype == torch.float8_e4m3fn
                and N % 64 == 0
                and K % 128 == 0
            )
        else:
            flashinfer_ok = False

        # Under torch.compile / CUDA-graph capture, the M<32 (FlashInfer) vs
        # M>=32 (DeepGEMM) choice must be made at RUNTIME, not frozen at trace
        # time. A Python ``if M < 32`` (or ``torch.compiler.is_compiling()``)
        # gate would bake the branch into the graph and drop FlashInfer's
        # low-batch path. Route through the opaque ``blockscale_gemm_dispatch``
        # custom op (like vLLM's ``dynamic_flashinfer_deepgemm_blockscale_gemm``)
        # which branches on the runtime M internally. The eager path below keeps
        # its buffer-reuse fast path.
        if _is_compiling():
            output = getattr(torch.ops, _FK_NS).blockscale_gemm_dispatch(
                input_2d, weight_fp8, weight_scale_inv, flashinfer_ok,
            )
            if bias is not None:
                output = output + bias
            return output.view(*in_shape[:-1], N)

        if flashinfer_ok and M < self._FLASHINFER_M_THRESHOLD:
            output = torch.empty(
                M, N, dtype=torch.bfloat16, device=input_2d.device,
            )
            getattr(torch.ops, _FK_NS).flashinfer_blockscale_gemm(
                input_2d, weight_fp8, weight_scale_inv, output,
            )
            if bias is not None:
                output = output + bias
            return output.view(*in_shape[:-1], N)

        # ---- fused single-op GEMV (M == 1) --------------------------------
        # At M == 1 the GEMM is a bandwidth-bound GEMV that DeepGEMM spends a
        # full 128-row M-tile on, and the activation quantization folds into its
        # prologue -- so the whole forward becomes ONE GPU op reading the weight
        # exactly once, which is the measured floor for this window (9.2 us of
        # the 17.4 us row on [1,2048]x[4096,2048], against 21.4 us for
        # quantize + DeepGEMM).  Anything the kernel cannot express -- another
        # weight-scale layout, K % 512, a non-contiguous activation -- falls
        # through to the path below unchanged.
        if (M == 1 and input_2d.is_contiguous()
                and input_2d.dtype == torch.bfloat16
                and self._gemv_eligible(K, N, weight_fp8, weight_scale_inv)):
            out = self._gemv_out
            if out is None:
                out = self._gemv_out = torch.empty(
                    1, N, dtype=torch.bfloat16, device=input_2d.device)
            fn = _GEMV_FN
            if fn is None:
                fn = _resolve_gemv_fn()
            fn(input_2d, weight_fp8, weight_scale_inv, out, _QUANT_EPS,
               _FP8_MIN, _FP8_MAX, -1)
            if bias is not None:
                out = out + bias
            return out if len(in_shape) == 2 else out.view(*in_shape[:-1], N)

        # ---- eager DeepGEMM path ------------------------------------------
        # Eager only (the compile path returned above), so buffer reuse is
        # always safe here.  ``input_scale`` is column-major, matching the SF
        # layout DeepGEMM expects (``vllm/.../fp8_utils.py:289-294``); the
        # quantizer picks that up from its strides.
        device = input_2d.device
        if self._a_buf is not None and M <= self._a_buf.shape[0]:
            q_input = self._a_buf[:M]
            output = self._o_buf[:M]
        elif self._pf is not None and M <= self._pf.a.shape[0]:
            q_input = self._pf.a[:M]
            output = self._pf.o[:M]
        else:
            q_input = None
            output = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        scratch_q, input_scale, packed_sf = self._quant_scratch(
            M, K, device, q_input is None)
        if q_input is None:
            q_input = scratch_q

        fn = _QUANT_FN
        if fn is None:
            fn = _resolve_quant_fn()
        fn(input_2d, q_input, input_scale, _QUANT_EPS, _FP8_MIN, _FP8_MAX)
        # With ``packed_sf`` the quantizer already emitted the exact packed
        # UE8M0 layout ``fp8_gemm_nt`` wants, so DeepGEMM's internal SF cast --
        # a whole extra kernel launch -- is skipped.  Bit-identical either way:
        # the scales are exact powers of two, so the cast is just an exponent
        # extraction (verified against the fp32-SF path over all captured
        # shapes).
        _DG_FP8_GEMM_NT(
            (q_input, input_scale),
            (weight_fp8, weight_scale_inv),
            output,
            disable_ue8m0_cast=packed_sf or not _use_ue8m0(),
        )

        if bias is not None:
            output = output + bias
            return output.view(*in_shape[:-1], N)
        return output if len(in_shape) == 2 else output.view(*in_shape[:-1], N)


def postprocess_fp8_weights(weight_fp8: torch.Tensor,
                            scale_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize FP8 weights to UE8M0 scale format and transform scale layout
    for DeepGEMM compatibility. Must be called once after weight loading.

    Matches vLLM's requant_weight_ue8m0_inplace + deepgemm_post_process_fp8_weight_block:
    dequantize in float32 for precision, re-quantize with UE8M0 power-of-two scales,
    then transform scale layout.  Handles non-block-aligned shapes via padding.
    """
    N, K = weight_fp8.shape
    block_size = Fp8Linear.BLOCK_SIZE

    scale_rows = math.ceil(N / block_size)
    scale_cols = math.ceil(K / block_size)
    scale = scale_inv[:scale_rows, :scale_cols].to(torch.float32)

    w_padded = weight_fp8
    need_n_pad = (block_size - N % block_size) % block_size
    need_k_pad = (block_size - K % block_size) % block_size
    if need_n_pad or need_k_pad:
        w_padded = torch.nn.functional.pad(
            weight_fp8.view(torch.int8),
            (0, need_k_pad, 0, need_n_pad),
        ).view(torch.float8_e4m3fn)

    if need_n_pad or need_k_pad:
        w_view = w_padded.view(
            math.ceil(N / block_size + need_n_pad / block_size), block_size,
            math.ceil(K / block_size + need_k_pad / block_size), block_size,
        )
    else:
        w_view = w_padded.view(scale_rows, block_size, scale_cols, block_size)

    w_f32 = w_view.to(torch.float32) * scale[:, None, :, None]

    w_f32_flat = w_f32.reshape(-1, w_f32.shape[2] * block_size)
    if need_n_pad or need_k_pad:
        w_f32_flat = w_f32_flat[:N, :K].contiguous()

    # Mirror vLLM: ``use_ue8m0`` and ``disable_ue8m0_cast`` are both keyed off
    # the same oracle (``is_deep_gemm_e8m0_used``), so the requant + the
    # SF layout transform agree. Keeps weights consistent with what the GEMM
    # kernel expects on this build/arch.
    use_ue8m0 = _is_deep_gemm_e8m0_used()
    w_fp8_new, scale_ue8m0 = deep_gemm.per_block_cast_to_fp8(
        w_f32_flat, use_ue8m0=use_ue8m0,
    )

    recipe = (1, block_size, block_size)
    scale_transformed = deep_gemm.transform_sf_into_required_layout(
        sf=scale_ue8m0.unsqueeze(0),
        mn=N,
        k=K,
        recipe=recipe,
        num_groups=1,
        is_sfa=False,
        disable_ue8m0_cast=not use_ue8m0,
    ).squeeze(0)

    return w_fp8_new, scale_transformed


def postprocess_fp8_weights_batched(weight_fp8: torch.Tensor,
                                    scale_inv: torch.Tensor) -> None:
    """Re-quantize 3D MoE weights [E, N, K] to UE8M0 scales in-place,
    then transform scale layout for DeepGEMM. Matches vLLM's
    requant_weight_ue8m0_inplace + deepgemm_post_process_fp8_weight_block."""
    assert weight_fp8.ndim == 3
    E, N, K = weight_fp8.shape
    block_size = Fp8Linear.BLOCK_SIZE

    scale_rows = math.ceil(N / block_size)
    scale_cols = math.ceil(K / block_size)

    use_ue8m0 = _is_deep_gemm_e8m0_used()

    for idx in range(E):
        w_q = weight_fp8[idx]
        s_old = scale_inv[idx, :scale_rows, :scale_cols]

        s_float = s_old.to(torch.float32)
        s_exp = torch.repeat_interleave(s_float, block_size, dim=0)[:N]
        s_exp = torch.repeat_interleave(s_exp, block_size, dim=1)[:, :K]
        w_dq = w_q.to(torch.float32) * s_exp

        w_requant, s_requant = deep_gemm.per_block_cast_to_fp8(
            w_dq, use_ue8m0=use_ue8m0,
        )
        w_q.copy_(w_requant)
        s_old.copy_(s_requant)

    # The requant loop above already wrote UNPACKED per-block UE8M0 fp32 scales
    # into ``scale_inv[:, :scale_rows, :scale_cols]`` — the exact [E, N/128, K/128]
    # fp32 layout the Triton MoE grouped GEMM (MoeGroupedGemm, used by
    # VllmFusedExperts) reads via strides. ``transform_sf_into_required_layout``
    # is a DeepGEMM-only SF layout (see vLLM quant_utils.py:428); on Blackwell
    # (``use_ue8m0``) it PACKS 4 UE8M0 exponents per int32, shrinking the last dim
    # (e.g. 48 -> 12) into a layout the Triton kernel does not consume — so applying
    # it here corrupts the scale (and its old in-place ``copy_`` even crashed on the
    # shape change). Only run the transform on the non-UE8M0 (Hopper) path, where it
    # preserves shape, to keep that path bit-identical.
    if not use_ue8m0:
        recipe = (1, block_size, block_size)
        scale_transformed = deep_gemm.transform_sf_into_required_layout(
            sf=scale_inv[:, :scale_rows, :scale_cols],
            mn=N,
            k=K,
            recipe=recipe,
            num_groups=E,
            is_sfa=False,
            disable_ue8m0_cast=True,
        )
        scale_inv[:, :scale_rows, :scale_cols].copy_(scale_transformed)
