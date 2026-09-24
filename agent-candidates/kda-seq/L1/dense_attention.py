"""Dense (non-paged) multi-head attention with a tiny-sequence fast path.

Same ``__init__``/``forward`` contract as the baseline. The module is a single
dispatcher over two regimes:

  **tiny sequence extent** — ``head_dim == 64``, fp16/bf16, self-attention
    (``q``/``k``/``v`` all the same shape), sequence extent at most 8, no explicit
    mask, and row addresses aligned to the packed-pair load the kernel issues. This
    is the temporal-attention shape family (``[144, 2..6, 16, 64]``): thousands of
    independent problems only a few tokens long. PyTorch's SDPA heuristic hands it
    to a cuDNN flash kernel tiled ``128x64``, so more than 96 % of every tile is
    masked padding and the measured latency (~75 us) is pure per-problem tile
    overhead — it is identical for a 2-token and a 5-token sequence. A custom
    warp-per-``(batch, head)`` kernel serves it instead.

  **everything else** — the baseline's ``forward`` decision tree, reproduced
    unchanged, so every shape, dtype, mask and backend the fast path does not
    explicitly claim behaves exactly as it does today. Long sequences, ``head_dim``
    other than 64, fp32, any explicit mask, the flash-attention callable and
    FlexAttention all land here deliberately; cuDNN already runs the large
    joint-attention shapes at ~65 % of dense peak and is not worth displacing.

The dispatch predicate is an explicit conjunction of sufficient conditions. Every
stride reaches the kernel as a runtime argument: in this shape family ``value`` is
routinely a ``transpose(0, 1)`` view of a fused ``qkv`` slice, and normalising it to
a contiguous copy would cost more than the whole target latency.

The launch geometry is fixed rather than configurable, so the module cannot be made
to run a grid below the GPU's SM count. The fast path is additionally restricted by
``backend`` *name* to ``"auto"``,
``"sdpa"`` and ``"cudnn"``. Routing state is not a sufficient test: an unrecognised
string resolves through the ``"auto"`` branch, which on Blackwell produces exactly
the state ``"cudnn"`` produces, so a typo would otherwise be accelerated as though
it had been a deliberate choice. Such instances keep the baseline's behaviour
byte-for-byte.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# cuDNN's SDPA kernels are limited to head_dim <= 128 ("head_dim should be no
# more than 128" in sdp_utils.cpp); larger heads must use EFFICIENT/MATH.
_CUDNN_MAX_HEAD_DIM = 128

# Fast-path domain. head_dim 64 in a 2-byte dtype is exactly 32 lanes x 2 elements,
# which is what makes a warp-per-row load a single fully-utilised 128-byte burst.
_TINY_HEAD_DIM = 64
_TINY_MAX_SEQ = 8
_TINY_SCALE_DEFAULT = _TINY_HEAD_DIM ** -0.5

# The only ``backend`` values the fast path claims. "flash_attn" and "flex" are
# recognised but route to kernels with their own conventions, and an unrecognised
# string is a caller error: the baseline resolves it through its "auto" branch, so
# it must keep behaving exactly like that rather than being quietly accelerated.
# Membership is decided from the *string*, because the routing state alone cannot
# distinguish "cudnn" from a typo (on Blackwell both end up on the cuDNN path).
_TINY_CLAIMED_BACKENDS = frozenset({"auto", "sdpa", "cudnn"})

# Launch geometry, fixed. At the captured batch*heads = 2304 pairs, 8 warps per
# block is 288 CTAs against 148 SMs -- 1.95x the SM count, which is the margin this
# regime needs, since a grid at or below the SM count leaves the machine unable to
# overlap anything. 16 warps would be 144 CTAs and 32 would be 72: both below the SM
# count, and both therefore inadmissible however they happen to measure.
#
# These are deliberately *not* environment-configurable. A sweep over ten
# configurations found no measurable difference between any of them (recorded in
# profile/tiny_seq_attn_v2_scaled_query/tiny_family_sweep.csv), so a runtime knob
# would buy nothing while making it possible for the shipped module to run a geometry
# that is ruled out. Configurability belongs in the standalone profiling harness under
# profile/, which takes both as argv and is where rejected sweep points should live.
_TINY_WARPS_PER_BLOCK = 8
# Pair index -> (batch, head) mapping: 0 keeps heads innermost (pair = b*H + h), so
# the head axis -- the smaller stride -- is walked fastest.
_TINY_HEAD_MAJOR = 0
# Grid size the fixed geometry produces, as a function of the problem. Kept as a
# function so the SM-count relationship stays checkable from a test rather than
# living only in this comment.
_SM_COUNT_B200 = 148


def _grid_ctas(batch: int, heads: int) -> int:
    """CTAs the fixed geometry launches for a ``batch x heads`` problem."""
    pairs = batch * heads
    return -(-pairs // _TINY_WARPS_PER_BLOCK)
# Escape hatch: serve everything from the baseline decision tree. Read once, here,
# so it can never affect the per-call cost -- it exists to A/B the fast path
# against the path it replaces, and to fall back in one step if it misbehaves.
_TINY_DISABLED = os.environ.get("DENSE_ATTN_DISABLE_FAST_PATH", "") not in ("", "0")


def _resolve_flash_attn_func():
    """Return the flash-attention callable for Ampere/Hopper.

    Same order as vllm-omni's CUDA FA resolver: FA3 (fa3-fwd) >
    FA3 (source-built flash_attn_interface) > FA2.
    """
    for mod in ("fa3_fwd_interface", "flash_attn_interface"):
        try:
            return __import__(mod, fromlist=["flash_attn_func"]).flash_attn_func
        except (ImportError, ModuleNotFoundError):
            pass
    from flash_attn import flash_attn_func
    return flash_attn_func


def _build_tiny_extension():
    """Compile the tiny-sequence kernel at import time.

    Compilation must never happen inside ``forward``: it would dominate any
    measurement and, because the toolchain starts helper processes, sits near the
    benchmark's thread-count tripwire. A build failure degrades to ``None``, which
    disables the fast path and leaves the baseline decision tree serving
    everything, rather than making the module unimportable.
    """
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().with_name("_tiny_seq_attn.cu")
    build_dir = os.environ.get("DENSE_ATTN_BUILD_DIR")
    if build_dir is None:
        build_dir = str(Path(__file__).resolve().parents[2] / ".torch_extensions")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="dense_attn_tiny_seq",
        sources=[str(source)],
        # No fast-math: the softmax exponential stays the accurate libdevice one,
        # which keeps the numerics comparable to the cuDNN reference for free.
        extra_cuda_cflags=["-O3", "-lineinfo"],
        extra_cflags=["-O3"],
        build_directory=build_dir,
        verbose=False,
    )


def _warm_tiny_extension(ext) -> None:
    """Launch every kernel specialisation once so the first real call is warm.

    Covers both dtypes, both mask modes and all supported sequence extents on a
    single-pair problem. Purely a warm-up: the results are discarded.
    """
    for dtype in (torch.float16, torch.bfloat16):
        for seq in range(1, _TINY_MAX_SEQ + 1):
            probe = torch.zeros((1, seq, 1, _TINY_HEAD_DIM), dtype=dtype, device="cuda")
            for causal in (True, False):
                ext.tiny_seq_attn(probe, probe, probe, _TINY_SCALE_DEFAULT, causal,
                                  _TINY_WARPS_PER_BLOCK, _TINY_HEAD_MAJOR)
    torch.cuda.synchronize()


# Why the fast path is unavailable, when it is. Degrading to the baseline tree on a
# toolchain or driver problem is the right behaviour, but degrading *silently* hides
# real breakage behind a plausible-looking 1.0x, so the reason is kept for tests and
# for anyone asking why nothing got faster.
_TINY_EXT_ERROR: str | None = None

try:
    if _TINY_DISABLED:
        _TINY_EXT = None
        _TINY_EXT_ERROR = "disabled by DENSE_ATTN_DISABLE_FAST_PATH"
    elif not torch.cuda.is_available():
        _TINY_EXT = None
        _TINY_EXT_ERROR = "no CUDA device available at import"
    else:
        _TINY_EXT = _build_tiny_extension()
        _warm_tiny_extension(_TINY_EXT)
except Exception as exc:  # noqa: BLE001 - serve everything from the baseline tree instead
    _TINY_EXT = None
    _TINY_EXT_ERROR = f"{type(exc).__name__}: {exc}"


def _rows_aligned(t: torch.Tensor) -> bool:
    """Whether every ``(batch, seq, head)`` row of *t* starts on a packed-pair
    boundary.

    A unit ``head_dim`` stride alone is not enough: an odd outer stride or an odd
    storage offset leaves rows only 2-byte aligned, and the kernel loads two
    elements at a time. ``data_ptr`` already folds in the storage offset. Every
    stride seen in practice satisfies this, but that is a property of those
    layouts rather than a guarantee, so it is checked rather than assumed.
    """
    s = t.stride()
    return (len(s) == 4
            and s[3] == 1
            and not (s[0] & 1) and not (s[1] & 1) and not (s[2] & 1)
            and not (t.data_ptr() & 3))


class DenseAttention(nn.Module):
    """Dense multi-head attention.

    Input layout: (batch, seq_len, num_heads, head_dim).

    Args:
        backend: Which kernel to use.
            ``"auto"`` selects flash-attention on Ampere/Hopper when
            available, SDPA everywhere else.
            ``"sdpa"`` always uses ``F.scaled_dot_product_attention``
            (PyTorch's heuristic chooses among flash/cuDNN/mem_eff/math).
            ``"flash_attn"`` always uses the flash-attention package.
            ``"cudnn"`` pins the cuDNN flash backend via
            ``torch.nn.attention.sdpa_kernel`` (with MATH fallback for
            masks cuDNN can't handle). Required to get cuDNN flash
            through ``torch.compile`` on Blackwell.
    """

    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None

        if backend == "sdpa":
            self._finish_init(backend)
            return

        if backend == "cudnn":
            self.use_cudnn_kernel = True
            self._finish_init(backend)
            return

        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            self._finish_init(backend)
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            self._finish_init(backend)
            return

        # backend == "auto": flash-attn on Ampere/Hopper (80<=cc<100); cuDNN flash
        # on Blackwell (cc>=100), where PyTorch's SDPA heuristic otherwise picks
        # FA2 (~3.6x slower than cuDNN for large joint-attention shapes on B200).
        # This mirrors vllm-omni's platform selector, which pins cuDNN/TRTLLM on
        # Blackwell. The cuDNN forward path already falls back to mem-efficient/MATH
        # for shapes/masks cuDNN rejects, so this is safe as a default.
        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True
        self._finish_init(backend)

    def _finish_init(self, backend: str) -> None:
        """Resolve everything the dispatcher can decide from ``backend`` alone.

        The fast path may only claim an instance whose routing computes plain
        scaled-dot-product attention. The flash-attention callable has its own
        argument and output conventions, and FlexAttention consumes a ``BlockMask``
        through the ``attn_mask`` slot and compiles a kernel for one exact shape;
        intercepting either would change observable behaviour.

        Routing state is necessary but not sufficient. An unrecognised ``backend``
        string falls through to the ``"auto"`` branch, which on Blackwell produces
        exactly the routing state ``"cudnn"`` does -- so an instance built from a
        typo is indistinguishable from a deliberate one by state alone. Such an
        instance keeps the baseline's behaviour and is excluded from the fast path
        by name.

        ``backend`` is required rather than defaulted: a default would mean a caller
        who forgot it silently re-widened eligibility, which is the exact failure
        this argument exists to prevent.
        """
        self._tiny_ok = (_TINY_EXT is not None
                         and self.fa_func is None
                         and not self.use_flex_kernel
                         and backend in _TINY_CLAIMED_BACKENDS)

    def forward(
        self,
        query,
        key,
        value,
        softmax_scale=None,
        causal=False,
        attn_mask: torch.Tensor | None = None,
    ):
        # Tiny-sequence fast path. Sufficient conditions only, tested as a flat
        # conjunction of attribute and tuple comparisons: comparing all three
        # shapes pins self-attention, equal head counts and equal head dims in one
        # go, and anything not claimed here falls through untouched.
        if self._tiny_ok and attn_mask is None:
            dtype = query.dtype
            shape = query.shape
            # ``len(shape) == 4`` comes first: every term after it indexes the
            # shape, and a rank-3 input would raise IndexError out of the dispatcher
            # rather than reaching the baseline tree, which is the one thing an
            # unclaimed configuration must always do. A rank-5 tensor whose axis-3
            # stride happens to be 1 would otherwise satisfy every other term and
            # reach the kernel.
            if (len(shape) == 4
                    and (dtype is torch.float16 or dtype is torch.bfloat16)
                    and key.dtype is dtype and value.dtype is dtype
                    and shape[3] == _TINY_HEAD_DIM
                    and 0 < shape[1] <= _TINY_MAX_SEQ
                    and shape[0] > 0 and shape[2] > 0
                    and key.shape == shape and value.shape == shape
                    and query.is_cuda
                    and key.device == query.device
                    and value.device == query.device
                    # The kernel is a plain extension call with no autograd node, so
                    # anything that wants gradients has to go through SDPA, which
                    # has them. Checked as three attribute reads before any
                    # heavier test, and false for every benchmarked input.
                    and not query.requires_grad
                    and not key.requires_grad
                    and not value.requires_grad
                    and _rows_aligned(query)
                    and _rows_aligned(key)
                    and _rows_aligned(value)):
                # An explicitly supplied 0.0 is a real scale, so this is an
                # ``is None`` test rather than a truthiness test.
                scale = (_TINY_SCALE_DEFAULT if softmax_scale is None
                         else float(softmax_scale))
                # Already (batch, seq, heads, head_dim), carrying the same strides
                # the baseline's own trailing permute produces.
                return _TINY_EXT.tiny_seq_attn(
                    query, key, value, scale, bool(causal),
                    _TINY_WARPS_PER_BLOCK, _TINY_HEAD_MAJOR)

        if self.fa_func is not None and attn_mask is None and query.dtype != torch.float32:
            out = self.fa_func(
                query, key, value,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out

        # SDPA handles both the masked case and the plain causal/non-causal case.
        # Custom masks force is_causal=False; FlashAttn does not support arbitrary masks.
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        if self.use_flex_kernel:
            # FlexAttention generates a fused Triton fwd+bwd kernel autotuned
            # for the exact (B, H, S_q, S_kv, D) shape and the user-provided
            # mask. ``attn_mask`` here is repurposed to accept a
            # ``BlockMask`` (from ``create_block_mask``) instead of a dense
            # bool tensor. On B200 with chunked-suffix shapes
            # (Q=1024, KV=9216, D=64), the fused fwd+bwd is ~1.37x faster
            # than cuDNN flash with the equivalent dense mask
            # (microbenched). Same numerical agreement vs the fp32 MATH
            # reference (~1e-2 max-abs-diff in bf16, identical to cuDNN).
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            out = self._flex_fn(
                q, k, v,
                block_mask=attn_mask,
                scale=softmax_scale,
            )
        elif self.use_cudnn_kernel:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            # An explicit mask plus is_causal=True is ambiguous, and the two code
            # paths here would resolve it differently: this branch would hand both
            # to SDPA (which applies the causal mask *on top of* attn_mask), while
            # the non-cuDNN branch below drops is_causal and treats attn_mask as
            # authoritative. SDPA itself accepts the combination on this backend
            # rather than rejecting it, so nothing would surface the disagreement
            # -- reject it here instead of silently masking twice.
            if attn_mask is not None and causal:
                raise ValueError(
                    "DenseAttention: pass either attn_mask or causal=True, not both "
                    "(an explicit mask must already encode causality). Got "
                    f"attn_mask={tuple(attn_mask.shape)} with causal=True."
                )
            # The sdpa_kernel context below FORCES cuDNN, and on Blackwell (sm100,
            # cuDNN 9.19) the cuDNN flash kernel accepts the permuted, non-contiguous
            # q/k/v views directly -- so we skip the q/k/v .contiguous() clones (they
            # were a real cost: 3 clones/block x54 blocks). Verified bit-identical and
            # faster; if cuDNN ever rejects a layout it raises -> MATH fallback below.
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            # Try strict cuDNN first. Adding MATH as a fallback in the
            # ``sdpa_kernel`` list causes PyTorch's selection heuristic to
            # pick MATH over cuDNN (~10× slower) for inputs both can
            # handle. If cuDNN rejects (e.g. head_dim=16, fp32, or some
            # mask shape it doesn't support), fall back through MATH.
            #
            # head_dim > 128 is rejected by cuDNN unconditionally ("head_dim
            # should be no more than 128"), so route it straight to the backends
            # that can serve it. The try/except below only recovers in eager --
            # under torch.compile the RuntimeError surfaces during fake-tensor
            # tracing and aborts the whole graph rather than taking the handler,
            # which is how a head_dim=256 model (Gemma-2B in Pi0) failed to
            # compile at all.
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                # EFFICIENT_ATTENTION requires an additive bias in the query's
                # dtype ("invalid dtype for bias - should match query's dtype");
                # cuDNN tolerated an fp32 mask against bf16 q/k/v. A bool mask is
                # passed through -- coercing it would turn True/False into a
                # 1.0/0.0 additive bias.
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
            else:
                try:
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
                except RuntimeError:
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
        else:
            # SDPA accepts a boolean mask (True = attend) directly; only a float
            # (additive) mask needs dtype coercion. Coercing a bool mask to q.dtype
            # would turn True/False into a 1.0/0.0 additive bias (wrong semantics) --
            # e.g. the HunyuanVideo key-padding mask would then fail to mask padding
            # on non-cuDNN backends.
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False if attn_mask is not None else causal,
                scale=softmax_scale,
            )
        return out.permute(0, 2, 1, 3)
