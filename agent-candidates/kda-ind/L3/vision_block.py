"""Vision transformer block for Qwen VL models, with the glue kernels fused.

Same ``__init__``/``forward`` contract as the baseline, and the same module tree
(``norm1``, ``norm2``, ``attn.qkv``, ``attn.proj``, ``mlp.fc1``, ``mlp.fc2``), so
the harness' name-based ``load_state_dict`` shares every weight.

Where the time goes in the baseline, at a full encoder batch (N=20680 tokens,
measured on this B200): cuBLAS ``nvjet`` GEMMs 433 us, FlashAttention-4 249 us,
and 512 us of bandwidth-bound glue. The glue is the headroom. Both LayerNorms
and the q/k ``permute().contiguous()`` run at ~1.5 TB/s against the ~5.2 TB/s
this machine sustains on a well-written elementwise kernel, and the vendored
Triton rotary launches 41360 blocks that each touch ~2.3 KB. So the GEMMs and
the attention call are issued exactly as the baseline issues them, and only the
glue is rewritten:

===========================  ======================================
baseline                     here
===========================  ======================================
``F.layer_norm`` (norm1)     ``fused_layer_norm``
q/k permute + contiguous     deleted -- rotary works in place on the
                             qkv buffer and FlashAttention takes
                             strided views of it
``apply_rotary``             ``fused_rotary_qk_``
residual add + norm2         ``fused_add_layer_norm`` (one pass)
``act_fn`` on the fc1 output ``fused_activation_`` (in place)
===========================  ======================================

Eleven kernels become nine, and the two slowest glue kernels disappear.

Everything is behind a guard predicate (:meth:`VisionBlock._fusable`). Inputs the
fused path does not cover -- grad mode, a batch dimension above 1, an activation
that is not exactly one of the four recognized ones, absent or oddly-strided
rotary tables, mixed affine dtypes, a non-contiguous input -- fall back to the
baseline sequence, which is the correctness net for anything the captured shapes
do not exercise.

Four further optimizations were built or probed and measured; each has a report
under ``profile/`` and a runnable probe, so none has to be re-litigated from a
table of estimates:

- head_dim 72 -> 96 attention padding, which makes the FlashAttention call itself
  1.32x faster and the whole block slower in three of three A/B rounds, because
  staging into a wider buffer and compacting the result costs two extra passes
  over q|k|v. Rejected (``profile/head_padding_v1/REPORT.md``).
- a cheaper exact-``erf`` evaluation for the activation, bit-exact on 49.3% of
  finite bf16 inputs against this kernel's 99.983%, and slower anyway. Rejected
  (``profile/cheap_erf_v1/REPORT.md``).
- an L2-resident chunked MLP, slower at every chunk size and both shapes because
  the smaller-GEMM inefficiency dominates the traffic saving. Rejected
  (``profile/chunked_mlp_v1/REPORT.md``).
- CUDA-graph replay of the fused path, implemented here behind
  :data:`_ENABLE_CUDA_GRAPHS` (``profile/cuda_graph_v1/REPORT.md``).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.quickgelu import QuickGELU
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import VisionMLP

# Counts entries into the fused path. The tests assert that guarded-out inputs
# leave this untouched, which is stronger than checking they merely produce the
# right answer -- a fused path that silently handled batch>1 incorrectly would
# still match on the shapes where it happens to agree.
_FUSED_CALLS: dict[str, int] = {"n": 0}

# Activation kinds recognized by ``_activation_kernel``. Detection is by class
# identity plus the ``approximate`` attribute, never by inspecting the tensor.
_ACT_GELU_ERF = 0
_ACT_GELU_TANH = 1
_ACT_QUICKGELU = 2
_ACT_SILU = 3

# Tile shapes and warp counts, from the sweep in
# ``profile/phase1_probes/bench_kernels.py --sweep`` at N=20680.
#
# The consistent finding across all four kernels is that narrow programs win:
# every one of them is fastest at 2 or 4 warps and degrades monotonically past
# that (the plain LayerNorm goes 3.72 -> 2.99 -> 1.98 TB/s from 2 to 8 to 16
# warps). These are streaming kernels with one pass over their data, so what
# matters is how many independent programs are resident issuing loads, not how
# wide each one is.
#
# The awkward widths -- 1152 for the embedding, 36 for the rotary half -- cannot
# be an exact ``tl.arange``. The rotary covers its half-width with two exact
# power-of-two lane groups; the two norms keep one padded group, because that is
# what measured faster for them. See :func:`_split_pow2`.
_LN_ROWS = 2
_LN_WARPS = 2
_ADD_LN_ROWS = 4
_ADD_LN_WARPS = 4
_ROTARY_TOKENS = 1
_ROTARY_WARPS = 2
_ACT_BLOCK = 4096
_ACT_WARPS = 4

# CUDA-graph replay of the fused path, for token counts where it pays.
#
# Replay removes a roughly constant ~200 us of host launch cost and adds a copy
# cost proportional to the token count (caller inputs into graph-owned statics,
# plus a clone of the result so graph storage never escapes). So it wins exactly
# where the block is host-bound, which is the small end. Measured on this B200
# (``profile/phase1_probes/probe_cuda_graph.py``, and see
# ``profile/cuda_graph_v1/REPORT.md``):
#
#   N        eager     replay    delta   host saved   speedup vs baseline
#   -----  --------  --------  -------  -----------  ---------------------
#    1760   243.7 us  192.4 us   -51.2       -191.3   1.577x -> 1.997x
#    3072   253.9 us  274.4 us   +20.5       -194.5   1.475x -> 1.365x
#   20680   825.4 us  871.5 us   +46.1       -191.3   1.295x -> 1.227x
#   64680  3238.7 us 3462.0 us  +223.3       -198.6   1.195x -> 1.118x
#
# Replay saves the same ~190 us of host time at every shape, so where it loses it
# is not because the saving shrank -- it is because the *device* work grew, by
# more than the input copies account for (14.6 MB at N=3072 is ~3 us). The graph's
# private memory pool holds the captured intermediates apart from the caching
# allocator's, and past a few thousand tokens that costs more in locality than the
# launch cost it removes.
#
# Output is bit-identical to the eager path at every shape, so this is purely a
# performance gate. It sits between the measured win at 1760 and the measured loss
# at 3072 rather than at a round number chosen for looks.
_ENABLE_CUDA_GRAPHS = True
_CUDA_GRAPH_MAX_TOKENS = 2048


def _split_pow2(width: int) -> tuple[int, int]:
    """Split *width* into two power-of-two lane counts covering it exactly.

    ``tl.arange`` needs a power of two, and neither width that matters here is
    one: 1152 for the embedding, 36 for the rotary half. The obvious answer is to
    round up to the next power of two and mask, and an ncu pass
    (``profile/glue_kernels_v1/REPORT.md``) said that was costing the rotary
    real throughput -- 5.50 sectors per load request against 16.00 for a
    well-coalesced kernel, because a 32-lane request landing in the padding has
    only a handful of active lanes.

    Splitting exactly is possible for both shapes (1152 = 1024 + 128,
    36 = 32 + 4) at the cost of one extra load/store pair. Measured
    best-config-against-best-config at N=20680, it helps only the rotary:

    ======================  ==========  =========
    kernel                      padded      split
    ======================  ==========  =========
    rotary                     48.1 us  *42.1 us*
    LayerNorm                 *25.6 us*   31.7 us
    residual-add + LayerNorm  *35.9 us*   37.9 us
    ======================  ==========  =========

    So the rotary splits and the two norms stay padded. The difference is what
    each kernel is short of: the rotary was losing memory *requests* to
    fragmented lanes, which the split fixes, while the norms were never
    request-bound, and for them the extra load/store pair is just more
    instructions. Which is the general lesson -- "avoid masked padding" is not a
    rule, it is a hypothesis to measure per kernel.
    """
    if width & (width - 1) == 0:  # already a power of two: halve it
        return width // 2, width // 2
    low = 1 << (width.bit_length() - 1)
    return low, triton.next_power_of_2(width - low)


@triton.jit
def _tanh(x):
    # Triton 3.6 exposes no tanh intrinsic. tanh(x) == 2*sigmoid(2x) - 1 is
    # exact, and in fp32 lands far inside a bf16 ulp of libm's tanh.
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _layer_norm_kernel(
    X, W, B, Y,
    n_rows,
    stride_x, stride_y,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
):
    """LayerNorm over the last dim, bf16 in and out, fp32 reduction.

    Same order of operations as ``vectorized_layer_norm_kernel<BFloat16, float,
    false>``: fp32 accumulation, affine parameters applied in fp32, ``eps`` added
    to the variance rather than inside the square root.

    One padded lane group over the 1152-wide row, held in registers across the
    reduction. Two structural alternatives were tried and both measured slower:
    splitting the row into exact power-of-two lane groups (see :func:`_split_pow2`)
    and a two-pass version that holds no whole row, which ncu's register-pressure
    reading suggested and which came out ~1.6x slower
    (``profile/phase1_probes/probe_layernorm_2pass.py``).
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    col_ok = cols < N_COLS
    mask = (rows[:, None] < n_rows) & col_ok[None, :]

    x = tl.load(X + rows[:, None] * stride_x + cols[None, :],
                mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N_COLS
    centered = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / N_COLS
    rstd = tl.rsqrt(var + eps)

    w = tl.load(W + cols, mask=col_ok, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_ok, other=0.0).to(tl.float32)
    y = centered * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + rows[:, None] * stride_y + cols[None, :],
             y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _add_layer_norm_kernel(
    X, R, W, B, H, Y,
    n_rows,
    stride_x, stride_r, stride_h, stride_y,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
):
    """``h = x + r`` then ``LayerNorm(h)``, in one pass.

    Reads x and r, writes h (the block needs it again for the final residual)
    and the normalized result: 190 MB instead of the 238 MB the separate add and
    norm move, and one launch instead of two.

    The statistics come from ``h`` *after* it is rounded to the output dtype,
    because that is what the baseline norms -- it materializes a bf16 residual
    sum and hands that to ``F.layer_norm``. Computing them from the fp32 sum
    would be the more accurate thing to do and a different function.
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    col_ok = cols < N_COLS
    mask = (rows[:, None] < n_rows) & col_ok[None, :]

    x = tl.load(X + rows[:, None] * stride_x + cols[None, :],
                mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + rows[:, None] * stride_r + cols[None, :],
                mask=mask, other=0.0).to(tl.float32)
    h = (x + r).to(H.dtype.element_ty)
    tl.store(H + rows[:, None] * stride_h + cols[None, :], h, mask=mask)

    hf = h.to(tl.float32)
    mean = tl.sum(hf, axis=1) / N_COLS
    centered = tl.where(mask, hf - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / N_COLS
    rstd = tl.rsqrt(var + eps)

    w = tl.load(W + cols, mask=col_ok, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_ok, other=0.0).to(tl.float32)
    y = centered * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + rows[:, None] * stride_y + cols[None, :],
             y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rotary_qk_kernel(
    QKV, COS, SIN,
    n_tokens,
    stride_tok, stride_cos, stride_sin,
    HALF: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    TOKENS: tl.constexpr,
):
    """Non-interleaved RoPE applied in place to q and k inside the qkv buffer.

    ``QKV`` is the ``[tokens, 3 * heads * head_dim]`` projection output. q and k
    are adjacent, so the ``GROUPS = 2 * heads`` head-blocks at offsets
    ``g * head_dim`` cover exactly the q|k span and v is never touched. Each
    program takes whole tokens, so the ``lo`` and ``hi`` loads of consecutive
    groups jointly consume the full contiguous span rather than the 2.3 KB
    fragments the vendored rotary's ``BLOCK_M=8, BLOCK_H=2`` tiling reads.

    Math matches ``flash_attn.ops.triton.rotary`` for ``rotary_dim ==
    head_dim``: halves at ``[0, HALF)`` and ``[HALF, 2*HALF)``, everything
    upcast to fp32, ``o0 = x0*cos - x1*sin``, ``o1 = x0*sin + x1*cos``. The
    position index is the global token row, as it is on the baseline's
    non-varlen rotary call, so ``cu_seqlens`` plays no part.
    """
    tok = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    # The group axis is 2 * num_heads, which need not be a power of two, so it is
    # padded to BLOCK_G and masked. For the benched 16 heads BLOCK_G == GROUPS
    # == 32 and g_ok is uniformly true, so this costs nothing there; it exists so
    # a legitimate head count like 12 falls into a correct launch rather than a
    # tl.arange that will not compile.
    g = tl.arange(0, BLOCK_G)
    g_ok = g < GROUPS
    tok_ok = tok < n_tokens
    # Two exact lane groups over the half-width rather than one padded group; see
    # _split_pow2. For head_dim 72 this is 36 = 32 + 4, and the 32-lane part --
    # eight ninths of the traffic -- has no masked lanes at all.
    da = tl.arange(0, BLOCK_A)
    db = BLOCK_A + tl.arange(0, BLOCK_B)
    b_ok = db < HALF

    base = (tok[:, None, None] * stride_tok + g[None, :, None] * (2 * HALF))
    live = tok_ok[:, None, None] & g_ok[None, :, None]
    mask_a = live
    mask_b = live & b_ok[None, None, :]

    cos_a = tl.load(COS + tok[:, None] * stride_cos + da[None, :],
                    mask=tok_ok[:, None], other=0.0).to(tl.float32)[:, None, :]
    sin_a = tl.load(SIN + tok[:, None] * stride_sin + da[None, :],
                    mask=tok_ok[:, None], other=0.0).to(tl.float32)[:, None, :]
    cos_b = tl.load(COS + tok[:, None] * stride_cos + db[None, :],
                    mask=tok_ok[:, None] & b_ok[None, :], other=0.0
                    ).to(tl.float32)[:, None, :]
    sin_b = tl.load(SIN + tok[:, None] * stride_sin + db[None, :],
                    mask=tok_ok[:, None] & b_ok[None, :], other=0.0
                    ).to(tl.float32)[:, None, :]

    lo_a = base + da[None, None, :]
    lo_b = base + db[None, None, :]
    dt = QKV.dtype.element_ty

    xa0 = tl.load(QKV + lo_a, mask=mask_a, other=0.0).to(tl.float32)
    xa1 = tl.load(QKV + lo_a + HALF, mask=mask_a, other=0.0).to(tl.float32)
    tl.store(QKV + lo_a, (xa0 * cos_a - xa1 * sin_a).to(dt), mask=mask_a)
    tl.store(QKV + lo_a + HALF, (xa0 * sin_a + xa1 * cos_a).to(dt), mask=mask_a)

    xb0 = tl.load(QKV + lo_b, mask=mask_b, other=0.0).to(tl.float32)
    xb1 = tl.load(QKV + lo_b + HALF, mask=mask_b, other=0.0).to(tl.float32)
    tl.store(QKV + lo_b, (xb0 * cos_b - xb1 * sin_b).to(dt), mask=mask_b)
    tl.store(QKV + lo_b + HALF, (xb0 * sin_b + xb1 * cos_b).to(dt), mask=mask_b)


@triton.jit
def _activation_kernel(X, n_elem, KIND: tl.constexpr, BLOCK: tl.constexpr):
    """Elementwise activation in place over a flat buffer, fp32 math."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    if KIND == 0:
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
        y = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    elif KIND == 1:
        # tanh approximation: 0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
        y = 0.5 * x * (1.0 + _tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    elif KIND == 2:
        y = x * tl.sigmoid(1.702 * x)
    else:
        y = x * tl.sigmoid(x)
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Host-side wrappers.
# ---------------------------------------------------------------------------
def fused_layer_norm(x: torch.Tensor, weight: torch.Tensor,
                     bias: torch.Tensor, eps: float) -> torch.Tensor:
    """``F.layer_norm(x, (x.shape[-1],), weight, bias, eps)`` for a 2-D ``x``."""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_rows == 0:
        return out
    _layer_norm_kernel[(triton.cdiv(n_rows, _LN_ROWS),)](
        x, weight, bias, out,
        n_rows,
        x.stride(0), out.stride(0),
        eps,
        N_COLS=n_cols,
        BLOCK_N=triton.next_power_of_2(n_cols),
        ROWS=_LN_ROWS,
        num_warps=_LN_WARPS,
    )
    return out


def fused_add_layer_norm(x: torch.Tensor, residual: torch.Tensor,
                         weight: torch.Tensor, bias: torch.Tensor,
                         eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x + residual, LayerNorm(x + residual))`` for 2-D inputs."""
    n_rows, n_cols = x.shape
    h = torch.empty_like(x)
    out = torch.empty_like(x)
    if n_rows == 0:
        return h, out
    _add_layer_norm_kernel[(triton.cdiv(n_rows, _ADD_LN_ROWS),)](
        x, residual, weight, bias, h, out,
        n_rows,
        x.stride(0), residual.stride(0), h.stride(0), out.stride(0),
        eps,
        N_COLS=n_cols,
        BLOCK_N=triton.next_power_of_2(n_cols),
        ROWS=_ADD_LN_ROWS,
        num_warps=_ADD_LN_WARPS,
    )
    return h, out


def fused_rotary_qk_(qkv: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    """Rotate q and k in place inside a ``[tokens, 3, heads, head_dim]`` buffer.

    ``v`` (index 2 of the second axis) is left untouched.
    """
    n_tokens, three, heads, head_dim = qkv.shape
    assert three == 3 and head_dim == 2 * cos.shape[-1]
    if n_tokens == 0:
        return qkv
    half = head_dim // 2
    block_a, block_b = _split_pow2(half)
    _rotary_qk_kernel[(triton.cdiv(n_tokens, _ROTARY_TOKENS),)](
        qkv, cos, sin,
        n_tokens,
        qkv.stride(0), cos.stride(0), sin.stride(0),
        HALF=half,
        GROUPS=2 * heads,
        BLOCK_G=triton.next_power_of_2(2 * heads),
        BLOCK_A=block_a,
        BLOCK_B=block_b,
        TOKENS=_ROTARY_TOKENS,
        num_warps=_ROTARY_WARPS,
    )
    return qkv


def activation_kind(act_fn: Callable[[torch.Tensor], torch.Tensor]) -> int | None:
    """The kernel's activation code for *act_fn*, or ``None`` if unrecognized.

    Dispatch is on **exact** type identity, not ``isinstance``. A subclass may
    override ``forward`` to compute something else entirely while still passing
    an ``isinstance`` check, and substituting the built-in formula for it would
    silently compute a different function -- the same failure mode the draft
    rejects ``torch._addmm_activation`` for. Anything not recognized exactly
    returns ``None`` so the caller runs it eagerly.
    """
    # Plain functions first: they have no useful type to compare.
    if act_fn is F.silu:
        return _ACT_SILU
    if act_fn is F.gelu:
        # Called as act_fn(x), so the default approximate="none" applies.
        return _ACT_GELU_ERF
    cls = type(act_fn)
    if cls is QuickGELU:
        return _ACT_QUICKGELU
    if cls is nn.SiLU:
        return _ACT_SILU
    if cls is GELU or cls is nn.GELU:
        approximate = getattr(act_fn, "approximate", "none")
        if approximate == "none":
            return _ACT_GELU_ERF
        if approximate == "tanh":
            return _ACT_GELU_TANH
    return None


def activation_fingerprint(act_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Everything about *act_fn* that changes what it computes.

    Caching the resolved kind against the activation *object* alone is not
    enough: ``GELU`` reads ``self.approximate`` on every call, so mutating that
    attribute in place changes the function without changing the object. The
    fingerprint is compared by value, and the object itself is held alongside it
    so identity is checked too.
    """
    return (type(act_fn), getattr(act_fn, "approximate", None))


def fused_activation_(x: torch.Tensor,
                      act_fn: Callable[[torch.Tensor], torch.Tensor],
                      kind: int | None = None) -> torch.Tensor | None:
    """Apply *act_fn* in place to a contiguous *x*; ``None`` if unsupported."""
    if kind is None:
        kind = activation_kind(act_fn)
    if kind is None or not x.is_contiguous():
        return None
    n_elem = x.numel()
    if n_elem:
        _activation_kernel[(triton.cdiv(n_elem, _ACT_BLOCK),)](
            x, n_elem, KIND=kind, BLOCK=_ACT_BLOCK, num_warps=_ACT_WARPS,
        )
    return x


# ---------------------------------------------------------------------------
# Submodules. Both subclass the baseline so ``__init__`` -- and therefore the
# parameter names and shapes the harness loads by name -- is inherited untouched.
# ---------------------------------------------------------------------------
class FusedVisionAttention(VisionAttention):
    """Attention with the q/k gather deleted and RoPE done in place.

    The baseline materializes a permuted contiguous copy of q and k (124 us at
    N=20680) purely so the rotary and FlashAttention see a
    ``(2*batch, seq, heads, dim)`` tensor, then rotates it (101 us). Both go
    away: the rotary works in place on the projection output, and
    FlashAttention-4 takes strided views of that buffer -- stride
    ``(3*heads*head_dim, head_dim, 1)`` rather than ``(heads*head_dim,
    head_dim, 1)``. Verified on this build to give bit-identical output at
    identical runtime; ``profile/phase1_probes/probe_strided_fa.py`` is the
    check, and it is re-run as part of the test suite because this is a
    behaviour of a vendored CuTeDSL kernel rather than a documented guarantee.

    ``v`` needs no copy at all: batch_size is 1, so slicing the buffer already
    yields its rows in the token order q and k are in.

    Padding the head dim from 72 to 96 so FlashAttention gets an aligned tile was
    implemented, tested and measured this round, and is **not** used: it loses
    end to end in three of three A/B rounds despite the attention call itself
    being 1.32x faster. ``profile/head_padding_v1/REPORT.md`` has the numbers and
    ``profile/phase1_probes/probe_head_padding.py`` reproduces them.
    """

    def forward_fused(
        self, normed: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        n_tokens = normed.shape[0]
        qkv = self.qkv(normed)
        buf = qkv.view(n_tokens, 3, self.num_heads, self.head_dim)
        fused_rotary_qk_(buf, rotary_pos_emb_cos, rotary_pos_emb_sin)
        q, k, v = buf[:, 0], buf[:, 1], buf[:, 2]

        out = self.attn(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            softmax_scale=self.head_dim ** -0.5,
            causal=False,
            # Same reason as the baseline: FA4's auto split-KV heuristic picks a
            # kernel variant that fails to compile on SM100 in this build, and
            # balanced encoder self-attention never benefits from splitting.
            num_splits=1,
        )
        return self.proj(out.view(n_tokens, -1))


class FusedVisionMLP(VisionMLP):
    """MLP with the activation fused into a single in-place pass.

    The fc1 output is a fresh buffer, so writing the activation back into it is
    safe and saves an allocation plus a full read/write pair.
    """

    def forward_fused(self, x: torch.Tensor, act_kind: int) -> torch.Tensor:
        h = self.fc1(x)
        activated = fused_activation_(h, self.act_fn, kind=act_kind)
        if activated is None:  # pragma: no cover - guard makes this unreachable
            activated = self.act_fn(h)
        return self.fc2(activated)


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        # promote_fp32=False to match the baseline (and vLLM, whose vision blocks
        # use a plain ``nn.LayerNorm`` on the bf16 activations).
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.attn = FusedVisionAttention(embed_dim, num_heads)
        self.mlp = FusedVisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)
        # Resolved on the first forward rather than here, and re-resolved if the
        # activation object is replaced -- the same trap the baseline LayerNorm
        # documents for its fp32 weight cache. Caching on first use alone would
        # silently keep computing the old activation for a caller that swapped
        # ``mlp.act_fn`` afterwards. These are plain attributes, not buffers, so
        # they cannot appear in state_dict.
        self._act_kind: int | None = None
        self._act_src: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._act_print: tuple | None = None
        self._act_resolved = False
        # CUDA-graph state. Plain attributes, so they cannot reach state_dict.
        self._graph_cache: dict = {}
        self._graph_error: str | None = None

    def _fusable(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                 cos: torch.Tensor | None, sin: torch.Tensor | None,
                 max_seqlen: int | None) -> bool:
        """Whether the fused path covers this call.

        Deliberately narrow. Everything the captured shapes exercise passes;
        anything else takes the baseline sequence, which is the correctness net.
        """
        if torch.is_grad_enabled():
            # The Triton kernels write their outputs directly, so nothing on the
            # fused path is connected to the autograd graph -- a training caller
            # would get silently wrong gradients rather than an error. Falling
            # back whenever grad mode is on keeps that impossible. It costs the
            # benchmark nothing (``_time_module`` and the correctness rounds both
            # run under ``torch.no_grad()``) and costs a real inference caller
            # nothing either, since those run under no_grad or inference_mode.
            return False
        act_fn = self.mlp.act_fn
        fingerprint = activation_fingerprint(act_fn)
        if (not self._act_resolved or self._act_src is not act_fn
                or self._act_print != fingerprint):
            self._act_kind = activation_kind(act_fn)
            self._act_src = act_fn
            self._act_print = fingerprint
            self._act_resolved = True
        if self._act_kind is None:
            return False
        if not isinstance(max_seqlen, int):
            # None would make the baseline derive it with a device->host sync.
            return False
        if cos is None or sin is None:
            return False
        if x.dim() != 3 or x.shape[1] != 1:
            return False
        if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda:
            return False
        if not x.is_contiguous():
            return False
        if self.attn.tp_size != 1:
            return False
        head_dim = self.attn.head_dim
        if cos.shape[-1] * 2 != head_dim or sin.shape[-1] * 2 != head_dim:
            return False
        n_tokens = x.shape[0]
        if cos.shape[0] != n_tokens or sin.shape[0] != n_tokens:
            return False
        if cos.dim() != 2 or sin.dim() != 2:
            return False
        # The rotary kernel indexes cos/sin as ``row * stride(0) + lane``, so an
        # arbitrary row stride is fine but the last dim must be unit-stride.
        if cos.stride(-1) != 1 or sin.stride(-1) != 1:
            return False
        for norm in (self.norm1, self.norm2):
            if (norm.promote_fp32 or norm.weight is None or norm.bias is None
                    or norm.normalized_shape != (x.shape[-1],)):
                return False
            # On mixed affine/activation dtypes the baseline's F.layer_norm
            # raises a dtype mismatch on this build, so the fused kernel
            # quietly producing a bf16 answer would not be the same behaviour.
            # Both norms hold the dtype the harness casts them to.
            if (norm.weight.dtype != x.dtype or norm.bias.dtype != x.dtype
                    or norm.weight.stride(-1) != 1 or norm.bias.stride(-1) != 1):
                return False
        return True

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if not self._fusable(x, cu_seqlens, rotary_pos_emb_cos,
                             rotary_pos_emb_sin, max_seqlen):
            x = x + self.attn(
                self.norm1(x), cu_seqlens,
                rotary_pos_emb_cos, rotary_pos_emb_sin,
                max_seqlen,
            )
            return x + self.mlp(self.norm2(x))

        _FUSED_CALLS["n"] += 1
        if _ENABLE_CUDA_GRAPHS and x.shape[0] <= _CUDA_GRAPH_MAX_TOKENS:
            return self._forward_fused_graphed(
                x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen)
        return self._forward_fused_eager(
            x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen)

    def _forward_fused_eager(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """The fused computation itself, with no dispatch or caching in it.

        Separated out so it can be called directly during graph capture without
        re-entering the guard (which would recurse) or bumping the call counter.
        """
        n_tokens, _, embed_dim = x.shape
        rows = x.view(n_tokens, embed_dim)

        normed = fused_layer_norm(rows, self.norm1.weight, self.norm1.bias,
                                  self.norm1.eps)
        attn_out = self.attn.forward_fused(
            normed, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin,
            max_seqlen,
        )
        h, normed2 = fused_add_layer_norm(rows, attn_out, self.norm2.weight,
                                          self.norm2.bias, self.norm2.eps)
        mlp_out = self.mlp.forward_fused(normed2, self._act_kind)
        # mlp_out is fresh from fc2, so the final residual can land in it.
        return mlp_out.add_(h).view(n_tokens, 1, embed_dim)

    def _graph_key(self, x, cu_seqlens, cos, sin, max_seqlen):
        """Everything that changes what a captured graph would compute.

        A captured graph bakes in kernel launches against fixed addresses, so the
        key has to cover every property those launches depend on: each input's
        shape, stride, dtype and device (strides matter because the kernels index
        with them), the scalar ``max_seqlen`` baked into the attention launch, the
        activation the MLP dispatches to, and the *identity* of every parameter
        object. Parameter identity rather than value: replacing a parameter gives
        the captured kernels a stale address and must miss the key, whereas
        writing new values into the same parameter is safe to replay, because the
        captured kernels read that same storage.
        """
        def spec(t):
            return (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)

        return (
            spec(x), spec(cu_seqlens), spec(cos), spec(sin),
            int(max_seqlen),
            self._act_print, self._act_kind,
            self.norm1.eps, self.norm2.eps,
            tuple(id(p) for p in self.parameters()),
        )

    def _forward_fused_graphed(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Replay a captured graph of the fused path.

        Semantics are preserved by two rules. Caller inputs are copied *into*
        graph-owned tensors rather than the graph being pointed at the caller's
        (which would silently read whatever address the caller happened to pass
        last), and the result is returned as a fresh clone, so graph-owned storage
        never escapes and two consecutive calls cannot alias.
        """
        key = self._graph_key(x, cu_seqlens, rotary_pos_emb_cos,
                              rotary_pos_emb_sin, max_seqlen)
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture_graph(x, cu_seqlens, rotary_pos_emb_cos,
                                        rotary_pos_emb_sin, max_seqlen)
            if entry is None:  # capture unsupported here; stay eager for good
                return self._forward_fused_eager(
                    x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin,
                    max_seqlen)
            self._graph_cache[key] = entry
        entry["x"].copy_(x)
        entry["cu"].copy_(cu_seqlens)
        entry["cos"].copy_(rotary_pos_emb_cos)
        entry["sin"].copy_(rotary_pos_emb_sin)
        entry["graph"].replay()
        return entry["out"].clone()

    def _capture_graph(self, x, cu_seqlens, cos, sin, max_seqlen):
        """Capture the fused path over graph-owned input tensors, or ``None``."""
        statics = {
            "x": x.clone(), "cu": cu_seqlens.clone(),
            "cos": cos.clone(), "sin": sin.clone(),
        }
        try:
            # Warm up on a side stream: capture requires the caches the kernels
            # touch (Triton autotune state, the CuTeDSL launcher, cuBLAS
            # workspaces) to be populated already.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._forward_fused_eager(
                        statics["x"], statics["cu"], statics["cos"],
                        statics["sin"], max_seqlen)
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._forward_fused_eager(
                    statics["x"], statics["cu"], statics["cos"],
                    statics["sin"], max_seqlen)
        except Exception as exc:  # noqa: BLE001 - capture is best-effort
            self._graph_error = repr(exc)[:300]
            return None
        statics["graph"] = graph
        statics["out"] = out
        # Strong references to the parameters whose addresses the captured kernels
        # baked in. The cache key uses id(), and without holding the objects a
        # replaced parameter could be freed and a new one allocated at the same
        # address, making a stale entry look like a hit.
        statics["params"] = tuple(self.parameters())
        return statics
