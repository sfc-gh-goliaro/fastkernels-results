"""FLUX attention module (L2 composite) -- fused joint-QKV variant.

Same ``__init__`` / ``forward`` contract as the baseline. The difference is
structural: instead of projecting the image and text streams separately and then
copying and concatenating q/k/v, both projections write **directly into one joint
sequence buffer**, qk-norm and interleaved RoPE run as a **single Triton kernel**
over that buffer's ``q`` and ``k`` thirds, and attention reads strided views of
the thirds. That removes the ``.contiguous()`` copies of q/k, the three
concatenations, and the separate fp64->bf16 cos/sin cast, taking the dual-stream
kernel count from 17 to 6 with no q/k/v copies at all.

The GEMMs and the attention are left exactly where the baseline puts them --
cuBLAS via ``torch.addmm`` and cuDNN via the baseline's own ``DenseAttention``
submodule -- because both are already near roofline on B200 (QKV GEMM
~1.6 PFLOPS, cuDNN SDPA ~1.4 PFLOPS).

Everything the fused path is not built for takes ``_forward_reference``, which
reproduces the baseline op-for-op and doubles as the correctness oracle in tests.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import triton
import triton.language as tl

from ....infra.tp import _tp_size
from ..L1.rms_norm import RMSNorm as FP32RMSNorm
from ..L1.diffusion_rope import DiffusionRoPE
from ..L1.dense_attention import DenseAttention
from .parallel_linear import (
    QKVParallelLinear,
    RowParallelLinear,
)


def _tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather tensor across TP ranks along the given dimension."""
    import torch.distributed as dist
    tp = _tp_size()
    if tp <= 1:
        return tensor
    gather_list = [torch.empty_like(tensor) for _ in range(tp)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=dim)


# ---------------------------------------------------------------------------
# Fused qk-norm + interleaved RoPE
# ---------------------------------------------------------------------------
# The kernel walks a flattened iteration space of ``NROW = seq_len * heads`` rows
# of ``head_dim`` contiguous elements, where ``row = token * heads + head``. Each
# program owns ``ROWS_PER_PROGRAM`` consecutive rows, so its q-side load is one
# contiguous chunk and its k-side load is another such chunk a fixed distance
# away in the same buffer.
#
# When ``ROWS_PER_PROGRAM`` divides ``heads`` every row of a program belongs to
# the same token, which is what ``UNIFORM_TOKEN`` exploits: cos/sin load once per
# program as a single ``rotary_half``-wide vector instead of once per row. At
# ROWS_PER_PROGRAM=8 with head_dim=128 the per-program fp64 cos/sin tile would
# otherwise be 8 KB against 4 KB of q/k, i.e. the position table would dominate
# the bytes moved.
#
# Everything about the arithmetic below is chosen to reproduce the baseline
# bit for bit, not to be as accurate as possible. Three points matter, and each
# one was wrong in an obvious-looking spelling before it was measured:
#
#   * **The reduction order.** ``rms_norm_kernel`` at bf16 and hidden_size=128
#     launches ``vec_size = gcd(16/2, 128) = 8`` and
#     ``block_size = min(128/8, 256) = 16``, so CUDA thread ``t`` accumulates the
#     squares of elements ``[8t, 8t+8)`` in source order and ``cub::BlockReduce``
#     combines the 16 partials as an adjacent-pair tree. A single
#     ``tl.sum`` over the 128-wide axis uses a different tree and leaves ~1 element
#     in 435,000 off by one ULP -- which the rotation's subtraction then amplifies
#     into a much larger relative error on near-cancelling pairs. Reproducing the
#     order exactly makes the norm bit-identical, so there is nothing to amplify.
#     Verified in ``scratch/probe_bitexact_norm.py``.
#   * **The cos/sin cast.** The baseline casts the fp64 table with
#     ``cos.to(query.dtype)``, and ATen's ``double -> BFloat16`` goes through
#     ``float``. A direct fp64 -> bf16 narrowing rounds once instead of twice and
#     lands on the other side of a near-tie for ~3 values in 300,000 drawn from
#     ``randn`` -- not just at adversarial magnitudes. Spelling it
#     fp64 -> fp32 -> activation dtype reproduces ATen exactly. Verified in
#     ``scratch/probe_cos_cast.py``.
#   * **The intermediate rounding.** The norm result is rounded to the activation
#     dtype and widened again before the rotation, because the baseline
#     materialises a low-precision tensor between the two ops. Skipping it would
#     be *more* accurate but *different*.
#
# ``SWAP_FORM`` selects between two formulations of the interleaved (GPT-J)
# rotation. Form 0 pairs adjacent lanes with ``tl.split`` / ``tl.join``. Form 1
# mirrors the baseline rotary kernel's ``rk_swap`` + ``tl.where`` parity trick,
# re-reading the row at swapped lane offsets instead; the second read hits the
# same cache lines, and the formulation avoids the ``split``/``join`` codegen
# path entirely. Both are measured; only configurations that reproduce the
# reference chain are eligible to ship.


@triton.jit
def _sum_squares_like_rms_norm(x, ROWS: tl.constexpr, HEAD_DIM: tl.constexpr,
                               GROUPS: tl.constexpr):
    """Sum of squares in the exact order ``rms_norm_kernel`` accumulates it.

    ``GROUPS`` is the baseline's CUDA block size and ``HEAD_DIM // GROUPS`` its
    vector width, so group ``g`` corresponds to thread ``g`` and holds that
    thread's contiguous run of elements. Within a group the squares are added in
    source order; across groups they are combined as an adjacent-pair tree, which
    is what a ``shfl_down`` reduction with offsets 1, 2, 4, 8 leaves in lane 0.

    The lane extraction is register-only -- reshapes and ``tl.split`` are shuffles.
    Re-reading the row from memory to get the same control over the order would be
    far more expensive, since this kernel is L1-throughput-bound.
    """
    tl.static_assert(HEAD_DIM // GROUPS == 8, "expects the baseline's 8-wide vector")
    tl.static_assert(GROUPS == 16, "expects the baseline's 16-thread block")
    groups = tl.reshape(x, (ROWS, GROUPS, HEAD_DIM // GROUPS))
    even, odd = tl.split(tl.reshape(groups, (ROWS, GROUPS, 4, 2)))
    e0, e1 = tl.split(tl.reshape(even, (ROWS, GROUPS, 2, 2)))
    o0, o1 = tl.split(tl.reshape(odd, (ROWS, GROUPS, 2, 2)))
    a0, a4 = tl.split(e0)
    a2, a6 = tl.split(e1)
    a1, a5 = tl.split(o0)
    a3, a7 = tl.split(o1)

    part = a0 * a0
    part = part + a1 * a1
    part = part + a2 * a2
    part = part + a3 * a3
    part = part + a4 * a4
    part = part + a5 * a5
    part = part + a6 * a6
    part = part + a7 * a7

    lo, hi = tl.split(tl.reshape(part, (ROWS, 8, 2)))
    s8 = lo + hi
    lo, hi = tl.split(tl.reshape(s8, (ROWS, 4, 2)))
    s4 = lo + hi
    lo, hi = tl.split(tl.reshape(s4, (ROWS, 2, 2)))
    s2 = lo + hi
    lo, hi = tl.split(s2)
    return lo + hi


@triton.jit
def _fused_qk_norm_rope_kernel(
    QKV, Q_OUT, K_OUT, COS, SIN,
    W_Q, W_K, W_ADDED_Q, W_ADDED_K,
    n_rows, ctx_len, eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NORM_GROUPS: tl.constexpr,
    ROTARY_HALF: tl.constexpr,
    QKV_ROW: tl.constexpr,
    Q_OFFSET: tl.constexpr,
    K_OFFSET: tl.constexpr,
    OUT_ROW: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    ACT_DTYPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_CTX: tl.constexpr,
    IN_PLACE: tl.constexpr,
    UNIFORM_TOKEN: tl.constexpr,
    SWAP_FORM: tl.constexpr,
):
    row0 = tl.program_id(0) * ROWS_PER_PROGRAM
    lane = tl.arange(0, ROWS_PER_PROGRAM)
    rows = row0 + lane
    live = rows < n_rows
    cols = tl.arange(0, HEAD_DIM)

    if SWAP_FORM:
        # Form 1 needs cos/sin repeated per lane pair, matching the baseline
        # rotary kernel's ``rk_repeat``.
        rope_col = cols // 2
    else:
        rope_col = tl.arange(0, ROTARY_HALF)

    if UNIFORM_TOKEN:
        token0 = row0 // HEADS
        head = row0 - token0 * HEADS + lane
        token = token0 + lane * 0
        if HAS_ROPE:
            cos_t = tl.load(COS + token0 * ROTARY_HALF + rope_col)[None, :]
            sin_t = tl.load(SIN + token0 * ROTARY_HALF + rope_col)[None, :]
    else:
        token = rows // HEADS
        head = rows - token * HEADS
        if HAS_ROPE:
            rope_at = token[:, None] * ROTARY_HALF + rope_col[None, :]
            cos_t = tl.load(COS + rope_at, mask=live[:, None], other=1.0)
            sin_t = tl.load(SIN + rope_at, mask=live[:, None], other=0.0)

    if HAS_ROPE:
        # fp32 first: ATen narrows double -> float -> bfloat16, and rounding once
        # instead of twice disagrees on near-ties that randn actually produces.
        cos_t = cos_t.to(tl.float32).to(ACT_DTYPE).to(tl.float32)
        sin_t = sin_t.to(tl.float32).to(ACT_DTYPE).to(tl.float32)

    if HAS_CTX:
        is_text = (token < ctx_len)[:, None]

    row_at = token[:, None] * QKV_ROW + head[:, None] * HEAD_DIM
    if not IN_PLACE:
        out_at = token[:, None] * OUT_ROW + head[:, None] * HEAD_DIM

    for third in tl.static_range(2):
        if third == 0:
            base = QKV + Q_OFFSET + row_at
            w_ptr = W_Q
            w_added_ptr = W_ADDED_Q
            dst_base = Q_OUT
        else:
            base = QKV + K_OFFSET + row_at
            w_ptr = W_K
            w_added_ptr = W_ADDED_K
            dst_base = K_OUT

        src = base + cols[None, :]
        x = tl.load(src, mask=live[:, None], other=0.0).to(tl.float32)
        sum_squares = _sum_squares_like_rms_norm(
            x, ROWS_PER_PROGRAM, HEAD_DIM, NORM_GROUPS)
        inv_rms = tl.rsqrt(sum_squares / HEAD_DIM + eps)[:, None]

        w = tl.load(w_ptr + cols).to(tl.float32)[None, :]
        if HAS_CTX:
            w_added = tl.load(w_added_ptr + cols).to(tl.float32)[None, :]
            w = tl.where(is_text, w_added, w)
        normed = ((x * inv_rms) * w).to(ACT_DTYPE).to(tl.float32)

        if HAS_ROPE:
            if SWAP_FORM:
                swap = cols + ((cols + 1) % 2) * 2 - 1
                x_s = tl.load(base + swap[None, :],
                              mask=live[:, None], other=0.0).to(tl.float32)
                w_s = tl.load(w_ptr + swap).to(tl.float32)[None, :]
                if HAS_CTX:
                    w_added_s = tl.load(w_added_ptr + swap).to(tl.float32)[None, :]
                    w_s = tl.where(is_text, w_added_s, w_s)
                normed_s = ((x_s * inv_rms) * w_s).to(ACT_DTYPE).to(tl.float32)
                scaled_cos = normed * cos_t
                scaled_sin = normed_s * sin_t
                result = tl.where(cols[None, :] % 2 == 0,
                                  scaled_cos - scaled_sin,
                                  scaled_cos + scaled_sin)
            else:
                even, odd = tl.split(
                    tl.reshape(normed, (ROWS_PER_PROGRAM, ROTARY_HALF, 2)))
                result = tl.reshape(
                    tl.join(even * cos_t - odd * sin_t, odd * cos_t + even * sin_t),
                    (ROWS_PER_PROGRAM, HEAD_DIM))
        else:
            result = normed

        if IN_PLACE:
            tl.store(src, result.to(ACT_DTYPE), mask=live[:, None])
        else:
            tl.store(dst_base + out_at + cols[None, :], result.to(ACT_DTYPE),
                     mask=live[:, None])


_TL_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}

# The fast path is deliberately narrow: exactly the regime the benchmark captures.
# The kernel is written parameterically in heads, head_dim and dtype -- that is what
# keeps its indexing honest rather than hard-coded -- but admitting a configuration
# no captured case exercises would mean shipping arithmetic whose bit-exactness
# against the baseline has not been established, so everything else takes the
# reference path. Widening this needs new exactness evidence, not just a wider
# predicate.
_FUSED_DTYPE = torch.bfloat16
_FUSED_HEADS = 24
_FUSED_HEAD_DIM = 128

# Statically selected (rows_per_program, num_warps). Chosen in two stages, in this
# order, because speed is only a tie-breaker among configurations that reproduce
# the baseline: first every (tile, warps, formulation, buffer-policy) combination
# is checked against the reference chain (``scratch/test_fused_kernel.py``) and the
# ones that are not bit-exact are struck out -- one row spread over 8 warps gives
# Triton 256 threads for 128 elements and it reassociates the summation chain --
# then the survivors are ranked by whole-module latency on all four captured shapes
# (``scratch/probe_config.py``).
#
# ``num_warps=2`` wins now that the summation order is pinned: the reshape-and-split
# chain costs registers, and fewer warps leave more of them per lane. ``tile=4``
# divides ``heads``, which keeps the program-uniform cos/sin load. Splitting the
# choice by sequence length was worth under 1 %, so there is one configuration
# rather than a per-regime table.
#
# Runtime autotuning is deliberately absent: the kernel mutates its input, so an
# autotuner's trial launches would re-apply norm+RoPE to an already-rotated buffer
# and make both the timing and the result meaningless.
_ROWS_PER_PROGRAM = 4
_NUM_WARPS = 2
# Rotation formulation and buffer policy, settled by the same two-stage selection.
# Both buffer policies are exactness-eligible and both are validated end to end, so
# the choice between them is a free one; in place keeps the design's zero-copy
# property, and its stores land on lines that are still resident rather than
# reaching DRAM.
_SWAP_FORM = 0
_IN_PLACE = True


def _select_config(heads: int) -> tuple[int, int]:
    rows_per_program = _ROWS_PER_PROGRAM
    # The program-uniform token path requires rows_per_program to divide heads;
    # anything else would compute the wrong token for part of the tile, so shrink
    # to a tile size that is both a power of two and a divisor.
    while rows_per_program > 1 and heads % rows_per_program != 0:
        rows_per_program //= 2
    return rows_per_program, _NUM_WARPS


def _fused_qk_norm_rope(
    qkv: torch.Tensor,
    seq_len: int,
    heads: int,
    head_dim: int,
    q_offset: int,
    k_offset: int,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_added_q: torch.Tensor | None,
    weight_added_k: torch.Tensor | None,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    ctx_len: int,
    eps: float,
    in_place: bool | None = None,
    swap_form: int | None = None,
    rows_per_program: int | None = None,
    num_warps: int | None = None,
):
    """Apply qk-norm then interleaved RoPE to the q and k thirds of ``qkv``.

    ``qkv`` is 2-D ``[seq_len, qkv_width]``. With ``in_place`` the thirds are
    rewritten where they are and the return is ``(q_view, k_view)`` onto the same
    storage; otherwise q and k are written to two fresh contiguous
    ``[1, seq_len, heads, head_dim]`` buffers and those are returned.

    Rows ``[0, ctx_len)`` are the text stream and use ``weight_added_*``; rows
    ``[ctx_len, seq_len)`` are the image stream and use ``weight_*``.
    """
    n_rows = seq_len * heads
    has_ctx = ctx_len > 0
    has_rope = cos is not None
    tile, warps = _select_config(heads)
    if rows_per_program is not None:
        tile = rows_per_program
    if num_warps is not None:
        warps = num_warps
    if in_place is None:
        in_place = _IN_PLACE
    if swap_form is None:
        swap_form = _SWAP_FORM
    uniform_token = heads % tile == 0

    # The baseline's qk-norm launch geometry, which the kernel's summation order
    # reproduces: vec_size = gcd(16 / element_size, head_dim) and
    # block_size = min(head_dim / vec_size, 256).
    vector_width = math.gcd(16 // qkv.element_size(), head_dim)
    norm_groups = min(head_dim // vector_width, 256)

    if in_place:
        q_out = k_out = qkv
        out_row = 0
    else:
        q_out = torch.empty(1, seq_len, heads, head_dim,
                            dtype=qkv.dtype, device=qkv.device)
        k_out = torch.empty_like(q_out)
        out_row = heads * head_dim

    # Unused pointers still have to be tensors; the constexpr branches that would
    # dereference them are compiled out.
    cos_arg = qkv if cos is None else cos
    sin_arg = qkv if sin is None else sin
    added_q_arg = weight_q if weight_added_q is None else weight_added_q
    added_k_arg = weight_k if weight_added_k is None else weight_added_k

    with torch.cuda.device(qkv.device.index):
        _fused_qk_norm_rope_kernel[(triton.cdiv(n_rows, tile),)](
            qkv, q_out, k_out, cos_arg, sin_arg,
            weight_q, weight_k, added_q_arg, added_k_arg,
            n_rows, ctx_len, eps,
            HEADS=heads,
            HEAD_DIM=head_dim,
            NORM_GROUPS=norm_groups,
            ROTARY_HALF=head_dim // 2,
            QKV_ROW=qkv.stride(0),
            Q_OFFSET=q_offset,
            K_OFFSET=k_offset,
            OUT_ROW=out_row,
            ROWS_PER_PROGRAM=tile,
            ACT_DTYPE=_TL_DTYPE[qkv.dtype],
            HAS_ROPE=has_rope,
            HAS_CTX=has_ctx,
            IN_PLACE=in_place,
            UNIFORM_TOKEN=uniform_token,
            SWAP_FORM=bool(swap_form),
            num_warps=warps,
        )

    if in_place:
        joint = qkv.view(1, seq_len, qkv.shape[1])
        span = heads * head_dim
        return (joint[..., q_offset:q_offset + span].unflatten(-1, (heads, head_dim)),
                joint[..., k_offset:k_offset + span].unflatten(-1, (heads, head_dim)))
    return q_out, k_out


def _autocast_active(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # older signature takes no argument
        return bool(torch.is_autocast_enabled())


def _overrides_forward(module, reference_cls) -> bool:
    """True when ``module``'s ``forward`` is not the one ``reference_cls`` defines.

    The fused path replaces the submodules' ``forward`` bodies with its own
    arithmetic, so a subclass or an instance-level override would be silently
    bypassed -- the reference path would honour it and the fused path would not.
    Anything that is not exactly the expected implementation falls back.
    """
    if type(module) is not reference_cls:
        return True
    # A bound method carries the class function in ``__func__``; a plain function
    # assigned onto the instance carries nothing, and is itself the override.
    bound = module.forward
    return getattr(bound, "__func__", bound) is not reference_cls.forward


def _is_plain_qkv_linear(layer, heads: int, head_dim: int, dtype, device) -> bool:
    """True when ``layer`` is an unquantized QKV projection the fused path can
    drive with a single ``addmm`` into a row slice."""
    if layer is None or _overrides_forward(layer, QKVParallelLinear):
        return False
    if getattr(layer, "use_fp8", False):
        return False
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return False
    if weight.dtype != dtype or weight.device != device:
        return False
    if layer.num_heads != heads or layer.num_kv_heads != heads:
        return False
    if layer.head_size != head_dim:
        return False
    # The kernel's third offsets are derived from this identity; a width that does
    # not decompose into (q, k, v) blocks of the stated geometry would make it
    # read the wrong region.
    if weight.shape[0] != (layer.num_heads + 2 * layer.num_kv_heads) * head_dim:
        return False
    bias = getattr(layer, "bias", None)
    if bias is not None and (bias.dtype != dtype or bias.device != device
                             or bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
        return False
    return True


def _norm_weight(norm, head_dim: int, eps: float, dtype, device):
    """The norm's weight tensor if it is fusable, else ``None``.

    The weight has to already be in the activation dtype: ``RMSNorm.forward``
    casts it before calling its CUDA kernel, so loading a higher-precision weight
    straight into fp32 registers would not reproduce the baseline's arithmetic.
    """
    if norm is None or _overrides_forward(norm, FP32RMSNorm):
        return None
    if not getattr(norm, "elementwise_affine", False):
        return None
    if getattr(norm, "hidden_size", None) != head_dim:
        return None
    if norm.eps != eps:
        return None
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    if (weight.dtype != dtype or weight.device != device
            or weight.dim() != 1 or weight.shape[0] != head_dim):
        return None
    return weight


class FluxAttention(nn.Module):
    """Multi-head attention for FLUX diffusion transformer.

    Supports two modes controlled by constructor args:
    - Dual-stream (``added_kv_proj_dim is not None``): separate QKV for image
      and text streams, concatenated before attention, split after.
    - Single-stream / pre-only (``pre_only=True``): standard self-attention,
      no output projection (caller handles it).
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q = FP32RMSNorm(dim_head, eps=eps)
        self.norm_k = FP32RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            total_num_kv_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias,
                                  quant_config=quant_config),
                nn.Dropout(dropout),
            ])

        if added_kv_proj_dim is not None:
            self.norm_added_q = FP32RMSNorm(dim_head, eps=eps)
            self.norm_added_k = FP32RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                total_num_kv_heads=self.heads,
                bias=added_proj_bias if added_proj_bias is not None else True,
                quant_config=quant_config,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim, query_dim, bias=out_bias,
                quant_config=quant_config,
            )

        self.rope = DiffusionRoPE(is_neox_style=False)
        self.attn = DenseAttention()

    # -- shared helpers ----------------------------------------------------

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)
        return query, key

    # -- fused path --------------------------------------------------------

    def _fusable(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        """Return the fused path's parameters, or ``None`` to take the reference.

        Every condition here guards either memory safety (a geometry the kernel's
        offsets do not describe) or numerical fidelity (a dtype or rounding the
        kernel does not reproduce). Anything unrecognized falls back rather than
        being approximated.
        """
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return None
        if not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda:
            return None
        if _autocast_active(hidden_states.device.type):
            return None
        if _tp_size() != 1:
            return None

        dtype, device = hidden_states.dtype, hidden_states.device
        if dtype is not _FUSED_DTYPE:
            return None
        if hidden_states.dim() != 3 or hidden_states.shape[0] != 1:
            return None
        if not hidden_states.is_contiguous():
            return None

        head_dim = self.head_dim
        if head_dim != _FUSED_HEAD_DIM:
            return None
        heads = self.to_qkv.num_heads
        if heads != _FUSED_HEADS:
            return None
        if not _is_plain_qkv_linear(self.to_qkv, heads, head_dim, dtype, device):
            return None
        if hidden_states.shape[2] != self.to_qkv.weight.shape[1]:
            return None

        eps = self.norm_q.eps
        weight_q = _norm_weight(self.norm_q, head_dim, eps, dtype, device)
        weight_k = _norm_weight(self.norm_k, head_dim, eps, dtype, device)
        if weight_q is None or weight_k is None:
            return None

        image_rows = hidden_states.shape[1]
        # An empty image stream makes the baseline launch a zero-block norm and
        # fail; the fused path would quietly succeed on the remaining rows.
        if image_rows == 0:
            return None
        if encoder_hidden_states is None:
            # A dual-configured module called without an encoder stream is a
            # configuration the baseline itself rejects (it projects ``None``);
            # route it to the reference so it fails the same way.
            if self.added_kv_proj_dim is not None:
                return None
            ctx_len = 0
            weight_added_q = weight_added_k = None
            add_layer = None
        else:
            if self.added_kv_proj_dim is None:
                return None
            enc = encoder_hidden_states
            if not isinstance(enc, torch.Tensor) or enc.dim() != 3:
                return None
            if enc.shape[0] != 1 or enc.dtype != dtype or enc.device != device:
                return None
            if not enc.is_contiguous():
                return None
            add_layer = getattr(self, "add_kv_proj", None)
            if not _is_plain_qkv_linear(add_layer, heads, head_dim, dtype, device):
                return None
            if enc.shape[2] != add_layer.weight.shape[1]:
                return None
            if add_layer.weight.shape[0] != self.to_qkv.weight.shape[0]:
                return None
            # Both projections must carry a bias or neither: the joint buffer is
            # filled by two calls that have to agree on the epilogue.
            if (add_layer.bias is None) != (self.to_qkv.bias is None):
                return None
            weight_added_q = _norm_weight(getattr(self, "norm_added_q", None),
                                          head_dim, eps, dtype, device)
            weight_added_k = _norm_weight(getattr(self, "norm_added_k", None),
                                          head_dim, eps, dtype, device)
            if weight_added_q is None or weight_added_k is None:
                return None
            ctx_len = enc.shape[1]
            if ctx_len == 0:
                return None
        seq_len = ctx_len + image_rows

        # Every captured case supplies a rotary table; a norm-only call is not a
        # regime this path has exactness evidence for.
        if image_rotary_emb is None:
            return None
        # The kernel implements the interleaved (GPT-J) rotation only; the
        # half-split NeoX layout is a different permutation entirely.
        if _overrides_forward(self.rope, DiffusionRoPE) or not self.rope.interleaved:
            return None
        if not isinstance(image_rotary_emb, (list, tuple)) or len(image_rotary_emb) != 2:
            return None
        cos, sin = image_rotary_emb
        if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
            return None
        if cos.shape != sin.shape or cos.dtype != sin.dtype:
            return None
        # A complex table would be truncated to its real part by the baseline's
        # cast; Triton has no pointer type for it at all.
        if not cos.is_floating_point():
            return None
        # A 3-D table is squeezed by the baseline's RoPE module and a longer table
        # is legal there too, but neither is a captured shape, so both fall back
        # rather than being fused on the strength of an untested index path.
        if cos.dim() != 2 or cos.shape[0] != seq_len:
            return None
        if cos.device != device or sin.device != device:
            return None
        # Only full rotary: the kernel rotates every lane of the head.
        if 2 * cos.shape[-1] != head_dim:
            return None
        cos, sin = cos.contiguous(), sin.contiguous()

        return dict(heads=heads, head_dim=head_dim, ctx_len=ctx_len,
                    image_rows=image_rows, seq_len=seq_len, eps=eps,
                    dtype=dtype, device=device, add_layer=add_layer,
                    weight_q=weight_q, weight_k=weight_k,
                    weight_added_q=weight_added_q, weight_added_k=weight_added_k,
                    cos=cos, sin=sin)

    @staticmethod
    def _project_into(layer, x, destination):
        """``F.linear(x, layer.weight, layer.bias)`` written straight into
        ``destination``, which must be a full-width row slice.

        ``addmm`` into such a slice is bit-identical to ``F.linear`` and runs at
        the same speed, because a full-width row slice of a 2-D buffer has the same
        layout cuBLAS would have written on its own. A column slice does not: it is
        non-contiguous, and the two assertions below are what stop that from being
        written by accident, since ``out=`` would otherwise either fail obscurely or
        silently produce a different layout.
        """
        assert destination.shape[1] == layer.weight.shape[0], (
            f"destination is {destination.shape[1]} wide but the projection "
            f"produces {layer.weight.shape[0]}; only a full-width row slice is "
            "a valid GEMM destination")
        assert destination.is_contiguous(), (
            "destination row slice must be contiguous")
        rows = destination.shape[0]
        flat = x.reshape(rows, x.shape[-1])
        if layer.bias is None:
            torch.mm(flat, layer.weight.t(), out=destination)
        else:
            torch.addmm(layer.bias, flat, layer.weight.t(), out=destination)

    def _forward_fused(self, hidden_states, encoder_hidden_states, plan):
        heads, head_dim = plan["heads"], plan["head_dim"]
        seq_len, ctx_len = plan["seq_len"], plan["ctx_len"]
        span = heads * head_dim
        width = self.to_qkv.weight.shape[0]

        qkv = torch.empty(seq_len, width, dtype=plan["dtype"], device=plan["device"])
        # Advance a cursor only when a projection is actually issued, then check it
        # against the buffer's own row count. Comparing the two stream lengths to
        # their own sum would prove nothing; this catches a projection that was
        # skipped or given the wrong row range, which would otherwise leave
        # uninitialized rows for attention to read.
        written = 0
        if ctx_len:
            # Text rows first, matching the baseline's ``cat([encoder, image])``,
            # so the RoPE table indexes joint positions directly.
            self._project_into(plan["add_layer"], encoder_hidden_states,
                               qkv[written:written + ctx_len])
            written += ctx_len
        image_rows = hidden_states.shape[1]
        self._project_into(self.to_qkv, hidden_states,
                           qkv[written:written + image_rows])
        written += image_rows
        assert written == qkv.shape[0], (
            f"the projections wrote {written} of {qkv.shape[0]} joint-buffer rows")

        query, key = _fused_qk_norm_rope(
            qkv, seq_len, heads, head_dim, 0, span,
            plan["weight_q"], plan["weight_k"],
            plan["weight_added_q"], plan["weight_added_k"],
            plan["cos"], plan["sin"], ctx_len, plan["eps"],
        )
        value = qkv.view(1, seq_len, width)[..., 2 * span:3 * span].unflatten(
            -1, (heads, head_dim))

        # Spelled the way the baseline spells it. ``head_dim ** -0.5`` does not round
        # to the same double (0.08838834764831845 against 0.08838834764831843), and
        # while that difference turned out to change no bf16 output at the captured
        # shapes (``scratch/probe_cudnn_values.py``), passing the baseline's exact
        # value costs nothing and removes the question.
        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = self.attn(query, key, value,
                                  softmax_scale=softmax_scale, causal=False)
        hidden_states = hidden_states.flatten(2, 3).to(plan["dtype"])

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [ctx_len, seq_len - ctx_len], dim=1,
            )
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        if _tp_size() > 1:
            hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
        return hidden_states

    # -- reference path ----------------------------------------------------

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The baseline's ``forward``, op for op.

        Serves both as the fallback for every configuration the fused path is not
        built for and as the oracle the fused path is tested against.
        """
        num_heads = self.to_qkv.num_heads
        num_kv_heads = self.to_qkv.num_kv_heads

        qkv = self.to_qkv(hidden_states)
        q_size = num_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (num_heads, -1))
        key = key.unflatten(-1, (num_kv_heads, -1))
        value = value.unflatten(-1, (num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            add_num_heads = self.add_kv_proj.num_heads
            add_num_kv_heads = self.add_kv_proj.num_kv_heads

            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            add_q_size = add_num_heads * self.head_dim
            add_kv_size = add_num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (add_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (add_num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (add_num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        query, key = self._apply_rope(query, key, image_rotary_emb)

        softmax_scale = 1.0 / (self.head_dim ** 0.5)
        hidden_states = self.attn(query, key, value, softmax_scale=softmax_scale,
                                  causal=False)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1],
                 hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        else:
            if _tp_size() > 1:
                hidden_states = _tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states

    # -- entry point -------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        plan = self._fusable(hidden_states, encoder_hidden_states, image_rotary_emb)
        if plan is None:
            return self._forward_reference(
                hidden_states, encoder_hidden_states, image_rotary_emb)
        return self._forward_fused(hidden_states, encoder_hidden_states, plan)
