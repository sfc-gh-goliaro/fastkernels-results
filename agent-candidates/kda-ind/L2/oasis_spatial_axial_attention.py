"""Oasis spatial axial attention.

The baseline dispatches 31 CUDA kernels per call for only 87-101 us of GPU work, so
the operator is launch-bound rather than compute-bound and the optimization is
op-count reduction. This candidate keeps the baseline's math and collapses the call
into three dispatches --

    F.linear(x2d, W_qkv)              -> qkv [M, 3*heads*dim_head]
    fused_rope_attn(qkv, cos, sin)    -> ctx [M, heads*dim_head]
    F.linear(ctx, W_out, b_out)       -> [M, dim]

-- where M = bsz * time * height * width and the reshapes around them are views.

Everything outside the fast path's envelope falls back to a transcription of the
baseline, so the module stays a drop-in replacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_rotate_half

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - the eager path stays available
    _HAVE_TRITON = False


# Tile shape, warp count and pipeline depth of the fused kernel, hard-coded so no
# autotuning can run inside a timed region. Selected by profile/config_sweep_final.csv:
# all five captured shapes, balanced randomized round-robin blocks, the statistic being
# the median across blocks of the within-block paired worst-shape latency, with a
# bootstrap interval, an explicit tied set, and a predeclared tie-break.
#
# BLOCK_M = 256 covers S = 144 in a single query tile, so each (frame, head) is one CTA
# that loads K and V exactly once rather than once per query tile, and num_stages = 3
# matches the three key blocks the loop actually runs.
#
# Worth knowing before changing these. The margin is one CUDA-event tick: this
# configuration measures 29.66 us against a cluster of sixteen at 29.70-29.79. It led in
# two of three runs of the selection, and lost in the third -- a run where the whole level
# was about 1.45x slower, i.e. the machine was heavily contended, which plausibly penalizes
# a 194-register 12.5%-occupancy kernel more than a lighter one. So this is a recorded,
# auditable choice, not a tuned optimum, and the operator's cost is dispatch rather than
# the tile shape either way.
_BLOCK_M = 256
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 3

_LOG2_E = 1.4426950408889634


if _HAVE_TRITON:

    @triton.jit
    def _fp16_mul_add(a, b, c, d):
        """``fp16(fp16(a*b) + fp16(c*d))`` with both products separately rounded.

        Two `mul.rn.f16x2` and one `add.rn.f16x2`, emitted as inline PTX so the
        compiler cannot fuse a multiply into the add. `f16x2` handles two lanes per
        instruction, the same rate the contracted form would have achieved, so
        pinning the rounding costs no throughput.
        """
        return tl.inline_asm_elementwise(
            """{
                .reg .b32 lhs, rhs;
                mul.rn.f16x2 lhs, $1, $2;
                mul.rn.f16x2 rhs, $3, $4;
                add.rn.f16x2 $0, lhs, rhs;
            }""",
            "=r,r,r,r,r",
            [a, b, c, d],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )

    @triton.jit
    def _rope_interleaved(t, cos, sin, negate_even, BLOCK: tl.constexpr,
                          DH: tl.constexpr):
        """Interleaved (GPT-J style) rotary embedding on a [BLOCK, DH] tile.

        ``oasis_rotate_half`` swaps each adjacent pair and negates the even lane,
        so with the frequency table already ``repeat_interleave(2)``-ed

            out[j] = t[j] * cos[j] + sign[j] * t[j ^ 1] * sin[j]

        with ``sign[j] = -1`` on even ``j``.

        The rotated operand is built first and only then multiplied, so the
        association of the sign is fixed rather than left to the compiler, and the
        arithmetic itself is emitted as inline PTX. That is not decoration: the
        eager path rounds each product to fp16 and only *then* adds, and written as
        ordinary Triton the two products get contracted into a single fused
        multiply-add that rounds once. An explicit ``.to(tl.float16)``, an fp32
        round-trip, and an int16 bitcast barrier were all optimized away; separate
        ``mul.rn.f16x2`` and ``add.rn.f16x2`` instructions are what actually pin the
        two roundings, and they make this bit-identical to
        ``oasis_apply_rotary_emb`` rather than merely close to it.
        """
        pairs = tl.reshape(t, (BLOCK, DH // 2, 2))
        even, odd = tl.split(pairs)
        swapped = tl.reshape(tl.join(odd, even), (BLOCK, DH))
        swapped = tl.where(negate_even[None, :], -swapped, swapped)
        return _fp16_mul_add(t, cos, swapped, sin)

    @triton.jit
    def _fused_rope_attn(
        qkv_ptr, cos_ptr, sin_ptr, out_ptr,
        S: tl.constexpr, HEADS: tl.constexpr, DH: tl.constexpr,
        QKV_ROW: tl.constexpr, OUT_ROW: tl.constexpr, QK_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Rotary embedding + non-causal attention + output layout, fused.

        ``qkv`` is the contiguous ``[frames * S, 3 * HEADS * DH]`` projection, so
        for flattened row ``r = frame * S + s`` the Q/K/V tiles of head ``h`` live
        at ``r * QKV_ROW + h * DH`` plus 0, ``OUT_ROW`` and ``2 * OUT_ROW``. The
        context is written straight out in (head, dim) order, which is what the
        output projection consumes -- that is what removes the baseline's
        permute-plus-reshape copy instead of reproducing it.
        """
        pid_m = tl.program_id(0)
        frame = tl.program_id(1) // HEADS
        head = tl.program_id(1) % HEADS

        offs_d = tl.arange(0, DH)
        negate_even = (offs_d % 2) == 0
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        head_base = frame * S * QKV_ROW + head * DH
        q = tl.load(qkv_ptr + head_base + offs_m[:, None] * QKV_ROW + offs_d[None, :],
                    mask=mask_m[:, None], other=0.0)
        table_m = offs_m[:, None] * DH + offs_d[None, :]
        q = _rope_interleaved(
            q,
            tl.load(cos_ptr + table_m, mask=mask_m[:, None], other=0.0),
            tl.load(sin_ptr + table_m, mask=mask_m[:, None], other=0.0),
            negate_even, BLOCK_M, DH,
        )

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, DH], tl.float32)

        for start_n in tl.range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S
            kv_base = head_base + offs_n[:, None] * QKV_ROW + offs_d[None, :]
            k = tl.load(qkv_ptr + kv_base + OUT_ROW, mask=mask_n[:, None], other=0.0)
            v = tl.load(qkv_ptr + kv_base + 2 * OUT_ROW, mask=mask_n[:, None], other=0.0)
            table_n = offs_n[:, None] * DH + offs_d[None, :]
            k = _rope_interleaved(
                k,
                tl.load(cos_ptr + table_n, mask=mask_n[:, None], other=0.0),
                tl.load(sin_ptr + table_n, mask=mask_n[:, None], other=0.0),
                negate_even, BLOCK_N, DH,
            )

            # Padded keys are pushed to -inf *before* the row maximum is taken, so
            # they contribute nothing to the softmax denominator.
            qk = tl.dot(q, tl.trans(k)) * QK_SCALE
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_off = (frame * S + offs_m)[:, None] * OUT_ROW + head * DH + offs_d[None, :]
        tl.store(out_ptr + out_off, acc.to(tl.float16), mask=mask_m[:, None])


def _apply_rotary(cos: torch.Tensor, sin: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``oasis_apply_rotary_emb`` with the cosine/sine tables handed in already
    evaluated, so the axial table is built once per shape instead of per call.

    Line-for-line equivalent to the baseline helper otherwise, including the
    empty ``t_left`` slice and the trailing cast.
    """
    dtype = t.dtype
    rot_dim = cos.shape[-1]
    t_left = t[..., :0]
    t_middle = t[..., :rot_dim]
    t_right = t[..., rot_dim:]
    t_transformed = (t_middle * cos) + (oasis_rotate_half(t_middle) * sin)
    return torch.cat((t_left, t_transformed, t_right), dim=-1).to(dtype)


def _drop_rope_cache(module: "OasisSpatialAxialAttention", incompatible_keys) -> None:
    del incompatible_keys
    module._rope_cache = None


class OasisSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")
        self.dim_head = dim_head
        # Nothing here may be derived from a parameter *value*. The weights are
        # allocated with torch.empty and only overwritten afterwards, by a
        # load_state_dict that runs after __init__ -- so a pre-transposed copy or
        # a rotary table built now would be garbage that is never refreshed.
        # The rotary tables are built on the first forward instead; the cache is
        # a plain attribute so it stays out of state_dict() and out of _apply's
        # numeric conversion.
        self._rope_cache = None
        self.register_load_state_dict_post_hook(_drop_rope_cache)

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        # Module._apply rewrites parameter storage without running
        # load-state-dict hooks, so .to()/.half()/.float()/.cuda() would
        # otherwise leave a table built from the previous freqs in place.
        self._rope_cache = None
        return out

    def _rope_tables(self, height: int, width: int):
        """Return ``(cos, sin)`` for the axial rotary table of this spatial grid.

        The table is *reused*, never re-derived. ``rotary_emb.freqs`` is cast to
        fp16 before the first call, so ``get_axial_freqs`` runs its einsum in
        fp16 and yields angles up to ~402 -- a magnitude where the fp16 spacing
        is 0.25. Those angles are deterministic but numerically arbitrary:
        recomputing them in fp32 produces a completely different, and completely
        wrong, table. Only the module's own rotary submodule may produce it.

        The cache key covers every way the table can go stale: the grid size,
        parameter replacement and device moves (``data_ptr``), an in-place
        ``freqs.copy_()`` (``_version``), and a dtype change.
        """
        freqs_param = self.rotary_emb.freqs
        key = (height, width, freqs_param.data_ptr(), freqs_param.dtype,
               freqs_param.device, freqs_param._version)
        cached = self._rope_cache
        if cached is None or cached[0] != key:
            axial = self.rotary_emb.get_axial_freqs(height, width)
            cos = axial.cos().contiguous()
            sin = axial.sin().contiguous()
            cached = (key, cos, sin)
            self._rope_cache = cached
        return cached[1], cached[2]

    def _can_fuse(self, x: torch.Tensor, cos: torch.Tensor) -> bool:
        """Admit only what the fused kernel actually implements.

        Anything else -- fp32/bf16, CPU, a non-contiguous input, grad enabled, a
        partial rotary dimension, an odd or non-power-of-two head dimension, or a
        projection whose shape disagrees with (heads, dim_head) -- goes to the
        eager path.
        """
        dim_head = self.dim_head
        return (
            x.dtype is torch.float16
            and x.is_cuda
            and x.is_contiguous()
            and not torch.is_grad_enabled()
            and cos.dtype is torch.float16
            and cos.shape[-1] == dim_head
            and dim_head % 2 == 0
            and 16 <= dim_head <= 128
            and dim_head & (dim_head - 1) == 0
            and self.to_qkv.weight.shape[0] == 3 * self.heads * dim_head
            and self.to_qkv.weight.dtype is torch.float16
            and self.to_out.weight.dtype is torch.float16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        cos, sin = self._rope_tables(height, width)
        if _HAVE_TRITON and self._can_fuse(x, cos):
            return self._fused_forward(x, cos, sin)
        return self._eager_forward(x, cos, sin)

    def _fused_forward(self, x: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, dim = x.shape
        heads = self.heads
        dim_head = self.dim_head
        seq = height * width
        rows = bsz * time * seq
        inner = heads * dim_head

        # Both reshapes are views on contiguous memory, so the only dispatched
        # work is the two projections and the fused kernel between them.
        qkv = F.linear(x.reshape(rows, dim), self.to_qkv.weight)
        ctx = torch.empty((rows, inner), dtype=qkv.dtype, device=qkv.device)
        _fused_rope_attn[(triton.cdiv(seq, _BLOCK_M), bsz * time * heads)](
            qkv, cos, sin, ctx,
            S=seq, HEADS=heads, DH=dim_head,
            QKV_ROW=3 * inner, OUT_ROW=inner,
            QK_SCALE=dim_head ** -0.5 * _LOG2_E,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
            num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
        )
        return F.linear(ctx, self.to_out.weight, self.to_out.bias).view(
            bsz, time, height, width, dim)

    def _eager_forward(self, x: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)

        q = _apply_rotary(cos, sin, q)
        k = _apply_rotary(cos, sin, k)

        q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        out = self.attn(q, k, v, causal=False)
        out = out.reshape(bsz, time, height, width, self.heads, -1).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))
