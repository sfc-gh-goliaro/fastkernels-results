"""Oasis DiT blocks, fused.

The reference block issues ~105 launches per forward for ~830 us of GPU work
against ~1720 us of wall time, and its latency does not move with ``T`` -- so it
is dispatch bound, not compute bound, and launch count is what has to come down.

The nine GEMMs stay on cuBLAS (at ``M <= 864`` they already run near what those
shapes allow), the two adaLN projections merge into one call against a
concatenated weight, and everything else -- LayerNorm, modulate, gate, residual,
and both attentions including their rotary and layout shuffles -- collapses into
four Triton kernels.  That is 19 launches per forward instead of 105, and it
removes the 113-123 us cuDNN ``sm80`` kernel that the temporal axis (2304
independent ``T x T`` score matrices) was landing on.

The fast path is deliberately specialized to the captured regime and everything
else routes to a pure-PyTorch path that mirrors the reference op for op: other
shapes, dtypes or devices, a non-causal temporal axis, autograd, or a Triton that
will not compile.  Spatial attention additionally carries a middle option -- a
small rotary/layout kernel feeding ``F.scaled_dot_product_attention`` -- so that
losing the custom attention kernel costs a few launches rather than the whole
fused schedule.

The file is deliberately self-contained: the bench imports it with
``--standalone``, under which a relative import would resolve to the *reference*
module rather than to a sibling candidate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - no Triton is a supported configuration
    _HAS_TRITON = False

if _HAS_TRITON:
    # The root of every "Triton cannot build or run this" failure --
    # CompilationError, OutOfResources, PTXASError and friends all derive from it.
    # Catching this and nothing wider is what keeps a fallback from swallowing an
    # ordinary bug: a ValueError, a shape mismatch or an OOM has to propagate, or
    # the block answers a defect by silently running slower forever.
    try:
        from triton.errors import TritonError as _TritonError
    except ImportError:  # pragma: no cover - older Triton kept it here
        from triton.compiler.errors import CompilationError as _TritonError
else:
    class _TritonError(Exception):
        """Stand-in so the ``except`` clauses stay valid with no Triton at all."""


# Column layout of the merged adaLN projection output, in units of hidden size:
# the spatial projection's six chunks followed by the temporal projection's six.
_SHIFT_MSA, _SCALE_MSA, _GATE_MSA, _SHIFT_MLP, _SCALE_MLP, _GATE_MLP = range(6)
_TEMPORAL = 6

_LN_EPS = 1e-6


# ---------------------------------------------------------------------------
# Reference-equivalent helpers (used by the fallback path).
# ---------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved pair rotation: ``out[2i] = -x[2i+1]``, ``out[2i+1] = x[2i]``.

    Not the split-half GPT-NeoX form -- the two disagree everywhere, and the
    reference rotary is the interleaved one.
    """
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_middle, t_right = t[..., :rot_dim], t[..., rot_dim:]
    t_middle = (t_middle * freqs.cos()) + (_rotate_half(t_middle) * freqs.sin())
    return torch.cat((t_middle, t_right), dim=-1).to(dtype)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """SDPA on a ``(batch, seq, heads, dim)`` layout, as the reference calls it."""
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
        attn_mask=None, dropout_p=0.0, is_causal=causal, scale=None,
    )
    return out.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Submodule tree.  Names mirror the reference exactly: the bench shares weights
# through ``load_state_dict(..., strict=False)`` wrapped in a bare
# ``try/except: pass``, so a renamed or reshaped key is silently *not* shared and
# the block then runs on its own random weights with no diagnostic.
# ---------------------------------------------------------------------------
class _LayerNorm(nn.Module):
    """Affine-free LayerNorm with an fp32 reduction, as the reference has it.

    ``elementwise_affine=False`` means no weight and no bias, so this contributes
    nothing to the state dict -- it exists so the module tree carries the
    reference's four norm names, and so the fallback path and the reference share
    one definition of the reduction rather than two that could drift.
    """

    def __init__(self, normalized_shape: int, eps: float = _LN_EPS):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape, None, None, self.eps).to(x.dtype)


class _Matmul(nn.Module):
    """Stateless holder around ``F.linear``, as the reference's ``Linear`` has."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, weight, bias)


class _Linear(nn.Module):
    """``weight``/``bias`` plus the reference's ``matmul`` child.

    ``nn.Linear`` would give the same two state-dict keys but not the
    ``*.matmul`` name, and the module tree is part of the contract: anything that
    navigates by module path -- a hook, a wrapping policy, a quantization config --
    sees the reference's paths this way.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = _Matmul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matmul(x, self.weight, self.bias)


class _DenseAttention(nn.Module):
    """Stateless SDPA holder on the reference's ``(batch, seq, heads, dim)`` layout."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal: bool = False) -> torch.Tensor:
        return _sdpa(q, k, v, causal)


class _GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class _Attn(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_emb: nn.Module,
                 *, is_causal: bool = False):
        super().__init__()
        self.heads = heads
        self.to_qkv = _Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = _Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = _DenseAttention()
        self.is_causal = is_causal


class _MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = _Linear(in_features, hidden_features, bias=True)
        self.act = _GELU(approximate="tanh")
        self.fc2 = _Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Derived-tensor cache.
#
# Everything derived from a parameter is built on the first forward, never in
# ``__init__``: the bench's order is move-and-cast -> re-initialize garbage
# weights -> share weights, so anything captured in ``__init__`` would hold
# pre-share values.  The guard has to cover both of those mutations, and they
# move different things -- an in-place ``load_state_dict`` bumps ``_version``
# while leaving ``data_ptr`` alone, and ``p.data = p.data.to(f16)`` moves
# ``data_ptr`` while leaving ``_version`` alone -- so neither alone is enough.
# ---------------------------------------------------------------------------
def _signature(tensors) -> tuple:
    return tuple((t.device, t.dtype, t.data_ptr(), t._version) for t in tensors)


class _Cache:
    __slots__ = ("signature", "hw", "adaln_weight", "adaln_bias",
                 "spatial_cos", "spatial_sin", "temporal")

    def __init__(self, signature: tuple, hw: tuple[int, int]):
        self.signature = signature
        self.hw = hw
        self.temporal: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


# ---------------------------------------------------------------------------
# Triton kernels.
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _modulated(normed, scale, shift, dtype: tl.constexpr,
                   ROUND_F16: tl.constexpr):
        """``normed * (1 + scale) + shift``, in the reference's own precision.

        With ``ROUND_F16`` the whole chain runs in half exactly as the reference
        does: the normalized value is rounded first, and ``1 + scale`` is rounded
        before the multiply.  Both casts are explicit because a bare ``1.0`` is an
        fp32 literal in Triton and would silently promote the product, which is
        the fp32-intermediate variant rather than the reference's.
        """
        # Both branches return ``dtype``: a Triton device function must have one
        # return type, and returning fp16 from one arm and fp32 from the other is
        # a compile error rather than a promotion.
        if ROUND_F16:
            half = normed.to(dtype)
            return half * (scale + 1.0).to(dtype) + shift
        return (normed * (1.0 + scale.to(tl.float32))
                + shift.to(tl.float32)).to(dtype)

    @triton.jit
    def _norm_modulate_kernel(
        X, MOD, RESID, OUT,
        stride_frame, stride_channel, stride_position,
        HW: tl.constexpr, C: tl.constexpr, MOD_STRIDE: tl.constexpr,
        SHIFT: tl.constexpr, SCALE: tl.constexpr,
        EPS: tl.constexpr, ROUND_F16: tl.constexpr,
    ):
        """One token per program: read the strided input, emit both outputs.

        ``x`` arrives as a ``[B, T, C, H, W]``-contiguous buffer viewed as
        ``[B, T, H, W, C]``, so a channel walk is a stride-``H*W`` gather.  Only
        the first kernel pays for that: it writes ``RESID`` contiguous and every
        later pointwise kernel reads that instead.

        This is the general schedule -- it holds for any ``H*W``.  Where ``H*W``
        divides by the tile width, ``_norm_modulate_tiled_kernel`` is faster.
        """
        row = tl.program_id(0)
        frame = row // HW
        position = row % HW
        channels = tl.arange(0, C)

        x = tl.load(X + frame * stride_frame + position * stride_position
                    + channels * stride_channel)
        tl.store(RESID + row * C + channels, x)

        xf = x.to(tl.float32)
        mean = tl.sum(xf, 0) / C
        centered = xf - mean
        # Biased variance (divide by C, not C-1) with eps=1e-6, matching the
        # reference F.layer_norm call.
        var = tl.sum(centered * centered, 0) / C
        normed = centered * tl.rsqrt(var + EPS)

        base = frame * MOD_STRIDE
        shift = tl.load(MOD + base + SHIFT * C + channels)
        scale = tl.load(MOD + base + SCALE * C + channels)
        y = _modulated(normed, scale, shift, OUT.dtype.element_ty, ROUND_F16)
        tl.store(OUT + row * C + channels, y.to(OUT.dtype.element_ty))

    @triton.jit
    def _norm_modulate_tiled_kernel(
        X, MOD, RESID, OUT,
        stride_frame, stride_channel, stride_position,
        HW: tl.constexpr, C: tl.constexpr, MOD_STRIDE: tl.constexpr,
        SHIFT: tl.constexpr, SCALE: tl.constexpr,
        EPS: tl.constexpr, ROUND_F16: tl.constexpr, TOKENS: tl.constexpr,
    ):
        """A ``[C, TOKENS]`` tile: TOKENS consecutive positions of one frame.

        Positions inside a frame are adjacent in memory (position stride 1), so
        putting them in the tile's fast axis turns each channel's read into one
        wide transaction instead of TOKENS separate 2-byte loads.  Measured
        against the one-token schedule, device time per launch falls from a
        geometric mean of 4.54 us across the five captured shapes to 3.51 us, and
        from 5.91 us to 4.05 us at ``T=6``; the one-token kernel wastes 75 % of
        the sectors it fetches, using only 5.3 of every 32 bytes.

        Both outputs are ``[token, channel]``, so the tile is transposed before
        it is stored -- coalesced reads bought with an in-register transpose.
        Requires ``HW % TOKENS == 0`` so a tile never straddles a frame boundary.
        """
        pid = tl.program_id(0)
        tiles = HW // TOKENS
        frame = pid // tiles
        positions = (pid % tiles) * TOKENS + tl.arange(0, TOKENS)
        channels = tl.arange(0, C)

        x = tl.load(X + frame * stride_frame
                    + channels[:, None] * stride_channel
                    + positions[None, :] * stride_position)
        rows = frame * HW + positions
        tl.store(RESID + rows[:, None] * C + channels[None, :], tl.trans(x))

        xf = x.to(tl.float32)
        mean = tl.sum(xf, 0)[None, :] / C
        centered = xf - mean
        var = tl.sum(centered * centered, 0)[None, :] / C
        normed = centered * tl.rsqrt(var + EPS)

        base = frame * MOD_STRIDE
        shift = tl.load(MOD + base + SHIFT * C + channels)[:, None]
        scale = tl.load(MOD + base + SCALE * C + channels)[:, None]
        y = _modulated(normed, scale, shift, OUT.dtype.element_ty, ROUND_F16)
        tl.store(OUT + rows[:, None] * C + channels[None, :],
                 tl.trans(y.to(OUT.dtype.element_ty)))

    @triton.jit
    def _gate_norm_modulate_kernel(
        RESID, DELTA, MOD, OUT,
        HW: tl.constexpr, C: tl.constexpr, MOD_STRIDE: tl.constexpr,
        GATE: tl.constexpr, SHIFT: tl.constexpr, SCALE: tl.constexpr,
        EPS: tl.constexpr, HAS_NORM: tl.constexpr, ROUND_F16: tl.constexpr,
    ):
        """``resid += gate * delta``, then optionally norm-and-modulate it.

        Covers the three interior residual joins and the final one; the last
        sub-block has nothing left to feed, so it runs with ``HAS_NORM=False``.
        The residual is updated in place -- one program owns one whole row, and
        it has already loaded the row before storing it.
        """
        row = tl.program_id(0)
        frame = row // HW
        channels = tl.arange(0, C)
        base = frame * MOD_STRIDE

        resid = tl.load(RESID + row * C + channels)
        delta = tl.load(DELTA + row * C + channels)
        gate = tl.load(MOD + base + GATE * C + channels)
        # Half arithmetic, gate before the add, exactly as the reference does it.
        resid = resid + gate * delta
        tl.store(RESID + row * C + channels, resid)

        if HAS_NORM:
            xf = resid.to(tl.float32)
            mean = tl.sum(xf, 0) / C
            centered = xf - mean
            var = tl.sum(centered * centered, 0) / C
            normed = centered * tl.rsqrt(var + EPS)
            shift = tl.load(MOD + base + SHIFT * C + channels)
            scale = tl.load(MOD + base + SCALE * C + channels)
            y = _modulated(normed, scale, shift, OUT.dtype.element_ty, ROUND_F16)
            tl.store(OUT + row * C + channels, y.to(OUT.dtype.element_ty))

    @triton.jit
    def _rotary_pair(value, partner, cos, sin, even):
        """``value * cos + rotate_half(value) * sin`` for an interleaved rotary.

        ``partner`` is the same tile gathered at ``d ^ 1``, so the rotation is a
        lane-parity select rather than a shuffle.  Half throughout, as the
        reference is -- though Triton contracts the multiply-add into an FMA, so
        the result lands within one rounding step of the reference rather than
        bit-identical to it.
        """
        return value * cos + tl.where(even, -partner, partner) * sin

    @triton.jit
    def _spatial_rotary_prep_kernel(
        QKV, COS, SIN, Q, K, V,
        HW: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Apply the axial rotary and lay ``q``/``k``/``v`` out for SDPA.

        The fallback for spatial attention: it costs two extra launches plus the
        reshape after SDPA, against one for the fused kernel, but it leans on a
        vendor attention kernel instead of a hand-written one.  ``row = t*HW + p``
        with ``col = head*D + d`` is already the ``(batch, seq, heads, dim)``
        layout SDPA wants, so the three outputs need no permute.

        The rotary goes on ``q`` and ``k`` only -- the reference does not rotate
        values.
        """
        tile = tl.program_id(0)
        frame = tl.program_id(1)
        head = tl.program_id(2)

        d = tl.arange(0, D)
        partner = d ^ 1
        even = (d % 2) == 0
        positions = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        in_range = positions < HW
        rows = frame * HW + positions
        source = QKV + rows[:, None] * (3 * C) + head * D
        target = rows[:, None] * C + head * D + d[None, :]

        cos = tl.load(COS + positions[:, None] * D + d[None, :],
                      mask=in_range[:, None], other=1.0)
        sin = tl.load(SIN + positions[:, None] * D + d[None, :],
                      mask=in_range[:, None], other=0.0)

        q = tl.load(source + d[None, :], mask=in_range[:, None], other=0.0)
        q_partner = tl.load(source + partner[None, :], mask=in_range[:, None], other=0.0)
        tl.store(Q + target, _rotary_pair(q, q_partner, cos, sin, even[None, :]),
                 mask=in_range[:, None])

        k = tl.load(source + C + d[None, :], mask=in_range[:, None], other=0.0)
        k_partner = tl.load(source + C + partner[None, :], mask=in_range[:, None],
                            other=0.0)
        tl.store(K + target, _rotary_pair(k, k_partner, cos, sin, even[None, :]),
                 mask=in_range[:, None])

        tl.store(V + target,
                 tl.load(source + 2 * C + d[None, :], mask=in_range[:, None], other=0.0),
                 mask=in_range[:, None])

    @triton.jit
    def _spatial_attn_kernel(
        QKV, COS, SIN, OUT,
        HW: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Axial-rotary attention over the ``H*W`` positions of one frame.

        Reads straight out of the ``[N, 3C]`` QKV GEMM output and writes straight
        into an ``[N, C]`` buffer at ``row = frame*HW + position``,
        ``col = head*D + d`` -- which is the order ``to_out`` already wants, so
        no permute or copy kernels are needed on either side.
        """
        tile = tl.program_id(0)
        frame = tl.program_id(1)
        head = tl.program_id(2)

        d = tl.arange(0, D)
        partner = d ^ 1
        even = (d % 2) == 0
        positions = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        in_range = positions < HW

        q_rows = QKV + (frame * HW + positions)[:, None] * (3 * C) + head * D
        q = tl.load(q_rows + d[None, :], mask=in_range[:, None], other=0.0)
        q_partner = tl.load(q_rows + partner[None, :], mask=in_range[:, None], other=0.0)
        cos = tl.load(COS + positions[:, None] * D + d[None, :],
                      mask=in_range[:, None], other=1.0)
        sin = tl.load(SIN + positions[:, None] * D + d[None, :],
                      mask=in_range[:, None], other=0.0)
        q = _rotary_pair(q, q_partner, cos, sin, even[None, :])

        running_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        for start in range(0, HW, BLOCK_N):
            keys = start + tl.arange(0, BLOCK_N)
            key_ok = keys < HW
            row_base = (frame * HW + keys)[None, :] * (3 * C) + head * D
            # Load k transposed so the score matmul needs no in-kernel transpose.
            kt_rows = QKV + C + row_base
            kt = tl.load(kt_rows + d[:, None], mask=key_ok[None, :], other=0.0)
            kt_partner = tl.load(kt_rows + partner[:, None], mask=key_ok[None, :],
                                 other=0.0)
            cos_t = tl.load(COS + keys[None, :] * D + d[:, None],
                            mask=key_ok[None, :], other=1.0)
            sin_t = tl.load(SIN + keys[None, :] * D + d[:, None],
                            mask=key_ok[None, :], other=0.0)
            kt = _rotary_pair(kt, kt_partner, cos_t, sin_t, even[:, None])

            v = tl.load(QKV + 2 * C + (frame * HW + keys)[:, None] * (3 * C)
                        + head * D + d[None, :], mask=key_ok[:, None], other=0.0)

            scores = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
            scores = tl.where(key_ok[None, :], scores, float("-inf"))

            tile_max = tl.maximum(running_max, tl.max(scores, 1))
            probs = tl.exp(scores - tile_max[:, None])
            rescale = tl.exp(running_max - tile_max)
            running_sum = running_sum * rescale + tl.sum(probs, 1)
            acc = acc * rescale[:, None] + tl.dot(
                probs.to(QKV.dtype.element_ty), v, out_dtype=tl.float32)
            running_max = tile_max

        acc = acc / running_sum[:, None]
        tl.store(OUT + (frame * HW + positions)[:, None] * C + head * D + d[None, :],
                 acc.to(OUT.dtype.element_ty), mask=in_range[:, None])

    @triton.jit
    def _temporal_attn_kernel(
        QKV, COS, SIN, OUT, T,
        HW: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        """Causal attention along the frame axis for one ``(position, head)``.

        ``T <= BLOCK_T``, so the whole sequence is one tile: no key tiling and no
        online softmax.  Both the gather and the scatter step ``HW`` rows at a
        time -- holding this program's ``T`` outputs contiguously would put them
        at ``position*T + t`` instead of the ``t*HW + position`` that ``to_out``
        consumes.

        The score matmul is a legal ``tl.dot`` (``K = D``), but ``probs @ v`` is
        not: the 16-bit contraction has to be at least 16 wide and this one is
        ``BLOCK_T``.  At ``8 x 8 x 64`` an explicit broadcast-multiply-reduce is
        free, and the kernel is memory bound anyway.
        """
        position = tl.program_id(0)
        head = tl.program_id(1)

        d = tl.arange(0, D)
        partner = d ^ 1
        even = (d % 2) == 0
        frames = tl.arange(0, BLOCK_T)
        live = frames < T

        rows = (frames * HW + position)[:, None] * (3 * C) + head * D
        q = tl.load(QKV + rows + d[None, :], mask=live[:, None], other=0.0)
        q_partner = tl.load(QKV + rows + partner[None, :], mask=live[:, None], other=0.0)
        cos = tl.load(COS + frames[:, None] * D + d[None, :], mask=live[:, None],
                      other=1.0)
        sin = tl.load(SIN + frames[:, None] * D + d[None, :], mask=live[:, None],
                      other=0.0)
        q = _rotary_pair(q, q_partner, cos, sin, even[None, :])

        kt_rows = (frames * HW + position)[None, :] * (3 * C) + C + head * D
        kt = tl.load(QKV + kt_rows + d[:, None], mask=live[None, :], other=0.0)
        kt_partner = tl.load(QKV + kt_rows + partner[:, None], mask=live[None, :],
                             other=0.0)
        cos_t = tl.load(COS + frames[None, :] * D + d[:, None], mask=live[None, :],
                        other=1.0)
        sin_t = tl.load(SIN + frames[None, :] * D + d[:, None], mask=live[None, :],
                        other=0.0)
        kt = _rotary_pair(kt, kt_partner, cos_t, sin_t, even[:, None])

        v = tl.load(QKV + 2 * C + (frames * HW + position)[:, None] * (3 * C)
                    + head * D + d[None, :], mask=live[:, None], other=0.0)

        scores = tl.dot(q, kt, out_dtype=tl.float32) * SCALE
        query_idx = frames[:, None]
        key_idx = frames[None, :]
        # Clamping the query index keeps every row of the padded tail with at
        # least one live key.  An all -inf row would softmax to NaN, and NaN
        # anywhere in the output fails the bench outright even though those rows
        # are never stored.
        keep = (key_idx <= tl.minimum(query_idx, T - 1)) & (key_idx < T)
        scores = tl.where(keep, scores, float("-inf"))

        row_max = tl.max(scores, 1)
        probs = tl.exp(scores - row_max[:, None])
        probs = probs / tl.sum(probs, 1)[:, None]
        acc = tl.sum(probs[:, :, None] * v.to(tl.float32)[None, :, :], axis=1)

        tl.store(OUT + (frames * HW + position)[:, None] * C + head * D + d[None, :],
                 acc.to(OUT.dtype.element_ty), mask=live[:, None])


class SpatioTemporalDiTBlock(nn.Module):
    """Two spatial and two temporal residual sub-blocks with adaLN modulation."""

    # The regime the fused path is specialized to and measured on.  Anything
    # else routes to the reference path rather than running tile shapes, a
    # rotary folding and frame-count specializations that were never measured
    # for it.
    CAPTURED_HW = (9, 16)
    CAPTURED_FRAMES = frozenset({2, 3, 4, 5, 6})
    CAPTURED_HIDDEN = 1024
    CAPTURED_HEADS = 16

    # Tile shapes, chosen by sweeping them (the figures below are block-wide GPU
    # time at T=6).  Every extent handed to ``tl.arange`` has to be a power of
    # two, so the 144 spatial positions are covered by an online-softmax loop
    # over BLOCK_N with the last tile masked rather than by one padded 256-wide
    # score tile, which would want a 64 KB fp32 accumulator.
    SPATIAL_BLOCK_M = 64        # 119.2 us here against 127.0 at BLOCK_M = 32
    SPATIAL_BLOCK_N = 64        # 119.2 us here against 122.0 at 32, 137.6 at 128
    SPATIAL_WARPS = 4
    # The smallest power of two covering the captured frame counts, which keeps
    # the temporal kernel to a single specialization across all of them.
    TEMPORAL_BLOCK_T = 8
    # The temporal tile is only BLOCK_T x D elements, so more warps only leave
    # lanes idle: 119.1 us at one warp against 120.6 at two, 130.9 at four and
    # 162.0 at eight.
    TEMPORAL_WARPS = 1
    POINTWISE_WARPS = 4         # 119.1 us here against 119.8 at two, 127.1 at 16
    # Width of the first kernel's channel-by-token tile, and the fallback for an
    # ``H*W`` it does not divide.  Swept over {2, 4, 8, 16} x {4, 8} warps.
    FIRST_KERNEL_TOKENS = 4
    FIRST_KERNEL_WARPS = 4
    # "triton" uses the fused attention kernel; "sdpa" forces the
    # rotary-prep-plus-SDPA fallback.  A launch failure on the fused kernel
    # switches this instance to the fallback for good.
    SPATIAL_BACKEND = "triton"
    # Round to half after the norm and modulate in half, as the reference does,
    # rather than carrying fp32 through the modulate.  Measured identical on both
    # parity and wall time, so this takes the variant that reproduces the
    # reference's rounding instead of the one that improves on it.
    ROUND_F16 = True

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        # Spelled as the reference spells it so that ``inspect.signature``
        # agrees.  ``from __future__ import annotations`` keeps this a string, so
        # naming the reference class costs no import -- the bench passes the
        # instances in and this file stays self-contained.
        spatial_rotary_emb: OasisRotaryEmbedding,  # noqa: F821
        temporal_rotary_emb: OasisRotaryEmbedding,  # noqa: F821
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads

        # Declared in the reference's order so the module tree matches it name
        # for name.  The four norms hold no parameters and no buffers, so the
        # state dict stays exactly the reference's 20 keys.
        self.s_norm1 = _LayerNorm(hidden_size)
        self.s_attn = _Attn(hidden_size, num_heads, self.dim_head, spatial_rotary_emb)
        self.s_norm2 = _LayerNorm(hidden_size)
        self.s_mlp = _MLP(hidden_size, int(hidden_size * mlp_ratio))
        # Kept as a container purely so the projection weights keep the
        # ``..._modulation.1.*`` names the bench shares against.  The fused path
        # never calls it -- it runs one merged GEMM over both projections' rows.
        self.s_adaLN_modulation = nn.Sequential(
            nn.SiLU(), _Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = _LayerNorm(hidden_size)
        self.t_attn = _Attn(hidden_size, num_heads, self.dim_head, temporal_rotary_emb,
                            is_causal=is_causal)
        self.t_norm2 = _LayerNorm(hidden_size)
        self.t_mlp = _MLP(hidden_size, int(hidden_size * mlp_ratio))
        self.t_adaLN_modulation = nn.Sequential(
            nn.SiLU(), _Linear(hidden_size, 6 * hidden_size, bias=True))

        self._cache: _Cache | None = None
        self._fused_disabled = False
        self._spatial_triton_disabled = False

    # -- cached derived tensors ------------------------------------------
    def _cache_sources(self) -> tuple[torch.Tensor, ...]:
        return (
            self.s_adaLN_modulation[1].weight, self.s_adaLN_modulation[1].bias,
            self.t_adaLN_modulation[1].weight, self.t_adaLN_modulation[1].bias,
            self.s_attn.rotary_emb.freqs, self.t_attn.rotary_emb.freqs,
        )

    def _build_cache(self, signature: tuple, hw: tuple[int, int]) -> _Cache:
        cache = _Cache(signature, hw)
        s_lin, t_lin = self.s_adaLN_modulation[1], self.t_adaLN_modulation[1]
        cache.adaln_weight = torch.cat((s_lin.weight, t_lin.weight), dim=0)
        cache.adaln_bias = torch.cat((s_lin.bias, t_lin.bias), dim=0)

        # Taken from the rotary module rather than re-derived.  The tables are
        # built in half, and for the spatial axes the angle runs up to ~402 where
        # half spacing is 0.25 -- an fp32 recomputation lands ~30% of the table
        # outside the bench's own tolerance, so the rounding of every
        # intermediate step has to be the module's own.
        freqs = self.s_attn.rotary_emb.get_axial_freqs(*hw)
        cache.spatial_cos = freqs.cos().reshape(hw[0] * hw[1], -1).contiguous()
        cache.spatial_sin = freqs.sin().reshape(hw[0] * hw[1], -1).contiguous()
        self._cache = cache
        return cache

    def _temporal_tables(self, cache: _Cache, frames: int, dtype: torch.dtype,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tables = cache.temporal.get(frames)
        if tables is None:
            rotary = self.t_attn.rotary_emb
            # The rotary's own path takes its position dtype from the query, so
            # the angles are accumulated in half here too.
            positions = torch.arange(frames, device=device, dtype=dtype)
            freqs = rotary(positions, rotary.freqs)
            tables = (freqs.cos().contiguous(), freqs.sin().contiguous())
            cache.temporal[frames] = tables
        return tables

    # -- fused path ------------------------------------------------------
    def _fused_supported(self, x: torch.Tensor, c: torch.Tensor) -> bool:
        if not _HAS_TRITON or self._fused_disabled:
            return False
        if x.dim() != 5 or c.dim() != 3:
            return False
        batch, frames, height, width, channels = x.shape
        if (batch, height, width) != (1,) + self.CAPTURED_HW:
            return False
        if frames not in self.CAPTURED_FRAMES:
            return False
        if channels != self.CAPTURED_HIDDEN or self.num_heads != self.CAPTURED_HEADS:
            return False
        if channels != self.hidden_size or channels != self.num_heads * self.dim_head:
            return False
        if x.dtype is not torch.float16 or c.dtype is not x.dtype:
            return False
        if not x.is_cuda or c.device != x.device:
            return False
        if c.shape != (batch, frames, channels):
            return False
        if not self.t_attn.is_causal:
            return False
        # The attention kernels fold the rotary into a single multiply over the
        # whole head, so a rotary that does not span the head cannot be used.  The
        # widths follow from the frequency counts without building the tables:
        # ``get_axial_freqs`` over two axes concatenates two
        # ``repeat_interleave(2)`` blocks, and the temporal path is one such block.
        if 4 * self.s_attn.rotary_emb.freqs.numel() != self.dim_head:
            return False
        if 2 * self.t_attn.rotary_emb.freqs.numel() != self.dim_head:
            return False
        # The strided read models the offset as affine in (frame, channel,
        # position), which needs the two spatial axes to be adjacent.
        if x.stride(2) != width * x.stride(3):
            return False
        if torch.is_grad_enabled() and (
                x.requires_grad or c.requires_grad
                or any(p.requires_grad for p in self.parameters())):
            return False
        return True

    @staticmethod
    def _mlp_fused(mlp: _MLP, x: torch.Tensor) -> torch.Tensor:
        """The MLP as three direct dispatches.

        Same three launches as calling ``mlp`` would, but it skips four
        ``nn.Module.__call__`` frames per forward -- the module tree mirrors the
        reference's holders for anything that navigates by name, and the fused path
        should not pay for them when the block is host bound.
        """
        hidden = F.linear(x, mlp.fc1.weight, mlp.fc1.bias)
        hidden = F.gelu(hidden, approximate=mlp.act.approximate)
        return F.linear(hidden, mlp.fc2.weight, mlp.fc2.bias)

    def _spatial_attention_fused(self, qkv: torch.Tensor, attn_out: torch.Tensor,
                                 cache: _Cache, frames: int, positions: int) -> None:
        _spatial_attn_kernel[
            (triton.cdiv(positions, self.SPATIAL_BLOCK_M), frames, self.num_heads)](
            qkv, cache.spatial_cos, cache.spatial_sin, attn_out,
            HW=positions, C=self.hidden_size, D=self.dim_head,
            SCALE=self.dim_head ** -0.5,
            BLOCK_M=self.SPATIAL_BLOCK_M, BLOCK_N=self.SPATIAL_BLOCK_N,
            num_warps=self.SPATIAL_WARPS)

    def _spatial_attention_via_sdpa(self, qkv: torch.Tensor, attn_out: torch.Tensor,
                                    cache: _Cache, frames: int, positions: int) -> None:
        """Rotary and layout in one kernel, then let SDPA do the attention."""
        channels, dim_head = self.hidden_size, self.dim_head
        heads = self.num_heads
        shape = (frames * positions, channels)
        q, k, v = (torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
                   for _ in range(3))
        _spatial_rotary_prep_kernel[
            (triton.cdiv(positions, self.SPATIAL_BLOCK_M), frames, heads)](
            qkv, cache.spatial_cos, cache.spatial_sin, q, k, v,
            HW=positions, C=channels, D=dim_head, BLOCK_M=self.SPATIAL_BLOCK_M,
            num_warps=self.SPATIAL_WARPS)
        view = (frames, positions, heads, dim_head)
        out = _sdpa(q.view(view), k.view(view), v.view(view), causal=False)
        attn_out.copy_(out.reshape(shape))

    def _forward_fused(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        batch, frames, height, width, channels = x.shape
        positions = height * width
        tokens = frames * positions
        device, dtype = x.device, x.dtype
        heads, dim_head = self.num_heads, self.dim_head
        scale = dim_head ** -0.5

        cache = self._cache
        signature = _signature(self._cache_sources())
        if cache is None or cache.signature != signature or cache.hw != (height, width):
            cache = self._build_cache(signature, (height, width))
        temporal_cos, temporal_sin = self._temporal_tables(cache, frames, dtype, device)

        # One SiLU shared by both projections (the reference computes it twice),
        # then one GEMM over the concatenated projection rows.  F.silu stays on
        # aten rather than becoming a fifth Triton kernel: it is the same single
        # launch and bit-identical, but 6 us cheaper per forward on the host,
        # which is the side that binds here.
        cond = c.reshape(frames, channels)
        if not cond.is_contiguous():
            cond = cond.contiguous()
        mod = F.linear(F.silu(cond), cache.adaln_weight, cache.adaln_bias)
        mod_stride = mod.stride(0)

        resid = torch.empty((tokens, channels), dtype=dtype, device=device)
        normed = torch.empty((tokens, channels), dtype=dtype, device=device)
        attn_out = torch.empty((tokens, channels), dtype=dtype, device=device)
        first = dict(HW=positions, C=channels, MOD_STRIDE=mod_stride,
                     SHIFT=_SHIFT_MSA, SCALE=_SCALE_MSA, EPS=_LN_EPS,
                     ROUND_F16=self.ROUND_F16)
        strides = (x.stride(1), x.stride(4), x.stride(3))

        tile = self.FIRST_KERNEL_TOKENS
        if positions % tile == 0:
            _norm_modulate_tiled_kernel[(tokens // tile,)](
                x, mod, resid, normed, *strides, TOKENS=tile,
                num_warps=self.FIRST_KERNEL_WARPS, **first)
        else:
            _norm_modulate_kernel[(tokens,)](
                x, mod, resid, normed, *strides,
                num_warps=self.POINTWISE_WARPS, **first)

        def join(gate: int, shift: int, scale_col: int, delta: torch.Tensor,
                 has_norm: bool) -> None:
            """Fold ``delta`` into the residual and prepare the next input."""
            _gate_norm_modulate_kernel[(tokens,)](
                resid, delta, mod, normed,
                HW=positions, C=channels, MOD_STRIDE=mod_stride,
                GATE=gate, SHIFT=shift, SCALE=scale_col,
                EPS=_LN_EPS, HAS_NORM=has_norm, ROUND_F16=self.ROUND_F16,
                num_warps=self.POINTWISE_WARPS)

        qkv = F.linear(normed, self.s_attn.to_qkv.weight)
        if self.SPATIAL_BACKEND == "triton" and not self._spatial_triton_disabled:
            try:
                self._spatial_attention_fused(qkv, attn_out, cache, frames, positions)
            except _TritonError:
                # Nothing has been written to attn_out yet, so the fallback can
                # take over in place rather than abandoning the whole schedule.
                # Only Triton's own failures are caught: a shape bug here is a
                # defect and must surface, not become a silent slowdown.
                self._spatial_triton_disabled = True
                self._spatial_attention_via_sdpa(qkv, attn_out, cache, frames,
                                                 positions)
        else:
            self._spatial_attention_via_sdpa(qkv, attn_out, cache, frames, positions)
        join(_GATE_MSA, _SHIFT_MLP, _SCALE_MLP,
             F.linear(attn_out, self.s_attn.to_out.weight, self.s_attn.to_out.bias),
             True)

        join(_GATE_MLP, _TEMPORAL + _SHIFT_MSA, _TEMPORAL + _SCALE_MSA,
             self._mlp_fused(self.s_mlp, normed), True)

        qkv = F.linear(normed, self.t_attn.to_qkv.weight)
        _temporal_attn_kernel[(positions, heads)](
            qkv, temporal_cos, temporal_sin, attn_out, frames,
            HW=positions, C=channels, D=dim_head, SCALE=scale,
            BLOCK_T=self.TEMPORAL_BLOCK_T, num_warps=self.TEMPORAL_WARPS)
        join(_TEMPORAL + _GATE_MSA, _TEMPORAL + _SHIFT_MLP, _TEMPORAL + _SCALE_MLP,
             F.linear(attn_out, self.t_attn.to_out.weight, self.t_attn.to_out.bias),
             True)

        # Nothing consumes a modulated activation after the last join, so its
        # norm half is switched off and it only writes the residual.
        join(_TEMPORAL + _GATE_MLP, 0, 0, self._mlp_fused(self.t_mlp, normed), False)

        return resid.view(batch, frames, height, width, channels)

    # -- reference-equivalent path ---------------------------------------
    def _spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, height, width, _ = x.shape
        attn = self.s_attn
        q, k, v = F.linear(x, attn.to_qkv.weight).chunk(3, dim=-1)
        shape = (batch * frames, height, width, attn.heads, -1)
        q = q.reshape(shape).permute(0, 3, 1, 2, 4)
        k = k.reshape(shape).permute(0, 3, 1, 2, 4)
        v = v.reshape(shape).permute(0, 3, 1, 2, 4)

        freqs = attn.rotary_emb.get_axial_freqs(height, width)
        q = _apply_rotary(freqs, q)
        k = _apply_rotary(freqs, k)

        q = q.reshape(batch * frames, attn.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(batch * frames, attn.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(batch * frames, attn.heads, height * width, -1).transpose(1, 2)
        out = attn.attn(q, k, v, causal=False)
        out = out.reshape(batch, frames, height, width, -1)
        return F.linear(out.to(q.dtype), attn.to_out.weight, attn.to_out.bias)

    def _temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, height, width, _ = x.shape
        attn = self.t_attn
        q, k, v = F.linear(x, attn.to_qkv.weight).chunk(3, dim=-1)
        shape = (batch, frames, height, width, attn.heads, -1)
        flat = (batch * height * width, attn.heads, frames, -1)
        q = q.reshape(shape).permute(0, 2, 3, 4, 1, 5).reshape(flat)
        k = k.reshape(shape).permute(0, 2, 3, 4, 1, 5).reshape(flat)
        v = v.reshape(shape).permute(0, 2, 3, 4, 1, 5).reshape(flat)

        rotary = attn.rotary_emb
        positions = torch.arange(frames, device=q.device, dtype=q.dtype)
        freqs = rotary(positions, rotary.freqs)
        q = _apply_rotary(freqs, q)
        k = _apply_rotary(freqs, k)

        out = attn.attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                        causal=attn.is_causal)
        out = out.reshape(batch, height, width, frames, attn.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(batch, frames, height, width, -1)
        return F.linear(out.to(q.dtype), attn.to_out.weight, attn.to_out.bias)

    def _forward_reference(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s_mod = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + _gate(self._spatial_attention(
            _modulate(self.s_norm1(x), s_mod[_SHIFT_MSA], s_mod[_SCALE_MSA])),
            s_mod[_GATE_MSA])
        x = x + _gate(self.s_mlp(
            _modulate(self.s_norm2(x), s_mod[_SHIFT_MLP], s_mod[_SCALE_MLP])),
            s_mod[_GATE_MLP])

        t_mod = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + _gate(self._temporal_attention(
            _modulate(self.t_norm1(x), t_mod[_SHIFT_MSA], t_mod[_SCALE_MSA])),
            t_mod[_GATE_MSA])
        x = x + _gate(self.t_mlp(
            _modulate(self.t_norm2(x), t_mod[_SHIFT_MLP], t_mod[_SCALE_MLP])),
            t_mod[_GATE_MLP])
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self._fused_supported(x, c):
            try:
                return self._forward_fused(x, c)
            except _TritonError:
                # A Triton that cannot build or run these kernels is a
                # configuration problem, not a correctness one: fall back for good
                # rather than raise.  Anything else -- a ValueError, an
                # AssertionError, a shape mismatch, an OOM, a failure inside SDPA
                # -- propagates, because answering a programming error with a
                # permanent performance downgrade and no diagnostic is worse than
                # crashing.
                self._fused_disabled = True
        return self._forward_reference(x, c)
