"""Oasis VAE self-attention -- one fused Triton kernel between the two GEMMs.

The baseline spends ~23 kernels per forward and only three of them do real
work: the fused ``qkv`` GEMM, SDPA, and the output projection.  Everything in
between is layout plumbing -- ``chunk``, two ``reshape``/``permute`` pairs, the
axial rotary embedding (``cos``/``sin`` recomputed every call, ``rotate_half``
via ``reshape``/``unbind``/``stack``/``flatten``, then a ``cat``), and the
transpose that puts SDPA's ``(B, H, S, D)`` output back into ``(B, S, H*D)``
for the projection.  On the captured shapes that plumbing costs ~6x the
attention itself.

Two observations collapse it into a single kernel:

1. **The rotary is elementwise in ``(seq, head_dim)``.**  The baseline threads
   ``q`` through ``(B, fh, fw, H, D) -> permute -> (B, H, S, D) -> transpose``,
   but every one of those steps is a pure re-index: element ``(b, s, h, d)`` of
   the tensor SDPA finally sees *is* element ``(b, s, h*D + d)`` of the ``qkv``
   output, and the axial frequency it needs is ``freqs[s // fw, s % fw, d]``.
   So the whole permute chain is a no-op and the frequency table flattens to
   ``[S, rot_dim]`` -- no copies, no transposes, just an index.

2. **``rotate_half`` pairs adjacent lanes.**  ``freqs`` is
   ``repeat_interleave(2)``-ed, so ``cos``/``sin`` are constant within each
   ``(2m, 2m+1)`` pair and the table only needs ``D/2`` columns.  That matters:
   fp32 ``cos``/``sin`` at full ``[BN, D]`` width is *four times* the bytes of
   the fp16 ``K`` tile it rotates, and re-reading it on every inner iteration
   measured 2x slower than the attention math (98 us vs 48 us on the
   ``[6, 576, 1024]`` case).  Half-width tables plus ``tl.split``/``tl.join``
   for the lane swap keep the rotary essentially free.

:func:`_attn_rope_fwd` therefore reads ``q``/``k``/``v`` straight out of the
``qkv`` GEMM's output, applies the rotary to the ``q``/``k`` tiles in registers,
runs a flash-attention forward (online softmax, fp32 accumulators), and stores
``(B, S, H*D)`` -- exactly the layout the projection GEMM wants.  Forward drops
to three kernels.

Numerics follow the baseline exactly: it promotes ``t_middle * freqs.cos()`` to
fp32 (fp16 x fp32), sums in fp32 and rounds **once** on the closing ``.to()``,
which is what the kernel does before handing the tile to ``tl.dot``.

cuDNN's ``sdpa_sm100_flash`` kernel is faster than this Triton pipeline on the
attention proper (27 us vs 42 us on the ``[6, 576, 1024]`` case), but it cannot
read the un-rotated ``qkv`` buffer and cannot write the projection's layout, so
using it costs a rotary kernel in front and a transpose behind -- and with the
transpose added it is already the slower of the two (40 us).  Measured
end-to-end against that five-kernel arrangement (one fused rotary + SDPA +
transpose + the two GEMMs), which is itself ~3x the baseline: 114 us vs 125 us
at ``[6, 576, 1024]`` and 50 us vs 61 us at ``[1, 576, 1024]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

_LOG2E = 1.4426950408889634


@triton.jit
def _rope(t, cs, sn, M: tl.constexpr, D: tl.constexpr):
    """Rotate a ``[M, D]`` tile by half-width ``[M, D/2]`` cos/sin tables.

    ``tl.split`` peels the even/odd lanes of each ``(2m, 2m+1)`` pair apart, so
    the rotation is two fused multiply-adds and ``tl.join`` interleaves them
    back -- no shared-memory transpose, and the tables stay at half width.
    """
    D2: tl.constexpr = D // 2
    te, to = tl.split(tl.reshape(t, (M, D2, 2)).to(tl.float32))
    return tl.reshape(tl.join(te * cs - to * sn, to * cs + te * sn), (M, D))


@triton.jit
def _attn_rope_fwd(
    Y, COS, SIN, O, S, qk_scale,
    H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
):
    """Rotary + flash-attention forward, fused.  grid = (cdiv(S, BM), B * H).

    ``Y`` is the ``[B, S, 3 * H * D]`` qkv-projection output; ``O`` is
    ``[B, S, H * D]``.
    """
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    HD: tl.constexpr = H * D
    Y3: tl.constexpr = 3 * HD
    D2: tl.constexpr = D // 2

    offs_m = start_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_c = tl.arange(0, D2)
    ybase = Y + b * (S * Y3) + h * D
    ty = O.dtype.element_ty

    crow_m = offs_m[:, None] * D2 + offs_c[None, :]
    if EVEN_M:
        q = tl.load(ybase + offs_m[:, None] * Y3 + offs_d[None, :])
        cq = tl.load(COS + crow_m)
        sq = tl.load(SIN + crow_m)
    else:
        mm = (offs_m < S)[:, None]
        q = tl.load(ybase + offs_m[:, None] * Y3 + offs_d[None, :], mask=mm, other=0.0)
        cq = tl.load(COS + crow_m, mask=mm, other=1.0)
        sq = tl.load(SIN + crow_m, mask=mm, other=0.0)
    q = _rope(q, cq, sq, BM, D).to(ty)

    m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)

    for start_n in tl.range(0, S, BN):
        offs_n = start_n + tl.arange(0, BN)
        kp = ybase + HD + offs_n[:, None] * Y3 + offs_d[None, :]
        crow_n = offs_n[:, None] * D2 + offs_c[None, :]
        if EVEN_N:
            k = tl.load(kp)
            ck = tl.load(COS + crow_n)
            sk = tl.load(SIN + crow_n)
            v = tl.load(kp + HD)
        else:
            nm = (offs_n < S)[:, None]
            k = tl.load(kp, mask=nm, other=0.0)
            ck = tl.load(COS + crow_n, mask=nm, other=1.0)
            sk = tl.load(SIN + crow_n, mask=nm, other=0.0)
            v = tl.load(kp + HD, mask=nm, other=0.0)
        k = _rope(k, ck, sk, BN, D).to(ty)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if not EVEN_N:
            qk = tl.where((offs_n < S)[None, :], qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(ty), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    op = O + b * (S * HD) + offs_m[:, None] * HD + h * D + offs_d[None, :]
    if EVEN_M:
        tl.store(op, acc.to(ty))
    else:
        tl.store(op, acc.to(ty), mask=(offs_m < S)[:, None])


# (BLOCK_M, BLOCK_N, num_warps, num_stages), swept on B200 over the captured
# shapes.  Two regimes, selected by how many CTAs the grid can field:
#
# * enough work to fill the machine (the ``[6, 576, 1024]`` case, 96 (b, h)
#   pairs) -- the wide row tile wins, because ``K``/``V`` are re-read once per
#   row block and BM=128 halves those passes.  BN=32 with one stage keeps the
#   fp32 accumulator + rotary tables inside the register budget at that width.
# * too few (b, h) pairs to fill it (the ``[1, 576, 1024]`` case, 16 pairs) --
#   BM=128 would leave half the SMs idle, so the narrower tile with a deeper
#   software pipeline is ~1.4x faster instead.
_CFG_WIDE = (128, 32, 4, 1)
_CFG_NARROW = (64, 64, 4, 3)

_FAST_DTYPES = (torch.float16, torch.bfloat16)


def _pick_cfg(nbh, S, device):
    """Wide tile once ``BM=128`` still fields >= 1.5 waves of CTAs."""
    bm = _CFG_WIDE[0]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if nbh * triton.cdiv(S, bm) * 2 >= 3 * sms:
        return _CFG_WIDE
    return _CFG_NARROW


def _attn_rope(y, cos, sin, cfg, B, S, H, D):
    out = torch.empty((B, S, H * D), device=y.device, dtype=y.dtype)
    bm, bn, warps, stages, grid = cfg
    _attn_rope_fwd[(grid, B * H)](
        y, cos, sin, out, S, (D ** -0.5) * _LOG2E,
        H=H, D=D, BM=bm, BN=bn,
        EVEN_M=(S % bm == 0), EVEN_N=(S % bn == 0),
        num_warps=warps, num_stages=stages,
    )
    return out


class OasisVAEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        self.register_buffer(
            "rotary_freqs",
            self.rotary.get_axial_freqs(frame_height, frame_width),
            persistent=False,
        )
        self.attn = DenseAttention(backend="sdpa")
        self.head_dim = dim // num_heads
        self._tbl_key = None
        self._tbl = None
        self._cfgs = {}

    # -- half-width cos/sin table -----------------------------------------
    def _tables(self):
        """``([S, D/2], [S, D/2])`` fp32 cos/sin, or ``None`` if unsupported.

        Built lazily: ``rotary_freqs`` is filled in ``__init__`` but the module
        is moved/cast afterwards, and ``cos`` of a large angle is far too
        sensitive to rounding to precompute at the wrong precision.
        """
        f = self.rotary_freqs
        key = (f.data_ptr(), f.dtype, f.device, f._version, f.shape)
        if key != self._tbl_key:
            self._tbl = self._build_tables(f)
            self._tbl_key = key
        return self._tbl

    def _build_tables(self, f):
        D = self.head_dim
        rot = f.shape[-1]
        if D < 16 or D & (D - 1) or rot % 2 or rot > D or not f.is_cuda:
            return None
        if self.num_heads * D != self.qkv.weight.shape[1]:
            return None
        flat = f.reshape(-1, rot).float()
        if flat.shape[0] != self.frame_height * self.frame_width:
            return None
        # ``freqs`` is repeat_interleave(2)-ed, so each (2m, 2m+1) lane pair
        # shares an angle and the table needs only D/2 columns. Verify rather
        # than assume -- a different frequency layout must not fuse.
        if not torch.equal(flat[:, 0::2], flat[:, 1::2]):
            return None
        ang = flat[:, 0::2]
        cos = torch.ones((flat.shape[0], D // 2), device=f.device, dtype=torch.float32)
        sin = torch.zeros_like(cos)
        cos[:, : rot // 2] = ang.cos()
        sin[:, : rot // 2] = ang.sin()
        return cos, sin

    def _cfg(self, nbh, seq_len, device):
        """Launch recipe for this batch, memoized (cached past the first call)."""
        cfg = self._cfgs.get(nbh)
        if cfg is None:
            bm, bn, warps, stages = _pick_cfg(nbh, seq_len, device)
            cfg = (bm, bn, warps, stages, triton.cdiv(seq_len, bm))
            self._cfgs[nbh] = cfg
        return cfg

    # -- baseline path, for shapes/dtypes the fused kernel does not cover --
    def _forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (bsz, self.frame_height, self.frame_width, self.num_heads, -1)
        q = q.reshape(*shape).permute(0, 3, 1, 2, 4)
        k = k.reshape(*shape).permute(0, 3, 1, 2, 4)
        v = v.reshape(*shape).permute(0, 3, 1, 2, 4)
        q = oasis_apply_rotary_emb(self.rotary_freqs, q)
        k = oasis_apply_rotary_emb(self.rotary_freqs, k)
        seq_len = self.frame_height * self.frame_width
        q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        out = self.attn(q, k, v).reshape(bsz, seq_len, -1)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in _FAST_DTYPES or not x.is_cuda or x.dim() != 3:
            return self._forward_eager(x)
        tbl = self._tables()
        if tbl is None:
            return self._forward_eager(x)
        bsz, seq_len = x.shape[0], self.frame_height * self.frame_width
        if x.shape[1] != seq_len or bsz == 0:
            return self._forward_eager(x)
        H, D = self.num_heads, self.head_dim
        cos, sin = tbl
        y = F.linear(x, self.qkv.weight, self.qkv.bias)
        cfg = self._cfg(bsz * H, seq_len, x.device)
        out = _attn_rope(y, cos, sin, cfg, bsz, seq_len, H, D)
        return F.linear(out, self.proj.weight, self.proj.bias)
