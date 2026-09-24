"""Timestep and text projection embeddings for diffusion models (L2 composite).

All classes are self-contained implementations that produce weight names
identical to the corresponding diffusers classes for checkpoint compatibility.

Every captured shape is batch-1, so total GPU math is negligible and runtime is
dominated by per-launch latency.  The optimisation is therefore structural:

* the sinusoid frequency table depends only on ``__init__`` args, so it is
  hoisted into a buffer instead of being rebuilt (arange/log/exp/div) per call;
* ``Timesteps`` emits its embedding -- already in ``flip_sin_to_cos`` order --
  from one kernel, replacing two ``torch.cat``s and every temporary;
* ``TimestepEmbedding`` runs as GEMV+SiLU then GEMV, linked by a programmatic
  dependent launch so the second layer's weight stream -- which does not depend
  on the first layer's output -- issues while the first drains;
* the ``Combined*`` classes fuse *across* branches: one kernel computes the
  sinusoid inline and all first-layer GEMVs, a second sums every second-layer
  GEMV straight into the output.  25-ish launches per forward become 2.

All reductions accumulate in fp32 and cast only at the final store, matching
the baseline's fp32-then-cast ordering.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from ..L1.linear import Linear
from ..L1.silu import SiLU


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style)."""
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def _sinusoid_freqs(half_dim: int, downscale_freq_shift: float,
                    max_period: int = 10000) -> torch.Tensor:
    """``exp(exponent)`` from :func:`get_timestep_embedding`, same op order."""
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    return torch.exp(exponent)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _proj_chunk(t, FREQ, k0, HALF: tl.constexpr, BK: tl.constexpr,
                FLIP: tl.constexpr, SCALE: tl.constexpr):
    """``BK`` lanes of one row of the sinusoid embedding, in output order.

    Returns fp32 values that have been round-tripped through bf16 -- the
    baseline feeds ``time_proj`` output through ``.to(dtype)`` before the MLP,
    so the GEMV input is bf16-rounded there too.
    """
    i = k0 + tl.arange(0, BK)
    lo = i < HALF
    f = tl.load(FREQ + tl.where(lo, i, i - HALF))
    e = t * f * SCALE
    if FLIP:
        v = tl.where(lo, tl.cos(e), tl.sin(e))
    else:
        v = tl.where(lo, tl.sin(e), tl.cos(e))
    return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _timesteps_kernel(T, FREQ, OUT, HALF: tl.constexpr, NC: tl.constexpr,
                      BLOCK: tl.constexpr, FLIP: tl.constexpr, SCALE: tl.constexpr):
    """One row of the sinusoid embedding, written directly in final order."""
    b = tl.program_id(0)
    t = tl.load(T + b).to(tl.float32)
    i = tl.arange(0, BLOCK)
    m = i < HALF
    f = tl.load(FREQ + i, mask=m, other=0.0)
    e = t * f * SCALE
    s = tl.sin(e)
    c = tl.cos(e)
    o = OUT + b * NC + i
    if FLIP:
        tl.store(o, c, mask=m)
        tl.store(o + HALF, s, mask=m)
    else:
        tl.store(o, s, mask=m)
        tl.store(o + HALF, c, mask=m)


@triton.jit
def _gemv_silu_kernel(X, W, B, H, K: tl.constexpr, N, SX,
                      BN: tl.constexpr, BK: tl.constexpr, HAS_B: tl.constexpr,
                      PDL: tl.constexpr):
    """``H[b] = silu(W @ X[b] + B)`` -- one block of rows, fp32 accumulation."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < K
        xv = tl.load(X + b * SX + kk, mask=mk, other=0.0).to(tl.float32)
        wv = tl.load(W + rn[:, None] * K + kk[None, :],
                     mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], 1)
    if HAS_B:
        acc += tl.load(B + rn, mask=mn, other=0.0).to(tl.float32)
    tl.store(H + b * N + rn, (acc * tl.sigmoid(acc)).to(H.dtype.element_ty), mask=mn)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _gemv_kernel(H, W, B, Y, K: tl.constexpr, N, OUT_DTYPE: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, HAS_B: tl.constexpr,
                 PDL: tl.constexpr):
    """``Y[b] = W @ H[b] + B`` -- fp32 accumulation, cast once at the store."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros([BN], dtype=tl.float32)
    if PDL:
        gdc_wait()
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < K
        hv = tl.load(H + b * K + kk, mask=mk, other=0.0).to(tl.float32)
        wv = tl.load(W + rn[:, None] * K + kk[None, :],
                     mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(wv * hv[None, :], 1)
    if HAS_B:
        acc += tl.load(B + rn, mask=mn, other=0.0).to(tl.float32)
    tl.store(Y + b * N + rn, acc.to(OUT_DTYPE), mask=mn)


@triton.jit
def _combined_stage1_kernel(
    TS, GD, POOL, FREQ,
    W1T, B1T, W1G, B1G, W1X, B1X, H,
    N, BS, KT: tl.constexpr, KX: tl.constexpr, HALF: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, BKX: tl.constexpr,
    FLIP: tl.constexpr, SCALE: tl.constexpr, NBRANCH: tl.constexpr,
    PDL: tl.constexpr,
):
    """First MLP layer of every branch, with the sinusoid computed inline.

    Recomputing the ``KT``-wide sinusoid per block is a handful of
    transcendentals; it removes the ``time_proj`` launch, its output tensor and
    the guidance branch's copy of both.  ``H`` is ``[NBRANCH, B, N]`` fp32.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    wrow = rn[:, None] * KT
    t = tl.load(TS + b).to(tl.float32)

    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, KT, BK):
        xv = _proj_chunk(t, FREQ, k0, HALF, BK, FLIP, SCALE)
        wv = tl.load(W1T + wrow + (k0 + tl.arange(0, BK))[None, :],
                     mask=mn[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], 1)
    acc += tl.load(B1T + rn, mask=mn, other=0.0).to(tl.float32)
    tl.store(H + b * N + rn, (acc * tl.sigmoid(acc)).to(H.dtype.element_ty), mask=mn)

    if NBRANCH == 3:
        g = tl.load(GD + b).to(tl.float32)
        acc = tl.zeros([BN], dtype=tl.float32)
        for k0 in range(0, KT, BK):
            xv = _proj_chunk(g, FREQ, k0, HALF, BK, FLIP, SCALE)
            wv = tl.load(W1G + wrow + (k0 + tl.arange(0, BK))[None, :],
                         mask=mn[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], 1)
        acc += tl.load(B1G + rn, mask=mn, other=0.0).to(tl.float32)
        tl.store(H + (BS + b) * N + rn,
                 (acc * tl.sigmoid(acc)).to(H.dtype.element_ty), mask=mn)

    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, KX, BKX):
        kk = k0 + tl.arange(0, BKX)
        mk = kk < KX
        xv = tl.load(POOL + b * KX + kk, mask=mk, other=0.0).to(tl.float32)
        wv = tl.load(W1X + rn[:, None] * KX + kk[None, :],
                     mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], 1)
    acc += tl.load(B1X + rn, mask=mn, other=0.0).to(tl.float32)
    tl.store(H + ((NBRANCH - 1) * BS + b) * N + rn,
             (acc * tl.sigmoid(acc)).to(H.dtype.element_ty), mask=mn)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _gemv_accum(acc, H, W, N, rn, mn, BK: tl.constexpr):
    """Accumulate ``W @ H`` into ``acc`` -- one weight stream at a time.

    Kept in its own helper so only a single ``[BN, BK]`` tile is live per
    branch; loading all three branches inside one loop body triples register
    pressure and spills, which costs more than the extra loops save.
    """
    for k0 in range(0, N, BK):
        kk = k0 + tl.arange(0, BK)
        mk = kk < N
        wv = tl.load(W + rn[:, None] * N + kk[None, :],
                     mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(wv * tl.load(H + kk, mask=mk, other=0.0)[None, :], 1)
    return acc


@triton.jit
def _combined_stage2_kernel(
    H, W2T, W2G, W2X, BSUM, Y, N, BS, OUT_DTYPE: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, NBRANCH: tl.constexpr, PDL: tl.constexpr,
):
    """Sum every branch's second GEMV straight into the output.

    The baseline materialises three ``[B, N]`` tensors and adds them; here the
    three weight streams land in one fp32 accumulator, so the two adds and two
    of the three output tensors disappear.
    """
    pid = tl.program_id(0)
    b = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    mn = rn < N
    acc = tl.zeros([BN], dtype=tl.float32)
    if PDL:
        gdc_wait()
    acc = _gemv_accum(acc, H + b * N, W2T, N, rn, mn, BK)
    if NBRANCH == 3:
        acc = _gemv_accum(acc, H + (BS + b) * N, W2G, N, rn, mn, BK)
    acc = _gemv_accum(acc, H + ((NBRANCH - 1) * BS + b) * N, W2X, N, rn, mn, BK)
    acc += tl.load(BSUM + rn, mask=mn, other=0.0)
    tl.store(Y + b * N + rn, acc.to(OUT_DTYPE), mask=mn)


_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
             torch.float32: tl.float32}

# (BN, BK, num_warps, num_stages) per stage, tuned in ITERATIONS.md.
# Programmatic dependent launch: the second GEMV's weight stream does not
# depend on the first's output, so it can issue while stage 1 drains.
_PDL = True

_CFG_TS = 4                   # num_warps for the one-block sinusoid kernel
_CFG_GEMV1 = (4, 2048, 4, 4)
_CFG_GEMV2 = (4, 2048, 4, 2)
_CFG_S1 = (4, 2048, 1024, 4, 4)   # (BN, BK, BKX, warps, stages)
_CFG_S2 = (4, 2048, 4, 2)


class Timesteps(nn.Module):
    """Wraps get_timestep_embedding as an nn.Module."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        half = num_channels // 2
        # Input-independent: arange/log/exp/div hoisted out of forward.
        self.register_buffer("_freqs", _sinusoid_freqs(half, downscale_freq_shift),
                             persistent=False)
        self._fused = num_channels % 2 == 0 and half > 0

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if not (self._fused and timesteps.is_cuda and timesteps.dim() == 1
                and timesteps.is_contiguous() and timesteps.numel() > 0):
            return get_timestep_embedding(
                timesteps, self.num_channels,
                flip_sin_to_cos=self.flip_sin_to_cos,
                downscale_freq_shift=self.downscale_freq_shift,
                scale=self.scale,
            )
        half = self.num_channels // 2
        out = torch.empty((timesteps.shape[0], self.num_channels),
                          dtype=torch.float32, device=timesteps.device)
        _timesteps_kernel[(timesteps.shape[0],)](
            timesteps, self._freqs, out, half, self.num_channels,
            triton.next_power_of_2(half), bool(self.flip_sin_to_cos),
            float(self.scale), num_warps=_CFG_TS,
        )
        return out


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self._fused = act_fn == "silu"

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if not (self._fused and sample.is_cuda and sample.dim() == 2
                and sample.is_contiguous() and sample.shape[0] > 0):
            sample = self.linear_1(sample)
            sample = self.act(sample)
            sample = self.linear_2(sample)
            return sample

        w1, b1 = self.linear_1.weight, self.linear_1.bias
        w2, b2 = self.linear_2.weight, self.linear_2.bias
        bsz, k = sample.shape
        n = w1.shape[0]
        h = torch.empty((bsz, n), dtype=torch.float32, device=sample.device)
        bn, bk, nw, ns = _CFG_GEMV1
        _gemv_silu_kernel[(triton.cdiv(n, bn), bsz)](
            sample, w1, b1, h, k, n, k, bn,
            min(bk, triton.next_power_of_2(k)), b1 is not None, _PDL,
            num_warps=nw, num_stages=ns, launch_pdl=_PDL,
        )
        out = torch.empty((bsz, w2.shape[0]), dtype=sample.dtype, device=sample.device)
        bn, bk, nw, ns = _CFG_GEMV2
        _gemv_kernel[(triton.cdiv(w2.shape[0], bn), bsz)](
            h, w2, b2, out, n, w2.shape[0], _TL_DTYPE[sample.dtype], bn,
            min(bk, triton.next_power_of_2(n)), b2 is not None, _PDL,
            num_warps=nw, num_stages=ns, launch_pdl=_PDL,
        )
        return out


class _CombinedBase(nn.Module):
    """Shared fused forward for the two ``Combined*`` embedding classes."""

    _nbranch: int

    def _fusable(self, *tensors: torch.Tensor) -> bool:
        if not self._fused:
            return False
        return all(t.is_cuda and t.is_contiguous() and t.numel() > 0
                   for t in tensors)

    def _init_fused(self) -> None:
        tp, te = self.time_proj, self.timestep_embedder
        n = te.linear_2.weight.shape[0]
        # The fused kernels assume every branch shares the embedding dim, that
        # the sinusoid branches consume exactly num_channels inputs, and that
        # both layers are biased.  Anything else takes the reference path.
        self._fused = (
            tp.num_channels >= 2
            and tp.num_channels == triton.next_power_of_2(tp.num_channels)
            and all(m.linear_1.weight.shape[0] == n
                    and tuple(m.linear_2.weight.shape) == (n, n)
                    and m.linear_1.bias is not None and m.linear_2.bias is not None
                    for m in self._branches())
            and all(m.linear_1.weight.shape[1] == tp.num_channels
                    for m in self._branches()[:-1])
        )
        self._bsum_cache: torch.Tensor | None = None
        self.register_load_state_dict_post_hook(_drop_bsum)

    def _branches(self) -> tuple:
        if self._nbranch == 3:
            return (self.timestep_embedder, self.guidance_embedder, self.text_embedder)
        return (self.timestep_embedder, self.text_embedder)

    def _bsum(self) -> torch.Tensor:
        """``sum(linear_2.bias)`` over branches -- input-independent, so cached."""
        cached = self._bsum_cache
        if cached is None:
            cached = torch.stack([m.linear_2.bias.float() for m in self._branches()]).sum(0)
            self._bsum_cache = cached
        return cached

    def _fused_forward(self, timestep, guidance, pooled_projection):
        te, tx = self.timestep_embedder, self.text_embedder
        gu = self.guidance_embedder if self._nbranch == 3 else te
        tp = self.time_proj
        bsz = pooled_projection.shape[0]
        n = te.linear_2.weight.shape[0]
        nb = self._nbranch
        kt = tp.num_channels
        kx = tx.linear_1.weight.shape[1]

        h = torch.empty((nb * bsz, n), dtype=torch.float32,
                        device=pooled_projection.device)
        bn, bk, bkx, nw, ns = _CFG_S1
        _combined_stage1_kernel[(triton.cdiv(n, bn), bsz)](
            timestep, guidance, pooled_projection, tp._freqs,
            te.linear_1.weight, te.linear_1.bias,
            gu.linear_1.weight, gu.linear_1.bias,
            tx.linear_1.weight, tx.linear_1.bias, h,
            n, bsz, kt, kx, kt // 2, bn, min(bk, kt),
            min(bkx, triton.next_power_of_2(kx)),
            bool(tp.flip_sin_to_cos), float(tp.scale), nb, _PDL,
            num_warps=nw, num_stages=ns, launch_pdl=_PDL,
        )
        out = torch.empty((bsz, n), dtype=pooled_projection.dtype,
                          device=pooled_projection.device)
        bn, bk, nw, ns = _CFG_S2
        _combined_stage2_kernel[(triton.cdiv(n, bn), bsz)](
            h, te.linear_2.weight, gu.linear_2.weight, tx.linear_2.weight,
            self._bsum(), out, n, bsz, _TL_DTYPE[pooled_projection.dtype],
            bn, min(bk, triton.next_power_of_2(n)), nb, _PDL,
            num_warps=nw, num_stages=ns, launch_pdl=_PDL,
        )
        return out


def _drop_bsum(module, incompatible_keys=None, *args, **kwargs):
    """Invalidate the cached bias sum whenever new weights are loaded."""
    module._bsum_cache = None


class CombinedTimestepTextProjEmbeddings(_CombinedBase):
    """Combines sinusoidal timestep encoding with pooled text projection.

    Produces ``timestep_embedder`` + ``text_embedder`` weight names matching
    the diffusers checkpoint layout.
    """

    _nbranch = 2

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)
        self._init_fused()

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        if self._fusable(timestep, pooled_projection) and timestep.dim() == 1 \
                and pooled_projection.dim() == 2:
            return self._fused_forward(timestep, timestep, pooled_projection)
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(_CombinedBase):
    """Combines sinusoidal timestep + guidance encoding with pooled text projection.

    Adds a ``guidance_embedder`` on top of
    :class:`CombinedTimestepTextProjEmbeddings`.  Weight names match the
    diffusers checkpoint layout.
    """

    _nbranch = 3

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)
        self._init_fused()

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        if self._fusable(timestep, guidance, pooled_projection) and timestep.dim() == 1 \
                and guidance.dim() == 1 and pooled_projection.dim() == 2:
            return self._fused_forward(timestep, guidance, pooled_projection)
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections
