"""Timestep and text projection embeddings for diffusion models (L2 composite).

All classes are self-contained implementations that produce weight names
identical to the corresponding diffusers classes for checkpoint compatibility.

Every scored shape here is batch 1 (``timesteps bf16[1]``,
``pooled_projection bf16[1, 768]``), which makes all three operators
latency bound rather than FLOP bound -- and, unlike the elementwise L1 kernels,
*host* bound as well.  The benchmark enqueues a 265 MiB ``l2.zero_()`` before
``start.record()``, so the GPU runs ~40 us behind the host and the measured
window is ``max(device_time, host_time - 40us)``; the eager reference lands on
the host side of that max (measured: 249 us window against 299 us of host
dispatch for the 3-branch composite, 35.9 us against 74.7 us for ``Timesteps``).
See ITERATIONS.md for the probe.

So the whole per-call path -- validation, allocation, launch -- lives in
``timestep_embedding_kernels.cu`` behind a single pybind call per ``forward``,
and the op graph collapses to *one device op per module*:

  * ``Timesteps`` -- 1 kernel.  The sinusoid is written directly in its
    ``flip_sin_to_cos`` order, so no arange / exp / mul / sin / cos / cat / pad
    op reaches the device, and the frequency vector is recomputed in-kernel
    (128 ``expf``, free at this size) rather than cached and re-read.
  * ``TimestepEmbedding`` -- 1 kernel for both layers.
  * ``CombinedTimestep[Guidance]TextProjEmbeddings`` -- 1 kernel for the whole
    module: three branches, both sinusoids, both layers, no cat / cast / add op.

Both layers fit in one kernel because the second one is formulated as a sum of
rank-1 updates, ``out[:] = sum_k h[k] * W2T[k,:]``, rather than a row-wise dot
product.  In that form a block that owns a slice of the hidden dimension needs
only *its* slice of ``h``, which it computes itself from a few KB of
``linear_1.weight`` -- so stage 1's weight traffic overlaps the stage-2 stream
instead of running as a kernel in front of it.  The price is a cross-block sum,
paid with ``red.global.add.v4.f32`` into a persistent, self-zeroing fp32 scratch
plus a per-output-tile last-block finish; the 4-wide form is not optional, the
same reduction with scalar ``atomicAdd`` costs more than the fusion saves.

``linear_2.weight`` is used through a lazily built transpose, cached and keyed on
``data_ptr`` + version counter, so it is built once during the benchmark's warmup
and an in-place parameter update invalidates it.  Without it every lane of a warp
would read a different row of the row-major weight, 6 KB apart.

Round 1's two-kernel path (GEMV+bias+SiLU, then GEMV+bias) is still present and
still serves anything the fused tiling cannot cover; ``FK_TSE_PATH=1`` forces it.
Anything neither path covers -- CPU tensors, dtypes other than bf16/fp16/fp32,
batch > 1, odd ``embedding_dim``, a vector length that a warp's 16-byte lanes do
not tile exactly, missing bias, non-contiguous or unaligned weights -- falls back
to the reference composition below, which is the baseline code verbatim.
``forward`` reads the parameters through the usual attribute chain on every call
rather than caching pointers: at ~5 us that is free against the ~40 us of host
slack the harness leaves, and it cannot go stale when a parameter is reassigned.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

_CU = Path(__file__).with_name("timestep_embedding_kernels.cu")

_DECLS = """
#include <torch/extension.h>
at::Tensor fk_sinusoid(const at::Tensor &t, int64_t dim, bool flip, double shift,
                       double scale, double max_period);
at::Tensor fk_mlp(const at::Tensor &x, const at::Tensor &w1, const at::Tensor &b1,
                  const at::Tensor &w2, const at::Tensor &b2);
at::Tensor fk_combined3(const at::Tensor &timestep, const at::Tensor &guidance,
                        const at::Tensor &pooled,
                        const at::Tensor &tw1, const at::Tensor &tb1,
                        const at::Tensor &tw2, const at::Tensor &tb2,
                        const at::Tensor &gw1, const at::Tensor &gb1,
                        const at::Tensor &gw2, const at::Tensor &gb2,
                        const at::Tensor &xw1, const at::Tensor &xb1,
                        const at::Tensor &xw2, const at::Tensor &xb2,
                        int64_t proj_dim, bool flip, double shift, double scale,
                        double max_period);
at::Tensor fk_combined2(const at::Tensor &timestep, const at::Tensor &pooled,
                        const at::Tensor &tw1, const at::Tensor &tb1,
                        const at::Tensor &tw2, const at::Tensor &tb2,
                        const at::Tensor &xw1, const at::Tensor &xb1,
                        const at::Tensor &xw2, const at::Tensor &xb2,
                        int64_t proj_dim, bool flip, double shift, double scale,
                        double max_period);
int64_t fk_path_used();
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    src = _CU.read_text()
    # Build for this device only: the ambient TORCH_CUDA_ARCH_LIST in this
    # environment names 7 architectures, which multiplies the compile of ~50 tile
    # instantiations.  The arch goes into the extension name so a cached .so is
    # never reused on a different one.
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    tag = hashlib.sha1(src.encode()).hexdigest()[:10] + f"_sm{major}{minor}"
    return load_inline(
        name=f"fk_timestep_embedding_{tag}",
        cpp_sources=_DECLS,
        cuda_sources=src,
        functions=["fk_sinusoid", "fk_mlp", "fk_combined3", "fk_combined2",
                   "fk_path_used"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


_EXT = None
_TRIED = False


def _ext():
    """The built extension, or None if it cannot be built (no CUDA / no nvcc).

    Built at most once per process; a build failure is latched so the cost is paid
    once, but *absence of CUDA* is not, so a module constructed before any device
    exists still picks up the fast path later.  Every failure mode degrades to the
    reference path.
    """
    global _EXT, _TRIED
    if _EXT is None and not _TRIED and torch.cuda.is_available():
        _TRIED = True
        try:
            _EXT = _build()
        except Exception:  # noqa: BLE001 - reference path stays correct
            _EXT = None
    return _EXT


# ---------------------------------------------------------------------------
# Reference composition (baseline verbatim), used for every unsupported input.
# ---------------------------------------------------------------------------
def _ref_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
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


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style)."""
    ext = _ext()
    if ext is not None and timesteps.is_cuda:
        out = ext.fk_sinusoid(timesteps, embedding_dim, flip_sin_to_cos,
                              float(downscale_freq_shift), float(scale),
                              float(max_period))
        if out is not None:
            return out
    return _ref_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos,
                                   downscale_freq_shift, scale, max_period)


class Timesteps(nn.Module):
    """Wraps get_timestep_embedding as an nn.Module."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        ext = _ext()
        if ext is not None and timesteps.is_cuda:
            out = ext.fk_sinusoid(timesteps, self.num_channels, self.flip_sin_to_cos,
                                  float(self.downscale_freq_shift), float(self.scale),
                                  10000.0)
            if out is not None:
                return out
        return _ref_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        ext = _ext()
        if ext is not None and sample.is_cuda:
            l1, l2 = self.linear_1, self.linear_2
            b1, b2 = l1.bias, l2.bias
            if b1 is not None and b2 is not None:
                out = ext.fk_mlp(sample, l1.weight, b1, l2.weight, b2)
                if out is not None:
                    return out
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


def _mlp_params(embedder):
    """(w1, b1, w2, b2) for one branch, or None if a bias is absent."""
    l1, l2 = embedder.linear_1, embedder.linear_2
    if l1.bias is None or l2.bias is None:
        return None
    return l1.weight, l1.bias, l2.weight, l2.bias


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep encoding with pooled text projection.

    Produces ``timestep_embedder`` + ``text_embedder`` weight names matching
    the diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        ext = _ext()
        if ext is not None and pooled_projection.is_cuda:
            tp = self.time_proj
            pt = _mlp_params(self.timestep_embedder)
            px = _mlp_params(self.text_embedder)
            if pt is not None and px is not None:
                out = ext.fk_combined2(timestep, pooled_projection, *pt, *px,
                                       tp.num_channels, tp.flip_sin_to_cos,
                                       float(tp.downscale_freq_shift), float(tp.scale),
                                       10000.0)
                if out is not None:
                    return out
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep + guidance encoding with pooled text projection.

    Adds a ``guidance_embedder`` on top of
    :class:`CombinedTimestepTextProjEmbeddings`.  Weight names match the
    diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        ext = _ext()
        if ext is not None and pooled_projection.is_cuda:
            tp = self.time_proj
            pt = _mlp_params(self.timestep_embedder)
            pg = _mlp_params(self.guidance_embedder)
            px = _mlp_params(self.text_embedder)
            if pt is not None and pg is not None and px is not None:
                out = ext.fk_combined3(timestep, guidance, pooled_projection,
                                       *pt, *pg, *px,
                                       tp.num_channels, tp.flip_sin_to_cos,
                                       float(tp.downscale_freq_shift), float(tp.scale),
                                       10000.0)
                if out is not None:
                    return out
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections
