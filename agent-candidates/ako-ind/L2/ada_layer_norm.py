"""Adaptive Layer Norm modules for diffusion transformers (L2 composite).

AdaLayerNormZero: 6-output adaLN-Zero for dual-stream FLUX blocks.
AdaLayerNormZeroSingle: 3-output adaLN-Zero for single-stream FLUX blocks.

Two kernels total, down from five, chained with a *programmatic dependent
launch* so the boundary between them costs nothing.

* The x-pass is one Triton kernel.  Eager
  ``F.layer_norm(x) * (1 + scale[:, None]) + shift[:, None]`` is three kernels
  moving 6x the bytes of ``x`` (a full x-sized temporary per stage) where 2x
  suffices; here one CTA owns one row, loads x once, reduces mean and E[x^2] in
  a single fp32 reduction round and stores ``xhat * (1 + scale) + shift`` once.
* The SiLU is folded into the projection GEMV.  It is only ``embedding_dim``
  elements, so as a separate launch it is pure overhead -- and each GEMV CTA can
  recompute ``silu(emb)`` itself from an L2-resident 6KB vector for free.
* Both launches are chained with PDL (``launch_pdl=True`` plus
  ``gdc_launch_dependents`` / ``gdc_wait``), worth 4.1us on every case.  A kernel
  boundary costs ~4us here and it is GPU-side ramp and drain, not host launch
  latency, so CUDA graphs do not touch it -- but a dependent launch does.  This
  operator is an unusually good fit: the x-pass's whole read of ``x`` and *both*
  its moments are independent of the projection and only the final
  scale-and-store needs ``shift_msa``/``scale_msa``, so its wait sits after the
  reduction rather than at the top (worth 2us).  Collapsing the two kernels into
  one instead, with a grid-wide software barrier, was measured and loses: the
  barrier costs exactly as much as the launch it removes.

``scale``/``shift`` are read straight out of the flat projection output with
constexpr offsets, so the other chunks stay zero-traffic views exactly as
``chunk()`` returns them.  The projection weight read is irreducible (every
chunk is returned), so that is where the remaining time goes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _silu_gemv_kernel(W, E, B, OUT, N,
                      K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      NK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """OUT[n] = bias[n] + sum_k W[n, k] * silu(E[k]).

    Deliberately mask-free: N is a multiple of BN and K a multiple of BK (the
    caller checks both), and predicating the 108MiB W tile load costs 8us --
    more than the whole SiLU fusion saves.  Every CTA reloads the entire
    ``silu(emb)`` vector; it is a few KB and hits L2, and measuring against a
    pre-activated variant showed the fold is free.

    ``evict_first`` on the W load is worth 4us on the 6x projection and 2us on
    the 3x one -- it is what takes this kernel from 4us behind cuBLAS to level
    with it.  W is 108MiB against a ~126MiB L2 that the harness has just left
    full of dirty lines, and each weight element is read exactly once, so
    letting the stream claim L2 residency only costs eviction pressure.  The
    per-row reduction and the fp32 upcast are *not* the bottleneck: variants
    that drop either measured identical.

    This kernel carries ``launch_pdl``, so it may begin before the grid that
    produced ``emb`` has finished -- hence the ``gdc_wait`` before the first load.
    What that buys is the *ramp*: its CTAs are dispatched and resident while the
    predecessor drains.  ``gdc_launch_dependents`` at the tail lets the x-pass do
    the same behind this kernel.  Both intrinsics are no-ops when PDL is off.
    """
    gdc_wait()
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros([BN], dtype=tl.float32)
    for j in tl.static_range(NK):
        kk = j * BK + k
        e = tl.load(E + kk).to(tl.float32)
        v = e * tl.sigmoid(e)
        w = tl.load(W + n[:, None] * K + kk[None, :],
                    eviction_policy="evict_first").to(tl.float32)
        acc += tl.sum(w * v[None, :], 1)
    if HAS_BIAS:
        acc += tl.load(B + n).to(tl.float32)
    # evict_last: every x-pass CTA reads a 12KB slice of this, so it wants to
    # survive the x stream that is about to sweep through L2.
    tl.store(OUT + n, acc.to(OUT.dtype.element_ty), eviction_policy="evict_last")
    gdc_launch_dependents()


@triton.jit
def _adaln_kernel(X, Y, E,
                  stride_xb, stride_xm, stride_yb, stride_ym, stride_eb,
                  eps,
                  D: tl.constexpr, BD: tl.constexpr,
                  SHIFT_OFF: tl.constexpr, SCALE_OFF: tl.constexpr):
    """y[b, m, :] = layer_norm(x[b, m, :]) * (1 + scale[b]) + shift[b].

    Ordering is deliberate and is what makes the PDL chain pay: ``x`` and both
    moments come first because they do not depend on the projection at all, and
    ``gdc_wait`` -- the point at which this kernel is willing to block on the
    GEMV -- sits after them.  Putting the wait at the top instead costs 2us
    aggregate.
    """
    r = tl.program_id(0)
    b = tl.program_id(1)

    cols = tl.arange(0, BD)
    m = cols < D

    # One reduction round for both moments: two dependent block-wide reductions
    # (mean, then centred var) put an extra barrier on the critical path with
    # nothing to overlap it.
    x = tl.load(X + b * stride_xb + r * stride_xm + cols, mask=m, other=0.0,
                eviction_policy="evict_first").to(tl.float32)
    mean = tl.sum(x, 0) * (1.0 / D)
    var = tl.sum(x * x, 0) * (1.0 / D) - mean * mean

    # Everything above is producer-independent; block on the GEMV only here.
    gdc_wait()

    # scale/shift stay bf16 in registers until use -- holding them as fp32 at
    # BD=4096 costs 64 regs/thread and caps blocks/SM.  Eviction hints:
    # scale/shift are read by every CTA (evict_last), x and y are each touched
    # exactly once (evict_first) so they should not displace it.
    ep = E + b * stride_eb + cols
    shift = tl.load(ep + SHIFT_OFF, mask=m, other=0.0, eviction_policy="evict_last")
    scale = tl.load(ep + SCALE_OFF, mask=m, other=0.0, eviction_policy="evict_last")
    y = ((x - mean) * tl.rsqrt(var + eps) * (scale.to(tl.float32) + 1.0)
         + shift.to(tl.float32))
    tl.store(Y + b * stride_yb + r * stride_ym + cols,
             y.to(Y.dtype.element_ty), mask=m, eviction_policy="evict_first")


# GEMV tile: BN=8 rows/CTA with an exact BK=1024 K-loop measured best over
# BN in {2,4,8,16,32} x BK in {512,1024} x num_warps in {4,8}.
_GEMV_BN, _GEMV_BK, _GEMV_WARPS = 8, 1024, 4
_XPASS_WARPS = 4

# PDL is Hopper+ (the launch attribute goes through cuLaunchKernelEx); the
# device-side intrinsics are documented no-ops when it is off, so the only thing
# that needs gating is the launch flag.
_PDL: dict[int, bool] = {}


def _pdl(device: torch.device) -> bool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ok = _PDL.get(idx)
    if ok is None:
        ok = _PDL[idx] = torch.cuda.get_device_capability(idx)[0] >= 9
    return ok


def _silu_gemv(emb: torch.Tensor, weight: torch.Tensor,
               bias: torch.Tensor | None) -> torch.Tensor:
    N, K = weight.shape
    out = torch.empty((1, N), device=emb.device, dtype=emb.dtype)
    _silu_gemv_kernel[(N // _GEMV_BN,)](
        weight, emb, weight if bias is None else bias, out, N,
        K=K, BN=_GEMV_BN, BK=_GEMV_BK, NK=K // _GEMV_BK,
        HAS_BIAS=bias is not None, num_warps=_GEMV_WARPS,
        launch_pdl=_pdl(emb.device),
    )
    return out


def _adaln(x: torch.Tensor, emb: torch.Tensor, D: int,
           shift_off: int, scale_off: int, eps: float) -> torch.Tensor:
    y = torch.empty_like(x)
    _adaln_kernel[(x.shape[1], x.shape[0])](
        x, y, emb,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1), emb.stride(0),
        eps,
        D=D, BD=triton.next_power_of_2(D), SHIFT_OFF=shift_off,
        SCALE_OFF=scale_off, num_warps=_XPASS_WARPS,
        launch_pdl=_pdl(x.device),
    )
    return y


def _can_fuse_gemv(emb: torch.Tensor, weight: torch.Tensor) -> bool:
    """The GEMV kernel is mask-free: it needs a single emb row, a row-major
    weight, N % BN == 0 and K % BK == 0."""
    N, K = weight.shape
    return (emb.is_cuda and emb.ndim == 2 and emb.shape[0] == 1
            and emb.is_contiguous() and emb.shape[1] == K
            and weight.stride(1) == 1 and weight.stride(0) == K
            and N % _GEMV_BN == 0 and K % _GEMV_BK == 0)


def _can_fuse_xpass(x: torch.Tensor, emb: torch.Tensor, D: int) -> bool:
    return (x.is_cuda and x.ndim == 3 and x.shape[2] == D and x.stride(2) == 1
            and emb.ndim == 2 and emb.stride(1) == 1
            and emb.shape[0] == x.shape[0])


def _project(mod: nn.Module, emb: torch.Tensor) -> torch.Tensor:
    w, b = mod.linear.weight, mod.linear.bias
    if _can_fuse_gemv(emb, w):
        return _silu_gemv(emb, w, b)
    return mod.linear(mod.silu(emb))


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int | None = None,
                 norm_type="layer_norm", bias=True, promote_fp32: bool = True):
        super().__init__()
        self.emb = None

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            # promote_fp32=False (bf16 F.layer_norm already accumulates stats in
            # fp32) avoids a full fp32 up/down-cast; callers on bf16 pass False.
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )
        self._dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        D = self._dim
        emb = _project(self, emb)
        if _can_fuse_xpass(x, emb, D):
            # shift_msa @ 0 and scale_msa @ D are read inside the kernel; the
            # rest stay views into ``emb``, as ``emb.chunk(6, dim=1)`` returns.
            out = _adaln(x, emb, D, 0, D, self.norm.eps)
            return (out, emb[:, 2 * D:3 * D], emb[:, 3 * D:4 * D],
                    emb[:, 4 * D:5 * D], emb[:, 5 * D:6 * D])
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero) for single-stream blocks.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True,
                 promote_fp32: bool = True):
        super().__init__()

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )
        self._dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        D = self._dim
        emb = _project(self, emb)
        if _can_fuse_xpass(x, emb, D):
            out = _adaln(x, emb, D, 0, D, self.norm.eps)
            return out, emb[:, 2 * D:3 * D]
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa
