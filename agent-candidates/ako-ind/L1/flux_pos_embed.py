"""2D rotary position embeddings for FLUX (concatenated per-axis 1D embeddings).

Optimized drop-in replacement for the diffusers ``FluxPosEmbed`` reference in
``baseline.py``.

The reference builds, *per call*, one ``arange`` + fp64 ``pow`` + ``div`` per
axis, then an ``outer``, a ``polar``, two strided complex views and two ``cat``s
-- roughly 25-30 kernel launches for ~300K elements of real work. It is entirely
launch-bound. Two changes:

1. The inverse-frequency vector depends only on ``theta`` and ``axes_dim``, never
   on the input, so it is built once (lazily, per device) and cached as a single
   concatenated ``[D]`` fp64 table, where ``D = sum(axes_dim) // 2`` is the
   number of output columns.
2. One Triton kernel reads ``ids``, gathers the right ``ids`` column per output
   column (a ``[D]`` int32 column->axis map, also cached), and writes both
   contiguous ``[S, D]`` fp64 outputs in a single pass -- no complex
   intermediate, no strided views, no ``cat``.

Anything the fused path does not cover (CPU tensors, exotic dtypes/ranks) falls
back to the reference implementation below.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - Triton is present in the bench env
    _HAVE_TRITON = False


# ---------------------------------------------------------------------------
# Reference path (fallback) -- copied from diffusers' get_1d_rotary_pos_embed.
# ---------------------------------------------------------------------------
def _get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int | torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = False,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )
    freqs = torch.outer(pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


# ---------------------------------------------------------------------------
# Fused kernel.
# ---------------------------------------------------------------------------
if _HAVE_TRITON:

    @triton.jit
    def _flux_rope_kernel(
        ids_ptr,          # *bf16/fp16/fp32  [S, n_axes] (strided)
        cos_ptr,          # *fp64            [S, D] contiguous
        sin_ptr,          # *fp64            [S, D] contiguous
        inv_ptr,          # *fp64            [D]  concatenated inverse frequencies
        ax_ptr,           # *int32           [D]  column -> ids column
        S,
        stride_row,
        stride_col,
        D: tl.constexpr,
        D_P: tl.constexpr,      # D rounded up to a power of two
        BLOCK_S: tl.constexpr,
        FP64_TRIG: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
        rmask = rows < S
        cols = tl.arange(0, D_P)

        if D_P == D:
            inv = tl.load(inv_ptr + cols)
            ax = tl.load(ax_ptr + cols)
            pos = tl.load(
                ids_ptr + rows[:, None] * stride_row + ax[None, :] * stride_col,
                mask=rmask[:, None],
                other=0.0,
            )
        else:
            cmask = cols < D
            inv = tl.load(inv_ptr + cols, mask=cmask, other=0.0)
            ax = tl.load(ax_ptr + cols, mask=cmask, other=0)
            pos = tl.load(
                ids_ptr + rows[:, None] * stride_row + ax[None, :] * stride_col,
                mask=rmask[:, None] & cmask[None, :],
                other=0.0,
            )

        # The angle is formed in fp64 exactly as the reference does (fp32 pos
        # promoted against the fp64 inverse-frequency table).
        ang = pos.to(tl.float64) * inv[None, :]

        if FP64_TRIG:
            c = tl.cos(ang)
            s = tl.sin(ang)
        else:
            # fp64 range reduction, then fp32 trig on |x| <= pi: absolute error
            # stays ~1.5e-7 for any angle magnitude, ~70x inside the 1e-5 atol.
            #
            # NOTE: both constants must stay *inline*. Binding one to a local
            # (`two_pi = 6.283185307179586`) makes Triton materialize it as
            # fp32, and the fp32-rounded 2*pi then leaks k*1.5e-7 of error into
            # the reduced angle -- 1.7e-6 at |ang|=63, 2.9e-4 at |ang|=10240.
            # Inline, the literal is typed by its fp64 operand and stays exact.
            k = tl.extra.cuda.libdevice.rint(ang * 0.15915494309189533576888376337251)
            x = (ang - k * 6.283185307179586476925286766559).to(tl.float32)
            c = tl.cos(x).to(tl.float64)
            s = tl.sin(x).to(tl.float64)

        off = rows[:, None] * D + cols[None, :]
        if D_P == D:
            tl.store(cos_ptr + off, c, mask=rmask[:, None])
            tl.store(sin_ptr + off, s, mask=rmask[:, None])
        else:
            m = rmask[:, None] & (cols < D)[None, :]
            tl.store(cos_ptr + off, c, mask=m)
            tl.store(sin_ptr + off, s, mask=m)


class FluxPosEmbed(nn.Module):
    """2D rotary position embeddings for FLUX."""

    # Rows per program, warps per program, and whether the trig runs in fp64.
    # Measured on B200 in the harness' timing window (L2 flush + full launch
    # queue), median of 50, both captured shapes:
    #   fp64 trig  BLOCK_S=4  warps=4  ->  9.2 us
    #   fp32 trig  BLOCK_S=4  warps=4  ->  7.14 us   <- window floor
    # A pure 1.5 MiB *or* 4.5 MiB memset also measures 7.1 us, and a 1-element
    # Triton kernel measures 5.15 us, so 7.1 us is the harness' single-launch
    # floor rather than a bandwidth limit -- no further headroom here.
    BLOCK_S = 4
    FP64_TRIG = False
    NUM_WARPS = 4

    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        # Output column count and the column -> axis map are fixed at
        # construction; only the device-resident copies are built lazily.
        self._n_axes = len(self.axes_dim)
        self._dim = sum(d // 2 for d in self.axes_dim)
        self._dim_p = max(1, 1 << (self._dim - 1).bit_length())
        self._cache: dict = {}
        self._grid_div = self.BLOCK_S

    # -- table construction (once per device; never on the hot path) --------
    def _tables(self, device: torch.device):
        entry = self._cache.get(device)
        if entry is None:
            invs = []
            axis_of_col = []
            for i, dim in enumerate(self.axes_dim):
                # Bit-identical to the reference expression.
                inv = 1.0 / (
                    self.theta
                    ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)
                )
                invs.append(inv)
                axis_of_col += [i] * (dim // 2)
            inv_freq = torch.cat(invs) if len(invs) > 1 else invs[0]
            ax = torch.tensor(axis_of_col, dtype=torch.int32, device=device)
            entry = (inv_freq, ax)
            self._cache[device] = entry
        return entry

    # -- reference path ----------------------------------------------------
    def _forward_reference(self, ids: torch.Tensor):
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = torch.float32 if ids.device.type in ("mps", "npu") else torch.float64
        for i in range(n_axes):
            freqs_cis = _get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i],
                theta=self.theta, use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin

    def _fusable(self, ids: torch.Tensor) -> bool:
        return (
            _HAVE_TRITON
            and ids.is_cuda
            and ids.ndim == 2
            and ids.shape[-1] == self._n_axes
            and ids.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64)
            # The fused kernel has no backward; keep the reference path (which
            # is differentiable w.r.t. ids) whenever a graph is being built.
            and not (ids.requires_grad and torch.is_grad_enabled())
        )

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._fusable(ids):
            return self._forward_reference(ids)

        device = ids.device
        entry = self._cache.get(device)
        if entry is None:
            entry = self._tables(device)
        inv_freq, ax = entry

        S = ids.shape[0]
        D = self._dim
        freqs_cos = torch.empty((S, D), dtype=torch.float64, device=device)
        freqs_sin = torch.empty((S, D), dtype=torch.float64, device=device)
        stride_row, stride_col = ids.stride()
        _flux_rope_kernel[((S + self._grid_div - 1) // self._grid_div,)](
            ids, freqs_cos, freqs_sin, inv_freq, ax,
            S, stride_row, stride_col,
            D=D, D_P=self._dim_p, BLOCK_S=self.BLOCK_S, FP64_TRIG=self.FP64_TRIG,
            num_warps=self.NUM_WARPS,
        )
        return freqs_cos, freqs_sin
