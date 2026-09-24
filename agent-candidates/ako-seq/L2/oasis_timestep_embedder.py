"""Oasis timestep embedding: the whole module in two fused CUDA kernels.

The reference forward is ten device ops for ~6 rows of real work -- a per-call
``arange`` + ``exp`` to rebuild the 128-entry frequency table, an int64->fp32
cast, the broadcast outer product, ``cos``, ``sin``, a ``cat``, then
``addmm`` + SiLU + ``addmm`` -- so it is entirely device-op-count bound: under
the benchmark's timing loop (host far ahead of the device, which is draining a
253MiB ``l2.zero_()``) the window is pure device time and each op costs a whole
~2.05us stream quantum, while the 5MB of weights the math actually needs is
nearly free to read.  Measured on B200: the pool copy alone is 6.9us, one added
tiny op 11.1us, two 13.5us, ten 42.1us, and an ``F.linear`` reading 0.5MB and
one reading 4MB both land on the same 15.3us step.

So this is an op-elimination kernel, not an arithmetic one:

  * everything input-independent is hoisted out of the hot path -- the frequency
    table is built once per device and cached (:meth:`_freq_table`), and the
    weights are used in their natural ``[N, K]`` row-major layout, which is
    already what a warp-per-output-column dot product wants, so there is no
    pre-transposed copy to rebuild after the harness' ``load_state_dict``;
  * the ``[B, 256]`` sinusoidal embedding is built inside the first kernel in
    shared memory (128 x B ``sincosf`` spread over the block's threads) and
    consumed from there, so the cast/mul/cos/sin/cat chain and the intermediate
    itself both disappear;
  * bias + SiLU are the first kernel's epilogue;
  * both kernels launch with programmatic dependent launch, so each one's grid
    setup and producer-independent loads (kernel 2's entire 4MB weight slice)
    overlap the previous op on the stream.

Two device ops per call, measured at 13.3us (B<=4) / 15.4us (B>=5) against a
68-91us reference.  ``solution/oasis_tse.cu`` carries the per-kernel design notes
and the measurements behind each choice; ``ITERATIONS.md`` has the cost model,
the sweeps, and the dead ends.

One correctness note that is easy to get backwards: this GPU's ``F.linear``
computes fp32 in **TF32** (``allow_tf32`` is True), so the kernel rounds both
GEMM operands to tf32 round-to-nearest-even to match it.  An exact-fp32 kernel is
*more* accurate and fails the harness' atol 1e-5 / rtol 1e-3 window.  cuBLAS only
does this for M >= 2 -- at M == 1 it uses an exact-fp32 GEMV -- hence the
``_MIN_BATCH`` gate.

Anything the fast path does not cover -- CPU tensors, non-fp32 weights, autograd,
``frequency_embedding_size != 256``, ``hidden_size != 1024``, batch outside
2..8, no nvcc -- falls through to the reference implementation below, which is
the baseline composition over the frozen L1 winners.  All of those paths are
checked in ``probe/fallback.py``.
"""

from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

_CU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oasis_tse.cu")

_CPP_DECL = (
    "#include <torch/extension.h>\n"
    "at::Tensor fk_oasis_tse(const at::Tensor &t, const at::Tensor &freqs,\n"
    "                        const at::Tensor &w1, const at::Tensor &b1,\n"
    "                        const at::Tensor &w2, const at::Tensor &b2);\n"
    "void fk_oasis_set_cfg(int64_t wpb1, int64_t wpb2, int64_t pdl,\n"
    "                      int64_t mode);\n"
)

# Shapes the CUDA path is compiled for (kernel 2's K and its shared-memory
# staging buffer are compile-time constants).
_FREQ_DIM = 256
_HIDDEN = 1024
_MIN_BATCH = 2  # M == 1 hits cuBLAS's exact-fp32 GEMV; the kernel emulates TF32
_MAX_BATCH = 8

_EXT = None


def _build():
    from torch.utils.cpp_extension import load_inline

    with open(_CU_PATH) as fh:
        src = fh.read()
    # Build for exactly the local GPU: the kernel uses sm_90+ instructions
    # (griddepcontrol, cvt.rn.tf32.f32) that a wider inherited arch list would
    # reject in ptxas. Restored afterwards so no other extension's build sees it.
    major, minor = torch.cuda.get_device_capability()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major >= 9 else "")
    # Name carries a source hash: load_inline caches its build directory by
    # name, so this rebuilds exactly when the .cu changes and never otherwise.
    tag = hashlib.sha1(src.encode()).hexdigest()[:10]
    try:
        return load_inline(
            name=f"fk_oasis_tse_{tag}",
            cpp_sources=_CPP_DECL,
            cuda_sources=src,
            functions=["fk_oasis_tse", "fk_oasis_set_cfg"],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = _build()
    return _EXT


class OasisTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Linear(frequency_embedding_size, hidden_size, bias=True),
                SiLU(),
                Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Bound native entry point (None until/unless it can be built), and the
        # per-device frequency table. Both live outside the module's parameter /
        # buffer machinery: nothing here is state_dict state.
        object.__setattr__(self, "_fused", None)
        object.__setattr__(self, "_freqs", {})
        if frequency_embedding_size == _FREQ_DIM and hidden_size == _HIDDEN:
            if torch.cuda.is_available():
                try:
                    object.__setattr__(self, "_fused", _ext().fk_oasis_tse)
                except Exception:  # noqa: BLE001 - no nvcc: reference path
                    pass

    # -- input-independent table, built once per device -----------------------
    def _freq_table(self, device: torch.device) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        table = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
            / half,
        )
        self._freqs[device] = table
        return table

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        fused = self._fused
        if fused is not None:
            w1 = self.mlp[0].weight
            b1 = self.mlp[0].bias
            w2 = self.mlp[2].weight
            b2 = self.mlp[2].bias
            if (
                t.is_cuda
                and t.dtype is torch.int64
                and t.dim() == 1
                and _MIN_BATCH <= t.shape[0] <= _MAX_BATCH
                and t.is_contiguous()
                and b1 is not None
                and b2 is not None
                and w1.dtype is torch.float32
                and w2.dtype is torch.float32
                and b1.dtype is torch.float32
                and b2.dtype is torch.float32
                and w1.shape == (_HIDDEN, _FREQ_DIM)
                and w2.shape[1] == _HIDDEN
                and w1.is_contiguous()
                and w2.is_contiguous()
                and b1.is_contiguous()
                and b2.is_contiguous()
                and w1.device == t.device
                and not torch.is_grad_enabled()
            ):
                freqs = self._freqs.get(t.device)
                if freqs is None:
                    freqs = self._freq_table(t.device)
                return fused(t, freqs, w1, b1, w2, b2)
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x
