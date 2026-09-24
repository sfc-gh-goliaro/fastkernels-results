"""Oasis timestep embedding — single-call fused CUDA path.

The module is tiny (B<=6 rows, 5MB of weights, ~16 MFLOP) so eager PyTorch
issues ~11 kernels for the frequency table, the sinusoidal embedding, two GEMVs
and a SiLU.  Each *serialized* kernel launch costs ~2.2us end-to-end here
regardless of its size, so the baseline spends ~28us purely arriving at its work;
its 5MB of weights would stream in 0.6us at B200 peak.  Measured baseline: 52us.

So everything batch-independent is hoisted out of ``forward`` -- the log-spaced
frequency table, the parameter pointers, the output and hidden-activation
buffers, the precision policy -- and the remaining chain (t->float, cos/sin, the
cos|sin concat, addmm+bias, SiLU, addmm+bias) is collapsed into two hand-written
CUDA kernels issued by one pybind call, the second launched as programmatically
dependent on the first so its prologue overlaps the first's tail.

Anything the fused path does not cover (non-CUDA tensors, non-int64 ``t``,
batch outside 1..8, non-fp32/non-contiguous/misaligned weights, missing bias,
shared-memory budget exceeded) falls through to ``_eager``, which is the
baseline formulation verbatim.
"""

from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.silu import SiLU

_MAX_BATCH = 8
_SHARED_BUDGET = 48 * 1024

# Launch geometry (warps per block, rows per thread, __launch_bounds__ min
# blocks/SM) -- these must match the single K1_CASE / K2_CASE instantiated in
# oasis_te.cu.  Chosen by a 42-point sweep; see ITERATIONS.md for the full table
# and the runners-up.
_K1_GEOM = (8, 2, 1)
_K2_GEOM = (8, 2, 4)
# Programmatic dependent launch: let k2's blocks start (fetching their own code,
# bias and W2 tile) while k1's tail drains.  sm_90+; the biggest single win here.
_PDL = 1

_EXT = None


def _ext():
    """Compile / fetch the fused extension (once per process)."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oasis_te.cu")
        with open(src, "rb") as fh:
            tag = hashlib.sha1(fh.read()).hexdigest()[:12]
        # Build only for the device we run on: the ambient TORCH_CUDA_ARCH_LIST
        # reaches back to sm_75, where `cvt.rna.tf32.f32` does not exist.
        major, minor = torch.cuda.get_device_capability()
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            _EXT = load(
                name=f"oasis_te_{tag}_sm{major}{minor}",
                sources=[src],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False,
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _EXT


def _geom_ok(nk: int, tpb: int) -> bool:
    """Can the kernel's K-axis decomposition cover ``nk`` float4 with ``tpb`` threads?

    ``nk`` must be a power of two (the kernel splits a thread's row-group and
    k-slot with shifts rather than runtime integer divisions).  Then either the
    row is wider than a block (whole block on one row, ``nk/tpb`` passes) or
    several row-groups share a block (``tpb/nk`` groups, each a whole number of
    warps so the shuffle reduction stays inside one group).
    """
    if nk & (nk - 1):
        return False
    if nk >= tpb:
        return True
    return tpb % nk == 0 and nk >= 32


def _to_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round to TF32 (round-to-nearest, ties away) keeping the fp32 container."""
    i = x.contiguous().view(torch.int32)
    return ((i + 0x1000) & -0x2000).view(torch.float32)


def _tf32_matches(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> int:
    """1 if ``F.linear`` on this shape behaves like TF32, 0 if like exact fp32.

    cuBLAS only takes the TF32 tensor-core path for some shapes, so rather than
    assume, compare the reference against both candidate semantics evaluated in
    float64 and keep whichever it sits closer to.  Run once per (shape, batch).
    """
    ref = F.linear(x, w, bias).double()
    wd, bd = w.double(), bias.double()
    exact = x.double() @ wd.t() + bd
    quant = _to_tf32(x).double() @ _to_tf32(w).double().t() + bd
    d_exact = (ref - exact).abs().max().item()
    d_quant = (ref - quant).abs().max().item()
    return 1 if d_quant < d_exact else 0


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
        self._fn = None        # extension entry point once the fast path is live
        self._args = None      # (freqs_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
                               #  dim, half, hidden)  -- all plain ints
        self._params = None    # the tensors those pointers came from
        self._bufs = {}        # batch -> (work, work_ptr, flags)
        self._blocked = False  # fast path known unusable for this module

    def _invalidate(self):
        """Drop the cached pointers/buffers; the next forward re-derives them."""
        self._fn = None
        self._args = None
        self._params = None
        self._bufs = {}
        self._blocked = False

    def _apply(self, *args, **kwargs):
        # .to() / .cuda() / .float() replace the parameter tensors, which would
        # leave the cached device pointers dangling.
        self._invalidate()
        return super()._apply(*args, **kwargs)

    # -- baseline formulation, used for anything the fused path rejects --------
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

    def _eager(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x

    # -- one-time setup -------------------------------------------------------
    def _setup(self) -> bool:
        self._blocked = True
        fc1, fc2 = self.mlp[0], self.mlp[2]
        w1, b1, w2, b2 = fc1.weight, fc1.bias, fc2.weight, fc2.bias
        if b1 is None or b2 is None:
            return False
        params = (w1, b1, w2, b2)
        if not all(p.is_cuda and p.dtype is torch.float32 and p.is_contiguous()
                   for p in params):
            return False
        dim = int(self.frequency_embedding_size)
        hidden = int(w1.shape[0])
        if dim < 2 or hidden < 1:
            return False
        if tuple(w1.shape) != (hidden, dim) or tuple(w2.shape) != (hidden, hidden):
            return False
        if tuple(b1.shape) != (hidden,) or tuple(b2.shape) != (hidden,):
            return False
        # The kernels load weights as float4, so rows must be a multiple of 4
        # wide and 16B-aligned.  dim being a multiple of 4 also makes it even, so
        # the baseline's odd-dim zero-pad column cannot arise on this path.
        if dim % 4 or hidden % 4 or w1.data_ptr() % 16 or w2.data_ptr() % 16:
            return False
        nw1, rpt1, _ = _K1_GEOM
        nw2, rpt2, _ = _K2_GEOM
        if not _geom_ok(dim // 4, nw1 * 32) or not _geom_ok(hidden // 4, nw2 * 32):
            return False
        if (_MAX_BATCH * dim + nw1 * rpt1 * (_MAX_BATCH + 1)) * 4 > _SHARED_BUDGET:
            return False
        half = dim // 2
        # Built exactly as the baseline builds it, so the table is bit-identical.
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=w1.device)
            / half,
        ).contiguous()
        try:
            fn = _ext().oasis_fwd
        except Exception:
            return False
        self._freqs = freqs                                  # keep alive
        self._args = (freqs.data_ptr(), w1.data_ptr(), b1.data_ptr(),
                      w2.data_ptr(), b2.data_ptr(), dim, half, hidden)
        self._params = (w1, b1, w2, b2)
        self._fn = fn
        self._blocked = False
        return True

    def _make_buf(self, batch: int):
        """Per-batch scratch for the hidden activation, plus the precision flags.

        The hidden activation buffer is cached (it is purely internal), but the
        *output* is allocated per call: returning a reused buffer would alias
        across calls, and the allocation costs nothing that can be measured here
        (the harness's pre-iteration L2 flush leaves ~68us of host head start).
        """
        if not 1 <= batch <= _MAX_BATCH:
            return None
        dim, hidden = self._args[5], self._args[7]
        w1, b1, w2, b2 = self._params
        dev = w1.device
        work = torch.empty((batch, hidden), dtype=torch.float32, device=dev)
        with torch.no_grad():
            probe1 = torch.empty((batch, dim), device=dev).uniform_(-1.0, 1.0)
            probe2 = torch.empty((batch, hidden), device=dev).normal_(0.0, 0.3)
            tf1 = _tf32_matches(probe1, w1, b1)
            tf2 = _tf32_matches(probe2, w2, b2)
        buf = (work, work.data_ptr(), tf1 | (tf2 << 1) | (_PDL << 10))
        self._bufs[batch] = buf
        return buf

    # -- hot path -------------------------------------------------------------
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        fn = self._fn
        if fn is None:
            if self._blocked or not self._setup():
                return self._eager(t)
            fn = self._fn
        if t.dtype is not torch.int64 or not t.is_cuda or t.dim() != 1 or not t.is_contiguous():
            return self._eager(t)
        batch = t.shape[0]
        buf = self._bufs.get(batch)
        if buf is None:
            buf = self._make_buf(batch)
            if buf is None:
                return self._eager(t)
        a = self._args
        p = self._params
        # Cheap guard against a weight tensor having been swapped out from under
        # the cached pointers by something `_apply` does not see (e.g. a direct
        # `param.data = ...` assignment).
        if (p[0].data_ptr() != a[1] or p[1].data_ptr() != a[2]
                or p[2].data_ptr() != a[3] or p[3].data_ptr() != a[4]):
            self._invalidate()
            return self.forward(t)
        out = torch.empty((batch, a[7]), dtype=torch.float32, device=p[0].device)
        fn(t.data_ptr(), a[0], a[1], a[2], a[3], a[4], buf[1], out.data_ptr(),
           batch, a[5], a[6], a[7], buf[2])
        return out
