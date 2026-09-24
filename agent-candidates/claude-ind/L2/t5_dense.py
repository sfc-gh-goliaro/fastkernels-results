"""T5 feed-forward dense layers with TP sharding (L2).

T5DenseActDense: standard FFN (ColumnParallel -> act -> RowParallel).
T5DenseGatedActDense: gated FFN (MergedColumnParallel -> gate*up -> RowParallel).

``T5DenseGatedActDense`` has a fast path for the captured bf16 case: two Triton
GEMMs (TMA loads + tcgen05 MMA, warp-specialized, persistent) with the
``gelu_new`` gate activation fused into the first GEMM's epilogue.

Two details make the fusion work:

* The merged ``wi`` weight is viewed as ``[2, d_ff, d_model]``, so a single TMA
  load pulls the gate rows *and* the up rows of a tile.  One MMA accumulator of
  width ``2*BLOCK_N`` then holds both halves, which keeps the inner loop a
  single ``tl.dot`` (Triton's automatic warp specialization does not handle two
  accumulators) and halves the number of A-tile loads.
* The epilogue reproduces eager mode's *bf16* arithmetic op-by-op: every
  intermediate of ``NewGELUActivation`` is rounded back to bf16, and
  ``torch.pow(x, 3.0)`` on a bf16 tensor is ``x * x * x`` evaluated in bf16
  (i.e. with a rounding between the two multiplies), not in fp32.  Emulating
  that exactly is what makes the fused output bit-identical to the reference;
  an fp32-accurate activation drifts ~0.3% per element, which is enough to fail
  the bf16 tolerance on near-zero outputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import T5Config

from ..L1.gelu import GELU
from ..L1.silu import SiLU
from .parallel_linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


__targets__ = ["T5DenseGatedActDense", "T5DenseActDense"]


class NewGELUActivation(nn.Module):
    """GELU approximation matching HuggingFace's NewGELUActivation exactly."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def _get_act_fn(name: str) -> nn.Module:
    act_fns = {
        "relu": nn.ReLU(),
        "gelu": GELU(),
        "gelu_new": NewGELUActivation(),
        "silu": SiLU(),
    }
    if name in act_fns:
        return act_fns[name]
    raise ValueError(f"Unknown activation function: {name}")


# ---------------------------------------------------------------------------
# Triton fast path
# ---------------------------------------------------------------------------
try:
    import dataclasses

    import triton
    import triton.language as tl
    from triton.backends.nvidia.compiler import CUDAOptions
    from triton.language.extra import libdevice
    from triton.language.extra.cuda import gdc
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAVE_TRITON = True
    # Programmatic dependent launch: the second GEMM can be scheduled while the
    # first is draining.  It waits on ``griddepcontrol.wait`` before touching
    # any memory, so the dependency on ``h`` is still honored.
    _HAVE_PDL = any(f.name == "launch_pdl" for f in dataclasses.fields(CUDAOptions))
except Exception:  # pragma: no cover - torch build without triton
    _HAVE_TRITON = False
    _HAVE_PDL = False


if _HAVE_TRITON:
    _SCRATCH: dict = {}

    def _alloc_fn(size: int, alignment: int, stream):
        """Global scratch for Triton's TMA bookkeeping (reused across launches)."""
        buf = _SCRATCH.get(stream)
        if buf is None or buf.numel() < size:
            buf = torch.empty(size, device="cuda", dtype=torch.int8)
            _SCRATCH[stream] = buf
        return buf

    triton.set_allocator(_alloc_fn)

    @triton.jit
    def _rb(x):
        """Round an fp32 value to bf16 and back, like an eager bf16 op."""
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _gelu_new_mul(acc_g, acc_u):
        """bf16-faithful ``NewGELUActivation(gate) * up``."""
        g = _rb(acc_g)
        u = _rb(acc_u)
        p = _rb(_rb(g * g) * g)                      # torch.pow(g, 3.0) in bf16
        t = _rb(0.044715 * p)
        t = _rb(g + t)
        t = _rb(0.7978845608028654 * t)              # sqrt(2/pi)
        t = _rb(libdevice.tanh(t))
        t = _rb(1.0 + t)
        h = _rb(0.5 * g)
        h = _rb(h * t)
        return (h * u).to(tl.bfloat16)

    @triton.jit
    def _gated_mlp_fwd(a_desc, w_desc, h_desc, M, K, D,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       NS: tl.constexpr):
        """h = act(x @ Wg^T) * (x @ Wu^T), with x [M, K] and W [2, D, K]."""
        num_m = tl.cdiv(M, BM)
        total = num_m * tl.cdiv(D, BN)
        for tile in tl.range(tl.program_id(0), total, tl.num_programs(0),
                             flatten=True, warp_specialize=True):
            om = (tile % num_m) * BM
            on = (tile // num_m) * BN
            acc = tl.zeros((BM, 2 * BN), dtype=tl.float32)
            for k in tl.range(0, K, BK, num_stages=NS):
                b = w_desc.load([0, on, k]).reshape(2 * BN, BK)
                acc = tl.dot(a_desc.load([om, k]), b.T, acc)
            acc_g, acc_u = tl.split(tl.trans(acc.reshape(BM, 2, BN), 0, 2, 1))
            h_desc.store([om, on], _gelu_new_mul(acc_g, acc_u))

    @triton.jit
    def _gemm_nt_fwd(a_desc, b_desc, o_desc, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     NS: tl.constexpr, PDL: tl.constexpr):
        """out = a @ b^T with a [M, K] and b [N, K]."""
        if PDL:
            gdc.gdc_wait()
        num_m = tl.cdiv(M, BM)
        total = num_m * tl.cdiv(N, BN)
        for tile in tl.range(tl.program_id(0), total, tl.num_programs(0),
                             flatten=True, warp_specialize=True):
            om = (tile % num_m) * BM
            on = (tile // num_m) * BN
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in tl.range(0, K, BK, num_stages=NS):
                acc = tl.dot(a_desc.load([om, k]), b_desc.load([on, k]).T, acc)
            o_desc.store([om, on], acc.to(o_desc.dtype))

    _NUM_SMS = None

    def _num_sms() -> int:
        global _NUM_SMS
        if _NUM_SMS is None:
            _NUM_SMS = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        return _NUM_SMS

    # (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps), tuned on B200.
    _CFG_GATED = (128, 64, 64, 6, 8)
    _CFG_DOWN = (128, 128, 64, 6, 8)

    def _gated_mlp(x2d: torch.Tensor, w13: torch.Tensor, wo: torch.Tensor):
        M, K = x2d.shape
        D = w13.shape[1]

        h = torch.empty((M, D), device=x2d.device, dtype=x2d.dtype)
        BM, BN, BK, NS, NW = _CFG_GATED
        a_desc = TensorDescriptor(x2d, list(x2d.shape), list(x2d.stride()), [BM, BK])
        w_desc = TensorDescriptor(w13, list(w13.shape), list(w13.stride()), [2, BN, BK])
        h_desc = TensorDescriptor(h, list(h.shape), list(h.stride()), [BM, BN])
        ntiles = triton.cdiv(M, BM) * triton.cdiv(D, BN)
        _gated_mlp_fwd[(min(_num_sms(), ntiles),)](
            a_desc, w_desc, h_desc, M, K, D, BM, BN, BK, NS,
            num_stages=NS, num_warps=NW)

        N = wo.shape[0]
        y = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)
        BM, BN, BK, NS, NW = _CFG_DOWN
        ha_desc = TensorDescriptor(h, list(h.shape), list(h.stride()), [BM, BK])
        wo_desc = TensorDescriptor(wo, list(wo.shape), list(wo.stride()), [BN, BK])
        y_desc = TensorDescriptor(y, list(y.shape), list(y.stride()), [BM, BN])
        ntiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        extra = {"launch_pdl": True} if _HAVE_PDL else {}
        _gemm_nt_fwd[(min(_num_sms(), ntiles),)](
            ha_desc, wo_desc, y_desc, M, N, D, BM, BN, BK, NS, _HAVE_PDL,
            num_stages=NS, num_warps=NW, **extra)
        return y


def _tma_ok(t: torch.Tensor) -> bool:
    """TMA needs a 16B-aligned base and 16B-aligned row pitch."""
    return (t.is_contiguous() and t.data_ptr() % 16 == 0
            and (t.shape[-1] * t.element_size()) % 16 == 0)


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model, [config.d_ff, config.d_ff], bias=False,
        )
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)
        self._fast = bool(
            _HAVE_TRITON and config.dense_act_fn == "gelu_new"
            and not self.wi.use_fp8 and not self.wo.use_fp8
            and self.wi.bias is None and self.wo.bias is None
            and self.wo.tp_size == 1
        )

    def _fusable(self, hidden_states: torch.Tensor) -> bool:
        if not self._fast or hidden_states.dtype != torch.bfloat16:
            return False
        wi, wo = self.wi.weight, self.wo.weight
        if wi.dtype != torch.bfloat16 or wo.dtype != torch.bfloat16:
            return False
        D = wo.shape[1]
        if wi.shape[0] != 2 * D or hidden_states.shape[-1] != wi.shape[1]:
            return False
        if hidden_states.numel() == 0 or not hidden_states.is_cuda:
            return False
        return _tma_ok(hidden_states) and _tma_ok(wi) and _tma_ok(wo)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fusable(hidden_states):
            try:
                shape = hidden_states.shape
                x2d = hidden_states.reshape(-1, shape[-1])
                D = self.wo.weight.shape[1]
                y = _gated_mlp(x2d, self.wi.weight.view(2, D, shape[-1]),
                               self.wo.weight)
                return y.view(*shape[:-1], y.shape[-1])
            except Exception:  # pragma: no cover - fall back to eager
                self._fast = False
        gate_up = self.wi(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
