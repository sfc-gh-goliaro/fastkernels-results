"""T5 feed-forward dense layers with TP sharding (L2).

T5DenseActDense: standard FFN (ColumnParallel -> act -> RowParallel).
T5DenseGatedActDense: gated FFN (MergedColumnParallel -> gate*up -> RowParallel).

The gated variant's middle is where the eager reference bleeds time. After the
merged ``wi`` projection the reference does ``gate_up.chunk(2, -1)`` and then
evaluates ``0.5*g*(1 + tanh(sqrt(2/pi)*(g + 0.044715*g^3))) * u`` as separate
torch ops. On the captured shape ([1, 512, 4096], d_model 4096, d_ff 10240) that
is ~7 elementwise launches, each streaming a *strided* 10.5 MB half of the
21 MB ``gate_up`` buffer and writing another 10.5 MB back -- measured 93 us,
against 69 us for the ``wi`` GEMM and 46 us for ``wo``. Almost half the operator
is the activation.

``_act_mul`` replaces the whole chain with one Triton kernel: each program
takes a BLOCK-wide column slice of one row, loads the gate and up halves of that
slice (one pass over ``gate_up``), evaluates the activation in fp32, and stores
BLOCK results. 31.5 MB moved instead of ~150 MB, one launch instead of seven.

* The transcendental is one instruction. ``tanh`` is the hardware
  ``tanh.approx.f32`` (1 SFU op, ~2^-22 relative) as in ``L1/gelu.py``; plain
  ``gelu`` reuses that file's least-squares quintic and SiLU the
  ``0.5*v*(1 + tanh(0.5*v))`` form from ``L1/silu.py``.
* ``gelu_new`` reproduces the reference **bit-exactly** rather than accurately.
  That is not fussiness: the reference evaluates its polynomial as ~8 separate
  bf16 ops, and ``torch.pow(bf16, 3.0)`` is *repeated squaring in bf16*
  (measured: ``pow(g,3)`` equals ``rn(rn(g*g)*g)`` on 100% of elements and the
  fp32 cube on only 78%), so the reference's own ``gate`` term carries ~0.4%
  relative error. A single-rounding fp32 chain is much closer to exact and
  therefore *further* from the reference: propagated through ``wo``'s K=10240
  reduction it left only a 3-sigma margin (matched ratio 0.9968 against the
  harness's 0.99 floor). Rounding to bf16 after every op the reference performs
  costs nothing measurable (11.44 us vs 11.30) and takes the matched ratio to
  1.000000, which also means later changes get the whole tolerance budget.
* ``launch_pdl`` + ``gdc_wait()``: this kernel always runs downstream of the
  ``wi`` GEMM, so the grid is staged while that GEMM's tail drains.

The fast path is gated on fp16/bf16; fp32 tolerances are 1e-5/1e-3 and no
captured shape reaches it, so fp32 takes the eager path.

Both projections then keep cuBLAS, but hand it a K-contiguous ``[in, out]``
weight instead of ``F.linear``'s ``[out, in]``. Profiling both says nvjet keeps
the *same* kernel family and tile either way -- ``192x128_64x6_2x1_2cta_v_bz``
for ``wi``, ``128x128_64x8_2x2_2cta_h_bz`` for ``wo`` -- and only the operand
layout suffix changes, ``TNT`` -> ``NNT``. So this is not a tile-selection or
wave-quantization effect (``wi``'s 3 M-tiles of 192 still cover 576 rows for
M=512, 11% padding, in both); the NNT variant is simply cheaper for the same
tiling, presumably in the B-operand SMEM path. Measured standalone: wi
68.5 -> 64.5 us, wo 44.1 -> 42.1 us, and 104.5 -> 100.3 us for the three stages
back to back. ``preferred_blas_library`` (cublas vs cublaslt) makes no
difference on either shape, and nothing reachable from torch lets us ask for a
128-row M-tile to recover that 11%.

The transposed copies are built on first use and cached, which costs one extra
copy of each weight -- for this config 240 MiB per layer. That is the deliberate
trade: 4% of the operator for 2x the FFN weight footprint. Nothing here mutates
the parameters, so ``state_dict`` still round-trips. Invalidation needs two
mechanisms, because the ``data_ptr``/``_version`` key alone is *not* enough:
``parallel_linear``'s TP loaders write through ``param.data.copy_``, and ``.data``
returns a view with its own version counter, so the Parameter's never moves.
``_guard_weight_loader`` therefore wraps each loader to drop the cache.

Not pursued, with measurements (see ITERATIONS.md):
* A custom GEMM, fused or not. This shape is L2-bandwidth bound, not
  MMA bound -- ``wi`` moves ~1.1 GB L2->SMEM at 192x128 tiles in 64.5 us, i.e.
  ~17.7 TB/s -- and cuBLAS already sits at 1332 TFLOPS, which is the *maximum
  this shape family reaches at any M* (M=2048 with the same N,K gets 1308).
  A tuned Triton TMA/tcgen05 GEMM measured 693 TFLOPS at best over 72 configs,
  1.9x off, so fusing the epilogue into it would cost ~60 us to save 5.9. L1's
  ``linear.py`` recorded the same bf16-vs-nvjet gap.
* Split-K on ``wo`` through a batched GEMM: the partials do run faster
  (40.0 vs 46.1 us at split 2/4/8) but no reduction is cheap enough to keep it
  -- ``.sum(0)`` alone costs 10 us.
* A persistent ``(gate_up, h)`` workspace instead of fresh allocations: 0.1 us
  on the harness, i.e. nothing, so the buffers are not worth 31.5 MB held per
  layer.
* Wider/narrower ``act_mul`` tiles, ILP > 1, and reverse tile order, all
  measured inside the real three-stage chain (isolated timing of this kernel is
  launch-latency bound and reads 11 us for 5.9 us of work, so it cannot rank
  configs). BLOCK=2048 / 4 warps / no ILP is the best of 30; ILP 2 and 4 cost
  ~2 us, reverse order and 8 warps are neutral.

Where the time goes now, in situ (L2 flushed once, stages added one at a time):
``wi`` 64.5 us, ``act_mul`` 5.9, ``wo`` 29.7 (its 42 us standalone drops because
``h`` is still L2-resident) -- 100.3 us against 104.5 measured by the harness,
the difference being the harness's own 4 MB input copy inside the timed window.
Both GEMMs are at or above the best TFLOPS this shape family reaches at any M,
so what is left is not tuning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import T5Config
from triton.language.extra.cuda import gdc_wait

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
# Fused activation-and-multiply
# ---------------------------------------------------------------------------
# sqrt(2/pi), the reference's own tanh-argument scale.
_T1F = tl.constexpr(0.7978845608028654)
# Exact-GELU quintic from L1/gelu.py: y/2 fitted to logit(Phi(x)) so it feeds
# tanh directly. Worst |error| vs x*Phi(x) over [-8, 8] is 3.0e-5.
_A1 = tl.constexpr(0.79745782)
_A3 = tl.constexpr(0.037051035)
_A5 = tl.constexpr(-0.000358865)
# The quintic turns over at |x| ~ 11.02; capping x^2 keeps the tanh argument
# large and positive in the tail so gelu(x) -> x still holds.
_UCAP = tl.constexpr(64.0)

_ACT_IDS = {"relu": 0, "gelu": 1, "gelu_new": 2, "silu": 3}
_FAST_DTYPES = (torch.float16, torch.bfloat16)
# 2048/(32*4) = 16 elements per thread, the measured bandwidth plateau in
# L1/gelu.py. Insensitive here (512..4096 all land within 10%), so the widest
# block that still divides d_ff wins on launch count.
_BLOCKS = (2048, 1024, 512, 256, 128, 64, 32)
_WARPS = 4


@triton.jit
def _tanh_approx(x):
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rn(x, DT: tl.constexpr):
    """Round an fp32 value to the storage dtype and back, as a torch op would."""
    return x.to(DT).to(tl.float32)


@triton.jit
def _activate(x, ACT: tl.constexpr, DT: tl.constexpr):
    if ACT == 0:
        y = tl.maximum(x, 0.0)
    elif ACT == 1:
        # L1/gelu.py's fitted quintic, matching the GELU module the reference
        # calls for dense_act_fn="gelu".
        u = tl.minimum(x * x, _UCAP)
        h = 0.5 * x
        y = h * _tanh_approx(x * ((_A5 * u + _A3) * u + _A1)) + h
    elif ACT == 2:
        # NewGELUActivation, op for op, with the reference's rounding after each:
        #   pow(x, 3) -> rn(rn(x*x)*x)   (torch's integer-exponent path)
        #   then * 0.044715, + x, * sqrt(2/pi), tanh, + 1, * (0.5*x)
        # Every scalar stays fp32 -- torch does not narrow wrapped scalars --
        # and 0.5*x is exact, so only the seven roundings below are observable.
        c = _rn(_rn(x * x, DT) * x, DT)
        t = _rn(0.044715 * c, DT)
        t = _rn(x + t, DT)
        t = _rn(_T1F * t, DT)
        t = _rn(_tanh_approx(t), DT)
        t = _rn(1.0 + t, DT)
        y = _rn(0.5 * x, DT) * t
    else:
        h = 0.5 * x
        y = h * _tanh_approx(h) + h
    return y


@triton.jit
def _act_mul_kernel(GU, OUT, D, SGU, NBLK: tl.constexpr, ACT: tl.constexpr,
                    BLOCK: tl.constexpr):
    """out[r, j] = act(gate_up[r, j]) * gate_up[r, D + j], one pass over GU.

    ``NBLK`` is a constexpr so ``pid // NBLK`` is a compile-time reciprocal, and
    the flat pid keeps consecutive CTAs on consecutive columns of the same row
    (contiguous addresses) rather than striding a row apart.
    """
    pid = tl.program_id(0)
    row = (pid // NBLK).to(tl.int64)
    off = (pid % NBLK) * BLOCK + tl.arange(0, BLOCK)
    p = GU + row * SGU + off
    # The wi GEMM may still be draining; wait before reading its output.
    gdc_wait()
    g = tl.load(p).to(tl.float32)
    ub = tl.load(p + D)
    # Round the activation to the storage dtype before the gate multiply: the
    # reference's last activation op does, so its `* up` sees this operand too.
    a = _rn(_activate(g, ACT, ub.dtype), ub.dtype)
    tl.store(OUT + row * D + off, (a * ub.to(tl.float32)).to(ub.dtype))


def _act_block(d: int) -> int:
    """Widest tile that divides the half-width exactly; 0 if none does."""
    return next((b for b in _BLOCKS if d % b == 0), 0)


def _launch_act_mul(flat: torch.Tensor, out: torch.Tensor, d: int, block: int,
                    act_id: int) -> None:
    nblk = d // block
    _act_mul_kernel[(flat.shape[0] * nblk,)](flat, out, d, 2 * d, nblk, act_id,
                                             block, num_warps=_WARPS,
                                             launch_pdl=True)


def _act_mul(gate_up: torch.Tensor, act_id: int) -> torch.Tensor | None:
    """One-pass act(gate)*up. ``None`` when the fast path does not apply."""
    if act_id < 0 or gate_up.dtype not in _FAST_DTYPES or not gate_up.is_cuda:
        return None
    if not gate_up.is_contiguous():
        return None
    d = gate_up.shape[-1] // 2
    block = _act_block(d)
    if block == 0:
        return None
    flat = gate_up.reshape(-1, 2 * d)
    out = torch.empty((flat.shape[0], d), dtype=gate_up.dtype,
                      device=gate_up.device)
    if flat.shape[0]:
        _launch_act_mul(flat, out, d, block, act_id)
    return out.view(gate_up.shape[:-1] + (d,))


def _k_contig(owner, weight: torch.Tensor, slot: str) -> torch.Tensor:
    """A cached ``[in, out]`` copy of a ``[out, in]`` weight.

    Keyed on storage identity, dtype and version counter. That covers a
    ``.data`` reassignment (new pointer -- what a dtype cast does) and
    ``load_state_dict`` (which copies the Parameter itself, bumping its version),
    but *not* a write through ``param.data.copy_``: ``.data`` hands out a view
    with its own version counter, so the Parameter's is untouched. That is
    precisely how ``parallel_linear``'s TP loaders write, hence
    ``_guard_weight_loader`` below.
    """
    key = (weight.data_ptr(), weight._version, weight.shape, weight.dtype)
    cached = getattr(owner, slot)
    if cached is not None and cached[0] == key:
        return cached[1]
    t = weight.t().contiguous()
    setattr(owner, slot, (key, t))
    return t


def _guard_weight_loader(owner, linear: nn.Module, slot: str) -> None:
    """Drop ``owner.<slot>`` whenever *linear*'s weight loader runs.

    The loaders in ``parallel_linear`` write through ``param.data.narrow(...)
    .copy_(...)``, which no version counter observes, so a weight loaded after
    the first forward would otherwise be served from a stale transpose. Wrapping
    the loader is the only place that mutation is visible. The attribute lives on
    the Parameter and survives the ``p.data = p.data.to(dtype)`` that module
    preparation does.
    """
    weight = getattr(linear, "weight", None)
    inner = getattr(weight, "weight_loader", None)
    if inner is None:
        return

    def loader(*args, **kwargs):
        setattr(owner, slot, None)
        return inner(*args, **kwargs)

    weight.weight_loader = loader


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model, [config.d_ff, config.d_ff], bias=False,
        )
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)
        self._act_id = _ACT_IDS.get(config.dense_act_fn, -1)
        self._wi_kc = None
        self._wo_kc = None
        _guard_weight_loader(self, self.wi, "_wi_kc")
        _guard_weight_loader(self, self.wo, "_wo_kc")
        # Driving cuBLAS directly means bypassing both wrappers, so the fused
        # path needs the plain case on each: no fp8 quantization, no bias, and
        # no all-reduce to fold in after ``wo``.
        self._fused = (
            self._act_id >= 0
            and not self.wi.use_fp8 and self.wi.bias is None
            and not self.wo.use_fp8 and self.wo.bias is None
            and not (self.wo.reduce_results and self.wo.tp_size > 1)
        )

    def _forward_fused(self, hidden_states: torch.Tensor):
        wi_w, wo_w = self.wi.weight, self.wo.weight
        two_d, k = wi_w.shape
        d = two_d // 2
        if two_d % 2 or wo_w.shape[1] != d or hidden_states.shape[-1] != k:
            return None
        block = _act_block(d)
        if block == 0:
            return None
        x = hidden_states.reshape(-1, k)
        if not x.is_contiguous():
            x = x.contiguous()
        rows = x.shape[0]
        if rows == 0:
            return hidden_states.new_empty(hidden_states.shape[:-1]
                                           + (wo_w.shape[0],))
        gate_up = torch.empty((rows, two_d), dtype=x.dtype, device=x.device)
        torch.mm(x, _k_contig(self, wi_w, "_wi_kc"), out=gate_up)
        h = torch.empty((rows, d), dtype=x.dtype, device=x.device)
        _launch_act_mul(gate_up, h, d, block, self._act_id)
        out = torch.mm(h, _k_contig(self, wo_w, "_wo_kc"))
        return out.view(hidden_states.shape[:-1] + (out.shape[-1],))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (self._fused and hidden_states.is_cuda
                and hidden_states.dtype in _FAST_DTYPES):
            out = self._forward_fused(hidden_states)
            if out is not None:
                return out
        gate_up = self.wi(hidden_states)
        hidden_states = _act_mul(gate_up, self._act_id)
        if hidden_states is None:
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
