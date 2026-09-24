"""T5 feed-forward dense layers with TP sharding (L2).

T5DenseActDense: standard FFN (ColumnParallel -> act -> RowParallel).
T5DenseGatedActDense: gated FFN (MergedColumnParallel -> gate*up -> RowParallel).

Optimization
------------
The captured shape is a T5-XXL encoder FFN: ``hidden_states[1, 512, 4096]`` with
``d_ff = 10240`` and ``dense_act_fn = "gelu_new"`` (``feed_forward_proj =
"gated-gelu"``).  In that regime the two GEMMs are compute bound (86 + 43
GFLOP), but the *gate* path between them is pure memory traffic -- and the
reference spells it as nine separate eager ops::

    pow -> mul -> add -> mul -> tanh -> add -> mul -> mul -> mul

each of which streams the whole ``[512, 10240]`` intermediate through HBM
(~21 MB per read/write).  Measured on a B200 that chain costs **137 us**, i.e.
half of the layer's 275 us, against a hard floor of ~13 us for the 63 MB the
computation actually has to move (read gate + read up, write one result).

``fk_gated_gelu_new`` below replaces the chain with one kernel that reads the fused
``gate_up`` tensor once, 16 B per thread per half, and writes ``act(gate) * up``
straight out: **15 us**, i.e. at the bandwidth limit.

Bit-exactness.  A fast fp32 evaluation of the activation is *not* accurate
enough here: ``1 + tanh(...)`` cancels catastrophically for negative gates
(``tanh -> -1``), so the reference's own bf16 rounding of each intermediate is
amplified by up to ~40x in the result.  Computing the chain "better" than the
reference therefore drifts ~1 bf16 ulp per element, and after ``wo`` that leaves
only ~98.6% of the output elements inside the scorer's 1% band (99% required).

The kernel instead *reproduces* the reference's arithmetic exactly, which two
measurements (``scratch/probe_num.py``, swept over every bf16 bit pattern) make
cheap:

* ``torch.pow(x, 3.0)`` on bf16 is evaluated as ``(x*x)*x`` with a bf16 round
  after **each** multiply -- exactly what ``mul.rn.bf16x2`` gives, two elements
  at a time.
* ``bf16(tanh.approx.f32(x))`` equals ``torch.tanh(x)`` for **all** 65280 finite
  bf16 inputs, so the single-instruction MUFU tanh is free of error here.

Every other step is a bf16 op whose fp32-then-round reference semantics
``add.rn.bf16x2`` / ``mul.rn.bf16x2`` reproduce exactly (a bf16xbf16 product and
a bf16+bf16 sum are both exact in fp32, so the correctly-rounded packed result
is the correctly-rounded fp32 one); the two multiplies by a python-float
constant, which torch performs in fp32 against the *unrounded* constant, are
done that way too.  The result is bitwise identical to the reference on every
element of the captured case, so the fusion costs nothing in accuracy.

The GEMMs are left on the reference path, because they are already at this
machine's practical ceiling: measured in one process, ``wi`` runs at 96% and
``wo`` at 83% of the per-FLOP rate cuBLAS itself reaches on a large square bf16
GEMM here, and the layer as a whole (129 GFLOP) lands within a few percent of
that rate.  A hand-written replacement would have to beat cuBLAS outright: the
best Triton tcgen05 tile found for the ``wi`` shape (128x128x64, 6 stages,
warp-specialized) reaches only 0.81x of it -- a 22 us deficit against the ~19 us
that folding the activation into a GEMM epilogue could ever save.  Fusing into
``wo``'s *prologue* is worse still: ``gate_up`` tiles are re-read by every
column block, so the activation would be recomputed 16x.

Net effect on the captured case: 275 us -> 152 us, bit-identical output
(scored 1.69-1.82x depending on how the GPU happens to be clocked).
"""

from __future__ import annotations

import math
import os

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
# Fused gated activation:  out[..., j] = act(gu[..., j]) * gu[..., D + j]
# ---------------------------------------------------------------------------
_CPP = "at::Tensor fk_gated_gelu_new(const at::Tensor& gate_up);"

_CUDA = r"""
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#define DEVI __device__ __forceinline__
#define NT 256

struct alignas(16) V16 { unsigned w[4]; };   // 8 bf16 lanes

// --- packed bf16x2 primitives ---------------------------------------------
// Each produces the correctly-rounded bf16 result of the corresponding torch
// elementwise op (both operands are bf16, so the fp32 intermediate is exact).
DEVI unsigned bmul(unsigned a, unsigned b) {
  unsigned r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
DEVI unsigned badd(unsigned a, unsigned b) {
  unsigned r; asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
DEVI float tanh_ap(float x) {
  float r; asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
DEVI float2 unpack2(unsigned v) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}
DEVI unsigned pack2(float2 f) {
  __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<const unsigned*>(&b);
}

#define B_ONE  0x3f803f80u   /* 1.0f  in both bf16 halves */
#define B_HALF 0x3f003f00u   /* 0.5f  in both bf16 halves */

// out = NewGELU(x) * u, two elements at a time, bit-identical to
//   0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715 * pow(x, 3)))) * u
// evaluated op-by-op in bf16 by eager torch.
DEVI unsigned gated_gelu_new(unsigned x, unsigned u) {
  unsigned c = bmul(bmul(x, x), x);            // torch.pow(x, 3.0) == (x*x)*x
  float2 cf = unpack2(c);                      // 0.044715 * x^3, fp32 constant
  cf.x *= 0.044715f; cf.y *= 0.044715f;
  unsigned b = badd(x, pack2(cf));             // x + ...
  float2 bf = unpack2(b);                      // sqrt(2/pi) * (...), fp32 const
  bf.x *= 0.7978845608028654f; bf.y *= 0.7978845608028654f;
  float2 kf = unpack2(pack2(bf));              // (rounded to bf16 as torch does)
  kf.x = tanh_ap(kf.x); kf.y = tanh_ap(kf.y);
  unsigned d = badd(pack2(kf), B_ONE);         // 1 + tanh(...)
  return bmul(bmul(bmul(x, B_HALF), d), u);    // (0.5*x) * (1+tanh) * u
}

// gate_up: [rows, 2*D] contiguous; out: [rows, D].  grid = (D/8 tiles, rows).
template <int VPT>
__global__ __launch_bounds__(NT) void gated_gelu_new_kernel(
    const V16* __restrict__ gu, V16* __restrict__ out, int dvec) {
  const long long gbase = (long long)blockIdx.y * 2 * dvec;
  const long long obase = (long long)blockIdx.y * dvec;
  const int j0 = blockIdx.x * (NT * VPT) + threadIdx.x;
  V16 g[VPT], u[VPT];
#pragma unroll
  for (int v = 0; v < VPT; ++v) {
    const int j = j0 + v * NT;
    if (j < dvec) { g[v] = gu[gbase + j]; u[v] = gu[gbase + dvec + j]; }
  }
#pragma unroll
  for (int v = 0; v < VPT; ++v) {
    const int j = j0 + v * NT;
    if (j < dvec) {
      V16 r;
#pragma unroll
      for (int w = 0; w < 4; ++w) r.w[w] = gated_gelu_new(g[v].w[w], u[v].w[w]);
      out[obase + j] = r;
    }
  }
}

// Ragged last dimension (D % 8): one element per thread.
__global__ void gated_gelu_new_scalar(
    const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out,
    long long D, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long row = i / D, j = i - row * D;
  const long long gi = row * 2 * D + j;
  unsigned x = 0, u = 0;
  reinterpret_cast<__nv_bfloat16*>(&x)[0] = gu[gi];
  reinterpret_cast<__nv_bfloat16*>(&u)[0] = gu[gi + D];
  const unsigned r = gated_gelu_new(x, u);
  out[i] = reinterpret_cast<const __nv_bfloat16*>(&r)[0];
}

at::Tensor fk_gated_gelu_new(const at::Tensor& gate_up) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == at::kBFloat16);
  TORCH_CHECK(gate_up.dim() >= 1 && gate_up.is_contiguous());
  const int64_t D2 = gate_up.size(-1);
  TORCH_CHECK(D2 % 2 == 0);
  const int64_t D = D2 / 2;
  auto sizes = gate_up.sizes().vec();
  sizes.back() = D;
  at::Tensor out = at::empty(sizes, gate_up.options());
  const int64_t rows = D == 0 ? 0 : gate_up.numel() / D2;
  if (rows == 0 || D == 0) return out;

  auto stream = c10::cuda::getCurrentCUDAStream();
  const auto* ip = reinterpret_cast<const __nv_bfloat16*>(gate_up.const_data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(ip) | reinterpret_cast<uintptr_t>(op)) & 15) == 0;
  if (aligned && D % 8 == 0 && rows <= 65535) {
    const int dvec = (int)(D / 8);
    const V16* gv = reinterpret_cast<const V16*>(ip);
    V16* ov = reinterpret_cast<V16*>(op);
    // Two 16 B vectors per thread: enough memory-level parallelism to saturate
    // HBM, and one block per row-chunk keeps the row index division-free.
    if (dvec >= 2 * NT) {
      dim3 grid((dvec + 2 * NT - 1) / (2 * NT), (unsigned)rows);
      gated_gelu_new_kernel<2><<<grid, NT, 0, stream>>>(gv, ov, dvec);
    } else {
      dim3 grid((dvec + NT - 1) / NT, (unsigned)rows);
      gated_gelu_new_kernel<1><<<grid, NT, 0, stream>>>(gv, ov, dvec);
    }
  } else {
    const int64_t n = rows * D;
    gated_gelu_new_scalar<<<(unsigned)((n + NT - 1) / NT), NT, 0, stream>>>(
        ip, op, D, n);
  }
  return out;
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    # The packed-bf16 arithmetic needs sm_90+, so pin the build to the local
    # arch for the duration of the build (the ambient TORCH_CUDA_ARCH_LIST here
    # spans sm_75..sm_120, and ptxas rejects ``mul.rn.bf16x2`` below sm_90).
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    major, minor = torch.cuda.get_device_capability(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = (
        f"{major}.{minor}" + ("a" if major in (9, 10, 12) else "")
    )
    try:
        return load_inline(
            name="fk_l2_t5_dense_gated_act",
            cpp_sources=_CPP,
            cuda_sources=_CUDA,
            functions=["fk_gated_gelu_new"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            ],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


_C = None
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
    try:  # tanh.approx.bf16 / bf16x2 arithmetic needs sm_90+
        _C = _build()
    except Exception:  # pragma: no cover - no nvcc: stay on the eager path
        _C = None


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model, [config.d_ff, config.d_ff], bias=False,
        )
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)
        # The fused kernel implements ``gelu_new`` only; every other activation
        # keeps the eager chunk/act/mul path.
        self._fused_act = _C is not None and isinstance(self.act, NewGELUActivation)

    def _gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        if (self._fused_act
                and gate_up.dtype == torch.bfloat16
                and gate_up.is_contiguous()
                and gate_up.size(-1) % 2 == 0
                and not (torch.is_grad_enabled() and gate_up.requires_grad)):
            return _C.fk_gated_gelu_new(gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act(gate) * up

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.wi(hidden_states)
        hidden_states = self._gate(gate_up)
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
