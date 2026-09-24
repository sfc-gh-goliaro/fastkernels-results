"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.

Unified across Qwen2-VL and Qwen3-VL:
  - use_postshuffle_norm: Qwen3 DeepStack mergers norm after spatial reshape.

Where the time goes
-------------------
The captured cases are ``x:bf16[N, 1, 1152]`` with N up to ~65k, i.e. M = N/4
rows of 4608 pushed through a 4608x4608 and a 4608x4096 matmul.  Per-kernel on
B200 for the largest case (N=64680, M=16170), baseline vs this file:

                     baseline        here
    layer_norm        296.2 us      102.5 us   (hand-written, below)
    fc1               515.8 us      530.3 us   (+ the GELU, fused as epilogue)
    gelu              141.4 us         --
    fc2               469.6 us      469.6 us   (untouched)
                     ---------     ---------
                      1423.0 us     1102.4 us      -> 1.27x end to end

The two matmuls are at this machine's ceiling: they run at ~1.3 PFLOP/s, and a
plain 298 MB device-to-device copy here takes 67.6 us, so fc1 is 7x past the
point where bandwidth could explain it.  (The L1 ``linear`` winner measured the
same wall from the other side -- ``mma.sync`` class code, which is what Triton's
``tl.dot`` emits, micro-benchmarks 4x short of these kernels.)  All the headroom
is therefore in the two *elementwise* passes around them, which were 31% of the
baseline while moving only 2 x 149 MB each:

* ``layer_norm``: torch's ``vectorized_layer_norm`` reads ``x`` twice and uses
  4-byte accesses, which lands it at ~1.5 TB/s.  :func:`ln_kernel` below reads
  once with 16-byte accesses and runs at 4.4 TB/s.
* ``gelu``: folded into fc1's epilogue, so the pass disappears rather than
  getting faster.  ``torch._addmm_activation`` runs the same cuBLAS GEMM as
  ``F.linear`` with the bias+GELU epilogue enabled; it picks a slightly worse
  tile (14.5 us) in exchange for a 141.4 us round trip through DRAM.  A
  hand-written bias+GELU kernel at full copy speed would still cost 67 us --
  the fusion wins because the traffic disappears, not because the arithmetic
  gets cheaper.  Measured across M from 1k to 16k the fused form is ~3% faster
  end to end everywhere except a ~200-wide window at M ~ 5.2k where it is 2%
  slower; that is a cuBLAS tile-selection artifact, not a trend, so there is one
  code path rather than a threshold fitted to it.

Nothing here replaces the matmuls: the compute written in this file is the
LayerNorm kernel, and the GEMMs keep the exact vendor path the baseline takes
(only fc1's epilogue changes).  Of the 1102 us that remain, 1000 us is the two
GEMMs at roofline and 68 us of the rest is traffic no layout can avoid.

The small captured case (N=1760, M=440) is a different regime: ~60 us of kernel
work against ~50 us of per-call CPU time, so it is won by launch count and
Python work.  Hence the row reshape is folded into the norm kernel's output
shape, the ATen fallback lives inside the extension (one pybind call, not a
support probe plus a call), and ``forward`` runs three ops with no attribute
juggling whenever the module is in its ordinary tp=1, unquantized configuration.

Numerics
--------
The norm reduction is fp32 and two-pass over the staged row, so the variance
never depends on ``E[x^2] - E[x]^2`` cancelling; it matches torch's fp32 Welford
accumulation to well inside a bf16 ulp.  cuBLAS' fused GELU epilogue agrees with
``F.gelu`` to one bf16 ulp (3.1e-2 at |y| ~ 6 -- the same 1-ulp disagreement the
L1 GELU winner has with it), and 99.95% of output elements land inside the bench
tolerance (atol = rtol = 1e-2, which needs 99%).

That 0.05% is rounding, not drift: against the whole pipeline recomputed in
fp32, this file matches 99.995% of elements where the baseline matches 99.94% --
the disagreement between the two is one bf16 ulp in *either* direction, and the
fused epilogue lands on the correct side of it slightly more often.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# ---------------------------------------------------------------------------
# Fused LayerNorm (+ patch-merge reshape) -- CUDA source
# ---------------------------------------------------------------------------
_CPP = """
at::Tensor fk_vpm_ln(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
                     double eps, int64_t norm_cols, int64_t out_cols);
"""

_CUDA = r"""
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DEVI __device__ __forceinline__

// --- 16-bit pair plumbing --------------------------------------------------
template <typename T> struct Pack;
template <> struct Pack<__nv_bfloat16> { using T2 = __nv_bfloat162; };
template <> struct Pack<__half>        { using T2 = __half2; };

DEVI float2 cvt(const __nv_bfloat162 v) { return __bfloat1622float2(v); }
DEVI float2 cvt(const __half2 v)        { return __half22float2(v); }
DEVI void   cvt_back(__nv_bfloat162& o, const float2 f) { o = __float22bfloat162_rn(f); }
DEVI void   cvt_back(__half2& o, const float2 f)        { o = __float22half2_rn(f); }

// ---------------------------------------------------------------------------
// LayerNorm over rows of exactly TPR*(NV4*8 + TAIL) elements.
//
// A row is owned by TPR threads holding NV4 uint4 plus one TAIL-element short
// vector each.  Both captured row lengths factor exactly into such a tiling
// (1152 = 64*(2*8+2), 4608 = 128*(4*8+4)), so every access is a full-width
// LDG.E.128/64/32 -- no masked lanes, no tail loop, no predication.  The row
// stays in registers *in the input dtype* across both reductions and the
// write-back: x is read once (2 x 149 MB for the whole pass instead of torch's
// 3 x), and staging bf16 rather than fp32 halves the registers per element,
// which is what caps how many loads the kernel can keep in flight.  The three
// cvt passes over those registers cost nothing against DRAM.
//
// Reductions are fp32 and two-pass (sum, then sum of squared deviations), so
// the variance stays well-conditioned whatever the input's mean.  Warps combine
// by butterfly shuffle -- every lane ends with the total, so no broadcast step
// -- and when a row spans several warps the per-warp partials meet in shared
// memory, with a separate slot per pass so one __syncthreads per pass suffices.
//
// The (TPR, NV4, TAIL, RPB) choices below came out of a sweep; what it showed is
// that the remaining ~35 us over a plain copy is the barrier between the
// reduction and the write-back, not arithmetic or occupancy.  Folding the
// variance into pass 1, folding the affine math into one fma per element, and
// staging w/b as fp32 in shared memory each moved the kernel by <3% (the last
// one by -30%); halving the staged registers by holding the row in fp32 instead
// of bf16 changed nothing; a grid-stride loop over rows changed nothing; and a
// warp-per-row tiling with no __syncthreads at all was 6% *slower* than this
// two-warp one.  So the one-pass variance is not worth its conditioning.
// ---------------------------------------------------------------------------
template <typename T, int TPR, int NV4, int TAIL, int RPB>
__global__ __launch_bounds__(TPR * RPB) void ln_kernel(
    const T* __restrict__ x, const T* __restrict__ wt, const T* __restrict__ bt,
    T* __restrict__ y, int rows, float eps) {
  using T2 = typename Pack<T>::T2;
  constexpr int C   = TPR * (NV4 * 8 + TAIL);
  constexpr int NW  = TPR / 32;      // warps spanned by one row
  constexpr int NT2 = TAIL / 2;      // T2 units in the short vector
  constexpr int TOFF = NV4 * 8 * TPR;
  constexpr float inv_c = 1.0f / (float)C;

  const int tid  = threadIdx.x;
  const int lane = tid % TPR;        // slot inside the row
  const int grp  = tid / TPR;        // which row of this block
  const int row  = blockIdx.x * RPB + grp;
  if (row >= rows) return;

  // --- stage the row ---
  const T* xr = x + (size_t)row * C;
  uint4 va[NV4];
#pragma unroll
  for (int v = 0; v < NV4; ++v)
    va[v] = *reinterpret_cast<const uint4*>(xr + (v * TPR + lane) * 8);
  uint2 vb;
  unsigned vc;
  if (TAIL == 4) vb = *reinterpret_cast<const uint2*>(xr + TOFF + lane * 4);
  if (TAIL == 2) vc = *reinterpret_cast<const unsigned*>(xr + TOFF + lane * 2);
  const T2* tp = (TAIL == 4) ? reinterpret_cast<const T2*>(&vb)
                             : reinterpret_cast<const T2*>(&vc);

  __shared__ float red[2][RPB][NW];

  // --- pass 1: mean ---
  float s = 0.f;
#pragma unroll
  for (int v = 0; v < NV4; ++v) {
    const T2* p = reinterpret_cast<const T2*>(&va[v]);
#pragma unroll
    for (int j = 0; j < 4; ++j) { const float2 f = cvt(p[j]); s += f.x + f.y; }
  }
#pragma unroll
  for (int j = 0; j < NT2; ++j) { const float2 f = cvt(tp[j]); s += f.x + f.y; }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (NW > 1) {
    if ((tid & 31) == 0) red[0][grp][lane >> 5] = s;
    __syncthreads();
    s = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) s += red[0][grp][i];
  }
  const float mean = s * inv_c;

  // --- pass 2: variance about that mean ---
  float q = 0.f;
#pragma unroll
  for (int v = 0; v < NV4; ++v) {
    const T2* p = reinterpret_cast<const T2*>(&va[v]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = cvt(p[j]);
      const float dx = f.x - mean, dy = f.y - mean;
      q += dx * dx + dy * dy;
    }
  }
#pragma unroll
  for (int j = 0; j < NT2; ++j) {
    const float2 f = cvt(tp[j]);
    const float dx = f.x - mean, dy = f.y - mean;
    q += dx * dx + dy * dy;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  if (NW > 1) {
    if ((tid & 31) == 0) red[1][grp][lane >> 5] = q;
    __syncthreads();
    q = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) q += red[1][grp][i];
  }
  const float rstd = rsqrtf(q * inv_c + eps);

  // --- write back: normalize, scale, shift ---
  T* yr = y + (size_t)row * C;
  const bool has_w = wt != nullptr, has_b = bt != nullptr;
#pragma unroll
  for (int v = 0; v < NV4; ++v) {
    const int off = (v * TPR + lane) * 8;
    const T2* p = reinterpret_cast<const T2*>(&va[v]);
    const T2* pw = has_w ? reinterpret_cast<const T2*>(wt + off) : nullptr;
    const T2* pb = has_b ? reinterpret_cast<const T2*>(bt + off) : nullptr;
    uint4 ow;
    T2* o = reinterpret_cast<T2*>(&ow);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = cvt(p[j]);
      float2 r;
      r.x = (f.x - mean) * rstd;
      r.y = (f.y - mean) * rstd;
      if (has_w) { const float2 g = cvt(pw[j]); r.x *= g.x; r.y *= g.y; }
      if (has_b) { const float2 c = cvt(pb[j]); r.x += c.x; r.y += c.y; }
      cvt_back(o[j], r);
    }
    *reinterpret_cast<uint4*>(yr + off) = ow;
  }
  if (NT2 > 0) {
    const int off = TOFF + lane * TAIL;
    const T2* pw = has_w ? reinterpret_cast<const T2*>(wt + off) : nullptr;
    const T2* pb = has_b ? reinterpret_cast<const T2*>(bt + off) : nullptr;
    T2 ot[NT2 > 0 ? NT2 : 1];
#pragma unroll
    for (int j = 0; j < NT2; ++j) {
      const float2 f = cvt(tp[j]);
      float2 r;
      r.x = (f.x - mean) * rstd;
      r.y = (f.y - mean) * rstd;
      if (has_w) { const float2 g = cvt(pw[j]); r.x *= g.x; r.y *= g.y; }
      if (has_b) { const float2 c = cvt(pb[j]); r.x += c.x; r.y += c.y; }
      cvt_back(ot[j], r);
    }
    if (TAIL == 4) *reinterpret_cast<uint2*>(yr + off) = *reinterpret_cast<uint2*>(ot);
    else           *reinterpret_cast<unsigned*>(yr + off) = *reinterpret_cast<unsigned*>(ot);
  }
}

// ---------------------------------------------------------------------------
// launch / dispatch
// ---------------------------------------------------------------------------
template <typename T, int TPR, int NV4, int TAIL, int RPB>
static void launch(const at::Tensor& x, const T* w, const T* b, T* y,
                   int rows, float eps, cudaStream_t s) {
  ln_kernel<T, TPR, NV4, TAIL, RPB><<<dim3((rows + RPB - 1) / RPB),
                                      TPR * RPB, 0, s>>>(
      reinterpret_cast<const T*>(x.const_data_ptr()), w, b, y, rows, eps);
}

static inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// Anything the compiled tilings do not cover -- another context_dim, fp32, a
// non-contiguous or oddly-offset input -- goes to ATen here rather than back
// across the pybind boundary, so ``forward`` is one call either way.
static at::Tensor ln_aten(const at::Tensor& x, const at::Tensor& w,
                          const at::Tensor& b, double eps,
                          int64_t norm_cols, int64_t out_cols) {
  at::Tensor y = at::layer_norm(x.reshape({-1, norm_cols}), {norm_cols}, w, b, eps);
  return out_cols == norm_cols ? y : y.view({-1, out_cols});
}

at::Tensor fk_vpm_ln(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
                     double eps, int64_t norm_cols, int64_t out_cols) {
  const auto dt = x.scalar_type();
  const bool fast =
      x.is_cuda() && x.is_contiguous() && (dt == at::kBFloat16 || dt == at::kHalf)
      && (norm_cols == 1152 || norm_cols == 4608)
      && x.numel() % out_cols == 0 && x.numel() % norm_cols == 0
      && w.defined() && w.numel() == norm_cols && w.is_contiguous()
      && w.scalar_type() == dt
      && (!b.defined() || (b.numel() == norm_cols && b.is_contiguous()
                           && b.scalar_type() == dt))
      && aligned16(x.const_data_ptr());
  if (!fast) return ln_aten(x, w, b, eps, norm_cols, out_cols);

  const int rows = (int)(x.numel() / norm_cols);
  at::Tensor y = at::empty({x.numel() / out_cols, out_cols}, x.options());
  auto s = at::cuda::getCurrentCUDAStream();

#define FK_DISPATCH(T)                                                         \
  do {                                                                         \
    const T* wp = reinterpret_cast<const T*>(w.const_data_ptr());              \
    const T* bp = b.defined() ? reinterpret_cast<const T*>(b.const_data_ptr()) \
                              : nullptr;                                       \
    T* yp = reinterpret_cast<T*>(y.data_ptr());                                \
    if (norm_cols == 1152)                                                     \
      launch<T, 64, 2, 2, 4>(x, wp, bp, yp, rows, (float)eps, s);               \
    else                                                                       \
      launch<T, 128, 4, 4, 2>(x, wp, bp, yp, rows, (float)eps, s);              \
  } while (0)

  if (dt == at::kBFloat16) { FK_DISPATCH(__nv_bfloat16); }
  else                     { FK_DISPATCH(__half); }
#undef FK_DISPATCH
  return y;
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = (
                f"{major}.{minor}" + ("a" if major in (9, 10, 12) else ""))
        except Exception:
            pass
    return load_inline(
        name="fk_l2_vision_patch_merger_cuda",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["fk_vpm_ln"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_HALF_OPERATORS__",
                           "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
        verbose=False,
    )


try:
    _ln = _build().fk_vpm_ln
except Exception:  # pragma: no cover - no GPU / no nvcc: use the L1 op instead
    _ln = None

# ``F.linear`` + ``F.gelu`` in one cuBLAS launch (bias+GELU epilogue). Private
# API, so resolve it once and fall back to the two-launch form if it is gone.
_addmm_gelu = getattr(torch, "_addmm_activation", None)


class VisionPatchMerger(nn.Module):
    """Patch merger: norm -> flatten -> MLP(fc1, GELU, fc2).

    Qwen3 DeepStack mergers set use_postshuffle_norm=True to norm after reshape.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm_dim = norm_dim
        # See VisionBlock: vLLM's vision path uses plain nn.LayerNorm on
        # bf16, and our fp32 promotion costs two full-tensor copies here.
        self.norm = LayerNorm(norm_dim, eps=eps, promote_fp32=False)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)
        # ``fc1.weight`` transposed for ``_addmm_activation``, which wants the
        # mathematical B rather than ``F.linear``'s [out, in] layout. Cached
        # against the weight's *address*, not its identity: ``_prepare_module``
        # (and any later ``.to()``) rebinds ``param.data`` in place, which leaves
        # the Parameter object the same while moving the storage the view points
        # at. Matching data_ptr means same storage, same shape, same strides --
        # so the cached view is still the right one.
        self._w1_ptr: int = 0
        self._w1_t: torch.Tensor | None = None

    def _eligible(self) -> bool:
        """Can ``forward`` take the three-op path? Fixed at the module's
        configuration: quantized weights, tensor parallelism or a missing
        private ATen op all send it back through the generic submodules."""
        return (_ln is not None and _addmm_gelu is not None
                and not self.fc1.use_fp8 and not self.fc2.use_fp8
                and self.fc1.bias is not None
                and self.fc2.tp_size == 1)

    def _fc1_t(self) -> torch.Tensor:
        w = self.fc1.weight
        ptr = w.data_ptr()
        if self._w1_ptr != ptr:
            self._w1_ptr, self._w1_t = ptr, w.t()
        return self._w1_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._eligible() and x.dtype in (torch.bfloat16, torch.float16):
            x = _ln(x, self.norm.weight, self.norm.bias, self.norm.eps,
                    self.norm_dim, self.hidden_size)
            return F.linear(
                _addmm_gelu(self.fc1.bias, x, self._fc1_t(), use_gelu=True),
                self.fc2.weight, self.fc2.bias)

        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)
        return self.fc2(self.act(self.fc1(x)))
