"""Oasis VAE attention block -- one CUDA-graph replay per captured shape.

The five children (two LayerNorms, the attention, the MLP, and the two
residual adds the block writes itself) are each already tuned, and composing
them is what is left to pay for.  Measured on B200 at the captured shapes,
the composition spends:

* **B=6 (M=3456).**  102 us of real compute (four GEMMs 66.5 us, cuDNN flash
  19.7 us, fused qkv/RoPE 6.4 us, GELU 9.6 us) inside a 140 us window.  The
  38 us of overhead is 20 us of glue kernels -- two LayerNorm passes (11.2 us)
  and two residual adds (9.0 us), each a full read+write over a 7 MB
  activation -- plus ~17 us of inter-kernel gap, because nine host launches
  cannot be issued faster than the device drains them.
* **B=1 (M=576).**  45 us of compute inside a 116 us window.  Here the glue is
  *larger than the compute*: 41 us across eleven elementwise launches, because
  the captured B=1 input is a permuted view (stride ``(589824, 1, 576)``), so
  ``norm1`` misses L1's fused row kernel and takes its reference path -- an
  fp32 copy, ATen's ``vectorized_layer_norm``, and a cast back -- and the
  first residual add then *propagates* that layout, so ``norm2`` pays the same
  three kernels again.  The remaining 30 us is gap.

So this level does four things.

**One kernel for each seam.**  ``ln_k`` reads one or two 16-bit operands
through arbitrary strides and writes both the elementwise result and its
LayerNorm.  It serves the block's two seams:

* the entry (``Bs == nullptr``): one read of the caller's ``x`` produces
  ``norm1(x)`` *and* a contiguous static copy of ``x`` for the residual.  The
  strided load path is what removes all six of B=1's cast/norm kernels -- a
  permuted input costs this kernel nothing but uncoalesced sectors that L2
  absorbs (the whole tensor is 1.2 MB and every sector is read once).
* the middle (``Bs != nullptr``): ``h = x + attn`` and ``norm2(h)`` in one
  pass, so the add never spends its own read+write over the activation.

**One graph replay for the rest.**  ``qkv -> fused qkv-split/axial-RoPE ->
attention -> proj -> (x + attn)/norm2 -> fc1 -> GELU -> fc2`` is captured as a
single graph, so a call is three launches (the entry kernel, the replay, the
final add) instead of nine, and the inter-kernel gap falls from 17 us to ~1-5 us
at B=6 and from 30 us to ~8-10 us at B=1 -- what is left there is the launch
latency of the eight graph nodes themselves, most of which are cuBLAS/cuDNN and
so cannot be given a programmatic dependency.  The three *glue* kernels this
file owns are launched with PDL (``cudaGridDependencySynchronize`` against the
producer they read), which is worth 2 us at B=6; ``_attn_k`` is not, because
measured over interleaved processes it is worth 0.03 us there -- the producer it
would stage against is the 2.3 us RoPE kernel, which does not trigger
programmatic completion.  Two properties of the
frozen children make the capture straightforward:
the L2 attention checks ``is_current_stream_capturing`` and hands back its
eager body while an outer capture is in progress, so nothing tries to replay a
nested graph, and every buffer inside the block is private, so the
handout/refcount dance L2 needs for its own returned tensor is unnecessary
here -- the only tensor that escapes is the block's output.

Graphing the compute chain is measured, not assumed: replaying it costs
-5.1 us at M=3456 and -1.0 us at M=576 against the same eager sequence, i.e.
the cuBLAS/cuDNN kernels do *not* run slower from a graph at these shapes
(the L2 MLP's round-2 note that they do was measured for a 3-node graph that
also had to copy its input in; there is no copy here, and eight nodes amortize
the replay).

**One flash kernel for the attention, at the shape cuDNN under-fills.**
``F.scaled_dot_product_attention`` resolves to cuDNN's sm100 flash kernel here,
whose Q tile is 128 rows: ``B*H*ceil(576/128)`` is 480 CTAs at B=6 (fine) but
**80 CTAs on a 148-SM machine at B=1**, which is why B=1 attention costs
10.8 us for 1.36 GFLOP (0.125 PFLOP/s) against B=6's 19.9 us for 8.15 GFLOP.
``_attn_k`` is a Triton flash forward with a 64-row Q tile, so the grid is 144
CTAs at B=1 -- one clean wave -- and it measures 8.9 us against cuDNN's 10.8.
At B=6 the same kernel *loses* (23.3 against 20.1), so it ships only where its
grid fits in one wave, which is exactly the shape cuDNN leaves on the table:
60.4 us against 62.5 at B=1 (6/6 and 8/8 fresh processes), and +4.0 us at B=6 if
forced on.  Only the SDPA call is replaced -- ``qkv`` and ``proj`` stay the
frozen child's ``Linear``s and the RoPE stays the frozen child's kernel on the
frozen child's cos/sin table, so the rotation is still bit-exact against the
reference and the only new error is the tiling of the softmax (2.4e-04 on the
attention output; the block's ``max_abs_error`` does not move off 3.91e-03).

**K-major ``qkv``/``proj``.**  ``F.linear`` hands cuBLAS ``w.t()``, which for a
row-major ``[N, K]`` weight is non-contiguous, and it answers with a ``..._TNT``
nvjet kernel; contiguous, it picks ``..._NNT``.  The frozen L2 MLP already
rewrites its own ``fc1``/``fc2`` storage K-major for this, but the frozen L2
attention does not, so ``qkv`` and ``proj`` still ran ``TNT``.  ``_kmajor``
below rewrites those two parameters' own storage in place before capture -- same
``Parameter``, same shape, dtype and values, only the strides move, so there is
no second copy to invalidate and a later weight update cannot be missed.  Worth
1.2 us of device time at B=6 and 1.0 at B=1, which at B=6 is a whole dispatch
slot of measured time: 126.0 us against 128.0 (5/6 fresh processes, interleaved
against the same-process alternative).

Otherwise the GEMMs are untouched: the same four ``addmm`` calls with the same
arguments the children issue today, on cuBLASLt's default picks -- re-checked
here for ``qkv`` and ``proj`` specifically, at both M and in both layouts,
against every candidate the heuristic offers (no candidate is more than 0.28 us
from the default; see ITERATIONS.md).  Numerics are otherwise the children's --
fp32 reductions and fp32 affine in both norms, fp32 softmax and fp32 PV
accumulation in the attention, L1's fitted GELU quintic, one round to fp16 per
store -- so the benchmark's ``max_abs_error`` is unchanged at 3.91e-03 (one fp16
ULP at the output magnitude, entirely from the frozen GELU).

The output is a fresh allocation every call: the final ``h + mlp`` runs
eagerly after the replay, which is both one kernel cheaper than graphing the
add and then copying the static result out, and the reason no caller can ever
be handed an alias the next replay would overwrite.  The entry kernel is eager
for the mirror-image reason -- it is the only op that reads the caller's
pointer.  Feeding those two pointers *into* the graph instead (a captured
16-byte H2D memcpy node from pinned memory) would collapse the call to a single
launch, and it is unsound: the host is free to run many iterations ahead of the
device, so it would overwrite call N+1's pointers before call N's memcpy node
had executed.  Three launches is the floor for a race-free hand-off.

Net, per call against the composed frozen children (``scripts/bench.sh``):
132.2 -> 126.0 us at B=6 and 119.5 -> 62.6 us at B=1.

Everything off the captured path defers to the ordinary composition of the
frozen children -- including, for the attention, any shape whose grid does not
fit in one wave, which keeps cuDNN wherever it is the better kernel: an unseen
shape or stride, any dtype but fp16 (``ln_k`` is
instantiated for ``__half`` only, and no captured shape is anything else), a
non-3-D or empty or CPU input,
grad enabled, an outer capture already in progress, a normalized axis that does
not split into 8-element thread strips, a misaligned row start, or a weight
whose storage moved since capture (``p.data = ...`` bypasses ``_apply``, which
is what otherwise invalidates the plans).
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_vae_attention import OasisVAEAttention

_CAPTURING = torch.cuda.is_current_stream_capturing

# The frozen L2 attention's fused qkv-split/axial-RoPE kernel, reached through
# its own module so that this level can keep it (it writes q/k/v bit-exactly
# against the reference, at ~6.3 TB/s) while replacing only the SDPA call after
# it.  Absent under ``--standalone``, where the baseline L2 class is imported
# instead; the whole Triton-attention path then simply does not engage.
_L2_ATTN_MOD = sys.modules.get(OasisVAEAttention.__module__)
_qkv_rope_into = getattr(_L2_ATTN_MOD, "_qkv_rope_into", None)
_ROPE_BLOCK = int(getattr(_L2_ATTN_MOD, "_BLOCK", 256) or 256)

# Threads per row-CTA in ``ln_k``, i.e. how the normalized row is split.  The
# whole row lives in registers, so elements per thread sets both the register
# footprint (40 registers at 8, 64 at 16, 128 at 32) and how many bytes each
# resident thread keeps in flight.  128 threads / 8 elements per thread wins at
# both captured row counts and, unlike 64 / 16, wins *reproducibly*: 64 threads
# measured a slot faster at B=1 in one process (64.5 us) and a slot slower in
# the next (68.6 us), which is the same per-process bimodality the L2 MLP notes
# record for cuBLAS operand layouts.  Picked on the median of four independent
# processes, not one.
_LN_THREADS_ENV = os.environ.get("OASIS_L3_LN_THREADS")
# Launch the three glue kernels with Programmatic Dependent Launch.
_PDL = 0 if os.environ.get("OASIS_L3_NO_PDL") else 1
# Threads per CTA in the final residual add (8 elements each).
_ADD_THREADS = int(os.environ.get("OASIS_L3_ADD_THREADS", "256"))
# Rewrite ``attn.qkv.weight`` / ``attn.proj.weight`` K-major before capture, so
# cuBLAS serves them from an ``..._NNT`` nvjet kernel instead of ``..._TNT``.
_KMAJOR = 0 if os.environ.get("OASIS_L3_NO_KMAJOR") else 1


def _kmajor(w: torch.Tensor) -> None:
    """Rewrite ``w`` (shape ``[N, K]``) in place so that ``w.t()`` is contiguous.

    Same ``Parameter`` object, same shape, dtype, device and values -- only the
    strides change, from ``(K, 1)`` to ``(1, N)``.  ``F.linear`` hands cuBLAS
    ``w.t()``, which for a row-major weight is non-contiguous and answers with a
    ``..._TNT`` nvjet kernel; contiguous, it picks ``..._NNT``.  The frozen L2
    MLP already does exactly this for ``fc1``/``fc2`` (and measured it as both
    slightly faster and, more importantly, *stable* -- the ``w.t()`` path is
    bimodal by a whole 2.05 us dispatch slot per process), but the frozen L2
    attention does not do it for ``qkv``/``proj``.  Doing it here reaches into
    those two GEMMs without editing the child.

    Rewriting the parameter's own storage rather than caching a transposed copy
    is what makes this safe: there is no second copy to invalidate, so a later
    weight update cannot be missed.  Cost is one transpose kernel per weight on
    the first capture and no extra resident memory; the guard is one
    ``stride(0)`` read.
    """
    n, k = w.shape
    dst = torch.empty(k, n, dtype=w.dtype, device=w.device).t()
    with torch.no_grad():
        dst.copy_(w)
    w.data = dst


# ---------------------------------------------------------------------------
# Attention.
#
# The frozen L2 attention hands its q/k/v to ``F.scaled_dot_product_attention``,
# which on this box resolves to cuDNN's sm100 flash kernel: 8.15 GFLOP in 19.9 us
# at B=6 (2.76 TFLOP/s per SM) but only 1.36 GFLOP in 10.5-11.0 us at B=1, where
# B*H = 16 pairs times ceil(576/128) = 5 Q-tiles is **80 CTAs on a 148-SM
# machine** -- nearly half of it idle, and the reason B=1 attention costs as much
# as B=6's per FLOP.
#
# ``_attn_k`` is a flash forward whose Q tile is 64 rows, so the grid is
# B*H*ceil(576/64): 144 CTAs at B=1, one clean wave.  Measured against the same
# q/k/v (32 calls per graph replay, median of 20 replays): 9.1 us against cuDNN's
# 10.5 at B=1, and 23.3 against 20.1 at B=6 -- so it only ships where its grid
# fits in one wave, which is exactly the shape cuDNN leaves on the table.  Every
# other candidate measured *worse* than cuDNN at both shapes; see ITERATIONS.md
# for the table (flashinfer, flash_attn 2.8.3, SDPA's other backends, split-KV,
# and RoPE fused into this kernel).
#
# Softmax and the PV accumulation are fp32.  The scale carries log2(e) so the
# softmax is one ``ex2`` per element with no extra multiply over the [64, 64]
# score tile (measured 9.5 -> 9.1 us at B=1).
_LOG2E = 1.4426950408889634
_ATTN_BM = int(os.environ.get("OASIS_L3_ATTN_BM", "64"))
_ATTN_BN = int(os.environ.get("OASIS_L3_ATTN_BN", "64"))
_ATTN_WARPS = int(os.environ.get("OASIS_L3_ATTN_WARPS", "4"))
_ATTN_STAGES = int(os.environ.get("OASIS_L3_ATTN_STAGES", "3"))
# 0 disables it (cuDNN everywhere), 2 forces it on regardless of the grid rule.
_ATTN_MODE = int(os.environ.get("OASIS_L3_ATTN", "1"))


@triton.jit
def _attn_k(Q, K, V, O, sqb, sqs, sqh, sob, sos,
            S, H, SCALE, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    """out[b, s, h*D + d] = softmax(q k^T / sqrt(D)) v, one CTA per Q tile.

    q/k/v are read from the frozen RoPE pass's ``[B, S, H, D]`` destinations;
    the output is written straight out as ``[B, S, H*D]``, which is the layout
    ``proj`` consumes -- so nothing between here and the GEMM has to reshape or
    copy.
    """
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh - b * H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BN)
    base = b * sqb + h * sqh

    q = tl.load(Q + base + offs_m[:, None] * sqs + offs_d[None, :])

    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for start_n in range(0, S, BN):
        n = start_n + offs_n
        k = tl.load(K + base + n[:, None] * sqs + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + base + n[:, None] * sqs + offs_d[None, :])
        acc = tl.dot(p.to(v.dtype), v, acc, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + b * sob + h * D + offs_m[:, None] * sos + offs_d[None, :],
             acc.to(O.dtype.element_ty))


def _attn_run(q, k, v, out, heads: int) -> None:
    """Launch ``_attn_k`` for q/k/v shaped ``[B, S, H, D]`` into ``[B, S, H*D]``."""
    bsz, seq, _, hd = q.shape
    _attn_k[(triton.cdiv(seq, _ATTN_BM), bsz * heads)](
        q, k, v, out, q.stride(0), q.stride(1), q.stride(2),
        out.stride(0), out.stride(1), seq, heads,
        (hd ** -0.5) * _LOG2E, BM=_ATTN_BM, BN=_ATTN_BN, D=hd,
        num_warps=_ATTN_WARPS, num_stages=_ATTN_STAGES)


def _attn_wins(bsz: int, heads: int, seq: int, hd: int, dev: int) -> bool:
    """Should ``_attn_k`` serve this shape instead of cuDNN?

    The rule is the one the measurement supports: take it exactly when the grid
    is at most one wave, i.e. when cuDNN's larger Q tile would leave SMs idle
    and there is no second wave for it to amortize its higher per-CTA efficiency
    over.  At the captured shapes that is B=1 (144 CTAs) and not B=6 (864).
    """
    # ``_attn_k`` carries no masks, so both tiles have to divide the sequence,
    # and ``hd == 64`` is where it was measured -- nothing in the kernel needs
    # 64, but nothing here has been timed at another head dim either.
    if (_ATTN_MODE == 0 or hd != 64
            or seq % _ATTN_BM or seq % _ATTN_BN):
        return False
    if _ATTN_MODE == 2:
        return True
    try:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:
        return False
    return bsz * heads * (seq // _ATTN_BM) <= sms


def _ln_threads(n: int) -> int:
    """Row split for an ``n``-column normalization, or 0 if none fits."""
    order = ((int(_LN_THREADS_ENV),) if _LN_THREADS_ENV
             else (128, 256, 64, 32))
    for t in order:
        npt = n // t
        if n % t == 0 and npt % 8 == 0 and npt <= 32:
            return t
    return 0


_CPP_SRC = r"""
#include <torch/extension.h>

int64_t oasis_block_make(int64_t a, int64_t b, int64_t ho, int64_t y,
                         int64_t w, int64_t bb, int64_t rows, int64_t n,
                         int64_t sdiv, int64_t sb, int64_t ss, int64_t sd,
                         double eps, int64_t threads, int64_t dev,
                         int64_t kind, int64_t nelem, int64_t pslot,
                         int64_t pdl);
void oasis_block_run(int64_t handle, int64_t patch);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make", &oasis_block_make);
  m.def("run", &oasis_block_run);
}
"""

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>

namespace {

struct alignas(16) H8 { __half x[8]; };

// One 16-bit row op, fused: read one or two operands, optionally store their
// (rounded) fp16 sum, and store the LayerNorm of that value.
//
//   Bs == nullptr : val = A[row]                       (entry: copy + norm1)
//   Bs != nullptr : val = fp16(A[row] + Bs[row])       (residual + norm2)
//
// ``A`` is read through (sb, ss, sd) so a permuted caller tensor needs no
// preparatory contiguous() copy; everything this kernel writes is contiguous
// ``[rows, N]``.  ``HO`` may be null when the summed/copied value is not
// needed downstream.
//
// One CTA per normalized row, whole row resident in registers, fp32
// reduction by the shifted one-pass formula (subtract the row's own first
// element, then accumulate sum and sum-of-squares in the same pass so the two
// reduction trees pipeline instead of serializing a mean pass against a
// variance pass).  Shifting by a real data point keeps the
// ``sq/N - off*off`` cancellation on the scale of the row's spread.
template <int T, int NPT, bool PDL>
__global__ __launch_bounds__(T) void ln_k(
    const __half* __restrict__ A, const __half* __restrict__ Bs,
    __half* __restrict__ HO, __half* __restrict__ Y,
    const float* __restrict__ W, const float* __restrict__ BB,
    int rows, int sdiv, long long sb, long long ss, long long sd,
    float eps, float inv_n) {
  constexpr int N = T * NPT;
  constexpr int V = NPT / 8;
  const int tid = threadIdx.x;
  const bool vec = (sd == 1);
  const int row = blockIdx.x;
  long long base;
  if (sdiv) {
    const int b = row / sdiv;
    base = (long long)b * sb + (long long)(row - b * sdiv) * ss;
  } else {
    base = (long long)row * ss;
  }
  const long long obase = (long long)row * N;
  if (row >= rows) return;  // uniform across the CTA; grid is sized from rows

  __half raw[NPT];
  float v[NPT];

  if (PDL) {
    // Staged while the producer (the harness' input copy, or the proj GEMM)
    // is still draining; the grid is already resident by the time its output
    // is readable, which is the ~2 us of otherwise-idle device time a
    // dependent launch costs.
    cudaGridDependencySynchronize();
  }

  // The shift: element 0 of this row, after the same rounding every other
  // element gets, so the norm sees exactly the values HO stores.
  float shift;
  if (Bs == nullptr) {
    shift = __half2float(A[base]);
  } else {
    shift = __half2float(__float2half_rn(__half2float(A[base])
                                         + __half2float(Bs[obase])));
  }

  if (Bs == nullptr) {
    if (vec) {
      const H8* ap = reinterpret_cast<const H8*>(A + base);
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const H8 t = ap[i * T + tid];
#pragma unroll
        for (int j = 0; j < 8; ++j) raw[i * 8 + j] = t.x[j];
      }
    } else {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const long long e = (long long)(i * T + tid) * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) raw[i * 8 + j] = A[base + (e + j) * sd];
      }
    }
  } else {
    const H8* bp = reinterpret_cast<const H8*>(Bs + obase);
    if (vec) {
      const H8* ap = reinterpret_cast<const H8*>(A + base);
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const H8 t = ap[i * T + tid];
        const H8 u = bp[i * T + tid];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          raw[i * 8 + j] = __float2half_rn(__half2float(t.x[j])
                                           + __half2float(u.x[j]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const long long e = (long long)(i * T + tid) * 8;
        const H8 u = bp[i * T + tid];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          raw[i * 8 + j] = __float2half_rn(__half2float(A[base + (e + j) * sd])
                                           + __half2float(u.x[j]));
      }
    }
  }

  // The residual sum / input copy does not depend on the reduction, so issue
  // its store before reducing rather than in the output loop: it drains while
  // the row's two reduction trees run (worth 1.2 us of the 16.3 at B=6).
  if (HO != nullptr) {
#pragma unroll
    for (int i = 0; i < V; ++i) {
      H8 rr;
#pragma unroll
      for (int j = 0; j < 8; ++j) rr.x[j] = raw[i * 8 + j];
      reinterpret_cast<H8*>(HO + obase)[i * T + tid] = rr;
    }
  }

  float acc = 0.f, sq = 0.f;
#pragma unroll
  for (int i = 0; i < NPT; ++i) {
    const float d = __half2float(raw[i]) - shift;
    v[i] = d;
    acc += d;
    sq += d * d;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, o);
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
  }
  if (T > 32) {
    constexpr int NW = T / 32;
    __shared__ float sm[2 * NW];
    const int w = tid >> 5;
    if ((tid & 31) == 0) { sm[w] = acc; sm[NW + w] = sq; }
    __syncthreads();
    acc = 0.f; sq = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) { acc += sm[i]; sq += sm[NW + i]; }
  }
  const float off = acc * inv_n;
  const float var = sq * inv_n - off * off;
  const float rstd = 1.0f / sqrtf(fmaxf(var, 0.0f) + eps);

#pragma unroll
  for (int i = 0; i < V; ++i) {
    const int e = (i * T + tid) * 8;
    float g[8], c[8];
    if (W != nullptr) {
      *reinterpret_cast<float4*>(g) = *reinterpret_cast<const float4*>(W + e);
      *reinterpret_cast<float4*>(g + 4) = *reinterpret_cast<const float4*>(W + e + 4);
    }
    if (BB != nullptr) {
      *reinterpret_cast<float4*>(c) = *reinterpret_cast<const float4*>(BB + e);
      *reinterpret_cast<float4*>(c + 4) = *reinterpret_cast<const float4*>(BB + e + 4);
    }
    H8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float y = (v[i * 8 + j] - off) * rstd;
      if (W != nullptr) y *= g[j];
      if (BB != nullptr) y += c[j];
      o.x[j] = __float2half_rn(y);
    }
    reinterpret_cast<H8*>(Y + obase)[i * T + tid] = o;
  }
}

// out = a + b over a dense 16-bit buffer. Same rounding as torch.add on fp16
// (fp16 add is exact in fp32, so one round-to-nearest per element either way).
template <bool PDL>
__global__ void add_k(const __half* __restrict__ A, const __half* __restrict__ B,
                      __half* __restrict__ O, int nv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (PDL) cudaGridDependencySynchronize();
  if (i >= nv) return;
  const H8 a = reinterpret_cast<const H8*>(A)[i];
  const H8 b = reinterpret_cast<const H8*>(B)[i];
  H8 o;
#pragma unroll
  for (int j = 0; j < 8; j += 2) {
    const __half2 r = __hadd2(*reinterpret_cast<const __half2*>(a.x + j),
                              *reinterpret_cast<const __half2*>(b.x + j));
    *reinterpret_cast<__half2*>(o.x + j) = r;
  }
  reinterpret_cast<H8*>(O)[i] = o;
}

// ---------------------------------------------------------------------------
// Pre-bound launch plans.
//
// Every argument except the one pointer that changes per call is resolved once
// and kept here, so a launch from Python is a two-argument pybind call rather
// than a dozen tensor unpacks -- at the captured sizes the eager entry launch
// is pure latency, and this is the difference between ~1 us and ~4 us of host
// time in front of it.
// ---------------------------------------------------------------------------
struct Plan {
  const __half* a;
  const __half* b;
  __half* ho;
  __half* y;
  const float* w;
  const float* bb;
  int rows, n, threads, npt, sdiv, dev, nelem, pdl;
  long long sb, ss, sd;
  float eps, inv_n;
  int kind;   // 0 = ln, 1 = add
  int pslot;  // which bound pointer ``run``'s patch argument replaces
};

std::vector<Plan> g_plans;

// Programmatic Dependent Launch: every one of these kernels runs immediately
// downstream of something (the caller's input copy, the proj GEMM, fc2), and a
// dependent launch otherwise costs a fixed slice of idle device time before the
// first block issues. Launching with stream serialization lets the grid be
// staged while the producer drains; ``cudaGridDependencySynchronize`` in the
// kernel keeps the dependency honest.
template <typename K>
void launch_pdl(K kern, dim3 grid, dim3 block, cudaStream_t s, void** args) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  AT_CUDA_CHECK(cudaLaunchKernelExC(&cfg, (const void*)kern, args));
}

template <int T, int NPT>
void ln_launch(const Plan& p, const __half* a, const __half* b, cudaStream_t s) {
  if (p.pdl) {
    void* args[] = {(void*)&a, (void*)&b, (void*)&p.ho, (void*)&p.y,
                    (void*)&p.w, (void*)&p.bb, (void*)&p.rows, (void*)&p.sdiv,
                    (void*)&p.sb, (void*)&p.ss, (void*)&p.sd, (void*)&p.eps,
                    (void*)&p.inv_n};
    launch_pdl(ln_k<T, NPT, true>, dim3(p.rows), dim3(T), s, args);
  } else {
    ln_k<T, NPT, false><<<p.rows, T, 0, s>>>(
        a, b, p.ho, p.y, p.w, p.bb, p.rows, p.sdiv, p.sb, p.ss, p.sd, p.eps,
        p.inv_n);
  }
}

#define LN_CASE(T, NPT) \
  if (p.threads == (T) && p.npt == (NPT)) { ln_launch<T, NPT>(p, a, b, s); return; }

void ln_dispatch(const Plan& p, const __half* a, const __half* b,
                 cudaStream_t s) {
  LN_CASE(32, 8) LN_CASE(32, 16) LN_CASE(32, 32)
  LN_CASE(64, 8) LN_CASE(64, 16) LN_CASE(64, 32)
  LN_CASE(128, 8) LN_CASE(128, 16) LN_CASE(128, 32)
  LN_CASE(256, 8) LN_CASE(256, 16) LN_CASE(256, 32)
  TORCH_CHECK(false, "oasis_block: unsupported (threads, npt)");
}
#undef LN_CASE

}  // namespace

// (threads, npt) must satisfy threads * npt == n; npt a multiple of 8.
int64_t oasis_block_make(int64_t a, int64_t b, int64_t ho, int64_t y,
                         int64_t w, int64_t bb, int64_t rows, int64_t n,
                         int64_t sdiv, int64_t sb, int64_t ss, int64_t sd,
                         double eps, int64_t threads, int64_t dev,
                         int64_t kind, int64_t nelem, int64_t pslot,
                         int64_t pdl) {
  Plan p{};
  p.a = reinterpret_cast<const __half*>(a);
  p.b = reinterpret_cast<const __half*>(b);
  p.ho = reinterpret_cast<__half*>(ho);
  p.y = reinterpret_cast<__half*>(y);
  p.w = reinterpret_cast<const float*>(w);
  p.bb = reinterpret_cast<const float*>(bb);
  p.rows = (int)rows;
  p.n = (int)n;
  p.threads = (int)threads;
  p.npt = (int)(threads ? n / threads : 0);
  p.pdl = (int)pdl;
  p.sdiv = (int)sdiv;
  p.dev = (int)dev;
  p.nelem = (int)nelem;
  p.sb = sb; p.ss = ss; p.sd = sd;
  p.eps = (float)eps;
  p.inv_n = 1.0f / (float)n;
  p.kind = (int)kind;
  p.pslot = (int)pslot;
  if (kind == 0) {
    TORCH_CHECK(threads > 0 && n % threads == 0 && (n / threads) % 8 == 0,
                "oasis_block: bad row split");
  } else {
    TORCH_CHECK(threads > 0 && nelem % 8 == 0, "oasis_block: bad add split");
  }
  g_plans.push_back(p);
  return (int64_t)g_plans.size() - 1;
}

// ``patch`` replaces the one pointer that changes per call -- the slot named by
// the plan's ``pslot``.  It is written into the plan so that a launch recorded
// into a CUDA graph keeps the address it was captured with; 0 keeps the bound
// one.
void oasis_block_run(int64_t handle, int64_t patch) {
  TORCH_CHECK(handle >= 0 && (size_t)handle < g_plans.size(), "bad handle");
  Plan& p = g_plans[(size_t)handle];
  if (patch) {
    if (p.pslot == 0)      p.a = reinterpret_cast<const __half*>(patch);
    else if (p.pslot == 1) p.b = reinterpret_cast<const __half*>(patch);
    else                   p.ho = reinterpret_cast<__half*>(patch);
  }
  cudaStream_t s = at::cuda::getCurrentCUDAStream(p.dev);
  if (p.kind == 0) {
    ln_dispatch(p, p.a, p.b, s);
  } else {
    const int nv = p.nelem / 8;
    const int t = p.threads;
    const int blocks = (nv + t - 1) / t;
    if (p.pdl) {
      void* args[] = {(void*)&p.a, (void*)&p.b, (void*)&p.ho, (void*)&nv};
      launch_pdl(add_k<true>, dim3(blocks), dim3(t), s, args);
    } else {
      add_k<false><<<blocks, t, 0, s>>>(p.a, p.b, p.ho, nv);
    }
  }
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    tag = hashlib.md5((_CPP_SRC + _CUDA_SRC).encode()).hexdigest()[:10]
    # Build for the arch actually present rather than the whole
    # TORCH_CUDA_ARCH_LIST the image exports (7.5 ... 12.0+PTX): one arch
    # instead of six turns the one-time build from minutes into seconds.
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        return load_inline(
            name=f"oasis_vae_block_glue_{tag}",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--ftz=false"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


_ext = None
if torch.cuda.is_available() and not os.environ.get("OASIS_L3_NO_EXT"):
    try:
        _ext = _build()
    except Exception:  # pragma: no cover - fall back to the composed children
        _ext = None

_make = _ext.make if _ext is not None else None
_run = _ext.run if _ext is not None else None

# Which bound pointer ``run``'s second argument replaces.
_SLOT_A, _SLOT_B, _SLOT_OUT = 0, 1, 2


class _Plan:
    """Everything one captured (shape, stride, dtype) needs, resolved once."""

    __slots__ = ("graph", "entry", "tail", "bufs", "out_shape", "dev",
                 "mods", "pdicts", "objs", "ptrs", "affine", "abody")


class OasisVAEAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)
        self._dim = int(dim)
        self._seq_len = int(frame_height) * int(frame_width)
        # (shape, stride, dtype) -> _Plan, or None once capture has been tried
        # and declined, so an un-graphable shape is not re-attempted per call.
        self._plans = {}
        # fp32 affine casts we own, for a LayerNorm that does not cache its own.
        self._own_affine = {}

    def _apply(self, *args, **kwargs):
        # .to()/.cuda()/.half() move every buffer the captured graphs baked in,
        # so all of them stop being valid.
        self._plans = {}
        self._own_affine = {}
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        # A checkpoint load can replace a Parameter outright; the per-call guard
        # would catch it, but dropping the plans here means the graph can never
        # outlive the weights it was captured against even by one call.
        self._plans = {}
        self._own_affine = {}
        return super()._load_from_state_dict(*args, **kwargs)

    # ------------------------------------------------------------------
    # Eager composition: the fallback, and the correctness oracle.
    # ------------------------------------------------------------------
    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.norm1(x))
        return h + self.mlp(self.norm2(h))

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def _affine(self, norm):
        """The fp32 (weight, bias) pair ``ln_k`` reads, or None.

        ``norm`` promotes to fp32 for the reduction, so the affine is applied in
        fp32 too.  The candidate ``LayerNorm`` already keeps an fp32 cast of its
        parameters cached for its own row kernel -- reuse that, so the affine
        the graph reads is the same storage the child would have read.  When the
        child does not expose one (the baseline class, e.g. under
        ``--standalone``), cast here and hold it on ``_own_affine`` keyed on the
        parameter objects, which is what the plan's identity guard checks.

        Either way this inherits the cast's one documented weakness: an *in
        place* update to a norm weight (``w.mul_(...)``, same storage, same
        object) is not reflected until the cache is rebuilt. That is exactly the
        window ``L1.LayerNorm._w32`` already documents, so behaviour here
        matches the child's.
        """
        if not getattr(norm, "promote_fp32", True):
            return None
        cached = getattr(norm, "_fp32_affine", None)
        if cached is not None:
            w, b = cached()
        else:
            w, b = norm.weight, norm.bias
            own = self._own_affine.get(id(norm))
            if own is None or own[0] is not w or own[1] is not b:
                own = (w, b,
                       None if w is None else w.float().contiguous(),
                       None if b is None else b.float().contiguous())
                self._own_affine[id(norm)] = own
            w, b = own[2], own[3]
        if (w is None or b is None or w.dtype is not torch.float32
                or b.dtype is not torch.float32
                or not w.is_contiguous() or not b.is_contiguous()
                or w.numel() != self._dim or b.numel() != self._dim
                or (w.data_ptr() & 15) or (b.data_ptr() & 15)):
            return None
        return w, b

    def _attn_body(self, am, x, bsz: int, seq: int, dim: int, dev: int):
        """The attention half as ``x -> proj(attn(rope(qkv(x))))``, or None.

        Returns a callable only when ``_attn_k`` is the right kernel for this
        shape *and* every piece it needs is present: the frozen L2 attention's
        fused RoPE kernel, its cos/sin table, its per-shape q/k/v destinations,
        and its own fusion gate.  Otherwise the caller keeps ``am._eager``,
        which is the frozen child's own composition (same RoPE pass, cuDNN
        flash, ``proj``).

        Only the SDPA call is replaced.  ``qkv`` and ``proj`` stay the frozen
        child's ``Linear``s with the frozen child's arguments, and the RoPE pass
        stays the frozen child's kernel on the frozen child's table, so the
        rotation is still bit-exact against the reference.
        """
        heads = getattr(am, "num_heads", 0)
        if (_qkv_rope_into is None or not heads or dim % heads
                or not getattr(am, "_fuse_ok", False)
                or getattr(am, "rotary_freqs", None) is None
                or am.rotary_freqs.dtype is not torch.float32
                or not hasattr(am, "_dest") or not hasattr(am, "_build_cs")):
            return None
        hd = dim // heads
        if not _attn_wins(bsz, heads, seq, hd, dev):
            return None
        cs = am._cs
        if cs is None:
            cs = am._build_cs()
        qkvd = am._dest(bsz, seq, dim, x)
        # ``_attn_k`` reads these as contiguous [B, S, H, D]; the child builds
        # them that way, but the kernel would read past the end if it ever
        # stopped, so say so rather than assume it.
        if (len(qkvd) != 3
                or any(t.shape != (bsz, seq, heads, hd) or not t.is_contiguous()
                       or t.dtype is not x.dtype for t in qkvd)):
            return None
        aout = torch.empty(bsz, seq, dim, dtype=x.dtype, device=x.device)

        def body(h_in):
            qkv = am.qkv(h_in)
            q, k, v = qkvd
            _qkv_rope_into(qkv, cs, heads, q, k, v, _ROPE_BLOCK)
            _attn_run(q, k, v, aout, heads)
            return am.proj(aout)

        return body

    def _capture(self, x: torch.Tensor):
        if _make is None or not x.is_cuda or x.dim() != 3:
            return None
        bsz, seq, k_in = x.shape
        n = self._dim
        threads = _ln_threads(n)
        if (bsz <= 0 or seq <= 0 or seq != self._seq_len or k_in != n
                or x.dtype is not torch.float16 or threads == 0
                or torch.is_grad_enabled() or _CAPTURING()):
            return None
        sb, ss, sd = x.stride()
        if sd == 1:
            # The vectorized row load needs every row start 16B-aligned.
            if (x.data_ptr() & 15) or ((ss * 2) & 15) or ((sb * 2) & 15):
                return None
        elif ss != 1:
            # Neither axis unit-stride: the row is a gather with no locality to
            # lean on, and no captured shape looks like that. Defer.
            return None
        a1 = self._affine(self.norm1)
        a2 = self._affine(self.norm2)
        if a1 is None or a2 is None:
            return None
        aw1, ab1 = a1
        aw2, ab2 = a2
        am, mm = self.attn, self.mlp
        if (am.qkv.weight.dtype is not x.dtype
                or am.proj.weight.dtype is not x.dtype
                or mm.fc1.weight.dtype is not x.dtype
                or mm.fc2.weight.dtype is not x.dtype
                or mm.fc1.bias is None or mm.fc2.bias is None
                or am.proj.bias is None):
            return None

        # K-major once, before the warmup and the capture, so both the algo the
        # warmup picks and the kernel the graph records are the NNT one.  Only
        # the strides move, so the values the graph computes with are unchanged
        # and the identity guard below records the post-rewrite addresses.
        if _KMAJOR:
            for _w in (am.qkv.weight, am.proj.weight):
                if (_w.dim() == 2 and _w.is_cuda and _w.stride(1) == 1
                        and _w.stride(0) == _w.shape[1] and _w.shape[0] > 0
                        and _w.shape[1] > 0):
                    try:
                        _kmajor(_w)
                    except Exception:
                        # A failed transpose leaves the parameter untouched
                        # (``w.data`` is reseated only after the copy), so this
                        # costs the layout, not the capture.
                        break

        # The candidate L2 attention exposes its eager body; drive that rather
        # than its ``forward`` so capturing does not also poison its own
        # per-shape graph cache. Under ``--standalone`` (or if that candidate is
        # absent) the baseline class is imported instead and has no ``_eager``,
        # in which case its plain forward is already the eager body.
        attn_body = getattr(am, "_eager", am)
        dev = x.get_device()
        try:
            attn_body = self._attn_body(am, x, bsz, seq, n, dev) or attn_body
        except Exception:
            pass
        rows = bsz * seq
        opt = {"dtype": x.dtype, "device": x.device}
        # Static buffers, all allocated before capture so the graph records
        # their final addresses: the contiguous copy of x the residual needs,
        # norm1's output, the residual sum, and norm2's output.
        xs = torch.empty(bsz, seq, n, **opt)
        n1 = torch.empty(bsz, seq, n, **opt)
        h = torch.empty(bsz, seq, n, **opt)
        n2 = torch.empty(bsz, seq, n, **opt)
        # sdiv == 0 says "the rows are one unit-stride run", so the kernel skips
        # a division per CTA; true for every contiguous input and every
        # single-batch one.
        sdiv = 0 if (bsz == 1 or sb == seq * ss) else seq
        try:
            entry = _make(0, 0, xs.data_ptr(), n1.data_ptr(),
                          aw1.data_ptr(), ab1.data_ptr(), rows, n,
                          sdiv, sb, ss, sd, self.norm1.eps, threads,
                          dev, 0, 0, _SLOT_A, _PDL)
            mid = _make(xs.data_ptr(), 0, h.data_ptr(), n2.data_ptr(),
                        aw2.data_ptr(), ab2.data_ptr(), rows, n,
                        0, 0, n, 1, self.norm2.eps, threads,
                        dev, 0, 0, _SLOT_B, _PDL)
        except Exception:
            return None

        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                # Warm up off the capture stream so cuBLAS/cuDNN pick their
                # algorithms and take their workspaces, the L2 MLP rewrites its
                # weights K-major, the attention builds its cos/sin table and
                # its q/k/v destinations, and Triton compiles the GELU -- none
                # of which may happen during capture.
                for _ in range(3):
                    _run(entry, x.data_ptr())
                    warm = attn_body(n1)
                    _run(mid, warm.data_ptr())
                    torch.add(h, mm(n2))
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                attn_out = attn_body(n1)
                _run(mid, attn_out.data_ptr())
                mlp_out = mm(n2)
        except Exception:
            # Capture is best-effort; the composed children stay correct.
            return None
        if (not isinstance(attn_out, torch.Tensor)
                or not attn_out.is_contiguous()
                or attn_out.numel() != rows * n
                or attn_out.dtype is not x.dtype
                or not isinstance(mlp_out, torch.Tensor)
                or not mlp_out.is_contiguous()
                or mlp_out.numel() != rows * n
                or mlp_out.dtype is not x.dtype):
            return None
        # The graph node recorded for ``mid`` carries the attention-output
        # address it was captured with, so a replay never has to patch it again;
        # the ``mid`` handle itself is dead after this point.
        try:
            tail = _make(h.data_ptr(), mlp_out.data_ptr(), 0, 0, 0, 0,
                         rows, n, 0, 0, n, 1, 0.0, _ADD_THREADS, dev, 1,
                         rows * n, _SLOT_OUT, _PDL)
        except Exception:
            return None

        p = _Plan()
        p.graph = graph
        p.entry = entry
        p.tail = tail
        # The attention body owns the buffers the graph recorded addresses for
        # (the fused-RoPE destinations and this level's attention output), so it
        # has to outlive the capture, not just the closure it was built in.
        p.abody = attn_body
        # Every static buffer stays referenced: the graph replays into these
        # addresses and its private pool is only held open by them.
        p.bufs = (xs, n1, h, n2, attn_out, mlp_out)
        p.out_shape = (bsz, seq, n)
        p.dev = dev
        p.affine = (aw1, ab1, aw2, ab2)
        # Guard state. ``pdicts`` are the plain ``_parameters`` dicts, read
        # directly so the per-call check is a dict get rather than two trips
        # through ``nn.Module.__getattr__``; ``objs`` catches a replaced
        # Parameter and ``ptrs`` a reseated one (``p.data = other``, which does
        # not go through ``_apply``).
        # ``mods`` closes the hole a parameter-only guard leaves: replacing a
        # whole submodule (``blk.attn = other``) leaves the plan holding the old
        # module's ``_parameters`` dict, which would keep matching forever.
        p.mods = ((self._modules, "attn", am), (self._modules, "mlp", mm),
                  (self._modules, "norm1", self.norm1),
                  (self._modules, "norm2", self.norm2),
                  (am._modules, "qkv", am.qkv), (am._modules, "proj", am.proj),
                  (mm._modules, "fc1", mm.fc1), (mm._modules, "fc2", mm.fc2))
        p.pdicts = (am.qkv._parameters, am.proj._parameters,
                    mm.fc1._parameters, mm.fc2._parameters,
                    self.norm1._parameters, self.norm2._parameters)
        p.objs = self._objs(p.pdicts)
        # (tensor, address) for every weight the graph's GEMMs read. Checked by
        # value because ``p.data = other`` keeps the Parameter and only moves
        # its storage, which identity cannot see.
        p.ptrs = tuple((t, t.data_ptr()) for t in p.objs[:8] if t is not None)
        return p

    @staticmethod
    def _objs(pd):
        qkv, proj, fc1, fc2, nm1, nm2 = pd
        return (qkv.get("weight"), qkv.get("bias"),
                proj.get("weight"), proj.get("bias"),
                fc1.get("weight"), fc1.get("bias"),
                fc2.get("weight"), fc2.get("bias"),
                nm1.get("weight"), nm1.get("bias"),
                nm2.get("weight"), nm2.get("bias"))

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def _stale(self, p) -> bool:
        """Has anything the graph baked in been replaced or reseated?

        Three checks, cheapest first: a swapped submodule, a replaced Parameter
        (``mod.weight = other``), and a reseated one (``p.data = other``, which
        keeps the Parameter object and so is invisible to identity). Everything
        is read straight out of the ``_modules`` / ``_parameters`` dicts, which
        are plain dicts in ``__dict__``, rather than through
        ``nn.Module.__getattr__``.
        """
        for d, name, mod in p.mods:
            if d.get(name) is not mod:
                return True
        qkv, proj, fc1, fc2, nm1, nm2 = p.pdicts
        o = p.objs
        # Identity, one ``is`` at a time: a tuple compare would fall through to
        # Tensor.__eq__ (elementwise) the moment one of them differs.
        if (qkv.get("weight") is not o[0] or qkv.get("bias") is not o[1]
                or proj.get("weight") is not o[2] or proj.get("bias") is not o[3]
                or fc1.get("weight") is not o[4] or fc1.get("bias") is not o[5]
                or fc2.get("weight") is not o[6] or fc2.get("bias") is not o[7]
                or nm1.get("weight") is not o[8] or nm1.get("bias") is not o[9]
                or nm2.get("weight") is not o[10] or nm2.get("bias") is not o[11]):
            return True
        for t, addr in p.ptrs:
            if t.data_ptr() != addr:
                return True
        # The fp32 affine the kernels read by raw address must still be the one
        # captured; a replaced Parameter already failed the identity check
        # above, this catches the cache itself being rebuilt underneath us.
        aw1, ab1, aw2, ab2 = p.affine
        n1, n2 = self.norm1, self.norm2
        return (getattr(n1, "_w32", aw1) is not aw1
                or getattr(n1, "_b32", ab1) is not ab1
                or getattr(n2, "_w32", aw2) is not aw2
                or getattr(n2, "_b32", ab2) is not ab2)

    def _replay(self, p, x: torch.Tensor) -> torch.Tensor:
        _run(p.entry, x.data_ptr())
        p.graph.replay()
        # A fresh allocation per call, so nothing the next replay overwrites is
        # ever reachable from the caller.
        out = torch.empty(p.out_shape, dtype=x.dtype, device=x.device)
        _run(p.tail, out.data_ptr())
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.shape, x.stride(), x.dtype)
        p = self._plans.get(key, False)
        if p is not False:
            if (p is not None and x.is_cuda and not torch.is_grad_enabled()
                    and not _CAPTURING() and x.get_device() == p.dev
                    and not self._stale(p)):
                return self._replay(p, x)
        elif x.is_cuda and x.dim() == 3:
            p = self._capture(x)
            self._plans[key] = p
            if p is not None:
                return self._replay(p, x)
        return self._eager(x)
