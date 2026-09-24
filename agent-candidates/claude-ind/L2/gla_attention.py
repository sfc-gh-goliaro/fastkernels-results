"""Gated linear attention (covers both GLA and RetNet) -- optimized.

Same public surface as the baseline ``GatedLinearAttention``: identical
``__init__`` / ``forward`` signatures, identical parameter names (so a baseline
``state_dict`` loads verbatim), identical return tuple.

What changed relative to the baseline
-------------------------------------
1. **Fused projections.**  ``q/k/v/g`` all read the same activation, so their
   four weight matrices are concatenated once (lazily, after weights land) into
   a single ``[2*key + 2*value, hidden]`` matrix driven by one matmul.

2. **Hand-written single-token decode kernel.**  For a one-token step that
   starts from an empty recurrent state the GLA recurrence is exactly rank
   one::

       S = diag(exp(gk)) . 0 + k^T v = k^T v
       o = (q * scale) . S = scale * <q, k> * v

   The forget gate never participates and the ``[K, V]`` state never has to
   exist, so the kernel skips ``gk_proj`` / ``logsigmoid`` entirely and folds
   ``<q,k>``, the per-head RMSNorm and the swish output gate into one pass over
   the fused projection buffer.  The baseline instead runs the generic
   recurrent kernel, which materializes an ``[N, H, K, V]`` fp32 state (640 MB
   at B=256) that the caller immediately discards.  The whole step -- both
   matmuls and the fused kernel -- sits behind a single Python call.

3. **Own chunked prefill** (Triton, two kernels -- see the block comment above
   ``_gla_prep_kernel``).  The reference runs five kernels and materializes an
   ``[N_chunks, N, H, K, V]`` fp32 state, 8 GB at the captured prefill shape,
   which it then reads back.  Here the recurrent state stays in registers and
   is carried across chunks inside a single kernel, so it never reaches memory.

4. **Fused prefill epilogue** (``gla_norm_gate``): per-head RMSNorm times the
   swish gate in one warp-per-row pass, replacing three elementwise kernels and
   ~5 GB of round-trip traffic.  The wide gate projection (``gk_proj[1]`` plus
   logsigmoid plus normalizer) likewise collapses into one kernel.

5. **No host/device sync.**  The baseline's ``cu_seqlens`` branch calls
   ``lengths.max().item()`` every forward; that reduction + ``item()`` costs
   ~2.3 ms here and dominated the B=1 case.  The dispatch length is only
   needed once the packed batch is long enough to reach the chunk kernel, so
   it is computed lazily.

6. **No dead final state.**  The recurrence only materializes its final state
   when there is actually a cache to store it in.

Paths that carry a recurrent state in or out (cached prefill, decode with a
warm cache) and packed variable-length prefill keep the baseline's L1 kernels;
the fast paths above are all guarded and fall back to them.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.chunk_gla import ChunkGLA
from ..L1.chunk_retention import ChunkRetention
from ..L1.fused_recurrent_gla import FusedRecurrentGLA
from ..L1.fused_recurrent_retention import FusedRecurrentRetention
from ..L1.gla_recurrence import NaiveRecurrentGLA
from ..L1.linear import Linear
from ..L1.log_sigmoid import LogSigmoid
from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L1.silu import SiLU

_CHUNK_THRESHOLD = 64


# ---------------------------------------------------------------------------
# Chunked prefill (Triton)
#
# Two kernels instead of FLA's five, and no materialized per-chunk state.
#
# ``_gla_prep_kernel`` walks one chunk of BT tokens and emits everything the
# scan needs, so the log-gate cumulative sums are computed exactly once:
#
#     qi[i] = q[i] * scale * exp(G[i])          (G = in-chunk inclusive cumsum)
#     ka[j] = k[j] * exp(G[-1] - G[j])          (decayed key for the state)
#     A[i,j] = <qi[i], k[j] * exp(-G[j])>       for j <= i, else 0
#     gs     = G[-1]                            (per-chunk, per-channel decay)
#
# Both ``qi`` and ``ka`` carry factors <= 1, so nothing blows up; the ``kx``
# factor in ``A`` is the one that grows with the in-chunk decay and is clamped.
# ``A`` accumulates through a tf32 dot: the intra-chunk term dominates the
# output for realistic gates, and rounding its operands to bf16 there is what
# costs accuracy.  The inter-chunk dots stay bf16, matching the reference.
#
# ``_gla_scan_kernel`` owns one (sequence, head, value-tile) and carries the
# [K, BV] recurrent state in registers across chunks, so the [N, H, K, V] fp32
# state snapshots the reference writes (and re-reads) per chunk -- 8 GB at this
# shape -- never touch memory.
# ---------------------------------------------------------------------------
@triton.jit
def _gla_prep_kernel(q, k, g, qi, ka, A, gs, scale, T, H: tl.constexpr,
                     K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    NT = tl.num_programs(0)
    i_b = i_bh // H
    i_h = i_bh % H
    rows = i_t * BT + tl.arange(0, BT)
    m = (rows < T)[:, None]
    Aacc = tl.zeros((BT, BT), dtype=tl.float32)
    for i_k in range(0, tl.cdiv(K, BK)):
        cols = i_k * BK + tl.arange(0, BK)
        off = (i_b * T + rows[:, None]) * (H * K) + i_h * K + cols[None, :]
        b_q = tl.load(q + off, mask=m, other=0.0).to(tl.float32)
        b_k = tl.load(k + off, mask=m, other=0.0).to(tl.float32)
        b_g = tl.load(g + off, mask=m, other=0.0).to(tl.float32)
        b_G = tl.cumsum(b_g, axis=0)
        b_tot = tl.sum(b_g, axis=0)
        b_qi = b_q * (tl.exp(b_G) * scale)
        b_kx = b_k * tl.exp(tl.minimum(-b_G, 80.0))
        b_ka = b_k * tl.exp(b_tot[None, :] - b_G)
        tl.store(qi + off, b_qi.to(qi.dtype.element_ty), mask=m)
        tl.store(ka + off, b_ka.to(ka.dtype.element_ty), mask=m)
        Aacc += tl.dot(b_qi, tl.trans(b_kx), input_precision="tf32")
        tl.store(gs + (i_b * NT + i_t) * (H * K) + i_h * K + cols, b_tot)
    idx = tl.arange(0, BT)
    Aacc = tl.where(idx[:, None] >= idx[None, :], Aacc, 0.0)
    ao = (i_b * T + rows[:, None]) * (H * BT) + i_h * BT + idx[None, :]
    tl.store(A + ao, Aacc.to(A.dtype.element_ty), mask=m)


@triton.jit
def _gla_scan_kernel(qi, ka, A, v, gs, o, T, H: tl.constexpr, K: tl.constexpr,
                     V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, NT):
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    ck = tl.arange(0, K)
    cv = i_v * BV + tl.arange(0, BV)
    ja = tl.arange(0, BT)
    S = tl.zeros((K, BV), dtype=tl.float32)
    for i_t in range(0, NT):
        rows = i_t * BT + tl.arange(0, BT)
        m = (rows < T)[:, None]
        base = i_b * T + rows[:, None]
        offk = base * (H * K) + i_h * K + ck[None, :]
        offv = base * (H * V) + i_h * V + cv[None, :]
        b_qi = tl.load(qi + offk, mask=m, other=0.0)
        b_v = tl.load(v + offv, mask=m, other=0.0)
        b_A = tl.load(A + base * (H * BT) + i_h * BT + ja[None, :], mask=m, other=0.0)
        b_o = tl.dot(b_A, b_v)
        b_o += tl.dot(b_qi, S.to(b_qi.dtype))
        tl.store(o + offv, b_o.to(o.dtype.element_ty), mask=m)
        b_ka = tl.load(ka + offk, mask=m, other=0.0)
        b_tot = tl.load(gs + (i_b * NT + i_t) * (H * K) + i_h * K + ck)
        S = S * tl.exp(b_tot)[:, None] + tl.dot(tl.trans(b_ka), b_v)


_CHUNK_BT = 64
_CHUNK_BV = 64
_CHUNK_BK = 64
_PREP_CFG = (4, 2)   # (num_warps, num_stages)
_SCAN_CFG = (8, 2)


def _chunk_gla_fwd(q, k, v, g, scale):
    """o = chunked gated linear attention, no state carried in or out."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT, BV, BK = _CHUNK_BT, _CHUNK_BV, _CHUNK_BK
    NT = triton.cdiv(T, BT)
    qi = torch.empty_like(q)
    ka = torch.empty_like(k)
    A = torch.empty((B, T, H, BT), device=q.device, dtype=q.dtype)
    gs = torch.empty((B, NT, H, K), device=q.device, dtype=torch.float32)
    o = torch.empty((B, T, H, V), device=q.device, dtype=v.dtype)
    _gla_prep_kernel[(NT, B * H)](q, k, g, qi, ka, A, gs, scale, T, H, K, BT, BK,
                                  num_warps=_PREP_CFG[0], num_stages=_PREP_CFG[1])
    _gla_scan_kernel[(V // BV, B * H)](qi, ka, A, v, gs, o, T, H, K, V, BT, BV,
                                       NT, num_warps=_SCAN_CFG[0],
                                       num_stages=_SCAN_CFG[1])
    return o


def _chunk_supported(q, k, v, g) -> bool:
    K, V = q.shape[-1], v.shape[-1]
    return (q.is_cuda and q.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
            and g is not None and g.dtype == torch.bfloat16
            and K in (64, 128, 256, 512) and V % _CHUNK_BV == 0
            and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            and g.is_contiguous())



# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

using bf16 = __nv_bfloat16;
struct alignas(16) bf8 { bf16 v[8]; };

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Every thread leaves with the full block sum; smem is reusable afterwards.
__device__ __forceinline__ float block_sum(float v, float* smem) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < kWarps; ++i) t += smem[i];
  __syncthreads();
  return t;
}

__device__ __forceinline__ float swish(float g) {
  return __bfloat162float(__float2bfloat16(g / (1.f + __expf(-g))));
}

// ---------------------------------------------------------------------------
// Single-token GLA step: rank-one recurrence, per-head RMSNorm, swish gate.
// One block per (row, head), reading straight out of the fused q|k|v|g buffer.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void gla_decode_t1_kernel(
    const bf16* __restrict__ qkvg, const bf16* __restrict__ nw,
    bf16* __restrict__ out, const int64_t row_stride, const int H, const int K,
    const int V, const float scale, const float eps) {
  const int b = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int kd = H * K, vd = H * V;
  const bf16* row = qkvg + (int64_t)b * row_stride;
  const bf16* qp = row + h * K;
  const bf16* kp = row + kd + h * K;
  const bf16* vp = row + 2 * kd + h * V;
  const bf16* gp = row + 2 * kd + vd + h * V;
  bf16* op = out + (int64_t)b * vd + h * V;

  __shared__ float smem[kWarps];

  float a = 0.f;
  for (int i = tid; i < K; i += kThreads)
    a = __fmaf_rn(__bfloat162float(qp[i]), __bfloat162float(kp[i]), a);
  const float s = block_sum(a, smem) * scale;

  // o = s * v, rounded to bf16 exactly where the reference materializes it.
  float var = 0.f;
  for (int j = tid; j < V; j += kThreads) {
    const float t = __bfloat162float(__float2bfloat16(s * __bfloat162float(vp[j])));
    var = __fmaf_rn(t, t, var);
  }
  const float rs = rsqrtf(block_sum(var, smem) / (float)V + eps);

  for (int j = tid; j < V; j += kThreads) {
    const float t = __bfloat162float(__float2bfloat16(s * __bfloat162float(vp[j])));
    const float n =
        __bfloat162float(__float2bfloat16(t * rs * __bfloat162float(nw[j])));
    op[j] = __float2bfloat16(n * swish(__bfloat162float(gp[j])));
  }
}

// ---------------------------------------------------------------------------
// Prefill epilogue: per-head RMSNorm of the recurrence output times the swish
// output gate, in one pass.  One warp per (token, head) row, 16-byte accesses,
// so the reduction never leaves the warp.
// ---------------------------------------------------------------------------
template <int VPT>  // 16-byte vectors per lane
__global__ void norm_gate_vec_kernel(const bf16* __restrict__ o,
                                     const bf16* __restrict__ gt,
                                     const bf16* __restrict__ nw,
                                     bf16* __restrict__ y, const int64_t nrows,
                                     const float inv_v, const float eps) {
  const int64_t r = (int64_t)blockIdx.x * (kThreads >> 5) + (threadIdx.x >> 5);
  if (r >= nrows) return;
  const int lane = threadIdx.x & 31;
  const int V = VPT * 256;
  const bf16* orow = o + r * V;
  const bf16* grow = gt + r * V;

  bf8 ov[VPT], gv[VPT], wv[VPT];
  float var = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = (i * 32 + lane) * 8;
    ov[i] = *reinterpret_cast<const bf8*>(orow + c);
    gv[i] = *reinterpret_cast<const bf8*>(grow + c);
    wv[i] = *reinterpret_cast<const bf8*>(nw + c);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float f = __bfloat162float(ov[i].v[e]);
      var = __fmaf_rn(f, f, var);
    }
  }
  const float rs = rsqrtf(warp_sum(var) * inv_v + eps);

#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    bf8 res;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float n = __bfloat162float(__float2bfloat16(
          __bfloat162float(ov[i].v[e]) * rs * __bfloat162float(wv[i].v[e])));
      res.v[e] = __float2bfloat16(n * swish(__bfloat162float(gv[i].v[e])));
    }
    *reinterpret_cast<bf8*>(y + r * V + (i * 32 + lane) * 8) = res;
  }
}

__global__ __launch_bounds__(kThreads) void norm_gate_gen_kernel(
    const bf16* __restrict__ o, const bf16* __restrict__ gt,
    const bf16* __restrict__ nw, bf16* __restrict__ y, const int V,
    const float inv_v, const float eps) {
  const int64_t r = blockIdx.x;
  const int tid = threadIdx.x;
  const bf16* orow = o + r * V;
  const bf16* grow = gt + r * V;
  bf16* yrow = y + r * V;
  __shared__ float smem[kWarps];
  float var = 0.f;
  for (int j = tid; j < V; j += kThreads) {
    const float f = __bfloat162float(orow[j]);
    var = __fmaf_rn(f, f, var);
  }
  const float rs = rsqrtf(block_sum(var, smem) * inv_v + eps);
  for (int j = tid; j < V; j += kThreads) {
    const float n = __bfloat162float(__float2bfloat16(
        __bfloat162float(orow[j]) * rs * __bfloat162float(nw[j])));
    yrow[j] = __float2bfloat16(n * swish(__bfloat162float(grow[j])));
  }
}

// ---------------------------------------------------------------------------
// Low-rank forget gate: second gk_proj matmul (rank R, so nowhere near a
// tensor-core problem) fused with logsigmoid and the logit normalizer.  The
// reference spends a bias-GEMM plus two full-width elementwise passes here;
// this touches the 1280-wide output exactly once.
// ---------------------------------------------------------------------------
// Two adjacent output columns per thread so the bf16 stores are 4-byte wide,
// and fast-intrinsic transcendentals -- logsigmoid's log1pf/expf dominate this
// kernel otherwise, and the result is rounded to bf16 anyway.
template <int R, int BM, int BN>
__global__ __launch_bounds__(BN / 2) void gk_fused_kernel(
    const bf16* __restrict__ gl, const bf16* __restrict__ W2,
    const bf16* __restrict__ b2, bf16* __restrict__ out, const int M,
    const int KD, const float inv_norm) {
  constexpr int TH = BN / 2;
  __shared__ bf16 sw[BN * R];   // W2 rows [n0, n0+BN), contiguous in memory
  __shared__ bf16 sgl[BM * R];
  const int tid = threadIdx.x;
  const int n0 = blockIdx.y * BN;
  const int m0 = blockIdx.x * BM;
  const int nlim = min(BN, KD - n0);

  for (int idx = tid; idx < nlim * R; idx += TH)
    sw[idx] = W2[(size_t)n0 * R + idx];
  for (int idx = tid; idx < BM * R; idx += TH) {
    const int mm = idx / R;
    sgl[idx] = (m0 + mm < M) ? gl[(size_t)(m0 + mm) * R + (idx - mm * R)]
                             : __float2bfloat16(0.f);
  }
  __syncthreads();

  const int c0 = tid * 2;
  if (c0 >= nlim) return;
  const bool pair = (c0 + 1) < nlim;

  float w0[R], w1[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    w0[i] = __bfloat162float(sw[c0 * R + i]);
    w1[i] = pair ? __bfloat162float(sw[(c0 + 1) * R + i]) : 0.f;
  }
  const float b0 = __bfloat162float(b2[n0 + c0]);
  const float b1 = pair ? __bfloat162float(b2[n0 + c0 + 1]) : 0.f;

  const int mend = min(BM, M - m0);
  for (int mm = 0; mm < mend; ++mm) {
    float a0 = b0, a1 = b1;
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const float x = __bfloat162float(sgl[mm * R + i]);
      a0 = __fmaf_rn(x, w0[i], a0);
      a1 = __fmaf_rn(x, w1[i], a1);
    }
    const float t0 = __bfloat162float(__float2bfloat16(a0));
    const float t1 = __bfloat162float(__float2bfloat16(a1));
    const float l0 = fminf(t0, 0.f) - __logf(1.f + __expf(-fabsf(t0)));
    const float l1 = fminf(t1, 0.f) - __logf(1.f + __expf(-fabsf(t1)));
    const float r0 = __bfloat162float(__float2bfloat16(l0)) * inv_norm;
    const float r1 = __bfloat162float(__float2bfloat16(l1)) * inv_norm;
    bf16* dst = out + (size_t)(m0 + mm) * KD + n0 + c0;
    if (pair) {
      *reinterpret_cast<__nv_bfloat162*>(dst) =
          __nv_bfloat162(__float2bfloat16(r0), __float2bfloat16(r1));
    } else {
      dst[0] = __float2bfloat16(r0);
    }
  }
}

}  // namespace

// Whole single-token decode step behind one Python call.
torch::Tensor gla_decode(torch::Tensor hs, torch::Tensor wqkvg_t,
                         torch::Tensor wo_t, torch::Tensor nw, int64_t H,
                         int64_t K, int64_t V, double scale, double eps) {
  TORCH_CHECK(hs.scalar_type() == at::kBFloat16, "hidden_states must be bf16");
  const int64_t B = hs.size(0);
  const at::cuda::OptionalCUDAGuard guard(device_of(hs));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto x = hs.reshape({B, hs.size(-1)});
  auto qkvg = at::mm(x, wqkvg_t);
  auto y = at::empty({B, H * V}, x.options());
  if (B > 0) {
    dim3 grid((unsigned)B, (unsigned)H);
    gla_decode_t1_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(qkvg.data_ptr()),
        reinterpret_cast<const bf16*>(nw.data_ptr()),
        reinterpret_cast<bf16*>(y.data_ptr()), qkvg.stride(0), (int)H, (int)K,
        (int)V, (float)scale, (float)eps);
  }
  return at::mm(y, wo_t).view({B, 1, -1});
}

// gk = logsigmoid(gl @ W2^T + b2) / normalizer, fused.
torch::Tensor gk_fused(torch::Tensor gl, torch::Tensor W2, torch::Tensor b2,
                       double inv_norm) {
  TORCH_CHECK(gl.dim() == 2 && gl.is_contiguous(), "gl must be 2-D contiguous");
  TORCH_CHECK(gl.scalar_type() == at::kBFloat16, "gl must be bf16");
  const int64_t M = gl.size(0);
  const int64_t R = gl.size(1);
  const int64_t KD = W2.size(0);
  TORCH_CHECK(W2.size(1) == R && W2.is_contiguous(), "bad W2");
  auto out = at::empty({M, KD}, gl.options());
  if (M == 0) return out;
  const at::cuda::OptionalCUDAGuard guard(device_of(gl));
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int BM = 64, BN = 256;
  dim3 grid((unsigned)((M + BM - 1) / BM), (unsigned)((KD + BN - 1) / BN));
  const unsigned threads = BN / 2;
  const auto* glp = reinterpret_cast<const bf16*>(gl.data_ptr());
  const auto* wp = reinterpret_cast<const bf16*>(W2.data_ptr());
  const auto* bp = reinterpret_cast<const bf16*>(b2.data_ptr());
  auto* op = reinterpret_cast<bf16*>(out.data_ptr());
#define FK_LAUNCH_R(N)                                                      \
  gk_fused_kernel<N, BM, BN><<<grid, threads, 0, stream>>>(glp, wp, bp, op,      \
                                                      (int)M, (int)KD,      \
                                                      (float)inv_norm)
  if (R == 16) FK_LAUNCH_R(16);
  else if (R == 8) FK_LAUNCH_R(8);
  else if (R == 32) FK_LAUNCH_R(32);
  else TORCH_CHECK(false, "unsupported gate_low_rank_dim ", R);
#undef FK_LAUNCH_R
  return out;
}

// y = rmsnorm(o, nw) * silu(gate), normalizing over each run of V elements.
torch::Tensor gla_norm_gate(torch::Tensor o, torch::Tensor gate,
                            torch::Tensor nw, int64_t V, double eps) {
  TORCH_CHECK(o.scalar_type() == at::kBFloat16, "o must be bf16");
  TORCH_CHECK(o.numel() == gate.numel(), "o / gate size mismatch");
  auto oc = o.is_contiguous() ? o : o.contiguous();
  auto gc = gate.is_contiguous() ? gate : gate.contiguous();
  auto y = at::empty_like(gc);
  const int64_t nrows = oc.numel() / V;
  if (nrows == 0) return y;
  const at::cuda::OptionalCUDAGuard guard(device_of(o));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* op = reinterpret_cast<const bf16*>(oc.data_ptr());
  const auto* gp = reinterpret_cast<const bf16*>(gc.data_ptr());
  const auto* np = reinterpret_cast<const bf16*>(nw.data_ptr());
  auto* yp = reinterpret_cast<bf16*>(y.data_ptr());
  const float inv_v = 1.f / (float)V;
  const int vpt = (int)(V / 256);
  if (V % 256 == 0 && vpt >= 1 && vpt <= 4) {
    dim3 grid((unsigned)((nrows + (kThreads / 32) - 1) / (kThreads / 32)));
#define FK_LAUNCH_VPT(N)                                                   \
  norm_gate_vec_kernel<N>                                                  \
      <<<grid, kThreads, 0, stream>>>(op, gp, np, yp, nrows, inv_v, (float)eps)
    if (vpt == 1) FK_LAUNCH_VPT(1);
    else if (vpt == 2) FK_LAUNCH_VPT(2);
    else if (vpt == 3) FK_LAUNCH_VPT(3);
    else FK_LAUNCH_VPT(4);
#undef FK_LAUNCH_VPT
  } else {
    norm_gate_gen_kernel<<<(unsigned)nrows, kThreads, 0, stream>>>(
        op, gp, np, yp, (int)V, inv_v, (float)eps);
  }
  return y;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor gla_decode(torch::Tensor hs, torch::Tensor wqkvg_t,
                         torch::Tensor wo_t, torch::Tensor nw, int64_t H,
                         int64_t K, int64_t V, double scale, double eps);
torch::Tensor gla_norm_gate(torch::Tensor o, torch::Tensor gate,
                            torch::Tensor nw, int64_t V, double eps);
torch::Tensor gk_fused(torch::Tensor gl, torch::Tensor W2, torch::Tensor b2,
                       double inv_norm);
"""

_EXT = None


def _ext():
    """JIT-build (once) and return the extension module."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if prev is None and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            suffix = "a" if major in (9, 10, 12) else ""
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"
        try:
            _EXT = load_inline(
                name="fk_gla_attn",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["gla_decode", "gla_norm_gate", "gk_fused"],
                extra_cuda_cflags=["-O3", "--use_fast_math",
                                   "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
                verbose=False,
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    return _EXT



class GatedLinearAttention(nn.Module):
    """Unified L2 attention for GLA and RetNet.

    Args:
        hidden_size: Model hidden size.
        num_heads: Number of attention heads.
        expand_k: Key expansion ratio (GLA: 0.5, RetNet: 1.0).
        expand_v: Value expansion ratio (GLA: 1.0, RetNet: 2.0).
        decay_mode: Which forget-gate mechanism to use.
        gate_low_rank_dim: Low-rank dim for the GLA gate.
        gate_logit_normalizer: Normalizer applied after logsigmoid.
        use_rotary: Whether to apply rotary to q/k (RetNet uses this).
        rotary_base: Rotary base (theta).
        rotary_max_position: Max sequence length the rotary cache covers.
        norm_eps: RMSNorm epsilon for the per-head output norm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        decay_mode: Literal["learned_low_rank", "fixed_per_head"] = "learned_low_rank",
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_max_position: int = 8192,
        norm_eps: float = 1e-6,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        assert decay_mode in ("learned_low_rank", "fixed_per_head"), (
            f"unknown decay_mode: {decay_mode!r}"
        )
        self.num_heads = num_heads
        self.decay_mode = decay_mode
        self.use_rotary = use_rotary
        self.gate_logit_normalizer = gate_logit_normalizer

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

        if decay_mode == "learned_low_rank":
            self.gk_proj = nn.Sequential(
                Linear(hidden_size, gate_low_rank_dim, bias=False),
                Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
            self.log_sigmoid = LogSigmoid()
        else:
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            self.register_buffer("log_gamma", torch.log(gamma), persistent=False)

        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_k_dim,
                max_position_embeddings=rotary_max_position,
                rope_theta=rotary_base,
            )

        self.use_fast_kernels = use_fast_kernels
        self.naive_recurrence = NaiveRecurrentGLA()
        if use_fast_kernels:
            if decay_mode == "learned_low_rank":
                self.fused_recurrence = FusedRecurrentGLA()
                self.chunk = ChunkGLA()
            else:
                self.fused_recurrence = FusedRecurrentRetention()
                self.chunk = ChunkRetention()

        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.gate_act = SiLU()

        # Fast-path scratch; rebuilt lazily so it always reflects the live
        # parameters (weights arrive after __init__ via load_state_dict / .to()).
        self._hidden_size = hidden_size
        self._norm_eps = norm_eps
        self._scale = self.head_k_dim ** -0.5
        self._decode_cache: tuple | None = None
        self._norm_cache: torch.Tensor | None = None

    # -- lazily fused weights ---------------------------------------------
    def _invalidate(self) -> None:
        self._decode_cache = None
        self._norm_cache = None

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate()
        return super()._load_from_state_dict(*args, **kwargs)

    def _norm_weight(self) -> torch.Tensor:
        nw = self._norm_cache
        if nw is None:
            nw = self.g_norm_swish_gate.weight
            nw = nw.to(dtype=torch.bfloat16).contiguous()
            self._norm_cache = nw
        return nw

    def _decode_args(self):
        """(fn, qkvg_weight^T, o_weight^T, norm_weight) for the decode kernel."""
        c = self._decode_cache
        if c is None:
            w = torch.cat(
                (self.q_proj.weight, self.k_proj.weight,
                 self.v_proj.weight, self.g_proj.weight), dim=0,
            ).contiguous()
            c = (_ext().gla_decode, w.t(), self.o_proj.weight.t(),
                 self._norm_weight())
            self._decode_cache = c
        return c

    def _can_fuse_epilogue(self, o: torch.Tensor, g: torch.Tensor) -> bool:
        return (o.is_cuda and o.dtype == torch.bfloat16
                and g.dtype == torch.bfloat16
                and self.g_norm_swish_gate.elementwise_affine
                and o.numel() == g.numel())

    # -- gk helpers (unchanged semantics) ---------------------------------
    def _compute_gk(self, hidden_states: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Returns gk shaped [B, num_heads, T, head_k_dim] in log-space."""
        if self.decay_mode == "learned_low_rank":
            gk = self.gk_proj(hidden_states)
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
            return gk.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1
        ).expand(B, self.num_heads, T, self.head_k_dim)

    def _compute_gk_bthk(self, hidden_states: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Returns gk shaped [B, T, num_heads, head_k_dim] in log-space."""
        low, up = self.gk_proj[0], self.gk_proj[1]
        rank = low.weight.shape[0]
        if (hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
                and up.bias is not None and rank in (8, 16, 32)):
            gl = low(hidden_states).reshape(-1, rank).contiguous()
            gk = _ext().gk_fused(gl, up.weight, up.bias,
                                 1.0 / self.gate_logit_normalizer)
        else:
            gk = self.gk_proj(hidden_states)
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
        return gk.view(B, T, self.num_heads, self.head_k_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        B, T, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")

        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))
        # The final state is only ever read back out of a cache object; without
        # one there is no reason for the recurrence to materialize it.
        store_state = use_cache and past_key_values is not None

        # Rank-one single-token fast path (see module docstring).
        if (T == 1 and initial_state is None and not store_state
                and self.use_fast_kernels and not self.use_rotary
                and self.decay_mode == "learned_low_rank"
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.bfloat16):
            fn, wqkvg, wo, nw = self._decode_args()
            out = fn(hidden_states, wqkvg, wo, nw, self.num_heads,
                     self.head_k_dim, self.head_v_dim, self._scale,
                     self._norm_eps)
            return out, None, past_key_values

        max_seqlen = None
        if cu_seqlens is not None:
            if B != 1:
                raise ValueError("cu_seqlens prefill expects packed hidden_states with batch size 1")
            if T >= _CHUNK_THRESHOLD:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
            else:
                # Every per-sequence length is <= T < threshold anyway, so the
                # dispatch is already decided -- skip the device sync.
                max_seqlen = T

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        if self.use_rotary:
            offsets = None
            if past_key_values is not None:
                offsets = getattr(past_key_values, "seq_offsets", None)
            if cu_seqlens is not None:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                seg_start = torch.repeat_interleave(cu_seqlens[:-1], lengths)
                positions = torch.arange(T, device=q.device, dtype=torch.int64) - seg_start
                if isinstance(offsets, int):
                    positions = positions + offsets
                elif offsets is not None:
                    positions = positions + torch.repeat_interleave(
                        offsets.to(device=q.device, dtype=torch.int64), lengths)
                positions = positions.contiguous()
            else:
                local = torch.arange(T, device=q.device, dtype=torch.int64)
                if offsets is None:
                    positions = local.repeat(B)
                elif isinstance(offsets, int):
                    positions = (local + offsets).repeat(B)
                else:
                    positions = (offsets.to(device=q.device, dtype=torch.int64)
                                 .unsqueeze(1) + local.unsqueeze(0)).reshape(-1)
                    positions = positions.contiguous()
            q_flat = q.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            k_flat = k.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(B, T, self.num_heads, self.head_k_dim)
            k = k_flat.view(B, T, self.num_heads, self.head_k_dim)
        else:
            q = q.view(B, T, self.num_heads, self.head_k_dim)
            k = k.view(B, T, self.num_heads, self.head_k_dim)

        v = v.view(B, T, self.num_heads, self.head_v_dim)

        if self.use_fast_kernels and q.is_cuda:
            dispatch_len = max_seqlen if max_seqlen is not None else T
            if self.decay_mode == "learned_low_rank":
                gk_btHK = self._compute_gk_bthk(hidden_states, B, T)
                if dispatch_len >= _CHUNK_THRESHOLD:
                    if (cu_seqlens is None and initial_state is None
                            and not store_state
                            and _chunk_supported(q, k, v, gk_btHK)):
                        o = _chunk_gla_fwd(q, k, v, gk_btHK, self._scale)
                        final_state = None
                    else:
                        o, final_state = self.chunk(
                            q=q, k=k, v=v, g=gk_btHK,
                            initial_state=initial_state,
                            output_final_state=store_state,
                            cu_seqlens=cu_seqlens,
                        )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v, gk=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=store_state,
                        cu_seqlens=cu_seqlens,
                    )
            else:
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=store_state,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=store_state,
                        cu_seqlens=cu_seqlens,
                    )
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            gk = self._compute_gk(hidden_states, B, T)
            o, final_state = self.naive_recurrence(
                q, k, v, gk,
                initial_state=initial_state,
                output_final_state=store_state,
            )
            o = o.transpose(1, 2)

        if store_state:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state

        if self._can_fuse_epilogue(o, g):
            o = _ext().gla_norm_gate(o, g, self._norm_weight(),
                                     self.head_v_dim, self._norm_eps)
        else:
            o = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
            o = o.view(B, T, self.value_dim)
            o = o * self.gate_act(g)

        return self.o_proj(o), None, past_key_values
