"""Fused recurrent GLA — hand-written CUDA kernel (decode + short prefill).

The captured workload is GLA / RetNet decode against an ``[N, H, K, V]`` fp32
recurrent state that is read *and* written on every call
(``output_final_state=True``).  At ``K=256, V=512`` that state is 512 KiB per
(sequence, head), which dwarfs ``q/k/v/gk``, so the ``T == 1`` path is a pure
HBM streaming problem: read ``h0``, write ``ht = h0*exp(gk) + k v^T``, and fold
``o = (q*scale)^T ht`` into the same pass.

Two kernels:

* ``gla_t1_kernel`` (``T == 1``) -- grid is (V tile, N*H).  A thread owns four
  contiguous ``V`` lanes (``float4``) and a ``1/KSUB`` slice of ``K``; the ``o``
  partials of the ``KSUB`` slices are reduced through shared memory, so no
  second kernel and no cross-CTA reduction are needed.  Loads/stores use
  streaming cache hints since nothing is re-read.  This lands within ~7% of a
  pure ``float4`` device-to-device copy of the same bytes.
* ``gla_tn_kernel`` (``T > 1``) -- one warp per ``V`` column; the state tile
  stays in registers (8 ``K`` rows per lane).  ``q/k/gk`` are read straight from
  global as one 16 B vector per lane, two steps ahead, and the per-step ``o``
  reduction is a warp shuffle -- so the step loop has no shared memory, no
  barrier, and no branch.

Anything the fast path does not cover (``gk is None``, varlen, other
dtypes/shapes) falls back to FLA's Triton kernel.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from fla.ops.gla import fused_recurrent_gla

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float4 ld_stream(const float* p) {
  return __ldcs(reinterpret_cast<const float4*>(p));
}
__device__ __forceinline__ void st_stream(float* p, const float4& x) {
  __stwt(reinterpret_cast<float4*>(p), x);
}
__device__ __forceinline__ float4 ld_bf4(const bf16* p) {
  const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(p);
  float2 a = __bfloat1622float2(q[0]);
  float2 b = __bfloat1622float2(q[1]);
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ void st_bf4(bf16* p, float x, float y, float z, float w) {
  __nv_bfloat162 out[2] = {__floats2bfloat162_rn(x, y), __floats2bfloat162_rn(z, w)};
  *reinterpret_cast<float2*>(p) = *reinterpret_cast<const float2*>(out);
}
__device__ __forceinline__ void unpack8(const uint4& r, float* out) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(p[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

// ---------------------------------------------------------------------------
// T == 1.  grid = (V/(4*NL), N*H), block = NL*KSUB.
// Thread (lane, sub) owns v4 lane `blockIdx.x*NL + lane` and k slice `sub`.
// ---------------------------------------------------------------------------
template <int NL, int KSUB, int U, bool HAS_H0, bool STORE_HT>
__global__ __launch_bounds__(NL* KSUB) void gla_t1_kernel(
    const bf16* __restrict__ pq, const bf16* __restrict__ pk,
    const bf16* __restrict__ pv, const bf16* __restrict__ pgk,
    const float* __restrict__ h0, float* __restrict__ ht,
    bf16* __restrict__ o, int K, int V, float scale) {
  extern __shared__ float sm[];
  float* sq = sm;
  float* sg = sm + K;
  float* sk = sm + 2 * K;
  float* sred = sm + 3 * K;

  const int nh = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % NL, sub = tid / NL;
  {
    const size_t off = (size_t)nh * K;
    for (int i = tid; i < K; i += NL * KSUB) {
      sq[i] = __bfloat162float(pq[off + i]) * scale;
      sg[i] = __expf(__bfloat162float(pgk[off + i]));
      sk[i] = __bfloat162float(pk[off + i]);
    }
  }
  __syncthreads();

  const int j = blockIdx.x * NL + lane;  // float4 lane within V
  const int kc = K / KSUB;
  const int ks = sub * kc;
  const size_t sbase = (size_t)nh * K * V + (size_t)ks * V + 4 * (size_t)j;
  const float4 vf = ld_bf4(pv + (size_t)nh * V + 4 * j);
  const float* hp = h0 + sbase;
  float* hq = ht + sbase;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;

  for (int i = 0; i < kc; i += U) {
    float4 b[U];
    if (HAS_H0) {
#pragma unroll
      for (int u = 0; u < U; ++u) b[u] = ld_stream(hp + (size_t)(i + u) * V);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const float g = sg[ks + i + u], kv = sk[ks + i + u], qv = sq[ks + i + u];
      float4 s;
      if (HAS_H0) {
        s.x = b[u].x * g + kv * vf.x;
        s.y = b[u].y * g + kv * vf.y;
        s.z = b[u].z * g + kv * vf.z;
        s.w = b[u].w * g + kv * vf.w;
      } else {
        s.x = kv * vf.x;
        s.y = kv * vf.y;
        s.z = kv * vf.z;
        s.w = kv * vf.w;
      }
      if (STORE_HT) st_stream(hq + (size_t)(i + u) * V, s);
      a0 += qv * s.x;
      a1 += qv * s.y;
      a2 += qv * s.z;
      a3 += qv * s.w;
    }
  }

  if (KSUB > 1) {
    float* d = sred + (size_t)sub * NL * 4 + lane * 4;
    d[0] = a0; d[1] = a1; d[2] = a2; d[3] = a3;
    __syncthreads();
    if (sub == 0) {
#pragma unroll 1
      for (int s2 = 1; s2 < KSUB; ++s2) {
        const float* e = sred + (size_t)s2 * NL * 4 + lane * 4;
        a0 += e[0]; a1 += e[1]; a2 += e[2]; a3 += e[3];
      }
      st_bf4(o + (size_t)nh * V + 4 * j, a0, a1, a2, a3);
    }
  } else {
    st_bf4(o + (size_t)nh * V + 4 * j, a0, a1, a2, a3);
  }
}

// ---------------------------------------------------------------------------
// T > 1.  grid = (V, N*H), one warp per block: warp owns V column
// `blockIdx.x`, lane `l` owns the 8 K rows at `8*l`.  The o reduction is a warp
// shuffle, so the step loop has no shared memory and no barrier.
//
// The step loop is latency-bound (only N*H*V warps exist, and every warp
// re-reads the same q/k/gk row), so it is written as a branch-free steady state
// that keeps PF steps of q/k/gk/v in flight.  Guarding the prefetch with
// `if (t + PF < T)` inside the unrolled body instead costs ~25%: the compiler
// will not hoist a load out of a conditional block.
// ---------------------------------------------------------------------------
template <int PF, bool HAS_H0, bool STORE_HT>
__global__ __launch_bounds__(32) void gla_tn_kernel(
    const bf16* __restrict__ pq, const bf16* __restrict__ pk,
    const bf16* __restrict__ pv, const bf16* __restrict__ pgk,
    const float* __restrict__ h0, float* __restrict__ ht,
    bf16* __restrict__ o, int T, int H, int V, float scale) {
  constexpr int KPT = 8, K = 32 * KPT;
  const int nh = blockIdx.y, b = nh / H, h = nh % H;
  const int lane = threadIdx.x;
  const int v = blockIdx.x;
  const size_t hoff = (size_t)nh * K * V + (size_t)(lane * KPT) * V + v;

  float st[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i) st[i] = HAS_H0 ? h0[hoff + (size_t)i * V] : 0.f;

  // Stepping a uint4* (rather than casting a bf16* in the loop) is what keeps
  // these as 16 B loads; otherwise nvcc splits each into four dwords.
  const size_t qs = (size_t)H * K / 8, vs = (size_t)H * V;
  const size_t qrow = ((size_t)b * T * H + h) * K;
  const uint4* qp = reinterpret_cast<const uint4*>(pq + qrow) + lane;
  const uint4* kp = reinterpret_cast<const uint4*>(pk + qrow) + lane;
  const uint4* gp = reinterpret_cast<const uint4*>(pgk + qrow) + lane;
  const bf16* vp = pv + ((size_t)b * T * H + h) * V + v;
  bf16* op = o + ((size_t)b * T * H + h) * V + v;

  uint4 rq[PF], rk[PF], rg[PF];
  bf16 rv[PF];
#pragma unroll
  for (int d = 0; d < PF; ++d) {
    if (d < T) {
      rq[d] = qp[(size_t)d * qs];
      rk[d] = kp[(size_t)d * qs];
      rg[d] = gp[(size_t)d * qs];
      rv[d] = vp[(size_t)d * vs];
    }
  }

#define GLA_STEP(t_, lq, lk, lg, f)                                            \
  do {                                                                        \
    float acc = 0.f;                                                          \
    _Pragma("unroll")                                                         \
    for (int i = 0; i < KPT; ++i) {                                           \
      const float s = st[i] * __expf(lg[i]) + lk[i] * (f);                    \
      st[i] = s;                                                              \
      acc += lq[i] * s;                                                       \
    }                                                                         \
    _Pragma("unroll")                                                         \
    for (int off = 16; off; off >>= 1)                                        \
      acc += __shfl_down_sync(0xffffffffu, acc, off);                         \
    if (lane == 0) op[(size_t)(t_) * vs] = __float2bfloat16(acc * scale);     \
  } while (0)

  int t = 0;
  for (; t + 2 * PF <= T; t += PF) {
#pragma unroll
    for (int d = 0; d < PF; ++d) {
      float lq[KPT], lk[KPT], lg[KPT];
      unpack8(rq[d], lq);
      unpack8(rk[d], lk);
      unpack8(rg[d], lg);
      const float f = __bfloat162float(rv[d]);
      const int tn = t + d + PF;  // always in range here
      rq[d] = qp[(size_t)tn * qs];
      rk[d] = kp[(size_t)tn * qs];
      rg[d] = gp[(size_t)tn * qs];
      rv[d] = vp[(size_t)tn * vs];
      GLA_STEP(t + d, lq, lk, lg, f);
    }
  }
  // Drain what is still in registers (constant indices -- a runtime index into
  // rq/rk/rg would push them to local memory), then finish without prefetch.
#pragma unroll
  for (int d = 0; d < PF; ++d) {
    if (t + d < T) {
      float lq[KPT], lk[KPT], lg[KPT];
      unpack8(rq[d], lq);
      unpack8(rk[d], lk);
      unpack8(rg[d], lg);
      GLA_STEP(t + d, lq, lk, lg, __bfloat162float(rv[d]));
    }
  }
  for (int t2 = t + PF; t2 < T; ++t2) {
    float lq[KPT], lk[KPT], lg[KPT];
    unpack8(qp[(size_t)t2 * qs], lq);
    unpack8(kp[(size_t)t2 * qs], lk);
    unpack8(gp[(size_t)t2 * qs], lg);
    GLA_STEP(t2, lq, lk, lg, __bfloat162float(vp[(size_t)t2 * vs]));
  }
#undef GLA_STEP

  if (STORE_HT) {
#pragma unroll
    for (int i = 0; i < KPT; ++i) ht[hoff + (size_t)i * V] = st[i];
  }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
#define T1_LAUNCH(NL, KSUB, U)                                                 \
  do {                                                                         \
    dim3 grid(nv4 / (NL), NH);                                                 \
    size_t sh = (3 * (size_t)K + ((KSUB) > 1 ? (size_t)(KSUB) * (NL)*4 : 0)) * \
                sizeof(float);                                                 \
    if (has_h0) {                                                              \
      if (store_ht)                                                            \
        gla_t1_kernel<NL, KSUB, U, true, true>                                 \
            <<<grid, (NL) * (KSUB), sh, stream>>>(qd, kd, vd, gd, h0d, htd, od,\
                                                  K, V, scale);                \
      else                                                                     \
        gla_t1_kernel<NL, KSUB, U, true, false>                                \
            <<<grid, (NL) * (KSUB), sh, stream>>>(qd, kd, vd, gd, h0d, htd, od,\
                                                  K, V, scale);                \
    } else {                                                                   \
      if (store_ht)                                                            \
        gla_t1_kernel<NL, KSUB, U, false, true>                                \
            <<<grid, (NL) * (KSUB), sh, stream>>>(qd, kd, vd, gd, h0d, htd, od,\
                                                  K, V, scale);                \
      else                                                                     \
        gla_t1_kernel<NL, KSUB, U, false, false>                               \
            <<<grid, (NL) * (KSUB), sh, stream>>>(qd, kd, vd, gd, h0d, htd, od,\
                                                  K, V, scale);                \
    }                                                                          \
  } while (0)

#define TN_LAUNCH(PF)                                                          \
  do {                                                                         \
    dim3 grid(V, NH);                                                          \
    if (has_h0) {                                                              \
      if (store_ht)                                                            \
        gla_tn_kernel<PF, true, true><<<grid, 32, 0, stream>>>(                 \
            qd, kd, vd, gd, h0d, htd, od, T, H, V, scale);                     \
      else                                                                     \
        gla_tn_kernel<PF, true, false><<<grid, 32, 0, stream>>>(                \
            qd, kd, vd, gd, h0d, htd, od, T, H, V, scale);                     \
    } else {                                                                   \
      if (store_ht)                                                            \
        gla_tn_kernel<PF, false, true><<<grid, 32, 0, stream>>>(                \
            qd, kd, vd, gd, h0d, htd, od, T, H, V, scale);                     \
      else                                                                     \
        gla_tn_kernel<PF, false, false><<<grid, 32, 0, stream>>>(               \
            qd, kd, vd, gd, h0d, htd, od, T, H, V, scale);                     \
    }                                                                          \
  } while (0)

std::vector<at::Tensor> gla_fwd(const at::Tensor& q, const at::Tensor& k,
                                const at::Tensor& v, const at::Tensor& gk,
                                c10::optional<at::Tensor> h0_opt, double scale_in,
                                bool store_ht) {
  std::vector<at::Tensor> empty;
  if (q.scalar_type() != at::kBFloat16 || k.scalar_type() != at::kBFloat16 ||
      v.scalar_type() != at::kBFloat16 || gk.scalar_type() != at::kBFloat16)
    return empty;
  if (q.dim() != 4 || v.dim() != 4) return empty;
  if (!q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
      !gk.is_contiguous())
    return empty;

  const int B = q.size(0), T = q.size(1), H = q.size(2), K = q.size(3);
  const int V = v.size(3);
  if (k.size(3) != K || v.size(0) != B || v.size(1) != T || v.size(2) != H)
    return empty;
  const int nv4 = V / 4;
  if ((V & 3) != 0 || (nv4 % 32) != 0 || (K % 8) != 0) return empty;

  const bool has_h0 = h0_opt.has_value() && h0_opt->defined();
  at::Tensor h0;
  if (has_h0) {
    h0 = *h0_opt;
    if (h0.scalar_type() != at::kFloat || !h0.is_contiguous()) return empty;
    if (h0.dim() != 4 || h0.size(0) != B || h0.size(1) != H ||
        h0.size(2) != K || h0.size(3) != V)
      return empty;
  }
  if (T != 1 && (K != 256 || (V % 4) != 0)) return empty;  // T>1 kernel: K==256

  const int NH = B * H;
  if (NH > 65535) return empty;  // grid.y bound
  const float scale = (float)scale_in;
  auto stream = at::cuda::getCurrentCUDAStream();

  at::Tensor o = at::empty({B, T, H, V}, q.options());
  at::Tensor ht;
  if (store_ht) ht = at::empty({B, H, K, V}, q.options().dtype(at::kFloat));

  const bf16* qd = (const bf16*)q.data_ptr();
  const bf16* kd = (const bf16*)k.data_ptr();
  const bf16* vd = (const bf16*)v.data_ptr();
  const bf16* gd = (const bf16*)gk.data_ptr();
  const float* h0d = has_h0 ? h0.data_ptr<float>() : nullptr;
  float* htd = store_ht ? ht.data_ptr<float>() : nullptr;
  bf16* od = (bf16*)o.data_ptr();

  if (T == 1) {
    if (K % 64 == 0)
      T1_LAUNCH(32, 8, 8);
    else
      T1_LAUNCH(32, 8, 1);
  } else {
    TN_LAUNCH(2);
  }

  std::vector<at::Tensor> out;
  out.push_back(o);
  out.push_back(store_ht ? ht : at::Tensor());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &gla_fwd, "fused recurrent GLA forward");
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    bdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_gla_fr")
    os.makedirs(bdir, exist_ok=True)
    return load_inline(
        name="fk_fused_recurrent_gla_ext",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        build_directory=bdir,
        verbose=False,
    )


try:
    _C = _build()
except Exception:  # pragma: no cover - fall back to the Triton path
    _C = None


class FusedRecurrentGLA(nn.Module):
    """Hand-written CUDA fused-recurrent GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        gk: torch.Tensor | None = None,  # [B, T, H, K]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _C is not None and gk is not None and cu_seqlens is None:
            r = _C.fwd(q, k, v, gk, initial_state,
                       k.shape[-1] ** -0.5 if scale is None else scale,
                       output_final_state)
            if r:
                return r[0], (r[1] if output_final_state else None)
        return fused_recurrent_gla(
            q=q, k=k, v=v, gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
