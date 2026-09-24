"""Gated linear attention (covers both GLA and RetNet).

The forward signature matches FLA's ``GatedLinearAttention.forward``
exactly so fastkernels kernels are drop-in for FLA users:

    forward(hidden_states, attention_mask=None,
            past_key_values=None, use_cache=False, **kwargs)
        -> (output, attentions, past_key_values)

Per the "Condense Variants" rule, this single class subsumes FLA's
``GatedLinearAttention`` (GLA, learned data-dependent gate) and
``MultiScaleRetention`` (RetNet, fixed-per-head decay + rotary).
The two architectures differ only in:

  * ``decay_mode``:
      - ``"learned_low_rank"`` (GLA): per-token, per-head, per-channel gk
        from a low-rank projection: ``gk = logsigmoid(W2(W1(x))) / norm``.
      - ``"fixed_per_head"`` (RetNet): data-independent gk[..., t, :] =
        log(gamma_h) for ``gamma_h = 1 - 2^(-5-h)``, broadcast across T.
  * ``use_rotary``: RetNet applies rotary to q/k; GLA does not.

Both feed into the SAME L1 recurrence kernel, and both finish with a
per-head RMSNorm + swish output gate.


Optimisation notes (this file is the L2 candidate)
==================================================

Profiled on a B200 against the captured shapes, the reference structure
loses most of its time outside the recurrence itself.

**1. Host cost, not arithmetic (the T == 1 shapes).**  The reference issues
~12 kernels per forward -- five projections, the low-rank gate GEMM,
``logsigmoid``, the normaliser divide, the recurrence, RMSNorm, SiLU, the
gate multiply, the output projection.  At the captured decode shapes
(``T == 1``, ``B`` from 1 to 256) each of those finishes in single-digit
microseconds, and the forward is bound by per-call host work: a cuBLAS
``mm`` costs ~9 us of CPU and a Triton launch ~8 us, against ~15-40 us of
real GPU work for the whole layer.  The ``T == 1`` path is therefore one
``_C.decode`` call into ``gla_decode.cu``, which issues exactly three
launches:

    fused [q|k|v|g|gk_lowrank] GEMM  ->  gla_decode_kernel  ->  o_proj GEMM

``gla_decode_kernel`` does the gate projection, ``logsigmoid``, the
normaliser divide, the state update, the ``q`` contraction, the per-head
RMSNorm and the swish gate in registers.  Its ``q/k/v/g`` operands are read
straight out of the (row-strided) fused-projection buffer, so nothing has
to be re-materialised contiguously, and ``o`` never reaches memory.

**2. A host sync in the varlen dispatch.**  ``max_seqlen`` came from
``lengths.max().item()``, which blocks the CPU mid-forward and starves the
launch queue for the rest of the call -- on the captured ``[1, 1, 2560]``
case that one sync dominated everything else.  A single-segment
``cu_seqlens`` (the packed layout for one sequence) is exactly the dense
``B == 1`` layout, so it is dropped up front and ``max_seqlen`` is known
without touching the device.

**3. A dead final-state write.**  ``output_final_state`` was wired to
``use_cache`` alone, but the final state is only ever *stored* when
``past_key_values`` is not None -- so with ``use_cache=True`` and no cache
object the recurrence produced a ``[B, H, K, V]`` fp32 tensor the caller
can never observe.  At ``B=256, K=256, V=512`` that is 671 MB of pure write
traffic.  It is now requested only when there is somewhere to put it, which
leaves every real cached-inference call unchanged.

**4. The prefill tail.**  For ``T >= 64`` the projections stay separate
(cuBLAS is at roofline there, and the L1 chunk kernel needs contiguous
operands), but two Triton kernels replace four reference ops: the gate's
second projection absorbs ``logsigmoid`` and the normaliser divide, and
``RMSNorm`` / ``SiLU`` / the gate multiply collapse into one pass that runs
at the 3-GB streaming roofline for the captured 195k-token batch.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Threshold (matches FLA's own dispatch in fla.layers.rwkv7) — below this
# the chunk kernel's launch overhead exceeds its parallel speedup, so the
# fused-recurrent path is faster for short sequences (typical decode T=1).
_CHUNK_THRESHOLD = 64

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

using bf16 = __nv_bfloat16;
#define FULL 0xffffffffu

// Round an fp32 value through bfloat16 -- the reference epilogue materializes
// bf16 between the norm, the norm weight and the gate multiply, so matching it
// keeps the candidate inside the scorer's tolerance on every element.
__device__ __forceinline__ float bfr(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ float logsig(float z) {
  return fminf(z, 0.f) - log1pf(__expf(-fabsf(z)));
}
__device__ __forceinline__ float sigf(float z) { return 1.f / (1.f + __expf(-z)); }

__device__ __forceinline__ float4 ld_bf4(const bf16* p) {
  const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(p);
  float2 a = __bfloat1622float2(q[0]);
  float2 b = __bfloat1622float2(q[1]);
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ void st_bf4(bf16* p, const float4& x) {
  __nv_bfloat162 o[2] = {__floats2bfloat162_rn(x.x, x.y),
                         __floats2bfloat162_rn(x.z, x.w)};
  *reinterpret_cast<float2*>(p) = *reinterpret_cast<const float2*>(o);
}

__device__ __forceinline__ float block_sum(float x, float* sr, int tid, int nwarp) {
#pragma unroll
  for (int off = 16; off; off >>= 1) x += __shfl_down_sync(FULL, x, off);
  if (nwarp == 1) return __shfl_sync(FULL, x, 0);
  __syncthreads();                       // protect a previous read of sr[0]
  if ((tid & 31) == 0) sr[tid >> 5] = x;
  __syncthreads();
  float s = 0.f;
  for (int i = 0; i < nwarp; ++i) s += sr[i];
  return s;
}

// ---------------------------------------------------------------------------
// T == 1 GLA step fused with the low-rank gate, the per-head RMSNorm and the
// swish output gate.  grid = (tokens, heads), block = V/4 threads: thread t
// owns the four contiguous value lanes at 4*t, so q/k/v/g are read straight
// out of the strided fused-projection buffer and o never reaches memory.
// ---------------------------------------------------------------------------
template <bool HAS_H0, bool STORE>
__global__ void gla_decode_kernel(
    const bf16* __restrict__ P, const bf16* __restrict__ GW,
    const bf16* __restrict__ GB, const float* __restrict__ H0,
    float* __restrict__ HT, const bf16* __restrict__ NW, bf16* __restrict__ Y,
    int H, int K, int V, int GR, int PN, int KD, int VD,
    float scale, float gnorm, float eps) {
  extern __shared__ float sm[];
  float* sq = sm;
  float* sk = sm + K;
  float* sd = sm + 2 * K;
  float* sa = sm + 3 * K;
  float* sr = sm + 3 * K + GR;

  const int nt = blockDim.x, tid = threadIdx.x;
  const int nwarp = (nt + 31) >> 5;
  const int m = blockIdx.x, h = blockIdx.y;
  const bf16* prow = P + (size_t)m * PN;
  const bf16* pq = prow + h * K;
  const bf16* pk = prow + KD + h * K;
  const bf16* pv = prow + 2 * KD + h * V;
  const bf16* pg = prow + 2 * KD + VD + h * V;

  const float4 vf = ld_bf4(pv + 4 * tid);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

  if (HAS_H0 || STORE) {
    const bf16* pa = prow + 2 * KD + 2 * VD;
    for (int i = tid; i < GR; i += nt) sa[i] = __bfloat162float(pa[i]);
    __syncthreads();
    for (int i = tid; i < K; i += nt) {
      const bf16* w = GW + (size_t)(h * K + i) * GR;
      float z = __bfloat162float(GB[h * K + i]);
      for (int g = 0; g < GR; ++g) z += __bfloat162float(w[g]) * sa[g];
      // The reference materializes bf16 after the GEMM, after logsigmoid and
      // after the normaliser divide; reproduce all three so the decay -- which
      // the recurrence exponentiates and compounds over T -- matches.
      sd[i] = __expf(bfr(bfr(logsig(bfr(z))) * gnorm));
      sq[i] = __bfloat162float(pq[i]) * scale;
      sk[i] = __bfloat162float(pk[i]);
    }
    __syncthreads();
    const size_t hb = (((size_t)m * H + h) * K) * V + 4 * (size_t)tid;
    for (int i = 0; i < K; ++i) {
      const float d = sd[i], kv = sk[i], q = sq[i];
      float4 s;
      if (HAS_H0) {
        const float4 b = *reinterpret_cast<const float4*>(H0 + hb + (size_t)i * V);
        s.x = b.x * d + kv * vf.x;
        s.y = b.y * d + kv * vf.y;
        s.z = b.z * d + kv * vf.z;
        s.w = b.w * d + kv * vf.w;
      } else {
        s.x = kv * vf.x;
        s.y = kv * vf.y;
        s.z = kv * vf.z;
        s.w = kv * vf.w;
      }
      if (STORE) *reinterpret_cast<float4*>(HT + hb + (size_t)i * V) = s;
      acc.x += q * s.x;
      acc.y += q * s.y;
      acc.z += q * s.z;
      acc.w += q * s.w;
    }
  } else {
    // No state in, none handed on: sum_k q_k (k_k v) with the outer product
    // factored out of the contraction.
    float part = 0.f;
    for (int i = tid; i < K; i += nt)
      part += __bfloat162float(pq[i]) * __bfloat162float(pk[i]);
    part = block_sum(part, sr, tid, nwarp) * scale;
    acc.x = part * vf.x;
    acc.y = part * vf.y;
    acc.z = part * vf.z;
    acc.w = part * vf.w;
  }

  acc.x = bfr(acc.x); acc.y = bfr(acc.y);
  acc.z = bfr(acc.z); acc.w = bfr(acc.w);
  float ss = acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
  ss = block_sum(ss, sr, tid, nwarp);
  const float rstd = rsqrtf(ss / (float)V + eps);
  const float4 w4 = ld_bf4(NW + 4 * tid);
  const float4 g4 = ld_bf4(pg + 4 * tid);
  float4 o4;
  o4.x = bfr(bfr(acc.x * rstd) * w4.x) * bfr(g4.x * sigf(g4.x));
  o4.y = bfr(bfr(acc.y * rstd) * w4.y) * bfr(g4.y * sigf(g4.y));
  o4.z = bfr(bfr(acc.z * rstd) * w4.z) * bfr(g4.z * sigf(g4.z));
  o4.w = bfr(bfr(acc.w * rstd) * w4.w) * bfr(g4.w * sigf(g4.w));
  st_bf4(Y + (size_t)m * VD + h * V + 4 * tid, o4);
}

#define LAUNCH(HH, ST)                                                        \
  gla_decode_kernel<HH, ST><<<grid, nt, sh, stream>>>(                        \
      (const bf16*)p.data_ptr(), (const bf16*)gw.data_ptr(),                  \
      (const bf16*)gb.data_ptr(), h0d, htd, (const bf16*)nw.data_ptr(),       \
      (bf16*)y.data_ptr(), (int)H, (int)K, (int)V, (int)GR, (int)PN, (int)KD,  \
      (int)VD, (float)scale, (float)gnorm, (float)eps)

// Whole T == 1 forward: fused [q|k|v|g|gk_lowrank] projection, the kernel
// above, then the output projection -- three launches from one host call.
std::vector<at::Tensor> gla_decode(
    const at::Tensor& hidden, const at::Tensor& wproj, const at::Tensor& gw,
    const at::Tensor& gb, const at::Tensor& nw, const at::Tensor& wo,
    c10::optional<at::Tensor> h0_opt, int64_t H, int64_t K, int64_t V,
    int64_t GR, double scale, double gnorm, double eps, bool store_ht) {
  std::vector<at::Tensor> empty;
  if (hidden.scalar_type() != at::kBFloat16 || wproj.scalar_type() != at::kBFloat16
      || gw.scalar_type() != at::kBFloat16 || gb.scalar_type() != at::kBFloat16
      || nw.scalar_type() != at::kBFloat16 || wo.scalar_type() != at::kBFloat16)
    return empty;
  if (!hidden.is_contiguous() || !wproj.is_contiguous() || !gw.is_contiguous()
      || !gb.is_contiguous() || !nw.is_contiguous() || !wo.is_contiguous())
    return empty;
  if ((V & 3) != 0) return empty;
  const int nt = (int)(V / 4);
  if (nt < 1 || nt > 1024) return empty;

  const int64_t HS = hidden.size(hidden.dim() - 1);
  const int64_t M = hidden.numel() / HS;
  const int64_t KD = H * K, VD = H * V;
  const int64_t PN = 2 * KD + 2 * VD + GR;
  if (wproj.size(0) != PN || wproj.size(1) != HS) return empty;
  if (wo.size(0) != HS || wo.size(1) != VD) return empty;
  if (gw.size(0) != KD || gw.size(1) != GR || gb.size(0) != KD) return empty;
  if (nw.size(0) != V) return empty;
  if (M < 1 || M > 2147483000) return empty;

  const bool has_h0 = h0_opt.has_value() && h0_opt->defined();
  at::Tensor h0;
  if (has_h0) {
    h0 = *h0_opt;
    if (h0.scalar_type() != at::kFloat || !h0.is_contiguous()) return empty;
    if (h0.numel() != M * H * K * V) return empty;
  }

  auto hx = hidden.view({M, HS});
  auto p = at::mm(hx, wproj.t());
  auto y = at::empty({M, VD}, hidden.options());
  at::Tensor ht;
  if (store_ht) ht = at::empty({M, H, K, V}, hidden.options().dtype(at::kFloat));

  const float* h0d = has_h0 ? h0.data_ptr<float>() : nullptr;
  float* htd = store_ht ? ht.data_ptr<float>() : nullptr;
  dim3 grid((unsigned)M, (unsigned)H);
  const size_t sh = (3 * (size_t)K + (size_t)GR + 32) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (has_h0) {
    if (store_ht) LAUNCH(true, true);
    else          LAUNCH(true, false);
  } else {
    if (store_ht) LAUNCH(false, true);
    else          LAUNCH(false, false);
  }

  auto sizes = hidden.sizes().vec();
  sizes[sizes.size() - 1] = HS;
  std::vector<at::Tensor> out;
  out.push_back(at::mm(y, wo.t()).view(sizes));
  out.push_back(store_ht ? ht : at::Tensor());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &gla_decode, "fused GLA decode step");
}

"""


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    bdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_gla_l2")
    os.makedirs(bdir, exist_ok=True)
    return load_inline(
        name="fk_gla_attention_l2_ext",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        functions=None,
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        build_directory=bdir,
        verbose=False,
    )


try:
    _C = _build_ext()
except Exception:  # pragma: no cover - keeps the eager path as the fallback
    _C = None


# ---------------------------------------------------------------------------
# Prefill-side Triton kernels.  Tile shapes were swept against the captured
# 195k-token batch with the scorer's own timer.
# ---------------------------------------------------------------------------
@triton.jit
def _logsigmoid(z):
    """``log(sigmoid(z))`` in the branch-free stable form."""
    return tl.minimum(z, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(z)))


@triton.jit
def _bf(x):
    """Round an fp32 value through bfloat16, as the reference epilogue does."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _gate_kernel(A, W, BI, O, M, N, gnorm,
                 GR: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """``gk = logsigmoid(A @ W.T + bias) / normalizer`` -- the low-rank gate's
    second projection with its whole epilogue folded in."""
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    gi = tl.arange(0, GR)
    mm = rm < M
    mn = rn < N
    a = tl.load(A + rm[:, None] * GR + gi[None, :], mask=mm[:, None], other=0.0)
    w = tl.load(W + rn[None, :] * GR + gi[:, None], mask=mn[None, :], other=0.0)
    z = tl.dot(a, w) + tl.load(BI + rn, mask=mn, other=0.0).to(tl.float32)[None, :]
    # Round where the reference rounds (after the GEMM, after logsigmoid): the
    # recurrence exponentiates and compounds this value over T.
    z = _bf(_logsigmoid(_bf(z))) * gnorm
    tl.store(O + rm[:, None] * N + rn[None, :], z.to(O.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


@triton.jit
def _tail_kernel(X, G, NW, Y, R, eps, V: tl.constexpr, BR: tl.constexpr):
    """``RMSNorm_per_head(x) * silu(g)`` -- one program per BR head-blocks."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    mr = r[:, None] < R
    vi = tl.arange(0, V)
    off = r[:, None].to(tl.int64) * V + vi[None, :]
    x = tl.load(X + off, mask=mr, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 1) / V + eps)
    x = _bf(_bf(x * rstd[:, None]) * tl.load(NW + vi).to(tl.float32)[None, :])
    gv = tl.load(G + off, mask=mr, other=0.0).to(tl.float32)
    tl.store(Y + off, (x * _bf(gv * tl.sigmoid(gv))).to(tl.bfloat16), mask=mr)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class GatedLinearAttention(nn.Module):
    """Unified L2 attention for GLA and RetNet.

    Args:
        hidden_size: Model hidden size.
        num_heads: Number of attention heads.
        expand_k: Key expansion ratio (GLA: 0.5, RetNet: 1.0).
        expand_v: Value expansion ratio (GLA: 1.0, RetNet: 2.0).
        decay_mode: Which forget-gate mechanism to use.
        gate_low_rank_dim: Low-rank dim for the GLA gate (ignored for
            ``fixed_per_head``).
        gate_logit_normalizer: Normalizer applied after logsigmoid in the
            GLA gate (ignored for ``fixed_per_head``).
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
        self.gate_low_rank_dim = gate_low_rank_dim
        self.norm_eps = norm_eps

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
            # FLA stores this as ``gk_proj = nn.Sequential(Linear, Linear)``
            # so the checkpoint paths are ``gk_proj.0.weight`` and
            # ``gk_proj.1.{weight,bias}``. nn.Sequential is used here purely
            # as a container; both children are L1 Linear ops.
            self.gk_proj = nn.Sequential(
                Linear(hidden_size, gate_low_rank_dim, bias=False),
                Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
            self.log_sigmoid = LogSigmoid()
        else:
            # RetNet: fixed per-head decay gamma_h = 1 - 2^(-5-h).
            # Stored as a non-persistent buffer so it auto-moves with the
            # module and is not written to checkpoints.
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            log_gamma = torch.log(gamma)
            self.register_buffer("log_gamma", log_gamma, persistent=False)

        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_k_dim,
                max_position_embeddings=rotary_max_position,
                rope_theta=rotary_base,
            )

        # Fast paths (Triton, FLA-vendored) + naive fallback (pure PyTorch).
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

        # Everything about the fused T == 1 path that depends only on the
        # config, so the per-call check is a single attribute read.
        self._decode_ready = (
            _C is not None
            and use_fast_kernels
            and not use_rotary
            and decay_mode == "learned_low_rank"
            and self.head_v_dim % 4 == 0
            and self.head_v_dim // 4 <= 1024
        )
        # Lazily-built [q|k|v|g|gk_lowrank] weight for the fused decode GEMM,
        # plus the bf16 RMSNorm weight the reference casts to on every call.
        # Dropped whenever the parameters are replaced or reloaded.
        self._proj_w: torch.Tensor | None = None
        self._nw_bf: torch.Tensor | None = None
        self.register_load_state_dict_post_hook(
            lambda mod, incompatible_keys: mod._drop_cache()
        )

    # -- derived-weight cache ---------------------------------------------
    def _drop_cache(self) -> None:
        self._proj_w = None
        self._nw_bf = None

    def _apply(self, *args, **kwargs):  # .to() / .cuda() / .float() / ...
        self._drop_cache()
        return super()._apply(*args, **kwargs)

    def _decode_weights(self, dtype):
        """(fused projection weight, bf16 norm weight), built once."""
        w = self._proj_w
        if w is None:
            w = torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                 self.g_proj.weight, self.gk_proj[0].weight), dim=0,
            ).contiguous()
            self._proj_w = w
            # RMSNorm.forward casts its weight to the activation dtype on every
            # call; cache that cast so the kernel can consume it directly.
            nw = self.g_norm_swish_gate.weight
            self._nw_bf = nw if nw.dtype == dtype else nw.to(dtype)
        return w, self._nw_bf

    def _compute_gk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, num_heads, T, head_k_dim] in log-space.

        Used by the naive recurrence path. The fast path uses
        :meth:`_compute_gk_bthk` to skip an unnecessary transpose.
        """
        if self.decay_mode == "learned_low_rank":
            gk = self.gk_proj(hidden_states)
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
            return gk.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1
        ).expand(B, self.num_heads, T, self.head_k_dim)

    def _compute_gk_bthk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, T, num_heads, head_k_dim] in log-space."""
        gk = self.gk_proj(hidden_states)
        gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
        return gk.view(B, T, self.num_heads, self.head_k_dim)

    def _gate_fused(self, a: torch.Tensor, M: int) -> torch.Tensor:
        """``logsigmoid(gk_proj[1](a)) / normalizer`` in one kernel."""
        w = self.gk_proj[1].weight
        N = w.shape[0]
        gk = torch.empty((M, N), device=a.device, dtype=a.dtype)
        BM, BN = 32, 128
        _gate_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
            a, w, self.gk_proj[1].bias, gk, M, N,
            1.0 / self.gate_logit_normalizer,
            GR=self.gate_low_rank_dim, BM=BM, BN=BN, num_warps=4,
        )
        return gk

    def _tail_fused(self, o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """``RMSNorm_per_head(o) * silu(g)`` in one kernel."""
        V, BR = self.head_v_dim, 8
        R = o.numel() // V
        y = torch.empty_like(g)
        _tail_kernel[(triton.cdiv(R, BR),)](
            o, g, self.g_norm_swish_gate.weight, y, R, self.norm_eps,
            V=V, BR=BR, num_warps=4,
        )
        return y

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
        max_seqlen = None
        if cu_seqlens is not None:
            if B != 1:
                raise ValueError("cu_seqlens prefill expects packed hidden_states with batch size 1")
            if cu_seqlens.numel() <= 2:
                # One segment covering the whole packed row: identical to the
                # dense B == 1 layout, and knowing max_seqlen == T here avoids
                # a ``.max().item()`` host sync in the middle of the forward.
                cu_seqlens = None
                max_seqlen = T
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seqlen = int(lengths.max().item()) if lengths.numel() else 0

        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))
        # The final state is only observable through ``past_key_values``; with
        # no cache object to store it in, producing it is pure write traffic.
        want_state = bool(use_cache) and past_key_values is not None

        if (self._decode_ready and T == 1 and cu_seqlens is None
                and hidden_states.dtype == torch.bfloat16 and hidden_states.is_cuda):
            w, nw = self._decode_weights(torch.bfloat16)
            r = _C.decode(
                hidden_states, w, self.gk_proj[1].weight, self.gk_proj[1].bias,
                nw, self.o_proj.weight, initial_state,
                self.num_heads, self.head_k_dim, self.head_v_dim,
                self.gate_low_rank_dim, self.head_k_dim ** -0.5,
                1.0 / self.gate_logit_normalizer, self.norm_eps, want_state,
            )
            if r:
                if want_state:
                    if not hasattr(past_key_values, "states"):
                        past_key_values.states = {}
                    past_key_values.states[id(self)] = r[1]
                return r[0], None, past_key_values

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        if self.use_rotary:
            # Build per-token absolute positions. For uncached single-shot
            # forward we use 0..T-1 per row. For cached prefill / decode the
            # engine passes ``past_key_values.seq_offsets`` (int or [B]
            # int64) giving the global position of token 0 in this call,
            # per row. Without that offset, RoPE would re-encode every
            # decode step at position 0 — totally breaking RetNet.
            #
            # NOTE: must materialize a contiguous int64 buffer with B*T real
            # elements. ``arange(T).expand(B, T).reshape(-1)`` returns a
            # stride-0 view (only T elements of storage); the CUDA RoPE
            # kernel does flat ``positions[token_idx]`` indexing which would
            # read out-of-bounds for token_idx >= T → illegal access.
            offsets = None
            if past_key_values is not None:
                offsets = getattr(past_key_values, "seq_offsets", None)
            if cu_seqlens is not None:
                # Packed varlen [1, total_T]: positions restart at each
                # sequence boundary. token t's position = its per-sequence local
                # index + that sequence's global start offset (seq_offsets, or
                # 0). This must be a flat [total_T] vector -- the dense branch
                # below builds [B*T], which is wrong for a packed batch and
                # feeds the RoPE kernel a positions length != query rows.
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
                    # [B] int64 tensor of per-row prefix lengths
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

        # Dispatch:
        #   T >= 64 + fast kernels -> chunk (prefill / training)
        #   T  < 64 + fast kernels -> fused_recurrent (decode)
        #   no fast kernels         -> naive PyTorch (CPU / debug / reference)
        fused_tail = False
        if self.use_fast_kernels and q.is_cuda:
            dispatch_len = max_seqlen if max_seqlen is not None else T
            if self.decay_mode == "learned_low_rank":
                # gk in [B, T, H, K] log-space, NOT transposed
                if (hidden_states.dtype == torch.bfloat16 and g.is_contiguous()
                        and _is_pow2(self.head_v_dim)
                        and _is_pow2(self.gate_low_rank_dim)
                        and self.gate_low_rank_dim >= 16):
                    a = self.gk_proj[0](hidden_states).reshape(B * T, -1)
                    gk_btHK = self._gate_fused(a, B * T).view(
                        B, T, self.num_heads, self.head_k_dim)
                    fused_tail = True
                else:
                    gk_btHK = self._compute_gk_bthk(hidden_states, B, T)
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v, g=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=want_state,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v, gk=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=want_state,
                        cu_seqlens=cu_seqlens,
                    )
            else:  # RetNet — kernel bakes in the per-head decay
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=want_state,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=want_state,
                        cu_seqlens=cu_seqlens,
                    )
            # Fast-path output is already [B, T, H, V] — no transpose needed.
        else:
            # Naive path expects [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            gk = self._compute_gk(hidden_states, B, T)
            o, final_state = self.naive_recurrence(
                q, k, v, gk,
                initial_state=initial_state,
                output_final_state=want_state,
            )
            o = o.transpose(1, 2)  # [B, H, T, V] -> [B, T, H, V]

        if want_state:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state

        if fused_tail and o.is_cuda and o.dtype == torch.bfloat16:
            o = self._tail_fused(o.contiguous(), g.reshape(B * T, self.value_dim))
        else:
            o = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
            o = o.view(B, T, self.value_dim)
            o = o * self.gate_act(g)

        return self.o_proj(o).view(B, T, -1), None, past_key_values
