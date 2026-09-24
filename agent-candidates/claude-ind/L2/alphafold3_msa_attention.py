"""MSA pair-weighted averaging for AlphaFold3 (Algorithm 10).

Weighted averaging over the MSA representation using pair activations,
NOT key-query self-attention.

Reference: openfold3/core/model/layers/msa.py MSAPairWeightedAveraging

The captured workload is tiny (m [1, 8, 16, 64], z [1, 16, 16, 128], 8 heads x 8
channels), so the eager reference is entirely launch-bound: ~19 kernels whose
total arithmetic is ~2 MFMA. Everything below collapses the whole algorithm into
a *single* CUDA kernel -- one block per (batch, sequence, query-residue) triple,
each block producing one 64-wide output row:

  1. LayerNorm(z[i, :, :]) . linear_z  -> 8 head logits per key residue j.
     The LayerNorm is folded into the projection algebraically, so the raw z row
     feeds the GEMM directly:
         logit[h][j] = rstd_j * (z[j] . A[h] - mu_j * sumA[h]) + bz[h]
     with A[h] = ln_z.weight * linear_z.weight[h], sumA[h] = sum_c A[h][c] and
     bz[h] = sum_c ln_z.bias[c] * linear_z.weight[h][c] all precomputed on the
     host.
  2. softmax over j (+ the inf * (mask - 1) pair bias).
  3. LayerNorm(m[s, :, :]) once per block, shared by the value/gate paths.
  4. The value average is re-associated to avoid materializing v:
         o[h][c] = sum_d ( sum_j w[h][j] * m_norm[j][d] ) * linear_v[h*8+c][d]
     which costs 16*8*64 + 64*64 FMA instead of a full [16, 64] x [64, 64] GEMM
     per block.
  5. gate = sigmoid(m_norm[i] . linear_g), then linear_o -- both 64x64 matvecs
     reading their weights straight from L2 (each weight element is touched
     exactly once per block, so staging them through shared memory would only
     add a round trip).

All reductions run in fp32; the only bf16 rounding is on the inputs/weights and
the final store, so the result is at least as accurate as the reference chain.
"""

from __future__ import annotations

import os
from functools import partial

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.softmax import Softmax
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# ---------------------------------------------------------------------------
# Fused kernel.  Compile-time specialized for the captured configuration
# (N_res = 16, C_m = 64, C_z = 128, 8 heads x 8 channels); anything else falls
# back to the eager reference path in ``_ref_forward``.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <optional>

#define NR   16          // N_res
#define CZ   128         // C_z
#define CM   64          // C_m
#define NH   8           // no_heads
#define CHD  8           // c_hidden
#define EE   64          // no_heads * c_hidden
#define ZST  132         // padded row stride of the shared z tile
#define RST  68          // padded row stride of the shared 64-wide tiles
#define NTHREADS 128

// bf16 -> fp32 is an exact 16-bit left shift; doing it on the bit pattern keeps
// the inner loops free of any conversion instructions.
__device__ __forceinline__ void unpack8(const uint4 v, float* o) {
  const unsigned int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    o[2 * k]     = __uint_as_float(w[k] << 16);
    o[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u);
  }
}

__global__ __launch_bounds__(NTHREADS) void msa_pwa_kernel(
    const unsigned short* __restrict__ mp,    // [B, S, NR, CM]
    const unsigned short* __restrict__ zp,    // [B, NR, NR, CZ]
    const unsigned short* __restrict__ kp,    // [B, NR, NR] or null
    const float* __restrict__ ex,             // A | sumA | bz | ln_m.w | ln_m.b
    const unsigned short* __restrict__ wp,    // linear_v | linear_g | linear_o
    unsigned short* __restrict__ op,          // [B, S, NR, CM]
    int S, float inf, float eps_m, float eps_z) {
  __shared__ float zs[NR * ZST];     // raw z[i][j][c], fp32
  __shared__ float mns[NR * RST];    // LayerNorm(m[s])[j][d]
  __shared__ float ps[NH * RST];     // sum_j w[h][j] * m_norm[j][d]
  __shared__ float muz[NR], rsz[NR], bias[NR];
  __shared__ float ws[NH * NR];      // softmax weights
  __shared__ float ogs[EE];          // (value average) * gate

  const int tid = threadIdx.x;
  const int i   = blockIdx.x & (NR - 1);
  const int sq  = (blockIdx.x >> 4) % S;
  const int bb  = (blockIdx.x >> 4) / S;

  // ---- issue every global load up front so the DRAM/L2 latency of the three
  // weight matrices overlaps the LayerNorm + logit work below ----------------
  const int eh = tid & 1;            // which 32-wide half of the row
  const int er = tid >> 1;           // 0..63 output channel
  const unsigned short* wvp = wp +             (size_t)er * CM + eh * 32;
  const unsigned short* wgp = wp + 1 * EE * CM + (size_t)er * CM + eh * 32;
  const unsigned short* wop = wp + 2 * EE * CM + (size_t)er * EE + eh * 32;
  uint4 wv[4], wg[4], wo[4];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    wv[u] = *reinterpret_cast<const uint4*>(wvp + u * 8);
    wg[u] = *reinterpret_cast<const uint4*>(wgp + u * 8);
    wo[u] = *reinterpret_cast<const uint4*>(wop + u * 8);
  }

  const int h1 = tid >> 4;           // head handled in the logit GEMM
  const int k1 = tid & 15;           // lane's slice of the C_z reduction
  float av[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) av[u] = ex[h1 * CZ + k1 + 16 * u];
  const float sumA = ex[NH * CZ + h1];
  const float bz   = ex[NH * CZ + NH + h1];

  // ---- z row i: stage as fp32 + LayerNorm statistics -----------------------
  {
    const int jj = tid >> 3;
    const int c0 = (tid & 7) * 16;
    const unsigned short* zr =
        zp + ((size_t)(bb * NR + i) * NR + jj) * CZ + c0;
    float x[16];
    unpack8(*reinterpret_cast<const uint4*>(zr), x);
    unpack8(*reinterpret_cast<const uint4*>(zr + 8), x + 8);
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int u = 0; u < 16; ++u) { s1 += x[u]; s2 = fmaf(x[u], x[u], s2); }
#pragma unroll
    for (int u = 0; u < 16; u += 4)
      *reinterpret_cast<float4*>(&zs[jj * ZST + c0 + u]) =
          make_float4(x[u], x[u + 1], x[u + 2], x[u + 3]);
#pragma unroll
    for (int off = 1; off < 8; off <<= 1) {
      s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      s2 += __shfl_xor_sync(0xffffffffu, s2, off);
    }
    if ((tid & 7) == 0) {
      const float mu = s1 * (1.f / CZ);
      muz[jj] = mu;
      rsz[jj] = rsqrtf(fmaxf(s2 * (1.f / CZ) - mu * mu, 0.f) + eps_z);
    }
  }

  // ---- LayerNorm(m[s]) ----------------------------------------------------
  {
    const int jj = tid >> 3;
    const int d0 = (tid & 7) * 8;
    const unsigned short* mr =
        mp + ((size_t)(bb * S + sq) * NR + jj) * CM + d0;
    float x[8];
    unpack8(*reinterpret_cast<const uint4*>(mr), x);
    const float4 lw0 = *reinterpret_cast<const float4*>(ex + 1040 + d0);
    const float4 lw1 = *reinterpret_cast<const float4*>(ex + 1044 + d0);
    const float4 lb0 = *reinterpret_cast<const float4*>(ex + 1104 + d0);
    const float4 lb1 = *reinterpret_cast<const float4*>(ex + 1108 + d0);
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int u = 0; u < 8; ++u) { s1 += x[u]; s2 = fmaf(x[u], x[u], s2); }
#pragma unroll
    for (int off = 1; off < 8; off <<= 1) {
      s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      s2 += __shfl_xor_sync(0xffffffffu, s2, off);
    }
    const float mu = s1 * (1.f / CM);
    const float rs = rsqrtf(fmaxf(s2 * (1.f / CM) - mu * mu, 0.f) + eps_m);
    const float lw[8] = {lw0.x, lw0.y, lw0.z, lw0.w, lw1.x, lw1.y, lw1.z, lw1.w};
    const float lb[8] = {lb0.x, lb0.y, lb0.z, lb0.w, lb1.x, lb1.y, lb1.z, lb1.w};
    float y[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) y[u] = fmaf((x[u] - mu) * rs, lw[u], lb[u]);
#pragma unroll
    for (int u = 0; u < 8; u += 4)
      *reinterpret_cast<float4*>(&mns[jj * RST + d0 + u]) =
          make_float4(y[u], y[u + 1], y[u + 2], y[u + 3]);
  }

  if (tid < NR) {
    bias[tid] = (kp == nullptr)
        ? 0.f
        : inf * (__uint_as_float((unsigned int)kp[(size_t)(bb * NR + i) * NR + tid] << 16) - 1.f);
  }
  __syncthreads();

  // ---- pair-bias logits + softmax over the key residue --------------------
  float acc[NR];
#pragma unroll
  for (int j = 0; j < NR; ++j) acc[j] = 0.f;
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const int c = k1 + 16 * u;
    const float a = av[u];
#pragma unroll
    for (int j = 0; j < NR; ++j) acc[j] = fmaf(zs[j * ZST + c], a, acc[j]);
  }
#pragma unroll
  for (int off = 1; off < 16; off <<= 1)
#pragma unroll
    for (int j = 0; j < NR; ++j) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], off);

  float mx = -3.0e38f;
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    acc[j] = fmaf(rsz[j], acc[j] - muz[j] * sumA, bz) + bias[j];
    mx = fmaxf(mx, acc[j]);
  }
  float den = 0.f;
#pragma unroll
  for (int j = 0; j < NR; ++j) { acc[j] = __expf(acc[j] - mx); den += acc[j]; }
  if (k1 == 0) {
    const float inv = 1.f / den;
#pragma unroll
    for (int j = 0; j < NR; ++j) ws[h1 * NR + j] = acc[j] * inv;
  }

  // ---- gate (independent of the softmax, so no sync needed yet) -----------
  float gacc = 0.f;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    float g[8];
    unpack8(wg[u], g);
#pragma unroll
    for (int t = 0; t < 8; ++t)
      gacc = fmaf(mns[i * RST + eh * 32 + u * 8 + t], g[t], gacc);
  }
  gacc += __shfl_xor_sync(0xffffffffu, gacc, 1);
  const float gate = 1.f / (1.f + __expf(-gacc));
  __syncthreads();

  // ---- p[h][d] = sum_j w[h][j] * m_norm[j][d] -----------------------------
  {
    const int dg = tid & 15;
    float pa[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < NR; ++j) {
      const float wj = ws[h1 * NR + j];
#pragma unroll
      for (int u = 0; u < 4; ++u)
        pa[u] = fmaf(wj, mns[j * RST + dg + 16 * u], pa[u]);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) ps[h1 * RST + dg + 16 * u] = pa[u];
  }
  __syncthreads();

  // ---- value average, gated ----------------------------------------------
  {
    float oacc = 0.f;
    const int hrow = er >> 3;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      float w[8];
      unpack8(wv[u], w);
#pragma unroll
      for (int t = 0; t < 8; ++t)
        oacc = fmaf(ps[hrow * RST + eh * 32 + u * 8 + t], w[t], oacc);
    }
    oacc += __shfl_xor_sync(0xffffffffu, oacc, 1);
    if (eh == 0) ogs[er] = oacc * gate;
  }
  __syncthreads();

  // ---- output projection --------------------------------------------------
  {
    float yacc = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      float w[8];
      unpack8(wo[u], w);
#pragma unroll
      for (int t = 0; t < 8; ++t)
        yacc = fmaf(ogs[eh * 32 + u * 8 + t], w[t], yacc);
    }
    yacc += __shfl_xor_sync(0xffffffffu, yacc, 1);
    if (eh == 0)
      op[((size_t)(bb * S + sq) * NR + i) * CM + er] =
          __nv_bfloat16_raw(__float2bfloat16(yacc)).x;
  }
}

at::Tensor msa_pwa(at::Tensor ex, at::Tensor wt, double inf, double eps_m,
                   double eps_z, at::Tensor m, at::Tensor z,
                   std::optional<at::Tensor> mask) {
  TORCH_CHECK(m.is_cuda() && z.is_cuda(), "cuda tensors required");
  TORCH_CHECK(m.device() == ex.device() && m.device() == wt.device(),
              "packed weights live on a different device");
  TORCH_CHECK(m.scalar_type() == at::kBFloat16 && z.scalar_type() == at::kBFloat16,
              "bfloat16 required");
  TORCH_CHECK(m.dim() == 4 && z.dim() == 4, "rank-4 m/z required");
  TORCH_CHECK(m.size(2) == NR && m.size(3) == CM, "unsupported m shape");
  TORCH_CHECK(z.size(0) == m.size(0) && z.size(1) == NR && z.size(2) == NR &&
                  z.size(3) == CZ, "unsupported z shape");
  TORCH_CHECK(m.is_contiguous() && z.is_contiguous(), "contiguous m/z required");
  const unsigned short* kp = nullptr;
  if (mask.has_value() && mask->defined()) {
    const at::Tensor& mk = *mask;
    TORCH_CHECK(mk.is_cuda() && mk.scalar_type() == at::kBFloat16 &&
                    mk.is_contiguous() && mk.dim() == 3 && mk.size(0) == m.size(0) &&
                    mk.size(1) == NR && mk.size(2) == NR, "unsupported mask");
    kp = reinterpret_cast<const unsigned short*>(mk.data_ptr());
  }
  const int B = (int)m.size(0), S = (int)m.size(1);
  at::Tensor out = at::empty_like(m);
  msa_pwa_kernel<<<B * S * NR, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const unsigned short*>(m.data_ptr()),
      reinterpret_cast<const unsigned short*>(z.data_ptr()), kp,
      ex.data_ptr<float>(),
      reinterpret_cast<const unsigned short*>(wt.data_ptr()),
      reinterpret_cast<unsigned short*>(out.data_ptr()),
      S, (float)inf, (float)eps_m, (float)eps_z);
  return out;
}
"""

_CPP_SRC = """
at::Tensor msa_pwa(at::Tensor ex, at::Tensor wt, double inf, double eps_m,
                   double eps_z, at::Tensor m, at::Tensor z,
                   std::optional<at::Tensor> mask);
"""


def _pin_arch() -> None:
    """Build only for the local architecture (mirrors infra/cuda_ext.py)."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:  # noqa: BLE001 - leave torch's default arch list alone
        return
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    if not caps:
        return
    os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(
        f"{c}a" if c.split(".")[0] in ("9", "10", "12") else c for c in caps)


def _load_ext():
    from torch.utils.cpp_extension import load_inline
    _pin_arch()
    return load_inline(
        name="fk_af3_msa_pwa_v1",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["msa_pwa"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


_EXT = None
try:
    if torch.cuda.is_available():
        _EXT = _load_ext()
except Exception:  # noqa: BLE001 - no nvcc / unsupported arch -> eager fallback
    _EXT = None


class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10).

    Uses pair activations as weights (softmax over token dim) instead of
    key-query attention.  Parameter names match the checkpoint layout:
    linear_v, linear_g, linear_o (no nested mha).

    Args:
        c_m: MSA input channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Per-head hidden channel dimension
        no_heads: Number of attention heads
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

        # Fused path state: ``_call`` is a partial with the packed weights bound,
        # built on the first forward (weights are loaded after __init__).
        self._call = None
        self._fails = 0

    # -- fused path ---------------------------------------------------------
    @torch.no_grad()
    def _build_call(self, m: torch.Tensor):
        """Pack the weights for the fused kernel, or return None if unsupported."""
        if (_EXT is None or self.c_m != 64 or self.c_z != 128
                or self.c_hidden != 8 or self.no_heads != 8
                or not m.is_cuda or m.dtype is not torch.bfloat16
                or self.layer_norm_m.normalized_shape != (64,)
                or self.layer_norm_z.normalized_shape != (128,)
                or not self.layer_norm_m.promote_fp32
                or not self.layer_norm_z.promote_fp32):
            return None
        dev = m.device
        wz = self.linear_z.weight.detach().to(dev, torch.float32)        # [8, 128]
        zw = self.layer_norm_z.weight
        zb = self.layer_norm_z.bias
        zw = (torch.ones(128, device=dev) if zw is None
              else zw.detach().to(dev, torch.float32))
        zb = (torch.zeros(128, device=dev) if zb is None
              else zb.detach().to(dev, torch.float32))
        mw = self.layer_norm_m.weight
        mb = self.layer_norm_m.bias
        mw = (torch.ones(64, device=dev) if mw is None
              else mw.detach().to(dev, torch.float32))
        mb = (torch.zeros(64, device=dev) if mb is None
              else mb.detach().to(dev, torch.float32))
        a = wz * zw                                                      # [8, 128]
        ex = torch.cat([a.reshape(-1), a.sum(-1), (wz * zb).sum(-1),
                        mw.reshape(-1), mb.reshape(-1)]).contiguous()
        wt = torch.cat([self.linear_v.weight.detach().reshape(-1),
                        self.linear_g.weight.detach().reshape(-1),
                        self.linear_o.weight.detach().reshape(-1)]).to(
                            dev, torch.bfloat16).contiguous()
        call = partial(_EXT.msa_pwa, ex, wt, float(self.inf),
                       float(self.layer_norm_m.eps), float(self.layer_norm_z.eps))
        self._call = call
        return call

    # -- eager reference (fallback for any shape the kernel does not cover) --
    def _ref_forward(self, m, z, mask):
        n_res = z.shape[-2]
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)

        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)

        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        o = o * g
        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            z:    [*, N_res, N_res, C_z] pair embedding
            mask: [*, N_res, N_res] pair mask

        Returns:
            [*, N_seq, N_res, C_m] updated MSA embedding
        """
        if z is None:
            return m
        call = self._call
        if call is not None:
            try:
                return call(m, z, mask)
            except Exception:  # noqa: BLE001 - shape/device outside the fast path
                self._call = None
                self._fails += 1
        if self._fails < 2:
            call = self._build_call(m)
            if call is not None:
                try:
                    return call(m, z, mask)
                except Exception:  # noqa: BLE001
                    self._call = None
                    self._fails += 1
        return self._ref_forward(m, z, mask)
